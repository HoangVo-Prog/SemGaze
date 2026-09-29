
import os

import numpy as np
from PIL import Image
from torchvision import transforms
from .utils import compute_search_cdf, preprocess_fixations, filter_scanpath, select_fewshot_subject
from .utils import cutFixOnTarget
from .data import  Siamese_Triplet_Gaze


def resolve_air_image_path(image_id, record, image_root):
    """Resolve an AiR-D image without guessing names or extensions."""
    image_id = os.fspath(image_id)
    image_path = image_id if os.path.isabs(image_id) else os.path.join(image_root, image_id)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            "AiR-D image not found for image_id={!r}, question_id={!r}, "
            "subject_idx={!r}: {!r}".format(
                image_id,
                record.get("question_id"),
                record.get("subject_idx"),
                image_path))
    return os.path.abspath(image_path)


def load_air_question_embeddings(path):
    """Load the question_id -> 768-D embedding dictionary produced by AiR."""
    if not os.path.isfile(path):
        raise FileNotFoundError("AiR-D question embedding file not found: {!r}".format(path))
    embeddings = np.load(path, allow_pickle=True)
    if isinstance(embeddings, np.ndarray) and embeddings.shape == ():
        embeddings = embeddings.item()
    if not isinstance(embeddings, dict):
        raise ValueError("AiR-D question embedding file must contain a dictionary: {!r}".format(path))
    return embeddings


def _air_question_embedding(embeddings, question_id, record):
    """Look up one question embedding, accepting only exact/string-int equivalents."""
    candidates = [question_id]
    if isinstance(question_id, str):
        candidates.append(question_id.strip())
        try:
            candidates.append(int(question_id))
        except ValueError:
            pass
    elif isinstance(question_id, (int, np.integer)):
        candidates.append(str(int(question_id)))

    for key in candidates:
        if key in embeddings:
            embedding = np.asarray(embeddings[key], dtype=np.float32)
            if embedding.ndim != 1:
                raise ValueError(
                    "AiR-D question_id={!r}, image_id={!r}, subject_idx={!r}: "
                    "embedding must be a 1-D vector, got shape {}".format(
                        question_id, record.get("image_id"), record.get("subject_idx"), embedding.shape))
            return embedding

    raise KeyError(
        "AiR-D question embedding missing for question_id={!r}, image_id={!r}, "
        "subject_idx={!r}".format(
            question_id, record.get("image_id"), record.get("subject_idx")))


def validate_air_task_embedding_dim(hparams):
    """Enforce the existing UserEmbeddingNet.task_transform input width."""
    expected_dim = 768
    configured_dim = getattr(hparams.Data, "task_embedding_dim", expected_dim)
    if configured_dim != expected_dim:
        raise ValueError(
            "AiR-D Data.task_embedding_dim={} must match "
            "UserEmbeddingNet.task_transform input width {}".format(configured_dim, expected_dim))
    return expected_dim


def build_air_subject_mapping(records, num_subjects, subject_order=None):
    """Build one deterministic subject_idx mapping for all AiR-D splits."""
    raw_subjects = []
    for record in records:
        if "subject_idx" not in record:
            raise ValueError("AiR-D record is missing subject_idx (question_id={!r}, image_id={!r})".format(
                record.get("question_id"), record.get("image_id")))
        subject = record["subject_idx"]
        if isinstance(subject, bool) or not isinstance(subject, (int, np.integer)):
            raise ValueError("AiR-D subject_idx must be an integer, got {!r} (question_id={!r}, image_id={!r})".format(
                subject, record.get("question_id"), record.get("image_id")))
        raw_subjects.append(int(subject))

    unique_subjects = sorted(set(raw_subjects))
    if subject_order is not None:
        if len(set(subject_order)) != len(subject_order) or set(subject_order) != set(unique_subjects):
            raise ValueError(
                "AiR-D fewshot_subject must contain distinct retained subject_idx values; "
                "requested={!r}, retained={!r}".format(subject_order, unique_subjects))
        mapping = {subject: mapped for mapped, subject in enumerate(subject_order)}
    elif unique_subjects == list(range(len(unique_subjects))):
        mapping = {subject: subject for subject in unique_subjects}
    else:
        mapping = {subject: mapped for mapped, subject in enumerate(unique_subjects)}

    if len(mapping) > num_subjects or any(mapped >= num_subjects for mapped in mapping.values()):
        raise ValueError(
            "AiR-D has {} subjects but Data.num_subjects is {}; subject_idx values={!r}".format(
                len(mapping), num_subjects, unique_subjects))
    return mapping


def process_air_data(records,
                     subject_mapping,
                     question_embeddings,
                     dataset_root,
                     hparams,
                     image_root=None):
    """Convert one AiR-D split to full-scanpath SE-Net records."""
    expected_embedding_dim = validate_air_task_embedding_dim(hparams)
    if not records:
        return []

    image_root = image_root or os.path.join(dataset_root, hparams.Data.image_path)
    max_length = hparams.Data.max_traj_length
    processed = []

    for record in records:
        question_id = record.get("question_id")
        image_id = record.get("image_id")
        raw_subject = record.get("subject_idx")
        context = "question_id={!r}, image_id={!r}, subject_idx={!r}".format(
            question_id, image_id, raw_subject)
        if question_id is None or image_id is None or raw_subject is None:
            raise ValueError("AiR-D record is missing question_id/image_id/subject_idx ({})".format(context))
        if raw_subject not in subject_mapping:
            raise ValueError("AiR-D subject_idx={!r} has no global mapping ({})".format(raw_subject, context))

        try:
            height = float(record["height"])
            width = float(record["width"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("AiR-D record has invalid source dimensions ({})".format(context))
        if not np.isfinite(height) or not np.isfinite(width) or height <= 0 or width <= 0:
            raise ValueError("AiR-D source height/width must be positive ({})".format(context))

        x = np.asarray(record.get("X", []), dtype=np.float32)
        y = np.asarray(record.get("Y", []), dtype=np.float32)
        t_start = np.asarray(record.get("T_start", []), dtype=np.float32)
        t_end = np.asarray(record.get("T_end", []), dtype=np.float32)
        if not (len(x) == len(y)):
            raise ValueError("AiR-D X/Y lengths differ ({})".format(context))
        if not (len(t_start) == len(t_end) == len(x)):
            raise ValueError("AiR-D T_start/T_end lengths are incompatible with X/Y ({})".format(context))
        try:
            source_length = int(record["length"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("AiR-D length is invalid ({})".format(context))
        if source_length != len(x):
            raise ValueError(
                "AiR-D length={} is incompatible with fixation arrays of length {} ({})".format(
                    source_length, len(x), context))
        if source_length <= 0:
            raise ValueError("AiR-D source contains an empty scanpath ({})".format(context))

        durations = t_end - t_start
        if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or
                not np.all(np.isfinite(durations)) or np.any(durations < 0)):
            raise ValueError("AiR-D fixation coordinates/durations are invalid or duration is negative ({})".format(context))

        image_path = resolve_air_image_path(image_id, record, image_root)
        try:
            with Image.open(image_path) as image:
                image.convert("RGB")
        except Exception as exc:
            raise ValueError("AiR-D image is not a readable RGB image ({})".format(context)) from exc

        task_emb = _air_question_embedding(question_embeddings, question_id, record)
        if task_emb.shape[0] != expected_embedding_dim:
            raise ValueError(
                "AiR-D question embedding width {} != expected {} ({})".format(
                    task_emb.shape[0], expected_embedding_dim, context))

        kept_length = min(source_length, max_length)
        # AiR-D coordinates are source-pixel coordinates.  There is deliberately
        # no (X - 1, Y - 1) correction here.
        resized_x = x[:kept_length] * (hparams.Data.im_w / width)
        resized_y = y[:kept_length] * (hparams.Data.im_h / height)
        fixations = np.stack((resized_x, resized_y), axis=1).astype(np.float32).tolist()
        processed.append({
            "subject_id": int(subject_mapping[raw_subject]),
            "image_id": image_id,
            "img_name": image_id,
            "image_path": image_path,
            "question_id": question_id,
            "fixations": fixations,
            "duration": durations[:kept_length].astype(np.float32).tolist(),
            "task_emb": task_emb,
            "scanpath_length": kept_length,
        })

    return processed


def process_data(target_trajs,
                 dataset_root,
                 target_annos,
                 hparams,
                 device):

    print("using", hparams.Train.repr, 'dataset:', hparams.Data.name, 'TAP:',
          hparams.Data.TAP)

    # Rescale fixations and images if necessary
    if hparams.Data.name == 'OSIE':
        ori_h, ori_w = 600, 800
        rescale_flag = hparams.Data.im_h != ori_h
    elif hparams.Data.name == 'COCO-Search18' or hparams.Data.name == 'COCO-Freeview':
        ori_h, ori_w = 320, 512
        rescale_flag = hparams.Data.im_h != ori_h
    elif hparams.Data.name == 'MIT1003':
        rescale_flag = False # Use rescaled scanpaths
    elif hparams.Data.name == 'CAT2000':
        ori_h, ori_w = 1080, 1920
        rescale_flag = hparams.Data.im_h != ori_h
    else:
        print(f"dataset {hparams.Data.name} not supported")
        raise NotImplementedError
    if rescale_flag:
        print(
            f"Rescaling image and fixation to {hparams.Data.im_h}x{hparams.Data.im_w}"
        )
        size = (hparams.Data.im_h, hparams.Data.im_w)
        ratio_h = hparams.Data.im_h / ori_h
        ratio_w = hparams.Data.im_w / ori_w
        for traj in target_trajs:
            traj['X'] = np.array(traj['X']) * ratio_w
            traj['Y'] = np.array(traj['Y']) * ratio_h
            traj['rescaled'] = True


    size = (hparams.Data.im_h, hparams.Data.im_w)
    transform_train = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])

    valid_target_trajs = list(
        filter(lambda x: x['split'] == 'test', target_trajs))
    

    is_coco_dataset = hparams.Data.name == 'COCO-Search18' or hparams.Data.name == 'COCO-Freeview'


    target_init_fixs = {}
    for traj in target_trajs:
        key = traj['task'] + '*' + traj['name'] + '*' + traj['condition']
        if is_coco_dataset:
            # Force center initialization for COCO-Search18
            target_init_fixs[key] = (0.5, 0.5)  
        else:
            target_init_fixs[key] = (traj['X'][0] / hparams.Data.im_w,
                                     traj['Y'][0] / hparams.Data.im_h)
    cat_names = list(np.unique([x['task'] for x in target_trajs]))
    catIds = dict(zip(cat_names, list(range(len(cat_names)))))

    human_mean_cdf = None
    # training fixation data
    train_target_trajs = list(
        filter(lambda x: x['split'] == 'train', target_trajs))

    # fewshot_subject indicating subject ids for unseen subjects
    if hparams.Data.fewshot_subject[0] != -1:
        train_target_trajs = select_fewshot_subject(hparams.Train.log_dir,
            train_target_trajs, hparams.Data.fewshot_subject, 
            hparams.Data.num_fewshot, hparams.Data.num_subjects, hparams.Data.random_support, 'train')
        
    # print statistics
    traj_lens = list(map(lambda x: x['length'], train_target_trajs))
    avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
    print('average train scanpath length : {:.3f} (+/-{:.3f})'.format(
        avg_traj_len, std_traj_len))
    print('num of train trajs = {}'.format(len(train_target_trajs)))

    train_task_img_pair = np.unique([
        traj['task'] + '*' + traj['name'] + '*' + traj['condition']
        for traj in train_target_trajs
    ])
    train_fix_labels = preprocess_fixations(
        train_target_trajs,
        hparams.Data.patch_size,
        hparams.Data.patch_num,
        hparams.Data.im_h,
        hparams.Data.im_w,
        truncate_num=hparams.Data.max_traj_length,
        has_stop=hparams.Data.has_stop,
        sample_scanpath=False,
        min_traj_length_percentage=0,
        discretize_fix=hparams.Data.discretize_fix,
        remove_return_fixations=hparams.Data.remove_return_fixations,
        is_coco_dataset=is_coco_dataset,
    )

    # validation fixation data
    valid_target_trajs = list(
        filter(lambda x: x['split'] == 'test', target_trajs))
    
    
    # print statistics
    traj_lens = list(map(lambda x: x['length'], valid_target_trajs))
    avg_traj_len, std_traj_len = np.mean(traj_lens), np.std(traj_lens)
    print('average valid scanpath length : {:.3f} (+/-{:.3f})'.format(
        avg_traj_len, std_traj_len))
    print('num of valid trajs = {}'.format(len(valid_target_trajs)))

    
    if hparams.Data.TAP in ['TP', 'TAP']:
        tp_trajs = list(
        filter(
            lambda x: x['condition'] == 'present' and x['split'] == 'test',
            target_trajs))
        human_mean_cdf, _ = compute_search_cdf(
            tp_trajs, target_annos, hparams.Data.max_traj_length)
        print('target fixation prob (valid).:', human_mean_cdf)

    valid_fix_labels = preprocess_fixations(
        valid_target_trajs,
        hparams.Data.patch_size,
        hparams.Data.patch_num,
        hparams.Data.im_h,
        hparams.Data.im_w,
        truncate_num=hparams.Data.max_traj_length,
        has_stop=hparams.Data.has_stop,
        sample_scanpath=False,
        min_traj_length_percentage=0,
        discretize_fix=hparams.Data.discretize_fix,
        remove_return_fixations=hparams.Data.remove_return_fixations,
        is_coco_dataset=is_coco_dataset,
    )


    # original HAT code is generate training examples for each fixation, 
    # in SE-Net, we only select the whole scanpath
    train_fix_labels = filter_scanpath(train_fix_labels)
    valid_fix_labels = filter_scanpath(valid_fix_labels)
    
    
    train_HG_dataset = Siamese_Triplet_Gaze(dataset_root,
                                            train_fix_labels,
                                            target_annos,
                                            hparams.Data,
                                            transform_train,
                                            catIds,
                                            device,
                                            blur_action=True)
    valid_HG_dataset = Siamese_Triplet_Gaze(dataset_root,
                                            valid_fix_labels,
                                            target_annos,
                                            hparams.Data,
                                            transform_test,
                                            catIds,
                                            device,
                                            blur_action=True)

    if hparams.Data.TAP == ['TP', 'TAP']:
        cutFixOnTarget(target_trajs, target_annos)
    print("num of training and eval fixations = {}, {}".format(
        len(train_HG_dataset), len(valid_HG_dataset)))

    return {
        'catIds': catIds,
        'gaze_train': train_HG_dataset,
        'gaze_valid': valid_HG_dataset,
        'bbox_annos': target_annos,
        'valid_scanpaths': valid_target_trajs,
        'human_cdf': human_mean_cdf,
    }
