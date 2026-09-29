import sys
import os

sys.path.append('../common')

from common.dataset import (
    build_air_subject_mapping,
    load_air_question_embeddings,
    process_air_data,
    process_data,
    validate_air_task_embedding_dim,
)
from .models import UserEmbeddingNet
from common.utils import adjust_subjects, select_fewshot_subject
from common.data import Siamese_Triplet_Gaze
import json
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms


def _air_split_path(dataset_root, hparams, split):
    configured = getattr(hparams.Data, '{}_fix_path'.format(split), None)
    if configured is None:
        configured = getattr(hparams.Data, 'fix_path', None)
        if not isinstance(configured, str) or '{split}' not in configured:
            raise ValueError(
                "AiR-D requires explicit train/validation/test_fix_path values "
                "or a fix_path template containing {{split}}; missing {}_fix_path".format(split))
        try:
            configured = configured.format(split=split)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError("AiR-D invalid fix_path template: {!r}".format(configured)) from exc
    return configured if os.path.isabs(configured) else os.path.join(dataset_root, configured)


def _load_air_split(dataset_root, hparams, split):
    path = _air_split_path(dataset_root, hparams, split)
    if not os.path.isfile(path):
        raise FileNotFoundError("AiR-D {} split file not found: {!r}".format(split, path))
    with open(path, 'r') as json_file:
        records = json.load(json_file)
    if not isinstance(records, list):
        raise ValueError("AiR-D {} split must contain a JSON list: {!r}".format(split, path))
    return records


def _build_air_datasets(hparams, dataset_root, device, is_eval=False):
    validate_air_task_embedding_dim(hparams)
    split_paths = [_air_split_path(dataset_root, hparams, split)
                   for split in ('train', 'validation', 'test')]
    if len({os.path.normcase(os.path.realpath(path)) for path in split_paths}) != 3:
        raise ValueError("AiR-D train/validation/test split paths must be distinct: {!r}".format(split_paths))
    train_records = _load_air_split(dataset_root, hparams, 'train')
    valid_records = _load_air_split(dataset_root, hparams, 'validation')
    test_records = _load_air_split(dataset_root, hparams, 'test')
    if not train_records:
        raise ValueError("AiR-D train split is empty")
    if not valid_records:
        raise ValueError("AiR-D validation split is empty")

    excluded_subjects = set(hparams.Data.subject)
    if excluded_subjects != {-1}:
        train_records = [r for r in train_records if r.get('subject_idx') not in excluded_subjects]
        valid_records = [r for r in valid_records if r.get('subject_idx') not in excluded_subjects]
        test_records = [r for r in test_records if r.get('subject_idx') not in excluded_subjects]
        if not train_records:
            raise ValueError("AiR-D train split is empty after excluding subject_idx={!r}".format(
                sorted(excluded_subjects)))
        if not valid_records:
            raise ValueError("AiR-D validation split is empty after excluding subject_idx={!r}".format(
                sorted(excluded_subjects)))

    fewshot_subjects = hparams.Data.fewshot_subject if hparams.Data.fewshot_subject[0] != -1 else None
    air_support_export = is_eval and fewshot_subjects is not None
    if fewshot_subjects is not None:
        if hparams.Data.num_fewshot <= 0:
            raise ValueError("AiR-D Data.num_fewshot must be positive when fewshot_subject is active")
        train_records = [record for record in train_records if record.get('subject_idx') in fewshot_subjects]
        valid_records = [record for record in valid_records if record.get('subject_idx') in fewshot_subjects]
        test_records = [record for record in test_records if record.get('subject_idx') in fewshot_subjects]

    all_records = train_records + valid_records + test_records
    subject_mapping = build_air_subject_mapping(
        all_records, hparams.Data.num_subjects, subject_order=fewshot_subjects)

    if fewshot_subjects is not None:
        support_records = [dict(record, subject=record['subject_idx'], name=record['image_id'])
                           for record in train_records]
        train_records = select_fewshot_subject(
            hparams.Train.log_dir, support_records, fewshot_subjects,
            hparams.Data.num_fewshot, hparams.Data.num_subjects,
            hparams.Data.random_support, 'train')
        missing_subjects = set(fewshot_subjects) - {record['subject_idx'] for record in train_records}
        if missing_subjects:
            raise ValueError("AiR-D selected support images have no trials for subject_idx={!r}".format(
                sorted(missing_subjects)))
        if not valid_records:
            raise ValueError("AiR-D validation split is empty after fewshot_subject selection")

    embedding_path = hparams.Data.task_embedding_path
    if not os.path.isabs(embedding_path):
        embedding_path = os.path.join(dataset_root, embedding_path)
    question_embeddings = load_air_question_embeddings(embedding_path)
    image_root = hparams.Data.image_path
    if not os.path.isabs(image_root):
        image_root = os.path.join(dataset_root, image_root)

    train_data = process_air_data(
        train_records, subject_mapping, question_embeddings, dataset_root, hparams, image_root=image_root)
    valid_data = process_air_data(
        valid_records, subject_mapping, question_embeddings, dataset_root, hparams, image_root=image_root)
    # Load and validate test with the same mapping, but never use it as the
    # validation loader.  SE-Net's user-embedding flow has no test loader.
    process_air_data(
        test_records, subject_mapping, question_embeddings, dataset_root, hparams, image_root=image_root)

    train_subjects = {record['subject_id'] for record in train_data}
    one_shot_support = air_support_export and hparams.Data.num_fewshot == 1
    if len(train_subjects) < 2 and not one_shot_support:
        raise ValueError("AiR-D train split needs at least two subjects for triplets")

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
    train_dataset = Siamese_Triplet_Gaze(
        dataset_root, train_data, {}, hparams.Data, transform_train, {}, device,
        blur_action=True, air_split='train', air_support_export=air_support_export)
    valid_dataset = Siamese_Triplet_Gaze(
        dataset_root, valid_data, {}, hparams.Data, transform_test, {}, device,
        blur_action=True, air_split='validation', air_support_export=air_support_export)
    return {
        'catIds': {},
        'gaze_train': train_dataset,
        'gaze_valid': valid_dataset,
        'bbox_annos': {},
        'valid_scanpaths': valid_records,
        'human_cdf': None,
    }


def build(hparams, dataset_root, device, is_eval=False, split=1):
    dataset_name = hparams.Data.name
    air_support_export = (dataset_name == 'AiR-D' and is_eval
                          and hparams.Data.fewshot_subject[0] != -1)

    if dataset_name == 'AiR-D':
        dataset = _build_air_datasets(hparams, dataset_root, device, is_eval=is_eval)
        bbox_annos = dataset['bbox_annos']
        n_tasks = 2  # ntask != 1 activates the existing task-conditioning path.
    else:
        bbox_annos = np.load(
            join(dataset_root, 'bbox_annos.npy'),
            allow_pickle=True).item() if dataset_name == 'COCO-Search18' else {}

        with open(join(dataset_root, hparams.Data.fix_path), 'r') as json_file:
            human_scanpaths = json.load(json_file)
        if dataset_name == 'COCO-Search18':
            n_tasks = 18
        else:
            n_tasks = 1

        # hparams.Data.subject indicating which subjects are unseen subjects
        if hparams.Data.subject[0] != -1:
            print(f"skip subject {hparams.Data.subject} data!")
            human_scanpaths = adjust_subjects(human_scanpaths, hparams.Data.subject)

        # Filtering training data
        if hparams.Data.TAP == 'TP':
            human_scanpaths = list(
                filter(lambda x: x['condition'] == 'present', human_scanpaths))
            human_scanpaths = list(
                filter(lambda x: x['fixOnTarget'], human_scanpaths))
        elif hparams.Data.TAP == 'FV':
            human_scanpaths = list(
                filter(lambda x: x['condition'] == 'freeview', human_scanpaths))
            n_tasks = 1

        # process fixation data
        dataset = process_data(
            human_scanpaths,
            dataset_root,
            bbox_annos,
            hparams,
            device)

    batch_size = hparams.Train.batch_size
    n_workers = hparams.Train.n_workers

    tag = False if is_eval else True
    bs = batch_size // 2 if is_eval else batch_size
    train_HG_loader = DataLoader(dataset['gaze_train'],
                                 batch_size=bs,
                                 shuffle=tag,
                                 num_workers=n_workers,
                                 drop_last=not air_support_export,
                                 pin_memory=True)
    print('num of training batches =', len(train_HG_loader))

    
    valid_HG_loader = DataLoader(dataset['gaze_valid'],
                                    batch_size=batch_size//2,
                                    shuffle=False,
                                    num_workers=n_workers,
                                    drop_last=False,
                                    pin_memory=True)


    # Create model
    emb_size = hparams.Model.embedding_dim
    n_heads = hparams.Model.n_heads
    hidden_size = hparams.Model.hidden_dim


    model = UserEmbeddingNet(
            hparams.Data,
            num_decoder_layers=hparams.Model.n_dec_layers,
            hidden_dim=emb_size,
            nhead=n_heads,
            ntask=n_tasks,
            num_output_layers=hparams.Model.num_output_layers,
            train_encoder=hparams.Train.train_backbone,
            train_pixel_decoder=hparams.Train.train_pixel_decoder,
            dropout=hparams.Train.dropout,
            dim_feedforward=hidden_size,
            num_encoder_layers=hparams.Model.n_enc_layers)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=hparams.Train.adam_lr,
                                  betas=hparams.Train.adam_betas)

    # Load weights from checkpoint when available
    if len(hparams.Model.checkpoint) > 0:
        print(f"loading weights from {hparams.Model.checkpoint} in {hparams.Train.transfer_learn} setting.")
        ckp = torch.load(join(hparams.Train.log_dir, hparams.Model.checkpoint), map_location=device)

        model.load_state_dict(ckp['model'], strict=False)
        # optimizer.load_state_dict(ckp['optimizer'])
        global_step = ckp['step']
    else:
        global_step = 0

    if hparams.Train.parallel:
        model = torch.nn.DataParallel(model)

    bbox_annos = dataset['bbox_annos']
    human_cdf = dataset['human_cdf']


    if dataset_name == 'AiR-D':
        term_pos_weight = 1.0
    else:
        is_lasts = [x[5] for x in dataset['gaze_train'].fix_labels]
        term_pos_weight = len(is_lasts) / np.sum(is_lasts) - 1
    print("termination pos weight: {:.3f}".format(term_pos_weight))

    return (model, optimizer, train_HG_loader, valid_HG_loader, term_pos_weight, global_step)
