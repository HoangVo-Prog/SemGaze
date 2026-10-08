"""Read configured persisted splits; verify membership rather than reconstructing it."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from .schema import NormalizedRecord
from .semantic import semantic_from_prediction
from semgaze.model.config import default_section

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = REPO_ROOT / default_section('data')['split_root']
IMAGES_ROOT = REPO_ROOT / default_section('data')['images_root']


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _stimulus_ids(payload, key, *, source):
    values = payload.get(key)
    if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f'{source} must contain a list of nonempty string {key}')
    if len(values) != len(set(values)):
        raise ValueError(f'{source} contains duplicate {key}')
    return set(values)


def _master_stimulus_ids(master):
    train = _stimulus_ids(master, 'train_stimulus_ids', source='master split manifest')
    # New manifests describe one shared held-out image pool.  Keep accepting
    # the old name for checkpoints/splits produced by the previous protocol,
    # but never prefer it when the new field is present.
    eval_key = 'eval_stimulus_ids' if 'eval_stimulus_ids' in master else 'test_stimulus_ids'
    if eval_key not in master:
        raise ValueError(
            'master split manifest must contain eval_stimulus_ids '
            '(or legacy test_stimulus_ids)'
        )
    return train, _stimulus_ids(master, eval_key, source='master split manifest')


def read_persisted_splits(split_root=SPLIT_ROOT, *, data_config=None):
    root = Path(split_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = root / 'split_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    # prepare_split.py writes self-contained dataset manifests.  The combined
    # directory needs dataset selection before the model sees any records.
    if 'source_manifests' in manifest or 'source_file' in manifest:
        if data_config and data_config['dataset'] == 'all':
            from .joint import read_joint_splits
            return read_joint_splits(root, manifest, data_config)
        from .json_splits import read_current_json_splits
        return read_current_json_splits(root, manifest, data_config)
    master_path = root.parent / 'master_split_manifest.json'
    index_path = root.parent / 'split_index.json'
    master = json.loads(master_path.read_text(encoding='utf-8'))
    index = json.loads(index_path.read_text(encoding='utf-8'))

    master_train, master_eval = _master_stimulus_ids(master)
    if master_train & master_eval:
        raise ValueError('master train/eval stimulus leakage')
    if index['k_shot'] != [1, 5, 10] or index['num_exclusive_draws_per_k'] != 10:
        raise ValueError('expected K_eval=1/5/10 with 10 exclusive draws')
    entry = next((v for v in index['variants'] if v['variant'] == root.name), None)
    if entry is None or entry['manifest_sha256'] != sha256_file(manifest_path):
        raise ValueError('variant manifest differs from shared split_index')
    unseen_subjects = set(manifest['unseen_subject_ids'])
    if unseen_subjects != {7, 8, 9}:
        raise ValueError('unseen subjects must remain {7,8,9}')
    variant = manifest['variant']
    if variant != root.name:
        raise ValueError('manifest variant differs from configured path')
    duration_field = data_config['duration']['source_field'] if data_config else 'T'
    if data_config and (variant != data_config['variant'] or unseen_subjects != set(data_config['unseen_subjects'])):
        raise ValueError('unexpected variant/unseen subjects in manifest')
    if 'eval_stimulus_ids' in manifest:
        variant_eval = _stimulus_ids(manifest, 'eval_stimulus_ids', source='variant split manifest')
        if variant_eval != master_eval:
            raise ValueError('variant eval stimulus membership differs from master eval pool')
    source = REPO_ROOT / manifest['curated_source_file']
    if not manifest['curated_source_sha256'] == master['curated_source_sha256'] == index['source_sha256'] == sha256_file(source):
        raise ValueError('curated-source checksum differs from manifest')
    splits, ids, images = {}, set(), set()
    for split in ('train', 'test'):
        path = root / f'{split}.json'
        if not path.exists():
            raise ValueError(f'COCO-Search18 requires train.json and test.json; missing {path}')
        if sha256_file(path) != manifest['split_file_sha256'][split]:
            raise ValueError(f'{split}: split checksum differs from manifest')
        records = json.loads(path.read_text(encoding='utf-8'))
        split_images = {r['stimulus_id'] for r in records}
        declared_images = _stimulus_ids(manifest, f'{split}_stimulus_ids', source='variant split manifest')
        if ((split == 'test' and split_images != declared_images)
                or images & split_images):
            raise ValueError(f'{split}: stimulus membership/leakage mismatch')
        if split == 'train':
            if (not split_images <= master_train or split_images & master_eval
                    or not split_images <= declared_images
                    or not declared_images <= master_train):
                raise ValueError('train: stimulus membership leaks into master eval pool')
        elif not split_images <= master_eval:
            raise ValueError('test: stimulus membership is outside master eval pool')
        if split == 'test' and any(r.get('subject') not in unseen_subjects for r in records):
            raise ValueError('test.json contains a non-unseen-subject record')
        record_counts = entry.get('record_counts', entry.get('runtime_record_counts'))
        if not isinstance(record_counts, dict) or split not in record_counts:
            raise ValueError(f'split_index missing {split} record count')
        image_counts = entry.get('image_counts')
        if (isinstance(image_counts, dict) and split in image_counts
                and len(split_images) != image_counts[split]):
            raise ValueError(f'{split}: split_index image count differs from persisted records')
        if len(records) != record_counts[split]:
            raise ValueError('split_index counts differ from persisted records')
        images.update(split_images)
        for r in records:
            rid = r['record_id']
            if rid in ids or r['variant'] != variant or r['split'] != split:
                raise ValueError(f'{rid}: invalid record membership or duplicate ID')
            ids.add(rid)
            if (rid != f"coco_semgaze::{r['subject']}::{r['task']}::{r['name']}"
                    or r['stimulus_id'] != r['name'] or r['trial_key'] != f"{r['task']}::{r['name']}"):
                raise ValueError(f'{rid}: invalid canonical record identity')
            if not len(r['X']) == len(r['Y']) == len(r[duration_field]) or not r['X']:
                raise ValueError(f'{rid}: misaligned X/Y/T')
            semantic_from_prediction(r['prediction'], len(r['X']), rid)
        splits[split] = records
    train = {r['record_id']: r for r in splits['train']}
    train_images = {r['stimulus_id'] for r in splits['train']}
    test_images = {r['stimulus_id'] for r in splits['test']}
    if train_images & test_images:
        raise ValueError('train/test stimulus leakage')
    seen = {r['subject'] for r in splits['train']} - unseen_subjects
    if seen != set(manifest['seen_subject_ids']):
        raise ValueError('seen-subject membership mismatch')
    def check_entry(entry, rid, subject):
        if rid not in train:
            raise ValueError('frozen support is not a train record')
        r = train[rid]
        if r['subject'] != subject or r['name'] != entry['image_name'] or r['trial_key'] != entry['trial_key']:
            raise ValueError('frozen support cannot resolve to its declared trial/subject')
    for subject in seen:
        if not any(r['subject'] == subject for r in splits['train']):
            raise ValueError('seen subject has no train query')
    if set(manifest['support_draws']) != {'1', '5', '10'}:
        raise ValueError('expected canonical K_eval=1/5/10')
    for key in manifest['support_draws']:
        k = int(key)
        blocks = manifest['support_draws'][str(k)]
        if k not in (1, 5, 10) or len(blocks) != 10 or any(len(b) != k for b in blocks):
            raise ValueError('expected nonempty K-sized final support blocks')
        entries = [e for b in blocks for e in b]
        if len({e['image_name'] for e in entries}) != len(blocks) * k:
            raise ValueError('support images overlap across final draws')
        for e in entries:
            for u in sorted(unseen_subjects):
                check_entry(e, e['resolved_record_id_by_subject'][str(u)], u)
    # The master manifest is shared by all variants; the variant manifest owns
    # record checks and frozen final support blocks.
    identity = hashlib.sha256(''.join(sha256_file(p) for p in (manifest_path, master_path, index_path)).encode()).hexdigest()
    return splits, manifest, identity


class CocoSearch18Adapter:
    def __init__(self, images_root=IMAGES_ROOT, *, annotation_frame, duration_field="T"):
        """annotation_frame must explicitly declare original_image or {width,height}.

        Persisted records have no dimensions. A resized image must not silently
        redefine the annotation coordinate frame.
        """
        self.root = Path(images_root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.duration_field = duration_field
        if annotation_frame is None:
            raise ValueError('data.annotation_frame unresolved: declare original_image or verified width/height')
        self.annotation_frame = annotation_frame
        self.index = {}
        # data/images may hold both AiR and COCO.  Never select an AiR image
        # with the same basename as a COCO image.
        aliases = ('COCO_Search18', 'COCO-Search18', 'COCOSearch18', 'cocosearch18')
        scoped_roots = [self.root / name for name in aliases if (self.root / name).is_dir()]
        for base in scoped_roots or [self.root]:
            for path in base.rglob('*'):
                if path.is_file():
                    relative = path.relative_to(self.root).parts
                    if not scoped_roots and relative[0].lower() == 'air':
                        continue
                    self.index.setdefault(path.name, []).append(path)

    @lru_cache(maxsize=None)
    def image_info(self, name, condition):
        from PIL import Image
        matches = self.index.get(name, [])
        if len(matches) > 1:
            folder = 'tp' if condition == 'present' else 'ta'
            conditioned = [p for p in matches if folder in
                           (part.lower() for part in p.relative_to(self.root).parts[:-1])]
            if conditioned:
                matches = conditioned
        if len(matches) != 1:
            raise FileNotFoundError(f'{name}: expected one image under {self.root}, found {len(matches)}')
        with Image.open(matches[0]) as image:
            image_size = image.size
        frame = self.annotation_frame
        if frame == 'original_image':
            width, height = image_size
        elif isinstance(frame, dict) and set(frame) == {'width', 'height'}:
            width, height = frame['width'], frame['height']
        else:
            raise ValueError('unsupported annotation_frame declaration')
        return str(matches[0]), width, height

    def __call__(self, raw):
        rid = raw['record_id']
        image_path, width, height = self.image_info(raw['name'], raw['condition'])
        return NormalizedRecord(rid, raw['stimulus_id'], int(raw['subject']), image_path, width, height,
            raw['task'], raw['condition'], tuple(raw['X']), tuple(raw['Y']), tuple(raw[self.duration_field]),
            semantic_from_prediction(raw['prediction'], len(raw['X']), rid))
