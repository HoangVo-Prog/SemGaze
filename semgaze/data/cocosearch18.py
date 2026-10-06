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


def read_persisted_splits(split_root=SPLIT_ROOT, *, data_config=None):
    root = Path(split_root).resolve()
    if root.name == 'validation.json' or root.parent.name != 'split_95_5' or root.name not in ('all', 'tp_only', 'ta_only'):
        raise ValueError('obsolete COCO-Search18 split root; use data/COCO_Search18/split_95_5/<variant>')
    if not root.is_dir():
        raise FileNotFoundError(root)
    manifest_path = root / 'split_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    protocol = 'cocosearch18_semgaze_master_955_v1'
    master_path = root.parent / 'master_split_manifest.json'
    index_path = root.parent / 'split_index.json'
    master = json.loads(master_path.read_text(encoding='utf-8'))
    index = json.loads(index_path.read_text(encoding='utf-8'))
    if any(m.get('protocol_version') != protocol for m in (manifest, master, index)):
        raise ValueError('obsolete COCO-Search18 protocol; regenerate split_95_5')
    if set(manifest['split_files']) != {'train', 'test'} or set(manifest['split_file_sha256']) != {'train', 'test'}:
        raise ValueError('COCO-Search18 must expose train/test only')
    master_train, master_test = set(master['train_stimulus_ids']), set(master['test_stimulus_ids'])
    if master_train & master_test:
        raise ValueError('master train/test stimulus leakage')
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
        if split_images != set(manifest[f'{split}_stimulus_ids']) or images & split_images:
            raise ValueError(f'{split}: stimulus membership/leakage mismatch')
        master_images = master_train if split == 'train' else master_test
        if not split_images <= master_images or (variant == 'all' and split_images != master_images):
            raise ValueError(f'{split}: variant differs from shared master partition')
        if len(split_images) != entry['image_counts'][split] or len(records) != entry['record_counts'][split]:
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
        for path in self.root.rglob('*'):
            if path.is_file():
                self.index.setdefault(path.name, []).append(path)

    @lru_cache(maxsize=None)
    def image_info(self, name, condition):
        from PIL import Image
        matches = self.index.get(name, [])
        if len(matches) > 1:
            folder = 'tp' if condition == 'present' else 'ta'
            matches = [p for p in matches if p.relative_to(self.root).parts[0] == folder]
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
        return NormalizedRecord(rid, raw['stimulus_id'], raw['subject'], image_path, width, height,
            raw['task'], raw['condition'], tuple(raw['X']), tuple(raw['Y']), tuple(raw[self.duration_field]),
            semantic_from_prediction(raw['prediction'], len(raw['X']), rid))
