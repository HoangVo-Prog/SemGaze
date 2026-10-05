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
    manifest_path = root / 'split_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    unseen_subjects = set(manifest['unseen_subject_ids'])
    variant = manifest['variant']
    duration_field = data_config['duration']['source_field'] if data_config else 'T'
    if data_config and (variant != data_config['variant'] or unseen_subjects != set(data_config['unseen_subjects'])):
        raise ValueError('unexpected variant/unseen subjects in manifest')
    source = REPO_ROOT / manifest['curated_source_file']
    if sha256_file(source) != manifest['curated_source_sha256']:
        raise ValueError('curated-source checksum differs from manifest')
    splits, ids, images = {}, set(), set()
    for split in ('train', 'validation', 'test'):
        path = root / f'{split}.json'
        if sha256_file(path) != manifest['split_file_sha256'][split]:
            raise ValueError(f'{split}: split checksum differs from manifest')
        records = json.loads(path.read_text(encoding='utf-8'))
        split_images = {r['stimulus_id'] for r in records}
        if split_images != set(manifest[f'{split}_stimulus_ids']) or images & split_images:
            raise ValueError(f'{split}: stimulus membership/leakage mismatch')
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
    seen = {r['subject'] for rows in splits.values() for r in rows} - unseen_subjects
    if seen != set(manifest['seen_subject_ids']):
        raise ValueError('seen-subject membership mismatch')
    def check_entry(entry, rid, subject):
        r = train[rid]
        if r['subject'] != subject or r['name'] != entry['image_name'] or r['trial_key'] != entry['trial_key']:
            raise ValueError('frozen support cannot resolve to its declared trial/subject')
    for subject in seen:
        if not any(r['subject'] == subject for r in splits['validation']):
            raise ValueError('seen subject has no validation query')
        for key in manifest['validation_supports'][str(subject)]:
            k = int(key)
            block = manifest['validation_supports'][str(subject)][str(k)]
            if k < 1 or len(block) != k or len({e['image_name'] for e in block}) != k:
                raise ValueError('invalid validation support block')
            for e in block:
                check_entry(e, e['record_id'], subject)
    for key in manifest['support_draws']:
        k = int(key)
        blocks = manifest['support_draws'][str(k)]
        if k < 1 or not blocks or any(len(b) != k for b in blocks):
            raise ValueError('expected nonempty K-sized final support blocks')
        entries = [e for b in blocks for e in b]
        if len({e['image_name'] for e in entries}) != len(blocks) * k:
            raise ValueError('support images overlap across final draws')
        for e in entries:
            for u in sorted(unseen_subjects):
                check_entry(e, e['resolved_record_id_by_subject'][str(u)], u)
    return splits, manifest, sha256_file(manifest_path)


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
