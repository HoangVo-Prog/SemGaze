"""Dataset-aware adapter/reader for AiR + COCO-Search18 persisted splits.

The source image splits and support draws are *not* re-generated.  The COCO
loader is reused unchanged, and AiR is verified against its own source
manifest before combining normalized episode populations.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
import hashlib
import json

from .json_splits import read_current_json_splits, _repo_path, _read_json, _sha256
from .cocosearch18 import CocoSearch18Adapter
from .schema import NormalizedRecord
from .semantic import semantic_from_prediction

AIR = 'AiR'
COCO = 'COCO-Search18'
PREFIX = 'AiR::'


def qualified_subject(dataset, subject):
    return f'{PREFIX}{subject}' if dataset == AIR else int(subject)


def qualified_stimulus(dataset, name):
    return f'{PREFIX}{name}' if dataset == AIR else name


def unseen_subject_ids(config, manifest=None):
    if manifest is not None and 'datasets' in manifest:
        return frozenset(manifest['unseen_subject_ids'])
    subjects = set(config['unseen_subjects'])
    if config['dataset'] in ('all', AIR):
        subjects.update(qualified_subject(AIR, s) for s in config['air_unseen_subjects'])
    return frozenset(subjects)


def _assert_unique(values, label):
    if len(values) != len(set(values)):
        raise ValueError(f'{label}: duplicate identities')


def _verify_air(root, combined, cfg):
    """Validate all AiR split rows, support mappings and persisted provenance."""
    source_path = _repo_path(combined['source_manifests'][AIR]).resolve()
    source = _read_json(source_path)
    if source.get('dataset') != AIR or source.get('variant') != 'all':
        raise ValueError('AiR source manifest mismatch')
    configured_unseen = set(map(str, cfg['air_unseen_subjects']))
    if set(map(str, source['unseen_subject_ids'])) != configured_unseen:
        raise ValueError('data.air_unseen_subjects differs from persisted AiR protocol')
    if source.get('discarded_records') != 0 or not source.get('all_source_records_materialized'):
        raise ValueError('AiR source manifest is incomplete')
    if _sha256(_repo_path(source['source_file'])) != source['source_sha256']:
        raise ValueError('AiR source annotation checksum mismatch')
    train_images = set(source['train_stimulus_ids'])
    test_images = set(source['test_stimulus_ids'])
    if train_images & test_images:
        raise ValueError('AiR image leakage')
    rows = {}
    ids = set()
    seen = set(map(str, source['seen_subject_ids']))
    for split in ('train', 'test', 'test_seen'):
        file = root / f'{split}.json'
        if _sha256(file) != combined['split_file_sha256'][split]:
            raise ValueError(f'combined {split}.json checksum mismatch')
        all_rows = _read_json(file)
        if len(all_rows) != combined['record_counts'][split]:
            raise ValueError(f'combined {split} row count mismatch')
        selected = [r for r in all_rows if r.get('dataset') == AIR]
        if len(selected) != combined['per_dataset_record_counts'][AIR][split]:
            raise ValueError(f'AiR {split} row count mismatch')
        stimuli = set()
        for row in selected:
            rid, name, subject = row['record_id'], row['name'], str(row['subject'])
            qid = str(row['qid'])
            expected_role = 'unseen' if subject in configured_unseen else 'seen'
            if (rid in ids or rid != f'air_semgaze::{subject}::{qid}::{name}'
                    or row.get('variant') != 'all' or row.get('split') != split
                    or row.get('stimulus_id') != name
                    or row.get('global_stimulus_id') != f'AiR::{name}'
                    or row.get('global_subject_id') != f'AiR::{subject}'
                    or row.get('trial_key') != f'{qid}::{name}'
                    or row.get('condition') != 'vqa'
                    or row.get('subject_role') != expected_role
                    or row.get('train_query_eligible') is not (split == 'train' and expected_role == 'seen')):
                raise ValueError(f'{rid}: invalid AiR persisted record')
            if ((split == 'train' and subject not in (configured_unseen | seen))
                    or (split == 'test' and subject not in configured_unseen)
                    or (split == 'test_seen' and subject not in seen)):
                raise ValueError(f'{rid}: invalid AiR subject split')
            if not (len(row['X']) == len(row['Y']) == len(row[cfg['duration']['source_field']]) > 0):
                raise ValueError(f'{rid}: invalid fixation lengths')
            semantic_from_prediction(row['prediction'], len(row['X']), rid)
            ids.add(rid)
            stimuli.add(name)
        if split == 'train' and stimuli != train_images:
            raise ValueError('AiR train stimuli differ from source manifest')
        if split == 'test' and not stimuli <= test_images:
            raise ValueError('AiR test stimulus membership mismatch')
        if split == 'test_seen' and not stimuli <= test_images:
            raise ValueError('AiR test_seen stimulus membership mismatch')
        rows[split] = selected
    if ({r['name'] for r in rows['test']} | {r['name'] for r in rows['test_seen']}) != test_images:
        raise ValueError('AiR held-out stimulus coverage mismatch')
    train_by_id = {r['record_id']: r for r in rows['train']}
    for k in (1, 5, 10):
        blocks = source['support_draws'][str(k)]
        if len(blocks) != 10 or any(len(block) != k for block in blocks):
            raise ValueError('AiR frozen support draw layout mismatch')
        all_images = []
        for block in blocks:
            for entry in block:
                name = entry['image_name']
                all_images.append(name)
                if entry['dataset'] != AIR or name not in train_images or entry['trial_key'] != f"{entry['unit_id']}::{name}":
                    raise ValueError('AiR frozen support unit mismatch')
                mapping = entry['resolved_record_id_by_subject']
                if set(mapping) != configured_unseen:
                    raise ValueError('AiR frozen support subject map mismatch')
                for subject, rid in mapping.items():
                    record = train_by_id.get(rid)
                    if record is None or str(record['subject']) != subject or str(record['qid']) != str(entry['unit_id']) or record['name'] != name:
                        raise ValueError('AiR frozen support record mismatch')
        _assert_unique(all_images, f'AiR K={k} exclusive support images')
    return rows, source, source_path


def read_joint_splits(root, manifest, config):
    """Return *all* AiR and COCO train/test rows and a dataset-aware manifest."""
    root = Path(root).resolve()
    if manifest.get('dataset') != 'all' or manifest.get('resplit_after_merge') is not False:
        raise ValueError('joint training requires the persisted AiR + COCO union')
    if set(manifest.get('source_manifests', {})) != {AIR, COCO}:
        raise ValueError('combined split must contain exactly AiR and COCO-Search18')
    coco_config = dict(config, dataset=COCO)
    coco_rows, coco_manifest, coco_identity = read_current_json_splits(root, manifest, coco_config)
    air_rows, air_manifest, air_path = _verify_air(root, manifest, config)
    for split in ('train', 'test'):
        expected = manifest['record_counts'][split]
        if len(coco_rows[split]) + len(air_rows[split]) != expected:
            raise ValueError(f'{split}: joint loader dropped persisted records')
    raw = {
        split: sorted(coco_rows[split] + air_rows[split], key=lambda row: (row['dataset'], row['record_id']))
        for split in ('train', 'test')
    }
    subjects = set(coco_manifest['unseen_subject_ids']) | {
        qualified_subject(AIR, s) for s in air_manifest['unseen_subject_ids']
    }
    train_images = set(coco_manifest['train_stimulus_ids']) | {
        qualified_stimulus(AIR, s) for s in air_manifest['train_stimulus_ids']
    }
    test_images = set(coco_manifest['test_stimulus_ids']) | {
        qualified_stimulus(AIR, s) for s in air_manifest['test_stimulus_ids']
    }
    if train_images & test_images:
        raise ValueError('joint train/test stimulus leakage')
    runtime = {
        'dataset': 'all', 'variant': 'all',
        'datasets': {AIR: air_manifest, COCO: coco_manifest},
        'unseen_subject_ids': sorted(subjects, key=str),
        'train_stimulus_ids': sorted(train_images),
        'test_stimulus_ids': sorted(test_images),
        'support_draws': coco_manifest['support_draws'],
        'per_dataset_record_counts': manifest['per_dataset_record_counts'],
    }
    if manifest['support_sampling']['per_dataset_support_draws'][AIR] != air_manifest['support_draws']:
        raise ValueError('combined AiR support draws differ from source manifest')
    # COCO comparison is already performed by read_current_json_splits.
    digest = hashlib.sha256(('joint-v1|' + coco_identity + '|' + _sha256(air_path)
                             + '|' + _sha256(root / 'split_manifest.json')).encode()).hexdigest()
    return raw, runtime, digest


class AirAdapter:
    """Resolve images only within AiR and preserve raw gaze pixels/ms."""

    def __init__(self, images_root, *, annotation_frame='original_image', duration_field='T'):
        root = Path(images_root).resolve()
        candidates = [root / name for name in ('AiR', 'AIR', 'air')]
        found = [p for p in candidates if p.is_dir()]
        if len(found) != 1:
            raise FileNotFoundError(f'expected exactly one AiR image directory below {root}; got {found}')
        self.root = found[0]
        self.annotation_frame = annotation_frame
        self.duration_field = duration_field
        self.index = {}
        for path in self.root.rglob('*'):
            if path.is_file():
                self.index.setdefault(path.name, []).append(path)

    @lru_cache(maxsize=None)
    def image_info(self, name):
        from PIL import Image
        choices = self.index.get(name, [])
        if len(choices) != 1:
            raise FileNotFoundError(f'AiR image {name}: expected one file, found {len(choices)}')
        with Image.open(choices[0]) as img:
            size = img.size
        frame = self.annotation_frame
        if frame == 'original_image':
            width, height = size
        elif isinstance(frame, dict) and set(frame) == {'width', 'height'}:
            width, height = frame['width'], frame['height']
        else:
            raise ValueError('invalid AiR coordinate frame')
        return str(choices[0]), width, height

    def __call__(self, row):
        rid = row['record_id']
        path, width, height = self.image_info(row['name'])
        return NormalizedRecord(
            record_id=rid, stimulus_id=qualified_stimulus(AIR, row['name']),
            subject=qualified_subject(AIR, row['subject']), image_path=path,
            image_width=width, image_height=height, task=row['task'],
            condition='vqa', x_px=tuple(row['X']), y_px=tuple(row['Y']),
            duration_ms=tuple(row[self.duration_field]),
            semantic=semantic_from_prediction(row['prediction'], len(row['X']), rid),
            dataset=AIR,
        )


class JointAdapter:
    def __init__(self, images_root, *, annotation_frame, duration_field='T'):
        self.coco = CocoSearch18Adapter(images_root, annotation_frame=annotation_frame,
                                        duration_field=duration_field)
        self.air = AirAdapter(images_root, annotation_frame=annotation_frame,
                              duration_field=duration_field)

    def __call__(self, row):
        if row.get('dataset') == AIR:
            return self.air(row)
        if row.get('dataset') == COCO:
            return self.coco(row)
        raise ValueError(f'unsupported joint dataset: {row.get("dataset")}')
