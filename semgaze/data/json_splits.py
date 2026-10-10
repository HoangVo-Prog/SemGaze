"""Read the image-disjoint SemGaze JSON splits produced by prepare_split.py.

The combined `data/split/all` directory contains both AiR and COCO-Search18.
Selection happens before any model-facing conversion; the dataset-specific
manifest owns the frozen test support draws.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .semantic import semantic_from_prediction

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ('train', 'test', 'test_seen')
COCO = 'COCO-Search18'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def _ids(payload: dict, key: str) -> set[str]:
    values = payload.get(key)
    if (not isinstance(values, list) or
            any(not isinstance(v, str) or not v for v in values) or
            len(values) != len(set(values))):
        raise ValueError(f'manifest.{key} must contain unique nonempty strings')
    return set(values)


def _subjects(values, label: str) -> set[int]:
    if not isinstance(values, list):
        raise ValueError(f'{label} must be a list')
    result = set()
    for value in values:
        if type(value) is int:
            number = value
        elif isinstance(value, str) and value.isdecimal():
            number = int(value)
        else:
            raise ValueError(f'{label} has non-numeric COCO subject {value!r}')
        if number in result:
            raise ValueError(f'{label} contains duplicate COCO subject {number}')
        result.add(number)
    return result


def read_current_json_splits(root: Path, manifest: dict, data_config: dict | None = None):
    """Return dataset-filtered train/test rows, frozen manifest, and split identity.

    Verify the combined files before filtering, and enforce the selected source
    manifest's train/held-out image partition and frozen supports. `test_seen`
    is read and validated but intentionally not part of the unseen test return.
    """
    root = Path(root).resolve()
    from semgaze.model.config import default_section
    cfg = data_config if data_config is not None else default_section('data')
    dataset, variant = cfg['dataset'], cfg['variant']
    if dataset != COCO:
        raise ValueError(f'JSON split adapter currently supports {COCO}, not {dataset!r}')
    configured_unseen = _subjects(cfg['unseen_subjects'], 'data.unseen_subjects')
    if configured_unseen != {7, 8, 9}:
        raise ValueError('COCO unseen subjects must remain {7,8,9}')

    is_combined = manifest.get('dataset') == 'all'
    if is_combined:
        if manifest.get('resplit_after_merge') is not False:
            raise ValueError('combined split must preserve source dataset partitions')
        source_ref = manifest.get('source_manifests', {}).get(dataset)
        if not source_ref:
            raise ValueError(f'combined manifest has no source manifest for {dataset}')
        source_manifest_path = _repo_path(source_ref).resolve()
        source = _read_json(source_manifest_path)
        if variant != 'all':
            raise ValueError('combined data/split/all contains only variant=all')
        if _subjects(manifest['per_dataset_unseen_subject_ids'][dataset],
                     'combined per_dataset_unseen_subject_ids') != configured_unseen:
            raise ValueError('combined unseen subjects differ from config')
        declared_counts = manifest['per_dataset_record_counts'][dataset]
        if (manifest.get('record_counts') is None or
                manifest.get('support_sampling', {}).get('per_dataset_support_draws', {}).get(dataset)
                != source.get('support_draws')):
            raise ValueError('combined dataset support manifest/counts mismatch')
    else:
        source_manifest_path = root / 'split_manifest.json'
        source = manifest
        declared_counts = None
        if manifest.get('dataset') != dataset:
            raise ValueError(f'split dataset {manifest.get("dataset")!r} != {dataset!r}')

    if source.get('dataset') != dataset or source.get('variant') != variant:
        raise ValueError('source manifest dataset/variant differs from data config')
    if (not source.get('all_source_records_materialized') or
            source.get('discarded_records') != 0):
        raise ValueError('source split manifest does not preserve all source records')
    unseen = _subjects(source['unseen_subject_ids'], 'source.unseen_subject_ids')
    seen = _subjects(source['seen_subject_ids'], 'source.seen_subject_ids')
    if unseen != configured_unseen or unseen & seen:
        raise ValueError('source subject partition differs from config')
    train_ids = _ids(source, 'train_stimulus_ids')
    heldout_ids = _ids(source, 'test_stimulus_ids')
    if train_ids & heldout_ids:
        raise ValueError('train/held-out image overlap in source manifest')
    source_file = _repo_path(source['source_file'])
    
    rows_by_split = {}
    observed_ids, images_by_split = set(), {}
    for split in SPLITS:
        path = root / f'{split}.json'
        if _sha256(path) != manifest['split_file_sha256'][split]:
            raise ValueError(f'{split}.json checksum differs from split manifest')
        rows = _read_json(path)
        if not isinstance(rows, list):
            raise ValueError(f'{path} must contain a JSON list')
        if is_combined:
            if len(rows) != manifest['record_counts'][split]:
                raise ValueError(f'{split}: combined record count mismatch')
            selected = [r for r in rows if r.get('dataset') == dataset]
        else:
            selected = rows
        if declared_counts is not None and len(selected) != declared_counts[split]:
            raise ValueError(f'{split}: dataset-specific count differs from combined manifest')

        stimuli = set()
        for row in selected:
            rid = row['record_id']
            name = row['name']
            task = row['task']
            subject_raw = row['subject']
            if (type(subject_raw) is not int and
                    not (isinstance(subject_raw, str) and subject_raw.isdecimal())):
                raise ValueError(f'{rid}: COCO subject must be an integer')
            subject = int(subject_raw)
            if (rid in observed_ids or row.get('dataset') != dataset or
                    row.get('variant') != variant or row.get('split') != split or
                    row.get('stimulus_id') != name or
                    row.get('global_stimulus_id') != f'{dataset}::{name}' or
                    row.get('global_subject_id') != f'{dataset}::{subject_raw}' or
                    rid != f'coco_semgaze::{subject_raw}::{task}::{name}' or
                    row.get('trial_key') != f'{task}::{name}'):
                raise ValueError(f'{rid}: duplicate or malformed persisted record identity')
            observed_ids.add(rid)
            expected_role = 'unseen' if subject in unseen else 'seen'
            if (row.get('subject_role') != expected_role or
                    row.get('train_query_eligible') is not (split == 'train' and subject in seen)):
                raise ValueError(f'{rid}: inconsistent subject role or training eligibility')
            if (split == 'test' and subject not in unseen or
                    split == 'test_seen' and subject not in seen or
                    split == 'train' and subject not in (seen | unseen)):
                raise ValueError(f'{rid}: subject appears in the wrong split')
            if not (len(row['X']) == len(row['Y']) == len(row[cfg['duration']['source_field']]) > 0):
                raise ValueError(f'{rid}: misaligned fixation fields')
            semantic_from_prediction(row['prediction'], len(row['X']), rid)
            stimuli.add(name)
        rows_by_split[split] = selected
        images_by_split[split] = stimuli

    if images_by_split['train'] != train_ids:
        raise ValueError('persisted train images differ from source manifest')
    if (images_by_split['test'] | images_by_split['test_seen']) != heldout_ids:
        raise ValueError('persisted held-out images differ from source manifest')
    if images_by_split['train'] & heldout_ids:
        raise ValueError('train/held-out image leakage')

    train_by_id = {r['record_id']: r for r in rows_by_split['train']}
    blocks_by_k = source['support_draws']
    if set(blocks_by_k) != {'1', '5', '10'}:
        raise ValueError('frozen support draws must cover K=1,5,10')
    for k in (1, 5, 10):
        blocks = blocks_by_k[str(k)]
        if len(blocks) != 10 or any(len(block) != k for block in blocks):
            raise ValueError(f'K={k} must contain ten frozen draws of size K')
        names = [entry['image_name'] for block in blocks for entry in block]
        if len(names) != len(set(names)) or not set(names) <= train_ids:
            raise ValueError(f'K={k} frozen supports are duplicated or outside train')
        for block in blocks:
            for entry in block:
                if (entry.get('dataset') != dataset or
                        entry.get('trial_key') != f"{entry.get('unit_id')}::{entry['image_name']}"):
                    raise ValueError('frozen support entry has inconsistent dataset or trial')
                mapping = entry['resolved_record_id_by_subject']
                if set(mapping) != {str(s) for s in unseen}:
                    raise ValueError('frozen support subject mapping mismatch')
                for s in unseen:
                    rid = mapping[str(s)]
                    record = train_by_id.get(rid)
                    if (record is None or int(record['subject']) != s or
                            record['name'] != entry['image_name'] or
                            record['task'] != entry['unit_id']):
                        raise ValueError(f'frozen support {rid} does not resolve to train')

    # The original JSON encodes COCO subject IDs as strings in its manifest.
    # The downstream episode API uses integer COCO subject IDs.
    runtime_manifest = dict(source)
    runtime_manifest['unseen_subject_ids'] = sorted(unseen)
    runtime_manifest['seen_subject_ids'] = sorted(seen)
    provenance = [root / 'split_manifest.json', source_manifest_path]
    provenance += [root / f'{split}.json' for split in SPLITS]
    identity = hashlib.sha256(''.join(_sha256(p) for p in provenance).encode()).hexdigest()
    return {k: rows_by_split[k] for k in ('train', 'test')}, runtime_manifest, identity
