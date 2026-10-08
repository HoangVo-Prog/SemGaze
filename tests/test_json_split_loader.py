"""Small self-contained tests for the new combined/dataset-specific JSON split."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from semgaze.data import json_splits
from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter
from semgaze.data.fewshot import frozen_episode
from semgaze.model.config import resolve_config


def _dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(subject, name, split):
    unseen = subject in (7, 8, 9)
    return {
        'record_id': f'coco_semgaze::{subject}::car::{name}',
        'dataset': 'COCO-Search18', 'variant': 'all', 'name': name,
        'stimulus_id': name, 'global_stimulus_id': f'COCO-Search18::{name}',
        'global_subject_id': f'COCO-Search18::{subject}', 'subject': subject,
        'task': 'car', 'trial_key': f'car::{name}', 'split': split,
        'subject_role': 'unseen' if unseen else 'seen',
        'train_query_eligible': split == 'train' and not unseen,
        'condition': 'present', 'X': [5], 'Y': [6], 'T': [100],
        'prediction': {'fixations': [{'fixation': 1, 'what': 'car'}],
                       'regions': [{'fixations': [1], 'why': 'search'}],
                       'how': 'inspect'},
    }


def _fixture(tmp_path, monkeypatch, *, combined=True):
    monkeypatch.setattr(json_splits, 'REPO_ROOT', tmp_path)
    relative_source = Path('data/split/COCO_Search18/COCOSearch-18.json')
    source_sha = _dump(tmp_path / relative_source, [])
    root = tmp_path / 'data/split/all' if combined else tmp_path / 'data/split/COCO_Search18/split/all'
    selected_root = tmp_path / 'data/split/COCO_Search18/split/all'
    images = [f'{n:012d}.jpg' for n in range(100)]
    heldout = 'test_image.jpg'
    coco_rows = {'train': [_record(s, name, 'train') for name in images for s in (1, 7, 8, 9)],
                 'test': [_record(7, heldout, 'test')],
                 'test_seen': [_record(1, heldout, 'test_seen')]}
    draws = {}
    for k in (1, 5, 10):
        draws[str(k)] = []
        for draw in range(10):
            entries = []
            for name in images[draw*k:(draw+1)*k]:
                entries.append({'dataset': 'COCO-Search18', 'unit_id': 'car',
                    'image_name': name, 'trial_key': f'car::{name}',
                    'resolved_record_id_by_subject': {
                        str(s): f'coco_semgaze::{s}::car::{name}' for s in (7, 8, 9)}})
            draws[str(k)].append(entries)
    selected_manifest = {
        'dataset': 'COCO-Search18', 'variant': 'all',
        'source_file': str(relative_source), 'source_sha256': source_sha,
        'all_source_records_materialized': True, 'discarded_records': 0,
        'seen_subject_ids': ['1'], 'unseen_subject_ids': ['7','8','9'],
        'train_stimulus_ids': images, 'test_stimulus_ids': [heldout],
        'support_draws': draws,
    }
    selected_manifest['split_file_sha256'] = {
        split: _dump(selected_root / f'{split}.json', rows) for split, rows in coco_rows.items()}
    _dump(selected_root / 'split_manifest.json', selected_manifest)
    if combined:
        merged = {}
        for split in ('train', 'test', 'test_seen'):
            air = {'dataset': 'AiR', 'record_id': f'air::{split}', 'name': 'air.jpg'}
            merged[split] = [air, *coco_rows[split]]
        merged_sha = {split: _dump(root / f'{split}.json', merged[split]) for split in merged}
        master = {'dataset': 'all', 'resplit_after_merge': False,
            'source_manifests': {'COCO-Search18': 'data/split/COCO_Search18/split/all/split_manifest.json'},
            'per_dataset_unseen_subject_ids': {'COCO-Search18': ['7','8','9']},
            'record_counts': {s: len(rows) for s, rows in merged.items()},
            'per_dataset_record_counts': {'COCO-Search18': {s: len(rows) for s, rows in coco_rows.items()}},
            'support_sampling': {'per_dataset_support_draws': {'COCO-Search18': draws}},
            'split_file_sha256': merged_sha}
        _dump(root / 'split_manifest.json', master)
    return root


def _config(root, tmp_path):
    return resolve_config({'data': {'dataset': 'COCO-Search18', 'variant': 'all',
        'split_root': str(root), 'images_root': str(tmp_path/'images')}})['data']


@pytest.mark.parametrize('combined', [True, False])
def test_json_loader_keeps_coco_only_and_reuses_persisted_supports(tmp_path, monkeypatch, combined):
    root = _fixture(tmp_path, monkeypatch, combined=combined)
    rows, manifest, identity = read_persisted_splits(root, data_config=_config(root, tmp_path))
    assert len(rows['train']) == 400
    assert len(rows['test']) == 1
    assert all(r['dataset'] == 'COCO-Search18' for subset in rows.values() for r in subset)
    assert manifest['unseen_subject_ids'] == [7,8,9]
    assert len(identity) == 64
    train = {r['record_id']: SimpleNamespace(**r) for r in rows['train']}
    query = SimpleNamespace(**rows['test'][0])
    for k in (1, 5, 10):
        ep = frozen_episode(query, train, manifest, k, draw_id=0)
        assert len(ep.supports) == k
        assert all(s.subject == query.subject for s in ep.supports)


def test_json_loader_rejects_tampered_split(tmp_path, monkeypatch):
    root = _fixture(tmp_path, monkeypatch)
    with (root / 'test.json').open('a', encoding='utf-8') as output:
        output.write(' ')
    with pytest.raises(ValueError, match='checksum'):
        read_persisted_splits(root, data_config=_config(root, tmp_path))


def test_coco_image_lookup_scopes_and_casts_subject(tmp_path):
    Image = pytest.importorskip('PIL.Image')
    images = tmp_path / 'images'
    (images/'AiR').mkdir(parents=True)
    (images/'COCO_Search18'/'tp').mkdir(parents=True)
    (images/'COCO_Search18'/'ta').mkdir(parents=True)
    for target in (images/'AiR'/'same.jpg', images/'COCO_Search18'/'tp'/'same.jpg',
                   images/'COCO_Search18'/'ta'/'same.jpg'):
        Image.new('RGB', (64, 32)).save(target)
    adapter = CocoSearch18Adapter(images, annotation_frame='original_image')
    raw = _record(7, 'same.jpg', 'test')
    raw['subject'] = '7'
    record = adapter(raw)
    assert record.subject == 7
    assert Path(record.image_path).parts[-3:] == ('COCO_Search18','tp','same.jpg')
    assert (record.image_width, record.image_height) == (64, 32)
