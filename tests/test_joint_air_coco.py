"""Regression fixtures for the dataset-aware frozen few-shot protocol.

Run in repository venv: python -m pytest tests/test_joint_air_coco.py -q
No model weights/GPU required.
"""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from semgaze.data.schema import normalized_episode_from_dict
from semgaze.data.fewshot import TrainingEpisodeSampler, frozen_episode
from semgaze.data.joint import qualified_subject, qualified_stimulus, unseen_subject_ids


@pytest.fixture
def canonical():
    root = Path(__file__).resolve().parents[1]
    episode = normalized_episode_from_dict(json.loads(
        (root / 'tests/fixtures/flat_episode.json').read_text(encoding='utf-8')))
    return episode.query


def row(template, *, dataset, subject, image, rid, condition):
    return replace(template, dataset=dataset, subject=subject, stimulus_id=image,
                   record_id=rid, condition=condition)


def test_joint_subjects_never_collide():
    assert qualified_subject('AiR', '7') != qualified_subject('COCO-Search18', '7')
    assert qualified_stimulus('AiR', 'same.jpg') != qualified_stimulus('COCO-Search18', 'same.jpg')


def test_air_record_is_distinct_from_coco(canonical):
    air = row(canonical, dataset='AiR', subject='AiR::JY', image='AiR::a.jpg',
              rid='air_semgaze::JY::q1::a.jpg', condition='vqa')
    assert air.dataset == 'AiR'
    with pytest.raises(ValueError):
        replace(air, condition='present')
    with pytest.raises(ValueError):
        replace(canonical, dataset='AiR')


def test_sampler_joint_query_coverage_excludes_both_unseen(canonical):
    records = []
    for i in range(12):
        records.append(row(canonical, dataset='COCO-Search18', subject=0,
                           image=f'coco_{i}.jpg', rid=f'coco_seen_{i}', condition='present'))
        records.append(row(canonical, dataset='COCO-Search18', subject=7,
                           image=f'coco_unseen_{i}.jpg', rid=f'coco_unseen_{i}', condition='present'))
        records.append(row(canonical, dataset='AiR', subject='AiR::AB',
                           image=f'AiR::air_{i}.jpg', rid=f'air_seen_{i}', condition='vqa'))
        records.append(row(canonical, dataset='AiR', subject='AiR::JY',
                           image=f'AiR::air_unseen_{i}.jpg', rid=f'air_unseen_{i}', condition='vqa'))
    cfg = {'variant': 'all', 'dataset': 'all', 'unseen_subjects': [7, 8, 9],
           'air_unseen_subjects': ['JY', 'SC', 'YN'],
           'fewshot': {'k_values': [1, 5, 10],
                       'train_k_probabilities': [1/3, 1/3, 1/3],
                       'k_sampling_strategy': 'per_batch'}}
    sampler = TrainingEpisodeSampler(records, 123, data_config=cfg)
    assert sampler.query_count == 24
    assert {r.dataset for r in sampler.queries} == {'AiR', 'COCO-Search18'}
    for query in sampler.queries:
        episode = sampler.sample_for_query(query, k=10)
        assert all(s.dataset == query.dataset and s.subject == query.subject
                   for s in episode.supports)
        assert query.stimulus_id not in {s.stimulus_id for s in episode.supports}


def test_air_frozen_support_uses_qid_not_question_text(canonical):
    query = row(canonical, dataset='AiR', subject='AiR::JY', image='AiR::heldout.jpg',
                rid='air_semgaze::JY::question_12::heldout.jpg', condition='vqa')
    support = row(canonical, dataset='AiR', subject='AiR::JY', image='AiR::train.jpg',
                  rid='air_semgaze::JY::qid_1::train.jpg', condition='vqa')
    entry = {'dataset': 'AiR', 'unit_id': 'qid_1', 'image_name': 'train.jpg',
             'trial_key': 'qid_1::train.jpg',
             'resolved_record_id_by_subject': {'JY': support.record_id}}
    air_manifest = {'train_stimulus_ids': ['train.jpg'],
                    'test_stimulus_ids': ['heldout.jpg'],
                    'support_draws': {'1': [[entry]]}}
    manifest = {'datasets': {'AiR': air_manifest},
                'unseen_subject_ids': ['AiR::JY'],
                'train_stimulus_ids': ['AiR::train.jpg'],
                'test_stimulus_ids': ['AiR::heldout.jpg']}
    episode = frozen_episode(query, {support.record_id: support}, manifest, 1, draw_id=0)
    assert episode.supports == (support,)
    with pytest.raises(ValueError):
        frozen_episode(query, {}, manifest, 1, draw_id=0)
    with pytest.raises(ValueError):
        frozen_episode(query, {support.record_id: support}, manifest, 1, draw_id=1)


def test_joint_unseen_union_config():
    cfg = {'dataset': 'all', 'unseen_subjects': [7, 8, 9],
           'air_unseen_subjects': ['JY', 'SC', 'YN']}
    assert unseen_subject_ids(cfg) == {7, 8, 9, 'AiR::JY', 'AiR::SC', 'AiR::YN'}


def test_joint_reader_keeps_both_dataset_record_populations(monkeypatch, tmp_path):
    from semgaze.data import joint
    import hashlib

    (tmp_path / 'split_manifest.json').write_text('{}')
    (tmp_path / 'air_manifest.json').write_text('{"dataset":"AiR"}')
    manifest = {
        'dataset': 'all', 'resplit_after_merge': False,
        'source_manifests': {'AiR': str(tmp_path / 'air_manifest.json'),
                             'COCO-Search18': 'not_used_in_mock'},
        'record_counts': {'train': 4, 'test': 2},
        'per_dataset_record_counts': {'AiR': {'train': 2, 'test': 1},
                                      'COCO-Search18': {'train': 2, 'test': 1}},
        'support_sampling': {'per_dataset_support_draws': {'AiR': {'1': [], '5': [], '10': []}}},
    }
    def fake_coco(root, combo, cfg):
        assert cfg['dataset'] == 'COCO-Search18'
        return ({'train': [{'dataset': 'COCO-Search18', 'record_id': 'c1'},
                           {'dataset': 'COCO-Search18', 'record_id': 'c2'}],
                 'test': [{'dataset': 'COCO-Search18', 'record_id': 'ct'}]},
                {'unseen_subject_ids': [7, 8, 9], 'train_stimulus_ids': ['c.jpg'],
                 'test_stimulus_ids': ['ct.jpg'], 'support_draws': {'1': [], '5': [], '10': []}},
                'coco_identity')
    def fake_air(root, combo, cfg):
        return ({'train': [{'dataset': 'AiR', 'record_id': 'a1'},
                           {'dataset': 'AiR', 'record_id': 'a2'}],
                 'test': [{'dataset': 'AiR', 'record_id': 'at'}]},
                {'unseen_subject_ids': ['JY', 'SC', 'YN'],
                 'train_stimulus_ids': ['a.jpg'], 'test_stimulus_ids': ['at.jpg'],
                 'support_draws': {'1': [], '5': [], '10': []}},
                tmp_path / 'air_manifest.json')
    monkeypatch.setattr(joint, 'read_current_json_splits', fake_coco)
    monkeypatch.setattr(joint, '_verify_air', fake_air)
    raw, runtime, identity = joint.read_joint_splits(tmp_path, manifest, {'dataset': 'all'})
    assert {r['dataset'] for r in raw['train']} == {'AiR', 'COCO-Search18'}
    assert len(raw['train']) == 4
    assert len(raw['test']) == 2
    assert 'AiR::JY' in runtime['unseen_subject_ids']
    assert 'AiR::at.jpg' in runtime['test_stimulus_ids']
    assert len(identity) == 64
