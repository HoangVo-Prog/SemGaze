"""Persisted master split and full real-record coverage without image/model IO."""
from collections import Counter
from types import SimpleNamespace

import pytest

from semgaze.data.cocosearch18 import read_persisted_splits
from semgaze.data.fewshot import TrainingEpisodeSampler, frozen_episode
from semgaze.model.config import ROOT, resolve_config
from semgaze.training.loop import resolve_epoch_schedule


@pytest.fixture(scope='module')
def persisted():
    return {variant: read_persisted_splits(ROOT / 'data/COCO_Search18/split_95_5' / variant)
            for variant in ('all', 'tp_only', 'ta_only')}


def test_master_partition_and_frozen_draws(persisted):
    master, _, _ = persisted['all']
    assert {s: len(rows) for s, rows in master.items()} == {'train':26125, 'test':1375}
    image_sets = {s: {r['stimulus_id'] for r in rows} for s, rows in master.items()}
    assert [len(image_sets[s]) for s in ('train','test')] == [4048, 213]
    assert not image_sets['train'] & image_sets['test']
    for variant, (raw, manifest, identity) in persisted.items():
        assert set(raw) == {'train', 'test'} and len(identity) == 64
        assert 'validation_supports' not in manifest
        for split, rows in raw.items():
            assert {r['stimulus_id'] for r in rows} <= image_sets[split]
        train = {r['record_id']: SimpleNamespace(**r) for r in raw['train']}
        queries = [SimpleNamespace(**r) for r in raw['test'] if r['subject'] in {7,8,9}]
        for k in (1,5,10):
            blocks = manifest['support_draws'][str(k)]
            assert len(blocks) == 10
            assert len({e['image_name'] for block in blocks for e in block}) == 10*k
            for draw in range(10):
                for q in queries:
                    ep = frozen_episode(q, train, manifest, k, draw_id=draw)
                    assert len(ep.supports) == k
                    assert [s.stimulus_id for s in ep.supports] == [e['image_name'] for e in blocks[draw]]
                    assert {s.subject for s in ep.supports} == {q.subject}
                    assert all(s.split == 'train' for s in ep.supports)


def test_complete_real_query_universe_and_deterministic_resume(persisted):
    raw, _, _ = persisted['all']
    records = [SimpleNamespace(**r) for r in raw['train']]
    expected = {r.record_id for r in records if r.subject not in {7,8,9}}
    sampler = TrainingEpisodeSampler(records, 42)
    assert sampler.query_count == len(expected)
    first = [sampler.sample() for _ in range(101)]
    checkpoint = sampler.state_dict()
    continuation = [sampler.sample() for _ in range(13)]
    resumed = TrainingEpisodeSampler(records, 999)
    resumed.load_state_dict(checkpoint)
    assert [resumed.sample() for _ in range(13)] == continuation
    rest = [resumed.sample() for _ in range(resumed.remaining)]
    episodes = first + continuation + rest
    counts = Counter(e.query.record_id for e in episodes)
    assert set(counts) == expected and set(counts.values()) == {1}
    assert resumed.epoch_complete()
    assert {len(e.supports) for e in episodes} == set(range(1,11))
    assert all(e.query.split == 'train' for e in episodes)
    assert all(s.split == 'train' for e in episodes for s in e.supports)
    order = list(resumed.state_dict()['permutation'])
    resumed.start_epoch()
    assert resumed.state_dict()['permutation'] != order


@pytest.mark.parametrize('path', ['data/COCO_Search18/split/all',
                                  'data/COCO_Search18/split_95_5/all/validation.json'])
def test_obsolete_paths_fail_before_io(path):
    with pytest.raises(ValueError, match='obsolete'):
        read_persisted_splits(ROOT / path)


def test_schedule_tail_debug_cap_and_distributed_fail_closed(monkeypatch):
    cfg = resolve_config({'training': {'num_train_epochs':3,
        'per_device_train_batch_size':2, 'gradient_accumulation_steps':3}})
    assert resolve_epoch_schedule(cfg, 13) == 3
    assert cfg['training']['total_optimizer_updates'] == 9
    cfg['training']['max_steps'] = 4
    assert resolve_epoch_schedule(cfg, 13) == 3
    assert cfg['training']['total_optimizer_updates'] == 4
    monkeypatch.setenv('WORLD_SIZE','2')
    with pytest.raises(ValueError, match='distributed query coverage'):
        resolve_epoch_schedule(cfg, 13)
