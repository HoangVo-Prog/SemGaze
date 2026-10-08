"""Sampling contracts; collation doubles avoid loading a model or image files."""
from collections import Counter
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from semgaze.data.fewshot import TrainingEpisodeSampler
from semgaze.data.schema import FlatEpisode, normalized_episode_from_dict
from semgaze.model.config import ROOT, default_section
from semgaze.model.visual_cache import InferenceVisualCache
from semgaze.training import batching


@pytest.fixture
def sampler_factory():
    episode = normalized_episode_from_dict(json.loads((ROOT / 'tests/fixtures/flat_episode.json').read_text()))
    records = [replace(episode.query, subject=u, stimulus_id=f'{u}:{i}', record_id=f'{u}:{i}:{r}')
               for u in (1, 2) for i in range(12) for r in range(2)]
    data = default_section('data')
    return lambda seed=42: TrainingEpisodeSampler(records, seed, data_config=data)


@pytest.fixture
def bundle(monkeypatch):
    def collate(processor, episode, *args, image_cache, **kwargs):
        length = 100 + int(episode.query.stimulus_id.split(':')[1]) * 100
        image_cache[episode.query.record_id] = True
        return SimpleNamespace(inputs={'input_ids': SimpleNamespace(shape=(1, length))})
    monkeypatch.setattr(batching, 'collate_where', collate)
    monkeypatch.setattr(batching, 'pack_where_batch', lambda processor, episodes, samples, cache:
                        SimpleNamespace(episodes=episodes, samples=samples, image_cache=cache))
    return SimpleNamespace(processor=None, end_fix_id=0, context_limit=8192,
        config={'training': dict(per_device_train_batch_size=2, gradient_accumulation_steps=3,
                                 length_aware_batching=True, max_episode_retries=10)})


def test_coverage_and_reshuffle(sampler_factory):
    sampler = sampler_factory()
    orders = []
    for epoch in range(1, 4):
        sampler.start_epoch()
        episodes = [sampler.sample() for _ in range(sampler.query_count)]
        ids = [e.query.record_id for e in episodes]
        assert len(ids) == len(set(ids)) == sampler.query_count
        assert set(ids) == set(sampler.query_ids)
        assert sampler.epoch == epoch and sampler.epoch_complete()
        assert all(1 <= len(e.supports) <= 10 for e in episodes)
        orders.append(ids)
        with pytest.raises(StopIteration):
            sampler.sample()
    assert orders[0] != orders[1] != orders[2]
    with pytest.raises(ValueError, match='forced K'):
        sampler.sample(k=5)


@pytest.mark.parametrize('size', [1, 2, 4, 7])
@pytest.mark.parametrize('length_sort', [False, True])
def test_same_k_preserves_membership_tail_and_resume(bundle, sampler_factory, size, length_sort):
    bundle.config['training'].update(per_device_train_batch_size=size, length_aware_batching=length_sort)
    actual, reference = sampler_factory(), sampler_factory()
    covered = []
    while not actual.epoch_complete():
        n = min(size * 3, actual.remaining)
        expected = [reference.sample() for _ in range(n)]
        batches, original, rejected = batching.sample_optimizer_batches(bundle, actual)
        assert rejected == 0 and original == expected
        packed = [e for b in batches for e in b.episodes]
        assert sorted(packed, key=repr) == sorted(expected, key=repr)
        assert all(0 < len(b.episodes) <= size and len({len(e.supports) for e in b.episodes}) == 1 for b in batches)
        assert actual.state_dict() == reference.state_dict()
        covered.extend(e.query.record_id for e in original)
        resumed = sampler_factory(seed=999)
        resumed.load_state_dict(actual.state_dict())
        actual = resumed
    assert len(covered) == len(set(covered)) == actual.query_count


def test_k_is_uniform_per_query(sampler_factory):
    sampler = sampler_factory()
    counts = Counter()
    for _ in range(100):
        sampler.start_epoch()
        counts.update(len(sampler.sample().supports) for _ in range(sampler.query_count))
    assert set(counts) == set(range(1, 11))
    for n in counts.values():
        assert n / sum(counts.values()) == pytest.approx(0.1, abs=0.02)


def test_preprocessing_cache_toggle_does_not_change_episode_sampling(bundle, sampler_factory, monkeypatch):
    import torch
    mock_collate = batching.collate_where

    def collate_with_pixel_row(*args, **kwargs):
        native = mock_collate(*args, **kwargs)
        native.inputs['pixel_values'] = torch.zeros((1, 3, 448, 448))
        return native

    monkeypatch.setattr(batching, 'collate_where', collate_with_pixel_row)
    cached_sampler, baseline_sampler = sampler_factory(), sampler_factory()
    bundle.config['training'].update(cache_preprocessed_images=True, preprocessing_cache_max_entries=3)
    cached_batches, cached_episodes, cached_rejections = batching.sample_optimizer_batches(bundle, cached_sampler)
    cache = cached_batches[0].image_cache
    assert isinstance(cache, InferenceVisualCache)
    assert cache.max_entries == 3
    assert len(cache) <= 3
    assert len(cache.pixels) <= 3

    bundle.config['training']['cache_preprocessed_images'] = False
    baseline_batches, baseline_episodes, baseline_rejections = batching.sample_optimizer_batches(bundle, baseline_sampler)
    assert not isinstance(baseline_batches[0].image_cache, InferenceVisualCache)
    assert cached_episodes == baseline_episodes
    assert cached_rejections == baseline_rejections == 0
    assert cached_sampler.state_dict() == baseline_sampler.state_dict()
    cache.close()


def test_overflow_retains_query_and_k_and_rolls_back_on_failure(bundle, sampler_factory, monkeypatch):
    sampler, reference = sampler_factory(), sampler_factory()
    first = reference.sample()
    retry = reference.sample_for_query(first.query, k=len(first.supports))
    expected = [retry, reference.sample()]
    bundle.config['training']['gradient_accumulation_steps'] = 1
    original_collate = batching.collate_where
    calls = []
    def collate(processor, episode, *args, **kwargs):
        calls.append(episode)
        if len(calls) == 1:
            raise batching.WhereContextOverflowError('overflow')
        return original_collate(processor, episode, *args, **kwargs)
    monkeypatch.setattr(batching, 'collate_where', collate)
    _, accepted, retries = batching.sample_optimizer_batches(bundle, sampler)
    assert retries == 1 and accepted == expected
    assert calls[0].query == calls[1].query and len(calls[0].supports) == len(calls[1].supports)
    assert sampler.state_dict() == reference.state_dict()
    before = sampler.state_dict()
    calls.clear()
    def overflow(processor, episode, *args, **kwargs):
        calls.append(episode)
        raise batching.WhereContextOverflowError('overflow')
    monkeypatch.setattr(batching, 'collate_where', overflow)
    with pytest.raises(RuntimeError, match='K_train='):
        batching.sample_optimizer_batches(bundle, sampler)
    assert len(calls) == 10
    assert len({(e.query.record_id, len(e.supports)) for e in calls}) == 1
    assert sampler.state_dict() == before


def test_resume_rejects_old_and_corrupt_state(sampler_factory):
    sampler = sampler_factory()
    sampler.sample()
    state = sampler.state_dict()
    with pytest.raises(RuntimeError, match='partially consumed'):
        sampler.start_epoch()
    with pytest.raises(ValueError, match='query-coverage'):
        sampler.load_state_dict(state['rng'])
    state['permutation'][0] = state['permutation'][1]
    with pytest.raises(ValueError, match='permutation'):
        sampler.load_state_dict(state)


def test_benchmark_replays_identical_work_across_sizes(bundle, sampler_factory):
    workloads, states = [], []
    for size in (1, 2, 4):
        sampler = sampler_factory()
        bundle.config['training'].update(per_device_train_batch_size=size, gradient_accumulation_steps=8//size)
        batches, accepted, _ = batching.sample_optimizer_batches(bundle, sampler, sampling_group_size=4)
        workloads.append(accepted)
        states.append(sampler.state_dict())
    assert workloads[0] == workloads[1] == workloads[2]
    assert states[0] == states[1] == states[2]

def test_configured_training_k_distribution_and_zero_weight(sampler_factory):
    from collections import Counter
    from semgaze.model.config import default_section
    records = sampler_factory().records
    data = default_section('data')
    data['fewshot']['k_values'] = [1, 5, 10]
    data['fewshot']['train_k_probabilities'] = [0.6, 0.3, 0.1]
    sampler = TrainingEpisodeSampler(records, 42, data_config=data)
    counts = Counter()
    for _ in range(40):
        sampler.start_epoch()
        counts.update(len(sampler.sample().supports) for _ in range(sampler.query_count))
    assert set(counts) == {1, 5, 10}
    for k, p in zip((1, 5, 10), (0.6, 0.3, 0.1)):
        assert counts[k] / sum(counts.values()) == pytest.approx(p, abs=0.05)
    data['fewshot']['train_k_probabilities'] = [1.0, 0.0, 0.0]
    fixed = TrainingEpisodeSampler(records, 42, data_config=data)
    assert all(len(fixed.sample().supports) == 1 for _ in range(fixed.query_count))


def test_training_k_checkpoint_fails_closed_on_distribution_change(sampler_factory):
    from semgaze.model.config import default_section
    original = sampler_factory()
    original.sample()
    checkpoint = original.state_dict()
    data = default_section('data')
    data['fewshot']['k_values'] = [1]
    data['fewshot']['train_k_probabilities'] = [1.0]
    configured = TrainingEpisodeSampler(original.records, 42, data_config=data)
    with pytest.raises(ValueError, match='K distribution'):
        configured.load_state_dict(checkpoint)
    # Legacy checkpoints without explicit K fields resume only under legacy K distribution.
    legacy = dict(checkpoint)
    legacy.pop('k_values')
    legacy.pop('k_probabilities')
    baseline = sampler_factory()
    baseline.load_state_dict(legacy)
    assert baseline.state_dict()['cursor'] == checkpoint['cursor']
    with pytest.raises(ValueError, match='K distribution'):
        configured.load_state_dict(legacy)


def test_configured_single_k_uses_only_k1(sampler_factory):
    from semgaze.model.config import default_section
    data = default_section('data')
    data['fewshot']['k_values'] = [1]
    data['fewshot']['train_k_probabilities'] = [1.0]
    sampler = TrainingEpisodeSampler(sampler_factory().records, 42, data_config=data)
    assert all(len(sampler.sample().supports) == 1 for _ in range(sampler.query_count))
