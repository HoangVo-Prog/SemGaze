"""Sampling contracts; collation doubles avoid loading a model or image files."""
from collections import Counter
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from semgaze.data.fewshot import TrainingEpisodeSampler
from semgaze.data.schema import FlatEpisode, normalized_episode_from_dict
from semgaze.model.config import ROOT, default_section
from semgaze.training import batching


@pytest.fixture
def sampler_factory():
    episode = normalized_episode_from_dict(json.loads((ROOT / 'tests/fixtures/flat_episode.json').read_text()))
    records = [replace(episode.query, subject=u, stimulus_id=f'{u}:{i}', record_id=f'{u}:{i}:{r}')
               for u in (1, 2) for i in range(12) for r in range(2)]
    data = default_section('data')
    data['fewshot'].update(k_values=[1, 5, 10], train_k_probabilities=[0.25, 0.75, 0.0])
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


def reference_sample(sampler, k=None):
    """Original sampler, with only the optional K draw omitted for batch members."""
    rng = sampler.rng
    subject = rng.choice(sampler.subjects)
    if k is None:
        k = rng.choices(sampler.k_values, weights=sampler.probabilities, k=1)[0]
    query = rng.choice(sampler.queries[subject, k])
    images = rng.sample([i for i in sampler.images[subject] if i != query.stimulus_id], k)
    supports = [rng.choice(sampler.images[subject][i]) for i in images]
    rng.shuffle(supports)
    return FlatEpisode(tuple(supports), query)


def test_conditional_sampling_and_original_b1_rng_order(sampler_factory):
    actual, reference = sampler_factory(), sampler_factory()
    for k in [None, 1, 5, None] * 30:
        assert actual.sample(k=k) == reference_sample(reference, k)
        assert actual.state_dict() == reference.state_dict()
    for invalid in [0, 10, 2, True, 1.0]:
        state = actual.state_dict()
        with pytest.raises(ValueError, match='not enabled'):
            actual.sample(k=invalid)
        assert actual.state_dict() == state


@pytest.mark.parametrize('size', [1, 2, 4])
@pytest.mark.parametrize('length_sort', [False, True])
def test_same_k_preserves_batch_order_membership_and_resume(bundle, sampler_factory, size, length_sort):
    bundle.config['training'].update(per_device_train_batch_size=size, length_aware_batching=length_sort)
    actual, reference = sampler_factory(), sampler_factory()
    for _ in range(4):
        expected_groups = []
        for _ in range(3):
            first = reference_sample(reference)
            expected_groups.append([first] + [reference_sample(reference, len(first.supports)) for _ in range(size-1)])
        batches, original, rejected = batching.sample_optimizer_batches(bundle, actual)
        assert rejected == 0
        assert original == [e for group in expected_groups for e in group]
        for batch, expected in zip(batches, expected_groups):
            assert len(batch.episodes) == size
            assert {len(e.supports) for e in batch.episodes} == {len(expected[0].supports)}
            assert sorted(batch.episodes, key=repr) == sorted(expected, key=repr)
        assert actual.state_dict() == reference.state_dict()
    resumed = sampler_factory(seed=999)
    resumed.load_state_dict(actual.state_dict())
    assert batching.sample_optimizer_batches(bundle, resumed)[1:] == batching.sample_optimizer_batches(bundle, actual)[1:]


def test_k_draws_use_configured_batch_distribution(bundle, sampler_factory):
    sampler = sampler_factory()
    bundle.config['training']['gradient_accumulation_steps'] = 2000
    batches, _, _ = batching.sample_optimizer_batches(bundle, sampler)
    counts = Counter(len(b.episodes[0].supports) for b in batches)
    assert set(counts) == {1, 5}
    assert counts[1] / len(batches) == pytest.approx(0.25, abs=0.035)
    order = [len(b.episodes[0].supports) for b in batches]
    assert order != sorted(order)  # no global K sort across physical batches


def test_overflow_retries_keep_initial_k_and_bound_failure(bundle, sampler_factory, monkeypatch):
    sampler, reference = sampler_factory(), sampler_factory()
    rejected = reference_sample(reference)
    k = len(rejected.supports)
    bundle.config['training']['gradient_accumulation_steps'] = 1
    expected = [reference_sample(reference, k), reference_sample(reference, k)]
    original_collate = batching.collate_where
    calls = []
    def collate(processor, episode, *args, **kwargs):
        calls.append(episode)
        if len(calls) == 1:
            raise batching.WhereContextOverflowError('test overflow')
        return original_collate(processor, episode, *args, **kwargs)
    monkeypatch.setattr(batching, 'collate_where', collate)
    _, accepted, retries = batching.sample_optimizer_batches(bundle, sampler)
    assert retries == 1 and accepted == expected
    assert {len(e.supports) for e in calls} == {k}
    assert sampler.state_dict() == reference.state_dict()
    calls.clear()
    def always_overflow(processor, episode, *args, **kwargs):
        calls.append(len(episode.supports))
        raise batching.WhereContextOverflowError('test overflow')
    monkeypatch.setattr(batching, 'collate_where', always_overflow)
    with pytest.raises(RuntimeError, match='at K='):
        batching.sample_optimizer_batches(bundle, sampler)
    assert len(calls) == 10 and len(set(calls)) == 1


def test_benchmark_replays_same_real_episode_identities_across_sizes(bundle, sampler_factory):
    workloads, states, packed = [], [], []
    for size in (1, 2, 4):
        sampler = sampler_factory()
        bundle.config['training'].update(per_device_train_batch_size=size, gradient_accumulation_steps=8//size)
        batches, accepted, _ = batching.sample_optimizer_batches(bundle, sampler, sampling_group_size=4)
        workloads.append(accepted)
        states.append(sampler.state_dict())
        packed.append([e for batch in batches for e in batch.episodes])
        assert all(len({len(e.supports) for e in batch.episodes}) == 1 for batch in batches)
    assert workloads[0] == workloads[1] == workloads[2]
    assert packed[0] == packed[1] == packed[2]
    assert states[0] == states[1] == states[2]
    with pytest.raises(ValueError, match='sampling_group_size'):
        batching.sample_optimizer_batches(bundle, sampler, sampling_group_size=3)
