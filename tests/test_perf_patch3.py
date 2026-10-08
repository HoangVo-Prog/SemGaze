"""Per-batch K distribution and rollback/resume contract tests."""
import pytest
from semgaze.data.fewshot import TrainingEpisodeSampler
from semgaze.training.batching import sample_optimizer_batches
from semgaze.model.config import default_section
from test_same_k_batching import sampler_factory, bundle


@pytest.mark.parametrize('k_values,probabilities', [([1],[1.0]), ([10],[1.0]),
    ([1,10],[.5,.5]), (list(range(1,11)),[.1]*10), ([1,5,10],[.2,.3,.5])])
@pytest.mark.parametrize('batch_size,accumulation', [(1,2),(2,3),(4,2)])
def test_batch_policy_one_k_draw_per_physical_batch(bundle, sampler_factory, k_values, probabilities,
                                                    batch_size, accumulation):
    data = default_section('data')
    data['fewshot'].update(k_values=k_values,train_k_probabilities=probabilities,
                           k_sampling_strategy='per_batch')
    sampler = TrainingEpisodeSampler(sampler_factory().records, 42, data_config=data)
    bundle.config['data'] = data
    bundle.config['training'].update(per_device_train_batch_size=batch_size,
                                     gradient_accumulation_steps=accumulation)
    batches, episodes, rejected = sample_optimizer_batches(bundle,sampler)
    assert rejected == 0
    assert len(episodes) == batch_size * accumulation
    assert len(batches) == accumulation
    assert all(len({len(e.supports) for e in b.episodes}) == 1 for b in batches)
    assert all(len(b.episodes) == batch_size for b in batches)
    assert [len(e.supports) for e in episodes] == [len(e.supports) for b in batches for e in b.episodes]
    assert all(len(e.supports) in k_values for e in episodes)
    # The saved sampler also records the statistical sampling policy.
    assert sampler.state_dict()['k_sampling_strategy'] == 'per_batch'
    restored = TrainingEpisodeSampler(sampler_factory().records, 12345,data_config=data)
    restored.load_state_dict(sampler.state_dict())
    assert restored.sample_for_query(restored.queries[0],k=k_values[0]) == sampler.sample_for_query(sampler.queries[0],k=k_values[0])


def test_resume_fails_closed_on_policy_mismatch(sampler_factory):
    baseline=sampler_factory()
    state=baseline.state_dict()
    data=default_section('data')
    data['fewshot']['k_sampling_strategy']='per_batch'
    changed=TrainingEpisodeSampler(baseline.records, 42, data_config=data)
    with pytest.raises(ValueError,match='strategy'):
        changed.load_state_dict(state)

def test_static_training_feature_gate_rejects_dropout():
    import torch
    from semgaze.model.training_feature_gate import inspect_frozen_training_producers
    vision = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(p=.2))
    vision.requires_grad_(False)
    projector = torch.nn.Linear(4, 4).requires_grad_(False)
    result = inspect_frozen_training_producers(vision, projector)
    assert not result['static_gate_passed']
    assert result['blockers'] and not result['training_cache_enabled']

