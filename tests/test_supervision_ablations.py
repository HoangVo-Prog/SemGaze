"""CPU-only supervision-mode contract tests. Copy to tests/ in patched checkout.

Run: pytest -q tests/test_supervision_ablations.py
Mocks the expensive InternVL forward but exercises real training orchestration,
loss gating, optimizer param selection and state-transfer autodiff.
"""
from types import SimpleNamespace
import pytest
import torch


def _run(monkeypatch, where_weight, sem_weight):
    import semgaze.training.flat_step as step
    from semgaze.model.build import FlatModelBundle
    import semgaze.data.joint as joint
    import semgaze.semantic.flat.diagnostics as sem_diagnostics

    calls = {'where': 0, 'where_compute_loss': None, 'states_requested': None, 'semantic': 0}

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_where = torch.nn.Parameter(torch.tensor(1.5))
        def forward(self, *args, **kwargs):
            raise AssertionError('not used in CPU forwarding mock')

    model = FakeModel()
    projector = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        projector[0].weight.fill_(2.)
    config = {
        'data': {},
        'where': {'use_end_fix_token': True, 'context': {'max_total_sequence_length': 128}},
        'semantic': {'conditioning': 'shared_states'},
        'training': {
            'lambda_where': where_weight,
            'lambda_sem': sem_weight,
            'scheduled_sampling': {'enabled': False},
            'late_fixation_ce': {'enabled': False, 'alpha': .5},
        },
    }
    row = torch.nn.Parameter(torch.tensor(0.))
    bundle = FlatModelBundle(model, SimpleNamespace(), projector, 999, row, row,
                             True, config, {}, None)
    query = SimpleNamespace(x_px=[12, 40], record_id='mock-query')
    episode = SimpleNamespace(query=query, supports=[])
    monkeypatch.setattr(step, 'require_seen_training_episode', lambda e, u: None)
    monkeypatch.setattr(joint, 'unseen_subject_ids', lambda _: set())
    monkeypatch.setattr(sem_diagnostics, 'summarize_semantic_component_nll',
                        lambda *args, **kwargs: {'loss_what': None, 'loss_why': None, 'loss_how': None})

    def fake_where(bundle, episodes, **kwargs):
        calls['where'] += 1
        calls['where_compute_loss'] = kwargs['compute_loss']
        calls['states_requested'] = kwargs['return_fixation_states']
        # Differentiable ONLY through WHERE computation, distinct from SEM path.
        f = torch.stack([model.lora_where * 1, model.lora_where * 2]).reshape(2, 1)
        where_ce = model.lora_where.square() if kwargs['compute_loss'] else None
        return SimpleNamespace(
            states=[f] if kwargs['return_fixation_states'] else [f.new_empty((0, 1))],
            episode_losses=where_ce.reshape(1) if where_ce is not None else None,
            loss_where=where_ce,
            unweighted_episode_losses=where_ce.reshape(1) if where_ce is not None else None,
            positions=[torch.tensor([0, 1])] if kwargs['return_fixation_states'] else [torch.empty(0, dtype=torch.long)],
            batch=SimpleNamespace(metadata=[{'where_length': 9}],
               samples=[SimpleNamespace(inputs={'labels': torch.tensor([[999, 2, 999]])}, response_start=0)])
        )

    def fake_semantic(bundle, queries, projected_states, **kwargs):
        calls['semantic'] += 1
        assert kwargs['where'] is not None
        assert len(projected_states) == 1 and projected_states[0].requires_grad
        value = projected_states[0].square().mean()
        result = SimpleNamespace(loss=value, episode_losses=value.reshape(1), token_nll=value.reshape(1))
        return result, [torch.tensor([0, 1])], [{}]

    monkeypatch.setattr(step, 'forward_where_batch', fake_where)
    monkeypatch.setattr(step, 'forward_flat_batch', fake_semantic)
    capture = {}
    result = step.run_flat_training_step(bundle, episode, diagnostics=False, audit_capture=capture)
    return bundle, calls, result, capture


def test_v3_skips_semantic_but_supervises_where(monkeypatch):
    bundle, calls, result, capture = _run(monkeypatch, 1., 0.)
    assert calls['where'] == 1 and calls['where_compute_loss'] is True
    assert calls['states_requested'] is False
    assert calls['semantic'] == 0
    assert result['loss_flat'] is None and result['loss_where'] is not None
    assert bundle.projector[0].weight.grad is None
    assert bundle.model.lora_where.grad is not None
    assert all(p is not bundle.projector[0].weight for p in bundle.trainable_parameters())


def test_v4_preserves_differentiable_fixation_state_handoff(monkeypatch):
    bundle, calls, result, capture = _run(monkeypatch, 0., 1.)
    assert calls['where'] == 1 and calls['where_compute_loss'] is False
    assert calls['states_requested'] is True
    assert calls['semantic'] == 1
    assert result['loss_where'] is None and result['loss_flat'] is not None
    assert capture['loss_where'] is None
    # Inspect retained intermediate WHERE state gradients, not just shared LoRA grads.
    assert len(capture['state_gradients']) == 1
    assert torch.count_nonzero(capture['state_gradients'][0]) > 0
    assert torch.count_nonzero(bundle.projector[0].weight.grad) > 0
    assert torch.count_nonzero(bundle.model.lora_where.grad) > 0


def test_full_joint_loss_and_gradient_remain_additive(monkeypatch):
    bundle, calls, result, capture = _run(monkeypatch, 1., 1.)
    assert calls['where_compute_loss'] is True and calls['states_requested'] is True
    assert calls['semantic'] == 1
    assert torch.allclose(result['loss_total'], result['loss_where'] + result['loss_flat'])
    assert torch.count_nonzero(capture['state_gradients'][0]) > 0


def test_invalid_zero_objective_rejected(monkeypatch):
    with pytest.raises(ValueError, match='WHERE and/or semantic'):
        _run(monkeypatch, 0., 0.)


def test_v4_where_forward_skips_logits_and_keeps_states(monkeypatch):
    """Actual WHERE orchestration must not execute the LM head in V4."""
    import semgaze.where.forward as wf
    model = torch.nn.Module()
    model.lora_where = torch.nn.Parameter(torch.tensor(1.))
    model.get_output_embeddings = lambda: (_ for _ in ()).throw(AssertionError('WHERE head should not be read'))
    backbone_called = []

    def fake_backbone(model, inputs, **kwargs):
        backbone_called.append(True)
        h = torch.stack([model.lora_where * i for i in (1., 2., 3., 4.)]).reshape(1, 4, 1)
        return SimpleNamespace(last_hidden_state=h, image_hidden_states=h.new_zeros((1, 1, 2)))

    def fake_extract(hidden, input_ids, labels, end_fix_id, counts, *, return_unpadded=False):
        assert counts == [2] and end_fix_id == 999 and return_unpadded
        return [hidden[0, :2]], None, [torch.tensor([1, 3])]

    monkeypatch.setattr(wf, 'forward_backbone', fake_backbone)
    monkeypatch.setattr(wf, 'extract_query_states', fake_extract)
    monkeypatch.setattr(wf, 'to_model_device', lambda tensors, model: tensors)
    monkeypatch.setattr(wf, 'compute_selected_causal_nll', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError('WHERE CE must not execute in V4')))
    episode = SimpleNamespace(query=SimpleNamespace(x_px=[1., 2.]))
    batch = SimpleNamespace(
        episodes=(episode,),
        inputs={'input_ids': torch.tensor([[1, 999, 2, 999]]),
                'labels': torch.tensor([[-100, 999, 2, 999]]),
                'pixel_values': torch.zeros(1, 1)},
        metadata=[{'image_count': 1, 'query_image_index': 0}],
    )
    bundle = SimpleNamespace(model=model, end_fix_id=999, processor=SimpleNamespace(image_seq_length=1),
        context_limit=1024, config={'where': {'use_end_fix_token': True,
            'supervision': {'use_cache': False}}, 'semantic': {'conditioning': 'shared_states'}})
    result = wf.forward_where_batch(bundle, [episode], batch=batch,
                                    compute_loss=False, return_fixation_states=True)
    assert backbone_called and result.loss_where is None
    assert result.episode_losses is None and result.unweighted_episode_losses is None
    assert len(result.states) == 1 and result.states[0].requires_grad
    result.states[0].square().mean().backward()
    assert torch.count_nonzero(model.lora_where.grad) > 0
