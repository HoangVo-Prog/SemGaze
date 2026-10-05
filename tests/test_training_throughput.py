"""Real small HF InternVL/PEFT regression tests for the physical training path."""
from dataclasses import replace
import pytest
import torch

from test_model_path import tiny_bundle, episode
from throughput_reference import reference_forward
from semgaze.model.selected_loss import compute_selected_causal_nll
from semgaze.where.forward import forward_where_batch
from semgaze.semantic.flat.forward import forward_flat_batch, prepare_semantic_batch, frozen_vision_is_reusable
from semgaze.training.flat_step import run_flat_training_step
from semgaze.training.batching import sample_optimizer_batches
from semgaze.data.fewshot import TrainingEpisodeSampler
from semgaze.data.schema import FlatEpisode


def mixed_episode(episode):
    # K=2, shorter WHERE and semantic targets, N=3, distinct image contents/order.
    query = replace(episode.supports[0], record_id='mixed-query', stimulus_id='mixed-query')
    supports = (replace(episode.query, record_id='mixed-support-1', stimulus_id='mixed-support-1'),
                replace(episode.supports[0], record_id='mixed-support-2', stimulus_id='mixed-support-2'))
    return FlatEpisode(supports, query)


def deterministic_train(bundle):
    # Zero stochastic probabilities for numerical equivalence, keep training and
    # checkpoint recomputation active. Production dropout settings are unchanged.
    bundle.model.train()
    for module in bundle.model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    # Nonzero B makes early/middle/late A and B gradients meaningful.
    with torch.no_grad():
        for name, p in bundle.model.named_parameters():
            if 'lora_B' in name:
                p.normal_(std=0.02)


def grads(bundle):
    return {name: p.grad.clone() for name, p in list(bundle.model.named_parameters()) +
            [('projector.'+n, p) for n, p in bundle.projector.named_parameters()] if p.requires_grad}


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('checkpointing', [False, True])
def test_single_loss_state_and_all_trainable_gradient_parity(tmp_path, episode, dtype, checkpointing):
    bundle = tiny_bundle(tmp_path, episode)
    bundle.model.to(dtype=dtype)
    bundle.projector.to(dtype=dtype)
    deterministic_train(bundle)
    if checkpointing:
        bundle.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    old_w, old_s, old_total, old_f, _ = reference_forward(bundle, episode)
    old_total.backward()
    expected = grads(bundle)
    bundle.model.zero_grad(set_to_none=True)
    bundle.projector.zero_grad(set_to_none=True)
    where = forward_where_batch(bundle, [episode])
    flat, _, _ = forward_flat_batch(bundle, [episode.query], [bundle.projector(where.states[0])], where=where)
    total = where.loss_where + flat.loss
    total.backward()
    tolerance = dict(rtol=0.04, atol=0.002) if dtype == torch.bfloat16 else dict(rtol=2e-4, atol=2e-6)
    for actual, reference in [(where.loss_where, old_w), (flat.loss, old_s), (total, old_total), (where.states[0], old_f)]:
        torch.testing.assert_close(actual, reference, **tolerance)
    actual = grads(bundle)
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], **tolerance, msg=lambda msg: name+' '+msg)
    lora_names = [n for n in actual if 'lora_' in n]
    for name in [lora_names[0], lora_names[len(lora_names)//2], lora_names[-1]]:
        assert actual[name].ne(0).any()
    assert bundle.input_row.grad.ne(0).any() and bundle.output_row.grad.ne(0).any()
    assert bundle.projector[0].weight.grad.ne(0).any()


def test_episode_mean_is_not_token_mean():
    torch.manual_seed(7)
    hidden = torch.randn(2, 7, 8, requires_grad=True)
    head = torch.nn.Linear(8, 13)
    labels = torch.tensor([[-100, 1, -100, -100, -100, -100, -100],
                           [-100, 8, 9, 10, 11, 12, -100]])
    result = compute_selected_causal_nll(hidden, labels, head)
    expected = torch.stack([torch.nn.functional.cross_entropy(
        head(hidden[b, :-1]).float(), labels[b, 1:], ignore_index=-100) for b in range(2)])
    torch.testing.assert_close(result.episode_losses, expected)
    torch.testing.assert_close(result.loss, expected.mean())
    assert not torch.isclose(result.loss, result.token_nll.mean())
    assert result.token_counts.tolist() == [1, 5]


@pytest.mark.parametrize('checkpointing', [False, True])
def test_mixed_batch_equivalence_isolation_and_semantic_gradients(tmp_path, episode, checkpointing):
    bundle = tiny_bundle(tmp_path, episode)
    deterministic_train(bundle)
    if checkpointing:
        bundle.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    episodes = [episode, mixed_episode(episode)]
    singles = []
    for e in episodes:
        w = forward_where_batch(bundle, [e])
        s, _, _ = forward_flat_batch(bundle, [e.query], [bundle.projector(w.states[0])], where=w)
        singles.append((w, s))
    where = forward_where_batch(bundle, episodes)
    states = [bundle.projector(f) for f in where.states]
    fused, positions, _ = prepare_semantic_batch(bundle, [e.query for e in episodes], states, where=where)
    for b, (f, (single_w, _)) in enumerate(zip(where.states, singles)):
        torch.testing.assert_close(f, single_w.states[0], rtol=2e-4, atol=2e-6)
        assert len(f) == len(episodes[b].query.x_px)
        f.retain_grad()
        for t, pos in enumerate(positions[b]):
            torch.testing.assert_close(fused['inputs_embeds'][b, pos], states[b][t])
            assert fused['labels'][b, pos] == -100 and fused['attention_mask'][b, pos] == 1
        assert fused['labels'][b][fused['attention_mask'][b] == 0].eq(-100).all()
    flat, _, _ = forward_flat_batch(bundle, [e.query for e in episodes], states, where=where)
    torch.testing.assert_close(where.loss_where, torch.stack([w.loss_where for w, _ in singles]).mean())
    torch.testing.assert_close(flat.loss, torch.stack([s.loss for _, s in singles]).mean())
    # Semantic-only gradient must pass through both samples' WHERE fixation states.
    flat.loss.backward()
    assert all(f.grad is not None and f.grad.ne(0).any() for f in where.states)
    assert bundle.input_row.grad.ne(0).any()
    assert bundle.projector[0].weight.grad.ne(0).any()


def test_one_forward_per_branch_and_frozen_reuse(tmp_path, episode):
    bundle = tiny_bundle(tmp_path, episode)
    calls, vision_calls, head_shapes = [], [], []
    base = bundle.model.get_base_model()
    handles = [base.model.language_model.register_forward_pre_hook(lambda *args: calls.append(1)),
        base.model.vision_tower.register_forward_pre_hook(lambda *args: vision_calls.append(1)),
        base.lm_head.register_forward_pre_hook(lambda module, args: head_shapes.append(args[0].shape))]
    result = run_flat_training_step(bundle, [episode, mixed_episode(episode)])
    for handle in handles:
        handle.remove()
    assert len(calls) == 2 and len(vision_calls) == 1
    assert len(head_shapes) == 2 and all(len(s) == 2 for s in head_shapes)
    assert result['physical_batch_size'] == 2
    assert all(m['reused_query_vision'] for m in result['episodes'])
    base.model.vision_tower.requires_grad_(True)
    assert not frozen_vision_is_reusable(bundle.model)


def test_sampling_invariance_and_optimizer_membership(tmp_path, episode):
    bundle = tiny_bundle(tmp_path, episode)
    bundle.config['training'].update(per_device_train_batch_size=2, gradient_accumulation_steps=3)
    records = [replace(episode.query, subject=u, record_id=f'{u}:{i}', stimulus_id=f'{u}:{i}')
               for u in (1, 2) for i in range(12)]
    old, new = (TrainingEpisodeSampler(records, 42) for _ in range(2))
    expected = [old.sample() for _ in range(6)]
    batches, original, rejected = sample_optimizer_batches(bundle, new)
    assert original == expected and rejected == 0
    assert new.state_dict() == old.state_dict()
    actual = [e for b in batches for e in b.episodes]
    assert sorted(actual, key=repr) == sorted(expected, key=repr)
    assert all(len(b.episodes) == 2 for b in batches)


def test_mixed_batch_gradient_and_accumulation_equivalence(tmp_path, episode):
    bundle = tiny_bundle(tmp_path, episode)
    deterministic_train(bundle)
    episodes = [episode, mixed_episode(episode)]
    # Original episode means, accumulated without an optimizer update.
    for e in episodes:
        _, _, loss, _, _ = reference_forward(bundle, e)
        (loss / 2).backward()
    expected = grads(bundle)
    result = run_flat_training_step(bundle, episodes)
    for name, value in grads(bundle).items():
        torch.testing.assert_close(value, expected[name], rtol=4e-4, atol=3e-6)
    # Two physical B=2 microbatches with conventional scaling equal the same mean.
    run_flat_training_step(bundle, episodes, loss_scale=0.5)
    run_flat_training_step(bundle, episodes, loss_scale=0.5, zero_grad=False)
    for name, value in grads(bundle).items():
        torch.testing.assert_close(value, expected[name], rtol=4e-4, atol=3e-6)


def test_batch_checkpoint_round_trip(tmp_path, episode):
    from semgaze.model.checkpoint import save_checkpoint, restore_checkpoint_state
    from semgaze.training.flat_step import make_optimizer
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    bundle = tiny_bundle(tmp_path, episode)
    bundle.optimizer = make_optimizer(bundle)
    bundle.scheduler = torch.optim.lr_scheduler.LambdaLR(bundle.optimizer, lambda _: 1.0)
    run_flat_training_step(bundle, [episode, mixed_episode(episode)], optimizer_step=True)
    path = tmp_path / 'batched-checkpoint'
    save_checkpoint(bundle, path, split_manifest_identity='batch-fixture', step=1)
    fresh = tiny_bundle(tmp_path / 'loaded', episode)
    fresh.optimizer = make_optimizer(fresh)
    fresh.scheduler = torch.optim.lr_scheduler.LambdaLR(fresh.optimizer, lambda _: 1.0)
    set_peft_model_state_dict(fresh.model, load_file(str(path / 'adapter/adapter_model.safetensors')))
    assert restore_checkpoint_state(fresh, path, split_manifest_identity='batch-fixture', resume_optimizer=True) == 1
    bundle.model.eval()
    fresh.model.eval()
    with torch.no_grad():
        a = forward_where_batch(bundle, [episode, mixed_episode(episode)])
        b = forward_where_batch(fresh, [episode, mixed_episode(episode)])
    for left, right in zip(a.states, b.states):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_image_swap_cannot_contaminate_other_sample(tmp_path, episode):
    from semgaze.where.collator import collate_where_batch
    bundle = tiny_bundle(tmp_path, episode)
    bundle.model.eval()
    episodes = [episode, mixed_episode(episode)]
    batch = collate_where_batch(bundle.processor, episodes, bundle.end_fix_id)
    with torch.no_grad():
        old = forward_where_batch(bundle, episodes, batch=batch)
        # Only B's images change. A must be bitwise unaffected.
        offset = batch.metadata[1]['image_offset']
        batch.inputs['pixel_values'][offset:] = 0
        changed = forward_where_batch(bundle, episodes, batch=batch)
    torch.testing.assert_close(old.states[0], changed.states[0], rtol=0, atol=0)
    assert not torch.equal(old.states[1], changed.states[1])
