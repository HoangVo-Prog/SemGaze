"""Epoch scheduling, flat token-loss diagnostics, and persisted validation history."""
import json
from dataclasses import replace
import pytest
import torch

pytest.importorskip('peft')
from test_model_path import tiny_bundle, episode
from semgaze.evaluation import validation
from semgaze.evaluation import predictions
from semgaze.evaluation.validation import EVAL_KEYS, evaluate_validation_episode, validate_epoch
from semgaze.training import loop
from semgaze.training.loop import resolve_epoch_schedule, run_training_loop
from semgaze.training.flat_step import make_optimizer, run_flat_training_step
from semgaze.model.checkpoint import restore_checkpoint_state
from semgaze.model.config import validate_config


def validation_data(episode):
    supports = [replace(episode.supports[0], record_id=f'support-{i}', stimulus_id=f'image-{i}') for i in range(10)]
    entries = [{'record_id': s.record_id, 'image_name': s.stimulus_id, 'task': s.task} for s in supports]
    manifest = {'seen_subject_ids': [1], 'validation_stimulus_ids': [episode.query.stimulus_id],
                'validation_supports': {'1': {str(k): list(reversed(entries[:k])) for k in (1, 5, 10)}}}
    return {s.record_id: s for s in supports}, [episode.query], manifest


def test_validation_losses_are_flat_diagnostics_and_preserve_training(tmp_path, episode):
    bundle = tiny_bundle(tmp_path, episode)
    run_flat_training_step(bundle, episode)
    parameters = [(p, p.detach().clone(), None if p.grad is None else p.grad.clone())
                  for p in (*bundle.model.parameters(), *bundle.projector.parameters())]
    bundle.model.get_base_model().model.vision_tower.eval()
    modes = [m.training for root in (bundle.model, bundle.projector) for m in root.modules()]
    rng = torch.get_rng_state().clone()
    calls = []
    hook = bundle.model.get_base_model().model.register_forward_pre_hook(lambda *args: calls.append(torch.is_grad_enabled()))
    try:
        result = evaluate_validation_episode(bundle, episode)
    finally:
        hook.remove()
    assert calls == [False, False]  # one WHERE and one flat forward, no semantic fan-out
    assert all(result[key] > 0 for key in EVAL_KEYS)
    weighted = sum(result[f'eval_{c}'] * result[f'{c}_response_tokens'] for c in ('what', 'why', 'how'))
    assert result['eval_flat'] == pytest.approx(weighted / result['flat_response_tokens'], rel=1e-6)
    assert result['eval_total'] == pytest.approx(result['eval_where'] + result['eval_flat'])
    assert modes == [m.training for root in (bundle.model, bundle.projector) for m in root.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    for p, value, grad in parameters:
        assert torch.equal(p, value)
        assert p.grad is None if grad is None else torch.equal(p.grad, grad)
    unseen = replace(episode, supports=tuple(replace(s, subject=7) for s in episode.supports),
                     query=replace(episode.query, subject=7))
    with pytest.raises(ValueError, match='unseen'):
        evaluate_validation_episode(bundle, unseen)
    assert modes == [m.training for root in (bundle.model, bundle.projector) for m in root.modules()]


def test_validation_covers_full_seen_query_set_and_frozen_k_order(tmp_path, episode, monkeypatch):
    bundle = tiny_bundle(tmp_path, episode)
    train, queries, manifest = validation_data(episode)
    queries.append(replace(episode.query, record_id='second', stimulus_id='second-image'))
    manifest['validation_stimulus_ids'].append('second-image')
    queries.append(replace(episode.query, subject=7, record_id='unseen'))
    calls = []
    def evaluate(bundle, ep):
        assert not bundle.model.training and not torch.is_grad_enabled()
        k = len(ep.supports)
        assert [s.record_id for s in ep.supports] == [e['record_id'] for e in manifest['validation_supports']['1'][str(k)]]
        calls.append((ep.query.record_id, k))
        return {key: float(k) for key in EVAL_KEYS}
    monkeypatch.setattr(validation, '_episode_losses', evaluate)
    result = validate_epoch(bundle, train, queries, manifest)
    assert len(calls) == result['eval_episodes'] == 6
    assert result['eval_queries'] == 2
    assert result['eval_flat'] == pytest.approx(16 / 3)
    assert result['eval_by_k']['10']['eval_where'] == 10
    assert bundle.model.training
    with pytest.raises(ValueError, match='leakage'):
        validate_epoch(bundle, train | {queries[0].record_id: queries[0]}, queries, manifest)
    with pytest.raises(ValueError, match='cover every'):
        validate_epoch(bundle, train, [], manifest)
    def fail(*args):
        raise RuntimeError('overflow or forward failure')
    monkeypatch.setattr(validation, '_episode_losses', fail)
    with pytest.raises(RuntimeError): validate_epoch(bundle, train, queries, manifest)
    assert bundle.model.training


@pytest.mark.parametrize('max_steps,steps', [(4, None), (4, 0), (4, True)])
def test_epoch_schedule_requires_an_explicit_complete_epoch(tmp_path, episode, max_steps, steps):
    cfg = tiny_bundle(tmp_path, episode).config
    with pytest.raises(ValueError): resolve_epoch_schedule(cfg, max_steps, steps)


def test_config_accepts_step_evaluation(tmp_path, episode):
    cfg = tiny_bundle(tmp_path, episode).config
    cfg['evaluation']['strategy'] = 'steps'
    validate_config(cfg)


class FixedSampler:
    def __init__(self, episode):
        self.episode, self.calls = episode, 0

    def sample(self):
        self.calls += 1
        return self.episode

    def state_dict(self):
        return self.calls

    def load_state_dict(self, value):
        self.calls = value


def test_epoch_end_logs_checkpoint_and_resume(tmp_path, episode, monkeypatch, capsys):
    bundle = tiny_bundle(tmp_path, episode)
    predictions.resolve_prediction_settings(bundle.config, 4)
    resolve_epoch_schedule(bundle.config, 4, 2)
    bundle.optimizer = make_optimizer(bundle)
    bundle.scheduler = torch.optim.lr_scheduler.LambdaLR(bundle.optimizer, lambda _: 1)
    train, queries, manifest = validation_data(episode)
    evaluations = []
    def evaluate(*args):
        evaluations.append(len(bundle.trainer_history))
        return {**{key: float(i + 1) for i, key in enumerate(EVAL_KEYS)},
                'eval_queries': 1, 'eval_episodes': 3}
    monkeypatch.setattr(loop, 'validate_epoch', evaluate)
    monkeypatch.setattr(predictions, 'evaluate_where_episode', lambda *args: {'text': 'generated WHERE'})
    monkeypatch.setattr(predictions, 'evaluate_flat_episode', lambda *args, **kw: {'text': 'generated semantic'})
    # Real training/backward, optimizer and checkpoint saves; deterministic
    # validator here isolates epoch timing and resume duplication from inference.
    sampler = FixedSampler(episode)
    run_training_loop(bundle, sampler, train, queries, manifest, split_manifest_identity='fixture',
                      max_steps=4, save_every=1)
    assert len(evaluations) == 2
    summary = [r for r in bundle.trainer_history if r['event'] == 'epoch_validation']
    assert [r['step'] for r in summary] == [2, 4]
    assert [r['epoch'] for r in summary] == [1, 2]
    assert 'Epoch 1 | step 2' in capsys.readouterr().out
    logged = [json.loads(line) for line in (tmp_path / 'trainer_log.jsonl').read_text().splitlines()]
    assert logged == bundle.trainer_history
    checkpoint_history = json.loads((tmp_path / 'checkpoint-2/trainer_state.json').read_text())['log_history']
    assert checkpoint_history[-2] == summary[0]
    assert checkpoint_history[-1]['event'] == 'epoch_predictions'
    assert sampler.calls == 4  # predictions never draw extra training episodes
    assert [r['epoch'] for r in bundle.trainer_history if r['event'] == 'epoch_predictions'] == [1, 2]
    # Resume mid-epoch (step 3): preserve epoch-1 history; emit epoch 2 only.
    resumed = tiny_bundle(tmp_path / 'resumed', episode)
    predictions.resolve_prediction_settings(resumed.config, 4)
    resolve_epoch_schedule(resumed.config, 4, 2)
    resumed.optimizer = make_optimizer(resumed)
    resumed.scheduler = torch.optim.lr_scheduler.LambdaLR(resumed.optimizer, lambda _: 1)
    start = restore_checkpoint_state(resumed, tmp_path / 'checkpoint-3', split_manifest_identity='fixture',
                                     sampler=sampler, resume_optimizer=True)
    evaluations.clear()
    run_training_loop(resumed, sampler, train, queries, manifest, split_manifest_identity='fixture',
                      max_steps=4, save_every=1, start=start)
    assert len(evaluations) == 1
    assert [r['epoch'] for r in resumed.trainer_history if r['event'] == 'epoch_validation'] == [1, 2]
    assert [r['epoch'] for r in resumed.trainer_history if r['event'] == 'epoch_predictions'] == [1, 2]
    assert not (tmp_path / 'resumed/predictions/epoch-0001').exists()


def test_real_small_hf_epoch_validation(tmp_path, episode):
    bundle = tiny_bundle(tmp_path, episode)
    train, queries, manifest = validation_data(episode)
    result = validate_epoch(bundle, train, queries, manifest)
    assert result['eval_episodes'] == 3
    assert set(result['eval_by_k']) == {'1', '5', '10'}
    assert result['eval_total'] == pytest.approx(result['eval_where'] + result['eval_flat'])
    assert all(p.grad is None for p in bundle.trainable_parameters())
    print(loop.format_epoch_summary({'epoch': 1, 'step': 2, **result}))
