"""Epoch scheduling, flat token-loss diagnostics, and persisted validation history."""
import json
from dataclasses import replace
import pytest
import torch

pytest.importorskip('peft')
from test_model_path import tiny_bundle, episode
from semgaze.evaluation import test as validation
from semgaze.evaluation import predictions
from semgaze.evaluation.test import EVAL_KEYS, evaluate_test_episode, evaluate_test_epoch
from semgaze.training import loop
from semgaze.training.loop import resolve_epoch_schedule, run_training_loop
from semgaze.training.flat_step import make_optimizer, run_flat_training_step
from semgaze.model.checkpoint import restore_checkpoint_state
from semgaze.model.config import validate_config


def evaluation_data(episode):
    supports = [replace(episode.supports[0], subject=7, record_id=f'support-{i}', stimulus_id=f'image-{i}') for i in range(100)]
    entries = [{'image_name': s.stimulus_id, 'task': s.task,
                'resolved_record_id_by_subject': {'7': s.record_id}} for s in supports]
    manifest = {'seen_subject_ids': [1], 'unseen_subject_ids': [7,8,9],
                'test_stimulus_ids': [episode.query.stimulus_id],
                'train_stimulus_ids': [s.stimulus_id for s in supports],
                'support_draws': {str(k): [list(reversed(entries[d*k:(d+1)*k])) for d in range(10)] for k in (1,5,10)}}
    return {s.record_id: s for s in supports}, [replace(episode.query, subject=7)], manifest


def test_validation_losses_are_flat_diagnostics_and_preserve_training(tmp_path, episode):
    bundle = tiny_bundle(tmp_path, episode)
    bundle.config['test']['prediction']['execution'] = 'serial'  # retained reference API
    run_flat_training_step(bundle, episode)
    parameters = [(p, p.detach().clone(), None if p.grad is None else p.grad.clone())
                  for p in (*bundle.model.parameters(), *bundle.projector.parameters())]
    bundle.model.get_base_model().model.vision_tower.eval()
    modes = [m.training for root in (bundle.model, bundle.projector) for m in root.modules()]
    rng = torch.get_rng_state().clone()
    calls = []
    hook = bundle.model.get_base_model().model.register_forward_pre_hook(lambda *args: calls.append(torch.is_grad_enabled()))
    try:
        result = evaluate_test_episode(bundle, episode)
    finally:
        hook.remove()
    assert calls == [False, False]  # one WHERE and one flat forward, no semantic fan-out
    assert all(result[key] > 0 for key in EVAL_KEYS)
    weighted = sum(result[f'test_{c}'] * result[f'{c}_response_tokens'] for c in ('what', 'why', 'how'))
    assert result['test_flat'] == pytest.approx(weighted / result['flat_response_tokens'], rel=1e-6)
    assert result['test_total'] == pytest.approx(result['test_where'] + result['test_flat'])
    assert modes == [m.training for root in (bundle.model, bundle.projector) for m in root.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    for p, value, grad in parameters:
        assert torch.equal(p, value)
        assert p.grad is None if grad is None else torch.equal(p.grad, grad)
    unseen = replace(episode, supports=tuple(replace(s, subject=7) for s in episode.supports),
                     query=replace(episode.query, subject=7))
    assert evaluate_test_episode(bundle, unseen)['test_total'] > 0
    assert modes == [m.training for root in (bundle.model, bundle.projector) for m in root.modules()]


def test_validation_covers_full_seen_query_set_and_frozen_k_order(tmp_path, episode, monkeypatch):
    bundle = tiny_bundle(tmp_path, episode)
    bundle.config['test']['prediction']['execution'] = 'serial'  # retained reference API
    train, queries, manifest = evaluation_data(episode)
    queries.append(replace(queries[0], record_id='second', stimulus_id='second-image'))
    manifest['test_stimulus_ids'].append('second-image')
    queries.append(replace(episode.query, subject=1, record_id='seen'))
    calls = []
    def evaluate(bundle, ep):
        assert not bundle.model.training and not torch.is_grad_enabled()
        k = len(ep.supports)
        assert [s.record_id for s in ep.supports] in [[e['resolved_record_id_by_subject']['7'] for e in block] for block in manifest['support_draws'][str(k)]]
        calls.append((ep.query.record_id, k))
        return {key: float(k) for key in EVAL_KEYS}
    def evaluate_batch(bundle, episodes, **kwargs):
        return [evaluate(bundle, ep) for ep in episodes]
    monkeypatch.setattr(validation, 'batch_losses', evaluate_batch)
    result = evaluate_test_epoch(bundle, train, queries, manifest)
    assert len(calls) == result['test_episodes'] == 60
    assert result['test_queries'] == 2
    assert result['test_flat'] == pytest.approx(16 / 3)
    assert result['test_by_k']['10']['test_where'] == 10
    assert bundle.model.training
    with pytest.raises(ValueError, match='leakage'):
        evaluate_test_epoch(bundle, train | {queries[0].record_id: queries[0]}, queries, manifest)
    with pytest.raises(ValueError, match='no unseen-subject'):
        evaluate_test_epoch(bundle, train, [], manifest)
    def fail(*args, **kwargs):
        raise RuntimeError('overflow or forward failure')
    monkeypatch.setattr(validation, 'batch_losses', fail)
    with pytest.raises(RuntimeError): evaluate_test_epoch(bundle, train, queries, manifest)
    assert bundle.model.training


@pytest.mark.parametrize('query_count', [None, 0, True])
def test_epoch_schedule_requires_query_count(tmp_path, episode, query_count):
    cfg = tiny_bundle(tmp_path, episode).config
    with pytest.raises(ValueError):
        resolve_epoch_schedule(cfg, query_count)


def test_config_accepts_step_evaluation(tmp_path, episode):
    cfg = tiny_bundle(tmp_path, episode).config
    cfg['evaluation']['strategy'] = 'steps'
    cfg['evaluation']['eval_steps'] = 1000
    validate_config(cfg)


def test_step_evaluation_uses_configured_global_step_interval(tmp_path, episode, monkeypatch):
    bundle = tiny_bundle(tmp_path, episode)
    bundle.config['training']['num_train_epochs'] = 1
    bundle.config['evaluation'].update(strategy='steps', eval_steps=2)
    bundle.config['evaluation']['predictions'].update(train_batches=0, test_scope='none')
    bundle.config['checkpoint'].update(save_at_end=False, save_at_epoch_end=False)
    bundle.optimizer = make_optimizer(bundle)
    bundle.scheduler = torch.optim.lr_scheduler.LambdaLR(bundle.optimizer, lambda _: 1)
    evaluations = []

    monkeypatch.setattr(loop, 'run_flat_training_step', lambda *a, **kw: {'loss_total': 1.0})
    monkeypatch.setattr(loop, 'evaluate_test_epoch', lambda *a: evaluations.append(1) or
                        dict.fromkeys(EVAL_KEYS, 1.0) | {'test_queries': 1, 'test_episodes': 1})

    run_training_loop(bundle, FixedSampler(episode, count=5), {}, [], {},
                      split_manifest_identity='fixture', max_steps=None, save_every=None)

    assert [entry['step'] for entry in bundle.trainer_history if entry['event'] == 'epoch_test'] == [2, 4]
    assert len(evaluations) == 2


class FixedSampler:
    """Small finite coverage sampler for loop/model integration tests."""
    def __init__(self, episode, count=2):
        self.episode, self.calls = episode, 0
        self.query_count, self.cursor, self.epoch = count, 0, 0

    @property
    def remaining(self):
        return self.query_count - self.cursor

    def epoch_complete(self):
        return self.epoch > 0 and self.cursor == self.query_count

    def start_epoch(self):
        assert self.epoch == 0 or self.epoch_complete()
        self.epoch += 1
        self.cursor = 0

    def sample(self):
        if not self.epoch:
            self.start_epoch()
        if not self.remaining:
            raise StopIteration
        self.calls += 1
        self.cursor += 1
        return replace(self.episode, query=replace(self.episode.query, record_id=f'q{self.cursor}'))

    def state_dict(self):
        return self.calls, self.cursor, self.epoch

    def load_state_dict(self, value):
        self.calls, self.cursor, self.epoch = value


def test_epoch_end_logs_checkpoint_and_resume(tmp_path, episode, monkeypatch, capsys):
    bundle = tiny_bundle(tmp_path, episode)
    bundle.config['training']['num_train_epochs'] = 2
    bundle.config['evaluation']['predictions'].update(train_batches=0, test_scope='none')
    bundle.optimizer = make_optimizer(bundle)
    bundle.scheduler = torch.optim.lr_scheduler.LambdaLR(bundle.optimizer, lambda _: 1)
    train, queries, manifest = evaluation_data(episode)
    evaluations = []
    def evaluate(*args):
        evaluations.append(1)
        return dict.fromkeys(EVAL_KEYS, 1.0) | {'test_queries':1,'test_episodes':30}
    monkeypatch.setattr(loop, 'evaluate_test_epoch', evaluate)
    sampler = FixedSampler(episode)
    run_training_loop(bundle, sampler, train, queries, manifest, split_manifest_identity='fixture',
                      max_steps=4, save_every=1)
    summary = [r for r in bundle.trainer_history if r['event'] == 'epoch_test']
    assert [r['step'] for r in summary] == [2,4]
    assert [r['epoch'] for r in summary] == [1,2]
    assert sampler.calls == 4
    resumed = tiny_bundle(tmp_path / 'resumed', episode)
    resumed.config = bundle.config
    resumed.optimizer = make_optimizer(resumed)
    resumed.scheduler = torch.optim.lr_scheduler.LambdaLR(resumed.optimizer, lambda _: 1)
    start = restore_checkpoint_state(resumed, tmp_path/'checkpoint-3', split_manifest_identity='fixture',
                                     sampler=sampler, resume_optimizer=True)
    assert sampler.cursor == 1 and sampler.epoch == 2
    evaluations.clear()
    run_training_loop(resumed, sampler, train, queries, manifest, split_manifest_identity='fixture',
                      max_steps=4, save_every=1, start=start)
    assert len(evaluations) == 1 and sampler.calls == 4
    assert [r['epoch'] for r in resumed.trainer_history if r['event'] == 'epoch_test'] == [1,2]


def test_debug_cutoff_does_not_evaluate_partial_epoch(tmp_path, episode, monkeypatch):
    bundle = tiny_bundle(tmp_path, episode)
    bundle.config['evaluation']['predictions'].update(train_batches=0, test_scope='none')
    bundle.config['checkpoint'].update(save_at_end=False)
    bundle.optimizer = make_optimizer(bundle)
    bundle.scheduler = torch.optim.lr_scheduler.LambdaLR(bundle.optimizer, lambda _: 1)
    monkeypatch.setattr(loop, 'evaluate_test_epoch', lambda *a: pytest.fail('partial epoch evaluation'))
    sampler = FixedSampler(episode, count=3)
    run_training_loop(bundle, sampler, {}, [], {}, split_manifest_identity='fixture', max_steps=2, save_every=None)
    assert sampler.cursor == 2 and not sampler.epoch_complete()


def test_tail_optimizer_window_keeps_all_queries_and_correct_weights(tmp_path, episode, monkeypatch):
    bundle = tiny_bundle(tmp_path, episode)
    bundle.config['training'].update(per_device_train_batch_size=2, gradient_accumulation_steps=2)
    bundle.config['evaluation']['predictions'].update(train_batches=0, test_scope='none')
    bundle.config['checkpoint'].update(save_at_end=False, save_at_epoch_end=False)
    bundle.optimizer = make_optimizer(bundle)
    bundle.scheduler = torch.optim.lr_scheduler.LambdaLR(bundle.optimizer, lambda _: 1)
    calls, evaluated = [], []
    sampler = FixedSampler(episode, count=5)
    def step(bundle, episodes, **kwargs):
        calls.append(([e.query.record_id for e in episodes], kwargs['loss_scale']))
        return {'loss_total': 1.0}
    def evaluate(*args):
        assert sampler.cursor == 5
        evaluated.append(1)
        return dict.fromkeys(EVAL_KEYS, 1.0) | {'test_queries':1, 'test_episodes':30}
    monkeypatch.setattr(loop, 'run_flat_training_step', step)
    monkeypatch.setattr(loop, 'evaluate_test_epoch', evaluate)
    run_training_loop(bundle, sampler, {}, [], {}, split_manifest_identity='fixture', max_steps=None, save_every=None)
    assert calls == [(['q1','q2'], 0.5), (['q3','q4'], 0.5), (['q5'], 1.0)]
    assert sampler.calls == 5 and evaluated == [1]
    assert bundle.scheduler.last_epoch == 2


def test_real_sampler_checkpoint_restores_next_episode_and_scheduler(tmp_path, episode):
    from semgaze.data.fewshot import TrainingEpisodeSampler
    from semgaze.model.checkpoint import save_checkpoint
    from semgaze.training.flat_step import make_scheduler
    records = [replace(episode.query, record_id=f'q{i}', stimulus_id=f'i{i}') for i in range(11)]
    sampler = TrainingEpisodeSampler(records, 42)
    bundle = tiny_bundle(tmp_path, episode)
    bundle.optimizer = make_optimizer(bundle)
    bundle.scheduler = make_scheduler(bundle, 11)
    run_flat_training_step(bundle, sampler.sample(), optimizer_step=True)
    save_checkpoint(bundle, tmp_path/'saved', split_manifest_identity='fixture', step=1, sampler=sampler)
    expected = [sampler.sample() for _ in range(5)]
    restored = TrainingEpisodeSampler(records, 999)
    resumed = tiny_bundle(tmp_path/'resumed', episode)
    resumed.optimizer = make_optimizer(resumed)
    resumed.scheduler = make_scheduler(resumed, 11)
    assert restore_checkpoint_state(resumed, tmp_path/'saved', split_manifest_identity='fixture',
                                    sampler=restored, resume_optimizer=True) == 1
    assert restored.cursor == 1 and restored.epoch == 1
    assert [restored.sample() for _ in range(5)] == expected
    assert resumed.scheduler.state_dict() == bundle.scheduler.state_dict()
    assert resumed.optimizer.param_groups[0]['lr'] == bundle.optimizer.param_groups[0]['lr']


def test_real_small_hf_epoch_test(tmp_path, episode):
    bundle = tiny_bundle(tmp_path, episode)
    bundle.config['test']['prediction']['execution'] = 'serial'  # retained reference API
    train, queries, manifest = evaluation_data(episode)
    result = evaluate_test_epoch(bundle, train, queries, manifest)
    assert result['test_episodes'] == 30
    assert set(result['test_by_k']) == {'1', '5', '10'}
    assert result['test_total'] == pytest.approx(result['test_where'] + result['test_flat'])
    assert all(p.grad is None for p in bundle.trainable_parameters())
    print(loop.format_epoch_summary({'epoch': 1, 'step': 2, **result}))
