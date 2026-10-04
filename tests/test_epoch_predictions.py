"""Epoch generation coverage, canonical conditioning and training isolation."""
import json
import random
from dataclasses import replace
from pathlib import Path
import pytest
import torch

pytest.importorskip('peft')
from test_model_path import tiny_bundle, episode
from test_epoch_validation import validation_data, FixedSampler
from semgaze.evaluation import predictions
from semgaze.evaluation.predictions import predict_epoch, resolve_prediction_settings, format_prediction_summary
from semgaze.evaluation import flat as flat_evaluation
from semgaze.model.config import validate_config
from semgaze.training import loop
from semgaze.training.flat_step import run_flat_training_step, make_optimizer
from semgaze.semantic.flat.target import build_flat_target
from semgaze.where.serialization import serialize_xyd_record
from semgaze.where.collator import collate_where


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def test_prediction_coverage_and_raw_gt_pred_files(tmp_path, episode, monkeypatch):
    bundle = tiny_bundle(tmp_path, episode)
    resolve_prediction_settings(bundle.config, 8)
    bundle.config['training']['per_device_train_batch_size'] = 2
    train, queries, manifest = validation_data(episode)
    queries.append(replace(episode.query, record_id='second', stimulus_id='second-image'))
    manifest['validation_stimulus_ids'].append('second-image')
    queries.append(replace(episode.query, subject=7, record_id='unseen'))
    calls = []
    def where(bundle, ep):
        assert not bundle.model.training and not torch.is_grad_enabled()
        calls.append(('where', ep.query.record_id, len(ep.supports)))
        return {'text': 'malformed raw WHERE', 'canonical_format_valid': False}
    def flat(bundle, ep, *, generation_budget):
        assert generation_budget == 8
        calls.append(('semantic', ep.query.record_id, len(ep.supports)))
        return {'text': 'unrepaired\nWHAT: text', 'state_source': 'teacher_forced_GT_XYD'}
    monkeypatch.setattr(predictions, 'evaluate_where_episode', where)
    monkeypatch.setattr(predictions, 'evaluate_flat_episode', flat)
    result = predict_epoch(bundle, [episode, episode], train, queries, manifest,
        epoch=1, step=2, split_manifest_identity='fixture')
    train_rows = read_records(result['prediction_files']['train'])
    val_rows = read_records(result['prediction_files']['validation'])
    assert len(train_rows) == 2
    assert len(val_rows) == result['validation_prediction_episodes'] == 6
    assert result['validation_prediction_queries'] == 2
    assert [(r['query_id'], r['k']) for r in val_rows] == [
        (q.record_id, k) for k in (1, 5, 10) for q in queries[:2]]
    assert [c[0] for c in calls] == ['where', 'semantic'] * 8
    for row in train_rows + val_rows:
        assert row['WHERE'] == {'GT': serialize_xyd_record(episode.query), 'PRED': 'malformed raw WHERE'}
        assert row['SEMANTIC'] == {'GT': build_flat_target(episode.query.semantic), 'PRED': 'unrepaired\nWHAT: text'}
        assert row['split_manifest_identity'] == 'fixture'
        assert row['epoch'] == 1 and row['step'] == 2
    for row in val_rows:
        assert row['support_ids'] == [e['record_id'] for e in manifest['validation_supports']['1'][str(row['k'])]]
    assert 'train: 1 batch (2 queries)' in format_prediction_summary(result)
    assert 'validation: 2 seen queries x K=1,5,10 (6 episodes)' in format_prediction_summary(result)
    assert not list(tmp_path.rglob('*.partial'))


def test_real_hf_predictions_generate_separately_and_preserve_training(tmp_path, episode, monkeypatch):
    bundle = tiny_bundle(tmp_path, episode)
    resolve_prediction_settings(bundle.config, 4)  # explicit small test budget
    run_flat_training_step(bundle, episode)
    train, queries, manifest = validation_data(episode)
    parameters = [(p, p.detach().clone(), None if p.grad is None else p.grad.clone())
                  for root in (bundle.model, bundle.projector) for p in root.parameters()]
    bundle.model.get_base_model().model.vision_tower.eval()
    modes = [m.training for root in (bundle.model, bundle.projector) for m in root.modules()]
    torch_rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    generate = bundle.model.generate
    generation_calls, gt_state_queries = [], []
    def observed_generate(**kwargs):
        assert 'labels' not in kwargs
        assert not torch.is_grad_enabled() and not bundle.model.training
        assert kwargs['do_sample'] is False
        if 'input_ids' in kwargs:
            generation_calls.append('WHERE')
            # Every free-running prefix ends before the query assistant target.
            ep = episode if len(generation_calls) == 1 else gt_episodes[len(gt_state_queries)]
            expected = collate_where(bundle.processor, ep, bundle.end_fix_id,
                                     bundle.context_limit, teacher_forcing=False)
            assert torch.equal(kwargs['input_ids'], expected.inputs['input_ids'])
        else:
            generation_calls.append('SEMANTIC')
            assert 'inputs_embeds' in kwargs and kwargs['max_new_tokens'] == 4
        return generate(**kwargs)
    from semgaze.data.fewshot import frozen_episode
    gt_episodes = [episode] + [frozen_episode(queries[0], train, manifest, k) for k in (1, 5, 10)]
    forward_where = flat_evaluation.forward_where
    def observed_gt_states(bundle, ep):
        gt_state_queries.append(ep)
        return forward_where(bundle, ep)
    monkeypatch.setattr(bundle.model, 'generate', observed_generate)
    monkeypatch.setattr(flat_evaluation, 'forward_where', observed_gt_states)
    prepare = flat_evaluation.prepare_flat_inputs
    def observed_semantic_inputs(bundle, query, states, *, generation_budget):
        inputs, positions, native = prepare(bundle, query, states, generation_budget=generation_budget)
        assert native.response_length == 0
        assert (native.inputs['labels'] == -100).all()
        assert build_flat_target(query.semantic) not in native.rendered_text
        assert len(positions) == len(query.x_px)
        return inputs, positions, native
    monkeypatch.setattr(flat_evaluation, 'prepare_flat_inputs', observed_semantic_inputs)
    result = predict_epoch(bundle, [episode], train, queries, manifest,
        epoch=1, step=2, split_manifest_identity='fixture')
    assert generation_calls == ['WHERE', 'SEMANTIC'] * 4
    assert gt_state_queries == gt_episodes
    for split in ('train', 'validation'):
        for row in read_records(result['prediction_files'][split]):
            assert row['where_generation']['oracle_length_conditioned'] is True
            assert row['semantic_generation']['state_source'] == 'teacher_forced_GT_XYD'
            assert row['semantic_generation']['gold_groups_supplied'] is True
            assert row['semantic_generation']['inserted_state_count'] == 4
            assert isinstance(row['SEMANTIC']['PRED'], str)
    assert modes == [m.training for root in (bundle.model, bundle.projector) for m in root.modules()]
    assert torch.equal(torch_rng, torch.get_rng_state()) and python_rng == random.getstate()
    for p, value, grad in parameters:
        assert torch.equal(p, value)
        assert p.grad is None if grad is None else torch.equal(p.grad, grad)
    print(format_prediction_summary(result))


def test_train_predictions_stop_at_first_batch_not_accumulation(tmp_path, episode, monkeypatch):
    bundle = tiny_bundle(tmp_path, episode)
    resolve_prediction_settings(bundle.config, 4)
    loop.resolve_epoch_schedule(bundle.config, 2, 2)
    bundle.config['training'].update(per_device_train_batch_size=2, gradient_accumulation_steps=3)
    bundle.optimizer = make_optimizer(bundle)
    bundle.scheduler = torch.optim.lr_scheduler.LambdaLR(bundle.optimizer, lambda _: 1)
    class SequentialSampler(FixedSampler):
        def sample(self):
            ep = super().sample()
            return replace(ep, query=replace(ep.query, record_id=f'train-query-{self.calls}'))
    sampler = SequentialSampler(episode)
    train, queries, manifest = validation_data(episode)
    monkeypatch.setattr(loop, 'run_flat_training_step', lambda *a, **kw: {'loss_total': 1.0})
    monkeypatch.setattr(loop, 'validate_epoch', lambda *a: {
        **{key: 1.0 for key in loop.EVAL_KEYS}, 'eval_queries': 1, 'eval_episodes': 3})
    monkeypatch.setattr(predictions, 'evaluate_where_episode', lambda *a: {'text': 'where'})
    monkeypatch.setattr(predictions, 'evaluate_flat_episode', lambda *a, **kw: {'text': 'semantic'})
    loop.run_training_loop(bundle, sampler, train, queries, manifest,
        split_manifest_identity='fixture', max_steps=2, save_every=2)
    assert sampler.calls == 12  # exactly the training draws, zero generation draws
    summary = bundle.trainer_history[-1]
    assert summary['event'] == 'epoch_predictions'
    assert [r['query_id'] for r in read_records(summary['prediction_files']['train'])] == [
        'train-query-7', 'train-query-8']  # first batch of final step, no second batch
    assert len(read_records(summary['prediction_files']['validation'])) == 3


def test_prediction_failure_restores_modes_rng_and_keeps_partial_file(tmp_path, episode, monkeypatch):
    bundle = tiny_bundle(tmp_path, episode)
    resolve_prediction_settings(bundle.config, 4)
    train, queries, manifest = validation_data(episode)
    rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    modes = [m.training for root in (bundle.model, bundle.projector) for m in root.modules()]
    def fail(*args):
        torch.rand(3)
        random.random()
        raise RuntimeError('generation failed')
    monkeypatch.setattr(predictions, 'evaluate_where_episode', fail)
    with pytest.raises(RuntimeError, match='generation failed'):
        predict_epoch(bundle, [episode], train, queries, manifest,
            epoch=1, step=2, split_manifest_identity='fixture')
    assert torch.equal(rng, torch.get_rng_state()) and python_rng == random.getstate()
    assert modes == [m.training for root in (bundle.model, bundle.projector) for m in root.modules()]
    assert not bundle.trainer_history
    assert (tmp_path / 'predictions/epoch-0001/train.jsonl.partial').exists()
    assert not (tmp_path / 'predictions/epoch-0001/train.jsonl').exists()


@pytest.mark.parametrize('budget', [None, 0, -1, True])
def test_prediction_budget_is_explicit(tmp_path, episode, budget):
    config = tiny_bundle(tmp_path, episode).config
    with pytest.raises(ValueError, match='semantic_max_new_tokens'):
        resolve_prediction_settings(config, budget)
    resolve_prediction_settings(config, 8)
    validate_config(config)
    config['evaluation']['predictions']['train_batches'] = 2
    with pytest.raises(ValueError, match='one train batch'):
        validate_config(config)
