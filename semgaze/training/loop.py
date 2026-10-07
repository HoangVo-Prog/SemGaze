"""Query-coverage epochs; optimizer windows never cross epoch boundaries."""
import json
import inspect
from pathlib import Path
import time
import torch
from semgaze.evaluation.test import EVAL_KEYS, evaluate_test_epoch
from semgaze.evaluation.predictions import predict_epoch, format_prediction_summary, resolve_prediction_settings
from semgaze.model.checkpoint import save_checkpoint
from semgaze.training.flat_step import run_flat_training_step, clip_and_check_gradients
from semgaze.training.batching import sample_optimizer_batches
from semgaze.evaluation.progress import RollingRate, format_eta, format_finish_time
from semgaze.model.visual_cache import InferenceVisualCache
from semgaze.evaluation.cache import ProjectedWhereCache


def _format_loss(value):
    return '...' if value is None else f'{value:.4f}'


def format_train_step(entry, max_steps=None):
    """Human-readable console rendering for a structured train-step event."""
    epoch_index = entry.get('epoch_index', entry.get('epoch', 0))
    epoch_total = entry.get('epoch_total', '?')
    epoch_step = entry.get('epoch_step', '?')
    epoch_steps = entry.get('epoch_steps', '?')
    progress = entry.get('epoch_progress_pct')
    progress_text = '...' if progress is None else f'{progress:.1f}%'
    learning_rate = entry.get('learning_rate')
    lr_text = '...' if learning_rate is None else f'{learning_rate:.3g}'
    text = (f"[TRAIN] epoch {epoch_index}/{epoch_total} | {epoch_step}/{epoch_steps} | "
            f"{progress_text} | global_step={entry['step']}\n"
            f"  loss={_format_loss(entry.get('loss_total'))} | "
            f"where={_format_loss(entry.get('loss_where'))} | "
            f"flat={_format_loss(entry.get('loss_flat'))} | lr={lr_text}\n"
            f"  rejected={entry.get('rejected_where_episodes', 0)} | "
            f"ETA epoch={format_eta(entry.get('epoch_eta_sec'))} | "
            f"finish~{format_finish_time(entry.get('epoch_eta_sec'))}")
    if entry.get('epoch_complete'):
        text += '\n        epoch complete'
    return text


def require_single_process(config):
    """Fail closed: this trainer does not implement uneven-rank DDP joins."""
    import os
    world = int(os.environ.get('WORLD_SIZE', '1'))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world = max(world, torch.distributed.get_world_size())
    if world != 1 or config['runtime'].get('world_size', 1) != 1:
        raise ValueError('distributed query coverage is unsupported by this trainer; use one process (no padding or dropping queries)')


def resolve_epoch_schedule(config, query_count):
    """Derive optimizer horizon from full coverage; max_steps only caps debugging."""
    require_single_process(config)
    t = config['training']
    if type(query_count) is not int or query_count < 1:
        raise ValueError('positive query_count required to derive a coverage epoch')
    episodes = t['per_device_train_batch_size'] * t['gradient_accumulation_steps']
    updates = (query_count + episodes - 1) // episodes
    t['query_count'] = query_count
    t['optimizer_updates_per_epoch'] = updates
    t['total_optimizer_updates'] = updates * t['num_train_epochs']
    if t['max_steps'] is not None:
        t['total_optimizer_updates'] = min(t['total_optimizer_updates'], t['max_steps'])
    return updates


def format_epoch_summary(entry):
    losses = ' | '.join(f'{key}={entry[key]:.6f}' for key in EVAL_KEYS)
    return (f"[EVAL][LOSS] Epoch {entry['epoch']} | step {entry['step']} | test: "
            f"{entry['test_queries']} unseen queries x K={entry.get('k_values', [])} ({entry['test_episodes']} episodes)\n"
            f"  response NLL (episode mean): {losses}\n"
            f"  complete traversals: {entry.get('test_loss_traversals', '?')}\n"
            '  WHAT/WHY/HOW: diagnostic sections of one flat response; generation-quality metrics not configured')


def run_training_loop(bundle, sampler, train_by_id, test_records, manifest, *,
                      split_manifest_identity, max_steps, save_every, start=0):
    t = bundle.config['training']
    updates_per_epoch = resolve_epoch_schedule(bundle.config, sampler.query_count)
    max_steps = t['total_optimizer_updates'] if max_steps is None else min(max_steps, t['total_optimizer_updates'])

    evaluation = bundle.config['evaluation']
    settings = resolve_prediction_settings(bundle.config) if evaluation['strategy'] != 'no' else evaluation['predictions']
    prediction_capacity = settings['train_batches'] * t['per_device_train_batch_size']
    recent_episodes = []
    checkpoint = bundle.config['checkpoint']
    logging = bundle.config['logging']
    if not 0 <= start <= max_steps or (save_every is not None and save_every < 1):
        raise ValueError('invalid resume step or save interval')
    window = t['per_device_train_batch_size'] * t['gradient_accumulation_steps']
    consumed_updates = ((sampler.epoch - 1) * updates_per_epoch +
                        (sampler.cursor + window - 1) // window) if sampler.epoch else 0
    if consumed_updates != start or (sampler.cursor % window and not sampler.epoch_complete()):
        raise ValueError('resume step/cursor do not match completed optimizer windows')
    output = Path(bundle.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Resume carries the full structured history into the new run directory.
    log_path = output / logging['filename']
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as stream:
        for entry in bundle.trainer_history:
            stream.write(json.dumps(entry, allow_nan=False) + '\n')
    with log_path.open('a', encoding='utf-8') as stream:
        epoch_rate = RollingRate()
        timed_epoch = None
        for step in range(start, max_steps):
            if sampler.epoch == 0 or sampler.epoch_complete():
                sampler.start_epoch()
            if timed_epoch != sampler.epoch:
                epoch_rate.reset()
                timed_epoch = sampler.epoch
            step_started = time.perf_counter()
            losses, rejected = [], 0
            component_losses = {'loss_where': [], 'loss_flat': []}
            diagnostic_losses = {key: [] for key in ('loss_what', 'loss_why', 'loss_how')}
            batches, sampled_episodes, rejected = sample_optimizer_batches(bundle, sampler)
            prediction_batch = sampled_episodes[-prediction_capacity:] if prediction_capacity else []
            weights = [len(batch.episodes) / len(sampled_episodes) for batch in batches]
            interval = t.get('gradient_diagnostics_every', 0)
            diagnose = step == start or (interval > 0 and (step + 1) % interval == 0)
            for micro, batch in enumerate(batches):
                result = run_flat_training_step(bundle, batch.episodes, zero_grad=(micro == 0),
                    loss_scale=weights[micro], where_batch=batch,
                    diagnostics=diagnose and micro == len(batches) - 1)
                losses.append(result['loss_total'] * weights[micro])
                for key in component_losses:
                    if key in result and result[key] is not None:
                        component_losses[key].append(result[key] * weights[micro])
                for key in diagnostic_losses:
                    if key in result and result[key] is not None:
                        diagnostic_losses[key].append(result[key] * weights[micro])
            if prediction_capacity:
                recent_episodes = (recent_episodes + prediction_batch)[-prediction_capacity:]
            clip_and_check_gradients(bundle)
            bundle.optimizer.step()
            bundle.scheduler.step()
            global_step = step + 1
            step_time = time.perf_counter() - step_started
            epoch_step = min(updates_per_epoch, (sampler.cursor + window - 1) // window)
            epoch_complete = sampler.epoch_complete()
            epoch_rate.update(epoch_step)
            epoch_eta = epoch_rate.eta(updates_per_epoch - epoch_step)
            entry = {'event': 'train_step', 'step': global_step,
                     'epoch': sampler.epoch - 1 + sampler.cursor / sampler.query_count,
                     'epoch_index': sampler.epoch, 'epoch_total': t['num_train_epochs'],
                     'epoch_step': epoch_step, 'epoch_steps': updates_per_epoch,
                     'epoch_progress_pct': 100 * epoch_step / updates_per_epoch,
                     'epoch_eta_sec': epoch_eta, 'epoch_complete': epoch_complete,
                     'query_cursor': sampler.cursor, 'query_count': sampler.query_count,
                     'loss_total': float(sum(losses)),
                     'rejected_where_episodes': rejected,
                     'learning_rate': float(bundle.optimizer.param_groups[0]['lr']),
                     'step_time_sec': step_time, 'physical_batch_size': t['per_device_train_batch_size'],
                     'gradient_accumulation_steps': t['gradient_accumulation_steps'],
                     'episodes_per_second': len(sampled_episodes) / step_time}
            for key, values in component_losses.items():
                entry[key] = float(sum(values)) if values else None
            for key, values in diagnostic_losses.items():
                if values:
                    entry[key] = float(sum(values))
            if (global_step % logging['every_steps'] == 0 or global_step == max_steps or
                    epoch_complete):
                bundle.trainer_history.append(entry)
                stream.write(json.dumps(entry, allow_nan=False) + '\n')
                stream.flush()
                print(format_train_step(entry, max_steps), flush=True)
            epoch_end = epoch_complete
            should_evaluate = ((evaluation['strategy'] == 'epoch' and epoch_end) or
                               (evaluation['strategy'] == 'steps' and
                                global_step % evaluation['eval_steps'] == 0))
            if should_evaluate:
                evaluation_started = time.perf_counter()
                print(f"[EVAL] starting epoch {sampler.epoch} evaluation", flush=True)
                has_predictions = settings['train_batches'] or settings['test_scope'] != 'none'
                cache_settings = bundle.config['test']['cache']
                shared_visual = InferenceVisualCache(
                    preprocessing=cache_settings['support_preprocessing'],
                    features=cache_settings['frozen_visual_features'],
                    max_entries=cache_settings['max_entries']) if has_predictions else None
                projected = ProjectedWhereCache(bundle=bundle,
                    split_manifest_identity=split_manifest_identity,
                    cycle_id=(sampler.epoch, step + 1)) if has_predictions else None
                try:
                    eval_kwargs = dict(visual_cache=shared_visual, projected_r_cache=projected,
                                       cycle_id=(sampler.epoch, step + 1))
                    eval_signature = inspect.signature(evaluate_test_epoch)
                    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in eval_signature.parameters.values()):
                        eval_kwargs = {k: v for k, v in eval_kwargs.items() if k in eval_signature.parameters}
                    summary = {'event': 'epoch_test', 'step': step + 1,
                               'epoch': sampler.epoch,
                               **evaluate_test_epoch(bundle, train_by_id, test_records, manifest, **eval_kwargs)}
                    if not has_predictions:
                        summary['evaluation_time_sec'] = time.perf_counter() - evaluation_started
                    bundle.trainer_history.append(summary)
                    stream.write(json.dumps(summary, allow_nan=False) + '\n')
                    stream.flush()
                    print(format_epoch_summary(summary), flush=True)
                    if has_predictions:
                        prediction_kwargs = dict(visual_cache=shared_visual, projected_r_cache=projected)
                        prediction_signature = inspect.signature(predict_epoch)
                        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in prediction_signature.parameters.values()):
                            prediction_kwargs = {k: v for k, v in prediction_kwargs.items() if k in prediction_signature.parameters}
                        predictions = predict_epoch(bundle, recent_episodes, train_by_id, test_records,
                            manifest, epoch=summary['epoch'], step=step + 1,
                            split_manifest_identity=split_manifest_identity,
                            **prediction_kwargs)
                        predictions['evaluation_time_sec'] = time.perf_counter() - evaluation_started
                        bundle.trainer_history.append(predictions)
                        stream.write(json.dumps(predictions, allow_nan=False) + '\n')
                        stream.flush()
                        print(format_prediction_summary(predictions), flush=True)
                        if predictions.get('metrics_artifact'):
                            metric_event = {'event': 'epoch_metrics', 'step': step + 1,
                                            'epoch': summary['epoch'], 'split': 'test',
                                            'metrics_file': predictions['metrics_artifact'].get('metrics_path'),
                                            'metric_namespaces': ('[EVAL][METRIC][WHERE]', '[EVAL][METRIC][SEM]')}
                            bundle.trainer_history.append(metric_event)
                            stream.write(json.dumps(metric_event, allow_nan=False) + '\n')
                            stream.flush()
                            print('[EVAL][METRIC][SEM] frozen semantic metrics complete', flush=True)
                finally:
                    if projected is not None:
                        projected.close()
                    if shared_visual is not None:
                        shared_visual.close()
                print('[EVAL] complete', flush=True)
            # Save after test and generation, including at epoch ends even when the
            # ordinary step-based save interval does not land on the boundary.
            if (epoch_end and checkpoint['save_at_epoch_end']) or (save_every is not None and (step + 1) % save_every == 0) or (step + 1 == max_steps and checkpoint['save_at_end']):
                save_checkpoint(bundle, output / f'checkpoint-{step + 1}',
                    split_manifest_identity=split_manifest_identity, step=step + 1, sampler=sampler)
