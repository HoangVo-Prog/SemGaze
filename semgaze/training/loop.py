"""Query-coverage epochs; optimizer windows never cross epoch boundaries."""
import json
from pathlib import Path
import time
import torch
from semgaze.evaluation.test import EVAL_KEYS, evaluate_test_epoch
from semgaze.evaluation.predictions import predict_epoch, format_prediction_summary, resolve_prediction_settings
from semgaze.model.checkpoint import save_checkpoint
from semgaze.training.flat_step import run_flat_training_step, clip_and_check_gradients
from semgaze.training.batching import sample_optimizer_batches


def _format_loss(value):
    return '...' if value is None else f'{value:.4f}'


def format_train_step(entry, max_steps=None):
    """Human-readable console rendering for a structured train-step event."""
    step_text = str(entry['step']) if max_steps is None else f"{entry['step']}/{max_steps}"
    learning_rate = entry.get('learning_rate')
    lr_text = '...' if learning_rate is None else f'{learning_rate:.3g}'
    return (f"[TRAIN] step {step_text} | epoch {entry['epoch']:.3f}\n"
            f"  loss={_format_loss(entry.get('loss_total'))} | "
            f"where={_format_loss(entry.get('loss_where'))} | "
            f"flat={_format_loss(entry.get('loss_flat'))} | lr={lr_text}\n"
            f"  rejected={entry.get('rejected_where_episodes', 0)} | "
            f"time={entry.get('step_time_sec', 0.0):.2f}s")


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
            f"  test loss time: {entry.get('test_time_sec', 0.0):.2f}s\n"
            '  WHAT/WHY/HOW: diagnostic sections of one flat response; generation-quality metrics not configured')


def run_training_loop(bundle, sampler, train_by_id, test_records, manifest, *,
                      split_manifest_identity, max_steps, save_every, start=0):
    t = bundle.config['training']
    updates_per_epoch = resolve_epoch_schedule(bundle.config, sampler.query_count)
    max_steps = t['total_optimizer_updates'] if max_steps is None else min(max_steps, t['total_optimizer_updates'])
    if bundle.config['evaluation']['strategy'] not in ('epoch', 'no'):
        raise ValueError('COCO evaluation must follow completed coverage epochs')
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
        for step in range(start, max_steps):
            if sampler.epoch == 0 or sampler.epoch_complete():
                sampler.start_epoch()
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
            entry = {'event': 'train_step', 'step': global_step,
                     'epoch': sampler.epoch - 1 + sampler.cursor / sampler.query_count,
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
            if global_step % logging['every_steps'] == 0 or global_step == max_steps:
                bundle.trainer_history.append(entry)
                stream.write(json.dumps(entry, allow_nan=False) + '\n')
                stream.flush()
                print(format_train_step(entry, max_steps), flush=True)
            epoch_end = sampler.epoch_complete()
            should_evaluate = ((evaluation['strategy'] == 'epoch' and epoch_end) or
                               (evaluation['strategy'] == 'steps' and
                                global_step % evaluation['eval_steps'] == 0))
            if should_evaluate:
                evaluation_started = time.perf_counter()
                print(f"[EVAL] step {step + 1} | starting test evaluation", flush=True)
                summary = {'event': 'epoch_test', 'step': step + 1,
                           'epoch': sampler.epoch,
                           **evaluate_test_epoch(bundle, train_by_id, test_records, manifest)}
                has_predictions = settings['train_batches'] or settings['test_scope'] != 'none'
                if not has_predictions:
                    summary['evaluation_time_sec'] = time.perf_counter() - evaluation_started
                bundle.trainer_history.append(summary)
                stream.write(json.dumps(summary, allow_nan=False) + '\n')
                stream.flush()
                print(format_epoch_summary(summary), flush=True)
                if has_predictions:
                    predictions = predict_epoch(bundle, recent_episodes, train_by_id, test_records,
                        manifest, epoch=summary['epoch'], step=step + 1,
                        split_manifest_identity=split_manifest_identity)
                    predictions['evaluation_time_sec'] = time.perf_counter() - evaluation_started
                    bundle.trainer_history.append(predictions)
                    stream.write(json.dumps(predictions, allow_nan=False) + '\n')
                    stream.flush()
                    print(format_prediction_summary(predictions), flush=True)
                print(f"[EVAL] done | total_time={time.perf_counter() - evaluation_started:.2f}s", flush=True)
            # Save after test and generation, including at epoch ends even when the
            # ordinary step-based save interval does not land on the boundary.
            if (epoch_end and checkpoint['save_at_epoch_end']) or (save_every is not None and (step + 1) % save_every == 0) or (step + 1 == max_steps and checkpoint['save_at_end']):
                save_checkpoint(bundle, output / f'checkpoint-{step + 1}',
                    split_manifest_identity=split_manifest_identity, step=step + 1, sampler=sampler)
