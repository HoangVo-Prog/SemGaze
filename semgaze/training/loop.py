"""Optimizer-step epochs for the existing with-replacement episode sampler."""
import json
from pathlib import Path
import time
import torch
from semgaze.evaluation.validation import EVAL_KEYS, validate_epoch
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


def resolve_epoch_schedule(config, max_steps, steps_per_epoch=None):
    steps = steps_per_epoch if steps_per_epoch is not None else config['training'].get('steps_per_epoch')
    if type(steps) is not int or steps < 1:
        raise ValueError('declare training.steps_per_epoch or --steps-per-epoch for the episodic sampler')
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError('max_steps must be positive')
    config['training']['steps_per_epoch'] = steps
    return steps


def format_epoch_summary(entry):
    losses = ' | '.join(f'{key}={entry[key]:.6f}' for key in EVAL_KEYS)
    return (f"[EVAL][LOSS] Epoch {entry['epoch']} | step {entry['step']} | validation: "
            f"{entry['eval_queries']} seen queries x K={entry.get('k_values', [])} ({entry['eval_episodes']} episodes)\n"
            f"  response NLL (episode mean): {losses}\n"
            f"  validation loss time: {entry.get('validation_time_sec', 0.0):.2f}s\n"
            '  WHAT/WHY/HOW: diagnostic sections of one flat response; generation-quality metrics not configured')


def run_training_loop(bundle, sampler, train_by_id, validation_records, manifest, *,
                      split_manifest_identity, max_steps, save_every, start=0):
    t = bundle.config['training']
    steps_per_epoch = resolve_epoch_schedule(bundle.config, max_steps)
    evaluation = bundle.config['evaluation']
    settings = resolve_prediction_settings(bundle.config) if evaluation['strategy'] != 'no' else evaluation['predictions']
    prediction_capacity = settings['train_batches'] * t['per_device_train_batch_size']
    recent_episodes = []
    checkpoint = bundle.config['checkpoint']
    logging = bundle.config['logging']
    if not 0 <= start <= max_steps or (save_every is not None and save_every < 1):
        raise ValueError('invalid resume step or save interval')
    episodes_per_step = t['per_device_train_batch_size'] * t['gradient_accumulation_steps']
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
            step_started = time.perf_counter()
            losses, rejected = [], 0
            component_losses = {'loss_where': [], 'loss_flat': []}
            diagnostic_losses = {key: [] for key in ('loss_what', 'loss_why', 'loss_how')}
            batches, sampled_episodes, rejected = sample_optimizer_batches(bundle, sampler)
            prediction_batch = sampled_episodes[:prediction_capacity]
            interval = t.get('gradient_diagnostics_every', 0)
            diagnose = step == start or (interval > 0 and (step + 1) % interval == 0)
            for micro, batch in enumerate(batches):
                result = run_flat_training_step(bundle, batch.episodes, zero_grad=(micro == 0),
                    loss_scale=1 / t['gradient_accumulation_steps'], where_batch=batch,
                    diagnostics=diagnose and micro == len(batches) - 1)
                losses.append(result['loss_total'])
                for key in component_losses:
                    if key in result and result[key] is not None:
                        component_losses[key].append(result[key])
                for key in diagnostic_losses:
                    if key in result and result[key] is not None:
                        diagnostic_losses[key].append(result[key])
            if prediction_capacity:
                recent_episodes = (recent_episodes + prediction_batch)[-prediction_capacity:]
            clip_and_check_gradients(bundle)
            bundle.optimizer.step()
            bundle.scheduler.step()
            step_time = time.perf_counter() - step_started
            entry = {'event': 'train_step', 'step': step + 1,
                     'epoch': (step + 1) / steps_per_epoch,
                     'loss_total': float(sum(losses) / len(losses)),
                     'rejected_where_episodes': rejected,
                     'learning_rate': float(bundle.optimizer.param_groups[0]['lr']),
                     'step_time_sec': step_time, 'physical_batch_size': t['per_device_train_batch_size'],
                     'gradient_accumulation_steps': t['gradient_accumulation_steps'],
                     'episodes_per_second': episodes_per_step / step_time}
            for key, values in component_losses.items():
                entry[key] = float(sum(values) / len(values)) if values else None
            for key, values in diagnostic_losses.items():
                if values:
                    entry[key] = float(sum(values) / len(values))
            if (step + 1) % logging['every_steps'] == 0 or step + 1 == max_steps:
                bundle.trainer_history.append(entry)
                stream.write(json.dumps(entry, allow_nan=False) + '\n')
                stream.flush()
                print(format_train_step(entry, max_steps), flush=True)
            epoch_end = (step + 1) % steps_per_epoch == 0
            should_evaluate = (evaluation['strategy'] == 'epoch' and epoch_end) or (evaluation['strategy'] == 'steps' and (step + 1) % evaluation['every_steps'] == 0)
            if should_evaluate:
                evaluation_started = time.perf_counter()
                print(f"[EVAL] step {step + 1} | starting validation", flush=True)
                summary = {'event': 'epoch_validation', 'step': step + 1,
                           'epoch': (step + 1) // steps_per_epoch,
                           **validate_epoch(bundle, train_by_id, validation_records, manifest)}
                has_predictions = settings['train_batches'] or settings['validation_scope'] != 'none'
                if not has_predictions:
                    summary['evaluation_time_sec'] = time.perf_counter() - evaluation_started
                bundle.trainer_history.append(summary)
                stream.write(json.dumps(summary, allow_nan=False) + '\n')
                stream.flush()
                print(format_epoch_summary(summary), flush=True)
                if has_predictions:
                    if len(recent_episodes) != prediction_capacity:
                        raise ValueError('not enough training batches before evaluation for configured predictions')
                    predictions = predict_epoch(bundle, recent_episodes, train_by_id, validation_records,
                        manifest, epoch=summary['epoch'], step=step + 1,
                        split_manifest_identity=split_manifest_identity)
                    predictions['evaluation_time_sec'] = time.perf_counter() - evaluation_started
                    bundle.trainer_history.append(predictions)
                    stream.write(json.dumps(predictions, allow_nan=False) + '\n')
                    stream.flush()
                    print(format_prediction_summary(predictions), flush=True)
                print(f"[EVAL] done | total_time={time.perf_counter() - evaluation_started:.2f}s", flush=True)
            # Save after validation and generation, including at epoch ends even when the
            # ordinary step-based save interval does not land on the boundary.
            if (epoch_end and checkpoint['save_at_epoch_end']) or (save_every is not None and (step + 1) % save_every == 0) or (step + 1 == max_steps and checkpoint['save_at_end']):
                save_checkpoint(bundle, output / f'checkpoint-{step + 1}',
                    split_manifest_identity=split_manifest_identity, step=step + 1, sampler=sampler)
