"""Optimizer-step epochs for the existing with-replacement episode sampler."""
import json
from pathlib import Path
import torch
from semgaze.evaluation.validation import EVAL_KEYS, validate_epoch
from semgaze.evaluation.predictions import predict_epoch, format_prediction_summary, resolve_prediction_settings
from semgaze.model.checkpoint import save_checkpoint
from semgaze.training.flat_step import run_flat_training_step
from semgaze.where.collator import WhereContextOverflowError


def resolve_epoch_schedule(config, max_steps, steps_per_epoch=None):
    steps = steps_per_epoch if steps_per_epoch is not None else config['training'].get('steps_per_epoch')
    if type(steps) is not int or steps < 1:
        raise ValueError('declare training.steps_per_epoch or --steps-per-epoch for the episodic sampler')
    if type(max_steps) is not int or max_steps < 1 or max_steps % steps:
        raise ValueError('max_steps must be a positive multiple of steps_per_epoch (complete epochs)')
    config['training']['steps_per_epoch'] = steps
    config.setdefault('evaluation', {'strategy': 'epoch', 'k_values': [1, 5, 10], 'loss_aggregation': 'episode_mean'})
    return steps


def format_epoch_summary(entry):
    losses = ' | '.join(f'{key}={entry[key]:.6f}' for key in EVAL_KEYS)
    return (f"Epoch {entry['epoch']} | step {entry['step']} | validation: "
            f"{entry['eval_queries']} seen queries x K=1,5,10 ({entry['eval_episodes']} episodes)\n"
            f"  response NLL (episode mean): {losses}\n"
            '  WHAT/WHY/HOW: diagnostic sections of one flat response; generation-quality metrics not configured')


def run_training_loop(bundle, sampler, train_by_id, validation_records, manifest, *,
                      split_manifest_identity, max_steps, save_every, start=0):
    t = bundle.config['training']
    steps_per_epoch = resolve_epoch_schedule(bundle.config, max_steps)
    resolve_prediction_settings(bundle.config)
    if not 0 <= start <= max_steps or save_every < 1:
        raise ValueError('invalid resume step or save interval')
    episodes_per_step = t['per_device_train_batch_size'] * t['gradient_accumulation_steps']
    output = Path(bundle.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Resume carries the full structured history into the new run directory.
    log_path = output / 'trainer_log.jsonl'
    with log_path.open('w', encoding='utf-8') as stream:
        for entry in bundle.trainer_history:
            stream.write(json.dumps(entry, allow_nan=False) + '\n')
    with log_path.open('a', encoding='utf-8') as stream:
        for step in range(start, max_steps):
            losses, rejected = [], 0
            prediction_batch = []
            for micro in range(episodes_per_step):
                while True:
                    episode = sampler.sample()
                    try:
                        result = run_flat_training_step(bundle, episode, zero_grad=(micro == 0),
                            loss_scale=1 / episodes_per_step)
                        losses.append(float(result['loss_total']))
                        # Retain only the first batch, excluding accumulated later
                        # batches. No tensors/graphs or extra sampler draws retained.
                        if len(prediction_batch) < t['per_device_train_batch_size']:
                            prediction_batch.append(episode)
                        break
                    except WhereContextOverflowError:
                        rejected += 1
                        if rejected >= 1000:
                            raise RuntimeError('1000 complete WHERE episode overflows; inspect context feasibility')
            torch.nn.utils.clip_grad_norm_(bundle.trainable_parameters(), t['max_grad_norm'], error_if_nonfinite=True)
            bundle.optimizer.step()
            bundle.scheduler.step()
            entry = {'event': 'train_step', 'step': step + 1,
                     'epoch': (step + 1) / steps_per_epoch,
                     'loss_total': sum(losses) / len(losses), 'rejected_where_episodes': rejected}
            bundle.trainer_history.append(entry)
            stream.write(json.dumps(entry, allow_nan=False) + '\n')
            stream.flush()
            print(json.dumps(entry), flush=True)
            epoch_end = (step + 1) % steps_per_epoch == 0
            if epoch_end:
                print(f'Epoch {(step + 1) // steps_per_epoch} complete; validating all seen queries at K=1,5,10...', flush=True)
                summary = {'event': 'epoch_validation', 'step': step + 1,
                           'epoch': (step + 1) // steps_per_epoch,
                           **validate_epoch(bundle, train_by_id, validation_records, manifest)}
                bundle.trainer_history.append(summary)
                stream.write(json.dumps(summary, allow_nan=False) + '\n')
                stream.flush()
                print(format_epoch_summary(summary), flush=True)
                predictions = predict_epoch(bundle, prediction_batch, train_by_id, validation_records,
                    manifest, epoch=summary['epoch'], step=step + 1,
                    split_manifest_identity=split_manifest_identity)
                bundle.trainer_history.append(predictions)
                stream.write(json.dumps(predictions, allow_nan=False) + '\n')
                stream.flush()
                print(format_prediction_summary(predictions), flush=True)
            # Save after validation and generation, including at epoch ends even when the
            # ordinary step-based save interval does not land on the boundary.
            if epoch_end or (step + 1) % save_every == 0 or step + 1 == max_steps:
                save_checkpoint(bundle, output / f'checkpoint-{step + 1}',
                    split_manifest_identity=split_manifest_identity, step=step + 1, sampler=sampler)
