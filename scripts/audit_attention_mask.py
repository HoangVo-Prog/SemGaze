#!/usr/bin/env python3
"""Audit whether WHERE right-padding masks change the Qwen3 CUDA path.

This is a diagnostic-only A/B experiment. It reuses the real SemGaze flat
training pipeline, a real pre-collated B=2 same-K training batch, the real WHERE
forward, semantic forward, projector, loss computation and one joint backward.
The only candidate change is that the WHERE text LM receives no attention mask.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_a100 import (  # noqa: E402
    machine_metadata,
    memory_snapshot,
    save_json,
    validate_real_config,
    load_real_sampler,
)
from semgaze.model.config import load_config, resolve_config  # noqa: E402
from semgaze.model.build import build_flat_model_bundle  # noqa: E402
from semgaze.training.batching import sample_optimizer_batches  # noqa: E402
from semgaze.training.flat_step import run_flat_training_step  # noqa: E402
from semgaze.training.profiling import runtime_metadata  # noqa: E402
from semgaze.where.collator import pack_where_batch  # noqa: E402

SCHEMA_VERSION = 1
VARIANTS = (
    ('baseline', False, 'current WHERE attention-mask behavior'),
    ('causal_only', True, 'diagnostic WHERE attention_mask=None'),
)
PROFILER_PATTERNS = (
    'scaled_dot_product_flash_attention',
    'scaled_dot_product_efficient_attention',
    'scaled_dot_product_attention',
    'bmm',
    'matmul',
    'cublaslt',
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def finite_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def tensor_digest(tensor):
    """Content hash for reproducibility metadata, not numerical comparison."""
    import torch
    host = tensor.detach().cpu().contiguous()
    header = json.dumps({'shape': list(host.shape), 'dtype': str(host.dtype)}, sort_keys=True).encode()
    if host.is_floating_point():
        payload = host.to(torch.float32).numpy().tobytes()
    else:
        payload = host.numpy().tobytes()
    value = hashlib.sha256()
    value.update(header)
    value.update(payload)
    return value.hexdigest()


def capture_rng_state(device):
    import torch
    state = {
        'python': random.getstate(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state(device),
    }
    try:
        import numpy as np
    except Exception:
        state['numpy'] = None
    else:
        state['numpy'] = np.random.get_state()
    return state


def restore_rng_state(state, device):
    import torch
    random.setstate(state['python'])
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state(state['torch_cuda'], device)
    if state.get('numpy') is not None:
        try:
            import numpy as np
        except Exception:
            pass
        else:
            np.random.set_state(state['numpy'])


def clear_gradients(bundle):
    bundle.model.zero_grad(set_to_none=True)
    bundle.projector.zero_grad(set_to_none=True)


def assert_where_causal_only_safe(where_batch, *, pad_token_id, end_fix_id):
    """Fail unless no consumed WHERE token can depend on a right-pad token."""
    import torch
    mask = where_batch.inputs['attention_mask']
    labels = where_batch.inputs['labels']
    input_ids = where_batch.inputs['input_ids']
    if mask.ndim != 2 or labels.shape != mask.shape or input_ids.shape != mask.shape:
        raise AssertionError('WHERE input_ids, attention_mask and labels must share [B,L] shape')
    lengths = mask.sum(dim=1)
    expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < lengths[:, None]
    if not torch.equal(mask.bool(), expected):
        raise AssertionError('WHERE attention_mask is not strict right-padding')
    if not torch.all(labels[~mask.bool()] == -100):
        raise AssertionError('WHERE labels at padding positions are not all -100')
    if not torch.all(input_ids[~mask.bool()] == pad_token_id):
        raise AssertionError('WHERE input_ids at padding positions are not all pad_token_id')
    metadata_lengths = torch.tensor([row['where_length'] for row in where_batch.metadata], device=mask.device)
    if not torch.equal(lengths.to(metadata_lengths.dtype), metadata_lengths):
        raise AssertionError('WHERE metadata lengths differ from attention-mask prefix lengths')
    valid_label_positions = labels != -100
    if bool((valid_label_positions & ~mask.bool()).any()):
        raise AssertionError('supervised WHERE labels include padding positions')
    if bool(valid_label_positions[:, 0].any()):
        raise AssertionError('first WHERE token cannot be supervised by causal loss')
    predictor_rows, predictor_positions = (labels[:, 1:] != -100).nonzero(as_tuple=True)
    if predictor_rows.numel() and not bool((predictor_positions < lengths[predictor_rows]).all()):
        raise AssertionError('causal predictor states for WHERE loss are not entirely inside real prefixes')
    consumed = (input_ids == end_fix_id) & valid_label_positions
    if bool((consumed & ~mask.bool()).any()):
        raise AssertionError('extracted query END_FIX states include padding positions')
    for index, (count, row) in enumerate(zip(consumed.sum(dim=1).tolist(), where_batch.metadata)):
        if count != row['fixation_count']:
            raise AssertionError(f'episode {index}: consumed END_FIX count differs from query fixation count')
        if row['response_start'] + row['response_length'] > row['where_length']:
            raise AssertionError(f'episode {index}: response span extends past real WHERE prefix')
    return {
        'strict_right_padding': True,
        'prefix_lengths': lengths.tolist(),
        'padding_tokens': int(mask.numel() - lengths.sum().item()),
        'labels_on_padding_all_ignored': True,
        'input_padding_token_id': int(pad_token_id),
        'consumed_end_fix_counts': consumed.sum(dim=1).tolist(),
    }


class CudaEventProfiler:
    def __init__(self, torch_module, device):
        self.torch = torch_module
        self.device = device
        self.events = []

    @contextmanager
    def stage(self, name):
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((name, start, end))

    def finish(self):
        self.torch.cuda.synchronize(self.device)
        times = {}
        for name, start, end in self.events:
            times[name] = times.get(name, 0.0) + start.elapsed_time(end) / 1000.0
        return times


class RecordFunctionProfiler:
    def __init__(self, torch_module):
        self.torch = torch_module

    def stage(self, name):
        return self.torch.profiler.record_function(name)


def select_real_b2_batch(bundle, sampler, *, batch_case, max_windows):
    if bundle.config['training']['per_device_train_batch_size'] != 2:
        raise ValueError('attention-mask audit requires physical B=2')
    rejected_total = 0
    inspected = 0
    equal_candidates = {}
    equal_candidate_count = 0
    for window in range(max_windows):
        if sampler.epoch == 0 or sampler.epoch_complete():
            sampler.start_epoch()
        batches, accepted, rejected = sample_optimizer_batches(bundle, sampler, sampling_group_size=2)
        rejected_total += rejected
        for batch in batches:
            inspected += 1
            lengths = [row['where_length'] for row in batch.metadata]
            if batch_case == 'equal':
                if len(batch.episodes) == 2 and len({len(episode.supports) for episode in batch.episodes}) == 1 and lengths[0] == lengths[1]:
                    return batch, {
                        'search_windows': window + 1,
                        'physical_batches_inspected': inspected,
                        'rejected_where_episodes': rejected_total,
                        'sampler_epoch': sampler.epoch,
                        'sampler_cursor': sampler.cursor,
                        'selection_case': batch_case,
                        'repacked_from_real_samples': False,
                    }
                # Exact equal-length physical batches are uncommon when queries
                # are sampled independently. Keep every real collated sample,
                # including singleton K buckets, and repack the first matching
                # same-K/same-length pair across source windows.
                for episode, sample in zip(batch.episodes, batch.samples):
                    equal_candidate_count += 1
                    key = (len(episode.supports), int(sample.inputs['input_ids'].shape[1]))
                    previous = equal_candidates.get(key)
                    if previous is None:
                        equal_candidates[key] = (episode, sample, batch.image_cache, window + 1)
                        continue
                    first_episode, first_sample, first_cache, first_window = previous
                    if first_episode.query.record_id == episode.query.record_id:
                        continue
                    cache = dict(first_cache)
                    cache.update(batch.image_cache)
                    matched = pack_where_batch(
                        bundle.processor,
                        [first_episode, episode],
                        [first_sample, sample],
                        cache,
                    )
                    return matched, {
                        'search_windows': window + 1,
                        'physical_batches_inspected': inspected,
                        'rejected_where_episodes': rejected_total,
                        'sampler_epoch': sampler.epoch,
                        'sampler_cursor': sampler.cursor,
                        'selection_case': batch_case,
                        'repacked_from_real_samples': True,
                        'source_windows': [first_window, window + 1],
                        'source_length_key': list(key),
                        'real_collated_samples_considered': equal_candidate_count,
                    }
                continue
            if len(batch.episodes) != 2:
                continue
            if len({len(episode.supports) for episode in batch.episodes}) != 1:
                continue
            unequal = lengths[0] != lengths[1]
            if batch_case == 'unequal' and not unequal:
                continue
            return batch, {
                'search_windows': window + 1,
                'physical_batches_inspected': inspected,
                'rejected_where_episodes': rejected_total,
                'sampler_epoch': sampler.epoch,
                'sampler_cursor': sampler.cursor,
                'selection_case': batch_case,
            }
    if batch_case == 'equal':
        raise RuntimeError(
            f'no B=2 equal-length same-K pair found among {equal_candidate_count} '
            f'real collated samples and {len(equal_candidates)} unique '
            f'(K, sequence_length) keys in {max_windows} optimizer windows'
        )
    raise RuntimeError(f'no B=2 {batch_case}-length same-K training batch found in {max_windows} optimizer windows')


def batch_report(batch):
    lengths = [row['where_length'] for row in batch.metadata]
    maximum = max(lengths)
    padding = [maximum - value for value in lengths]
    return {
        'B': len(batch.episodes),
        'K': [len(episode.supports) for episode in batch.episodes],
        'same_K': len({len(episode.supports) for episode in batch.episodes}) == 1,
        'record_ids': [episode.query.record_id for episode in batch.episodes],
        'subjects': [episode.query.subject for episode in batch.episodes],
        'support_ids': [[support.record_id for support in episode.supports] for episode in batch.episodes],
        'sequence_lengths': lengths,
        'max_sequence_length': maximum,
        'padding_tokens_by_episode': padding,
        'total_padding_tokens': sum(padding),
        'padding_fraction': sum(padding) / (len(lengths) * maximum),
        'fixation_counts': [row['fixation_count'] for row in batch.metadata],
        'image_counts': [row['image_count'] for row in batch.metadata],
        'image_paths': [list(row['image_paths']) for row in batch.metadata],
        'tensor_sha256': {key: tensor_digest(batch.inputs[key]) for key in
                          ('input_ids', 'attention_mask', 'labels', 'pixel_values')},
    }


def collect_gradients(bundle):
    lora, projector = {}, {}
    for name, parameter in bundle.model.named_parameters():
        if parameter.requires_grad and 'lora_' in name:
            lora[name] = None if parameter.grad is None else parameter.grad.detach().clone()
    for name, parameter in bundle.projector.named_parameters():
        projector['projector.' + name] = None if parameter.grad is None else parameter.grad.detach().clone()
    rows = {
        'input_row': None if bundle.input_row.grad is None else bundle.input_row.grad.detach().clone(),
    }
    if bundle.output_row is bundle.input_row:
        rows['output_row'] = rows['input_row']
    else:
        rows['output_row'] = None if bundle.output_row.grad is None else bundle.output_row.grad.detach().clone()
    return {'lora': lora, 'projector': projector, 'end_fix_rows': rows}


def one_training_backward(bundle, batch, *, disable_where_attention_mask, rng_state, device):
    import torch
    restore_rng_state(rng_state, device)
    clear_gradients(bundle)
    bundle.model.train()
    bundle.projector.train()
    capture = {}
    profiler = CudaEventProfiler(torch, device)
    torch.cuda.reset_peak_memory_stats(device)
    with profiler.stage('total_step'):
        run_flat_training_step(
            bundle,
            batch.episodes,
            where_batch=batch,
            diagnostics=True,
            profiler=profiler,
            audit_disable_where_attention_mask=disable_where_attention_mask,
            audit_capture=capture,
        )
    times = profiler.finish()
    capture['gradients'] = collect_gradients(bundle)
    return {
        'stage_seconds': times,
        'memory': memory_snapshot(torch, device),
        'capture': capture,
    }


def run_timed_variant(bundle, batch, *, disable_where_attention_mask, rng_state, device, warmup_steps, steps):
    rows, parity_run = [], None
    for index in range(warmup_steps + steps):
        row = one_training_backward(
            bundle,
            batch,
            disable_where_attention_mask=disable_where_attention_mask,
            rng_state=rng_state,
            device=device,
        )
        if index >= warmup_steps:
            rows.append({key: value for key, value in row.items() if key != 'capture'})
            parity_run = row
    if not rows:
        raise ValueError('at least one measured timing step is required')
    clear_gradients(bundle)
    stages = sorted({stage for row in rows for stage in row['stage_seconds']})
    summary = {}
    for stage in stages:
        values = [row['stage_seconds'].get(stage, 0.0) for row in rows]
        summary[stage] = {
            'mean_sec': statistics.mean(values),
            'median_sec': statistics.median(values),
            'per_episode_mean_sec': statistics.mean(values) / len(batch.episodes),
            'samples': values,
        }
    peak_allocated = max(row['memory']['peak_memory_allocated_bytes'] for row in rows)
    peak_reserved = max(row['memory']['peak_memory_reserved_bytes'] for row in rows)
    return {
        'timing': summary,
        'timing_rows': rows,
        'peak_memory_allocated_bytes': int(peak_allocated),
        'peak_memory_reserved_bytes': int(peak_reserved),
        'parity_capture': parity_run['capture'],
    }


def summarize_capture(capture):
    gradients = capture['gradients']
    return {
        'where_episode_losses': [float(value) for value in capture['where_episode_losses'].detach().cpu()],
        'loss_where': float(capture['loss_where'].detach().cpu()),
        'loss_flat': float(capture['loss_flat'].detach().cpu()),
        'loss_total': float(capture['loss_total'].detach().cpu()),
        'where_state_shapes': [list(value.shape) for value in capture['where_states']],
        'state_gradient_shapes': [None if value is None else list(value.shape)
                                  for value in capture['state_gradients']],
        'rng_state_after_where_cpu_sha256': tensor_digest(capture['rng_state_after_where_cpu']),
        'rng_state_after_where_cuda_sha256': (
            tensor_digest(capture['rng_state_after_where_cuda'])
            if 'rng_state_after_where_cuda' in capture else None
        ),
        'gradient_tensor_counts': {name: len(values) for name, values in gradients.items()},
        'missing_gradient_counts': {
            name: sum(1 for value in values.values() if value is None)
            for name, values in gradients.items()
        },
    }


def profiler_time_us(event):
    for name in ('self_cuda_time_total', 'cuda_time_total', 'self_device_time_total', 'device_time_total'):
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def collect_profiler_evidence(prof):
    evidence = {pattern: {'count': 0, 'time_us': 0.0, 'ops': {}} for pattern in PROFILER_PATTERNS}
    events = list(prof.key_averages())
    try:
        events += list(prof.events())
    except Exception:
        pass
    for event in events:
        name = getattr(event, 'key', None) or getattr(event, 'name', '')
        lower = name.lower()
        count = int(getattr(event, 'count', 1) or 1)
        time_us = profiler_time_us(event)
        for pattern in PROFILER_PATTERNS:
            if pattern in lower:
                row = evidence[pattern]
                row['count'] += count
                row['time_us'] += time_us
                op = row['ops'].setdefault(name, {'count': 0, 'time_us': 0.0})
                op['count'] += count
                op['time_us'] += time_us
    for row in evidence.values():
        row['ops'] = dict(sorted(row['ops'].items(), key=lambda item: item[1]['time_us'], reverse=True)[:25])
    return evidence


def run_profiler_trace(bundle, batch, *, disable_where_attention_mask, rng_state, device, trace_path):
    import torch
    from torch.profiler import ProfilerActivity, profile
    restore_rng_state(rng_state, device)
    clear_gradients(bundle)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    stage_profiler = RecordFunctionProfiler(torch)
    with profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=False) as prof:
        run_flat_training_step(
            bundle,
            batch.episodes,
            where_batch=batch,
            diagnostics=False,
            profiler=stage_profiler,
            audit_disable_where_attention_mask=disable_where_attention_mask,
        )
    torch.cuda.synchronize(device)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    evidence = collect_profiler_evidence(prof)
    clear_gradients(bundle)
    return {'trace_path': str(trace_path), 'kernel_evidence': evidence}


def compare_tensor(a, b, *, atol, rtol):
    import torch
    if a is None or b is None:
        return {
            'status': 'both_missing' if a is None and b is None else 'missing_mismatch',
            'passed': False,
            'numel': 0,
            'max_absolute_difference': None,
            'mean_absolute_difference': None,
            'max_relative_difference': None,
            'reference_relative_difference': None,
            'atol': atol,
            'rtol': rtol,
        }
    if tuple(a.shape) != tuple(b.shape):
        return {
            'status': 'shape_mismatch',
            'passed': False,
            'shape': [list(a.shape), list(b.shape)],
            'atol': atol,
            'rtol': rtol,
        }
    left = a.detach().cpu().to(torch.float64)
    right = b.detach().cpu().to(torch.float64)
    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    if not finite:
        return {'status': 'non_finite', 'passed': False, 'shape': [list(a.shape), list(b.shape)],
                'atol': atol, 'rtol': rtol}
    diff = (left - right).abs()
    denom = torch.maximum(torch.maximum(left.abs(), right.abs()), torch.full_like(diff, 1e-12))
    reference = left.abs().max().clamp_min(1e-12)
    threshold = atol + rtol * right.abs()
    return {
        'status': 'ok',
        'passed': bool((diff <= threshold).all()),
        'shape': list(a.shape),
        'dtype': [str(a.dtype), str(b.dtype)],
        'numel': int(diff.numel()),
        'max_absolute_difference': finite_float(diff.max().item()) if diff.numel() else 0.0,
        'mean_absolute_difference': finite_float(diff.mean().item()) if diff.numel() else 0.0,
        'max_relative_difference': finite_float((diff / denom).max().item()) if diff.numel() else 0.0,
        'reference_relative_difference': finite_float(diff.max().item() / reference.item()) if diff.numel() else 0.0,
        'atol': atol,
        'rtol': rtol,
    }


def aggregate_comparisons(items):
    failures = [name for name, row in items.items() if not row['passed']]
    compared = [row for row in items.values() if row.get('status') == 'ok']
    total = sum(row['numel'] for row in compared)
    weighted_mean = (sum(row['mean_absolute_difference'] * row['numel'] for row in compared) / total
                     if total else None)
    worst_name, worst = None, None
    for name, row in items.items():
        value = row.get('max_absolute_difference')
        if value is not None and (worst is None or value > worst):
            worst_name, worst = name, value
    return {
        'passed': not failures,
        'failure_count': len(failures),
        'failures': failures[:50],
        'tensor_count': len(items),
        'missing_in_both': [name for name, row in items.items() if row.get('status') == 'both_missing'][:50],
        'max_absolute_difference': worst,
        'mean_absolute_difference': weighted_mean,
        'max_relative_difference': max((row.get('max_relative_difference') or 0.0) for row in compared) if compared else None,
        'worst_tensor': worst_name,
        'items': items,
    }


def compare_tensor_list(left, right, *, atol, rtol, prefix):
    if len(left) != len(right):
        return {'passed': False, 'failure_count': 1, 'failures': ['list_length_mismatch'],
                'left_count': len(left), 'right_count': len(right), 'items': {}}
    return aggregate_comparisons({
        f'{prefix}_{index}': compare_tensor(a, b, atol=atol, rtol=rtol)
        for index, (a, b) in enumerate(zip(left, right))
    })


def compare_tensor_dict(left, right, *, atol, rtol):
    names = sorted(set(left) | set(right))
    return aggregate_comparisons({
        name: compare_tensor(left.get(name), right.get(name), atol=atol, rtol=rtol)
        for name in names
    })


def compare_variants(baseline, candidate, args):
    a, b = baseline, candidate
    comparisons = {
        'where_episode_loss': aggregate_comparisons({
            'where_episode_losses': compare_tensor(a['where_episode_losses'], b['where_episode_losses'],
                                                   atol=args.loss_atol, rtol=args.loss_rtol),
        }),
        'where_total_loss': aggregate_comparisons({
            'loss_where': compare_tensor(a['loss_where'], b['loss_where'],
                                         atol=args.loss_atol, rtol=args.loss_rtol),
        }),
        'end_fix_hidden_states': compare_tensor_list(a['where_states'], b['where_states'],
                                                     atol=args.tensor_atol, rtol=args.tensor_rtol,
                                                     prefix='episode'),
        'semantic_loss': aggregate_comparisons({
            'loss_flat': compare_tensor(a['loss_flat'], b['loss_flat'],
                                        atol=args.loss_atol, rtol=args.loss_rtol),
        }),
        'total_joint_loss': aggregate_comparisons({
            'loss_total': compare_tensor(a['loss_total'], b['loss_total'],
                                         atol=args.loss_atol, rtol=args.loss_rtol),
        }),
        'semantic_rng_state_after_where': aggregate_comparisons({
            'cpu_rng_state': compare_tensor(a['rng_state_after_where_cpu'], b['rng_state_after_where_cpu'],
                                            atol=0.0, rtol=0.0),
            'cuda_rng_state': compare_tensor(a.get('rng_state_after_where_cuda'), b.get('rng_state_after_where_cuda'),
                                             atol=0.0, rtol=0.0),
        }),
        'lora_gradients': compare_tensor_dict(a['gradients']['lora'], b['gradients']['lora'],
                                              atol=args.grad_atol, rtol=args.grad_rtol),
        'projector_gradients': compare_tensor_dict(a['gradients']['projector'], b['gradients']['projector'],
                                                   atol=args.grad_atol, rtol=args.grad_rtol),
        'end_fix_row_gradients': compare_tensor_dict(a['gradients']['end_fix_rows'], b['gradients']['end_fix_rows'],
                                                     atol=args.grad_atol, rtol=args.grad_rtol),
        'end_fix_state_path_gradients': compare_tensor_list(a['state_gradients'], b['state_gradients'],
                                                            atol=args.grad_atol, rtol=args.grad_rtol,
                                                            prefix='episode'),
    }
    return {'passed': all(row['passed'] for row in comparisons.values()), 'comparisons': comparisons}


def speedups(baseline, candidate):
    result = {}
    base_times = baseline['timing']
    cand_times = candidate['timing']
    for stage in sorted(set(base_times) & set(cand_times)):
        a = base_times[stage]['mean_sec']
        b = cand_times[stage]['mean_sec']
        result[stage] = a / b if b > 0 else None
    return result


def summarize_kernel_names(evidence):
    names = set()
    for row in evidence.values():
        names.update(row.get('ops', {}).keys())
    return names


def profiler_has_relevant_evidence(profiler):
    for name in ('baseline', 'causal_only'):
        evidence = profiler.get(name, {}).get('kernel_evidence', {})
        if not any(row.get('count', 0) > 0 for row in evidence.values()):
            return False
    return True


def final_verdict(parity, speedup, profiler, min_speedup):
    if not parity['passed']:
        return 'INVALID'
    if not profiler or any('kernel_evidence' not in profiler.get(key, {}) for key in ('baseline', 'causal_only')):
        return 'INCONCLUSIVE'
    if not profiler_has_relevant_evidence(profiler):
        return 'INCONCLUSIVE'
    relevant = [value for key, value in speedup.items() if key in ('where_forward', 'backward') and value is not None]
    if not relevant:
        return 'INCONCLUSIVE'
    return 'SUPPORTED' if max(relevant) >= min_speedup else 'NOT SUPPORTED'


def write_summary(path, report):
    parity = report.get('parity', {})
    speed = report.get('speedup', {})
    batch = report['batch']
    lines = [
        '# Attention Mask Audit',
        '',
        f'Verdict: **{report["final_verdict"]}**',
        '',
        f'B={batch["B"]}, K={batch["K"]}, lengths={batch["sequence_lengths"]}, padding={batch["padding_tokens_by_episode"]}',
        f'GC enabled: `{report["gradient_checkpointing"]}`',
        '',
        '| Stage | Baseline ms/episode | Causal-only ms/episode | Speedup |',
        '|---|---:|---:|---:|',
    ]
    baseline = report['variants']['baseline']['timing']
    causal = report['variants']['causal_only']['timing']
    for stage in ('where_forward', 'semantic_forward', 'backward', 'total_step'):
        if stage in baseline and stage in causal:
            lines.append(f'| {stage} | {baseline[stage]["per_episode_mean_sec"] * 1000:.3f} | '
                         f'{causal[stage]["per_episode_mean_sec"] * 1000:.3f} | '
                         f'{speed.get(stage, float("nan")):.4g} |')
    lines += [
        '',
        f'Loss/gradient parity: `{"PASS" if parity.get("passed") else "FAIL"}`',
        f'Peak allocated VRAM: baseline={report["variants"]["baseline"]["peak_memory_allocated_bytes"] / 2**30:.3f} GiB, '
        f'causal-only={report["variants"]["causal_only"]["peak_memory_allocated_bytes"] / 2**30:.3f} GiB',
        f'Peak reserved VRAM: baseline={report["variants"]["baseline"]["peak_memory_reserved_bytes"] / 2**30:.3f} GiB, '
        f'causal-only={report["variants"]["causal_only"]["peak_memory_reserved_bytes"] / 2**30:.3f} GiB',
        '',
        'Profiler traces:',
        f'- baseline: `{report["profiler"]["baseline"]["trace_path"]}`',
        f'- causal-only: `{report["profiler"]["causal_only"]["trace_path"]}`',
        '',
        'See `report.json` for per-tensor parity, kernel evidence, RNG/batch metadata, and raw timing samples.',
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; this diagnostic has no CPU performance fallback')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = load_config(args.config)
    override = {
        'training': {
            'per_device_train_batch_size': 2,
            'gradient_accumulation_steps': 1,
        },
        'runtime': {'device': args.device, 'output_dir': str(args.output_dir / 'model')},
    }
    if args.gradient_checkpointing != 'config':
        override['training']['gradient_checkpointing'] = args.gradient_checkpointing == 'on'
    config = resolve_config(config, override)
    for field in ('images_root', 'split_root'):
        value = getattr(args, field)
        if value is not None:
            config['data'][field] = str(value.resolve())
    validate_real_config(config)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('bf16 CUDA device required for configured flat throughput audit')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        'schema_version': SCHEMA_VERSION,
        'created_at': timestamp(),
        'status': 'running',
        'config_source': str(args.config.resolve()),
        'resolved_config': config,
        'machine': machine_metadata(),
        'device': args.device,
        'gradient_checkpointing': config['training']['gradient_checkpointing'],
        'timing': {'warmup_steps': args.warmup_steps, 'measured_steps': args.steps,
                   'source': 'CUDA events', 'normalized': 'per physical episode'},
        'tolerances': {
            'loss_atol': args.loss_atol,
            'loss_rtol': args.loss_rtol,
            'tensor_atol': args.tensor_atol,
            'tensor_rtol': args.tensor_rtol,
            'grad_atol': args.grad_atol,
            'grad_rtol': args.grad_rtol,
            'min_material_speedup': args.min_speedup,
        },
    }
    save_json(args.output_dir / 'report.json', report)
    sampler, dataset = load_real_sampler(config)
    report['dataset'] = dataset
    bundle = build_flat_model_bundle(config=config, output_dir=args.output_dir / 'model')
    report['model_metadata'] = dict(bundle.diagnostics) | runtime_metadata(bundle)
    batch, selection = select_real_b2_batch(bundle, sampler, batch_case=args.case, max_windows=args.max_windows)
    report['batch_selection'] = selection
    report['batch'] = batch_report(batch)
    report['safety_assertions'] = assert_where_causal_only_safe(
        batch,
        pad_token_id=bundle.processor.tokenizer.pad_token_id,
        end_fix_id=bundle.end_fix_id,
    )
    rng_state = capture_rng_state(device)
    raw_captures = {}
    report['variants'] = {}
    for name, disable_mask, description in VARIANTS:
        print(f'[AUDIT] timing {name}: {description}', flush=True)
        variant = run_timed_variant(
            bundle,
            batch,
            disable_where_attention_mask=disable_mask,
            rng_state=rng_state,
            device=device,
            warmup_steps=args.warmup_steps,
            steps=args.steps,
        )
        raw_captures[name] = variant.pop('parity_capture')
        report['variants'][name] = {'description': description, **variant,
                                    'parity_snapshot': summarize_capture(raw_captures[name])}
        save_json(args.output_dir / 'report.json', report)
    report['parity'] = compare_variants(raw_captures['baseline'], raw_captures['causal_only'], args)
    report['speedup'] = speedups(report['variants']['baseline'], report['variants']['causal_only'])
    report['profiler'] = {}
    for name, disable_mask, _description in VARIANTS:
        print(f'[AUDIT] profiler trace {name}', flush=True)
        report['profiler'][name] = run_profiler_trace(
            bundle,
            batch,
            disable_where_attention_mask=disable_mask,
            rng_state=rng_state,
            device=device,
            trace_path=args.output_dir / f'{name}_trace.json',
        )
        save_json(args.output_dir / 'report.json', report)
    baseline_kernels = summarize_kernel_names(report['profiler']['baseline']['kernel_evidence'])
    causal_kernels = summarize_kernel_names(report['profiler']['causal_only']['kernel_evidence'])
    report['observed_profiler_kernel_evidence'] = {
        'baseline_relevant_ops': sorted(baseline_kernels),
        'causal_only_relevant_ops': sorted(causal_kernels),
        'only_in_baseline': sorted(baseline_kernels - causal_kernels),
        'only_in_causal_only': sorted(causal_kernels - baseline_kernels),
        'patterns_checked': list(PROFILER_PATTERNS),
        'note': 'Backend evidence comes from torch.profiler operation/kernel names, not only _attn_implementation.',
    }
    report['final_verdict'] = final_verdict(report['parity'], report['speedup'], report['profiler'], args.min_speedup)
    report['status'] = 'complete'
    report['finished_at'] = timestamp()
    save_json(args.output_dir / 'report.json', report)
    write_summary(args.output_dir / 'summary.md', report)
    print(f'[AUDIT] report: {args.output_dir / "report.json"}', flush=True)
    print(f'[AUDIT] summary: {args.output_dir / "summary.md"}', flush=True)
    print(f'[AUDIT] final verdict: {report["final_verdict"]}', flush=True)
    return 0 if report['final_verdict'] != 'INVALID' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/flat_throughput.yaml')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'runs' / ('attention-mask-audit-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--images-root', type=Path)
    parser.add_argument('--split-root', type=Path)
    parser.add_argument('--case', choices=('unequal', 'equal', 'any'), default='unequal',
                        help='B=2 WHERE length relationship to select from real same-K training batches')
    parser.add_argument('--max-windows', type=int, default=200,
                        help='maximum B=2 optimizer windows to inspect while selecting a real batch')
    parser.add_argument('--warmup-steps', type=int, default=1)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--gradient-checkpointing', choices=('config', 'on', 'off'), default='config')
    parser.add_argument('--loss-atol', type=float, default=5e-3)
    parser.add_argument('--loss-rtol', type=float, default=5e-3)
    parser.add_argument('--tensor-atol', type=float, default=2e-2)
    parser.add_argument('--tensor-rtol', type=float, default=2e-2)
    parser.add_argument('--grad-atol', type=float, default=5e-2)
    parser.add_argument('--grad-rtol', type=float, default=5e-2)
    parser.add_argument('--min-speedup', type=float, default=1.10,
                        help='material speedup threshold for WHERE/backward/total timing')
    args = parser.parse_args(argv)
    if args.max_windows < 1 or args.warmup_steps < 0 or args.steps < 1:
        parser.error('max-windows/steps must be positive and warmup nonnegative')
    if any(value < 0 for value in (args.loss_atol, args.loss_rtol, args.tensor_atol,
                                   args.tensor_rtol, args.grad_atol, args.grad_rtol)):
        parser.error('tolerances must be nonnegative')
    if args.min_speedup <= 1:
        parser.error('min-speedup must be greater than 1')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output-dir must be new or empty')
    try:
        return run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            'schema_version': SCHEMA_VERSION,
            'status': 'failed',
            'error': f'{type(error).__name__}: {error}',
            'traceback': traceback.format_exc(),
            'finished_at': timestamp(),
        }
        save_json(args.output_dir / 'report.json', report)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
