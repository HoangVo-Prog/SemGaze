#!/usr/bin/env python3
"""Checkpoint-exact test parity, fresh-process sweeps, and full CUDA reports."""
import argparse
import json
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_a100 import (save_json, machine_metadata, timestamp, digest,
                                    memory_snapshot, sequence_statistics)

METRICS = ('test_where', 'test_flat', 'test_what', 'test_why', 'test_how', 'test_total')


def compare_reports(reference, candidate, tolerance):
    if reference.get('status') != 'complete' or candidate.get('status') != 'complete':
        raise ValueError('parity requires two completed runs')
    for key in ('checkpoint_sha256', 'workload_sha256', 'split_manifest_identity', 'scientific_config_sha256'):
        if reference.get(key) is None or reference[key] != candidate.get(key):
            raise ValueError(f'incomparable parity reports: {key}')
    def index(report):
        rows = report['episodes']
        result = {row['episode_id']: row for row in rows}
        if len(result) != len(rows) or not rows:
            raise ValueError('duplicate or empty parity episodes')
        return result
    a, b = index(reference), index(candidate)
    if a.keys() != b.keys():
        raise ValueError('missing parity episode or K')
    differences = {key: [] for key in METRICS}
    for key in a:
        for field in ('K', 'support_ids', 'state_count'):
            if a[key][field] != b[key][field]:
                raise ValueError(f'{key}: parity {field} mismatch')
        for metric in METRICS:
            import math
            difference = abs(a[key][metric] - b[key][metric])
            if not math.isfinite(difference):
                raise ValueError('non-finite parity metric')
            differences[metric].append(difference)
    stats = {key: dict(max_absolute_difference=max(v), mean_absolute_difference=statistics.mean(v),
                      episode_with_max_difference=list(a)[v.index(max(v))],
                      tolerance=tolerance, passed=max(v) <= tolerance) for key, v in differences.items()}
    speedup = {k: reference['by_k'][k]['wall_time_sec']/row['wall_time_sec']
               for k, row in candidate.get('by_k', {}).items() if k in reference.get('by_k', {})}
    return dict(passed=all(v['passed'] for v in stats.values()), episodes=len(a), metrics=stats,
                speedup_by_k=speedup,
                overall_speedup=(reference['total_wall_time_sec']/candidate['total_wall_time_sec']
                                 if candidate.get('total_wall_time_sec') else None))


def write_parity(directory, result):
    save_json(directory / 'parity_quick.json', result)
    lines = [f'# Episode parity: {"PASS" if result["passed"] else "FAIL"}', '',
             '| Metric | Max absolute difference | Mean absolute difference | Tolerance | Pass |',
             '|---|---:|---:|---:|---|']
    for key, row in result['metrics'].items():
        lines.append(f'| {key} | {row["max_absolute_difference"]:.8g} | {row["mean_absolute_difference"]:.8g} | {row["tolerance"]} | {row["passed"]} |')
    lines += ['', f'Measured overall speedup: {result.get("overall_speedup")}',
              f'Measured speedup by K: {result.get("speedup_by_k")}',
              'Timing ratios include configured cache/bucketing and diagnostic overhead; compare identical profiling settings.']
    (directory / 'parity_quick.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def summarize_batches(batches, wall, memory):
    rows = [{'episodes': [e | {'physical_batch_index': i} for e in batch]}
            for i, batch in enumerate(batches)]
    padding = sequence_statistics(rows)
    sizes = [len(b) for b in batches]
    count = sum(sizes)
    return dict(episodes=count, batch_count=len(sizes), mean_actual_batch_size=statistics.mean(sizes),
        min_actual_batch_size=min(sizes), max_actual_batch_size=max(sizes), wall_time_sec=wall,
        episodes_per_second=count/wall, seconds_per_episode=wall/count, padding=padding, **memory)


def write_report(directory, report):
    save_json(directory / 'report.json', report)
    if report.get('status') == 'complete':
        name = 'serial' if report['mode'] == 'serial' else 'batched'
        save_json(directory / f'{name}_{report["scope"]}.json', report)
    lines = ['# Test benchmark', '', f'Status: {report["status"]}', '',
             '| Mode | K | Physical B | Episodes | WHERE pad eff. | Semantic pad eff. | Peak alloc GiB | Peak reserved GiB | Episodes/s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for run in ([report['reference_summary']] if report.get('reference_summary') else []) + [report]:
        for k, row in run.get('by_k', {}).items():
            pad = row['padding']
            lines.append(f'| {run["mode"]} | {k} | {run["batch_size_by_k"][k]} | {row["episodes"]} | '
                f'{1-pad["where_padding_fraction"]:.4f} | {1-pad["semantic_padding_fraction"]:.4f} | '
                f'{row["peak_memory_allocated_bytes"]/2**30:.3f} | {row["peak_memory_reserved_bytes"]/2**30:.3f} | {row["episodes_per_second"]:.4f} |')
    lines += ['', f'Overall episodes/s: {report.get("episodes_per_second")}',
              f'Cache: `{report.get("cache")}`', f'Parity: `{report.get("parity", "pending comparison")}`',
              f'Stage profiling (inclusive): `{report.get("stage_times", {})}`',
              f'Largest measured stage: `{report.get("remaining_bottleneck", "not profiled; use --profile")}`',
              f'Error: {report.get("error", "none")}', '',
              'Useful batch sizes and remaining bottlenecks require comparisons across measured runs. '
              'Prefix KV caching is disabled. Report includes cold invocation caches after kernel warmup.']
    if report.get('tensor_trace'):
        lines += ['', f'FIRST_DIVERGENCE = {report["tensor_trace"].get("FIRST_DIVERGENCE", "unresolved")}',
                  'See tensor_trace.json for the untimed input checks and tensor comparisons.']
    (directory / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def checkpoint_digest(path):
    import hashlib
    files = sorted(p for p in path.rglob('*') if p.is_file() and
                   (p.suffix in ('.safetensors', '.json', '.jinja', '.txt', '.model')))
    result = {}
    for file in files:
        value = hashlib.sha256()
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                value.update(chunk)
        result[file.relative_to(path).as_posix()] = value.hexdigest()
    return digest(result)


def prediction_parity(bundle, workloads, sizes, budget, directory):
    from copy import deepcopy
    from semgaze.evaluation.predictions import prediction_batches
    from semgaze.evaluation.test import test_mode
    from semgaze.model.visual_cache import InferenceVisualCache
    settings = bundle.config['test']['prediction']
    previous = deepcopy(settings)
    results = {}
    try:
        with test_mode(bundle):
            for execution in ('serial', 'same_k_batched'):
                settings.update(execution=execution, batch_size_by_k=sizes)
                cache = InferenceVisualCache(preprocessing=execution != 'serial', features=execution != 'serial')
                rows = []
                try:
                    for k, episodes in workloads.items():
                        for ep, (where, semantic) in prediction_batches(bundle, episodes, budget=budget, cache=cache):
                            rows.append(dict(episode_id=f'{k}:{ep.query.record_id}', where=where, semantic=semantic))
                finally:
                    cache.close()
                results[execution] = rows
        differences = [a['episode_id'] for a,b in zip(results['serial'], results['same_k_batched']) if a != b]
        result = dict(passed=not differences, different_episode_ids=differences,
                      semantic_max_new_tokens=budget, do_sample=False, **results)
        save_json(directory / 'prediction_parity.json', result)
        (directory / 'prediction_parity.md').write_text(
            f'# Greedy prediction parity: {"PASS" if result["passed"] else "FAIL"}\n\n'
            f'Different episode IDs: {differences}\n', encoding='utf-8')
        return result['passed']
    finally:
        settings.clear()
        settings.update(previous)


def run(args):
    import torch
    from semgaze.model.config import load_config, resolve_config
    from semgaze.model.checkpoint import load_checkpoint_bundle
    from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter
    from semgaze.data.fewshot import frozen_episode
    from semgaze.evaluation.test import (test_queries, test_mode, batch_losses,
                                              serial_reference_losses, EpisodeAccumulator)
    from semgaze.evaluation.batching import schedule_batches, episode_id
    from semgaze.model.visual_cache import InferenceVisualCache
    from semgaze.training.profiling import TrainingProfiler, runtime_metadata
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    preprocess_cache = (args.preprocess_cache or args.cache) == 'on' and args.mode != 'serial'
    visual_cache = (args.visual_cache or args.cache) == 'on' and args.mode != 'serial'
    report = dict(timestamp=timestamp(), environment=machine_metadata(), status='running', mode=args.mode,
                  scope=args.scope, config_path=str(args.config), checkpoint=str(args.checkpoint),
                  episodes=[], by_k={}, inference_mode=True, same_k_batching=args.mode != 'serial',
                  length_bucketing=args.bucket and args.mode != 'serial', visual_cache=visual_cache,
                  preprocess_cache=preprocess_cache, support_cache=preprocess_cache, prefix_kv_cache=False,
                  profiling=args.profile, warmup_batches_per_k=args.warmup)
    git_status = report['environment'].get('git_status')
    report['environment']['dirty_working_tree'] = None if git_status is None else bool(git_status)
    cache, bundle, active_k, active_b = None, None, None, None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA server required; no CPU performance fallback')
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        props = torch.cuda.get_device_properties(device)
        report['environment'].update(gpu_name=props.name, gpu_total_memory=props.total_memory,
                                     cuda_version=torch.version.cuda)
        config = load_config(args.checkpoint / 'resolved_config.json')
        if args.config:
            # The checkpoint owns the scientific configuration, tokenizer and weights.
            overrides = load_config(args.config)
            config = resolve_config(config, {'test': overrides['test']})
        config['runtime']['device'] = args.device
        data = config['data']
        raw, manifest, identity = read_persisted_splits(ROOT / data['split_root'], data_config=data)
        adapter = CocoSearch18Adapter(ROOT / data['images_root'], annotation_frame=data['annotation_frame'],
                                     duration_field=data['duration']['source_field'])
        train = {r['record_id']: adapter(r) for r in raw['train']}
        queries = list(test_queries(train, [adapter(r) for r in raw['test']], manifest, data['unseen_subjects']))
        canonical_count = len(queries)
        if args.scope == 'quick':
            # Deterministic per-subject length-spread selection; same identities in all modes.
            chosen = []
            for subject in sorted({q.subject for q in queries}):
                pool = sorted((q for q in queries if q.subject == subject),
                              key=lambda q: (len(q.x_px), len(str(q.semantic)), q.record_id))
                indices = sorted({round(i*(len(pool)-1)/(args.quick_per_subject-1))
                                  for i in range(args.quick_per_subject)})
                chosen.extend(pool[i] for i in indices)
            ids = {q.record_id for q in chosen}
            queries = [q for q in queries if q.record_id in ids]
        ks = config['evaluation']['k_values']
        if args.k is not None:
            if args.k not in ks:
                raise ValueError('requested sweep K absent from checkpoint evaluation config')
            ks = [args.k]
        sizes = {str(k): 1 if args.mode == 'serial' else getattr(args, f'batch_k{k}', None) or
                 config['test']['loss']['batch_size_by_k'].get(k,
                 config['test']['loss']['batch_size_by_k'].get(str(k))) for k in ks}
        if args.batch_size:
            sizes = {str(k): args.batch_size for k in ks}
        settings = dict(execution='same_k_batched', batch_size_by_k=sizes,
                        bucket_by_length=args.bucket, bucket_window=args.bucket_window)
        if args.mode == 'serial':
            settings.update(execution='serial', bucket_by_length=False)
            sizes = {str(k): 1 for k in ks}
            settings['batch_size_by_k'] = sizes
        report.update(batch_size_by_k=sizes, canonical_query_count=canonical_count, query_count=len(queries),
            k_values=ks, episode_count_total=len(queries)*len(ks)*10, episode_count_by_k={str(k): len(queries)*10 for k in ks},
            split_manifest_identity=identity, checkpoint_sha256=checkpoint_digest(args.checkpoint),
            scientific_config_sha256=digest({k: config[k] for k in ('data', 'model', 'where', 'state', 'semantic', 'training')}),
            base_model=config['model']['base_model'], adapter=config['model']['initialization_adapter'])
        bundle = load_checkpoint_bundle(args.checkpoint, split_manifest_identity=identity,
                    output_dir=directory / 'model', runtime=config['runtime'])
        bundle.config = config
        config['test']['cache']['frozen_visual_features'] = visual_cache
        report['model_metadata'] = runtime_metadata(bundle)
        report['dtype'] = report['model_metadata']['dtype']
        report['attention_backend'] = report['model_metadata']['text_attention']
        workloads = {k: [frozen_episode(q, train, manifest, k, draw_id=draw, unseen_subjects=data['unseen_subjects']) for draw in range(10) for q in queries] for k in ks}
        report['workload_sha256'] = digest([{'episode_id': episode_id(e), 'support_ids': [s.record_id for s in e.supports]}
                                          for es in workloads.values() for e in es])
        profiler = TrainingProfiler(device) if args.profile else None
        overall = EpisodeAccumulator()
        with test_mode(bundle):
            # Kernel warmup includes all configured K. No warmup cache is retained.
            for k, episodes in workloads.items():
                active_k, active_b = k, sizes[str(k)]
                for _ in range(args.warmup):
                    first = next(schedule_batches(bundle, episodes[:sizes[str(k)]], settings))
                    if args.mode == 'serial':
                        serial_reference_losses(bundle, first.episodes[0])
                    else:
                        batch_losses(bundle, first.episodes, batch=first)
            torch.cuda.synchronize(device)
            cache = InferenceVisualCache(preprocessing=preprocess_cache, features=visual_cache,
                                        max_entries=config['test']['cache']['max_entries'])
            start = time.perf_counter()
            for k, episodes in workloads.items():
                active_k, active_b = k, sizes[str(k)]
                torch.cuda.reset_peak_memory_stats(device)
                k_start = time.perf_counter()
                batches = []
                for batch_index, batch in enumerate(schedule_batches(bundle, episodes, settings, cache=cache, profiler=profiler)):
                    if args.mode == 'serial':
                        ep = batch.episodes[0]
                        rows = [serial_reference_losses(bundle, ep, batch=batch)]
                    else:
                        rows = batch_losses(bundle, batch.episodes, batch=batch, cache=cache, profiler=profiler)
                    for row in rows:
                        row['physical_batch_index'] = batch_index
                        overall.add(row['episode_id'], row)
                    report['episodes'].extend(rows)
                    batches.append(rows)
                    print(f'[BENCH] K={k} episodes={sum(map(len,batches))}/{len(episodes)} B={len(rows)}', flush=True)
                torch.cuda.synchronize(device)
                wall = time.perf_counter()-k_start
                summary = summarize_batches(batches, wall, memory_snapshot(torch, device))
                ordered = {row['episode_id']: row for batch in batches for row in batch}
                canonical = [ordered[episode_id(e)] for e in episodes]
                baseline = [canonical[i:i+sizes[str(k)]] for i in range(0,len(canonical),sizes[str(k)])]
                summary['canonical_order_padding'] = summarize_batches(baseline, wall, {})['padding']
                report['by_k'][str(k)] = summary
            torch.cuda.synchronize(device)
            wall = time.perf_counter()-start
        expected = [episode_id(e) for es in workloads.values() for e in es]
        report['metrics'] = overall.finish(expected)
        ordered = {e['episode_id']: e for e in report['episodes']}
        report['episodes'] = [ordered[key] for key in expected]
        report['episode_ids'] = expected
        report.update(status='complete', total_wall_time_sec=wall, episodes_per_second=len(expected)/wall,
            seconds_per_episode=wall/len(expected), cache=cache.statistics(),
            stage_times=profiler.times if profiler else {},
            peak_memory_allocated_bytes=max(r['peak_memory_allocated_bytes'] for r in report['by_k'].values()),
            peak_memory_reserved_bytes=max(r['peak_memory_reserved_bytes'] for r in report['by_k'].values()))
        if args.reference:
            reference = json.loads(args.reference.read_text())
            report['parity'] = compare_reports(reference, report, args.tolerance)
            report['reference_summary'] = {key: reference[key] for key in
                ('mode', 'batch_size_by_k', 'by_k', 'total_wall_time_sec', 'episodes_per_second')}
            write_parity(directory, report['parity'])
            if not report['parity']['passed']:
                report['status'] = 'parity_failed'
        if profiler and profiler.times:
            report['remaining_bottleneck'] = max(profiler.times, key=profiler.times.get)
        if args.trace_parity and (not args.audit_stage or report['status'] == 'parity_failed'):
            cache.close()  # Trace memory/time must not enter benchmark telemetry.
            try:
                report['tensor_trace'] = trace_parity(bundle, workloads, settings, report, args.tolerance,
                                                      reference=reference if args.reference else None)
            except Exception as error:
                report['tensor_trace'] = dict(status='failed', FIRST_DIVERGENCE='unresolved',
                    error=str(error), traceback=traceback.format_exc())
            save_json(directory / 'tensor_trace.json', report['tensor_trace'])
        if args.prediction_parity_budget is not None:
            cache.close()  # Do not retain the measured run's visual tensors.
            report['prediction_parity_passed'] = prediction_parity(bundle, workloads, sizes,
                args.prediction_parity_budget, directory)
            if not report['prediction_parity_passed']:
                report['status'] = 'prediction_parity_failed'
    except Exception as error:
        report.update(status='oom' if isinstance(error, torch.cuda.OutOfMemoryError) else 'failed',
            error=str(error), traceback=traceback.format_exc(), failed_k=active_k, requested_physical_batch=active_b)
        if torch.cuda.is_available():
            report['failure_memory'] = memory_snapshot(torch, args.device)
    finally:
        if cache is not None:
            cache.close()
        write_report(directory, report)
    return 0 if report['status'] == 'complete' else 1


def tensor_difference(a, b, tolerance):
    """CPU summaries only; the scalar parity tolerance is never changed."""
    import torch
    result = dict(shape=[list(a.shape), list(b.shape)], dtype=[str(a.dtype), str(b.dtype)],
                  allclose=False, bitwise_equal=False, max_abs_diff=None, mean_abs_diff=None,
                  max_relative_error=None, first_differing_index=None, atol=tolerance, rtol=0)
    if a.shape != b.shape:
        return result
    a, b = a.detach().cpu(), b.detach().cpu()
    # Chunk vocabulary-sized tensors to avoid multiple full FP32 trace buffers.
    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
    maximum, total, relative, first, close, finite = 0., 0., 0., None, True, True
    for start in range(0, a.numel(), 65536):
        x, y = flat_a[start:start+65536].double(), flat_b[start:start+65536].double()
        finite = finite and bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        diff = (x-y).abs()
        if first is None and bool((x != y).any()):
            first = start + int((x != y).nonzero()[0, 0])
        maximum = max(maximum, float(diff.max()))
        total += float(diff.sum())
        relative = max(relative, float((diff / x.abs().clamp_min(1e-12)).max()))
        close = close and bool((diff <= tolerance).all())
    if first is not None:
        index, remaining = [], first
        for size in reversed(a.shape):
            index.append(remaining % size)
            remaining //= size
        result['first_differing_index'] = list(reversed(index))
    result.update(allclose=close and finite, bitwise_equal=first is None and a.dtype == b.dtype,
                  max_abs_diff=maximum if finite else None,
                  mean_abs_diff=total/max(1, a.numel()) if finite else None,
                  max_relative_error=relative if finite else None, finite=finite)
    return result


def trace_inputs(bundle, alone, batch, row):
    """Check logical equality and each packing invariant BEFORE either forward."""
    import torch
    from semgaze.evaluation.batching import episode_id
    checks = {}
    def check(name, value):
        checks[name] = 'PASS' if bool(value) else 'FAIL'
    a, b = alone.episodes[0], batch.episodes[row]
    ma, mb = alone.metadata[0], batch.metadata[row]
    n = ma['where_length']
    offset, count = mb['image_offset'], mb['image_count']
    check('episode_id', episode_id(a) == episode_id(b))
    check('K', len(a.supports) == len(b.supports))
    check('support_ids', [s.record_id for s in a.supports] == [s.record_id for s in b.supports])
    check('real_sequence_length', n == mb['where_length'])
    for key in ('input_ids', 'attention_mask', 'labels'):
        check(key, torch.equal(alone.inputs[key][0, :n], batch.inputs[key][row, :n]))
    for name, packed in (('alone', alone), ('batch', batch)):
        for i, meta in enumerate(packed.metadata):
            tail = slice(meta['where_length'], None)
            for key, value in (('attention_mask', 0), ('labels', -100),
                               ('input_ids', bundle.processor.tokenizer.pad_token_id)):
                check(f'{name}_{i}_padding_{key}', (packed.inputs[key][i, tail] == value).all())
            start, size = meta['image_offset'], meta['image_count']
            check(f'{name}_{i}_image_offsets', start == sum(m['image_count'] for m in packed.metadata[:i])
                  and meta['query_image_index'] == start+size-1
                  and tuple(meta['support_image_indices']) == tuple(range(start, start+size-1)))
            check(f'{name}_{i}_pixel_slice', torch.equal(packed.inputs['pixel_values'][start:start+size],
                                                        packed.samples[i].inputs['pixel_values']))
    check('pixel_values', torch.equal(alone.inputs['pixel_values'], batch.inputs['pixel_values'][offset:offset+count]))
    check('image_paths', alone.samples[0].image_paths == batch.samples[row].image_paths)
    check('image_ordering', tuple(mb['image_paths']) == tuple(s.image_path for s in (*b.supports, b.query)))
    image_a = (alone.inputs['input_ids'][0] == bundle.processor.image_token_id).nonzero().flatten()
    image_b = (batch.inputs['input_ids'][row] == bundle.processor.image_token_id).nonzero().flatten()
    check('image_token_boundaries', torch.equal(image_a, image_b) and len(image_b) == count*bundle.processor.image_seq_length)
    positions = lambda packed, i: (packed.inputs['labels'][i, 1:] != -100).nonzero().flatten()
    check('supervised_WHERE_token_positions', torch.equal(positions(alone, 0), positions(batch, row)))
    check('state_count', len(a.query.x_px) == len(b.query.x_px) == int((batch.inputs['labels'][row] == bundle.end_fix_id).sum()))
    return dict(checks=checks, passed=all(v == 'PASS' for v in checks.values()),
                episode_id=episode_id(a), K=len(a.supports), support_ids=[s.record_id for s in a.supports],
                image_paths=list(alone.samples[0].image_paths),
                image_offsets=[ma['image_offset'], offset], image_count=count,
                real_lengths=[m['where_length'] for m in batch.metadata],
                target_row=row, supervised_positions=positions(alone, 0).tolist(),
                position_convention='zero-based predictor t for supervised label t+1')


def capture_trace(bundle, batch, row, *, serial, cache, semantic):
    """Observe real evaluator calls. All patches/hooks are scoped and restored.

    Only the target's supervised logits are copied to host; no tensors go to JSON.
    Raising at the WHERE return prevents downstream semantic work on WHERE failure.
    """
    from contextlib import ExitStack
    from unittest.mock import patch
    import torch
    import semgaze.evaluation.test as test
    import semgaze.where.forward as where_module
    import semgaze.semantic.flat.forward as flat_module
    native = bundle.model.get_base_model().model
    tensors, details = {}, dict(vision_calls=[], tensor_devices={}, semantic_natives=[])
    phase, labels, length = 'where', batch.inputs['labels'], batch.metadata[row]['where_length']
    def keep(name, value):
        details['tensor_devices'][name] = str(value.device)
        tensors[name] = value.detach().to('cpu', copy=True)
    def selected_slice():
        counts = (labels[:, 1:] != -100).sum(dim=1).tolist()
        return slice(sum(counts[:row]), sum(counts[:row+1]))
    def positions():
        return (labels[row, 1:] != -100).nonzero().flatten()
    original_vision = native.get_image_features
    def vision(*args, **kwargs):
        result = original_vision(*args, **kwargs)
        pixels = args[0] if args else kwargs['pixel_values']
        details['vision_calls'].append(dict(phase=phase, pixel_shape=list(pixels.shape),
            feature_shape=list(result.pooler_output.shape), dtype=str(result.pooler_output.dtype),
            device=str(result.pooler_output.device), requires_grad=result.pooler_output.requires_grad,
            options={k: str(v) for k, v in kwargs.items() if k != 'pixel_values'}))
        return result
    def lm_pre(module, args, kwargs):
        keep(f'{phase}_fused_inputs', kwargs['inputs_embeds'][row, :length])
        for key in ('attention_mask', 'position_ids', 'cache_position'):
            value = kwargs.get(key)
            if isinstance(value, torch.Tensor):
                if value.ndim == 2:
                    value = value[min(row, value.shape[0]-1), :length]
                elif value.ndim == 1:
                    value = value[:length]
                keep(f'{phase}_model_{key}', value)
            else:
                details[f'{phase}_model_{key}'] = str(value)
    def backbone_post(module, args, output):
        keep(f'{phase}_hidden', output.last_hidden_state[row, positions().to(output.last_hidden_state.device)])
        if phase == 'where' and output.image_hidden_states is not None:
            m = batch.metadata[row]
            keep('visual_features', output.image_hidden_states[m['image_offset']:m['image_offset']+m['image_count']])
    def head_post(module, args, output):
        if output.ndim == 3:  # native serial semantic head: all positions
            value = output[row, positions().to(output.device)]
        else:  # selected head: concatenated supervised positions, in episode order
            value = output[selected_slice()]
        keep(f'{phase}_logits', value)
    original_fuse = cache.fuse if cache is not None else None
    def fuse(*args, **kwargs):
        result = original_fuse(*args, **kwargs)
        if result[1] is not None:
            m = batch.metadata[row]
            keep('visual_features', result[1][m['image_offset']:m['image_offset']+m['image_count']])
            details['cached_features_require_grad'] = result[1].requires_grad
        return result
    def selected_wrapper(original):
        def observe(*args, **kwargs):
            result = original(*args, **kwargs)
            keep(f'{phase}_nll', result.token_nll[selected_slice()])
            keep(f'{phase}_episode_loss', result.episode_losses[row])
            return result
        return observe
    class WhereFinished(Exception):
        pass
    original_where = test.forward_where_batch
    def where(*args, **kwargs):
        nonlocal phase
        result = original_where(*args, **kwargs)
        keep('query_END_FIX_states', result.states[row])
        if not semantic:
            raise WhereFinished()
        phase = 'semantic'
        return result
    def projector_post(module, args, output):
        counts = [len(e.query.x_px) for e in batch.episodes]
        keep('projected_R_states', output[sum(counts[:row]):sum(counts[:row+1])])
    def prepared(original, is_serial):
        def observe(*args, **kwargs):
            nonlocal labels, length
            result = original(*args, **kwargs)
            inputs = result[0]
            labels = inputs['labels'].detach().cpu()
            length = int(inputs['attention_mask'][row].sum())
            keep('semantic_labels', labels[row, :length])
            keep('semantic_supervised_positions', positions())
            keep('semantic_attention_mask', inputs['attention_mask'][row, :length])
            details['semantic_padding_valid'] = bool((labels[row, length:] == -100).all() and
                (inputs['attention_mask'][row, length:] == 0).all())
            details['response_offsets'] = list(result[2].response_offsets if is_serial else result[2][row]['response_offsets'])
            return result
        return observe
    original_collate = flat_module.collate_native
    def collate(*args, **kwargs):
        result = original_collate(*args, **kwargs)
        details['semantic_natives'].append(result.inputs['input_ids'].tolist())
        return result
    original_components = test.component_metrics
    component_index = 0
    def components(nll, offsets, semantic_record):
        nonlocal component_index
        result = original_components(nll, offsets, semantic_record)
        if component_index == row:
            keep('semantic_nll', nll)
            details['component_metrics'] = result
            start, ranges = 0, {}
            for name in ('what', 'why', 'how'):
                end = start + result[f'{name}_response_tokens']
                ranges[name] = [start, end]
                start = end
            details['component_token_ranges_half_open'] = ranges
        component_index += 1
        return result
    with ExitStack() as stack:
        stack.enter_context(patch.object(native, 'get_image_features', vision))
        stack.enter_context(patch.object(test, 'forward_where_batch', where))
        stack.enter_context(patch.object(where_module, 'compute_selected_causal_nll', selected_wrapper(where_module.compute_selected_causal_nll)))
        if cache is not None:
            stack.enter_context(patch.object(cache, 'fuse', fuse))
        for module, hook, pre in ((native.language_model, lm_pre, True), (native, backbone_post, False),
                                  (bundle.model.get_output_embeddings(), head_post, False)):
            handle = module.register_forward_pre_hook(hook, with_kwargs=True) if pre else module.register_forward_hook(hook)
            stack.callback(handle.remove)
        if semantic:
            stack.callback(bundle.projector.register_forward_hook(projector_post).remove)
            stack.enter_context(patch.object(test, 'prepare_flat_inputs', prepared(test.prepare_flat_inputs, True)))
            stack.enter_context(patch.object(flat_module, 'prepare_semantic_batch', prepared(flat_module.prepare_semantic_batch, False)))
            stack.enter_context(patch.object(flat_module, 'collate_native', collate))
            stack.enter_context(patch.object(flat_module, 'compute_selected_causal_nll', selected_wrapper(flat_module.compute_selected_causal_nll)))
            stack.enter_context(patch.object(test, 'component_metrics', components))
        try:
            if serial:
                values = test.serial_reference_losses(bundle, batch.episodes[0], batch=batch)
                if semantic:
                    keep('semantic_episode_loss', torch.tensor(values['test_flat']))
            else:
                test.batch_losses(bundle, batch.episodes, batch=batch, cache=cache)
        except WhereFinished:
            pass
    if semantic:
        tensors['semantic_input_ids'] = torch.tensor(details.pop('semantic_natives')[row][0])
    else:
        details.pop('semantic_natives')
    return tensors, details


def trace_parity(bundle, workloads, settings, report, tolerance, *, reference=None):
    import torch
    from semgaze.evaluation.test import test_mode, batch_losses
    from semgaze.evaluation.batching import schedule_batches, episode_id
    from semgaze.model.visual_cache import InferenceVisualCache
    from semgaze.where.collator import to_model_device
    parity = report.get('parity', {})
    metrics = parity.get('metrics', {})
    where_passed = metrics.get('test_where', {}).get('passed', False)
    semantic = where_passed and any(not v['passed'] for k, v in metrics.items() if k != 'test_where')
    metric = 'test_where' if not semantic else next(k for k, v in metrics.items() if not v['passed'])
    target = metrics.get(metric, {}).get('episode_with_max_difference')
    if target is None:
        # Explicit stand-alone trace: pick a short/long same-K pair.
        k = next(iter(workloads))
        rows = sorted((r for r in report['episodes'] if r['K'] == k), key=lambda r: (r['where_length'], r['episode_id']))
        target = rows[0]['episode_id']
        ids = {target, rows[-1]['episode_id']}
        trace_workloads = {k: [e for e in workloads[k] if episode_id(e) in ids]}
        settings = dict(settings, execution='same_k_batched', batch_size_by_k={str(k): 2}, bucket_by_length=False)
    else:
        trace_workloads = workloads
    episode = next(e for es in workloads.values() for e in es if episode_id(e) == target)
    cache = InferenceVisualCache(preprocessing=report['preprocess_cache'], features=report['visual_cache'],
                                 max_entries=bundle.config['test']['cache']['max_entries'])
    result = dict(status='running', target_episode=target, semantic_trace=semantic,
                  scalar_tolerance=tolerance, reference='serial cache-off, target alone',
                  FIRST_DIVERGENCE='unresolved', replayed_batches=0)
    try:
        with test_mode(bundle):
            # Replay the original scheduler/cache history, including previous K groups.
            # This preserves cache hits, feature producer batch shapes and LRU eviction.
            found = False
            for episodes in trace_workloads.values():
                for batch in schedule_batches(bundle, episodes, settings, cache=cache):
                    ids = [episode_id(e) for e in batch.episodes]
                    if target in ids:
                        found = True
                        break
                    if report['preprocess_cache'] or report['visual_cache']:
                        batch_losses(bundle, batch.episodes, batch=batch, cache=cache)
                        result['replayed_batches'] += 1
                if found:
                    break
            if not found:
                raise ValueError('trace target missing from schedule')
            row = ids.index(target)
            off = InferenceVisualCache(preprocessing=False, features=False)
            try:
                alone = next(schedule_batches(bundle, [episode], dict(execution='serial',
                    batch_size_by_k={str(len(episode.supports)): 1}), cache=off))
                result['batch_episode_ids'] = ids
                result['inputs'] = trace_inputs(bundle, alone, batch, row)
                if not result['inputs']['passed']:
                    result.update(status='complete', FIRST_DIVERGENCE='inputs.' + next(
                        k for k, v in result['inputs']['checks'].items() if v == 'FAIL'))
                    return result
                a, da = capture_trace(bundle, alone, 0, serial=True, cache=None, semantic=semantic)
                b, db = capture_trace(bundle, batch, row, serial=False, cache=cache, semantic=semantic)
            finally:
                off.close()
            result['execution'] = dict(reference=da, candidate=db)
            result['where_loss_effect'] = tensor_difference(a['where_episode_loss'], b['where_episode_loss'], tolerance)
            result['where_loss_values'] = [float(a['where_episode_loss']), float(b['where_episode_loss'])]
            measured = next(r for r in report['episodes'] if r['episode_id'] == target)['test_where']
            result['candidate_replay_loss_diff'] = abs(measured - float(b['where_episode_loss']))
            if reference is not None:
                measured_reference = next(r for r in reference['episodes'] if r['episode_id'] == target)['test_where']
                result['reference_replay_loss_diff'] = abs(measured_reference - float(a['where_episode_loss']))
            stages = ['visual_features', 'where_fused_inputs', 'where_hidden', 'where_logits', 'where_nll',
                      'where_episode_loss']
            if semantic:
                stages += ['query_END_FIX_states', 'projected_R_states', 'semantic_input_ids', 'semantic_labels',
                           'semantic_attention_mask', 'semantic_supervised_positions', 'semantic_fused_inputs',
                           'semantic_hidden', 'semantic_logits', 'semantic_nll', 'semantic_episode_loss']
            comparisons, first_bitwise, first = {}, None, None
            for stage in stages:
                if stage not in a or stage not in b:
                    result['unavailable_stage'] = stage
                    first = 'unresolved: ' + stage
                    break
                diff = tensor_difference(a[stage], b[stage], 0 if stage in (
                    'semantic_input_ids', 'semantic_labels', 'semantic_attention_mask', 'semantic_supervised_positions') else tolerance)
                comparisons[stage] = diff
                if stage == 'visual_features' and a[stage].shape == b[stage].shape:
                    result['visual_features_per_image'] = [dict(path=path, **tensor_difference(x, y, tolerance))
                        for path, x, y in zip(alone.samples[0].image_paths, a[stage], b[stage])]
                if not diff['bitwise_equal'] and first_bitwise is None:
                    first_bitwise = stage
                if not diff['allclose']:
                    first = stage
                    break
            result['tensors'] = comparisons
            result['FIRST_BITWISE_MISMATCH'] = first_bitwise
            result['FIRST_DIVERGENCE'] = first or ('semantic_component_losses' if semantic and
                any(abs(da['component_metrics'][key]-db['component_metrics'][key]) > tolerance
                    for key in ('test_flat', 'test_what', 'test_why', 'test_how')) else 'none at tensor threshold')
            result['model_input_checks'] = {}
            for key in ('where_model_attention_mask', 'where_model_position_ids', 'where_model_cache_position'):
                passed = tensor_difference(a[key], b[key], 0)['allclose'] if key in a and key in b else (
                    key not in a and key not in b and da.get(key) == db.get(key))
                result['model_input_checks'][key] = 'PASS' if passed else 'FAIL'
            if any(v == 'FAIL' for v in result['model_input_checks'].values()):
                result['FIRST_DIVERGENCE'] = 'WHERE model attention/position inputs'
            material = not result['where_loss_effect']['allclose']
            result['mismatch_class'] = ('material episode loss mismatch under unchanged parity criterion' if material else
                'tensor numerical mismatch without WHERE loss parity failure' if first_bitwise else 'bitwise equal at observed stages')
            if semantic:
                result['semantic_components'] = dict(reference=da.get('component_metrics'), candidate=db.get('component_metrics'))
                result['semantic_reduction'] = ('compute_selected_causal_nll: sum NLL per episode / episode token count; '
                    'EpisodeAccumulator: sum episode metrics / episode count. No global token weighting.')
                result['semantic_token_counts'] = [len(a['semantic_nll']), len(b['semantic_nll'])]
                result['semantic_response_offsets_equal'] = da['response_offsets'] == db['response_offsets']
                result['semantic_padding_checks'] = [da['semantic_padding_valid'], db['semantic_padding_valid']]
                result['semantic_component_token_ranges'] = [da['component_token_ranges_half_open'], db['component_token_ranges_half_open']]
                result['semantic_episode_mean_checks'] = [tensor_difference(x['semantic_nll'].mean(),
                    x['semantic_episode_loss'], tolerance) for x in (a, b)]
            # Only a failed visual-cache branch justifies additional vision probes.
            if report['visual_cache'] and not parity.get('passed', True):
                pixels = to_model_device({'pixel_values': batch.inputs['pixel_values']}, bundle.model)['pixel_values']
                native = bundle.model.get_base_model().model
                m = batch.metadata[row]
                direct = native.get_image_features(pixels, return_dict=True).pooler_output
                one = native.get_image_features(pixels[m['query_image_index']:m['query_image_index']+1], return_dict=True).pooler_output
                result['visual_cache_audit'] = dict(
                    direct_vs_cached=tensor_difference(direct[m['image_offset']:m['image_offset']+m['image_count']], b['visual_features'], tolerance),
                    query_image_alone_vs_image_batch=tensor_difference(one[0], direct[m['query_image_index']], tolerance),
                    feature_shape_per_image=list(direct.shape[1:]), dtype=str(direct.dtype), device=str(direct.device),
                    feature_reuse_key='(image_path, pixel dtype, pixel device); invocation-scoped',
                    reuse_keys=[list(key) for key in cache.visual if key[0] in batch.samples[row].image_paths],
                    detached='no explicit detach; inference_mode, cloned cache rows, requires_grad=False',
                    operations='both invoke native.get_image_features (vision_tower then multi_modal_projector); '
                               'cache deduplicates misses and bypasses native image insertion via masked_scatter',
                    ordering='sample, support, query; keys expanded back to original image order',
                    fusion_offsets=result['inputs']['image_offsets'],
                    semantic_reuse='visual flag also controls WHERE query-feature reuse in semantic preparation')
            result['status'] = 'complete'
            return result
    finally:
        cache.close()


def audit_verdict(experiments):
    failed = next((r for r in experiments if r['status'] != 'complete'), None)
    verdict = dict(category='H. unresolved', confidence='low', first_divergent_stage='not observed',
        minimum_reproducer='none', evidence='All requested comparisons passed on this workload.',
        ruled_out=[r['audit_stage'] for r in experiments if r['status'] == 'complete'],
        next_implementation_target='No fix justified; run full scope if this was quick scope.')
    if failed is None:
        return verdict
    stage = failed['audit_stage']
    trace = failed.get('tensor_trace', {})
    verdict.update(first_divergent_stage=trace.get('FIRST_DIVERGENCE', 'unresolved'),
                   minimum_reproducer=dict(stage=stage, command=failed.get('command'),
                       target_episode=trace.get('target_episode'), batch=trace.get('batch_episode_ids')),
                   evidence=failed.get('error') or failed.get('parity'),
                   next_implementation_target='Resolve execution/trace failure before attributing a cause.')
    if failed['status'] != 'parity_failed':
        return verdict
    verdict['evidence'] = '; '.join(f'{name} max diff={value["max_absolute_difference"]:.8g} '
        f'(tolerance={value["tolerance"]}, episode={value["episode_with_max_difference"]})'
        for name, value in failed['parity']['metrics'].items() if not value['passed'])
    categories = dict(B0='A. serial-vs-batched implementation mismatch at B=1',
        B1='B. physical batching / padding / packing mismatch', C='C. bucketing / scheduling mismatch',
        D='D. preprocessing-cache mismatch', E='E. visual-feature-cache mismatch')
    targets = dict(B0='serial_reference_losses vs batch_losses around shared forward_where_batch',
        B1='WHERE packing and InternVL forward', C='schedule_batches packing/order',
        D='InferenceVisualCache.processor_for', E='InferenceVisualCache.fuse/get_image_features and feature insertion')
    verdict.update(category=categories[stage], confidence='high' if stage in ('D', 'E') else 'medium',
                   next_implementation_target=targets[stage])
    if failed['parity']['metrics']['test_where']['passed']:
        verdict.update(category='G. semantic-path mismatch after WHERE parity', confidence='medium',
                       next_implementation_target='serial prepare_flat_inputs/native loss vs forward_flat_batch/selected loss')
        if trace.get('FIRST_DIVERGENCE') in ('visual_features', 'where_fused_inputs', 'where_hidden',
                'where_logits', 'where_nll', 'query_END_FIX_states'):
            verdict.update(category=categories[stage], confidence='medium',
                           next_implementation_target=targets[stage] +
                           '; WHERE scalar passed, but upstream tensors diverge before semantic inputs')
    elif (stage == 'B1' and trace.get('inputs', {}).get('passed') and trace.get('model_input_checks') and
          all(v == 'PASS' for v in trace['model_input_checks'].values()) and
          trace.get('FIRST_DIVERGENCE') in ('visual_features', 'where_hidden', 'where_logits', 'where_nll', 'where_episode_loss')):
        verdict.update(category='F. batch-shape-dependent model numerics', confidence='medium',
            next_implementation_target='Candidate: batch-shape-dependent BF16/CUDA kernels at ' + trace['FIRST_DIVERGENCE'] +
            '; not automatically acceptable. Inspect model operation before any precision/tolerance decision.')
    if trace.get('status') != 'complete' or max(trace.get('candidate_replay_loss_diff', 0),
            trace.get('reference_replay_loss_diff', 0)) > failed['parity']['metrics']['test_where']['tolerance']:
        verdict['confidence'] = 'low'
    if trace.get('status') == 'complete' and trace.get('FIRST_DIVERGENCE') == 'none at tensor threshold':
        verdict.update(category='H. unresolved', confidence='low',
                       next_implementation_target='Scalar failure did not reproduce at captured tensor stages; inspect replay and first bitwise mismatch.')
    return verdict


def write_audit_report(directory, report):
    save_json(directory / 'report.json', report)
    lines = ['# Test parity isolation audit', '', f'Status: {report["status"]}', '',
             '| Experiment | Status | B by K | Preprocess | Visual | Bucket | Episodes/s | Wall s | Alloc GiB | Reserved GiB |',
             '|---|---|---|---|---|---|---:|---:|---:|---:|']
    for r in report['experiments']:
        lines.append(f'| {r["audit_stage"]} | {r["status"]} | {r.get("batch_size_by_k")} | '
            f'{r.get("preprocess_cache")} | {r.get("visual_cache")} | {r.get("length_bucketing")} | '
            f'{r.get("episodes_per_second")} | {r.get("total_wall_time_sec")} | '
            f'{r.get("peak_allocated_GiB")} | {r.get("peak_reserved_GiB")} |')
    for r in report['experiments']:
        if r.get('parity'):
            lines += ['', f'{r["audit_stage"]} versus A:', '', '| Metric | Max diff | Mean diff | Worst episode | Pass |',
                      '|---|---:|---:|---|---|']
            for key, value in r['parity']['metrics'].items():
                lines.append(f'| {key} | {value["max_absolute_difference"]:.8g} | {value["mean_absolute_difference"]:.8g} | '
                             f'{value["episode_with_max_difference"]} | {value["passed"]} |')
            lines.append('')
    lines += ['', f'Skipped after stop: {report["skipped_experiments"]}',
              'Performance excludes tensor trace and cache-history replay. Tensor allclose uses the unchanged scalar '
              'tolerance with rtol=0 as a diagnostic threshold; it does not establish scientific acceptability.',
              'See the failing experiment tensor_trace.json for input PASS/FAIL checks, tensor summaries and relative errors.']
    trace = next((r['tensor_trace'] for r in report['experiments'] if 'tensor_trace' in r), {})
    lines += ['', f'FIRST_DIVERGENCE = {trace.get("FIRST_DIVERGENCE", "not observed")}',
              '', '## Root-Cause Verdict', '']
    for key, value in report['verdict'].items():
        if key == 'minimum_reproducer' and isinstance(value, dict):
            value = (f'stage {value["stage"]}, target {value["target_episode"]}, batch {value["batch"]}; '
                     f'exact argv in the {value["stage"]} experiment command field in report.json')
        lines.append(f'- {key.replace("_", " ")}: {value}')
    (directory / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def audit_suite(args):
    """Fresh-process experiments reuse run()/compare_reports(); stop on first failure."""
    stages = [('A', 'serial', False, False, False), ('B0', 'same-k-batched', False, False, False),
              ('B1', 'same-k-batched', False, False, False), ('C', 'same-k-batched', False, False, True),
              ('D', 'same-k-batched', True, False, True), ('E', 'same-k-batched', True, True, True)]
    report = dict(status='running', scope=args.scope, experiments=[], tolerance=args.tolerance)
    try:
        for stage, mode, preprocess, visual, bucket in stages:
            directory = args.output_dir / stage
            command = [sys.executable, str(Path(__file__).resolve()), '--checkpoint', str(args.checkpoint.resolve()),
                '--output-dir', str(directory.resolve()), '--mode', mode, '--scope', args.scope,
                '--device', args.device, '--cache', 'off', '--preprocess-cache', 'on' if preprocess else 'off',
                '--visual-cache', 'on' if visual else 'off', '--audit-stage', stage, '--trace-parity',
                '--quick-per-subject', str(args.quick_per_subject), '--warmup', str(args.warmup),
                '--tolerance', str(args.tolerance), '--bucket-window', str(args.bucket_window)]
            if args.config:
                command += ['--config', str(args.config.resolve())]
            if args.k is not None:
                command += ['--k', str(args.k)]
            if args.profile:
                command += ['--profile']
            if stage in ('A', 'B0'):
                command += ['--batch-size', '1']
            else:
                for k, default in ((1, 4), (5, 2), (10, 2)):
                    command += [f'--batch-k{k}', str(args.batch_size or getattr(args, f'batch_k{k}') or default)]
            if bucket:
                command += ['--bucket']
            if stage != 'A':
                command += ['--reference', str((args.output_dir / 'A' / 'report.json').resolve())]
            print(f'[AUDIT] {stage}: preprocess={preprocess} visual={visual} bucket={bucket}', flush=True)
            completed = subprocess.run(command, cwd=ROOT)
            path = directory / 'report.json'
            result = json.loads(path.read_text()) if path.exists() else dict(status='failed',
                error=f'worker exited {completed.returncode} without a report')
            if completed.returncode and result['status'] == 'complete':
                result.update(status='failed', error=f'worker exited {completed.returncode} after writing report')
            result.update(audit_stage=stage, command=command)
            for kind in ('allocated', 'reserved'):
                value = result.get(f'peak_memory_{kind}_bytes')
                result[f'peak_{kind}_GiB'] = value / 2**30 if value is not None else None
            if stage == 'A' and result['status'] == 'complete':
                result['parity'] = compare_reports(result, result, args.tolerance)
            report['experiments'].append(result)
            if completed.returncode or result['status'] != 'complete':
                break
        report['status'] = 'complete' if len(report['experiments']) == len(stages) and all(
            r['status'] == 'complete' for r in report['experiments']) else 'stopped'
    except Exception as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
    finally:
        baseline = report['experiments'][0] if report['experiments'] else {}
        for key in ('environment', 'checkpoint', 'config_path', 'checkpoint_sha256', 'workload_sha256',
                    'scientific_config_sha256', 'split_manifest_identity', 'dtype', 'attention_backend',
                    'episode_count_total', 'episode_ids', 'episode_count_by_k'):
            report[key] = baseline.get(key)
        report['skipped_experiments'] = [s[0] for s in stages[len(report['experiments']):]]
        report['verdict'] = audit_verdict(report['experiments'])
        if report.get('error'):
            report['verdict'].update(evidence=report['error'], category='H. unresolved', confidence='low')
        write_audit_report(args.output_dir, report)
    return 0 if report['status'] == 'complete' else 1


def sweep(args, argv):
    if args.k is None:
        raise ValueError('--sweep requires --k')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    # Reconstruct child arguments without the coordinator's candidate controls.
    base = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg in ('--sweep', '--output-dir'):
            skip = True
            continue
        base.append(arg)
    for size in args.sweep:
        directory = args.output_dir / f'k{args.k}-b{size}'
        command = [sys.executable, str(Path(__file__).resolve()), *base,
                   '--batch-size', str(size), '--output-dir', str(directory)]
        completed = subprocess.run(command, cwd=ROOT)
        if (directory / 'report.json').exists():
            report = json.loads((directory / 'report.json').read_text())
        else:
            report = dict(status='failed', mode=args.mode, scope=args.scope,
                          batch_size_by_k={str(args.k):size},
                          error=f'worker exited {completed.returncode} without a report')
            directory.mkdir(parents=True, exist_ok=True)
            write_report(directory, report)
        results.append(report)
        if completed.returncode and report['status'] != 'oom':
            break
    useful = [r for r in results if r['status'] == 'complete' and
              r['peak_memory_reserved_bytes'] <= args.memory_fraction*r['environment']['gpu_total_memory']]
    best = max(useful, key=lambda r: r['episodes_per_second']) if useful else None
    summary = dict(K=args.k, candidates=results, memory_fraction=args.memory_fraction,
                   chosen_batch_size=best['batch_size_by_k'][str(args.k)] if best else None,
                   choice_status='measured throughput with reserved-memory headroom; full parity still required')
    save_json(args.output_dir / 'sweep.json', summary)
    lines = [f'# K={args.k} batch sweep', '', '| B | Status | Episodes/s | Peak allocated | Peak reserved |',
             '|---:|---|---:|---:|---:|']
    for r in results:
        lines.append(f'| {r.get("batch_size_by_k",{}).get(str(args.k))} | {r["status"]} | {r.get("episodes_per_second")} | {r.get("peak_memory_allocated_bytes")} | {r.get("peak_memory_reserved_bytes")} |')
    lines += ['', f'Measured choice: {summary["chosen_batch_size"]}. {summary["choice_status"]}.']
    (args.output_dir / 'sweep.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return 0 if best else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('serial', 'same-k-batched'), default='same-k-batched')
    parser.add_argument('--scope', choices=('quick', 'full'), default='quick')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-k1', type=int)
    parser.add_argument('--batch-k5', type=int)
    parser.add_argument('--batch-k10', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--sweep', type=lambda s: [int(x) for x in s.split(',')])
    parser.add_argument('--memory-fraction', type=float, default=0.9)
    parser.add_argument('--cache', choices=('on', 'off'), default='on')
    parser.add_argument('--preprocess-cache', choices=('on', 'off'),
                        help='override --cache for decoded images and preprocessing only')
    parser.add_argument('--visual-cache', choices=('on', 'off'),
                        help='override --cache for vision features and semantic query-feature reuse')
    parser.add_argument('--audit-parity-suite', action='store_true',
                        help='run A/B0/B1/C/D/E in order; stop and trace the first failure')
    parser.add_argument('--trace-parity', action='store_true',
                        help='untimed input/tensor trace; automatically used on suite parity failure')
    parser.add_argument('--audit-stage', choices=('A', 'B0', 'B1', 'C', 'D', 'E'), help=argparse.SUPPRESS)
    parser.add_argument('--bucket', action='store_true')
    parser.add_argument('--bucket-window', type=int, default=64)
    parser.add_argument('--quick-per-subject', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--tolerance', type=float, default=0.002)
    parser.add_argument('--compare', type=Path, nargs=2, metavar=('SERIAL', 'BATCHED'))
    parser.add_argument('--prediction-parity-budget', type=int,
                        help='after timing, compare serial/batched greedy predictions using this explicit semantic budget')
    args = parser.parse_args(argv)
    if args.tolerance < 0 or not 0 < args.memory_fraction < 1:
        parser.error('invalid tolerance or memory fraction')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.compare:
        result = compare_reports(*(json.loads(p.read_text()) for p in args.compare), args.tolerance)
        write_parity(args.output_dir, result)
        return 0 if result['passed'] else 1
    if any(args.output_dir.iterdir()):
        parser.error('choose an empty output directory for a measured run or sweep')
    if args.checkpoint is None:
        parser.error('--checkpoint is required for measured runs')
    if args.quick_per_subject < 2 or args.bucket_window < 1 or args.warmup < 0 or any(
            v is not None and v < 1 for v in (args.batch_k1,args.batch_k5,args.batch_k10,args.batch_size,args.prediction_parity_budget)):
        parser.error('positive batch/window sizes, at least two quick queries per subject, nonnegative warmup required')
    if args.sweep and (args.mode != 'same-k-batched' or any(b < 1 for b in args.sweep)):
        parser.error('sweep requires positive batched candidates')
    if args.audit_parity_suite:
        if args.sweep or args.reference or args.prediction_parity_budget is not None:
            parser.error('audit suite cannot be combined with sweep, reference or prediction parity')
        if any(v == 1 for v in (args.batch_k1, args.batch_k5, args.batch_k10, args.batch_size)):
            parser.error('audit B1 requires physical B>1; B0 already measures B=1')
        return audit_suite(args)
    return sweep(args, argv) if args.sweep else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
