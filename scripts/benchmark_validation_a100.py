#!/usr/bin/env python3
"""Checkpoint-exact validation parity, fresh-process sweeps, and full CUDA reports."""
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

METRICS = ('eval_where', 'eval_flat', 'eval_what', 'eval_why', 'eval_how', 'eval_total')


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
    lines = ['# Validation benchmark', '', f'Status: {report["status"]}', '',
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
    from semgaze.evaluation.validation import validation_mode
    from semgaze.model.visual_cache import InferenceVisualCache
    settings = bundle.config['validation']['prediction']
    previous = deepcopy(settings)
    results = {}
    try:
        with validation_mode(bundle):
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
    from semgaze.evaluation.validation import (validation_queries, validation_mode, batch_losses,
                                              serial_reference_losses, EpisodeAccumulator)
    from semgaze.evaluation.batching import schedule_batches, episode_id
    from semgaze.model.visual_cache import InferenceVisualCache
    from semgaze.training.profiling import TrainingProfiler, runtime_metadata
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    cache_enabled = args.cache == 'on' and args.mode != 'serial'
    report = dict(timestamp=timestamp(), environment=machine_metadata(), status='running', mode=args.mode,
                  scope=args.scope, config_path=str(args.config), checkpoint=str(args.checkpoint),
                  episodes=[], by_k={}, inference_mode=True, same_k_batching=args.mode != 'serial',
                  length_bucketing=args.bucket and args.mode != 'serial', visual_cache=cache_enabled,
                  support_cache=cache_enabled, prefix_kv_cache=False,
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
            config = resolve_config(config, {'validation': overrides['validation']})
        config['runtime']['device'] = args.device
        data = config['data']
        raw, manifest, identity = read_persisted_splits(ROOT / data['split_root'], data_config=data)
        adapter = CocoSearch18Adapter(ROOT / data['images_root'], annotation_frame=data['annotation_frame'],
                                     duration_field=data['duration']['source_field'])
        train = {r['record_id']: adapter(r) for r in raw['train']}
        queries = list(validation_queries(train, [adapter(r) for r in raw['validation']], manifest, data['unseen_subjects']))
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
                 config['validation']['loss']['batch_size_by_k'].get(k,
                 config['validation']['loss']['batch_size_by_k'].get(str(k))) for k in ks}
        if args.batch_size:
            sizes = {str(k): args.batch_size for k in ks}
        settings = dict(execution='same_k_batched', batch_size_by_k=sizes,
                        bucket_by_length=args.bucket, bucket_window=args.bucket_window)
        if args.mode == 'serial':
            settings.update(execution='serial', bucket_by_length=False)
            sizes = {str(k): 1 for k in ks}
            settings['batch_size_by_k'] = sizes
        report.update(batch_size_by_k=sizes, canonical_query_count=canonical_count, query_count=len(queries),
            k_values=ks, episode_count_total=len(queries)*len(ks), episode_count_by_k={str(k): len(queries) for k in ks},
            split_manifest_identity=identity, checkpoint_sha256=checkpoint_digest(args.checkpoint),
            scientific_config_sha256=digest({k: config[k] for k in ('data', 'model', 'where', 'state', 'semantic', 'training')}),
            base_model=config['model']['base_model'], adapter=config['model']['initialization_adapter'])
        bundle = load_checkpoint_bundle(args.checkpoint, split_manifest_identity=identity,
                    output_dir=directory / 'model', runtime=config['runtime'])
        bundle.config = config
        config['validation']['cache']['frozen_visual_features'] = cache_enabled
        report['model_metadata'] = runtime_metadata(bundle)
        report['dtype'] = report['model_metadata']['dtype']
        report['attention_backend'] = report['model_metadata']['text_attention']
        workloads = {k: [frozen_episode(q, train, manifest, k, unseen_subjects=data['unseen_subjects']) for q in queries] for k in ks}
        report['workload_sha256'] = digest([{'episode_id': episode_id(e), 'support_ids': [s.record_id for s in e.supports]}
                                          for es in workloads.values() for e in es])
        profiler = TrainingProfiler(device) if args.profile else None
        overall = EpisodeAccumulator()
        with validation_mode(bundle):
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
            cache = InferenceVisualCache(preprocessing=cache_enabled, features=cache_enabled,
                                        max_entries=config['validation']['cache']['max_entries'])
            start = time.perf_counter()
            for k, episodes in workloads.items():
                active_k, active_b = k, sizes[str(k)]
                torch.cuda.reset_peak_memory_stats(device)
                k_start = time.perf_counter()
                batches = []
                for batch in schedule_batches(bundle, episodes, settings, cache=cache, profiler=profiler):
                    if args.mode == 'serial':
                        ep = batch.episodes[0]
                        rows = [serial_reference_losses(bundle, ep, batch=batch)]
                    else:
                        rows = batch_losses(bundle, batch.episodes, batch=batch, cache=cache, profiler=profiler)
                    for row in rows:
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
    return sweep(args, argv) if args.sweep else run(args)


if __name__ == '__main__':
    raise SystemExit(main())
