#!/usr/bin/env python3
"""Benchmark the real SemGaze training pipeline on one CUDA device.

Run from a prepared repository on the server. No fixture, tiny model, CPU
fallback, model tuning, or test-suite imports. Each variant runs in a fresh
process. --plan-only writes unmeasured JSON/CSV templates without loading CUDA.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import time
import traceback
import csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA_VERSION = 1
STAGES = ('sampling_collation', 'where_forward', 'semantic_preparation',
          'semantic_forward', 'backward', 'gradient_clipping', 'optimizer_step', 'scheduler_step')
METRICS = ('episodes_per_second', 'seconds_per_episode', 'step_time_mean_sec',
           'step_time_median_sec', 'step_time_p95_sec', 'total_measured_sec',
           'peak_memory_allocated_bytes', 'peak_memory_reserved_bytes',
           *(f'{stage}_mean_sec' for stage in STAGES))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def command_output(command):
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=15)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def machine_metadata():
    packages = {}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'safetensors', 'Pillow'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source_files = sorted((ROOT / 'semgaze').rglob('*.py')) + [Path(__file__)]
    return dict(hostname=socket.gethostname(), platform=platform.platform(),
        python=sys.version, executable=sys.executable, cpu_count=os.cpu_count(), packages=packages,
        git_commit=command_output(['git', 'rev-parse', 'HEAD']),
        git_status=command_output(['git', 'status', '--short']),
        source_sha256=digest({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in source_files}),
        environment={key: os.environ.get(key) for key in (
            'CUDA_VISIBLE_DEVICES', 'PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF',
            'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUBLAS_WORKSPACE_CONFIG')})


def variant_matrix(effective_batch, include_off=False):
    if type(effective_batch) is not int or effective_batch < 2 or effective_batch % 2:
        raise ValueError('effective batch must be a positive multiple of 2 to compare B=1 and B=2')
    return [dict(id=f'b{b}_gc_{"on" if gc else "off"}', physical_batch_size=b,
                 gradient_accumulation_steps=effective_batch // b, gradient_checkpointing=gc,
                 effective_optimizer_batch=effective_batch)
            for gc in ([True, False] if include_off else [True]) for b in (1, 2)]


def validate_real_config(config):
    model, training = config['model'], config['training']
    if model['base_model'] != 'OpenGVLab/InternVL3_5-8B-HF':
        raise ValueError('benchmark requires OpenGVLab/InternVL3_5-8B-HF')
    if training['precision'] != 'bf16':
        raise ValueError('benchmark requires configured bf16 precision')
    if config['where']['supervision']['use_cache']:
        raise ValueError('checkpointing-ON variants require where.supervision.use_cache: false')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('run directly with python on one GPU, not under distributed torchrun')


def empty_result(variant):
    return dict(variant, status='not_run', failure_phase=None, failure_stage=None, error=None,
                measured_steps_completed=0, measured_episodes=0,
                metrics=dict.fromkeys(METRICS), workload_groups=[], steps=[], warmup_steps=[],
                memory={}, model_metadata={}, measured_K_histogram={})


def scheduler_horizon(config, benchmark_steps):
    training = config['training']
    horizon = training['max_steps']
    if horizon is None and training['epochs'] is not None and training['steps_per_epoch'] is not None:
        horizon = training['epochs'] * training['steps_per_epoch']
    if horizon is not None and horizon < benchmark_steps:
        raise ValueError('benchmark warmup + steps exceeds the configured training horizon; reduce benchmark steps')
    return benchmark_steps if horizon is None else horizon


class BoundaryProfiler:
    """Synchronized wall-clock stage times; nested parent stages are inclusive.

    Synchronization is used only at explicit profiling boundaries. No allocator
    empty_cache calls. The outer step includes CPU collation and device transfers.
    """
    def __init__(self, torch_module, device):
        self.torch, self.device = torch_module, device
        self.times = {}
        self.failed_stage = None

    @contextmanager
    def stage(self, name):
        try:
            self.torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            yield
            self.torch.cuda.synchronize(self.device)
        except BaseException:
            if self.failed_stage is None:
                self.failed_stage = name
            raise
        else:
            self.times[name] = self.times.get(name, 0.0) + time.perf_counter() - start


def memory_snapshot(torch_module, device):
    return {key: int(fn(device)) for key, fn in (
        ('memory_allocated_bytes', torch_module.cuda.memory_allocated),
        ('memory_reserved_bytes', torch_module.cuda.memory_reserved),
        ('peak_memory_allocated_bytes', torch_module.cuda.max_memory_allocated),
        ('peak_memory_reserved_bytes', torch_module.cuda.max_memory_reserved))}


def episode_identity(episode):
    return dict(query_id=episode.query.record_id, subject=episode.query.subject,
                support_ids=[s.record_id for s in episode.supports])


def summarize(result):
    """Only completed measured steps contribute to rates; failed peaks remain visible."""
    rows = result['steps']
    result['measured_steps_completed'] = len(rows)
    result['measured_episodes'] = sum(len(row['episodes']) for row in rows)
    result['workload_sha256'] = digest(result['workload_groups'])
    result['measured_K_histogram'] = dict(Counter(str(e['K']) for row in rows for e in row['episodes']))
    metrics = dict.fromkeys(METRICS)
    if rows:
        times = [r['step_time_sec'] for r in rows]
        total = sum(times)
        n = result['measured_episodes']
        metrics.update(episodes_per_second=n / total, seconds_per_episode=total / n,
            step_time_mean_sec=statistics.mean(times), step_time_median_sec=statistics.median(times),
            step_time_p95_sec=sorted(times)[max(0, math.ceil(0.95 * len(times)) - 1)], total_measured_sec=total)
        for stage in STAGES:
            metrics[f'{stage}_mean_sec'] = statistics.mean(r['stage_seconds'].get(stage, 0) for r in rows)
    # Never label model-loading or warmup peaks as measured-step peaks.
    measured_memory = result['memory'].get('measured', {})
    for key in ('peak_memory_allocated_bytes', 'peak_memory_reserved_bytes'):
        metrics[key] = measured_memory.get(key)
    result['metrics'] = metrics
    result['rates_are_partial'] = result['status'] != 'ok' and bool(rows)


def load_real_sampler(config):
    """Same persisted split, normalization and sampler as train_flat.main."""
    from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter
    from semgaze.data.fewshot import TrainingEpisodeSampler
    data = config['data']
    raw, manifest, identity = read_persisted_splits(ROOT / data['split_root'], data_config=data)
    adapter = CocoSearch18Adapter(ROOT / data['images_root'], annotation_frame=data['annotation_frame'],
                                  duration_field=data['duration']['source_field'])
    records = [adapter(r) for r in raw['train'] if r['subject'] not in data['unseen_subjects']]
    return TrainingEpisodeSampler(records, config['experiment']['seed'], data_config=data), dict(
        split_manifest_sha256=identity, train_record_count=len(records),
        train_subjects=sorted({r.subject for r in records}), split='train',
        k_values=data['fewshot']['k_values'], k_probabilities=data['fewshot']['train_k_probabilities'])


def run_measured_step(bundle, sampler, step, profiler, result):
    from semgaze.training.batching import sample_optimizer_batches
    from semgaze.training.flat_step import run_flat_training_step, clip_and_check_gradients
    t = bundle.config['training']
    profiler.times = {}
    episode_rows, losses = [], []
    with profiler.stage('step'):
        with profiler.stage('sampling_collation'):
            batches, accepted, rejected = sample_optimizer_batches(bundle, sampler)
        result['workload_groups'].append([episode_identity(e) for e in accepted])
        result['inflight_episodes'] = [
            {k: v for k, v in m.items() if k != 'semantic_target'} | episode_identity(e)
            for batch in batches for e, m in zip(batch.episodes, batch.metadata)]
        interval = t.get('gradient_diagnostics_every', 0)
        diagnose = step == 0 or (interval > 0 and (step + 1) % interval == 0)
        for micro, batch in enumerate(batches):
            output = run_flat_training_step(bundle, batch.episodes, where_batch=batch,
                zero_grad=micro == 0, loss_scale=1 / len(batches), profiler=profiler,
                diagnostics=diagnose and micro == len(batches)-1)
            for e, metadata in zip(batch.episodes, output['episodes']):
                episode_rows.append(metadata | episode_identity(e) | {'physical_batch_index': micro})
            losses.append(output['loss_total'])
        with profiler.stage('gradient_clipping'):
            clip_and_check_gradients(bundle)
        with profiler.stage('optimizer_step'):
            bundle.optimizer.step()
        with profiler.stage('scheduler_step'):
            bundle.scheduler.step()
    # CUDA is already synchronized by the outer boundary. Report I/O is excluded.
    loss = float(sum(losses) / len(losses))
    if not math.isfinite(loss):
        raise RuntimeError('non-finite total loss')
    result.pop('inflight_episodes', None)
    return dict(optimizer_step=step+1, step_time_sec=profiler.times['step'],
                stage_seconds=dict(profiler.times), episodes=episode_rows,
                rejected_where_episodes=rejected, loss_total=loss)


def worker(request_path):
    request = json.loads(Path(request_path).read_text(encoding='utf-8'))
    variant, config = request['variant'], request['config']
    directory = Path(request['variant_dir'])
    result_path = directory / 'result.json'
    result = empty_result(variant) | {'started_at': timestamp(), 'status': 'running',
                                    'resolved_config': config, 'failure_phase': 'preflight'}
    torch = profiler = device = None
    measured_started = False
    save_json(result_path, result)
    try:
        import torch
        from semgaze.model.build import build_flat_model_bundle
        from semgaze.training.flat_step import make_optimizer, make_scheduler
        from semgaze.training.profiling import runtime_metadata
        validate_real_config(config)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: this benchmark has no CPU or tiny-model fallback')
        device = torch.device(config['runtime']['device'])
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('selected CUDA device does not support bf16')
        props = torch.cuda.get_device_properties(device)
        result['machine'] = machine_metadata() | dict(
            gpu_name=props.name, gpu_total_memory_bytes=props.total_memory,
            gpu_uuid=str(getattr(props, 'uuid', 'unknown')), gpu_compute_capability=list(torch.cuda.get_device_capability(device)),
            target_is_a100_40gb='A100' in props.name and 35 * 2**30 <= props.total_memory <= 45 * 2**30,
            torch_cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
            torch_threads=torch.get_num_threads(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
            nvidia_smi=command_output(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total', '--format=csv']))
        result['failure_phase'] = 'dataset_loading'
        sampler, result['dataset'] = load_real_sampler(config)
        result['failure_phase'] = 'model_loading'
        bundle = build_flat_model_bundle(config=config, output_dir=directory / 'model')
        bundle.optimizer = make_optimizer(bundle)
        result['scheduler_horizon_steps'] = scheduler_horizon(config, request['warmup_steps'] + request['steps'])
        bundle.scheduler = make_scheduler(bundle, result['scheduler_horizon_steps'])
        result['model_metadata'] = dict(bundle.diagnostics) | runtime_metadata(bundle)
        result['memory']['after_load'] = memory_snapshot(torch, device)
        profiler = BoundaryProfiler(torch, device)
        torch.cuda.synchronize(device)  # warmup profiling boundary
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(request['warmup_steps'] + request['steps']):
            measured = step >= request['warmup_steps']
            result['failure_phase'] = 'measurement' if measured else 'warmup'
            result['attempted_optimizer_step'] = step + 1
            if step == request['warmup_steps']:
                result['memory']['warmup'] = memory_snapshot(torch, device)
                torch.cuda.synchronize(device)  # measured-window boundary
                torch.cuda.reset_peak_memory_stats(device)
                measured_started = True
            row = run_measured_step(bundle, sampler, step, profiler, result)
            row['phase'] = result['failure_phase']
            row['phase_memory'] = memory_snapshot(torch, device)
            result['steps' if measured else 'warmup_steps'].append(row)
            result['memory']['measured' if measured else 'warmup'] = row['phase_memory']
            summarize(result)
            save_json(result_path, result)
            print(f"[{variant['id']}] {row['phase']} step {step+1}: "
                  f"{row['step_time_sec']:.3f}s, {len(row['episodes'])/row['step_time_sec']:.3f} episodes/s", flush=True)
            print('  ' + json.dumps({key: [e[key] for e in row['episodes']] for key in
                                    ('K', 'where_length', 'semantic_length', 'image_count', 'fixation_count')}), flush=True)
        result.update(status='ok', failure_phase=None)
        result['model_metadata'].update(runtime_metadata(bundle))
        observed = set(result['measured_K_histogram'])
        result['missing_measured_K_values'] = [k for k, p in zip(result['dataset']['k_values'], result['dataset']['k_probabilities'])
                                               if p > 0 and str(k) not in observed]
    except BaseException as exc:
        oom_type = getattr(getattr(torch, 'cuda', None), 'OutOfMemoryError', ())
        result.update(status='oom' if isinstance(exc, oom_type) else 'failed',
                      error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(),
                      failure_stage=profiler.failed_stage if profiler else None)
        if torch is not None and device is not None:
            try:
                result['memory']['measured' if measured_started else 'failure_before_measurement'] = memory_snapshot(torch, device)
            except Exception:
                pass
        result['measured_peak_includes_failed_step'] = measured_started
        print(f"[{variant['id']}] {result['status']}: {result['error']}", flush=True)
    finally:
        result['finished_at'] = timestamp()
        summarize(result)
        save_json(result_path, result)
    return 0 if result['status'] == 'ok' else 1


def check_comparability(results):
    """Compare accepted episodes/order at common optimizer steps, including partial runs."""
    populated = [r for r in results if r['workload_groups']]
    if len(populated) < 2:
        return {'status': 'not_checked', 'reason': 'fewer than two variants reached real episode collation',
                'variants_checked': [r['id'] for r in populated]}
    reference = max(populated, key=lambda r: len(r['workload_groups']))
    differences = []
    for row in populated:
        n = min(len(row['workload_groups']), len(reference['workload_groups']))
        if row['workload_groups'][:n] != reference['workload_groups'][:n]:
            differences.append(row['id'])
        for key in ('base_commit', 'adapter_file_sha256'):
            if row['model_metadata'].get(key) != reference['model_metadata'].get(key):
                differences.append(row['id'] + ':' + key)
        if row.get('dataset') != reference.get('dataset'):
            differences.append(row['id'] + ':dataset')
    return {'status': 'mismatch' if differences else 'matched_common_prefix',
            'reference_variant': reference['id'], 'differences': differences,
            'variants_checked': [r['id'] for r in populated]}


def write_outputs(directory, report):
    report['comparison'] = check_comparability(report['variants'])
    save_json(directory / 'results.json', report)
    columns = ['id', 'status', 'physical_batch_size', 'gradient_accumulation_steps',
               'gradient_checkpointing', 'effective_optimizer_batch', 'measured_steps_completed',
               'measured_episodes', *METRICS, 'measured_K_histogram', 'where_lengths', 'semantic_lengths',
               'image_counts', 'fixation_counts', 'failure_phase', 'failure_stage', 'error',
               'workload_sha256', 'rates_are_partial', 'inflight_episodes']
    with (directory / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in report['variants']:
            row = {key: result.get(key) for key in columns}
            row.update(result['metrics'])
            row['measured_K_histogram'] = json.dumps(result['measured_K_histogram'], sort_keys=True)
            row['inflight_episodes'] = json.dumps(result.get('inflight_episodes', []))
            episodes = [e for step in result['steps'] for e in step['episodes']]
            for column, key in (('where_lengths', 'where_length'), ('semantic_lengths', 'semantic_length'),
                                ('image_counts', 'image_count'), ('fixation_counts', 'fixation_count')):
                row[column] = json.dumps([e[key] for e in episodes])
            writer.writerow(row)


def launch_worker(request_path, directory):
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(request_path)]
    with (directory / 'console.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            process.stdout.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/flat_throughput.yaml')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs' / ('a100-benchmark-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    parser.add_argument('--warmup-steps', type=int, default=3, help='optimizer updates excluded from summary')
    parser.add_argument('--steps', type=int, default=30, help='measured optimizer updates per variant')
    parser.add_argument('--effective-batch-size', type=int, help='defaults to configured B*accumulation; must be divisible by 2')
    parser.add_argument('--include-checkpointing-off', action='store_true', help='also attempt B=1/B=2 without checkpointing; record OOM and continue')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--images-root', type=Path, help='explicit server path override')
    parser.add_argument('--split-root', type=Path, help='explicit server path override')
    parser.add_argument('--plan-only', action='store_true', help='write not_run templates; no model/data/CUDA execution')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    if args.steps < 1 or args.warmup_steps < 1:
        parser.error('steps and warmup-steps must be positive')
    if not (args.device == 'cuda' or (args.device.startswith('cuda:') and args.device[5:].isdigit())):
        parser.error('device must be cuda or cuda:N; CPU execution is not supported')
    from semgaze.model.config import load_config, resolve_config
    try:
        config = load_config(args.config)
        validate_real_config(config)
        scheduler_horizon(config, args.warmup_steps + args.steps)
        effective = args.effective_batch_size if args.effective_batch_size is not None else (
            config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps'])
        matrix = variant_matrix(effective, args.include_checkpointing_off)
    except ValueError as exc:
        parser.error(str(exc))
    directory = args.output_dir.resolve()
    if directory.exists() and any(directory.iterdir()):
        parser.error('output-dir must be new or empty; existing results will not be overwritten')
    directory.mkdir(parents=True, exist_ok=True)
    for field in ('images_root', 'split_root'):
        value = getattr(args, field)
        if value is not None:
            config['data'][field] = str(value.resolve())
    config['runtime']['device'] = args.device
    report = dict(schema_version=SCHEMA_VERSION, created_at=timestamp(), plan_only=args.plan_only,
        machine=machine_metadata(), config_source=str(args.config.resolve()), source_config=config,
        benchmark=dict(warmup_steps=args.warmup_steps, measured_steps=args.steps,
                       effective_batch_size=effective, seed=config['experiment']['seed'], world_size=1,
                       timing='synchronized wall-clock; includes sampling/collation, excludes report I/O and model/data loading',
                       stages_are_inclusive=True, fresh_process_per_variant=True),
        variants=[empty_result(v) for v in matrix])
    write_outputs(directory, report)
    failed = False
    for index, variant in enumerate(matrix):
        child_dir = directory / variant['id']
        child_config = resolve_config(config, {'training': {
            'per_device_train_batch_size': variant['physical_batch_size'],
            'gradient_accumulation_steps': variant['gradient_accumulation_steps'],
            'gradient_checkpointing': variant['gradient_checkpointing']},
            'runtime': {'output_dir': str(child_dir / 'model')}})
        request_path = child_dir / 'request.json'
        save_json(request_path, dict(variant=variant, config=child_config, variant_dir=str(child_dir),
                                    warmup_steps=args.warmup_steps, steps=args.steps))
        if args.plan_only:
            continue
        print(f"[RUN] {variant['id']}: physical B={variant['physical_batch_size']}, "
              f"accumulation={variant['gradient_accumulation_steps']}", flush=True)
        try:
            code = launch_worker(request_path, child_dir)
        except KeyboardInterrupt:
            code = 130
        result_path = child_dir / 'result.json'
        row = json.loads(result_path.read_text(encoding='utf-8')) if result_path.exists() else empty_result(variant)
        row['worker_exit_code'] = code
        if row['status'] in ('not_run', 'running') or (code != 0 and row['status'] == 'ok'):
            row.update(status='failed', error=f'worker exited with code {code}; inspect {child_dir / "console.log"}')
        summarize(row)
        report['variants'][index] = row
        failed |= row['status'] != 'ok'
        write_outputs(directory, report)
        print(f"[RESULT] {row['id']}: {row['status']} {json.dumps(row['metrics'])}", flush=True)
        if code == 130:
            break
    report['finished_at'] = timestamp()
    write_outputs(directory, report)
    print(f"{'Unmeasured plan' if args.plan_only else 'Results'}: {directory / 'results.json'}\nCSV: {directory / 'results.csv'}", flush=True)
    return 1 if failed or report['comparison']['status'] == 'mismatch' else 0


if __name__ == '__main__':
    raise SystemExit(main())
