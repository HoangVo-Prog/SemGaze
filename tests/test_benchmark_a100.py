"""Offline tests of server instrumentation; no CUDA performance validation."""
import csv
import json
from types import SimpleNamespace

import pytest

from scripts import benchmark_a100 as bench
from semgaze.model.config import ROOT, load_config


def result_row(variant=None):
    result = bench.empty_result(variant or bench.variant_matrix(2)[0])
    result['status'] = 'ok'
    result['workload_groups'] = [[{'query_id': 'a', 'support_ids': ['b'], 'subject': 1}]]
    result['steps'] = [dict(step_time_sec=2.0, stage_seconds={'backward': 1.0}, episodes=[
        dict(K=1, where_length=100, semantic_length=50, image_count=2, fixation_count=3, physical_batch_index=0),
        dict(K=1, where_length=600, semantic_length=90, image_count=2, fixation_count=8,
             physical_batch_index=1 if result['physical_batch_size'] == 1 else 0)])]
    result['memory'] = {'measured': {'peak_memory_allocated_bytes': 1000, 'peak_memory_reserved_bytes': 2000}}
    bench.summarize(result)
    return result


def test_matrix_fixes_effective_batch_without_serial_physical_fallback():
    variants = bench.variant_matrix(6, True)
    assert [v['id'] for v in variants] == ['b1_gc_on', 'b2_gc_on', 'b1_gc_off', 'b2_gc_off']
    assert [v['gradient_accumulation_steps'] for v in variants] == [6, 3, 6, 3]
    assert all(v['physical_batch_size'] * v['gradient_accumulation_steps'] == 6 for v in variants)
    for value in (0, 1, 3, True):
        with pytest.raises(ValueError):
            bench.variant_matrix(value)
    variants = bench.variant_matrix(4, include_b4=True)
    assert [v['physical_batch_size'] for v in variants] == [1, 2, 4]
    assert [v['gradient_accumulation_steps'] for v in variants] == [4, 2, 1]
    assert all(v['sampling_group_size'] == 4 for v in variants)
    with pytest.raises(ValueError):
        bench.variant_matrix(6, include_b4=True)


def test_sequence_statistics_use_physical_batches_and_token_weighted_padding():
    single = result_row()['metrics']
    assert single['where_padding_fraction'] == single['semantic_padding_fraction'] == 0
    row = result_row(bench.variant_matrix(2)[1])
    metrics = row['metrics']
    assert metrics['where_tokens'] == 700
    assert metrics['where_padded_token_slots'] == 1200
    assert metrics['where_padding_tokens'] == 500
    assert metrics['where_padding_fraction'] == pytest.approx(500/1200)
    assert metrics['where_length_mean'] == metrics['where_length_median'] == 350
    assert metrics['where_length_p95'] == metrics['where_length_max'] == 600
    assert metrics['semantic_padding_tokens'] == 40
    # Batch index zero in a different optimizer update must form its own group.
    short = {'episodes': [dict(K=5, where_length=10, semantic_length=20, physical_batch_index=0)]}
    combined = bench.sequence_statistics(row['steps'] + [short])
    assert combined['where_padding_fraction'] == pytest.approx(500/1210)
    row['steps'][0]['episodes'][1]['K'] = 5
    with pytest.raises(ValueError, match='mixed-K'):
        bench.sequence_statistics(row['steps'])


def test_summary_counts_episodes_and_preserves_partial_oom_peaks():
    row = result_row()
    assert row['metrics']['episodes_per_second'] == 1.0
    assert row['metrics']['seconds_per_episode'] == 1.0
    assert row['metrics']['step_time_mean_sec'] == 2.0
    assert row['metrics']['backward_mean_sec'] == 1.0
    assert row['measured_K_histogram'] == {'1': 2}
    row['status'] = 'oom'
    row['memory']['measured']['peak_memory_allocated_bytes'] = 3000
    bench.summarize(row)
    assert row['rates_are_partial'] and row['metrics']['peak_memory_allocated_bytes'] == 3000
    row['steps'] = []
    row['memory'] = {'warmup': {'peak_memory_allocated_bytes': 9000}}
    bench.summarize(row)
    assert row['metrics']['episodes_per_second'] is None
    assert row['metrics']['peak_memory_allocated_bytes'] is None


def test_json_csv_and_common_workload_verification(tmp_path):
    first, second = result_row(), result_row(bench.variant_matrix(2)[1])
    assert bench.check_comparability([first])['status'] == 'not_checked'
    report = {'variants': [first, second]}
    bench.write_outputs(tmp_path, report)
    assert json.loads((tmp_path / 'results.json').read_text())['comparison']['status'] == 'matched_common_prefix'
    with (tmp_path / 'results.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert rows[1]['episodes_per_second'] == '1.0'
    assert json.loads(rows[1]['where_lengths']) == [100, 600]
    second['workload_groups'][0][0]['support_ids'] = ['changed']
    assert bench.check_comparability([first, second])['status'] == 'mismatch'


def test_plan_only_has_no_fabricated_measurements(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, 'machine_metadata', lambda: {'test': True})
    monkeypatch.setattr(bench, 'launch_worker', lambda *a: pytest.fail('plan must not launch CUDA workers'))
    directory = tmp_path / 'plan'
    assert bench.main(['--plan-only', '--include-checkpointing-off', '--include-b4',
                       '--effective-batch-size', '4', '--output-dir', str(directory)]) == 0
    report = json.loads((directory / 'results.json').read_text())
    assert len(report['variants']) == 6 and report['plan_only']
    assert all(v['status'] == 'not_run' and all(x is None for x in v['metrics'].values()) for v in report['variants'])
    requests = [json.loads(p.read_text()) for p in directory.glob('*/request.json')]
    source = load_config(ROOT / 'configs/flat_throughput.yaml')
    for request in requests:
        assert request['config']['model'] == source['model']
        assert request['config']['data'] == source['data']
        assert request['config']['semantic'] == source['semantic']
        for key, value in source['training'].items():
            if key not in ('per_device_train_batch_size', 'gradient_accumulation_steps', 'gradient_checkpointing'):
                assert request['config']['training'][key] == value
    with pytest.raises(SystemExit):
        bench.main(['--plan-only', '--output-dir', str(directory)])


def test_parent_continues_after_oom_and_process_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, 'machine_metadata', lambda: {})
    calls = []
    def launch(request_path, directory):
        request = json.loads(request_path.read_text())
        calls.append(request['variant']['id'])
        if request['variant']['physical_batch_size'] in (1, 4):
            row = bench.empty_result(request['variant']) | {'status': 'oom', 'error': 'test OOM'}
            bench.save_json(directory / 'result.json', row)
        # Second worker simulates a process killed before writing its result.
        return 1
    monkeypatch.setattr(bench, 'launch_worker', launch)
    assert bench.main(['--output-dir', str(tmp_path / 'run'), '--include-b4', '--effective-batch-size', '4']) == 1
    rows = json.loads((tmp_path / 'run/results.json').read_text())['variants']
    assert calls == ['b1_gc_on', 'b2_gc_on', 'b4_gc_on']
    assert [r['status'] for r in rows] == ['oom', 'failed', 'oom']


def test_profiler_failure_records_inner_boundary_without_post_failure_sync():
    calls = []
    profiler = bench.BoundaryProfiler(SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda d: calls.append(d))), 'fake')
    with pytest.raises(RuntimeError):
        with profiler.stage('step'):
            with profiler.stage('where_forward'):
                raise RuntimeError('test')
    assert profiler.failed_stage == 'where_forward'
    assert calls == ['fake', 'fake']


@pytest.mark.parametrize('fail_phase', ['warmup', 'measurement'])
def test_worker_reports_oom_and_resets_measurement_peak(tmp_path, monkeypatch, fail_phase):
    torch = pytest.importorskip('torch')
    from semgaze.model import build
    from semgaze.training import flat_step, profiling
    resets = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda d: None)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda d: SimpleNamespace(name='TEST ONLY', total_memory=40*2**30))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda d: (8, 0))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda d: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda d: resets.append(1))
    monkeypatch.setattr(bench, 'machine_metadata', lambda: {})
    monkeypatch.setattr(bench, 'command_output', lambda a: None)
    monkeypatch.setattr(bench, 'load_real_sampler', lambda c: (object(), {'k_values': [1], 'k_probabilities': [1]}))
    monkeypatch.setattr(build, 'build_flat_model_bundle', lambda **k: SimpleNamespace(diagnostics={}))
    monkeypatch.setattr(flat_step, 'make_optimizer', lambda b: object())
    monkeypatch.setattr(flat_step, 'make_scheduler', lambda *a: object())
    monkeypatch.setattr(profiling, 'runtime_metadata', lambda b: {})
    monkeypatch.setattr(bench, 'memory_snapshot', lambda *a: {'peak_memory_allocated_bytes': 123, 'peak_memory_reserved_bytes': 456})
    def step(bundle, sampler, number, profiler, result):
        if number == (0 if fail_phase == 'warmup' else 1):
            profiler.failed_stage = 'backward'
            raise torch.cuda.OutOfMemoryError('synthetic unit-test failure')
        return result_row()['steps'][0]
    monkeypatch.setattr(bench, 'run_measured_step', step)
    config = load_config(ROOT / 'configs/flat_throughput.yaml')
    path = tmp_path / 'request.json'
    bench.save_json(path, dict(variant=bench.variant_matrix(2)[0], config=config,
                              variant_dir=str(tmp_path), warmup_steps=1, steps=1))
    assert bench.worker(path) == 1
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['status'] == 'oom' and result['failure_phase'] == fail_phase
    assert result['failure_stage'] == 'backward'
    assert len(resets) == (1 if fail_phase == 'warmup' else 2)
    assert result['metrics']['episodes_per_second'] is None
    assert result['metrics']['peak_memory_allocated_bytes'] == (None if fail_phase == 'warmup' else 123)


def test_real_config_preserves_current_adapter_mode_and_requires_bf16():
    config = load_config(ROOT / 'configs/flat_throughput.yaml')
    bench.validate_real_config(config)
    config['model']['adapter_load_mode'] = 'fresh'
    bench.validate_real_config(config)
    config['training']['precision'] = 'fp32'
    with pytest.raises(ValueError, match='bf16'):
        bench.validate_real_config(config)


def test_scheduler_horizon_preserves_configured_schedule():
    config = load_config(ROOT / 'configs/flat_throughput.yaml')
    assert bench.scheduler_horizon(config, 33) == 33
    config['training']['max_steps'] = 1000
    assert bench.scheduler_horizon(config, 33) == 1000
    with pytest.raises(ValueError, match='exceeds'):
        bench.scheduler_horizon(config, 1001)
    config['training'].update(max_steps=None, total_optimizer_updates=1000)
    assert bench.scheduler_horizon(config, 33) == 1000


@pytest.mark.parametrize('physical_batch', [1, 2, 4])
def test_offline_step_uses_real_training_batch_and_accumulation(tmp_path, physical_batch):
    """CPU graph integration only; stage synchronization is a test double."""
    pytest.importorskip('peft')
    import torch
    from test_model_path import tiny_bundle
    from semgaze.data.schema import normalized_episode_from_dict
    from semgaze.training.flat_step import make_optimizer, make_scheduler
    episode = normalized_episode_from_dict(json.loads((ROOT / 'tests/fixtures/flat_episode.json').read_text()))
    bundle = tiny_bundle(tmp_path, episode)
    bundle.config['training'].update(per_device_train_batch_size=physical_batch,
                                     gradient_accumulation_steps=4 // physical_batch)
    bundle.optimizer = make_optimizer(bundle)
    bundle.scheduler = make_scheduler(bundle, 2)
    calls = []
    hook = bundle.model.get_base_model().model.language_model.register_forward_pre_hook(
        lambda model, args, kwargs: calls.append(kwargs['inputs_embeds'].shape[0]), with_kwargs=True)
    profiler = bench.BoundaryProfiler(SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda d: None)), 'cpu-test')
    result = bench.empty_result(next(v for v in bench.variant_matrix(4, include_b4=True)
                                   if v['physical_batch_size'] == physical_batch))
    try:
        from test_epoch_validation import FixedSampler
        row = bench.run_measured_step(bundle, FixedSampler(episode, count=4), 0, profiler, result)
    finally:
        hook.remove()
    assert calls == [physical_batch] * (2 * (4 // physical_batch))
    assert len(row['episodes']) == 4 and row['episodes'][0]['K'] == 1
    assert row['episodes'][0]['where_length'] > 0 and row['episodes'][0]['semantic_length'] > 0
    assert row['episodes'][0]['support_ids'] == [episode.supports[0].record_id]
    assert set(bench.STAGES).issubset(row['stage_seconds'])
    assert bundle.scheduler.last_epoch == 1
    assert bundle.input_row.grad.ne(0).any()
