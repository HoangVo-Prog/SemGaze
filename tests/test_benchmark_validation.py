"""Report correctness and failure semantics without production weights/CUDA."""
from copy import deepcopy
import json
import pytest
from scripts import benchmark_test_a100 as benchmark


def report():
    return dict(status='complete', checkpoint_sha256='weights', workload_sha256='episodes',
        split_manifest_identity='split', scientific_config_sha256='config',
        episodes=[dict(episode_id=f'{k}:q{i}', K=k, support_ids=list(range(k)), state_count=i+1,
                       **{m: float(i+1) for m in benchmark.METRICS}) for k in (1,5,10) for i in range(2)])


def test_episode_parity_not_aggregate_only(tmp_path):
    reference, actual = report(), report()
    actual['episodes'].reverse()
    result = benchmark.compare_reports(reference, actual, 1e-5)
    assert result['passed']
    actual['episodes'][0]['test_flat'] += .01
    actual['episodes'][1]['test_flat'] -= .01
    result = benchmark.compare_reports(reference, actual, 1e-5)
    assert not result['passed'] and result['metrics']['test_flat']['max_absolute_difference'] > .009
    benchmark.write_parity(tmp_path, result)
    assert 'FAIL' in (tmp_path/'parity_quick.md').read_text()
    assert not json.loads((tmp_path/'parity_quick.json').read_text())['passed']


@pytest.mark.parametrize('field,value', [('checkpoint_sha256','different'), ('workload_sha256','different'),
                                      ('status','oom'), ('split_manifest_identity','different')])
def test_incomparable_reports_rejected(field, value):
    a,b = report(),report()
    b[field] = value
    with pytest.raises(ValueError):
        benchmark.compare_reports(a,b,.001)


def test_parity_missing_duplicate_nonfinite_and_state_failures():
    for mutation in (lambda r: r['episodes'].pop(),
                     lambda r: r['episodes'].append(r['episodes'][0]),
                     lambda r: r['episodes'][0].update(state_count=100),
                     lambda r: r['episodes'][0].update(test_flat=float('nan'))):
        changed = report()
        mutation(changed)
        with pytest.raises(ValueError):
            benchmark.compare_reports(report(), changed, .001)


def test_padding_uses_real_partial_batch_sizes():
    batches = [[dict(K=1,where_length=n,semantic_length=n+1) for n in (2,4,3,1)],
               [dict(K=1,where_length=8,semantic_length=9)]]
    row = benchmark.summarize_batches(batches, 2, {})
    assert row['episodes'] == 5 and row['mean_actual_batch_size'] == 2.5
    assert row['padding']['where_tokens'] == 18
    assert row['padding']['where_padded_token_slots'] == 24
    assert row['episodes_per_second'] == 2.5


def test_cuda_absence_writes_failure_report(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(benchmark, 'machine_metadata', lambda: {})
    assert benchmark.main(['--checkpoint','absent','--output-dir',str(tmp_path)]) == 1
    result = json.loads((tmp_path/'report.json').read_text())
    assert result['status'] == 'failed' and 'CUDA server required' in result['error']
    assert (tmp_path/'report.md').exists()
    assert 'episodes_per_second' not in result


def test_sweep_records_oom_and_selects_throughput_with_headroom(tmp_path, monkeypatch):
    def worker(command, **kwargs):
        from pathlib import Path
        from types import SimpleNamespace
        size = int(command[command.index('--batch-size')+1])
        directory = Path(command[command.index('--output-dir')+1])
        result = dict(status='oom' if size == 8 else 'complete', batch_size_by_k={'1':size},
                      environment={'gpu_total_memory':100}, peak_memory_reserved_bytes=95 if size == 4 else 70,
                      episodes_per_second={1:1,2:3,4:4,8:0}[size], peak_memory_allocated_bytes=60)
        benchmark.save_json(directory/'report.json',result)
        return SimpleNamespace(returncode=1 if size==8 else 0)
    monkeypatch.setattr(benchmark.subprocess,'run',worker)
    assert benchmark.main(['--checkpoint','fixture','--output-dir',str(tmp_path),'--k','1','--sweep','1,2,4,8']) == 0
    result=json.loads((tmp_path/'sweep.json').read_text())
    assert result['chosen_batch_size'] == 2
    assert len(result['candidates']) == 4 and result['candidates'][-1]['status']=='oom'


def test_report_pipeline_with_cpu_model_and_mock_cuda_telemetry(tmp_path, monkeypatch):
    """Exercise report wiring, NOT a CUDA benchmark or performance measurement."""
    from dataclasses import replace
    from types import SimpleNamespace
    from pathlib import Path
    import torch
    from test_model_path import tiny_bundle
    from test_epoch_validation import evaluation_data
    from semgaze.data.schema import normalized_episode_from_dict
    from semgaze.data import cocosearch18
    from semgaze.model import checkpoint
    episode = normalized_episode_from_dict(json.loads((benchmark.ROOT/'tests/fixtures/flat_episode.json').read_text()))
    bundle = tiny_bundle(tmp_path,episode)
    train,queries,manifest=evaluation_data(episode)
    queries.append(replace(queries[0],record_id='q2',stimulus_id='q2'))
    manifest['test_stimulus_ids'].append('q2')
    raw=dict(train=[dict(record_id=r.record_id,record=r) for r in train.values()],test=queries)
    monkeypatch.setattr(cocosearch18,'read_persisted_splits',lambda *a,**kw:(raw,manifest,'fixture'))
    monkeypatch.setattr(cocosearch18,'CocoSearch18Adapter',lambda *a,**kw:lambda r:r['record'] if isinstance(r,dict) else r)
    monkeypatch.setattr(checkpoint,'load_checkpoint_bundle',lambda *a,**kw:bundle)
    monkeypatch.setattr(benchmark,'machine_metadata',lambda:{'test_only':True})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'set_device',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *a:SimpleNamespace(name='MOCK_CPU_NOT_A_BENCHMARK',total_memory=1000))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda *a:None)
    monkeypatch.setattr(benchmark,'memory_snapshot',lambda *a,**kw:dict(peak_memory_allocated_bytes=0,peak_memory_reserved_bytes=0))
    saved=tmp_path/'checkpoint'
    saved.mkdir()
    (saved/'resolved_config.json').write_text(json.dumps(bundle.config))
    reports=[]
    for mode in ('serial','same-k-batched'):
        output=tmp_path/mode
        reference_args = ['--reference',str(tmp_path/'serial/report.json')] if mode != 'serial' else []
        assert benchmark.main(['--checkpoint',str(saved),'--output-dir',str(output),'--mode',mode,
                              '--batch-k1','2','--batch-k5','2','--batch-k10','2','--bucket',*reference_args])==0
        report=json.loads((output/'report.json').read_text())
        assert report['episode_count_total']==len(report['episodes'])==60
        assert set(report['by_k'])=={'1','5','10'}
        assert all(r['mean_actual_batch_size']==(1 if mode=='serial' else 2) for r in report['by_k'].values())
        if mode != 'serial':
            assert report['parity']['passed'] and report['reference_summary']['mode']=='serial'
            assert '| serial |' in (output/'report.md').read_text()
        reports.append(report)
    assert benchmark.compare_reports(*reports,1e-5)['passed']
