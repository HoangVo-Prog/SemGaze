"""Independent prediction K and pooled-vs-task evaluation regression tests."""
import json
from types import SimpleNamespace

from semgaze.evaluation import predictions
from semgaze.evaluation.records import evaluation_group, resolve_prediction_draw_plan


def _manifest():
    return {'support_draws': {str(k): [[], []] for k in (1, 5, 10)}}


def _config(where_k, semantic_k, *, use_draws=True):
    return {'evaluation': {
        'k_values': [1, 5, 10], 'draw': 2, 'semantic_use_draws': use_draws,
        'predictions': {'where_k_values': where_k, 'semantic_k_values': semantic_k}}}


def test_independent_and_overlapping_prediction_k():
    where, semantic, wd, sd, active = resolve_prediction_draw_plan(
        _config([1, 5], [5, 10], use_draws=False), _manifest(),
        enabled_where=True, enabled_semantic=True)
    assert where == [1, 5] and semantic == [5, 10]
    assert wd == {'1': 2, '5': 2, '10': 0}
    assert sd == {'1': 0, '5': 1, '10': 1}
    assert active == {'1': 2, '5': 2, '10': 1}
    assert sum(active.values()) == 5 # shared K=5 draw=0 runs both branches


def test_disabled_branch_and_legacy_k_fallback():
    where, semantic, wd, sd, active = resolve_prediction_draw_plan(
        _config(None, None), _manifest(),
        enabled_where=False, enabled_semantic=True)
    assert where == [] and semantic == [1, 5, 10]
    assert all(v == 0 for v in wd.values())
    assert sd == active == {'1': 2, '5': 2, '10': 2}


def test_evaluation_group_distinguishes_conditions():
    def query(dataset, condition):
        return SimpleNamespace(dataset=dataset, condition=condition, record_id='example')
    assert evaluation_group(query('COCO-Search18', 'absent')) == 'TA'
    assert evaluation_group(query('COCO-Search18', 'present')) == 'TP'
    assert evaluation_group(query('AiR', 'vqa')) == 'VQA'


def test_joint_grouping_and_pooled_overall_do_not_average_group_scores(tmp_path, monkeypatch):
    queries = [SimpleNamespace(record_id='a', dataset='COCO-Search18', condition='absent'),
               SimpleNamespace(record_id='b', dataset='COCO-Search18', condition='present'),
               SimpleNamespace(record_id='c', dataset='AiR', condition='vqa')]
    artifact_path = tmp_path / 'predictions.jsonl'
    artifact_path.write_text(''.join(json.dumps({
        'split': 'test', 'query_id': q.record_id, 'K': 1, 'draw_id': 0,
        'dataset': q.dataset, 'where_generation': {'text': ''}}) + '\n'
        for q in queries), encoding='utf-8')
    config = {'data': {'dataset': 'all'}, 'evaluation': {
        'predictions': {'where': True, 'semantic': False},
        'metrics': {'enabled': True, 'where': {'sed': True}, 'semantic': {}}}}
    original = predictions.score_prediction_artifact
    calls = []

    def lightweight(*args, **kwargs):
        if kwargs.get('_subset', False):
            selected = tuple(q.record_id for q in kwargs['queries'])
            calls.append(selected)
            # The real scorer, not mean(subset scores), handles the pooled set.
            return {'selected': list(selected), 'n': len(selected)}
        return original(*args, **kwargs)

    monkeypatch.setattr(predictions, 'score_prediction_artifact', lightweight)
    result = original(artifact_path, queries=queries, config=config,
                      output_dir=tmp_path, checkpoint='test')
    assert result['test_queries_by_group'] == {'TA': 1, 'TP': 1, 'VQA': 1}
    assert result['by_group']['TA']['selected'] == ['a']
    assert result['by_group']['TP']['selected'] == ['b']
    assert result['by_group']['VQA']['selected'] == ['c']
    assert result['overall']['selected'] == ['a', 'b', 'c']
    assert result['by_dataset']['COCO-Search18']['selected'] == ['a', 'b']
    assert (tmp_path / 'metrics_by_group.json').is_file()
    assert ('a', 'b', 'c') in calls
