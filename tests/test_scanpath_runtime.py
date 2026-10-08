"""Frozen WHERE metric integration and failure-policy checks."""
import json
from types import SimpleNamespace

import pytest

from semgaze.evaluation.metrics_scanpath import CoordinateProtocolError, score_scanpath_pair
from semgaze.evaluation.predictions import score_prediction_artifact


def _query(record_id, subject, xs=(840.0, 420.0), ys=(525.0, 262.5)):
    return SimpleNamespace(record_id=record_id, subject=subject, x_px=xs, y_px=ys,
                           duration_ms=(100.0,) * len(xs), image_width=1680,
                           image_height=1050, semantic={})


def _config():
    return {'evaluation': {'draw': 1, 'semantic_use_draws': True, 'metrics': {
        'enabled': True, 'where': {'scanmatch': True, 'multimatch': False, 'sed': True},
        'semantic': {}}}, 'where': {'end_fix_token': '<END_FIX>'}}


def _row(q, prediction, *, draw=0, subject=None):
    return {'split': 'test', 'query_id': q.record_id, 'subject': q.subject if subject is None else subject,
            'K': 1, 'draw_id': draw, 'WHERE': {'PRED': prediction},
            'where_generation': {'under_generated': False}}


def test_frozen_where_metrics_recover_malformed_and_apply_empty_policy(tmp_path):
    q = _query('q', 1)
    rows = [_row(q, 'prefix (50,50,100)<END_FIX>, (25,25,100)<END_FIX>'),
            _row(q, 'no fixations', draw=1)]
    path = tmp_path / 'predictions.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    config = _config()
    config['evaluation']['draw'] = 2
    artifact = score_prediction_artifact(path, queries=[q], config=config, output_dir=tmp_path)
    assert artifact['by_k']['1']['draws']['0']['valid'] == 1
    assert artifact['by_k']['1']['draws']['0']['where_format_failure_count'] == 1
    assert artifact['by_k']['1']['draws']['1']['SM'] == 0
    assert artifact['by_k']['1']['draws']['1']['SED'] == 2


def test_frozen_where_pairing_rejects_subject_mismatch(tmp_path):
    q = _query('q', 1)
    path = tmp_path / 'predictions.jsonl'
    path.write_text(json.dumps(_row(q, '(50,50,100)<END_FIX>', subject=2)) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='subject mismatch'):
        score_prediction_artifact(path, queries=[q], config=_config(), output_dir=tmp_path)


def test_invalid_gt_is_dataset_integrity_error():
    with pytest.raises(CoordinateProtocolError):
        score_scanpath_pair([(840.0, 525.0, 100.0), (420.0, 262.5, float('nan'))],
                            [(50, 50, 100)], require_multimatch=False)


def test_multimatch_keeps_all_five_components_and_duration_seconds():
    class Reference:
        def docomparison(self, gt, pred, screensize):
            assert list(gt['duration']) == [0.1, 0.001, 0.001]
            assert list(pred['duration']) == [0.1, 0.001, 0.001]
            assert screensize == [512, 320]
            return [1, 2, 3, 4, 5]

    score = score_scanpath_pair([(840.0, 525.0, 100.0)], [(50, 50, 100)],
                                multimatch_module=Reference())
    assert score['mm_components'] == [1, 2, 3, 4, 5]
    assert score['mm'] == 3
