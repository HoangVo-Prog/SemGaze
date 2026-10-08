"""Dependency-free guard against the duplicate-subject-anchor regression."""
from pathlib import Path
import importlib.util

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('joint_patcher', root / 'apply_joint_patch.py')
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)

source = '''def _prediction_record(query):
    record = {
            'query_id': query.record_id, 'subject': query.subject,
    }
    return record

def predict_epoch(bundle, train_batch, train_by_id, test_records, manifest, settings, queries, k, draw, counts):
    if any(ep.query.subject in bundle.config['data']['unseen_subjects'] for ep in train_batch):
        raise ValueError('unseen')
    queries = test_queries(train_by_id, test_records, manifest, bundle.config['data']['unseen_subjects']) if settings['test_scope'] == 'all_unseen' else ()
    episodes = [frozen_episode(query, train_by_id, manifest, k, draw_id=draw,
                                    unseen_subjects=bundle.config['data']['unseen_subjects']) for query in queries]
    result = {
            'test_prediction_episodes': counts['test'], 'k_values': list(k_values),
    }
    return result

def score_prediction_artifact(path, queries, config, output_dir, checkpoint='integrated', split_manifest_identity=''):
    metrics_config = config.get('evaluation', {}).get('metrics', {})
    for row in ():
        query = queries[0]
        if int(row.get('subject', query.subject)) != int(query.subject):
            raise ValueError('wrong subject')
    for row in ():
        query = queries[0]
        if int(row.get('subject', query.subject)) != int(query.subject):
            raise ValueError('wrong subject')
    gt_x, gt_y, gt_d, gt_dims = (), (), (), ()
    invalid_gt = (not gt_x or len(gt_x) != len(gt_y) or
                                  len(gt_x) != len(gt_d) or gt_dims != (1680, 1050))
    return invalid_gt
'''
fixed = patcher.patch_predictions(source)
compile(fixed, '<patched prediction fixture>', 'exec')
assert fixed.count("if str(row.get('subject', query.subject)) != str(query.subject):") == 2
assert "if int(row.get('subject', query.subject)) != int(query.subject):" not in fixed
assert "'dataset': getattr(query, 'dataset', 'COCO-Search18')" in fixed
assert 'if config.get(\'data\', {}).get(\'dataset\') == \'all\':' in fixed
assert 'metrics_by_dataset.json' in fixed
try:
    patcher.patch_predictions(source.replace("            raise ValueError('wrong subject')", "            raise ValueError('wrong subject')\n        if int(row.get('subject', query.subject)) != int(query.subject):\n            raise ValueError('extra wrong subject')", 1))
except ValueError as exc:
    assert 'expected exactly 2 anchors, found 3' in str(exc)
else:
    raise AssertionError('duplicate anchor count check must fail closed')
print('PASS: 2 WHERE/semantic subject anchors replaced, 3 anchors rejected, Python compiled')
