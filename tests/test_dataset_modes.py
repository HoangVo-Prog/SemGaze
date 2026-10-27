"""Single-dataset selector regressions. No VLM/GPU required."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from semgaze.model.config import ROOT, load_config, resolve_config
from semgaze.data.fewshot import TrainingEpisodeSampler
from semgaze.data.joint import unseen_subject_ids

MODES = {
    'vqa': ('AiR', 'all', 'data/split/AiR/split/all'),
    'coco': ('COCO-Search18', 'all', 'data/split/COCO_Search18/split/all'),
    'ta': ('COCO-Search18', 'ta', 'data/split/COCO_Search18/split/ta'),
    'tp': ('COCO-Search18', 'tp', 'data/split/COCO_Search18/split/tp'),
}

@pytest.mark.parametrize('name', MODES)
@pytest.mark.parametrize('end_fix', [True, False])
def test_main_config_matrix(name, end_fix):
    dataset, variant, split = MODES[name]
    path = ROOT / 'configs/single_dataset' / (
        f'main_{name}_{"with_end_fix" if end_fix else "no_end_fix"}.yaml')
    cfg = load_config(path)
    assert cfg['data']['dataset'] == dataset
    assert cfg['data']['variant'] == variant
    assert cfg['data']['split_root'] == split
    assert cfg['model']['mtl_lora']['enabled'] is False
    assert cfg['training']['backward']['one_joint_backward'] is True
    assert cfg['semantic']['conditioning'] == 'explicit_scanpath_text'
    assert cfg['where']['use_end_fix_token'] is end_fix
    assert cfg['where']['supervision']['supervise_query_end_fix'] is end_fix
    assert cfg['state']['readout_position'] == (
        'query_end_fix_token_position' if end_fix else 'last_duration_numeric_token')

@pytest.mark.parametrize('dataset,variant', [('AiR','tp'),('AiR','ta'),
    ('all','tp'),('all','ta'),('COCO-Search18','vqa')])
def test_reject_unsupported_variant(dataset, variant):
    with pytest.raises(ValueError, match='variant'):
        resolve_config({'data': {'dataset': dataset, 'variant': variant}})

@pytest.mark.parametrize('name', MODES)
def test_sampler_keeps_only_selected_dataset_and_seen_subjects(name):
    dataset, variant, _ = MODES[name]
    cfg = load_config(ROOT / 'configs/single_dataset' /
        f'main_{name}_no_end_fix.yaml')['data']
    def record(subject, image, *, ds=dataset, condition=None):
        if condition is None:
            condition = ('vqa' if ds == 'AiR' else
                         'present' if variant == 'tp' else 'absent')
        return SimpleNamespace(dataset=ds, subject=subject, condition=condition,
            stimulus_id=image, record_id=f'{subject}:{image}')
    seen = 'AiR::AA' if dataset == 'AiR' else 1
    unseen = 'AiR::JY' if dataset == 'AiR' else 7
    rows = [record(seen, 'one'), record(seen, 'two'), record(unseen, 'three')]
    sampler = TrainingEpisodeSampler(rows, 42, data_config=cfg)
    assert sampler.query_count == 2
    sampler.start_epoch()
    e = sampler.sample()
    assert e.query.subject == seen
    assert all(s.subject == seen for s in e.supports)
    wrong_ds = 'COCO-Search18' if dataset == 'AiR' else 'AiR'
    wrong_subject = 1 if wrong_ds == 'COCO-Search18' else 'AiR::AA'
    with pytest.raises(ValueError, match='wrong dataset'):
        TrainingEpisodeSampler(rows + [record(wrong_subject,'wrong',ds=wrong_ds)],
                               42, data_config=cfg)

@pytest.mark.parametrize('variant,incorrect', [('tp','absent'),('ta','present')])
def test_reject_wrong_condition_in_sampler(variant, incorrect):
    cfg = load_config(ROOT / 'configs/single_dataset' /
        f'main_{variant}_no_end_fix.yaml')['data']
    rows = [SimpleNamespace(dataset='COCO-Search18',subject=1,
             condition=incorrect,stimulus_id='a',record_id='a')]
    with pytest.raises(ValueError, match='variant condition'):
        TrainingEpisodeSampler(rows, 42, data_config=cfg)


def test_air_manifest_without_record_counts_is_valid_schema(tmp_path, monkeypatch):
    """Exercise real _verify_air (incl. checksums/holds/draws), not a stub."""
    import hashlib
    import json
    import semgaze.data.joint as joint
    root = tmp_path / 'air'
    root.mkdir()
    image_names = [f'im-{i}.jpg' for i in range(100)]
    train_rows = []
    seen = ('AA',)
    unseen = ('JY', 'SC', 'YN')
    def make(subject, image, split, qid='42'):
        rid = f'air_semgaze::{subject}::{qid}::{image}'
        return dict(dataset='AiR', variant='all', split=split,
          record_id=rid, name=image, subject=subject, qid=qid,
          stimulus_id=image, global_stimulus_id=f'AiR::{image}',
          global_subject_id=f'AiR::{subject}', trial_key=f'{qid}::{image}',
          condition='vqa', subject_role=('unseen' if subject in unseen else 'seen'),
          train_query_eligible=(split=='train' and subject not in unseen),
          X=[1], Y=[2], T=[100], prediction={})
    for im in image_names:
        for sub in (*unseen, *seen):
            train_rows.append(make(sub, im, 'train'))
    test_rows = [make(sub, 'heldout.jpg','test') for sub in unseen]
    test_seen = [make('AA','heldout.jpg','test_seen')]
    draws = {}
    for k in (1,5,10):
        draws[str(k)] = [[dict(dataset='AiR', image_name=im,
             trial_key=f'42::{im}',unit_id='42',
             resolved_record_id_by_subject={sub:f'air_semgaze::{sub}::42::{im}'
                                            for sub in unseen})
             for im in image_names[d*k:(d+1)*k]] for d in range(10)]
    file_hashes = {}
    for split, rows in [('train',train_rows),('test',test_rows),('test_seen',test_seen)]:
        p = root / f'{split}.json'
        p.write_text(json.dumps(rows), encoding='utf-8')
        file_hashes[split] = hashlib.sha256(p.read_bytes()).hexdigest()
    source = root / 'source.json'
    source.write_text('[]', encoding='utf-8')
    manifest = dict(dataset='AiR',variant='all', source_file=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        train_stimulus_ids=image_names,test_stimulus_ids=['heldout.jpg'],
        unseen_subject_ids=list(unseen),seen_subject_ids=list(seen),
        all_source_records_materialized=True,discarded_records=0,
        support_draws=draws,split_file_sha256=file_hashes)
    (root/'split_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(joint, 'semantic_from_prediction', lambda *args: None)
    cfg = load_config(ROOT / 'configs/single_dataset/main_vqa_no_end_fix.yaml')['data']
    verified, source_manifest, path = joint._verify_air(root, manifest, cfg)
    assert len(verified['train']) == 400
    assert len(verified['test']) == 3
    assert len(verified['test_seen']) == 1
    assert path == root/'split_manifest.json'
    from semgaze.data.cocosearch18 import read_persisted_splits
    raw, runtime, identity = read_persisted_splits(root, data_config=cfg)
    assert len(raw['train']) == 400
    assert runtime['dataset'] == 'AiR'
    assert runtime['unseen_subject_ids'] == ['AiR::JY','AiR::SC','AiR::YN']
    assert len(identity) == 64
    # Combined AiR + COCO has explicit record_counts, unlike standalone AiR.
    # Preserve those count guards when selecting only AiR out of the union.
    combined_root = tmp_path / 'combined'
    combined_root.mkdir()
    combined_counts = {}
    combined_hashes = {}
    for split, rows in [('train', train_rows), ('test', test_rows), ('test_seen', test_seen)]:
        merged = [*rows, {'dataset': 'COCO-Search18', 'record_id': 'placeholder'}]
        p = combined_root / f'{split}.json'
        p.write_text(json.dumps(merged), encoding='utf-8')
        combined_counts[split] = len(merged)
        combined_hashes[split] = hashlib.sha256(p.read_bytes()).hexdigest()
    combined = dict(dataset='all', source_manifests={'AiR': str(root/'split_manifest.json')},
        record_counts=combined_counts,
        per_dataset_record_counts={'AiR': {s: len(rows) for s, rows in
            [('train',train_rows), ('test',test_rows), ('test_seen',test_seen)]}},
        split_file_sha256=combined_hashes)
    union_rows, _, _ = joint._verify_air(combined_root, combined, cfg)
    assert len(union_rows['train']) == 400
    combined['record_counts']['train'] += 1
    with pytest.raises(ValueError, match='combined train row count mismatch'):
        joint._verify_air(combined_root, combined, cfg)
    # Cross-dataset row injection, even with an updated matching checksum,
    # must fail rather than silently enter the AiR-only training universe.
    altered = [dict(train_rows[0], dataset='COCO-Search18'), *train_rows[1:]]
    (root/'train.json').write_text(json.dumps(altered), encoding='utf-8')
    manifest['split_file_sha256']['train'] = hashlib.sha256((root/'train.json').read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='invalid AiR persisted record'):
        joint._verify_air(root, manifest, cfg)
