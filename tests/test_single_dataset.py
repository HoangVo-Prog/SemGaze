"""VQA-only selector regression tests (no GPU required)."""
from types import SimpleNamespace

from semgaze.data.cocosearch18 import read_persisted_splits
from semgaze.data.fewshot import TrainingEpisodeSampler, frozen_episode
from semgaze.data.joint import unseen_subject_ids
from semgaze.model.config import ROOT, load_config, resolve_config


def test_vqa_config_and_subjects():
    config = load_config(ROOT / 'configs/single_dataset/main_vqa_no_end_fix.yaml')
    assert config['data']['dataset'] == 'AiR'
    assert config['data']['split_root'] == 'data/split/AiR/split/all'
    assert 'mtl_lora' not in config['model']
    assert config['training']['backward']['one_joint_backward'] is True
    assert unseen_subject_ids(config['data']) == frozenset(
        {'AiR::JY', 'AiR::SC', 'AiR::YN'})


def test_coco_variant_is_not_blocked_by_config():
    config = resolve_config({'data': {'dataset': 'COCO-Search18', 'variant': 'ta',
        'split_root': 'data/split/COCO_Search18/split/ta'}})
    assert config['data']['variant'] == 'ta'


def test_air_sampler_excludes_unseen_subjects():
    config = load_config(ROOT / 'configs/single_dataset/main_vqa_no_end_fix.yaml')
    def rec(subject, name):
        return SimpleNamespace(dataset='AiR', subject=subject,
            stimulus_id='AiR::' + name, record_id=subject + '::' + name)
    records = [rec('AiR::AA', 'a.jpg'), rec('AiR::AA', 'b.jpg'),
               rec('AiR::JY', 'c.jpg')]
    sampler = TrainingEpisodeSampler(records, 42, data_config=config['data'])
    assert sampler.query_count == 2
    sampler.start_epoch()
    ep = sampler.sample()
    assert ep.query.subject == 'AiR::AA'
    assert all(s.subject == 'AiR::AA' for s in ep.supports)


def test_standalone_air_split_and_frozen_supports(tmp_path, monkeypatch):
    import semgaze.data.joint as joint
    root = tmp_path / 'air'
    root.mkdir()
    (root / 'split_manifest.json').write_text(
        '{"dataset":"AiR","source_file":"unused"}')
    for split in ('train', 'test', 'test_seen'):
        (root / (split + '.json')).write_text('[]')
    local = {'dataset': 'AiR', 'variant': 'all',
        'unseen_subject_ids': ['JY', 'SC', 'YN'], 'seen_subject_ids': ['AA'],
        'train_stimulus_ids': ['train.jpg'], 'test_stimulus_ids': ['test.jpg'],
        'support_draws': {'1': [[{
            'dataset': 'AiR', 'image_name': 'train.jpg',
            'resolved_record_id_by_subject': {'JY': 'support'}}]]}}
    def verify(source_root, manifest, cfg):
        assert source_root == root.resolve()
        assert manifest['dataset'] == cfg['dataset'] == 'AiR'
        return {'train': [], 'test': [], 'test_seen': []}, local, root / 'split_manifest.json'
    monkeypatch.setattr(joint, '_verify_air', verify)
    rows, manifest, identity = read_persisted_splits(
        root, data_config={'dataset': 'AiR', 'variant': 'all'})
    assert len(identity) == 64
    assert set(rows) == {'train', 'test', 'test_seen'}
    assert manifest['unseen_subject_ids'] == ['AiR::JY', 'AiR::SC', 'AiR::YN']
    assert manifest['seen_subject_ids'] == ['AiR::AA']
    assert manifest['train_stimulus_ids'] == ['AiR::train.jpg']
    assert manifest['datasets']['AiR']['test_stimulus_ids'] == ['test.jpg']
    query = SimpleNamespace(dataset='AiR', subject='AiR::JY',
                            stimulus_id='AiR::test.jpg', record_id='q')
    support = SimpleNamespace(dataset='AiR', subject='AiR::JY',
                              stimulus_id='AiR::train.jpg', record_id='support')
    ep = frozen_episode(query, {'support': support}, manifest, 1, draw_id=0)
    assert ep.supports == (support,)
