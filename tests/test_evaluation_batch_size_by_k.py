"""Per-K physical evaluation batching: config, loss, prediction, standalone override."""
import json
from types import SimpleNamespace

import pytest

from semgaze.evaluation import batching, predictions
from semgaze.model.config import resolve_config


def _episode(k, identity):
    return SimpleNamespace(
        supports=(None,) * k,
        query=SimpleNamespace(record_id=identity, x_px=()),
        draw_id=None,
    )


def _sizes(execution='same_k_batched'):
    return {
        'execution': execution,
        'batch_size_by_k': {'1': 1, '5': 1, '10': 1},
        'bucket_by_length': False,
        'bucket_window': 64,
    }


def test_precedence_and_legacy_compatibility():
    settings = _sizes()
    settings['batch_size_by_k']['5'] = 6
    overrides = {'1': 3, '10': 2}
    assert batching.physical_size(settings, 1, batch_size=4, batch_size_by_k=overrides) == 3
    assert batching.physical_size(settings, 5, batch_size=4, batch_size_by_k=overrides) == 4
    assert batching.physical_size(settings, 10, batch_size=4, batch_size_by_k=overrides) == 2
    assert batching.physical_size(settings, 5, batch_size_by_k=overrides) == 6
    assert batching.physical_size(settings, 5) == 6
    assert batching.physical_size(settings, 1, batch_size=4) == 4
    settings['execution'] = 'serial'
    assert batching.physical_size(settings, 1, batch_size=4, batch_size_by_k=overrides) == 1


def test_config_normalizes_keys_and_merges_partial_overrides():
    base = resolve_config({'evaluation': {
        'batch_size': 4,
        'batch_size_by_k': {1: 3, 10: 1},
    }})
    assert base['evaluation']['batch_size_by_k'] == {'1': 3, '10': 1}
    assert resolve_config({})['evaluation']['batch_size_by_k'] == {}
    updated = resolve_config(base, {'evaluation': {'batch_size_by_k': {'5': 2}}})
    assert updated['evaluation']['batch_size'] == 4
    assert updated['evaluation']['batch_size_by_k'] == {'1': 3, '5': 2, '10': 1}


@pytest.mark.parametrize('mapping', [None, 3, [], '1: 4', {'0': 1}, {'11': 1},
                                     {'abc': 1}, {'01': 1}, {'1': 0},
                                     {'1': -1}, {'1': True}, {'1': 1.5}, {'1': '2'}])
def test_invalid_per_k_config_fails_fast(mapping):
    with pytest.raises(ValueError, match='evaluation.batch_size_by_k'):
        resolve_config({'evaluation': {'batch_size_by_k': mapping}})


def test_per_k_physical_loss_scheduler_has_no_drops(monkeypatch):
    episodes = ([_episode(1, key) for key in 'abcde']
                + [_episode(5, key) for key in 'fgh']
                + [_episode(10, key) for key in 'ij'])
    monkeypatch.setattr(batching, 'collate_where',
                        lambda *a, **kw: SimpleNamespace(inputs={'input_ids': SimpleNamespace(shape=(1, 5))}))
    monkeypatch.setattr(batching, 'pack_where_batch',
                        lambda processor, group, samples, cache: SimpleNamespace(episodes=group))
    bundle = SimpleNamespace(processor=None, end_fix_id=1, context_limit=8192, config={})
    batches = list(batching.schedule_batches(bundle, episodes, _sizes(), cache={},
                   batch_size=4, batch_size_by_k={'1': 3, '5': 2, '10': 1}))
    assert [(len(b.episodes[0].supports), len(b.episodes)) for b in batches] == [
        (1, 3), (1, 2), (5, 2), (5, 1), (10, 1), (10, 1)]
    assert [ep.query.record_id for b in batches for ep in b.episodes] == list('abcdefghij')


def test_per_k_where_and_semantic_prediction_batches(monkeypatch):
    episodes = ([_episode(1, key) for key in 'abcde']
                + [_episode(5, key) for key in 'fgh']
                + [_episode(10, key) for key in 'ij'])
    config = {
        'evaluation': {'batch_size': 4, 'batch_size_by_k': {'1': 3, '5': 2, '10': 1}},
        'test': {'prediction': {**_sizes(), 'bucket_window': 64}},
    }
    bundle = SimpleNamespace(config=config)
    where_calls = []
    semantic_calls = []

    def generate_where(_bundle, group, *, cache, **kwargs):
        where_calls.append((len(group[0].supports), len(group)))
        return [{'text': 'WHERE'} for _ in group]

    def generate_semantics(_bundle, group, *, generation_budget, cache, projected_r_cache):
        assert generation_budget == 32
        semantic_calls.append((len(group[0].supports), len(group)))
        return [{'text': 'SEMANTIC'} for _ in group]

    monkeypatch.setattr(predictions, 'generate_where_batch', generate_where)
    monkeypatch.setattr(predictions, 'evaluate_flat_batch', generate_semantics)
    rows = list(predictions.prediction_batches(bundle, episodes, budget=32, cache={}, path='both'))
    assert [ep.query.record_id for ep, _ in rows] == list('abcdefghij')
    assert where_calls == [(1, 3), (1, 2), (5, 2), (5, 1), (10, 1), (10, 1)]
    assert semantic_calls == where_calls
    assert all(pred[0]['text'] == 'WHERE' and pred[1]['text'] == 'SEMANTIC' for _, pred in rows)


def test_prediction_window_respects_large_per_k_override(monkeypatch):
    episodes = [_episode(1, key) for key in 'abcd']
    bundle = SimpleNamespace(config={
        'evaluation': {'batch_size': None, 'batch_size_by_k': {'1': 4}},
        'test': {'prediction': {**_sizes(), 'bucket_window': 2}},
    })
    calls = []

    def generate_where(_bundle, group, *, cache, **kwargs):
        calls.append(len(group))
        return [{'text': 'WHERE'} for _ in group]

    monkeypatch.setattr(predictions, 'generate_where_batch', generate_where)
    rows = list(predictions.prediction_batches(bundle, episodes, budget=8, cache={}, path='where'))
    assert calls == [4]
    assert len(rows) == 4


def test_standalone_override_per_k_fallback_to_checkpoint_scalar(tmp_path):
    from evaluate_flat import _load_evaluation_config

    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'resolved_config.json').write_text(
        json.dumps(resolve_config({'evaluation': {'batch_size': 4}})), encoding='utf-8')
    override = tmp_path / 'override.yaml'
    override.write_text(
        'evaluation:\n  batch_size_by_k:\n    1: 3\n    5: 2\n    10: 1\n',
        encoding='utf-8',
    )
    actual = _load_evaluation_config(checkpoint, override)
    assert actual['evaluation']['batch_size'] == 4
    assert actual['evaluation']['batch_size_by_k'] == {'1': 3, '5': 2, '10': 1}
    for k, expected in ((1, 3), (5, 2), (10, 1)):
        assert batching.physical_size(actual['test']['loss'], k, batch_size=4,
               batch_size_by_k=actual['evaluation']['batch_size_by_k']) == expected
        assert batching.physical_size(actual['test']['prediction'], k, batch_size=4,
               batch_size_by_k=actual['evaluation']['batch_size_by_k']) == expected
