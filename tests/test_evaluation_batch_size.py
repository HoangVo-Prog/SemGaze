"""A single evaluation batch size controls loss and generation for every K."""
import json
from types import SimpleNamespace

import pytest

from semgaze.evaluation import batching, predictions
from semgaze.model.config import resolve_config


def _episode(k, key):
    return SimpleNamespace(
        supports=(None,) * k,
        query=SimpleNamespace(record_id=key, x_px=()),
        draw_id=None,
    )


def test_global_batch_size_config_and_legacy_fallback():
    config = resolve_config({'evaluation': {'batch_size': 3}})
    assert config['evaluation']['batch_size'] == 3
    assert resolve_config({})['evaluation']['batch_size'] is None
    settings = {'execution': 'same_k_batched', 'batch_size_by_k': {'1': 1, '5': 4}}
    assert batching.physical_size(settings, 5) == 4
    assert batching.physical_size(settings, 1, batch_size=3) == 3
    assert batching.physical_size(settings, 5, batch_size=3) == 3
    settings['execution'] = 'serial'
    assert batching.physical_size(settings, 5, batch_size=3) == 1


@pytest.mark.parametrize('invalid', [0, -1, True, 2.5, '2'])
def test_global_batch_size_rejects_invalid_values(invalid):
    with pytest.raises(ValueError, match='evaluation.batch_size'):
        resolve_config({'evaluation': {'batch_size': invalid}})


def test_common_batch_size_reaches_loss_scheduler_and_prediction(monkeypatch):
    episodes = [_episode(1, 'a'), _episode(5, 'b'), _episode(1, 'c'),
                _episode(5, 'd'), _episode(1, 'e')]
    old_sizes = {'1': 1, '5': 1}
    loss_settings = {'execution': 'same_k_batched', 'batch_size_by_k': old_sizes,
                     'bucket_by_length': False, 'bucket_window': 64}
    monkeypatch.setattr(batching, 'collate_where',
                        lambda *a, **kw: SimpleNamespace(
                            inputs={'input_ids': SimpleNamespace(shape=(1, 10))}))
    monkeypatch.setattr(batching, 'pack_where_batch',
                        lambda processor, group, samples, cache: SimpleNamespace(episodes=group))
    bundle = SimpleNamespace(processor=None, end_fix_id=1, context_limit=8192, config={})
    loss_batches = list(batching.schedule_batches(
        bundle, episodes, loss_settings, cache={}, batch_size=2))
    assert [(len(batch.episodes[0].supports), len(batch.episodes))
            for batch in loss_batches] == [(1, 2), (1, 1), (5, 2)]
    assert sorted(e.query.record_id for b in loss_batches for e in b.episodes) == list('abcde')

    bundle.config = {'evaluation': {'batch_size': 2}, 'test': {'prediction': {
        'execution': 'same_k_batched', 'batch_size_by_k': old_sizes,
        'bucket_by_length': False, 'bucket_window': 64}}}
    calls = []

    def generate_where(bundle, group, *, cache, **kwargs):
        calls.append(('where', len(group[0].supports), len(group)))
        return [{'text': 'WHERE'} for _ in group]

    def generate_semantics(bundle, group, *, generation_budget, cache, projected_r_cache):
        assert generation_budget == 8
        calls.append(('semantic', len(group[0].supports), len(group)))
        return [{'text': 'SEMANTIC'} for _ in group]

    monkeypatch.setattr(predictions, 'generate_where_batch', generate_where)
    monkeypatch.setattr(predictions, 'evaluate_flat_batch', generate_semantics)
    produced = list(predictions.prediction_batches(bundle, episodes, budget=8, cache={}, path='both'))
    assert [episode.query.record_id for episode, _ in produced] == list('abcde')
    assert calls == [('where', 1, 2), ('semantic', 1, 2),
                     ('where', 1, 1), ('semantic', 1, 1),
                     ('where', 5, 2), ('semantic', 5, 2)]
    assert all(generated[0]['text'] == 'WHERE' and generated[1]['text'] == 'SEMANTIC'
               for _, generated in produced)


def test_standalone_eval_config_can_override_checkpoint(tmp_path):
    from evaluate_flat import _load_evaluation_config

    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'resolved_config.json').write_text(
        json.dumps(resolve_config({})), encoding='utf-8')
    override = tmp_path / 'evaluation.yaml'
    override.write_text('evaluation:\n  batch_size: 4\n', encoding='utf-8')
    resolved = _load_evaluation_config(checkpoint, override)
    assert resolved['evaluation']['batch_size'] == 4
    assert resolved['test']['loss']['batch_size_by_k']['5'] == 1
    assert resolved['test']['prediction']['batch_size_by_k']['5'] == 1
