import pytest
from test_model_path import tiny_bundle, episode
from test_epoch_validation import evaluation_data
from semgaze.evaluation import test as loss_module, predictions
from semgaze.model.visual_cache import InferenceVisualCache


@pytest.mark.parametrize('external', [False, True])
@pytest.mark.parametrize('phase', ['loss', 'prediction'])
@pytest.mark.parametrize('fail', [False, True])
def test_high_level_cache_ownership(tmp_path, episode, monkeypatch, external, phase, fail):
    bundle = tiny_bundle(tmp_path, episode)
    train, queries, manifest = evaluation_data(episode)
    bundle.config['evaluation']['k_values'] = [1]
    manifest['support_draws']['1'] = manifest['support_draws']['1'][:1]
    predictions.resolve_prediction_settings(bundle.config, 2)
    cache = InferenceVisualCache()
    module = loss_module if phase == 'loss' else predictions
    monkeypatch.setattr(module, 'InferenceVisualCache', lambda **kw: cache)

    def losses(bundle, episodes, **kwargs):
        assert kwargs['cache'] is cache and not cache.closed
        if fail:
            raise RuntimeError('injected')
        return [dict.fromkeys(loss_module.EVAL_KEYS, 1.) for _ in episodes]

    def generated(bundle, episodes, **kwargs):
        assert kwargs['cache'] is cache and not cache.closed
        if fail:
            raise RuntimeError('injected')
        for ep in episodes:
            yield ep, ({'text': 'where'}, {'text': 'semantic'})

    monkeypatch.setattr(loss_module, 'batch_losses', losses)
    monkeypatch.setattr(predictions, 'prediction_batches', generated)
    kwargs = {'visual_cache': cache} if external else {}

    def run():
        if phase == 'loss':
            return loss_module.evaluate_test_epoch(bundle, train, queries, manifest, **kwargs)
        return predictions.predict_epoch(bundle, [], train, queries, manifest,
            epoch=1, step=1, split_manifest_identity='fixture', **kwargs)

    if fail:
        with pytest.raises(RuntimeError, match='injected'):
            run()
    else:
        run()
    assert cache.closed == (not external)
    cache.close()
