from dataclasses import replace
import pytest
import torch
from test_model_path import tiny_bundle, episode
from semgaze.evaluation.cache import ProjectedWhereCache, episode_key
from semgaze.evaluation.test import test_mode as evaluation_mode


def test_exact_episode_keys(episode):
    support = episode.supports[0]
    second = replace(support, record_id='second', stimulus_id='second-image')
    base = replace(episode, supports=(support, second), draw_id=0)
    variants = [
        replace(base, query=replace(base.query, record_id='new-query')),
        replace(base, query=replace(base.query, x_px=(0., *base.query.x_px[1:]))),
        replace(base, supports=(support,)), replace(base, draw_id=1),
        replace(base, supports=(replace(support, record_id='new-support'), second)),
        replace(base, supports=(replace(support, duration_ms=(1., *support.duration_ms[1:])), second)),
        replace(base, supports=(second, support)),
    ]
    assert episode_key(base) == episode_key(replace(base))
    assert len({episode_key(e) for e in [base, *variants]}) == len(variants) + 1


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
def test_exact_cpu_storage_and_closed_cache(tmp_path, episode, dtype):
    bundle = tiny_bundle(tmp_path, episode)
    cache = ProjectedWhereCache(bundle=bundle, split_manifest_identity='split', cycle_id=(1, 2))
    with evaluation_mode(bundle):
        r = torch.arange(len(episode.query.x_px) * 32, dtype=dtype).reshape(-1, 32)
        cache.put(episode, r)
        stored = cache.get(episode)
        assert stored.device.type == 'cpu' and stored.dtype == dtype
        assert torch.equal(stored, r) and stored.data_ptr() != r.data_ptr()
        assert cache.get(replace(episode, draw_id=1)) is None
        cache.close()
        cache.close()
        assert not cache._entries
        with pytest.raises(ValueError, match='closed'):
            cache.get(episode)


@pytest.mark.parametrize('field', ['model', 'projector', 'processor'])
def test_object_provenance(tmp_path, episode, field):
    bundle = tiny_bundle(tmp_path, episode)
    cache = ProjectedWhereCache(bundle=bundle, split_manifest_identity='split', cycle_id=(1, 2))
    other = tiny_bundle(tmp_path, episode)
    setattr(bundle, field, getattr(other, field))
    with pytest.raises(ValueError, match='provenance'):
        cache.validate(bundle)


def test_split_cycle_and_parameter_mutation(tmp_path, episode):
    bundle = tiny_bundle(tmp_path, episode)
    cache = ProjectedWhereCache(bundle=bundle, split_manifest_identity='split', cycle_id=(1, 2))
    for kwargs in ({'split_manifest_identity': 'other'}, {'cycle_id': (2, 3)}):
        with pytest.raises(ValueError, match='provenance'):
            cache.validate(bundle, **kwargs)
    cache.validate(bundle, manifest={'split': 1})
    with pytest.raises(ValueError, match='manifest'):
        cache.validate(bundle, manifest={'split': 2})
    with torch.no_grad():
        next(bundle.projector.parameters()).add_(1)
    with pytest.raises(ValueError, match='provenance'):
        cache.validate(bundle)
