"""CPU semantic-native cache tests; no network/model required."""
from types import SimpleNamespace
from test_model_path import episode
import torch
from semgaze.semantic.flat.collation_cache import SemanticNativeLRU


def test_text_cache_hits_preserve_exact_fields_and_are_non_aliasing():
    cache = SemanticNativeLRU(max_entries=1)
    inputs = {'input_ids': torch.tensor([[1,2,3]]),
              'attention_mask': torch.ones(1,3,dtype=torch.long),
              'labels': torch.tensor([[-100,2,3]]),
              'pixel_values': torch.rand(1,3,448,448)}
    from semgaze.where.collator import NativeBatch
    native = NativeBatch(inputs, (1,2), 1, 2, 'text', ('image',), ((0,1),(1,2)))
    cache.put(('a',), native)
    pixels = torch.rand(1,3,448,448)
    one = cache.get(('a',), pixels=pixels)
    assert one.inputs['pixel_values'] is pixels
    assert one.boundaries == native.boundaries and one.response_offsets == native.response_offsets
    for key in ('input_ids','attention_mask','labels'):
        torch.testing.assert_close(one.inputs[key],native.inputs[key],atol=0,rtol=0)
    one.inputs['input_ids'][0,0] = 999
    two = cache.get(('a',), pixels=pixels)
    assert two.inputs['input_ids'][0,0] == 1
    cache.put(('b',), native)
    assert cache.get(('a',), pixels=pixels) is None
    assert cache.statistics()['evictions'] == 1
    cache.close()

def test_cached_support_pixels_match_native_processor(tmp_path, episode):
    from test_model_path import tiny_bundle
    from semgaze.where.collator import collate_where
    from semgaze.model.visual_cache import InferenceVisualCache
    bundle = tiny_bundle(tmp_path, episode)
    cache = InferenceVisualCache(preprocessing=True,features=False,max_entries=16)
    reference = collate_where(bundle.processor,episode,bundle.end_fix_id,bundle.context_limit,
                              config=bundle.config,image_cache={})
    first = collate_where(bundle.processor,episode,bundle.end_fix_id,bundle.context_limit,
                          config=bundle.config,image_cache=cache)
    second = collate_where(bundle.processor,episode,bundle.end_fix_id,bundle.context_limit,
                           config=bundle.config,image_cache=cache)
    for key in ('input_ids','attention_mask','labels','pixel_values'):
        torch.testing.assert_close(first.inputs[key],reference.inputs[key],rtol=0,atol=0)
        torch.testing.assert_close(second.inputs[key],reference.inputs[key],rtol=0,atol=0)
    assert cache.role_hits['support'] >= len(episode.supports)
    cache.close()


def test_semantic_text_cache_hit_preserves_fused_embeddings_and_labels(tmp_path, episode):
    from test_model_path import tiny_bundle
    from semgaze.where.forward import forward_where_batch
    from semgaze.semantic.flat.forward import prepare_semantic_batch
    bundle = tiny_bundle(tmp_path, episode)
    where = forward_where_batch(bundle, [episode])
    states = [bundle.projector(where.states[0])]
    cache = SemanticNativeLRU(max_entries=4)
    baseline, bp, _ = prepare_semantic_batch(bundle,[episode.query],states,where=where)
    _, _, _ = prepare_semantic_batch(bundle,[episode.query],states,where=where,semantic_cache=cache)
    reused, rp, _ = prepare_semantic_batch(bundle,[episode.query],states,where=where,semantic_cache=cache)
    for key in baseline:
        torch.testing.assert_close(baseline[key],reused[key],rtol=0,atol=0)
    assert bp == rp and cache.hits == 1 and cache.misses == 1

