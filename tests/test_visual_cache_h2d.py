from dataclasses import replace
import pytest
import torch
from test_model_path import tiny_bundle, episode
from semgaze.evaluation.test import test_mode as evaluation_mode
from semgaze.model.visual_cache import InferenceVisualCache
from semgaze.where.collator import collate_where, pack_where_batch
from semgaze.where.forward import forward_where_batch
from semgaze.where.generation import generate_where_batch


@pytest.mark.parametrize('k', [1, 5, 10])
def test_miss_only_transfer_order_and_repeated_images(tmp_path, episode, monkeypatch, k):
    bundle = tiny_bundle(tmp_path, episode)
    cache = InferenceVisualCache()
    paths = [f'image-{i}' for i in range(k + 1)]
    pixels = torch.stack([torch.full((3, 448, 448), i / 10.) for i in range(k + 1)])
    native = bundle.model.get_base_model().model
    transferred, computed = [], []
    transfer, vision = cache._transfer_misses, native.get_image_features
    monkeypatch.setattr(cache, '_transfer_misses', lambda x, b: transferred.append(x.clone()) or transfer(x, b))
    def observe(x, **kwargs):
        computed.append(x.clone())
        return vision(x, **kwargs)
    with evaluation_mode(bundle):
        baseline = vision(pixels, return_dict=True).pooler_output
        monkeypatch.setattr(native, 'get_image_features', observe)
        warmed = list(range(0, k + 1, 2))
        cache.get_or_compute_visual_features(bundle, paths=[paths[i] for i in warmed], host_pixel_values=pixels[warmed])
        transferred.clear(); computed.clear()
        actual = cache.get_or_compute_visual_features(bundle, paths=paths + [paths[-1]],
            host_pixel_values=torch.cat((pixels, pixels[-1:])))
        missing = list(range(1, k + 1, 2))
        assert len(transferred) == len(computed) == 1
        assert torch.equal(transferred[0], pixels[missing])
        assert torch.equal(computed[0], pixels[missing])
        # Vision matrix kernels may differ between the full and miss sub-batch.
        torch.testing.assert_close(actual[:-1], baseline, rtol=0, atol=2e-7)
        assert torch.equal(actual[-1], actual[-2])
        transferred.clear(); computed.clear()
        hit = cache.get_or_compute_visual_features(bundle, paths=paths, host_pixel_values=pixels)
        assert torch.equal(hit, actual[:-1])
        assert transferred == computed == []
    for feature in cache.visual.values():
        assert feature.untyped_storage().nbytes() == feature.numel() * feature.element_size()


def test_forward_and_generation_exclude_pixels_before_transfer(tmp_path, episode, monkeypatch):
    from semgaze.where import forward, generation
    bundle = tiny_bundle(tmp_path, episode)
    cache = InferenceVisualCache(max_entries=1)
    transfers = []
    original = forward.to_model_device
    def text_only(inputs, model):
        assert 'pixel_values' not in inputs
        return original(inputs, model)
    monkeypatch.setattr(forward, 'to_model_device', text_only)
    monkeypatch.setattr(generation, 'to_model_device', text_only)
    transfer = cache._transfer_misses
    monkeypatch.setattr(cache, '_transfer_misses', lambda p, b: transfers.append(len(p)) or transfer(p, b))
    with evaluation_mode(bundle):
        forward_where_batch(bundle, [episode], visual_cache=cache)
        # Query survives the visual LRU, then another collation evicts its
        # independent preprocessing entry. Its visual entry pins the CPU row.
        query = episode.query.image_path
        assert query in cache.visual_pixels
        cache.pixels.clear()
        old = cache.misses['preprocessing']
        from semgaze.where.collator import collate_native
        from semgaze.where.conversation import image_user
        collate_native(bundle.processor, [image_user('query')], [query], image_cache=cache)
        assert cache.misses['preprocessing'] == old
        generate_where_batch(bundle, [episode], cache=cache)
        assert transfers == [len(episode.supports) + 1, len(episode.supports)]


@pytest.mark.parametrize('change', ['model', 'vision', 'projector', 'processor', 'image_processor',
                                  'preprocessing', 'dtype', 'device', 'sequence', 'token', 'trainable'])
def test_incompatible_visual_context_rejected(tmp_path, episode, change):
    bundle = tiny_bundle(tmp_path, episode)
    cache = InferenceVisualCache()
    with evaluation_mode(bundle):
        cache.validate(bundle)
        native = bundle.model.get_base_model().model
        if change in ('model', 'processor'):
            other = tiny_bundle(tmp_path, episode)
            other.model.eval()
            setattr(bundle, change, getattr(other, change))
        elif change in ('vision', 'projector'):
            import copy
            name = 'vision_tower' if change == 'vision' else 'multi_modal_projector'
            setattr(native, name, copy.deepcopy(getattr(native, name)))
        elif change == 'image_processor':
            import copy
            bundle.processor.image_processor = copy.deepcopy(bundle.processor.image_processor)
        elif change == 'preprocessing':
            bundle.processor.image_processor.size = {'height': 224, 'width': 224}
        elif change == 'dtype':
            bundle.model.double()
        elif change == 'device':
            bundle.model.to('meta')
        elif change == 'sequence':
            bundle.processor.image_seq_length += 1
        elif change == 'token':
            bundle.processor.image_token_id += 1
        elif change == 'trainable':
            next(native.vision_tower.parameters()).requires_grad_(True)
        with pytest.raises(ValueError, match='incompatible|inference'):
            cache.validate(bundle)


def test_host_preprocessing_mismatch_rejected(tmp_path, episode):
    bundle = tiny_bundle(tmp_path, episode)
    cache = InferenceVisualCache()
    with evaluation_mode(bundle):
        pixels = torch.zeros(1, 3, 448, 448)
        cache.get_or_compute_visual_features(bundle, paths=['a'], host_pixel_values=pixels)
        with pytest.raises(ValueError, match='preprocessing'):
            cache.get_or_compute_visual_features(bundle, paths=['a'], host_pixel_values=pixels + 1)
