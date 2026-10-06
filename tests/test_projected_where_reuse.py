import pytest
import torch
from test_model_path import tiny_bundle, episode
from semgaze.evaluation.cache import ProjectedWhereCache
from semgaze.evaluation.test import batch_losses, test_mode as evaluation_mode
from semgaze.evaluation.flat import evaluate_flat_batch
from semgaze.model.visual_cache import InferenceVisualCache
from semgaze.where.collator import collate_where, pack_where_batch


def test_loss_capture_and_all_hit_skips_gt_where_and_projector(tmp_path, episode, monkeypatch):
    bundle = tiny_bundle(tmp_path, episode)
    visual = InferenceVisualCache()
    projected = ProjectedWhereCache(bundle=bundle, split_manifest_identity='split', cycle_id=(1, 1))
    samples = [collate_where(bundle.processor, episode, bundle.end_fix_id, bundle.context_limit,
                             config=bundle.config, image_cache=visual)]
    batch = pack_where_batch(bundle.processor, [episode], samples, visual)
    with evaluation_mode(bundle):
        batch_losses(bundle, [episode], batch=batch, cache=visual, projected_r_cache=projected)
        assert projected.contains(episode)
        where_calls, projector_calls = [], []
        original_where = __import__('semgaze.evaluation.flat', fromlist=['forward_where_batch']).forward_where_batch
        original_projector = bundle.projector.forward
        monkeypatch.setattr('semgaze.evaluation.flat.forward_where_batch',
                            lambda *args, **kwargs: where_calls.append(1) or original_where(*args, **kwargs))
        monkeypatch.setattr(bundle.projector, 'forward',
                            lambda *args, **kwargs: projector_calls.append(1) or original_projector(*args, **kwargs))
        result = evaluate_flat_batch(bundle, [episode], generation_budget=2, cache=visual,
                                     projected_r_cache=projected)
        assert len(result) == 1
        assert where_calls == [] and projector_calls == []
        assert projected.get(episode).dtype == bundle.projector[0].weight.dtype
