from dataclasses import dataclass
import torch
from contextlib import nullcontext
from semgaze.state.extractor import extract_query_states, extract_duration_numeric_states
from .collator import collate_where_batch, to_model_device
from .weighting import build_fixation_target_weights
from .serialization import active_end_fix_token
from semgaze.model.selected_loss import forward_backbone, compute_selected_causal_nll


@dataclass
class WhereOutput:
    loss_where: object
    states: object
    batch: object
    positions: object
    episode_losses: object = None
    query_image_features: object = None
    unweighted_episode_losses: object = None


def forward_where(bundle, episode):
    result = forward_where_batch(bundle, [episode])
    return WhereOutput(result.loss_where, result.states[0], result.batch.samples[0], result.positions[0])


def forward_where_batch(bundle, episodes, *, batch=None, profiler=None, visual_cache=None, use_cache=None,
                        audit_disable_where_attention_mask=False, fixation_weighting=None):
    stage = profiler.stage if profiler is not None else lambda name: nullcontext()
    with stage('where_collation'):
        batch = batch if batch is not None else collate_where_batch(
            bundle.processor, episodes, bundle.end_fix_id, bundle.context_limit, config=bundle.config, image_cache=visual_cache)
    if tuple(episodes) != batch.episodes:
        raise ValueError('precollated WHERE batch does not match requested episodes')
    cached_features = visual_cache is not None and visual_cache.features
    inputs = to_model_device({k: v for k, v in batch.inputs.items()
                              if k != 'labels' and not (cached_features and k == 'pixel_values')}, bundle.model)
    with stage('where_forward'):
        features = None
        if visual_cache is not None:
            inputs, features = visual_cache.fuse(bundle, inputs, [p for s in batch.samples for p in s.image_paths],
                                                  host_pixel_values=batch.inputs['pixel_values'])
        if audit_disable_where_attention_mask:
            inputs = {k: v for k, v in inputs.items() if k != 'attention_mask'}
        outputs = forward_backbone(bundle.model, inputs, use_cache=(bundle.config['where']['supervision']['use_cache']
                                  if use_cache is None else use_cache))
        target_weights = build_fixation_target_weights(
            batch, episodes, config=fixation_weighting,
            end_fix_token=active_end_fix_token(bundle.config))
        loss = compute_selected_causal_nll(outputs.last_hidden_state, batch.inputs['labels'],
            bundle.model.get_output_embeddings(), target_weights=target_weights)
    counts = [len(e.query.x_px) for e in episodes]
    if bundle.config['semantic']['conditioning'] == 'explicit_scanpath_text':
        # No WHERE hidden-state extraction or semantic hand-off in V2.
        states = [outputs.last_hidden_state.new_empty((0, outputs.last_hidden_state.shape[-1])) for _ in episodes]
        positions = [torch.empty(0, dtype=torch.long) for _ in episodes]
    elif bundle.config['where']['use_end_fix_token']:
        states, _, positions = extract_query_states(outputs.last_hidden_state, batch.inputs['input_ids'],
            batch.inputs['labels'], bundle.end_fix_id, counts, return_unpadded=True)
    else:
        states, _, positions = extract_duration_numeric_states(outputs.last_hidden_state,
            batch.samples, counts, return_unpadded=True)
    features = outputs.image_hidden_states if features is None else features
    if features.shape[:2] != (sum(m['image_count'] for m in batch.metadata), bundle.processor.image_seq_length):
        raise ValueError('native image feature groups differ from collated image ordering')
    return WhereOutput(loss.loss, states, batch, positions,
                        loss.episode_losses, [features[m['query_image_index']] for m in batch.metadata],
                        loss.unweighted_episode_losses)
