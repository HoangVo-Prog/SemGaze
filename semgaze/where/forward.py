from dataclasses import dataclass
from contextlib import nullcontext
from semgaze.state.extractor import extract_query_states
from .collator import collate_where_batch, to_model_device
from semgaze.model.selected_loss import forward_backbone, compute_selected_causal_nll


@dataclass
class WhereOutput:
    loss_where: object
    states: object
    batch: object
    positions: object
    episode_losses: object = None
    query_image_features: object = None


def forward_where(bundle, episode):
    result = forward_where_batch(bundle, [episode])
    return WhereOutput(result.loss_where, result.states[0], result.batch.samples[0], result.positions[0])


def forward_where_batch(bundle, episodes, *, batch=None, profiler=None):
    stage = profiler.stage if profiler is not None else lambda name: nullcontext()
    with stage('where_collation'):
        batch = batch if batch is not None else collate_where_batch(
            bundle.processor, episodes, bundle.end_fix_id, bundle.context_limit, config=bundle.config)
    if tuple(episodes) != batch.episodes:
        raise ValueError('precollated WHERE batch does not match requested episodes')
    inputs = to_model_device({k: v for k, v in batch.inputs.items() if k != 'labels'}, bundle.model)
    with stage('where_forward'):
        outputs = forward_backbone(bundle.model, inputs, use_cache=bundle.config['where']['supervision']['use_cache'])
        loss = compute_selected_causal_nll(outputs.last_hidden_state, batch.inputs['labels'], bundle.model.get_output_embeddings())
    counts = [len(e.query.x_px) for e in episodes]
    states, mask, positions = extract_query_states(outputs.last_hidden_state, batch.inputs['input_ids'],
        batch.inputs['labels'], bundle.end_fix_id, counts)
    features = outputs.image_hidden_states
    if features.shape[:2] != (sum(m['image_count'] for m in batch.metadata), bundle.processor.image_seq_length):
        raise ValueError('native image feature groups differ from collated image ordering')
    return WhereOutput(loss.loss, [states[b, :n] for b, n in enumerate(counts)], batch, positions,
                       loss.episode_losses, [features[m['query_image_index']] for m in batch.metadata])
