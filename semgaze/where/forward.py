from dataclasses import dataclass
from semgaze.state.extractor import extract_query_states
from .collator import collate_where, to_model_device


@dataclass
class WhereOutput:
    loss_where: object
    states: object
    batch: object
    positions: object


def forward_where(bundle, episode):
    batch = collate_where(bundle.processor, episode, bundle.end_fix_id, bundle.context_limit, config=bundle.config)
    inputs = to_model_device(batch.inputs, bundle.model)
    outputs = bundle.model(**inputs, output_hidden_states=True, use_cache=bundle.config['where']['supervision']['use_cache'], return_dict=True)
    states, mask, positions = extract_query_states(outputs.hidden_states[-1], inputs['input_ids'],
        inputs['labels'], bundle.end_fix_id, [len(episode.query.x_px)])
    if not mask.all():
        raise ValueError('single-episode extraction unexpectedly padded')
    return WhereOutput(outputs.loss, states[0], batch, positions[0])
