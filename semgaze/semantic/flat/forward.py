from semgaze.where.collator import collate_native, to_model_device
from semgaze.where.conversation import image_user
from semgaze.state.insertion import insert_states
from .prompt import build_flat_prompt
from .target import build_flat_target


class SemanticContextOverflowError(ValueError):
    def __init__(self, **diagnostics):
        self.diagnostics = {'semantic_mode': 'flat_single_output', **diagnostics}
        super().__init__(str(self.diagnostics))


def prepare_flat_inputs(bundle, query, states, *, generation_budget=None):
    prompt = build_flat_prompt(query)
    target = build_flat_target(query.semantic) if generation_budget is None else None
    native = collate_native(bundle.processor, [image_user(prompt.text)], [query.image_path], target, prompt=prompt)
    n, m = len(query.x_px), len(query.semantic.why_groups)
    native_length = native.inputs['input_ids'].shape[1]
    budget = native_length + n + (generation_budget or 0)
    if budget > bundle.context_limit:
        raise SemanticContextOverflowError(sample_id=query.record_id,
            native_prompt_length=native.response_start, number_of_inserted_states=n,
            number_of_gold_groups=m, final_prompt_length=native.response_start + n,
            target_length=native_length - native.response_start if target is not None else None,
            generation_budget=generation_budget, context_limit=bundle.context_limit)
    if states.shape != (n, bundle.projector[0].in_features):
        raise ValueError('full chronological R shape differs from [N,d_model]')
    inputs = to_model_device(native.inputs, bundle.model)
    if (inputs['input_ids'] == bundle.end_fix_id).any():
        raise ValueError('semantic conversation must not contain END_FIX')
    embeddings = bundle.model.get_input_embeddings()(inputs['input_ids'])
    fused, positions = insert_states(embeddings, inputs['attention_mask'], inputs['labels'], states, native.boundaries)
    fused['pixel_values'] = inputs['pixel_values']
    if generation_budget is not None:
        fused.pop('labels')
    return fused, positions, native


def forward_flat(bundle, query, states):
    inputs, positions, native = prepare_flat_inputs(bundle, query, states)
    outputs = bundle.model(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
    return outputs.loss, positions
