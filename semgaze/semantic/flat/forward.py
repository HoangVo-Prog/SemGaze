from semgaze.where.collator import collate_native, to_model_device
from semgaze.where.conversation import image_user
from semgaze.state.insertion import insert_states, insert_states_batch
import torch
from contextlib import nullcontext
from .prompt import build_flat_prompt
from .target import build_flat_target
from semgaze.model.selected_loss import forward_backbone, compute_selected_causal_nll


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
    loss, positions, _ = forward_flat_batch(bundle, [query], [states])
    return loss.loss, positions[0]


def frozen_vision_is_reusable(model):
    """Frozen is necessary, but stochastic frozen features cannot be reused exactly."""
    native = model.get_base_model().model
    for producer in (native.vision_tower, native.multi_modal_projector):
        if any(p.requires_grad for p in producer.parameters()):
            return False
        for module in producer.modules():
            if module.training:
                for attr in ('p', 'drop_prob', 'attention_dropout'):
                    value = getattr(module, attr, 0)
                    if isinstance(value, (int, float)) and value > 0:
                        return False
    return True


def prepare_semantic_batch(bundle, queries, states, *, where=None):
    """One native prompt per sample, followed by differentiable batched assembly."""
    if not queries or len(queries) != len(states):
        raise ValueError('query/state batch counts differ')
    reuse = (where is not None and bundle.config['training'].get('reuse_query_vision', True)
             and frozen_vision_is_reusable(bundle.model))
    natives, embeddings, metadata = [], [], []
    image_cache = where.batch.image_cache if where is not None else {}
    for b, (query, state) in enumerate(zip(queries, states)):
        prompt = build_flat_prompt(query)
        native = collate_native(bundle.processor, [image_user(prompt.text)], [query.image_path],
            build_flat_target(query.semantic), prompt=prompt, image_cache=image_cache)
        n = len(query.x_px)
        length = native.inputs['input_ids'].shape[1]
        if length + n > bundle.context_limit:
            raise SemanticContextOverflowError(sample_id=query.record_id, native_prompt_length=native.response_start,
                number_of_inserted_states=n, number_of_gold_groups=len(query.semantic.why_groups),
                final_prompt_length=native.response_start+n, target_length=length-native.response_start,
                generation_budget=None, context_limit=bundle.context_limit)
        if state.shape != (n, bundle.projector[0].in_features):
            raise ValueError('full chronological R shape differs from [N,d_model]')
        ids = native.inputs['input_ids']
        if bool((ids == bundle.end_fix_id).any()):
            raise ValueError('semantic conversation must not contain END_FIX')
        embedding = bundle.model.get_input_embeddings()(ids.to(bundle.input_row.device))
        if reuse:
            # Feature identity includes native preprocessing. Compare on host before reuse.
            sample = where.batch.samples[b]
            if sample.image_paths[-1] != query.image_path or not torch.equal(
                    sample.inputs['pixel_values'][-1:], native.inputs['pixel_values']):
                raise ValueError('WHERE/semantic query preprocessing differs; cannot reuse features')
            features = where.query_image_features[b].to(embedding)
            mask = (ids == bundle.processor.image_token_id)
            if int(mask.sum()) != features.shape[0]:
                raise ValueError('semantic image placeholders do not match query features')
            embedding = embedding.masked_scatter(mask.to(embedding.device).unsqueeze(-1), features)
        embeddings.append(embedding)
        natives.append(native)
        metadata.append(dict(semantic_length=length+n,
            supervised_semantic_tokens=int((native.inputs['labels'][:, 1:] != -100).sum()),
            reused_query_vision=reuse))
    fused, positions = insert_states_batch(embeddings, [s.inputs['attention_mask'] for s in natives],
        [s.inputs['labels'] for s in natives], states, [s.boundaries for s in natives])
    # Keep host labels for selected index discovery; move only the backbone inputs.
    if not reuse:
        fused['pixel_values'] = torch.cat([s.inputs['pixel_values'] for s in natives])
    return fused, positions, metadata


def forward_flat_batch(bundle, queries, states, *, where=None, profiler=None):
    stage = profiler.stage if profiler is not None else lambda name: nullcontext()
    with stage('semantic_preparation'):
        inputs, positions, metadata = prepare_semantic_batch(bundle, queries, states, where=where)
    host_labels = inputs['labels']
    with stage('semantic_forward'):
        outputs = forward_backbone(bundle.model, to_model_device(
            {k: v for k, v in inputs.items() if k != 'labels'}, bundle.model))
        loss = compute_selected_causal_nll(outputs.last_hidden_state, host_labels, bundle.model.get_output_embeddings())
    return loss, positions, metadata
