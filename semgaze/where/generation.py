import torch
from .collator import collate_where, to_model_device, WhereContextOverflowError
from .serialization import serialize_xyd, parse_xyd_output
from .collator import pack_where_batch


def decode_response(tokenizer, ids):
    ids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
    if tokenizer.eos_token_id in ids:
        ids = ids[:ids.index(tokenizer.eos_token_id)]
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def left_pad_generation(inputs, pad_token_id):
    """Convert shared right-padded collation to decoder-only generation layout."""
    result = dict(inputs)
    mask = inputs['attention_mask']
    for key in ('input_ids', 'inputs_embeds', 'attention_mask'):
        if key not in inputs:
            continue
        value = inputs[key]
        target = torch.full_like(value, pad_token_id if key == 'input_ids' else 0)
        for b in range(len(mask)):
            selected = value[b][mask[b].bool()]
            target[b, -len(selected):] = selected
        result[key] = target
    result.pop('labels', None)
    return result


def where_generation_budget(bundle, episode):
    n = len(episode.query.x_px)
    return max(64, len(bundle.processor.tokenizer.encode(serialize_xyd([(99, 99, 999)] * n,
        bundle.config['where']['end_fix_token']), add_special_tokens=False)) + 16)


@torch.inference_mode()
def generate_where_batch(bundle, episodes, *, cache=None, precomputed_samples=None):
    if not episodes or len({len(e.supports) for e in episodes}) != 1:
        raise ValueError('WHERE generation requires same-K episodes')
    if precomputed_samples is not None and len(precomputed_samples) != len(episodes):
        raise ValueError('precomputed WHERE samples must align with episodes')
    tokenizer = bundle.processor.tokenizer
    budgets = [where_generation_budget(bundle, e) for e in episodes]
    results = [None] * len(episodes)
    # Preserve the exact per-episode max_new_tokens. Different oracle lengths can
    # have different budgets, so they form separate physical generation groups.
    for budget in dict.fromkeys(budgets):
        indices = [i for i, value in enumerate(budgets) if value == budget]
        group = [episodes[i] for i in indices]
        samples = ([precomputed_samples[i] for i in indices]
                   if precomputed_samples is not None else
                   [collate_where(bundle.processor, e, bundle.end_fix_id, bundle.context_limit,
                    teacher_forcing=False, config=bundle.config, image_cache=cache) for e in group])
        batch = pack_where_batch(bundle.processor, group, samples, cache)
        inputs = left_pad_generation(batch.inputs, tokenizer.pad_token_id)
        length = inputs['input_ids'].shape[1]
        if length + budget > bundle.context_limit:
            raise WhereContextOverflowError('frozen WHERE prefix + generation budget exceeds context limit')
        host_pixels = inputs['pixel_values']
        cached_features = cache is not None and cache.features
        inputs = to_model_device({k: v for k, v in inputs.items()
                                  if not (cached_features and k == 'pixel_values')}, bundle.model)
        # Retain IDs for the original generation prefix/output contract while HF
        # uses cached multimodal embeddings only on its initial generation step.
        if cache is not None and cache.features:
            fused, _ = cache.fuse(bundle, inputs, [p for s in samples for p in s.image_paths],
                                  host_pixel_values=host_pixels)
            inputs = fused | {'input_ids': inputs['input_ids']}
        output = bundle.model.generate(**inputs, max_new_tokens=budget, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
        for row, index in enumerate(indices):
            text = decode_response(tokenizer, output[row, length:])
            results[index] = {'text': text, 'max_new_tokens': budget,
                **parse_xyd_output(text, len(episodes[index].query.x_px), bundle.config['where']['end_fix_token'])}
    return results


@torch.no_grad()
def generate_where(bundle, episode, *, cache=None):
    if cache is not None:
        bundle.model.eval()
        return generate_where_batch(bundle, [episode], cache=cache)[0]
    bundle.model.eval()
    n = len(episode.query.x_px)
    tokenizer = bundle.processor.tokenizer
    token = bundle.config['where']['end_fix_token']
    budget = max(64, len(tokenizer.encode(serialize_xyd([(99, 99, 999)] * n, token), add_special_tokens=False)) + 16)
    batch = collate_where(bundle.processor, episode, bundle.end_fix_id, bundle.context_limit, teacher_forcing=False, config=bundle.config)
    inputs = to_model_device(batch.inputs, bundle.model)
    inputs.pop('labels')
    prompt_length = inputs['input_ids'].shape[1]
    if prompt_length + budget > bundle.context_limit:
        raise WhereContextOverflowError('frozen WHERE prefix + generation budget exceeds context limit')
    output = bundle.model.generate(**inputs, max_new_tokens=budget, do_sample=False,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    text = decode_response(tokenizer, output[0, prompt_length:])
    return {'text': text, 'max_new_tokens': budget, **parse_xyd_output(text, n, token)}
