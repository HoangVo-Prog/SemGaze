import torch
from semgaze.where.forward import forward_where
from semgaze.where.generation import decode_response
from semgaze.semantic.flat.forward import prepare_flat_inputs
from semgaze.semantic.flat.parser import parse_flat_output
from semgaze.where.forward import forward_where_batch
from semgaze.where.collator import collate_where, pack_where_batch, to_model_device
from semgaze.where.generation import left_pad_generation
from semgaze.semantic.flat.forward import prepare_semantic_batch


@torch.inference_mode()
def evaluate_flat_batch(bundle, episodes, *, generation_budget, cache=None):
    if type(generation_budget) is not int or generation_budget <= 0:
        raise ValueError('declare a positive semantic generation budget explicitly')
    if not episodes or len({len(e.supports) for e in episodes}) != 1:
        raise ValueError('semantic prediction requires same-K episodes')
    samples = [collate_where(bundle.processor, e, bundle.end_fix_id, bundle.context_limit,
               config=bundle.config, image_cache=cache) for e in episodes]
    batch = pack_where_batch(bundle.processor, episodes, samples, cache)
    where = forward_where_batch(bundle, episodes, batch=batch, visual_cache=cache, use_cache=False)
    states = bundle.projector(torch.cat(where.states)).split([len(e.query.x_px) for e in episodes])
    inputs, positions, _ = prepare_semantic_batch(bundle, [e.query for e in episodes], states,
        where=where, generation_budget=generation_budget,
        reuse_query_vision=bundle.config['validation']['cache']['frozen_visual_features'])
    tokenizer = bundle.processor.tokenizer
    inputs = to_model_device(left_pad_generation(inputs, tokenizer.pad_token_id), bundle.model)
    output = bundle.model.generate(**inputs, max_new_tokens=generation_budget, do_sample=False,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    if output.shape[1] > generation_budget:
        raise RuntimeError('installed HF inputs_embeds generation returned an unexpected token prefix')
    results = []
    for b, episode in enumerate(episodes):
        text = decode_response(tokenizer, output[b])
        results.append({'text': text, 'state_source': 'teacher_forced_GT_XYD', 'gold_groups_supplied': True,
            'max_new_tokens': generation_budget, 'inserted_state_count': len(positions[b]),
            **parse_flat_output(text, len(episode.query.x_px), len(episode.query.semantic.why_groups))})
    return results


@torch.no_grad()
def evaluate_flat_episode(bundle, episode, *, generation_budget):
    if type(generation_budget) is not int or generation_budget <= 0:
        raise ValueError('declare a positive semantic generation budget explicitly')
    bundle.model.eval()
    bundle.projector.eval()
    where = forward_where(bundle, episode)
    states = bundle.projector(where.states)
    inputs, positions, _ = prepare_flat_inputs(bundle, episode.query, states, generation_budget=generation_budget)
    tokenizer = bundle.processor.tokenizer
    output = bundle.model.generate(**inputs, max_new_tokens=generation_budget, do_sample=False,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    # HF inputs_embeds-only generation returns newly generated assistant IDs.
    if output.shape[1] > generation_budget:
        raise RuntimeError('installed HF inputs_embeds generation returned an unexpected token prefix')
    text = decode_response(tokenizer, output[0])
    return {'text': text, 'state_source': 'teacher_forced_GT_XYD', 'gold_groups_supplied': True,
            'max_new_tokens': generation_budget, 'inserted_state_count': len(positions),
            **parse_flat_output(text, len(episode.query.x_px), len(episode.query.semantic.why_groups))}
