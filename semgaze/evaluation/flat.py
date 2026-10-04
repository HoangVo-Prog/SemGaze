import torch
from semgaze.where.forward import forward_where
from semgaze.where.generation import decode_response
from semgaze.semantic.flat.forward import prepare_flat_inputs
from semgaze.semantic.flat.parser import parse_flat_output


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
