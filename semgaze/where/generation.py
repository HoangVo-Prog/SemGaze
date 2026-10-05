import torch
from .collator import collate_where, to_model_device, WhereContextOverflowError
from .serialization import serialize_xyd, parse_xyd_output


def decode_response(tokenizer, ids):
    ids = ids.tolist() if hasattr(ids, 'tolist') else list(ids)
    if tokenizer.eos_token_id in ids:
        ids = ids[:ids.index(tokenizer.eos_token_id)]
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


@torch.no_grad()
def generate_where(bundle, episode):
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
