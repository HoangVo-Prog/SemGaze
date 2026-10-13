"""Native InternVL one-image/one-tile collation, with exact assistant-only labels."""
import torch
from torch.nn.utils.rnn import pad_sequence

from semgaze.where.collator import collate_native, to_model_device
from semgaze.where.conversation import image_user
from semgaze.where.generation import decode_response
from .prompt import build_baseline_prompt, semantic_target


def make_native(processor, query, style, *, training):
    prompt = build_baseline_prompt(query, style)
    target = semantic_target(query) if training else None
    native = collate_native(processor, [image_user(prompt.text)], [query.image_path], target)
    if native.image_paths != (query.image_path,):
        raise ValueError('semantic baseline unexpectedly loaded multiple images')
    if tuple(native.inputs['pixel_values'].shape) != (1, 3, 448, 448):
        raise ValueError('expected exactly one native 448x448 tile')
    if training:
        labels = native.inputs['labels'][0]
        if not ((labels[:native.response_start] == -100).all() and
                (labels[native.response_start:] != -100).all()):
            raise ValueError('assistant-only training labels are not a contiguous suffix')
        if int((labels != -100).sum()) != native.response_length:
            raise ValueError('assistant-only response length mismatch')
    return native


def pack_natives(processor, natives, *, left_pad=False, training=False):
    """Keep pixel order synchronized with text order; do not make any support slots."""
    if not natives:
        raise ValueError('cannot pack empty batch')
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError('native InternVL tokenizer requires a pad token')
    tensors = {}
    for key, pad in [('input_ids', pad_id), ('attention_mask', 0)]:
        rows = [n.inputs[key][0] for n in natives]
        if left_pad:
            length = max(x.numel() for x in rows)
            tensors[key] = torch.stack([torch.nn.functional.pad(x, (length - x.numel(), 0), value=pad)
                                        for x in rows])
        else:
            tensors[key] = pad_sequence(rows, batch_first=True, padding_value=pad)
    if training:
        tensors['labels'] = pad_sequence([n.inputs['labels'][0] for n in natives],
                                         batch_first=True, padding_value=-100)
        if bool((tensors['labels'][tensors['attention_mask'] == 0] != -100).any()):
            raise ValueError('padding may not contribute to training loss')
    tensors['pixel_values'] = torch.cat([n.inputs['pixel_values'] for n in natives], dim=0)
    if tensors['pixel_values'].shape[0] != len(natives):
        raise ValueError('one image per baseline query required')
    return tensors


def model_inputs(inputs, model):
    return to_model_device(inputs, model)


def decode_generated(tokenizer, output_ids, input_ids, *, max_new_tokens):
    """Handle both standard HF prefix-return and generation-only model implementations."""
    length = input_ids.shape[-1]
    if output_ids.shape[-1] >= length and torch.equal(output_ids[..., :length], input_ids):
        new = output_ids[..., length:]
    elif output_ids.shape[-1] <= max_new_tokens:
        new = output_ids
    else:
        raise RuntimeError('HF generate output has neither input prefix nor generation-only layout')
    if new.shape[-1] > max_new_tokens:
        raise RuntimeError('semantic generation exceeded requested budget')
    return [decode_response(tokenizer, sequence) for sequence in new]
