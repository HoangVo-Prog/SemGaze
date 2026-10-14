"""Conservative, causally correct sparse fixation-level scheduled sampling.

One GT-prefix-conditioned fixation replacement per selected episode; generate
that fixation autoregressively to END_FIX, then teacher-force only the other GT
fixations using the mixed history. The existing GT WHERE path supplies semantic
states; sampled-history tokens never receive CE supervision.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import re
from dataclasses import dataclass
from collections import Counter

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from semgaze.model.selected_loss import compute_selected_causal_nll, forward_backbone
from semgaze.where.collator import (collate_where, collate_native, pack_where_batch,
                                    to_model_device)
from semgaze.where.conversation import build_where_conversation
from semgaze.where.serialization import serialize_xyd, serialize_xyd_record
from semgaze.where.coordinates import coordinate_bin
from semgaze.where.duration import duration_bin
from semgaze.where.weighting import mask_replaced_fixation, build_fixation_target_weights


class _StopAtEndFix(StoppingCriteria):
    def __init__(self, end_fix_id: int, prompt_length: int):
        self.end_fix_id = end_fix_id
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        # Only one episode per sampling call.
        return bool((input_ids[0, self.prompt_length:] == self.end_fix_id).any())


def _episode_rng(seed: int, global_step: int, record_id: str):
    key = f'{seed}|{global_step}|{record_id}|semgaze-fixation-ss'.encode('utf-8')
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], 'big'))


def sampling_probability(config, global_step, total_optimizer_updates):
    base = min(float(config['probability']), float(config['max_probability']))
    warmup = max(1, round(float(config['warmup_ratio']) * total_optimizer_updates))
    return base * min(1.0, max(0, global_step + 1) / warmup)


def _gt_fixations(record):
    return [(coordinate_bin(x, record.image_width), coordinate_bin(y, record.image_height), duration_bin(d))
            for x, y, d in zip(record.x_px, record.y_px, record.duration_ms)]


def _parse_generated_fixation(text: str, end_fix_token: str):
    pattern = r'^\(\s*(\d{1,2})\s*,\s*(\d{1,2})\s*,\s*(\d{1,3})\s*\)\s*' + re.escape(end_fix_token) + r'$'
    match = re.fullmatch(pattern, text.strip())
    if match is None:
        return None
    result = tuple(map(int, match.groups()))
    return result if (0 <= result[0] <= 99 and 0 <= result[1] <= 99 and 0 <= result[2] <= 999) else None


def _write_ss_debug_trace(bundle, episode, *, step, replace_index, gt, prefix,
                          prompt, input_ids, generated, end_fix_id, tokenizer,
                          status, max_new_tokens, cache):
    """Optional observational trace; NO impact on acceptance, loss, or decoding.

    Set SEMGAZE_SS_TRACE_PATH=/path/to/ss_trace.jsonl to enable. Writes only
    short text tails, token IDs, and local sample identity; never image tensors.
    """
    path = os.environ.get('SEMGAZE_SS_TRACE_PATH')
    if not path:
        return
    end_pos = generated.index(end_fix_id) if end_fix_id in generated else None
    eos_id = tokenizer.eos_token_id
    trace = {
        'event': 'ss_generation_trace', 'step': step,
        'record_id': episode.query.record_id, 'K': len(episode.supports),
        'replacement_index': replace_index, 'gt_replaced_fixation': gt[replace_index-1],
        'prefix_tail_text': prefix[-200:], 'prefix_token_ids_tail': input_ids[0, -96:].tolist(),
        'new_token_ids': generated,
        'generated_text_with_special_tokens': tokenizer.decode(
            generated, skip_special_tokens=False, clean_up_tokenization_spaces=False),
        'generated_text_without_special_tokens': tokenizer.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False),
        'end_fix_token_id': end_fix_id, 'contains_end_fix_id': end_pos is not None,
        'end_fix_position': end_pos,
        'eos_position': generated.index(eos_id) if eos_id in generated else None,
        'generation_length': len(generated), 'max_new_tokens': max_new_tokens,
        'finish_reason_inferred': ('end_fix' if end_pos is not None else
            'eos' if eos_id in generated else 'max_new_tokens_or_other'),
        'status': status,
    }
    # Compare contexts at the exact boundary before the replacement tuple. This
    # deliberately uses collate_where's canonical native teacher-forced encoding.
    try:
        canonical = collate_where(bundle.processor, episode, bundle.end_fix_id,
            bundle.context_limit, teacher_forcing=True, config=bundle.config,
            image_cache=cache)
        canonical_text = serialize_xyd(gt, bundle.config['where']['end_fix_token'])
        char_position = len(prefix)
        positions = canonical.response_offsets
        before = sum(hi <= char_position for lo, hi in positions)
        straddling = [(i, lo, hi) for i, (lo, hi) in enumerate(positions)
                      if lo < char_position < hi]
        canonical_ids = canonical.inputs['input_ids'][0, :canonical.response_start + before].tolist()
        actual_ids = input_ids[0].tolist()
        first_difference = next((i for i in range(min(len(canonical_ids), len(actual_ids)))
                                 if canonical_ids[i] != actual_ids[i]), None)
        trace['gt_prefix_parity'] = {
            'canonical_response_startswith_prefix': canonical_text.startswith(prefix),
            'canonical_prefix_boundary_char': char_position,
            'canonical_token_boundary_straddles': straddling[:4],
            'token_ids_equal': canonical_ids == actual_ids,
            'canonical_token_count': len(canonical_ids),
            'ss_token_count': len(actual_ids),
            'first_mismatch_index': first_difference,
            'canonical_tail_ids': canonical_ids[-48:],
            'ss_tail_ids': actual_ids[-48:],
        }
    except Exception as exc:
        trace['gt_prefix_parity'] = {'error_type': type(exc).__name__,
                                     'error_message': str(exc)[:500]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(trace, ensure_ascii=False) + '\n')


@torch.no_grad()
def generate_one_fixation(bundle, episode, *, replace_index, cache=None, max_new_tokens=48,
                          debug_global_step=None):
    """Prefix contains GT fixation 1..j-1, but NO GT fixation j or later."""
    tokenizer = bundle.processor.tokenizer
    token = bundle.config['where']['end_fix_token']
    gt = _gt_fixations(episode.query)
    prefix = serialize_xyd(gt[:replace_index - 1], token)[:-1] + ','
    # The native conversation has the full multimodal prompt, but no query GT response.
    prompt = collate_where(bundle.processor, episode, bundle.end_fix_id,
                           bundle.context_limit, teacher_forcing=False,
                           config=bundle.config, image_cache=cache)
    suffix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    input_ids = torch.cat((prompt.inputs['input_ids'], torch.tensor([suffix_ids], dtype=torch.long)), dim=1)
    length = input_ids.shape[1]
    if length + max_new_tokens > bundle.context_limit:
        return None, 'context_overflow'
    inputs = dict(prompt.inputs)
    inputs['input_ids'] = input_ids
    inputs['attention_mask'] = torch.ones_like(input_ids)
    inputs.pop('labels', None)
    inputs = to_model_device(inputs, bundle.model)
    was_training = bundle.model.training
    try:
        bundle.model.eval()
        output = bundle.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([_StopAtEndFix(bundle.end_fix_id, length)]),
        )
    finally:
        bundle.model.train(was_training)
    generated = output[0, length:].tolist()
    if bundle.end_fix_id not in generated:
        status = 'no_end_fix'
        fixation = None
    else:
        truncated = generated[:generated.index(bundle.end_fix_id) + 1]
        text = tokenizer.decode(truncated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        fixation = _parse_generated_fixation(text, token)
        status = 'valid' if fixation is not None else 'invalid_format'
    if os.environ.get('SEMGAZE_SS_TRACE_PATH'):
        _write_ss_debug_trace(bundle, episode, step=debug_global_step,
            replace_index=replace_index, gt=gt, prefix=prefix, prompt=prompt,
            input_ids=input_ids, generated=generated,
            end_fix_id=bundle.end_fix_id, tokenizer=tokenizer, status=status,
            max_new_tokens=max_new_tokens, cache=cache)
    return fixation, status


def _mixed_native(bundle, episode, gt, replace_index, prediction, cache=None):
    mixed = list(gt)
    mixed[replace_index - 1] = prediction
    response = serialize_xyd(mixed, bundle.config['where']['end_fix_token'])
    messages, images = build_where_conversation(episode, bundle.config['where']['end_fix_token'],
        support_prompt_style=bundle.config['where']['support_prompt_style'])
    native = collate_native(bundle.processor, messages, images, response, image_cache=cache)
    if native.inputs['input_ids'].shape[1] > bundle.context_limit:
        raise ValueError('mixed context overflow')
    mask_replaced_fixation(native, response, replace_index,
                           end_fix_token=bundle.config['where']['end_fix_token'])
    if int((native.inputs['labels'][:, :native.response_start] != -100).sum()) != 0:
        raise ValueError('SS accidentally supervised support context')
    return native


@dataclass
class ScheduledSamplingResult:
    episode_losses: torch.Tensor
    applied: torch.Tensor
    diagnostics: dict


def scheduled_sampling_loss(bundle, episodes, *, global_step, weighting=None, image_cache=None):
    """Return differentiable WHERE losses for selected episodes; zeros elsewhere."""
    config = bundle.config['training']['scheduled_sampling']
    device = bundle.input_row.device
    count = len(episodes)
    accepted = []
    failures = Counter()
    p = sampling_probability(config, global_step, bundle.config['training']['total_optimizer_updates'])
    eligible = attempted = changed = 0
    positions = Counter()
    for index, episode in enumerate(episodes):
        T = len(episode.query.x_px)
        if T < 3:
            failures['too_short'] += 1
            continue
        eligible += 1
        rng = _episode_rng(config['seed'], global_step, episode.query.record_id)
        if rng.random() >= p:
            continue
        attempted += 1
        j = rng.randint(2, T - 1)
        prediction, status = generate_one_fixation(
            bundle, episode, replace_index=j, cache=image_cache,
            max_new_tokens=int(config['max_new_tokens']), debug_global_step=global_step)
        if prediction is None:
            failures[status] += 1
            continue
        try:
            sample = _mixed_native(bundle, episode, _gt_fixations(episode.query), j, prediction, cache=image_cache)
        except ValueError as err:
            failures['collation_failed'] += 1
            continue
        accepted.append((index, episode, sample))
        changed += int(prediction != _gt_fixations(episode.query)[j - 1])
        positions[str(j)] += 1
    mask = torch.zeros(count, dtype=torch.bool, device=device)
    if not accepted:
        return ScheduledSamplingResult(torch.zeros(count, device=device), mask,
            {'ss_eligible': eligible, 'ss_attempted': attempted, 'ss_accepted': 0,
             'ss_probability': p, 'ss_changed': 0, 'ss_positions': {},
             'ss_failure_reasons': dict(failures)})
    indices, subepisodes, samples = zip(*accepted)
    mixed_batch = pack_where_batch(bundle.processor, list(subepisodes), list(samples), image_cache)
    inputs = to_model_device({k: v for k, v in mixed_batch.inputs.items() if k != 'labels'}, bundle.model)
    outputs = forward_backbone(bundle.model, inputs, use_cache=False)
    weights = (build_fixation_target_weights(mixed_batch, subepisodes,
        config=weighting, end_fix_token=bundle.config['where']['end_fix_token'])
        if weighting is not None else None)
    selected = compute_selected_causal_nll(outputs.last_hidden_state,
        mixed_batch.inputs['labels'], bundle.model.get_output_embeddings(),
        target_weights=weights)
    index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    mask[index_tensor] = True
    losses = torch.zeros(count, device=device, dtype=selected.episode_losses.dtype).index_copy(
        0, index_tensor, selected.episode_losses)
    return ScheduledSamplingResult(losses, mask,
        {'ss_eligible': eligible, 'ss_attempted': attempted, 'ss_accepted': len(accepted),
         'ss_probability': p, 'ss_changed': changed, 'ss_positions': dict(positions),
         'ss_failure_reasons': dict(failures),
         'ss_loss_mixed_mean': float(selected.episode_losses.detach().mean())})
