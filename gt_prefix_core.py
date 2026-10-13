"""Token-exact GT-prefix preparation + dependency-light paired scanpath diagnostics.

This file deliberately changes no training code. The prefix is cut after the
p-th *supervised* END_FIX token in the native teacher-forced WHERE sequence.
"""
from dataclasses import replace
from math import hypot
import re

import torch


def native_gt_prefix(native, *, prefix_count, end_fix_id, tokenizer, target, end_fix_token):
    """Return (native sample with masked labels, exact forced text, prefix token count).

    The assistant prefix is selected from the native tokenization of the GT
    target; no retokenization at assistant-prefix boundaries and no EOS leakage.
    """
    ids = native.inputs['input_ids']
    labels = native.inputs['labels']
    if ids.ndim != 2 or ids.size(0) != 1 or labels.shape != ids.shape:
        raise ValueError('expected a single native teacher-forced WHERE sample')
    end_positions = torch.nonzero(labels[0] == end_fix_id, as_tuple=False).flatten().tolist()
    n = len(end_positions)
    if not 0 < prefix_count < n:
        raise ValueError(f'GT prefix {prefix_count} requires 1 <= prefix < N={n}')
    response_start = native.response_start
    stop = end_positions[prefix_count - 1] + 1
    if stop <= response_start or stop >= ids.size(1):
        raise ValueError('invalid GT fixation prefix offset')
    boundaries = [m.end() for m in re.finditer(re.escape(end_fix_token), target)]
    if len(boundaries) != n:
        raise ValueError('GT END_FIX count does not match supervised token count')
    forced = target[:boundaries[prefix_count - 1]]
    actual = tokenizer.decode(ids[0, response_start:stop].tolist(),
                              skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if actual != forced:
        raise ValueError('native GT prefix tokenization is not losslessly decodable')
    inputs = dict(native.inputs)
    inputs['input_ids'] = ids[:, :stop].clone()
    inputs['attention_mask'] = native.inputs['attention_mask'][:, :stop].clone()
    inputs['labels'] = torch.full_like(inputs['input_ids'], -100)
    sample = replace(native, inputs=inputs, response_length=0, response_offsets=())
    return sample, forced, stop - response_start


def xy_distance(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def suffix_diagnostics(gt, pred, prefix):
    """Unforced-suffix diagnostics. Truncated/invalid generations remain visible.

    A collapse is >=3 consecutive identical XY among generated suffix points.
    GT-prefix points are NEVER counted as predicted or scored as correct.
    """
    if prefix < 0 or prefix >= len(gt):
        raise ValueError('prefix must precede the last GT fixation')
    usable = min(len(gt), len(pred))
    errors = [xy_distance(gt[t], pred[t]) for t in range(prefix, usable)]
    duration = [abs(gt[t][2] - pred[t][2]) for t in range(prefix, usable)]
    repeats = [pred[t][:2] == pred[t-1][:2]
               for t in range(prefix+1, usable)]
    movements = [xy_distance(pred[t], pred[t-1]) for t in range(prefix+1, usable)]
    gt_movements = [xy_distance(gt[t], gt[t-1]) for t in range(prefix+1, usable)]
    first_collapse = None
    for t in range(prefix + 2, usable):
        if pred[t][:2] == pred[t-1][:2] == pred[t-2][:2]:
            first_collapse = t + 1  # human one-based fixation index
            break
    absorbing = (first_collapse is not None and
                 all(pred[t][:2] == pred[first_collapse - 1][:2]
                     for t in range(first_collapse, usable)))
    return {
        'expected_suffix_count': len(gt)-prefix,
        'generated_suffix_count': max(0, usable-prefix),
        'full_suffix': usable == len(gt),
        'mean_xy_error': sum(errors)/len(errors) if errors else None,
        'mean_duration_mae': sum(duration)/len(duration) if duration else None,
        'xy_repeated_transitions': sum(repeats),
        'xy_transition_count': len(repeats),
        'xy_repeat_rate': sum(repeats)/len(repeats) if repeats else None,
        'predicted_suffix_movement': sum(movements)/len(movements) if movements else None,
        'gt_suffix_movement': sum(gt_movements)/len(gt_movements) if gt_movements else None,
        'collapse3': first_collapse is not None,
        'collapse_at_t': first_collapse,
        'absorbing_collapse': bool(absorbing),
    }


def compare_suffix(gt, free, intervention, prefix):
    """Paired errors at *exactly the same available suffix indices*."""
    usable = min(len(gt), len(free), len(intervention))
    before = [xy_distance(gt[t], free[t]) for t in range(prefix, usable)]
    after = [xy_distance(gt[t], intervention[t]) for t in range(prefix, usable)]
    return {'paired_positions': len(before),
            'free_xy_error': sum(before)/len(before) if before else None,
            'prefix_xy_error': sum(after)/len(after) if after else None,
            'delta_xy_error': (sum(after)-sum(before))/len(before) if before else None}
