"""Response-character grounded fixation weighting and replaced-span masking.

Only query assistant response labels are touched; support/padding remain -100.
All positions are label positions, *not* shifted-logit positions.
"""
import re
import torch
from semgaze.where.serialization import serialize_xyd_record


def fixation_spans(response: str, end_fix_token: str = '<END_FIX>'):
    pattern = re.compile(r'\(\s*\d{2}\s*,\s*\d{2}\s*,\s*\d{3}\s*\)\s*' + re.escape(end_fix_token))
    spans = [(match.start(), match.end()) for match in pattern.finditer(response)]
    if not spans:
        raise ValueError('expected canonical fixation spans in response')
    return tuple(spans)


def _supervised_offsets(native):
    """Map original assistant response offsets even after SS masks some labels.

    collate_native produces consecutive response token positions starting at
    response_start; masking replaces their labels with -100, not their offsets.
    Synthetic legacy callers without response_start retain strict unmasked behavior.
    """
    labels = native.inputs['labels'][0]
    if hasattr(native, 'response_start'):
        start = native.response_start
        stop = start + len(native.response_offsets)
        if start < 0 or stop > len(labels):
            raise ValueError('assistant response bounds do not align')
        return [(start + i, pair) for i, pair in enumerate(native.response_offsets)
                if labels[start + i] != -100]
    indices = (labels != -100).nonzero(as_tuple=True)[0].tolist()
    if len(indices) != len(native.response_offsets):
        raise ValueError('supervision indices/response character offsets disagree')
    return list(zip(indices, native.response_offsets))


def build_fixation_target_weights(batch, episodes, *, config, end_fix_token='<END_FIX>'):
    """Return CPU [B,L] weights, neutral on non-fixation tokens.

    BPE tokens straddling span boundaries keep neutral weight (and are masked in SS
    if they overlap replaced content). This avoids fictitious token/fixation alignment.
    """
    if config is None or not config.get('enabled', False):
        return None
    alpha = float(config['alpha'])
    mode = config['mode']
    threshold = int(config['threshold'])
    if alpha < 0 or mode not in ('late_only', 'linear'):
        raise ValueError('invalid fixation weighting configuration')
    weights = torch.ones(batch.inputs['labels'].shape, dtype=torch.float32)
    for b, (native, ep) in enumerate(zip(batch.samples, episodes)):
        response = serialize_xyd_record(ep.query, end_fix_token)
        spans = fixation_spans(response, end_fix_token)
        if len(spans) != len(ep.query.x_px):
            raise ValueError('fixation count/span mismatch')
        for label_position, (a, z) in _supervised_offsets(native):
            belongs = [t for t, (lo, hi) in enumerate(spans, 1) if lo <= a < z <= hi]
            if len(belongs) != 1:
                continue
            t = belongs[0]
            weights[b, label_position] = (1.0 + alpha if t > threshold else 1.0) if mode == 'late_only' else (1.0 + alpha * (t - 1) / max(len(spans) - 1, 1))
    return weights


def mask_replaced_fixation(native, response: str, replacement_index: int, *, end_fix_token='<END_FIX>'):
    """Mutate an SS *newly collated* NativeBatch, not cached GT samples.

    Conservatively ignore all tokens overlapping the generated fixation, including
    tokens which straddle a boundary. EOS/separators remain supervised when safe.
    """
    spans = fixation_spans(response, end_fix_token)
    if not 1 <= replacement_index <= len(spans):
        raise ValueError('replacement fixation index out of range')
    left, right = spans[replacement_index - 1]
    masked = 0
    for position, (a, z) in _supervised_offsets(native):
        if a < right and z > left:
            native.inputs['labels'][0, position] = -100
            masked += 1
    if not masked:
        raise ValueError('no labels masked for predicted fixation')
    if int((native.inputs['labels'] != -100).sum()) == 0:
        raise ValueError('no remaining GT supervision')
    return masked
