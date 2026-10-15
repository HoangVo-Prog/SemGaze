import torch
from torch.nn.utils.rnn import pad_sequence


def extract_query_states(hidden, input_ids, labels, end_fix_id, counts, *, return_unpadded=False):
    if hidden.shape[:2] != input_ids.shape or labels.shape != input_ids.shape:
        raise ValueError('hidden/input/label shapes do not align')
    if len(counts) != hidden.shape[0]:
        raise ValueError('one fixation count is required per sample')
    selected, positions = [], []
    for b, n in enumerate(counts):
        pos = torch.where((input_ids[b] == end_fix_id) & (labels[b] != -100))[0]
        if len(pos) != n:
            raise ValueError(f'query {b}: expected {n} END_FIX states, found {len(pos)}')
        selected.append(hidden[b, pos.to(hidden.device)])
        positions.append(pos)
    if return_unpadded:
        return selected, None, positions
    padded = pad_sequence(selected, batch_first=True)
    mask = torch.arange(padded.shape[1], device=hidden.device)[None, :] < torch.tensor(counts, device=hidden.device)[:, None]
    return padded, mask, positions


def extract_duration_numeric_states(hidden, samples, counts, *, return_unpadded=False):
    """Read out the last token covering GT duration's final character.

    Positions come from native response token offsets, not digit-count heuristics.
    Reject tokenizers merging the final duration digit with trailing punctuation.
    """
    import re
    if len(samples) != len(counts) or len(samples) != hidden.shape[0]:
        raise ValueError('V1 sample/count batch mismatch')
    selected, positions = [], []
    for b, (native, n) in enumerate(zip(samples, counts)):
        start = native.response_start
        if not native.response_offsets:
            raise ValueError('V1 requires teacher-forced response offsets')
        # The native assistant suffix is the final response; locate its start using
        # the maximum response offset and its EOS. The canonical text is supplied by
        # callers as native.ablation_query_text (one per query, not support).
        text = getattr(native, 'ablation_query_text', None)
        if text is None:
            raise ValueError('native.ablation_query_text is required for V1 readout')
        matches = list(re.finditer(r'\(\s*\d{2}\s*,\s*\d{2}\s*,\s*(\d{3})\s*\)', text))
        if len(matches) != n:
            raise ValueError(f'query {b}: expected {n} GT duration spans, found {len(matches)}')
        chosen = []
        for match in matches:
            final_char = match.end(1)-1
            covering = [(i,a,z) for i,(a,z) in enumerate(native.response_offsets)
                        if a <= final_char < z]
            if len(covering) != 1:
                raise ValueError('cannot uniquely align final duration digit to token')
            i,a,z = covering[0]
            # A BPE token may cover a digit plus adjacent punctuation. This
            # remains the last token COVERING the terminal duration digit; it
            # is not a punctuation-only token. No retokenization is permitted.
            pos = start + i
            if native.inputs['labels'][0, pos] == -100:
                raise ValueError('V1 readout escaped query assistant supervision')
            chosen.append(pos)
        p = torch.tensor(chosen, dtype=torch.long)
        positions.append(p)
        selected.append(hidden[b, p.to(hidden.device)])
    if return_unpadded:
        return selected, None, positions
    padded = pad_sequence(selected, batch_first=True)
    mask = torch.arange(padded.shape[1], device=hidden.device)[None,:] < torch.tensor(counts, device=hidden.device)[:,None]
    return padded, mask, positions
