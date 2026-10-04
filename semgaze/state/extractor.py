import torch
from torch.nn.utils.rnn import pad_sequence


def extract_query_states(hidden, input_ids, labels, end_fix_id, counts):
    if hidden.shape[:2] != input_ids.shape or labels.shape != input_ids.shape:
        raise ValueError('hidden/input/label shapes do not align')
    selected, positions = [], []
    for b, n in enumerate(counts):
        pos = torch.where((input_ids[b] == end_fix_id) & (labels[b] != -100))[0]
        if len(pos) != n:
            raise ValueError(f'query {b}: expected {n} END_FIX states, found {len(pos)}')
        selected.append(hidden[b, pos])
        positions.append(pos)
    padded = pad_sequence(selected, batch_first=True)
    mask = torch.arange(padded.shape[1], device=hidden.device)[None, :] < torch.tensor(counts, device=hidden.device)[:, None]
    return padded, mask, positions
