import torch


def insert_states(inputs_embeds, attention_mask, labels, states, boundaries):
    """Interleave one unpadded episode. No token ID is assigned to a state."""
    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        raise ValueError('insertion expects one episode')
    if states.ndim != 2 or states.shape != (len(boundaries), inputs_embeds.shape[-1]):
        raise ValueError('state count/dimension mismatch')
    length = inputs_embeds.shape[1]
    if any(type(p) is not int or not 0 < p < length for p in boundaries) or list(boundaries) != sorted(set(boundaries)):
        raise ValueError('insertion boundaries must be unique and chronological')
    if attention_mask.shape != (1, length) or labels.shape != (1, length):
        raise ValueError('embedding/mask/labels lengths differ')
    chunks, masks, targets, positions, previous = [], [], [], [], 0
    for t, boundary in enumerate(boundaries):
        if labels[0, boundary - 1] != -100 or labels[0, boundary] != -100:
            raise ValueError('state boundary must be inside ignored user context')
        chunks.extend((inputs_embeds[:, previous:boundary], states[t:t + 1].to(inputs_embeds).unsqueeze(0)))
        masks.extend((attention_mask[:, previous:boundary], attention_mask.new_ones((1, 1))))
        targets.extend((labels[:, previous:boundary], labels.new_full((1, 1), -100)))
        positions.append(boundary + t)
        previous = boundary
    chunks.append(inputs_embeds[:, previous:])
    masks.append(attention_mask[:, previous:])
    targets.append(labels[:, previous:])
    return {'inputs_embeds': torch.cat(chunks, 1), 'attention_mask': torch.cat(masks, 1),
            'labels': torch.cat(targets, 1)}, tuple(positions)
