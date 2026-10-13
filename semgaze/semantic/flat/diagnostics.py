"""Read-only WHAT / WHY / HOW component CE diagnostics for flat response."""
import torch
from semgaze.semantic.flat.target import flatten_text


def summarize_semantic_component_nll(token_nll, episodes, metadata):
    """Partition already-computed selected token NLL. Does not change objective."""
    counts = [int(row['supervised_semantic_tokens']) for row in metadata]
    if len(counts) != len(episodes) or sum(counts) != token_nll.numel():
        raise ValueError('semantic NLL / episode token alignment mismatch')
    per_section = {'what': [], 'why': [], 'how': []}
    cursor = 0
    for episode, row, n in zip(episodes, metadata, counts):
        offsets = row['response_offsets']
        if len(offsets) != n:
            raise ValueError('semantic response offset count mismatch')
        semantic = episode.query.semantic
        what_text = '\n'.join(f'WHAT {i}: {flatten_text(x)}' for i, x in enumerate(semantic.what, 1)) + '\n'
        why_text = '\n'.join(f'WHY {i}: {flatten_text(g.why)}' for i, g in enumerate(semantic.why_groups, 1)) + '\n'
        why_start, how_start = len(what_text), len(what_text + why_text)
        nll = token_nll[cursor:cursor + n]
        cursor += n
        section = torch.tensor([0 if a < why_start else 1 if a < how_start else 2
                                for a, _ in offsets], dtype=torch.long, device=nll.device)
        for section_id, name in enumerate(('what', 'why', 'how')):
            chosen = nll[section == section_id]
            if not chosen.numel():
                raise ValueError(f'empty semantic {name.upper()} supervision')
            per_section[name].append(chosen.detach().mean())
    return {f'loss_{name}': torch.stack(values).mean() for name, values in per_section.items()}
