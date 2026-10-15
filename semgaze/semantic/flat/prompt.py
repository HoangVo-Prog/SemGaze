from dataclasses import dataclass
from semgaze.data.semantic import validate_semantic
from semgaze.where.coordinates import coordinate_bin
from semgaze.where.duration import duration_bin

INSTRUCTION = '''Describe the visual exploration represented by these states.

For every fixation, provide one concise WHAT description of the specific visible object or image region being inspected.

For every provided gold WHY group, provide one concise shared task-relevant search or exploration rationale for that group.

Finally, provide one concise HOW description of the overall visual search or exploration strategy across the complete trajectory.

Return only the following flat line format:
WHAT 1: ...
...
WHAT {N}: ...
WHY 1: ...
...
WHY {M}: ...
HOW: ...

Rules:
- Return exactly {N} WHAT lines, then exactly {M} WHY lines, then exactly one HOW line.
- Keep WHAT indices in fixation order 1..{N}.
- Keep WHY indices in gold-group order 1..{M}.
- Use the provided gold WHY groups; do not predict, rewrite, merge, split, or output group membership.
- Keep every WHAT, WHY, and HOW value on one line.
- Do not output JSON, Python objects, lists, bullets, code fences, or extra explanation.'''


@dataclass(frozen=True)
class FlatPrompt:
    text: str
    boundaries: tuple[int, ...]  # character offsets; never serialized/tokenized markers


def build_flat_prompt(query, *, conditioning="shared_states"):
    n, m = len(query.x_px), len(query.semantic.why_groups)
    validate_semantic(query.semantic, n, query.record_id)
    if conditioning not in ('shared_states', 'explicit_scanpath_text'):
        raise ValueError('unknown semantic conditioning')
    text = f'Task:\n{query.task}\n\n' + ('Exploration states:\n' if conditioning == 'shared_states' else 'Scanpath:\n')
    boundaries = []
    for t in range(1, n + 1):
        text += f'Fixation {t}:'
        if conditioning == 'shared_states':
            boundaries.append(len(text))
        else:
            i = t-1
            x = coordinate_bin(query.x_px[i], query.image_width)
            y = coordinate_bin(query.y_px[i], query.image_height)
            d = duration_bin(query.duration_ms[i])
            text += f' ({x:02d}, {y:02d}, {d:03d})'
        text += '\n'
    text += '\nGold WHY groups:\n'
    text += '\n'.join(f"Group {m}: fixations {', '.join(map(str, g.members))}"
                      for m, g in enumerate(query.semantic.why_groups, 1))
    instruction = INSTRUCTION if conditioning == 'shared_states' else INSTRUCTION.replace(
        'represented by these states', 'described by this scanpath')
    text += '\n\n' + instruction.format(N=n, M=m)
    return FlatPrompt(text, tuple(boundaries))
