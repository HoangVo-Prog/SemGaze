from dataclasses import dataclass
from semgaze.data.semantic import validate_semantic

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


def build_flat_prompt(query):
    n, m = len(query.x_px), len(query.semantic.why_groups)
    validate_semantic(query.semantic, n, query.record_id)
    text = f'Task:\n{query.task}\n\nExploration states:\n'
    boundaries = []
    for t in range(1, n + 1):
        text += f'Fixation {t}:'
        boundaries.append(len(text))
        text += '\n'
    text += '\nGold WHY groups:\n'
    text += '\n'.join(f"Group {m}: fixations {', '.join(map(str, g.members))}"
                      for m, g in enumerate(query.semantic.why_groups, 1))
    text += '\n\n' + INSTRUCTION.format(N=n, M=m)
    return FlatPrompt(text, tuple(boundaries))
