"""Oracle-GT textual conditioning with an identical information payload for both styles."""
from dataclasses import dataclass

from semgaze.data.semantic import validate_semantic
from semgaze.semantic.flat.prompt import INSTRUCTION
from semgaze.semantic.flat.target import build_flat_target
from semgaze.where.coordinates import coordinate_bin
from semgaze.where.duration import duration_bin

PROMPT_STYLES = ('origin', 'concise')

CONCISE_INSTRUCTION = '''Given the image, task, ground-truth fixations and gold WHY groups, describe the visual exploration.
Write one concise WHAT about the visible region at each fixation, one task-relevant WHY for each provided gold group, and one HOW for the overall trajectory.

Return only this flat format:
WHAT 1: ...
...
WHAT {N}: ...
WHY 1: ...
...
WHY {M}: ...
HOW: ...

Exactly {N} WHAT, then {M} WHY, then one HOW. Follow chronological fixation order and listed gold-group order. Do not predict, change, merge, split, or print group membership. Keep each value on one line. No JSON, bullets, lists, fences, or extra text.'''


@dataclass(frozen=True)
class BaselinePrompt:
    text: str
    fixation_values: tuple[tuple[int, int, int], ...]
    group_members: tuple[tuple[int, ...], ...]


def build_baseline_prompt(query, style):
    if style not in PROMPT_STYLES:
        raise ValueError(f'invalid semantic baseline prompt style: {style!r}')
    n = len(query.x_px)
    validate_semantic(query.semantic, n, query.record_id)
    values = tuple((coordinate_bin(x, query.image_width), coordinate_bin(y, query.image_height), duration_bin(d))
                   for x, y, d in zip(query.x_px, query.y_px, query.duration_ms))
    groups = tuple(tuple(group.members) for group in query.semantic.why_groups)
    m = len(groups)
    # IMPORTANT: this factual prefix is shared byte-for-byte by origin and concise.
    # Never read semantic.what, semantic.how, or any group.why while constructing it.
    facts = [f'Task:\n{query.task}', f'Fixation count: {n}',
             'Exploration states (ground-truth fixation coordinates and dwell durations):']
    facts.extend(f'Fixation {i}: X={x:02d}, Y={y:02d}, D={d:03d}'
                 for i, (x, y, d) in enumerate(values, 1))
    facts.extend(['', f'Gold WHY group count: {m}', 'Gold WHY groups:'])
    facts.extend(f"Group {i}: fixations {', '.join(map(str, members))}"
                 for i, members in enumerate(groups, 1))
    instruction = INSTRUCTION if style == 'origin' else CONCISE_INSTRUCTION
    return BaselinePrompt('\n'.join(facts) + '\n\n' + instruction.format(N=n, M=m), values, groups)


def semantic_target(query):
    """Exactly the original SemGaze flat supervision, never part of user inputs."""
    return build_flat_target(query.semantic)
