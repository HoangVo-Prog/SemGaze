COCO_TARGETS = frozenset(('bottle', 'bowl', 'car', 'chair', 'clock', 'cup', 'fork', 'keyboard',
    'knife', 'laptop', 'microwave', 'mouse', 'oven', 'potted plant', 'sink', 'stop sign', 'toilet', 'tv'))


def build_where_prompt(task, n):
    if n <= 0:
        raise ValueError('positive requested fixation count required')
    if task in COCO_TARGETS:
        intro = f'while searching for a {task}.'
        consider = 'Consider the search target, visual saliency, and how attention naturally flows during visual search.'
    else:
        intro = f'while performing the following task: {task}.'
        consider = 'Consider the task, visual saliency, semantic importance, and how attention naturally flows while performing the task.'
    return f'''Analyze this image and predict a human eye movement scanpath {intro}
A scanpath is the temporal sequence of fixation points showing where a person looks over time.
{consider}

Generate a scanpath of exactly {n} fixation points in temporal order, with fixation duration, as a list-style sequence of tuples: (x, y, t)
- x: horizontal position (0-100, 0=left, 100=right)
- y: vertical position (0-100, 0=top, 100=bottom)
- t: fixation duration in milliseconds (0-999)
- Points should be ordered from first fixation to last fixation.
- Append <END_FIX> immediately after every fixation tuple.

Output ONLY the list-style scanpath sequence:
[(51, 46, 221)<END_FIX>, (38, 28, 754)<END_FIX>, ...]'''
