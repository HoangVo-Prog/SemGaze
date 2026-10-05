"""Held-out, seen-subject loss diagnostics from the existing flat objective.

WHAT/WHY/HOW partition the supervised flat response tokens, not model forwards or
training objectives. No generation-quality metric is defined here.
"""
from contextlib import contextmanager
import math
import torch
from torch.nn import functional as F
from semgaze.data.fewshot import frozen_episode
from semgaze.data.schema import UNSEEN_SUBJECTS
from semgaze.where.forward import forward_where
from semgaze.semantic.flat.forward import prepare_flat_inputs
from semgaze.semantic.flat.target import flatten_text

EVAL_KEYS = ('eval_where', 'eval_what', 'eval_why', 'eval_how', 'eval_flat', 'eval_total')


@contextmanager
def validation_mode(bundle):
    # Preserve even deliberately mixed submodule modes, also when validation fails.
    modes = [(module, module.training) for root in (bundle.model, bundle.projector) for module in root.modules()]
    try:
        bundle.model.eval()
        bundle.projector.eval()
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes:
            module.training = mode


def flat_component_nll(logits, labels, response_offsets, semantic):
    """Slice the standard shifted response NLL without changing tokenization.

    Prefixes and intervening newlines belong to their line's section; native EOS
    belongs to HOW. A BPE token crossing a section boundary belongs to the section
    containing its first character. Every supervised token is counted exactly once.
    """
    if logits.shape[:2] != labels.shape or labels.shape[0] != 1:
        raise ValueError('expected aligned logits/labels for one flat response')
    valid = labels[:, 1:] != -100
    targets = labels[:, 1:][valid]
    if targets.numel() != len(response_offsets) or not len(response_offsets):
        raise ValueError('flat response offsets and shifted supervised tokens differ')
    # Select supervised positions before casting to float32, avoiding a second
    # full-context float32 logits buffer on the 8B model.
    nll = F.cross_entropy(logits[:, :-1][valid].float(), targets, reduction='none')
    what_text = '\n'.join(f'WHAT {t}: {flatten_text(w)}' for t, w in enumerate(semantic.what, 1)) + '\n'
    why_text = '\n'.join(f'WHY {m}: {flatten_text(g.why)}' for m, g in enumerate(semantic.why_groups, 1)) + '\n'
    why_start, how_start = len(what_text), len(what_text + why_text)
    section_ids = torch.tensor([0 if start < why_start else 1 if start < how_start else 2
                                for start, end in response_offsets], device=nll.device)
    result = {'eval_flat': float(nll.mean()), 'flat_response_tokens': len(response_offsets)}
    for i, component in enumerate(('what', 'why', 'how')):
        selected = nll[section_ids == i]
        if not selected.numel():
            raise ValueError(f'no supervised tokens in {component} section')
        result[f'eval_{component}'] = float(selected.mean())
        result[f'{component}_response_tokens'] = selected.numel()
    return result


def _episode_losses(bundle, episode):
    if episode.query.subject in bundle.config['data']['unseen_subjects']:
        raise ValueError('unseen subjects cannot contribute epoch validation/model selection')
    where = forward_where(bundle, episode)
    states = bundle.projector(where.states)
    inputs, _, native = prepare_flat_inputs(bundle, episode.query, states)
    output = bundle.model(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
    values = flat_component_nll(output.logits, inputs['labels'], native.response_offsets, episode.query.semantic)
    if not math.isclose(values['eval_flat'], float(output.loss), rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError('diagnostic flat token reduction differs from native response loss')
    values['eval_flat'] = float(output.loss)
    values['eval_where'] = float(where.loss_where)
    t = bundle.config['training']
    values['eval_total'] = t['lambda_where'] * values['eval_where'] + t['lambda_sem'] * values['eval_flat']
    if not all(math.isfinite(values[key]) for key in EVAL_KEYS):
        raise RuntimeError('non-finite validation loss')
    return values


def evaluate_validation_episode(bundle, episode):
    with validation_mode(bundle):
        return _episode_losses(bundle, episode)


def validation_queries(train_by_id, validation_records, manifest, unseen_subjects=None):
    """Shared eligibility checks for loss validation and epoch predictions."""
    unseen_subjects = set(manifest.get('unseen_subject_ids', UNSEEN_SUBJECTS) if unseen_subjects is None else unseen_subjects)
    queries = tuple(r for r in validation_records if r.subject not in unseen_subjects)
    seen = set(manifest['seen_subject_ids'])
    if not queries or {r.subject for r in queries} != seen or seen & unseen_subjects:
        raise ValueError('validation must cover every manifest seen subject and exclude unseen subjects')
    if len({r.record_id for r in queries}) != len(queries):
        raise ValueError('duplicate validation query record')
    if any(r.record_id in train_by_id for r in queries) or (
            {r.stimulus_id for r in queries} & {r.stimulus_id for r in train_by_id.values()}):
        raise ValueError('train/validation query leakage')
    if any(r.stimulus_id not in manifest['validation_stimulus_ids'] for r in queries):
        raise ValueError('query is outside persisted validation membership')
    return queries


def validate_epoch(bundle, train_by_id, validation_records, manifest):
    """All seen validation queries, each with frozen train supports at K=1/5/10.

    Mean per-episode response losses match the trainer's episode mean. Per-K
    summaries remain explicit; this is monitoring, not a model-selection metric.
    """
    queries = validation_queries(train_by_id, validation_records, manifest, bundle.config['data']['unseen_subjects'])
    k_values = bundle.config['evaluation']['k_values']
    sums = {key: 0.0 for key in EVAL_KEYS}
    by_k = {}
    with validation_mode(bundle):
        for k in k_values:
            subtotal = {key: 0.0 for key in EVAL_KEYS}
            for query in queries:
                episode = frozen_episode(query, train_by_id, manifest, k, unseen_subjects=bundle.config['data']['unseen_subjects'])
                values = _episode_losses(bundle, episode)
                for key in EVAL_KEYS:
                    subtotal[key] += values[key]
                    sums[key] += values[key]
            by_k[str(k)] = {key: value / len(queries) for key, value in subtotal.items()}
            by_k[str(k)]['episodes'] = len(queries)
    count = len(queries) * len(k_values)
    return {**{key: value / count for key, value in sums.items()}, 'eval_by_k': by_k,
            'k_values': list(k_values), 'eval_queries': len(queries), 'eval_episodes': count,
            'eval_loss_aggregation': 'episode_mean', 'eval_kind': 'teacher_forced_response_nll',
            'eval_component_definition': 'flat_response_token_sections; prefixes/newlines included; EOS in HOW',
            'generation_quality_metrics': 'not configured: project protocol choices remain unresolved'}
