"""Held-out, unseen-subject loss diagnostics from the existing flat objective.

WHAT/WHY/HOW partition the supervised flat response tokens, not model forwards or
training objectives. No generation-quality metric is defined here.
"""
from contextlib import contextmanager
import math
import time
import torch
from torch.nn import functional as F
from semgaze.data.fewshot import frozen_episode
from semgaze.data.schema import UNSEEN_SUBJECTS
from semgaze.where.forward import forward_where_batch
from semgaze.semantic.flat.forward import forward_flat_batch
from semgaze.evaluation.batching import schedule_batches, episode_id
from semgaze.model.visual_cache import InferenceVisualCache
from semgaze.semantic.flat.forward import prepare_flat_inputs
from semgaze.semantic.flat.target import flatten_text
from semgaze.evaluation.progress import format_duration

EVAL_KEYS = ('test_where', 'test_what', 'test_why', 'test_how', 'test_flat', 'test_total')


@contextmanager
def test_mode(bundle):
    # Preserve even deliberately mixed submodule modes, also when test fails.
    modes = [(module, module.training) for root in (bundle.model, bundle.projector) for module in root.modules()]
    try:
        bundle.model.eval()
        bundle.projector.eval()
        with torch.inference_mode():
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
    return component_metrics(nll, response_offsets, semantic)


def component_metrics(nll, response_offsets, semantic):
    """Partition one existing NLL vector using the original character boundaries."""
    if nll.numel() != len(response_offsets) or not len(response_offsets):
        raise ValueError('flat response offsets and selected NLL differ')
    what_text = '\n'.join(f'WHAT {t}: {flatten_text(w)}' for t, w in enumerate(semantic.what, 1)) + '\n'
    why_text = '\n'.join(f'WHY {m}: {flatten_text(g.why)}' for m, g in enumerate(semantic.why_groups, 1)) + '\n'
    why_start, how_start = len(what_text), len(what_text + why_text)
    section_ids = torch.tensor([0 if start < why_start else 1 if start < how_start else 2
                                for start, end in response_offsets], device=nll.device)
    result = {'test_flat': float(nll.mean()), 'flat_response_tokens': len(response_offsets)}
    for i, component in enumerate(('what', 'why', 'how')):
        selected = nll[section_ids == i]
        if not selected.numel():
            raise ValueError(f'no supervised tokens in {component} section')
        result[f'test_{component}'] = float(selected.mean())
        result[f'{component}_response_tokens'] = selected.numel()
    return result


def serial_reference_losses(bundle, episode, *, batch=None):
    where = forward_where_batch(bundle, [episode], batch=batch, use_cache=False)
    states = bundle.projector(where.states[0])
    inputs, _, native = prepare_flat_inputs(bundle, episode.query, states)
    output = bundle.model(**inputs, output_hidden_states=False, use_cache=False, return_dict=True)
    values = flat_component_nll(output.logits, inputs['labels'], native.response_offsets, episode.query.semantic)
    if not math.isclose(values['test_flat'], float(output.loss), rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError('diagnostic flat token reduction differs from native response loss')
    values['test_flat'] = float(output.loss)
    values['test_where'] = float(where.loss_where)
    t = bundle.config['training']
    values['test_total'] = t['lambda_where'] * values['test_where'] + t['lambda_sem'] * values['test_flat']
    values.update(episode_id=episode_id(episode), K=len(episode.supports),
                  support_ids=[s.record_id for s in episode.supports], state_count=len(where.states[0]),
                  where_length=where.batch.metadata[0]['where_length'],
                  semantic_length=inputs['attention_mask'].shape[1])
    if not all(math.isfinite(values[key]) for key in EVAL_KEYS):
        raise RuntimeError('non-finite test loss')
    return values


def batch_losses(bundle, episodes, *, batch=None, cache=None, profiler=None):
    if not episodes or len({len(e.supports) for e in episodes}) != 1:
        raise ValueError('test requires a nonempty same-K batch')
    where = forward_where_batch(bundle, episodes, batch=batch, visual_cache=cache,
                                profiler=profiler, use_cache=False)
    counts = [len(e.query.x_px) for e in episodes]
    states = bundle.projector(torch.cat(where.states)).split(counts)
    flat, positions, metadata = forward_flat_batch(bundle, [e.query for e in episodes], states,
        where=where, profiler=profiler, response_offsets=True,
        reuse_query_vision=bundle.config.get('test', {}).get('cache', {}).get('frozen_visual_features', True))
    # One transfer per batch. Diagnostic Python loops do no forward or CE work.
    nll = flat.token_nll.cpu()
    where_losses, flat_losses = where.episode_losses.cpu(), flat.episode_losses.cpu()
    values, offset = [], 0
    for b, (ep, meta) in enumerate(zip(episodes, metadata)):
        count = meta['supervised_semantic_tokens']
        row = component_metrics(nll[offset:offset+count], meta['response_offsets'], ep.query.semantic)
        offset += count
        if len(where.states[b]) != counts[b] or len(positions[b]) != counts[b]:
            raise ValueError('query state-count mismatch')
        row.update(test_where=float(where_losses[b]), test_flat=float(flat_losses[b]),
                   episode_id=episode_id(ep), K=len(ep.supports), query_id=ep.query.record_id,
                   support_ids=[s.record_id for s in ep.supports], state_count=counts[b],
                   fixation_count=counts[b], where_length=where.batch.metadata[b]['where_length'],
                   semantic_length=meta['semantic_length'])
        t = bundle.config['training']
        row['test_total'] = t['lambda_where'] * row['test_where'] + t['lambda_sem'] * row['test_flat']
        if not all(math.isfinite(row[key]) for key in EVAL_KEYS):
            raise RuntimeError('non-finite test loss')
        values.append(row)
    if offset != nll.numel():
        raise ValueError('semantic token NLL count mismatch')
    return values


def _episode_losses(bundle, episode):
    return batch_losses(bundle, [episode])[0]


class EpisodeAccumulator:
    def __init__(self):
        self.rows = {}
        self.sums = {key: 0.0 for key in EVAL_KEYS}

    def add(self, key, values):
        if key in self.rows:
            raise ValueError('duplicate episode metric')
        if not all(math.isfinite(values[k]) for k in EVAL_KEYS):
            raise ValueError('non-finite test metric')
        self.rows[key] = values
        for name in EVAL_KEYS:
            self.sums[name] += values[name]

    def finish(self, expected_ids):
        if set(expected_ids) != set(self.rows) or len(expected_ids) != len(self.rows) or not self.rows:
            raise ValueError('metric accumulator count mismatch or missing episode')
        return {key: value / len(self.rows) for key, value in self.sums.items()}


def evaluate_test_episode(bundle, episode):
    with test_mode(bundle):
        return _episode_losses(bundle, episode)


def test_queries(train_by_id, test_records, manifest, unseen_subjects=None):
    """Shared eligibility checks for final test queries."""
    unseen_subjects = set(manifest.get('unseen_subject_ids', UNSEEN_SUBJECTS) if unseen_subjects is None else unseen_subjects)
    queries = tuple(r for r in test_records if r.subject in unseen_subjects)
    if not queries:
        raise ValueError('test split has no unseen-subject queries')
    if len({r.record_id for r in queries}) != len(queries):
        raise ValueError('duplicate test query record')
    if any(r.record_id in train_by_id for r in queries) or (
            {r.stimulus_id for r in queries} & {r.stimulus_id for r in train_by_id.values()}):
        raise ValueError('train/test query leakage')
    if any(r.stimulus_id not in manifest['test_stimulus_ids'] for r in queries):
        raise ValueError('query is outside persisted test membership')
    return queries




def evaluate_test_epoch(bundle, train_by_id, test_records, manifest, *, episode_callback=None, profiler=None):
    """Full unseen-subject test queries at each of 10 frozen draws for K=1/5/10."""
    queries = test_queries(train_by_id, test_records, manifest, bundle.config['data']['unseen_subjects'])
    k_values = bundle.config['evaluation']['k_values']
    settings = bundle.config['test']['loss']
    cache_settings = bundle.config['test']['cache']
    cache = InferenceVisualCache(preprocessing=cache_settings['support_preprocessing'],
        features=cache_settings['frozen_visual_features'], max_entries=cache_settings['max_entries'])
    overall, by_k, k_times, batch_stats = EpisodeAccumulator(), {}, {}, {}
    expected = [f'{k}:{draw}:{q.record_id}' for k in k_values
                for draw in range(len(manifest['support_draws'][str(k)])) for q in queries]
    started = time.perf_counter()
    print(f'[TEST][LOSS] queries={len(queries)} | K={k_values} | episodes={len(expected)}', flush=True)
    try:
        with test_mode(bundle):
            for k in k_values:
                k_started = time.perf_counter()
                subtotal, stats, keys = EpisodeAccumulator(), [], []
                for draw in range(len(manifest['support_draws'][str(k)])):
                    episodes = [frozen_episode(q, train_by_id, manifest, k, draw_id=draw,
                        unseen_subjects=bundle.config['data']['unseen_subjects']) for q in queries]
                    for batch in schedule_batches(bundle, episodes, settings, cache=cache, profiler=profiler):
                        rows = batch_losses(bundle, batch.episodes, batch=batch, cache=cache, profiler=profiler)
                        if len(rows) != len(batch.episodes):
                            raise ValueError('test loss batch lost episodes')
                        for ep, row in zip(batch.episodes, rows):
                            key = f'{k}:{draw}:{ep.query.record_id}'
                            row = dict(row, draw_id=draw, episode_id=key)
                            subtotal.add(key, row)
                            overall.add(key, row)
                            keys.append(key)
                            if episode_callback is not None:
                                episode_callback(row)
                        stats.append({'physical_batch_size': len(rows), **{
                            f'{branch}_{kind}_tokens': (sum(lengths) if kind == 'real' else len(rows)*max(lengths))
                            for branch in ('where', 'semantic')
                            for lengths in ([row.get(f'{branch}_length', 0) for row in rows],)
                            for kind in ('real', 'padded')}})
                    print(f'[TEST][LOSS] K={k} | draw={draw + 1}/10 | queries={len(queries)}', flush=True)
                by_k[str(k)] = subtotal.finish(keys) | {'episodes': len(keys)}
                k_times[str(k)] = time.perf_counter() - k_started
                batch_stats[str(k)] = stats
        cache_stats = cache.statistics()
    finally:
        cache.close()
    return {**overall.finish(expected), 'test_by_k': by_k, 'k_values': list(k_values),
            'test_queries': len(queries), 'test_episodes': len(expected),
            'test_time_sec': time.perf_counter()-started, 'test_k_time_sec': k_times,
            'test_batches': batch_stats, 'test_cache': cache_stats,
            'test_loss_aggregation': 'episode_mean', 'test_kind': 'teacher_forced_response_nll',
            'test_component_definition': 'flat_response_token_sections; prefixes/newlines included; EOS in HOW',
            'generation_quality_metrics': 'not configured: project protocol choices remain unresolved'}
