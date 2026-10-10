"""Deterministic scheduling over the shared training WhereBatch representation."""
from semgaze.where.collator import collate_where, pack_where_batch
from contextlib import nullcontext


def episode_id(episode):
    draw = '' if episode.draw_id is None else f'{episode.draw_id}:'
    return f'{len(episode.supports)}:{draw}{episode.query.record_id}'


def physical_size(settings, k, *, batch_size=None, batch_size_by_k=None):
    # Per-K evaluation override > common evaluation size > legacy branch size.
    overrides = {} if batch_size_by_k is None else batch_size_by_k
    size = overrides.get(str(k), overrides.get(k))
    if size is None and str(k) not in overrides and k not in overrides:
        if batch_size is not None:
            size = batch_size
        else:
            mapping = settings['batch_size_by_k']
            size = mapping.get(k, mapping.get(str(k)))
    if type(size) is not int or size < 1:
        raise ValueError(f'missing or invalid test physical batch size for K={k}')
    return 1 if settings.get('execution') == 'serial' else size


def schedule_batches(bundle, episodes, settings, *, cache=None, profiler=None,
                     batch_size=None, batch_size_by_k=None):
    """Bounded windows; exact native lengths; no episode drops or cross-K moves."""
    episodes = tuple(episodes)
    ids = [episode_id(e) for e in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate test episode')
    cache = {} if cache is None else cache
    stage = profiler.stage if profiler is not None else lambda name: nullcontext()
    seen = set()
    for k in dict.fromkeys(len(e.supports) for e in episodes):
        group = [e for e in episodes if len(e.supports) == k]
        size = physical_size(settings, k, batch_size=batch_size, batch_size_by_k=batch_size_by_k)
        window = max(size, settings.get('bucket_window', 64)) if settings.get('bucket_by_length') else size
        for start in range(0, len(group), window):
            with stage('test_collation'):
                prepared = [(e, collate_where(bundle.processor, e, bundle.end_fix_id, bundle.context_limit,
                             config=bundle.config, image_cache=cache)) for e in group[start:start+window]]
            if settings.get('bucket_by_length'):
                prepared.sort(key=lambda pair: (pair[1].inputs['input_ids'].shape[1], len(pair[0].query.x_px)))
            for offset in range(0, len(prepared), size):
                pairs = prepared[offset:offset+size]
                batch = pack_where_batch(bundle.processor, [e for e, _ in pairs], [n for _, n in pairs], cache)
                if len({len(e.supports) for e in batch.episodes}) != 1:
                    raise ValueError('mixed K in test batch')
                for e in batch.episodes:
                    key = episode_id(e)
                    if key in seen:
                        raise ValueError('duplicate scheduled episode')
                    seen.add(key)
                yield batch
    if seen != set(ids):
        raise ValueError('missing test episode')
