"""Build physical batches without changing query coverage or K sampling."""
from semgaze.where.collator import collate_where, pack_where_batch, WhereContextOverflowError
from semgaze.model.visual_cache import InferenceVisualCache


def _sample_optimizer_batches(bundle, sampler, *, sampling_group_size=None):
    """Bucket one optimizer window by independently sampled K; retain every tail.

    An optimizer window contains at most B*G episodes. Mixed K can require more
    physical forwards than G; each is weighted by its actual episode count.
    sampling_group_size is a benchmark compatibility argument only.
    """
    t = bundle.config['training']
    size = t['per_device_train_batch_size']
    count = size * t['gradient_accumulation_steps']
    group_size = size if sampling_group_size is None else sampling_group_size
    if type(group_size) is not int or group_size < size or group_size % size or count % group_size:
        raise ValueError('sampling_group_size must be a multiple of physical B and divide the optimizer batch')
    original_episodes, rejected = [], 0
    # A single bounded native cache now handles support AND query processing
    # throughout this optimizer window; false retains the original dict path.
    cache = (InferenceVisualCache(preprocessing=True, features=False,
             max_entries=t.get('preprocessing_cache_max_entries', 32))
             if t.get('cache_preprocessed_images', False) else {})
    # Every call consumes the next query exactly once. Episodes are then
    # bucketed by their already sampled K solely so the collator can batch them.
    sampled_all = []
    target = min(count, sampler.remaining)
    if not target:
        raise StopIteration("query-coverage epoch is complete; start the next epoch explicitly")
    policy = bundle.config.get('data', {}).get('fewshot', {}).get(
        'k_sampling_strategy', t.get('k_sampling_strategy', 'per_episode'))
    if policy not in ('per_episode', 'per_batch'):
        raise ValueError('unsupported K sampling strategy')
    batch_k = None
    for episode_index in range(target):
        if policy == 'per_batch':
            # Each contiguous physical group makes one K draw, including tails.
            if episode_index % size == 0:
                batch_k = sampler.draw_k()
            query = sampler.next_query()
            episode = sampler.sample_for_query(query, k=batch_k)
        else:
            episode = sampler.sample()
        k = len(episode.supports)
        retries = 0
        while True:
            # Native preprocessing cache is deterministic, so retaining a
            # processed image from a rejected support realization is harmless.
            # Legacy dict caches keep transactional behavior.
            candidate_cache = cache if isinstance(cache, InferenceVisualCache) else dict(cache)
            try:
                native = collate_where(bundle.processor, episode, bundle.end_fix_id,
                    bundle.context_limit, config=bundle.config, image_cache=candidate_cache)
            except WhereContextOverflowError:
                rejected += 1
                retries += 1
                if retries >= t['max_episode_retries']:
                    raise RuntimeError(f'query {episode.query.record_id}, K_train={k}: no feasible context found after {retries} support realizations; query not trained')
                episode = sampler.sample_for_query(episode.query, k=k)
                continue
            cache = candidate_cache
            sampled_all.append((episode, native))
            original_episodes.append(episode)
            break
    groups = {}
    if policy == 'per_batch':
        # Do not regroup same-K batches across physical boundaries. Group order
        # must remain aligned with its one K draw (including optimizer tails).
        for start in range(0, len(sampled_all), size):
            groups[start] = sampled_all[start:start + size]
    else:
        for pair in sampled_all:
            groups.setdefault(len(pair[0].supports), []).append(pair)
    batches = []
    for group in groups.values():
        if t.get('length_aware_batching', True):
            group.sort(key=lambda pair: (pair[1].inputs['input_ids'].shape[1] // 256,
                                         len(pair[0].query.x_px)))
        for i in range(0, len(group), size):
            part = group[i:i + size]
            batches.append(pack_where_batch(bundle.processor, [e for e, _ in part],
                [n for _, n in part], cache))
    # `cache` has been used by native WHERE collation throughout this
    # window; it already contains reusable support and query CPU pixel rows.
    return batches, original_episodes, rejected


def sample_optimizer_batches(bundle, sampler, *, sampling_group_size=None):
    state = sampler.state_dict()
    try:
        return _sample_optimizer_batches(bundle, sampler, sampling_group_size=sampling_group_size)
    except Exception:
        sampler.load_state_dict(state)
        raise
