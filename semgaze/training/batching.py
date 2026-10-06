"""Build physical batches without changing query coverage or K sampling."""
from semgaze.where.collator import collate_where, pack_where_batch, WhereContextOverflowError


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
    original_episodes, rejected, cache = [], 0, {}
    # Every call consumes the next query exactly once. Episodes are then
    # bucketed by their already sampled K solely so the collator can batch them.
    sampled_all = []
    target = min(count, sampler.remaining)
    if not target:
        raise StopIteration("query-coverage epoch is complete; start the next epoch explicitly")
    for _ in range(target):
        episode = sampler.sample()
        k = len(episode.supports)
        retries = 0
        while True:
            candidate_cache = dict(cache)
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
    return batches, original_episodes, rejected


def sample_optimizer_batches(bundle, sampler, *, sampling_group_size=None):
    state = sampler.state_dict()
    try:
        return _sample_optimizer_batches(bundle, sampler, sampling_group_size=sampling_group_size)
    except Exception:
        sampler.load_state_dict(state)
        raise
