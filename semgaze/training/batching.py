"""Sample one K per physical batch, preserving conditional episode sampling."""
from semgaze.where.collator import collate_where, pack_where_batch, WhereContextOverflowError


def sample_optimizer_batches(bundle, sampler, *, sampling_group_size=None):
    """Build homogeneous-K batches without resampling K on context overflow.

    The benchmark alone may request a larger same-K sampling group, split into
    physical batches, to replay identical episodes across physical batch sizes.
    Production defaults to one independent K draw per physical batch. No queues
    or sampler state beyond the existing checkpointed RNG are introduced.
    """
    t = bundle.config['training']
    size = t['per_device_train_batch_size']
    count = size * t['gradient_accumulation_steps']
    group_size = size if sampling_group_size is None else sampling_group_size
    if type(group_size) is not int or group_size < size or group_size % size or count % group_size:
        raise ValueError('sampling_group_size must be a multiple of physical B and divide the optimizer batch')
    groups, original_episodes, rejected, cache = [], [], 0, {}
    for _ in range(count // group_size):
        sampled, k = [], None
        while len(sampled) < group_size:
            episode = sampler.sample() if k is None else sampler.sample(k=k)
            if k is None:
                k = len(episode.supports)
            if len(episode.supports) != k:
                raise ValueError('sampler returned a different K within a physical batch')
            candidate_cache = dict(cache)
            try:
                native = collate_where(bundle.processor, episode, bundle.end_fix_id,
                    bundle.context_limit, config=bundle.config, image_cache=candidate_cache)
            except WhereContextOverflowError:
                rejected += 1
                if rejected >= t['max_episode_retries']:
                    raise RuntimeError(f'{rejected} complete WHERE episode overflows at K={k}; inspect context feasibility')
                continue
            cache = candidate_cache
            sampled.append((episode, native))
            original_episodes.append(episode)
        if t.get('length_aware_batching', True):
            # Never move episodes across independently sampled K groups.
            sampled.sort(key=lambda pair: (pair[1].inputs['input_ids'].shape[1] // 256,
                                          len(pair[0].query.x_px)))
        groups.append(sampled)
    batches = [pack_where_batch(bundle.processor, [e for e, _ in group[i:i+size]],
        [n for _, n in group[i:i+size]], cache)
        for group in groups for i in range(0, group_size, size)]
    return batches, original_episodes, rejected
