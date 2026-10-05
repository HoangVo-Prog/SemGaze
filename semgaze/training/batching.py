"""Group only already-sampled episodes within a single optimizer minibatch."""
from semgaze.where.collator import collate_where, pack_where_batch, WhereContextOverflowError


def sample_optimizer_batches(bundle, sampler):
    t = bundle.config['training']
    count = t['per_device_train_batch_size'] * t['gradient_accumulation_steps']
    sampled, rejected, cache = [], 0, {}
    while len(sampled) < count:
        episode = sampler.sample()
        candidate_cache = dict(cache)
        try:
            native = collate_where(bundle.processor, episode, bundle.end_fix_id,
                bundle.context_limit, config=bundle.config, image_cache=candidate_cache)
        except WhereContextOverflowError:
            rejected += 1
            if rejected >= t['max_episode_retries']:
                raise RuntimeError(f'{rejected} complete WHERE episode overflows; inspect context feasibility')
            continue
        cache = candidate_cache
        sampled.append((episode, native))
    original_episodes = [e for e, _ in sampled]
    if t.get('length_aware_batching', True):
        # Stable K-aware/coarse length grouping. Never crosses an optimizer step,
        # drops a sample, resamples K, or changes support order.
        sampled.sort(key=lambda pair: (len(pair[0].supports), pair[1].inputs['input_ids'].shape[1] // 256,
                                      len(pair[0].query.x_px)))
    size = t['per_device_train_batch_size']
    batches = [pack_where_batch(bundle.processor, [e for e, _ in sampled[i:i+size]],
        [n for _, n in sampled[i:i+size]], cache) for i in range(0, count, size)]
    return batches, original_episodes, rejected
