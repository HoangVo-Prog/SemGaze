"""CPU real-HF contracts for validation scheduling, exact loss and generation."""
from dataclasses import replace
import json
import pytest
import torch
from test_model_path import tiny_bundle, episode
from test_training_throughput import mixed_episode
from test_epoch_validation import validation_data
from semgaze.evaluation import validation
from semgaze.evaluation.batching import schedule_batches, episode_id
from semgaze.model.visual_cache import InferenceVisualCache
from semgaze.where.generation import generate_where, generate_where_batch
from semgaze.evaluation.flat import evaluate_flat_episode, evaluate_flat_batch
from semgaze.evaluation.predictions import predict_epoch, resolve_prediction_settings


def same_k_pair(episode):
    other = mixed_episode(episode)
    return [episode, replace(other, supports=other.supports[:len(episode.supports)])]


@pytest.mark.parametrize('bucket', [False, True])
def test_scheduler_all_k_complete_and_partial(tmp_path, episode, bucket):
    bundle = tiny_bundle(tmp_path, episode)
    train, queries, manifest = validation_data(episode)
    from semgaze.data.fewshot import frozen_episode
    episodes = [frozen_episode(replace(queries[0], record_id=f'q{i}'), train, manifest, k)
                for i in range(5) for k in (1, 5, 10)]
    settings = dict(batch_size_by_k={1: 4, 5: 2, 10: 3}, bucket_by_length=bucket, bucket_window=5)
    batches = list(schedule_batches(bundle, episodes, settings))
    assert all(len({len(e.supports) for e in b.episodes}) == 1 for b in batches)
    assert sorted(episode_id(e) for b in batches for e in b.episodes) == sorted(map(episode_id, episodes))
    assert [len(b.episodes) for b in batches] == [4, 1, 2, 2, 1, 3, 2]
    with pytest.raises(ValueError, match='duplicate'):
        list(schedule_batches(bundle, episodes+[episodes[0]], settings))


def test_episode_accumulator_unequal_batches():
    acc = validation.EpisodeAccumulator()
    for batch in ([1., 2., 3., 4.], [100.]):
        for value in batch:
            acc.add(str(value), {key: value for key in validation.EVAL_KEYS})
    assert acc.finish(['1.0','2.0','3.0','4.0','100.0'])['eval_total'] == 22
    with pytest.raises(ValueError, match='count'):
        acc.finish(['1.0'])
    with pytest.raises(ValueError, match='duplicate'):
        acc.add('1.0', {key: 1 for key in validation.EVAL_KEYS})
    with pytest.raises(ValueError, match='non-finite'):
        acc.add('bad', {key: float('nan') for key in validation.EVAL_KEYS})


def test_batch_size_config_survives_json_and_yaml_keys():
    from semgaze.model.config import resolve_config
    config = resolve_config({'validation': {'loss': {'batch_size_by_k': {1: 4, 5: 2, 10: 1}}}})
    assert config == resolve_config(json.loads(json.dumps(config)))
    changed = resolve_config(config, {'validation': {'loss': {'batch_size_by_k': {1: 8}}}})
    assert changed['validation']['loss']['batch_size_by_k'] == {'1':8,'5':2,'10':1}


def test_synthetic_state_and_image_mapping_isolation(tmp_path, episode):
    from semgaze.state.extractor import extract_query_states
    from semgaze.state.insertion import insert_states_batch
    bundle=tiny_bundle(tmp_path,episode)
    eps=same_k_pair(episode)
    batch=next(schedule_batches(bundle,eps,dict(batch_size_by_k={1:2})))
    ids,labels=batch.inputs['input_ids'],batch.inputs['labels']
    hidden=torch.stack([torch.full((ids.shape[1],32),float(i+1)) for i in range(2)])
    counts=[len(e.query.x_px) for e in eps]
    states,mask,positions=extract_query_states(hidden,ids,labels,bundle.end_fix_id,counts)
    for b,n in enumerate(counts):
        assert states[b,:n].eq(b+1).all() and mask[b].sum()==n
        indices=batch.metadata[b]['support_image_indices']+(batch.metadata[b]['query_image_index'],)
        torch.testing.assert_close(batch.inputs['pixel_values'][list(indices)],batch.samples[b].inputs['pixel_values'])
    # Explicit synthetic insertion distinguishes A/B even with different N.
    embeddings=[torch.zeros(1,n+2,32) for n in counts]
    fused,inserted=insert_states_batch(embeddings,[torch.ones(1,n+2,dtype=torch.long) for n in counts],
        [torch.full((1,n+2),-100) for n in counts],[states[b,:n] for b,n in enumerate(counts)],
        [tuple(range(1,n+1)) for n in counts])
    for b,pos in enumerate(inserted):
        assert fused['inputs_embeds'][b,list(pos)].eq(b+1).all()


def test_cache_rejects_training_and_bounds_storage(tmp_path,episode):
    from semgaze.where.collator import to_model_device
    bundle=tiny_bundle(tmp_path,episode)
    cache=InferenceVisualCache(max_entries=1)
    batch=next(schedule_batches(bundle,[episode],dict(batch_size_by_k={1:1}),cache=cache))
    inputs=to_model_device(batch.inputs,bundle.model)
    with pytest.raises(ValueError,match='inference'):
        cache.fuse(bundle,inputs,batch.samples[0].image_paths)
    with validation.validation_mode(bundle):
        cache.fuse(bundle,inputs,batch.samples[0].image_paths)
    assert len(cache.visual)==len(cache.pixels)==len(cache)==1
    row=next(iter(cache.visual.values()))
    assert row.untyped_storage().nbytes()==row.numel()*row.element_size()


@pytest.mark.parametrize('cache_enabled', [False, True])
def test_native_serial_component_parity_and_reuse_contract(tmp_path, episode, cache_enabled):
    bundle = tiny_bundle(tmp_path, episode)
    episodes = same_k_pair(episode)
    cache = InferenceVisualCache(preprocessing=cache_enabled, features=cache_enabled, max_entries=2)
    settings = dict(batch_size_by_k={len(episode.supports): 2})
    calls, heads = [], []
    with validation.validation_mode(bundle):
        reference = [validation.serial_reference_losses(bundle, e) for e in episodes]
        hook = bundle.model.get_base_model().model.register_forward_pre_hook(lambda *a: calls.append(1))
        head = bundle.model.get_output_embeddings().register_forward_pre_hook(lambda m,a: heads.append(a[0].shape))
        try:
            batch = next(schedule_batches(bundle, episodes, settings, cache=cache))
            actual = validation.batch_losses(bundle, episodes, batch=batch, cache=cache)
        finally:
            hook.remove(); head.remove()
        assert len(calls) == 2 and all(len(shape) == 2 for shape in heads)
        for expected, observed in zip(reference, actual):
            for metric in validation.EVAL_KEYS:
                assert observed[metric] == pytest.approx(expected[metric], abs=3e-6)
            assert observed['state_count'] == expected['state_count']
        again = validation.batch_losses(bundle, episodes, batch=batch, cache=cache)
        for a,b in zip(actual,again):
            assert a['eval_total'] == b['eval_total']
        if cache_enabled:
            assert cache.statistics()['visual']['hits'] > 0
            assert cache.statistics()['preprocessing']['hits'] > 0
    cache.close()
    assert not cache.visual and not cache.pixels and not cache
    assert all(p.grad is None for p in bundle.trainable_parameters())


def test_epoch_b4_plus_b1_is_episode_mean(tmp_path, episode):
    bundle = tiny_bundle(tmp_path, episode)
    train, queries, manifest = validation_data(episode)
    queries = [replace(queries[0], record_id=f'q{i}', stimulus_id=f'val{i}') for i in range(5)]
    manifest['validation_stimulus_ids'] = [q.stimulus_id for q in queries]
    bundle.config['validation']['loss'].update(batch_size_by_k={1: 4, 5: 4, 10: 4}, bucket_by_length=True)
    rows = []
    result = validation.validate_epoch(bundle, train, queries, manifest, episode_callback=rows.append)
    assert len(rows) == result['eval_episodes'] == 15
    for key in validation.EVAL_KEYS:
        assert result[key] == pytest.approx(sum(row[key] for row in rows)/len(rows))
    assert len({r['episode_id'] for r in rows}) == 15


@pytest.mark.parametrize('cache_enabled', [False, True])
def test_generation_matches_serial_with_left_padding_and_budgets(tmp_path, episode, cache_enabled):
    bundle = tiny_bundle(tmp_path, episode)
    episodes = same_k_pair(episode)
    # Add a second same-budget query with a longer textual task to exercise true
    # padded WHERE generation, plus the other fixation count/budget subgroup.
    episodes.append(replace(episode, query=replace(episode.query, task='find a very different object', record_id='third')))
    cache = InferenceVisualCache(preprocessing=cache_enabled, features=cache_enabled)
    calls = []
    generate = bundle.model.generate
    def observed(**kwargs):
        calls.append(kwargs['attention_mask'].shape[0])
        return generate(**kwargs)
    with validation.validation_mode(bundle):
        old_where = [generate_where(bundle, e) for e in episodes]
        old_flat = [evaluate_flat_episode(bundle, e, generation_budget=6) for e in episodes]
        bundle.model.generate = observed
        new_where = generate_where_batch(bundle, episodes, cache=cache)
        new_flat = evaluate_flat_batch(bundle, episodes, generation_budget=6, cache=cache)
    assert max(calls) == 3 and any(b > 1 for b in calls)
    assert new_where == old_where
    assert new_flat == old_flat


def test_batched_epoch_prediction_order_coverage(tmp_path, episode):
    bundle = tiny_bundle(tmp_path, episode)
    resolve_prediction_settings(bundle.config, 3)
    train, queries, manifest = validation_data(episode)
    queries += [replace(queries[0], record_id='second', stimulus_id='second')]
    manifest['validation_stimulus_ids'].append('second')
    bundle.config['validation']['prediction'].update(batch_size_by_k={1: 2, 5: 2, 10: 2}, bucket_by_length=True)
    result = predict_epoch(bundle, [episode], train, queries, manifest, epoch=1, step=1, split_manifest_identity='test')
    rows = [json.loads(line) for line in open(result['prediction_files']['validation'], encoding='utf-8')]
    assert [(r['query_id'],r['k']) for r in rows] == [(q.record_id,k) for k in (1,5,10) for q in queries]
    assert all(r['semantic_generation']['inserted_state_count'] == 4 for r in rows)
