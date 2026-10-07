"""Epoch qualitative outputs through the two canonical autoregressive paths."""
import json
from pathlib import Path
import random
import time
import torch
from semgaze.data.fewshot import frozen_episode
from semgaze.evaluation.flat import evaluate_flat_episode, evaluate_flat_batch
from semgaze.where.generation import generate_where_batch
from semgaze.evaluation.batching import physical_size
from semgaze.model.visual_cache import InferenceVisualCache
from semgaze.evaluation.where import evaluate_where_episode
from semgaze.evaluation.test import test_mode, test_queries
from semgaze.semantic.flat.target import build_flat_target
from semgaze.where.serialization import serialize_xyd_record
from semgaze.evaluation.progress import (RollingRate, format_eta, format_finish_time,
                                          progress_interval, should_report)
from semgaze.evaluation.metrics_semantic import (build_bertscorer, build_cider_r_scorer,
    score_prediction_semantics)
from semgaze.evaluation.records import (assert_key_sets_equal, implementation_head,
                                         resolve_evaluation_draw_counts, write_metrics_artifact)


def resolve_prediction_settings(config, semantic_max_new_tokens=None):
    settings = config.setdefault('evaluation', {}).setdefault('predictions', {})
    settings.setdefault('train_batches', 1)
    settings.setdefault('test_scope', 'all_unseen')
    if type(settings['train_batches']) is not int or settings['train_batches'] < 0 or settings['test_scope'] not in ('all_unseen', 'none'):
        raise ValueError('predictions require nonnegative train_batches and test_scope all_unseen or none')
    if not settings['train_batches'] and settings['test_scope'] == 'none':
        return settings
    budget = semantic_max_new_tokens if semantic_max_new_tokens is not None else settings.get('semantic_max_new_tokens')
    if type(budget) is not int or budget < 1:
        raise ValueError('declare evaluation.predictions.semantic_max_new_tokens or --semantic-max-new-tokens explicitly')
    settings['semantic_max_new_tokens'] = budget
    return settings


def _prediction_record(bundle, episode, *, epoch, step, split, index, budget, split_manifest_identity, generated=None):
    where = evaluate_where_episode(bundle, episode) if generated is None else generated[0]
    # Canonical semantic inference constructs R from GT WHERE, never predicted
    # fixations. Only the semantic response is autoregressively generated here.
    semantic = evaluate_flat_episode(bundle, episode, generation_budget=budget) if generated is None else generated[1]
    query = episode.query
    where_diagnostics = dict(where)
    semantic_diagnostics = dict(semantic)
    where_text = where_diagnostics.pop('text')
    semantic_text = semantic_diagnostics.pop('text')
    return {'epoch': epoch, 'step': step, 'split': split, 'prediction_index': index,
            'split_manifest_identity': split_manifest_identity,
            'query_id': query.record_id, 'subject': query.subject,
            'stimulus_id': query.stimulus_id, 'task': query.task, 'k': len(episode.supports),
            'K': len(episode.supports), 'support_ids': [s.record_id for s in episode.supports],
            'annotation_width': query.image_width, 'annotation_height': query.image_height,
            'gt_raw_pixels': [[x, y] for x, y in zip(query.x_px, query.y_px)],
            'gt_duration_ms_raw': list(query.duration_ms),
            'WHERE': {'GT': serialize_xyd_record(query, bundle.config['where']['end_fix_token']), 'PRED': where_text},
            'SEMANTIC': {'GT': build_flat_target(query.semantic), 'PRED': semantic_text},
            'where_generation': where_diagnostics, 'semantic_generation': semantic_diagnostics}


def prediction_batches(bundle, episodes, *, budget, cache, path='both', projected_r_cache=None):
    """Yield canonical-order predictions while grouping same-K work internally.

    Index-based restoration permits repeated supplied training episodes.
    """
    settings = bundle.config['test']['prediction']
    window = settings['bucket_window']
    for start in range(0, len(episodes), window):
        block = episodes[start:start+window]
        results = {}
        for k in dict.fromkeys(len(e.supports) for e in block):
            indices = [i for i, e in enumerate(block) if len(e.supports) == k]
            if settings['bucket_by_length']:
                from semgaze.where.collator import collate_where
                indices.sort(key=lambda i: collate_where(bundle.processor, block[i], bundle.end_fix_id,
                    bundle.context_limit, teacher_forcing=False, config=bundle.config,
                    image_cache=cache).inputs['input_ids'].shape[1])
            size = physical_size(settings, k) if str(k) in settings['batch_size_by_k'] or k in settings['batch_size_by_k'] else 1
            for offset in range(0, len(indices), size):
                selected = indices[offset:offset+size]
                group = [block[i] for i in selected]
                if settings['execution'] == 'serial':
                    generated = [(evaluate_where_episode(bundle, e) if path in ('where', 'both') else None,
                                  (evaluate_flat_batch(bundle, [e], generation_budget=budget, cache=cache,
                                      projected_r_cache=projected_r_cache)[0] if projected_r_cache is not None else
                                   evaluate_flat_episode(bundle, e, generation_budget=budget))
                                  if path in ('semantic', 'both') else None) for e in group]
                else:
                    where = generate_where_batch(bundle, group, cache=cache) if path in ('where', 'both') else [None]*len(group)
                    semantic = evaluate_flat_batch(bundle, group, generation_budget=budget, cache=cache,
                                                   projected_r_cache=projected_r_cache) if path in ('semantic', 'both') else [None]*len(group)
                    generated = [({'oracle_length_conditioned': True, **w} if w is not None else None, f)
                                 for w, f in zip(where, semantic)]
                results.update(zip(selected, generated))
        if set(results) != set(range(len(block))):
            raise ValueError('prediction coverage mismatch')
        for i, e in enumerate(block):
            yield e, results[i]


def predict_epoch(bundle, train_batch, train_by_id, test_records, manifest, *,
                  epoch, step, split_manifest_identity, visual_cache=None, projected_r_cache=None):
    """Exactly one supplied train batch; every unseen test query at every K.

    Episodes use same-K physical batches and retain full multi-image
    contexts. No training sampler is consumed. Partial files remain identifiable
    on failure and no successful history event is emitted for incomplete coverage.
    """
    settings = resolve_prediction_settings(bundle.config)
    batch_size = bundle.config['training']['per_device_train_batch_size']
    if len(train_batch) > batch_size * settings['train_batches']:
        raise ValueError('train predictions require the configured number of complete batches')
    if any(ep.query.subject in bundle.config['data']['unseen_subjects'] for ep in train_batch):
        raise ValueError('unseen query in train prediction batch')
    queries = test_queries(train_by_id, test_records, manifest, bundle.config['data']['unseen_subjects']) if settings['test_scope'] == 'all_unseen' else ()
    k_values = bundle.config['evaluation']['k_values']
    draw_counts = resolve_evaluation_draw_counts(bundle.config['evaluation']['draw'], manifest, k_values)
    tag = f'epoch-{epoch:04d}' if bundle.config['evaluation']['strategy'] == 'epoch' else f'step-{step:08d}'
    directory = Path(bundle.output_dir).resolve() / 'predictions' / tag
    directory.mkdir(parents=True, exist_ok=True)
    paths, counts = {}, {}
    prediction_by_k = {}
    prediction_started = time.perf_counter()
    print(f"[EVAL][PRED] starting training predictions | episodes={len(train_batch)}", flush=True)
    if queries:
        print(f"[EVAL][PRED] starting test predictions | queries={len(queries)} | "
              f"K={list(k_values)} | draws={sum(draw_counts.values())} | "
              f"total_episodes={sum(len(queries) * draw_counts[str(k)] for k in k_values)}", flush=True)
    else:
        print('[EVAL][PRED] test predictions disabled', flush=True)
    cache_settings = bundle.config['test']['cache']
    owns_cache = visual_cache is None
    cache = InferenceVisualCache(preprocessing=cache_settings['support_preprocessing'],
        features=cache_settings['frozen_visual_features'], max_entries=cache_settings['max_entries']) if owns_cache else visual_cache
    prediction_total = sum(len(queries) * draw_counts[str(k)] for k in k_values)
    prediction_rate = RollingRate()
    prediction_completed = 0
    python_rng = random.getstate()
    try:
        # Greedy inference normally consumes no RNG. Preserve it explicitly so
        # even runtime-specific generation internals cannot change training draws.
        with torch.random.fork_rng(), test_mode(bundle):
            cache.validate(bundle)
            if projected_r_cache is not None:
                projected_r_cache.validate(bundle, split_manifest_identity=split_manifest_identity,
                                           cycle_id=(epoch, step), manifest=manifest)
            for split in ('train', 'test'):
                path = directory / f'{split}.jsonl'
                partial = path.with_suffix('.jsonl.partial')
                count = 0
                with partial.open('w', encoding='utf-8') as stream:
                    if split == 'train':
                        total = len(train_batch)
                        interval = progress_interval(total)
                        train_rate = RollingRate()
                        for episode, generated in prediction_batches(bundle, train_batch,
                                budget=settings['semantic_max_new_tokens'], cache=cache,
                                projected_r_cache=projected_r_cache):
                            count += 1
                            record = _prediction_record(bundle, episode, epoch=epoch, step=step,
                                split=split, index=count, budget=settings['semantic_max_new_tokens'],
                                split_manifest_identity=split_manifest_identity, generated=generated)
                            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                            stream.flush()
                            train_rate.update(count)
                            if should_report(count, total, interval):
                                eta = train_rate.eta(total - count)
                                print(f"[EVAL][PRED][TRAIN] {count}/{total} | "
                                      f"{100 * count / total:.1f}%", flush=True)
                                print(f"                     ETA={format_eta(eta)} | "
                                      f"finish~{format_finish_time(eta)}", flush=True)
                    else:
                        for k_position, k in enumerate(k_values, 1):
                            k_started = time.perf_counter()
                            interval = progress_interval(len(queries))
                            draw_count = draw_counts[str(k)]
                            draw_times = []
                            k_rate = RollingRate()
                            k_total = len(queries) * draw_count
                            for draw in range(draw_count):
                                draw_started = time.perf_counter()
                                draw_rate = RollingRate()
                                print(f"[EVAL][PRED] K={k} ({k_position}/{len(k_values)}) | "
                                      f"draw={draw + 1}/{draw_count} | "
                                      f"starting | queries={len(queries)}", flush=True)
                                episodes = [frozen_episode(query, train_by_id, manifest, k, draw_id=draw,
                                    unseen_subjects=bundle.config['data']['unseen_subjects']) for query in queries]
                                for query_index, (episode, generated) in enumerate(prediction_batches(bundle, episodes,
                                        budget=settings['semantic_max_new_tokens'], cache=cache,
                                        projected_r_cache=projected_r_cache), 1):
                                    count += 1
                                    record = _prediction_record(bundle, episode, epoch=epoch, step=step,
                                        split=split, index=count, budget=settings['semantic_max_new_tokens'],
                                        split_manifest_identity=split_manifest_identity, generated=generated)
                                    record['draw_id'] = draw
                                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                                    stream.flush()
                                    draw_rate.update(query_index)
                                    k_completed = draw * len(queries) + query_index
                                    k_rate.update(k_completed)
                                    prediction_completed += 1
                                    prediction_rate.update(prediction_completed)
                                    if should_report(query_index, len(queries), interval):
                                        draw_eta = draw_rate.eta(len(queries) - query_index)
                                        k_eta = k_rate.eta(k_total - k_completed)
                                        prediction_eta = prediction_rate.eta(prediction_total - prediction_completed)
                                        finish_eta = prediction_eta if prediction_eta is not None else draw_eta
                                        print(f"[EVAL][PRED] K={k} ({k_position}/{len(k_values)}) | "
                                              f"draw={draw + 1}/{draw_count} | "
                                              f"{query_index}/{len(queries)} | "
                                              f"{100 * query_index / len(queries):.1f}%", flush=True)
                                        print(f"             ETA draw={format_eta(draw_eta)} | "
                                              f"ETA K={format_eta(k_eta)} | "
                                              f"ETA prediction={format_eta(prediction_eta)} | "
                                              f"finish~{format_finish_time(finish_eta)}", flush=True)
                                draw_time = time.perf_counter() - draw_started
                                draw_times.append(draw_time)
                                print(f"[EVAL][PRED] K={k} ({k_position}/{len(k_values)}) | "
                                      f"draw={draw + 1}/{draw_count} | complete", flush=True)
                            k_time = time.perf_counter() - k_started
                            prediction_by_k[str(k)] = {
                                'episodes': len(queries) * draw_count,
                                'draws': draw_count,
                                'complete_query_traversals': draw_count,
                                'draw_time_sec': draw_times,
                                'prediction_k_time_sec': k_time,
                            }
                            print(f"[EVAL][PRED] K={k} ({k_position}/{len(k_values)}) | "
                                  f"draws={draw_count} | complete", flush=True)
                partial.replace(path)
                counts[split], paths[split] = count, str(path)
    finally:
        if owns_cache:
            cache.close()
        random.setstate(python_rng)
    prediction_time = time.perf_counter() - prediction_started
    metrics_artifact = None
    if paths.get('test') and bundle.config.get('evaluation', {}).get('metrics', {}).get('enabled'):
        metrics_artifact = score_prediction_artifact(paths['test'], queries=queries,
            config=bundle.config, output_dir=directory,
            checkpoint=getattr(bundle, 'checkpoint_path', 'integrated'),
            split_manifest_identity=split_manifest_identity)
    print('[EVAL][PRED] complete', flush=True)
    return {'event': 'epoch_predictions', 'epoch': epoch, 'step': step,
            'train_prediction_batches': settings['train_batches'], 'train_prediction_episodes': counts['train'],
            'train_prediction_source': 'final_training_batches_before_evaluation',
            'test_prediction_queries': len(queries),
            'test_prediction_episodes': counts['test'], 'k_values': list(k_values),
            'evaluation_draw': bundle.config['evaluation']['draw'],
            'prediction_files': paths, 'semantic_max_new_tokens': settings['semantic_max_new_tokens'],
            'prediction_time_sec': prediction_time, 'prediction_by_k': prediction_by_k,
            'metrics_artifact': metrics_artifact,
            'test_prediction_traversals': sum(value['complete_query_traversals']
                                              for value in prediction_by_k.values()),
            'where_conditioning': 'oracle_length; no query GT trajectory',
            'semantic_conditioning': 'teacher_forced_GT_XYD states; gold WHY groups'}


def format_prediction_summary(entry):
    paths = entry['prediction_files']
    k_text = ','.join(map(str, entry['k_values']))
    batches = entry['train_prediction_batches']
    batch_word = 'batch' if batches == 1 else 'batches'
    return (f"[EVAL][PRED] epoch {entry['epoch']} | step {entry['step']} | "
            "autoregressive predictions complete:\n"
            f"  train: {batches} {batch_word} ({entry['train_prediction_episodes']} queries) -> {paths['train']}\n"
            f"  test: {entry['test_prediction_queries']} unseen queries x K={k_text} "
            f"({entry['test_prediction_episodes']} episodes) -> {paths['test']}\n"
            '  Each record: WHERE GT/PRED; SEMANTIC GT/PRED. '
            'WHERE is free-running; semantic uses GT WHERE states + gold WHY groups.')


def score_prediction_artifact(path, *, queries, config, output_dir, checkpoint='integrated',
                              split_manifest_identity=''):
    """Rescore frozen semantic predictions without loading or generating with the VLM."""
    metrics_config = config.get('evaluation', {}).get('metrics', {})
    if not metrics_config.get('enabled'):
        return None
    semantic_config = metrics_config.get('semantic', {})
    if not (semantic_config.get('bertscore', {}).get('enabled') or
            semantic_config.get('cider_r', {}).get('enabled')):
        # Probability metrics are produced during the test loss traversal and
        # cannot be reconstructed from frozen prediction text.
        return None
    bertscorer = bert_provenance = cider_r = cider_provenance = None
    if semantic_config.get('bertscore', {}).get('enabled'):
        options = {key: value for key, value in semantic_config['bertscore'].items()
                   if key not in ('enabled', 'batch_size')}
        bertscorer, bert_provenance = build_bertscorer(**options)
    if semantic_config.get('cider_r', {}).get('enabled'):
        cider_r, cider_provenance = build_cider_r_scorer(
            reference_root=Path(__file__).resolve().parents[2] / 'third_party' / 'cider_r',
            n=semantic_config['cider_r'].get('n', 4), k_r=semantic_config['cider_r'].get('k_r', 0.8))
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    refs = {}
    by_id = {q.record_id: q for q in queries}
    grouped = {}
    for row in rows:
        if row.get('split') != 'test':
            continue
        key = (row.get('query_id', row.get('record_id')), int(row.get('K', row.get('k'))), int(row['draw_id']))
        query = by_id.get(key[0])
        if query is None:
            raise ValueError(f'missing query reference for frozen prediction {key}')
        refs[key] = query.semantic
        grouped.setdefault((key[1], key[2]), []).append(row)
    # Frozen prediction artifacts are allowed to be inspected independently;
    # each observed draw must nevertheless contain every eligible test query,
    # exactly once, and joins remain key based.
    expected_query_ids = {q.record_id for q in queries}
    for (k, draw), group in grouped.items():
        assert_key_sets_equal(group)
        observed_query_ids = {row.get('query_id', row.get('record_id')) for row in group}
        if observed_query_ids != expected_query_ids:
            missing = sorted(expected_query_ids - observed_query_ids)
            extra = sorted(observed_query_ids - expected_query_ids)
            raise ValueError(f'incomplete frozen prediction draw K={k}, draw={draw}: '
                             f'missing={missing!r}, extra={extra!r}')
    by_k = {}
    for (k, draw), group in sorted(grouped.items()):
        by_k.setdefault(str(k), {'draws': {}})['draws'][str(draw)] = score_prediction_semantics(
            group, {(row.get('query_id', row.get('record_id')), k, draw): refs[(row.get('query_id', row.get('record_id')), k, draw)] for row in group},
            bertscorer=bertscorer, cider_r=cider_r,
            batch_size=semantic_config.get('bertscore', {}).get('batch_size', 64))
    for block in by_k.values():
        draws = list(block['draws'].values())
        block['draw_mean'] = {name: sum(d[name] for d in draws) / len(draws)
                              for name in draws[0] if name.startswith('eval_') and
                              all(isinstance(d.get(name), (int, float)) for d in draws)} if draws else {}
    artifact = write_metrics_artifact(Path(output_dir) / 'metrics.json', checkpoint=checkpoint,
        config=config.get('_resolved_config_path', 'resolved_config.json'),
        split_manifest_identity=split_manifest_identity, by_k=by_k,
        provenance={'bertscore': bert_provenance, 'cider_r': cider_provenance,
                    'scanpath': 'blocked_by_coordinate_protocol',
                    'probability': 'not_integrated; tokenizer outcome unresolved'},
        implementation_head=implementation_head(), evaluation_draw=config['evaluation']['draw'])
    artifact['metrics_path'] = str(Path(output_dir) / 'metrics.json')
    return artifact

