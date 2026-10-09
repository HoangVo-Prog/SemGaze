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
from semgaze.where.serialization import parse_xyd_output
from semgaze.evaluation.metrics_scanpath import (
    CoordinateProtocolError, DatasetIntegrityError, aggregate_scanpath_draw, score_scanpath_pair,
)
from semgaze.evaluation.progress import RollingRate, format_eta, progress_interval, should_report
from semgaze.evaluation.metrics_semantic import (build_bertscorer, build_cider_r_scorer,
    aggregate_semantic_metrics, score_prediction_semantics)
from semgaze.evaluation.records import (assert_key_sets_equal, implementation_head,
                                         resolve_evaluation_draw_counts,
                                         resolve_semantic_draw_counts, write_metrics_artifact)


def resolve_prediction_settings(config, semantic_max_new_tokens=None):
    settings = config.setdefault('evaluation', {}).setdefault('predictions', {})
    settings.setdefault('train_batches', 1)
    settings.setdefault('test_scope', 'all_unseen')
    settings.setdefault('where', True)
    settings.setdefault('semantic', True)
    for branch in ('where', 'semantic'):
        if type(settings[branch]) is not bool:
            raise ValueError(f'evaluation.predictions.{branch} must be boolean')
    if type(settings['train_batches']) is not int or settings['train_batches'] < 0 or settings['test_scope'] not in ('all_unseen', 'none'):
        raise ValueError('predictions require nonnegative train_batches and test_scope all_unseen or none')
    if (not settings['where'] and not settings['semantic']) or (
            not settings['train_batches'] and settings['test_scope'] == 'none'):
        return settings
    if not settings['semantic']:
        # WHERE-only generation needs no semantic token budget.
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
    record = {'epoch': epoch, 'step': step, 'split': split, 'prediction_index': index,
            'split_manifest_identity': split_manifest_identity,
            'query_id': query.record_id, 'subject': query.subject,
            'dataset': getattr(query, 'dataset', 'COCO-Search18'),
            'stimulus_id': query.stimulus_id, 'task': query.task, 'k': len(episode.supports),
            'K': len(episode.supports), 'support_ids': [s.record_id for s in episode.supports],
            'annotation_width': query.image_width, 'annotation_height': query.image_height,
            'gt_raw_pixels': [[x, y] for x, y in zip(query.x_px, query.y_px)],
            'gt_duration_ms_raw': list(query.duration_ms)}
    if where is not None:
        where_diagnostics = dict(where)
        where_text = where_diagnostics.pop('text')
        record.update({
            'WHERE': {'GT': serialize_xyd_record(query, bundle.config['where']['end_fix_token']), 'PRED': where_text},
            'where_generation': where_diagnostics,
        })
    # A WHERE record can outlive the single semantic traversal when semantic
    # draws are disabled.  Do not serialize a placeholder or a duplicate
    # semantic prediction for those later WHERE draws.
    if semantic is not None:
        semantic_diagnostics = dict(semantic)
        semantic_text = semantic_diagnostics.pop('text')
        record.update({
            'SEMANTIC': {'GT': build_flat_target(query.semantic), 'PRED': semantic_text},
            'semantic_generation': semantic_diagnostics,
        })
    return record


def prediction_batches(bundle, episodes, *, budget, cache, path='both', projected_r_cache=None,
                       on_window_ready=None):
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
            precomputed = {}
            if settings['bucket_by_length']:
                from semgaze.where.collator import collate_where

                def sort_length(i):
                    sample = collate_where(bundle.processor, block[i], bundle.end_fix_id,
                        bundle.context_limit, teacher_forcing=False, config=bundle.config,
                        image_cache=cache)
                    if path in ('where', 'both') and settings['execution'] != 'serial':
                        precomputed[i] = sample
                    return sample.inputs['input_ids'].shape[1]

                indices.sort(key=sort_length)
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
                    if path in ('where', 'both'):
                        where = (generate_where_batch(bundle, group, cache=cache,
                            precomputed_samples=[precomputed[i] for i in selected])
                            if precomputed else generate_where_batch(bundle, group, cache=cache))
                    else:
                        where = [None] * len(group)
                    semantic = evaluate_flat_batch(bundle, group, generation_budget=budget, cache=cache,
                                                   projected_r_cache=projected_r_cache) if path in ('semantic', 'both') else [None]*len(group)
                    generated = [({'oracle_length_conditioned': True, **w} if w is not None else None, f)
                                 for w, f in zip(where, semantic)]
                results.update(zip(selected, generated))
        if set(results) != set(range(len(block))):
            raise ValueError('prediction coverage mismatch')
        # The entire inference window is ready BEFORE its records are yielded.
        # A yielded-record clock measures disk/serialization bursts, not inference.
        if on_window_ready is not None:
            on_window_ready(len(block))
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
    if any(ep.query.subject in set(manifest['unseen_subject_ids']) for ep in train_batch):
        raise ValueError('unseen query in train prediction batch')
    queries = test_queries(train_by_id, test_records, manifest) if settings['test_scope'] == 'all_unseen' else ()
    k_values = bundle.config['evaluation']['k_values']
    where_enabled, semantic_enabled = settings['where'], settings['semantic']
    if not (where_enabled or semantic_enabled):
        raise ValueError('predict_epoch called with both prediction branches disabled')
    where_draw_counts = resolve_evaluation_draw_counts(bundle.config['evaluation']['draw'], manifest, k_values)
    semantic_use_draws = bundle.config.get('evaluation', {}).get('semantic_use_draws', True)
    semantic_draw_counts = (resolve_semantic_draw_counts(where_draw_counts, semantic_use_draws)
                            if semantic_enabled else {str(k): 0 for k in k_values})
    active_draw_counts = (where_draw_counts if where_enabled else semantic_draw_counts)
    train_path = 'both' if where_enabled and semantic_enabled else ('where' if where_enabled else 'semantic')
    tag = f'epoch-{epoch:04d}' if bundle.config['evaluation']['strategy'] == 'epoch' else f'step-{step:08d}'
    directory = Path(bundle.output_dir).resolve() / 'predictions' / tag
    directory.mkdir(parents=True, exist_ok=True)
    paths, counts = {}, {}
    prediction_by_k = {}
    prediction_started = time.perf_counter()
    print(f"[EVAL][PRED] starting training predictions | episodes={len(train_batch)}", flush=True)
    if queries:
        total = sum(len(queries) * active_draw_counts[str(k)] for k in k_values)
        semantic_total = sum(len(queries) * semantic_draw_counts[str(k)] for k in k_values)
        semantic_draw_label = (str(sum(semantic_draw_counts.values())) if semantic_enabled
                               else 'OFF')
        where_draw_label = str(sum(where_draw_counts.values())) if where_enabled else 'OFF'
        print(f"[EVAL][PRED] starting test predictions | queries={len(queries)} | "
              f"K={list(k_values)} | WHERE draws={where_draw_label} | "
              f"semantic draws={semantic_draw_label} | total_episodes={total} | "
              f"semantic_episodes={semantic_total}", flush=True)
    else:
        print('[EVAL][PRED] test predictions disabled', flush=True)
    cache_settings = bundle.config['test']['cache']
    owns_cache = visual_cache is None
    cache = InferenceVisualCache(preprocessing=cache_settings['support_preprocessing'],
        features=cache_settings['frozen_visual_features'], max_entries=cache_settings['max_entries']) if owns_cache else visual_cache
    # The combined file has one record per active prediction episode.
    # Semantic-only runs do not perform WHERE generation.
    prediction_total = sum(len(queries) * active_draw_counts[str(k)] for k in k_values)
    prediction_rate = RollingRate(min_observations=2)
    prediction_completed = 0  # records written, for display/coverage
    prediction_ready = 0      # records inferred in completed windows, for ETA

    def test_window_ready(window_size):
        nonlocal prediction_ready
        prediction_ready += window_size
        if prediction_ready > prediction_total:
            raise RuntimeError('prediction window readiness exceeded planned evaluation coverage')
        prediction_rate.update(prediction_ready)

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
                        train_rate = RollingRate(min_observations=2)
                        train_ready = 0

                        def train_window_ready(window_size):
                            nonlocal train_ready
                            train_ready += window_size
                            if train_ready > total:
                                raise RuntimeError('train prediction window exceeds requested coverage')
                            train_rate.update(train_ready)

                        for episode, generated in prediction_batches(bundle, train_batch,
                                budget=settings['semantic_max_new_tokens'], cache=cache,
                                path=train_path, projected_r_cache=projected_r_cache,
                                on_window_ready=train_window_ready):
                            count += 1
                            record = _prediction_record(bundle, episode, epoch=epoch, step=step,
                                split=split, index=count, budget=settings['semantic_max_new_tokens'],
                                split_manifest_identity=split_manifest_identity, generated=generated)
                            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                            stream.flush()
                            if should_report(count, total, interval):
                                eta = train_rate.eta(total - train_ready)
                                print(f"[EVAL][PRED][TRAIN] {count}/{total} "
                                      f"({100 * count / total:.1f}%) | "
                                      f"ETA train predictions={format_eta(eta)}", flush=True)
                    else:
                        # Exclude earlier train generation from the test ETA.
                        prediction_rate.reset()
                        prediction_ready = 0
                        for k_position, k in enumerate(k_values, 1):
                            k_started = time.perf_counter()
                            interval = progress_interval(len(queries))
                            draw_count = active_draw_counts[str(k)]
                            where_draw_count = where_draw_counts[str(k)] if where_enabled else 0
                            semantic_draw_count = semantic_draw_counts[str(k)]
                            draw_times = []
                            semantic_label = str(semantic_draw_count) if semantic_enabled else 'OFF'
                            where_label = str(where_draw_count) if where_enabled else 'OFF'
                            print(f"[EVAL][PRED] K={k} ({k_position}/{len(k_values)}) | "
                                  f"WHERE draws={where_label} | semantic draws={semantic_label}", flush=True)
                            for draw in range(draw_count):
                                draw_started = time.perf_counter()
                                print(f"[EVAL][PRED] K={k} ({k_position}/{len(k_values)}) | "
                                      f"draw={draw + 1}/{draw_count} | "
                                      f"starting | queries={len(queries)}", flush=True)
                                episodes = [frozen_episode(query, train_by_id, manifest, k, draw_id=draw,
                                    unseen_subjects=manifest['unseen_subject_ids']) for query in queries]
                                prediction_path = (
                                    'both' if where_enabled and draw < semantic_draw_count else
                                    'where' if where_enabled else 'semantic')
                                for query_index, (episode, generated) in enumerate(prediction_batches(bundle, episodes,
                                        budget=settings['semantic_max_new_tokens'], cache=cache, path=prediction_path,
                                        projected_r_cache=projected_r_cache,
                                        on_window_ready=test_window_ready), 1):
                                    count += 1
                                    record = _prediction_record(bundle, episode, epoch=epoch, step=step,
                                        split=split, index=count, budget=settings['semantic_max_new_tokens'],
                                        split_manifest_identity=split_manifest_identity, generated=generated)
                                    record['draw_id'] = draw
                                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                                    stream.flush()
                                    prediction_completed += 1
                                    if should_report(query_index, len(queries), interval):
                                        # ETA is sampled when each *inference window* finishes.
                                        # Writing the window's records cannot fake a high rate.
                                        prediction_eta = prediction_rate.eta(
                                            prediction_total - prediction_ready)
                                        print(f"[EVAL][PRED] {prediction_completed}/{prediction_total} "
                                              f"({100 * prediction_completed / prediction_total:.1f}%) | "
                                              f"K={k} ({k_position}/{len(k_values)}) "
                                              f"draw={draw + 1}/{draw_count} [{query_index}/{len(queries)}] | "
                                              f"path={prediction_path} | "
                                              f"ETA test predictions={format_eta(prediction_eta)}", flush=True)
                                draw_time = time.perf_counter() - draw_started
                                draw_times.append(draw_time)
                                print(f"[EVAL][PRED] K={k} ({k_position}/{len(k_values)}) | "
                                      f"draw={draw + 1}/{draw_count} | complete", flush=True)
                            k_time = time.perf_counter() - k_started
                            prediction_by_k[str(k)] = {
                                'episodes': len(queries) * draw_count,
                                'draws': draw_count,
                                'where_draws': where_draw_count,
                                'semantic_draws': semantic_draw_count,
                                'semantic_episodes': len(queries) * semantic_draw_count,
                                'complete_query_traversals': draw_count,
                                'semantic_query_traversals': semantic_draw_count,
                                'draw_time_sec': draw_times,
                                'prediction_k_time_sec': k_time,
                            }
                            print(f"[EVAL][PRED] K={k} ({k_position}/{len(k_values)}) | "
                                  f"WHERE draws={where_label} | semantic draws={semantic_label} | complete", flush=True)
                partial.replace(path)
                counts[split], paths[split] = count, str(path)
    finally:
        if owns_cache:
            cache.close()
        random.setstate(python_rng)
    prediction_time = time.perf_counter() - prediction_started
    metrics_artifact = None
    if queries and paths.get('test') and bundle.config.get('evaluation', {}).get('metrics', {}).get('enabled'):
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
            'test_queries_by_dataset': {
                ds: sum(getattr(q, 'dataset', 'COCO-Search18') == ds for q in queries)
                for ds in sorted({getattr(q, 'dataset', 'COCO-Search18') for q in queries})},
            'evaluation_draw': bundle.config['evaluation']['draw'],
            'semantic_use_draws': semantic_use_draws if semantic_enabled else False,
            'prediction_branches': {'where': where_enabled, 'semantic': semantic_enabled},
            'prediction_files': paths, 'semantic_max_new_tokens': settings['semantic_max_new_tokens'],
            'prediction_time_sec': prediction_time, 'prediction_by_k': prediction_by_k,
            'metrics_artifact': metrics_artifact,
            'test_prediction_traversals': sum(value['complete_query_traversals']
                                              for value in prediction_by_k.values()),
            'test_semantic_prediction_episodes': sum(value['semantic_episodes']
                                                      for value in prediction_by_k.values()),
            'test_semantic_prediction_traversals': sum(value['semantic_query_traversals']
                                                        for value in prediction_by_k.values()),
            **({'where_conditioning': 'oracle_length; no query GT trajectory'} if where_enabled else {}),
            **({'semantic_conditioning': 'teacher_forced_GT_XYD states; gold WHY groups'}
               if semantic_enabled else {})}


def format_prediction_summary(entry):
    paths = entry['prediction_files']
    k_text = ','.join(map(str, entry['k_values']))
    batches = entry['train_prediction_batches']
    batch_word = 'batch' if batches == 1 else 'batches'
    branches = entry.get('prediction_branches', {'where': True, 'semantic': True})
    enabled = ', '.join(branch.upper() for branch in ('where', 'semantic') if branches[branch])
    return (f"[EVAL][PRED] epoch {entry['epoch']} | step {entry['step']} | "
            f"autoregressive predictions complete: {enabled}\n"
            f"  train: {batches} {batch_word} ({entry['train_prediction_episodes']} queries) -> {paths['train']}\n"
            f"  test: {entry['test_prediction_queries']} unseen queries x K={k_text} "
            f"({entry['test_prediction_episodes']} episodes) -> {paths['test']}\n"
            f"  semantic test predictions: {entry.get('test_semantic_prediction_episodes', 0)} episodes\n"
            '  Records contain only the enabled prediction branches. '
            'WHERE is free-running; semantic uses GT WHERE states + gold WHY groups.')


def score_prediction_artifact(path, *, queries, config, output_dir, checkpoint='integrated',
                              split_manifest_identity=''):
    """Rescore frozen predictions without loading or generating with the VLM.

    WHERE metrics consume the exact records emitted by ``predict_epoch``.  The
    query id is the join key and the subject is checked before scoring, which
    prevents an image-level human average from silently replacing a
    personalized pair.
    """
    metric_opts = config.get('evaluation', {}).get('metrics', {})
    active_opts = config.get('evaluation', {}).get('predictions', {})
    where_opts = metric_opts.get('where', {})
    sem_opts = metric_opts.get('semantic', {})
    score_where = active_opts.get('where', True) and any(
        where_opts.get(k) for k in ('scanmatch', 'multimatch', 'sed'))
    score_sem = active_opts.get('semantic', True) and (
        sem_opts.get('bertscore', {}).get('enabled') or
        sem_opts.get('cider_r', {}).get('enabled'))
    if not metric_opts.get('enabled') or not (score_where or score_sem):
        return None
    if config.get('data', {}).get('dataset') == 'all':
        import copy
        from collections import Counter
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with Path(path).open('r', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        counts = Counter(getattr(q, 'dataset', 'COCO-Search18') for q in queries)
        artifacts = {}
        for dataset in ('AiR', 'COCO-Search18'):
            selected_queries = [q for q in queries if getattr(q, 'dataset', 'COCO-Search18') == dataset]
            if not selected_queries:
                raise ValueError(f'joint metrics missing test queries: {dataset}')
            subdir = output / dataset.replace('-', '_')
            subdir.mkdir(parents=True, exist_ok=True)
            selected_rows = [r for r in rows if r.get('split') == 'test' and
                             r.get('dataset', 'COCO-Search18') == dataset]
            subset_path = subdir / 'test_predictions.jsonl'
            with subset_path.open('w', encoding='utf-8') as stream:
                for row in selected_rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            scoped_config = copy.deepcopy(config)
            scoped_config['data']['dataset'] = dataset
            artifact = score_prediction_artifact(subset_path, queries=selected_queries,
                config=scoped_config, output_dir=subdir, checkpoint=checkpoint,
                split_manifest_identity=split_manifest_identity)
            artifacts[dataset] = artifact
        result = {'metrics_path': str(output / 'metrics_by_dataset.json'),
                  'by_dataset': artifacts, 'test_queries_by_dataset': dict(counts),
                  'aggregation': 'separate_dataset_metrics_no_joint_scalar'}
        (output / 'metrics_by_dataset.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
    metrics_config = config.get('evaluation', {}).get('metrics', {})
    if not metrics_config.get('enabled'):
        return None
    pred_config = config.get('evaluation', {}).get('predictions', {})
    where_config = metrics_config.get('where', {})
    scanpath_enabled = pred_config.get('where', True) and any(
        bool(where_config.get(name)) for name in ('scanmatch', 'multimatch', 'sed'))
    semantic_config = metrics_config.get('semantic', {})
    semantic_enabled = pred_config.get('semantic', True) and bool(
        semantic_config.get('bertscore', {}).get('enabled') or
        semantic_config.get('cider_r', {}).get('enabled'))
    if not scanpath_enabled and not semantic_enabled:
        # Probability metrics are produced during the test loss traversal and
        # cannot be reconstructed from frozen prediction text.
        return None
    bertscorer = bert_provenance = cider_r = cider_provenance = None
    if semantic_enabled and semantic_config.get('bertscore', {}).get('enabled'):
        options = {key: value for key, value in semantic_config['bertscore'].items()
                   if key not in ('enabled', 'batch_size')}
        bertscorer, bert_provenance = build_bertscorer(**options)
    if semantic_enabled and semantic_config.get('cider_r', {}).get('enabled'):
        cider_r, cider_provenance = build_cider_r_scorer(
            reference_root=Path(__file__).resolve().parents[2] / 'third_party' / 'cider_r',
            n=semantic_config['cider_r'].get('n', 4), k_r=semantic_config['cider_r'].get('k_r', 0.8))
    with Path(path).open('r', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    # In single-traversal semantic mode the combined file still contains all
    # WHERE draws, but only draw zero carries a semantic prediction. Keep the
    # two row populations separate so WHERE metrics still see every draw.
    test_rows = [row for row in rows if row.get('split') == 'test']
    semantic_rows = ([row for row in test_rows if 'semantic_generation' in row]
                     if semantic_enabled else [])
    refs = {}
    by_id = {q.record_id: q for q in queries}
    where_grouped = {}
    semantic_grouped = {}
    for row in test_rows:
        key = (row.get('query_id', row.get('record_id')), int(row.get('K', row.get('k'))), int(row['draw_id']))
        query = by_id.get(key[0])
        if query is None:
            raise ValueError(f'missing query reference for frozen prediction {key}')
        if str(row.get('subject', query.subject)) != str(query.subject):
            raise ValueError(f'subject mismatch for frozen prediction {key}')
        if scanpath_enabled:
            where_grouped.setdefault((key[1], key[2]), []).append(row)
    for row in semantic_rows:
        key = (row.get('query_id', row.get('record_id')), int(row.get('K', row.get('k'))), int(row['draw_id']))
        query = by_id.get(key[0])
        if query is None:
            raise ValueError(f'missing query reference for frozen prediction {key}')
        if str(row.get('subject', query.subject)) != str(query.subject):
            raise ValueError(f'subject mismatch for frozen prediction {key}')
        refs[key] = query.semantic
        semantic_grouped.setdefault((key[1], key[2]), []).append(row)
    # Frozen prediction artifacts are allowed to be inspected independently;
    # each observed draw must nevertheless contain every eligible test query,
    # exactly once, and joins remain key based.
    expected_query_ids = {q.record_id for q in queries}
    for (k, draw), group in where_grouped.items():
        assert_key_sets_equal(group)
        observed_query_ids = {row.get('query_id', row.get('record_id')) for row in group}
        if observed_query_ids != expected_query_ids:
            missing = sorted(expected_query_ids - observed_query_ids)
            extra = sorted(observed_query_ids - expected_query_ids)
            raise ValueError(f'incomplete frozen prediction draw K={k}, draw={draw}: '
                             f'missing={missing!r}, extra={extra!r}')
    for (k, draw), group in semantic_grouped.items():
        assert_key_sets_equal(group)
        observed_query_ids = {row.get('query_id', row.get('record_id')) for row in group}
        if observed_query_ids != expected_query_ids:
            missing = sorted(expected_query_ids - observed_query_ids)
            extra = sorted(observed_query_ids - expected_query_ids)
            raise ValueError(f'incomplete semantic prediction draw K={k}, draw={draw}: '
                             f'missing={missing!r}, extra={extra!r}')
    by_k = {}
    grouped_keys = sorted((set(where_grouped) if scanpath_enabled else set()) |
                          (set(semantic_grouped) if semantic_enabled else set()))
    for (k, draw) in grouped_keys:
        where_group = where_grouped.get((k, draw), [])
        semantic_group = semantic_grouped.get((k, draw), [])
        block = by_k.setdefault(str(k), {'draws': {}})
        draw_metrics = {}
        if scanpath_enabled:
            scores = []
            for row in where_group:
                query = by_id[row.get('query_id', row.get('record_id'))]
                try:
                    gt_x, gt_y, gt_d = query.x_px, query.y_px, query.duration_ms
                    gt_dims = (query.image_width, query.image_height)
                    invalid_gt = (not gt_x or len(gt_x) != len(gt_y) or
                                  len(gt_x) != len(gt_d) or
                                  (getattr(query, 'dataset', 'COCO-Search18') == 'COCO-Search18'
                                   and gt_dims != (1680, 1050)))
                except (AttributeError, TypeError):
                    invalid_gt = True
                if invalid_gt:
                    raise DatasetIntegrityError(
                        f'invalid GT dimensions for query {query.record_id!r}')
                where = row.get('where_generation', {})
                text = row.get('WHERE', {}).get('PRED', where.get('text', ''))
                parsed = parse_xyd_output(text, len(gt_x), config.get('where', {}).get('end_fix_token', '<END_FIX>'))
                # Preserve parser recovery: malformed format with recovered
                # fixations is a scored query and remains a format failure.
                prediction = parsed.get('fixations', [])
                if not prediction:
                    gt_symbols = len(gt_x)
                    score = {'sm': 0.0, 'mm': 0.0, 'sed': gt_symbols,
                             'sm_no_duration': 0.0, 'sm_with_duration': 0.0,
                             'mm_components': [0.0] * 5, 'valid': False,
                             'format_failure': True, 'mm_numeric_failure': False}
                else:
                    score = score_scanpath_pair(
                        list(zip(gt_x, gt_y, gt_d)), prediction,
                        gt_image_width=query.image_width, gt_image_height=query.image_height,
                        require_multimatch=bool(where_config.get('multimatch', False)),
                        compute_multimatch=bool(where_config.get('multimatch', False)),
                        compute_scanmatch=bool(where_config.get('scanmatch', False)),
                        compute_sed=bool(where_config.get('sed', False)))
                    score['format_failure'] = not bool(parsed.get('canonical_format_valid'))
                scores.append(score)
            draw_metrics = aggregate_scanpath_draw(scores)
            draw_metrics.update({
                'SM': draw_metrics['eval_where_sm'],
                'MM': draw_metrics['eval_where_mm'],
                'SED': draw_metrics['eval_where_sed'],
                'valid': draw_metrics['where_valid_count'],
                'failed': len(scores) - draw_metrics['where_valid_count'],
                'query_count': len(scores),
                         'under_generation_count': sum(bool(r.get('where_generation', {}).get('under_generated')) for r in where_group),
            })
            print(f"[EVAL][METRIC][WHERE] K={k} draw={draw} SM={draw_metrics['SM']:.6f} "
                  f"MM={draw_metrics['MM']:.6f} SED={draw_metrics['SED']:.6f} "
                  f"valid={draw_metrics['valid']} failed={draw_metrics['failed']}", flush=True)
        if semantic_enabled and semantic_group:
            semantic_metrics = score_prediction_semantics(
                semantic_group, {(row.get('query_id', row.get('record_id')), k, draw): refs[(row.get('query_id', row.get('record_id')), k, draw)] for row in semantic_group},
                bertscorer=bertscorer, cider_r=cider_r,
                batch_size=semantic_config.get('bertscore', {}).get('batch_size', 64))
            draw_metrics.update(semantic_metrics)
        block['draws'][str(draw)] = draw_metrics
    semantic_use_draws = config.get('evaluation', {}).get('semantic_use_draws', True)
    for block in by_k.values():
        draws = list(block['draws'].values())
        semantic_draws = [draw for draw in draws if any(name.startswith('eval_sem_') for name in draw)]
        if semantic_enabled:
            block['semantic_draw_count'] = len(semantic_draws)
            block['semantic_metric_aggregation'] = 'draw_mean' if semantic_use_draws else 'single_traversal'
            block['draw_mean'] = aggregate_semantic_metrics(semantic_draws, use_draws=semantic_use_draws)
        else:
            block['draw_mean'] = {}
        if scanpath_enabled and draws:
            for name in ('SM', 'MM', 'SED', 'valid', 'failed', 'query_count',
                         'under_generation_count', 'mm_numeric_failure_count',
                         'where_format_failure_count', 'where_valid_count'):
                values = [d[name] for d in draws if isinstance(d.get(name), (int, float))]
                if values:
                    block['draw_mean'][name] = sum(values) / len(values)
    artifact = write_metrics_artifact(Path(output_dir) / 'metrics.json', checkpoint=checkpoint,
        config=config.get('_resolved_config_path', 'resolved_config.json'),
        split_manifest_identity=split_manifest_identity, by_k=by_k,
        provenance={**({'bertscore': bert_provenance, 'cider_r': cider_provenance}
                       if semantic_enabled else {}),
                    'scanpath': {
                        'enabled': scanpath_enabled,
                        'multimatch_gaze_version': '0.1.3',
                        'query_pairing': 'same_subject_canonical_query_id',
                        'frame': [512, 320],
                        'scanmatch': {'Xres': 512, 'Yres': 320, 'Xbin': 16, 'Ybin': 12,
                                      'Offset': [0, 0], 'TempBin_ms': 50, 'Threshold': 3.5},
                        'sed': {'grid': [5, 5], 'normalization': 'raw_edit_distance'},
                        'query_counts': {
                            'draws': sum(len(block['draws']) for block in by_k.values()),
                            'queries': sum(int(d.get('query_count', 0))
                                           for block in by_k.values() for d in block['draws'].values()),
                            'valid': sum(int(d.get('valid', 0))
                                         for block in by_k.values() for d in block['draws'].values()),
                            'failed': sum(int(d.get('failed', 0))
                                          for block in by_k.values() for d in block['draws'].values()),
                            'under_generation': sum(int(d.get('under_generation_count', 0))
                                                   for block in by_k.values() for d in block['draws'].values()),
                        },
                        'gate_b1': 'passed',
                        'gate_b2': 'passed',
                        'inverse': 'deepgaze_vl_predict_scanpath_round',
                        'inverse_source': 'DeepGaze-VL/predict_scanpath.py:238-244',
                        'isp_preprocess_sha256': '81aa0754346a382f1f818270645c934af275b6f5e96f6f44b97049108e44512f',
                        'isp_evaluation_sha256': 'd68c1c0554c8195785084f34e4b7f74b06ca085dc3deff7c729517eee79cdb2a',
                        'deepgaze_prediction_sha256': '4f059a30e829ff7414924d1841f20909c7b776d5227767b5abeda863e872108f',
                    },
                    'probability': 'not_integrated; tokenizer outcome unresolved'},
        implementation_head=implementation_head(), evaluation_draw=config['evaluation']['draw'])
    artifact['metrics_path'] = str(Path(output_dir) / 'metrics.json')
    return artifact

