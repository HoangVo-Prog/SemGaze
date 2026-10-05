"""Epoch qualitative outputs through the two canonical autoregressive paths."""
import json
from pathlib import Path
import random
import time
import torch
from semgaze.data.fewshot import frozen_episode
from semgaze.evaluation.flat import evaluate_flat_episode
from semgaze.evaluation.where import evaluate_where_episode
from semgaze.evaluation.validation import validation_mode, validation_queries
from semgaze.semantic.flat.target import build_flat_target
from semgaze.where.serialization import serialize_xyd_record
from semgaze.evaluation.progress import format_duration, progress_interval, should_report


def resolve_prediction_settings(config, semantic_max_new_tokens=None):
    settings = config.setdefault('evaluation', {}).setdefault('predictions', {})
    settings.setdefault('train_batches', 1)
    settings.setdefault('validation_scope', 'all_seen')
    if type(settings['train_batches']) is not int or settings['train_batches'] < 0 or settings['validation_scope'] not in ('all_seen', 'none'):
        raise ValueError('predictions require nonnegative train_batches and validation_scope all_seen or none')
    if not settings['train_batches'] and settings['validation_scope'] == 'none':
        return settings
    budget = semantic_max_new_tokens if semantic_max_new_tokens is not None else settings.get('semantic_max_new_tokens')
    if type(budget) is not int or budget < 1:
        raise ValueError('declare evaluation.predictions.semantic_max_new_tokens or --semantic-max-new-tokens explicitly')
    settings['semantic_max_new_tokens'] = budget
    return settings


def _prediction_record(bundle, episode, *, epoch, step, split, index, budget, split_manifest_identity):
    where = evaluate_where_episode(bundle, episode)
    # Canonical semantic inference constructs R from GT WHERE, never predicted
    # fixations. Only the semantic response is autoregressively generated here.
    semantic = evaluate_flat_episode(bundle, episode, generation_budget=budget)
    query = episode.query
    return {'epoch': epoch, 'step': step, 'split': split, 'prediction_index': index,
            'split_manifest_identity': split_manifest_identity,
            'query_id': query.record_id, 'subject': query.subject,
            'stimulus_id': query.stimulus_id, 'task': query.task, 'k': len(episode.supports),
            'support_ids': [s.record_id for s in episode.supports],
            'WHERE': {'GT': serialize_xyd_record(query, bundle.config['where']['end_fix_token']), 'PRED': where.pop('text')},
            'SEMANTIC': {'GT': build_flat_target(query.semantic), 'PRED': semantic.pop('text')},
            'where_generation': where, 'semantic_generation': semantic}


def predict_epoch(bundle, train_batch, train_by_id, validation_records, manifest, *,
                  epoch, step, split_manifest_identity):
    """Exactly one supplied train batch; every seen validation query at every K.

    Episodes are executed sequentially, like training, to retain full multi-image
    contexts. No training sampler is consumed. Partial files remain identifiable
    on failure and no successful history event is emitted for incomplete coverage.
    """
    settings = resolve_prediction_settings(bundle.config)
    batch_size = bundle.config['training']['per_device_train_batch_size']
    if len(train_batch) != batch_size * settings['train_batches']:
        raise ValueError('train predictions require the configured number of complete batches')
    if any(ep.query.subject in bundle.config['data']['unseen_subjects'] for ep in train_batch):
        raise ValueError('unseen query in train prediction batch')
    queries = validation_queries(train_by_id, validation_records, manifest, bundle.config['data']['unseen_subjects']) if settings['validation_scope'] == 'all_seen' else ()
    k_values = bundle.config['evaluation']['k_values']
    tag = f'epoch-{epoch:04d}' if bundle.config['evaluation']['strategy'] == 'epoch' else f'step-{step:08d}'
    directory = Path(bundle.output_dir).resolve() / 'predictions' / tag
    directory.mkdir(parents=True, exist_ok=True)
    paths, counts = {}, {}
    prediction_by_k = {}
    prediction_started = time.perf_counter()
    print(f"[EVAL][PRED] starting training predictions | episodes={len(train_batch)}", flush=True)
    if queries:
        print(f"[EVAL][PRED] starting validation predictions | queries={len(queries)} | "
              f"K={list(k_values)} | total_episodes={len(queries) * len(k_values)}", flush=True)
    else:
        print('[EVAL][PRED] validation predictions disabled', flush=True)
    python_rng = random.getstate()
    try:
        # Greedy inference normally consumes no RNG. Preserve it explicitly so
        # even runtime-specific generation internals cannot change training draws.
        with torch.random.fork_rng(), validation_mode(bundle):
            for split in ('train', 'validation'):
                path = directory / f'{split}.jsonl'
                partial = path.with_suffix('.jsonl.partial')
                count = 0
                split_started = time.perf_counter()
                with partial.open('w', encoding='utf-8') as stream:
                    if split == 'train':
                        total = len(train_batch)
                        interval = progress_interval(total)
                        for episode in train_batch:
                            count += 1
                            record = _prediction_record(bundle, episode, epoch=epoch, step=step,
                                split=split, index=count, budget=settings['semantic_max_new_tokens'],
                                split_manifest_identity=split_manifest_identity)
                            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                            stream.flush()
                            if should_report(count, total, interval):
                                elapsed = time.perf_counter() - split_started
                                print(f"[EVAL][PRED][TRAIN] {count}/{total} | "
                                      f"{100 * count / total:.1f}% | elapsed={format_duration(elapsed)} | "
                                      f"ETA={format_duration(elapsed / count * (total - count))}", flush=True)
                    else:
                        for k in k_values:
                            k_started = time.perf_counter()
                            interval = progress_interval(len(queries))
                            for query_index, query in enumerate(queries, 1):
                                count += 1
                                episode = frozen_episode(query, train_by_id, manifest, k,
                                    unseen_subjects=bundle.config['data']['unseen_subjects'])
                                record = _prediction_record(bundle, episode, epoch=epoch, step=step,
                                    split=split, index=count, budget=settings['semantic_max_new_tokens'],
                                    split_manifest_identity=split_manifest_identity)
                                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                                stream.flush()
                                if should_report(query_index, len(queries), interval):
                                    elapsed = time.perf_counter() - k_started
                                    print(f"[EVAL][PRED] K={k} | {query_index}/{len(queries)} | "
                                          f"{100 * query_index / len(queries):.1f}% | "
                                          f"elapsed={format_duration(elapsed)} | "
                                          f"ETA={format_duration(elapsed / query_index * (len(queries) - query_index))}", flush=True)
                            k_time = time.perf_counter() - k_started
                            prediction_by_k[str(k)] = {'episodes': len(queries), 'prediction_k_time_sec': k_time}
                            print(f"[EVAL][PRED] K={k} | done | time={k_time:.2f}s", flush=True)
                partial.replace(path)
                counts[split], paths[split] = count, str(path)
    finally:
        random.setstate(python_rng)
    prediction_time = time.perf_counter() - prediction_started
    print(f"[EVAL][PRED] done | time={prediction_time:.2f}s", flush=True)
    return {'event': 'epoch_predictions', 'epoch': epoch, 'step': step,
            'train_prediction_batches': settings['train_batches'], 'train_prediction_episodes': counts['train'],
            'train_prediction_source': 'final_training_batches_before_evaluation',
            'validation_prediction_queries': len(queries),
            'validation_prediction_episodes': counts['validation'], 'k_values': list(k_values),
            'prediction_files': paths, 'semantic_max_new_tokens': settings['semantic_max_new_tokens'],
            'prediction_time_sec': prediction_time, 'prediction_by_k': prediction_by_k,
            'where_conditioning': 'oracle_length; no query GT trajectory',
            'semantic_conditioning': 'teacher_forced_GT_XYD states; gold WHY groups'}


def format_prediction_summary(entry):
    paths = entry['prediction_files']
    k_text = ','.join(map(str, entry['k_values']))
    batches = entry['train_prediction_batches']
    batch_word = 'batch' if batches == 1 else 'batches'
    return (f"[EVAL][PRED] epoch {entry['epoch']} | step {entry['step']} | "
            f"autoregressive predictions | time={entry.get('prediction_time_sec', 0.0):.2f}s:\n"
            f"  train: {batches} {batch_word} ({entry['train_prediction_episodes']} queries) -> {paths['train']}\n"
            f"  validation: {entry['validation_prediction_queries']} seen queries x K={k_text} "
            f"({entry['validation_prediction_episodes']} episodes) -> {paths['validation']}\n"
            '  Each record: WHERE GT/PRED; SEMANTIC GT/PRED. '
            'WHERE is free-running; semantic uses GT WHERE states + gold WHY groups.')
