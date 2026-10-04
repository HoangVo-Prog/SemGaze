"""Epoch qualitative outputs through the two canonical autoregressive paths."""
import json
from pathlib import Path
import random
import torch
from semgaze.data.fewshot import frozen_episode
from semgaze.data.schema import K_VALUES, UNSEEN_SUBJECTS
from semgaze.evaluation.flat import evaluate_flat_episode
from semgaze.evaluation.where import evaluate_where_episode
from semgaze.evaluation.validation import validation_mode, validation_queries
from semgaze.semantic.flat.target import build_flat_target
from semgaze.where.serialization import serialize_xyd_record


def resolve_prediction_settings(config, semantic_max_new_tokens=None):
    settings = config.setdefault('evaluation', {}).setdefault('predictions', {})
    settings.setdefault('train_batches', 1)
    settings.setdefault('validation_scope', 'all_seen')
    if type(settings['train_batches']) is not int or settings['train_batches'] != 1 or settings['validation_scope'] != 'all_seen':
        raise ValueError('epoch predictions require one train batch and all seen validation queries')
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
            'WHERE': {'GT': serialize_xyd_record(query), 'PRED': where.pop('text')},
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
    if len(train_batch) != batch_size:
        raise ValueError('train predictions require exactly one complete prediction batch')
    if any(ep.query.subject in UNSEEN_SUBJECTS for ep in train_batch):
        raise ValueError('unseen query in train prediction batch')
    queries = validation_queries(train_by_id, validation_records, manifest)
    directory = Path(bundle.output_dir).resolve() / 'predictions' / f'epoch-{epoch:04d}'
    directory.mkdir(parents=True, exist_ok=True)
    paths, counts = {}, {}
    python_rng = random.getstate()
    try:
        # Greedy inference normally consumes no RNG. Preserve it explicitly so
        # even runtime-specific generation internals cannot change training draws.
        with torch.random.fork_rng(), validation_mode(bundle):
            for split, episodes in (
                ('train', iter(train_batch)),
                ('validation', (frozen_episode(q, train_by_id, manifest, k)
                                for k in K_VALUES for q in queries)),
            ):
                path = directory / f'{split}.jsonl'
                partial = path.with_suffix('.jsonl.partial')
                count = 0
                with partial.open('w', encoding='utf-8') as stream:
                    for count, episode in enumerate(episodes, 1):
                        record = _prediction_record(bundle, episode, epoch=epoch, step=step,
                            split=split, index=count, budget=settings['semantic_max_new_tokens'],
                            split_manifest_identity=split_manifest_identity)
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                        stream.flush()
                partial.replace(path)
                counts[split], paths[split] = count, str(path)
    finally:
        random.setstate(python_rng)
    return {'event': 'epoch_predictions', 'epoch': epoch, 'step': step,
            'train_prediction_batches': 1, 'train_prediction_episodes': counts['train'],
            'train_prediction_source': 'first_batch_of_final_optimizer_step',
            'validation_prediction_queries': len(queries),
            'validation_prediction_episodes': counts['validation'], 'k_values': list(K_VALUES),
            'prediction_files': paths, 'semantic_max_new_tokens': settings['semantic_max_new_tokens'],
            'where_conditioning': 'oracle_length; no query GT trajectory',
            'semantic_conditioning': 'teacher_forced_GT_XYD states; gold WHY groups'}


def format_prediction_summary(entry):
    paths = entry['prediction_files']
    return (f"Epoch {entry['epoch']} | step {entry['step']} | autoregressive predictions:\n"
            f"  train: 1 batch ({entry['train_prediction_episodes']} queries) -> {paths['train']}\n"
            f"  validation: {entry['validation_prediction_queries']} seen queries x K=1,5,10 "
            f"({entry['validation_prediction_episodes']} episodes) -> {paths['validation']}\n"
            '  Each record: WHERE GT/PRED; SEMANTIC GT/PRED. '
            'WHERE is free-running; semantic uses GT WHERE states + gold WHY groups.')
