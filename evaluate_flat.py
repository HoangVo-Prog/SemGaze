"""Standalone held-out SemGaze evaluation from a complete training checkpoint.

The checkpoint is authoritative for model/data/training. --config accepts either
an evaluation-only YAML mapping or the complete configs/flat_single.yaml and
imports only runtime/evaluation/test overrides. The integrated repository
LL and prediction/scanpath/semantic metric implementations are reused.
Loss is optional via evaluation.compute_loss or --skip-loss.

python evaluate_flat.py \
  --checkpoint runs/YOUR_RUN/checkpoints/YOUR_CHECKPOINT \
  --config configs/flat_single.yaml \
  --output-dir runs/eval_metrics_only \
  --skip-loss

"""
import argparse
import json
import time
from pathlib import Path

from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter
from semgaze.data.joint import JointAdapter
from semgaze.data.fewshot import frozen_episode
from semgaze.model.checkpoint import load_checkpoint_bundle
from semgaze.model.config import ROOT, load_config, resolve_config, resolve_output_dir, write_run_config
from semgaze.model.visual_cache import InferenceVisualCache
from semgaze.evaluation.test import evaluate_test_epoch, test_mode, test_queries
from semgaze.evaluation.batching import schedule_batches
from semgaze.evaluation.progress import RollingRate, format_eta, progress_interval, should_report
from semgaze.evaluation.metrics_probability import (
    score_probability_batch, aggregate_probability_draw, aggregate_probability_k,
)
from semgaze.evaluation.predictions import (
    _prediction_record, prediction_batches, resolve_prediction_settings,
    score_prediction_artifact,
)
from semgaze.evaluation.records import (
    assert_key_sets_equal, expected_episode_keys, resolve_evaluation_draw_counts,
    resolve_semantic_draw_counts,
)


ALLOWED_OVERRIDE_SECTIONS = frozenset({'runtime', 'evaluation', 'test'})


def _load_evaluation_config(checkpoint, path):
    """Retain checkpoint identity and selectively overlay runtime evaluation controls."""
    checkpoint_config = load_config(checkpoint / 'resolved_config.json')
    if path is None:
        return checkpoint_config
    import yaml
    source = Path(path)
    if source.suffix.lower() == '.json':
        override_payload = json.loads(source.read_text(encoding='utf-8'))
    else:
        override_payload = yaml.safe_load(source.read_text(encoding='utf-8'))
    if not isinstance(override_payload, dict):
        raise ValueError('--config must be a YAML/JSON mapping')
    # A full training YAML is a valid input: immutable training sections are
    # deliberately ignored, rather than replacing weights/data provenance.
    selected = {key: override_payload[key] for key in ALLOWED_OVERRIDE_SECTIONS
                if key in override_payload}
    if any(not isinstance(v, dict) for v in selected.values()):
        raise ValueError('runtime, evaluation and test overrides must be mappings')
    ignored = sorted(set(override_payload) - ALLOWED_OVERRIDE_SECTIONS)
    if ignored:
        print('[CONFIG] Ignoring checkpoint-owned --config sections: '
              + ', '.join(ignored), flush=True)
    return resolve_config(checkpoint_config, selected)


def _prediction_plan(config, manifest, *, enabled_where, enabled_semantic):
    ks = config['evaluation']['k_values']
    if any(k not in (1, 5, 10) for k in ks):
        raise ValueError('frozen_episode currently supports evaluation K in {1, 5, 10}')
    # Validate persisted draws before allocating the backbone, including
    # loss-free/semantic-only runs.
    where_counts = resolve_evaluation_draw_counts(config['evaluation']['draw'], manifest, ks)
    semantic_counts = (resolve_semantic_draw_counts(where_counts, config['evaluation']['semantic_use_draws'])
                       if enabled_semantic else {str(k): 0 for k in ks})
    return where_counts, semantic_counts


def _validate_frozen_supports(train, queries, manifest, counts):
    """Fail before loading the 8B model if a persisted support is unresolved."""
    for k, count in counts.items():
        for draw_id in range(count):
            for query in queries:
                frozen_episode(query, train, manifest, int(k), draw_id=draw_id,
                               unseen_subjects=manifest['unseen_subject_ids'])


def _generate_predictions(bundle, train, queries, manifest, identity, output_dir, *,
                          enabled_where, enabled_semantic, budget, draw_counts, semantic_counts):
    """Canonical prediction records and complete (query_id, K, draw_id) coverage."""
    if not (enabled_where or enabled_semantic):
        return None
    cache_opts = bundle.config['test']['cache']
    cache = InferenceVisualCache(
        preprocessing=cache_opts['support_preprocessing'],
        features=cache_opts['frozen_visual_features'],
        max_entries=cache_opts['max_entries'],
    )
    destination = output_dir / 'predictions.jsonl'
    partial = destination.with_suffix('.jsonl.partial')
    k_values = bundle.config['evaluation']['k_values']
    counts = {}
    written = 0
    # Match training ETA: measure full inference windows, NOT the much faster
    # loop serializing previously generated rows. Distinct K/path workloads
    # mean this whole-prediction ETA is an estimate, not a deadline.
    total = sum(len(queries) * (draw_counts[str(k)] if enabled_where
                                else semantic_counts[str(k)]) for k in k_values)
    progress = RollingRate(min_observations=2)
    ready = 0
    started = time.perf_counter()

    def window_ready(window_size):
        nonlocal ready
        ready += window_size
        if ready > total:
            raise RuntimeError('prediction windows exceed planned evaluation episodes')
        progress.update(ready)

    print(f'[EVAL][PRED] starting | queries={len(queries)} | K={list(k_values)} '
          f'| planned_episodes={total} | WHERE={enabled_where} '
          f'| semantic={enabled_semantic}', flush=True)
    try:
        with test_mode(bundle):
            cache.validate(bundle)
            with partial.open('w', encoding='utf-8') as stream:
                for k in k_values:
                    traversal_count = draw_counts[str(k)] if enabled_where else semantic_counts[str(k)]
                    for draw_id in range(traversal_count):
                        run_semantic = enabled_semantic and draw_id < semantic_counts[str(k)]
                        path = 'both' if enabled_where and run_semantic else (
                            'where' if enabled_where else 'semantic')
                        episodes = [frozen_episode(query, train, manifest, k, draw_id=draw_id,
                                    unseen_subjects=manifest['unseen_subject_ids']) for query in queries]
                        key = f'K={k},draw={draw_id}'
                        rates = {'queries': 0}
                        if enabled_where:
                            rates['where_under_generated'] = 0
                        if run_semantic:
                            rates['flat_format_valid'] = 0
                        observed = []
                        interval = progress_interval(len(queries))
                        for episode, generated in prediction_batches(
                                bundle, episodes, budget=budget, cache=cache, path=path,
                                on_window_ready=window_ready):
                            row = _prediction_record(
                                bundle, episode, epoch=None, step=None, split='test',
                                index=written + 1, budget=budget,
                                split_manifest_identity=identity, generated=generated)
                            row['draw_id'] = draw_id
                            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                            observed.append(row)
                            if enabled_where:
                                rates['where_under_generated'] += int(
                                    bool(row['where_generation']['under_generated']))
                            if run_semantic:
                                rates['flat_format_valid'] += int(
                                    bool(row['semantic_generation']['flat_format_valid']))
                            written += 1
                            rates['queries'] += 1
                            if should_report(rates['queries'], len(queries), interval):
                                eta = progress.eta(total - ready)
                                print(f'[EVAL][PRED] {written}/{total} '
                                      f'({100 * written / total:.1f}%) '
                                      f'| K={k} draw={draw_id + 1}/{traversal_count} '
                                      f'[{rates["queries"]}/{len(queries)}] '
                                      f'| path={path} | ETA test predictions={format_eta(eta)}',
                                      flush=True)
                        expected = [{'query_id': q.record_id, 'K': k, 'draw_id': draw_id}
                                    for q in queries]
                        assert_key_sets_equal(observed, expected)
                        if rates['queries'] != len(queries):
                            raise ValueError(f'incomplete prediction coverage: {key}')
                        counts[key] = rates | {
                            label + '_rate': value / rates['queries']
                            for label, value in rates.items() if label != 'queries'}
                        print(f'[EVAL][PRED] {key} | {rates["queries"]}/{len(queries)} complete',
                              flush=True)
        expected_all = expected_episode_keys(
            [q.record_id for q in queries], k_values,
            draw_counts if enabled_where else semantic_counts)
        if written != len(expected_all) or ready != written:
            raise ValueError(f'prediction episode coverage mismatch: written={written}, '
                             f'inferred={ready}, expected={len(expected_all)}')
        partial.replace(destination)
        print(f'[EVAL][PRED] complete | episodes={written}/{total} '
              f'| elapsed={time.perf_counter() - started:.1f}s', flush=True)
    finally:
        cache.close()
    (output_dir / 'validity_summary.json').write_text(
        json.dumps(counts, indent=2, allow_nan=False), encoding='utf-8')
    return destination


def _evaluate_probability_metrics_only(bundle, train, queries, manifest, output_dir, draw_counts):
    """Compute canonical COCO-Search18 LL without WHERE/semantic loss forwards.

    Reuses the same Outcome-B scorer and test.loss batching policy as
    evaluate_test_epoch. The scorer performs the GT-context WHERE forward
    needed for token log-probabilities; it never computes NLL loss objectives.
    """
    probability = bundle.config['evaluation']['metrics']['probability']
    if probability.get('ig'):
        raise ValueError('IG is not implemented in the canonical evaluation path')
    if not probability.get('ll'):
        return None
    if probability.get('outcome') != 'B':
        raise ValueError('canonical LL currently supports only metrics.probability.outcome: B')

    settings = bundle.config['test']['loss']
    cache_options = bundle.config['test']['cache']
    cache = InferenceVisualCache(
        preprocessing=cache_options['support_preprocessing'],
        features=cache_options['frozen_visual_features'],
        max_entries=cache_options['max_entries'],
    )
    coco_queries = [q for q in queries
                    if getattr(q, 'dataset', 'COCO-Search18') == 'COCO-Search18']
    by_k = {}
    total = sum(len(coco_queries) * draw_counts[str(k)]
                for k in bundle.config['evaluation']['k_values'])
    completed = 0
    progress = RollingRate(min_observations=2)
    started = time.perf_counter()
    print(f'[EVAL][LL] starting | COCO queries={len(coco_queries)} '
          f'| K={bundle.config["evaluation"]["k_values"]} '
          f'| planned_episodes={total}', flush=True)
    try:
        with test_mode(bundle):
            cache.validate(bundle)
            for k in bundle.config['evaluation']['k_values']:
                block = {'draws': {}}
                for draw_id in range(draw_counts[str(k)]):
                    print(f'[EVAL][LL] K={k} draw={draw_id + 1}/{draw_counts[str(k)]} '
                          f'| COCO queries={len(coco_queries)} | starting', flush=True)
                    episodes = [frozen_episode(q, train, manifest, k, draw_id=draw_id,
                                unseen_subjects=manifest['unseen_subject_ids']) for q in coco_queries]
                    rows = []
                    draw_completed = 0
                    interval = progress_interval(len(coco_queries))
                    next_report = interval
                    for batch in schedule_batches(bundle, episodes, settings, cache=cache):
                        scored = score_probability_batch(bundle, batch.episodes, visual_cache=cache)
                        if len(scored) != len(batch.episodes):
                            raise ValueError('LL scorer returned incomplete physical batch')
                        rows.extend(scored)
                        draw_completed += len(scored)
                        completed += len(scored)
                        # Unlike prediction generation, LL batches return only
                        # after inference, so this is a true completed-work rate.
                        progress.update(completed)
                        if draw_completed >= next_report or draw_completed == len(coco_queries):
                            print(f'[EVAL][LL] {completed}/{total} '
                                  f'({100 * completed / total:.1f}%) '
                                  f'| K={k} draw={draw_id + 1}/{draw_counts[str(k)]} '
                                  f'[{draw_completed}/{len(coco_queries)}] '
                                  f'| ETA LL={format_eta(progress.eta(total - completed))}',
                                  flush=True)
                            next_report = (draw_completed // interval + 1) * interval
                    expected = [{'query_id': q.record_id, 'K': k, 'draw_id': draw_id}
                                for q in coco_queries]
                    assert_key_sets_equal(rows, expected)
                    block['draws'][str(draw_id)] = aggregate_probability_draw(rows)
                    print(f'[EVAL][LL] K={k} draw={draw_id + 1}/{draw_counts[str(k)]} '
                          '| complete', flush=True)
                block['draw_mean'] = aggregate_probability_k(list(block['draws'].values()))
                by_k[str(k)] = block
    finally:
        cache.close()
    if completed != total:
        raise ValueError(f'LL evaluation coverage mismatch: {completed} != {total}')
    print(f'[EVAL][LL] complete | episodes={completed}/{total} '
          f'| elapsed={time.perf_counter() - started:.1f}s', flush=True)
    artifact = {
        'metric': 'LL (Outcome B)',
        'dataset': 'COCO-Search18',
        'loss_computed': False,
        'draw': bundle.config['evaluation']['draw'],
        'by_k': by_k,
    }
    path = output_dir / 'probability_metrics.json'
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding='utf-8')
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Checkpoint directory with resolved_config.json, adapter, processor and tensors')
    parser.add_argument('--config', type=Path,
                        help='Full configs/flat_single.yaml or evaluation/runtime/test override YAML')
    parser.add_argument('--output-dir', type=Path,
                        help='Fresh output directory (never write into the training checkpoint)')
    parser.add_argument('--split', choices=('test',), default='test')
    parser.add_argument('--dry-run', action='store_true',
                        help='Check YAML, split and frozen supports without loading GPU weights')
    loss_options = parser.add_mutually_exclusive_group()
    loss_options.add_argument('--skip-loss', dest='loss_override', action='store_false',
                              help='Skip all WHERE/semantic loss forward passes; still compute enabled metrics')
    loss_options.add_argument('--compute-loss', dest='loss_override', action='store_true',
                              help='Force loss computation (overrides evaluation.compute_loss: false)')
    parser.set_defaults(loss_override=None)
    parser.add_argument('--path', choices=('where', 'semantic', 'both', 'none'),
                        help='Optional override of evaluation.predictions.where/semantic')
    parser.add_argument('--semantic-max-new-tokens', type=int,
                        help='Override evaluation.predictions.semantic_max_new_tokens')
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    try:
        config = _load_evaluation_config(checkpoint, args.config)
        if args.path is not None:
            config['evaluation']['predictions'].update(
                where=args.path in ('where', 'both'),
                semantic=args.path in ('semantic', 'both'))
        if args.semantic_max_new_tokens is not None:
            config['evaluation']['predictions']['semantic_max_new_tokens'] = args.semantic_max_new_tokens
        config = resolve_config(config)
        compute_loss = config['evaluation'].get('compute_loss', True)
        if type(compute_loss) is not bool:
            raise ValueError('evaluation.compute_loss must be true or false')
        if args.loss_override is not None:
            compute_loss = args.loss_override
        config['evaluation']['compute_loss'] = compute_loss
        predictions = resolve_prediction_settings(config)
        enabled_where = predictions['where'] and predictions['test_scope'] == 'all_unseen'
        enabled_semantic = predictions['semantic'] and predictions['test_scope'] == 'all_unseen'
        budget = predictions['semantic_max_new_tokens']
        if not compute_loss and not (enabled_where or enabled_semantic):
            raise ValueError('loss is disabled and no test prediction branch is enabled; '
                             'enable WHERE/semantic or --compute-loss')
        metrics = config['evaluation']['metrics']
        probability = metrics['probability']
        if metrics['enabled'] and enabled_where and probability.get('ig'):
            raise ValueError('IG is not implemented in the current evaluate_test_epoch; '
                             'disable evaluation.metrics.probability.ig')
        if metrics['enabled'] and enabled_where and (probability.get('ll') or probability.get('ig')):
            if probability.get('outcome') != 'B':
                raise ValueError('canonical LL currently supports only metrics.probability.outcome: B')
        if args.output_dir is not None:
            config['runtime']['output_dir'] = str(args.output_dir.expanduser().resolve())
        elif args.config is None or _has_no_explicit_output_dir(args.config):
            # The checkpoint's saved runtime.output_dir points at the training
            # run. Never accidentally reuse it as a standalone eval directory.
            config['runtime']['output_dir'] = None
        output_dir = resolve_output_dir(config)
        if output_dir.resolve() == checkpoint or output_dir.resolve() in checkpoint.parents or checkpoint in output_dir.parents:
            raise ValueError('evaluation output directory must not be the checkpoint or its parent')
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError('choose an empty evaluation output directory')
    except (ValueError, KeyError, FileNotFoundError) as exc:
        parser.error(str(exc))

    data = config['data']
    raw, manifest, identity = read_persisted_splits(ROOT / data['split_root'], data_config=data)
    adapter_cls = JointAdapter if data['dataset'] == 'all' else CocoSearch18Adapter
    adapter = adapter_cls(ROOT / data['images_root'], annotation_frame=data['annotation_frame'],
                          duration_field=data['duration']['source_field'])
    train = {r['record_id']: adapter(r) for r in raw['train']}
    tests = [adapter(r) for r in raw['test']]
    queries = test_queries(train, tests, manifest)
    try:
        draw_counts, semantic_counts = _prediction_plan(
            config, manifest, enabled_where=enabled_where,
            enabled_semantic=enabled_semantic)
    except ValueError as exc:
        parser.error(str(exc))
    _validate_frozen_supports(train, queries, manifest, draw_counts)

    print(f'[EVAL] checkpoint={checkpoint} | queries={len(queries)} | '
          f'K={config["evaluation"]["k_values"]} | draws={config["evaluation"]["draw"]} | '
          f'WHERE={enabled_where} | semantic={enabled_semantic} | '
          f'compute_loss={compute_loss}', flush=True)
    if args.dry_run:
        print('[EVAL] dry-run PASSED; no model loaded or files written', flush=True)
        return
    bundle = load_checkpoint_bundle(checkpoint, split_manifest_identity=identity,
                                    output_dir=output_dir, runtime=config['runtime'])
    bundle.config = config
    write_run_config(config, output_dir)
    policy = {
        'checkpoint': str(checkpoint),
        'config_override': str(args.config) if args.config else None,
        'checkpoint_model_and_data_authoritative': True,
        'split_manifest_identity': identity,
        'prediction_branches': {'where': enabled_where, 'semantic': enabled_semantic},
        'compute_loss': compute_loss,
        'k_values': config['evaluation']['k_values'],
        'draw': config['evaluation']['draw'],
        'semantic_use_draws': config['evaluation']['semantic_use_draws'],
        'semantic_max_new_tokens': budget if enabled_semantic else None,
        'where_conditioning': 'oracle_length; no query GT trajectory' if enabled_where else None,
        'semantic_conditioning': 'teacher_forced_GT_XYD_states_and_gold_WHY_groups'
                                 if enabled_semantic else None,
        'standalone_strategy_note': 'strategy/eval_epoch/eval_steps control training schedules, not this standalone run',
        'probability_scope': 'COCO-Search18 only (not AiR)',
    }
    (output_dir / 'evaluation_config.json').write_text(
        json.dumps(policy, indent=2), encoding='utf-8')

    probability_metrics_path = None
    if compute_loss:
        # Canonical teacher-forced losses, including LL if enabled.
        loss_result = evaluate_test_epoch(bundle, train, tests, manifest)
        (output_dir / 'test_metrics.json').write_text(
            json.dumps(loss_result, indent=2, allow_nan=False), encoding='utf-8')
        print(f'[EVAL][LOSS] test_total={loss_result["test_total"]:.6f}', flush=True)
    else:
        print('[EVAL][LOSS] skipped: no WHERE/semantic NLL or total loss computed',
              flush=True)
        # LL is a separate metric, not the WHERE loss: keep it enabled without
        # running the expensive semantic/WHERE loss-evaluation traversal.
        if metrics['enabled'] and enabled_where and probability.get('ll'):
            probability_metrics_path = _evaluate_probability_metrics_only(
                bundle, train, queries, manifest, output_dir, draw_counts)

    prediction_path = _generate_predictions(
        bundle, train, queries, manifest, identity, output_dir,
        enabled_where=enabled_where, enabled_semantic=enabled_semantic,
        budget=budget, draw_counts=draw_counts, semantic_counts=semantic_counts)
    metrics_artifact = None
    if prediction_path is not None and metrics['enabled']:
        print('[EVAL][METRIC] scoring frozen predictions | starting', flush=True)
        metric_started = time.perf_counter()
        metrics_artifact = score_prediction_artifact(
            prediction_path, queries=queries, config=config, output_dir=output_dir,
            checkpoint=checkpoint, split_manifest_identity=identity)
        print(f'[EVAL][METRIC] complete '
              f'| elapsed={time.perf_counter() - metric_started:.1f}s', flush=True)
    (output_dir / 'evaluation_summary.json').write_text(json.dumps({
        'checkpoint': str(checkpoint), 'output_dir': str(output_dir),
        'test_metrics': str(output_dir / 'test_metrics.json') if compute_loss else None,
        'compute_loss': compute_loss,
        'probability_metrics': probability_metrics_path,
        'prediction_file': str(prediction_path) if prediction_path is not None else None,
        'prediction_metrics': metrics_artifact,
        'prediction_branches': policy['prediction_branches'],
    }, indent=2, allow_nan=False), encoding='utf-8')
    print(f'[EVAL] complete | results={output_dir}', flush=True)


def _has_no_explicit_output_dir(path):
    """Check the user's override rather than inherited checkpoint runtime."""
    import yaml
    payload = Path(path).read_text(encoding='utf-8')
    parsed = json.loads(payload) if Path(path).suffix.lower() == '.json' else yaml.safe_load(payload)
    return not isinstance(parsed, dict) or not isinstance(parsed.get('runtime'), dict) or \
        parsed['runtime'].get('output_dir') is None


if __name__ == '__main__':
    main()
