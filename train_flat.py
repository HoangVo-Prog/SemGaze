"""Train the configured flat model using standard teacher-forced objectives."""
import argparse
import json
from pathlib import Path
import torch
from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter
from semgaze.data.joint import JointAdapter, AirAdapter
from semgaze.data.fewshot import TrainingEpisodeSampler
from semgaze.model.build import build_flat_model_bundle
from semgaze.model.config import ROOT, load_config, write_run_config
from semgaze.model.checkpoint import load_checkpoint_bundle, restore_checkpoint_state
from semgaze.model.staged_init import validate_staged_config, adapt_checkpoint_for_new_run
from semgaze.model.mtl_lora import checkpoint_task_context_fn
from semgaze.training.flat_step import make_optimizer, make_scheduler
from semgaze.training.loop import resolve_epoch_schedule, run_training_loop, require_single_process
from semgaze.evaluation.predictions import resolve_prediction_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/flat_single.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-steps', type=int, help='Optional optimizer-update cap; may stop mid-epoch')
    parser.add_argument('--semantic-max-new-tokens', type=int, help='Explicit epoch semantic generation budget')
    parser.add_argument('--num-train-epochs', '--epochs', dest='num_train_epochs', type=int)
    parser.add_argument('--save-every', type=int, help='Save every N optimizer steps; 0 disables interval saves')
    parser.add_argument('--preflight-only', action='store_true',
                        help='Validate selected split, image paths, train sampling and evaluation supports without loading the VLM')
    parser.add_argument('--resume', type=Path, help='Strict optimizer/scheduler/sampler resume')
    parser.add_argument('--init-from-checkpoint', type=Path,
                        help='Weight-only initialization for a new adaptation run')
    args = parser.parse_args()
    overrides = {}
    for section, key, value in (
        ('runtime', 'output_dir', str(args.output_dir) if args.output_dir else None),
        ('training', 'max_steps', args.max_steps), ('training', 'num_train_epochs', args.num_train_epochs),
        ('checkpoint', 'save_every', args.save_every)):
        if value is not None:
            overrides.setdefault(section, {})[key] = value
    if args.resume is not None and args.init_from_checkpoint is not None:
        parser.error('choose --resume or --init-from-checkpoint')
    if args.resume is not None:
        overrides['checkpoint_loading'] = {'resume_from': str(args.resume), 'init_from': None}
    elif args.init_from_checkpoint is not None:
        overrides['checkpoint_loading'] = {'init_from': str(args.init_from_checkpoint), 'resume_from': None}
    if args.semantic_max_new_tokens is not None:
        overrides['evaluation'] = {'predictions': {'semantic_max_new_tokens': args.semantic_max_new_tokens}}
    try:
        config = load_config(args.config, overrides)
        require_single_process(config)
        if config['data']['dataset'] in ('AiR', 'all') and config['data']['variant'] != 'all':
            raise ValueError('AiR/combined optimization Q_train requires variant=all')
        if config['evaluation']['strategy'] != 'no':
            resolve_prediction_settings(config)
    except ValueError as exc:
        parser.error(str(exc))
    loading = config['checkpoint_loading']
    resume_path = args.resume or (Path(loading['resume_from']) if loading['resume_from'] else None)
    init_path = args.init_from_checkpoint or (Path(loading['init_from']) if loading['init_from'] else None)
    if resume_path is not None and init_path is not None:
        parser.error('use either --resume or --init-from-checkpoint, not both')
    from semgaze.model.config import resolve_output_dir
    output_dir = resolve_output_dir(config)
    if not args.preflight_only and output_dir is not None and output_dir.exists() and any(output_dir.iterdir()):
        parser.error('choose an empty output directory; resume reads from --resume')
    if init_path is not None:
        # Fail incompatible stage transitions before loading the 8B model.
        previous = load_config(init_path / 'resolved_config.json')
        validate_staged_config(previous, config)
    data = config['data']
    raw, manifest, identity = read_persisted_splits(ROOT / data['split_root'], data_config=data)
    adapter_cls = {'all': JointAdapter, 'AiR': AirAdapter,
                   'COCO-Search18': CocoSearch18Adapter}[data['dataset']]
    adapter = adapter_cls(ROOT / data['images_root'], annotation_frame=data['annotation_frame'],
                          duration_field=data['duration']['source_field'])
    records = [adapter(r) for r in raw['train']]
    eval_subject_scope = config['evaluation'].get('subject_scope', 'unseen')
    test_rows = (raw['test'] if eval_subject_scope == 'unseen'
                 else raw['test_seen'] if eval_subject_scope == 'seen'
                 else raw['test'] + raw['test_seen'])
    test_records = [adapter(r) for r in test_rows]
    from collections import Counter
    from semgaze.data.joint import unseen_subject_ids
    counts_train = Counter(r.dataset for r in records)
    counts_test = Counter(r.dataset for r in test_records)
    expected_datasets = ({'AiR', 'COCO-Search18'} if data['dataset'] == 'all'
                         else {data['dataset']})
    if set(counts_train) != expected_datasets or set(counts_test) != expected_datasets:
        raise ValueError(f'dataset loader selected {set(counts_train)} train and '
                         f'{set(counts_test)} test; expected {expected_datasets}')
    if data['dataset'] == 'COCO-Search18' and data['variant'] in ('tp', 'ta'):
        expected_condition = {'tp': 'present', 'ta': 'absent'}[data['variant']]
        if any(r.condition != expected_condition for r in records + test_records):
            raise ValueError(f'COCO {data["variant"]} loader contains the wrong condition')
    unseen = unseen_subject_ids(data, manifest)
    eligible = Counter(r.dataset for r in records if r.subject not in unseen)
    print(f'[DATA] train={dict(counts_train)} eligible_queries={dict(eligible)} '
          f'test={dict(counts_test)}', flush=True)
    sampler = TrainingEpisodeSampler(records, config['experiment']['seed'], data_config=data)
    resolve_epoch_schedule(config, sampler.query_count)
    if args.preflight_only:
        from semgaze.data.fewshot import evaluation_episode
        from semgaze.evaluation.test import test_queries
        train_by_id = {r.record_id: r for r in records}
        if len(train_by_id) != len(records):
            raise ValueError('duplicate training record_id after conversion')
        # Train support construction is independent of held-out frozen draws.
        for dataset_name in sorted(expected_datasets):
            query = next((r for r in sampler.queries if r.dataset == dataset_name), None)
            if query is None:
                raise ValueError(f'{dataset_name}: no seen-subject optimization query')
            for k in data['fewshot']['k_values']:
                sampler.sample_for_query(query, k=k)
        scopes = ('unseen', 'seen') if eval_subject_scope == 'both' else (eval_subject_scope,)
        checked = 0
        # Check every held-out subject/dataset for every requested K and draw.
        # Small CPU workload: at most 29 subjects x 3 K x 10 draws.
        for scope in scopes:
            scope_records = [adapter(r) for r in raw['test' if scope == 'unseen' else 'test_seen']]
            queries = test_queries(train_by_id, scope_records, manifest, scope=scope)
            if not queries:
                raise ValueError(f'no {scope} held-out queries for dataset={data["dataset"]}')
            representative = {}
            for query in queries:
                representative.setdefault((query.dataset, query.subject), query)
            for query in representative.values():
                for k in sorted(set(config['evaluation']['k_values'] +
                                    (config['evaluation']['predictions'].get('where_k_values') or []) +
                                    (config['evaluation']['predictions'].get('semantic_k_values') or []))):
                    for draw_id in range(config['evaluation']['draw']):
                        evaluation_episode(query, train_by_id, manifest, k, draw_id=draw_id,
                                           unseen_subjects=manifest['unseen_subject_ids'])
                        checked += 1
        print(f'[PREFLIGHT] PASS | dataset={data["dataset"]} variant={data["variant"]} '
              f'| Q_train={sampler.query_count} | eval_support_episodes={checked} '
              f'| split_identity={identity} | VLM_NOT_LOADED', flush=True)
        return
    max_steps = config['training']['total_optimizer_updates']
    source_checkpoint = resume_path or init_path
    if source_checkpoint is not None:
        bundle = load_checkpoint_bundle(source_checkpoint, split_manifest_identity=identity, output_dir=output_dir, runtime=config['runtime'],
                                        precision=config['training']['precision'])
        # Load and verify the checkpoint in its original architecture first.
        if resume_path is not None:
            if bundle.config['model'] != config['model'] or bundle.config['where']['end_fix_token'] != config['where']['end_fix_token'] or bundle.config['where']['use_end_fix_token'] != config['where']['use_end_fix_token'] or bundle.config['semantic']['conditioning'] != config['semantic']['conditioning']:
                raise ValueError('strict resume model/token settings differ from checkpoint')
        if resume_path is not None and any(
            bundle.config['training'].get(name, {}).get('enabled', False)
            for name in ('scheduled_sampling', 'late_fixation_ce')
        ):
            raise ValueError('Cannot strictly resume a checkpoint trained with removed loss objectives; '
                             'use the original code or initialize weights with --init-from-checkpoint')
        if resume_path is not None:
            for key in ('lambda_where', 'lambda_sem'):
                if bundle.config['training'][key] != config['training'][key]:
                    raise ValueError(f'strict resume cannot change training.{key}; start a new run')
        if resume_path is not None and bundle.config['training']['optimizer'] != config['training']['optimizer']:
            raise ValueError('cannot restore optimizer moments into a different optimizer implementation')
        if resume_path is not None:
            for key in ('per_device_train_batch_size', 'gradient_accumulation_steps', 'scheduler',
                        'warmup_ratio', 'learning_rate', 'total_optimizer_updates', 'num_train_epochs'):
                if bundle.config['training'][key] != config['training'][key]:
                    raise ValueError(f'resume training.{key} differs; start a new run to change it')
        if init_path is not None:
            import random
            random.seed(config['experiment']['seed'])
            torch.manual_seed(config['experiment']['seed'])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config['experiment']['seed'])
            adapt_checkpoint_for_new_run(bundle, config, source_checkpoint=init_path)
        else:
            # Strict resume does not permit architecture/objective changes.
            bundle.config = config
            bundle.context_limit = config['where']['context']['max_total_sequence_length']
    else:
        bundle = build_flat_model_bundle(config=config, output_dir=output_dir)
    if source_checkpoint is not None:
        if config['training']['gradient_checkpointing']:
            checkpoint_kwargs = {'use_reentrant': False}
            if config['model']['mtl_lora']['enabled']:
                checkpoint_kwargs['context_fn'] = checkpoint_task_context_fn
            bundle.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
        else:
            bundle.model.gradient_checkpointing_disable()
    bundle.config['evaluation'] = config['evaluation']
    bundle.config['runtime'].update(split_manifest_identity=identity, world_size=1)
    output_dir = bundle.output_dir
    write_run_config(bundle.config, output_dir)
    if init_path is not None:
        (output_dir / 'preflight.json').write_text(
            json.dumps(bundle.diagnostics, indent=2), encoding='utf-8')
    bundle.optimizer = make_optimizer(bundle)
    bundle.scheduler = make_scheduler(bundle, max_steps)
    start = 0
    if resume_path is not None:
        start = restore_checkpoint_state(bundle, resume_path, split_manifest_identity=identity,
                                          sampler=sampler, resume_optimizer=True)
    run_training_loop(bundle, sampler, {r.record_id: r for r in records}, test_records, manifest,
        split_manifest_identity=identity, max_steps=max_steps, save_every=config['checkpoint']['save_every'], start=start)


if __name__ == '__main__':
    main()
