"""Train the configured flat model after its fixture gradient smoke."""
import argparse
import json
from pathlib import Path
import torch
from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter
from semgaze.data.joint import JointAdapter
from semgaze.data.fewshot import TrainingEpisodeSampler
from semgaze.data.schema import normalized_episode_from_dict
from semgaze.model.build import build_flat_model_bundle
from semgaze.model.config import ROOT, load_config, write_run_config
from semgaze.model.checkpoint import load_checkpoint_bundle, restore_checkpoint_state
from semgaze.training.flat_step import run_flat_training_step, make_optimizer, make_scheduler
from semgaze.training.loop import resolve_epoch_schedule, run_training_loop, require_single_process
from semgaze.evaluation.predictions import resolve_prediction_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/flat_single.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-steps', type=int, help='Debug/smoke optimizer-update cap; may stop mid-epoch')
    parser.add_argument('--semantic-max-new-tokens', type=int, help='Explicit epoch semantic generation budget')
    parser.add_argument('--num-train-epochs', '--epochs', dest='num_train_epochs', type=int)
    parser.add_argument('--save-every', type=int, help='Save every N optimizer steps; 0 disables interval saves')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    overrides = {}
    for section, key, value in (
        ('runtime', 'output_dir', str(args.output_dir) if args.output_dir else None),
        ('training', 'max_steps', args.max_steps), ('training', 'num_train_epochs', args.num_train_epochs),
        ('checkpoint', 'save_every', args.save_every)):
        if value is not None:
            overrides.setdefault(section, {})[key] = value
    if args.semantic_max_new_tokens is not None:
        overrides['evaluation'] = {'predictions': {'semantic_max_new_tokens': args.semantic_max_new_tokens}}
    try:
        config = load_config(args.config, overrides)
        require_single_process(config)
        if config['data']['variant'] != 'all':
            raise ValueError('optimization Q_train requires the all-variant train.json')
        if config['evaluation']['strategy'] != 'no':
            resolve_prediction_settings(config)
    except ValueError as exc:
        parser.error(str(exc))
    from semgaze.model.config import resolve_output_dir
    output_dir = resolve_output_dir(config)
    if output_dir is not None and output_dir.exists() and any(output_dir.iterdir()):
        parser.error('choose an empty output directory; resume reads from --resume')
    data = config['data']
    raw, manifest, identity = read_persisted_splits(ROOT / data['split_root'], data_config=data)
    adapter_cls = JointAdapter if data['dataset'] == 'all' else CocoSearch18Adapter
    adapter = adapter_cls(ROOT / data['images_root'], annotation_frame=data['annotation_frame'],
                          duration_field=data['duration']['source_field'])
    records = [adapter(r) for r in raw['train']]
    test_records = [adapter(r) for r in raw['test']]
    from collections import Counter
    from semgaze.data.joint import unseen_subject_ids
    counts_train = Counter(r.dataset for r in records)
    counts_test = Counter(r.dataset for r in test_records)
    if data['dataset'] == 'all' and (set(counts_train) != {'AiR', 'COCO-Search18'}
                                    or set(counts_test) != {'AiR', 'COCO-Search18'}):
        raise ValueError('joint dataset loader omitted AiR or COCO-Search18')
    unseen = unseen_subject_ids(data, manifest)
    eligible = Counter(r.dataset for r in records if r.subject not in unseen)
    print(f'[DATA] train={dict(counts_train)} eligible_queries={dict(eligible)} '
          f'test={dict(counts_test)}', flush=True)
    sampler = TrainingEpisodeSampler(records, config['experiment']['seed'], data_config=data)
    resolve_epoch_schedule(config, sampler.query_count)
    max_steps = config['training']['total_optimizer_updates']
    if args.resume:
        bundle = load_checkpoint_bundle(args.resume, split_manifest_identity=identity, output_dir=output_dir, runtime=config['runtime'],
                                        precision=config['training']['precision'])
        # A resume retains optimizer schedule and episode-window geometry.
        if bundle.config['model'] != config['model'] or bundle.config['where']['end_fix_token'] != config['where']['end_fix_token']:
            raise ValueError('resume model/token settings differ from saved weight structure; initialize a new run instead')
        if bundle.config['training']['optimizer'] != config['training']['optimizer']:
            raise ValueError('cannot restore optimizer moments into a different optimizer implementation')
        for key in ('per_device_train_batch_size', 'gradient_accumulation_steps', 'scheduler',
                    'warmup_ratio', 'learning_rate', 'total_optimizer_updates', 'num_train_epochs'):
            if bundle.config['training'][key] != config['training'][key]:
                raise ValueError(f'resume training.{key} differs; start a new run to change it')
        bundle.config = config
        bundle.context_limit = config['where']['context']['max_total_sequence_length']
    else:
        bundle = build_flat_model_bundle(config=config, output_dir=output_dir)
    if args.resume:
        if config['training']['gradient_checkpointing']:
            bundle.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
            bundle.model.gradient_checkpointing_disable()
    bundle.config['evaluation'] = config['evaluation']
    bundle.config['runtime'].update(split_manifest_identity=identity, world_size=1)
    output_dir = bundle.output_dir
    write_run_config(bundle.config, output_dir)
    fixture = normalized_episode_from_dict(json.loads((ROOT / config['smoke']['fixture']).read_text()))
    smoke = run_flat_training_step(bundle, fixture)  # never updates parameters
    (output_dir / 'gradient_smoke.json').write_text(json.dumps(
        {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in smoke.items()}, indent=2))
    bundle.optimizer = make_optimizer(bundle)
    bundle.scheduler = make_scheduler(bundle, max_steps)
    start = 0
    if args.resume:
        start = restore_checkpoint_state(bundle, args.resume, split_manifest_identity=identity,
                                          sampler=sampler, resume_optimizer=True)
    run_training_loop(bundle, sampler, {r.record_id: r for r in records}, test_records, manifest,
        split_manifest_identity=identity, max_steps=max_steps, save_every=config['checkpoint']['save_every'], start=start)


if __name__ == '__main__':
    main()
