"""Train flat_single_output only after a released-model fixture gradient smoke."""
import argparse
import json
from pathlib import Path
import torch
from transformers import get_cosine_schedule_with_warmup
from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter
from semgaze.data.fewshot import TrainingEpisodeSampler
from semgaze.data.schema import normalized_episode_from_dict, UNSEEN_SUBJECTS
from semgaze.model.build import build_flat_model_bundle
from semgaze.model.config import ROOT, load_config, write_run_config
from semgaze.model.checkpoint import load_checkpoint_bundle, restore_checkpoint_state
from semgaze.training.flat_step import run_flat_training_step, make_optimizer
from semgaze.training.loop import resolve_epoch_schedule, run_training_loop
from semgaze.evaluation.predictions import resolve_prediction_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/flat_single.yaml')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, required=True, help='Explicit optimizer-step training horizon')
    parser.add_argument('--steps-per-epoch', type=int, help='Optimizer steps per epoch; overrides training.steps_per_epoch')
    parser.add_argument('--semantic-max-new-tokens', type=int, help='Explicit epoch semantic generation budget')
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.max_steps <= 0 or args.save_every <= 0:
        parser.error('max-steps and save-every must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('choose an empty output directory; resume reads from --resume')
    config = load_config(args.config)
    try:
        steps_per_epoch = resolve_epoch_schedule(config, args.max_steps, args.steps_per_epoch)
        resolve_prediction_settings(config, args.semantic_max_new_tokens)
    except ValueError as exc:
        parser.error(str(exc))
    raw, manifest, identity = read_persisted_splits()
    adapter = CocoSearch18Adapter(annotation_frame=config['data'].get('annotation_frame'))
    records = [adapter(r) for r in raw['train'] if r['subject'] not in UNSEEN_SUBJECTS]
    validation_records = [adapter(r) for r in raw['validation'] if r['subject'] not in UNSEEN_SUBJECTS]
    sampler = TrainingEpisodeSampler(records, config['experiment']['seed'])
    if args.resume:
        bundle = load_checkpoint_bundle(args.resume, split_manifest_identity=identity, output_dir=args.output_dir)
        # A resume config may not silently replace the checkpoint's training protocol.
        from semgaze.model.config import resolve_config
        for section in ('experiment', 'data', 'model', 'where', 'state', 'semantic', 'training', 'evaluation'):
            if bundle.config[section] != resolve_config(config)[section]:
                raise ValueError(f'resume config differs in {section}')
        saved_horizon = bundle.config['runtime']['max_steps']
        if saved_horizon != args.max_steps:
            raise ValueError('resume must preserve scheduler training horizon')
    else:
        bundle = build_flat_model_bundle(args.config, output_dir=args.output_dir)
    bundle.config['training']['steps_per_epoch'] = steps_per_epoch
    bundle.config['evaluation'] = config['evaluation']
    bundle.config['runtime'].update(max_steps=args.max_steps, save_every=args.save_every,
                                    split_manifest_identity=identity, world_size=1)
    write_run_config(bundle.config, args.output_dir)
    fixture = normalized_episode_from_dict(json.loads((ROOT / 'tests/fixtures/flat_episode.json').read_text()))
    smoke = run_flat_training_step(bundle, fixture)  # never updates parameters
    (args.output_dir / 'gradient_smoke.json').write_text(json.dumps(
        {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in smoke.items()}, indent=2))
    bundle.optimizer = make_optimizer(bundle)
    t = bundle.config['training']
    bundle.scheduler = get_cosine_schedule_with_warmup(bundle.optimizer,
        num_warmup_steps=int(args.max_steps * t['warmup_ratio']), num_training_steps=args.max_steps)
    start = 0
    if args.resume:
        start = restore_checkpoint_state(bundle, args.resume, split_manifest_identity=identity,
                                          sampler=sampler, resume_optimizer=True)
    run_training_loop(bundle, sampler, {r.record_id: r for r in records}, validation_records, manifest,
        split_manifest_identity=identity, max_steps=args.max_steps, save_every=args.save_every, start=start)


if __name__ == '__main__':
    main()
