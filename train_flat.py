"""Train flat_single_output only after a released-model fixture gradient smoke."""
import argparse
import json
from pathlib import Path
import torch
from transformers import get_cosine_schedule_with_warmup
from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter
from semgaze.data.fewshot import TrainingEpisodeSampler
from semgaze.data.schema import normalized_episode_from_dict
from semgaze.model.build import build_flat_model_bundle
from semgaze.model.config import ROOT, load_config, write_run_config
from semgaze.model.checkpoint import save_checkpoint, load_checkpoint_bundle, restore_checkpoint_state
from semgaze.training.flat_step import run_flat_training_step, make_optimizer
from semgaze.where.collator import WhereContextOverflowError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/flat_single.yaml')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-steps', type=int, required=True, help='Explicit optimizer-step training horizon')
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.max_steps <= 0 or args.save_every <= 0:
        parser.error('max-steps and save-every must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('choose an empty output directory; resume reads from --resume')
    config = load_config(args.config)
    raw, manifest, identity = read_persisted_splits()
    adapter = CocoSearch18Adapter(annotation_frame=config['data'].get('annotation_frame'))
    records = [adapter(r) for r in raw['train']]
    sampler = TrainingEpisodeSampler(records, config['experiment']['seed'])
    if args.resume:
        bundle = load_checkpoint_bundle(args.resume, split_manifest_identity=identity, output_dir=args.output_dir)
        # A resume config may not silently replace the checkpoint's training protocol.
        from semgaze.model.config import resolve_config
        for section in ('experiment', 'data', 'model', 'where', 'state', 'semantic', 'training'):
            if bundle.config[section] != resolve_config(config)[section]:
                raise ValueError(f'resume config differs in {section}')
        saved_horizon = bundle.config['runtime']['max_steps']
        if saved_horizon != args.max_steps:
            raise ValueError('resume must preserve scheduler training horizon')
    else:
        bundle = build_flat_model_bundle(args.config, output_dir=args.output_dir)
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
    episodes_per_step = t['per_device_train_batch_size'] * t['gradient_accumulation_steps']
    if type(episodes_per_step) is not int or episodes_per_step < 1:
        raise ValueError('positive episode batch/accumulation counts required')
    for step in range(start, args.max_steps):
        losses, rejected = [], 0
        for micro in range(episodes_per_step):
            while True:
                episode = sampler.sample()
                try:
                    result = run_flat_training_step(bundle, episode, zero_grad=(micro == 0),
                        loss_scale=1 / episodes_per_step)
                    losses.append(float(result['loss_total']))
                    break
                except WhereContextOverflowError:
                    rejected += 1  # resample the entire episode under the same probability law
                    if rejected >= 1000:
                        raise RuntimeError('1000 complete WHERE episode overflows; inspect context feasibility')
        torch.nn.utils.clip_grad_norm_(bundle.trainable_parameters(), t['max_grad_norm'], error_if_nonfinite=True)
        bundle.optimizer.step()
        bundle.scheduler.step()
        print(json.dumps({'step': step + 1, 'loss_total': sum(losses) / len(losses), 'rejected_where_episodes': rejected}), flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == args.max_steps:
            save_checkpoint(bundle, args.output_dir / f'checkpoint-{step + 1}',
                split_manifest_identity=identity, step=step + 1, sampler=sampler)


if __name__ == '__main__':
    main()
