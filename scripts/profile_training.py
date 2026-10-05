"""Explicit profile: tiny CPU engineering benchmark or configured released model.

Use the same episodes, seed, warmup and measurement count for each variant.
The baseline uses the original full-logit model path (see throughput_reference).
"""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from semgaze.training.profiling import TrainingProfiler, runtime_metadata
from semgaze.training.flat_step import make_optimizer, make_scheduler, clip_and_check_gradients, run_flat_training_step
from semgaze.data.schema import normalized_episode_from_dict
from throughput_reference import reference_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/flat_throughput.yaml')
    parser.add_argument('--variant', choices=['baseline', 'optimized'], default='baseline')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--checkpointing', choices=['on', 'off'], default='on')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workload', type=Path, help='JSON array of normalized episodes; identical ordered workload for comparisons')
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1 or args.warmup < 0:
        parser.error('batch-size/steps must be positive and warmup nonnegative')
    episode = normalized_episode_from_dict(json.loads((ROOT / 'tests/fixtures/flat_episode.json').read_text()))
    episodes = ([normalized_episode_from_dict(e) for e in json.loads(args.workload.read_text())]
                if args.workload else [episode] * args.batch_size)
    if len(episodes) != args.batch_size:
        parser.error('workload must contain exactly batch-size episodes')
    if args.tiny:
        from test_model_path import tiny_bundle
        bundle = tiny_bundle(args.output.parent, episode)
    else:
        from semgaze.model.build import build_flat_model_bundle, ModelPreflightError
        from semgaze.model.config import load_config
        config = load_config(args.config)
        if (config['model']['base_model'] != 'OpenGVLab/InternVL3_5-8B-HF'
                or config['model']['initialization_adapter'] != 'DeepGaze-VL/model/visual_search_adapter'
                or config['training']['precision'] != 'bf16'):
            parser.error('acceptance requires InternVL3_5-8B-HF, released DeepGaze adapter and bf16; use flat_throughput.yaml')
        try:
            bundle = build_flat_model_bundle(config=config)
        except ModelPreflightError as exc:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'status': 'blocked', 'reason': str(exc),
                'cuda_available': torch.cuda.is_available(), 'torch_version': torch.__version__}, indent=2))
            raise SystemExit(str(exc))
    if args.checkpointing == 'on':
        bundle.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        bundle.model.gradient_checkpointing_disable()
    bundle.optimizer = make_optimizer(bundle)
    bundle.scheduler = make_scheduler(bundle, args.warmup + args.steps)
    profiler = TrainingProfiler(bundle.input_row.device)
    metadata = []
    for step in range(args.warmup + args.steps):
        if step == args.warmup:
            profiler.times.clear()
            if profiler.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(profiler.device)
        with profiler.stage('total'):
            metadata = []
            if args.variant == 'baseline':
                for b in range(args.batch_size):
                    result = reference_step(bundle, episodes[b], zero_grad=b == 0,
                        loss_scale=1 / args.batch_size, profiler=profiler)
                    metadata.extend(result['episodes'])
            else:
                result = run_flat_training_step(bundle, episodes,
                                               diagnostics=False, profiler=profiler)
                metadata = result['episodes']
            with profiler.stage('optimizer'):
                clip_and_check_gradients(bundle)
                bundle.optimizer.step()
                bundle.scheduler.step()
    seconds = profiler.times['total']
    report = dict(variant=args.variant, physical_batch_size=1 if args.variant == 'baseline' else args.batch_size,
        optimizer_episode_count=args.batch_size, tiny_model=args.tiny, device=str(profiler.device),
        measured_steps=args.steps, warmup_steps=args.warmup, checkpointing=args.checkpointing,
        K=[len(e.supports) for e in episodes], fixations=[len(e.query.x_px) for e in episodes],
        images=[len(e.supports)+1 for e in episodes],
        episodes_per_second=args.steps * args.batch_size / seconds,
        seconds_per_episode=seconds / (args.steps * args.batch_size),
        stage_seconds=profiler.times, memory_bytes=profiler.memory(),
        episodes=metadata, runtime=runtime_metadata(bundle))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
