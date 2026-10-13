"""Train a non-personalized semantic-only InternVL LoRA baseline."""
import argparse
from pathlib import Path
import yaml
from semgaze.model.config import ROOT
from semgaze.semantic.baselines.data import load_queries
from semgaze.semantic.baselines.model import BASE_MODEL, load_baseline_model
from semgaze.semantic.baselines.scoring import dump_json
from semgaze.semantic.baselines.training import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/semantic_sft.yaml')
    p.add_argument('--semgaze-config', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--prompt-style', choices=('origin', 'concise'))
    p.add_argument('--images-root')
    p.add_argument('--split-root')
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--effective-batch-size', type=int)
    p.add_argument('--gradient-accumulation-steps', type=int)
    p.add_argument('--learning-rate', type=float)
    p.add_argument('--warmup-ratio', type=float)
    p.add_argument('--seed', type=int)
    p.add_argument('--save-every', type=int)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--device')
    p.add_argument('--precision', choices=('bf16', 'fp32'))
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if cfg.get('method') != 'semantic_sft':
        p.error('SFT requires method: semantic_sft')
    style = args.prompt_style or cfg.get('prompt_style', 'origin')
    source = args.semgaze_config or ROOT / cfg.get('semgaze_config', 'configs/flat_single.yaml')
    overrides = dict(cfg.get('data', {}))
    if args.images_root:
        overrides['images_root'] = args.images_root
    if args.split_root:
        overrides['split_root'] = args.split_root
    records, _, identity, _ = load_queries(source, data_overrides=overrides)
    batch_size = args.batch_size or cfg.get('batch_size', 1)
    accum = args.gradient_accumulation_steps or cfg.get('gradient_accumulation_steps', 8)
    if args.effective_batch_size is not None:
        if args.gradient_accumulation_steps is not None or args.effective_batch_size % batch_size:
            p.error('effective batch size must divide by physical batch size; choose either effective or accumulation')
        accum = args.effective_batch_size // batch_size
    # Seed BEFORE LoRA construction, not only before the data-shuffle loop.
    seed = args.seed if args.seed is not None else cfg.get('seed', 42)
    import random
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = args.device or cfg.get('device', 'cuda')
    precision = args.precision or cfg.get('precision', 'bf16')
    model, processor, count = load_baseline_model(
        method='semantic_sft', device=device, precision=precision, training=True,
        gradient_checkpointing=cfg.get('gradient_checkpointing', True))
    report = train(model, processor, records, identity=identity, output_dir=args.output_dir,
                   prompt_style=style, epochs=args.epochs or cfg.get('epochs', 2),
                   batch_size=batch_size, accumulation=accum,
                   lr=args.learning_rate or cfg.get('learning_rate', 1e-4),
                   warmup_ratio=args.warmup_ratio if args.warmup_ratio is not None else cfg.get('warmup_ratio', 0.1),
                   seed=seed,
                   save_every=args.save_every if args.save_every is not None else cfg.get('save_every', 100),
                   max_steps=args.max_steps)
    report.update(model_id=BASE_MODEL, precision=precision, device=device,
                  trainable_parameter_count=count)
    dump_json(args.output_dir / 'run_metadata.json', report)
    print('SFT complete:', report['checkpoint'], 'optimizer updates:', report['optimizer_step'])


if __name__ == '__main__':
    main()
