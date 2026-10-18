"""Evaluate a previously trained semantic-only InternVL LoRA checkpoint.

This entrypoint NEVER trains or mutates the adapter. It delegates all generation,
query validation, and canonical WHAT/WHY/HOW scoring to the existing
``evaluate_semantic_baseline.py`` evaluator.

Minimal usage:
    python evaluate_semantic_sft.py --checkpoint runs/semsft_origin/final

The evaluation directory defaults to the sibling ``eval_<checkpoint-name>``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='SFT output/final or checkpoint-step-N directory containing a PEFT adapter')
    parser.add_argument('--output-dir', type=Path,
                        help='Fresh evaluation directory (default: next to checkpoint)')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/semantic_sft.yaml')
    parser.add_argument('--semgaze-config', type=Path)
    parser.add_argument('--prompt-style', choices=('origin', 'concise'),
                        help='Inferred from saved checkpoint metadata by default')
    parser.add_argument('--images-root')
    parser.add_argument('--split-root')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--precision', choices=('bf16', 'fp32'))
    parser.add_argument('--device')
    parser.add_argument('--resume', action='store_true',
                        help='Resume an interrupted evaluation only when its settings match')
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        parser.error(f'checkpoint is not a directory: {checkpoint}')
    metadata_file = checkpoint / 'semantic_sft_checkpoint.json'
    if not metadata_file.is_file():
        parser.error(f'missing semantic SFT checkpoint metadata: {metadata_file}')
    if not (checkpoint / 'adapter_config.json').is_file():
        parser.error('checkpoint has no adapter_config.json (expected a PEFT LoRA adapter)')
    if not any((checkpoint / name).is_file() for name in
               ('adapter_model.safetensors', 'adapter_model.bin')):
        parser.error('checkpoint has no adapter_model.safetensors / adapter_model.bin')
    try:
        metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        parser.error(f'invalid semantic SFT checkpoint metadata: {exc}')
    if metadata.get('method') != 'semantic_sft':
        parser.error('checkpoint metadata method is not semantic_sft')
    if not metadata.get('split_manifest_identity'):
        parser.error('checkpoint is missing split_manifest_identity')
    training_style = metadata.get('prompt_style')
    if training_style not in ('origin', 'concise'):
        parser.error('checkpoint has no valid training prompt_style')
    if args.prompt_style is not None and args.prompt_style != training_style:
        parser.error(f'prompt_style differs from checkpoint training style ({training_style})')

    output = (args.output_dir if args.output_dir is not None
              else checkpoint.parent / f'eval_{checkpoint.name}').expanduser().resolve()
    if output == checkpoint or checkpoint in output.parents or output == checkpoint.parent:
        parser.error('evaluation output must not overwrite the checkpoint or its directory')
    if not args.config.is_file():
        parser.error(f'baseline config not found: {args.config}')
    evaluator = ROOT / 'evaluate_semantic_baseline.py'
    if not evaluator.is_file():
        parser.error(f'existing canonical evaluator not found: {evaluator}')

    command = [sys.executable, str(evaluator),
               '--config', str(args.config),
               '--method', 'semantic_sft',
               '--checkpoint', str(checkpoint),
               '--prompt-style', training_style,
               '--output-dir', str(output)]
    for name in ('semgaze_config', 'images_root', 'split_root', 'batch_size',
                 'max_new_tokens', 'precision', 'device'):
        value = getattr(args, name)
        if value is not None:
            command.extend(['--' + name.replace('_', '-'), str(value)])
    if args.resume:
        command.append('--resume')

    # The canonical evaluator enforces exact split identity before loading 8B
    # weights. Do not silently bypass this check for changed data manifests.
    return subprocess.run(command, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
