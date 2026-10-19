"""Run zero-shot or LoRA-SFT semantic baseline inference and canonical scoring."""
import argparse
from pathlib import Path
import yaml

from semgaze.model.config import ROOT
from semgaze.semantic.baselines.data import load_queries
from semgaze.semantic.baselines.inference import predict
from semgaze.semantic.baselines.model import BASE_MODEL, load_baseline_model
from semgaze.semantic.baselines.scoring import dump_json, scoring_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/semantic_zero_shot.yaml')
    p.add_argument('--semgaze-config', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--method', choices=('semantic_zero_shot', 'semantic_sft'))
    p.add_argument('--prompt-style', choices=('origin', 'concise'))
    p.add_argument('--checkpoint', type=Path, help='LoRA adapter directory from semantic-only SFT')
    p.add_argument('--images-root')
    p.add_argument('--split-root')
    p.add_argument('--subject-scope', choices=('seen', 'unseen', 'both'),
                   help='Override config evaluation.subject_scope')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--max-new-tokens', type=int)
    p.add_argument('--device')
    p.add_argument('--precision', choices=('bf16', 'fp32'))
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    scope = args.subject_scope or cfg.get('evaluation', {}).get('subject_scope', 'unseen')
    if scope not in ('seen', 'unseen', 'both'):
        p.error('evaluation.subject_scope must be seen, unseen or both')
    if scope == 'both':
        _run_both_baseline_scopes(args)
        return
    method = args.method or cfg.get('method', 'semantic_zero_shot')
    style = args.prompt_style or cfg.get('prompt_style', 'origin')
    if (method == 'semantic_sft') != (args.checkpoint is not None):
        p.error('SFT evaluation needs --checkpoint; zero-shot must not load a checkpoint')
    source = args.semgaze_config or ROOT / cfg.get('semgaze_config', 'configs/flat_single.yaml')
    overrides = dict(cfg.get('data', {}))
    if args.images_root:
        overrides['images_root'] = args.images_root
    if args.split_root:
        overrides['split_root'] = args.split_root
    _, queries, identity, data = load_queries(source, data_overrides=overrides,
                                               subject_scope=scope)
    precision = args.precision or cfg.get('precision', 'bf16')
    device = args.device or cfg.get('device', 'cuda')
    ckpt = None
    if method == 'semantic_sft':
        import json
        ckpt = json.loads((args.checkpoint / 'semantic_sft_checkpoint.json').read_text(encoding='utf-8'))
        if ckpt['split_manifest_identity'] != identity:
            p.error('SFT checkpoint split identity does not match evaluation data')
        if ckpt['prompt_style'] != style:
            p.error('matched-style SFT requires evaluating with training prompt_style')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    resolved = {'method': method, 'prompt_style': style, 'model_id': BASE_MODEL,
                'checkpoint': str(args.checkpoint) if args.checkpoint else None,
                'split_manifest_identity': identity, 'data': data,
                'subject_scope': scope,
                'source_split_json': str(ROOT / data['split_root'] /
                                         ('test_seen.json' if scope == 'seen' else 'test.json')),
                'precision': precision, 'device': device, 'metrics': cfg.get('metrics', {})}
    old = output / 'resolved_config.json'
    if args.resume and old.exists():
        import json
        if json.loads(old.read_text()) != resolved:
            p.error('resumed evaluation data/model configuration differs')
    if old.exists():
        import json
        if json.loads(old.read_text()) != resolved:
            p.error('existing output directory has a different resolved configuration')
    dump_json(old, resolved)
    model, processor, count = load_baseline_model(
        method=method, device=device, precision=precision, adapter_path=args.checkpoint)
    budget = args.max_new_tokens or cfg.get('max_new_tokens', 512)
    rows = predict(model, processor, queries, method=method, prompt_style=style, output_dir=output,
                   max_new_tokens=budget, batch_size=args.batch_size or cfg.get('batch_size', 1),
                   resume=args.resume, precision=precision, model_id=BASE_MODEL,
                   run_info=({'checkpoint': str(args.checkpoint),
                              'epoch': ckpt['checkpoint_epoch'],
                              'optimizer_step': ckpt['checkpoint_optimizer_step'],
                              'trainable_parameter_count': ckpt['trainable_parameters']} if ckpt else {}))
    scoring_report(rows, queries, cfg.get('metrics', {}), output, method=method)
    dump_json(output / 'evaluation_provenance.json', {
        **resolved, 'trainable_parameter_count': count,
        'query_count': len(queries), 'decode': 'greedy', 'max_new_tokens': budget,
        'reference_type': 'identical held-out normalized GT WHAT/WHY/HOW'})
    print('Baseline evaluation complete:', output, '| queries=', len(rows))


def _run_both_baseline_scopes(args):
    """Evaluate both subject groups into disjoint output directories."""
    import json
    import subprocess
    import sys
    dest = args.output_dir.expanduser().resolve()
    if dest.exists() and any(dest.iterdir()):
        raise ValueError('both baseline evaluation requires an empty output directory')
    dest.mkdir(parents=True, exist_ok=True)
    for role in ('unseen', 'seen'):
        cmd = [sys.executable, str(Path(__file__).resolve()),
               '--config', str(args.config), '--output-dir', str(dest / role),
               '--subject-scope', role]
        for argname, value in (
            ('--semgaze-config', args.semgaze_config),
            ('--method', args.method), ('--prompt-style', args.prompt_style),
            ('--checkpoint', args.checkpoint), ('--images-root', args.images_root),
            ('--split-root', args.split_root), ('--batch-size', args.batch_size),
            ('--max-new-tokens', args.max_new_tokens), ('--device', args.device),
            ('--precision', args.precision)):
            if value is not None:
                cmd.extend([argname, str(value)])
        if args.resume:
            cmd.append('--resume')
        subprocess.run(cmd, check=True)
    (dest / 'evaluation_provenance.json').write_text(json.dumps({
        'subject_scope': 'both', 'sources': {
            role: str(dest / role / 'evaluation_provenance.json')
            for role in ('unseen', 'seen')},
        'note': 'separate metrics for distinct heldout subjects',
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
