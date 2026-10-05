"""Export frozen-episode predictions and validity; metric choices remain gated."""
import argparse
import json
from pathlib import Path
from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter
from semgaze.data.fewshot import frozen_episode
from semgaze.model.checkpoint import load_checkpoint_bundle
from semgaze.model.config import ROOT, load_config, write_run_config
from semgaze.evaluation.predictions import prediction_batches
from semgaze.evaluation.validation import validation_mode
from semgaze.model.visual_cache import InferenceVisualCache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--config', type=Path, help='Evaluation/runtime overrides of checkpoint configuration')
    parser.add_argument('--split', choices=('validation', 'test'), required=True)
    parser.add_argument('--path', choices=('where', 'semantic', 'both'), required=True)
    parser.add_argument('--semantic-max-new-tokens', type=int,
                        help='Must be explicitly selected; canonical semantic decoding budget is not frozen')
    args = parser.parse_args()
    config = load_config(args.checkpoint / 'resolved_config.json')
    if args.config:
        import yaml
        overrides = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        if set(overrides) - {'runtime', 'evaluation', 'validation'}:
            parser.error('evaluation overrides may change runtime/evaluation/validation; model/data settings belong to the checkpoint')
        from semgaze.model.config import resolve_config
        config = resolve_config(config, overrides)
    if args.semantic_max_new_tokens is not None:
        config['evaluation']['predictions']['semantic_max_new_tokens'] = args.semantic_max_new_tokens
    budget = config['evaluation']['predictions']['semantic_max_new_tokens']
    if args.path in ('semantic', 'both') and (type(budget) is not int or budget <= 0):
        parser.error('declare evaluation.predictions.semantic_max_new_tokens or --semantic-max-new-tokens')
    output_dir = args.output_dir or (Path(config['runtime']['output_dir']) if config['runtime']['output_dir'] else None)
    if output_dir is None:
        parser.error('declare runtime.output_dir or --output-dir')
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error('choose an empty output directory')
    data = config['data']
    raw, manifest, identity = read_persisted_splits(ROOT / data['split_root'], data_config=data)
    bundle = load_checkpoint_bundle(args.checkpoint, split_manifest_identity=identity, output_dir=output_dir,
                                    runtime=config['runtime'])
    config['runtime']['output_dir'] = str(output_dir.resolve())
    bundle.config = config
    write_run_config(config, output_dir)
    adapter = CocoSearch18Adapter(ROOT / data['images_root'], annotation_frame=data['annotation_frame'],
                                  duration_field=data['duration']['source_field'])
    train = {r['record_id']: adapter(r) for r in raw['train']}
    queries = [adapter(r) for r in raw[args.split] if
               (r['subject'] in data['unseen_subjects']) == (args.split == 'test')]
    policy = {'split': args.split, 'path': args.path, 'semantic_max_new_tokens': budget,
              'do_sample': False, 'split_manifest_identity': identity,
              'report_kind': 'prediction_and_validity_diagnostics',
              'metrics_gate': 'SM/MM/SED implementations and invalid treatment, semantic metrics and selection scalar require explicit protocol choices'}
    (output_dir / 'evaluation_config.json').write_text(json.dumps(policy, indent=2), encoding='utf-8')
    summary = {}
    cache_settings = config['validation']['cache']
    cache = InferenceVisualCache(preprocessing=cache_settings['support_preprocessing'],
        features=cache_settings['frozen_visual_features'], max_entries=cache_settings['max_entries'])
    with validation_mode(bundle), (output_dir / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for k in config['evaluation']['k_values']:
            draws = range(len(manifest['support_draws'][str(k)])) if args.split == 'test' else (None,)
            draw_rates = []
            for draw_id in draws:
                counts = {'queries': 0}
                if args.path in ('where', 'both'):
                    counts['where_under_generated'] = 0
                if args.path in ('semantic', 'both'):
                    counts['flat_format_valid'] = 0
                episodes = [frozen_episode(query, train, manifest, k, draw_id=draw_id,
                            unseen_subjects=data['unseen_subjects']) for query in queries]
                for episode, generated in prediction_batches(bundle, episodes, budget=budget, cache=cache, path=args.path):
                    query = episode.query
                    result = {'record_id': query.record_id, 'subject': query.subject, 'k': k, 'draw_id': draw_id,
                              'support_ids': [s.record_id for s in episode.supports]}
                    if args.path in ('where', 'both'):
                        result['where'] = generated[0]
                        counts['where_under_generated'] += int(result['where']['under_generated'])
                    if args.path in ('semantic', 'both'):
                        result['semantic'] = generated[1]
                        counts['flat_format_valid'] += int(result['semantic']['flat_format_valid'])
                    counts['queries'] += 1
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                if not counts['queries']:
                    raise ValueError('no eligible evaluation queries')
                rates = {key + '_rate': value / counts['queries'] for key, value in counts.items() if key != 'queries'}
                draw_rates.append(rates)
                summary[f'K={k},draw={draw_id}'] = counts | rates
            summary[f'K={k},draw_mean'] = {key: sum(r[key] for r in draw_rates) / len(draw_rates) for key in draw_rates[0]}
    cache.close()
    (output_dir / 'validity_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
