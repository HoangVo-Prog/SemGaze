"""Export frozen-episode predictions and validity; metric choices remain gated."""
import argparse
import json
from pathlib import Path
from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter
from semgaze.data.fewshot import frozen_episode
from semgaze.data.schema import UNSEEN_SUBJECTS
from semgaze.model.checkpoint import load_checkpoint_bundle
from semgaze.evaluation.where import evaluate_where_episode
from semgaze.evaluation.flat import evaluate_flat_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), required=True)
    parser.add_argument('--path', choices=('where', 'semantic', 'both'), required=True)
    parser.add_argument('--semantic-max-new-tokens', type=int,
                        help='Must be explicitly selected; canonical semantic decoding budget is not frozen')
    args = parser.parse_args()
    if args.path in ('semantic', 'both') and (args.semantic_max_new_tokens is None or args.semantic_max_new_tokens <= 0):
        parser.error('declare --semantic-max-new-tokens for semantic generation')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('choose an empty output directory')
    raw, manifest, identity = read_persisted_splits()
    bundle = load_checkpoint_bundle(args.checkpoint, split_manifest_identity=identity, output_dir=args.output_dir)
    adapter = CocoSearch18Adapter(annotation_frame=bundle.config['data'].get('annotation_frame'))
    train = {r['record_id']: adapter(r) for r in raw['train']}
    queries = [adapter(r) for r in raw[args.split] if
               (r['subject'] in UNSEEN_SUBJECTS) == (args.split == 'test')]
    policy = {'split': args.split, 'path': args.path, 'semantic_max_new_tokens': args.semantic_max_new_tokens,
              'do_sample': False, 'split_manifest_identity': identity,
              'report_kind': 'prediction_and_validity_diagnostics',
              'metrics_gate': 'SM/MM/SED implementations and invalid treatment, semantic metrics and selection scalar require explicit protocol choices'}
    (args.output_dir / 'evaluation_config.json').write_text(json.dumps(policy, indent=2), encoding='utf-8')
    summary = {}
    with (args.output_dir / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for k in (1, 5, 10):
            draws = range(10) if args.split == 'test' else (None,)
            draw_rates = []
            for draw_id in draws:
                counts = {'queries': 0}
                if args.path in ('where', 'both'):
                    counts['where_under_generated'] = 0
                if args.path in ('semantic', 'both'):
                    counts['flat_format_valid'] = 0
                for query in queries:
                    episode = frozen_episode(query, train, manifest, k, draw_id=draw_id)
                    result = {'record_id': query.record_id, 'subject': query.subject, 'k': k, 'draw_id': draw_id,
                              'support_ids': [s.record_id for s in episode.supports]}
                    if args.path in ('where', 'both'):
                        result['where'] = evaluate_where_episode(bundle, episode)
                        counts['where_under_generated'] += int(result['where']['under_generated'])
                    if args.path in ('semantic', 'both'):
                        result['semantic'] = evaluate_flat_episode(bundle, episode, generation_budget=args.semantic_max_new_tokens)
                        counts['flat_format_valid'] += int(result['semantic']['flat_format_valid'])
                    counts['queries'] += 1
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                if not counts['queries']:
                    raise ValueError('no eligible evaluation queries')
                rates = {key + '_rate': value / counts['queries'] for key, value in counts.items() if key != 'queries'}
                draw_rates.append(rates)
                summary[f'K={k},draw={draw_id}'] = counts | rates
            summary[f'K={k},draw_mean'] = {key: sum(r[key] for r in draw_rates) / len(draw_rates) for key in draw_rates[0]}
    (args.output_dir / 'validity_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
