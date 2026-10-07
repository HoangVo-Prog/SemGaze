"""Export frozen-episode predictions and validity; metric choices remain gated."""
import argparse
import json
from pathlib import Path
from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter
from semgaze.data.fewshot import frozen_episode
from semgaze.model.checkpoint import load_checkpoint_bundle
from semgaze.model.config import ROOT, load_config, write_run_config
from semgaze.evaluation.predictions import prediction_batches
from semgaze.evaluation.test import test_mode
from semgaze.model.visual_cache import InferenceVisualCache
from semgaze.evaluation.metrics_semantic import (build_bertscorer, build_cider_r_scorer,
    score_prediction_semantics)
from semgaze.evaluation.records import (assert_key_sets_equal, expected_episode_keys,
                                         implementation_head, resolve_evaluation_draw_counts,
                                         write_metrics_artifact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--config', type=Path, help='Evaluation/runtime overrides of checkpoint configuration')
    parser.add_argument('--split', choices=('test',), required=True)
    parser.add_argument('--path', choices=('where', 'semantic', 'both'), required=True)
    parser.add_argument('--semantic-max-new-tokens', type=int,
                        help='Must be explicitly selected; canonical semantic decoding budget is not frozen')
    args = parser.parse_args()
    config = load_config(args.checkpoint / 'resolved_config.json')
    if args.config:
        import yaml
        overrides = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        if set(overrides) - {'runtime', 'evaluation', 'test'}:
            parser.error('evaluation overrides may change runtime/evaluation settings; model/data settings belong to the checkpoint')
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
    queries = [adapter(r) for r in raw['test'] if r['subject'] in data['unseen_subjects']]
    policy = {'split': args.split, 'path': args.path, 'semantic_max_new_tokens': budget,
              'evaluation_draw': config['evaluation']['draw'],
              'do_sample': False, 'split_manifest_identity': identity,
              'report_kind': 'prediction_and_validity_diagnostics',
              'metrics_gate': 'SM/MM/SED implementations and invalid treatment, semantic metrics and selection scalar require explicit protocol choices'}
    (output_dir / 'evaluation_config.json').write_text(json.dumps(policy, indent=2), encoding='utf-8')
    summary = {}
    metrics_config = config['evaluation'].get('metrics', {})
    probability_config = metrics_config.get('probability', {})
    if metrics_config.get('enabled') and (probability_config.get('ll') or probability_config.get('ig')):
        parser.error('standalone frozen prediction evaluation cannot reconstruct LL/IG without saved probability sufficient statistics')
    semantic_config = metrics_config.get('semantic', {})
    semantic_enabled = bool(semantic_config.get('bertscore', {}).get('enabled') or
                            semantic_config.get('cider_r', {}).get('enabled'))
    if metrics_config.get('enabled') and semantic_enabled and args.path == 'where':
        parser.error('semantic metrics require --path semantic or --path both')
    bertscorer = None
    bert_provenance = None
    cider_r = None
    cider_provenance = None
    if metrics_config.get('enabled') and semantic_config.get('bertscore', {}).get('enabled'):
        bert_options = {key: value for key, value in semantic_config['bertscore'].items()
                        if key not in ('enabled', 'batch_size')}
        bertscorer, bert_provenance = build_bertscorer(**bert_options)
    if metrics_config.get('enabled') and semantic_config.get('cider_r', {}).get('enabled'):
        cider_r, cider_provenance = build_cider_r_scorer(
            reference_root=ROOT / 'third_party' / 'cider_r',
            n=semantic_config['cider_r'].get('n', 4),
            k_r=semantic_config['cider_r'].get('k_r', 0.8))
    metrics_by_k = {str(k): {'draws': {}} for k in config['evaluation']['k_values']}
    draw_counts = resolve_evaluation_draw_counts(config['evaluation']['draw'], manifest,
                                                 config['evaluation']['k_values'])
    cache_settings = config['test']['cache']
    cache = InferenceVisualCache(preprocessing=cache_settings['support_preprocessing'],
        features=cache_settings['frozen_visual_features'], max_entries=cache_settings['max_entries'])
    with test_mode(bundle), (output_dir / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for k in config['evaluation']['k_values']:
            draws = range(draw_counts[str(k)])
            draw_rates = []
            for draw_id in draws:
                counts = {'queries': 0}
                if args.path in ('where', 'both'):
                    counts['where_under_generated'] = 0
                if args.path in ('semantic', 'both'):
                    counts['flat_format_valid'] = 0
                episodes = [frozen_episode(query, train, manifest, k, draw_id=draw_id,
                            unseen_subjects=data['unseen_subjects']) for query in queries]
                draw_prediction_rows = []
                for episode, generated in prediction_batches(bundle, episodes, budget=budget, cache=cache, path=args.path):
                    query = episode.query
                    result = {'record_id': query.record_id, 'query_id': query.record_id, 'subject': query.subject,
                              'split': args.split, 'K': k, 'k': k, 'draw_id': draw_id,
                              'support_ids': [s.record_id for s in episode.supports]}
                    if args.path in ('where', 'both'):
                        result['where'] = generated[0]
                        counts['where_under_generated'] += int(result['where']['under_generated'])
                    if args.path in ('semantic', 'both'):
                        result['semantic'] = generated[1]
                        counts['flat_format_valid'] += int(result['semantic']['flat_format_valid'])
                    counts['queries'] += 1
                    draw_prediction_rows.append(result)
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                if not counts['queries']:
                    raise ValueError('no eligible evaluation queries')
                rates = {key + '_rate': value / counts['queries'] for key, value in counts.items() if key != 'queries'}
                draw_rates.append(rates)
                summary[f'K={k},draw={draw_id}'] = counts | rates
                if metrics_config.get('enabled') and (bertscorer is not None or cider_r is not None):
                    expected_rows = [{'query_id': q.record_id, 'K': k, 'draw_id': draw_id}
                                     for q in queries]
                    assert_key_sets_equal(draw_prediction_rows, expected_rows)
                    references = {(q.record_id, k, draw_id): q.semantic for q in queries}
                    metrics_by_k[str(k)]['draws'][str(draw_id)] = score_prediction_semantics(
                        draw_prediction_rows, references, bertscorer=bertscorer, cider_r=cider_r,
                        batch_size=semantic_config.get('bertscore', {}).get('batch_size', 64))
            summary[f'K={k},draw_mean'] = {key: sum(r[key] for r in draw_rates) / len(draw_rates) for key in draw_rates[0]}
    cache.close()
    (output_dir / 'validity_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if metrics_config.get('enabled') and (bertscorer is not None or cider_r is not None):
        expected = expected_episode_keys(
            [q.record_id for q in queries], config['evaluation']['k_values'],
            draw_counts)
        observed = []
        for k, block in metrics_by_k.items():
            for draw_id in block['draws']:
                observed.extend({'query_id': q.record_id, 'K': int(k), 'draw_id': int(draw_id)}
                                 for q in queries)
        if (bertscorer is not None or cider_r is not None) and set(
                (str(r['query_id']), int(r['K']), int(r['draw_id'])) for r in observed) != expected:
            raise ValueError('metrics artifact does not cover every configured test episode')
        for k, block in metrics_by_k.items():
            draws = list(block['draws'].values())
            if draws:
                numeric = sorted({name for draw in draws for name, value in draw.items()
                                  if name.startswith('eval_') and isinstance(value, (int, float))})
                block['draw_mean'] = {name: sum(draw[name] for draw in draws if isinstance(draw.get(name), (int, float))) /
                                      sum(isinstance(draw.get(name), (int, float)) for draw in draws)
                                      for name in numeric if any(isinstance(draw.get(name), (int, float)) for draw in draws)}
            else:
                block['draw_mean'] = {}
        write_metrics_artifact(output_dir / 'metrics.json', checkpoint=args.checkpoint,
            config=args.config or args.checkpoint / 'resolved_config.json',
            split_manifest_identity=identity, by_k=metrics_by_k,
            provenance={'bertscore': bert_provenance, 'cider_r': cider_provenance,
                        'scanpath': {
                            'gate_b1': 'passed',
                            'gate_b2': 'passed',
                            'inverse': 'deepgaze_vl_predict_scanpath_round',
                            'inverse_source': 'DeepGaze-VL/predict_scanpath.py:238-244',
                            'isp_preprocess_sha256': '81aa0754346a382f1f818270645c934af275b6f5e96f6f44b97049108e44512f',
                            'isp_evaluation_sha256': 'd68c1c0554c8195785084f34e4b7f74b06ca085dc3deff7c729517eee79cdb2a',
                            'deepgaze_prediction_sha256': '4f059a30e829ff7414924d1841f20909c7b776d5227767b5abeda863e872108f',
                        },
                        'probability': 'not_integrated; tokenizer outcome unresolved'},
            implementation_head=implementation_head(), evaluation_draw=config['evaluation']['draw'])


if __name__ == '__main__':
    main()
