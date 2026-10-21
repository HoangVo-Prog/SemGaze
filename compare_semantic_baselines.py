"""Re-score frozen SemGaze and independent baselines on exactly matched queries."""
import argparse
import csv
import json
from pathlib import Path
import yaml

from semgaze.model.config import ROOT
from semgaze.semantic.baselines.data import load_queries
from semgaze.semantic.baselines.scoring import (
    SUBSETS, Scorers, coverage_hash, dump_json, evaluate_rows,
    load_jsonl, verify_complete,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--semgaze-config', type=Path, default=ROOT / 'configs/flat_single.yaml')
    p.add_argument('--metrics-config', type=Path, default=ROOT / 'configs/semantic_zero_shot.yaml')
    p.add_argument('--semgaze-predictions', type=Path, required=True)
    p.add_argument('--baseline', type=Path, nargs='+', required=True,
                   help='One or more baseline evaluation directories containing predictions.jsonl')
    p.add_argument('--reference-k', type=int, default=1)
    p.add_argument('--reference-draw', type=int, default=0)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--semgaze-metrics-root', type=Path,
                   help='Optional SemGaze evaluator root containing AiR/metrics.json and COCO_Search18/metrics.json')
    p.add_argument('--parity-atol', type=float, default=1e-7)
    p.add_argument('--allow-generation-budget-mismatch', action='store_true')
    p.add_argument('--images-root')
    p.add_argument('--split-root')
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    overrides = {}
    if a.images_root:
        overrides['images_root'] = a.images_root
    if a.split_root:
        overrides['split_root'] = a.split_root
    _, queries, identity, data = load_queries(a.semgaze_config, data_overrides=overrides)
    chosen = verify_complete(load_jsonl(a.semgaze_predictions), queries,
                             method='SemGaze', reference_k=a.reference_k,
                             reference_draw=a.reference_draw)
    rows_by_method = {'SemGaze': chosen}
    configuration = {'SemGaze': {'K': a.reference_k, 'draw_id': a.reference_draw,
                                  'conditioning': 'fewshot WHERE hidden states projected from teacher-forced GT XYD'}}
    observed_sem_budgets = {row['semantic_generation'].get('max_new_tokens') for row in chosen}
    if len(observed_sem_budgets) != 1:
        raise ValueError('selected SemGaze draw uses inconsistent semantic generation budgets')
    deviations = []
    if observed_sem_budgets != {a.max_new_tokens}:
        deviations.append({'model': 'SemGaze', 'issue': 'max_new_tokens mismatch',
                           'observed': list(observed_sem_budgets), 'expected': a.max_new_tokens})
    for directory in a.baseline:
        directory = directory.resolve()
        metadata = json.loads((directory / 'resolved_config.json').read_text())
        if metadata['split_manifest_identity'] != identity:
            raise ValueError(f'{directory}: split identity mismatch')
        method, style = metadata['method'], metadata['prompt_style']
        if method not in ('semantic_zero_shot', 'semantic_sft') or style not in ('origin', 'concise'):
            raise ValueError('unknown baseline method/prompt style')
        label = f'{method}/{style}'
        if label in rows_by_method:
            raise ValueError('comparison received duplicate experiment label')
        predictions = verify_complete(load_jsonl(directory / 'predictions.jsonl'), queries,
                                      method=method)
        if any(row['prompt_style'] != style for row in predictions):
            raise ValueError('prediction prompt style differs from resolved config')
        if method == 'semantic_sft':
            ckpt_path = Path(metadata['checkpoint'])
            info = json.loads((ckpt_path / 'semantic_sft_checkpoint.json').read_text())
            if info['prompt_style'] != style or info['split_manifest_identity'] != identity:
                raise ValueError('SFT checkpoint style/split mismatch')
        budgets = {row['generation']['max_new_tokens'] for row in predictions}
        if budgets != {a.max_new_tokens}:
            deviations.append({'model': label, 'issue': 'max_new_tokens mismatch',
                               'observed': list(budgets), 'expected': a.max_new_tokens})
        configuration[label] = {'method': method, 'style': style,
                                'checkpoint': metadata.get('checkpoint'),
                                'conditioning': 'textual GT XYD + gold groups, no support examples'}
        rows_by_method[label] = predictions
    if deviations and not a.allow_generation_budget_mismatch:
        raise ValueError(f'mismatched decoding budgets, refusing fair comparison: {deviations}')
    metric_cfg = yaml.safe_load(a.metrics_config.read_text())['metrics']
    scorer = Scorers(metric_cfg)
    try:
        metrics = {method: evaluate_rows(rows, queries, scorer)
                   for method, rows in rows_by_method.items()}
        provenance = scorer.provenance
    finally:
        scorer.close()
    parity = {'status': 'not_checked_no_original_metrics_artifact'}
    if a.semgaze_metrics_root is not None:
        parity = {'status': 'checked', 'dataset_scores': {}}
        for dataset in ('AiR', 'COCO-Search18'):
            current = metrics['SemGaze'].get(dataset)
            if current is None:
                continue
            artifact = a.semgaze_metrics_root / dataset.replace('-', '_') / 'metrics.json'
            original = json.loads(artifact.read_text(encoding='utf-8'))
            if original.get('split_manifest_identity') != identity:
                raise ValueError(f'{dataset}: original evaluator split identity mismatch')
            frozen = original['by_k'][str(a.reference_k)]['draws'][str(a.reference_draw)]
            observed = current['metrics']
            common = sorted(set(k for k in frozen if k.startswith('eval_sem_')) &
                            set(k for k in observed if k.startswith('eval_sem_')))
            if not common:
                raise ValueError(f'{dataset}: no common semantic metrics for original parity check')
            differences = {key: abs(float(frozen[key]) - float(observed[key]))
                           for key in common if frozen[key] is not None and observed[key] is not None}
            failures = {k: diff for k, diff in differences.items() if diff > a.parity_atol}
            if failures:
                raise ValueError(f'{dataset}: canonical SemGaze metric parity failed: {failures}')
            parity['dataset_scores'][dataset] = {'metrics_checked': common,
                                                   'maximum_error': max(differences.values(), default=0.0),
                                                   'atol': a.parity_atol}
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / 'comparison_metrics.json', metrics)
    fields = ['method', 'subset', 'query_count', 'parse_failure_count',
              *(f'{branch}_{name}' for branch in ('overall',)
                for name in ('bleu1', 'bleu4', 'meteor', 'rouge_l', 'cider_r', 'bertscore_f1'))]
    with (out / 'comparison_table.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, partitions in metrics.items():
            for subset in (*SUBSETS, 'Overall'):
                if subset not in partitions:
                    continue
                item = partitions[subset]
                raw = item['metrics']
                values = {'method': name, 'subset': subset,
                          'query_count': item['query_count'],
                          'parse_failure_count': raw['semantic_parse_failure_count']}
                for branch in ('overall',):
                    for metric in ('bleu1', 'bleu4', 'meteor', 'rouge_l', 'cider_r', 'bertscore_f1'):
                        values[f'{branch}_{metric}'] = raw.get(f'eval_sem_{branch}_{metric}')
                writer.writerow(values)
    dump_json(out / 'comparison_protocol.json', {
        'data_config': data, 'split_manifest_identity': identity,
        'query_count': len(queries), 'coverage_sha256': coverage_hash(queries),
        'query_ids': sorted(q.record_id for q in queries),
        'queries_by_dataset': {'AiR': sum(q.dataset == 'AiR' for q in queries),
                               'COCO-Search18': sum(q.dataset == 'COCO-Search18' for q in queries)},
        'query_subsets': {s: sum((q.dataset == 'AiR' and q.condition == 'vqa') if s == 'VQA'
                                   else (q.dataset == 'COCO-Search18' and
                                         q.condition == ('present' if s == 'TP' else 'absent'))
                                   for q in queries) for s in SUBSETS},
        'semgaze_predictions': str(a.semgaze_predictions),
        'reference_k': a.reference_k, 'reference_draw': a.reference_draw,
        'models': configuration, 'metric_config': metric_cfg,
        'metric_provenance': provenance, 'decoding': {'greedy': True, 'budget': a.max_new_tokens},
        'parity_deviations': deviations, 'existing_evaluator_parity': parity,
        'conditioning_limitation': 'SemGaze uses latent teacher-forced GT WHERE states which may carry '
                                   'few-shot support information; baselines see explicit text GT WHERE without support.',
        'scoring': 'semgaze.evaluation.metrics_semantic.score_prediction_semantics '
                   '(complete corpus per method and subset, no missing predictions dropped)',
    })
    print('Comparison complete:', out, '| population:', len(queries))


if __name__ == '__main__':
    main()
