"""Re-score frozen InterGaze / InternVL-LoRA-SFT / zero-shot semantic predictions.

Reads gold WHAT/WHY/HOW exclusively from the persisted test.json or
 test_seen.json. No checkpoints, images, VLM inference, CUDA or training code.
Uses the repository's exact BLEU-1/4, Java METEOR, ROUGE-L and CIDEr-R scorers.

Usage:
 python evaluate_semantic_predictions_offline.py \
   --predictions runs/eval/predictions.jsonl --subject-scope unseen \
   --split-root data/split/all --output-dir runs/offline_scores --k 1
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

GROUPS = ('TA', 'TP', 'VQA')
METRICS = (('B1', 'bleu1'), ('B4', 'bleu4'), ('METEOR', 'meteor'),
           ('ROUGE', 'rouge_l'), ('CIDEr-R', 'cider_r'))


@dataclass(frozen=True)
class ReferenceQuery:
    record_id: str
    semantic: object
    dataset: str
    condition: str
    subject: str
    stimulus_id: str
    task: str
    gt_fixation_count: int
    gt_group_count: int


def read_json_records(path, *, label):
    """Accept JSON lists/wrappers, id-keyed JSON maps, or NDJSON/JSONL."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in ('.jsonl', '.ndjson'):
        with path.open(encoding='utf-8-sig') as source:
            entries = [json.loads(line) for line in source if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        if isinstance(payload, list):
            entries = payload
        elif isinstance(payload, dict):
            for key in ('predictions', 'rows', 'results', 'records', 'data'):
                if isinstance(payload.get(key), list):
                    entries = payload[key]
                    break
            else:
                # Also accept {record_id: {text: ..., ...}, ...}.
                if payload and all(isinstance(value, dict) for value in payload.values()):
                    entries = [dict(value, query_id=value.get('query_id', value.get('record_id', key)))
                               for key, value in payload.items()]
                else:
                    raise ValueError(f'{label}: unsupported JSON container in {path}')
        else:
            raise ValueError(f'{label}: expected a JSON list, mapping or JSONL in {path}')
    if not isinstance(entries, list) or not entries or not all(isinstance(e, dict) for e in entries):
        raise ValueError(f'{label}: expected a nonempty list of JSON objects in {path}')
    return entries


def load_references(path, *, scope):
    """Use semantic annotations inside the raw held-out split; no image adapters."""
    from semgaze.data.semantic import semantic_from_prediction

    result = {}
    for raw in read_json_records(path, label='ground truth'):
        rid = raw.get('record_id')
        if not isinstance(rid, str) or not rid or rid in result:
            raise ValueError(f'{path}: missing or duplicate GT record_id {rid!r}')
        if raw.get('subject_role', scope) != scope:
            raise ValueError(f'{rid}: GT subject_role does not match {scope}')
        split = 'test_seen' if scope == 'seen' else 'test'
        if raw.get('split', split) != split:
            raise ValueError(f'{rid}: GT split mismatch (expected {split})')
        ds = raw.get('dataset', 'COCO-Search18')
        cond = str(raw.get('condition', '')).lower()
        if (ds == 'AiR' and cond != 'vqa') or (ds == 'COCO-Search18' and cond not in ('present', 'absent')):
            raise ValueError(f'{rid}: invalid dataset/condition {ds}/{cond}')
        if not isinstance(raw.get('prediction'), Mapping) or not isinstance(raw.get('X'), list):
            raise ValueError(f'{rid}: GT requires semantic prediction object and X list')
        semantic = semantic_from_prediction(raw['prediction'], len(raw['X']), rid)
        # AiR stimulus identifiers are dataset-qualified by the joint adapter.
        stimulus = raw.get('global_stimulus_id') if ds == 'AiR' else raw.get('stimulus_id')
        if stimulus is None:
            stimulus = f"AiR::{raw['name']}" if ds == 'AiR' else raw['name']
        subject = raw.get('global_subject_id') if ds == 'AiR' else raw.get('subject')
        result[rid] = ReferenceQuery(
            rid, semantic, ds, cond, str(subject), str(stimulus), str(raw['task']),
            len(raw['X']), len(semantic.why_groups))
    return result


def group_for(query):
    if query.dataset == 'AiR':
        return 'VQA'
    return 'TP' if query.condition == 'present' else 'TA'


def _extract_prediction(row, *, rid):
    """Never read the GT-annotation `prediction` field as a model output object."""
    if 'SEMANTIC' in row:
        value = row['SEMANTIC']
        if not isinstance(value, Mapping):
            raise ValueError(f'{rid}: SEMANTIC must be a GT/PRED mapping')
        result = value.get('PRED')
    else:
        keys = ('text', 'predicted_text', 'semantic_prediction', 'semantic_pred', 'output')
        present = [key for key in keys if key in row]
        if present:
            result = row[present[0]]
        elif 'prediction' in row and isinstance(row['prediction'], (str, type(None))):
            result = row['prediction']
        elif 'where_generation' in row and 'semantic_generation' not in row:
            return False, None  # WHERE-only record in a multi-draw InterGaze file.
        else:
            raise ValueError(f'{rid}: no semantic text in SEMANTIC.PRED/text/output')
    if result is not None and not isinstance(result, str):
        raise ValueError(f'{rid}: semantic text must be string or null')
    return True, result


def normalize_predictions(rows, reference_ids, *, selected_ks=None, selected_draws=None,
                          strict_extras=True, parser=None):
    """Return {(K,draw): {query_id: canonical row}} with missing text retained."""
    if parser is None:
        from semgaze.semantic.flat.parser import parse_flat_output
        parser = parse_flat_output
    grouped = defaultdict(dict)
    for row in rows:
        rid = row.get('query_id', row.get('record_id'))
        if not isinstance(rid, str) or not rid:
            raise ValueError('prediction row is missing query_id/record_id')
        if rid not in reference_ids:
            if strict_extras:
                raise ValueError(f'extra prediction query_id outside selected GT split: {rid}')
            continue
        k = row.get('K', row.get('k', 0))
        draw = row.get('draw_id', 0)
        if type(k) not in (int, str) or type(draw) not in (int, str):
            raise ValueError(f'{rid}: K/draw_id must be integers')
        k, draw = int(k), int(draw)
        if k < 0 or draw < 0:
            raise ValueError(f'{rid}: K/draw_id must be nonnegative')
        if selected_ks is not None and k not in selected_ks:
            continue
        if selected_draws is not None and draw not in selected_draws:
            continue
        is_sem, text = _extract_prediction(row, rid=rid)
        if not is_sem:
            continue
        key = (k, draw)
        if rid in grouped[key]:
            raise ValueError(f'duplicate semantic prediction {rid} at K={k}, draw={draw}')
        ref = reference_ids[rid]
        for name, expected in (('dataset', ref.dataset), ('condition', ref.condition),
                               ('subject', ref.subject), ('stimulus_id', ref.stimulus_id),
                               ('task', ref.task), ('gt_fixation_count', ref.gt_fixation_count),
                               ('gt_group_count', ref.gt_group_count)):
            if name in row and str(row[name]) != str(expected):
                # AiR baseline stimulus_id is raw basename (the joint adapter
                # uses AiR::basename). Accept only that documented representation.
                if not (name == 'stimulus_id' and ref.dataset == 'AiR'
                        and str(row[name]) == ref.stimulus_id.removeprefix('AiR::')):
                    raise ValueError(f'{rid}: prediction {name} mismatch (GT={expected!r}, pred={row[name]!r})')
        if row.get('split') not in (None, 'test', 'test_seen'):
            raise ValueError(f'{rid}: unsupported prediction split {row["split"]!r}')
        canonical = dict(row, query_id=rid, K=k, draw_id=draw,
                         SEMANTIC={**(row.get('SEMANTIC', {}) or {}), 'PRED': text})
        # Reparse the original text: a stale/missing saved parser result must not
        # suppress syntax errors; malformed generations stay in scoring corpus.
        canonical['semantic_generation'] = parser(text or '', ref.gt_fixation_count, ref.gt_group_count)
        grouped[key][rid] = canonical
    if not grouped:
        raise ValueError('no semantic predictions for the requested K/draw/scope')
    expected = set(reference_ids)
    for (k, draw), indexed in grouped.items():
        missing = expected - set(indexed)
        if missing:
            sample = sorted(missing)[:5]
            raise ValueError(f'incomplete semantic predictions K={k} draw={draw}: '
                             f'{len(missing)} missing of {len(expected)}, example={sample}')
    return dict(sorted(grouped.items()))


def score_scope(grouped, references, scorer, *, label, scope, gt_path, prediction_path):
    """Score complete original text per query/group/K/draw. Overall is rescored."""
    partitions = {name: [q for q in references.values() if group_for(q) == name] for name in GROUPS}
    partitions['Overall'] = list(references.values())
    # Dataset corpora are optional reports, with independently recomputed scores.
    partitions['COCO-Search18'] = [q for q in references.values() if q.dataset == 'COCO-Search18']
    partitions['AiR'] = [q for q in references.values() if q.dataset == 'AiR']
    blocks = {name: {'by_k': {}} for name, queries in partitions.items() if queries}
    for (k, draw), index in grouped.items():
        for name, queries in partitions.items():
            if not queries:
                continue
            rows = [index[q.record_id] for q in queries]
            ref_map = {(q.record_id, k, draw): q.semantic for q in queries}
            metrics = scorer.score(rows, ref_map)
            block = blocks[name]['by_k'].setdefault(str(k), {'draws': {}})
            block['draws'][str(draw)] = metrics
    for entry in blocks.values():
        for k, block in entry['by_k'].items():
            draws = list(block['draws'].values())
            block['semantic_draw_count'] = len(draws)
            block['semantic_metric_aggregation'] = 'draw_mean' if len(draws) > 1 else 'single_traversal'
            block['draw_mean'] = {
                metric: sum(d[metric] for d in draws) / len(draws)
                for metric in draws[0] if metric.startswith('eval_') and
                all(isinstance(d.get(metric), (float, int)) for d in draws)}
    return {
        'schema_version': 1, 'method': label, 'subject_scope': scope,
        'ground_truth_file': str(gt_path), 'ground_truth_sha256': hashlib.sha256(Path(gt_path).read_bytes()).hexdigest(),
        'predictions_file': str(prediction_path),
        'predictions_sha256': hashlib.sha256(Path(prediction_path).read_bytes()).hexdigest(),
        'query_count': len(references),
        'query_ids_sha256': hashlib.sha256('\n'.join(sorted(references)).encode('utf-8')).hexdigest(),
        'query_count_by_group': {name: len(partitions[name]) for name in GROUPS},
        'scored_k_draw': [{'K': k, 'draw_id': draw} for k, draw in grouped],
        'coverage_by_k_draw': [
            {'K': k, 'draw_id': draw, 'query_count': len(rows),
             'format_failures': sum(not row['semantic_generation']['flat_format_valid']
                                    for row in rows.values()),
             'empty_predictions': sum(not row['SEMANTIC']['PRED'] for row in rows.values())}
            for (k, draw), rows in grouped.items()],
        'overall': blocks['Overall'],
        'by_group': {name: blocks.get(name) for name in GROUPS},
        'by_dataset': {name: blocks.get(name) for name in ('COCO-Search18', 'AiR')},
        'aggregation': 'corpus_rescore_per_group_and_overall; draw_mean_is_mean_over_complete_draws',
    }


class CanonicalSemanticScorer:
    """Reuse the existing SemGaze scorer, never a reimplementation or model."""
    def __init__(self, *, bleu1=True, bleu4=True, rouge_l=True, cider_r=True,
                 meteor=True, bertscore=False, meteor_jar_path=None,
                 report_per_branch=False):
        from semgaze.evaluation.metrics_semantic import build_bertscorer, build_cider_r_scorer
        from semgaze.evaluation.metrics_text_overlap import build_meteor_scorer
        self._score = None
        self.meteor = self.cider = self.bert = None
        self.provenance = {'scoring_unit': 'one_raw_WHAT_WHY_HOW_text_per_query',
                           'bleu1': bleu1, 'bleu4': bleu4, 'rouge_l': rouge_l,
                           'cider_r': cider_r, 'meteor': meteor, 'bertscore': bertscore}
        self.options = dict(bleu1=bleu1, bleu4=bleu4, rouge_l=rouge_l,
                            report_per_branch=report_per_branch)
        try:
            if cider_r:
                self.cider, self.provenance['cider_r_implementation'] = build_cider_r_scorer()
            if meteor:
                self.meteor, self.provenance['meteor_implementation'] = build_meteor_scorer(
                    jar_path=meteor_jar_path)
            if bertscore:
                self.bert, self.provenance['bertscore_implementation'] = build_bertscorer()
        except Exception:
            self.close()
            raise

    def score(self, rows, refs):
        from semgaze.evaluation.metrics_semantic import score_prediction_semantics
        return score_prediction_semantics(
            rows, refs, bertscorer=self.bert, cider_r=self.cider,
            bleu1=self.options['bleu1'], bleu4=self.options['bleu4'],
            rouge_l=self.options['rouge_l'], meteor=self.meteor,
            report_per_branch=self.options['report_per_branch'])

    def close(self):
        if self.meteor is not None:
            self.meteor.close()
            self.meteor = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_table(result, output_dir):
    """Exactly 21 columns: Method + 5 metrics per TA, TP, VQA and Overall."""
    output_dir = Path(output_dir)
    columns = ['Method'] + [f'{group} {label} ↑' for group in (*GROUPS, 'Overall')
                            for label, _ in METRICS]
    table = []
    for k in sorted(result['overall']['by_k'], key=int):
        row = [result['method'] if int(k) == 0 else f"{result['method']} (K={k})"]
        for group in (*GROUPS, 'Overall'):
            artifact = result['overall'] if group == 'Overall' else result['by_group'][group]
            scores = artifact['by_k'][k]['draw_mean'] if artifact and k in artifact['by_k'] else {}
            for _, metric in METRICS:
                value = scores.get('eval_sem_overall_' + metric)
                row.append(f'{value:.4f}' if isinstance(value, (int, float)) else '-')
        table.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'semantic_metrics_table.csv').open('w', encoding='utf-8', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(table)
    content = ['| ' + ' | '.join(columns) + ' |',
               '|:-------' + '|--------:' * (len(columns) - 1) + '|']
    content.extend('| ' + ' | '.join(row) + ' |' for row in table)
    (output_dir / 'semantic_metrics_table.md').write_text('\n'.join(content) + '\n', encoding='utf-8')
    return {'columns': columns, 'rows': table}


def evaluate_scope(*, predictions, references_path, scope, output_dir, scorer,
                   selected_ks=None, selected_draws=None, label='InterGaze', strict_extras=True):
    refs = load_references(references_path, scope=scope)
    raw = read_json_records(predictions, label='predictions')
    grouped = normalize_predictions(raw, refs, selected_ks=selected_ks,
                                    selected_draws=selected_draws,
                                    strict_extras=strict_extras)
    result = score_scope(grouped, refs, scorer, label=label, scope=scope,
                         gt_path=references_path, prediction_path=predictions)
    result['metric_provenance'] = scorer.provenance
    result['prediction_format'] = ('intergaze' if any('SEMANTIC' in row and
        ('K' in row or 'k' in row) for group in grouped.values() for row in group.values())
        else 'semantic_baseline_or_generic')
    result['completeness'] = 'all_test_queries_once_per_scored_K_draw'
    out = Path(output_dir)
    _write_json(out / 'metrics.json', result)
    _write_json(out / 'metrics_by_group.json', result['by_group'])
    _write_json(out / 'metrics_by_dataset.json', result['by_dataset'])
    write_table(result, out)
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--predictions', type=Path,
                   help='JSON / JSONL prediction file; for --subject-scope both it must cover both splits')
    p.add_argument('--predictions-seen', type=Path,
                   help='Optional separate prediction file for seen')
    p.add_argument('--predictions-unseen', type=Path,
                   help='Optional separate prediction file for unseen')
    p.add_argument('--split-root', type=Path, default=Path('data/split/all'),
                   help='Folder containing test.json and test_seen.json')
    p.add_argument('--subject-scope', choices=('seen', 'unseen', 'both'), default='unseen')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--method', default='auto',
                   help='Table label: auto detects InterGaze or semantic_sft / semantic_zero_shot')
    p.add_argument('--k', action='append', type=int, help='Filter to one K; repeat for multiple K')
    p.add_argument('--draw', action='append', type=int, help='Filter to one draw_id; repeat for multiple draws')
    p.add_argument('--no-bleu1', action='store_true')
    p.add_argument('--no-bleu4', action='store_true')
    p.add_argument('--no-rouge-l', action='store_true')
    p.add_argument('--no-cider-r', action='store_true')
    p.add_argument('--no-meteor', action='store_true')
    p.add_argument('--meteor-jar-path', type=Path)
    p.add_argument('--bertscore', action='store_true')
    p.add_argument('--report-per-branch', action='store_true',
                   help='Also report diagnostic WHAT/WHY/HOW metrics, without replacing Overall')
    return p


def _name_from_predictions(path):
    rows = read_json_records(path, label='predictions')
    methods = {row.get('method') for row in rows if isinstance(row.get('method'), str)}
    if len(methods) > 1:
        raise ValueError(f'mixed method labels in {path}: {sorted(methods)}')
    if methods:
        return methods.pop()
    if any('SEMANTIC' in row for row in rows):
        return 'InterGaze'
    return 'semantic_predictions'


def main(argv=None):
    args = build_parser().parse_args(argv)
    scopes = ('seen', 'unseen') if args.subject_scope == 'both' else (args.subject_scope,)
    paths = {scope: (args.predictions_seen if scope == 'seen' else args.predictions_unseen)
             or args.predictions for scope in scopes}
    if any(path is None for path in paths.values()):
        raise SystemExit('Missing predictions: supply --predictions or the required scope-specific file(s)')
    if args.k and any(k < 0 for k in args.k):
        raise SystemExit('--k must be nonnegative')
    if args.draw and any(draw < 0 for draw in args.draw):
        raise SystemExit('--draw must be nonnegative')
    label = args.method if args.method != 'auto' else _name_from_predictions(next(iter(paths.values())))
    # This script lives at repository root, so no model/config imports are needed.
    root = Path(__file__).resolve().parent
    split_root = args.split_root if args.split_root.is_absolute() else root / args.split_root
    try:
        with CanonicalSemanticScorer(
                bleu1=not args.no_bleu1, bleu4=not args.no_bleu4,
                rouge_l=not args.no_rouge_l, cider_r=not args.no_cider_r,
                meteor=not args.no_meteor, bertscore=args.bertscore,
                meteor_jar_path=args.meteor_jar_path,
                report_per_branch=args.report_per_branch) as scorer:
            outputs = {}
            for scope in scopes:
                file = split_root / ('test_seen.json' if scope == 'seen' else 'test.json')
                out = args.output_dir / scope if args.subject_scope == 'both' else args.output_dir
                # When a shared prediction file spans both scopes, exclude the
                # other known scope's record_ids, but never ignore unknown IDs.
                if args.subject_scope == 'both' and paths['seen'] == paths['unseen']:
                    other = 'unseen' if scope == 'seen' else 'seen'
                    other_file = split_root / ('test_seen.json' if other == 'seen' else 'test.json')
                    other_refs = load_references(other_file, scope=other)
                    current_refs = load_references(file, scope=scope)
                    accepted = set(other_refs) | set(current_refs)
                    raw = read_json_records(paths[scope], label='predictions')
                    selected_rows = []
                    for row in raw:
                        rid = row.get('query_id', row.get('record_id'))
                        if rid not in accepted:
                            raise ValueError(f'unknown prediction query_id in combined file: {rid!r}')
                        if rid in current_refs:
                            selected_rows.append(row)
                    # Normalize/score without writing filtered prediction data.
                    grouped = normalize_predictions(
                        selected_rows, current_refs, selected_ks=set(args.k) if args.k else None,
                        selected_draws=set(args.draw) if args.draw else None)
                    result = score_scope(grouped, current_refs, scorer, label=label,
                        scope=scope, gt_path=file, prediction_path=paths[scope])
                    result['metric_provenance'] = scorer.provenance
                    result['completeness'] = 'all_test_queries_once_per_scored_K_draw'
                    _write_json(out / 'metrics.json', result)
                    _write_json(out / 'metrics_by_group.json', result['by_group'])
                    _write_json(out / 'metrics_by_dataset.json', result['by_dataset'])
                    write_table(result, out)
                else:
                    result = evaluate_scope(
                        predictions=paths[scope], references_path=file, scope=scope,
                        output_dir=out, scorer=scorer,
                        selected_ks=set(args.k) if args.k else None,
                        selected_draws=set(args.draw) if args.draw else None, label=label)
                outputs[scope] = str(out / 'metrics.json')
                print(f'[SEMANTIC][OFFLINE] {scope} | queries={result["query_count"]} '
                      f'| K/draw={result["scored_k_draw"]} | table={out / "semantic_metrics_table.md"}', flush=True)
            if args.subject_scope == 'both':
                _write_json(args.output_dir / 'evaluation_summary.json', {
                    'method': label, 'subject_scope': 'both', 'separate_scopes': outputs,
                    'note': 'Seen and unseen metrics were scored independently; never pooled.'})
    except (KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(f'Offline semantic evaluation failed: {exc}') from exc
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
