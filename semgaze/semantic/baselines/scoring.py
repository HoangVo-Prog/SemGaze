"""Strict query join and re-scoring through the *existing* SemGaze corpus scorers."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from semgaze.evaluation.metrics_semantic import (
    build_bertscorer, build_cider_r_scorer, score_prediction_semantics,
)
from semgaze.evaluation.metrics_text_overlap import build_meteor_scorer
from semgaze.semantic.flat.parser import parse_flat_output
from .prompt import semantic_target

SUBSETS = ('TP', 'TA', 'VQA')


def dump_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False,
                                     default=str) + '\n', encoding='utf-8')


def load_jsonl(path):
    path = Path(path)
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_complete(rows, queries, *, method, reference_k=None, reference_draw=None):
    """Reject duplicate/missing IDs; no cherry-picking unparseable generations."""
    by_id = {q.record_id: q for q in queries}
    if len(by_id) != len(queries):
        raise ValueError('duplicate normalized test query')
    selected = []
    for row in rows:
        if method == 'SemGaze':
            if row.get('split') != 'test' or int(row.get('K', row.get('k', -1))) != reference_k or int(row.get('draw_id', -1)) != reference_draw:
                continue
            if 'semantic_generation' not in row:
                raise ValueError('selected SemGaze comparison row has no semantic prediction')
        selected.append(row)
    seen = set()
    for row in selected:
        rid = row.get('query_id')
        if rid in seen:
            raise ValueError(f'duplicate {method} prediction query_id: {rid}')
        seen.add(rid)
        q = by_id.get(rid)
        if q is None:
            raise ValueError(f'extraneous/mismatched {method} query_id: {rid}')
        for key, expected in [('subject', q.subject), ('dataset', q.dataset),
                              ('stimulus_id', q.stimulus_id), ('task', q.task)]:
            if key in row and str(row[key]) != str(expected):
                raise ValueError(f'{method} {rid}: {key} mismatch')
        if method == 'SemGaze':
            expected_pairs = [[x, y] for x, y in zip(q.x_px, q.y_px)]
            for key, expected in [('gt_raw_pixels', expected_pairs),
                                  ('gt_duration_ms_raw', list(q.duration_ms)),
                                  ('annotation_width', q.image_width),
                                  ('annotation_height', q.image_height)]:
                if key not in row or row[key] != expected:
                    raise ValueError(f'{rid}: frozen SemGaze {key} mismatch')
            if row.get('SEMANTIC', {}).get('GT') != semantic_target(q):
                raise ValueError(f'{rid}: SemGaze semantic ground truth mismatch')
        else:
            for key, expected in [('gt_fixation_count', len(q.x_px)),
                                  ('gt_group_count', len(q.semantic.why_groups)),
                                  ('condition', q.condition)]:
                if row.get(key) != expected:
                    raise ValueError(f'{rid}: baseline {key} mismatch')
            if row.get('method') != method or not row.get('prompt_style'):
                raise ValueError(f'{rid}: mismatched baseline method/style')
            parsed = parse_flat_output(row['text'], len(q.x_px), len(q.semantic.why_groups))
            if row.get('semantic_generation') != json.loads(json.dumps(parsed, default=str)):
                # JSON round-trip retains int-key dictionary semantics as string keys.
                if row.get('semantic_generation') != json.loads(json.dumps(parsed)):
                    raise ValueError(f'{rid}: baseline parsed output differs from canonical parser')
    missing = set(by_id) - seen
    if missing or len(selected) != len(by_id):
        raise ValueError(f'incomplete {method} population: missing={len(missing)} '
                         f'expected={len(by_id)}, observed={len(selected)}')
    # One stable normalized-query order for all models and corpora.
    selected_by_id = {row['query_id']: row for row in selected}
    return [selected_by_id[q.record_id] for q in queries]


def subset_name(query):
    if query.dataset == 'AiR' and query.condition == 'vqa':
        return 'VQA'
    if query.dataset == 'COCO-Search18' and query.condition == 'present':
        return 'TP'
    if query.dataset == 'COCO-Search18' and query.condition == 'absent':
        return 'TA'
    raise ValueError(f'{query.record_id}: unrecognized dataset-condition subset')


class Scorers:
    def __init__(self, config):
        self.options = config
        self.bertscorer = self.cider = self.meteor = None
        self.provenance = {'bleu4': 'SemGaze corpus BLEU-4', 'rouge_l': 'SemGaze ROUGE-L'}
        if config.get('bertscore', False):
            self.bertscorer, self.provenance['bertscore'] = build_bertscorer()
        if config.get('cider_r', True):
            self.cider, self.provenance['cider_r'] = build_cider_r_scorer()
        if config.get('meteor', True):
            self.meteor, self.provenance['meteor'] = build_meteor_scorer(
                jar_path=config.get('meteor_jar_path'))

    def score(self, rows, queries):
        if not rows:
            return None
        if len(rows) != len(queries):
            raise ValueError('metric population differs from normalized query population')
        lookup = {q.record_id: q for q in queries}
        # K=0/draw=0 are scorer-internal identity placeholders ONLY: baseline
        # predictions and model prompts never contain K or support contexts.
        canonical = [dict(row, K=0, draw_id=0) for row in rows]
        references = {(q.record_id, 0, 0): q.semantic for q in queries}
        return score_prediction_semantics(
            canonical, references, bertscorer=self.bertscorer, cider_r=self.cider,
            bleu4=self.options.get('bleu4', True),
            rouge_l=self.options.get('rouge_l', True), meteor=self.meteor,
            batch_size=int(self.options.get('bertscore_batch_size', 64)))

    def close(self):
        if self.meteor is not None:
            self.meteor.close()


def evaluate_rows(rows, queries, scorers):
    """Never combine AiR and COCO into an unmarked headline scalar."""
    by_id = {r['query_id']: r for r in rows}
    result = {}
    for subset in (*SUBSETS, 'COCO-Search18', 'AiR'):
        chosen = [q for q in queries if (subset_name(q) == subset if subset in SUBSETS
                                         else q.dataset == subset)]
        if chosen:
            result[subset] = {'query_count': len(chosen),
                              'metrics': scorers.score([by_id[q.record_id] for q in chosen], chosen)}
    return result


def coverage_hash(queries):
    return hashlib.sha256('\n'.join(sorted(q.record_id for q in queries)).encode()).hexdigest()


def scoring_report(rows, queries, scorer_opts, output_dir, *, method):
    output_dir = Path(output_dir)
    scorers = Scorers(scorer_opts)
    try:
        result = evaluate_rows(rows, queries, scorers)
        dump_json(output_dir / 'metrics_by_subset.json', {k: v for k, v in result.items() if k in SUBSETS})
        dump_json(output_dir / 'metrics.json', {k: v for k, v in result.items() if k not in SUBSETS})
        dump_json(output_dir / 'validity_summary.json', {
            'method': method, 'total_queries': len(rows),
            'valid': sum(bool(r['semantic_generation'].get('flat_format_valid')) for r in rows),
            'invalid': sum(not bool(r['semantic_generation'].get('flat_format_valid')) for r in rows),
            'parse_error_frequencies': dict(Counter(
                error for r in rows for error in r['semantic_generation'].get('errors', ()))),
            'coverage_sha256': coverage_hash(queries), 'scorer_provenance': scorers.provenance,
        })
        return result
    finally:
        scorers.close()
