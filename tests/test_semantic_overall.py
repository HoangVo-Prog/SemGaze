"""CPU-only tests for complete structured semantic scoring and group tables."""
import unittest
from types import SimpleNamespace

from semgaze.evaluation.metrics_semantic import (
    score_prediction_semantics, _structured_unit_from_prediction)
from semgaze.evaluation.metrics_text_overlap import (
    score_bleu1_corpus, score_bleu4_corpus, score_rouge_l_corpus)
from semgaze.evaluation.semantic_table import make_semantic_metric_table, write_semantic_metrics_tables
from semgaze.semantic.flat.target import build_flat_target


def gt():
    return SimpleNamespace(what=['red bus', 'green tree'],
                           why_groups=[SimpleNamespace(members=(1, 2), why='looking for a bus')],
                           how='searched left then right')


def row(qid, pred, *, valid=True, baseline=False):
    data = {'query_id': qid, 'K': 1, 'draw_id': 0,
            'semantic_generation': {'flat_format_valid': valid}}
    if baseline:
        data['text'] = pred
    else:
        data['SEMANTIC'] = {'GT': build_flat_target(gt()), 'PRED': pred}
    return data


class StructuredSemanticsTest(unittest.TestCase):
    def test_complete_query_is_one_document_and_scores_perfect(self):
        full = build_flat_target(gt())
        refs = {('q1', 1, 0): gt()}
        metrics = score_prediction_semantics([row('q1', full)], refs,
                                             bleu1=True, bleu4=True, rouge_l=True)
        self.assertEqual(metrics['semantic_overall_unit_count'], 1)
        self.assertNotIn('eval_sem_what_bleu4', metrics)
        for name in ('bleu1', 'bleu4', 'rouge_l'):
            self.assertAlmostEqual(metrics[f'eval_sem_overall_{name}'], 1.0)

    def test_malformed_raw_string_is_scored_without_repair(self):
        full = build_flat_target(gt())
        wrong = 'HOW: searched left then right\nWHAT 1: red bus'
        refs = {('q1', 1, 0): gt()}
        unit = _structured_unit_from_prediction(row('q1', wrong, valid=False), gt())
        self.assertNotEqual(unit['candidate'], unit['reference'])
        metrics = score_prediction_semantics([row('q1', wrong, valid=False)], refs,
                                             bleu1=True, bleu4=True, rouge_l=True)
        self.assertEqual(metrics['semantic_parse_failure_count'], 1)
        self.assertEqual(metrics['semantic_overall_missing_count'], 0)
        self.assertLess(metrics['eval_sem_overall_bleu1'], 1.0)
        self.assertLess(metrics['eval_sem_overall_rouge_l'], 1.0)

    def test_empty_prediction_kept_in_denominator(self):
        full = build_flat_target(gt())
        refs = {(q, 1, 0): gt() for q in ('q1', 'q2')}
        metrics = score_prediction_semantics(
            [row('q1', full), row('q2', None)], refs,
            bleu1=True, bleu4=True, rouge_l=True)
        self.assertEqual(metrics['semantic_overall_unit_count'], 2)
        self.assertEqual(metrics['semantic_overall_missing_count'], 1)
        self.assertAlmostEqual(metrics['eval_sem_overall_rouge_l'], 0.5)
        self.assertLess(metrics['eval_sem_overall_bleu1'], 1)

    def test_baseline_text_and_gt_guard(self):
        raw = build_flat_target(gt())
        unit = _structured_unit_from_prediction(row('q1', raw, baseline=True), gt())
        self.assertEqual(unit['candidate'], unit['reference'])
        bad = row('q1', raw)
        bad['SEMANTIC']['GT'] = 'contaminated GT'
        with self.assertRaisesRegex(ValueError, 'GT mismatch'):
            _structured_unit_from_prediction(bad, gt())

    def test_corpus_bleu1_not_sample_macro(self):
        units = [{'reference': 'a b c d', 'candidate': 'a b c d'},
                 {'reference': 'z', 'candidate': None}]
        self.assertLess(score_bleu1_corpus(units), 1.0)
        self.assertLess(score_bleu4_corpus(units), 1.0)
        self.assertAlmostEqual(score_rouge_l_corpus(units), 0.5)

    def test_group_table_reads_recomputed_overall(self):
        def artifact(b1, b4, mt, rouge, cider):
            return {'by_k': {'1': {'draw_mean': {
                f'eval_sem_overall_{key}': v for key, v in zip(
                    ('bleu1', 'bleu4', 'meteor', 'rouge_l', 'cider_r'),
                    (b1, b4, mt, rouge, cider))}}}}
        groups = {'TA': artifact(.4, .2, .3, .5, 1.0),
                  'TP': artifact(.6, .4, .5, .7, 1.4),
                  'VQA': artifact(.2, .1, .2, .3, .8)}
        fields, rows = make_semantic_metric_table(
            groups, artifact(.51, .29, .44, .58, 1.25), method='main(+end_fix)')
        self.assertEqual(len(fields), 21)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], 'main(+end_fix) (K=1)')
        self.assertEqual(rows[0][-5:], ['0.5100', '0.2900', '0.4400', '0.5800', '1.2500'])
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            paths = write_semantic_metrics_tables(directory, groups,
                artifact(.51, .29, .44, .58, 1.25), method='main(+end_fix)')
            self.assertEqual(len(Path(paths['csv']).read_text().splitlines()), 2)
            self.assertIn('Overall CIDEr-R ↑', Path(paths['markdown']).read_text())


if __name__ == '__main__':
    unittest.main()
