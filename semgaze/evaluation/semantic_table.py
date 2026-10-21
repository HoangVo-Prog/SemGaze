"""Publication-ready TA/TP/VQA/Overall metric tables from canonical corpus scores.

Never average group-level BLEU/CIDEr: Overall is separately rescored on the
union of predictions by score_prediction_artifact.
"""
from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

GROUPS = ('TA', 'TP', 'VQA', 'Overall')
SCORES = (('B1', 'bleu1'), ('B4', 'bleu4'), ('METEOR', 'meteor'),
          ('ROUGE', 'rouge_l'), ('CIDEr-R', 'cider_r'))


def _scores_by_k(artifact):
    if artifact is None:
        return {}
    return {str(k): block.get('draw_mean', {})
            for k, block in artifact.get('by_k', {}).items()}


def make_semantic_metric_table(by_group, overall, *, method='SemGaze'):
    """Return (headers, data_rows) with values rounded to four decimals.

    `-` denotes a group not in the dataset or a disabled metric. Each row is
    one support-K, after the evaluator's configured draw aggregation.
    """
    corpora = {group: _scores_by_k(by_group.get(group)) for group in GROUPS[:-1]}
    corpora['Overall'] = _scores_by_k(overall)
    k_values = sorted(corpora['Overall'], key=lambda value: int(value))
    fields = ['Method'] + [f'{group} {label} ↑' for group in GROUPS
                           for label, _ in SCORES]
    rows = []
    for k in k_values:
        scores = corpora['Overall'].get(k, {})
        if not any(name.startswith('eval_sem_overall_') for name in scores):
            continue
        row = [f'{method} (K={k})']
        for group in GROUPS:
            metrics = corpora[group].get(k, {})
            for _, metric in SCORES:
                value = metrics.get('eval_sem_overall_' + metric)
                row.append(f'{value:.4f}' if isinstance(value, (int, float)) else '-')
        rows.append(row)
    return fields, rows


def write_semantic_metrics_tables(root, by_group, overall, *, method='SemGaze'):
    """Write standalone CSV/Markdown table without resampling/averaging groups."""
    headers, rows = make_semantic_metric_table(by_group, overall, method=method)
    if not rows:
        return {}
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    csv_file = root / 'semantic_metrics_table.csv'
    markdown_file = root / 'semantic_metrics_table.md'
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    csv_file.write_text(buffer.getvalue(), encoding='utf-8')
    md = ['| ' + ' | '.join(headers) + ' |',
          '|:-------' + '|--------:' * (len(headers) - 1) + '|']
    md += ['| ' + ' | '.join(row) + ' |' for row in rows]
    markdown_file.write_text('\n'.join(md) + '\n', encoding='utf-8')
    return {'csv': str(csv_file), 'markdown': str(markdown_file)}
