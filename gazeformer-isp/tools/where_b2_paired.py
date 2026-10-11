#!/usr/bin/env python3
"""Select reproducible K10 B2 paired cohorts and summarize canonical WHERE diagnostics.

Run from repository root.  Does NOT edit production code or prediction artifacts.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path: Path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def historics(path: Path, draw_id=0):
    rows = []
    for r in jsonl(path):
        if r.get('dataset', 'COCO-Search18') != 'COCO-Search18':
            continue
        if int(r.get('K', r.get('k', -1))) != 10 or int(r.get('draw_id', 0)) != draw_id:
            continue
        if not all(k in r for k in ('query_id', 'subject', 'support_ids', 'WHERE', 'where_generation')):
            raise ValueError(f'Historical K10 row missing fields: {r.get("query_id")}')
        rows.append(r)
    ids = [r['query_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate historical (query,K,draw) rows')
    return rows


def classify(row, *, end_fix_token='<END_FIX>'):
    from semgaze.where.serialization import parse_xyd_output
    pred = row['WHERE']['PRED']
    n = row['where_generation']['requested_fixation_count']
    if not isinstance(pred, str) or not isinstance(n, int) or n <= 0:
        raise ValueError('missing prediction text / oracle fixation length')
    parsed = parse_xyd_output(pred, n, end_fix_token)
    declared = row['where_generation'].get('canonical_format_valid')
    if declared is not None and bool(declared) != parsed['canonical_format_valid']:
        raise ValueError(f'Historical parser mismatch for {row["query_id"]}')
    if parsed['canonical_format_valid']:
        return 'valid'
    if pred and not pred.startswith('[(') and parse_xyd_output(
            '[(' + pred, n, end_fix_token)['canonical_format_valid']:
        return 'prefix_only'
    return 'other_invalid'


def check_supported_historical(row, label):
    if int(row['subject']) not in (7, 8, 9):
        raise ValueError(f'Unexpected subject {row["subject"]}')
    if len(row['support_ids']) != 10 or len(set(row['support_ids'])) != 10:
        raise ValueError(f'Bad frozen K10 support IDs for {row["query_id"]}')
    return {'query_id': row['query_id'], 'subject': int(row['subject']), 'category': label,
            'requested_fixation_count': row['where_generation']['requested_fixation_count'],
            'ordered_support_ids': row['support_ids']}


def excluded_ids(path):
    if not path:
        return set()
    if path.suffix == '.json':
        payload = json.loads(path.read_text(encoding='utf-8'))
        return set(payload.get('query_ids', []))
    return {l.strip() for l in path.read_text(encoding='utf-8').splitlines()
            if l.strip() and not l.lstrip().startswith('#')}


def make_selection(rows, *, mode, seed=2027, per_subject=30,
                   failures_per_subject=10, excluded=frozenset(),
                   end_fix_token='<END_FIX>'):
    labeled = [(r, classify(r, end_fix_token=end_fix_token)) for r in rows
               if r['query_id'] not in excluded]
    if mode == 'audit16':
        # Reproduce original audit's documented rank exactly: subject 8,
        # N in {3,4}, then query ID. Batches remain historical file independent.
        def rank(pair):
            r, _ = pair
            n = r['where_generation']['requested_fixation_count']
            return (str(r['subject']) != '8', n not in (3, 4), str(r['query_id']))
        bad = sorted((p for p in labeled if p[1] == 'prefix_only'), key=rank)[:12]
        good = sorted((p for p in labeled if p[1] == 'valid'), key=rank)[:4]
        if len(bad) != 12 or len(good) != 4:
            raise ValueError(f'audit16 needs 12 strict prefix-only failures / 4 valid; got {len(bad)}/{len(good)}')
        return bad + good
    if mode != 'balanced90':
        raise ValueError('unknown cohort mode')
    selected = []
    for subject in (7, 8, 9):
        bad = [p for p in labeled if p[1] == 'prefix_only' and int(p[0]['subject']) == subject]
        good = [p for p in labeled if p[1] == 'valid' and int(p[0]['subject']) == subject]
        for items, suffix in ((bad, 'bad'), (good, 'good')):
            rng = random.Random(int(hashlib.sha256(f'{seed}:{subject}:{suffix}'.encode()).hexdigest()[:16], 16))
            items.sort(key=lambda p: p[0]['query_id'])
            rng.shuffle(items)
        take_bad = min(len(bad), failures_per_subject)
        take_good = per_subject - take_bad
        # If there are fewer valid controls, fill with more prefix-only errors,
        # but never silently include unrelated malformed categories.
        if len(good) < take_good:
            take_good = len(good)
            take_bad = per_subject - take_good
        if len(bad) < take_bad or len(good) < take_good or take_good < 1:
            raise ValueError(f'subject {subject}: cannot select {per_subject} prefix/valid queries '
                             f'after exclusions (bad={len(bad)}, valid={len(good)})')
        selected.extend(bad[:take_bad] + good[:take_good])
    if len(selected) != per_subject * 3:
        raise AssertionError('incomplete balanced cohort')
    return selected


def select(args):
    rows = historics(args.predictions_jsonl, args.draw_id)
    selected = make_selection(rows, mode=args.mode, seed=args.seed,
        per_subject=args.per_subject, failures_per_subject=args.failures_per_subject,
        excluded=excluded_ids(args.exclude), end_fix_token=args.end_fix_token)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('selection output directory must be empty')
    entries = [check_supported_historical(r, category) for r,category in selected]
    ids = [r['query_id'] for r in entries]
    if len(ids) != len(set(ids)):
        raise AssertionError('duplicate query identities in selection')
    (out/'query_ids.txt').write_text('\n'.join(ids)+'\n', encoding='utf-8')
    metadata = {'mode':args.mode, 'query_ids':ids, 'entries':entries,
        'seed':args.seed, 'draw_id':args.draw_id, 'original_sha256':sha256(args.predictions_jsonl),
        'excluded_file':str(args.exclude) if args.exclude else None,
        'category_counts':dict(Counter(r['category'] for r in entries)),
        'subject_counts':{str(s):dict(Counter(r['category'] for r in entries if r['subject']==s))
                          for s in (7,8,9)}}
    (out/'selection.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+'\n',encoding='utf-8')
    with (out/'selection.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['query_id','subject','category','requested_fixation_count'])
        w.writeheader();w.writerows({k:v for k,v in r.items() if k!='ordered_support_ids'} for r in entries)
    print(f'{args.mode}: {len(ids)} selected; breakdown: {json.dumps(metadata["subject_counts"])}')
    print(f'Use --query-ids-file {out / "query_ids.txt"}')


def compare_pair(row, base, *, score_fn=None, truth=None):
    if row['query_id'] != base['query_id'] or row['support_ids'] != base['support_ids']:
        raise ValueError('query ID or exact ordered K10 supports differ across variants')
    if row['K'] != 10 or base['K'] != 10 or row['draw_id'] != base['draw_id']:
        raise ValueError('K/draw changed across paired variants')
    if row['requested_fixation_count'] != base['requested_fixation_count']:
        raise ValueError('requested fixation count changed across paired variants')
    if row['checkpoint'] != base['checkpoint'] or row['split_manifest_identity'] != base['split_manifest_identity']:
        raise ValueError('checkpoint/split mismatch across variants')
    if row['generation'] != base['generation']:
        raise ValueError('generation settings changed across variants')
    b, c = bool(base['canonical_format_valid']), bool(row['canonical_format_valid'])
    entry = {'query_id':row['query_id'], 'subject':int(row['subject']),
        'category_historical': 'prefix_only' if base['historical_prefix_bad'] else 'historically_valid_or_other',
        'baseline_valid':b, 'concise_valid':c,
        'fixed':not b and c, 'broken':b and not c,
        'baseline_starts_with_open':bool(base['starts_with_open']),
        'concise_starts_with_open':bool(row['starts_with_open']),
        'baseline_accepted_count':base['accepted_fixation_count'],
        'concise_accepted_count':row['accepted_fixation_count'],
        'baseline_under_generated':bool(base['under_generated']),
        'concise_under_generated':bool(row['under_generated']),
        'requested_fixation_count':row['requested_fixation_count'],
        'tuple_equal':base['fixations']==row['fixations'],
        'baseline_tokens':base['prompt_token_length'], 'concise_tokens':row['prompt_token_length']}
    if score_fn:
        for label, source in (('original',base), ('concise',row)):
            score=score_fn(truth,source)
            for key in ('sm','mm','sed'):
                entry[f'{label}_{key}']=score[key]
            entry[f'{label}_scanpath_score']=score
        for key in ('sm','mm','sed'):
            entry[f'delta_{key}']=entry[f'concise_{key}']-entry[f'original_{key}']
    return entry


def metric_score(truth, observation):
    from semgaze.evaluation.metrics_scanpath import score_scanpath_pair
    xy=truth['gt_raw_pixels']
    ds=truth['gt_duration_ms_raw']
    if len(xy) != len(ds) or not xy:
        raise ValueError(f'invalid historical GT pixels/durations: {truth["query_id"]}')
    gt=[(x,y,d) for (x,y),d in zip(xy,ds)]
    score=score_scanpath_pair(gt, observation['fixations'],
        gt_image_width=int(truth['annotation_width']),
        gt_image_height=int(truth['annotation_height']),
        require_multimatch=True, compute_multimatch=True,
        compute_scanmatch=True, compute_sed=True)
    score['format_failure']=not bool(observation['canonical_format_valid'])
    return score


def group_summary(rows):
    result={'N':len(rows), 'baseline_valid':sum(r['baseline_valid'] for r in rows),
            'concise_valid':sum(r['concise_valid'] for r in rows),
            'fixed':sum(r['fixed'] for r in rows), 'broken':sum(r['broken'] for r in rows),
            'baseline_prefix_valid':sum(r['baseline_starts_with_open'] for r in rows),
            'concise_prefix_valid':sum(r['concise_starts_with_open'] for r in rows),
            'baseline_under_generated':sum(r['baseline_under_generated'] for r in rows),
            'concise_under_generated':sum(r['concise_under_generated'] for r in rows),
            'tuple_unchanged':sum(r['tuple_equal'] for r in rows),
            'baseline_mean_prompt_tokens':round(statistics.mean(r['baseline_tokens'] for r in rows),1),
            'concise_mean_prompt_tokens':round(statistics.mean(r['concise_tokens'] for r in rows),1)}
    if 'original_sm' in rows[0]:
        for label in ('original','concise'):
            from semgaze.evaluation.metrics_scanpath import aggregate_scanpath_draw
            aggregate=aggregate_scanpath_draw([r[f'{label}_scanpath_score'] for r in rows])
            for metric in ('sm','mm','sed'):
                result[f'{label}_{metric}']=round(aggregate[f'eval_where_{metric}'],6)
        for metric in ('sm','mm','sed'):
            result[f'delta_{metric}']=round(result[f'concise_{metric}']-result[f'original_{metric}'],6)
    return result


def report(args):
    rows=list(jsonl(args.diagnostics))
    by_query=defaultdict(dict)
    for r in rows:
        if r['condition'] not in ('A5_original10','B2_concise'):
            continue
        q=r['query_id']
        if r['condition'] in by_query[q]:
            raise ValueError(f'duplicate observation: {q} / {r["condition"]}')
        by_query[q][r['condition']]=r
    if not by_query or any(set(variants)!= {'A5_original10','B2_concise'} for variants in by_query.values()):
        raise ValueError('require paired A5_original10 and B2_concise for every query')
    truths={r['query_id']:r for r in historics(args.predictions_jsonl,args.draw_id)}
    if not set(by_query).issubset(truths):
        raise ValueError('missing historical GT queries')
    pairs=[]
    for q,variants in by_query.items():
        base,concise=variants['A5_original10'],variants['B2_concise']
        if base['historical_prediction'] != truths[q]['WHERE']['PRED']:
            raise ValueError(f'baseline historical prediction differs: {q}')
        if list(base['support_ids']) != list(truths[q]['support_ids']):
            raise ValueError(f'historical frozen support mismatch: {q}')
        pairs.append(compare_pair(concise,base,truth=truths[q],
            score_fn=None if args.skip_metrics else metric_score))
    if args.selection:
        selection=json.loads(args.selection.read_text(encoding='utf-8'))
        if set(by_query)!=set(selection['query_ids']):
            raise ValueError('runtime queries differ from saved selection')
        if sha256(args.predictions_jsonl)!=selection['original_sha256']:
            raise ValueError('historical prediction JSONL hash changed')
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('report output directory must be empty')
    summary={'all':group_summary(pairs),
             'subjects':{str(subject):group_summary([r for r in pairs if r['subject']==subject])
                for subject in (7,8,9) if any(r['subject']==subject for r in pairs)}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    (out/'query_transitions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs),encoding='utf-8')
    keys=[k for k in pairs[0] if k not in ('original_scanpath_score','concise_scanpath_score')]
    with (out/'query_transitions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows({k:v for k,v in row.items() if k in keys} for row in pairs)
    lines=['# Paired B1 original vs B2 concise', '',
           'All pairs share K=10, draw, checkpoint, ordered frozen supports and decoding settings.',
           'Scanpath metrics (if computed) use production ISP adapters and parser-recovered tuples. '
           'This is a selected cohort, not an unbiased official evaluation.', '',
           '| Group | N | Valid B1 | Valid B2 | Fixed | Broken | Under-generated B1 | Under-generated B2 | Tuple unchanged | SM delta | MM delta | SED delta |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, group in [('all',summary['all'])]+[(s,g) for s,g in summary['subjects'].items()]:
        lines.append('| '+ ' | '.join(str(x) for x in [name, group['N'],group['baseline_valid'],group['concise_valid'],
            group['fixed'],group['broken'],group['baseline_under_generated'],group['concise_under_generated'],
            group['tuple_unchanged'],group.get('delta_sm','NOT SCORED'),group.get('delta_mm','NOT SCORED'),
            group.get('delta_sed','NOT SCORED')])+' |')
    lines+=['','SM/MM higher is better; SED lower is better. Positive deltas mean B2 minus B1.',
            'For malformed nonempty predictions, the production parser recovers fixation tuples for metrics.',
            'A larger canonical-format rate alone is NOT evidence of improved scanpath quality.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sel=sub.add_parser('select',help='write exact reproducible query cohort and metadata')
    sel.add_argument('--predictions-jsonl',type=Path,required=True)
    sel.add_argument('--mode',choices=('audit16','balanced90'),required=True)
    sel.add_argument('--draw-id',type=int,default=0)
    sel.add_argument('--seed',type=int,default=2027)
    sel.add_argument('--per-subject',type=int,default=30)
    sel.add_argument('--failures-per-subject',type=int,default=10)
    sel.add_argument('--exclude',type=Path,help='exclusion selection.json or query_ids.txt')
    sel.add_argument('--end-fix-token',default='<END_FIX>')
    sel.add_argument('--output-dir',type=Path,required=True)
    sel.set_defaults(func=select)
    rep=sub.add_parser('report',help='score paired records with canonical ISP metrics')
    rep.add_argument('--diagnostics',type=Path,required=True)
    rep.add_argument('--predictions-jsonl',type=Path,required=True)
    rep.add_argument('--selection',type=Path)
    rep.add_argument('--draw-id',type=int,default=0)
    rep.add_argument('--skip-metrics',action='store_true',help='format-only output if ISP reference dependencies are unavailable')
    rep.add_argument('--output-dir',type=Path,required=True)
    rep.set_defaults(func=report)
    a=parser.parse_args(argv)
    if a.command=='select' and (a.per_subject<1 or a.failures_per_subject<0 or
                                a.failures_per_subject>=a.per_subject):
        parser.error('per-subject>=1; failures-per-subject in [0,per-subject)')
    a.func(a)


if __name__=='__main__':
    main()
