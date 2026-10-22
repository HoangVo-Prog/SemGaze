"""Inference-only K-shot scanpath transfer to ScanDiff OSIE/COCO-FreeView.

Uses a SemGaze flat checkpoint with its original model configuration, plus
same-observer K-shot references from the target dataset's TRAIN split. No
checkpoint training/updates and no semantic annotations are required.

Subject split defaults to the OFFICIAL FS-PSP main-paper split: OSIE
0..9 seen / 10..14 unseen; COCO-FreeView 3..9 seen / 0..2 unseen.
Raw ScanDiff subject numbering is verified (0-based or 1-based) before
mapping these identities. ``seen`` denotes FS-PSP's target observer group,
NOT source-checkpoint observers. Query fixation COUNT is oracle-conditioned.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby
import re
import hashlib
import json
import math
from pathlib import Path
import random

from semgaze.data.schema import FlatEpisode
from semgaze.evaluation import metrics_scanpath as scanpath_metrics
from semgaze.evaluation.metrics_scanpath import (
    CoordinateAdapter, aggregate_scanpath_draw, score_scanpath_pair,
)
from semgaze.evaluation.test import test_mode
from semgaze.model.checkpoint import load_checkpoint_bundle
from semgaze.model.visual_cache import InferenceVisualCache
from semgaze.where.generation import generate_where_batch
from semgaze.where.serialization import active_end_fix_token, serialize_xyd_record


FRAMES = {'osie': (800, 600), 'coco_freeview': (1680, 1050)}
# Screen sizes in FS-PSP's official ISP/{dataset}/GazeformerISP/src/opts.py.
EVAL_FRAMES = {'osie': (512, 384), 'coco_freeview': (512, 320)}
LABELS = {'osie': 'OSIE', 'coco_freeview': 'COCO-FreeView'}
# Neutral (non-search) instruction; deliberately distinct from COCO target labels.
FREEVIEW_TASK = 'free viewing'

# Source: cvlab-stonybrook/few-shot-scanpath (CVPR 2025, FS-PSP):
# ISP/OSIE/GazeformerISP/README.md, bash/train.sh
# ISP/COCO_FV/GazeformerISP/README.md, bash/train.sh
# These are main-paper Table 1 splits, NOT supplemental second split.
FSPSP_MAIN_SPLIT = {
    'osie': {'total_subjects': 15, 'unseen_zero_based': (10, 11, 12, 13, 14)},
    'coco_freeview': {'total_subjects': 10, 'unseen_zero_based': (0, 1, 2)},
}
FSPSP_SPLIT_SOURCE = 'https://github.com/cvlab-stonybrook/few-shot-scanpath'


@dataclass(frozen=True)
class TransferRecord:
    record_id: str
    stimulus_id: str
    subject: str
    image_path: str
    image_width: int  # Annotation frame, NOT resized pixel-tensor dimensions.
    image_height: int
    task: str
    condition: str
    x_px: tuple[float, ...]
    y_px: tuple[float, ...]
    duration_ms: tuple[float, ...]
    semantic: None
    dataset: str


def scan_file(root: Path, dataset: str, split: str) -> Path:
    if dataset == 'osie':
        return root / 'processed_fixations' / f'osie_fixations_{split}.json'
    return root / f'COCOFreeView_fixations_{split}.json'


def resolve_root(parent: Path, dataset: str, explicit: Path | None) -> Path:
    candidates = ([explicit] if explicit is not None else
                  [parent / dataset, parent / 'data' / dataset, parent,
                   parent / dataset / dataset])
    for root in candidates:
        if scan_file(root, dataset, 'train').is_file():
            return root.resolve()
    raise FileNotFoundError(f'{dataset}: cannot find ScanDiff train JSON in {candidates}')


def read_split(root: Path, dataset: str, split: str, max_fixations: int):
    path = scan_file(root, dataset, split)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, list):
        raise ValueError(f'{path}: expected a JSON list')
    width, height = FRAMES[dataset]
    images_root = (root / 'images').resolve()
    records = []
    info = dict(source_rows=len(payload), empty_skipped=0, truncated_sequences=0)
    for index, item in enumerate(payload):
        if item.get('split') != split:
            raise ValueError(f'{path} row {index}: split mismatch')
        name = item['name']
        subject = str(item['subject'])
        if not isinstance(name, str) or not name or not subject.strip():
            raise ValueError(f'{path} row {index}: invalid subject or image')
        xs, ys, ts = item['X'], item['Y'], item['T']
        if not isinstance(xs, list) or not isinstance(ys, list) or not isinstance(ts, list):
            raise ValueError(f'{path} row {index}: X/Y/T must be lists')
        if len(xs) != len(ys) or len(xs) != len(ts):
            raise ValueError(f'{path} row {index}: X/Y/T length mismatch')
        if not xs:
            info['empty_skipped'] += 1
            continue
        if len(xs) > max_fixations:
            info['truncated_sequences'] += 1
        points = [(float(x), float(y), float(t))
                  for x, y, t in zip(xs[:max_fixations], ys[:max_fixations], ts[:max_fixations])]
        if any(not all(math.isfinite(v) for v in p) or not
               (0 <= p[0] < width and 0 <= p[1] < height and p[2] >= 0)
               for p in points):
            raise ValueError(f'{path} row {index}: out-of-frame/nonfinite fixation; '
                             f'expected {width}x{height} original coordinate frame, T in ms')
        image = (images_root / name).resolve()
        if not image.is_relative_to(images_root) or not image.is_file():
            raise FileNotFoundError(f'{path} row {index}: missing image: {image}')
        records.append(TransferRecord(
            f'{dataset}::{split}::{subject}::{index}', name, subject, str(image),
            width, height, FREEVIEW_TASK, 'freeview',
            tuple(p[0] for p in points), tuple(p[1] for p in points),
            tuple(p[2] for p in points), None, LABELS[dataset],
        ))
    if not records:
        raise ValueError(f'{path}: no usable fixation records')
    info['usable_rows'] = len(records)
    return records, info


def load_dataset(root: Path, dataset: str, max_fixations: int):
    train, train_info = read_split(root, dataset, 'train', max_fixations)
    test, test_info = read_split(root, dataset, 'test', max_fixations)
    overlap = {r.stimulus_id for r in train} & {r.stimulus_id for r in test}
    if overlap:
        raise ValueError(f'{dataset}: train/test image leakage: {len(overlap)} stimuli')
    return train, test, {'train': train_info, 'test': test_info}


def parse_unseen_subjects(value: str | None) -> set[str]:
    if not value:
        raise ValueError('manual protocol requires explicit --unseen-subjects '
                         'or dataset-specific --osie-unseen-subjects / '
                         '--coco-freeview-unseen-subjects')
    values = {v.strip() for v in value.split(',') if v.strip()}
    if not values:
        raise ValueError('target unseen-subject group cannot be empty')
    return values


def fspsp_main_subject_split(dataset: str, train, test, id_base='auto'):
    """Translate official FS-PSP 0-based observer IDs to ScanDiff raw IDs.

    Unlike guessing 'last 30%', this checks the FULL subject universe. A
    malformed/partial/different subject set fails loudly, rather than silently
    assigning an unrelated ScanDiff observer to a named FS-PSP identity.
    """
    if id_base not in ('auto', '0', '1'):
        raise ValueError('FS-PSP subject ID base must be auto, 0, or 1')
    ref = FSPSP_MAIN_SPLIT[dataset]
    n = ref['total_subjects']
    found_train = {r.subject for r in train}
    found_test = {r.subject for r in test}
    full = found_train | found_test
    candidates = {
        base: {str(x) for x in range(base, base + n)} for base in (0, 1)
    }
    if id_base == 'auto':
        matches = [base for base in (0, 1) if full == candidates[base]]
        if len(matches) != 1:
            raise ValueError(
                f'{dataset}: ScanDiff observer IDs {sorted(full)} do not match '
                f'the full official FS-PSP population 0..{n-1} or 1..{n}; '
                'cannot safely map observer identities. Check the JSON source, '
                'then specify --fspsp-subject-id-base 0/1 if applicable. '
                'Do NOT use manual IDs to claim FS-PSP main-table parity.'
            )
        base = matches[0]
    else:
        base = int(id_base)
        if full != candidates[base]:
            raise ValueError(f'{dataset}: --fspsp-subject-id-base {base} expects '
                             f'{sorted(candidates[base])}, observed {sorted(full)}')
    # Main-paper Table 1 unseen IDs, shifted only if ScanDiff numbers 1..N.
    unseen = {str(x + base) for x in ref['unseen_zero_based']}
    seen = candidates[base] - unseen
    if not unseen <= found_test or not seen <= found_test:
        raise ValueError(f'{dataset}: target test split lacks a complete FS-PSP '
                         f'partition: missing {sorted(candidates[base] - found_test)}')
    if not candidates[base] <= found_train:
        raise ValueError(f'{dataset}: target train split lacks per-observer supports '
                         f'for {sorted(candidates[base] - found_train)}')
    return unseen, {
        'protocol': 'fspsp_main_table1',
        'source': FSPSP_SPLIT_SOURCE,
        'reference_subject_id_base': 0,
        'scandiff_subject_id_base': base,
        'official_unseen_zero_based': list(ref['unseen_zero_based']),
        'target_unseen_ids': sorted(unseen, key=int),
        'target_seen_ids': sorted(seen, key=int),
        'mapped_from_zero_based': base == 1,
    }


def partition_queries(train, test, unseen: set[str], scope: str):
    test_ids = {r.subject for r in test}
    train_ids = {r.subject for r in train}
    if not unseen <= test_ids:
        raise ValueError(f'unseen IDs {sorted(unseen - test_ids)} not in test; '
                         f'available subject IDs: {sorted(test_ids)}')
    if not test_ids - unseen:
        raise ValueError('seen partition would be empty')
    if not test_ids <= train_ids:
        raise ValueError(f'target subjects without same-observer TRAIN supports: '
                         f'{sorted(test_ids - train_ids)}')
    groups = {}
    for name in ('seen', 'unseen'):
        if scope in (name, 'both'):
            groups[name] = [q for q in test if (q.subject in unseen) == (name == 'unseen')]
            if not groups[name]:
                raise ValueError(f'no queries in target {name} partition')
    return groups


def make_support_plan(train, query_subjects, k_values, draws, seed,
                      *, shared_stimuli=True):
    """Select TRAIN supports; FS-PSP samples the same K image names per group.

    Their dataset.py::select_fewshot_subject shuffles image names first and
    then gathers scanpaths from ALL selected observers on those same images.
    We use SHA256 seeds and sorted candidates to be deterministic across
    runs/Python hash randomization. Exact paper sample identities require the
    authors' saved ``sampling/{K}_shot/sample_{seed}.json`` files, if available.
    """
    query_subjects = set(query_subjects)
    if not query_subjects:
        raise ValueError('no observers to build support sets')
    unique_images = defaultdict(dict)
    for r in train:
        unique_images[r.subject].setdefault(r.stimulus_id, r)
    plan = {}
    if shared_stimuli:
        common_images = set.intersection(
            *(set(unique_images[s]) for s in sorted(query_subjects)))
        ordered_images = sorted(common_images)
        if len(ordered_images) < max(k_values) * draws:
            raise ValueError(
                f'FS-PSP shared-stimulus support pool has {len(ordered_images)} '
                f'images across {sorted(query_subjects)}; need at least '
                f'{max(k_values) * draws}. Check ScanDiff split/subject coverage.'
            )
        for k in k_values:
            rng = hashlib.sha256(
                f'fspsp-shared-support-v1|{seed}|{train[0].dataset}|K={k}|'
                f'{",".join(sorted(query_subjects))}'.encode()).digest()
            names = list(ordered_images)
            random.Random(int.from_bytes(rng, 'big')).shuffle(names)
            for draw in range(draws):
                selected = names[draw*k:(draw+1)*k]
                for subject in sorted(query_subjects):
                    plan[(subject, k, draw)] = tuple(
                        unique_images[subject][name] for name in selected)
    else:
        for subject in sorted(query_subjects):
            pool = unique_images[subject]
            for k in k_values:
                if len(pool) < k * draws:
                    raise ValueError(f'{subject}: requires {k * draws} distinct train '
                                     f'images for K={k} draws={draws}; only {len(pool)}')
                seed_text = f'transfer-freeview-v1|{seed}|{train[0].dataset}|{subject}|K={k}'
                digest = hashlib.sha256(seed_text.encode()).digest()
                names = sorted(pool)
                random.Random(int.from_bytes(digest, 'big')).shuffle(names)
                for draw in range(draws):
                    plan[(subject, k, draw)] = tuple(
                        pool[name] for name in names[draw * k:(draw + 1) * k])
    return plan



def parse_batch_sizes_by_k(spec, k_values, fallback):
    """Explicit per-K physical batch sizes; preserve --batch-size as fallback.

    Accept e.g. ['1:4', '5:2', '10:1'] or ['1=4,5=2,10=1'].
    Extra valid K overrides are allowed when evaluating a K subset; the
    returned mapping only contains requested K values.
    """
    if type(fallback) is not int or fallback < 1:
        raise ValueError('--batch-size must be a positive integer')
    entries = []
    for part in spec or ():
        entries.extend(token for token in re.split(r'[,\s]+', part.strip()) if token)
    overrides = {}
    for entry in entries:
        match = re.fullmatch(r'(\d+)\s*[:=]\s*(\d+)', entry)
        if match is None:
            raise ValueError(f'invalid --batch-size-by-k item {entry!r}; '
                             'expected K:B or K=B, such as 1:4,5:2,10:1')
        k, size = (int(part) for part in match.groups())
        if not 1 <= k <= 10:
            raise ValueError(f'--batch-size-by-k requires K within 1..10: {k}')
        if k in overrides:
            raise ValueError(f'duplicate batch-size override for K={k}')
        if size < 1:
            raise ValueError(f'physical batch size for K={k} must be positive')
        overrides[k] = size
    return {k: overrides.get(k, fallback) for k in k_values}


def scheduled_episode_batches(episodes, batch_size, *, bucket_by_length=True):
    """Yield same-K physical batches, optionally grouped by query length.

    generate_where_batch normally subdivides mixed-length batches by generation
    budget. Grouping oracle-length queries up front avoids internal microbatches
    and improves utilization. Each episode is included exactly once.
    """
    if batch_size < 1:
        raise ValueError('batch size must be positive')
    if not episodes:
        return
    if len({len(ep.supports) for ep in episodes}) != 1:
        raise ValueError('physical batch scheduler must receive a single K')
    indexed = list(enumerate(episodes))
    if bucket_by_length:
        indexed.sort(key=lambda pair: (len(pair[1].query.x_px), pair[0]))
        grouped = (list(rows) for _, rows in groupby(
            indexed, key=lambda pair: len(pair[1].query.x_px)))
    else:
        grouped = (indexed,)
    for group in grouped:
        for start in range(0, len(group), batch_size):
            selected = group[start:start + batch_size]
            yield [ep for _, ep in selected]


def episodes_for(queries, plan, k, draw):
    return [FlatEpisode(plan[(q.subject, k, draw)], q, draw_id=draw) for q in queries]


@lru_cache(maxsize=2)
def _osie_scanmatch(with_duration: bool):
    # Same ScanMatch reference/version as the audited SemGaze metrics, with
    # OSIE-specific Yres=384 from the official FS-PSP evaluator.
    ScanMatch = scanpath_metrics._load_reference_scanmatch()
    kwargs = dict(Xres=512, Yres=384, Xbin=16, Ybin=12,
                  Offset=(0, 0), Threshold=3.5)
    if with_duration:
        kwargs['TempBin'] = 50
    return ScanMatch(**kwargs)


def score_osie_pair(query, prediction):
    """OSIE reference frame = 512x384 (NOT the COCO 512x320 frame)."""
    import numpy as np

    raw_w, raw_h = FRAMES['osie']
    width, height = EVAL_FRAMES['osie']
    gt = [(float(x)*width/raw_w, float(y)*height/raw_h, float(t))
          for x, y, t in zip(query.x_px, query.y_px, query.duration_ms)]
    pred = [(float(round(int(x)/100.0*width)),
             float(round(int(y)/100.0*height)), float(t))
            for x, y, t in prediction]
    if not pred:
        return dict(sm=0.0, sm_no_duration=0.0, sm_with_duration=0.0,
                    mm=0.0, mm_components=[0.0]*5, sed=len(gt), valid=False,
                    format_failure=True, mm_numeric_failure=False)
    for points in (gt, pred):
        if any(not 0 <= x < width or not 0 <= y < height or t < 0
               for x, y, t in points):
            raise ValueError('out-of-frame OSIE scanpath after 512x384 scaling')

    sm = {}
    for temporal in (False, True):
        matcher = _osie_scanmatch(temporal)
        a = np.asarray(gt if temporal else [(x, y) for x, y, _ in gt], dtype=float)
        b = np.asarray(pred if temporal else [(x, y) for x, y, _ in pred], dtype=float)
        match_gt = matcher.fixationToSequence(a).astype(np.int32)
        match_pred = matcher.fixationToSequence(b).astype(np.int32)
        value = float(matcher.match(match_gt, match_pred)[0])
        if not math.isfinite(value):
            if np.array_equal(match_gt, match_pred):
                value = 1.0  # Official FS-PSP identical-sequence NaN extension.
            else:
                raise ValueError('ScanMatch returned nonfinite OSIE score')
        sm[temporal] = value

    # Reuse the audited MultiMatch wrapper's padding/second-unit convention,
    # but pass the correct OSIE screen size instead of hardcoded 512x320.
    scorer = scanpath_metrics._multimatch_module()
    to_structured = scanpath_metrics._multimatch_points
    mm_gt = to_structured([(x, y, t/1000.0) for x, y, t in gt])
    mm_pred = to_structured([(x, y, t/1000.0) for x, y, t in pred])
    mm = [float(v) for v in scorer.docomparison(
        mm_gt, mm_pred, screensize=[width, height])]
    if len(mm) != 5:
        raise ValueError('OSIE MultiMatch must return five components')
    mm_failed = any(not math.isfinite(v) for v in mm)
    if mm_failed:
        mm = [0.0]*5  # Keep valid evaluation coverage, as SemGaze does.

    vame = scanpath_metrics._load_reference_vame()
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    sed = int(vame.string_edit_distance(
        canvas, np.asarray(gt, dtype=float), np.asarray(pred, dtype=float)))
    return dict(sm=scanpath_metrics._hmean([sm[False], sm[True]]),
                sm_no_duration=sm[False], sm_with_duration=sm[True],
                mm=sum(mm)/5.0, mm_components=mm, sed=sed,
                valid=True, format_failure=False, mm_numeric_failure=mm_failed)


def score_prediction(query, generated):
    if query.dataset == LABELS['osie']:
        return score_osie_pair(query, generated['fixations'])
    # COCO-FreeView shares the official 512x320 target frame with the audited
    # SemGaze/COCO-Search18 scorer; explicitly override ORIGINAL GT dimensions.
    return score_scanpath_pair(
        list(zip(query.x_px, query.y_px, query.duration_ms)),
        generated['fixations'], adapter=CoordinateAdapter(),
        gt_coordinate_space='raw_coco',
        gt_image_width=query.image_width, gt_image_height=query.image_height,
    )


def summarize_draws(scores, under_generated, format_failures):
    ordered = sorted(scores)
    per_draw = {str(i): aggregate_scanpath_draw(scores[i]) for i in ordered}
    values = list(per_draw.values())
    return {
        'sm': sum(v['eval_where_sm'] for v in values) / len(values),
        'mm': sum(v['eval_where_mm'] for v in values) / len(values),
        'sed': sum(v['eval_where_sed'] for v in values) / len(values),
        'queries_per_draw': len(scores[ordered[0]]),
        'n_draws': len(ordered),
        'under_generated_rate': under_generated / sum(len(v) for v in scores.values()),
        'noncanonical_format_rate': format_failures / sum(len(v) for v in scores.values()),
        'by_draw': per_draw,
    }


def write_table(path: Path, results):
    rows = [
        '| Dataset | Target partition | n-shot | SM ↑ | MM ↑ | SED ↓ | Queries | Draws |',
        '|:--|:--|--:|--:|--:|--:|--:|--:|',
    ]
    for dataset, scopes in results.items():
        for scope, k_blocks in scopes.items():
            for k, block in sorted(k_blocks.items(), key=lambda pair: int(pair[0])):
                rows.append(f"| {LABELS[dataset]} | {scope} | {k} | "
                            f"{block['sm']:.4f} | {block['mm']:.4f} | {block['sed']:.4f} | "
                            f"{block['queries_per_draw']} | {block['n_draws']} |")
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8')


def write_fspsp_table(path: Path, results):
    """Print the Table-1 layout ONLY for FS-PSP's held-out unseen group."""
    targets = ('osie', 'coco_freeview')
    rows = [
        '| n-shot | OSIE SM ↑ | OSIE MM ↑ | OSIE SED ↓ | '
        'COCO-FreeView SM ↑ | COCO-FreeView MM ↑ | COCO-FreeView SED ↓ |',
        '|:--:|--:|--:|--:|--:|--:|--:|',
    ]
    available = sorted({int(k) for d in targets for k in results.get(d, {}).get('unseen', {})})
    for k in available:
        row = [f'n = {k}']
        for d in targets:
            block = results.get(d, {}).get('unseen', {}).get(str(k))
            row += ([f"{block['sm']:.4f}", f"{block['mm']:.4f}", f"{block['sed']:.4f}"]
                    if block is not None else ['—', '—', '—'])
        rows.append('| ' + ' | '.join(row) + ' |')
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8')


def run(args):
    if not args.k_values or len(set(args.k_values)) != len(args.k_values) or any(
            not 1 <= k <= 10 for k in args.k_values):
        raise ValueError('K must be unique integers in [1,10]')
    if args.draws < 1 or args.batch_size < 1 or not 1 <= args.max_fixations <= 10:
        raise ValueError('draws, batch-size must be positive; max-fixations in [1,10]')
    physical_batch_sizes = parse_batch_sizes_by_k(
        args.batch_size_by_k, args.k_values, args.batch_size)
    checkpoint = args.checkpoint.expanduser().resolve()
    meta = json.loads((checkpoint / 'metadata.json').read_text(encoding='utf-8'))
    identity = meta.get('split_manifest_identity')
    if not identity:
        raise ValueError('checkpoint missing source split_manifest_identity')
    names = list(FRAMES) if args.dataset == 'both' else [args.dataset]
    prepared = {}
    protocol = {
        'checkpoint': str(checkpoint), 'checkpoint_split_manifest_identity': identity,
        'no_finetuning': True, 'oracle_query_fixation_count': True,
        'subject_scope_meaning': 'FS-PSP target-dataset seen/unseen groups (not source-checkpoint identities)',
        'subject_split_protocol': args.split_protocol,
        'support_sampling_protocol': 'shared_training_stimuli_across_target_observers',
        'support_split': 'train', 'query_split': 'test',
        'fixed_annotation_frames': FRAMES, 'metric_frames': EVAL_FRAMES,
        'max_fixations': args.max_fixations,
        'k_values': args.k_values, 'draws': args.draws, 'seed': args.seed,
        'batch_size_by_k': {str(k): v for k, v in physical_batch_sizes.items()},
        'bucket_by_fixation_count': args.bucket_by_fixation_count,
        'datasets': {},
    }
    for name in names:
        root = resolve_root(args.scandiff_data.expanduser().resolve(), name,
                            getattr(args, name + '_root'))
        train, test, stats = load_dataset(root, name, args.max_fixations)
        override = getattr(args, name + '_unseen_subjects') or args.unseen_subjects
        if args.split_protocol == 'fspsp-main':
            if override is not None:
                raise ValueError('FS-PSP main protocol uses fixed official observer '
                                 'identities; remove --unseen-subjects flags or use '
                                 '--split-protocol manual for a non-paper split')
            unseen, split_detail = fspsp_main_subject_split(
                name, train, test, args.fspsp_subject_id_base)
        else:
            unseen = parse_unseen_subjects(override)
            split_detail = {
                'protocol': 'manual_NOT_FSPSP_TABLE1',
                'target_unseen_ids': sorted(unseen),
            }
        groups = partition_queries(train, test, unseen, args.scope)
        # The official FS-PSP sampler selects the same K image names for all
        # new observers, rather than a different set per observer. For ``both``
        # we build separate group-local support pools; never contaminate a
        # seen or unseen partition with another partition's samples.
        plans = {
            scope: make_support_plan(
                train, {q.subject for q in queries}, args.k_values, args.draws,
                args.seed, shared_stimuli=True)
            for scope, queries in groups.items()
        }
        prepared[name] = (groups, plans)
        protocol['datasets'][name] = {
            'root': str(root), 'annotation_frame': FRAMES[name],
            'split_detail': split_detail,
            'all_target_subject_ids': sorted({q.subject for q in test}),
            'target_seen_subject_ids': sorted({q.subject for q in test} - unseen),
            'target_unseen_subject_ids': sorted(unseen),
            'queries': {key: len(value) for key, value in groups.items()},
            'data_stats': stats,
            'support_ids_by_partition': {
                scope: {f'{s}|K={k}|draw={d}': [r.record_id for r in supports]
                        for (s, k, d), supports in plan.items()}
                for scope, plan in plans.items()
            },
        }
        print(f"[TRANSFER][PREFLIGHT] {LABELS[name]}: {protocol['datasets'][name]['queries']}; "
              f"subjects={protocol['datasets'][name]['all_target_subject_ids']}; "
              f"train={len(train)}; frame={FRAMES[name]}", flush=True)
    print(f'[TRANSFER][BATCH] physical sizes by K={physical_batch_sizes}; '
          f'bucket_by_fixation_count={args.bucket_by_fixation_count}', flush=True)
    if args.dry_run:
        print('[TRANSFER] DRY RUN PASSED (no model loaded, no output written)', flush=True)
        return

    out = args.output_dir.expanduser().resolve()
    if out == checkpoint or out in checkpoint.parents or checkpoint in out.parents:
        raise ValueError('output directory must not overwrite the checkpoint')
    if out.exists() and any(out.iterdir()):
        raise ValueError('choose an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'transfer_protocol.json').write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')

    bundle = load_checkpoint_bundle(checkpoint, split_manifest_identity=identity,
                                    output_dir=out, runtime={'device': args.device})
    cache = InferenceVisualCache(preprocessing=True, features=False, max_entries=256)
    results, n_written = {}, 0
    partial = out / 'predictions.jsonl.partial'
    try:
        with test_mode(bundle):
            cache.validate(bundle)
            with partial.open('w', encoding='utf-8') as stream:
                for name in names:
                    groups, plans = prepared[name]
                    results[name] = {}
                    all_draw_scores = {k: defaultdict(list) for k in args.k_values}
                    all_under = defaultdict(int)
                    all_bad_format = defaultdict(int)
                    for scope, queries in groups.items():
                        results[name][scope] = {}
                        for k in args.k_values:
                            draw_scores = {}
                            under_generated, bad_format = 0, 0
                            for draw in range(args.draws):
                                episodes = episodes_for(queries, plans[scope], k, draw)
                                scores = []
                                k_batch_size = physical_batch_sizes[k]
                                draw_processed = 0
                                for batch in scheduled_episode_batches(
                                        episodes, k_batch_size,
                                        bucket_by_length=args.bucket_by_fixation_count):
                                    generated = generate_where_batch(bundle, batch, cache=cache)
                                    if len(generated) != len(batch):
                                        raise RuntimeError('batch coverage mismatch')
                                    for ep, pred in zip(batch, generated):
                                        q = ep.query
                                        score = score_prediction(q, pred)
                                        scores.append(score)
                                        under_generated += int(pred['under_generated'])
                                        bad_format += int(not pred['canonical_format_valid'])
                                        row = {
                                            'dataset': LABELS[name], 'target_partition': scope,
                                            'query_id': q.record_id, 'subject': q.subject,
                                            'stimulus_id': q.stimulus_id, 'K': k, 'draw_id': draw,
                                            'support_ids': [r.record_id for r in ep.supports],
                                            'annotation_width': q.image_width,
                                            'annotation_height': q.image_height,
                                            'gt_fixations_raw': list(zip(q.x_px, q.y_px, q.duration_ms)),
                                            'gt_where': serialize_xyd_record(
                                                q, active_end_fix_token(bundle.config)),
                                            'where_pred': pred['text'],
                                            'pred_fixations_bins': pred['fixations'],
                                            'requested_fixation_count': pred['requested_fixation_count'],
                                            'accepted_fixation_count': pred['accepted_fixation_count'],
                                            'under_generated': pred['under_generated'],
                                            'canonical_format_valid': pred['canonical_format_valid'],
                                            'metrics': score,
                                        }
                                        stream.write(json.dumps(
                                            row, ensure_ascii=False, allow_nan=False) + '\n')
                                        n_written += 1
                                    draw_processed += len(batch)
                                    print(f'[TRANSFER] {LABELS[name]} {scope} '
                                          f'K={k} draw={draw+1}/{args.draws} '
                                          f'batch_size={k_batch_size}: '
                                          f'{draw_processed}/{len(episodes)}',
                                          flush=True)
                                if len(scores) != len(queries):
                                    raise RuntimeError('missing predictions within a draw')
                                draw_scores[draw] = scores
                                all_draw_scores[k][draw].extend(scores)
                            all_under[k] += under_generated
                            all_bad_format[k] += bad_format
                            summary = summarize_draws(draw_scores, under_generated, bad_format)
                            results[name][scope][str(k)] = summary
                            print(f"[TRANSFER] {LABELS[name]} {scope} K={k}: "
                                  f"SM={summary['sm']:.4f} MM={summary['mm']:.4f} "
                                  f"SED={summary['sed']:.4f}", flush=True)
                    if len(groups) > 1:
                        # Full target-dataset row for the paper's OSIE/COCO-FreeView columns.
                        # Aggregate QUERY predictions, not seen/unseen macro means.
                        results[name]['all'] = {
                            str(k): summarize_draws(all_draw_scores[k], all_under[k],
                                                    all_bad_format[k])
                            for k in args.k_values
                        }
        partial.replace(out / 'predictions.jsonl')
    finally:
        cache.close()
    protocol['completed_predictions'] = n_written
    (out / 'transfer_protocol.json').write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'metrics.json').write_text(
        json.dumps({'results': results}, indent=2, ensure_ascii=False, allow_nan=False),
        encoding='utf-8')
    write_table(out / 'metrics.md', results)
    if args.split_protocol == 'fspsp-main' and args.scope in ('unseen', 'both'):
        write_fspsp_table(out / 'metrics_fspsp_table.md', results)
    print(f'[TRANSFER] complete: {n_written} rows; results in {out}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--scandiff-data', type=Path, required=True,
                   help='ScanDiff/data directory, OR the parent of OSIE/COCO-FreeView roots')
    p.add_argument('--osie-root', type=Path)
    p.add_argument('--coco-freeview-root', type=Path)
    p.add_argument('--dataset', choices=('osie', 'coco_freeview', 'both'), default='both')
    p.add_argument('--scope', choices=('seen', 'unseen', 'both'), default='unseen',
                   help='unseen matches FS-PSP main Table 1; seen is a separate appendix evaluation')
    p.add_argument('--split-protocol', choices=('fspsp-main', 'manual'),
                   default='fspsp-main',
                   help='FS-PSP main-paper subject split (default), or explicit nonpaper IDs')
    p.add_argument('--fspsp-subject-id-base', choices=('auto', '0', '1'),
                   default='auto', help='Verify ScanDiff subjects are 0..N-1 or 1..N')
    p.add_argument('--unseen-subjects', type=str,
                   help='ONLY for --split-protocol manual; comma-separated raw target subject IDs')
    p.add_argument('--osie-unseen-subjects', type=str)
    p.add_argument('--coco-freeview-unseen-subjects', type=str)
    p.add_argument('--k-values', nargs='+', type=int, default=[1, 5, 10])
    p.add_argument('--draws', type=int, default=1)
    p.add_argument('--max-fixations', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=1,
                   help='physical batch size fallback for K values without an override')
    p.add_argument('--batch-size-by-k', nargs='+', default=None,
                   help='per-K physical batches, e.g. 1:4,5:2,10:1 or 1=4 5=2 10=1')
    p.add_argument('--bucket-by-fixation-count', action=argparse.BooleanOptionalAction,
                   default=True, help='group same-length query episodes to avoid mixed-budget '
                                      'microbatches inside generate_where_batch (default on)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
