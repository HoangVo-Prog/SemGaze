"""Semantic metric bookkeeping and frozen scorer settings.

No parser repair or GT text injection occurs here.  Missing expected units stay
in the denominator with a deterministic zero contribution.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import sys
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence


BRANCHES = ("what", "why", "how")


def normalize_text(text):
    return " ".join(str(text).strip().split()) if text is not None else ""


def _units_for(record: Mapping, branch: str):
    value = record['semantic'][branch]
    if not isinstance(value, Mapping) or 'gt' not in value or 'pred' not in value:
        raise ValueError('semantic units require explicit gt/pred fields')
    gt, pred = value['gt'], value['pred']
    if branch == 'how':
        gt, pred = [gt], [pred]
    if not isinstance(gt, (list, tuple)) or not isinstance(pred, (list, tuple)):
        raise ValueError('WHAT/WHY must be aligned arrays')
    if not gt:
        raise ValueError('dataset-invalid empty semantic references')
    return [{'gt': g, 'pred': pred[i] if i < len(pred) else None} for i, g in enumerate(gt)]


def collect_semantic_units(records: Iterable[Mapping], branch: str):
    branch = branch.lower()
    if branch not in BRANCHES:
        raise ValueError(f'unknown semantic branch: {branch}')
    result = []
    for row in records:
        query_id = row.get('query_id', row.get('record_id'))
        if not isinstance(query_id, str) or not query_id:
            raise ValueError('missing query identity')
        for index, value in enumerate(_units_for(row, branch)):
            reference = value['gt']
            if not isinstance(reference, str) or not normalize_text(reference):
                raise ValueError('dataset-invalid semantic reference')
            candidate = value['pred']
            candidate = normalize_text(candidate) if isinstance(candidate, str) else None
            candidate = candidate or None
            result.append({'unit_id': f'{query_id}::{branch.upper()}::{index}',
                           'reference': normalize_text(reference), 'candidate': candidate,
                           'missing': candidate is None})
    if len({u['unit_id'] for u in result}) != len(result):
        raise ValueError('semantic unit IDs are not unique; score only one K/draw/branch')
    return result


def semantic_units_from_prediction(row: Mapping, query_semantic):
    # Existing parser dictionaries use integer keys in memory, strings in JSON.
    from dataclasses import asdict, is_dataclass
    gt = asdict(query_semantic) if is_dataclass(query_semantic) else query_semantic
    parsed = row.get('semantic_generation', row.get('semantic', {}))
    def candidates(branch, n):
        values = parsed.get(branch, {})
        return [values.get(i, values.get(str(i))) for i in range(1, n+1)]
    canonical = {'query_id': row.get('query_id', row.get('record_id')), 'semantic': {
        'what': {'gt': list(gt['what']), 'pred': candidates('what', len(gt['what']))},
        'why': {'gt': [g['why'] for g in gt['why_groups']], 'pred': candidates('why', len(gt['why_groups']))},
        'how': {'gt': gt['how'], 'pred': parsed.get('how')}}}
    return {branch: collect_semantic_units([canonical], branch) for branch in BRANCHES}


def build_bertscorer(*, package_version="0.3.13", lang="en", model_type="roberta-large",
                     num_layers=17, idf=False, rescale_with_baseline=False,
                     use_fast_tokenizer=False, device=None):
    try:
        import bert_score
    except ImportError as exc:
        raise RuntimeError("bert-score==0.3.13 is required for BERTScore metrics") from exc
    from importlib.metadata import version
    frozen = (package_version, lang, model_type, num_layers, idf, rescale_with_baseline, use_fast_tokenizer)
    if frozen != ('0.3.13', 'en', 'roberta-large', 17, False, False, False):
        raise ValueError('canonical BERTScore settings are frozen')
    if version('bert-score') != package_version:
        raise RuntimeError('BERTScore distribution must be 0.3.13')
    scorer = bert_score.BERTScorer(lang=lang, model_type=model_type, num_layers=num_layers, idf=idf,
                                 rescale_with_baseline=rescale_with_baseline,
                                 use_fast_tokenizer=use_fast_tokenizer, device=device)
    import inspect
    scorer_source = Path(inspect.getsourcefile(bert_score.BERTScorer) or '')
    scorer_hash = hashlib.sha256(scorer_source.read_bytes()).hexdigest() if scorer_source.is_file() else None
    return scorer, {'package': 'bert-score', 'version': version('bert-score'),
        'transformers_version': version('transformers'), 'model': model_type, 'layer': num_layers,
        'lang': lang, 'idf': idf, 'rescale_with_baseline': rescale_with_baseline,
        'use_fast_tokenizer': use_fast_tokenizer, 'official_scorer_hash': scorer_hash}


def score_bertscore_branch(units: Sequence[Mapping], scorer, *, batch_size=64):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive BERTScore batch_size required")
    if not units:
        return {"f1": None, "unit_count": 0, "missing_count": 0}
    values = [0.0] * len(units)
    active = [(i, u) for i, u in enumerate(units) if u.get("candidate") is not None]
    for start in range(0, len(active), batch_size):
        chunk = active[start:start + batch_size]
        candidates = [u["candidate"] for _, u in chunk]
        references = [u["reference"] for _, u in chunk]
        _, _, f1 = scorer.score(candidates, references)
        values_for_chunk = f1.detach().cpu().tolist() if hasattr(f1, "detach") else list(f1)
        if len(values_for_chunk) != len(chunk) or not all(math.isfinite(float(v)) for v in values_for_chunk):
            raise RuntimeError("BERTScore returned invalid sample scores")
        for (index, _), value in zip(chunk, values_for_chunk):
            values[index] = float(value)
    return {"f1": sum(values) / len(values), "unit_count": len(units),
            "missing_count": sum(bool(u.get("missing")) for u in units)}


def score_cider_r_branch(units: Sequence[Mapping], scorer, tokenizer=None):
    if tokenizer is None:
        raise ValueError('canonical CIDEr-R requires authors PTBTokenizer')
    if not units:
        raise ValueError('empty reference corpus')
    if len({u['unit_id'] for u in units}) != len(units):
        raise ValueError('duplicate semantic unit')
    candidates = {u['unit_id']: [{'caption': u['candidate'] or ''}] for u in units}
    references = {u['unit_id']: [{'caption': u['reference']}] for u in units}
    candidates = tokenizer.tokenize(candidates)
    references = tokenizer.tokenize(references)
    if set(candidates) != set(references) or set(candidates) != {u['unit_id'] for u in units}:
        raise RuntimeError('PTB tokenizer lost semantic units')
    # All reference documents enter a single scorer call, including missing candidates.
    # Authors' empty-candidate repetition penalty is NaN; only those invalid
    # candidates receive the contract's deterministic zero extension below.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        _, raw = scorer.compute_score(references, candidates)
    if len(raw) != len(units):
        raise RuntimeError('CIDEr-R lost sample scores')
    values = []
    for unit, value in zip(units, raw):
        missing = unit['candidate'] is None or not candidates[unit['unit_id']][0].strip()
        value = 0.0 if missing else float(value)
        if not math.isfinite(value):
            raise RuntimeError('nonfinite CIDEr-R for a valid candidate')
        values.append(value)
    return {'score': sum(values)/len(values), 'sample_scores': values, 'unit_count': len(units),
            'missing_count': sum(u['candidate'] is None for u in units)}


def build_cider_r_scorer(*, reference_root=None, n=4, k_r=0.8):
    if (n, k_r) != (4, 0.8):
        raise ValueError('CIDEr-R parameters are frozen')
    import shutil
    if not shutil.which('java'):
        raise RuntimeError('authors PTBTokenizer requires Java on PATH')
    # Namespace import prevents an installed ordinary pycocoevalcap shadowing the vendor.
    from third_party.cider_r.pycocoevalcap.ciderR.ciderR import CiderR
    from third_party.cider_r.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from semgaze.evaluation.records import file_hash
    root = Path(reference_root) if reference_root is not None else \
        Path(__file__).resolve().parents[2] / 'third_party/cider_r'
    if not root.is_dir():
        raise FileNotFoundError(f'vendored CIDEr-R root missing: {root}')
    files = {p.relative_to(root).as_posix(): file_hash(p) for p in root.rglob('*')
             if p.is_file() and p.suffix in ('.py', '.jar', '.txt')}
    return (CiderR(n=4), PTBTokenizer()), {
        'implementation': 'vendored_authors', 'n': 4, 'k_r': 0.8, 'length_coefficient': 0.2,
        'final_scale': 10, 'tokenizer': 'stanford_ptb_reference',
        'upstream': 'gabrielsantosrv/coco-caption---My-changes',
        'commit': 'b5f27535299eacb2bb4b599ac239841625030c8d', 'file_sha256': files}


def score_prediction_semantics(rows, references_by_key, *, bertscorer=None, cider_r=None,
                               batch_size=64):
    """Score complete semantic corpora from frozen prediction rows.

    ``references_by_key`` maps ``(query_id, K, draw_id)`` to normalized query
    semantic annotations.  No generation or model object is accessed.
    """
    branches = {branch: [] for branch in BRANCHES}
    parse_failures = 0
    for row in rows:
        key = (str(row.get("query_id", row.get("record_id"))), int(row.get("K", row.get("k"))), int(row["draw_id"]))
        if key not in references_by_key:
            raise ValueError(f"missing semantic reference for prediction key {key}")
        parsed = row.get("semantic_generation", row.get("semantic", {}))
        parse_failures += int(not parsed.get("flat_format_valid", True))
        units = semantic_units_from_prediction(row, references_by_key[key])
        for branch in BRANCHES:
            branches[branch].extend(units[branch])
    result = {"semantic_parse_failure_count": parse_failures}
    if cider_r is None:
        cider_scorer, cider_tokenizer = None, None
    elif isinstance(cider_r, tuple):
        cider_scorer, cider_tokenizer = cider_r
    else:
        cider_scorer, cider_tokenizer = cider_r, None
    for branch, units in branches.items():
        if bertscorer is not None:
            result[f"eval_sem_{branch}_bertscore_f1"] = score_bertscore_branch(units, bertscorer, batch_size=batch_size)["f1"]
        if cider_scorer is not None:
            result[f"eval_sem_{branch}_cider_r"] = score_cider_r_branch(units, cider_scorer, cider_tokenizer)["score"]
        result[f"semantic_{branch}_unit_count"] = len(units)
        result[f"semantic_{branch}_missing_count"] = sum(u.get("candidate") is None for u in units)
    return result


def aggregate_semantic_draw(records: Sequence[Mapping], *, bertscorer=None,
                            cider_scorer=None, cider_tokenizer=None,
                            batch_size=64):
    """Score one complete ``(K, draw_id)`` semantic corpus.

    ``cider_scorer`` may also be the ``(scorer, tokenizer)`` tuple returned by
    :func:`build_cider_r_scorer`.  Keeping the tokenizer alongside the scorer
    is required because CIDEr-R document frequency is built over this whole
    draw, after PTB tokenization.
    """
    if isinstance(cider_scorer, tuple):
        if cider_tokenizer is not None:
            raise ValueError('CIDEr-R tokenizer supplied twice')
        cider_scorer, cider_tokenizer = cider_scorer
    result = {}
    for branch in BRANCHES:
        units = collect_semantic_units(records, branch)
        if bertscorer is not None:
            result[f"eval_sem_{branch}_bertscore_f1"] = score_bertscore_branch(
                units, bertscorer, batch_size=batch_size)["f1"]
        if cider_scorer is not None:
            result[f"eval_sem_{branch}_cider_r"] = score_cider_r_branch(
                units, cider_scorer, cider_tokenizer)["score"]
        result[f"semantic_{branch}_unit_count"] = len(units)
        result[f"semantic_{branch}_missing_count"] = sum(u["missing"] for u in units)
    return result
