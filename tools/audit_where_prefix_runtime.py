#!/usr/bin/env python3
"""Read-only raw-token audit of SemGaze WHERE generation (no-attn @ 5adafa2e).

Run at repository root, using the SAME original checkpoint/config/split as evaluation.
This observer temporarily wraps model.generate, returns its exact original object,
and never edits production predictions, parsers, ground truth, or model weights.

Example:
  python tools/audit_where_prefix_runtime.py \\
    --checkpoint runs/RUN/checkpoints/epoch_2 \\
    --predictions-jsonl runs/standalone/predictions.jsonl \\
    --output-dir runs/prefix_audit --tests A B C D E --batch-size 4

Use --query-id (repeatable) and --draw-id for manifest-only selection when no
existing JSONL is available. The original JSONL is strongly recommended: it
allows verification of the historical frozen support set and predictions.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback

import torch

BASE_COMMIT = '5adafa2ec9c7831e19eb771b11fad83fa203a9e9'
TESTS = {
    'A': ('serial', 1, False),
    'B': ('batched', 1, False),
    'C': ('batched', 1, True),
    'D': ('batched', 4, True),
    'E': ('batched', 4, False),
}


def git_sha() -> str | None:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def ids_list(ids):
    return ids.detach().cpu().tolist() if hasattr(ids, 'detach') else list(ids)


def returned_sequences(output):
    return output.sequences if hasattr(output, 'sequences') else output


def production_decode(tokenizer, ids):
    ids = ids_list(ids)
    eos = tokenizer.eos_token_id
    if eos is not None and eos in ids:
        ids = ids[:ids.index(eos)]
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def validate_left_padding(ids, mask, pad_id):
    """Assert 0*1* mask, true-token positioning, and masked ID==pad."""
    ids, mask = ids_list(ids), ids_list(mask)
    if len(ids) != len(mask) or not ids or any(m not in (0, 1) for m in mask):
        raise AssertionError('invalid attention_mask shape/value')
    first = next((i for i, m in enumerate(mask) if m), len(mask))
    if first == len(mask) or any(mask[:first]) or any(m == 0 for m in mask[first:]):
        raise AssertionError('not contiguous left-padding')
    if any(x != pad_id for x in ids[:first]):
        raise AssertionError('left-padding token ID mismatch')
    return first, len(ids)-first


def token_window(tokenizer, ids, start, stop):
    ids = ids_list(ids)
    start, stop = max(start, 0), min(stop, len(ids))
    chunk = ids[start:stop]
    return {'span': [start, stop], 'ids': chunk,
            'tokens': tokenizer.convert_ids_to_tokens(chunk),
            'decoded': tokenizer.decode(chunk, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)}


def inspect_one(tokenizer, input_ids, attention_mask, returned_ids, *, budget,
                expected_input_ids=None, expected_mask=None, pad_id=None):
    """Pure token-level checks; no inference/model dependency."""
    prompt = ids_list(input_ids)
    mask = ids_list(attention_mask)
    seq = ids_list(returned_ids)
    if pad_id is None:
        pad_id = tokenizer.pad_token_id
    n_padding, n_real = validate_left_padding(prompt, mask, pad_id)
    if expected_input_ids is not None and prompt != ids_list(expected_input_ids):
        raise AssertionError('generation query-row does not match prepared episode input_ids')
    if expected_mask is not None and mask != ids_list(expected_mask):
        raise AssertionError('generation mask does not match prepared episode')
    L = len(prompt)
    mismatch = next((i for i, (a, b) in enumerate(zip(prompt, seq)) if a != b), None)
    if mismatch is None and len(seq) < L:
        mismatch = len(seq)
    echo = len(seq) >= L and seq[:L] == prompt
    unpadded_echo = len(seq) >= n_real and seq[:n_real] == prompt[n_padding:]
    structure = ('FULL_PADDED_PROMPT_ECHO' if echo else
                 'UNPADDED_PROMPT_ECHO' if unpadded_echo else 'NONMATCHING_OR_NEW_TOKENS_ONLY')
    # Exactly the production slice, even when alignment is not established.
    sliced = seq[L:]
    prod_text = production_decode(tokenizer, sliced)
    possible_start = L if echo else n_real if unpadded_echo else None
    possible_ids = seq[possible_start:] if possible_start is not None else None
    possible_text = production_decode(tokenizer, possible_ids) if possible_ids is not None else None
    eos = tokenizer.eos_token_id
    eos_at = [i for i, v in enumerate(seq) if v == eos and i >= (possible_start or 0)] if eos is not None else []
    result = {
        'input_ids': prompt, 'attention_mask': mask, 'returned_sequence_ids': seq,
        'prefix_width_L': L, 'true_prompt_length': n_real, 'left_pad_length': n_padding,
        'returned_length': len(seq), 'requested_generation_budget': budget,
        'full_prompt_prefix_matches': echo, 'first_prompt_mismatch': mismatch,
        'sequence_structure': structure, 'candidate_continuation_offset': possible_start,
        'production_continuation_ids': sliced,
        'production_continuation_tokens': tokenizer.convert_ids_to_tokens(sliced),
        'production_decoded_text': prod_text, 'production_starts_with_open': prod_text.startswith('[('),
        'candidate_continuation_ids': possible_ids,
        'candidate_decoded_text': possible_text,
        'candidate_starts_with_open': possible_text.startswith('[(') if possible_text is not None else None,
        'token_boundary_window': token_window(tokenizer, seq, L-12, L+20),
        'decoded_boundary_window': token_window(tokenizer, seq, L-5, L+10)['decoded'],
        'first_100_decoded_characters': prod_text[:100],
        'raw_token_first_window': token_window(tokenizer, seq, possible_start or 0,
                                               (possible_start or 0)+20),
        'eos_positions_after_candidate_start': eos_at,
        'inputs_embeds_supplied': None,
    }
    return result


@contextmanager
def capture_generate(model, tokenizer, expected_calls, sink):
    """Pass-through observer. Crucially never changes kwargs or return value."""
    original = model.generate
    counter = 0

    def spy(*args, **kwargs):
        nonlocal counter
        if args:
            raise AssertionError('unexpected positional generate arguments in SemGaze')
        if counter >= len(expected_calls):
            raise AssertionError('more physical generate calls than anticipated')
        planned = expected_calls[counter]
        counter += 1
        ids = kwargs['input_ids']
        mask = kwargs['attention_mask']
        if ids.shape[0] != len(planned['query_ids']):
            raise AssertionError('physical batch query count mismatch')
        if int(kwargs['max_new_tokens']) != planned['budget']:
            raise AssertionError('budget-based group order mismatch')
        # Check caller-provided inputs BEFORE executing the model.
        for row in range(ids.shape[0]):
            validate_left_padding(ids[row], mask[row], tokenizer.pad_token_id)
            if ids_list(ids[row]) != planned['reference_input_ids'][row]:
                raise AssertionError(f'wrong query-to-row mapping: {planned["query_ids"][row]}')
            if ids_list(mask[row]) != planned['reference_masks'][row]:
                raise AssertionError('attention mask differs from independently packed episode')
        result = original(*args, **kwargs)
        sequences = returned_sequences(result)
        if sequences.ndim != 2 or sequences.shape[0] != ids.shape[0]:
            raise AssertionError('unexpected HF generation result rank/batch rows')
        for row, query_id in enumerate(planned['query_ids']):
            evidence = inspect_one(tokenizer, ids[row], mask[row], sequences[row],
                                   budget=planned['budget'],
                                   expected_input_ids=planned['reference_input_ids'][row],
                                   expected_mask=planned['reference_masks'][row])
            evidence.update(query_id=query_id, batch_row=row,
                actual_physical_batch_size=int(ids.shape[0]),
                physical_batch_query_ids=list(planned['query_ids']),
                inputs_embeds_supplied=('inputs_embeds' in kwargs),
                pixel_values_supplied=('pixel_values' in kwargs))
            sink.append(evidence)
        return result

    model.generate = spy
    try:
        yield
        if counter != len(expected_calls):
            raise AssertionError(f'missing physical generation calls: {counter}/{len(expected_calls)}')
    finally:
        model.generate = original


def _planned_group(bundle, episodes, *, cache, serial):
    from semgaze.where.collator import collate_where, pack_where_batch
    from semgaze.where.generation import where_generation_budget, left_pad_generation
    tokenizer = bundle.processor.tokenizer
    samples = [collate_where(bundle.processor, e, bundle.end_fix_id, bundle.context_limit,
                             teacher_forcing=False, config=bundle.config, image_cache=cache)
               for e in episodes]
    if serial:
        assert len(episodes) == 1
        return samples, [{'budget': where_generation_budget(bundle, episodes[0]),
            'query_ids': [episodes[0].query.record_id],
            'reference_input_ids': [ids_list(samples[0].inputs['input_ids'][0])],
            'reference_masks': [ids_list(samples[0].inputs['attention_mask'][0])]}]
    budgets = [where_generation_budget(bundle, e) for e in episodes]
    planned = []
    for budget in dict.fromkeys(budgets):
        positions = [i for i, b in enumerate(budgets) if b == budget]
        packed = pack_where_batch(bundle.processor, [episodes[i] for i in positions],
                                  [samples[i] for i in positions], cache)
        padded = left_pad_generation(packed.inputs, tokenizer.pad_token_id)
        planned.append({'budget': budget,
            'query_ids': [episodes[i].query.record_id for i in positions],
            'reference_input_ids': [ids_list(row) for row in padded['input_ids']],
            'reference_masks': [ids_list(row) for row in padded['attention_mask']]})
    return samples, planned


def run_condition(bundle, episodes, label, *, batch_size, cache_features, checkpoint, commit):
    from semgaze.where.generation import generate_where, generate_where_batch
    from semgaze.evaluation.test import test_mode
    from semgaze.model.visual_cache import InferenceVisualCache
    from semgaze.where.serialization import parse_xyd_output
    tokenizer = bundle.processor.tokenizer
    condition_rows = []
    records = []
    cache = (InferenceVisualCache(preprocessing=True, features=True, max_entries=256)
             if cache_features else None)
    try:
        with test_mode(bundle):
            if cache is not None:
                with torch.inference_mode():
                    cache.validate(bundle)
            for start in range(0, len(episodes), batch_size):
                group = episodes[start:start+batch_size]
                samples, planned = _planned_group(bundle, group, cache=cache,
                                                   serial=(label == 'A'))
                observed = []
                with capture_generate(bundle.model, tokenizer, planned, observed):
                    if label == 'A':
                        result = [generate_where(bundle, group[0])]
                    else:
                        result = generate_where_batch(bundle, group, cache=cache,
                                                      precomputed_samples=samples)
                by_id = {r['query_id']: r for r in observed}
                if set(by_id) != {ep.query.record_id for ep in group}:
                    raise AssertionError('physical batch restoration lost/duplicated a query')
                for episode, production in zip(group, result):
                    query_id = episode.query.record_id
                    item = by_id[query_id]
                    if production['text'] != item['production_decoded_text']:
                        raise AssertionError('observer text differs from production decode/extraction')
                    requested = len(episode.query.x_px)
                    parsed = parse_xyd_output(production['text'], requested,
                                        bundle.config['where']['end_fix_token'])
                    for key in ('requested_fixation_count', 'accepted_fixation_count',
                                'under_generated', 'canonical_format_valid'):
                        if parsed[key] != production[key]:
                            raise AssertionError(f'parser differs from production: {key}')
                    item.update(condition=label, checkpoint=str(checkpoint), git_commit=commit,
                        K=len(episode.supports), draw_id=episode.draw_id,
                        subject=episode.query.subject,
                        ordered_support_ids=[s.record_id for s in episode.supports],
                        requested_fixation_count=requested,
                        accepted_fixation_count=parsed['accepted_fixation_count'],
                        under_generated=parsed['under_generated'],
                        canonical_format_valid=parsed['canonical_format_valid'],
                        production_extraction_identical=True, production_result=production)
                    records.append(item)
                condition_rows.append({'condition': label, 'physical_size': len(group),
                    'generate_calls': len(planned),
                    'actual_generation_batch_sizes': [len(p['query_ids']) for p in planned],
                    'ids': [e.query.record_id for e in group]})
    finally:
        if cache is not None:
            cache.close()
    return records, condition_rows


def _prefix_only(row, *, end_fix_token='<END_FIX>'):
    from semgaze.where.serialization import parse_xyd_output
    pred = row.get('WHERE', {}).get('PRED', '')
    n = row.get('where_generation', {}).get('requested_fixation_count')
    return (isinstance(pred, str) and pred and not pred.startswith('[(') and
            isinstance(n, int) and n > 0 and
            parse_xyd_output('[(' + pred, n, end_fix_token)['canonical_format_valid'])


def load_original_records(path, k, draw, query_ids, *, prefix_count, controls,
                          end_fix_token='<END_FIX>'):
    rows = []
    if path is None:
        return rows
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get('K', row.get('k', -1))) != k:
                continue
            if row.get('dataset', 'COCO-Search18') != 'COCO-Search18':
                continue
            if int(row.get('draw_id', 0)) != draw:
                continue
            if query_ids and row['query_id'] not in query_ids:
                continue
            if 'WHERE' not in row or 'support_ids' not in row:
                continue
            rows.append(row)
    if query_ids:
        candidates = rows
    else:
        bad = [r for r in rows if _prefix_only(r, end_fix_token=end_fix_token)]
        good = [r for r in rows if r.get('where_generation', {}).get('canonical_format_valid')]
        def rank(row):
            n = row.get('where_generation', {}).get('requested_fixation_count')
            return (str(row.get('subject')) != '8', n not in (3, 4), str(row['query_id']))
        candidates = sorted(bad, key=rank)[:prefix_count] + sorted(good, key=rank)[:controls]
    indexed = {r['query_id']: r for r in candidates}
    if len(indexed) != len(candidates):
        raise ValueError('duplicate original prediction rows for selected K/draw')
    if query_ids and set(indexed) != set(query_ids):
        raise ValueError('some requested query IDs absent from historical prediction JSONL')
    return candidates


def summarize(records, original_by_query, conditions):
    summary = []
    for name in conditions:
        observed = [r for r in records if r['condition'] == name]
        reproduced = [r for r in observed if
            _original_bad(original_by_query.get(r['query_id'])) and
            not r['production_starts_with_open']]
        alignment = {s: sum(r['sequence_structure'] == s for r in observed)
                     for s in ('FULL_PADDED_PROMPT_ECHO','UNPADDED_PROMPT_ECHO',
                               'NONMATCHING_OR_NEW_TOKENS_ONLY')}
        summary.append(dict(condition=name, executed=bool(observed), queries=len(observed),
                            reproduced_missing_prefix=len(reproduced),
                            alignment=alignment))
    return summary


def _original_bad(row):
    return bool(row) and not row.get('WHERE', {}).get('PRED', '').startswith('[(')


def verdicts(records, summary):
    """Conservative observation-level verdicts, never inferred from aggregate rates."""
    names = ('H1 Incorrect output slicing', 'H2 Tokenizer decoding corruption',
             'H3 Left-padding/batched generation', 'H4 Visual cache/inputs_embeds',
             'H5 Model-generated malformed prefix')
    outcome = {key: ('NOT EXECUTED', 'No model-generated tokens available.') for key in names}
    if not records:
        return outcome
    bad = [r for r in records if not r['production_starts_with_open']]
    misaligned_with_open = [r for r in records if r['sequence_structure'] ==
                            'UNPADDED_PROMPT_ECHO' and r['candidate_starts_with_open'] and
                            not r['production_starts_with_open']]
    if misaligned_with_open:
        outcome[names[0]] = ('INCONCLUSIVE',
            f'{len(misaligned_with_open)} cases appear to echo the unpadded prompt and '
            'candidate extraction recovers the opening. Confirm the exact HF/InternVL '
            'output structure for this version before attributing the difference to slicing.')
    elif all(r['full_prompt_prefix_matches'] for r in records):
        outcome[names[0]] = ('RULED OUT FOR TESTED CASES',
            'Each returned sequence exactly echoes the full provided input_ids[0:L].')
    else:
        outcome[names[0]] = ('INCONCLUSIVE',
            'Some returned sequences do not echo the full prompt. Inspect alternate '
            'prefix structure, model/version contract, and boundary token IDs.')
    # Token piece concatenation is not guaranteed for byte-level BPE; do NOT
    # claim a decoding bug (or rule one out) solely from convert_ids_to_tokens.
    outcome[names[1]] = ('INCONCLUSIVE',
        'Tokenizer decode is reproduced on the raw token IDs. Compare exact '
        'first token IDs and their native decoded spans; token strings alone '
        'cannot establish decoder corruption.')
    indexed = {(r['condition'], r['query_id']): r for r in records}
    def compare(a, b):
        common = [qid for cond, qid in indexed if cond == a and (b, qid) in indexed]
        if not common:
            return None
        diff = [qid for qid in common if
                indexed[a, qid]['production_continuation_ids'] !=
                indexed[b, qid]['production_continuation_ids']]
        return len(common), diff
    comparisons = [('A', 'B'), ('B', 'E')]
    present = [(a,b,compare(a,b)) for a,b in comparisons if compare(a,b) is not None]
    diffs = [(a,b,x) for a,b,(count,qs) in present for x in qs]
    if diffs:
        outcome[names[2]] = ('CONFIRMED',
            f'Uncached serial/batch execution changed the raw continuation for '
            f'{len(diffs)} query/pair comparisons, e.g. {diffs[:3]}. This is '
            'execution-path sensitivity, not by itself a padding defect.')
    elif present:
        outcome[names[2]] = ('RULED OUT FOR TESTED CASES',
            'Uncached serial/batched raw continuation IDs agree in available '
            f'matched comparisons: {[(a,b,z[0]) for a,b,z in present]}.')
    else:
        outcome[names[2]] = ('INCONCLUSIVE', 'No matched uncached serial/batch comparisons.')
    cache_comparisons = [('B', 'C'), ('E', 'D')]
    cp = [(a,b,compare(a,b)) for a,b in cache_comparisons if compare(a,b) is not None]
    cache_diffs = [(a,b,x) for a,b,(_,qs) in cp for x in qs]
    if cache_diffs:
        outcome[names[3]] = ('CONFIRMED',
            f'Cache status changed raw continuation tokens on matched queries '
            f'{cache_diffs[:4]}; inspect fusion/positions and cache initialization.')
    elif cp:
        outcome[names[3]] = ('RULED OUT FOR TESTED CASES',
            f'Raw tokens agree with cache off/on for matched comparisons '
            f'{[(a,b,z[0]) for a,b,z in cp]}.')
    else:
        outcome[names[3]] = ('INCONCLUSIVE', 'No matched cache-off/on observations.')
    generated_bad = [r for r in bad if r['full_prompt_prefix_matches']]
    if generated_bad:
        outcome[names[4]] = ('CONFIRMED',
            f'{len(generated_bad)} observations have exact prompt echo, and '
            'their generated continuation itself lacks the opening. This '
            'does NOT identify the prompt/training factor responsible.')
    elif bad:
        outcome[names[4]] = ('INCONCLUSIVE',
            'Malformed production text exists, but returned-prompt alignment '
            'is not confirmed for those observations.')
    else:
        outcome[names[4]] = ('INCONCLUSIVE',
            'No missing-prefix predictions reproduced; the historical '
            'execution context may differ.')
    return outcome


def write_report(directory, metadata, summary, records, original_by_query, skipped):
    table = ['| Test | Queries | Historical missing-prefix reproduced | Prefix alignment |',
             '|---|---:|---:|---|']
    for item in summary:
        table.append(f'| {item["condition"]} | {item["queries"]} | '
                     f'{item["reproduced_missing_prefix"]} | {item["alignment"]} |')
    for item in skipped:
        table.append(f'| {item["condition"]} | NOT EXECUTED | — | {item["reason"]} |')
    examples = [r for r in records if _original_bad(original_by_query.get(r['query_id']))][:3]
    examples += [r for r in records if r not in examples][:max(0, 3-len(examples))]
    evidence = ['### C. Raw token evidence']
    if not examples:
        evidence.append('NOT EXECUTED — no model token observations.')
    for r in examples:
        evidence.extend([f'- `{r["condition"]}` `{r["query_id"]}`: '
            f'prefix={r["sequence_structure"]}; L={r["prefix_width_L"]}; '
            f'raw first tokens={r["raw_token_first_window"]["ids"]}; '
            f'production first tokens={r["production_continuation_ids"][:12]}; '
            f'production first chars={r["first_100_decoded_characters"]!r}'])
    outcomes = verdicts(records, summary)
    reason = outcomes['H1 Incorrect output slicing'][1]
    if any(outcomes[k][0] == 'CONFIRMED' for k in ('H1 Incorrect output slicing',
                                                  'H5 Model-generated malformed prefix')):
        root = 'Token-level evidence is decisive for at least some tested rows: ' + reason + ' ' + outcomes['H5 Model-generated malformed prefix'][1]
    elif records:
        root = ('Original missing-prefix outcome remains INCONCLUSIVE; '
                'inspect per-row raw token windows and reproduce original batch composition.')
    else:
        root = 'NOT EXECUTED. Whether `[( ` was generated or removed remains unknown.'
    text = '\n'.join([
        '# SemGaze WHERE prefix runtime audit',
        '### A. Execution environment',
        '```json', json.dumps(metadata, ensure_ascii=False, indent=2), '```',
        '### B. Reproduction results', *table, '', *evidence, '',
        '### D. Hypothesis verdicts',
        *(f'- {name}: **{value[0]}** — {value[1]}' for name, value in outcomes.items()),
        '', '### E. Root-cause conclusion', root,
        '', '### F. Recommended fix',
        ('Inspect the confirmed extraction defect and adjust the offset only '
         'for the documented output contract.' if outcomes['H1 Incorrect output slicing'][0] == 'CONFIRMED' else
         'No production patch without a specific demonstrated implementation defect.'),
        '', 'Caveat: historical physical batch grouping is not reconstructed by this selected-query run. '
        'If the anomaly does not reproduce, compare original bucket/window neighbors, K, draw, '
        'versions, and cache settings before drawing conclusions.'
    ])
    (directory / 'REPORT.md').write_text(text+'\n', encoding='utf-8')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, help='evaluation override YAML/JSON (checkpoint authoritative)')
    p.add_argument('--predictions-jsonl', type=Path, help='original frozen evaluation predictions')
    p.add_argument('--query-id', action='append', default=[])
    p.add_argument('--draw-id', type=int, default=0)
    p.add_argument('--k', type=int, choices=[1, 5, 10], default=10)
    p.add_argument('--prefix-count', type=int, default=12)
    p.add_argument('--valid-controls', type=int, default=4)
    p.add_argument('--tests', nargs='+', choices=list(TESTS), default=list(TESTS))
    p.add_argument('--batch-size', type=int, default=4,
                   help='physical chunk size for D/E (default 4)')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--continue-on-oom', action='store_true')
    a = p.parse_args(argv)
    if a.batch_size < 1 or a.draw_id < 0 or a.prefix_count < 0 or a.valid_controls < 0:
        p.error('nonnegative draw/selection counts and positive batch size required')
    if not a.predictions_jsonl and not a.query_id:
        p.error('supply original --predictions-jsonl or one or more --query-id')
    out = a.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        p.error('choose an empty output directory; will not overwrite prior audit')
    out.mkdir(parents=True, exist_ok=True)
    metadata = {'checkpoint': str(a.checkpoint.resolve()), 'requested_ref': 'no-attn',
        'reference_head_at_implementation': BASE_COMMIT, 'git_commit': git_sha(),
        'python': platform.python_version(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'requested_k': a.k, 'draw_id': a.draw_id, 'tests_requested': a.tests,
        'original_jsonl': str(a.predictions_jsonl) if a.predictions_jsonl else None,
        'original_batch_grouping_reconstructed': False}
    try:
        import transformers, peft
        metadata.update(transformers=transformers.__version__, peft=peft.__version__)
        from evaluate_flat import _load_evaluation_config
        from semgaze.model.config import ROOT, resolve_config
        from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter
        from semgaze.data.joint import JointAdapter
        from semgaze.evaluation.test import test_queries
        from semgaze.data.fewshot import frozen_episode
        from semgaze.model.checkpoint import load_checkpoint_bundle
        config = resolve_config(_load_evaluation_config(a.checkpoint, a.config))
        data = config['data']
        raw, manifest, identity = read_persisted_splits(ROOT / data['split_root'], data_config=data)
        adapter_cls = JointAdapter if data['dataset'] == 'all' else CocoSearch18Adapter
        adapter = adapter_cls(ROOT / data['images_root'], annotation_frame=data['annotation_frame'],
                              duration_field=data['duration']['source_field'])
        train = {r['record_id']: adapter(r) for r in raw['train']}
        tests = [adapter(r) for r in raw['test']]
        queries = {r.record_id: r for r in test_queries(train, tests, manifest)}
        rows = load_original_records(a.predictions_jsonl, a.k, a.draw_id, set(a.query_id),
                      prefix_count=a.prefix_count, controls=a.valid_controls,
                      end_fix_token=config['where']['end_fix_token'])
        if rows:
            selected_ids = [r['query_id'] for r in rows]
        else:
            selected_ids = list(dict.fromkeys(a.query_id))
        if not selected_ids:
            raise ValueError('no matching prefix failures/valid controls in original file')
        if any(q not in queries for q in selected_ids):
            raise ValueError('requested query not in current frozen unseen test split')
        original_by_query = {r['query_id']: r for r in rows}
        episodes = [frozen_episode(queries[q], train, manifest, a.k, draw_id=a.draw_id,
                            unseen_subjects=manifest['unseen_subject_ids']) for q in selected_ids]
        for e in episodes:
            historical = original_by_query.get(e.query.record_id)
            if historical is not None and [s.record_id for s in e.supports] != historical['support_ids']:
                raise ValueError(f'FROZEN SUPPORT MISMATCH for {e.query.record_id}: '
                                 f'checkpoint/split/draw/preprocessing may differ')
        metadata['selected_query_ids'] = selected_ids
        metadata['model_name'] = config['model']['base_model']
        metadata['generation_decoding'] = {'do_sample': False, 'use_cache': True, 'budget_policy': 'oracle_N_tokenized_serialization_plus_16_min_64'}
        metadata['evaluation_prediction_settings'] = config['test']['prediction']
        metadata['cache_settings_original'] = config['test']['cache']
        metadata['data_settings'] = data
        metadata['frozen_supports_verified_against_original_jsonl'] = bool(rows)
        metadata['split_manifest_identity'] = identity
        bundle = load_checkpoint_bundle(a.checkpoint, split_manifest_identity=identity,
                                        output_dir=out, runtime=config['runtime'])
        bundle.config = config
        bundle.model.eval()
        all_records, physical_groups, skipped = [], [], []
        for name in a.tests:
            execution, fixed_batch, use_cache = TESTS[name]
            size = a.batch_size if name in ('D', 'E') else fixed_batch
            try:
                rec, groups = run_condition(bundle, episodes, name, batch_size=size,
                    cache_features=use_cache, checkpoint=a.checkpoint, commit=metadata['git_commit'])
                all_records.extend(rec)
                physical_groups.extend(groups)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if not (a.continue_on_oom and 'out of memory' in str(exc).lower()):
                    raise
                skipped.append({'condition': name, 'reason': f'OOM: {str(exc)[:250]}'})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        with (out / 'diagnostics.jsonl').open('w', encoding='utf-8') as fp:
            for r in all_records:
                fp.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')
        (out / 'physical_groups.json').write_text(json.dumps(physical_groups, indent=2), encoding='utf-8')
        summary = summarize(all_records, original_by_query, a.tests)
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as fp:
            writer = csv.DictWriter(fp, fieldnames=['condition','executed','queries','reproduced_missing_prefix','alignment'])
            writer.writeheader()
            for s in summary:
                writer.writerow(s)
        metadata['test_conditions_executed'] = [s['condition'] for s in summary if s['executed']]
        metadata['test_conditions_not_executed'] = skipped
        write_report(out, metadata, summary, all_records, original_by_query, skipped)
        print('Diagnostics complete:', out)
    except Exception as exc:
        metadata['status'] = 'FAILED BEFORE COMPLETION'
        metadata['exception'] = repr(exc)
        (out / 'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write_report(out, metadata, [], [], {}, [{'condition': x, 'reason': repr(exc)} for x in a.tests])
        raise
    finally:
        (out / 'environment.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
                                               encoding='utf-8')


if __name__ == '__main__':
    main()
