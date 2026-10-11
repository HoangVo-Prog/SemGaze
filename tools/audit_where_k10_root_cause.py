#!/usr/bin/env python3
"""Controlled, read-only WHERE K=10 interventions. Run from the SemGaze repo root.

All outputs are isolated in --output-dir. Runtime always uses a single episode,
uncached visual processing, greedy decoding, and the checkpoint's own split.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OPENING = '[('
INSTRUCTION = 'Analyze this image and predict a human eye movement scanpath'


def digest(value):
    if isinstance(value, str):
        value = value.encode('utf-8')
    return hashlib.sha256(value).hexdigest()


def support_ids(episode):
    return [record.record_id for record in episode.supports]


def nested_supports(episode):
    from semgaze.data.schema import FlatEpisode
    s = episode.supports
    if len(s) != 10:
        raise ValueError('controlled K ablation requires exactly 10 original supports')
    return {name: FlatEpisode(tuple(items), episode.query, episode.draw_id) for name, items in (
        ('A1_first1', s[:1]), ('A2_last1', s[-1:]), ('A3_first5', s[:5]),
        ('A4_last5', s[-5:]), ('A5_original10', s))}


def reordered_supports(episode, variant, seed=2027):
    from semgaze.data.schema import FlatEpisode
    items = list(episode.supports)
    if variant == 'C_reverse':
        items.reverse()
    elif variant in ('C_perm1', 'C_perm2'):
        # Deterministic per query and seed, irrespective of Python hash randomization.
        salt = int(digest(f'{seed}:{episode.query.record_id}:{variant}')[:16], 16)
        random.Random(salt).shuffle(items)
        if items == list(episode.supports):
            items = items[1:] + items[:1]
    elif variant != 'C_original':
        raise ValueError(f'unknown ordering variant: {variant}')
    return FlatEpisode(tuple(items), episode.query, episode.draw_id)


def transform_messages(messages, episode, variant):
    """Change only support-user text. Keep roles, image placeholders, GT and query."""
    import copy
    output = copy.deepcopy(messages)
    k = len(episode.supports)
    if len(output) != 2*k+1:
        raise AssertionError('unexpected WHERE conversation layout')
    for index, record in enumerate(episode.supports):
        position = 2*index
        msg = output[position]
        if msg['role'] != 'user' or [x['type'] for x in msg['content']] != ['image', 'text']:
            raise AssertionError('unexpected support multimodal message structure')
        if variant == 'B2_concise':
            value = (f'Support reference. Task: {record.task}. '
                     f'Fixation count: {len(record.x_px)}. Follow the query response format.')
        elif variant == 'B3_minimal':
            value = f'Task: {record.task}.'
        elif variant == 'B1_original':
            value = msg['content'][1]['text']
        else:
            raise ValueError(f'unknown prompt variant: {variant}')
        msg['content'][1]['text'] = value
        if output[position+1] != messages[position+1]:
            raise AssertionError('support GT modified unexpectedly')
    if output[-1] != messages[-1]:
        raise AssertionError('query text/image changed unexpectedly')
    if [m['role'] for m in output] != [m['role'] for m in messages]:
        raise AssertionError('message role ordering changed')
    return output


def select_query_rows(predictions_jsonl, draw_id, query_ids, *, prefix_count=12, controls=4,
                      end_fix_token='<END_FIX>'):
    from tools.audit_where_prefix_runtime import load_original_records, _prefix_only
    rows = load_original_records(predictions_jsonl, 10, draw_id, set(query_ids),
                                 prefix_count=prefix_count, controls=controls,
                                 end_fix_token=end_fix_token)
    if not rows:
        raise ValueError('no matching historical COCO K=10 predictions')
    if not query_ids:
        bad = [r for r in rows if _prefix_only(r, end_fix_token=end_fix_token)]
        good = [r for r in rows if r not in bad]
        if len(bad) < 1 or len(good) < 4:
            raise ValueError('need historical failures and at least four valid K10 controls')
    return rows


def pilot_rows(rows, *, n_bad=4, n_good=2, end_fix_token='<END_FIX>'):
    from tools.audit_where_prefix_runtime import _prefix_only
    bad = [r for r in rows if _prefix_only(r, end_fix_token=end_fix_token)]
    good = [r for r in rows if r not in bad]
    if not bad or not good:
        raise ValueError('pilot requires both prefix failures and valid controls')
    return bad[:n_bad] + good[:n_good]


def verify_historical(episode, historical, checkpoint):
    if episode.query.record_id != historical['query_id']:
        raise ValueError('query ID mismatch')
    if episode.query.subject != historical['subject'] and str(episode.query.subject) != str(historical['subject']):
        raise ValueError('historical subject mismatch')
    if int(historical.get('K', historical.get('k', -1))) != 10:
        raise ValueError('historical K mismatch')
    if int(historical.get('draw_id', 0)) != episode.draw_id:
        raise ValueError('historical draw mismatch')
    if support_ids(episode) != historical['support_ids']:
        raise ValueError(f'frozen support identity/order mismatch for {episode.query.record_id}')
    request = historical.get('where_generation', {}).get('requested_fixation_count')
    if request != len(episode.query.x_px):
        raise ValueError('historical requested fixation count mismatch')
    if not checkpoint.is_dir() or not (checkpoint / 'resolved_config.json').exists():
        raise ValueError('checkpoint directory/config not found')
    return True


def source_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def native_prefill(processor, messages, paths, prefix):
    """Tokenizer+multimodal processor receives REAL assistant prefix pre-generation."""
    import torch
    from semgaze.where.collator import load_images
    native = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # This is before native tokenization and before model.generate, not postprocessing.
    full = native + prefix
    encoded = processor(text=[full], images=load_images(paths), return_tensors='pt',
        crop_to_patches=False, size={'height': 448, 'width': 448}, add_special_tokens=False)
    ids = encoded['input_ids'][0].tolist()
    if tuple(encoded['pixel_values'].shape) != (len(paths), 3, 448, 448):
        raise AssertionError('prefilled prompt changed the image tiling policy')
    if ids.count(processor.image_token_id) != len(paths) * processor.image_seq_length:
        raise AssertionError('prefilled prompt changed native image expansion')
    text = processor.tokenizer.decode(ids, skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)
    if not text.endswith(prefix):
        raise AssertionError('assistant prefix is not the end of tokenized prompt')
    if processor.tokenizer.encode(text, add_special_tokens=False) != ids:
        raise AssertionError('native prefill tokenization not losslessly decodable')
    return encoded, text, full


def make_input(bundle, episode, variant='B1_original', prefill=''):
    from semgaze.where.conversation import build_where_conversation
    from semgaze.where.collator import collate_native
    original_messages, paths = build_where_conversation(
        episode, bundle.config['where']['end_fix_token'])
    messages = transform_messages(original_messages, episode, variant)
    native_text = bundle.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    if prefill:
        raw, rendered, actual_text = native_prefill(bundle.processor, messages, paths, prefill)
        inputs = dict(raw)
        inputs['attention_mask'] = raw['attention_mask']
    else:
        batch = collate_native(bundle.processor, messages, paths, target=None, image_cache=None)
        inputs, rendered, actual_text = dict(batch.inputs), batch.rendered_text, native_text
    return {'inputs': inputs, 'rendered': rendered, 'native_text': actual_text,
            'messages': messages, 'paths': paths, 'prefill': prefill}


def create_observation(bundle, episode, condition, prepared, out, *, history=None, kind='manipulated'):
    import torch
    from semgaze.where.collator import to_model_device
    from semgaze.where.generation import where_generation_budget, decode_response
    from semgaze.where.serialization import parse_xyd_output
    tokenizer = bundle.processor.tokenizer
    ins = to_model_device(prepared['inputs'], bundle.model)
    ins.pop('labels', None)
    L = int(ins['input_ids'].shape[1])
    budget = where_generation_budget(bundle, episode)
    if L + budget > bundle.context_limit:
        raise ValueError(f'{condition}: prompt+budget exceeds context_limit')
    with torch.inference_mode():
        returned = bundle.model.generate(**ins, max_new_tokens=budget, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    ids = returned.sequences[0].tolist() if hasattr(returned, 'sequences') else returned[0].tolist()
    if ids[:L] != ins['input_ids'][0].tolist():
        raise AssertionError(f'{condition}: generate did not echo exact input IDs; stop before slicing')
    new_ids = ids[L:]
    decoded = decode_response(tokenizer, new_ids)
    response = prepared['prefill'] + decoded
    n = len(episode.query.x_px)
    parsed = parse_xyd_output(response, n, bundle.config['where']['end_fix_token'])
    messages = prepared['messages']
    query_text = messages[-1]['content'][1]['text']
    pos = prepared['rendered'].rfind(query_text)
    if pos == -1:
        raise AssertionError('cannot locate final query instruction')
    query_token_span = None
    try:
        native_tokens = tokenizer(prepared['rendered'], add_special_tokens=False,
                                  return_offsets_mapping=True)
        if native_tokens['input_ids'] == ins['input_ids'][0].tolist():
            offsets = native_tokens['offset_mapping']
            covered = [i for i, (a,b) in enumerate(offsets)
                       if a < pos+len(query_text) and b > pos]
            if covered:
                query_token_span = [covered[0], covered[-1]+1]
    except (TypeError, KeyError, ValueError):
        pass  # Char span and query message index remain exact.
    prefix_path = out / 'prompts' / f'{digest(prepared["rendered"])}.txt'
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    if not prefix_path.exists():
        prefix_path.write_text(prepared['rendered'], encoding='utf-8')
    row = {
        'query_id': episode.query.record_id, 'subject': episode.query.subject,
        'dataset': episode.query.dataset, 'K': len(episode.supports), 'draw_id': episode.draw_id,
        'condition': condition, 'episode_kind': kind, 'support_ids': support_ids(episode),
        'support_n_fixations': [len(s.x_px) for s in episode.supports],
        'requested_fixation_count': n, 'prompt_hash': digest(prepared['rendered']),
        'native_prompt_hash': digest(prepared['native_text']),
        'input_id_hash': digest(json.dumps(ins['input_ids'][0].tolist(), separators=(',', ':'))),
        'prompt_file': str(prefix_path.relative_to(out)), 'prompt_token_length': L,
        'instruction_repetitions': prepared['rendered'].count(INSTRUCTION),
        'query_char_span': [pos, pos+len(query_text)], 'query_token_span': query_token_span,
        'query_message_index': 2*len(episode.supports),
        'generation': {'do_sample': False, 'use_cache': True,
                       'visual_cache': False, 'batch_size': 1, 'max_new_tokens': budget},
        'prefill': prepared['prefill'], 'prefill_applied_before_generation': bool(prepared['prefill']),
        'first_generated_token_id': new_ids[0] if new_ids else None,
        'first_generated_token_text': tokenizer.convert_ids_to_tokens(new_ids[0]) if new_ids else None,
        'first_generated_token_decoded': tokenizer.decode(new_ids[:1], skip_special_tokens=False,
            clean_up_tokenization_spaces=False),
        'raw_generated_ids': new_ids, 'raw_generated_text': decoded,
        'response': response, 'first_100_output_chars': response[:100],
        'starts_with_open': response.startswith(OPENING),
        'continuation_starts_with_open': decoded.startswith(OPENING),
        'continuation_matches_prefill_suffix': True if prepared['prefill'] else None,
        'historical_prefix_bad': bool(history) and not history['WHERE']['PRED'].startswith(OPENING),
        'historical_prediction': history['WHERE']['PRED'] if history else None,
        **parsed,
    }
    if prepared['prefill'] and decoded.startswith(OPENING):
        row['prefill_duplicated_by_continuation'] = True
    else:
        row['prefill_duplicated_by_continuation'] = False
    # For prefilled responses the FULL canonical string, not the suffix, is checked.
    del ins, returned
    return row


def first_token_scores(bundle, episode, prepared, opening=OPENING, observed=None, top_k=8):
    """One-step generation scores; candidate chain scored via true token IDs.

    Never treats the literal characters '[(' as assumed single-token IDs.
    Each step recomputes the full multimodal prefix; no full-vocab logit history.
    """
    import torch
    from semgaze.where.collator import to_model_device
    tokenizer = bundle.processor.tokenizer
    base = to_model_device(prepared['inputs'], bundle.model)
    base.pop('labels', None)
    ids = base['input_ids']
    mask = base['attention_mask']
    result = {}
    targets = {'valid_opening': tokenizer.encode(opening, add_special_tokens=False)}
    if not targets['valid_opening'] or tokenizer.decode(targets['valid_opening'],
        skip_special_tokens=False, clean_up_tokenization_spaces=False) != opening:
        raise ValueError('opening tokenization does not exactly round-trip')
    if observed:
        match = re.match(r'\d+', observed)
        if match:
            digits = match.group(0)
            number_ids = tokenizer.encode(digits, add_special_tokens=False)
            if tokenizer.decode(number_ids, skip_special_tokens=False,
                clean_up_tokenization_spaces=False) == digits:
                targets['competing_numeric_opening'] = number_ids
    first_step = None
    for label, candidate_ids in targets.items():
        acc, stages = 0.0, []
        for step, tok in enumerate(candidate_ids):
            appended = candidate_ids[:step]
            if ids.shape[1] + len(appended) + 1 > bundle.context_limit:
                raise ValueError('candidate scoring exceeds context limit')
            if appended:
                extra = torch.tensor([appended], dtype=ids.dtype, device=ids.device)
                extra_mask = torch.ones_like(extra)
                current = dict(base, input_ids=torch.cat([ids, extra], dim=-1),
                               attention_mask=torch.cat([mask, extra_mask], dim=-1))
            else:
                current = base
            with torch.inference_mode():
                response = bundle.model.generate(**current, do_sample=False, max_new_tokens=1,
                    use_cache=True, return_dict_in_generate=True, output_scores=True,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                scores = response.scores[0][0].float()
                log_probs = torch.log_softmax(scores, dim=-1)
            acc += float(log_probs[tok].item())
            stages.append({'token_id': tok, 'token': tokenizer.convert_ids_to_tokens(tok),
                'log_probability': float(log_probs[tok].item())})
            if step == 0 and first_step is None:
                scores_top = torch.topk(log_probs, min(top_k, log_probs.numel()))
                first_step = {'actual_first_step_greedy_id': int(torch.argmax(log_probs).item()),
                    'actual_first_step_greedy_decoded': tokenizer.decode([int(torch.argmax(log_probs).item())],
                        skip_special_tokens=False, clean_up_tokenization_spaces=False),
                    'top_candidates': [{'id': int(i), 'token': tokenizer.convert_ids_to_tokens(int(i)),
                        'decoded': tokenizer.decode([int(i)], skip_special_tokens=False,
                          clean_up_tokenization_spaces=False), 'probability': float(torch.exp(v).item()),
                        'log_probability': float(v.item())}
                        for i,v in zip(scores_top.indices.tolist(), scores_top.values)]}
            del response, log_probs, scores
        result[label] = {'candidate_ids': candidate_ids, 'steps': stages,
            'sequence_log_probability': acc, 'sequence_probability': __import__('math').exp(acc),
            'scoring_method': 'one-step greedy HF generate scores; conditional token-sequence logprob'}
    return first_step | {'candidate_scores': result}


def write_report(out, metadata, observations, skips):
    by_cond = defaultdict(list)
    by_query = defaultdict(dict)
    for r in observations:
        if r.get('condition', '').startswith('E_'):
            continue
        by_cond[r['condition']].append(r)
        by_query[r['query_id']][r['condition']] = r
    summary = []
    for cond, vals in sorted(by_cond.items()):
        matched = [(r, by_query[r['query_id']].get('A5_original10')) for r in vals]
        corrected = sum(bool(base and base['historical_prefix_bad'] and
                         not base['canonical_format_valid'] and r['canonical_format_valid'])
                        for r, base in matched)
        damaged = sum(bool(base and not base['historical_prefix_bad'] and
                        base['canonical_format_valid'] and not r['canonical_format_valid'])
                      for r, base in matched)
        summary.append({'condition': cond, 'queries': len(vals), 'original_failures_corrected': corrected,
            'original_valid_broken': damaged,
            'canonical_valid': sum(r['canonical_format_valid'] for r in vals),
            'mean_prompt_tokens': round(sum(r['prompt_token_length'] for r in vals)/len(vals), 1)})
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (out/'summary.csv').open('w', newline='', encoding='utf-8') as fp:
        wr = csv.DictWriter(fp, fieldnames=list(summary[0]) if summary else [
            'condition','queries','original_failures_corrected','original_valid_broken',
            'canonical_valid','mean_prompt_tokens'])
        wr.writeheader(); wr.writerows(summary)
    cond = set(by_cond)
    def changed(names):
        return any((by_query[q].get(n, {}).get('canonical_format_valid') !=
                    by_query[q]['A5_original10']['canonical_format_valid']) for q in by_query
                   if 'A5_original10' in by_query[q] for n in names if n in by_query[q])
    h = {}
    h['H1 K/context sensitivity'] = ('CONFIRMED' if changed(('A1_first1','A2_last1','A3_first5','A4_last5')) else
        'NOT EXECUTED' if 'A1_first1' not in cond else 'RULED OUT FOR TESTED CASES')
    h['H2 repeated instructions'] = ('CONFIRMED' if changed(('B2_concise','B3_minimal')) else
        'NOT EXECUTED' if 'B2_concise' not in cond else 'RULED OUT FOR TESTED CASES')
    h['H3 support order'] = ('CONFIRMED' if changed(('C_reverse','C_perm1','C_perm2')) else
        'NOT EXECUTED' if 'C_reverse' not in cond else 'RULED OUT FOR TESTED CASES')
    rescues = any((x.get('A5_original10') and x.get('D2_prefill') and
                   not x['A5_original10']['canonical_format_valid'] and
                   x['D2_prefill']['canonical_format_valid']) for x in by_query.values())
    h['H4 assistant prefill'] = ('CONFIRMED' if rescues else
        'NOT EXECUTED' if 'D2_prefill' not in cond else 'RULED OUT FOR TESTED CASES')
    for key, value in list(h.items()):
        # These are *conditional comparison* verdicts, never proof of a unique root cause.
        if value == 'CONFIRMED' and key.startswith('H1'):
            h[key] = 'CONFIRMED (K/support-subset sensitivity; training cause INCONCLUSIVE)'
    matrix = ['| Query ID | Historical K10 | '+' | '.join(sorted(cond))+' |',
              '|---|---|'+'|'.join(['---']*len(cond))+'|']
    for q, tests in by_query.items():
        historical = 'bad' if next(iter(tests.values()))['historical_prefix_bad'] else 'valid'
        matrix.append('| '+str(q)+' | '+historical+' | '+' | '.join(
            'valid' if tests.get(name, {}).get('canonical_format_valid') else
            'invalid' if name in tests else '—' for name in sorted(cond))+' |')
    lines = ['# SemGaze WHERE K10 root cause: controlled runtime report',
        '## A. Prompt construction',
        'Original support user turns repeat the full WHERE instruction K times; the last query is another full instruction. '
        'Support assistant turns contain serialized GT fixations; query last.',
        '## B. Environment and reproduction', '```json',json.dumps(metadata,ensure_ascii=False,indent=2),'```',
        '## C. Paired ablations',
        '| Condition | Queries | Originally failing corrected | Originally valid broken | Valid | Mean prompt tokens |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f'| {r["condition"]} | {r["queries"]} | {r["original_failures_corrected"]} | '
            f'{r["original_valid_broken"]} | {r["canonical_valid"]}/{r["queries"]} | {r["mean_prompt_tokens"]} |')
    lines += ['## D. Query transitions', *matrix, '## E. First token scoring',
       'Read `first_token_scores.jsonl` for the actual token IDs, top candidates and conditional sequence scores.',
       '## F. Hypothesis verdicts']
    lines += [f'- {key}: **{value}**' for key,value in h.items()]
    lines += ['## G. Causal interpretation',
       'A change under an intervention establishes sensitivity for those exact frozen episodes, not a unique cause. '
       'A fixed checkpoint cannot establish that K=1-only training caused K10 failures. '
       'An assistant prefill only forces the opening; it does not show that the unprefilled model would produce it.',
       '## H. Recommendation',
       'Compare prefix rescue, previously valid controls, accepted counts and fixation tuple changes in diagnostics '
       'before shipping any inference intervention. No production files have been changed.',
       '## Unexecuted/failed comparisons',json.dumps(skips,ensure_ascii=False,indent=2)]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--config', type=Path)
    ap.add_argument('--predictions-jsonl', type=Path, required=True)
    ap.add_argument('--query-id', action='append', default=[])
    ap.add_argument('--draw-id', type=int, default=0)
    ap.add_argument('--prefix-count', type=int, default=12)
    ap.add_argument('--valid-controls', type=int, default=4)
    ap.add_argument('--stage', choices=('pilot','disambiguation','confirmation','all'), default='pilot')
    ap.add_argument('--experiments', nargs='+', choices=('A','B','C','D','E'))
    ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--continue-on-oom', action='store_true')
    a = ap.parse_args(argv)
    if a.draw_id < 0 or a.prefix_count < 1 or a.valid_controls < 4:
        ap.error('draw nonnegative; prefix-count>=1 and valid-controls>=4')
    if a.stage == 'all':
        selected = ('A','B','C','D','E')
    else:
        selected = tuple(a.experiments or ({'pilot':['A','B','D'],
                      'disambiguation':['C','E'], 'confirmation':['A','B','C','D']}[a.stage]))
    out = a.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        ap.error('output directory must be new or empty')
    out.mkdir(parents=True, exist_ok=True)
    meta = {'repo_commit':source_sha(),'checkpoint':str(a.checkpoint.resolve()),
        'checkpoint_config_hash':digest((a.checkpoint/'resolved_config.json').read_bytes()),
        'python':platform.python_version(),'draw_id':a.draw_id,'stage':a.stage,'experiments':selected,
        'batch_size':1,'visual_cache':False,'decoding':'greedy','seed':a.seed}
    observations, scores, skips = [], [], []
    try:
        import torch, transformers, peft
        from evaluate_flat import _load_evaluation_config
        from semgaze.model.config import resolve_config
        from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter
        from semgaze.data.joint import JointAdapter
        from semgaze.data.fewshot import frozen_episode
        from semgaze.evaluation.test import test_queries
        from semgaze.model.checkpoint import load_checkpoint_bundle
        from semgaze.where.conversation import build_where_conversation
        from semgaze.where.generation import where_generation_budget
        meta.update(torch=torch.__version__, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                    transformers=transformers.__version__, peft=peft.__version__)
        config = resolve_config(_load_evaluation_config(a.checkpoint,a.config))
        data = config['data']
        raw, manifest, identity = read_persisted_splits(ROOT/data['split_root'], data_config=data)
        adapter = (JointAdapter if data['dataset']=='all' else CocoSearch18Adapter)(
            ROOT/data['images_root'], annotation_frame=data['annotation_frame'],
            duration_field=data['duration']['source_field'])
        train = {r['record_id']:adapter(r) for r in raw['train']}
        tests = {q.record_id:q for q in test_queries(train, [adapter(r) for r in raw['test']], manifest)
                 if q.dataset == 'COCO-Search18'}
        original = select_query_rows(a.predictions_jsonl,a.draw_id,a.query_id,
            prefix_count=a.prefix_count,controls=a.valid_controls,
            end_fix_token=config['where']['end_fix_token'])
        selected_rows = pilot_rows(original, end_fix_token=config['where']['end_fix_token']) if a.stage in ('pilot','disambiguation') else original
        by_history = {r['query_id']:r for r in selected_rows}
        episodes = []
        for historical in selected_rows:
            q = tests[historical['query_id']]
            ep = frozen_episode(q,train,manifest,10,draw_id=a.draw_id,
                                unseen_subjects=manifest['unseen_subject_ids'])
            verify_historical(ep,historical,a.checkpoint)
            episodes.append(ep)
        meta.update(selected_query_ids=[x.query.record_id for x in episodes],
                    split_manifest_identity=identity, original_file=str(a.predictions_jsonl.resolve()),
                    original_file_sha256=digest(a.predictions_jsonl.read_bytes()),
                    checkpoint_training_k=data['fewshot']['k_values'],
                    checkpoint_training_k_probabilities=data['fewshot']['train_k_probabilities'],
                    first_support_ids=support_ids(episodes[0]),
                    episode_count=len(episodes),
                    where_prompt_example=build_where_conversation(episodes[0])[0][-1]['content'][1]['text'],
                    original_repetition_count=len(episodes[0].supports)+1,
                    original_k1_k5_nested_in_k10={})
        if a.stage == 'pilot':
            print('[PILOT] selected',len(episodes),'queries; historical verification complete',flush=True)
        bundle=load_checkpoint_bundle(a.checkpoint,split_manifest_identity=identity,
                                      output_dir=out,runtime=config['runtime'])
        bundle.config=config; bundle.model.eval()
        for idx, ep in enumerate(episodes):
            hist = by_history[ep.query.record_id]
            condition_outputs = {}
            def run(name, variant_ep, *, prompt_variant='B1_original', prefill='', episode_kind='manipulated'):
                prep = make_input(bundle, variant_ep, prompt_variant, prefill)
                result = create_observation(bundle,variant_ep,name,prep,out,history=hist,kind=episode_kind)
                result.update(git_commit=meta['repo_commit'], checkpoint=meta['checkpoint'],
                              split_manifest_identity=identity)
                observations.append(result); condition_outputs[name]=result
                print(f'[{idx+1}/{len(episodes)}] {name} {ep.query.record_id}: '
                      f'valid={result["canonical_format_valid"]}, prefix={result["starts_with_open"]}',flush=True)
                return result
            try:
                base = run('A5_original10',ep,episode_kind='persisted')
                if 'A' in selected:
                    nested = nested_supports(ep)
                    for name, e in nested.items():
                        if name != 'A5_original10': run(name,e)
                    for k in (1,5):
                        try:
                            independent = frozen_episode(ep.query,train,manifest,k,draw_id=a.draw_id,
                                                unseen_subjects=manifest['unseen_subject_ids'])
                            name=f'A_official{k}'
                            run(name,independent,episode_kind='persisted_independent')
                            meta['original_k1_k5_nested_in_k10'][f'{ep.query.record_id}:{k}'] = (
                               support_ids(independent)==support_ids(nested['A1_first1' if k==1 else 'A3_first5']))
                        except (KeyError,ValueError) as exc:
                            skips.append({'query_id':ep.query.record_id,'condition':f'A_official{k}',
                                          'reason':str(exc)})
                if 'B' in selected:
                    run('B2_concise',ep,prompt_variant='B2_concise')
                    run('B3_minimal',ep,prompt_variant='B3_minimal')
                if 'C' in selected:
                    for name in ('C_reverse','C_perm1','C_perm2'):
                        run(name,reordered_supports(ep,name,a.seed))
                if 'D' in selected:
                    after = run('D2_prefill',ep,prefill=OPENING)
                    after['baseline_tuple_count_delta'] = (after['accepted_fixation_count']-
                                                            base['accepted_fixation_count'])
                    after['tuples_equal_to_baseline'] = after['fixations']==base['fixations']
                    after['baseline_common_tuple_prefix'] = next((j for j, (x,y) in enumerate(
                        zip(after['fixations'],base['fixations'])) if x!=y),
                        min(len(after['fixations']),len(base['fixations'])))
                if 'E' in selected:
                    for k in (1,5,10):
                        sc_ep = ep if k==10 else frozen_episode(ep.query,train,manifest,k,
                            draw_id=a.draw_id, unseen_subjects=manifest['unseen_subject_ids'])
                        prep=make_input(bundle,sc_ep)
                        result=first_token_scores(bundle,sc_ep,prep,observed=base['raw_generated_text'] if k==10 else None)
                        scores.append({'query_id':ep.query.record_id,'subject':ep.query.subject,
                            'K':k,'support_ids':support_ids(sc_ep),'prompt_token_length':int(prep['inputs']['input_ids'].shape[1]),
                            'baseline_first_generated_token_id':base['first_generated_token_id'] if k==10 else None,
                            'first_generated_token_id': result['actual_first_step_greedy_id'],
                            'first_generated_token_decoded': result['actual_first_step_greedy_decoded'],
                            **result})
            except (torch.cuda.OutOfMemoryError,RuntimeError) as exc:
                if not (a.continue_on_oom and 'out of memory' in str(exc).lower()):
                    raise
                skips.append({'query_id':ep.query.record_id,'reason':f'OOM: {exc}'})
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            finally:
                for name, target in [('diagnostics.jsonl', observations),('first_token_scores.jsonl',scores)]:
                    with (out/name).open('w',encoding='utf-8') as stream:
                        for record in target:stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
        baselines=[r for r in observations if r['condition']=='A5_original10']
        meta['baseline_prefix_failures_reproduced']=sum(
            r['historical_prefix_bad'] and not r['starts_with_open'] for r in baselines)
        meta['baseline_originally_bad_count']=sum(r['historical_prefix_bad'] for r in baselines)
        meta['baseline_control_valid_count']=sum(
            not r['historical_prefix_bad'] and r['canonical_format_valid'] for r in baselines)
        meta['observations']=len(observations);meta['first_token_score_rows']=len(scores)
        write_report(out,meta,observations,skips)
    except Exception:
        (out/'failure_traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        (out/'environment.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':
    main()
