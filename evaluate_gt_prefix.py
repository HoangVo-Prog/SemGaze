"""Controlled GT-prefix intervention on the frozen SemGaze SFT test split.

Run from repository root (no retraining, same checkpoint, same K/draw support).

python evaluate_gt_prefix.py \
  --checkpoint runs/2e/checkpoint-6050 \
  --output-dir runs/gt_prefix_sft_test \
  --prefixes 1 3 5 --k-values 1 5 10 --draw 1 --batch-size 1

Two conditions for each eligible (query, K, draw, p):
  free:   model generates all N fixations, GT length N is given in the prompt
  gt_p:   GT first p fixations are teacher-forced into query assistant prefix;
          model generates only suffix p+1..N; same support and GT length N.

The GT prefix is never counted in accuracy/collapse metrics. This is a causal
prefix-intervention diagnostic, NOT a model retraining or SM/MM/SED report.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import statistics
import time

import torch

from evaluate_flat import _load_evaluation_config, _prediction_plan, _validate_frozen_supports
from gt_prefix_core import native_gt_prefix, suffix_diagnostics, compare_suffix, xy_distance
from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter
from semgaze.data.joint import JointAdapter
from semgaze.data.fewshot import frozen_episode
from semgaze.model.checkpoint import load_checkpoint_bundle
from semgaze.model.config import ROOT, resolve_config
from semgaze.model.visual_cache import InferenceVisualCache
from semgaze.evaluation.batching import physical_size
from semgaze.evaluation.test import test_mode, test_queries
from semgaze.evaluation.records import implementation_head, expected_episode_keys
from semgaze.where.collator import collate_where, pack_where_batch, to_model_device, WhereContextOverflowError
from semgaze.where.generation import (
    decode_response, left_pad_generation, generate_where_batch, where_generation_budget,
)
from semgaze.where.serialization import serialize_xyd_record, parse_xyd_output


def _name_dataset(query):
    return getattr(query, 'dataset', 'COCO-Search18')


def _native_suffix_generation(bundle, episodes, prefix, *, cache):
    """Exact native prefix with the existing generation/cache implementation."""
    if not episodes:
        return []
    if len({len(ep.supports) for ep in episodes}) != 1:
        raise ValueError('GT-prefix batching may not mix K')
    token = bundle.config['where']['end_fix_token']
    tokenizer = bundle.processor.tokenizer
    prepared = []
    for ep in episodes:
        target = serialize_xyd_record(ep.query, token)
        native = collate_where(bundle.processor, ep, bundle.end_fix_id, bundle.context_limit,
                               teacher_forcing=True, config=bundle.config, image_cache=cache)
        sample, forced, used = native_gt_prefix(
            native, prefix_count=prefix, end_fix_id=bundle.end_fix_id,
            tokenizer=tokenizer, target=target, end_fix_token=token)
        # Preserve original token ceiling (full-length oracle budget), despite
        # having supplied p GT fixations. At least 1 generation token is needed.
        remaining_budget = max(1, where_generation_budget(bundle, ep) - used)
        prepared.append((ep, sample, forced, used, remaining_budget))
    results = [None] * len(prepared)
    for budget in dict.fromkeys(item[4] for item in prepared):
        indices = [i for i, item in enumerate(prepared) if item[4] == budget]
        subset = [prepared[i] for i in indices]
        group = [item[0] for item in subset]
        samples = [item[1] for item in subset]
        batch = pack_where_batch(bundle.processor, group, samples, cache)
        inputs = left_pad_generation(batch.inputs, tokenizer.pad_token_id)
        prompt_length = inputs['input_ids'].shape[1]
        if prompt_length + budget > bundle.context_limit:
            raise WhereContextOverflowError('GT-prefix query + generation budget exceeds context limit')
        pixels = inputs['pixel_values']
        cached_features = cache is not None and cache.features
        inputs = to_model_device({k: v for k, v in inputs.items()
                                  if not (cached_features and k == 'pixel_values')}, bundle.model)
        if cached_features:
            fused, _ = cache.fuse(bundle, inputs, [p for s in samples for p in s.image_paths],
                                  host_pixel_values=pixels)
            inputs = fused | {'input_ids': inputs['input_ids']}
        output = bundle.model.generate(**inputs, max_new_tokens=budget,
            do_sample=False, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, use_cache=True)
        for row, index in enumerate(indices):
            ep, _, forced, used, _ = prepared[index]
            tail = decode_response(tokenizer, output[row, prompt_length:])
            full = forced + tail
            parsed = parse_xyd_output(full, len(ep.query.x_px), token)
            # A GT prefix must match the serialized GT exactly in token space.
            expected = parse_xyd_output(serialize_xyd_record(ep.query, token),
                                        len(ep.query.x_px), token)['fixations']
            if parsed['fixations'][:prefix] != expected[:prefix]:
                raise RuntimeError('GT-prefix corruption: forced fixations were not recovered')
            results[index] = {
                'text': full, **parsed, 'oracle_length_conditioned': True,
                'forced_prefix_count': prefix, 'forced_prefix_tokens': used,
                'generated_continuation_text': tail,
                'generated_fixation_count': max(0, len(parsed['fixations']) - prefix),
                'remaining_max_new_tokens': budget,
            }
    return results


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _average(items, key):
    vals = [x[key] for x in items if x.get(key) is not None]
    return statistics.mean(vals) if vals else None


def _bootstrap_ci(pairs, seed=2027, n=1000):
    """Cluster bootstrap over query IDs (all draws of one query resampled together)."""
    grouped = defaultdict(list)
    for x in pairs:
        if x['delta_xy_error'] is not None:
            grouped[x['query_id']].append(x['delta_xy_error'])
    keys = list(grouped)
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sampled = [grouped[keys[rng.randrange(len(keys))]] for _ in keys]
        values = [v for block in sampled for v in block]
        means.append(statistics.mean(values))
    means.sort()
    return [means[int(.025*n)], means[min(n-1,int(.975*n))]]


def _summarize(rows, prefixes, total_episode_count):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['dataset'], row['K'], row['prefix'])].append(row)
    groups = []
    for (dataset, k, prefix), block in sorted(grouped.items()):
        n = len(block)
        groups.append({
            'dataset': dataset, 'K': k, 'prefix': prefix, 'eligible_episodes': n,
            'free_complete_rate': sum(x['free_full_suffix'] for x in block)/n,
            'prefix_complete_rate': sum(x['prefix_full_suffix'] for x in block)/n,
            'free_collapse3_rate': sum(x['free_collapse3'] for x in block)/n,
            'gt_prefix_collapse3_rate': sum(x['prefix_collapse3'] for x in block)/n,
            'free_absorbing_collapse_rate': sum(x['free_absorbing_collapse'] for x in block)/n,
            'gt_prefix_absorbing_collapse_rate': sum(x['prefix_absorbing_collapse'] for x in block)/n,
            'free_mean_xy_error': _average(block, 'free_xy_error'),
            'gt_prefix_mean_xy_error': _average(block, 'prefix_xy_error'),
            'paired_delta_xy_error': _average(block, 'delta_xy_error'),
            'delta_bootstrap_95pct_ci': _bootstrap_ci(block),
            'paired_with_at_least_one_position': sum(x['paired_positions'] > 0 for x in block),
            'free_mean_suffix_movement': _average(block,'free_suffix_movement'),
            'gt_prefix_mean_suffix_movement': _average(block,'prefix_suffix_movement'),
            'recovery_among_free_suffix_collapses': (
                sum(x['free_collapse3'] and not x['prefix_collapse3'] for x in block) /
                sum(x['free_collapse3'] for x in block)
                if any(x['free_collapse3'] for x in block) else None),
            'free_collapse_pre_prefix_episodes': sum(x['free_collapse_before_prefix'] for x in block),
        })
    return {'total_free_episodes': total_episode_count, 'prefixes': list(prefixes),
            'interpretation': 'negative paired_delta_xy_error is better; '
                              'collapse is measured ONLY on generated suffix positions',
            'groups': groups}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, help='Evaluation-only overrides (same semantics as evaluate_flat.py)')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--prefixes', type=int, nargs='+', default=[1, 3, 5])
    p.add_argument('--k-values', type=int, nargs='+', default=None)
    p.add_argument('--draw', type=int, default=None, help='Persisted support draws per K; default checkpoint evaluation.draw')
    p.add_argument('--batch-size', type=int, default=1, help='Safe physical maximum; respects existing per-K config')
    p.add_argument('--dataset', choices=('all', 'AiR', 'COCO-Search18'), default='all')
    p.add_argument('--where-support-prompt-style', choices=('full','concise'),
                   help='Same as checkpoint unless overridden; compare runs only with identical style')
    p.add_argument('--dry-run', action='store_true', help='Check split/supports without loading GPU')
    p.add_argument('--max-queries', type=int, default=None,
                   help='Smoke tests only; omit for complete scientific test evaluation')
    args = p.parse_args(argv)
    if args.batch_size < 1 or not args.prefixes or len(set(args.prefixes)) != len(args.prefixes) or min(args.prefixes) < 1:
        p.error('batch-size must be positive and prefixes must be distinct positive integers')
    if args.max_queries is not None and args.max_queries < 1:
        p.error('--max-queries must be positive')
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == checkpoint or output_dir in checkpoint.parents or checkpoint in output_dir.parents:
        p.error('output directory may not be checkpoint, checkpoint parent or its child')
    if output_dir.exists() and any(output_dir.iterdir()):
        p.error('output directory must be new/empty to prevent accidental mixed checkpoints')
    config = _load_evaluation_config(checkpoint, args.config)
    if args.k_values is not None:
        config['evaluation']['k_values'] = args.k_values
    if args.draw is not None:
        config['evaluation']['draw'] = args.draw
    if args.where_support_prompt_style is not None:
        config['where']['support_prompt_style'] = args.where_support_prompt_style
    config = resolve_config(config)
    # Read-only: all test identities and supports come from the existing manifest.
    data = config['data']
    raw, manifest, identity = read_persisted_splits(ROOT / data['split_root'], data_config=data)
    adapter_cls = JointAdapter if data['dataset'] == 'all' else CocoSearch18Adapter
    adapter = adapter_cls(ROOT / data['images_root'], annotation_frame=data['annotation_frame'],
                          duration_field=data['duration']['source_field'])
    train = {r['record_id']: adapter(r) for r in raw['train']}
    tests = [adapter(r) for r in raw['test']]
    queries = test_queries(train, tests, manifest)
    if args.dataset != 'all':
        queries = [q for q in queries if _name_dataset(q) == args.dataset]
    if args.max_queries is not None:
        queries = queries[:args.max_queries]
    if not queries:
        p.error('no eligible queries')
    draw_counts, _ = _prediction_plan(config, manifest, enabled_where=True, enabled_semantic=False)
    _validate_frozen_supports(train, queries, manifest, draw_counts)
    print(f'[GT-PREFIX] checkpoint={checkpoint} queries={len(queries)} '
          f'K={config["evaluation"]["k_values"]} draws={draw_counts} '
          f'prefixes={args.prefixes} split=test dry_run={args.dry_run}',flush=True)
    if args.dry_run:
        print('[GT-PREFIX] dry-run PASS: no model or GPU loaded; no files written',flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_checkpoint_bundle(checkpoint, split_manifest_identity=identity,
                                    output_dir=output_dir, runtime=config['runtime'])
    bundle.config = config
    cache_opts = config['test']['cache']
    cache = InferenceVisualCache(preprocessing=cache_opts['support_preprocessing'],
        features=cache_opts['frozen_visual_features'], max_entries=cache_opts['max_entries'])
    provenance = {
        'checkpoint': str(checkpoint), 'source_head': implementation_head(),
        'split_manifest_identity': identity, 'split': 'test',
        'dataset_filter': args.dataset, 'max_queries': args.max_queries,
        'K': list(config['evaluation']['k_values']), 'draws': draw_counts,
        'prefixes': list(args.prefixes), 'physical_batch_size_max': args.batch_size,
        'where_support_prompt_style': config['where']['support_prompt_style'],
        'oracle_length_conditioned': True, 'free_generation': 'unchanged generate_where_batch',
        'prefix_construction': 'native teacher-forced END_FIX token slicing',
        'model_updates': False, 'semantic_generation': False,
    }
    (output_dir/'protocol.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    results_file = output_dir/'interventions.jsonl'
    partial = results_file.with_suffix('.jsonl.partial')
    summary_rows, point_rows = [], []
    free_count = 0
    start = time.monotonic()
    try:
        with test_mode(bundle):
            cache.validate(bundle)
            with partial.open('w', encoding='utf-8') as stream:
                for k in config['evaluation']['k_values']:
                    physical = min(args.batch_size, physical_size(config['test']['prediction'], k,
                        batch_size=config['evaluation'].get('batch_size'),
                        batch_size_by_k=config['evaluation'].get('batch_size_by_k', {})))
                    for draw in range(draw_counts[str(k)]):
                        episodes = [frozen_episode(q, train, manifest, k, draw_id=draw,
                                    unseen_subjects=manifest['unseen_subject_ids']) for q in queries]
                        for offset in range(0, len(episodes), physical):
                            block = episodes[offset:offset+physical]
                            free_out = generate_where_batch(bundle, block, cache=cache)
                            prefix_outputs = {}
                            for prefix in args.prefixes:
                                selected = [(i,e) for i,e in enumerate(block) if len(e.query.x_px) > prefix]
                                output = _native_suffix_generation(bundle,
                                    [ep for _,ep in selected], prefix, cache=cache)
                                prefix_outputs[prefix] = dict(zip((i for i,_ in selected), output))
                            for i, (ep, free) in enumerate(zip(block, free_out)):
                                q=ep.query
                                gt_text=serialize_xyd_record(q, config['where']['end_fix_token'])
                                gt=parse_xyd_output(gt_text,len(q.x_px),config['where']['end_fix_token'])['fixations']
                                key={'query_id':q.record_id,'dataset':_name_dataset(q),'subject':q.subject,
                                     'task':q.task,'K':k,'draw_id':draw,
                                     'support_ids':[s.record_id for s in ep.supports],
                                     'split_manifest_identity':identity}
                                def save(condition, result, prefix):
                                    row=dict(key, condition=condition, forced_prefix_count=prefix,
                                             gt_fixation_count=len(gt),
                                             WHERE={'GT':gt_text,'PRED':result['text']},
                                             where_generation={field:v for field,v in result.items() if field!='text'})
                                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                                free=dict(free,oracle_length_conditioned=True,forced_prefix_count=0)
                                save('free',free,0)
                                free_count+=1
                                for prefix in args.prefixes:
                                    if i not in prefix_outputs[prefix]:
                                        continue
                                    intervention=prefix_outputs[prefix][i]
                                    save(f'gt_prefix_{prefix}', intervention, prefix)
                                    d0=suffix_diagnostics(gt,free['fixations'],prefix)
                                    d1=suffix_diagnostics(gt,intervention['fixations'],prefix)
                                    paired=compare_suffix(gt,free['fixations'],intervention['fixations'],prefix)
                                    free_before=free['fixations'][:prefix]
                                    free_pre_error=(statistics.mean(xy_distance(a,b)
                                        for a,b in zip(gt[:prefix],free_before)) if len(free_before)==prefix else None)
                                    free_pre_collapse=any(free_before[j][:2] == free_before[j-1][:2] == free_before[j-2][:2]
                                        for j in range(2,len(free_before)))
                                    summary_rows.append(dict(key,prefix=prefix,
                                        free_pre_prefix_xy_error=free_pre_error,
                                        free_collapse_before_prefix=free_pre_collapse,
                                        free_suffix_movement=d0['predicted_suffix_movement'],
                                        prefix_suffix_movement=d1['predicted_suffix_movement'],
                                        free_full_suffix=d0['full_suffix'],prefix_full_suffix=d1['full_suffix'],
                                        free_collapse3=d0['collapse3'], prefix_collapse3=d1['collapse3'],
                                        free_absorbing_collapse=d0['absorbing_collapse'],
                                        prefix_absorbing_collapse=d1['absorbing_collapse'],
                                        free_collapse_at_t=d0['collapse_at_t'],
                                        prefix_collapse_at_t=d1['collapse_at_t'], **paired))
                                    for t in range(prefix,len(gt)):
                                        fp=free['fixations']; ip=intervention['fixations']
                                        point_rows.append({'dataset':_name_dataset(q),'query_id':q.record_id,
                                            'K':k,'draw_id':draw,'prefix':prefix,'t':t+1,
                                            'free_xy_error':xy_distance(gt[t],fp[t]) if len(fp)>t else None,
                                            'gt_prefix_xy_error':xy_distance(gt[t],ip[t]) if len(ip)>t else None,
                                            'free_xy_repeat':fp[t][:2]==fp[t-1][:2] if len(fp)>t else None,
                                            'gt_prefix_xy_repeat':ip[t][:2]==ip[t-1][:2] if len(ip)>t else None})
                            stream.flush()
                            done=offset+len(block)
                            if done==len(episodes) or done%50<physical:
                                print(f'[GT-PREFIX] K={k} draw={draw} {done}/{len(episodes)} '
                                      f'elapsed={time.monotonic()-start:.0f}s',flush=True)
        partial.replace(results_file)
    finally:
        cache.close()
    expected=expected_episode_keys([q.record_id for q in queries],
        config['evaluation']['k_values'],draw_counts)
    if free_count != len(expected):
        raise RuntimeError(f'free-running coverage failure: got {free_count}, expected {len(expected)}')
    _write_csv(output_dir/'paired_queries.csv',summary_rows)
    _write_csv(output_dir/'paired_fixations.csv',point_rows)
    artifact=_summarize(summary_rows,args.prefixes,free_count)
    (output_dir/'summary.json').write_text(json.dumps(artifact,indent=2,allow_nan=False),encoding='utf-8')
    print(f'[GT-PREFIX] DONE | free episodes={free_count} | '
          f'paired comparisons={len(summary_rows)} | output={output_dir}',flush=True)

if __name__=='__main__':
    main()
