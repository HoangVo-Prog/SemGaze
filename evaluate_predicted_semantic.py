"""K=1 semantic inference from frozen WHERE JSONL. Never generates WHERE.

Run from the SemGaze repository root. Does not alter GT-conditioned evaluation.
Only query PRED XYD is replayed. WHY-group membership is selectable:
  off (default): one trajectory-level WHY with a dedicated no-group prompt;
  on: original GT WHY-group membership, without GT trajectory tokens.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import re

from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter
from semgaze.data.joint import JointAdapter, AirAdapter
from semgaze.data.fewshot import evaluation_episode
from semgaze.data.schema import FlatEpisode
from semgaze.data.semantic import NormalizedSemantic, WhyGroup
from semgaze.evaluation.metrics_semantic import semantic_metrics_enabled
from semgaze.evaluation.predictions import score_prediction_artifact
from semgaze.evaluation.test import test_mode, test_queries
from semgaze.model.checkpoint import load_checkpoint_bundle
from semgaze.model.config import ROOT, resolve_config, write_run_config
from semgaze.semantic.flat.parser import parse_flat_output
from semgaze.semantic.flat.prompt import FlatPrompt, build_flat_prompt
from semgaze.where.coordinates import coordinate_bin
from semgaze.where.duration import duration_bin
from semgaze.semantic.flat.target import build_flat_target
from semgaze.state.insertion import insert_states
from semgaze.where.collator import collate_native, to_model_device, WhereContextOverflowError
from semgaze.where.conversation import image_user
from semgaze.where.forward import forward_where_batch
from semgaze.where.generation import decode_response
from semgaze.where.serialization import serialize_xyd, serialize_xyd_record, active_end_fix_token
from evaluate_flat import _load_evaluation_config


class InvalidPrediction(ValueError):
    pass


def filter_k1_records(path):
    """All three K/draw conditions are mandatory. No oracle K fallback."""
    selected, total, discarded = [], 0, Counter()
    with Path(path).open(encoding='utf-8') as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{lineno}: invalid JSON') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{lineno}: expected a JSON object')
            if any(type(row.get(key)) is not int or row[key] != 1
                   for key in ('K', 'k')) or type(row.get('draw_id')) is not int or row['draw_id'] != 0:
                discarded['not_K1_draw0'] += 1
                continue
            if not isinstance(row.get('query_id'), str) or not row['query_id']:
                raise ValueError(f'{path}:{lineno}: K=1 query_id is missing')
            selected.append(row)
    ids = [r['query_id'] for r in selected]
    if len(ids) != len(set(ids)):
        raise ValueError(f'{path}: duplicate K=1 query_id')
    return selected, {'all_records': total, 'k1_records': len(selected),
                      'excluded': dict(discarded)}


def parse_stored_prediction(row, token):
    """Read the COMPLETE raw WHERE.PRED; never cap length to GT fixation count."""
    pred = row.get('WHERE', {}).get('PRED')
    if not isinstance(pred, str) or not pred:
        raise InvalidPrediction('missing_where_pred')
    if not token:
        raise ValueError('no-END_FIX checkpoints require a separate validated parser')
    pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*' + re.escape(token)
    matches = re.findall(pattern, pred)
    fixes = [tuple(int(v) for v in match) for match in matches]
    if not fixes or any(not (0 <= x <= 99 and 0 <= y <= 99 and 0 <= d <= 999)
                        for x, y, d in fixes):
        raise InvalidPrediction('no_valid_predicted_fixations')
    if serialize_xyd(fixes, token) != pred:
        raise InvalidPrediction('noncanonical_where_pred')
    generated = row.get('where_generation', {})
    parsed = generated.get('fixations')
    # Legacy parser caps diagnostics at oracle N. The full WHERE.PRED is primary.
    if parsed is not None:
        if not isinstance(parsed, list) or fixes[:len(parsed)] != [tuple(v) for v in parsed]:
            raise InvalidPrediction('where_text_parser_disagreement')
        if len(parsed) > len(fixes):
            raise InvalidPrediction('where_parser_longer_than_raw')
    return fixes, {'legacy_parsed_fixation_count': len(parsed) if parsed is not None else None,
                   'parsed_list_truncated': parsed is not None and len(parsed) < len(fixes)}


def shadow_query_from_predicted(query, fixations):
    """Only image/task/observer identity and PRED XYD enter semantic inference.

    The placeholder texts NEVER enter a prompt or target. One WHOLE-trajectory
    WHY group is a deterministic grouping of PRED positions, not GT membership.
    """
    if not fixations:
        raise InvalidPrediction('empty_predicted_fixations')
    n = len(fixations)
    placeholder = 'unused-target-placeholder'
    shadow_sem = NormalizedSemantic(
        what=(placeholder,) * n,
        why_groups=(WhyGroup(tuple(range(1, n + 1)), placeholder),),
        how=placeholder)
    return replace(query,
        x_px=tuple(x * query.image_width / 100.0 for x, _, _ in fixations),
        y_px=tuple(y * query.image_height / 100.0 for _, y, _ in fixations),
        duration_ms=tuple(d for _, _, d in fixations), semantic=shadow_sem)


def assert_replay_identity(shadow, fixations, token):
    expected = serialize_xyd(fixations, token)
    actual = serialize_xyd_record(shadow, token)
    if actual != expected:
        raise InvalidPrediction('coordinate_duration_roundtrip_mismatch')
    return actual


NO_GOLD_GROUP_INSTRUCTION = """Describe the visual exploration along the supplied scanpath.

For every fixation, provide one concise WHAT description of the specific visible object or image region being inspected.

Provide exactly one WHY statement summarizing the overall task-relevant rationale for the complete predicted trajectory. Do not divide the trajectory into groups or describe group membership.

Finally, provide one concise HOW description of the visual search or exploration strategy across the complete trajectory.

Return only the following flat line format:
WHAT 1: ...
...
WHAT {N}: ...
WHY 1: ...
HOW: ...

Rules:
- Return exactly {N} WHAT lines, one WHY line, then one HOW line.
- Keep WHAT indices in fixation order 1..{N}.
- WHY 1 must describe the full trajectory, not a subsection of fixations.
- Keep each value on one line.
- Do not output group membership, JSON, Python objects, lists, bullets, code fences, or extra explanation."""


def shadow_query_for_group_mode(query, fixations, *, gold_why_groups):
    """Map PRED to the semantic query without leaking GT XYD in either mode.

    When gold WHY groups are enabled, predicted and GT trajectory lengths
    must agree for the *original* index memberships to remain well-defined.
    No padding/truncation or matching of fixations is performed.
    """
    shadow = shadow_query_from_predicted(query, fixations)
    if gold_why_groups:
        if len(fixations) != len(query.x_px):
            raise InvalidPrediction('gold_groups_require_equal_fixation_counts')
        placeholder = 'unused-target-placeholder'
        semantic = replace(shadow.semantic, why_groups=tuple(
            WhyGroup(tuple(group.members), placeholder)
            for group in query.semantic.why_groups))
        shadow = replace(shadow, semantic=semantic)
    return shadow


def no_gold_group_prompt(shadow, conditioning):
    """Dedicated prompt: no gold WHY membership/count, no synthetic group list.

    The internal one-group placeholder exists solely for the existing flat
    parser; it is NOT exposed as a grouping annotation in the user prompt.
    """
    if conditioning not in ('shared_states', 'explicit_scanpath_text'):
        raise ValueError(f'unsupported semantic conditioning: {conditioning!r}')
    n = len(shadow.x_px)
    if n < 1 or len(shadow.y_px) != n or len(shadow.duration_ms) != n:
        raise ValueError('predicted semantic query requires aligned XYD')
    text = f'Task:\n{shadow.task}\n\n'
    text += 'Exploration states:\n' if conditioning == 'shared_states' else 'Scanpath:\n'
    boundaries = []
    for t in range(n):
        text += f'Fixation {t + 1}:'
        if conditioning == 'shared_states':
            boundaries.append(len(text))
        else:
            x = coordinate_bin(shadow.x_px[t], shadow.image_width)
            y = coordinate_bin(shadow.y_px[t], shadow.image_height)
            d = duration_bin(shadow.duration_ms[t])
            text += f' ({x:02d}, {y:02d}, {d:03d})'
        text += '\n'
    text += '\n' + NO_GOLD_GROUP_INSTRUCTION.format(N=n)
    return FlatPrompt(text, tuple(boundaries))


def prediction_only_prompt(shadow, conditioning, *, gold_why_groups=False):
    """Prompt mode selector; with-gold reuses the canonical training prompt."""
    if gold_why_groups:
        return build_flat_prompt(shadow, conditioning=conditioning)
    return no_gold_group_prompt(shadow, conditioning)


def validate_alignment(row, query, episode, token, identity):
    if row.get('split') not in (None, 'test'):
        raise ValueError(f'{query.record_id}: input is not a held-out test prediction')
    if 'split_manifest_identity' in row and row['split_manifest_identity'] != identity:
        raise ValueError(f'{query.record_id}: split manifest identity mismatch')
    for key, value in (('dataset', query.dataset), ('subject', query.subject),
                       ('stimulus_id', query.stimulus_id), ('task', query.task),
                       ('condition', query.condition)):
        if str(row.get(key)) != str(value):
            raise ValueError(f'{query.record_id}: {key} does not match test manifest')
    if row.get('support_ids') != [r.record_id for r in episode.supports]:
        raise ValueError(f'{query.record_id}: original few-shot supports do not match frozen draw')
    if row.get('annotation_width') != query.image_width or row.get('annotation_height') != query.image_height:
        raise ValueError(f'{query.record_id}: annotation dimensions mismatch')
    pixels = row.get('gt_raw_pixels')
    durations = row.get('gt_duration_ms_raw')
    if pixels != [[x, y] for x, y in zip(query.x_px, query.y_px)] or durations != list(query.duration_ms):
        raise ValueError(f'{query.record_id}: JSONL GT does not match reference annotation')
    if row.get('WHERE', {}).get('GT') != serialize_xyd_record(query, token):
        raise ValueError(f'{query.record_id}: WHERE.GT does not match reference annotation')


def semantic_failure_record(row, query, reason, *, gold_why_groups=False):
    return {'query_id': query.record_id, 'split': 'test', 'K': 1, 'k': 1, 'draw_id': 0,
        'dataset': query.dataset, 'condition': query.condition, 'subject': query.subject,
        'stimulus_id': query.stimulus_id, 'task': query.task, 'support_ids': row['support_ids'],
        'source_file': row.get('_source_file'), 'scanpath_source': 'predicted_file',
        'semantic_prompt_mode': 'gold_why_groups' if gold_why_groups else 'no_gold_why_groups',
        'WHERE': row.get('WHERE', {'GT': None, 'PRED': None}),
        'SEMANTIC': {'GT': build_flat_target(query.semantic), 'PRED': ''},
        'semantic_generation': {'state_source': 'precomputed_PRED_XYD',
                                'gold_groups_supplied': False, 'gold_groups_requested': gold_why_groups,
                                'why_group_source': 'not_applied_skipped', 'flat_format_valid': False,
                                'what': {}, 'why': {}, 'how': None,
                                'errors': [reason], 'skipped_reason': reason}}


def replay_generate_semantic(bundle, episode, predicted, budget, *, gold_why_groups=False):
    """Causal WHERE replay with predicted query tokens, then semantic generation."""
    import torch
    from semgaze.where.generation import left_pad_generation
    query = episode.query
    if budget <= 0:
        raise ValueError('positive semantic generation budget required')
    token = active_end_fix_token(bundle.config)
    shadow = shadow_query_for_group_mode(query, predicted, gold_why_groups=gold_why_groups)
    assert_replay_identity(shadow, predicted, token)
    replay_episode = FlatEpisode(episode.supports, shadow, draw_id=0)
    mode = bundle.config['semantic']['conditioning']
    if mode == 'shared_states':
        # Teacher-forcing is applied ONLY to the precomputed PRED tokens.
        where = forward_where_batch(bundle, [replay_episode], compute_loss=False,
                                    use_cache=False)
        if len(where.states[0]) != len(predicted):
            raise RuntimeError('replayed WHERE state count != predicted fixation count')
        states = bundle.projector(where.states[0])
    elif mode == 'explicit_scanpath_text':
        states = None
    else:
        raise ValueError(f'unsupported semantic conditioning: {mode}')
    prompt = prediction_only_prompt(shadow, mode, gold_why_groups=gold_why_groups)
    native = collate_native(bundle.processor, [image_user(prompt.text)],
                            [shadow.image_path], None, prompt=prompt)
    inputs = to_model_device(native.inputs, bundle.model)
    inserted = 0
    if mode == 'shared_states':
        embeds = bundle.model.get_input_embeddings()(inputs['input_ids'])
        fused, positions = insert_states(embeds, inputs['attention_mask'],
                                         inputs['labels'], states, native.boundaries)
        fused['pixel_values'] = inputs['pixel_values']
        inputs = fused
        inserted = len(positions)
    if inputs['attention_mask'].shape[1] + budget > bundle.context_limit:
        raise InvalidPrediction('semantic_context_overflow')
    inputs = to_model_device(left_pad_generation(inputs, bundle.processor.tokenizer.pad_token_id), bundle.model)
    outputs = bundle.model.generate(**inputs, max_new_tokens=budget, do_sample=False,
                                    eos_token_id=bundle.processor.tokenizer.eos_token_id,
                                    pad_token_id=bundle.processor.tokenizer.pad_token_id,
                                    use_cache=True)
    length = inputs['input_ids'].shape[1] if mode == 'explicit_scanpath_text' else 0
    if mode == 'explicit_scanpath_text' and not torch.equal(outputs[0, :length], inputs['input_ids'][0]):
        raise RuntimeError('semantic generation lost explicit input prefix')
    raw = decode_response(bundle.processor.tokenizer, outputs[0, length:])
    if inserted != (len(predicted) if mode == 'shared_states' else 0):
        raise RuntimeError('semantic state insertion did not reflect predicted N')
    parsed = parse_flat_output(raw, len(predicted), len(shadow.semantic.why_groups))
    parsed.update({'text': raw, 'state_source': 'replayed_precomputed_PRED_XYD',
                   'gold_groups_supplied': gold_why_groups,
                   'gold_groups_requested': gold_why_groups,
                   'why_group_source': ('GT_membership_only' if gold_why_groups else 'predicted_whole_trajectory'),
                   'why_group_count': len(shadow.semantic.why_groups),
                   'semantic_prompt_mode': 'gold_why_groups' if gold_why_groups else 'no_gold_why_groups',
                   'inserted_state_count': inserted, 'max_new_tokens': budget,
                   'predicted_fixation_count': len(predicted)})
    return parsed


def replay_generate_semantic_batch(bundle, items, budget, *, gold_why_groups=False):
    """Batched PRED-only WHERE replay and batched semantic generation.

    ``items`` is an ordered list of (original_episode, predicted_XYD). The
    returned list retains that order, including sample-local InvalidPrediction
    and WhereContextOverflowError values. No WHERE generation is invoked.
    """
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from semgaze.state.insertion import insert_states_batch
    from semgaze.where.collator import collate_where, pack_where_batch
    from semgaze.where.generation import left_pad_generation

    if not items:
        return []
    if type(budget) is not int or budget <= 0:
        raise ValueError('positive semantic generation budget required')
    if any(len(episode.supports) != 1 for episode, _ in items):
        raise ValueError('predicted semantic batch is K=1 only')
    mode = bundle.config['semantic']['conditioning']
    if mode not in ('shared_states', 'explicit_scanpath_text'):
        raise ValueError(f'unsupported semantic conditioning: {mode}')
    token = active_end_fix_token(bundle.config)
    tokenizer = bundle.processor.tokenizer
    results = [None] * len(items)
    work = []
    # A bad prediction/overlong prompt must not invalidate other queries in
    # the same physical batch; all errors are recorded at their input index.
    for index, (episode, predicted) in enumerate(items):
        try:
            shadow = shadow_query_for_group_mode(
                episode.query, predicted, gold_why_groups=gold_why_groups)
            assert_replay_identity(shadow, predicted, token)
            replay_episode = FlatEpisode(episode.supports, shadow, draw_id=0)
            prompt = prediction_only_prompt(shadow, mode, gold_why_groups=gold_why_groups)
            native = collate_native(bundle.processor, [image_user(prompt.text)],
                                    [shadow.image_path], None, prompt=prompt)
            # Include one inserted continuous state per PRED fixation.
            semantic_length = native.inputs['attention_mask'].shape[1]
            state_count = len(predicted) if mode == 'shared_states' else 0
            if semantic_length + state_count + budget > bundle.context_limit:
                raise InvalidPrediction('semantic_context_overflow')
            where_sample = (collate_where(bundle.processor, replay_episode,
                            bundle.end_fix_id, bundle.context_limit,
                            teacher_forcing=True, config=bundle.config)
                            if mode == 'shared_states' else None)
            work.append((index, replay_episode, predicted, native, where_sample))
        except (InvalidPrediction, WhereContextOverflowError) as exc:
            results[index] = exc
    if not work:
        return results

    # Readout positions are computed from *PRED* query assistant tokens. Each
    # sample can have a different number of PRED fixations and WHERE length.
    if mode == 'shared_states':
        replay_episodes = [entry[1] for entry in work]
        samples = [entry[4] for entry in work]
        batch = pack_where_batch(bundle.processor, replay_episodes, samples)
        where = forward_where_batch(bundle, replay_episodes, batch=batch,
                                    compute_loss=False, use_cache=False)
        if len(where.states) != len(work):
            raise RuntimeError('batched WHERE replay lost predictions')
        states = []
        for entry, r in zip(work, where.states):
            if r.ndim != 2 or r.shape[0] != len(entry[2]):
                raise RuntimeError('batched WHERE state count != PRED fixation count')
            states.append(bundle.projector(r))
    else:
        states = [None] * len(work)

    # Follow the original prepare_semantic_batch assembly semantics, but with
    # the separately selected no-gold/gold prompt for each PRED trajectory.
    if mode == 'shared_states':
        embedding_module = bundle.model.get_input_embeddings()
        embeddings, masks, labels, boundaries = [], [], [], []
        for (_, _, _, native, _), state in zip(work, states):
            native_inputs = to_model_device(native.inputs, bundle.model)
            ids = native_inputs['input_ids']
            if bool((ids == bundle.end_fix_id).any()):
                raise RuntimeError('semantic prompt contains END_FIX')
            embeds = embedding_module(ids)
            embeddings.append(embeds)
            masks.append(native_inputs['attention_mask'])
            labels.append(native_inputs['labels'])
            boundaries.append(native.boundaries)
        fused, positions = insert_states_batch(embeddings, masks, labels,
                                               states, boundaries)
        inputs = {'inputs_embeds': fused['inputs_embeds'],
                  'attention_mask': fused['attention_mask'],
                  'pixel_values': torch.cat([entry[3].inputs['pixel_values'] for entry in work])}
    else:
        native_inputs = [entry[3].inputs for entry in work]
        inputs = {
            'input_ids': pad_sequence([v['input_ids'][0] for v in native_inputs],
                batch_first=True, padding_value=tokenizer.pad_token_id),
            'attention_mask': pad_sequence([v['attention_mask'][0] for v in native_inputs],
                batch_first=True, padding_value=0),
            'pixel_values': torch.cat([v['pixel_values'] for v in native_inputs]),
        }
        positions = [() for _ in work]

    # Convert right-padded physical batches to left-padded generation inputs.
    # The shared-state and explicit-text paths preserve the same conventions
    # as evaluate_flat_batch, including the output-prefix interpretation.
    inputs = to_model_device(left_pad_generation(inputs, tokenizer.pad_token_id), bundle.model)
    output = bundle.model.generate(**inputs, max_new_tokens=budget, do_sample=False,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        use_cache=True)
    if output.ndim != 2 or output.shape[0] != len(work):
        raise RuntimeError('semantic generation returned an invalid physical batch')
    prefix_length = inputs['input_ids'].shape[1] if mode == 'explicit_scanpath_text' else 0
    if mode == 'explicit_scanpath_text':
        if output.shape[1] <= prefix_length or not torch.equal(
                output[:, :prefix_length], inputs['input_ids']):
            raise RuntimeError('batched semantic generation lost explicit input prefixes')
    if output.shape[1] - prefix_length > budget:
        raise RuntimeError('batched semantic generation exceeded new-token budget')
    for row_index, (index, _, predicted, native, _) in enumerate(work):
        inserted = len(positions[row_index])
        if inserted != (len(predicted) if mode == 'shared_states' else 0):
            raise RuntimeError('batched semantic state insertion != predicted fixation count')
        raw = decode_response(tokenizer, output[row_index, prefix_length:])
        shadow = work[row_index][1].query
        parsed = parse_flat_output(raw, len(predicted), len(shadow.semantic.why_groups))
        parsed.update({'text': raw, 'state_source': 'replayed_precomputed_PRED_XYD',
                       'gold_groups_supplied': gold_why_groups,
                       'gold_groups_requested': gold_why_groups,
                       'why_group_source': ('GT_membership_only' if gold_why_groups
                                            else 'predicted_whole_trajectory'),
                       'why_group_count': len(shadow.semantic.why_groups),
                       'semantic_prompt_mode': ('gold_why_groups' if gold_why_groups
                                                else 'no_gold_why_groups'),
                       'inserted_state_count': inserted, 'max_new_tokens': budget,
                       'predicted_fixation_count': len(predicted)})
        results[index] = parsed
    if any(r is None for r in results):
        raise RuntimeError('batched semantic results did not cover every input')
    return results


def _input_universe(raw, adapter, train, manifest, scope):
    key = 'test_seen' if scope == 'seen' else 'test'
    queries = test_queries(train, [adapter(row) for row in raw[key]], manifest, scope=scope)
    return {q.record_id: q for q in queries}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='evaluation/runtime/test override, as in evaluate_flat.py')
    parser.add_argument('--where-predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--subject-scope', choices=('auto', 'seen', 'unseen'), default='auto')
    parser.add_argument('--semantic-max-new-tokens', type=int)
    parser.add_argument('--batch-size', type=int,
                        help='Maximum K=1 episodes per WHERE replay and semantic generation batch (default: 1)')
    parser.add_argument('--branch-metrics', action='store_true')
    parser.add_argument('--gold-why-groups', choices=('on', 'off'),
                        help='Use GT WHY-group membership (on) or a dedicated whole-trajectory WHY prompt (off; default)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    destination = args.output_dir.expanduser().resolve()
    if destination == checkpoint or destination in checkpoint.parents or checkpoint in destination.parents:
        parser.error('output directory must be outside checkpoint tree')
    if destination.exists() and any(destination.iterdir()):
        parser.error('output directory must be empty')
    config = _load_evaluation_config(checkpoint, args.config)
    group_settings = config['evaluation'].get('predicted_semantic', {})
    if not isinstance(group_settings, dict) or type(group_settings.get('gold_why_groups', False)) is not bool:
        parser.error('evaluation.predicted_semantic.gold_why_groups must be a boolean')
    gold_why_groups = (args.gold_why_groups == 'on') if args.gold_why_groups is not None else group_settings.get('gold_why_groups', False)
    config['evaluation'].setdefault('predicted_semantic', {})['gold_why_groups'] = gold_why_groups
    batch_value = group_settings.get('batch_size', 1)
    if type(batch_value) is not int or batch_value < 1:
        parser.error('evaluation.predicted_semantic.batch_size must be an integer >= 1')
    batch_size = args.batch_size if args.batch_size is not None else batch_value
    if type(batch_size) is not int or batch_size < 1:
        parser.error('--batch-size must be an integer >= 1')
    config['evaluation']['predicted_semantic']['batch_size'] = batch_size
    budget = (args.semantic_max_new_tokens if args.semantic_max_new_tokens is not None
              else config['evaluation']['predictions']['semantic_max_new_tokens'])
    if type(budget) is not int or budget <= 0:
        parser.error('semantic generation budget must be positive')
    config['evaluation']['predictions'].update(where=False, semantic=True,
                                               semantic_k_values=[1], where_k_values=[1],
                                               semantic_max_new_tokens=budget)
    config['evaluation']['semantic_use_draws'] = False
    config['evaluation']['draw'] = 1
    config['evaluation']['metrics']['where'].update(scanmatch=False, multimatch=False, sed=False)
    if args.branch_metrics:
        config['evaluation']['metrics']['semantic']['report_per_branch'] = True
    config['runtime']['output_dir'] = str(destination)
    config = resolve_config(config)
    source = args.where_predictions.expanduser().resolve()
    rows, stats = filter_k1_records(source)
    if not rows:
        parser.error('no K=1 draw=0 records in prediction input')
    data = config['data']
    raw, manifest, identity = read_persisted_splits(ROOT / data['split_root'], data_config=data)
    adapter_class = {'all': JointAdapter, 'AiR': AirAdapter,
                     'COCO-Search18': CocoSearch18Adapter}[data['dataset']]
    adapter = adapter_class(ROOT / data['images_root'], annotation_frame=data['annotation_frame'],
                            duration_field=data['duration']['source_field'])
    train = {r['record_id']: adapter(r) for r in raw['train']}
    input_ids = {r['query_id'] for r in rows}
    populations = {scope: _input_universe(raw, adapter, train, manifest, scope)
                   for scope in ('unseen', 'seen')}
    if args.subject_scope == 'auto':
        candidates = [scope for scope, by_id in populations.items() if input_ids <= set(by_id)]
        if len(candidates) != 1:
            sizes = {k: len(v) for k, v in populations.items()}
            raise ValueError(f'input query IDs must be a subset of exactly ONE test population; '
                             f'population sizes={sizes}; input={len(input_ids)}; '
                             f'overlap={ {k: len(set(v) & input_ids) for k, v in populations.items()} }')
        scope = candidates[0]
    else:
        scope = args.subject_scope
    by_id = populations[scope]
    missing, unexpected = set(by_id) - input_ids, input_ids - set(by_id)
    if unexpected:
        raise ValueError(f'{scope}: unexpected query IDs={len(unexpected)}: {sorted(unexpected)[:10]}')
    token = active_end_fix_token(config)
    if not token:
        raise ValueError('this replay mode requires checkpoint WHERE <END_FIX> token')
    episodes = {}
    for row in rows:
        query = by_id[row['query_id']]
        episode = evaluation_episode(query, train, manifest, 1, draw_id=0,
                                     unseen_subjects=manifest['unseen_subject_ids'])
        validate_alignment(row, query, episode, token, identity)
        episodes[query.record_id] = episode
    stats.update({'source_file': str(source), 'scope': scope, 'matched_queries': len(episodes),
                  'expected_test_queries': len(by_id), 'missing_query_count': len(missing),
                  'missing_query_ids': sorted(missing),
                  'manifest_identity': identity,
                  'gold_why_groups': gold_why_groups, 'batch_size': batch_size,
                  'semantic_prompt_mode': 'gold_why_groups' if gold_why_groups else 'no_gold_why_groups',
                  'dataset_groups': dict(Counter(
                      'VQA' if q.condition == 'vqa' else 'TP' if q.condition == 'present' else 'TA'
                      for q in by_id.values()))})
    if args.dry_run:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return
    destination.mkdir(parents=True, exist_ok=True)
    config['evaluation']['subject_scope'] = scope
    bundle = load_checkpoint_bundle(checkpoint, split_manifest_identity=identity,
                                    output_dir=destination, runtime=config['runtime'])
    bundle.config = config
    write_run_config(config, destination)
    (destination / 'evaluation_config.json').write_text(json.dumps({
        'checkpoint': str(checkpoint),
        'where_predictions_path': str(source),
        'scanpath_source': 'predicted_file',
        'K': 1, 'draw_id': 0, 'subject_scope': scope,
        'gold_why_groups': gold_why_groups,
        'batch_size': batch_size,
        'semantic_prompt_mode': 'gold_why_groups' if gold_why_groups else 'no_gold_why_groups',
        'semantic_max_new_tokens': budget,
        'split_manifest_identity': identity,
        'full_test_manifest_scoring': True,
    }, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    output_path = destination / 'predictions.jsonl'
    errors = Counter()
    success = 0
    # Missing frozen WHERE records count as failures, never fabricated predictions.
    if missing:
        errors['missing_where_record'] = len(missing)
    try:
        with output_path.open('w', encoding='utf-8') as stream, test_mode(bundle):
            # Group only K=1 draw=0 input rows; every physical forward receives
            # at most batch_size episodes. Output is restored to JSONL order.
            for chunk_start in range(0, len(rows), batch_size):
                chunk = rows[chunk_start:chunk_start + batch_size]
                records = [None] * len(chunk)
                valid = []
                for local_index, row in enumerate(chunk):
                    query = by_id[row['query_id']]
                    episode = episodes[query.record_id]
                    row['_source_file'] = source.name
                    try:
                        fixed, details = parse_stored_prediction(row, token)
                        valid.append((local_index, row, query, episode, fixed, details))
                    except InvalidPrediction as exc:
                        reason = str(exc)
                        errors[reason] += 1
                        record = semantic_failure_record(row, query, reason,
                                                         gold_why_groups=gold_why_groups)
                        record['validation'] = {}
                        records[local_index] = record
                if valid:
                    if batch_size == 1:
                        # Exact backward-compatible serial path when configured 1.
                        (_, _, _, ep, fixed, _) = valid[0]
                        try:
                            inferred = [replay_generate_semantic(
                                bundle, ep, fixed, budget, gold_why_groups=gold_why_groups)]
                        except (InvalidPrediction, WhereContextOverflowError) as exc:
                            inferred = [exc]
                    else:
                        inferred = replay_generate_semantic_batch(
                            bundle, [(entry[3], entry[4]) for entry in valid], budget,
                            gold_why_groups=gold_why_groups)
                    if len(inferred) != len(valid):
                        raise RuntimeError('semantic inference lost batched query alignment')
                    for (local_index, row, query, episode, fixed, details), sem in zip(valid, inferred):
                        if isinstance(sem, (InvalidPrediction, WhereContextOverflowError)):
                            reason = str(sem)
                            errors[reason] += 1
                            record = semantic_failure_record(row, query, reason,
                                                             gold_why_groups=gold_why_groups)
                        else:
                            record = semantic_failure_record(row, query, '',
                                                             gold_why_groups=gold_why_groups)
                            record['SEMANTIC']['PRED'] = sem['text']
                            record['semantic_generation'] = {k: v for k, v in sem.items() if k != 'text'}
                            success += 1
                        record['validation'] = details
                        records[local_index] = record
                if any(record is None for record in records):
                    raise RuntimeError('missing row in batched output restoration')
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                index = chunk_start + len(chunk)
                if index // 50 != chunk_start // 50 or index == len(rows):
                    print(f'[PRED-SEM] {index}/{len(rows)} | processed={success} | '
                          f'skipped={sum(errors.values())} | batch_size={batch_size}', flush=True)
            # Full-manifest scoring: an absent query is a documented failure
            # with NO synthetic WHERE.PRED or semantic output.
            for query_id in sorted(missing):
                query = by_id[query_id]
                episode = evaluation_episode(query, train, manifest, 1, draw_id=0,
                                             unseen_subjects=manifest['unseen_subject_ids'])
                absent = {'support_ids': [s.record_id for s in episode.supports],
                          '_source_file': source.name}
                record = semantic_failure_record(absent, query, 'missing_where_record', gold_why_groups=gold_why_groups)
                record['validation'] = {'absent_from_input_jsonl': True}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    except BaseException:
        output_path.rename(destination / 'predictions.jsonl.incomplete')
        raise
    stats['processed'] = success
    stats['invalid_or_skipped'] = sum(errors.values())
    stats['invalid_reasons'] = dict(errors)
    stats['semantic_scanpath_source'] = 'precomputed WHERE.PRED (no WHERE generation)'
    stats['semantic_why_group_source'] = ('GT membership, enabled explicitly; no GT scanpath tokens'
                                         if gold_why_groups else 'whole predicted trajectory; no GT group count or membership')
    (destination / 'validation_summary.json').write_text(json.dumps(stats, indent=2,
        ensure_ascii=False, allow_nan=False), encoding='utf-8')
    scores = None
    if config['evaluation']['metrics']['enabled'] and semantic_metrics_enabled(config['evaluation']['metrics']['semantic']):
        scores = score_prediction_artifact(output_path, queries=list(by_id.values()),
             config=config, output_dir=destination, checkpoint=checkpoint,
             split_manifest_identity=identity)
    (destination / 'evaluation_summary.json').write_text(json.dumps({
        'prediction_file': str(output_path), 'validation_summary': str(destination / 'validation_summary.json'),
        'scores': scores,
        'protocol': ('predicted_scanpath_with_GT_WHY_membership' if gold_why_groups
                     else 'predicted_scanpath_no_GT_WHY_membership'),
        'gold_why_groups': gold_why_groups,
        'batch_size': batch_size,
        'semantic_prompt_mode': 'gold_why_groups' if gold_why_groups else 'no_gold_why_groups',
        'not_directly_comparable_to_gold_group_conditioned_GT_evaluation': not gold_why_groups,
    }, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'[PRED-SEM] complete | processed={success} skipped={sum(errors.values())} | {destination}', flush=True)


if __name__ == '__main__':
    main()
