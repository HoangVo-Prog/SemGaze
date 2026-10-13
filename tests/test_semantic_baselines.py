"""CPU-friendly contracts; native model forward/8B greedy decoding need CUDA smoke."""
from dataclasses import replace
from types import SimpleNamespace
import json

import pytest
import torch
from semgaze.data.schema import NormalizedRecord
from semgaze.data.semantic import WhyGroup, NormalizedSemantic
from semgaze.semantic.baselines.prompt import (
    build_baseline_prompt, semantic_target, CONCISE_INSTRUCTION,
)
from semgaze.semantic.flat.parser import parse_flat_output
from semgaze.semantic.flat.prompt import INSTRUCTION
from semgaze.semantic.baselines.native import pack_natives, decode_generated
from semgaze.semantic.baselines.scoring import verify_complete, subset_name
from semgaze.evaluation.metrics_semantic import score_prediction_semantics
from semgaze.where.coordinates import coordinate_bin
from semgaze.where.duration import duration_bin


def query():
    return NormalizedRecord(
        record_id='coco_semgaze::7::bottle::fixture.jpg', stimulus_id='fixture.jpg',
        subject=7, image_path='data/images/fixture.jpg', image_width=100, image_height=100,
        task='Find the bottle', condition='present',
        x_px=(42.49, -3, 999), y_px=(30.5, 200, -40),
        duration_ms=(179.5, -4, 1000.5), dataset='COCO-Search18',
        semantic=NormalizedSemantic(
            what=('SECRET_WHAT_A', 'SECRET_WHAT_B', 'SECRET_WHAT_C'),
            why_groups=(WhyGroup((1, 2), 'SECRET_WHY_A'),
                        WhyGroup((3,), 'SECRET_WHY_B')),
            how='SECRET_HOW'))


def test_prompt_parity_and_no_reference_leakage():
    q = query()
    orig = build_baseline_prompt(q, 'origin')
    short = build_baseline_prompt(q, 'concise')
    assert orig.fixation_values == short.fixation_values
    assert orig.group_members == short.group_members == ((1, 2), (3,))
    assert orig.text.split('\n\nDescribe the visual exploration')[0] == short.text.split('\n\nGiven the image')[0]
    for prompt in (orig.text, short.text):
        assert q.task in prompt
        assert 'Fixation count: 3' in prompt
        assert 'Gold WHY group count: 2' in prompt
        assert prompt.count('Fixation 1: X=') == 1
        assert prompt.count('Group 1: fixations 1, 2') == 1
        assert 'X=42, Y=30, D=180' in prompt
        assert 'X=00, Y=99, D=000' in prompt
        assert 'X=99, Y=00, D=999' in prompt
        assert all(v not in prompt for v in ('SECRET_WHAT_A', 'SECRET_WHAT_B', 'SECRET_WHAT_C',
                                              'SECRET_WHY_A', 'SECRET_WHY_B', 'SECRET_HOW'))
    assert INSTRUCTION.format(N=3, M=2) in orig.text
    assert CONCISE_INSTRUCTION.format(N=3, M=2) in short.text
    assert orig.fixation_values == tuple(zip(
        (coordinate_bin(x, 100) for x in q.x_px),
        (coordinate_bin(y, 100) for y in q.y_px),
        (duration_bin(d) for d in q.duration_ms)))
    assert semantic_target(q) == ('WHAT 1: SECRET_WHAT_A\nWHAT 2: SECRET_WHAT_B\n'
                                  'WHAT 3: SECRET_WHAT_C\nWHY 1: SECRET_WHY_A\n'
                                  'WHY 2: SECRET_WHY_B\nHOW: SECRET_HOW')


def test_invalid_style_and_format_are_rejected_not_repaired():
    with pytest.raises(ValueError):
        build_baseline_prompt(query(), 'full')
    target = semantic_target(query())
    assert parse_flat_output(target, 3, 2)['flat_format_valid']
    for broken in ('WHAT 1: a\nWHY 1: b\nHOW: c',
                   target.replace('WHAT 2:', 'WHAT 1:'),
                   target + '\nExtra notes: more'):
        item = parse_flat_output(broken, 3, 2)
        assert not item['flat_format_valid'] and item['errors']


def test_rounding_edges_reuse_canonical_where_codec():
    assert coordinate_bin(0.5, 100) == 0  # two-stage rounding and Python half-to-even
    assert coordinate_bin(1.5, 100) == 2
    assert duration_bin(0.5) == 0
    assert duration_bin(1.5) == 2
    assert duration_bin(999.5) == 999


def test_pack_native_masks_labels_and_preserves_single_tile_per_query():
    def sample(token, length):
        ids = torch.tensor([[token] * length])
        labels = torch.full_like(ids, -100)
        labels[0, -2:] = token
        return SimpleNamespace(inputs={'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                                       'labels': labels, 'pixel_values': torch.zeros(1, 3, 448, 448)})
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    result = pack_natives(processor, [sample(5, 4), sample(6, 6)], training=True)
    assert result['input_ids'].shape == (2, 6)
    assert result['pixel_values'].shape == (2, 3, 448, 448)
    assert (result['labels'][0, 4:] == -100).all()
    assert int((result['labels'] != -100).sum()) == 4
    left = pack_natives(processor, [sample(5, 4), sample(6, 6)], left_pad=True)
    assert left['input_ids'][0].tolist() == [0, 0, 5, 5, 5, 5]


def test_generated_prefix_split_with_budget_larger_than_prompt():
    class T:
        eos_token_id = 99
        def decode(self, ids, **kwargs):
            return ','.join(map(str, ids))
    prefix = torch.tensor([[11, 12, 13]])
    full = torch.tensor([[11, 12, 13, 22, 23]])
    only = torch.tensor([[22, 23]])
    assert decode_generated(T(), full, prefix, max_new_tokens=512) == ['22,23']
    assert decode_generated(T(), only, prefix, max_new_tokens=512) == ['22,23']


def baseline_row(q, text):
    return {'query_id': q.record_id, 'subject': q.subject, 'dataset': q.dataset,
            'stimulus_id': q.stimulus_id, 'task': q.task, 'condition': q.condition,
            'method': 'semantic_zero_shot', 'prompt_style': 'origin',
            'gt_fixation_count': len(q.x_px),
            'gt_group_count': len(q.semantic.why_groups), 'text': text,
            'semantic_generation': json.loads(json.dumps(parse_flat_output(text, 3, 2)))}


def test_duplicate_missing_mismatch_fail_closed():
    q = query()
    row = baseline_row(q, semantic_target(q))
    assert verify_complete([row], [q], method='semantic_zero_shot') == [row]
    with pytest.raises(ValueError, match='duplicate'):
        verify_complete([row, row], [q], method='semantic_zero_shot')
    with pytest.raises(ValueError, match='incomplete'):
        verify_complete([], [q], method='semantic_zero_shot')
    with pytest.raises(ValueError, match='mismatch'):
        verify_complete([{**row, 'task': 'incorrect'}], [q], method='semantic_zero_shot')
    assert subset_name(q) == 'TP'
    assert subset_name(replace(q, condition='absent')) == 'TA'


def test_canonical_scoring_counts_all_missing_semantic_units():
    q = query()
    bad = baseline_row(q, 'WHAT 1: visible')
    output = score_prediction_semantics(
        [dict(bad, K=0, draw_id=0)], {(q.record_id, 0, 0): q.semantic},
        bleu4=True, rouge_l=True)
    assert output['semantic_parse_failure_count'] == 1
    assert output['semantic_what_unit_count'] == 3
    assert output['semantic_what_missing_count'] == 2
    assert output['semantic_why_missing_count'] == 2
    assert output['semantic_how_missing_count'] == 1
    assert output['eval_sem_what_bleu4'] >= 0
    assert output['eval_sem_what_rouge_l'] >= 0


def test_semgaze_selected_draw_enforces_complete_population():
    q = query()
    original = baseline_row(q, semantic_target(q))
    selected = {'query_id': q.record_id, 'subject': q.subject, 'dataset': q.dataset,
                'stimulus_id': q.stimulus_id, 'task': q.task, 'split': 'test', 'K': 1,
                'draw_id': 0, 'annotation_width': q.image_width,
                'annotation_height': q.image_height,
                'gt_raw_pixels': [[x, y] for x, y in zip(q.x_px, q.y_px)],
                'gt_duration_ms_raw': list(q.duration_ms),
                'SEMANTIC': {'GT': semantic_target(q), 'PRED': original['text']},
                'semantic_generation': original['semantic_generation']}
    assert verify_complete([selected], [q], method='SemGaze', reference_k=1,
                           reference_draw=0) == [selected]
    with pytest.raises(ValueError, match='incomplete'):
        verify_complete([selected], [q], method='SemGaze', reference_k=5, reference_draw=0)
    with pytest.raises(ValueError, match='duplicate'):
        verify_complete([selected, selected], [q], method='SemGaze', reference_k=1,
                        reference_draw=0)


def test_offline_end_to_end_scoring_artifacts(tmp_path):
    from semgaze.semantic.baselines.scoring import scoring_report
    q = query()
    row = baseline_row(q, semantic_target(q))
    rows = verify_complete([row], [q], method='semantic_zero_shot')
    result = scoring_report(rows, [q], {'bleu4': True, 'rouge_l': True,
                                         'cider_r': False, 'meteor': False},
                            tmp_path, method='semantic_zero_shot')
    assert result['TP']['query_count'] == 1
    assert result['COCO-Search18']['query_count'] == 1
    for name in ('metrics.json', 'metrics_by_subset.json', 'validity_summary.json'):
        assert (tmp_path / name).is_file()
    summary = json.loads((tmp_path / 'validity_summary.json').read_text())
    assert summary['valid'] == 1 and summary['invalid'] == 0
