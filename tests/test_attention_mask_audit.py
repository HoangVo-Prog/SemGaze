from types import SimpleNamespace

import pytest
import torch

from scripts import audit_attention_mask as audit


def test_causal_only_safety_requires_strict_right_padding():
    batch = SimpleNamespace(
        inputs={
            'attention_mask': torch.tensor([[1, 1, 1], [1, 0, 1]]),
            'labels': torch.tensor([[-100, 7, -100], [-100, -100, 7]]),
            'input_ids': torch.tensor([[3, 7, 0], [3, 0, 7]]),
        },
        metadata=(
            {'where_length': 3, 'fixation_count': 1, 'response_start': 1, 'response_length': 1},
            {'where_length': 2, 'fixation_count': 1, 'response_start': 1, 'response_length': 1},
        ),
    )
    with pytest.raises(AssertionError, match='strict right-padding'):
        audit.assert_where_causal_only_safe(batch, pad_token_id=0, end_fix_id=7)
    batch.inputs['attention_mask'] = torch.tensor([[1, 1, 1], [1, 1, 0]])
    batch.inputs['input_ids'] = torch.tensor([[3, 7, 0], [3, 7, 0]])
    batch.inputs['labels'] = torch.tensor([[-100, 7, -100], [-100, 7, -100]])
    result = audit.assert_where_causal_only_safe(batch, pad_token_id=0, end_fix_id=7)
    assert result['strict_right_padding']
    assert result['prefix_lengths'] == [3, 2]
    assert result['padding_tokens'] == 1


def test_verdict_requires_parity_and_relevant_profiler_evidence():
    parity = {'passed': True}
    speed = {'where_forward': 1.2, 'backward': 1.0, 'total_step': 2.0}
    no_ops = {'baseline': {'kernel_evidence': {}}, 'causal_only': {'kernel_evidence': {}}}
    assert audit.final_verdict({'passed': False}, speed, no_ops, 1.1) == 'INVALID'
    assert audit.final_verdict(parity, speed, no_ops, 1.1) == 'INCONCLUSIVE'
    evidence = {'scaled_dot_product_attention': {'count': 1}}
    profiler = {
        'baseline': {'kernel_evidence': evidence},
        'causal_only': {'kernel_evidence': evidence},
    }
    assert audit.final_verdict(parity, speed, profiler, 1.1) == 'SUPPORTED'
    assert audit.final_verdict(parity, {'where_forward': 1.0, 'backward': 1.0}, profiler, 1.1) == 'NOT SUPPORTED'
