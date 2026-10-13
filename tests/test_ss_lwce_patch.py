"""Run in SemGaze checkout: pytest -q tests/test_ss_lwce_patch.py"""
from dataclasses import dataclass
from types import SimpleNamespace
import torch
import pytest

from semgaze.model.selected_loss import compute_selected_causal_nll
from semgaze.where.serialization import serialize_xyd
from semgaze.where.weighting import fixation_spans, mask_replaced_fixation


def test_shifted_weighted_ce_and_disabled_parity():
    torch.manual_seed(7)
    hidden = torch.randn(2, 6, 5, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3, 4, 1], [-100, -100, -100, 2, 3, -100]])
    head = torch.nn.Linear(5, 6)
    original = compute_selected_causal_nll(hidden, labels, head)
    w = torch.ones_like(labels, dtype=torch.float32)
    identical = compute_selected_causal_nll(hidden, labels, head, target_weights=w)
    torch.testing.assert_close(original.episode_losses, identical.episode_losses)
    w[0, 4] = 1.5
    weighted = compute_selected_causal_nll(hidden, labels, head, target_weights=w)
    assert torch.isfinite(weighted.episode_losses).all()
    torch.testing.assert_close(weighted.unweighted_episode_losses, original.episode_losses)
    weighted.loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()


def test_mask_only_replaced_fixation_with_straddling_token():
    response = serialize_xyd([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
    spans = fixation_spans(response)
    assert len(spans) == 3
    # Synthetic token offsets: first, second, boundary-straddling, third, EOS.
    offsets = [(spans[0][0], spans[0][1]), (spans[1][0], spans[1][1]),
               (spans[1][1] - 1, spans[2][0] + 1),
               (spans[2][0] + 1, spans[2][1]), (len(response), len(response) + 1)]
    native = SimpleNamespace(
        inputs={'labels': torch.tensor([[-100, 4, 5, 6, 7, 8]])},
        response_offsets=offsets,
    )
    masked = mask_replaced_fixation(native, response, 2)
    assert masked == 2
    assert native.inputs['labels'][0].tolist() == [-100, 4, -100, -100, 7, 8]


def test_missing_end_fix_fails_closed():
    with pytest.raises(ValueError):
        fixation_spans('[(01, 02, 003)]')


def test_replacement_range_validated():
    response = serialize_xyd([(1, 2, 3)])
    native = SimpleNamespace(inputs={'labels': torch.tensor([[4]])}, response_offsets=[(0, len(response))])
    with pytest.raises(ValueError):
        mask_replaced_fixation(native, response, 2)


def test_lwce_spans_and_neutral_eos():
    from semgaze.where.weighting import build_fixation_target_weights
    record = SimpleNamespace(x_px=(1, 4, 7), y_px=(2, 5, 8), duration_ms=(3, 6, 9),
                             image_width=100, image_height=100)
    episode = SimpleNamespace(query=record)
    response = serialize_xyd([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
    spans = fixation_spans(response)
    offsets = [*spans, (len(response), len(response) + 1)]
    native = SimpleNamespace(
        inputs={'labels': torch.tensor([[-100, 4, 5, 6, 7]])},
        response_offsets=offsets,
    )
    batch = SimpleNamespace(inputs={'labels': native.inputs['labels']}, samples=[native])
    late = build_fixation_target_weights(batch, [episode],
        config={'enabled': True, 'mode': 'late_only', 'alpha': 0.5, 'threshold': 2})
    torch.testing.assert_close(late, torch.tensor([[1., 1., 1., 1.5, 1.]]))
    linear = build_fixation_target_weights(batch, [episode],
        config={'enabled': True, 'mode': 'linear', 'alpha': 0.5, 'threshold': 2})
    torch.testing.assert_close(linear, torch.tensor([[1., 1., 1.25, 1.5, 1.]]))
    assert build_fixation_target_weights(batch, [episode], config={'enabled': False}) is None


def test_ss_masked_offsets_remain_aligned_for_lwce():
    from semgaze.where.weighting import _supervised_offsets
    response = serialize_xyd([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
    spans = fixation_spans(response)
    # Native assistant response positions are contiguous, including predicted tuple.
    native = SimpleNamespace(
        inputs={'labels': torch.tensor([[-100, 10, 11, 12, 13, 14]])},
        response_start=1,
        response_offsets=[spans[0], spans[1], spans[2], (len(response), len(response)+1)],
    )
    # One extra supervised token beyond declared offsets is invalid; correct fixture.
    native.inputs['labels'] = torch.tensor([[-100, 10, 11, 12, 13]])
    assert mask_replaced_fixation(native, response, 2) == 1
    remaining = list(_supervised_offsets(native))
    assert [p for p, _ in remaining] == [1, 3, 4]
    rec = SimpleNamespace(x_px=(1, 4, 7), y_px=(2, 5, 8), duration_ms=(3, 6, 9),
                          image_width=100, image_height=100)
    batch = SimpleNamespace(inputs={'labels': native.inputs['labels']}, samples=[native])
    from semgaze.where.weighting import build_fixation_target_weights
    weights = build_fixation_target_weights(batch, [SimpleNamespace(query=rec)],
          config={'enabled': True, 'alpha': 0.5, 'mode': 'late_only', 'threshold': 2})
    assert weights[0, 2] == 1  # masked predicted fixation is neutral
    assert weights[0, 3] == 1.5  # last GT fixation retains late weight
