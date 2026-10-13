"""Lightweight SS policy tests; require installed transformers dependency."""
import pytest
pytest.importorskip('transformers')
from semgaze.where.ss import _episode_rng, _parse_generated_fixation, sampling_probability


def test_ss_deterministic_policy():
    a = _episode_rng(7, 13, 'image123')
    b = _episode_rng(7, 13, 'image123')
    assert [a.random() for _ in range(10)] == [b.random() for _ in range(10)]
    assert sampling_probability({'probability': 0.4, 'max_probability': 0.3, 'warmup_ratio': 0.1}, 0, 100) == pytest.approx(0.03)
    assert sampling_probability({'probability': 0.4, 'max_probability': 0.3, 'warmup_ratio': 0.1}, 10, 100) == pytest.approx(0.3)


def test_ss_validated_complete_fixations_only():
    assert _parse_generated_fixation('(03, 99, 089)<END_FIX>', '<END_FIX>') == (3, 99, 89)
    assert _parse_generated_fixation('(100, 10, 100)<END_FIX>', '<END_FIX>') is None
    assert _parse_generated_fixation('(02, 02, 100)', '<END_FIX>') is None
    assert _parse_generated_fixation('(02, 02, 100)<END_FIX>, (01, 01, 010)<END_FIX>', '<END_FIX>') is None
