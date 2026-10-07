import math

import pytest

from semgaze.evaluation.metrics_probability import (
    aggregate_probability_draw, aggregate_probability_k, digit_logprob,
    score_gt_transition, score_probability_query,
)
from semgaze.evaluation.metrics_scanpath import CoordinateAdapter, CoordinateProtocolError, _levenshtein, _sed_symbols
from semgaze.evaluation.metrics_semantic import (collect_semantic_units, normalize_text,
                                                 score_bertscore_branch, score_cider_r_branch,
                                                 semantic_units_from_prediction)
from semgaze.evaluation.records import (assert_key_sets_equal, expected_episode_keys,
                                         resolve_evaluation_draw_counts)


def test_historical_prediction_inverse_and_sed_boundaries():
    with pytest.raises(CoordinateProtocolError):
        CoordinateAdapter()
    adapter = CoordinateAdapter(prediction_inverse="deepgaze_historical_round")
    assert adapter.prediction_bins([0, 99], [0, 99]) == [(0, 0), (507, 317)]
    assert _sed_symbols([(0, 0, 10), (101, 63, 10), (102, 64, 10), (511, 319, 10)]) == "aagz"
    assert _levenshtein("abc", "ac") == 1


def test_digit_normalization_and_transition():
    assert digit_logprob([0.0] * 10, 0, list(range(10))) == pytest.approx(-math.log(10))
    scored = score_gt_transition(9, 10, [[0.0] * 10] * 4, list(range(10)))
    assert scored["ll_fix"] == pytest.approx(-4 * math.log(10))
    assert score_probability_query([])["defined"] is False


def test_probability_undefined_draws_are_null_semantics():
    undefined = aggregate_probability_draw([score_probability_query([])])
    assert undefined["eval_where_ll"] is None
    assert undefined["probability_undefined_query_count"] == 1
    assert aggregate_probability_k([undefined])["eval_where_ig"] is None


def test_semantic_units_are_unique_and_whitespace_only_normalized():
    assert normalize_text("  a\n b  ") == "a b"
    units = collect_semantic_units([{"query_id": "q", "semantic": {
        "what": {"gt": ["one", "two"], "pred": ["1", "2"]},
        "why": {"gt": ["reason"], "pred": [None]},
        "how": {"gt": "summary", "pred": "answer"},
    }}], "what")
    assert [u["unit_id"] for u in units] == ["q::WHAT::0", "q::WHAT::1"]


def test_identity_join_is_key_based():
    left = [{"query_id": "q", "K": 1, "draw_id": 0}]
    right = [{"record_id": "q", "k": 1, "draw_id": 0}]
    assert assert_key_sets_equal(left, right) == {("q", 1, 0)}
    with pytest.raises(ValueError):
        assert_key_sets_equal(left, [{"query_id": "other", "K": 1, "draw_id": 0}])
    assert expected_episode_keys(["q"], [1], {"1": ["draw0", "draw1"]}) == {
        ("q", 1, 0), ("q", 1, 1)}


def test_configured_evaluation_draw_uses_prefix_of_persisted_draws():
    manifest = {'support_draws': {'1': [['draw0'], ['draw1'], ['draw2']]}}
    assert resolve_evaluation_draw_counts(2, manifest, [1]) == {'1': 2}


def test_semantic_prediction_parser_accepts_integer_and_json_string_keys():
    query = {
        "what": ("look", "move"),
        "why_groups": ({"why": "because"},),
        "how": "search",
    }
    row = {"query_id": "q", "semantic_generation": {
        "what": {1: "look", "2": "move"},
        "why": {"1": "because"},
        "how": "search",
    }}
    units = semantic_units_from_prediction(row, query)
    assert [u["unit_id"] for u in units["what"]] == ["q::WHAT::0", "q::WHAT::1"]
    assert units["why"][0]["candidate"] == "because"
    assert units["how"][0]["candidate"] == "search"


class _FakeBertScorer:
    def __init__(self):
        self.calls = []

    def score(self, candidates, references):
        self.calls.append((list(candidates), list(references)))
        # A deterministic stand-in for the tensor returned by bert-score.
        class Values:
            def __init__(self, values): self.values = values
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return self.values
        return None, None, Values([1.0 if c == r else 0.5 for c, r in zip(candidates, references)])


def test_bertscore_batch_size_invariance_and_missing_zero():
    units = [
        {"unit_id": "q::WHAT::0", "reference": "a", "candidate": "a", "missing": False},
        {"unit_id": "q::WHAT::1", "reference": "b", "candidate": None, "missing": True},
        {"unit_id": "q::WHAT::2", "reference": "c", "candidate": "x", "missing": False},
    ]
    one = score_bertscore_branch(units, _FakeBertScorer(), batch_size=1)
    many = score_bertscore_branch(units, _FakeBertScorer(), batch_size=64)
    assert one == many
    assert one["f1"] == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert one["missing_count"] == 1


class _FakeTokenizer:
    def tokenize(self, captions):
        return {key: [value[0]["caption"]] for key, value in captions.items()}


class _FakeCider:
    def compute_score(self, references, candidates):
        assert set(references) == set(candidates)
        return 0.5, [1.0, 0.0]


def test_cider_r_keeps_missing_reference_unit_in_corpus():
    units = [
        {"unit_id": "q::WHAT::0", "reference": "a", "candidate": "a", "missing": False},
        {"unit_id": "q::WHAT::1", "reference": "b", "candidate": None, "missing": True},
    ]
    result = score_cider_r_branch(units, _FakeCider(), _FakeTokenizer())
    assert result["score"] == pytest.approx(0.5)
    assert result["missing_count"] == 1
