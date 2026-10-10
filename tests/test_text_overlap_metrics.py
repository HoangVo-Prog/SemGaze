"""Reference-parity primitives, missing-unit policy, and frozen-metric integration."""
from types import SimpleNamespace
import json
import math

import pytest

from semgaze.evaluation.metrics_text_overlap import (
    score_bleu4_corpus, score_rouge_l_corpus, Meteor15,
)
from semgaze.evaluation.metrics_semantic import score_prediction_semantics


def _unit(reference, candidate, index=0):
    return dict(unit_id=f"q::WHAT::{index}", reference=reference,
                candidate=candidate, missing=candidate is None)


def test_rouge_l_llada_beta_and_missing():
    assert score_rouge_l_corpus([_unit("alpha beta", "alpha beta")]) == pytest.approx(1)
    assert score_rouge_l_corpus([_unit("alpha beta", "alpha")]) == pytest.approx(
        (1 + 1.2 ** 2) * 1 * 0.5 / (0.5 + 1.2 ** 2))
    assert score_rouge_l_corpus([_unit("alpha beta", "alpha beta"),
                                 _unit("alpha beta", None, 1)]) == pytest.approx(0.5)


def test_corpus_bleu4_not_average_of_sentence_bleu():
    exact = _unit("a b c d", "a b c d")
    missing = _unit("w x y z", None, 1)
    assert score_bleu4_corpus([exact]) == pytest.approx(1, abs=1e-8)
    # One exact candidate, one missing candidate. Global BP = exp(1 - 8/4).
    assert score_bleu4_corpus([exact, missing]) == pytest.approx(math.exp(-1), abs=1e-8)
    assert score_bleu4_corpus([missing]) == 0.0
    assert score_bleu4_corpus([]) is None


class _MeteorStub:
    def __init__(self):
        self.called_with = []

    def score_units(self, units):
        self.called_with.append(list(units))
        return 0.25


def test_scoring_bleu_rouge_and_meteor_on_each_flat_branch():
    query_semantic = {
        "what": ("a b c d", "w x y z"),
        "why_groups": ({"why": "a b c d"},),
        "how": "a b c d",
    }
    row = {"query_id": "q", "K": 1, "draw_id": 0,
           "semantic_generation": {"flat_format_valid": False,
                                   "what": {"1": "a b c d"},
                                   "why": {"1": "a b c d"}, "how": "a b c d"}}
    meteor = _MeteorStub()
    result = score_prediction_semantics([row], {("q", 1, 0): query_semantic},
                                        bleu4=True, rouge_l=True, meteor=meteor)
    assert result["eval_sem_what_bleu4"] == pytest.approx(math.exp(-1), abs=1e-8)
    assert result["eval_sem_what_rouge_l"] == pytest.approx(0.5)
    assert result["eval_sem_what_rouge"] == result["eval_sem_what_rouge_l"]
    assert result["eval_sem_why_bleu4"] == pytest.approx(1, abs=1e-8)
    assert result["eval_sem_how_meteor"] == pytest.approx(0.25)
    assert result["semantic_what_missing_count"] == 1
    assert result["semantic_parse_failure_count"] == 1
    assert [len(x) for x in meteor.called_with] == [2, 1, 1]


def test_meteor_sanitization():
    assert Meteor15._sanitize(' hello\n|||' + ' world  ') == 'hello world'


def test_training_and_standalone_shared_frozen_prediction_scorer(tmp_path):
    pytest.importorskip("torch")
    from semgaze.evaluation.predictions import score_prediction_artifact
    semantic = {"what": ("a b c d",), "why_groups": ({"why": "a b c d"},),
                "how": "a b c d"}
    query = SimpleNamespace(record_id="q", subject="7", semantic=semantic)
    path = tmp_path / "test.jsonl"
    with path.open("w") as stream:
        for draw in (0, 1):
            row = {"split": "test", "query_id": "q", "K": 1, "draw_id": draw,
                   "subject": "7", "semantic_generation": {
                       "what": {"1": "a b c d"}, "why": {"1": "a b c d"},
                       "how": "a b c d", "flat_format_valid": True}}
            stream.write(json.dumps(row) + "\n")
    config = {"data": {"dataset": "COCO-Search18"}, "evaluation": {
        "draw": 2, "semantic_use_draws": True,
        "predictions": {"where": False, "semantic": True},
        "metrics": {"enabled": True, "where": {}, "semantic": {
            "bleu4": {"enabled": True}, "rouge_l": {"enabled": True},
            "meteor": {"enabled": False}, "bertscore": {"enabled": False},
            "cider_r": {"enabled": False}}}}}
    artifact = score_prediction_artifact(
        path, queries=[query], config=config, output_dir=tmp_path,
        checkpoint="test-checkpoint")
    scores = artifact["by_k"]["1"]
    assert len(scores["draws"]) == 2
    assert scores["draw_mean"]["eval_sem_what_bleu4"] == pytest.approx(1, abs=1e-8)
    assert scores["draw_mean"]["eval_sem_what_rouge_l"] == 1
    assert scores["semantic_draw_count"] == 2
    assert artifact["metric_provenance"]["bleu4"]["aggregation"] == "corpus"
