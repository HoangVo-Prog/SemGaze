"""Executable Gate B1/B2 parity checks against the local reference trees."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from semgaze.evaluation.metrics_scanpath import (
    COCO_RAW_HEIGHT,
    COCO_RAW_WIDTH,
    CoordinateAdapter,
    ISP_HEIGHT,
    ISP_WIDTH,
    VERIFIED_PREDICTION_INVERSE,
    CoordinateProtocolError,
    score_scanpath_pair,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "isp_coordinate_pairs.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gate_b_fixture_proves_raw_gt_scale_and_subject_mapping():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    adapter = CoordinateAdapter()
    frame = payload["transform"]
    assert (frame["reference_width"], frame["reference_height"]) == (512, 320)
    assert payload["provenance"]["gt_transform"] == (
        "x_isp = x_raw * 512 / 1680; y_isp = y_raw * 320 / 1050"
    )
    for row in payload["records"]:
        assert adapter.semgaze_subject_to_isp(row["raw_subject"]) == row["reference_subject"]
        assert adapter.isp_subject_to_semgaze(row["reference_subject"]) == row["raw_subject"]
        valid = [
            (x, y)
            for x, y in zip(row["raw_x"], row["raw_y"])
            if 0 <= x < frame["source_width"] and 0 <= y < frame["source_height"]
        ]
        converted = adapter.raw_gt(
            [x for x, _ in valid], [y for _, y in valid],
            image_width=frame["source_width"], image_height=frame["source_height"],
        )
        n = row["reference_count"]
        assert [x for x, _ in converted[:n]] == pytest.approx(row["reference_x"][:n], abs=1e-10)
        assert [y for _, y in converted[:n]] == pytest.approx(row["reference_y"][:n], abs=1e-10)
        if row["kind"] == "full":
            assert len(converted) == n


def test_gate_b_fixture_records_measured_population_caveat():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    counts = payload["measured_match_counts"]
    assert counts["present_semgaze_records"] == counts["key_matches"] == 14000
    assert counts["full_source_length_matches"] == 13167
    assert counts["valid_prefix_coordinate_matches"] == 14000
    assert counts["sequence_population_mismatches"] == 833
    assert counts["filtered_source_records"] == 27
    assert counts["coordinate_content_mismatches_after_validity_filter"] == 0
    # The local TP file has records with shorter fixation populations. Their
    # measured common prefixes still prove the coordinate scale; this test
    # prevents the caveat from being accidentally rewritten as full parity.
    assert any(row["kind"] == "isp_prefix" for row in payload["records"])


def test_gate_b1_boundary_formula_is_direct_and_unquantized():
    adapter = CoordinateAdapter()
    boundary = json.loads(FIXTURE.read_text(encoding="utf-8"))["boundary_probes"]
    for row in boundary:
        converted = adapter.raw_gt(
            [row["raw"][0]], [row["raw"][1]],
            image_width=COCO_RAW_WIDTH, image_height=COCO_RAW_HEIGHT,
        )
        assert converted[0] == pytest.approx(row["isp"])


def test_gate_b2_uses_authoritative_deepgaze_inverse():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    adapter = CoordinateAdapter()
    assert adapter.prediction_inverse == VERIFIED_PREDICTION_INVERSE
    assert payload["prediction_inverse"]["cases"]
    for row in payload["prediction_inverse"]["cases"]:
        assert adapter.prediction_bins([row["bin_x"]], [row["bin_y"]]) == [
            (row["pixel_x"], row["pixel_y"])
        ]
    assert adapter.prediction_bins([99], [99]) == [(507, 317)]
    assert (ROOT / "DeepGaze-VL" / "predict_scanpath.py").exists()


def test_gate_source_hashes_are_present_and_stable():
    expected = {
        "isp_preprocess": "isp-senet/ISP/COCO_Search18/GazeformerISP/src/preprocess/preprocess_fixations.py",
        "isp_evaluation": "isp-senet/ISP/COCO_Search18/GazeformerISP/src/utils/evaluation.py",
        "isp_dataset": "isp-senet/ISP/COCO_Search18/GazeformerISP/src/dataset/dataset.py",
        "isp_test": "isp-senet/ISP/COCO_Search18/GazeformerISP/src/test.py",
        "gazeformer_preprocess": "gazeformer-isp/COCO_Search18/GazeformerISP/src/preprocess/preprocess_fixations.py",
        "gazeformer_dataset": "gazeformer-isp/COCO_Search18/GazeformerISP/src/dataset/dataset.py",
        "gazeformer_evaluation": "gazeformer-isp/COCO_Search18/GazeformerISP/src/utils/evaluation.py",
        "gazeformer_test": "gazeformer-isp/COCO_Search18/GazeformerISP/src/test.py",
        "deepgaze_prediction": "DeepGaze-VL/predict_scanpath.py",
        "deepgaze_evaluation": "DeepGaze-VL/evaluate_vllm_unified.py",
    }
    hashes = json.loads(FIXTURE.read_text(encoding="utf-8"))["source_hashes"]
    for key, path in expected.items():
        digest = hashes[key]
        assert _sha256(ROOT / path) == digest


def test_scanpath_wrapper_uses_isp_scanmatch_and_empty_failure_extension():
    # The bins below invert to the scaled GT points exactly. MultiMatch is an
    # optional package in this checkout, so this fixture exercises the local
    # ScanMatch/VAME path while explicitly using the deterministic dependency
    # failure extension.
    gt = [(840.0, 525.0, 100.0), (420.0, 262.5, 100.0)]
    prediction = [(50, 50, 100), (25, 25, 100)]
    score = score_scanpath_pair(gt, prediction, require_multimatch=False)
    assert score["sm"] == pytest.approx(1.0)
    assert score["sed"] == 0
    assert score["mm"] == 0.0
    empty = score_scanpath_pair(gt, [], require_multimatch=False)
    assert empty["sm"] == empty["mm"] == 0.0
    assert empty["sed"] == 2
    with pytest.raises(CoordinateProtocolError):
        score_scanpath_pair([(-1.0, 10.0, 100.0)], [(0, 0, 100)], require_multimatch=False)
