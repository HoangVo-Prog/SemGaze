"""Executable COCO-Search18 coordinate/remapping parity fixture.

The fixture is generated from the local SemGaze source records and the local
ISP-SENet TP records.  It deliberately tests the reference frame conversion
without routing GT through the model's action grid.
"""

import json
import hashlib
from pathlib import Path

import pytest


FIXTURE = Path(__file__).parent / "fixtures" / "isp_coordinate_pairs.json"


def test_local_isp_coordinate_transform_and_subject_remap():
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    frame = payload["transform"]
    sx = frame["reference_width"] / frame["source_width"]
    sy = frame["reference_height"] / frame["source_height"]

    assert payload["subject_mapping"] == "reference_subject = source_subject - 1"
    assert (frame["reference_width"], frame["reference_height"]) == (512, 320)

    for probe in payload["boundary_probes"]:
        assert probe["isp"] == pytest.approx(
            [probe["raw"][0] * sx, probe["raw"][1] * sy], abs=1e-12
        )

    root = FIXTURE.parents[2]
    source_path = root / payload["source"]
    reference_path = root / payload["reference"]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == payload["source_sha256"]
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == payload["reference_sha256"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    source_by_key = {
        (r["name"], r["task"], r["condition"], r["subject"]): r
        for r in source
        if r.get("condition") == "present"
    }
    reference_by_key = {
        (r["name"], r["task"], r["condition"], r["subject"]): r
        for r in reference
    }

    for row in payload["records"]:
        assert row["reference_subject"] == row["raw_subject"] - 1
        valid = [
            (x, y, t)
            for x, y, t in zip(row["raw_x"], row["raw_y"], row["raw_t_ms"])
            if 0 <= x < frame["source_width"] and 0 <= y < frame["source_height"]
        ]
        n = row["reference_count"]
        assert n == len(row["reference_x"]) == len(row["reference_y"]) == len(row["reference_t_ms"])
        assert n <= len(valid)
        assert row["reference_x"] == pytest.approx([x * sx for x, _, _ in valid[:n]], abs=1e-12)
        assert row["reference_y"] == pytest.approx([y * sy for _, y, _ in valid[:n]], abs=1e-12)
        assert row["reference_t_ms"] == [t for _, _, t in valid[:n]]
        source_row = source_by_key[(row["name"], row["task"], row["condition"], row["raw_subject"])]
        reference_row = reference_by_key[(row["name"], row["task"], row["condition"], row["reference_subject"])]
        assert source_row["X"] == row["raw_x"]
        assert source_row["Y"] == row["raw_y"]
        assert source_row["T"] == row["raw_t_ms"]
        assert reference_row["X"] == row["reference_x"]
        assert reference_row["Y"] == row["reference_y"]
        assert reference_row["T"][:n] == row["reference_t_ms"]


def test_full_local_isp_target_present_parity():
    """Exercise every target-present key in the two checked-in source trees."""
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    frame = payload["transform"]
    sx = frame["reference_width"] / frame["source_width"]
    sy = frame["reference_height"] / frame["source_height"]
    root = FIXTURE.parents[2]
    source = json.loads((root / payload["source"]).read_text(encoding="utf-8"))
    reference = json.loads((root / payload["reference"]).read_text(encoding="utf-8"))
    source_rows = [row for row in source if row.get("condition") == "present"]
    source_by_key = {
        (row["name"], row["task"], row["condition"], row["subject"]): row
        for row in source_rows
    }
    reference_by_key = {
        (row["name"], row["task"], row["condition"], row["subject"]): row
        for row in reference
    }
    counts = {"full": 0, "prefix": 0, "invalid_records": 0, "invalid_points": 0}
    assert len(source_rows) == 14_000
    from semgaze.evaluation.metrics_scanpath import CoordinateAdapter
    adapter = CoordinateAdapter()
    for source_row in source_rows:
        key = (source_row["name"], source_row["task"], source_row["condition"], source_row["subject"] - 1)
        assert adapter.semgaze_subject_to_isp(source_row["subject"]) == key[3]
        assert key in reference_by_key
        reference_row = reference_by_key[key]
        valid = [
            (x, y, t)
            for x, y, t in zip(source_row["X"], source_row["Y"], source_row["T"])
            if 0 <= x < frame["source_width"] and 0 <= y < frame["source_height"]
        ]
        if len(valid) != len(source_row["X"]):
            counts["invalid_records"] += 1
            counts["invalid_points"] += len(source_row["X"]) - len(valid)
        n = reference_row["length"]
        assert len(reference_row["X"]) == len(reference_row["Y"]) == n
        expected = valid[:n]
        assert n <= len(valid)
        assert reference_row["X"] == pytest.approx([x * sx for x, _, _ in expected], abs=1e-10)
        assert reference_row["Y"] == pytest.approx([y * sy for _, y, _ in expected], abs=1e-10)
        assert reference_row["T"][:n] == [t for _, _, t in expected]
        if n == len(source_row["X"]):
            counts["full"] += 1
        else:
            counts["prefix"] += 1
    assert counts == {"full": 13_167, "prefix": 833, "invalid_records": 27, "invalid_points": 41}

