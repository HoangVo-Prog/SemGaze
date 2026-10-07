"""Gated ISP-SENet scanpath metrics.

The production wrapper is intentionally unavailable until the owner resolves
the required raw-GT parity and quantized prediction-bin inverse protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence


class CoordinateProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class CoordinateAdapter:
    """Explicit ISP frame declaration; no default inverse is provided."""
    width: int = 512
    height: int = 320
    prediction_inverse: str | None = None

    def __post_init__(self):
        if (self.width, self.height) != (512, 320):
            raise CoordinateProtocolError("ISP COCO-Search18 frame must be 512x320")
        if self.prediction_inverse is None:
            raise CoordinateProtocolError("predicted-bin -> ISP-frame representative requires owner decision")

    def raw_gt(self, xs: Sequence[float], ys: Sequence[float], *, image_width: int, image_height: int):
        if image_width <= 0 or image_height <= 0 or len(xs) != len(ys):
            raise CoordinateProtocolError("raw coordinates and dimensions are invalid")
        values = [(float(x) * self.width / image_width, float(y) * self.height / image_height)
                  for x, y in zip(xs, ys)]
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in values):
            raise CoordinateProtocolError("raw coordinates must be finite")
        return values

    def prediction_bins(self, xs: Sequence[int], ys: Sequence[int]):
        if self.prediction_inverse != "deepgaze_historical_round":
            raise CoordinateProtocolError("prediction-bin inverse is not historically verified")
        if len(xs) != len(ys) or any(type(v) is not int or not 0 <= v <= 99 for v in (*xs, *ys)):
            raise CoordinateProtocolError("predicted coordinates must be integer bins in 0..99")
        return [(round(x / 100.0 * self.width), round(y / 100.0 * self.height)) for x, y in zip(xs, ys)]


def _sed_symbols(points, *, width=512, height=320, n=5):
    wstep, hstep = width // n, height // n
    return "".join(chr(97 + (int(x) // wstep) + (int(y) // hstep) * n) for x, y, _ in points)


def _levenshtein(a: str, b: str):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        new = [i]
        for j, cb in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = new
    return row[-1]


def _hmean(values):
    values = [float(v) for v in values]
    if any(v <= 0 or not math.isfinite(v) for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def score_scanpath_pair(*args, **kwargs):
    raise RuntimeError("BLOCKED: raw-GT/ISP parity and prediction-bin inverse owner decision are unresolved")


def invalid_generation_contribution(gt_symbol_length: int, *, empty: bool):
    """Apply the contract's deterministic WHERE failure extension."""
    if type(gt_symbol_length) is not int or gt_symbol_length < 0:
        raise ValueError("GT symbol length must be a nonnegative integer")
    if empty:
        return {"sm": 0.0, "mm": 0.0, "sed": gt_symbol_length,
                "valid": False, "format_failure": True}
    raise ValueError("non-empty malformed paths must be scored after parser recovery")


def aggregate_scanpath_draw(scores: Iterable[dict]):
    scores = list(scores)
    if not scores:
        raise ValueError("cannot aggregate an empty draw")
    return {"eval_where_sm": sum(s["sm"] for s in scores) / len(scores),
            "eval_where_mm": sum(s["mm"] for s in scores) / len(scores),
            "eval_where_sed": sum(s["sed"] for s in scores) / len(scores),
            "mm_numeric_failure_count": sum(bool(s.get("mm_numeric_failure")) for s in scores)}
