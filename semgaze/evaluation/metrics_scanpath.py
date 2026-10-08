"""COCO-Search18 scanpath metrics using the audited ISP reference.

The coordinate contracts in this module are explicit. ISP's COCO evaluator
consumes a 512x320 frame, SemGaze ground-truth annotations are in the
original 1680x1050 frame, and generated WHERE coordinates are integer values
on the 0..99 grid. The prediction-grid inverse is taken from the local
``DeepGaze-VL/predict_scanpath.py`` source; it is not inferred from the
lossy forward quantizer.

Metric functions consume already parsed records. They never load a model or
generate text. MultiMatch is imported lazily because the audited package is
an optional runtime dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
import math
import numbers
from pathlib import Path
import sys
import types
from typing import Any, Iterable, Sequence


class CoordinateProtocolError(ValueError):
    """Raised when a record cannot be adapted to the audited frame."""


class DatasetIntegrityError(CoordinateProtocolError):
    """Raised when ground-truth data is missing or dimensionally invalid."""


class ScanpathMetricDependencyError(RuntimeError):
    """Raised when an audited optional runtime dependency is unavailable."""


ISP_WIDTH = 512
ISP_HEIGHT = 320
COCO_RAW_WIDTH = 1680
COCO_RAW_HEIGHT = 1050
VERIFIED_PREDICTION_INVERSE = "deepgaze_vl_predict_scanpath_round"
# Read compatibility for artifacts produced by the earlier gated adapter.
LEGACY_PREDICTION_INVERSES = frozenset(
    {"deepgaze_historical_round", VERIFIED_PREDICTION_INVERSE}
)
ISP_SCANMATCH_SOURCE = "third_party/isp_reference/COCO_Search18/scanmatch.py"
ISP_VAME_SOURCE = "third_party/isp_reference/COCO_Search18/visual_attention_metrics.py"
DEEPGAZE_INVERSE_SOURCE = "DeepGaze-VL/predict_scanpath.py:238-244"

# Constructed lazily because importing this module must remain cheap for
# training and frozen-artifact inspection.
_SCANMATCH_REFERENCES: dict[bool, Any] = {}
_VAME_REFERENCE: Any | None = None


@dataclass(frozen=True)
class CoordinateAdapter:
    """Adapt SemGaze COCO records to the audited ISP coordinate protocol."""

    width: int = ISP_WIDTH
    height: int = ISP_HEIGHT
    prediction_inverse: str | None = VERIFIED_PREDICTION_INVERSE

    def __post_init__(self):
        if (self.width, self.height) != (ISP_WIDTH, ISP_HEIGHT):
            raise CoordinateProtocolError("ISP COCO-Search18 frame must be 512x320")
        # Configurations written while the gate was unresolved used null. The
        # source audit now establishes the default, so normalize that legacy
        # value to the verified protocol rather than leaving a false blocker.
        if self.prediction_inverse is None:
            object.__setattr__(self, "prediction_inverse", VERIFIED_PREDICTION_INVERSE)
        if self.prediction_inverse not in LEGACY_PREDICTION_INVERSES:
            raise CoordinateProtocolError(
                "prediction-bin inverse is not established by an audited reference"
            )

    @staticmethod
    def semgaze_subject_to_isp(subject: int) -> int:
        """Map persisted SemGaze subject IDs (1..10) to ISP IDs (0..9)."""
        if type(subject) is not int or not 1 <= subject <= 10:
            raise CoordinateProtocolError("SemGaze subject ID must be an integer in 1..10")
        return subject - 1

    @staticmethod
    def isp_subject_to_semgaze(subject: int) -> int:
        """Map ISP's zero-based subject IDs back to SemGaze IDs."""
        if type(subject) is not int or not 0 <= subject <= 9:
            raise CoordinateProtocolError("ISP subject ID must be an integer in 0..9")
        return subject + 1

    def raw_gt(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        *,
        image_width: int = COCO_RAW_WIDTH,
        image_height: int = COCO_RAW_HEIGHT,
    ) -> list[tuple[float, float]]:
        """Scale raw annotation pixels directly into the ISP frame.

        The active ISP COCO evaluation trajectory reads preprocessed 512x320
        ``X``/``Y`` values directly. SemGaze's raw records therefore use the
        exact scale ``x*512/1680`` and ``y*320/1050`` with no clipping or
        intermediate 0..99 quantization.
        """
        if image_width <= 0 or image_height <= 0 or len(xs) != len(ys):
            raise CoordinateProtocolError("raw coordinates and dimensions are invalid")
        try:
            values = [
                (float(x) * self.width / image_width, float(y) * self.height / image_height)
                for x, y in zip(xs, ys)
            ]
        except (TypeError, ValueError, OverflowError) as exc:
            raise CoordinateProtocolError("raw coordinates must be numeric") from exc
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in values):
            raise CoordinateProtocolError("raw coordinates must be finite")
        return values

    def prediction_bins(
        self, xs: Sequence[int], ys: Sequence[int]
    ) -> list[tuple[int, int]]:
        """Apply DeepGaze-VL's integer-bin-to-pixel representative."""
        if len(xs) != len(ys) or any(
            isinstance(v, bool)
            or not isinstance(v, numbers.Integral)
            or not 0 <= int(v) <= 99
            for v in (*xs, *ys)
        ):
            raise CoordinateProtocolError("predicted coordinates must be integer bins in 0..99")
        # The reference uses Python's round through int(round(...)); retain
        # its ties-to-even behavior exactly.
        return [
            (
                int(round(x / 100.0 * self.width)),
                int(round(y / 100.0 * self.height)),
            )
            for x, y in zip(xs, ys)
        ]


def _coerce_points(points: Sequence[Any], *, name: str) -> list[tuple[float, float, float]]:
    """Normalize tuple or NumPy structured-array points without changing values."""
    result: list[tuple[float, float, float]] = []
    for point in points:
        try:
            if hasattr(point, "dtype") and getattr(point.dtype, "names", None):
                x, y, d = point["start_x"], point["start_y"], point["duration"]
            else:
                x, y, d = point[:3]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise CoordinateProtocolError(
                f"{name} fixation must contain (x, y, duration)"
            ) from exc
        try:
            values = (float(x), float(y), float(d))
        except (TypeError, ValueError, OverflowError) as exc:
            raise CoordinateProtocolError(
                f"{name} fixation values must be numeric"
            ) from exc
        if not all(math.isfinite(v) for v in values):
            raise CoordinateProtocolError(f"{name} fixation values must be finite")
        result.append(values)
    return result


def _prediction_points(
    points: Sequence[Any], adapter: CoordinateAdapter
) -> list[tuple[float, float, float]]:
    """Convert parsed ``(x_bin, y_bin, duration_ms)`` tuples to ISP pixels."""
    bins: list[tuple[int, int]] = []
    durations: list[float] = []
    for point in points:
        try:
            if hasattr(point, "dtype") and getattr(point.dtype, "names", None):
                x, y, d = point["start_x"], point["start_y"], point["duration"]
            else:
                x, y, d = point[:3]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise CoordinateProtocolError(
                "prediction fixation must contain (x, y, duration)"
            ) from exc
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, numbers.Integral)
            or not isinstance(y, numbers.Integral)
        ):
            raise CoordinateProtocolError("predicted coordinates must be integer bins in 0..99")
        x, y = int(x), int(y)
        try:
            duration = float(d)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CoordinateProtocolError("predicted durations must be numeric") from exc
        if not math.isfinite(duration):
            raise CoordinateProtocolError("predicted durations must be finite")
        bins.append((x, y))
        durations.append(duration)
    pixels = adapter.prediction_bins([x for x, _ in bins], [y for _, y in bins])
    return [(float(x), float(y), d) for (x, y), d in zip(pixels, durations)]


def _load_reference_scanmatch():
    """Load the vendored ScanMatch class without optional plotting imports."""
    try:
        module = importlib.import_module("third_party.isp_reference.COCO_Search18.scanmatch")
    except (ImportError, ModuleNotFoundError):
        path = Path(__file__).resolve().parents[2] / ISP_SCANMATCH_SOURCE
        if not path.exists():
            raise ScanpathMetricDependencyError(f"audited ScanMatch source is missing: {path}")
        from importlib import util as importlib_util

        spec = importlib_util.spec_from_file_location("semgaze_isp_scanmatch", path)
        if spec is None or spec.loader is None:
            raise ScanpathMetricDependencyError(f"cannot load audited ScanMatch source: {path}")
        module = importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.ScanMatch


@lru_cache(maxsize=4096)
def _cached_gt_scanmatch_sequence(scorer, gt, with_duration):
    """Reuse exact GT sequence across draws; scorer identity is part of key."""
    import numpy as np

    values = (np.asarray(gt, dtype=float) if with_duration else
              np.asarray([(x, y) for x, y, _ in gt], dtype=float))
    result = scorer.fixationToSequence(values).astype(np.int32)
    result.setflags(write=False)
    return result


def _scanmatch_score(
    gt: list[tuple[float, float, float]],
    pred: list[tuple[float, float, float]],
    *,
    with_duration: bool,
) -> float:
    import numpy as np

    ScanMatch = _load_reference_scanmatch()
    kwargs = {
        "Xres": ISP_WIDTH,
        "Yres": ISP_HEIGHT,
        "Xbin": 16,
        "Ybin": 12,
        "Offset": (0, 0),
        "Threshold": 3.5,
    }
    if with_duration:
        kwargs["TempBin"] = 50
    # The reference object is immutable after construction.  Reuse it across
    # queries so metric evaluation never pays a per-query setup cost.
    key = bool(with_duration)
    scorer = _SCANMATCH_REFERENCES.get(key)
    if scorer is None:
        scorer = ScanMatch(**kwargs)
        _SCANMATCH_REFERENCES[key] = scorer
    # ISP consumes milliseconds directly; predictions are never cached.
    pred_array = (np.asarray(pred, dtype=float) if with_duration else
                  np.asarray([(x, y) for x, y, _ in pred], dtype=float))
    seq_gt = _cached_gt_scanmatch_sequence(scorer, tuple(gt), bool(with_duration))
    seq_pred = scorer.fixationToSequence(pred_array).astype(np.int32)
    return float(scorer.match(seq_gt, seq_pred)[0])


def _multimatch_module(module=None):
    if module is not None:
        return module
    try:
        return importlib.import_module("multimatch_gaze")
    except ImportError as exc:
        raise ScanpathMetricDependencyError(
            "MultiMatch requires audited multimatch-gaze==0.1.3"
        ) from exc


def _multimatch_points(points: list[tuple[float, float, float]]):
    import numpy as np

    padded = list(points)
    padded.extend([(1.0, 1.0, 0.001)] * max(0, 3 - len(padded)))
    return np.asarray(
        padded,
        dtype={
            "names": ("start_x", "start_y", "duration"),
            "formats": ("f8", "f8", "f8"),
        },
    )


def _multimatch_score(gt, pred, module=None):
    scorer = _multimatch_module(module)
    try:
        compare = scorer.docomparison
    except AttributeError as exc:
        raise ScanpathMetricDependencyError(
            "multimatch-gaze==0.1.3 does not expose audited docomparison API"
        ) from exc
    result = compare(
        _multimatch_points([(x, y, d / 1000.0) for x, y, d in gt]),
        _multimatch_points([(x, y, d / 1000.0) for x, y, d in pred]),
        screensize=[ISP_WIDTH, ISP_HEIGHT],
    )
    values = [float(v) for v in result]
    if len(values) != 5:
        raise CoordinateProtocolError("MultiMatch reference must return five components")
    if any(not math.isfinite(v) for v in values):
        return None, "audited MultiMatch returned a non-finite component"
    return values, None


def _load_reference_vame():
    """Load the exact vendored VAME module with plotting imports stubbed.

    The audited ``string_edit_distance`` function only needs NumPy, while the
    reference file imports plotting/OpenCV helpers for unrelated saliency
    functions. Keeping those imports lazy lets the parity wrapper run in a
    minimal CPU environment without replacing the reference implementation.
    """
    global _VAME_REFERENCE
    if _VAME_REFERENCE is not None:
        return _VAME_REFERENCE
    path = Path(__file__).resolve().parents[2] / ISP_VAME_SOURCE
    if not path.exists():
        raise ScanpathMetricDependencyError(f"audited VAME source is missing: {path}")
    import importlib.util

    spec = importlib.util.spec_from_file_location("semgaze_isp_vame", path)
    if spec is None or spec.loader is None:
        raise ScanpathMetricDependencyError(f"cannot load audited VAME source: {path}")
    saved = {name: sys.modules.get(name) for name in ("matplotlib", "matplotlib.pyplot", "cv2")}
    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules.setdefault("matplotlib", matplotlib_stub)
    sys.modules.setdefault("matplotlib.pyplot", pyplot_stub)
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    _VAME_REFERENCE = module
    return module


@lru_cache(maxsize=1)
def _vame_stimulus():
    """VAME only reads frame shape: share the identical zero-filled ISP frame."""
    import numpy as np

    return np.zeros((ISP_HEIGHT, ISP_WIDTH, 3), dtype=np.float32)


def _vame_sed_score(gt, pred) -> int:
    import numpy as np

    module = _load_reference_vame()
    stimulus = _vame_stimulus()
    gt_array = np.asarray(gt, dtype=float)
    pred_array = np.asarray(pred, dtype=float)
    return int(module.string_edit_distance(stimulus, gt_array, pred_array))


def _sed_symbols(points, *, width=ISP_WIDTH, height=ISP_HEIGHT, n=5):
    wstep, hstep = width // n, height // n
    return "".join(
        chr(97 + (int(x) // wstep) + (int(y) // hstep) * n)
        for x, y, _ in points
    )


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
    try:
        from scipy.stats import hmean
        return float(hmean(values))
    except ImportError:  # keep lightweight coordinate-only environments usable
        return len(values) / sum(1.0 / v for v in values)


def _require_in_frame(points: Sequence[tuple[float, float, float]], *, name: str):
    """Reject malformed GT instead of silently letting reference clipping repair it."""
    for x, y, duration in points:
        if not (0.0 <= x < ISP_WIDTH and 0.0 <= y < ISP_HEIGHT):
            raise CoordinateProtocolError(
                f"{name} fixation lies outside the audited 512x320 frame: {(x, y)!r}"
            )
        if duration < 0:
            raise CoordinateProtocolError(f"{name} fixation duration must be nonnegative")


def score_scanpath_pair(
    gt_points: Sequence[Any],
    prediction_points: Sequence[Any],
    *,
    adapter: CoordinateAdapter | None = None,
    gt_coordinate_space: str = "raw_coco",
    gt_image_width: int = COCO_RAW_WIDTH,
    gt_image_height: int = COCO_RAW_HEIGHT,
    multimatch_module=None,
    require_multimatch: bool = True,
    compute_multimatch: bool = True,
    compute_scanmatch: bool = True,
    compute_sed: bool = True,
) -> dict[str, Any]:
    """Score one same-subject GT/prediction pair with audited ISP behavior.

    GT points are ``(x, y, duration_ms)`` in the raw COCO frame by default;
    predictions are parsed ``(x_bin, y_bin, duration_ms)`` tuples. Empty
    predictions use the deterministic zero/GT-symbol extension. Set
    ``require_multimatch=False`` only for ScanMatch/coordinate fixtures when
    the optional MultiMatch package is unavailable.
    """
    adapter = adapter or CoordinateAdapter()
    if gt_coordinate_space not in ("raw_coco", "isp"):
        raise CoordinateProtocolError("gt_coordinate_space must be raw_coco or isp")
    raw_gt = _coerce_points(gt_points, name="GT")
    if not raw_gt:
        raise CoordinateProtocolError("GT scanpath must contain at least one fixation")
    if gt_coordinate_space == "raw_coco":
        xy = adapter.raw_gt(
            [x for x, _, _ in raw_gt],
            [y for _, y, _ in raw_gt],
            image_width=gt_image_width,
            image_height=gt_image_height,
        )
        gt = [(px, py, d) for (_, _, d), (px, py) in zip(raw_gt, xy)]
    else:
        gt = raw_gt
    _require_in_frame(gt, name="GT")
    pred = _prediction_points(prediction_points, adapter)
    _require_in_frame(pred, name="prediction")
    gt_symbols = _sed_symbols(gt)
    if not pred:
        return {
            "sm": 0.0,
            "sm_no_duration": 0.0,
            "sm_with_duration": 0.0,
            "mm": 0.0,
            "mm_components": [0.0] * 5,
            "sed": len(gt_symbols),
            "valid": False,
            "format_failure": True,
            "mm_numeric_failure": False,
        }
    sm_no = _scanmatch_score(gt, pred, with_duration=False) if compute_scanmatch else 0.0
    sm_with = _scanmatch_score(gt, pred, with_duration=True) if compute_scanmatch else 0.0
    sed = _vame_sed_score(gt, pred) if compute_sed else 0
    if not compute_multimatch:
        mm_components, mm_failure = [0.0] * 5, None
    else:
        try:
            mm_components, mm_failure = _multimatch_score(
                gt, pred, module=multimatch_module
            )
        except ScanpathMetricDependencyError:
            if require_multimatch:
                raise
            mm_components, mm_failure = None, "optional MultiMatch dependency unavailable"
    if mm_components is None:
        # A model-side/non-runtime degeneracy follows LOCK 5: retain the
        # query and contribute deterministic zeros to all five components.
        # Production callers normally leave require_multimatch=True, which
        # surfaces a missing audited dependency before scoring.
        if require_multimatch and mm_failure == "optional MultiMatch dependency unavailable":
            raise ScanpathMetricDependencyError(mm_failure)
        mm_components = [0.0] * 5
        mm_numeric_failure = mm_failure != "optional MultiMatch dependency unavailable"
    else:
        mm_numeric_failure = False
    return {
        "sm": _hmean([sm_no, sm_with]),
        "sm_no_duration": sm_no,
        "sm_with_duration": sm_with,
        "mm": sum(mm_components) / 5.0,
        "mm_components": mm_components,
        "sed": int(sed),
        "valid": True,
        "format_failure": False,
        "mm_numeric_failure": mm_numeric_failure,
    }


def invalid_generation_contribution(gt_symbol_length: int, *, empty: bool):
    """Apply the contract's deterministic WHERE failure extension."""
    if type(gt_symbol_length) is not int or gt_symbol_length < 0:
        raise ValueError("GT symbol length must be a nonnegative integer")
    if empty:
        return {
            "sm": 0.0,
            "mm": 0.0,
            "sed": gt_symbol_length,
            "valid": False,
            "format_failure": True,
        }
    raise ValueError("non-empty malformed paths must be scored after parser recovery")


def aggregate_scanpath_draw(scores: Iterable[dict]):
    """Reduce complete same-subject query scores for one ``(K, draw)``."""
    scores = list(scores)
    if not scores:
        raise ValueError("cannot aggregate an empty draw")
    component_rows = []
    for score in scores:
        values = score.get("mm_components")
        if values is None:
            values = [score["mm"]] * 5
        if len(values) != 5:
            raise ValueError("each scanpath score must contain five MultiMatch components")
        component_rows.append(values)
    mm_components = [
        sum(float(row[i]) for row in component_rows) / len(component_rows)
        for i in range(5)
    ]
    # Keep compatibility with older frozen rows that only stored the combined
    # ``sm`` value. Avoid an eager ``dict.get`` default here: callers may store
    # the duration-specific fields without duplicating the combined scalar.
    sm_no = sum(
        float(s["sm_no_duration"] if "sm_no_duration" in s else s["sm"])
        for s in scores
    ) / len(scores)
    sm_with = sum(
        float(s["sm_with_duration"] if "sm_with_duration" in s else s["sm"])
        for s in scores
    ) / len(scores)
    return {
        "eval_where_sm": _hmean([sm_no, sm_with]),
        "eval_where_mm": sum(mm_components) / 5.0,
        "eval_where_sed": sum(float(s["sed"]) for s in scores) / len(scores),
        "eval_diag_where_scanmatch_no_duration": sm_no,
        "eval_diag_where_scanmatch_with_duration": sm_with,
        "eval_diag_where_multimatch_components": mm_components,
        "mm_numeric_failure_count": sum(bool(s.get("mm_numeric_failure")) for s in scores),
        "where_format_failure_count": sum(bool(s.get("format_failure")) for s in scores),
        "where_valid_count": sum(bool(s.get("valid", True)) for s in scores),
    }
