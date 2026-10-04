import re
from .coordinates import coordinate_bin
from .duration import duration_bin

END_FIX = "<END_FIX>"


def serialize_xyd(fixations):
    return "[" + ", ".join(f"({x:02d}, {y:02d}, {d:03d}){END_FIX}" for x, y, d in fixations) + "]"


def serialize_xyd_record(record):
    return serialize_xyd((coordinate_bin(x, record.image_width), coordinate_bin(y, record.image_height),
                          duration_bin(d)) for x, y, d in zip(record.x_px, record.y_px, record.duration_ms))


def parse_xyd_output(text, n):
    """Boundary-aware fallback extraction per WHERE §12.4, plus exact-format diagnostic."""
    if n <= 0:
        raise ValueError("positive requested fixation count required")
    matches = re.findall(r"\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)\s*<END_FIX>", text)
    values = [(max(0, min(99, int(x))), max(0, min(99, int(y))), max(0, min(999, int(d))))
              for x, y, d in matches]
    return {"fixations": values[:n], "accepted_fixation_count": len(values), "requested_fixation_count": n,
            "under_generated": len(values) < n,
            "canonical_format_valid": bool(values) and text == serialize_xyd(values)}
