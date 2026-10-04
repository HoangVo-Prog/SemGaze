import math


def duration_bin(milliseconds):
    if not math.isfinite(milliseconds):
        raise ValueError("finite dwell duration in milliseconds required")
    return max(0, min(999, round(milliseconds)))
