import math


def coordinate_bin(value, original_size):
    if not math.isfinite(value) or not math.isfinite(original_size) or original_size <= 0:
        raise ValueError("finite coordinate and positive original dimension required")
    return max(0, min(99, int(round(round(100 * value / original_size, 1)))))
