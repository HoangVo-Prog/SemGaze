"""Small, dependency-free helpers for readable evaluation progress logs."""
import math


def progress_interval(total, *, target_updates=20, maximum=100):
    """Return a bounded interval that avoids one log line per example."""
    if total <= 0:
        return 1
    return max(1, min(maximum, max(25, math.ceil(total / target_updates))))


def format_duration(seconds):
    """Format wall-clock seconds compactly for a terminal log."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    return f'{minutes:02d}:{seconds:02d}'


def should_report(current, total, interval):
    return current == total or current % interval == 0
