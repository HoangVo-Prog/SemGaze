"""Small, dependency-free helpers for readable progress logs."""
import math
import time


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


class RollingRate:
    """Stable completed-units/second estimate for progress ETAs.

    The estimator accepts an explicit monotonic timestamp so tests and callers
    can update it without synchronizing a device.  A few observations are
    required before reporting an ETA; this avoids presenting a misleading
    first-step estimate for highly variable episodes.
    """

    def __init__(self, *, alpha=0.2, min_observations=3):
        if not 0 < alpha <= 1:
            raise ValueError('alpha must be in (0, 1]')
        if type(min_observations) is not int or min_observations < 1:
            raise ValueError('min_observations must be a positive integer')
        self.alpha = alpha
        self.min_observations = min_observations
        self.reset()

    def reset(self, *, now=None):
        self.last_units = 0
        self.last_time = time.perf_counter() if now is None else now
        self.rate = None
        self.observations = 0

    def update(self, completed_units, *, now=None):
        if type(completed_units) is not int or completed_units < self.last_units:
            raise ValueError('completed_units must be a nondecreasing integer')
        current_time = time.perf_counter() if now is None else now
        units = completed_units - self.last_units
        seconds = current_time - self.last_time
        if units and seconds > 0:
            instant_rate = units / seconds
            self.rate = (instant_rate if self.rate is None else
                         self.alpha * instant_rate + (1 - self.alpha) * self.rate)
            self.observations += 1
        self.last_units = completed_units
        self.last_time = current_time
        return self

    def eta(self, remaining_units):
        if type(remaining_units) is not int or remaining_units < 0:
            return None
        if remaining_units == 0:
            return 0.0
        if self.rate is None or self.observations < self.min_observations:
            return None
        return remaining_units / self.rate if self.rate > 0 else None


def format_eta(seconds):
    """Human-friendly relative ETA; independent of server timezone."""
    if seconds is None:
        return 'estimating...'
    if seconds <= 0:
        return '0m'
    minutes = max(1, math.ceil(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours}h {minutes}m' if minutes else f'{hours}h'
    return f'{minutes}m' 
