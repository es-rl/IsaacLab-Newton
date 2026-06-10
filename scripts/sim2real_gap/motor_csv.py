"""Helpers for raw motor CSV benchmark inputs."""

from __future__ import annotations

import numpy as np


def normalize_motor_csv_time_seconds(time_s: np.ndarray) -> np.ndarray:
    """Normalize motor CSV timestamps to monotonic seconds.

    Some historical G1 captures have a mixed-unit ``time_s`` column: the first
    chunk is stored in microseconds, while the rest is stored in seconds.
    Convert only the microsecond-scale samples so analysis interpolation sees a
    normal 500 Hz time base.
    """
    if time_s.size <= 1:
        return time_s

    normalized = time_s.copy()
    large_mask = np.abs(normalized) > 1000.0
    if large_mask.any():
        candidate = normalized.copy()
        candidate[large_mask] = candidate[large_mask] / 1e6

        diffs = np.diff(candidate)
        duration = candidate[-1] - candidate[0]
        if np.all(diffs > 0.0) and 0.0 < duration < 3600.0:
            return candidate

    return normalized
