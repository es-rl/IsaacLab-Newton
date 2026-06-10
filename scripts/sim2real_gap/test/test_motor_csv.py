# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for raw motor CSV helpers."""

from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from motor_csv import normalize_motor_csv_time_seconds  # noqa: E402


def test_normalizes_mixed_microsecond_and_second_timestamps() -> None:
    raw = np.array([0.0, 1949.9, 3925.9, 997938.3, 0.9999345, 1.0019218])

    normalized = normalize_motor_csv_time_seconds(raw)

    assert np.allclose(normalized, [0.0, 0.0019499, 0.0039259, 0.9979383, 0.9999345, 1.0019218])
    assert np.all(np.diff(normalized) > 0.0)


def test_leaves_normal_second_timestamps_unchanged() -> None:
    raw = np.array([0.0, 0.002, 0.004, 0.006])

    normalized = normalize_motor_csv_time_seconds(raw)

    assert np.array_equal(normalized, raw)
