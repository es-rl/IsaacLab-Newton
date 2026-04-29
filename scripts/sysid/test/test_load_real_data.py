# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for scripts.sysid.data_loading.load_real_data."""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

# Make the scripts/sysid/ directory importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from data_loading import load_real_data  # noqa: E402


def _write_motion(dirpath: str, n_ctrl: int, n_state: int, n_joints: int = 4) -> None:
    """Write joint_list.txt + control.csv + state_motor.csv for a fake motion."""
    joint_names = [f"joint_{i}" for i in range(n_joints)]
    with open(os.path.join(dirpath, "joint_list.txt"), "w") as f:
        f.write("\n".join(joint_names) + "\n")

    def _write_csv(filename: str, n_rows: int) -> None:
        with open(os.path.join(dirpath, filename), "w") as f:
            f.write("timestamp,positions\n")
            for i in range(n_rows):
                positions = [0.0] * n_joints
                f.write(f'{i * 2000},"{positions}"\n')  # 2000 us = 500 Hz

    _write_csv("control.csv", n_ctrl)
    _write_csv("state_motor.csv", n_state)


def test_aligned_rows_returns_normally() -> None:
    """500-row control + 500-row state (h1-shape) should load without error."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_motion(tmp, n_ctrl=500, n_state=500)
        ctrl, state = load_real_data(tmp, target_dt=0.002, joint_names=["joint_0", "joint_1"])
        assert ctrl.shape == (500, 2)
        assert state.shape == (500, 2)


def test_off_by_one_rows_returns_normally() -> None:
    """Tolerate a single-row delta (rounding artifacts)."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_motion(tmp, n_ctrl=500, n_state=499)
        ctrl, state = load_real_data(tmp, target_dt=0.002, joint_names=["joint_0"])
        assert ctrl.shape == (499, 1)
        assert state.shape == (499, 1)


def test_mismatched_rows_raises_value_error() -> None:
    """SO-101-shape async data (50 Hz ctrl, 500 Hz state) → row counts differ
    by 9x. Must raise ValueError instead of silently truncating."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_motion(tmp, n_ctrl=100, n_state=900)
        with pytest.raises(ValueError) as exc_info:
            load_real_data(tmp, target_dt=0.002, joint_names=["joint_0"])
        msg = str(exc_info.value)
        assert "100 rows" in msg and "900 rows" in msg
        assert "nearest-timestamp" in msg.lower()


def test_empty_state_raises() -> None:
    """Empty state CSV (0 rows) vs. populated control CSV → still raises."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_motion(tmp, n_ctrl=10, n_state=0)
        with pytest.raises(ValueError):
            load_real_data(tmp, target_dt=0.002, joint_names=["joint_0"])
