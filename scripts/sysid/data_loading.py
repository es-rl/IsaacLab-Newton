# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-robot data loading utilities for the SysID toolbox.

Extracted from run_sysid.py so it can be unit-tested without launching the
SimulationApp. run_sysid.py imports load_real_data from this module.
"""

from __future__ import annotations

import ast
import csv
import os

import numpy as np
from scipy.interpolate import interp1d


def _log(msg: str) -> None:
    """Lightweight logger that writes to stdout. Mirrors run_sysid.log_message
    behavior for messages emitted from this module before/without sim launch."""
    print(msg, flush=True)


def load_real_data(data_dir: str, target_dt: float, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load real-robot control + state data for the specified joints.

    Args:
        data_dir: Directory with control.csv, state_motor.csv, joint_list.txt.
        target_dt: Target timestep for resampling.
        joint_names: Joint names to extract (determines column order and count).

    Returns:
        commanded: (T, N) commanded positions for the specified joints.
        measured:  (T, N) actual positions for the specified joints.

    Raises:
        ValueError: If control.csv and state_motor.csv have row counts that
            differ by more than 1. The toolbox previously truncated silently
            via min(len(ctrl), len(state)), which produced time-stretched
            garbage on async-collector data (e.g. SO-101 hirate at 50 Hz
            control / 500 Hz state). Customers with mismatched-rate data
            must align their CSVs by nearest-timestamp pairing in their
            data-prep pipeline before feeding into sysid.
    """
    num_joints = len(joint_names)

    joint_list_path = os.path.join(data_dir, "joint_list.txt")
    with open(joint_list_path) as f:
        all_joint_names = [line.strip() for line in f if line.strip()]

    joint_indices = []
    for name in joint_names:
        joint_indices.append(all_joint_names.index(name))

    # control.csv → commanded positions
    ctrl_times, ctrl_pos = [], []
    with open(os.path.join(data_dir, "control.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ctrl_times.append(float(row["timestamp"]))
            positions = ast.literal_eval(row["positions"])
            ctrl_pos.append([positions[i] for i in joint_indices])

    # state_motor.csv → actual positions
    state_times, state_pos = [], []
    with open(os.path.join(data_dir, "state_motor.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            state_times.append(float(row["timestamp"]))
            positions = ast.literal_eval(row["positions"])
            state_pos.append([positions[i] for i in joint_indices])

    ctrl_pos = np.array(ctrl_pos)
    state_pos = np.array(state_pos)

    # Row-count alignment: previous behavior truncated silently to
    # min(len(ctrl), len(state)), which mapped state row i (recorded at real
    # time i*state_dt) onto sim time i*ctrl_dt — producing time-stretched
    # garbage when control and state rates differ. Now we raise loudly.
    delta = abs(len(ctrl_pos) - len(state_pos))
    if delta > 1:
        raise ValueError(
            f"control.csv ({len(ctrl_pos)} rows) and state_motor.csv "
            f"({len(state_pos)} rows) differ by {delta} rows. SysID assumes "
            "row index = wall-clock time (i.e. row i of each represents the "
            "same instant). Align your CSVs by nearest-timestamp pairing "
            "before sysid (see SO-101 data-prep example)."
        )

    # Tolerated 0- or 1-row delta: trim to common length.
    min_len = min(len(ctrl_pos), len(state_pos))
    ctrl_pos = ctrl_pos[:min_len]
    state_pos = state_pos[:min_len]

    # Timestamps are in microseconds
    ctrl_times = np.array(ctrl_times[:min_len])
    real_dt = float(np.median(np.diff(ctrl_times))) / 1e6

    # Resample if needed
    if abs(real_dt - target_dt) > 1e-6:
        T = ctrl_pos.shape[0]
        duration = T * real_dt
        new_len = int(round(duration / target_dt))
        old_times = np.linspace(0, duration, T, endpoint=False)
        new_times = np.linspace(0, duration, new_len, endpoint=False)
        new_times = new_times[new_times <= old_times[-1]]

        new_ctrl = np.zeros((len(new_times), num_joints))
        new_state = np.zeros((len(new_times), num_joints))
        for j in range(num_joints):
            new_ctrl[:, j] = interp1d(old_times, ctrl_pos[:, j], kind="linear")(new_times)
            new_state[:, j] = interp1d(old_times, state_pos[:, j], kind="linear")(new_times)
        ctrl_pos = new_ctrl
        state_pos = new_state
        _log(f"Resampled {T} ({1 / real_dt:.0f}Hz) → {len(new_times)} ({1 / target_dt:.0f}Hz)")

    _log(f"Loaded real data: {len(ctrl_pos)} steps, {num_joints} joints")
    return ctrl_pos, state_pos
