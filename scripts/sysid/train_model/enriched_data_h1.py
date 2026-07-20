# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H1 right-arm SAGE-format dataset loader.

Reads the SAGE-format trajectories produced by ``run_benchmark.py``
auto-conversion (control.csv + state_motor.csv + joint_list.txt per
motion). Mirrors G1's ``enriched_data_g1.load_experiment`` shape so the
downstream Tier-1-enriched windowing logic can be a near-direct port.

The toolbox converter parses ``positions/velocities/torques`` as Python-list
strings inside the CSV; we use ``ast.literal_eval`` for a one-shot decode
rather than pulling pandas here.

Joint order: read from ``joint_list.txt`` (canonical order
``right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow``
per ``convert_h1_chirp_to_csv.JOINT_MAP``).
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np

CANONICAL_JOINT_ORDER = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
]
N_JOINTS = len(CANONICAL_JOINT_ORDER)


def _parse_list_field(s: str) -> list[float]:
    return list(ast.literal_eval(s))


_VALID_ROW_TYPES = {"STATE_MOTOR", "STATE_CONTROL", "CONTROL"}


def _load_csv_array(path: Path, value_col: int = 2) -> np.ndarray:
    """Return ``(T, n_joints)`` from a SAGE state_motor or control CSV.

    Accepts ``CONTROL`` (toolbox converter output for control.csv),
    ``STATE_CONTROL`` (legacy alias), and ``STATE_MOTOR`` (state file).
    """
    rows = []
    with open(path) as f:
        rdr = csv.reader(f)
        next(rdr)
        for row in rdr:
            if not row or row[0] not in _VALID_ROW_TYPES:
                continue
            rows.append(_parse_list_field(row[value_col]))
    return np.array(rows, dtype=np.float32)


def load_motion(motion_dir: Path) -> dict[str, np.ndarray]:
    """Load one SAGE-format motion directory.

    Returns a dict with keys:
        ``positions`` (T, 4), ``velocities`` (T, 4), ``torques`` (T, 4)  — from state_motor.csv
        ``targets``   (T, 4)                                              — from control.csv
        ``joint_names``                                                    — from joint_list.txt

    Raises if state and control row counts differ; the toolbox concat_motor_csvs
    truncates to align lengths, so a mismatch indicates a stale or partial dir.
    """
    motion_dir = Path(motion_dir)
    joint_names = (motion_dir / "joint_list.txt").read_text().strip().splitlines()
    # joint_list.txt is the ground truth (alphabetical from the toolbox converter).
    # We trust it and surface the order on the returned dict so callers can
    # record it on the checkpoint for deploy-time alignment.
    if set(joint_names) != set(CANONICAL_JOINT_ORDER):
        raise ValueError(
            f"{motion_dir.name}: joint_list.txt has unexpected joints {joint_names}, "
            f"expected the four right-arm joints {CANONICAL_JOINT_ORDER}"
        )
    state_path = motion_dir / "state_motor.csv"
    ctrl_path = motion_dir / "control.csv"

    pos, vel, trq = [], [], []
    with open(state_path) as f:
        rdr = csv.reader(f)
        next(rdr)
        for row in rdr:
            if not row or row[0] != "STATE_MOTOR":
                continue
            pos.append(_parse_list_field(row[2]))
            vel.append(_parse_list_field(row[3]))
            trq.append(_parse_list_field(row[4]))

    targets = _load_csv_array(ctrl_path, value_col=2)
    positions = np.array(pos, dtype=np.float32)
    velocities = np.array(vel, dtype=np.float32)
    torques = np.array(trq, dtype=np.float32)

    n = min(positions.shape[0], targets.shape[0])
    if positions.shape[0] != targets.shape[0]:
        # Toolbox aligns by truncating; replicate that for safety.
        positions = positions[:n]
        velocities = velocities[:n]
        torques = torques[:n]
        targets = targets[:n]

    return {
        "positions": positions,
        "velocities": velocities,
        "torques": torques,
        "targets": targets,
        "joint_names": joint_names,
    }


def load_split(split_json: Path) -> dict[str, list[str]]:
    """Read the multijoint_h1_arm_split.json manifest."""
    with open(split_json) as f:
        manifest = json.load(f)
    return {
        "train": list(manifest["train"]),
        "val": list(manifest["val"]),
        "test": list(manifest["test"]),
    }


def _strip_motor_suffix(name: str) -> str:
    """``C01_f0_1-0_5_a0_05_motor.csv`` -> ``C01_f0_1-0_5_a0_05``."""
    if name.endswith("_motor.csv"):
        return name[: -len("_motor.csv")]
    if name.endswith(".csv"):
        return name[: -len(".csv")]
    return name


def load_split_motions(
    sage_root: Path,
    split_json: Path,
    role: str,
) -> list[tuple[str, dict[str, np.ndarray]]]:
    """Return list of (motion_name, motion_dict) for the given split role.

    ``sage_root`` is the dir holding per-motion SAGE folders, e.g.
    ``~/data/h1/sage_motions``.
    The split JSON's names include ``_motor.csv`` suffix; we strip it before
    looking up directory names.
    """
    sage_root = Path(sage_root)
    split = load_split(split_json)
    if role not in split:
        raise ValueError(f"role must be one of {list(split.keys())}, got {role!r}")
    out = []
    for raw in split[role]:
        name = _strip_motor_suffix(raw)
        motion_dir = sage_root / name
        if not motion_dir.is_dir():
            raise FileNotFoundError(f"missing SAGE dir: {motion_dir}")
        out.append((name, load_motion(motion_dir)))
    return out
