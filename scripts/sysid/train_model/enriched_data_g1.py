# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-experiment G1 multi-joint data loader with explicit train/val/probe split.

Each MJ experiment directory is loaded independently and resampled to 500 Hz.
Supports experiment-level splitting so the C-configs can be held out as a
distribution-shift probe set.

Timestamp convention: per-experiment CSVs store timestamps in microseconds
(float64), which are divided by 1e6 to produce seconds before resampling.
The loader skips non-data rows (EVENT rows) by filtering on the ``type``
column, unlike the multijoint_all reader which has no EVENT rows.
"""

from __future__ import annotations

import ast
import csv
import json
import os

import numpy as np


def _resample_to_grid(
    timestamps: np.ndarray,
    data: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Resample each column of *data* onto *t_grid* via linear interpolation.

    Parameters
    ----------
    timestamps : (N,) source timestamps in seconds
    data       : (N, C) source values
    t_grid     : (M,) target uniform timestamps

    Returns
    -------
    resampled : (M, C) float64
    """
    n_cols = data.shape[1]
    out = np.empty((len(t_grid), n_cols), dtype=np.float64)
    for c in range(n_cols):
        out[:, c] = np.interp(t_grid, timestamps, data[:, c])
    return out


def _parse_list_string(s: str) -> list[float]:
    return ast.literal_eval(s)


def _read_per_exp_state(path: str):
    """Read state_motor.csv, skipping non-STATE_MOTOR rows.

    Returns
    -------
    timestamps : (N,) float64 — seconds (converted from microseconds)
    positions  : (N, 4) float64
    velocities : (N, 4) float64
    torques    : (N, 4) float64
    """
    timestamps, positions, velocities, torques = [], [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("type") and row["type"] != "STATE_MOTOR":
                continue
            timestamps.append(float(row["timestamp"]))
            positions.append(_parse_list_string(row["positions"]))
            velocities.append(_parse_list_string(row["velocities"]))
            torques.append(_parse_list_string(row["torques"]))
    return (
        np.array(timestamps, dtype=np.float64) / 1e6,
        np.array(positions, dtype=np.float64),
        np.array(velocities, dtype=np.float64),
        np.array(torques, dtype=np.float64),
    )


def _read_per_exp_control(path: str):
    """Read control.csv, skipping non-CONTROL rows.

    Returns
    -------
    timestamps : (N,) float64 — seconds (converted from microseconds)
    targets    : (N, 4) float64
    """
    timestamps, targets = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("type") and row["type"] != "CONTROL":
                continue
            timestamps.append(float(row["timestamp"]))
            targets.append(_parse_list_string(row["positions"]))
    return (
        np.array(timestamps, dtype=np.float64) / 1e6,
        np.array(targets, dtype=np.float64),
    )


def load_experiment(exp_dir: str) -> dict[str, np.ndarray]:
    """Load one MJ experiment directory and resample all signals to 500 Hz.

    Parameters
    ----------
    exp_dir : path to a single MJ* experiment directory containing
              ``state_motor.csv`` and ``control.csv``.

    Returns
    -------
    dict with keys ``positions``, ``velocities``, ``torques``, ``targets``,
    each an (T, 4) float32 array at 500 Hz over the overlapping time window.
    """
    state_path = os.path.join(exp_dir, "state_motor.csv")
    control_path = os.path.join(exp_dir, "control.csv")

    ts_state, positions, velocities, torques = _read_per_exp_state(state_path)
    ts_control, targets = _read_per_exp_control(control_path)

    t_start = max(ts_state[0], ts_control[0])
    t_end = min(ts_state[-1], ts_control[-1])

    dt = 1.0 / 500.0
    t_grid = np.arange(t_start, t_end, dt)

    positions_r = _resample_to_grid(ts_state, positions, t_grid)
    velocities_r = _resample_to_grid(ts_state, velocities, t_grid)
    torques_r = _resample_to_grid(ts_state, torques, t_grid)
    targets_r = _resample_to_grid(ts_control, targets, t_grid)

    return {
        "positions": positions_r.astype(np.float32),
        "velocities": velocities_r.astype(np.float32),
        "torques": torques_r.astype(np.float32),
        "targets": targets_r.astype(np.float32),
    }


def _is_c_config(name: str) -> bool:
    """Identify back-bent C-config experiments.

    Checks for ``_C_`` as a middle segment or ``_C`` as a trailing suffix.
    Both patterns are present in the G1 MJ dataset.
    """
    return "_C_" in name or name.endswith("_C")


def split_experiments(
    names: list[str],
    val_count: int = 2,
    seed: int = 0,
) -> tuple[list[str], list[str], list[str]]:
    """Partition MJ experiment names into train / val / probe sets.

    C-config experiments (back-bent posture, identified by ``_C_`` substring
    or ``_C`` suffix) are always placed in the probe set for distribution-shift
    evaluation. The remaining A/B experiments are randomly split into val
    (``val_count`` experiments) and train (the rest).

    Parameters
    ----------
    names     : list of experiment names (bare names, not full paths)
    val_count : how many A/B experiments to hold out for validation
    seed      : RNG seed for reproducible val selection

    Returns
    -------
    train : sorted list of training experiment names
    val   : sorted list of validation experiment names
    probe : sorted list of C-config probe experiment names
    """
    c_probe = sorted(n for n in names if _is_c_config(n))
    ab = sorted(n for n in names if not _is_c_config(n))

    rng = np.random.default_rng(seed)
    ab_shuffled = ab.copy()
    rng.shuffle(ab_shuffled)

    val = sorted(ab_shuffled[:val_count])
    train = sorted(ab_shuffled[val_count:])

    return train, val, c_probe


def list_mj_experiments(experiments_root: str) -> list[str]:
    """List all MJ* subdirectory names under ``experiments_root``."""
    return sorted(
        d
        for d in os.listdir(experiments_root)
        if d.startswith("MJ") and os.path.isdir(os.path.join(experiments_root, d))
    )


def list_experiments(experiments_root: str, prefixes: tuple[str, ...] = ("MJ",)) -> list[str]:
    """List subdirectories under ``experiments_root`` matching any given prefix.

    Used by v2 loaders (prefixes=("T_","H_")) alongside the legacy MJ helper.
    """
    return sorted(
        d
        for d in os.listdir(experiments_root)
        if any(d.startswith(p) for p in prefixes) and os.path.isdir(os.path.join(experiments_root, d))
    )


def split_v2(names: list[str]) -> tuple[list[str], list[str]]:
    """Partition v2 names into (train, held_out) by T_/H_ prefix. Names without
    a recognized prefix are rejected.
    """
    train: list[str] = []
    held: list[str] = []
    for n in names:
        if n.startswith("T_"):
            train.append(n)
        elif n.startswith("H_"):
            held.append(n)
        else:
            raise ValueError(f"split_v2: {n!r} has no T_/H_ prefix")
    return sorted(train), sorted(held)


def filter_by_joint_count(
    experiments_root: str,
    names: list[str],
    expected_n_joints: int,
) -> list[str]:
    """Keep only experiment dir names whose ``joint_list.txt`` has the expected count.

    Useful when arm (4 joints) and leg (6 joints) v2 experiments share an
    input root with overlapping ``T_*`` / ``H_*`` prefixes. Reading the
    joint list is O(N) over the candidate set and avoids loading the full
    state CSV just to discover a dimension mismatch.

    Args:
        experiments_root: Directory containing the per-experiment dirs.
        names: Candidate dir names (relative to ``experiments_root``).
        expected_n_joints: Required joint count; dirs whose
            ``joint_list.txt`` has any other count are dropped.

    Returns:
        Sorted list of names whose joint count matches.
    """
    keep: list[str] = []
    for name in names:
        jl = os.path.join(experiments_root, name, "joint_list.txt")
        if not os.path.isfile(jl):
            continue
        with open(jl) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) == expected_n_joints:
            keep.append(name)
    return sorted(keep)


def load_split_json(path: str) -> tuple[list[str], list[str]]:
    """Load a v2 train/test split manifest produced for arm sweeps.

    The manifest must contain ``train`` and ``test`` keys, each a non-empty
    list of experiment dir names (without the data root). Names must be
    disjoint across the two lists.

    Args:
        path: Path to the JSON manifest file.

    Returns:
        ``(train_names, test_names)`` — both alphabetically sorted.

    Raises:
        ValueError: when keys are missing, lists are empty, or names overlap.
    """
    with open(path) as f:
        manifest = json.load(f)
    for key in ("train", "test"):
        if key not in manifest:
            raise ValueError(f"{path}: manifest missing required key {key!r}")
        if not isinstance(manifest[key], list):
            raise ValueError(f"{path}: {key!r} must be a list, got {type(manifest[key]).__name__}")
        if not manifest[key]:
            raise ValueError(f"{path}: {key!r} list is empty")
    overlap = sorted(set(manifest["train"]) & set(manifest["test"]))
    if overlap:
        raise ValueError(f"{path}: train/test overlap on {overlap}")
    return sorted(manifest["train"]), sorted(manifest["test"])


# ----------------------------------------------------------------------------
# Leg v2 helpers (used by gen_multijoint_leg + warm_start_fulltorque_enriched_leg).
# Arm v2's list_experiments() and split_v2() are reused as-is.
# ----------------------------------------------------------------------------
LEG_PREFIXES: tuple[str, str] = ("T_", "H_")


def is_leg_experiment(name: str) -> bool:
    """Return True for v2 leg experiment dir names (``T_*`` or ``H_*``)."""
    return name.startswith(LEG_PREFIXES)
