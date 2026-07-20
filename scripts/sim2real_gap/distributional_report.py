# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate distributional sim-to-real reports from matching SAGE motions."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable

import numpy as np
from distributional_metrics import compute_feature_metrics, load_sage_motion_features

DISTRIBUTIONAL_FIELDNAMES = (
    "scope",
    "robot_name",
    "motion_source",
    "motion_name",
    "joint_name",
    "feature",
    "included_in_aggregate",
    "sim_samples",
    "real_samples",
    "wasserstein_raw",
    "wasserstein_normalized",
    "mmd_rff_squared",
    "mmd_bandwidth",
    "real_mean",
    "real_scale",
)


def _aggregate_row(rows: list[dict[str, object]], *, scope: str, **identity: object) -> dict[str, object]:
    included = [row for row in rows if row["included_in_aggregate"] is True]
    if not included:
        raise ValueError(f"Cannot create {scope} aggregate without position or velocity metrics")
    return {
        "scope": scope,
        **identity,
        "joint_name": "*",
        "feature": "all",
        "included_in_aggregate": True,
        "sim_samples": "",
        "real_samples": "",
        "wasserstein_raw": "",
        "wasserstein_normalized": float(np.mean([row["wasserstein_normalized"] for row in included])),
        "mmd_rff_squared": float(np.mean([row["mmd_rff_squared"] for row in included])),
        "mmd_bandwidth": "",
        "real_mean": "",
        "real_scale": "",
    }


def compute_motion_source_rows(
    *,
    result_folder: str,
    robot_name: str,
    motion_source: str,
    motion_names: str,
    num_features: int = 256,
    seed: int = 0,
    chunk_size: int = 1024,
) -> list[dict[str, object]]:
    """Compute detailed and aggregate distributional rows for one motion source."""
    sim_root = os.path.join(result_folder, "sim", robot_name, motion_source)
    real_root = os.path.join(result_folder, "real", robot_name, motion_source)
    if motion_names == "*":
        selected_motions = sorted(
            name for name in os.listdir(sim_root) if os.path.isfile(os.path.join(sim_root, name, "state_motor.csv"))
        )
    else:
        selected_motions = [name.strip() for name in motion_names.split(",") if name.strip()]

    all_rows: list[dict[str, object]] = []
    motion_aggregates: list[dict[str, object]] = []
    for motion_name in selected_motions:
        sim_dir = os.path.join(sim_root, motion_name)
        real_dir = os.path.join(real_root, motion_name)
        if not os.path.isdir(sim_dir) or not os.path.isdir(real_dir):
            raise FileNotFoundError(f"Missing matching sim/real motion directories for {motion_source}/{motion_name}")

        sim_features = load_sage_motion_features(sim_dir)
        real_features = load_sage_motion_features(real_dir)
        real_joint_indices = {name: index for index, name in enumerate(real_features.joint_names)}
        common_joints = [name for name in sim_features.joint_names if name in real_joint_indices]
        if not common_joints:
            raise ValueError(f"No common joints for {motion_source}/{motion_name}")

        sim_joint_indices = {name: index for index, name in enumerate(sim_features.joint_names)}
        motion_rows: list[dict[str, object]] = []
        feature_pairs = (
            ("position", sim_features.positions, real_features.positions, True),
            ("velocity", sim_features.velocities, real_features.velocities, True),
        )
        if sim_features.commands is not None and real_features.commands is not None:
            feature_pairs += (("command", sim_features.commands, real_features.commands, False),)

        for joint_offset, joint_name in enumerate(common_joints):
            sim_index = sim_joint_indices[joint_name]
            real_index = real_joint_indices[joint_name]
            for feature_offset, (feature_name, sim_values, real_values, included) in enumerate(feature_pairs):
                metrics = compute_feature_metrics(
                    sim_values[:, sim_index],
                    real_values[:, real_index],
                    num_features=num_features,
                    seed=seed + joint_offset * len(feature_pairs) + feature_offset,
                    chunk_size=chunk_size,
                )
                motion_rows.append(
                    {
                        "scope": "feature",
                        "robot_name": robot_name,
                        "motion_source": motion_source,
                        "motion_name": motion_name,
                        "joint_name": joint_name,
                        "feature": feature_name,
                        "included_in_aggregate": included,
                        "sim_samples": metrics.sim_samples,
                        "real_samples": metrics.real_samples,
                        "wasserstein_raw": metrics.wasserstein_raw,
                        "wasserstein_normalized": metrics.wasserstein_normalized,
                        "mmd_rff_squared": metrics.mmd_rff_squared,
                        "mmd_bandwidth": metrics.bandwidth,
                        "real_mean": metrics.real_mean,
                        "real_scale": metrics.real_scale,
                    }
                )

        motion_aggregate = _aggregate_row(
            motion_rows,
            scope="motion",
            robot_name=robot_name,
            motion_source=motion_source,
            motion_name=motion_name,
        )
        all_rows.extend(motion_rows)
        all_rows.append(motion_aggregate)
        motion_aggregates.append(motion_aggregate)

    if motion_aggregates:
        all_rows.append(
            _aggregate_row(
                motion_aggregates,
                scope="group",
                robot_name=robot_name,
                motion_source=motion_source,
                motion_name="*",
            )
        )
    return all_rows


def write_distributional_metrics_csv(path: str, rows: Iterable[dict[str, object]]) -> None:
    """Write stable detail/group/overall distributional metrics to CSV."""
    rows = list(rows)
    group_rows = [row for row in rows if row["scope"] == "group"]
    if group_rows:
        rows.append(
            _aggregate_row(
                group_rows,
                scope="overall",
                robot_name=group_rows[0]["robot_name"],
                motion_source="*",
                motion_name="*",
            )
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DISTRIBUTIONAL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
