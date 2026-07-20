# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for distributional SAGE report generation."""

from __future__ import annotations

import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from distributional_report import compute_motion_source_rows, write_distributional_metrics_csv  # noqa: E402


def _write_motion(path: str, offset: float) -> None:
    os.makedirs(path)
    with open(os.path.join(path, "joint_list.txt"), "w") as file:
        file.write("joint_0\njoint_1\n")
    with open(os.path.join(path, "state_motor.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["type", "timestamp", "positions", "velocities", "torques"])
        for index in range(5):
            writer.writerow(
                [
                    "STATE_MOTOR",
                    index * 0.1,
                    [index + offset, index + 1.0 + offset],
                    [index * 0.5 + offset, index * 0.25 + offset],
                    [0.0, 0.0],
                ]
            )
    with open(os.path.join(path, "control.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["type", "timestamp", "positions"])
        for index in range(5):
            writer.writerow(["CONTROL", index * 0.1, [index, index + 1.0]])


def test_report_contains_feature_motion_group_and_overall_rows(tmp_path) -> None:
    result_folder = tmp_path / "benchmark"
    sim_motion = result_folder / "sim" / "robot" / "custom" / "motion_implicit"
    real_motion = result_folder / "real" / "robot" / "custom" / "motion_implicit"
    _write_motion(str(sim_motion), offset=0.5)
    _write_motion(str(real_motion), offset=0.0)

    rows = compute_motion_source_rows(
        result_folder=str(result_folder),
        robot_name="robot",
        motion_source="custom",
        motion_names="motion_implicit",
        num_features=64,
        seed=4,
        chunk_size=2,
    )

    feature_rows = [row for row in rows if row["scope"] == "feature"]
    assert len(feature_rows) == 6
    assert {row["feature"] for row in feature_rows} == {"position", "velocity", "command"}
    assert all(row["included_in_aggregate"] is False for row in feature_rows if row["feature"] == "command")
    assert [row["scope"] for row in rows[-2:]] == ["motion", "group"]
    assert rows[-1]["wasserstein_normalized"] > 0.0

    output_path = tmp_path / "analysis" / "distributional_metrics.csv"
    write_distributional_metrics_csv(str(output_path), rows)
    with open(output_path) as file:
        output_rows = list(csv.DictReader(file))

    assert output_rows[-1]["scope"] == "overall"
    assert output_rows[-1]["motion_source"] == "*"
