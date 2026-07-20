# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for objective-aware CMA-ES result and log schemas."""

from __future__ import annotations

import csv
import os
import sys

import pytest
import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from optimizer import CMAESOptimizer  # noqa: E402


def _make_optimizer(tmp_path, objective: str) -> CMAESOptimizer:
    config_path = tmp_path / "bounds.yaml"
    with open(config_path, "w") as file:
        yaml.safe_dump(
            {
                "joint_types": ["elbow", "wrist"],
                "mirror": False,
                "parameters": {"armature": {"lower": 0.0, "upper": 1.0}},
                "cmaes": {"sigma": 0.2, "max_iterations": 2, "epsilon": 0.01},
            },
            file,
        )
    optimizer = CMAESOptimizer(str(config_path), num_envs=4, device="cpu", objective=objective)
    optimizer._best_score = 0.25
    optimizer._best_params = torch.tensor([0.75, 0.6])
    optimizer.params[:] = torch.tensor([[0.75, 0.6], [0.5, 0.4], [0.4, 0.3], [0.3, 0.2]])
    optimizer.scores[:] = torch.tensor([0.25, 0.5, 0.6, 0.75])
    optimizer._score_steps = 1
    return optimizer


def test_mse_schema_remains_unchanged(tmp_path) -> None:
    optimizer = _make_optimizer(tmp_path, "mse")

    result = optimizer.get_best_params()
    log_path = tmp_path / "mse.csv"
    optimizer.log_generation(str(log_path))
    with open(log_path) as file:
        rows = list(csv.reader(file))

    assert result["best_mse"] == pytest.approx(0.25)
    assert "best_score" not in result
    assert rows[0][:4] == ["generation", "best_mse", "mean_mse", "min_mse"]


@pytest.mark.parametrize(
    ("objective", "metric_key"),
    (("wasserstein", "best_wasserstein"), ("mmd", "best_mmd_rff_squared")),
)
def test_distributional_schema_identifies_objective(tmp_path, objective: str, metric_key: str) -> None:
    optimizer = _make_optimizer(tmp_path, objective)

    result = optimizer.get_best_params()
    log_path = tmp_path / f"{objective}.csv"
    optimizer.log_generation(str(log_path))
    with open(log_path) as file:
        rows = list(csv.reader(file))

    assert result["objective"] == objective
    assert result["best_score"] == pytest.approx(0.25)
    assert result[metric_key] == pytest.approx(0.25)
    assert "best_mse" not in result
    assert rows[0][:5] == ["generation", "objective", "best_score", "mean_score", "min_score"]
    assert rows[1][1] == objective


def test_invalid_objective_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported objective"):
        _make_optimizer(tmp_path, "invalid")
