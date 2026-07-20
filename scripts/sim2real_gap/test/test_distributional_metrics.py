# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for distributional sim-to-real metrics."""

from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from distributional_metrics import (  # noqa: E402
    compute_batched_distributional_score,
    compute_feature_metrics,
    load_sage_joint_velocities,
    load_sage_motion_features,
    mmd_rff_squared,
    wasserstein_distance_1d,
)


def test_wasserstein_identity_shift_and_symmetry() -> None:
    reference = torch.tensor([-2.0, -0.5, 1.0, 3.0])
    shifted = reference + 2.5

    assert wasserstein_distance_1d(reference, reference) == pytest.approx(0.0)
    assert wasserstein_distance_1d(reference, shifted) == pytest.approx(2.5)
    assert wasserstein_distance_1d(shifted, reference) == pytest.approx(2.5)


def test_wasserstein_supports_unequal_and_permuted_samples() -> None:
    first = torch.tensor([0.0, 1.0])
    second = torch.tensor([2.0, 0.0, 1.0, 1.0])

    forward = wasserstein_distance_1d(first, second)
    permuted = wasserstein_distance_1d(first.flip(0), second[[2, 0, 3, 1]])

    assert forward == pytest.approx(0.5)
    assert permuted == pytest.approx(forward)


def test_feature_metrics_are_reproducible_and_detect_shift() -> None:
    reference = np.linspace(-1.0, 1.0, 257)
    identical = compute_feature_metrics(reference[::-1].copy(), reference, num_features=128, seed=7)
    shifted = compute_feature_metrics(reference + 0.75, reference, num_features=128, seed=7)
    shifted_again = compute_feature_metrics(reference + 0.75, reference, num_features=128, seed=7)

    assert identical.wasserstein_normalized == pytest.approx(0.0)
    assert identical.mmd_rff_squared == pytest.approx(0.0, abs=1.0e-12)
    assert shifted.wasserstein_raw == pytest.approx(0.75)
    assert shifted.mmd_rff_squared > 0.0
    assert shifted.mmd_rff_squared == pytest.approx(shifted_again.mmd_rff_squared)


def test_mmd_is_batched_symmetric_and_non_negative() -> None:
    reference = torch.linspace(-1.0, 1.0, 101)
    simulations = torch.stack((reference, reference + 0.5))

    scores, bandwidth = mmd_rff_squared(simulations, reference, num_features=64, seed=3, chunk_size=17)
    reverse, _ = mmd_rff_squared(reference + 0.5, reference, bandwidth=bandwidth, num_features=64, seed=3)

    assert scores.shape == (2,)
    assert torch.all(scores >= 0.0)
    assert scores[0] == pytest.approx(0.0, abs=1.0e-12)
    assert scores[1] == pytest.approx(reverse[0])


def test_batched_objectives_average_position_and_velocity() -> None:
    real_position = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    real_velocity = torch.tensor([[0.0], [0.5], [1.0], [1.5]])
    sim_position = torch.stack((real_position, real_position + 1.0))
    sim_velocity = torch.stack((real_velocity, real_velocity + 0.5))

    wasserstein = compute_batched_distributional_score(
        sim_position,
        real_position,
        sim_velocity,
        real_velocity,
        objective="wasserstein",
    )
    mmd = compute_batched_distributional_score(
        sim_position,
        real_position,
        sim_velocity,
        real_velocity,
        objective="mmd",
        num_features=64,
        seed=2,
        chunk_size=2,
    )

    assert wasserstein[0] == pytest.approx(0.0)
    assert wasserstein[1] > 0.0
    assert mmd[0] == pytest.approx(0.0, abs=1.0e-12)
    assert mmd[1] > 0.0


def test_metrics_reject_empty_and_non_finite_samples() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        wasserstein_distance_1d(torch.tensor([]), torch.tensor([0.0]))
    with pytest.raises(ValueError, match="non-finite"):
        compute_feature_metrics(np.array([np.nan]), np.array([0.0]))


def _write_sage_motion(path: str, *, timestamps: list[float], microseconds: bool) -> None:
    with open(os.path.join(path, "joint_list.txt"), "w") as file:
        file.write("joint_b\njoint_a\n")

    scale = 1.0e6 if microseconds else 1.0
    with open(os.path.join(path, "state_motor.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["type", "timestamp", "positions", "velocities", "torques"])
        for index, timestamp in enumerate(timestamps):
            writer.writerow(["STATE_MOTOR", timestamp * scale, [index, index + 10], [index + 20, index + 30], [0, 0]])

    with open(os.path.join(path, "control.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["type", "timestamp", "positions"])
        for index, timestamp in enumerate(timestamps):
            writer.writerow(["CONTROL", timestamp * scale, [index + 40, index + 50]])

    with open(os.path.join(path, "event.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["type", "timestamp", "event"])
        writer.writerow(["EVENT", timestamps[1] * scale, "MOTION_START"])
        writer.writerow(["EVENT", timestamps[-2] * scale, "DISABLE"])


@pytest.mark.parametrize("microseconds", [False, True])
def test_sage_loader_preserves_joint_order_and_crops_event_window(tmp_path, microseconds: bool) -> None:
    _write_sage_motion(str(tmp_path), timestamps=[0.0, 0.5, 1.0, 1.5], microseconds=microseconds)

    features = load_sage_motion_features(str(tmp_path))

    assert features.joint_names == ("joint_b", "joint_a")
    assert np.array_equal(features.positions, [[1.0, 11.0], [2.0, 12.0]])
    assert np.array_equal(features.velocities, [[21.0, 31.0], [22.0, 32.0]])
    assert np.array_equal(features.commands, [[41.0, 51.0], [42.0, 52.0]])
    raw_velocities = load_sage_joint_velocities(str(tmp_path), ["joint_a"])
    assert np.array_equal(raw_velocities[:, 0], [30.0, 31.0, 32.0, 33.0])
