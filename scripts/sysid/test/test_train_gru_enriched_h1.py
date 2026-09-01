# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the deployable H1 enriched residual training layout."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_TRAIN_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_model")
sys.path.insert(0, _TRAIN_MODEL_DIR)

from enriched_contract import (  # noqa: E402
    H1_DEPLOYABLE_FEATURE_LAYOUT,
    H1_LEGACY_FEATURE_LAYOUT,
    make_h1_deployable_metadata,
)
from enriched_normalization import save_stats  # noqa: E402
from train_gru_enriched_h1 import build_windowed  # noqa: E402


def _motion() -> dict[str, np.ndarray | list[str]]:
    position = np.arange(24, dtype=np.float32).reshape(6, 4) / 10.0
    velocity = np.full((6, 4), 0.2, dtype=np.float32)
    target = position + 0.1
    torque = np.arange(24, dtype=np.float32).reshape(6, 4)
    return {
        "positions": position,
        "velocities": velocity,
        "targets": target,
        "torques": torque,
        "joint_names": [
            "right_shoulder_pitch",
            "right_shoulder_roll",
            "right_shoulder_yaw",
            "right_elbow",
        ],
    }


def test_deployable20_omits_bias_and_uses_pd_residual_target() -> None:
    def unexpected_bias(*_args):
        raise AssertionError("deployable20 must not request qfrc_bias")

    motion = _motion()
    windows, targets = build_windowed(
        [("motion", motion)],
        window_len=3,
        bias_fn=unexpected_bias,
        residual_target=True,
        feature_layout=H1_DEPLOYABLE_FEATURE_LAYOUT,
    )

    assert windows.shape == (2, 3, 20)
    assert targets.shape == (2, 3, 4)
    np.testing.assert_allclose(windows[0, :, 0:4], motion["positions"][:3])
    np.testing.assert_allclose(windows[0, :, 4:8], 0.1, atol=1e-6)
    np.testing.assert_allclose(windows[0, :, 8:12], 0.2)
    np.testing.assert_allclose(windows[0, :, 12:16], 5.7, atol=1e-5)
    np.testing.assert_allclose(windows[0, 0, 16:20], 0.0)
    np.testing.assert_allclose(windows[0, 1:, 16:20], motion["torques"][:2])
    np.testing.assert_allclose(targets[0], motion["torques"][:3] - 5.7, atol=1e-5)


def test_legacy24_requires_and_inserts_bias_provider() -> None:
    bias = np.full((6, 4), 7.0, dtype=np.float32)
    windows, _ = build_windowed(
        [("motion", _motion())],
        window_len=3,
        bias_fn=lambda *_args: bias,
        residual_target=True,
        feature_layout=H1_LEGACY_FEATURE_LAYOUT,
    )
    assert windows.shape == (2, 3, 24)
    np.testing.assert_allclose(windows[0, :, 16:20], 7.0)

    with pytest.raises(ValueError, match="qfrc_bias provider"):
        build_windowed(
            [("motion", _motion())],
            window_len=3,
            residual_target=True,
            feature_layout=H1_LEGACY_FEATURE_LAYOUT,
        )


def test_stats_sidecar_persists_deployment_metadata(tmp_path) -> None:
    metadata = make_h1_deployable_metadata(
        hidden_size=128,
        num_layers=2,
        force_bound=5.0,
        kp=60.0,
        kd=1.5,
    )
    path = tmp_path / "stats.json"
    save_stats(
        {"mean": np.zeros(20, dtype=np.float32), "std": np.ones(20, dtype=np.float32)},
        str(path),
        metadata=metadata,
    )
    payload = json.loads(path.read_text())
    assert payload["metadata"]["feature_layout"] == H1_DEPLOYABLE_FEATURE_LAYOUT
    assert payload["metadata"]["input_size"] == 20
    assert payload["metadata"]["recommended_residual_scale"] == 0.33
