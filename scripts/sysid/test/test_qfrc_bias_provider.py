# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the qfrc_bias provider and vendored resampler in ``train_model/``.

These cover the paths that need neither MuJoCo nor the trajectory datasets: the
pure-numpy ``_resample_to_grid`` helper and the ``make_bias_provider`` cache /
validation branches (``model_xml=None``).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_TRAIN_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train_model")
sys.path.insert(0, _TRAIN_MODEL_DIR)

from enriched_data_g1 import _resample_to_grid  # noqa: E402
from precompute_qfrc_bias import make_bias_provider  # noqa: E402


def test_resample_to_grid_linear_interpolation():
    ts = np.array([0.0, 1.0, 2.0])
    data = np.array([[0.0, 100.0], [10.0, 200.0], [20.0, 300.0]])
    grid = np.array([0.5, 1.5])
    out = _resample_to_grid(ts, data, grid)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[:, 0], [5.0, 15.0])
    np.testing.assert_allclose(out[:, 1], [150.0, 250.0])


def test_provider_cache_hit_returns_cache(tmp_path):
    name = "motion_a"
    bias = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(tmp_path / f"{name}_qfrc_bias.npy", bias)
    provider = make_bias_provider(model_xml=None, joint_names=[], cache_root=str(tmp_path))
    pos = np.zeros((3, 4), dtype=np.float32)
    out = provider(name, pos, pos)
    np.testing.assert_array_equal(out, bias)
    assert out.dtype == np.float32


def test_provider_stale_cache_length_mismatch_raises(tmp_path):
    name = "motion_b"
    np.save(tmp_path / f"{name}_qfrc_bias.npy", np.zeros((5, 4), dtype=np.float32))
    provider = make_bias_provider(model_xml=None, joint_names=[], cache_root=str(tmp_path))
    pos = np.zeros((3, 4), dtype=np.float32)  # traj length 3 != cached length 5
    with pytest.raises(ValueError, match=r"qfrc_bias len 5 != traj len 3"):
        provider(name, pos, pos)


def test_provider_no_cache_no_model_raises(tmp_path):
    provider = make_bias_provider(model_xml=None, joint_names=[], cache_root=str(tmp_path))
    pos = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(FileNotFoundError):
        provider("missing_motion", pos, pos)
