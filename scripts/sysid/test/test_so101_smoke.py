# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-101 robot enablement smoke tests (no GPU required).

These tests load configs and the actuator template at the YAML/dict level
only - they do NOT require a SimulationApp, GPU, or `isaaclab` imports.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# scripts/sysid/test/ -> scripts/sysid -> scripts -> repo root
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_REPO, "input"))


# Hardcoded here (NOT imported from run_sysid.py) - run_sysid.py is not
# import-safe without a SimulationApp running.
SO101_JOINT_NAMES = [
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
]


def test_so101_run_config_loads() -> None:
    """`load_run_cfg("so101")` returns a non-empty config with the
    expected top-level keys."""
    from run_configs import load_run_cfg

    cfg = load_run_cfg("so101")
    assert cfg, "so101.yaml not found at input/run_configs/so101/so101.yaml"
    for key in ("simulation", "benchmark", "actuator", "sysid"):
        assert key in cfg, f"missing top-level key: {key}"


def test_so101_actuator_sysid_fit_loads() -> None:
    """`so101_implicit.yaml` parses and ships with the fitted SO-101
    balanced SysID values."""
    from actuator_models import load_actuator_params

    params = load_actuator_params("so101/so101_implicit.yaml")
    assert "stiffness" in params and "damping" in params, (
        "so101_implicit.yaml missing stiffness/damping"
    )
    assert params["effort_limit_sim"] == 3.35
    assert params["velocity_limit_sim"] == 30.0
    assert params["motor_lag_ms"] == 0.0

    expected = {
        "stiffness": {
            "Rotation": 48.469227,
            "Pitch": 45.524723,
            "Elbow": 11.299391,
            "Wrist_Pitch": 47.66135,
            "Wrist_Roll": 66.163048,
            "Jaw": 45.479378,
        },
        "damping": {
            "Rotation": 2.500566,
            "Pitch": 3.24012,
            "Elbow": 0.428969,
            "Wrist_Pitch": 3.836192,
            "Wrist_Roll": 2.504375,
            "Jaw": 2.820747,
        },
        "armature": {
            "Rotation": 0.079202,
            "Pitch": 0.072242,
            "Elbow": 0.038962,
            "Wrist_Pitch": 0.042249,
            "Wrist_Roll": 0.049878,
            "Jaw": 0.053242,
        },
        "dynamic_friction": {
            "Rotation": 0.350446,
            "Pitch": 0.228862,
            "Elbow": 0.346933,
            "Wrist_Pitch": 0.439954,
            "Wrist_Roll": 0.118686,
            "Jaw": 0.198292,
        },
        "viscous_friction": {
            "Rotation": 0.951254,
            "Pitch": 1.033873,
            "Elbow": 0.599379,
            "Wrist_Pitch": 0.960546,
            "Wrist_Roll": 1.493437,
            "Jaw": 0.839063,
        },
    }
    for param_name, values in expected.items():
        assert params[param_name] == values


def test_so101_sysid_bounds_loads() -> None:
    """`so101_sysid_bounds.yaml` parses with toolbox-conventional schema."""
    import yaml

    bounds_path = os.path.join(
        _REPO, "input", "run_configs", "so101", "so101_sysid_bounds.yaml"
    )
    with open(bounds_path) as f:
        bounds = yaml.safe_load(f)

    assert "joint_types" in bounds, "missing joint_types"
    assert sorted(bounds["joint_types"]) == sorted(SO101_JOINT_NAMES), (
        f"joint_types {bounds['joint_types']} does not match SO101_JOINT_NAMES"
    )
    assert "parameters" in bounds and "cmaes" in bounds
    for param in ("armature", "dynamic_friction", "viscous_friction"):
        assert param in bounds["parameters"], f"missing parameter {param}"
        assert "lower" in bounds["parameters"][param]
        assert "upper" in bounds["parameters"][param]


def test_so101_joint_configs_present() -> None:
    """sim2real_gap joint list configs exist with the expected joints."""
    import yaml

    yaml_path = os.path.join(
        _REPO, "scripts", "sim2real_gap", "configs", "so101_joints.yaml"
    )
    txt_path = os.path.join(
        _REPO, "scripts", "sim2real_gap", "configs", "so101_valid_joints.txt"
    )

    with open(yaml_path) as f:
        joints_yaml = yaml.safe_load(f)
    assert sorted(joints_yaml["joints"]) == sorted(SO101_JOINT_NAMES)

    with open(txt_path) as f:
        joints_txt = [line.strip() for line in f if line.strip()]
    assert sorted(joints_txt) == sorted(SO101_JOINT_NAMES)


def test_h1_run_config_still_loads() -> None:
    """Regression: existing h1 robot still works after our additions."""
    from run_configs import load_run_cfg

    cfg = load_run_cfg("h1")
    assert cfg, "h1.yaml not found"
    assert "simulation" in cfg
