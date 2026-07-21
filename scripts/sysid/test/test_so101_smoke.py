# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-101 robot enablement smoke tests (no GPU required).

These tests load configs and the actuator template at the YAML/dict level
only - they do NOT require a SimulationApp, GPU, or `isaaclab` imports.
"""

from __future__ import annotations

import ast
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# scripts/sysid/test/ -> scripts/sysid -> scripts -> repo root
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_REPO, "input"))


_SYSID_PATH = os.path.join(_REPO, "scripts", "sysid", "run_sysid.py")

# Legacy data labels retained in joint_list.txt and the SysID bounds.
SO101_DATA_JOINT_NAMES = [
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
]

SO101_JOINT_NAME_MAP = {
    "Rotation": "shoulder_pan",
    "Pitch": "shoulder_lift",
    "Elbow": "elbow_flex",
    "Wrist_Pitch": "wrist_flex",
    "Wrist_Roll": "wrist_roll",
    "Jaw": "gripper",
}


def test_so101_run_config_loads() -> None:
    """`load_run_cfg("so101")` returns a non-empty config with the
    expected top-level keys."""
    from run_configs import load_run_cfg

    cfg = load_run_cfg("so101")
    assert cfg, "so101.yaml not found at input/run_configs/so101/so101.yaml"
    for key in ("simulation", "benchmark", "actuator", "sysid"):
        assert key in cfg, f"missing top-level key: {key}"


def test_so101_actuator_template_loads() -> None:
    """`so101_implicit.yaml` parses and ships as a clean template (no
    fitted values)."""
    from actuator_models import load_actuator_params

    params = load_actuator_params("so101/so101_implicit.yaml")
    assert "stiffness" in params and "damping" in params, "so101_implicit.yaml missing stiffness/damping"
    # Template must ship with zero friction/armature so customer's CMA-ES
    # fit is not biased by stale fitted values.
    for key in ("armature", "dynamic_friction", "viscous_friction"):
        block = params.get(key, {})
        if isinstance(block, dict):
            for joint, value in block.items():
                assert value == 0.0, (
                    f"so101_implicit.yaml shipping with non-zero {key}[{joint}]={value} - "
                    "template must be clean (zero friction/armature)"
                )
        else:
            assert block == 0.0, f"so101_implicit.yaml shipping with non-zero {key}={block} - template must be clean"


def test_so101_sysid_bounds_loads() -> None:
    """`so101_sysid_bounds.yaml` parses with toolbox-conventional schema."""
    import yaml

    bounds_path = os.path.join(_REPO, "input", "run_configs", "so101", "so101_sysid_bounds.yaml")
    with open(bounds_path) as f:
        bounds = yaml.safe_load(f)

    assert "joint_types" in bounds, "missing joint_types"
    assert sorted(bounds["joint_types"]) == sorted(SO101_DATA_JOINT_NAMES), (
        f"joint_types {bounds['joint_types']} does not match SO101_DATA_JOINT_NAMES"
    )
    assert "parameters" in bounds and "cmaes" in bounds
    for param in ("armature", "dynamic_friction", "viscous_friction"):
        assert param in bounds["parameters"], f"missing parameter {param}"
        assert "lower" in bounds["parameters"][param]
        assert "upper" in bounds["parameters"][param]


def test_so101_joint_configs_present() -> None:
    """sim2real_gap joint list configs exist with the expected joints."""
    import yaml

    yaml_path = os.path.join(_REPO, "scripts", "sim2real_gap", "configs", "so101_joints.yaml")
    txt_path = os.path.join(_REPO, "scripts", "sim2real_gap", "configs", "so101_valid_joints.txt")

    with open(yaml_path) as f:
        joints_yaml = yaml.safe_load(f)
    assert sorted(joints_yaml["joints"]) == sorted(SO101_DATA_JOINT_NAMES)

    with open(txt_path) as f:
        joints_txt = [line.strip() for line in f if line.strip()]
    assert sorted(joints_txt) == sorted(SO101_DATA_JOINT_NAMES)


def test_so101_sysid_maps_legacy_data_names_to_usd_prims() -> None:
    """The SysID runtime preserves old data labels while targeting new USD prims."""
    with open(_SYSID_PATH) as file:
        tree = ast.parse(file.read())

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "SO101_JOINT_NAME_MAP" for target in node.targets):
            assert ast.literal_eval(node.value) == SO101_JOINT_NAME_MAP
            return
    raise AssertionError("Could not find SO101_JOINT_NAME_MAP in run_sysid.py")


def test_h1_run_config_still_loads() -> None:
    """Regression: existing h1 robot still works after our additions."""
    from run_configs import load_run_cfg

    cfg = load_run_cfg("h1")
    assert cfg, "h1.yaml not found"
    assert "simulation" in cfg
