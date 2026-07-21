# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-101 benchmark dispatch smoke tests (no GPU required).

These tests verify the SO-101 wiring in
``scripts/sim2real_gap/newton_benchmark.py`` at the text/file level only -
they do NOT import the module (which triggers ``AppLauncher``) and do NOT
require a SimulationApp or GPU.
"""

from __future__ import annotations

import ast
import os
import re

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
# scripts/sim2real_gap/test/ -> scripts/sim2real_gap -> scripts -> repo root
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

# Hardcoded here (NOT imported from newton_benchmark.py) - that module is
# not import-safe without a SimulationApp running.
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


_BENCH_PATH = os.path.join(_REPO, "scripts", "sim2real_gap", "newton_benchmark.py")

# Match the so101 entry inside _BENCHMARK_ROBOT_CONFIGS: the dict key,
# its scene_cfg_cls value, and the actuator_yaml string.
_DISPATCH_ENTRY_RE = re.compile(
    r'"so101":\s*\{\s*'
    r'"scene_cfg_cls":\s*So101BenchmarkSceneCfg,\s*'
    r'"actuator_yaml":\s*"(?P<yaml>[^"]+)"',
    re.DOTALL,
)

# Match the actuator yaml argument inside So101BenchmarkSceneCfg's
# load_implicit_actuator_cfg(...) call.
_SCENE_ACTUATOR_RE = re.compile(
    r"class So101BenchmarkSceneCfg.*?"
    r'load_implicit_actuator_cfg\(\s*"(?P<yaml>[^"]+)"',
    re.DOTALL,
)


def _load_literal_assignment(name: str):
    with open(_BENCH_PATH) as file:
        tree = ast.parse(file.read())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Could not find literal assignment for {name}")


def test_so101_registered_in_benchmark_dispatch() -> None:
    """``so101`` is registered in ``_BENCHMARK_ROBOT_CONFIGS`` and points at
    ``So101BenchmarkSceneCfg`` with a well-formed actuator_yaml entry."""
    with open(_BENCH_PATH) as f:
        source = f.read()

    assert "class So101BenchmarkSceneCfg" in source, "newton_benchmark.py is missing So101BenchmarkSceneCfg"
    match = _DISPATCH_ENTRY_RE.search(source)
    assert match, "_BENCHMARK_ROBOT_CONFIGS does not contain a well-formed 'so101' entry"


def test_so101_dispatch_actuator_yaml_matches_scene_cfg() -> None:
    """The actuator YAML referenced inside ``So101BenchmarkSceneCfg`` and the
    one registered in ``_BENCHMARK_ROBOT_CONFIGS["so101"]["actuator_yaml"]``
    must agree.

    Catches drift if either site is renamed without updating the other.
    """
    with open(_BENCH_PATH) as f:
        source = f.read()

    dispatch_match = _DISPATCH_ENTRY_RE.search(source)
    scene_match = _SCENE_ACTUATOR_RE.search(source)
    assert dispatch_match, "could not find so101 dispatch entry"
    assert scene_match, "could not find load_implicit_actuator_cfg call inside So101BenchmarkSceneCfg"

    assert dispatch_match.group("yaml") == scene_match.group("yaml"), (
        f"actuator_yaml drift: dispatch entry says '{dispatch_match.group('yaml')}' "
        f"but So101BenchmarkSceneCfg loads '{scene_match.group('yaml')}'"
    )


def test_so101_benchmark_usd_exists() -> None:
    """The SO-101 USD asset referenced by the benchmark scene cfg is present."""
    usd_path = os.path.join(_REPO, "input", "robot_models", "so101", "so101_no_camera_new_calib.usd")
    assert os.path.isfile(usd_path), f"missing SO-101 USD: {usd_path}"

    with open(_BENCH_PATH) as file:
        source = file.read()
    assert "so101_no_camera_new_calib.usd" in source


def test_so101_benchmark_maps_legacy_data_names_to_usd_prims() -> None:
    """Legacy recording labels map to the calibrated USD joint prims."""
    assert _load_literal_assignment("_SO101_JOINT_NAME_MAP") == SO101_JOINT_NAME_MAP


def test_so101_sage_configs_exist() -> None:
    """``configs/so101_joints.yaml`` and ``so101_valid_joints.txt`` exist
    and list the six legacy recording labels in order."""
    yaml_path = os.path.join(_REPO, "scripts", "sim2real_gap", "configs", "so101_joints.yaml")
    txt_path = os.path.join(_REPO, "scripts", "sim2real_gap", "configs", "so101_valid_joints.txt")
    assert os.path.isfile(yaml_path), f"missing {yaml_path}"
    assert os.path.isfile(txt_path), f"missing {txt_path}"

    with open(yaml_path) as f:
        joints_yaml = yaml.safe_load(f)
    assert joints_yaml.get("joints") == SO101_DATA_JOINT_NAMES, (
        f"so101_joints.yaml joints {joints_yaml.get('joints')} does not match expected order {SO101_DATA_JOINT_NAMES}"
    )

    with open(txt_path) as f:
        joints_txt = [line.strip() for line in f if line.strip()]
    assert joints_txt == SO101_DATA_JOINT_NAMES, (
        f"so101_valid_joints.txt {joints_txt} does not match expected order {SO101_DATA_JOINT_NAMES}"
    )
