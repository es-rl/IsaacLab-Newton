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

import os

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
# scripts/sim2real_gap/test/ -> scripts/sim2real_gap -> scripts -> repo root
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))

# Hardcoded here (NOT imported from newton_benchmark.py) - that module is
# not import-safe without a SimulationApp running.
SO101_JOINT_NAMES = [
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
]


def test_so101_registered_in_benchmark_dispatch() -> None:
    """``so101`` is registered in ``_BENCHMARK_ROBOT_CONFIGS`` and points at
    ``So101BenchmarkSceneCfg``."""
    bench_path = os.path.join(_REPO, "scripts", "sim2real_gap", "newton_benchmark.py")
    with open(bench_path) as f:
        source = f.read()

    assert "class So101BenchmarkSceneCfg" in source, "newton_benchmark.py is missing So101BenchmarkSceneCfg"
    assert '"so101":' in source, "newton_benchmark.py is missing the 'so101' dispatch key"
    assert "So101BenchmarkSceneCfg," in source, "_BENCHMARK_ROBOT_CONFIGS does not reference So101BenchmarkSceneCfg"


def test_so101_benchmark_usd_exists() -> None:
    """The SO-101 USD asset referenced by the benchmark scene cfg is present."""
    usd_path = os.path.join(_REPO, "input", "robot_models", "so101", "so101.usd")
    assert os.path.isfile(usd_path), f"missing SO-101 USD: {usd_path}"


def test_so101_sage_configs_exist() -> None:
    """``configs/so101_joints.yaml`` and ``so101_valid_joints.txt`` exist
    and list the six USD joint names in order."""
    yaml_path = os.path.join(_REPO, "scripts", "sim2real_gap", "configs", "so101_joints.yaml")
    txt_path = os.path.join(_REPO, "scripts", "sim2real_gap", "configs", "so101_valid_joints.txt")
    assert os.path.isfile(yaml_path), f"missing {yaml_path}"
    assert os.path.isfile(txt_path), f"missing {txt_path}"

    with open(yaml_path) as f:
        joints_yaml = yaml.safe_load(f)
    assert joints_yaml.get("joints") == SO101_JOINT_NAMES, (
        f"so101_joints.yaml joints {joints_yaml.get('joints')} does not match expected order {SO101_JOINT_NAMES}"
    )

    with open(txt_path) as f:
        joints_txt = [line.strip() for line in f if line.strip()]
    assert joints_txt == SO101_JOINT_NAMES, (
        f"so101_valid_joints.txt {joints_txt} does not match expected order {SO101_JOINT_NAMES}"
    )
