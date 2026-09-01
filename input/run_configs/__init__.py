# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run configuration loader.

Loads default per-robot YAML configs or an explicitly selected YAML path.
CLI arguments always override values set in the config file.
"""

import os

import yaml

_CONFIGS_DIR = os.path.dirname(__file__)


def load_run_cfg(robot_name: str, config_path: str | None = None) -> dict:
    """Load a robot run config or an explicitly selected YAML file.

    By default, looks for ``<robot_name>/<robot_name>.yaml`` inside the
    run-config directory and then the legacy flat ``<robot_name>.yaml``.
    When *config_path* is supplied, it is resolved from the current working
    directory (or used as an absolute path) and must exist.
    """
    if config_path is not None:
        path = os.path.abspath(os.path.expanduser(config_path))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Run config not found: {path}")
    else:
        path = os.path.join(_CONFIGS_DIR, robot_name, f"{robot_name}.yaml")
        if not os.path.isfile(path):
            path = os.path.join(_CONFIGS_DIR, f"{robot_name}.yaml")
        if not os.path.isfile(path):
            return {}

    with open(path) as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Run config must contain a YAML mapping: {path}")
    return config
