"""Run configuration loader.

Loads per-robot YAML configs from input/run_configs/<robot>.yaml.
CLI arguments always override values set in the config file.
"""

import os

import yaml

_CONFIGS_DIR = os.path.dirname(__file__)


def load_run_cfg(robot_name: str) -> dict:
    """Load the full run config for a robot.

    Looks for ``<robot_name>/<robot_name>.yaml`` inside the run_configs
    directory (e.g. ``h1/h1.yaml``).  Falls back to the flat
    ``<robot_name>.yaml`` for backwards compatibility.

    Returns the parsed YAML dict, or {} if the file doesn't exist.
    """
    # New layout: h1/h1.yaml
    path = os.path.join(_CONFIGS_DIR, robot_name, f"{robot_name}.yaml")
    if not os.path.isfile(path):
        # Fallback: flat layout
        path = os.path.join(_CONFIGS_DIR, f"{robot_name}.yaml")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}
