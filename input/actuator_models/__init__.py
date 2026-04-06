# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actuator model configs and loader utilities.

Isaac Lab / Omniverse modules must not be imported at module level — they
require the SimulationApp to be instantiated first. All isaaclab imports
are deferred to function bodies.
"""

import os

import yaml

_ACTUATOR_MODELS_DIR = os.path.dirname(__file__)


def load_actuator_params(yaml_file: str) -> dict:
    """Load raw actuator parameters from a YAML file.

    Returns the full dict including non-physics keys like motor_lag_ms.

    Args:
        yaml_file: Filename (relative to actuator_models/) or absolute path.
    """
    if not os.path.isabs(yaml_file):
        yaml_file = os.path.join(_ACTUATOR_MODELS_DIR, yaml_file)

    with open(yaml_file) as f:
        return yaml.safe_load(f)


def _filter_param(value, joint_names_expr: list[str]):
    """Filter a dict-valued parameter to only include patterns matching joint_names_expr.

    If value is a scalar or None, returns as-is. If value is a dict with regex keys,
    keeps only entries whose pattern could match one of the joint_names_expr patterns.
    This prevents Isaac Lab from raising ValueError for unmatched regex patterns
    when a per-joint YAML is shared across multiple single-joint actuator groups.
    """
    import re

    if not isinstance(value, dict) or not joint_names_expr:
        return value
    # Heuristic: both YAML keys and joint_names_expr use ".*_{suffix}" patterns.
    # We match on the suffix portion. This assumes joint names don't have
    # overlapping suffixes (e.g. "elbow" doesn't appear in "elbow_flex").
    filtered = {}
    for pattern, v in value.items():
        # Check if this YAML pattern could match any joint covered by joint_names_expr.
        # Both are regexes matching joint names like "right_shoulder_pitch".
        # A simple heuristic: extract the suffix from the YAML pattern (e.g. "shoulder_pitch"
        # from ".*_shoulder_pitch") and check if any joint_names_expr contains it.
        for jne in joint_names_expr:
            # If the core part of the pattern appears in the joint expression, keep it
            pattern_core = pattern.replace(".*", "").strip("_")
            jne_core = jne.replace(".*", "").strip("_")
            if pattern_core == jne_core or re.search(pattern_core, jne_core) or re.search(jne_core, pattern_core):
                filtered[pattern] = v
                break
    return filtered if filtered else value


def load_implicit_actuator_cfg(yaml_file: str, joint_names_expr: list[str]):
    """Load an ImplicitActuatorCfg from a YAML file.

    Parameters may be scalars (applied to all joints) or dicts with regex
    keys for per-joint values::

        stiffness: 60.0                   # scalar — same for all joints
        armature:                          # dict — per-joint values
          ".*_elbow": 0.018
          ".*_shoulder_pitch": 0.020

    Args:
        yaml_file: Filename (relative to actuator_models/) or absolute path.
        joint_names_expr: Joint name regex patterns for the actuator group.

    Returns:
        An ImplicitActuatorCfg populated from the YAML parameters.
    """
    from isaaclab.actuators import ImplicitActuatorCfg

    if not os.path.isabs(yaml_file):
        yaml_file = os.path.join(_ACTUATOR_MODELS_DIR, yaml_file)

    with open(yaml_file) as f:
        params = yaml.safe_load(f)

    return ImplicitActuatorCfg(
        joint_names_expr=joint_names_expr,
        effort_limit_sim=params.get("effort_limit_sim"),
        velocity_limit_sim=params.get("velocity_limit_sim"),
        stiffness=_filter_param(params["stiffness"], joint_names_expr),
        damping=_filter_param(params["damping"], joint_names_expr),
        armature=_filter_param(params.get("armature"), joint_names_expr),
        friction=_filter_param(params.get("friction"), joint_names_expr),
        dynamic_friction=_filter_param(params.get("dynamic_friction"), joint_names_expr),
        viscous_friction=_filter_param(params.get("viscous_friction"), joint_names_expr),
    )


def load_dc_motor_cfg(yaml_file: str, joint_names_expr: list[str]):
    """Load a DCMotorCfg from a YAML file.

    Parameters may be scalars (applied to all joints) or dicts with regex
    keys for per-joint values (same format as load_implicit_actuator_cfg).

    Args:
        yaml_file: Filename (relative to actuator_models/) or absolute path.
        joint_names_expr: Joint name regex patterns for the actuator group.

    Returns:
        A DCMotorCfg populated from the YAML parameters.
    """
    from isaaclab.actuators import DCMotorCfg

    if not os.path.isabs(yaml_file):
        yaml_file = os.path.join(_ACTUATOR_MODELS_DIR, yaml_file)

    with open(yaml_file) as f:
        params = yaml.safe_load(f)

    return DCMotorCfg(
        joint_names_expr=joint_names_expr,
        saturation_effort=params["saturation_effort"],
        effort_limit=params["effort_limit"],
        velocity_limit=params["velocity_limit"],
        armature=params.get("armature"),
        friction=params.get("friction"),
        stiffness=params["stiffness"],
        damping=params["damping"],
    )


def load_fmu_actuator_cfg(
    yaml_file: str,
    fmu_path: str,
    joint_names_expr: list[str],
    fmu_step_size: float = 0.002,
):
    """Load an ActuatorNetFMUCfg from a YAML file and FMU path.

    The YAML provides PD gains and limits (used for the internal PD torque
    prediction). The FMU path points to a CoSimulation FMU (e.g. Nvidia
    neural-ODE motor model) that corrects the PD output.

    Args:
        yaml_file: Filename (relative to actuator_models/) or absolute path.
        fmu_path: Path to FMU directory or .fmu archive (relative to
            actuator_models/ or absolute).
        joint_names_expr: Joint name regex patterns for the actuator group.
        fmu_step_size: Communication step size [s] for the FMU solver.

    Returns:
        An ActuatorNetFMUCfg populated from the YAML and FMU path.
    """
    from actuator_fmu import ActuatorNetFMUCfg

    if not os.path.isabs(yaml_file):
        yaml_file = os.path.join(_ACTUATOR_MODELS_DIR, yaml_file)
    if not os.path.isabs(fmu_path):
        fmu_path = os.path.join(_ACTUATOR_MODELS_DIR, fmu_path)

    with open(yaml_file) as f:
        params = yaml.safe_load(f)

    return ActuatorNetFMUCfg(
        joint_names_expr=joint_names_expr,
        stiffness=params["stiffness"],
        damping=params["damping"],
        effort_limit=params.get("effort_limit", params.get("effort_limit_sim")),
        velocity_limit=params.get("velocity_limit", params.get("velocity_limit_sim")),
        armature=params.get("armature"),
        friction=params.get("friction"),
        fmu_path=fmu_path,
        fmu_step_size=fmu_step_size,
    )
