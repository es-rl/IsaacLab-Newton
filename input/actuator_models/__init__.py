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
        stiffness=params["stiffness"],
        damping=params["damping"],
        armature=params.get("armature"),
        friction=params.get("friction"),
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
