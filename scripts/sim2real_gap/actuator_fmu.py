# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FMU-based hybrid actuator model.

Wraps an Ansys Twin Builder FMU (Functional Mock-up Unit) exported as an
FMI 2.0 ModelExchange model.  The FMU takes joint position, velocity, and
a PD-predicted torque as inputs, and outputs a corrected "true" torque.

Requires the ``fmpy`` package (``pip install fmpy``).
"""

from __future__ import annotations

import ctypes
import logging
import os
from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators.actuator_pd import IdealPDActuator
from isaaclab.actuators.actuator_pd_cfg import IdealPDActuatorCfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@configclass
class ActuatorNetFMUCfg(IdealPDActuatorCfg):
    """Configuration for an FMU-based hybrid actuator model.

    The FMU is expected to follow the Ansys Twin Builder convention:

    * **Inputs**: ``position`` [rad], ``velocity`` [rad/s], ``torque_pred`` [N-m]
    * **Output**: ``torque_true`` [N-m]
    """

    class_type: type | str = "scripts.sim2real_gap.actuator_fmu:ActuatorNetFMU"

    fmu_path: str = MISSING
    """Path to the FMU directory (unzipped) or ``.fmu`` archive."""

    fmu_step_size: float = 0.002
    """Communication step size [s] for the FMU solver. Should match the
    physics dt of the simulation."""


# ---------------------------------------------------------------------------
# Actuator
# ---------------------------------------------------------------------------

class ActuatorNetFMU(IdealPDActuator):
    """Hybrid actuator that corrects PD torques through an FMU model.

    The compute flow is:

    1. Compute PD torque the same way as :class:`IdealPDActuator`.
    2. Feed ``(position, velocity, torque_pred)`` into the FMU.
    3. Use the FMU output ``torque_true`` as the applied effort.

    One FMU instance is created per joint. Each instance maintains its own
    internal solver state across timesteps (the FMU is stateful / dynamic).
    """

    cfg: ActuatorNetFMUCfg
    """The configuration of the actuator model."""

    def __init__(self, cfg: ActuatorNetFMUCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        try:
            import fmpy
            from fmpy.fmi2 import FMU2Model  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"fmpy is required for ActuatorNetFMU. Install with: pip install fmpy\n"
                f"Original error: {e}"
            ) from e

        self._fmpy = fmpy

        fmu_path = cfg.fmu_path
        if not os.path.isabs(fmu_path):
            fmu_path = os.path.abspath(fmu_path)

        self._fmu_path = fmu_path
        self._step_size = cfg.fmu_step_size

        # Read FMU model description once
        self._model_desc = fmpy.read_model_description(fmu_path)

        # Build variable reference maps
        self._vr_map = {}  # name -> valueReference
        for var in self._model_desc.modelVariables:
            self._vr_map[var.name] = var.valueReference

        # Validate expected variables exist
        for name in ("position", "velocity", "torque_pred", "torque_true"):
            if name not in self._vr_map:
                raise ValueError(
                    f"FMU at {fmu_path} is missing expected variable '{name}'. "
                    f"Available: {list(self._vr_map.keys())}"
                )

        self._vr_position = self._vr_map["position"]
        self._vr_velocity = self._vr_map["velocity"]
        self._vr_torque_pred = self._vr_map["torque_pred"]
        self._vr_torque_true = self._vr_map["torque_true"]

        # Get number of continuous states from model description
        self._nx = getattr(self._model_desc, "numberOfContinuousStates", 0) or 0

        # Create one FMU instance per (env, joint).
        # For benchmark usage num_envs is typically 1.
        self._fmu_instances: list[list] = []  # [env][joint]
        self._time = 0.0

        for _env in range(self._num_envs):
            env_instances = []
            for _joint in range(self.num_joints):
                instance = self._create_fmu_instance()
                env_instances.append(instance)
            self._fmu_instances.append(env_instances)

        logger.info(
            f"ActuatorNetFMU: loaded {self._num_envs * self.num_joints} FMU instances "
            f"from {fmu_path}"
        )

    def _create_fmu_instance(self):
        """Create and initialize a single FMU ModelExchange instance."""
        from fmpy import extract
        from fmpy.fmi2 import FMU2Model

        # Extract if it's a .fmu file; if directory, use directly
        if os.path.isdir(self._fmu_path):
            unzip_dir = self._fmu_path
        else:
            unzip_dir = extract(self._fmu_path)

        guid = self._model_desc.guid
        model_id = self._model_desc.modelExchange.modelIdentifier

        instance = FMU2Model(
            guid=guid,
            unzipDirectory=unzip_dir,
            modelIdentifier=model_id,
        )
        instance.instantiate()
        instance.setupExperiment(startTime=0.0)
        instance.enterInitializationMode()
        instance.exitInitializationMode()
        # Enter continuous-time mode for ModelExchange
        instance.enterContinuousTimeMode()
        instance.fmi2SetTime(instance.component, 0.0)

        return instance

    def _terminate_instances(self):
        """Clean up all FMU instances."""
        for env_instances in self._fmu_instances:
            for inst in env_instances:
                try:
                    inst.terminate()
                    inst.freeInstance()
                except Exception:
                    pass
        self._fmu_instances = []

    def __del__(self):
        self._terminate_instances()

    """
    Operations.
    """

    def reset(self, env_ids):
        """Reset FMU instances for specified environments."""
        # env_ids may be a slice (e.g. slice(None) for all envs) or a sequence
        if isinstance(env_ids, slice):
            env_ids = range(*env_ids.indices(self._num_envs))
        for env_id in env_ids:
            for j in range(self.num_joints):
                try:
                    self._fmu_instances[env_id][j].terminate()
                    self._fmu_instances[env_id][j].freeInstance()
                except Exception:
                    pass
                self._fmu_instances[env_id][j] = self._create_fmu_instance()
        self._time = 0.0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """Compute torques: PD prediction fed through FMU for correction.

        Args:
            control_action: The joint action instance.
            joint_pos: Current joint positions [rad]. Shape is (num_envs, num_joints).
            joint_vel: Current joint velocities [rad/s]. Shape is (num_envs, num_joints).

        Returns:
            The computed control action with corrected joint efforts.
        """
        # Step 1: compute PD torque (same as IdealPDActuator)
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        pd_torque = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts

        # Step 2: feed through FMU instances (CPU / numpy)
        pos_np = joint_pos.detach().cpu().numpy()
        vel_np = joint_vel.detach().cpu().numpy()
        pd_np = pd_torque.detach().cpu().numpy()

        output = np.zeros_like(pd_np)

        for env_idx in range(self._num_envs):
            for j_idx in range(self.num_joints):
                inst = self._fmu_instances[env_idx][j_idx]

                # Set inputs via FMI2 C API
                vrs = (ctypes.c_uint32 * 3)(
                    self._vr_position, self._vr_velocity, self._vr_torque_pred
                )
                vals = (ctypes.c_double * 3)(
                    float(pos_np[env_idx, j_idx]),
                    float(vel_np[env_idx, j_idx]),
                    float(pd_np[env_idx, j_idx]),
                )
                inst.fmi2SetReal(inst.component, vrs, 3, vals)

                # Advance time
                new_time = self._time + self._step_size
                inst.fmi2SetTime(inst.component, new_time)

                # RK4 step on continuous states
                if self._nx > 0:
                    dt = self._step_size
                    nx = self._nx

                    x0 = (ctypes.c_double * nx)()
                    inst.fmi2GetContinuousStates(inst.component, x0, nx)

                    # k1 = f(t, x0)
                    k1 = (ctypes.c_double * nx)()
                    inst.fmi2GetDerivatives(inst.component, k1, nx)

                    # k2 = f(t + dt/2, x0 + dt/2 * k1)
                    x_tmp = (ctypes.c_double * nx)(
                        *(x0[i] + 0.5 * dt * k1[i] for i in range(nx))
                    )
                    inst.fmi2SetContinuousStates(inst.component, x_tmp, nx)
                    inst.fmi2SetTime(inst.component, self._time + 0.5 * dt)
                    k2 = (ctypes.c_double * nx)()
                    inst.fmi2GetDerivatives(inst.component, k2, nx)

                    # k3 = f(t + dt/2, x0 + dt/2 * k2)
                    x_tmp = (ctypes.c_double * nx)(
                        *(x0[i] + 0.5 * dt * k2[i] for i in range(nx))
                    )
                    inst.fmi2SetContinuousStates(inst.component, x_tmp, nx)
                    k3 = (ctypes.c_double * nx)()
                    inst.fmi2GetDerivatives(inst.component, k3, nx)

                    # k4 = f(t + dt, x0 + dt * k3)
                    x_tmp = (ctypes.c_double * nx)(
                        *(x0[i] + dt * k3[i] for i in range(nx))
                    )
                    inst.fmi2SetContinuousStates(inst.component, x_tmp, nx)
                    inst.fmi2SetTime(inst.component, new_time)
                    k4 = (ctypes.c_double * nx)()
                    inst.fmi2GetDerivatives(inst.component, k4, nx)

                    # x_new = x0 + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
                    x_new = (ctypes.c_double * nx)(
                        *(x0[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
                          for i in range(nx))
                    )
                    inst.fmi2SetContinuousStates(inst.component, x_new, nx)

                # Read output
                vr_out = (ctypes.c_uint32 * 1)(self._vr_torque_true)
                val_out = (ctypes.c_double * 1)()
                inst.fmi2GetReal(inst.component, vr_out, 1, val_out)
                output[env_idx, j_idx] = val_out[0]

        self._time += self._step_size

        # Step 3: convert back to tensor
        self.computed_effort = torch.tensor(output, dtype=joint_pos.dtype, device=joint_pos.device)

        # Clip the computed effort based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)

        # Return torques
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action
