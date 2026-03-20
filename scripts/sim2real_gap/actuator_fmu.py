# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FMU-based hybrid actuator model (CoSimulation).

Wraps an FMI 2.0 CoSimulation FMU that takes joint position, commanded
position, and a PD-predicted torque as inputs, and outputs a corrected
"true" torque.  The FMU has its own internal solver and is advanced via
``doStep()``.

.. note::
    The FMU's variable names (``position``, ``velocity``, ``torque_pred``)
    follow the Ansys Twin Builder training convention where the
    ``velocity`` slot actually carries the **commanded position** [rad],
    not angular velocity.  ``torque_pred`` is the Unitree controller's
    estimated torque; in sim we approximate it with PD torque
    (``kp * position_error - kd * velocity``).

Requires the ``fmpy`` package (``pip install fmpy``).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg

# Register this module under the isaaclab namespace so that FactoryBase's
# __init_subclass__ check (requires cls.__module__ to start with "isaaclab")
# accepts our actuator class without needing to live inside the isaaclab tree.
_orig_module_name = __name__
__name__ = "isaaclab.actuators.actuator_fmu"
sys.modules[__name__] = sys.modules[_orig_module_name]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@configclass
class ActuatorNetFMUCfg(IdealPDActuatorCfg):
    """Configuration for an FMU-based hybrid actuator model.

    The FMU is expected to expose these FMI variables:

    * **Inputs**: ``position`` [rad], ``velocity`` (commanded position [rad]),
      ``torque_pred`` (kp * position_error [N-m])
    * **Output**: ``torque_true`` [N-m]
    """

    class_type: type | str = "isaaclab.actuators.actuator_fmu:ActuatorNetFMU"

    fmu_path: str = MISSING
    """Path to the FMU ``.fmu`` archive or unzipped directory."""

    fmu_step_size: float = 0.002
    """Communication step size [s] for ``doStep()``. Should match the
    physics dt of the simulation."""


# ---------------------------------------------------------------------------
# Actuator
# ---------------------------------------------------------------------------

class ActuatorNetFMU(IdealPDActuator):
    """Hybrid actuator that corrects PD torques through a CoSimulation FMU.

    The compute flow is:

    1. Compute PD torque the same way as :class:`IdealPDActuator`.
    2. Feed ``(position, velocity, torque_pred)`` into the FMU.
    3. Call ``doStep()`` to advance the FMU's internal solver.
    4. Use the FMU output ``torque_true`` as the applied effort.

    One FMU instance is created per (env, joint). Each instance maintains
    its own internal state across timesteps (the FMU is stateful / dynamic).
    """

    cfg: ActuatorNetFMUCfg
    """The configuration of the actuator model."""

    def __init__(self, cfg: ActuatorNetFMUCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        try:
            import fmpy
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

        # Validate that the FMU supports CoSimulation
        if self._model_desc.coSimulation is None:
            raise ValueError(
                f"FMU at {fmu_path} does not support CoSimulation. "
                f"Only CoSimulation FMUs are supported by ActuatorNetFMU."
            )

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

        # Create one FMU instance per (env, joint).
        self._fmu_instances: list[list] = []  # [env][joint]
        self._time = 0.0

        for _env in range(self._num_envs):
            env_instances = []
            for _joint in range(self.num_joints):
                instance = self._create_fmu_instance()
                env_instances.append(instance)
            self._fmu_instances.append(env_instances)

        logger.info(
            f"ActuatorNetFMU [CoSimulation]: loaded "
            f"{self._num_envs * self.num_joints} FMU instances from {fmu_path}"
        )

    def _create_fmu_instance(self):
        """Create and initialize a single CoSimulation FMU instance."""
        from fmpy import extract
        from fmpy.fmi2 import FMU2Slave

        # Extract if it's a .fmu file; if directory, use directly
        if os.path.isdir(self._fmu_path):
            unzip_dir = self._fmu_path
        else:
            unzip_dir = extract(self._fmu_path)

        instance = FMU2Slave(
            guid=self._model_desc.guid,
            unzipDirectory=unzip_dir,
            modelIdentifier=self._model_desc.coSimulation.modelIdentifier,
        )
        instance.instantiate()
        instance.setupExperiment(startTime=0.0)
        instance.enterInitializationMode()
        instance.exitInitializationMode()

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

        The FMU was trained with Ansys Twin Builder convention where:
        - ``position`` = actual joint position
        - ``velocity`` = **commanded position** (not angular velocity)
        - ``torque_pred`` = PD torque (approximation of Unitree controller estimate)

        Args:
            control_action: The joint action instance.
            joint_pos: Current joint positions [rad]. Shape is (num_envs, num_joints).
            joint_vel: Current joint velocities [rad/s]. Shape is (num_envs, num_joints).

        Returns:
            The computed control action with corrected joint efforts.
        """
        # Step 1: compute torque_pred as PD torque (kp * pos_error - kd * vel).
        # The FMU was trained with the Unitree controller's torque estimate as
        # torque_pred; PD is our best approximation in sim-in-the-loop.
        error_pos = control_action.joint_positions - joint_pos
        torque_pred = self.stiffness * error_pos - self.damping * joint_vel

        # Step 2: feed through FMU instances (CPU / numpy)
        # FMU input mapping (Ansys Twin Builder convention):
        #   "position"    <- actual joint position
        #   "velocity"    <- commanded position (NOT angular velocity)
        #   "torque_pred" <- PD torque (best approx of Unitree controller estimate)
        pos_np = joint_pos.detach().cpu().numpy()
        cmd_np = control_action.joint_positions.detach().cpu().numpy()
        tp_np = torque_pred.detach().cpu().numpy()

        output = np.zeros_like(tp_np)

        for env_idx in range(self._num_envs):
            for j_idx in range(self.num_joints):
                inst = self._fmu_instances[env_idx][j_idx]

                # Set inputs (Ansys convention: "velocity" = commanded position)
                inst.setReal([self._vr_position], [float(pos_np[env_idx, j_idx])])
                inst.setReal([self._vr_velocity], [float(cmd_np[env_idx, j_idx])])
                inst.setReal([self._vr_torque_pred], [float(tp_np[env_idx, j_idx])])

                # Advance the FMU's internal solver
                inst.doStep(
                    currentCommunicationPoint=self._time,
                    communicationStepSize=self._step_size,
                )

                # Read output
                output[env_idx, j_idx] = inst.getReal([self._vr_torque_true])[0]

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
