# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Regression test for fixed-base articulation root-velocity indexing.

Exists because vendored ``articulation_data.py`` historically had a fixed-base
branch using ``[:, 0, 0]`` indexing for ``_sim_bind_root_com_vel_w`` while the
current Newton API returns a 1-D array (per environment) for both fixed-base
and floating-base. Any future vendor sync that re-introduces the ``[:, 0, 0]``
form will fail this test.

Companion sync-resilient static check lives at
``scripts/sysid/test/test_articulation_static_check.py``.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
import torch
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager  # noqa: F401

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def _make_fixed_base_cfg() -> ArticulationCfg:
    """Build a minimal fixed-base articulation config (Franka panda).

    Uses Franka because it is a known fixed-base asset already exercised by
    the rest of the test suite (see ``test_initialization_fixed_base`` in
    ``test/assets/test_articulation.py``).
    """
    return ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={".*": 0.0},
        ),
        actuators={},
    )


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.isaacsim_ci
def test_fixed_base_root_velocity_binding_is_one_dim(device: str) -> None:
    """After patch: ``_sim_bind_root_com_vel_w`` is shape ``(num_envs, 6)`` for
    fixed-base, matching floating-base. Pre-patch fixed-base used
    ``[:, 0, 0]`` which crashed with ``IndexError: tuple index out of range``
    at ``_create_simulation_bindings``.
    """
    sim_cfg = SimulationCfg(
        device=device,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(),
            num_substeps=2,
        ),
    )
    with build_simulation_context(sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        articulation_cfg = _make_fixed_base_cfg()
        articulation = Articulation(articulation_cfg)

        # Trigger _create_simulation_bindings: pre-patch this raised IndexError
        # for fixed-base assets because of the [:, 0, 0] indexing.
        sim.reset()

        assert articulation.is_initialized
        assert articulation.is_fixed_base, "Franka should be fixed-base for this test"

        # The patch unifies indexing to [:, 0]; the resulting binding is per-
        # environment 6-D linear+angular velocity.
        vel_binding = articulation.data._sim_bind_root_com_vel_w
        assert vel_binding is not None, "_sim_bind_root_com_vel_w must be set after reset"
        # Shape sanity: (num_envs, 6). This is the post-patch contract; the
        # pre-patch fixed-base form ``[:, 0, 0]`` would have produced a 1-D
        # tensor (or crashed with IndexError on 1-D input).
        vel_torch = vel_binding if isinstance(vel_binding, torch.Tensor) else torch.as_tensor(vel_binding)
        assert vel_torch.ndim == 2, f"expected 2-D (num_envs, 6), got shape {tuple(vel_torch.shape)}"
        assert vel_torch.shape[-1] == 6, f"expected last dim 6, got {tuple(vel_torch.shape)}"
