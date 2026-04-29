# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility shims for IsaacLab actuator backends.

The Newton-backend ``ActuatorBase`` (in ``source/isaaclab_newton/...actuators/
actuator_base.py``) lacks the public ``joint_indices`` property that the
upstream ``isaaclab`` ``ActuatorBase`` exposes. Downstream callers must accept
either spelling. This module hosts the fallback in one place so individual
call sites stay clean and testable.

Once the Newton-backend ``ActuatorBase`` adds the public property upstream,
remove this helper and inline the public read.
"""

from __future__ import annotations

from typing import Any


def get_joint_indices(actuator: Any) -> Any:
    """Return the joint indices for an actuator.

    Prefers the public ``joint_indices`` property and falls back to the
    private ``_joint_indices`` attribute when the public property is missing
    (as is the case on the Newton-backend ``ActuatorBase``).

    Args:
        actuator: An ``ActuatorBase`` (or subclass) instance.

    Returns:
        The joint indices tensor/array/slice, as exposed by either the public
        ``joint_indices`` property or the private ``_joint_indices`` attribute.

    Raises:
        AttributeError: If neither attribute exists. Surfaces a clear failure
            for unexpected actuator types instead of silently returning None.
    """
    public = getattr(actuator, "joint_indices", None)
    if public is not None:
        return public
    return actuator._joint_indices
