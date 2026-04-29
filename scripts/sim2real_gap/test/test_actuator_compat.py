# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``scripts.sim2real_gap.actuator_compat.get_joint_indices``."""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from actuator_compat import get_joint_indices  # noqa: E402


class _ActuatorWithPublic:
    """Both attributes set; helper must prefer the public property."""

    joint_indices = [1, 2, 3]
    _joint_indices = [9, 9, 9]  # would be wrong if used


class _ActuatorPrivateOnly:
    """Newton-backend ActuatorBase shape: only ``_joint_indices``."""

    _joint_indices = [4, 5, 6]


class _ActuatorNoIndices:
    """Neither attribute present; helper must raise."""


def test_prefers_public_property_when_present() -> None:
    actuator = _ActuatorWithPublic()
    assert get_joint_indices(actuator) == [1, 2, 3]


def test_falls_back_to_private_when_public_missing() -> None:
    actuator = _ActuatorPrivateOnly()
    assert get_joint_indices(actuator) == [4, 5, 6]


def test_raises_attribute_error_when_neither_present() -> None:
    actuator = _ActuatorNoIndices()
    with pytest.raises(AttributeError):
        get_joint_indices(actuator)
