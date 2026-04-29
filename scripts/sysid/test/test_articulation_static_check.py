# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sync-resilient static check for the vendored articulation fix.

The full integration test lives at
``source/isaaclab_newton/test/test_articulation_fixed_base_indexing.py``,
inside the vendored boundary — a vendor sync may wipe it. This static
check parses the patched source file and asserts the buggy ``[:, 0, 0]``
indexing is not present in the ``_sim_bind_root_com_vel_w`` block,
catching regressions even if the vendored test is removed.
"""

from __future__ import annotations

import os
import re

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_TARGET = os.path.join(
    _REPO_ROOT,
    "source",
    "isaaclab_newton",
    "isaaclab_newton",
    "assets",
    "articulation",
    "articulation_data.py",
)


def test_root_com_vel_w_does_not_use_three_dim_indexing() -> None:
    """Source must not contain ``_sim_bind_root_com_vel_w[..][:, 0, 0]``
    in the bindings block (lines ~1215-1230)."""
    assert os.path.isfile(_TARGET), f"Vendored source not found at {_TARGET}"
    with open(_TARGET) as f:
        source = f.read()

    # Locate the _sim_bind_root_com_vel_w binding block.
    block_start = source.find("self._sim_bind_root_com_vel_w = self._root_view.get_root_velocities")
    assert block_start != -1, "could not locate root_com_vel_w binding line"
    # Look for the buggy [:, 0, 0] indexing within ~30 lines of the binding.
    block = source[block_start : block_start + 2000]
    assert not re.search(r"\[:,\s*0,\s*0\]", block), (
        "Found buggy [:, 0, 0] indexing near _sim_bind_root_com_vel_w. "
        "A vendor sync likely re-introduced the fixed-base bug. Re-apply "
        "the patch: collapse the if/else to use [:, 0] unconditionally."
    )
