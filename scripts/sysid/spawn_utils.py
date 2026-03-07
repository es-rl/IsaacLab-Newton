# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn utility: USD spawner with fixed-base joint."""

from pxr import Gf, Usd, UsdPhysics

from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import clone
from isaaclab.sim.utils.stage import get_current_stage


@clone
def spawn_from_usd_with_fixed_base(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    base_link_name: str = "pelvis",
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from USD file and add a fixed joint to the base link.

    This function wraps the standard spawn_from_usd and adds a fixed joint
    between the world frame and the robot's base link after spawning.

    Args:
        prim_path: The prim path to spawn the asset at.
        cfg: The USD file configuration.
        translation: The translation to apply to the prim.
        orientation: The orientation (w, x, y, z) to apply to the prim.
        base_link_name: Name of the base link to attach fixed joint to. Defaults to "pelvis".
        **kwargs: Additional keyword arguments.

    Returns:
        The prim of the spawned asset.
    """
    # First, spawn the USD normally
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)

    # Get the stage
    stage = get_current_stage()

    # Construct path to base link
    base_link_path = f"{prim_path}/{base_link_name}"
    base_link_prim = stage.GetPrimAtPath(base_link_path)

    if not base_link_prim or not base_link_prim.IsValid():
        # Auto-detect: use first child prim as base link
        children = prim.GetChildren()
        base_link_prim = None
        for child in children:
            if child.GetTypeName() in ("Xform", "Mesh", "Scope"):
                base_link_prim = child
                break
        if not children:
            # No children — attach fixed joint directly to the root prim
            base_link_prim = prim
        elif base_link_prim is None:
            base_link_prim = children[0]
        print(f"[INFO] base_link '{base_link_name}' not found, using '{base_link_prim.GetPath()}'")
        base_link_path = str(base_link_prim.GetPath())

    # Ensure RigidBodyAPI is applied
    if not base_link_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"[INFO] Applying RigidBodyAPI to {base_link_path}")
        UsdPhysics.RigidBodyAPI.Apply(base_link_prim)

    # Create the fixed joint
    fixed_joint_path = base_link_prim.GetPath().AppendChild("FixedJoint")

    # Check if joint already exists
    if stage.GetPrimAtPath(fixed_joint_path):
        print(f"[INFO] Fixed joint already exists at {fixed_joint_path}")
        return prim

    # Define the fixed joint
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, fixed_joint_path)

    # Set body1 to the base link (body0 is world by default)
    fixed_joint.CreateBody1Rel().SetTargets([base_link_prim.GetPath()])

    # Set local poses (identity transform - no offset)
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # Identity quaternion (w, x, y, z)
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # Enable the joint
    fixed_joint.CreateJointEnabledAttr().Set(True)

    print(f"[INFO] Created fixed joint at {fixed_joint_path}")

    return prim
