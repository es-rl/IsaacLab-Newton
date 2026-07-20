# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Precompute per-motion ``qfrc_bias`` feature caches for the enriched GRU trainers.

The 24-feature enriched layout (see ``TRAINING_SCRIPTS.md``) includes a
``qfrc_bias`` block: MuJoCo's bias force ``C(q, q_dot)*q_dot + g(q)`` — the
Coriolis/centrifugal + gravity generalized forces (actuation, external, and
passive forces excluded). MuJoCo populates ``mjData.qfrc_bias`` during the
forward pipeline, so for each recorded timestep we set ``qpos``/``qvel`` to the
measured joint state, run :func:`mujoco.mj_forward`, and read the bias for the
arm DOFs.

This regenerates the ``<motion>_qfrc_bias.npy`` files that
``train_gru_enriched_g1.py`` / ``train_gru_enriched_h1.py`` load via
``--qfrc-bias-root``. Output arrays are ``(T, 4)`` float32 in canonical
``[shoulder_pitch, shoulder_roll, shoulder_yaw, elbow]`` order, one file per
motion, length-aligned to that motion's ``positions``.

**Correctness requirements**

- Pass the *same* MuJoCo model XML used to train/deploy the checkpoint. ``qfrc_bias``
  is fully model-dependent (masses, inertias, gravity, joint frames); a mismatched
  model silently corrupts the feature.
- ``--joint-names`` must be the four arm joint names *as they appear in the MJCF*,
  given in canonical order. Non-arm joints are left at the model's default pose, so
  use a fixed-base arm-only model (or one whose remaining joints don't affect the
  arm's bias) for a faithful result.

Usage (G1)::

    ./isaaclab.sh -p scripts/sysid/train_model/precompute_qfrc_bias.py \\
        --robot g1 \\
        --model-xml /path/to/g1_right_arm.xml \\
        --joint-names right_shoulder_pitch right_shoulder_roll right_shoulder_yaw right_elbow \\
        --data-root ~/data/g1/experiments \\
        --split-json enriched_split_g1.json --roles train test \\
        --out-dir ~/data/g1/qfrc_bias

Usage (H1)::

    ./isaaclab.sh -p scripts/sysid/train_model/precompute_qfrc_bias.py \\
        --robot h1 \\
        --model-xml /path/to/h1_right_arm.xml \\
        --joint-names right_shoulder_pitch right_shoulder_roll right_shoulder_yaw right_elbow \\
        --sage-root ~/data/h1/sage \\
        --split-json enriched_split_h1.json --roles train val test \\
        --out-dir ~/data/h1/qfrc_bias
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Co-located loaders (same directory as this script).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _resolve_joint_indices(model, joint_names: list[str]) -> tuple[list[int], list[int]]:
    """Map MJCF joint names to ``qpos`` and ``dof`` addresses (assumes 1-DOF hinges)."""
    import mujoco

    scalar_types = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    qpos_idx, dof_idx = [], []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(
                f"joint {name!r} not found in model; available joints: "
                f"{[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]}"
            )
        if int(model.jnt_type[jid]) not in scalar_types:
            raise ValueError(
                f"joint {name!r} has jnt_type={int(model.jnt_type[jid])}; qfrc_bias "
                "computation assumes scalar (hinge/slide) joints, one qpos/qvel each."
            )
        qpos_idx.append(int(model.jnt_qposadr[jid]))
        dof_idx.append(int(model.jnt_dofadr[jid]))
    return qpos_idx, dof_idx


def compute_qfrc_bias(
    model,
    data,
    qpos_idx: list[int],
    dof_idx: list[int],
    positions: np.ndarray,
    velocities: np.ndarray,
) -> np.ndarray:
    """Compute ``qfrc_bias`` for the arm DOFs over one motion.

    Args:
        model: A ``mujoco.MjModel``.
        data: A ``mujoco.MjData`` bound to ``model``.
        qpos_idx: ``qpos`` addresses of the arm joints, canonical order.
        dof_idx: ``dof`` addresses of the arm joints, canonical order.
        positions: Joint positions [rad], shape [T, 4], canonical order.
        velocities: Joint velocities [rad/s], shape [T, 4], canonical order.

    Returns:
        Bias forces [N·m], shape [T, 4], float32, canonical order.
    """
    import mujoco

    n = positions.shape[0]
    out = np.empty((n, len(dof_idx)), dtype=np.float32)
    for t in range(n):
        data.qpos[qpos_idx] = positions[t]
        data.qvel[dof_idx] = velocities[t]
        mujoco.mj_forward(model, data)  # fills data.qfrc_bias
        out[t] = data.qfrc_bias[dof_idx]
    return out


def make_bias_provider(
    model_xml: str | None,
    joint_names: list[str],
    cache_root: str | None = None,
    save_cache: bool = False,
    recompute: bool = False,
):
    """Build a ``fn(name, positions, velocities) -> (T, 4)`` ``qfrc_bias`` provider.

    Used by the enriched trainers to obtain the ``qfrc_bias`` feature per motion
    directly from the trajectory data. Resolution order per motion:

    1. Unless ``recompute`` is set: if ``cache_root/<name>_qfrc_bias.npy`` exists,
       load it (validating its length against the trajectory) and return.
    2. Else if ``model_xml`` is set, compute it with MuJoCo (``mj_forward`` →
       ``data.qfrc_bias``) and, when ``save_cache`` is set, write it to
       ``cache_root``.
    3. Else raise ``FileNotFoundError``.

    Args:
        model_xml: Path to the MuJoCo MJCF used to compute the bias (same model
            as train/deploy). ``None`` disables computation (cache-only mode).
        joint_names: The four arm joint names in the MJCF, canonical order.
        cache_root: Directory of ``<name>_qfrc_bias.npy`` caches (read, and
            written when ``save_cache`` is set). ``None`` disables caching.
        save_cache: Write freshly computed bias arrays back to ``cache_root``.
        recompute: Ignore any existing cache and always compute (requires
            ``model_xml``); combine with ``save_cache`` to overwrite stale caches.
    """
    import warnings

    model = data = qpos_idx = dof_idx = None
    if model_xml:
        import mujoco

        model = mujoco.MjModel.from_xml_path(os.path.expanduser(model_xml))
        data = mujoco.MjData(model)
        qpos_idx, dof_idx = _resolve_joint_indices(model, joint_names)
        if model.nv != len(dof_idx):
            warnings.warn(
                f"model has nv={model.nv} DOFs but only {len(dof_idx)} arm joints are "
                "set; non-arm DOFs stay at their model defaults and perturb qfrc_bias. "
                "Use a fixed-base arm-only MJCF for a faithful result.",
                stacklevel=2,
            )

    resolved_cache = os.path.expanduser(cache_root) if cache_root else None

    def _check_len(name: str, bias: np.ndarray, expected: int, source: str) -> None:
        if bias.shape[0] != expected:
            raise ValueError(
                f"{name}: {source} qfrc_bias len {bias.shape[0]} != traj len {expected}"
                + (". Delete the stale cache or pass recompute=True." if source == "cached" else "")
            )

    def provider(name: str, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        expected = positions.shape[0]
        if resolved_cache and not recompute:
            cpath = os.path.join(resolved_cache, f"{name}_qfrc_bias.npy")
            if os.path.isfile(cpath):
                bias = np.load(cpath).astype(np.float32)
                _check_len(name, bias, expected, "cached")
                return bias
        if model is None:
            raise FileNotFoundError(
                f"qfrc_bias for {name!r} not found under {resolved_cache!r} and no "
                "model_xml given to compute it. Pass --model-xml (and --joint-names "
                "if your MJCF differs) to compute from the trajectory data."
            )
        bias = compute_qfrc_bias(model, data, qpos_idx, dof_idx, positions, velocities)
        _check_len(name, bias, expected, "computed")
        if save_cache and resolved_cache:
            os.makedirs(resolved_cache, exist_ok=True)
            np.save(os.path.join(resolved_cache, f"{name}_qfrc_bias.npy"), bias)
        return bias

    return provider


def _enumerate_g1(data_root: str, split_json: str, roles: list[str]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(motion_name, positions, velocities)`` for the G1 split roles."""
    from enriched_data_g1 import load_experiment, load_split_json

    train, test = load_split_json(split_json)
    role_map = {"train": train, "test": test}
    motions = []
    for role in roles:
        if role not in role_map:
            raise ValueError(f"G1 split has roles {list(role_map)}, got {role!r}")
        for name in role_map[role]:
            exp = load_experiment(os.path.join(data_root, name))
            # G1 CSV columns are already in canonical 4-joint order.
            motions.append((name, exp["positions"], exp["velocities"]))
    return motions


def _enumerate_h1(
    sage_root: str, split_json: str, roles: list[str], joint_names: list[str]
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(motion_name, positions, velocities)`` for the H1 split roles.

    SAGE columns are alphabetical; reorder them to match ``joint_names`` (canonical).
    """
    from enriched_data_h1 import load_split_motions

    motions = []
    for role in roles:
        for name, m in load_split_motions(sage_root, split_json, role):
            jn = list(m["joint_names"])
            try:
                perm = [jn.index(j) for j in joint_names]
            except ValueError as exc:
                raise ValueError(
                    f"{name}: --joint-names {joint_names} not all present in motion joint_names {jn}"
                ) from exc
            motions.append((name, m["positions"][:, perm], m["velocities"][:, perm]))
    return motions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", choices=["g1", "h1"], required=True)
    parser.add_argument("--model-xml", required=True, help="MuJoCo MJCF for the arm")
    parser.add_argument(
        "--joint-names",
        nargs=4,
        required=True,
        help="Four arm joint names (MJCF), canonical order [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow]",
    )
    parser.add_argument("--split-json", required=True, help="Split manifest (enriched_split_*.json)")
    parser.add_argument(
        "--roles",
        nargs="+",
        default=["train", "test"],
        help="Split roles to process (g1: train/test; h1: train/val/test)",
    )
    parser.add_argument("--out-dir", required=True, help="Directory to write <motion>_qfrc_bias.npy")
    # Robot-specific data roots.
    parser.add_argument("--data-root", help="G1: dir of per-experiment folders")
    parser.add_argument("--sage-root", help="H1: dir of per-motion SAGE folders")
    args = parser.parse_args()

    import importlib.util

    if importlib.util.find_spec("mujoco") is None:
        sys.exit(
            "mujoco is required: pip install mujoco  (run via ./isaaclab.sh -p if using the Isaac Lab environment)."
        )

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.robot == "g1":
        if not args.data_root:
            parser.error("--data-root is required for --robot g1")
        motions = _enumerate_g1(os.path.expanduser(args.data_root), args.split_json, args.roles)
    else:
        if not args.sage_root:
            parser.error("--sage-root is required for --robot h1")
        motions = _enumerate_h1(os.path.expanduser(args.sage_root), args.split_json, args.roles, args.joint_names)

    # Standalone precompute: always compute fresh and overwrite the cache.
    provider = make_bias_provider(args.model_xml, args.joint_names, cache_root=out_dir, save_cache=True, recompute=True)
    print(f"[qfrc_bias] {len(motions)} motions -> {out_dir}")
    for name, positions, velocities in motions:
        bias = provider(name, positions, velocities)
        print(f"  {name}: {bias.shape} -> {name}_qfrc_bias.npy")
    print("[qfrc_bias] done.")


if __name__ == "__main__":
    main()
