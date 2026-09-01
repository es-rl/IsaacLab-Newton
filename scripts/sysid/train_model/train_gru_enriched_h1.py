# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the deployable H1 right-arm enriched residual GRU.

The default feature contract is a 20-input coupled four-joint layout:
[position, position_error, velocity, PD_hint, previous_torque]. It deliberately
omits qfrc_bias because the original H1 training MJCF is not available at
deployment. The default target is measured torque minus the factory PD estimate
with kp=60 and kd=1.5.

The legacy 24-input layout remains available for historical reproduction only
and requires --feature-layout legacy24 with matching qfrc_bias caches or the
original --model-xml.

Usage::

    ./isaaclab.sh -p scripts/sysid/train_model/train_gru_enriched_h1.py \
        --epochs 15 \
        --out ~/data/h1/models/h1_arm_enriched_residual_deployable20.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Co-located flat modules (the shared enriched_* files in this directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from enriched_contract import (  # noqa: E402
    H1_DEPLOYABLE_FEATURE_LAYOUT,
    H1_DEPLOYABLE_INPUT_SIZE,
    H1_LEGACY_FEATURE_LAYOUT,
    H1_LEGACY_INPUT_SIZE,
    make_h1_deployable_metadata,
)
from enriched_data_h1 import CANONICAL_JOINT_ORDER, N_JOINTS, load_split_motions  # noqa: E402
from enriched_model import ForceResidualGRU  # noqa: E402
from enriched_normalization import fit_stats, save_stats  # noqa: E402
from precompute_qfrc_bias import make_bias_provider  # noqa: E402

HIDDEN_SIZE = 128
NUM_LAYERS = 2
FORCE_BOUND = 25.0
KP_FACTORY = 60.0
KD_FACTORY = 1.5
# --kp-factory / --kd-factory override the sysid_pd_hint gains (e.g. lower H1
# kp to 40/1.0 to mirror G1's PD authority) for regime tests.
BATCH_SIZE = 64
WINDOW_LEN = 200
EPOCHS_DEFAULT = 15
LR = 1e-3
WEIGHT_DECAY = 1e-5

DATA_ROOT = os.path.expanduser("~/data/h1/sage")
QFRC_BIAS_ROOT = os.path.expanduser("~/data/h1/qfrc_bias")
SAVE_DIR = os.path.expanduser("~/data/h1/models")
DEFAULT_SPLIT = os.path.join(str(Path(__file__).resolve().parent), "enriched_split_h1.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _to_canonical(motion: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reorder SAGE alphabetical columns to [pitch, roll, yaw, elbow]."""
    csv_order = motion["joint_names"]
    if csv_order == CANONICAL_JOINT_ORDER:
        return motion
    perm = [csv_order.index(jn) for jn in CANONICAL_JOINT_ORDER]
    out = dict(motion)
    out["positions"] = motion["positions"][:, perm]
    out["velocities"] = motion["velocities"][:, perm]
    out["torques"] = motion["torques"][:, perm]
    out["targets"] = motion["targets"][:, perm]
    out["joint_names"] = list(CANONICAL_JOINT_ORDER)
    return out


def build_windowed(
    named_motions: list[tuple[str, dict[str, np.ndarray]]],
    window_len: int,
    bias_fn=None,
    residual_target: bool = False,
    kp_factory: float = KP_FACTORY,
    kd_factory: float = KD_FACTORY,
    tau_clip: float | None = None,
    tau_sim_root: str | None = None,
    feature_layout: str = H1_DEPLOYABLE_FEATURE_LAYOUT,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice motions into non-overlapping windows using a versioned feature layout.

    The deployable layout is [q, pe, v, sysid_pd_hint, prev_torque]. The legacy
    layout additionally inserts qfrc_bias before prev_torque and requires a
    bias provider. When residual_target=True the target is tau - sysid_pd_hint.

    When tau_sim_root is set, the target becomes tau_real - tau_sim where
    tau_sim is precomputed by running the deploy-time SysID-equipped simulation
    on each motion.
    """
    xs, ys = [], []
    for exp_name, motion in named_motions:
        m = _to_canonical(motion)
        q = m["positions"].astype(np.float32)
        qt = m["targets"].astype(np.float32)
        v = m["velocities"].astype(np.float32)
        tau = m["torques"].astype(np.float32)
        if tau_clip is not None:
            tau = np.clip(tau, -tau_clip, tau_clip)
        pe = qt - q
        sysid_pd = kp_factory * pe - kd_factory * v
        prev_torque = np.zeros_like(tau)
        prev_torque[1:] = tau[:-1]

        feature_blocks = [q, pe, v, sysid_pd]
        if feature_layout == H1_LEGACY_FEATURE_LAYOUT:
            if bias_fn is None:
                raise ValueError("legacy24 feature layout requires a qfrc_bias provider")
            feature_blocks.append(bias_fn(exp_name, q, v))
        elif feature_layout != H1_DEPLOYABLE_FEATURE_LAYOUT:
            raise ValueError(f"Unknown H1 feature layout: {feature_layout}")
        feature_blocks.append(prev_torque)
        feats = np.concatenate(feature_blocks, axis=1).astype(np.float32)

        if tau_sim_root is not None:
            tau_sim_path = os.path.join(tau_sim_root, f"{exp_name}_tau_sim.npy")
            if not os.path.isfile(tau_sim_path):
                raise FileNotFoundError(
                    f"--tau-sim-root set but {tau_sim_path} missing. Precompute with precompute_tau_sim_residual.py"
                )
            tau_sim = np.load(tau_sim_path).astype(np.float32)
            if tau_sim.shape != tau.shape:
                raise ValueError(f"{exp_name}: tau_sim shape {tau_sim.shape} != tau {tau.shape}")
            target = tau - tau_sim
        elif residual_target:
            target = tau - sysid_pd
        else:
            target = tau

        for start in range(0, q.shape[0] - window_len + 1, window_len):
            xs.append(feats[start : start + window_len])
            ys.append(target[start : start + window_len])

    return np.stack(xs), np.stack(ys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sage-root", default=DATA_ROOT, help="Directory of per-motion SAGE folders (the trajectory data)."
    )
    parser.add_argument("--split-json", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--model-xml",
        default=None,
        help="MuJoCo MJCF for the H1 right arm. When set, qfrc_bias is "
        "computed from --sage-root trajectories instead of loaded "
        "from --qfrc-bias-root caches.",
    )
    parser.add_argument(
        "--joint-names",
        nargs=4,
        default=list(CANONICAL_JOINT_ORDER),
        help="Four arm joint names in the MJCF, canonical order [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow].",
    )
    parser.add_argument(
        "--qfrc-bias-root",
        default=QFRC_BIAS_ROOT,
        help="Directory of <motion>_qfrc_bias.npy caches. Used when a "
        "cache exists; otherwise --model-xml computes and (with "
        "--save-qfrc-cache) writes here.",
    )
    parser.add_argument(
        "--save-qfrc-cache", action="store_true", help="Write computed qfrc_bias arrays to --qfrc-bias-root."
    )
    parser.add_argument(
        "--recompute-qfrc",
        action="store_true",
        help="Ignore existing qfrc_bias caches and recompute from "
        "--model-xml (combine with --save-qfrc-cache to overwrite).",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument(
        "--feature-layout",
        choices=(H1_DEPLOYABLE_FEATURE_LAYOUT, H1_LEGACY_FEATURE_LAYOUT),
        default=H1_DEPLOYABLE_FEATURE_LAYOUT,
        help="deployable20 omits qfrc_bias; legacy24 requires matching bias caches or the original MJCF.",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(SAVE_DIR, "h1_arm_enriched_residual_deployable20.pt"),
    )
    parser.add_argument(
        "--stats-out",
        default=os.path.join(SAVE_DIR, "h1_arm_enriched_residual_deployable20_stats.json"),
    )
    parser.add_argument(
        "--residual-target",
        default=True,
        action="store_true",
        help=(
            "Train target = tau - (KP_FACTORY*pe - KD_FACTORY*v). Pair with "
            "pd_plus_gru:true deploy at kp=60/kd=1.5 so runtime sum equals tau."
        ),
    )
    parser.add_argument(
        "--full-torque-target",
        dest="residual_target",
        action="store_false",
        help="Train against full measured torque instead of the default PD residual target.",
    )
    parser.add_argument(
        "--joint-loss-weights",
        type=str,
        default=None,
        help=(
            "Comma-separated 4 floats in canonical order "
            "[shoulder_pitch, shoulder_roll, shoulder_yaw, elbow]. Multiplies "
            "per-joint squared error before mean. Use to prioritize the joint "
            "that dominates baseline τ RMSE (e.g. '1,1,5,1' biases shoulder_yaw)."
        ),
    )
    parser.add_argument(
        "--torque-magnitude-weighted",
        action="store_true",
        help=(
            "Multiply per-sample squared error by |target| so high-magnitude "
            "samples dominate. Stacks with --joint-loss-weights."
        ),
    )
    parser.add_argument(
        "--kp-factory", type=float, default=KP_FACTORY, help="kp used for sysid_pd_hint feature + residual subtraction."
    )
    parser.add_argument(
        "--kd-factory", type=float, default=KD_FACTORY, help="kd used for sysid_pd_hint feature + residual subtraction."
    )
    parser.add_argument(
        "--tau-clip",
        type=float,
        default=None,
        help=(
            "Clip real motor torque to ±tau_clip Nm before training. Use to "
            "mask saturation spikes (e.g. shoulder_yaw at -26 Nm against "
            "real-robot mechanical stop) that contaminate the supervised target."
        ),
    )
    parser.add_argument(
        "--force-bound",
        type=float,
        default=FORCE_BOUND,
        help=(
            "Force bound applied to model output AND training target clamp. "
            "Lower values (e.g. 5.0) constrain the GRU to small residual "
            "corrections, preventing closed-loop saturation cascade at deploy."
        ),
    )
    parser.add_argument(
        "--tau-sim-root",
        type=str,
        default=None,
        help=(
            "Track 1 SysID-aware residual (underperformed in practice; kept "
            "for repro): directory with "
            "precomputed <motion>_tau_sim.npy files. When set, target = "
            "tau_real - tau_sim. Hypothesized to fix training-vs-deploy basis "
            "mismatch; in practice underperforms v28's tau_real - "
            "sysid_pd_hint target by 2-4 pp on val5. Kept for repro / future "
            "use. Pair with pd_plus_gru:true at deploy."
        ),
    )
    args = parser.parse_args()
    args.out = os.path.abspath(os.path.expanduser(args.out))
    args.stats_out = os.path.abspath(os.path.expanduser(args.stats_out))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)

    train_pairs = load_split_motions(args.sage_root, args.split_json, role="train")
    val_pairs = load_split_motions(args.sage_root, args.split_json, role="val")
    print(f"Train ({len(train_pairs)}) / Val ({len(val_pairs)})")

    tau_sim_root = os.path.expanduser(args.tau_sim_root) if args.tau_sim_root else None
    if tau_sim_root is not None and not os.path.isdir(tau_sim_root):
        raise FileNotFoundError(f"--tau-sim-root not a directory: {tau_sim_root}")

    input_size = H1_DEPLOYABLE_INPUT_SIZE
    bias_fn = None
    if args.feature_layout == H1_LEGACY_FEATURE_LAYOUT:
        input_size = H1_LEGACY_INPUT_SIZE
        bias_fn = make_bias_provider(
            args.model_xml,
            args.joint_names,
            args.qfrc_bias_root,
            save_cache=args.save_qfrc_cache,
            recompute=args.recompute_qfrc,
        )
        if args.model_xml:
            mode = "recompute (ignore cache)" if args.recompute_qfrc else "cache if present, else compute"
            print(f"qfrc_bias: {mode} via {args.model_xml}")
        else:
            print(f"qfrc_bias: load matching legacy caches from {args.qfrc_bias_root}")
    else:
        print(f"qfrc_bias: omitted by {H1_DEPLOYABLE_FEATURE_LAYOUT} layout")

    Xtr, Ytr = build_windowed(
        train_pairs,
        WINDOW_LEN,
        bias_fn,
        args.residual_target,
        args.kp_factory,
        args.kd_factory,
        args.tau_clip,
        tau_sim_root=tau_sim_root,
        feature_layout=args.feature_layout,
    )
    Xva, Yva = build_windowed(
        val_pairs,
        WINDOW_LEN,
        bias_fn,
        args.residual_target,
        args.kp_factory,
        args.kd_factory,
        args.tau_clip,
        tau_sim_root=tau_sim_root,
        feature_layout=args.feature_layout,
    )
    if args.tau_clip is not None:
        print(f"[tau-clip] real τ clipped to ±{args.tau_clip} Nm before residual computation")
    if args.kp_factory != KP_FACTORY or args.kd_factory != KD_FACTORY:
        print(f"[kp/kd] sysid_pd_hint computed with kp={args.kp_factory}, kd={args.kd_factory}")
    if tau_sim_root is not None:
        print(
            f"[Track 1] SysID-aware residual: target = tau_real - tau_sim from "
            f"{tau_sim_root} (deploy with pd_plus_gru:true)"
        )
    elif args.residual_target:
        print("[residual] target = tau - sysid_pd_hint (deploy with pd_plus_gru:true)")
    print(f"Train windows: {Xtr.shape}, Val windows: {Xva.shape}, input_size={input_size}")

    norm_stats = fit_stats(Xtr.reshape(-1, input_size))
    print(f"Norm stats: std range [{norm_stats['std'].min():.3f}, {norm_stats['std'].max():.3f}]")

    Xtr_n = (Xtr - norm_stats["mean"]) / norm_stats["std"]
    Xva_n = (Xva - norm_stats["mean"]) / norm_stats["std"]

    gru = ForceResidualGRU(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_joints=N_JOINTS,
        force_bound=args.force_bound,
        dropout=0.1,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(gru.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    if args.force_bound != FORCE_BOUND:
        print(f"[force_bound] using {args.force_bound} Nm (vs default {FORCE_BOUND})")

    Xtr_t = torch.from_numpy(Xtr_n).to(DEVICE)
    Ytr_t = torch.from_numpy(Ytr).to(DEVICE).clamp(-args.force_bound, args.force_bound)
    Xva_t = torch.from_numpy(Xva_n).to(DEVICE)
    Yva_t = torch.from_numpy(Yva).to(DEVICE).clamp(-args.force_bound, args.force_bound)

    # Joint-loss-weight tensor: shape [1, 1, 4], broadcasts over (batch, window, joint).
    if args.joint_loss_weights:
        jw = [float(x) for x in args.joint_loss_weights.split(",")]
        if len(jw) != N_JOINTS:
            raise ValueError(f"--joint-loss-weights expects {N_JOINTS} values, got {len(jw)}")
        joint_w = torch.tensor(jw, dtype=torch.float32, device=DEVICE).view(1, 1, N_JOINTS)
        print(f"[loss] joint weights {jw} (canonical: pitch, roll, yaw, elbow)")
    else:
        joint_w = None

    n_train = Xtr_t.shape[0]
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        gru.train()
        perm = torch.randperm(n_train, device=DEVICE)
        running, nb = 0.0, 0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            xb = Xtr_t[idx]
            yb = Ytr_t[idx]
            pred, _ = gru(xb)
            sq = (pred - yb) ** 2
            if args.torque_magnitude_weighted:
                sq = sq * (yb.abs() + 0.1)
            if joint_w is not None:
                sq = sq * joint_w
            loss = torch.mean(sq)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(gru.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach())
            nb += 1
        tr = running / max(1, nb)

        gru.eval()
        with torch.no_grad():
            pred_va, _ = gru(Xva_t)
            va = float(torch.mean((pred_va - Yva_t) ** 2))
            abs_mean = float(pred_va.abs().mean())
        print(f"Epoch {epoch:2d}: train {tr:.4f}, val {va:.4f}, |pred|_mean {abs_mean:.3f}")

        if va < best_val:
            best_val = va
            metadata = None
            if args.feature_layout == H1_DEPLOYABLE_FEATURE_LAYOUT and args.residual_target:
                metadata = make_h1_deployable_metadata(
                    hidden_size=HIDDEN_SIZE,
                    num_layers=NUM_LAYERS,
                    force_bound=args.force_bound,
                    kp=args.kp_factory,
                    kd=args.kd_factory,
                )
            torch.save(
                {
                    "model": gru.state_dict(),
                    "epoch": epoch,
                    "val_loss": va,
                    "metadata": metadata,
                },
                args.out,
            )
            save_stats(norm_stats, args.stats_out, metadata=metadata)

    print(f"\nEnriched warm start done. Best val MSE: {best_val:.4f}. Saved to {args.out}")

    # TorchScript the best-val checkpoint for newton_benchmark.py deployment.
    # Reload the best state explicitly so the scripted module matches the
    # checkpoint that was selected by val MSE, not the last epoch's state.
    best_ckpt = torch.load(args.out, map_location=DEVICE, weights_only=True)
    gru.load_state_dict(best_ckpt["model"])
    gru.eval()
    script_path = args.out.replace(".pt", "_script.pt")
    scripted = torch.jit.script(gru)
    scripted.save(script_path)
    print(f"Scripted checkpoint saved to {script_path}")


if __name__ == "__main__":
    main()
