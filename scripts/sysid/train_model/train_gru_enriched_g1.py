# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the 24-feat enriched GRU at v1 Optuna-winning hparams on the v2 train pool.

Mirrors training/train_optuna_best.py but reads the v2 split manifest +
v2 Row 10 qfrc_bias cache. Hparams are the v1 Tier-1 Optuna winner
(see optuna_enriched.db trial 15) so the change vs v2 Enriched is purely
the training pipeline (hidden=192 not 128, force_bound=15 not 25, lr=8e-4
not 1e-3, batch=32 not 64, dropout=0.225 not 0.1, 25 epochs not 15).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Co-located flat modules (the shared enriched_* files in this directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enriched_data_g1 import (
    load_experiment,
    load_split_json,
)
from enriched_model import ForceResidualGRU
from enriched_normalization import fit_stats, save_stats
from precompute_qfrc_bias import make_bias_provider

# Optuna-winning config (v1 Tier 1, trial 15)
HIDDEN_SIZE = 192
NUM_LAYERS = 2
DROPOUT = 0.225
LR = 8.0e-4
WEIGHT_DECAY = 3.2e-6
BATCH_SIZE = 32
FORCE_BOUND = 15.0

INPUT_SIZE = 24
KP = 40.0
KD = 1.0
WINDOW_LEN = 200
EPOCHS = 25

# Canonical right-arm joint names for the qfrc_bias MuJoCo computation. Override
# with --joint-names if your MJCF uses different names.
DEFAULT_ARM_JOINTS = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
]

DATA_ROOT = os.path.expanduser("~/data/g1/experiments")
DEFAULT_SPLIT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enriched_split_g1.json")
DEFAULT_QFRC_BIAS_ROOT = os.path.expanduser("~/data/g1/qfrc_bias")
SAVE_DIR = os.path.expanduser("~/data/g1/models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_windowed(pairs, window_len, bias_fn):
    xs, ys = [], []
    for exp_name, exp in pairs:
        q = exp["positions"].astype(np.float32)
        qt = exp["targets"].astype(np.float32)
        v = exp["velocities"].astype(np.float32)
        tau = exp["torques"].astype(np.float32)
        pe = qt - q
        sysid_pd = KP * pe - KD * v
        qfrc_bias = bias_fn(exp_name, q, v)
        prev_torque = np.zeros_like(tau)
        prev_torque[1:] = tau[:-1]
        feats = np.concatenate([q, pe, v, sysid_pd, qfrc_bias, prev_torque], axis=1).astype(np.float32)
        n = q.shape[0]
        for s in range(0, n - window_len + 1, window_len):
            xs.append(feats[s : s + window_len])
            ys.append(tau[s : s + window_len])
    return np.stack(xs), np.stack(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-json", default=DEFAULT_SPLIT_JSON)
    parser.add_argument(
        "--data-root",
        default=DATA_ROOT,
        help="Directory of per-experiment motion folders (positions/velocities/torques/targets).",
    )
    parser.add_argument(
        "--model-xml",
        default=None,
        help="MuJoCo MJCF for the G1 right arm. When set, qfrc_bias is "
        "computed from --data-root trajectories instead of loaded "
        "from --qfrc-bias-root caches.",
    )
    parser.add_argument(
        "--joint-names",
        nargs=4,
        default=DEFAULT_ARM_JOINTS,
        help="Four arm joint names in the MJCF, canonical order [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow].",
    )
    parser.add_argument(
        "--qfrc-bias-root",
        default=DEFAULT_QFRC_BIAS_ROOT,
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
    parser.add_argument("--val-count", type=int, default=2)
    parser.add_argument("--val-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out", default=os.path.join(SAVE_DIR, "g1_fulltorque_optuna_best_v2_row10.pt"))
    parser.add_argument("--stats-out", default=os.path.join(SAVE_DIR, "g1_fulltorque_optuna_best_v2_row10_stats.json"))
    parser.add_argument("--script-out", default=None)
    args = parser.parse_args()

    set_global_seed(args.seed)
    os.makedirs(SAVE_DIR, exist_ok=True)
    if args.script_out is None:
        base, ext = os.path.splitext(args.out)
        args.script_out = f"{base}_script{ext}"
    print(f"Training seed: {args.seed}, validation seed: {args.val_seed}")

    train_pool, _test = load_split_json(args.split_json)
    rng = np.random.default_rng(args.val_seed)
    shuffled = sorted(train_pool)
    rng.shuffle(shuffled)
    val_names = sorted(shuffled[: args.val_count])
    train_names = sorted(shuffled[args.val_count :])
    print(f"Split {args.split_json}: {len(train_pool)} train pool → {len(train_names)} train + {len(val_names)} val")
    print(f"  Train: {train_names}")
    print(f"  Val:   {val_names}")

    data_root = os.path.expanduser(args.data_root)
    train_pairs = [(n, load_experiment(os.path.join(data_root, n))) for n in train_names]
    val_pairs = [(n, load_experiment(os.path.join(data_root, n))) for n in val_names]

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
        # G1 data has no per-motion joint metadata: columns are trusted to be in
        # --joint-names order. A mismatch silently corrupts qfrc_bias.
        print(f"[warn] G1: assuming data columns are in --joint-names order {args.joint_names}")
    else:
        print(f"qfrc_bias: load from {args.qfrc_bias_root} (no --model-xml; compute disabled)")

    Xtr, Ytr = build_windowed(train_pairs, WINDOW_LEN, bias_fn)
    Xva, Yva = build_windowed(val_pairs, WINDOW_LEN, bias_fn)
    print(f"Train windows: {Xtr.shape}, Val windows: {Xva.shape}")

    stats = fit_stats(Xtr.reshape(-1, INPUT_SIZE))
    Xtr_n = (Xtr - stats["mean"]) / stats["std"]
    Xva_n = (Xva - stats["mean"]) / stats["std"]

    gru = ForceResidualGRU(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_joints=4,
        force_bound=FORCE_BOUND,
        dropout=DROPOUT,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(gru.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    Xtr_t = torch.from_numpy(Xtr_n).to(DEVICE)
    Ytr_t = torch.from_numpy(Ytr).to(DEVICE).clamp(-FORCE_BOUND, FORCE_BOUND)
    Xva_t = torch.from_numpy(Xva_n).to(DEVICE)
    Yva_t = torch.from_numpy(Yva).to(DEVICE).clamp(-FORCE_BOUND, FORCE_BOUND)

    n_tr = Xtr_t.shape[0]
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        gru.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        running, nb = 0.0, 0
        for i in range(0, n_tr, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            pred, _ = gru(Xtr_t[idx])
            loss = torch.mean((pred - Ytr_t[idx]) ** 2)
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
        print(f"Epoch {epoch:2d}: train {tr:.4f}, val {va:.4f}")
        if va < best_val:
            best_val = va
            torch.save(
                {
                    "model": gru.state_dict(),
                    "epoch": epoch,
                    "val_loss": va,
                    "seed": args.seed,
                    "val_seed": args.val_seed,
                    "train_names": train_names,
                    "val_names": val_names,
                },
                args.out,
            )
            save_stats(stats, args.stats_out)

    ckpt = torch.load(args.out, map_location=DEVICE, weights_only=False)
    gru.load_state_dict(ckpt["model"])
    gru.eval()
    export_model = gru.to("cpu").eval()
    x_example = torch.zeros(1, 1, INPUT_SIZE, dtype=torch.float32)
    h_example = torch.zeros(NUM_LAYERS, 1, HIDDEN_SIZE, dtype=torch.float32)
    with torch.no_grad():
        scripted = torch.jit.trace(export_model, (x_example, h_example))
    scripted.save(args.script_out)
    print(f"Saved TorchScript to {args.script_out}")
    print(f"\nDone. Best val MSE: {best_val:.6f}. Saved to {args.out}")


if __name__ == "__main__":
    main()
