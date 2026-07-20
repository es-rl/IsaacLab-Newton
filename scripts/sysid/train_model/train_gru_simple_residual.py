#!/usr/bin/env python3
"""
Hybrid residual GRU for torque prediction.

Trains on the residual:  tau_residual = tau_real - (kp * pos_error - kd * vel)

Instead of predicting full motor torque, the GRU learns only what the simple
PD model cannot explain.  Since the residual is much smaller than full torque,
prediction errors compound far less during closed-loop simulation in Newton.

Same architecture and TBPTT training as train_torque_gru_sysid_stateful.py:
  - TorqueGRU (GRU + linear head), 3 inputs -> 1 output
  - StreamManager with N_STREAMS parallel CSV streams
  - Stateful hidden state carried within each CSV, reset at boundaries
  - Optuna hyperparameter search + final retrain

Additional features:
  - Physics params (kp, kd) loaded from implicit actuator YAML
  - Two-pass data loading: compute residual stats, then normalize
  - Optional Gaussian noise injection for robustness to sim state drift
  - Stats JSON tagged with "model_type": "hybrid_residual"

Inputs : [position, position_error, velocity]
Output : torque_residual
"""

import argparse
import gc
import glob
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import yaml
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

# ─── Config ───────────────────────────────────────────────────────────────────

COLS_IN = ["position", "position_error", "velocity"]
COL_OUT = "torque"
COL_RESIDUAL = "torque_residual"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_STREAMS    = 64
MIN_CSV_LEN  = 400     # skip CSVs shorter than this (must fit at least 2 chunks)
VAL_CSV_FRAC = 0.15
N_LOAD_WORKERS = 24

N_TRIAL_EPOCHS = 30
N_FINAL_EPOCHS = 150
N_TRIALS       = 30

SEED = 42

# Warm-start Optuna (same priors as non-hybrid GRU baseline).
DEFAULT_PARAMS = {
    "hidden_dim":   256,
    "num_layers":   4,
    "chunk_len":    100,
    "lr":           1.758e-3,
    "dropout":      0.156,
    "huber_delta":  1.293,
    "weight_decay": 1.99e-5,
}

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
SAVE_DIR = os.path.join(_PROJECT_ROOT, "input", "actuator_models", "h1")

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark        = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Module-level globals set by main() after CLI parsing.
DATA_DIRS: list[str]       = []
COLS_IN_RESOLVED: list[str] = list(COLS_IN)
COL_OUT_RESOLVED: str       = COL_OUT
PHYSICS_PARAMS: dict        = {}
JOINT_TYPE: str             = "all"
NOISE_STD: float            = 0.0


# ─── YAML loading ─────────────────────────────────────────────────────────────

def _resolve_joint_param(
    regex_dict: dict, joint_type: str | None, param_name: str,
) -> float:
    """Match joint_type against regex keys in YAML dict.

    The YAML has entries like {".*_elbow": 0.021208, ".*_shoulder_pitch": 0.001867}.
    We try matching a synthetic full joint name (e.g. "right_elbow") against each
    regex key.
    """
    if joint_type is None:
        return float(next(iter(regex_dict.values())))

    for pattern, value in regex_dict.items():
        if re.search(pattern, f"right_{joint_type}") or re.search(pattern, joint_type):
            return float(value)

    raise ValueError(
        f"No match for joint_type={joint_type!r} in {param_name} regex dict: "
        f"{list(regex_dict.keys())}"
    )


def load_physics_params(yaml_path: str, joint_type: str | None) -> dict:
    """Load kp, kd, and per-joint mechanical params from an implicit actuator YAML.

    Returns dict with keys: kp, kd, armature, dynamic_friction, viscous_friction.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    result = {
        "kp": float(cfg["stiffness"]),
        "kd": float(cfg["damping"]),
    }

    for key in ("armature", "dynamic_friction", "viscous_friction"):
        val = cfg.get(key, 0.0)
        if isinstance(val, dict):
            result[key] = _resolve_joint_param(val, joint_type, key)
        else:
            result[key] = float(val)

    return result


# ─── Data loading (two-pass) ──────────────────────────────────────────────────

def _csv_to_raw_df(
    path: str, cols_in: list[str], col_torque: str,
) -> pd.DataFrame | None:
    """Parse one CSV -> raw DataFrame (no normalization)."""
    try:
        df = pd.read_csv(path, usecols=cols_in + [col_torque]).dropna()
        if len(df) < MIN_CSV_LEN:
            return None
        return df
    except Exception as exc:
        print(f"  [skip] {path}: {exc}")
        return None


def compute_residual_and_stats(
    all_csvs: list[str],
    cols_in: list[str],
    col_torque: str,
    kp: float,
    kd: float,
    dynamic_friction: float = 0.0,
    viscous_friction: float = 0.0,
) -> dict:
    """Pass 1: load all CSVs raw, compute tau_residual, compute normalization stats.

    The residual is: tau_real - (kp * pos_error - kd * vel)
    Optionally subtracts friction terms if provided (non-zero):
        - dynamic_friction * sign(vel)  (Coulomb friction)
        - viscous_friction * vel        (viscous damping)

    Returns dict mapping column_name -> {"mean", "std", "p1", "p99"}.
    """
    all_dfs: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N_LOAD_WORKERS) as pool:
        futures = {pool.submit(_csv_to_raw_df, p, cols_in, col_torque): p
                   for p in all_csvs}
        done = 0
        for future in as_completed(futures):
            df = future.result()
            if df is not None:
                all_dfs.append(df)
            done += 1
            if done % 20 == 0 or done == len(futures):
                print(f"  [pass 1] {done}/{len(futures)} CSVs  "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)

    if not all_dfs:
        raise RuntimeError("No valid CSVs loaded — check --data-dirs and column names")

    combined = pd.concat(all_dfs, ignore_index=True)

    # Compute residual: tau_real - PD_torque - friction_torque
    col_pos_err = cols_in[1]
    col_vel     = cols_in[2]
    pd_torque = kp * combined[col_pos_err] - kd * combined[col_vel]
    friction_torque = 0.0
    if dynamic_friction != 0.0:
        friction_torque = friction_torque + dynamic_friction * np.sign(combined[col_vel])
    if viscous_friction != 0.0:
        friction_torque = friction_torque + viscous_friction * combined[col_vel]
    combined[COL_RESIDUAL] = combined[col_torque] - pd_torque - friction_torque

    # Compute stats for all columns
    stats: dict[str, dict] = {}
    for col in cols_in + [COL_RESIDUAL]:
        vals = combined[col].to_numpy()
        stats[col] = {
            "mean": float(np.mean(vals)),
            "std":  float(max(np.std(vals), 1e-8)),   # guard against zero std
            "p1":   float(np.percentile(vals, 1)),
            "p99":  float(np.percentile(vals, 99)),
        }

    n_samples = len(combined)
    print(f"\n[stats] computed from {n_samples:,} samples across {len(all_dfs)} CSVs")
    for col, s in stats.items():
        print(f"  {col:25s}  mean={s['mean']:+.4f}  std={s['std']:.4f}  "
              f"p1={s['p1']:+.4f}  p99={s['p99']:+.4f}")
    print()

    return stats


def _csv_to_array_hybrid(
    path: str,
    cols_in: list[str],
    col_torque: str,
    kp: float,
    kd: float,
    stats: dict,
    dynamic_friction: float = 0.0,
    viscous_friction: float = 0.0,
) -> np.ndarray | None:
    """Pass 2: parse one CSV -> (N, 4) float32 array [pos, pos_err, vel, residual].

    Residual is computed on raw data, then all columns are clip-normalized.
    """
    try:
        df = pd.read_csv(path, usecols=cols_in + [col_torque]).dropna()
        if len(df) < MIN_CSV_LEN:
            return None

        # Compute residual (raw physical units)
        col_pos_err = cols_in[1]
        col_vel     = cols_in[2]
        pd_torque = kp * df[col_pos_err] - kd * df[col_vel]
        friction_torque = 0.0
        if dynamic_friction != 0.0:
            friction_torque = friction_torque + dynamic_friction * np.sign(df[col_vel])
        if viscous_friction != 0.0:
            friction_torque = friction_torque + viscous_friction * df[col_vel]
        df[COL_RESIDUAL] = df[col_torque] - pd_torque - friction_torque

        # Clip and normalize all columns
        for col in cols_in + [COL_RESIDUAL]:
            s = stats[col]
            df[col] = df[col].clip(s["p1"], s["p99"])
            df[col] = (df[col] - s["mean"]) / s["std"]

        return df[cols_in + [COL_RESIDUAL]].to_numpy(dtype=np.float32)
    except Exception as exc:
        print(f"  [skip] {path}: {exc}")
        return None


def load_raw_arrays(
    data_dirs: list[str],
    cols_in: list[str],
    col_torque: str,
    kp: float,
    kd: float,
    dynamic_friction: float = 0.0,
    viscous_friction: float = 0.0,
) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """Two-pass data loading: compute stats, then normalize.

    Returns (train_arrays, val_arrays, stats).
    """
    all_csvs: list[str] = []
    for d in data_dirs:
        all_csvs.extend(
            sorted(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))
        )
    print(f"[data] found {len(all_csvs)} CSV files in {len(data_dirs)} directories")

    if not all_csvs:
        raise RuntimeError(f"No CSV files found in: {data_dirs}")

    rng = np.random.default_rng(SEED)
    rng.shuffle(all_csvs)
    n_val    = max(1, int(len(all_csvs) * VAL_CSV_FRAC))
    val_csvs = all_csvs[:n_val]
    tr_csvs  = all_csvs[n_val:]

    # Pass 1: compute stats from all CSVs
    print("[pass 1] computing residual statistics ...")
    stats = compute_residual_and_stats(
        all_csvs, cols_in, col_torque, kp, kd,
        dynamic_friction=dynamic_friction, viscous_friction=viscous_friction,
    )

    # Pass 2: re-load with normalization
    print("[pass 2] loading and normalizing ...")
    loader = partial(
        _csv_to_array_hybrid,
        cols_in=cols_in, col_torque=col_torque,
        kp=kp, kd=kd, stats=stats,
        dynamic_friction=dynamic_friction, viscous_friction=viscous_friction,
    )

    def _load_list(paths: list[str], label: str) -> list[np.ndarray]:
        results: dict[int, np.ndarray] = {}
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=N_LOAD_WORKERS) as pool:
            futures = {pool.submit(loader, p): i for i, p in enumerate(paths)}
            done = 0
            for future in as_completed(futures):
                idx = futures[future]
                arr = future.result()
                if arr is not None:
                    results[idx] = arr
                done += 1
                if done % 20 == 0 or done == len(paths):
                    print(f"  [{label}] {done}/{len(paths)} CSVs  "
                          f"({time.perf_counter() - t0:.1f}s)", flush=True)
        return [results[i] for i in sorted(results)]

    tr_arrays  = _load_list(tr_csvs,  "train")
    val_arrays = _load_list(val_csvs, "val")

    print(f"  [train] {len(tr_arrays)} CSVs, "
          f"{sum(len(a) for a in tr_arrays):,} samples")
    print(f"  [val]   {len(val_arrays)} CSVs, "
          f"{sum(len(a) for a in val_arrays):,} samples")
    return tr_arrays, val_arrays, stats


def _arrays_to_gpu(arrays: list[np.ndarray], label: str) -> list[torch.Tensor]:
    """Upload list of float32 numpy arrays -> list of BF16 GPU tensors."""
    tensors = [
        torch.from_numpy(a).to(device=DEVICE, dtype=torch.bfloat16)
        for a in arrays
    ]
    total_gb = sum(t.untyped_storage().nbytes() for t in tensors) / 1e9
    print(f"  {label}: {len(tensors)} tensors, {total_gb:.2f} GB on GPU")
    return tensors


# Global cache — loaded once, shared across all Optuna trials.
_cache: dict = {}


def get_gpu_data() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if "tr" not in _cache:
        tr_arrays, val_arrays, stats = load_raw_arrays(
            DATA_DIRS, COLS_IN_RESOLVED, COL_OUT_RESOLVED,
            PHYSICS_PARAMS["kp"], PHYSICS_PARAMS["kd"],
            dynamic_friction=PHYSICS_PARAMS.get("dynamic_friction", 0.0),
            viscous_friction=PHYSICS_PARAMS.get("viscous_friction", 0.0),
        )
        _cache["stats"] = stats

        print("[gpu] uploading to VRAM (BF16) ...")
        _cache["tr"]  = _arrays_to_gpu(tr_arrays,  "train")
        _cache["val"] = _arrays_to_gpu(val_arrays, "val")

        del tr_arrays, val_arrays
        torch.cuda.empty_cache()
        if DEVICE.type == "cuda":
            used  = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[gpu] VRAM used: {used:.2f} / {total:.1f} GB")

    return _cache["tr"], _cache["val"]


# ─── Stream manager ───────────────────────────────────────────────────────────

class StreamManager:
    """Manages N_STREAMS parallel CSV streams for stateful TBPTT training.

    Each stream maintains a position pointer into a CSV.  When it exhausts
    that CSV it picks the next one from a shuffled pool and flags itself for
    a hidden-state reset.
    """

    def __init__(
        self,
        tensors:   list[torch.Tensor],
        n_streams: int,
        chunk_len: int,
    ):
        self.tensors   = tensors
        self.n         = n_streams
        self.chunk_len = chunk_len

        order = torch.randperm(len(tensors)).tolist()
        self._pool     = order * ((n_streams // max(len(order), 1)) + 2)
        self._next_csv = 0

        self._csv_idx    = [0] * n_streams
        self._pos        = [0] * n_streams
        self._reset_flag = [True] * n_streams

        for i in range(n_streams):
            self._assign(i)

    def _assign(self, i: int):
        self._csv_idx[i] = self._pool[self._next_csv % len(self._pool)]
        self._next_csv  += 1
        self._pos[i]     = 0

    def next_chunk(self) -> tuple[torch.Tensor, torch.Tensor, list[bool]]:
        xs, ys     = [], []
        reset_mask       = list(self._reset_flag)
        self._reset_flag = [False] * self.n

        for i in range(self.n):
            arr = self.tensors[self._csv_idx[i]]
            end = self._pos[i] + self.chunk_len

            if end > len(arr):
                self._assign(i)
                arr          = self.tensors[self._csv_idx[i]]
                end          = self.chunk_len
                reset_mask[i] = True

            xs.append(arr[self._pos[i]:end, :3])   # (chunk_len, 3) BF16
            ys.append(arr[self._pos[i]:end,  3])   # (chunk_len,)   BF16
            self._pos[i] = end

        x = torch.stack(xs)          # (N, chunk_len, 3)
        y = torch.stack(ys).float()  # (N, chunk_len)  float32
        return x, y, reset_mask

    def steps_per_epoch(self) -> int:
        total = sum(len(t) for t in self.tensors)
        return max(1, total // (self.n * self.chunk_len))


# ─── Model ────────────────────────────────────────────────────────────────────

class TorqueGRU(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=3,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, h_new = self.gru(x, h)
        pred = self.head(out).squeeze(-1)
        return pred, h_new


# ─── Epoch runners ────────────────────────────────────────────────────────────

def train_epoch(
    model:      nn.Module,
    tr_tensors: list[torch.Tensor],
    chunk_len:  int,
    optimizer:  torch.optim.Optimizer,
    loss_fn:    nn.Module,
    noise_std:  float = 0.0,
) -> tuple[float, float]:
    model.train()
    mgr   = StreamManager(tr_tensors, N_STREAMS, chunk_len)
    steps = mgr.steps_per_epoch()

    h: torch.Tensor | None = None

    running_loss = torch.zeros(1, device=DEVICE)
    running_sq   = torch.zeros(1, device=DEVICE)
    n_total      = 0

    for _ in range(steps):
        x, y, reset_mask = mgr.next_chunk()

        # TBPTT: detach gradient graph but keep hidden state values
        if h is not None:
            h = h.detach()
            if any(reset_mask):
                mask = torch.tensor(reset_mask, device=DEVICE)
                h[:, mask, :] = 0.0

        # Noise injection: add Gaussian noise to normalized inputs
        if noise_std > 0.0:
            x = x + torch.randn_like(x) * noise_std

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred, h = model(x, h)
            pred = pred.float()
            loss = loss_fn(pred, y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            running_sq   += ((pred.detach() - y) ** 2).sum()
            running_loss += loss.detach()
        n_total += N_STREAMS * chunk_len

    # Denormalize RMSE using computed residual stats
    residual_std = _cache["stats"][COL_RESIDUAL]["std"]
    rmse_nm = ((running_sq.item() / n_total) ** 0.5) * residual_std
    return running_loss.item() / steps, rmse_nm


@torch.inference_mode()
def eval_stateful(
    model:       nn.Module,
    val_tensors: list[torch.Tensor],
    chunk_len:   int,
) -> float:
    """Evaluate each val CSV as a continuous stream with h carried forward."""
    model.eval()
    total_sq = torch.zeros(1, device=DEVICE)
    total_n  = 0

    for arr in val_tensors:
        h = None
        n = len(arr)

        for pos in range(0, n, chunk_len):
            end = min(pos + chunk_len, n)
            x   = arr[pos:end, :3].unsqueeze(0)
            y   = arr[pos:end,  3].float()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred, h = model(x, h)
            h    = h.detach()
            pred = pred.squeeze(0).float()

            total_sq += ((pred - y) ** 2).sum()
            total_n  += len(y)

    residual_std = _cache["stats"][COL_RESIDUAL]["std"]
    return ((total_sq.item() / total_n) ** 0.5) * residual_std


# ─── Optuna objective ─────────────────────────────────────────────────────────

def objective(trial: optuna.Trial) -> float:
    hidden_dim   = trial.suggest_categorical("hidden_dim",  [128, 256, 512])
    num_layers   = trial.suggest_int(        "num_layers",  2, 4)
    chunk_len    = trial.suggest_categorical("chunk_len",   [50, 100, 150, 200])
    lr           = trial.suggest_float(      "lr",          1e-4, 5e-3, log=True)
    dropout      = trial.suggest_float(      "dropout",     0.0, 0.3)
    huber_delta  = trial.suggest_float(      "huber_delta", 0.3, 2.0)
    weight_decay = trial.suggest_float(      "weight_decay",1e-6, 1e-3, log=True)

    tr_tensors, val_tensors = get_gpu_data()

    model     = TorqueGRU(hidden_dim, num_layers, dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_TRIAL_EPOCHS, eta_min=lr * 0.1
    )
    loss_fn   = nn.HuberLoss(delta=huber_delta)

    best_val = float("inf")
    try:
        for epoch in range(N_TRIAL_EPOCHS):
            train_epoch(model, tr_tensors, chunk_len, optimizer, loss_fn,
                        noise_std=NOISE_STD)
            scheduler.step()

            val_rmse = eval_stateful(model, val_tensors, chunk_len)
            best_val = min(best_val, val_rmse)

            trial.report(val_rmse, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    except torch.cuda.OutOfMemoryError:
        print(f"  [trial {trial.number}] OOM "
              f"(hidden={hidden_dim}, layers={num_layers}, chunk={chunk_len})",
              flush=True)
        raise optuna.exceptions.TrialPruned()
    finally:
        del model, optimizer, loss_fn, scheduler
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return best_val


# ─── Final retraining ─────────────────────────────────────────────────────────

def retrain_best(params: dict, epochs: int = N_FINAL_EPOCHS) -> float:
    print(f"\n{'='*60}")
    print(f"Final retraining (hybrid residual, stateful TBPTT) — {epochs} epochs")
    print(f"Params: {params}")
    print(f"Physics: kp={PHYSICS_PARAMS['kp']}, kd={PHYSICS_PARAMS['kd']}")
    print(f"Noise std: {NOISE_STD}")
    print("=" * 60)

    tr_tensors, val_tensors = get_gpu_data()
    stats = _cache["stats"]

    hd  = params["hidden_dim"]
    nl  = params["num_layers"]
    cl  = params["chunk_len"]
    lr  = params["lr"]
    do  = params["dropout"]
    hd_ = params["huber_delta"]
    wd  = params["weight_decay"]

    model     = TorqueGRU(hd, nl, do).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.05
    )
    loss_fn   = nn.HuberLoss(delta=hd_)

    best_val_rmse = float("inf")
    best_state    = None
    model_filename = f"hybrid_residual_{JOINT_TYPE}_best.pt"

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_rmse = train_epoch(
            model, tr_tensors, cl, optimizer, loss_fn, noise_std=NOISE_STD,
        )
        val_rmse = eval_stateful(model, val_tensors, cl)
        scheduler.step()
        dt = time.perf_counter() - t0

        print(
            f"  ep {epoch:3d}/{epochs}"
            f"  loss {tr_loss:.4f}"
            f"  tr_rmse {tr_rmse:.4f} Nm"
            f"  val_rmse {val_rmse:.4f} Nm"
            f"  {dt:.1f}s",
            flush=True,
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state,
                       os.path.join(SAVE_DIR, model_filename))
            print(f"    -> saved  val_rmse={best_val_rmse:.4f} Nm", flush=True)

    # Evaluate best checkpoint on full dataset
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    # Use larger inference chunks to avoid tiny batch-size-one GPU launches.
    full_rmse = eval_stateful(model, tr_tensors + val_tensors, max(cl, 4096))

    model_path = os.path.join(SAVE_DIR, model_filename)
    stats_filename = model_filename.replace("_best.pt", "_stats.json")
    stats_path = os.path.join(SAVE_DIR, stats_filename)

    torch.save(best_state, model_path)

    def _j(v):
        if isinstance(v, (np.integer,  np.int64)):               return int(v)
        if isinstance(v, (np.floating, np.float32, np.float64)): return float(v)
        return v

    with open(stats_path, "w") as f:
        json.dump(
            {
                "model_type":           "hybrid_residual",
                "subtract_friction":    PHYSICS_PARAMS.get("dynamic_friction", 0.0) != 0.0
                                        or PHYSICS_PARAMS.get("viscous_friction", 0.0) != 0.0,
                "physics_params":       PHYSICS_PARAMS,
                "params":               {k: _j(v) for k, v in params.items()},
                "normalization":        stats,
                "best_val_rmse_nm":     best_val_rmse,
                "full_dataset_rmse_nm": full_rmse,
                "cols_in":              COLS_IN_RESOLVED,
                "col_out":              COL_RESIDUAL,
                "n_streams":            N_STREAMS,
                "noise_std":            NOISE_STD,
                "data_dirs":            DATA_DIRS,
                "joint_type":           JOINT_TYPE,
            },
            f,
            indent=2,
        )

    print(f"\n  Best val RMSE     : {best_val_rmse:.4f} Nm  (held-out)")
    print(f"  Full dataset RMSE : {full_rmse:.4f} Nm  (train+val)")
    print(f"  Model saved       : {model_path}")
    print(f"  Stats saved       : {stats_path}")
    return best_val_rmse


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global N_TRIAL_EPOCHS, N_FINAL_EPOCHS, N_TRIALS, N_STREAMS, NOISE_STD, SAVE_DIR
    global DATA_DIRS, COLS_IN_RESOLVED, COL_OUT_RESOLVED, PHYSICS_PARAMS, JOINT_TYPE

    parser = argparse.ArgumentParser(
        description="Hybrid residual GRU — trains on tau_residual = tau_real - (kp*pos_err - kd*vel)"
    )

    # Data sources
    parser.add_argument("--data-dirs", type=str, nargs="+", required=True,
                        help="Directories containing training CSV files")
    parser.add_argument("--implicit-yaml", type=str, required=True,
                        help="Path to implicit actuator YAML (e.g., h1_arm_implicit.yaml)")
    parser.add_argument("--joint-type", type=str, default=None,
                        help="Joint type for per-joint params and column prefix "
                             "(e.g., 'elbow', 'shoulder_pitch'). "
                             "If set, CSV columns become {jt}_position, etc.")

    # Training control
    parser.add_argument("--trials",       type=int, default=N_TRIALS,
                        help=f"Number of Optuna trials (default {N_TRIALS})")
    parser.add_argument("--trial-epochs", type=int, default=N_TRIAL_EPOCHS,
                        help=f"Epochs per trial (default {N_TRIAL_EPOCHS})")
    parser.add_argument("--final-epochs", type=int, default=N_FINAL_EPOCHS,
                        help=f"Epochs for final retrain (default {N_FINAL_EPOCHS})")
    parser.add_argument("--noise-std",    type=float, default=0.0,
                        help="Gaussian noise std on normalized inputs during training "
                             "(default: 0, recommended: 0.01-0.05)")
    parser.add_argument("--n-streams",    type=int, default=N_STREAMS,
                        help=f"Parallel TBPTT streams (default {N_STREAMS})")
    parser.add_argument("--study-name",   type=str,
                        default="hybrid_residual_gru")
    parser.add_argument("--storage",      type=str, default=None,
                        help="Optuna storage URL, e.g. sqlite:///optuna.db")
    parser.add_argument("--save-dir",     type=str, default=None,
                        help="Output directory for model .pt and _stats.json "
                             "(default: input/actuator_models/h1/)")
    parser.add_argument("--skip-optuna",  action="store_true",
                        help="Skip Optuna; retrain with DEFAULT_PARAMS")
    args = parser.parse_args()

    # Set globals from CLI
    N_TRIAL_EPOCHS = args.trial_epochs
    N_FINAL_EPOCHS = args.final_epochs
    N_TRIALS       = args.trials
    N_STREAMS      = args.n_streams
    NOISE_STD      = args.noise_std
    DATA_DIRS      = args.data_dirs
    JOINT_TYPE     = args.joint_type or "all"
    if args.save_dir:
        SAVE_DIR = args.save_dir
        os.makedirs(SAVE_DIR, exist_ok=True)

    # Resolve column names based on joint_type
    if args.joint_type:
        jt = args.joint_type
        COLS_IN_RESOLVED = [f"{jt}_position", f"{jt}_position_error", f"{jt}_velocity"]
        COL_OUT_RESOLVED = f"{jt}_torque"
    else:
        COLS_IN_RESOLVED = list(COLS_IN)
        COL_OUT_RESOLVED = COL_OUT

    # Load physics params from YAML
    PHYSICS_PARAMS = load_physics_params(args.implicit_yaml, args.joint_type)

    print(f"{'='*60}")
    print(f"Hybrid Residual GRU Training")
    print(f"{'='*60}")
    print(f"Device     : {DEVICE}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU        : {props.name}")
        print(f"VRAM       : {props.total_memory / 1e9:.1f} GB")
        print(f"PyTorch    : {torch.__version__}")
    print(f"Joint type : {JOINT_TYPE}")
    print(f"Streams    : {N_STREAMS}")
    print(f"Noise std  : {NOISE_STD}")
    print(f"Save dir   : {SAVE_DIR}")
    print(f"Columns    : {COLS_IN_RESOLVED} -> {COL_RESIDUAL}")
    print(f"\nPhysics params (from {args.implicit_yaml}):")
    print(f"  kp (stiffness)       = {PHYSICS_PARAMS['kp']}")
    print(f"  kd (damping)         = {PHYSICS_PARAMS['kd']}")
    print(f"  armature             = {PHYSICS_PARAMS['armature']}")
    print(f"  dynamic_friction     = {PHYSICS_PARAMS['dynamic_friction']}")
    print(f"  viscous_friction     = {PHYSICS_PARAMS['viscous_friction']}")
    residual_eq = (f"tau_residual = tau_real - ({PHYSICS_PARAMS['kp']} * pos_error "
                   f"- {PHYSICS_PARAMS['kd']} * velocity)")
    if PHYSICS_PARAMS.get("dynamic_friction", 0.0) != 0.0:
        residual_eq += f" - {PHYSICS_PARAMS['dynamic_friction']} * sign(vel)"
    if PHYSICS_PARAMS.get("viscous_friction", 0.0) != 0.0:
        residual_eq += f" - {PHYSICS_PARAMS['viscous_friction']} * vel"
    print(f"\nResidual: {residual_eq}")
    print()

    if args.skip_optuna:
        retrain_best(DEFAULT_PARAMS, epochs=N_FINAL_EPOCHS)
        return

    study = optuna.create_study(
        study_name     = args.study_name,
        direction      = "minimize",
        sampler        = TPESampler(seed=SEED, multivariate=True),
        pruner         = HyperbandPruner(
            min_resource=5, max_resource=N_TRIAL_EPOCHS, reduction_factor=3
        ),
        storage        = args.storage,
        load_if_exists = True,
    )

    study.enqueue_trial(DEFAULT_PARAMS)

    print(f"[optuna] starting {N_TRIALS} trials  "
          f"(hybrid residual, stateful TBPTT, N_STREAMS={N_STREAMS})")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Best trial #{best.number}  val RMSE: {best.value:.4f} Nm")
    for k, v in best.params.items():
        print(f"  {k:16s}: {v}")

    results_path = os.path.join(SAVE_DIR, f"{args.study_name}_optuna_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "best_value":  best.value,
                "best_params": best.params,
                "n_trials":    len(study.trials),
            },
            f,
            indent=2,
        )
    print(f"Study results -> {results_path}")

    final_rmse = retrain_best(best.params, epochs=N_FINAL_EPOCHS)
    print(f"\nDone.  Final best val RMSE: {final_rmse:.4f} Nm")


if __name__ == "__main__":
    main()
