#!/usr/bin/env python3
"""
Stateful GRU for torque prediction.

Uses Truncated Backpropagation Through Time (TBPTT):
  - N_STREAMS parallel CSV streams, each processed in temporal order
  - Hidden state h carried forward within each CSV
  - h reset to zero only when a stream finishes a CSV and starts a new one
  - Gradients detached every CHUNK_LEN steps — prevents exploding gradients
  - Loss computed over ALL timesteps in each chunk (not just the last)

This ensures the GRU trains on realistic non-zero hidden states,
matching the continuous step-by-step inference regime in the actuator.

Normalization stats are computed from data (two-pass loading), not hardcoded.

Inputs : [position, position_error, velocity]
Output : torque
"""

import argparse
import gc
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

# ─── Config ───────────────────────────────────────────────────────────────────

COLS_IN = ["position", "position_error", "velocity"]
COL_OUT = "torque"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_STREAMS    = 64
MIN_CSV_LEN  = 400     # skip CSVs shorter than this (must fit at least 2 chunks)
VAL_CSV_FRAC = 0.15
N_LOAD_WORKERS = 24

N_TRIAL_EPOCHS = 30
N_FINAL_EPOCHS = 150
N_TRIALS       = 30

SEED = 42

# Warm-start Optuna with known-good priors.
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
JOINT_TYPE: str             = "all"
NOISE_STD: float            = 0.0


# ─── Data loading (two-pass) ─────────────────────────────────────────────────

def _csv_to_raw_df(
    path: str, cols_in: list[str], col_out: str,
) -> pd.DataFrame | None:
    """Parse one CSV -> raw DataFrame (no normalization)."""
    try:
        df = pd.read_csv(path, usecols=cols_in + [col_out]).dropna()
        if len(df) < MIN_CSV_LEN:
            return None
        return df
    except Exception as exc:
        print(f"  [skip] {path}: {exc}")
        return None


def compute_stats(
    all_csvs: list[str],
    cols_in: list[str],
    col_out: str,
) -> dict:
    """Pass 1: load all CSVs raw, compute normalization stats.

    Returns dict mapping column_name -> {"mean", "std", "p1", "p99"}.
    """
    all_dfs: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N_LOAD_WORKERS) as pool:
        futures = {pool.submit(_csv_to_raw_df, p, cols_in, col_out): p
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

    stats: dict[str, dict] = {}
    for col in cols_in + [col_out]:
        vals = combined[col].to_numpy()
        stats[col] = {
            "mean": float(np.mean(vals)),
            "std":  float(max(np.std(vals), 1e-8)),
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


def _csv_to_array(
    path: str,
    cols_in: list[str],
    col_out: str,
    stats: dict,
) -> np.ndarray | None:
    """Pass 2: parse one CSV -> (N, 4) float32 array [pos, pos_err, vel, torque].

    All columns are clip-normalized using computed stats.
    """
    try:
        df = pd.read_csv(path, usecols=cols_in + [col_out]).dropna()
        if len(df) < MIN_CSV_LEN:
            return None

        for col in cols_in + [col_out]:
            s = stats[col]
            df[col] = df[col].clip(s["p1"], s["p99"])
            df[col] = (df[col] - s["mean"]) / s["std"]

        return df[cols_in + [col_out]].to_numpy(dtype=np.float32)
    except Exception as exc:
        print(f"  [skip] {path}: {exc}")
        return None


def load_raw_arrays(
    data_dirs: list[str],
    cols_in: list[str],
    col_out: str,
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
    print("[pass 1] computing normalization statistics ...")
    stats = compute_stats(all_csvs, cols_in, col_out)

    # Pass 2: re-load with normalization
    print("[pass 2] loading and normalizing ...")
    loader = partial(_csv_to_array, cols_in=cols_in, col_out=col_out, stats=stats)

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

    # Denormalize RMSE using computed torque stats
    torque_std = _cache["stats"][COL_OUT_RESOLVED]["std"]
    rmse_nm = ((running_sq.item() / n_total) ** 0.5) * torque_std
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

    torque_std = _cache["stats"][COL_OUT_RESOLVED]["std"]
    return ((total_sq.item() / total_n) ** 0.5) * torque_std


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
    print(f"Final retraining (stateful TBPTT) — {epochs} epochs")
    print(f"Params: {params}")
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
    model_filename = f"gru_{JOINT_TYPE}_stateful_best.pt"

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
    full_rmse = eval_stateful(model, tr_tensors + val_tensors, cl)

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
                "model_type":           "gru",
                "params":               {k: _j(v) for k, v in params.items()},
                "normalization":        stats,
                "best_val_rmse_nm":     best_val_rmse,
                "full_dataset_rmse_nm": full_rmse,
                "cols_in":              COLS_IN_RESOLVED,
                "col_out":              COL_OUT_RESOLVED,
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
    global DATA_DIRS, COLS_IN_RESOLVED, COL_OUT_RESOLVED, JOINT_TYPE

    parser = argparse.ArgumentParser(
        description="Stateful GRU torque prediction — TBPTT"
    )

    # Data sources
    parser.add_argument("--data-dirs", type=str, nargs="+", required=True,
                        help="Directories containing training CSV files")
    parser.add_argument("--joint-type", type=str, default=None,
                        help="Joint type for column prefix "
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
                             "(default: 0)")
    parser.add_argument("--n-streams",    type=int, default=N_STREAMS,
                        help=f"Parallel TBPTT streams (default {N_STREAMS})")
    parser.add_argument("--study-name",   type=str,
                        default="torque_gru_stateful")
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

    print(f"{'='*60}")
    print(f"Stateful GRU Torque Training")
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
    print(f"Columns    : {COLS_IN_RESOLVED} -> {COL_OUT_RESOLVED}")
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
          f"(stateful TBPTT, N_STREAMS={N_STREAMS})")
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
