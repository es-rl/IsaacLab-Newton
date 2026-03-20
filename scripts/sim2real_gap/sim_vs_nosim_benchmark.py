# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""No-sim actuator model evaluation — feeds real data directly through the model.

Bypasses Newton / Isaac Lab entirely. Real position, velocity, and commanded
position are fed directly into the actuator model and the predicted torque is
compared against measured real torque.

Supported model types:
  - ``implicit``: Pure PD control (kp * error - kd * vel)
  - ``dcmotor``: DC motor model (PD + saturation/effort limits)
  - ``lstm`` / ``gru``: Stateful LSTM/GRU neural network (single model)
  - ``lstm_perjoint`` / ``gru_perjoint``: Per-joint LSTM/GRU models
  - ``fmu``: FMI 2.0 CoSimulation FMU (Ansys Twin Builder)

Optionally overlays sim-in-the-loop results from a prior benchmark run
(``--sim-results``) so you can compare all three on the same plot:
  real data  |  no-sim (direct model)  |  sim-in-the-loop

Outputs: per-file PNG plots, summary CSV, and a PDF report with config info,
summary table, and all plots.

Supports parquet files (benchtop motor data) and motor CSVs.

Usage:
    python sim_vs_nosim_benchmark.py --robot-name teststand
    python sim_vs_nosim_benchmark.py --config input/run_configs/h1/h1.yaml --model-type lstm
    python sim_vs_nosim_benchmark.py --robot-name teststand --sim-results output/sim2real_benchmark
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

sys.path.insert(0, os.path.join(REPO_ROOT, "input"))
sys.path.insert(0, os.path.join(REPO_ROOT, "input", "run_configs"))

from actuator_models import load_actuator_params  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from run_configs import load_run_cfg  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_parquet(path):
    """Load a single parquet file -> (time_s, position, velocity, torque, commanded)."""
    import pandas as pd

    df = pd.read_parquet(path)
    time_raw = df["time"].values.astype(np.float64)

    time_s = time_raw - time_raw[0]
    if time_s[-1] > 1e6:
        time_s = time_s / 1e6
    if time_s[-1] > 1e3:
        time_s = time_s / 1e3

    position = df["position"].values.astype(np.float64)
    velocity = df["velocity"].values.astype(np.float64)
    torque = df["torque"].values.astype(np.float64)

    if "commanded_position" in df.columns:
        commanded = df["commanded_position"].values.astype(np.float64)
    elif "position_error" in df.columns:
        commanded = position + df["position_error"].values.astype(np.float64)
    else:
        raise ValueError(f"No commanded_position or position_error in {path}")

    return time_s, position, velocity, torque, commanded


def load_motor_csv(path, joint_name="elbow"):
    """Load a *_motor.csv file -> (time_s, position, velocity, torque, commanded)."""
    import pandas as pd

    df = pd.read_csv(path)
    time_s = df["time_s"].values.astype(np.float64)
    time_s = time_s - time_s[0]

    position = df[f"{joint_name}_position"].values.astype(np.float64)
    velocity = df[f"{joint_name}_velocity"].values.astype(np.float64)
    torque = df[f"{joint_name}_torque"].values.astype(np.float64)

    if f"{joint_name}_position_error" in df.columns:
        pos_err = df[f"{joint_name}_position_error"].values.astype(np.float64)
        commanded = position + pos_err
    else:
        commanded = position.copy()

    return time_s, position, velocity, torque, commanded


def find_data_files(data_dir):
    """Find parquet and motor CSV files in a directory.

    Prefers parquet files. Motor CSVs are only included if no parquet
    exists for the same motion (avoids duplicate processing).
    """
    parquets = {}
    csvs = {}
    for f in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, f)
        # Derive base name for dedup
        name = os.path.splitext(f)[0]
        for suffix in ("_combined_clean", "_combined", "_clean", "_motor"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        if f.endswith(".parquet"):
            parquets[name] = ("parquet", full)
        elif f.endswith("_motor.csv"):
            csvs[name] = ("csv", full)

    # Parquets take priority; only use CSV if no parquet for that motion
    merged = dict(parquets)
    for name, entry in csvs.items():
        if name not in merged:
            merged[name] = entry

    return list(merged.values())


def load_sage_state_motor(state_motor_csv, joint_name, joint_list_path=None):
    """Load sim torque from a SAGE state_motor.csv file.

    Returns (time_s, torque) arrays for the requested joint.
    """
    # Read joint order
    joint_names = None
    if joint_list_path and os.path.exists(joint_list_path):
        with open(joint_list_path) as f:
            joint_names = [line.strip() for line in f if line.strip()]

    joint_idx = None
    if joint_names:
        for i, name in enumerate(joint_names):
            if name == joint_name or name.endswith(f"_{joint_name}"):
                joint_idx = i
                break

    times = []
    torques = []
    with open(state_motor_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["timestamp"])
            torque_list = ast.literal_eval(row["torques"])
            times.append(ts / 1e6)  # microseconds -> seconds
            if joint_idx is not None and joint_idx < len(torque_list):
                torques.append(torque_list[joint_idx])
            elif len(torque_list) == 1:
                torques.append(torque_list[0])
            else:
                torques.append(0.0)

    time_s = np.array(times)
    time_s = time_s - time_s[0]
    return time_s, np.array(torques)


def find_sim_torque(sim_results_dir, robot_name, motion_source, motion_name, joint_name):
    """Try to find sim-in-the-loop torque data from a benchmark output directory.

    Searches for state_motor.csv matching the motion name under any actuator
    subfolder.
    """
    sim_base = os.path.join(sim_results_dir, "sim", robot_name, motion_source)
    if not os.path.isdir(sim_base):
        return None, None

    # Search for motion_name under any actuator subfolder
    for actuator_dir in os.listdir(sim_base):
        motion_dir = os.path.join(sim_base, actuator_dir, motion_name)
        state_csv = os.path.join(motion_dir, "state_motor.csv")
        joint_list = os.path.join(motion_dir, "joint_list.txt")
        if os.path.isfile(state_csv):
            return load_sage_state_motor(state_csv, joint_name, joint_list)

    # Also check direct (no actuator subfolder)
    motion_dir = os.path.join(sim_base, motion_name)
    state_csv = os.path.join(motion_dir, "state_motor.csv")
    joint_list = os.path.join(motion_dir, "joint_list.txt")
    if os.path.isfile(state_csv):
        return load_sage_state_motor(state_csv, joint_name, joint_list)

    return None, None


# ---------------------------------------------------------------------------
# Actuator models (open-loop, no Isaac Lab)
# ---------------------------------------------------------------------------

def eval_pd(time_s, position, velocity, commanded, kp, kd):
    """Pure PD torque: kp * (cmd - pos) - kd * vel."""
    pos_err = commanded - position
    return kp * pos_err - kd * velocity


def eval_dcmotor(time_s, position, velocity, commanded, kp, kd,
                 saturation_effort=None, effort_limit=None):
    """DC motor model: PD torque clamped by saturation and effort limits."""
    torque = eval_pd(time_s, position, velocity, commanded, kp, kd)
    if saturation_effort is not None:
        torque = np.clip(torque, -saturation_effort, saturation_effort)
    if effort_limit is not None:
        torque = np.clip(torque, -effort_limit, effort_limit)
    return torque


def eval_lstm(time_s, position, velocity, commanded, network_file,
              stats_file=None):
    """Run LSTM/GRU model open-loop (no Isaac Lab).

    Loads the PyTorch model directly and runs step-by-step inference with
    hidden state carried across timesteps. Supports both LSTM and GRU.

    Args:
        time_s: Timestamps [s].
        position: Joint positions [rad].
        velocity: Joint velocities [rad/s].
        commanded: Commanded positions [rad].
        network_file: Path to .pt checkpoint (state_dict or TorchScript).
        stats_file: Optional path to sidecar stats JSON for normalization.
    """
    import json as _json

    import torch
    import torch.nn as nn

    if not os.path.isabs(network_file):
        network_file = os.path.join(REPO_ROOT, "input", "actuator_models", network_file)

    device = torch.device("cpu")

    # Load normalization stats if available
    norm_stats = None
    if stats_file is None:
        # Auto-detect sidecar stats JSON
        base, _ = os.path.splitext(network_file)
        for candidate in [base + "_stats.json",
                          base.removesuffix("_scripted") + "_stats.json"]:
            if os.path.isfile(candidate):
                stats_file = candidate
                break

    if stats_file and os.path.isfile(stats_file):
        with open(stats_file) as f:
            raw = _json.load(f)
        norm_raw = raw.get("normalization", raw)
        # Strip joint prefix to get canonical keys
        norm_stats = {}
        for key, val in norm_raw.items():
            for suffix in ["position_error", "position", "velocity", "torque"]:
                if key.endswith(suffix):
                    norm_stats[suffix] = val
                    break
        print(f"  LSTM/GRU stats: {os.path.basename(stats_file)}")

    # Try loading as TorchScript first, fall back to state_dict
    try:
        model = torch.jit.load(network_file, map_location=device)
        model.eval()
        # Detect architecture from TorchScript
        is_lstm = hasattr(model, "lstm") and isinstance(
            getattr(model, "lstm", None), (nn.LSTM, nn.GRU)
        )
        # Get hidden dim/layers from model parameters
        for name, param in model.named_parameters():
            if "weight_ih_l0" in name:
                hidden_dim = param.shape[0] // (4 if "lstm" in name.lower() else 3)
                break
        else:
            hidden_dim = 256
        num_layers = max(
            int(name.split("_l")[1].split(".")[0])
            for name, _ in model.named_parameters()
            if "_l" in name and ("lstm" in name or "gru" in name)
        ) + 1
    except Exception:
        # Load as state_dict and reconstruct
        state_dict = torch.load(network_file, map_location=device, weights_only=True)
        has_lstm = any(k.startswith("lstm.") for k in state_dict)
        has_gru = any(k.startswith("gru.") for k in state_dict)
        rnn_prefix = "lstm" if has_lstm else "gru"
        gate_factor = 4 if has_lstm else 3

        weight_key = f"{rnn_prefix}.weight_ih_l0"
        input_size = state_dict[weight_key].shape[1]
        hidden_dim = state_dict[weight_key].shape[0] // gate_factor
        num_layers = max(
            int(k.split("_l")[1].split(".")[0])
            for k in state_dict if k.startswith(f"{rnn_prefix}.") and "_l" in k
        ) + 1

        rnn_cls = nn.LSTM if has_lstm else nn.GRU

        class _RNNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = rnn_cls(
                    input_size=input_size, hidden_size=hidden_dim,
                    num_layers=num_layers, batch_first=True,
                )
                self.head = nn.Linear(hidden_dim, 1)

            def forward(self, x, h=None):
                out, h_new = self.rnn(x, h)
                return self.head(out).squeeze(-1), h_new

        model = _RNNModel()
        # Remap keys if needed
        mapped = {}
        for k, v in state_dict.items():
            mapped[k.replace(f"{rnn_prefix}.", "rnn.")] = v
        mapped = {k.replace("head.", "head."): v for k, v in mapped.items()}
        model.load_state_dict(mapped)
        model.eval()

    # Prepare inputs
    pos_err = commanded - position
    n = len(time_s)

    def _normalize(arr, key):
        if norm_stats and key in norm_stats:
            s = norm_stats[key]
            clipped = np.clip(arr, s["p1"], s["p99"])
            return (clipped - s["mean"]) / max(s["std"], 1e-8)
        return arr

    def _denormalize(arr, key):
        if norm_stats and key in norm_stats:
            s = norm_stats[key]
            return arr * s["std"] + s["mean"]
        return arr

    pos_norm = _normalize(position, "position")
    err_norm = _normalize(pos_err, "position_error")
    vel_norm = _normalize(velocity, "velocity")

    # Step-by-step inference with hidden state
    output = np.zeros(n)
    h = None

    with torch.no_grad():
        for i in range(n):
            x = torch.tensor(
                [[[pos_norm[i], err_norm[i], vel_norm[i]]]],
                dtype=torch.float32, device=device,
            )
            try:
                # TorchScript models use (x, (h, c)) signature
                if h is None:
                    h0 = torch.zeros(num_layers, 1, hidden_dim, device=device)
                    c0 = torch.zeros(num_layers, 1, hidden_dim, device=device)
                    pred, (h_new, c_new) = model(x, (h0, c0))
                else:
                    pred, (h_new, c_new) = model(x, h)
                h = (h_new, c_new)
            except (TypeError, RuntimeError):
                # Plain GRU model uses (x, h) signature
                if h is None:
                    pred, h = model(x)
                else:
                    pred, h = model(x, h)

            val = pred.item()
            output[i] = _denormalize(np.array([val]), "torque")[0]

    return output


def eval_fmu_cosim(time_s, position, commanded, torque_pred,
                   fmu_path, step_size=None):
    """Run CoSimulation FMU open-loop.

    FMU input mapping (Ansys Twin Builder convention):
      "position"    <- actual joint position
      "velocity"    <- commanded position (NOT angular velocity)
      "torque_pred" <- kp * position_error (P-only)
    """
    import fmpy
    from fmpy import extract
    from fmpy.fmi2 import FMU2Slave

    if not os.path.isabs(fmu_path):
        fmu_path = os.path.join(REPO_ROOT, "input", "actuator_models", fmu_path)

    model_desc = fmpy.read_model_description(fmu_path)

    if model_desc.coSimulation is None:
        from benchmark_fmu import run_fmu_fmpy
        return run_fmu_fmpy(fmu_path, time_s, position, velocity, torque_pred,
                            step_size=step_size)

    if os.path.isdir(fmu_path):
        unzip_dir = fmu_path
    else:
        unzip_dir = extract(fmu_path)

    vr_map = {v.name: v.valueReference for v in model_desc.modelVariables}

    inst = FMU2Slave(
        guid=model_desc.guid,
        unzipDirectory=unzip_dir,
        modelIdentifier=model_desc.coSimulation.modelIdentifier,
    )
    inst.instantiate()
    inst.setupExperiment(startTime=0.0)
    inst.enterInitializationMode()
    inst.exitInitializationMode()

    n = len(time_s)
    output = np.zeros(n)
    t = 0.0

    vr_pos = vr_map["position"]
    vr_vel = vr_map["velocity"]
    vr_tp = vr_map["torque_pred"]
    vr_tt = vr_map["torque_true"]

    for i in range(n):
        if step_size is not None:
            dt = step_size
        elif i == 0:
            dt = time_s[1] - time_s[0] if n > 1 else 0.002
        else:
            dt = time_s[i] - time_s[i - 1]

        inst.setReal([vr_pos], [float(position[i])])
        inst.setReal([vr_vel], [float(commanded[i])])  # "velocity" = commanded position
        inst.setReal([vr_tp], [float(torque_pred[i])])

        inst.doStep(currentCommunicationPoint=t, communicationStepSize=dt)

        output[i] = inst.getReal([vr_tt])[0]
        t += dt

    inst.terminate()
    inst.freeInstance()
    return output


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(time_s, real_torque, nosim_torque, pd_torque,
                    title, save_path, nosim_label="No-sim model",
                    sim_time=None, sim_torque=None):
    """Plot comparison with optional sim-in-the-loop overlay.

    Panels:
      1. Torque overlay (real, no-sim model, PD baseline, [sim-in-loop])
      2. Error vs real (no-sim, PD, [sim-in-loop])
      3. Scatter: no-sim vs real
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    has_sim = sim_time is not None and sim_torque is not None

    # --- Panel 1: torque overlay ---
    axes[0].plot(time_s, real_torque, label="Real", alpha=0.8, lw=0.6, color="tab:blue")
    axes[0].plot(time_s, nosim_torque, label=f"{nosim_label} (no-sim)",
                 alpha=0.8, lw=0.6, color="tab:orange")
    if has_sim:
        axes[0].plot(sim_time, sim_torque, label=f"{nosim_label} (sim-in-loop)",
                     alpha=0.7, lw=0.6, color="tab:red", linestyle="--")
    axes[0].plot(time_s, pd_torque, label="PD baseline",
                 alpha=0.4, lw=0.5, color="tab:green")
    axes[0].set_ylabel("Torque [N-m]")
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # --- Panel 2: error ---
    nosim_err = nosim_torque - real_torque
    pd_err = pd_torque - real_torque
    nosim_rmse = np.sqrt(np.mean(nosim_err ** 2))
    pd_rmse = np.sqrt(np.mean(pd_err ** 2))

    axes[1].plot(time_s, nosim_err,
                 label=f"No-sim error (RMSE={nosim_rmse:.4f})",
                 alpha=0.8, lw=0.5, color="tab:orange")
    axes[1].plot(time_s, pd_err,
                 label=f"PD error (RMSE={pd_rmse:.4f})",
                 alpha=0.4, lw=0.5, color="tab:green")

    if has_sim:
        # Interpolate sim torque to real time grid for error comparison
        from scipy.interpolate import interp1d
        sim_interp = interp1d(sim_time, sim_torque, kind="linear",
                              bounds_error=False, fill_value="extrapolate")
        sim_on_real = sim_interp(time_s)
        sim_err = sim_on_real - real_torque
        sim_rmse = np.sqrt(np.mean(sim_err ** 2))
        axes[1].plot(time_s, sim_err,
                     label=f"Sim error (RMSE={sim_rmse:.4f})",
                     alpha=0.7, lw=0.5, color="tab:red", linestyle="--")

    axes[1].set_ylabel("Error [N-m]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # --- Panel 3: scatter ---
    axes[2].scatter(real_torque, nosim_torque, s=0.5, alpha=0.3,
                    label=f"No-sim (RMSE={nosim_rmse:.4f})", color="tab:orange")
    if has_sim:
        axes[2].scatter(real_torque, sim_on_real, s=0.5, alpha=0.3,
                        label=f"Sim (RMSE={sim_rmse:.4f})", color="tab:red")

    lim = max(abs(real_torque).max(), abs(nosim_torque).max()) * 1.1
    if lim > 0:
        axes[2].plot([-lim, lim], [-lim, lim], "k--", lw=0.5, label="ideal")
        axes[2].set_xlim(-lim, lim)
        axes[2].set_ylim(-lim, lim)
    axes[2].set_xlabel("Real torque [N-m]")
    axes[2].set_ylabel("Predicted torque [N-m]")
    axes[2].set_aspect("equal")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def _collect_version_info():
    """Gather version strings for Isaac Lab, Newton, Isaac Sim, and fmpy."""
    info = {}
    try:
        import isaaclab
        info["isaaclab"] = getattr(isaaclab, "__version__", "unknown")
    except ImportError:
        pass
    try:
        import isaaclab_newton
        info["isaaclab_newton"] = getattr(isaaclab_newton, "__version__", "unknown")
    except ImportError:
        pass
    try:
        import isaacsim
        info["isaacsim"] = getattr(isaacsim, "__version__", "unknown")
    except ImportError:
        pass
    try:
        import fmpy
        info["fmpy"] = getattr(fmpy, "__version__", "unknown")
    except ImportError:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def _generate_pdf_report(pdf_path, summary_rows, plot_figures, run_cfg, args,
                         model_type, kp, kd, fmu_path, versions):
    """Generate a PDF report with summary table, config info, and all plots."""
    from datetime import datetime

    has_sim = any("sim_rmse" in r for r in summary_rows)

    with PdfPages(pdf_path) as pdf:
        # --- Page 1: Title + config info ---
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.95, "No-Sim Actuator Model Evaluation Report",
                 ha="center", va="top", fontsize=16, fontweight="bold")
        fig.text(0.5, 0.91, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 ha="center", va="top", fontsize=10, color="gray")

        # Build info text
        lines = []
        lines.append(f"Robot: {args.robot_name}")
        lines.append(f"Model type: {model_type}")
        lines.append(f"PD gains: kp={kp}, kd={kd}")
        if fmu_path:
            lines.append(f"FMU: {fmu_path}")
        lines.append(f"Data: {args.data_dir or 'from config'}")
        lines.append(f"Files evaluated: {len(summary_rows)}")
        lines.append("")

        # Actuator config
        act_cfg = run_cfg.get("actuator", {})
        if act_cfg:
            lines.append("Actuator config:")
            for k, v in act_cfg.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        # Simulation config
        sim_cfg = run_cfg.get("simulation", {})
        if sim_cfg:
            lines.append("Simulation config:")
            for k, v in sim_cfg.items():
                lines.append(f"  {k}: {v}")
            lines.append("")

        # Versions
        if versions:
            lines.append("Software versions:")
            for k, v in versions.items():
                lines.append(f"  {k}: {v}")

        fig.text(0.08, 0.84, "\n".join(lines), ha="left", va="top",
                 fontsize=9, fontfamily="monospace",
                 transform=fig.transFigure)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: Summary table ---
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.95, "RMSE Summary (N-m)",
                 ha="center", va="top", fontsize=14, fontweight="bold")

        col_labels = ["File", "No-sim", "PD"]
        if has_sim:
            col_labels.append("Sim")
        col_labels.append("Improve")

        table_data = []
        for row in summary_rows:
            r = [row["file"], f"{row['nosim_rmse']:.4f}", f"{row['pd_rmse']:.4f}"]
            if has_sim:
                r.append(f"{row.get('sim_rmse', 0):.4f}")
            r.append(f"{row['improvement_pct']:.1f}%")
            table_data.append(r)

        ax = fig.add_axes([0.05, 0.05, 0.9, 0.85])
        ax.axis("off")
        table = ax.table(cellText=table_data, colLabels=col_labels,
                         loc="upper center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)

        # Style header row
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")
        # Left-align file names
        for i in range(1, len(table_data) + 1):
            table[i, 0].set_text_props(ha="left")
        # Color improvement column
        imp_col = len(col_labels) - 1
        for i, row in enumerate(summary_rows, start=1):
            val = row["improvement_pct"]
            if val > 5:
                table[i, imp_col].set_facecolor("#C6EFCE")
            elif val < -5:
                table[i, imp_col].set_facecolor("#FFC7CE")

        pdf.savefig(fig)
        plt.close(fig)

        # --- Remaining pages: one plot per page ---
        for fig in plot_figures:
            pdf.savefig(fig, dpi=150)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="No-sim actuator model evaluation — real data fed directly through model",
    )
    parser.add_argument("--robot-name", type=str, default=None,
                        help="Robot name (loads run config from input/run_configs/)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to run config YAML (alternative to --robot-name)")
    parser.add_argument("--data-dir", "--data-path", type=str, default=None,
                        dest="data_dir",
                        help="Directory with parquet or motor CSV files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots")
    parser.add_argument("--model-type", type=str, default=None,
                        help="Override model type: implicit, dcmotor, lstm, lstm_perjoint, fmu")
    parser.add_argument("--joint-name", type=str, default="elbow",
                        help="Joint name for motor CSVs")
    parser.add_argument("--kp", type=float, default=None, help="Override PD stiffness")
    parser.add_argument("--kd", type=float, default=None, help="Override PD damping")
    parser.add_argument("--network-file", type=str, default=None,
                        help="Override LSTM/GRU network .pt file")
    parser.add_argument("--fmu-path", type=str, default=None, help="Override FMU path")
    parser.add_argument("--fmu-step-size", type=float, default=None,
                        help="FMU step size [s]")
    parser.add_argument("--sim-results", type=str, default=None,
                        help="Benchmark output dir to overlay sim-in-the-loop results "
                             "(e.g. output/sim2real_benchmark)")
    parser.add_argument("--motion-source", type=str, default=None,
                        help="Motion source subfolder in sim results (default: from run config)")
    args = parser.parse_args()

    # Load run config: --config (direct YAML path) or --robot-name (lookup)
    if args.config:
        import yaml
        config_path = args.config
        if not os.path.isabs(config_path):
            config_path = os.path.join(REPO_ROOT, config_path)
        with open(config_path) as f:
            run_cfg = yaml.safe_load(f) or {}
        # Infer robot name from config path if not provided
        if args.robot_name is None:
            args.robot_name = os.path.basename(os.path.dirname(config_path))
    else:
        if args.robot_name is None:
            args.robot_name = "teststand"
        run_cfg = load_run_cfg(args.robot_name)
    act_cfg = run_cfg.get("actuator", {})
    bench_cfg = run_cfg.get("benchmark", {})

    # Resolve model type
    model_type = args.model_type or act_cfg.get("model_type", "implicit")

    # Resolve PD gains
    yaml_file = act_cfg.get("yaml_file")
    if yaml_file:
        params = load_actuator_params(yaml_file)
    else:
        params = {}
    kp = args.kp if args.kp is not None else params.get("stiffness", 60.0)
    kd = args.kd if args.kd is not None else params.get("damping", 1.5)

    # Resolve data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = bench_cfg.get("motion_files", "")
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(REPO_ROOT, data_dir)

    # Resolve output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(REPO_ROOT, "output", "nosim_eval", args.robot_name, model_type)
    os.makedirs(output_dir, exist_ok=True)

    # Resolve FMU path
    fmu_path = args.fmu_path or act_cfg.get("fmu_path")
    fmu_step = args.fmu_step_size or act_cfg.get("fmu_step_size")

    # Resolve network file (LSTM/GRU)
    network_file = args.network_file or act_cfg.get("network_file")
    network_files = act_cfg.get("network_files", {})  # per-joint dict

    # DC motor params
    saturation_effort = params.get("saturation_effort")
    effort_limit = params.get("effort_limit")

    # Sim results overlay
    sim_results_dir = args.sim_results or bench_cfg.get("output_folder")
    if sim_results_dir and not os.path.isabs(sim_results_dir):
        sim_results_dir = os.path.join(REPO_ROOT, sim_results_dir)
    motion_source = args.motion_source or bench_cfg.get("motion_name", "custom")

    # Find data files
    data_files = find_data_files(data_dir)
    if not data_files:
        print(f"No parquet or motor CSV files found in {data_dir}")
        sys.exit(1)

    # Normalize aliases
    if model_type in ("gru", "gru_perjoint"):
        model_type = model_type.replace("gru", "lstm")

    model_label = {
        "implicit": "PD", "dcmotor": "DCMotor", "fmu": "FMU",
        "lstm": "LSTM/GRU", "lstm_perjoint": "LSTM/GRU (per-joint)",
    }.get(model_type, model_type.upper())

    print(f"No-sim evaluation: {model_type} (kp={kp}, kd={kd})")
    print(f"Data: {data_dir} ({len(data_files)} files)")
    if model_type == "fmu" and fmu_path:
        print(f"FMU: {fmu_path}")
    if model_type in ("lstm", "lstm_perjoint"):
        nf = network_file or network_files
        print(f"Network: {nf}")
    if sim_results_dir and os.path.isdir(str(sim_results_dir)):
        print(f"Sim overlay: {sim_results_dir}")
    print(f"Output: {output_dir}")
    print()

    summary_rows = []
    plot_figures = []

    for file_type, file_path in data_files:
        name = os.path.splitext(os.path.basename(file_path))[0]
        for suffix in ("_combined_clean", "_combined", "_clean", "_motor"):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break

        print(f"Processing: {name}")

        # Load real data
        if file_type == "parquet":
            time_s, position, velocity, real_torque, commanded = load_parquet(file_path)
        else:
            time_s, position, velocity, real_torque, commanded = load_motor_csv(
                file_path, args.joint_name
            )

        # PD baseline (always computed)
        pd_torque = eval_pd(time_s, position, velocity, commanded, kp, kd)

        # Model torque (no-sim)
        if model_type == "implicit":
            nosim_torque = pd_torque.copy()
        elif model_type == "dcmotor":
            nosim_torque = eval_dcmotor(
                time_s, position, velocity, commanded, kp, kd,
                saturation_effort=saturation_effort,
                effort_limit=effort_limit,
            )
        elif model_type in ("lstm", "lstm_perjoint"):
            # For lstm_perjoint, pick the network matching joint_name
            nf = network_file
            if model_type == "lstm_perjoint" and network_files:
                # Try exact match, then suffix match
                nf = network_files.get(args.joint_name)
                if nf is None:
                    for jt, path in network_files.items():
                        if args.joint_name.endswith(jt) or jt.endswith(args.joint_name):
                            nf = path
                            break
            if not nf:
                print(f"  ERROR: no network_file configured for '{args.joint_name}', skipping")
                continue
            print(f"  Running LSTM/GRU (no-sim, network={os.path.basename(nf)})...")
            nosim_torque = eval_lstm(
                time_s, position, velocity, commanded, nf,
            )
        elif model_type == "fmu":
            if not fmu_path:
                print("  ERROR: no fmu_path configured, skipping")
                continue
            # Use data timestep for FMU step size (not physics dt from config)
            if args.fmu_step_size is not None:
                data_step = args.fmu_step_size
            elif len(time_s) > 1:
                data_step = float(np.median(np.diff(time_s[:100])))
            else:
                data_step = 0.0005
            # FMU torque_pred = kp * position_error (P-only).
            fmu_torque_pred = kp * (commanded - position)
            print(f"  Running FMU (no-sim, kp={kp}, dt={data_step:.6f}s)...")
            nosim_torque = eval_fmu_cosim(
                time_s, position, commanded, fmu_torque_pred,
                fmu_path, step_size=data_step,
            )
        else:
            print(f"  WARNING: '{model_type}' not supported for no-sim eval, using PD")
            nosim_torque = pd_torque.copy()

        # Try to load sim-in-the-loop results
        sim_time, sim_torque = None, None
        if sim_results_dir and os.path.isdir(str(sim_results_dir)):
            sim_time, sim_torque = find_sim_torque(
                sim_results_dir, args.robot_name, motion_source, name, args.joint_name
            )
            if sim_torque is not None:
                print(f"  Found sim-in-loop data ({len(sim_torque)} samples)")

        # Metrics
        nosim_rmse = np.sqrt(np.mean((nosim_torque - real_torque) ** 2))
        pd_rmse = np.sqrt(np.mean((pd_torque - real_torque) ** 2))
        improve = (1 - nosim_rmse / pd_rmse) * 100 if pd_rmse > 0 else 0

        row = {
            "file": name,
            "nosim_rmse": nosim_rmse,
            "pd_rmse": pd_rmse,
            "improvement_pct": improve,
        }

        print(f"  No-sim  RMSE={nosim_rmse:.4f} N-m")
        if model_type != "implicit":
            print(f"  PD      RMSE={pd_rmse:.4f} N-m  (improvement: {improve:.1f}%)")

        if sim_torque is not None:
            from scipy.interpolate import interp1d
            sim_interp = interp1d(sim_time, sim_torque, kind="linear",
                                  bounds_error=False, fill_value="extrapolate")
            sim_on_real = sim_interp(time_s)
            sim_rmse = np.sqrt(np.mean((sim_on_real - real_torque) ** 2))
            row["sim_rmse"] = sim_rmse
            print(f"  Sim     RMSE={sim_rmse:.4f} N-m")

        summary_rows.append(row)

        # Plot
        plot_path = os.path.join(output_dir, f"{name}.png")
        fig = plot_comparison(
            time_s, real_torque, nosim_torque, pd_torque,
            f"{name} — {model_label}", plot_path,
            nosim_label=model_label,
            sim_time=sim_time, sim_torque=sim_torque,
        )
        plot_figures.append(fig)
        print(f"  Plot: {plot_path}")
        print()

    # Summary table
    if summary_rows:
        has_sim = any("sim_rmse" in r for r in summary_rows)
        print("=" * 90)
        hdr = f"{'File':<40s} {'No-sim':>10s} {'PD':>10s}"
        if has_sim:
            hdr += f" {'Sim':>10s}"
        hdr += f" {'Improve':>8s}"
        print(hdr)
        print("-" * 90)
        for row in summary_rows:
            line = (
                f"{row['file']:<40s} "
                f"{row['nosim_rmse']:>10.4f} "
                f"{row['pd_rmse']:>10.4f}"
            )
            if has_sim:
                line += f" {row.get('sim_rmse', 0):>10.4f}"
            line += f" {row['improvement_pct']:>7.1f}%"
            print(line)
        print("=" * 90)

        csv_path = os.path.join(output_dir, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            all_keys = list(summary_rows[0].keys())
            if has_sim and "sim_rmse" not in all_keys:
                all_keys.append("sim_rmse")
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary: {csv_path}")

        # PDF report
        versions = _collect_version_info()
        pdf_path = os.path.join(output_dir, "report.pdf")
        _generate_pdf_report(
            pdf_path, summary_rows, plot_figures, run_cfg, args,
            model_type, kp, kd, fmu_path, versions,
        )
        print(f"Report: {pdf_path}")

        # Close all plot figures
        for fig in plot_figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
