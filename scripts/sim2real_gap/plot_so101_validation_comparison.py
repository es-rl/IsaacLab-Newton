#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Plot SO-101 held-out validation traces.

Compares real SAGE data against two Newton replay outputs:

- upstream Feetech baseline
- balanced SysID fit

The script is intentionally offline: it reads benchmark CSV outputs and does
not launch Isaac Sim/Newton.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


DEFAULT_MOTIONS = [
    "backlash_detection",
    "coupled_joints",
    "custom_motion",
    "diagonal_sweep",
    "frequency_sweep",
    "friction_gravity",
    "hold_under_gravity",
    "square_pattern",
]


@dataclass(frozen=True)
class Run:
    label: str
    root: Path
    color: str
    linestyle: str = "-"
    linewidth: float = 1.4


def _parse_vec(value: str) -> list[float]:
    return [float(x) for x in ast.literal_eval(value)]


def _time_to_seconds(timestamp: float) -> float:
    return timestamp / 1e6 if timestamp > 1000 else timestamp


def _read_joint_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _read_state(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: list[float] = []
    positions: list[list[float]] = []
    torques: list[list[float]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("type") and row["type"] != "STATE_MOTOR":
                continue
            times.append(_time_to_seconds(float(row["timestamp"])))
            positions.append(_parse_vec(row["positions"]))
            torques.append(_parse_vec(row["torques"]))
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(torques, dtype=np.float64),
    )


def _motion_dirs(root: Path, motion: str) -> tuple[Path, Path]:
    real_dir = root / "real" / "so101" / "custom" / motion
    sim_dir = root / "sim" / "so101" / "custom" / "so101_implicit" / motion
    return real_dir, sim_dir


def _common_grid(time_arrays: list[np.ndarray], sample_dt: float) -> np.ndarray:
    t0 = max(float(t[0]) for t in time_arrays)
    t1 = min(float(t[-1]) for t in time_arrays)
    if t1 <= t0:
        raise ValueError(f"No overlapping timestamp range: start={t0}, end={t1}")
    return np.arange(t0, t1 + 1e-12, sample_dt, dtype=np.float64)


def _interp(signal_t: np.ndarray, signal_y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    out = np.zeros((len(grid), signal_y.shape[1]), dtype=np.float64)
    for idx in range(signal_y.shape[1]):
        out[:, idx] = np.interp(grid, signal_t, signal_y[:, idx])
    return out


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a - b))))


def _axis_limits(series: list[np.ndarray], pad_frac: float = 0.08) -> tuple[float, float]:
    finite = [s[np.isfinite(s)] for s in series if s.size]
    if not finite:
        return -1.0, 1.0
    flat = np.concatenate(finite)
    lo = float(np.percentile(flat, 1.0))
    hi = float(np.percentile(flat, 99.0))
    span = max(hi - lo, 1e-4)
    return lo - pad_frac * span, hi + pad_frac * span


def _motion_data(runs: list[Run], motion: str, plot_dt: float, rmse_dt: float):
    real_dir, _ = _motion_dirs(runs[0].root, motion)
    joints = _read_joint_names(real_dir / "joint_list.txt")
    real_t, real_pos, real_tau = _read_state(real_dir / "state_motor.csv")

    sim = {}
    for run in runs:
        _, sim_dir = _motion_dirs(run.root, motion)
        sim_t, sim_pos, sim_tau = _read_state(sim_dir / "state_motor.csv")
        sim[run.label] = (run, sim_t, sim_pos, sim_tau)

    plot_grid = _common_grid([real_t] + [value[1] for value in sim.values()], plot_dt)
    rmse_grid = _common_grid([real_t] + [value[1] for value in sim.values()], rmse_dt)

    real_plot = (_interp(real_t, real_pos, plot_grid), _interp(real_t, real_tau, plot_grid))
    real_rmse = (_interp(real_t, real_pos, rmse_grid), _interp(real_t, real_tau, rmse_grid))

    sim_plot = {}
    sim_rmse = {}
    metrics = {}
    for label, (run, sim_t, sim_pos, sim_tau) in sim.items():
        sim_plot[label] = (
            run,
            _interp(sim_t, sim_pos, plot_grid),
            _interp(sim_t, sim_tau, plot_grid),
        )
        sim_rmse_pos = _interp(sim_t, sim_pos, rmse_grid)
        sim_rmse_tau = _interp(sim_t, sim_tau, rmse_grid)
        sim_rmse[label] = (sim_rmse_pos, sim_rmse_tau)
        metrics[label] = {
            "pos_rmse_rad": _rmse(sim_rmse_pos, real_rmse[0]),
            "torque_rmse_nm": _rmse(sim_rmse_tau, real_rmse[1]),
        }

    return joints, plot_grid - plot_grid[0], real_plot, sim_plot, metrics


def _plot_motion(
    pdf: PdfPages,
    out_dir: Path,
    runs: list[Run],
    motion: str,
    plot_dt: float,
    rmse_dt: float,
) -> list[dict[str, object]]:
    joints, t, real, sims, metrics = _motion_data(runs, motion, plot_dt, rmse_dt)

    baseline = runs[0].label
    best = runs[1].label
    pos_reduction = (metrics[baseline]["pos_rmse_rad"] - metrics[best]["pos_rmse_rad"]) / metrics[baseline][
        "pos_rmse_rad"
    ] * 100.0
    torque_reduction = (
        (metrics[baseline]["torque_rmse_nm"] - metrics[best]["torque_rmse_nm"])
        / metrics[baseline]["torque_rmse_nm"]
        * 100.0
    )

    fig, axes = plt.subplots(len(joints), 2, figsize=(15, 13), sharex=True)
    fig.suptitle(
        f"SO-101 held-out validation - {motion}\n"
        f"balanced fit vs upstream baseline: position RMSE {pos_reduction:.1f}% lower, "
        f"torque RMSE {torque_reduction:.1f}% lower",
        fontsize=12,
        y=0.995,
    )

    real_pos, real_tau = real
    for joint_idx, joint in enumerate(joints):
        ax_q = axes[joint_idx, 0]
        ax_tau = axes[joint_idx, 1]
        ax_q.plot(t, real_pos[:, joint_idx], color="black", lw=1.8, label="real" if joint_idx == 0 else None)
        ax_tau.plot(t, real_tau[:, joint_idx], color="black", lw=1.8, label="real" if joint_idx == 0 else None)

        q_series = [real_pos[:, joint_idx]]
        tau_series = [real_tau[:, joint_idx]]
        for label, (run, sim_pos, sim_tau) in sims.items():
            ax_q.plot(
                t,
                sim_pos[:, joint_idx],
                color=run.color,
                linestyle=run.linestyle,
                lw=run.linewidth,
                label=label if joint_idx == 0 else None,
            )
            ax_tau.plot(t, sim_tau[:, joint_idx], color=run.color, linestyle=run.linestyle, lw=run.linewidth)
            q_series.append(sim_pos[:, joint_idx])
            tau_series.append(sim_tau[:, joint_idx])

        ax_q.set_ylabel(f"{joint}\nq [rad]", fontsize=8)
        ax_tau.set_ylabel("torque [Nm]", fontsize=8)
        ax_q.set_ylim(*_axis_limits(q_series))
        ax_tau.set_ylim(*_axis_limits(tau_series))
        ax_q.grid(alpha=0.25)
        ax_tau.grid(alpha=0.25)
        if joint_idx == 0:
            ax_q.set_title("position")
            ax_tau.set_title("torque")
            ax_q.legend(loc="upper right", fontsize=8, framealpha=0.9)

    axes[-1, 0].set_xlabel("time [s]")
    axes[-1, 1].set_xlabel("time [s]")
    plt.tight_layout(rect=[0, 0, 1, 0.982])

    pdf.savefig(fig, bbox_inches="tight")
    png_path = out_dir / f"so101_val8__{motion}.png"
    fig.savefig(png_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for label in [run.label for run in runs]:
        rows.append(
            {
                "motion": motion,
                "model": label,
                "pos_rmse_rad": metrics[label]["pos_rmse_rad"],
                "torque_rmse_nm": metrics[label]["torque_rmse_nm"],
            }
        )
    print(f"wrote {png_path}")
    return rows


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-root",
        default="output/sim2real_benchmark_so101_upstream_val8_20260522",
    )
    parser.add_argument(
        "--fit-root",
        default="output/sim2real_benchmark_so101_balanced_42train_val8_20260522",
    )
    parser.add_argument("--out-dir", default="output/so101_plots_val8_20260522")
    parser.add_argument("--motions", nargs="*", default=DEFAULT_MOTIONS)
    parser.add_argument("--plot-dt", type=float, default=0.02, help="Plot interpolation timestep in seconds.")
    parser.add_argument("--rmse-dt", type=float, default=0.005, help="RMSE interpolation timestep in seconds.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        Run("upstream baseline", Path(args.baseline_root).expanduser(), "0.45", "--", 1.2),
        Run("balanced SysID", Path(args.fit_root).expanduser(), "tab:green", "-", 1.7),
    ]

    all_rows: list[dict[str, object]] = []
    pdf_path = out_dir / "so101_val8_real_vs_upstream_vs_balanced.pdf"
    with PdfPages(pdf_path) as pdf:
        for motion in args.motions:
            all_rows.extend(_plot_motion(pdf, out_dir, runs, motion, args.plot_dt, args.rmse_dt))

    metrics_path = out_dir / "so101_val8_plot_metrics.csv"
    _write_metrics(metrics_path, all_rows)
    print(f"wrote {pdf_path}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
