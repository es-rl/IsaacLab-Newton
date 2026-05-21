#!/usr/bin/env python3
"""Generate H1 arm plots comparing real, PD, lag20 SysID, and v28 PD+GRU.

The benchmark/analysis numbers remain the source of truth. This script uses
the SAGE ``metrics_summary.xlsx`` files for plot title RMSE labels, then draws
time traces from the benchmark CSV outputs for inspection.
"""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402


DEFAULT_REAL_ROOT = Path(
    "output/h1_arm_gru_v28_test10_20260521/real/"
    "h1_arm_v28_lag20_gripperpos_s033/multijoint_gripper_test10"
)
DEFAULT_PD_ROOT = Path(
    "output/h1_arm_repro_test10_20260521/sim/"
    "h1_arm_pd_baseline_yawclamped_gripperpos/multijoint_gripper_test10/h1_arm_implicit"
)
DEFAULT_SYSID_ROOT = Path(
    "output/h1_arm_repro_test10_20260521/sim/"
    "h1_arm_lag20_sysid_gripperpos/multijoint_gripper_test10/"
    "h1_arm_post_velsync_sysid_gen10_lag20"
)
DEFAULT_GRU_ROOT = Path(
    "output/h1_arm_gru_v28_test10_20260521/sim/"
    "h1_arm_v28_lag20_gripperpos_s033/multijoint_gripper_test10/"
    "h1_arm_post_velsync_sysid_gen10_lag20"
)
DEFAULT_PD_METRICS = Path("output/h1_arm_repro_analysis_pd_test10_20260521/metrics_summary.xlsx")
DEFAULT_SYSID_METRICS = Path("output/h1_arm_repro_analysis_lag20_sysid_test10_20260521/metrics_summary.xlsx")
DEFAULT_GRU_METRICS = Path("output/h1_arm_gru_v28_analysis_test10_20260521/metrics_summary.xlsx")
DEFAULT_OUTPUT_DIR = Path("output/h1_plots_current/arm_real_pd_lag20sysid_v28gru_labeled")

JOINT_LABELS = {
    "right_elbow": "Right elbow",
    "right_shoulder_pitch": "Right shoulder pitch",
    "right_shoulder_roll": "Right shoulder roll",
    "right_shoulder_yaw": "Right shoulder yaw",
}

MODEL_COLORS = {
    "real": "black",
    "pd": "#7f7f7f",
    "sysid": "#1f77b4",
    "gru": "#d62728",
}

MODEL_LABELS = {
    "real": "Real",
    "pd": "PD baseline",
    "sysid": "lag20 SysID",
    "gru": "v28 PD+GRU",
}

METRIC_SHEETS = {
    "position": "positions_rmse_compare",
    "velocity": "velocities_rmse_compare",
    "torque": "torques_rmse_compare",
}


def _time_to_seconds(values: np.ndarray) -> np.ndarray:
    values = values.astype(float)
    if values.size and np.nanmax(values) > 1000.0:
        values = values / 1e6
    return values - values[0]


def _read_joint_list(motion_dir: Path) -> list[str]:
    return [line.strip() for line in (motion_dir / "joint_list.txt").read_text().splitlines() if line.strip()]


def _read_state(motion_dir: Path) -> dict[str, np.ndarray | list[str]]:
    times = []
    positions = []
    velocities = []
    torques = []
    with (motion_dir / "state_motor.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["timestamp"]))
            positions.append(ast.literal_eval(row["positions"]))
            velocities.append(ast.literal_eval(row["velocities"]))
            torques.append(ast.literal_eval(row["torques"]))
    return {
        "time": _time_to_seconds(np.asarray(times, dtype=float)),
        "positions": np.asarray(positions, dtype=float),
        "velocities": np.asarray(velocities, dtype=float),
        "torques": np.asarray(torques, dtype=float),
        "joints": _read_joint_list(motion_dir),
    }


def _series(data: dict[str, np.ndarray | list[str]], joint: str, key: str) -> np.ndarray:
    joints = data["joints"]
    assert isinstance(joints, list)
    idx = joints.index(joint)
    values = data[key]
    assert isinstance(values, np.ndarray)
    return values[:, idx]


def _load_metrics(path: Path) -> dict[str, pd.DataFrame]:
    return {metric: pd.read_excel(path, sheet_name=sheet, index_col=0) for metric, sheet in METRIC_SHEETS.items()}


def _metric_mean(metrics: dict[str, dict[str, pd.DataFrame]], model: str, metric: str, motion: str | None = None) -> float:
    df = metrics[model][metric]
    if motion is None:
        return float(df.to_numpy(dtype=float).mean())
    return float(df[motion].to_numpy(dtype=float).mean())


def _reduction(baseline: float, candidate: float) -> float:
    return 100.0 * (baseline - candidate) / baseline if baseline else 0.0


def _motion_summary(metrics: dict[str, dict[str, pd.DataFrame]], motion: str | None = None) -> dict[str, float]:
    out = {}
    for metric in METRIC_SHEETS:
        for model in ("pd", "sysid", "gru"):
            out[f"{model}_{metric}_rmse"] = _metric_mean(metrics, model, metric, motion)
        out[f"sysid_{metric}_reduction_vs_pd_percent"] = _reduction(
            out[f"pd_{metric}_rmse"], out[f"sysid_{metric}_rmse"]
        )
        out[f"gru_{metric}_reduction_vs_pd_percent"] = _reduction(
            out[f"pd_{metric}_rmse"], out[f"gru_{metric}_rmse"]
        )
        out[f"gru_{metric}_improvement_vs_sysid_percent"] = _reduction(
            out[f"sysid_{metric}_rmse"], out[f"gru_{metric}_rmse"]
        )
    return out


def _plot_motion(
    output_dir: Path,
    pdf: PdfPages,
    motion: str,
    data: dict[str, dict[str, np.ndarray | list[str]]],
    metrics: dict[str, dict[str, pd.DataFrame]],
):
    joints = data["real"]["joints"]
    assert isinstance(joints, list)
    summary = _motion_summary(metrics, motion)

    fig, axes = plt.subplots(len(joints), 2, figsize=(16, 11), sharex="col")
    if len(joints) == 1:
        axes = np.asarray([axes])

    title = (
        f"H1 arm v28 PD+GRU comparison - {motion}\n"
        f"Position: PD {summary['pd_position_rmse']:.4f}, "
        f"lag20 {summary['sysid_position_rmse']:.4f}, GRU {summary['gru_position_rmse']:.4f} "
        f"| GRU {summary['gru_position_reduction_vs_pd_percent']:.1f}% vs PD, "
        f"{summary['gru_position_improvement_vs_sysid_percent']:.1f}% vs lag20\n"
        f"Torque: PD {summary['pd_torque_rmse']:.4f}, "
        f"lag20 {summary['sysid_torque_rmse']:.4f}, GRU {summary['gru_torque_rmse']:.4f} "
        f"| GRU {summary['gru_torque_reduction_vs_pd_percent']:.1f}% vs PD, "
        f"{summary['gru_torque_improvement_vs_sysid_percent']:.1f}% vs lag20"
    )
    fig.suptitle(title, fontsize=10, fontweight="bold", y=0.985)

    for row, joint in enumerate(joints):
        label = JOINT_LABELS.get(joint, joint)
        for model in ("real", "pd", "sysid", "gru"):
            lw = 1.2 if model == "real" else 0.9
            alpha = 0.9 if model == "real" else 0.8
            axes[row, 0].plot(
                data[model]["time"],
                _series(data[model], joint, "positions"),
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
                linewidth=lw,
                alpha=alpha,
            )
            axes[row, 1].plot(
                data[model]["time"],
                _series(data[model], joint, "torques"),
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
                linewidth=lw,
                alpha=alpha,
            )
        axes[row, 0].set_ylabel(f"{label}\nrad")
        axes[row, 1].set_ylabel(f"{label}\nNm")
        axes[row, 0].grid(True, alpha=0.25)
        axes[row, 1].grid(True, alpha=0.25)
        if row == 0:
            axes[row, 0].set_title("Position")
            axes[row, 1].set_title("Torque")
            axes[row, 1].legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])

    png_path = output_dir / f"h1_arm_real_pd_lag20sysid_v28gru_{motion}.png"
    fig.savefig(png_path, dpi=160)
    pdf.savefig(fig)
    plt.close(fig)


def _write_metrics_csv(output_dir: Path, motions: list[str], metrics: dict[str, dict[str, pd.DataFrame]]) -> Path:
    rows = []
    for motion in motions:
        row = {"motion": motion}
        row.update(_motion_summary(metrics, motion))
        rows.append(row)
    overall = {"motion": "OVERALL"}
    overall.update(_motion_summary(metrics, None))
    rows.append(overall)

    csv_path = output_dir / "h1_arm_pd_lag20sysid_v28gru_labeled_plot_metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _plot_summary(output_dir: Path, metrics: dict[str, dict[str, pd.DataFrame]]) -> Path:
    values = {
        metric: [_metric_mean(metrics, model, metric) for model in ("pd", "sysid", "gru")]
        for metric in METRIC_SHEETS
    }
    units = {"position": "rad", "velocity": "rad/s", "torque": "Nm"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    colors = [MODEL_COLORS["pd"], MODEL_COLORS["sysid"], MODEL_COLORS["gru"]]
    labels = [MODEL_LABELS["pd"], MODEL_LABELS["sysid"], MODEL_LABELS["gru"]]
    for ax, metric in zip(axes, METRIC_SHEETS):
        bars = ax.bar(labels, values[metric], color=colors)
        for bar, value in zip(bars, values[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
        gru_vs_pd = _reduction(values[metric][0], values[metric][2])
        gru_vs_sysid = _reduction(values[metric][1], values[metric][2])
        ax.set_title(f"{metric.title()} RMSE\nGRU {gru_vs_pd:.1f}% vs PD, {gru_vs_sysid:.1f}% vs lag20")
        ax.set_ylabel(units[metric])
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("H1 arm benchmark summary: real vs PD vs lag20 SysID vs v28 PD+GRU", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    png_path = output_dir / "h1_arm_pd_lag20sysid_v28gru_labeled_rmse_summary.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--pd-root", type=Path, default=DEFAULT_PD_ROOT)
    parser.add_argument("--sysid-root", type=Path, default=DEFAULT_SYSID_ROOT)
    parser.add_argument("--gru-root", type=Path, default=DEFAULT_GRU_ROOT)
    parser.add_argument("--pd-metrics", type=Path, default=DEFAULT_PD_METRICS)
    parser.add_argument("--sysid-metrics", type=Path, default=DEFAULT_SYSID_METRICS)
    parser.add_argument("--gru-metrics", type=Path, default=DEFAULT_GRU_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    roots = {"real": args.real_root, "pd": args.pd_root, "sysid": args.sysid_root, "gru": args.gru_root}
    for name, root in roots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"{name} root does not exist: {root}")

    motions = sorted(p.name for p in args.real_root.iterdir() if (p / "state_motor.csv").is_file())
    if not motions:
        raise RuntimeError(f"No motions found under {args.real_root}")
    for model, root in roots.items():
        missing = [motion for motion in motions if not (root / motion / "state_motor.csv").is_file()]
        if missing:
            raise FileNotFoundError(f"{model} is missing state_motor.csv for: {missing}")

    metrics = {
        "pd": _load_metrics(args.pd_metrics),
        "sysid": _load_metrics(args.sysid_metrics),
        "gru": _load_metrics(args.gru_metrics),
    }

    pdf_path = args.output_dir / "h1_arm_real_pd_lag20sysid_v28gru_labeled_all_motions.pdf"
    with PdfPages(pdf_path) as pdf:
        for motion in motions:
            motion_data = {model: _read_state(root / motion) for model, root in roots.items()}
            _plot_motion(args.output_dir, pdf, motion, motion_data, metrics)

    csv_path = _write_metrics_csv(args.output_dir, motions, metrics)
    summary_path = _plot_summary(args.output_dir, metrics)

    print(f"PDF:     {pdf_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"CSV:     {csv_path.resolve()}")
    print(f"PNGs:    {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
