"""CMA-ES system identification for robot actuator parameters.

Runs PACE-style optimization: N parallel sim environments, each with different
candidate motor parameters, replaying real robot data and minimizing position
MSE vs real measured response.

Supported robots: h1 (default), ur10e

Input modes:
  - Motor CSVs:   raw *_motor.csv directory (auto-converted to SAGE format)
  - SAGE CSVs:    directory with control.csv + state_motor.csv
  - Single-joint: parquet chirp files (e.g. left_elbow sysid experiments)

Usage (config-driven — reads real_data_dir from h1.yaml sysid section):
    python scripts/sysid/run_sysid.py --robot-name h1 --headless

Usage (H1 motor CSVs — auto-converts to SAGE format):
    python scripts/sysid/run_sysid.py \
        --real-data-dir input/sysid_data/h1/arm_march2026/elbow \
        --output-dir output/sysid/h1/elbow \
        --headless

Usage (H1 SAGE CSVs):
    python scripts/sysid/run_sysid.py \
        --real-data-dir output/sim2real_benchmark/real/h1/arm_march2026/elbow \
        --output-dir output/sysid/h1/elbow \
        --headless

Usage (UR10e all-joints CSV):
    python scripts/sysid/run_sysid.py \
        --robot-name ur10e \
        --real-data-dir input/sysid_data/ur10e_csv \
        --physics-freq 500 --control-freq 500 \
        --output-dir output/sysid/ur10e \
        --headless
"""

import argparse
import ast
import csv
import glob
import os
import sys
import time

import yaml

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="CMA-ES system identification for robot actuators")

# --- Robot selection ---
parser.add_argument(
    "--robot-name",
    type=str,
    default="h1",
    choices=["h1", "h1_right_arm", "ur10e"],
    help="Robot name (default: h1).",
)

# --- Input data (choose one) ---
parser.add_argument(
    "--data-dir",
    type=str,
    default=None,
    help="Directory with parquet chirp files (single-joint mode). All *.parquet files loaded.",
)
parser.add_argument(
    "--data-files",
    type=str,
    default=None,
    help="Comma-separated parquet paths (single-joint mode, alternative to --data-dir).",
)
parser.add_argument(
    "--real-data-dir",
    type=str,
    default=None,
    help="Directory with control.csv + state_motor.csv + joint_list.txt (all-joints mode).",
)
parser.add_argument(
    "--joint-name",
    type=str,
    default="left_elbow",
    help="Sim joint name for single-joint mode (default: left_elbow).",
)

# --- Optimization ---
parser.add_argument("--num-envs", type=int, default=64, help="Parallel environments = CMA-ES population size")
parser.add_argument("--max-iter", type=int, default=200, help="Maximum CMA-ES generations")
parser.add_argument("--sigma", type=float, default=None, help="CMA-ES initial step size")
parser.add_argument("--epsilon", type=float, default=None, help="CMA-ES convergence threshold")
parser.add_argument(
    "--joints",
    type=str,
    default=None,
    help="Joint types to optimize: 'full' (all from bounds YAML) or comma-separated "
    "list (e.g., 'elbow' or 'elbow,shoulder_pitch'). Default: from run config or full.",
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Path to parameter bounds YAML (default: auto from --robot-name)",
)

# --- Simulation ---
parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: auto from --robot-name)")
parser.add_argument("--physics-freq", type=int, default=200, help="Physics frequency (Hz)")
parser.add_argument("--control-freq", type=int, default=200, help="Control replay frequency (Hz)")
parser.add_argument("--buffer-time", type=float, default=2.0, help="Settle time before replay (s)")
parser.add_argument("--max-trajectory-len", type=int, default=None, help="Max steps per trajectory (truncate)")
parser.add_argument("--log-interval", type=int, default=1, help="Log every N generations")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False

# Load per-robot run config; CLI args override config values
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "input"))
from run_configs import load_run_cfg

_run_cfg = load_run_cfg(args.robot_name)
_sim_cfg = _run_cfg.get("simulation", {})
_sysid_cfg = _run_cfg.get("sysid", {})

# Integer-defaulted args: override only if CLI left them at argparse default
for _attr, _default in [
    ("physics_freq", 200),
    ("control_freq", 200),
    ("num_envs", 64),
    ("max_iter", 200),
]:
    if getattr(args, _attr) == _default:
        if _attr in _sim_cfg:
            setattr(args, _attr, _sim_cfg[_attr])
        elif _attr in _sysid_cfg:
            setattr(args, _attr, _sysid_cfg[_attr])

# Float-defaulted args
if args.buffer_time == 2.0 and "buffer_time" in _sysid_cfg:
    args.buffer_time = _sysid_cfg["buffer_time"]

# None-defaulted args
for _attr in ("sigma", "epsilon", "real_data_dir", "joints"):
    if getattr(args, _attr) is None and _attr in _sysid_cfg:
        setattr(args, _attr, _sysid_cfg[_attr])

# Normalize --joints: None/"full" → None (all), comma-string → list, YAML list → list
if isinstance(args.joints, str):
    if args.joints.lower() == "full":
        args.joints = None
    else:
        args.joints = [j.strip() for j in args.joints.split(",")]
elif isinstance(args.joints, list):
    pass  # YAML already gave us a list
# else: None → use all joint types from bounds YAML

# Resolve defaults that depend on --robot-name
_RUN_CONFIGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "input", "run_configs")
)
_DEFAULT_CONFIGS = {
    "h1": os.path.join(_RUN_CONFIGS_DIR, "h1", "h1_arms_sysid_bounds.yaml"),
    "h1_right_arm": os.path.join(_RUN_CONFIGS_DIR, "h1", "h1_right_arm_sysid_bounds.yaml"),
    "ur10e": os.path.join(_RUN_CONFIGS_DIR, "ur10e", "ur10e_sysid_bounds.yaml"),
}
if args.config is None:
    # CLI > run config bounds_yaml > default
    _bounds_yaml = _sysid_cfg.get("bounds_yaml")
    if _bounds_yaml:
        args.config = os.path.join(_RUN_CONFIGS_DIR, _bounds_yaml)
    else:
        args.config = _DEFAULT_CONFIGS[args.robot_name]
if args.output_dir is None:
    # output/sysid/{robot_name}/{sysid.output_dir or joint names}
    _sysid_output_name = _sysid_cfg.get("output_dir")
    if not _sysid_output_name:
        _joints = args.joints
        if isinstance(_joints, list):
            _sysid_output_name = "_".join(_joints)
        else:
            _sysid_output_name = args.robot_name
    args.output_dir = os.path.join("output", "sysid", args.robot_name, _sysid_output_name)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Sim-dependent imports (after AppLauncher) ---

import numpy as np
import pandas as pd
import torch
import warp as wp
from collections import deque
from scipy.interpolate import interp1d
from tqdm import tqdm

import isaaclab.sim as sim_utils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "input"))
from actuator_models import load_actuator_params, load_implicit_actuator_cfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import H1_MINIMAL_CFG
from spawn_utils import spawn_from_usd_with_fixed_base

from optimizer import CMAESOptimizer

# Local H1 USD (avoids dependency on cloud-hosted Omniverse Nucleus server)
_H1_LOCAL_USD = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "input", "robot_models", "h1_minimal", "h1_minimal.usda")
)


def log_message(msg):
    print(f"[SYSID] {msg}")


# ---------------------------------------------------------------------------
# Scene config
# ---------------------------------------------------------------------------

@configclass
class SysidSceneCfg(InteractiveSceneCfg):
    """Scene with H1 robot for system identification."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = H1_MINIMAL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=H1_MINIMAL_CFG.spawn.replace(usd_path=_H1_LOCAL_USD, func=spawn_from_usd_with_fixed_base),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.5),
            joint_pos={
                ".*_hip_yaw": 0.0,
                ".*_hip_roll": 0.0,
                ".*_hip_pitch": -0.28,
                ".*_knee": 0.79,
                ".*_ankle": -0.52,
                "torso": 0.0,
                ".*_shoulder_pitch": 0.0,
                ".*_shoulder_roll": 0.0,
                ".*_shoulder_yaw": 0.0,
                ".*_elbow": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
                effort_limit_sim=300,
                stiffness={
                    ".*_hip_yaw": 50.0, ".*_hip_roll": 50.0,
                    ".*_hip_pitch": 100.0, ".*_knee": 100.0, "torso": 100.0,
                },
                damping={
                    ".*_hip_yaw": 5.0, ".*_hip_roll": 5.0,
                    ".*_hip_pitch": 5.0, ".*_knee": 5.0, "torso": 5.0,
                },
            ),
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle"],
                effort_limit_sim=100,
                stiffness={".*_ankle": 20.0},
                damping={".*_ankle": 4.0},
            ),
            "arms": load_implicit_actuator_cfg(
                "h1/h1_arm_implicit.yaml",
                [".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
            ),
        },
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


_UR10_USD_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "input", "robot_models", "ur10", "ur10", "ur10.usd"
))


@configclass
class Ur10eSysidSceneCfg(InteractiveSceneCfg):
    """Scene with UR10e robot for system identification."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_UR10_USD_PATH,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.712,
                "elbow_joint": 1.712,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
            },
        ),
        actuators={
            "arm": load_implicit_actuator_cfg(
                "ur10e/ur10e_implicit.yaml",
                [".*"],
            ),
        },
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# ---------------------------------------------------------------------------
# Robot-specific joint name lists
# ---------------------------------------------------------------------------

H1_ARM_JOINT_NAMES = [
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
]

H1_RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
]

UR10E_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

_ROBOT_CONFIGS = {
    "h1": {
        "scene_cfg_cls": SysidSceneCfg,
        "joint_names": H1_ARM_JOINT_NAMES,
        "actuator_yaml": "h1/h1_arm_implicit.yaml",
    },
    "h1_right_arm": {
        "scene_cfg_cls": SysidSceneCfg,
        "joint_names": H1_RIGHT_ARM_JOINT_NAMES,
        "actuator_yaml": "h1/h1_arm_implicit.yaml",
    },
    "ur10e": {
        "scene_cfg_cls": Ur10eSysidSceneCfg,
        "joint_names": UR10E_JOINT_NAMES,
        "actuator_yaml": "ur10e/ur10e_implicit.yaml",
    },
}


# ---------------------------------------------------------------------------
# Data loading: single-joint parquet chirp
# ---------------------------------------------------------------------------

def load_chirp_parquets(file_paths: list[str], target_dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Load and concatenate single-joint chirp parquet files.

    Each parquet has columns: time, position, commanded_position, ...
    Data is typically at 2000 Hz. Resampled to target_dt.

    Returns:
        commanded: (T,) array of commanded positions.
        measured:  (T,) array of actual measured positions.
    """
    all_cmd = []
    all_pos = []

    for path in sorted(file_paths):
        df = pd.read_parquet(path)
        log_message(f"  {os.path.basename(path)}: {len(df)} rows, "
                    f"{(df['time'].max() - df['time'].min()) / 1e9:.1f}s")

        times = df["time"].values / 1e9  # nanoseconds → seconds
        data_dt = float(np.median(np.diff(times)))

        cmd = df["commanded_position"].values
        pos = df["position"].values

        if abs(data_dt - target_dt) > 1e-6:
            duration = times[-1] - times[0]
            new_len = int(round(duration / target_dt))
            new_times = np.linspace(times[0], times[-1], new_len, endpoint=False)
            new_times = new_times[new_times <= times[-1]]
            f_cmd = interp1d(times, cmd, kind="linear")
            f_pos = interp1d(times, pos, kind="linear")
            cmd = f_cmd(new_times)
            pos = f_pos(new_times)

        all_cmd.append(cmd)
        all_pos.append(pos)

    commanded = np.concatenate(all_cmd)
    measured = np.concatenate(all_pos)
    log_message(f"Total: {len(commanded)} steps at {1/target_dt:.0f}Hz = {len(commanded) * target_dt:.1f}s")
    return commanded, measured


# ---------------------------------------------------------------------------
# Data loading: all-joints CSV
# ---------------------------------------------------------------------------

def load_real_data(data_dir: str, target_dt: float, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load all-joints data from CSVs (control.csv + state_motor.csv).

    Args:
        data_dir: Directory with control.csv, state_motor.csv, joint_list.txt.
        target_dt: Target timestep for resampling.
        joint_names: Joint names to extract (determines column order and count).

    Returns:
        commanded: (T, N) commanded positions for the specified joints.
        measured:  (T, N) actual positions for the specified joints.
    """
    num_joints = len(joint_names)

    joint_list_path = os.path.join(data_dir, "joint_list.txt")
    with open(joint_list_path) as f:
        all_joint_names = [line.strip() for line in f if line.strip()]

    joint_indices = []
    for name in joint_names:
        joint_indices.append(all_joint_names.index(name))

    # control.csv → commanded positions
    ctrl_times, ctrl_pos = [], []
    with open(os.path.join(data_dir, "control.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ctrl_times.append(float(row["timestamp"]))
            positions = ast.literal_eval(row["positions"])
            ctrl_pos.append([positions[i] for i in joint_indices])

    # state_motor.csv → actual positions
    state_times, state_pos = [], []
    with open(os.path.join(data_dir, "state_motor.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            state_times.append(float(row["timestamp"]))
            positions = ast.literal_eval(row["positions"])
            state_pos.append([positions[i] for i in joint_indices])

    ctrl_pos = np.array(ctrl_pos)
    state_pos = np.array(state_pos)

    # Timestamps are in microseconds
    ctrl_times = np.array(ctrl_times)
    real_dt = float(np.median(np.diff(ctrl_times))) / 1e6

    # Align lengths
    min_len = min(len(ctrl_pos), len(state_pos))
    ctrl_pos = ctrl_pos[:min_len]
    state_pos = state_pos[:min_len]

    # Resample if needed
    if abs(real_dt - target_dt) > 1e-6:
        T = ctrl_pos.shape[0]
        duration = T * real_dt
        new_len = int(round(duration / target_dt))
        old_times = np.linspace(0, duration, T, endpoint=False)
        new_times = np.linspace(0, duration, new_len, endpoint=False)
        new_times = new_times[new_times <= old_times[-1]]

        new_ctrl = np.zeros((len(new_times), num_joints))
        new_state = np.zeros((len(new_times), num_joints))
        for j in range(num_joints):
            new_ctrl[:, j] = interp1d(old_times, ctrl_pos[:, j], kind="linear")(new_times)
            new_state[:, j] = interp1d(old_times, state_pos[:, j], kind="linear")(new_times)
        ctrl_pos = new_ctrl
        state_pos = new_state
        log_message(f"Resampled {T} ({1/real_dt:.0f}Hz) → {len(new_times)} ({1/target_dt:.0f}Hz)")

    log_message(f"Loaded real data: {len(ctrl_pos)} steps, {num_joints} joints")
    return ctrl_pos, state_pos


# ---------------------------------------------------------------------------
# Auto-convert motor CSVs to SAGE format
# ---------------------------------------------------------------------------

def auto_convert_motor_csvs(data_dir: str, output_dir: str) -> str:
    """Detect and convert raw motor CSVs to SAGE format if needed.

    If data_dir already contains control.csv (SAGE format), returns data_dir
    unchanged. If it contains *_motor.csv files (directly or in subdirs),
    converts them to a single concatenated SAGE output under output_dir/converted_sage/.

    Returns:
        Path to use as real_data_dir (original or converted).
    """
    # Already SAGE format?
    if os.path.isfile(os.path.join(data_dir, "control.csv")):
        log_message(f"SAGE format detected in {data_dir}")
        return data_dir

    # Previously converted?
    converted_dir = os.path.join(output_dir, "converted_sage")
    if os.path.isfile(os.path.join(converted_dir, "control.csv")):
        log_message(f"Using previously converted SAGE data in {converted_dir}")
        return converted_dir

    from convert_h1_chirp_to_csv import (
        CSV_ORDER,
        find_motor_csvs,
        load_csv_with_commands,
        write_sage_output,
    )

    subdirs = find_motor_csvs(data_dir, recursive=True)
    if not subdirs:
        raise ValueError(
            f"No motor CSVs or SAGE files found in {data_dir}. "
            "Expected either control.csv + state_motor.csv or *_motor.csv files."
        )

    log_message(f"Auto-converting motor CSVs from {data_dir} ({len(subdirs)} dirs)")

    all_cmd_rows = []
    all_state_rows = []
    time_offset = 0.0
    total_files = 0

    for name, dir_path in subdirs:
        csvs = sorted(f for f in os.listdir(dir_path) if f.endswith("_motor.csv"))
        if not csvs:
            continue

        log_message(f"  {name}: {len(csvs)} files")

        for fname in csvs:
            path = os.path.join(dir_path, fname)
            time_s, commands, positions, velocities, torques = load_csv_with_commands(path)

            t0 = time_s[0]
            duration = time_s[-1] - t0

            for k in range(len(time_s)):
                t_abs = time_s[k] - t0 + time_offset
                cmd_pos = [float(commands[j][k]) for j in CSV_ORDER]
                state_pos = [float(positions[j][k]) for j in CSV_ORDER]
                state_vel = [float(velocities[j][k]) for j in CSV_ORDER]
                state_torque = [float(torques[j][k]) for j in CSV_ORDER]
                all_cmd_rows.append((t_abs, cmd_pos))
                all_state_rows.append((t_abs, state_pos, state_vel, state_torque))

            time_offset += duration + 0.1
            total_files += 1

    write_sage_output(converted_dir, all_cmd_rows, all_state_rows)
    log_message(f"Converted {total_files} motor CSVs → {converted_dir}")
    return converted_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    num_envs = args.num_envs
    physics_dt = 1.0 / args.physics_freq
    control_dt = 1.0 / args.control_freq
    divisor = args.physics_freq // args.control_freq

    if args.physics_freq % args.control_freq != 0:
        raise ValueError(f"physics_freq ({args.physics_freq}) must be divisible by control_freq ({args.control_freq})")

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Robot-specific config
    robot_cfg = _ROBOT_CONFIGS[args.robot_name]
    all_joint_names = robot_cfg["joint_names"]

    # Filter sysid joints if --joints specifies a subset
    if args.joints is not None:
        sysid_joint_names = [n for n in all_joint_names if any(jt in n for jt in args.joints)]
        if not sysid_joint_names:
            raise ValueError(
                f"--joints {args.joints} matched no sim joints from {all_joint_names}. "
                "Use joint type names like 'elbow', 'shoulder_pitch', etc."
            )
        log_message(f"Joint filter: {args.joints} → {sysid_joint_names}")
    else:
        sysid_joint_names = all_joint_names

    num_sysid_joints = len(sysid_joint_names)

    # Motor lag: sysid bounds YAML > actuator YAML > 0
    motor_lag_ms = float(cfg.get("fixed", {}).get("motor_lag_ms", 0.0))
    if motor_lag_ms == 0.0:
        actuator_params = load_actuator_params(robot_cfg["actuator_yaml"])
        motor_lag_ms = float(actuator_params.get("motor_lag_ms", 0.0))
    motor_lag_steps = round(motor_lag_ms / 1000.0 / control_dt)
    log_message(f"Robot: {args.robot_name}, {num_sysid_joints} joints")
    log_message(f"Motor lag: {motor_lag_ms}ms = {motor_lag_steps} control steps at {args.control_freq}Hz")

    # --- Detect mode and load data ---
    single_joint_mode = False

    if args.data_dir or args.data_files:
        # Single-joint parquet mode
        single_joint_mode = True
        if args.data_dir:
            parquet_files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
            if not parquet_files:
                raise ValueError(f"No .parquet files found in {args.data_dir}")
        else:
            parquet_files = [f.strip() for f in args.data_files.split(",")]

        log_message(f"MODE: Single-joint ({args.joint_name}), {len(parquet_files)} chirp files")
        commanded_1d, measured_1d = load_chirp_parquets(parquet_files, control_dt)

        if args.max_trajectory_len and len(commanded_1d) > args.max_trajectory_len:
            commanded_1d = commanded_1d[:args.max_trajectory_len]
            measured_1d = measured_1d[:args.max_trajectory_len]
            log_message(f"Truncated to {args.max_trajectory_len} steps")

        trajectory_len = len(commanded_1d)

    elif args.real_data_dir:
        # All-joints CSV mode — auto-convert motor CSVs if needed
        sage_dir = auto_convert_motor_csvs(args.real_data_dir, args.output_dir)
        log_message(f"MODE: All-joints (CSV)")

        # Determine which sysid joints actually exist in the real data.
        # When data has only one side (e.g. right arm), we load/score those joints
        # and mirror commanded positions + params to the other side.
        sage_joint_list_path = os.path.join(sage_dir, "joint_list.txt")
        with open(sage_joint_list_path) as f:
            available_data_joints = set(line.strip() for line in f if line.strip())
        data_joint_names = [n for n in sysid_joint_names if n in available_data_joints]
        if not data_joint_names:
            raise ValueError(
                f"No overlap between sysid joints {sysid_joint_names} and "
                f"data joints {sorted(available_data_joints)}"
            )
        if len(data_joint_names) < len(sysid_joint_names):
            mirrored_joints = [n for n in sysid_joint_names if n not in available_data_joints]
            log_message(f"Data has {len(data_joint_names)} of {len(sysid_joint_names)} sysid joints: {data_joint_names}")
            log_message(f"Mirroring params to {len(mirrored_joints)} joints: {mirrored_joints}")

        commanded_nj, measured_nj = load_real_data(sage_dir, control_dt, data_joint_names)

        if args.max_trajectory_len and len(commanded_nj) > args.max_trajectory_len:
            commanded_nj = commanded_nj[:args.max_trajectory_len]
            measured_nj = measured_nj[:args.max_trajectory_len]
            log_message(f"Truncated to {args.max_trajectory_len} steps")

        trajectory_len = len(commanded_nj)

    else:
        raise ValueError("Specify --data-dir / --data-files (single-joint) or --real-data-dir (all-joints)")

    log_message(f"Trajectory: {trajectory_len} steps at {args.control_freq}Hz = {trajectory_len * control_dt:.1f}s")

    # --- Setup simulation ---
    sim_cfg = SimulationCfg(
        dt=physics_dt,
        render_interval=divisor,
        gravity=(0.0, 0.0, -9.81),
    )
    _sim_section = _run_cfg.get("simulation", {})
    from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
    sim_cfg.physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator=_sim_section.get("integrator", "implicitfast"),
            ls_iterations=_sim_section.get("ls_iterations", 10),
            iterations=_sim_section.get("solver_iterations", 10),
        )
    )

    sim = SimulationContext(sim_cfg)
    SceneCfgCls = robot_cfg["scene_cfg_cls"]

    # Check that USD exists for UR10e
    if SceneCfgCls is Ur10eSysidSceneCfg and not os.path.isfile(_UR10_USD_PATH):
        raise FileNotFoundError(
            f"UR10 USD not found at {_UR10_USD_PATH}\n"
            "Place the UR10 USD asset at input/robot_models/ur10/ur10/ur10.usd"
        )

    scene_cfg = SceneCfgCls(num_envs=num_envs, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()

    robot = scene["robot"]
    device = robot.device

    # Joint index mapping
    joint_name_to_idx = {name: i for i, name in enumerate(robot.joint_names)}

    if single_joint_mode:
        if args.joint_name not in joint_name_to_idx:
            raise ValueError(f"Joint '{args.joint_name}' not found. Available: {robot.joint_names}")
        target_joint_idx = joint_name_to_idx[args.joint_name]
        log_message(f"Target joint: {args.joint_name} (sim index {target_joint_idx})")
        commanded_t = torch.tensor(commanded_1d, dtype=torch.float32, device=device)
        measured_t = torch.tensor(measured_1d, dtype=torch.float32, device=device)
        start_pos_scalar = commanded_t[0]
    else:
        commanded_t = torch.tensor(commanded_nj, dtype=torch.float32, device=device)
        measured_t = torch.tensor(measured_nj, dtype=torch.float32, device=device)
        start_pos_nj = commanded_t[0]  # (num_data_joints,)

        # Build data_joint_ids for scoring (sim indices of joints with real data)
        data_joint_ids = [joint_name_to_idx[n] for n in data_joint_names]
        data_joint_ids_tensor = torch.tensor(data_joint_ids, dtype=torch.long, device=device)

        # Build mirror mapping: command mirrored joints with same data as their counterpart
        mirror_joint_ids = []
        mirror_data_indices = []
        sysid_name_set = set(sysid_joint_names)
        data_name_set = set(data_joint_names)
        for i, dn in enumerate(data_joint_names):
            if dn.startswith("right_"):
                mn = "left_" + dn[6:]
            elif dn.startswith("left_"):
                mn = "right_" + dn[5:]
            else:
                continue
            if mn in sysid_name_set and mn not in data_name_set and mn in joint_name_to_idx:
                mirror_joint_ids.append(joint_name_to_idx[mn])
                mirror_data_indices.append(i)
        has_mirrors = len(mirror_joint_ids) > 0
        if has_mirrors:
            mirror_joint_ids_tensor = torch.tensor(mirror_joint_ids, dtype=torch.long, device=device)
            mirror_data_indices_tensor = torch.tensor(mirror_data_indices, dtype=torch.long, device=device)
            log_message(f"Mirror target joints: {[robot.joint_names[j] for j in mirror_joint_ids]}")

    # --- Create optimizer ---
    optimizer = CMAESOptimizer(
        config_path=args.config,
        num_envs=num_envs,
        device=str(device),
        sigma=args.sigma,
        max_iterations=args.max_iter,
        epsilon=args.epsilon,
        joint_types=args.joints,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Write run summary
    from datetime import datetime
    _act_cfg = _run_cfg.get("actuator", {})
    _act_model_type = _act_cfg.get("model_type", "implicit")
    _act_info = {"model_type": _act_model_type}
    _mt_aliases = {"gru": "lstm", "gru_perjoint": "lstm_perjoint"}
    _mt = _mt_aliases.get(_act_model_type, _act_model_type)
    if _mt in ("implicit", "dcmotor"):
        _yaml_file = _act_cfg.get("yaml_file")
        if _yaml_file:
            _act_info["yaml_file"] = _yaml_file
            try:
                from actuator_models import load_actuator_params
                _act_info["parameters"] = load_actuator_params(_yaml_file)
            except Exception:
                pass
    elif _mt == "lstm":
        _nf = _act_cfg.get("network_file")
        if _nf:
            _act_info["network_file"] = os.path.basename(_nf)
    elif _mt == "lstm_perjoint":
        _nfs = _act_cfg.get("network_files", {})
        if _nfs:
            _act_info["network_files"] = {jt: os.path.basename(f) for jt, f in _nfs.items()}

    _summary = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "robot_name": args.robot_name,
        "simulation": _run_cfg.get("simulation", {}),
        "sysid": {
            "bounds_yaml": os.path.basename(args.config),
            "real_data_dir": args.real_data_dir,
            "joints": optimizer.joint_types,
            "mirror": optimizer.mirror,
            "num_envs": args.num_envs,
            "max_iter": args.max_iter,
            "sigma": args.sigma,
            "epsilon": args.epsilon,
            "buffer_time": args.buffer_time,
            "physics_freq": args.physics_freq,
            "control_freq": args.control_freq,
        },
        "bounds": {
            "parameters": cfg.get("parameters", {}),
            "fixed": cfg.get("fixed", {}),
        },
        "actuator_model": _act_info,
    }
    _summary_path = os.path.join(args.output_dir, "run_summary.yaml")
    with open(_summary_path, "w") as f:
        yaml.dump(_summary, f, default_flow_style=False, sort_keys=False)
    log_message(f"Run summary saved to {_summary_path}")

    log_file = os.path.join(args.output_dir, "optimization_log.csv")

    # Build arm_joint_ids in optimizer's expand_all_envs order for param writing.
    # With mirroring (H1): right joints first, then left, following optimizer.joint_types order.
    arm_joint_ids = []
    if optimizer.mirror:
        for prefix in ("right_", "left_"):
            for jt in optimizer.joint_types:
                sim_name = f"{prefix}{jt}"
                if sim_name not in joint_name_to_idx:
                    raise ValueError(f"Sim joint '{sim_name}' not found. Available: {robot.joint_names}")
                arm_joint_ids.append(joint_name_to_idx[sim_name])
    else:
        for jt in optimizer.joint_types:
            matched = [n for n in sysid_joint_names if jt in n]
            if not matched:
                raise ValueError(f"Joint type '{jt}' matches no sim joint in {sysid_joint_names}")
            arm_joint_ids.append(joint_name_to_idx[matched[0]])

    arm_joint_ids_tensor = torch.tensor(arm_joint_ids, dtype=torch.int32, device=device)
    log_message(f"Sim robot: {len(robot.joint_names)} joints, sysid joint IDs: {arm_joint_ids}")

    # Build mapping from bounds YAML joint types to actual sim joint names
    # so best_params.yaml keys can be pasted directly into actuator YAMLs.
    # Mirrored (H1):      "shoulder_pitch" -> ".*_shoulder_pitch" (regex key)
    # Non-mirrored (UR10e): "shoulder_pan" -> "shoulder_pan_joint" (exact name)
    # Non-mirrored (H1 right arm): "right_shoulder_pitch" stays as-is (already exact)
    _joint_name_map = {}
    for jt in optimizer.joint_types:
        if jt in sysid_joint_names:
            # Exact match — no mapping needed
            continue
        if optimizer.mirror:
            # Mirrored: type is a suffix shared by left/right — use regex key
            _joint_name_map[jt] = f".*_{jt}"
        else:
            # Non-mirrored: find the sim joint name containing this type
            for sim_name in sysid_joint_names:
                if jt in sim_name:
                    _joint_name_map[jt] = sim_name
                    break

    log_message(f"CMA-ES: {optimizer.num_params} params, {num_envs} envs, "
                f"max {optimizer.max_iterations} gens")
    log_message(f"Parameters: {optimizer.param_names}")

    # --- Helpers ---
    def to_torch(data):
        return wp.to_torch(data) if isinstance(data, wp.array) else data

    def sim_step():
        _apply_stribeck_torque()
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(physics_dt)

    # Map from optimizer parameter name to the Isaac Lab sim write function.
    # Only parameters present in the bounds YAML's "parameters" section are optimized.
    #
    # Verified Newton model properties (newton._src.sim.model.Model):
    #   joint_armature                  — rotor inertia [kg·m²]
    #   joint_friction                  — Coulomb/dry friction [N·m]
    #   joint_target_ke                 — PD position gain kp [N·m/rad]
    #   joint_target_kd                 — PD velocity gain kd [N·m·s/rad]
    #   mujoco.dof_passive_damping      — passive viscous damping F = -d * qvel [N·m·s/rad]
    #                                     (MuJoCo-solver custom attribute, default 0.0)
    #
    # Isaac Lab API → Newton model mapping:
    #   write_joint_armature_to_sim()   → model.joint_armature
    #   write_joint_friction_to_sim()   → model.joint_friction
    #   write_joint_stiffness_to_sim()  → model.joint_target_ke  (PD kp)
    #   write_joint_damping_to_sim()    → model.joint_target_kd  (PD kd)
    #   (no Isaac Lab API)              → model.mujoco.dof_passive_damping  (custom writer below)
    _env_ids = torch.arange(num_envs, dtype=torch.int32, device=device)

    # --- Locate Newton model for direct property access (viscous damping) ---
    # Newton's passive viscous damping (model.mujoco.dof_passive_damping) has no
    # Isaac Lab write API, so we access the Newton model directly via BFS from
    # robot.root_view (an ArticulationView wrapping the Newton solver).
    _newton_model = None
    _needs_newton_model = "viscous_friction" in {n for n in optimizer.property_names}
    if _needs_newton_model:
        _visited = set()
        _queue = [("view", robot.root_view)]
        for _depth in range(4):
            _next_queue = []
            for _path, _obj in _queue:
                _oid = id(_obj)
                if _oid in _visited:
                    continue
                _visited.add(_oid)
                if hasattr(_obj, "joint_target_ke"):
                    _newton_model = _obj
                    log_message(f"Newton model found at root_view -> {_path} ({type(_obj).__name__})")
                    break
                for _attr in dir(_obj):
                    if _attr.startswith("__"):
                        continue
                    try:
                        _child = getattr(_obj, _attr)
                        if not callable(_child) and not isinstance(
                            _child, (int, float, str, bool, type(None))
                        ):
                            _next_queue.append((f"{_path}.{_attr}", _child))
                    except Exception:
                        pass
            if _newton_model is not None:
                break
            _queue = _next_queue

        if _newton_model is None:
            raise RuntimeError(
                "Cannot find Newton model (needed for viscous_friction). "
                "Remove 'viscous_friction' from bounds YAML or check Newton version."
            )

        # Verify the mujoco custom attribute namespace exists
        _mujoco_ns = getattr(_newton_model, "mujoco", None)
        if _mujoco_ns is None or not hasattr(_mujoco_ns, "dof_passive_damping"):
            raise RuntimeError(
                "Newton model has no 'mujoco.dof_passive_damping' attribute. "
                "Ensure Newton is using the MuJoCo Warp solver (solver_cfg.integrator = 'implicitfast')."
            )
        log_message(f"Newton mujoco.dof_passive_damping available (shape: {_newton_model.mujoco.dof_passive_damping.shape})")

    def _write_viscous_friction(vals: torch.Tensor):
        """Write passive viscous damping directly to Newton's mujoco.dof_passive_damping.

        vals: shape (num_envs, num_arm_joints) — per-env, per-joint damping values.
        The Newton array may be 2D (num_worlds, num_dofs) or 1D (total_dofs,).
        """
        dof_damping = _newton_model.mujoco.dof_passive_damping
        vals_np = vals.cpu().numpy()
        dof_np = dof_damping.numpy()
        num_dofs = len(robot.joint_names)
        if dof_np.ndim == 2:
            # Shape: (num_worlds, num_dofs)
            for env_i in range(vals_np.shape[0]):
                for j_idx, joint_id in enumerate(arm_joint_ids):
                    dof_np[env_i, joint_id] = vals_np[env_i, j_idx]
        else:
            # Shape: (num_worlds * num_dofs,) — flattened
            for env_i in range(vals_np.shape[0]):
                for j_idx, joint_id in enumerate(arm_joint_ids):
                    dof_np[env_i * num_dofs + joint_id] = vals_np[env_i, j_idx]
        dof_damping.assign(dof_np)

    # --- Stribeck friction support ---
    # When stribeck_velocity is being optimized, we disable Newton's built-in
    # Coulomb friction (joint_friction = 0) and instead apply a smooth Stribeck
    # friction model as an external torque each sim step:
    #   τ = -dynamic_friction * tanh(v / v_stribeck)
    # This smoothly transitions from 0 at v=0 to ±dynamic_friction at |v| >> v_s.
    _has_stribeck = "stribeck_velocity" in {n for n in optimizer.property_names}
    _stribeck_velocity_cache = None  # (num_envs, num_arm_joints) — set per generation

    def _cache_stribeck_velocity(vals: torch.Tensor):
        nonlocal _stribeck_velocity_cache
        _stribeck_velocity_cache = vals.clone()

    def _write_dynamic_friction_maybe_disable(vals: torch.Tensor):
        """Write dynamic_friction. If Stribeck is active, keep values cached but
        set Newton's joint_friction to 0 (friction applied externally instead)."""
        if _has_stribeck:
            # Cache the values for external Stribeck computation; zero out Newton's
            robot.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=torch.zeros_like(vals),
                joint_ids=arm_joint_ids_tensor, env_ids=_env_ids,
            )
        else:
            robot.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=vals, joint_ids=arm_joint_ids_tensor, env_ids=_env_ids,
            )

    # Cache for dynamic_friction values (needed for Stribeck torque computation)
    _dynamic_friction_cache = None

    def _cache_and_write_dynamic_friction(vals: torch.Tensor):
        nonlocal _dynamic_friction_cache
        _dynamic_friction_cache = vals.clone()
        _write_dynamic_friction_maybe_disable(vals)

    def _apply_stribeck_torque():
        """Compute and apply Stribeck friction as external torque each sim step."""
        if not _has_stribeck or _stribeck_velocity_cache is None or _dynamic_friction_cache is None:
            return
        vel = to_torch(robot.data.joint_vel)[:, arm_joint_ids]
        # τ = -f_c * tanh(v / v_s)
        friction_torque = -_dynamic_friction_cache * torch.tanh(vel / _stribeck_velocity_cache)
        # Apply as external effort on arm joints
        effort = to_torch(robot.data.joint_effort_target).clone()
        effort[:, arm_joint_ids] = friction_torque
        robot.set_joint_effort_target_index(target=effort)

    _PARAM_WRITERS = {
        "armature":          lambda vals: robot.write_joint_armature_to_sim_index(armature=vals, joint_ids=arm_joint_ids_tensor, env_ids=_env_ids),
        "dynamic_friction":  _cache_and_write_dynamic_friction,
        "viscous_friction":  _write_viscous_friction,
        "stribeck_velocity": _cache_stribeck_velocity,
        "stiffness":         lambda vals: robot.write_joint_stiffness_to_sim_index(stiffness=vals, joint_ids=arm_joint_ids_tensor, env_ids=_env_ids),
        "damping":           lambda vals: robot.write_joint_damping_to_sim_index(damping=vals, joint_ids=arm_joint_ids_tensor, env_ids=_env_ids),
    }

    def write_params_to_envs():
        for prop_name in optimizer.property_names:
            writer = _PARAM_WRITERS.get(prop_name)
            if writer is None:
                log_message(f"WARNING: No sim writer for parameter '{prop_name}', skipping")
                continue
            values = optimizer.expand_all_envs(prop_name)
            try:
                writer(values)
            except (AttributeError, TypeError) as e:
                log_message(f"WARNING: write {prop_name} to sim failed: {e}")

    def reset_envs():
        joint_pos = to_torch(robot.data.joint_pos).clone()
        joint_vel = to_torch(robot.data.joint_vel).clone()
        if single_joint_mode:
            joint_pos[:, target_joint_idx] = start_pos_scalar
            joint_vel[:, target_joint_idx] = 0.0
        else:
            joint_pos[:, data_joint_ids_tensor] = start_pos_nj.unsqueeze(0).expand(num_envs, -1)
            joint_vel[:, data_joint_ids_tensor] = 0.0
            if has_mirrors:
                joint_pos[:, mirror_joint_ids_tensor] = start_pos_nj[mirror_data_indices_tensor].unsqueeze(0).expand(num_envs, -1)
                joint_vel[:, mirror_joint_ids_tensor] = 0.0
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(physics_dt)

    # --- Optimization loop ---
    buffer_steps = int(args.buffer_time / control_dt)
    total_sim_steps = (buffer_steps + trajectory_len) * divisor
    log_message(f"Sim steps per generation: {total_sim_steps} "
                f"(buffer {buffer_steps * divisor} + trajectory {trajectory_len * divisor}) "
                f"x {num_envs} envs")

    pbar = tqdm(range(optimizer.max_iterations), desc="CMA-ES", unit="gen",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for gen in pbar:
        optimizer.sample_population()
        write_params_to_envs()
        reset_envs()

        # Inner progress bar for sim steps within a generation
        inner_pbar = tqdm(total=total_sim_steps, desc=f"  Gen {gen}", unit="step",
                          leave=False, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        # Buffer: hold at start to settle
        for step in range(buffer_steps * divisor):
            if step % divisor == 0:
                target = to_torch(robot.data.joint_pos).clone()
                if single_joint_mode:
                    target[:, target_joint_idx] = start_pos_scalar
                else:
                    target[:, data_joint_ids_tensor] = start_pos_nj.unsqueeze(0).expand(num_envs, -1)
                    if has_mirrors:
                        target[:, mirror_joint_ids_tensor] = start_pos_nj[mirror_data_indices_tensor].unsqueeze(0).expand(num_envs, -1)
                robot.set_joint_position_target_index(target=target)
            sim_step()
            inner_pbar.update(1)

        # Motor lag buffer
        if motor_lag_steps > 0:
            cmd_buffer = deque(maxlen=motor_lag_steps + 1)
            if single_joint_mode:
                for _ in range(motor_lag_steps):
                    cmd_buffer.append(start_pos_scalar.clone())
            else:
                for _ in range(motor_lag_steps):
                    cmd_buffer.append(start_pos_nj.clone())
        else:
            cmd_buffer = None

        # Replay trajectory
        for t in range(trajectory_len * divisor):
            index = t // divisor
            if index >= trajectory_len:
                break

            if t % divisor == 0:
                cmd = commanded_t[index]

                if cmd_buffer is not None:
                    cmd_buffer.append(cmd.clone() if cmd.dim() > 0 else cmd.clone())
                    delayed_cmd = cmd_buffer[0]
                else:
                    delayed_cmd = cmd

                target = to_torch(robot.data.joint_pos).clone()

                if single_joint_mode:
                    target[:, target_joint_idx] = delayed_cmd
                    robot.set_joint_position_target_index(target=target)
                    sim_pos = to_torch(robot.data.joint_pos)[:, target_joint_idx]
                    real_pos = measured_t[index]
                    diff = sim_pos - real_pos
                    optimizer.scores += diff * diff
                    optimizer._score_steps += 1
                else:
                    target[:, data_joint_ids_tensor] = delayed_cmd.unsqueeze(0).expand(num_envs, -1)
                    if has_mirrors:
                        target[:, mirror_joint_ids_tensor] = delayed_cmd[mirror_data_indices_tensor].unsqueeze(0).expand(num_envs, -1)
                    robot.set_joint_position_target_index(target=target)
                    sim_pos = to_torch(robot.data.joint_pos)[:, data_joint_ids_tensor]
                    optimizer.accumulate_score(sim_pos, measured_t[index])

            sim_step()
            inner_pbar.update(1)

        inner_pbar.close()

        converged = optimizer.evolve()
        pbar.set_postfix(best_mse=f"{optimizer._best_score:.6f}")

        if gen % args.log_interval == 0 or converged:
            best = optimizer.get_best_params()
            parts = [f"Gen {gen:4d} | best MSE: {optimizer._best_score:.6f}"]
            for prop in optimizer.property_names:
                vals = [best[prop][jt] for jt in optimizer.joint_types]
                parts.append(f"{prop}: {vals}")
            tqdm.write(f"[SYSID] {' | '.join(parts)}")
        optimizer.log_generation(log_file)

        if converged:
            tqdm.write(f"[SYSID] Converged at generation {gen}")
            break
    pbar.close()

    # --- Save results ---
    best_params = optimizer.get_best_params(joint_name_map=_joint_name_map)
    best_params["robot_name"] = args.robot_name
    if single_joint_mode:
        best_params["mode"] = "single_joint"
        best_params["joint_name"] = args.joint_name
        best_params["data_source"] = args.data_dir or args.data_files
        best_params["num_files"] = len(parquet_files)
    else:
        best_params["mode"] = "all_joints"
        best_params["data_source"] = args.real_data_dir

    best_params_file = os.path.join(args.output_dir, "best_params.yaml")
    with open(best_params_file, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False, sort_keys=False)
    log_message(f"\nBest parameters saved to {best_params_file}")
    log_message(f"Best MSE: {optimizer._best_score:.6f}")

    actuator_yaml = robot_cfg["actuator_yaml"]
    log_message(f"\n--- Copy to {actuator_yaml} ---")
    for prop in optimizer.property_names:
        mean_val = best_params.get(f"{prop}_mean", "N/A")
        per_joint = best_params.get(prop, {})
        log_message(f"{prop} (mean {mean_val}):")
        for jname, jval in per_joint.items():
            log_message(f"  {jname}: {jval}")

    # --- Timing summary ---
    elapsed = time.time() - t_start
    hours, rem = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds}s"
    else:
        time_str = f"{elapsed:.1f}s"
    log_message(f"\nTotal time: {time_str}")


if __name__ == "__main__":
    main()
    if simulation_app is not None:
        simulation_app.close()
