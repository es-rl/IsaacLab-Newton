"""Run SAGE-compatible joint motion benchmark under Newton physics.

Paths for motion data, real data, and output can be set in the run config
(input/run_configs/<robot>.yaml) or overridden via CLI arguments.

Usage (paths from run config):
    python scripts/sim2real_gap/run_benchmark.py --robot-name h1 --headless

Usage with motion file (CLI override):
    python scripts/sim2real_gap/run_benchmark.py \
        --robot-name h1 \
        --motion-files input/motion_files/h1/custom/arm_reach.txt \
        --output-folder output/sim2real_benchmark \
        --headless

Usage with real robot control.csv (replays exact real commands in sim):
    python scripts/sim2real_gap/run_benchmark.py \
        --robot-name h1 \
        --real-control-csv output/sim2real_benchmark/real/h1/custom/motion_stand_left_arm_swing/control.csv.bak \
        --motion-name motion_stand_left_arm_swing \
        --original-control-freq 100 \
        --headless
"""

import argparse
import ast
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "input"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sysid"))
from actuator_models import load_actuator_params
from run_configs import load_run_cfg

from isaaclab.app import AppLauncher

# Parse arguments before importing anything that needs the simulator
parser = argparse.ArgumentParser(description="SAGE joint motion benchmark (Newton backend)")
parser.add_argument("--robot-name", type=str, default="h1", help="Robot name (default: h1)")
parser.add_argument(
    "--motion-files",
    type=str,
    default=None,
    help="Path to motion file(s) or directory. Single file, comma-separated, or directory.",
)
parser.add_argument(
    "--real-control-csv",
    type=str,
    default=None,
    help="Path to real robot control.csv (SAGE format or bare CSV). "
    "Replays the exact commands sent to the real robot in simulation.",
)
parser.add_argument(
    "--motion-name",
    type=str,
    default=None,
    help="Motion name for output directory (required with --real-control-csv).",
)
parser.add_argument("--valid-joints-file", type=str, default=None, help="Path to valid joints file")
parser.add_argument("--output-folder", type=str, default=None, help="Path to output folder (default: from run config)")
parser.add_argument("--fix-root", action="store_true", default=True, help="Fix root joint (default: True)")
parser.add_argument("--physics-freq", type=int, default=200, help="Physics timestep frequency (Hz)")
parser.add_argument("--render-freq", type=int, default=200, help="Render timestep frequency (Hz)")
parser.add_argument("--control-freq", type=int, default=None, help="Control frequency (Hz)")
parser.add_argument(
    "--original-control-freq", type=int, default=None, help="Original control frequency of motion files (Hz)"
)
parser.add_argument("--kp", type=float, default=None, help="Override default joint stiffness")
parser.add_argument("--kd", type=float, default=None, help="Override default joint damping")
parser.add_argument(
    "--motor-lag-ms",
    type=float,
    default=None,
    help="Motor command lag in milliseconds. Delays position targets by N physics steps. "
    "If not set, reads motor_lag_ms from the arms actuator YAML (if present).",
)
parser.add_argument("--record-video", action="store_true", help="Record video")

# Add AppLauncher args (--headless, --device, etc.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Load per-robot run config; CLI args override config values
_run_cfg = load_run_cfg(args.robot_name)
_sim_cfg = _run_cfg.get("simulation", {})
_bench_cfg = _run_cfg.get("benchmark", {})

# Integer-defaulted args: override only if CLI left them at the argparse default
for _attr, _default, _section in [
    ("physics_freq", 200, _sim_cfg),
    ("render_freq", 200, _sim_cfg),
]:
    if getattr(args, _attr) == _default and _attr in _section:
        setattr(args, _attr, _section[_attr])

# Save CLI-explicit motion source flags before config merge
_cli_motion_files = args.motion_files
_cli_real_control_csv = args.real_control_csv

# None-defaulted args: override only if CLI left them as None
for _attr, _section in [
    ("control_freq", _sim_cfg),
    ("kp", _bench_cfg),
    ("kd", _bench_cfg),
    ("motor_lag_ms", _bench_cfg),
    ("motion_files", _bench_cfg),
    ("real_control_csv", _bench_cfg),
    ("original_control_freq", _bench_cfg),
    ("motion_name", _bench_cfg),
    ("output_folder", _bench_cfg),
]:
    if getattr(args, _attr) is None and _attr in _section:
        setattr(args, _attr, _section[_attr])

# motion_files and real_control_csv are mutually exclusive.
# If CLI explicitly set one, discard the other (which came from config).
if args.real_control_csv and args.motion_files:
    if _cli_real_control_csv is not None:
        args.motion_files = None
    elif _cli_motion_files is not None:
        args.real_control_csv = None

# motion_source from config's motion_name (used as parent folder in output)
args.motion_source = _bench_cfg.get("motion_name", "custom")

# Validate required args after config merge
if not args.output_folder:
    parser.error("--output-folder is required (set via CLI or run config benchmark.output_folder)")

# Launch Isaac Lab app (initializes Newton backend)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now safe to import simulation-dependent modules
from sage.simulation import get_motion_files, get_motion_name, log_message  # noqa: E402

from newton_benchmark import NewtonJointMotionBenchmark  # noqa: E402



def _write_run_summary(output_folder, robot_name, motion_source, run_cfg, args):
    """Write a run_summary.yaml with timestamp and config snapshot."""
    from datetime import datetime

    import yaml

    summary_dir = os.path.join(output_folder, "sim", robot_name, motion_source)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "run_summary.yaml")

    act_cfg = run_cfg.get("actuator", {})
    model_type = act_cfg.get("model_type", "implicit")

    # Build actuator_model section with actual params or network file info
    _type_aliases = {"gru": "lstm", "gru_perjoint": "lstm_perjoint"}
    mt = _type_aliases.get(model_type, model_type)

    actuator_info = {"model_type": model_type}

    if mt in ("implicit", "dcmotor"):
        yaml_file = act_cfg.get("yaml_file")
        if yaml_file:
            actuator_info["yaml_file"] = yaml_file
            try:
                actuator_info["parameters"] = load_actuator_params(yaml_file)
            except Exception:
                pass
    elif mt == "lstm":
        network_file = act_cfg.get("network_file")
        if network_file:
            actuator_info["network_file"] = os.path.basename(network_file)
    elif mt == "lstm_perjoint":
        network_files = act_cfg.get("network_files", {})
        if network_files:
            actuator_info["network_files"] = {
                jt: os.path.basename(f) for jt, f in network_files.items()
            }

    summary = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "robot_name": robot_name,
        "simulation": run_cfg.get("simulation", {}),
        "benchmark": run_cfg.get("benchmark", {}),
        "actuator_model": actuator_info,
        "cli_overrides": {},
    }

    # Record CLI overrides that differ from config defaults
    for attr in ("kp", "kd", "motor_lag_ms", "physics_freq", "render_freq", "control_freq"):
        val = getattr(args, attr, None)
        if val is not None:
            summary["cli_overrides"][attr] = val

    if not summary["cli_overrides"]:
        del summary["cli_overrides"]

    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    log_message(f"Run summary saved to {summary_path}")


def prepare_motor_csv_data(data_dir, output_folder, robot_name, motion_source="custom"):
    """Convert motor CSVs to SAGE real data and motion files for sim.

    Walks data_dir recursively for *_motor.csv files. For each subdirectory
    containing motor CSVs:
    - Converts to SAGE real format in output_folder/real/<robot>/<motion_source>/<motion_name>/
    - Creates a temp motion file (bare CSV of commanded positions) for sim playback

    Returns list of (motion_file_path, motion_name, is_temp) tuples.
    """
    from convert_h1_chirp_to_csv import (
        CANONICAL_ORDER,
        CSV_ORDER,
        find_motor_csvs,
        has_position_error,
        load_csv_with_commands,
        write_sage_output,
    )

    subdirs = find_motor_csvs(data_dir, recursive=True)
    if not subdirs:
        return []

    log_message(f"Found {len(subdirs)} directories with motor CSVs in {data_dir}")
    results = []

    for subdir_name, dir_path in subdirs:
        real_dir = os.path.join(output_folder, "real", robot_name, motion_source)

        # Check if this subdir has CSVs with position_error (direct mode)
        csvs = sorted(f for f in os.listdir(dir_path) if f.endswith("_motor.csv"))
        if not csvs:
            continue

        use_direct = has_position_error(os.path.join(dir_path, csvs[0]))

        if use_direct:
            import numpy as np

            # Per-file: each CSV becomes its own motion
            for fname in csvs:
                path = os.path.join(dir_path, fname)
                motion_name = fname.replace("_motor.csv", "")
                motion_real_dir = os.path.join(real_dir, motion_name)

                time_s, commands, positions, velocities, torques = load_csv_with_commands(path)
                t0 = time_s[0]
                time_s = time_s - t0
                dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 0.005

                # Skip real data conversion if already present
                real_control = os.path.join(motion_real_dir, "control.csv")
                if os.path.isfile(real_control):
                    log_message(f"  {motion_name}: real CSVs exist, skipping conversion")
                else:
                    all_cmd_rows = []
                    all_state_rows = []
                    for k in range(len(time_s)):
                        cmd_pos = [float(commands[j][k]) for j in CSV_ORDER]
                        state_pos = [float(positions[j][k]) for j in CSV_ORDER]
                        state_vel = [float(velocities[j][k]) for j in CSV_ORDER]
                        state_torque = [float(torques[j][k]) for j in CSV_ORDER]
                        all_cmd_rows.append((time_s[k], cmd_pos))
                        all_state_rows.append((time_s[k], state_pos, state_vel, state_torque))

                    write_sage_output(motion_real_dir, all_cmd_rows, all_state_rows)
                    log_message(f"  Converted {motion_name}: {len(time_s)} samples ({1/dt:.0f}Hz)")

                # Create temp motion file for sim playback (bare CSV of commanded positions)
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, prefix=f"motion_{motion_name}_"
                )
                tmp.write(",".join(CANONICAL_ORDER) + "\n")
                for k in range(len(time_s)):
                    cmd_pos = [float(commands[j][k]) for j in CSV_ORDER]
                    tmp.write(",".join(f"{v}" for v in cmd_pos) + "\n")
                tmp.close()

                results.append((tmp.name, motion_name, True))
        else:
            # No position_error — skip auto-convert (need chirp filename reconstruction)
            log_message(f"  SKIP {subdir_name}: no position_error columns (use convert_h1_chirp_to_csv.py manually)")

    return results


def control_csv_to_motion_file(control_csv_path, joint_list_path=None):
    """Convert a real robot control.csv to a motion file format.

    Supports two formats:
    - Bare CSV: joint header + comma-separated values (e.g., control.csv.bak)
    - SAGE format: type,timestamp,positions with list strings

    Returns path to a temporary motion file.
    """
    with open(control_csv_path) as f:
        first_line = f.readline().strip()

    # Detect format: SAGE format starts with "type,timestamp,positions"
    is_sage_format = first_line.startswith("type,")

    if not is_sage_format:
        # Bare CSV format — already a valid motion file (no embedded timestamps)
        log_message(f"Control CSV is bare format, using directly as motion file")
        return control_csv_path, None

    # SAGE format — extract positions and convert to bare CSV
    log_message(f"Control CSV is SAGE format, converting to motion file...")

    # Get joint names from joint_list.txt if available
    joint_names = None
    if joint_list_path and os.path.exists(joint_list_path):
        with open(joint_list_path) as f:
            joint_names = [line.strip() for line in f if line.strip()]
    else:
        # Try to find joint_list.txt in the same directory
        csv_dir = os.path.dirname(control_csv_path)
        jl_path = os.path.join(csv_dir, "joint_list.txt")
        if os.path.exists(jl_path):
            with open(jl_path) as f:
                joint_names = [line.strip() for line in f if line.strip()]

    # Parse SAGE control.csv (preserving timestamps for frequency detection)
    rows = []
    timestamps = []
    with open(control_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            positions = ast.literal_eval(row["positions"])
            rows.append(positions)
            timestamps.append(float(row["timestamp"]))

    if not rows:
        raise ValueError(f"No data rows in {control_csv_path}")

    num_joints = len(rows[0])
    if joint_names is None:
        # Fall back to generic names — but this won't match the robot
        raise ValueError(
            f"No joint_list.txt found near {control_csv_path}. "
            "Joint names are needed to map control positions to robot joints."
        )

    if len(joint_names) != num_joints:
        raise ValueError(
            f"joint_list.txt has {len(joint_names)} joints but control.csv has {num_joints} values per row"
        )

    # Detect original control frequency from timestamps (microseconds)
    detected_freq = None
    if len(timestamps) >= 2:
        # Timestamps are in microseconds; compute median dt
        dts = [timestamps[i + 1] - timestamps[i] for i in range(min(len(timestamps) - 1, 50))]
        median_dt_us = sorted(dts)[len(dts) // 2]
        if median_dt_us > 0:
            detected_freq = round(1e6 / median_dt_us, 1)
            duration_s = (timestamps[-1] - timestamps[0]) / 1e6
            log_message(f"Detected control frequency: {detected_freq:.1f} Hz "
                        f"(median dt={median_dt_us:.0f} us, duration={duration_s:.1f}s)")

    # Write temporary motion file
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix="motion_")
    tmp.write(",".join(joint_names) + "\n")
    for positions in rows:
        tmp.write(",".join(f"{v}" for v in positions) + "\n")
    tmp.close()

    log_message(f"Converted {len(rows)} control commands to motion file: {tmp.name}")
    return tmp.name, detected_freq


def _is_motor_csv_dir(path):
    """Check if path is a directory containing *_motor.csv files (directly or in subdirs)."""
    if not os.path.isdir(path):
        return False
    # Check direct children
    for f in os.listdir(path):
        if f.endswith("_motor.csv"):
            return True
    # Check one level of subdirs
    for d in os.listdir(path):
        sub = os.path.join(path, d)
        if os.path.isdir(sub):
            for f in os.listdir(sub):
                if f.endswith("_motor.csv"):
                    return True
    return False


def main():
    if args.real_control_csv and args.motion_files:
        raise ValueError("Specify either --motion-files or --real-control-csv, not both")
    if not args.real_control_csv and not args.motion_files:
        raise ValueError("Must specify either --motion-files or --real-control-csv")

    benchmark = NewtonJointMotionBenchmark(args)

    _write_run_summary(args.output_folder, args.robot_name, args.motion_source, _run_cfg, args)

    if args.real_control_csv:
        # Replay real robot commands in simulation
        motion_file, detected_freq = control_csv_to_motion_file(args.real_control_csv)

        # Auto-set original_control_freq from CSV timestamps if not provided by user
        if detected_freq is not None and args.original_control_freq is None:
            args.original_control_freq = detected_freq
            benchmark.original_control_freq = detected_freq
            log_message(f"Auto-set --original-control-freq {detected_freq:.1f} from control.csv timestamps")

        motion_name = args.motion_name or get_motion_name(args.real_control_csv)
        log_message(f"################### REPLAYING REAL COMMANDS: {motion_name} ###################")
        benchmark.set_motion(motion_file, motion_name)
        benchmark.run_benchmark()

        # Clean up temp file if we created one
        if motion_file != args.real_control_csv:
            os.unlink(motion_file)

    elif _is_motor_csv_dir(args.motion_files):
        # Auto-convert motor CSVs: populate real/ folder and create motion files for sim
        log_message(f"Detected motor CSVs in {args.motion_files} — auto-converting to SAGE format")
        prepared = prepare_motor_csv_data(args.motion_files, args.output_folder, args.robot_name, args.motion_source)

        if not prepared:
            raise ValueError(f"No convertible motor CSVs found in {args.motion_files}")

        log_message(f"Prepared {len(prepared)} motions for benchmark")

        for motion_file, motion_name, is_temp in prepared:
            try:
                log_message(f"################### PROCESSING {motion_name} ###################")
                benchmark.set_motion(motion_file, motion_name)
                benchmark.run_benchmark()
            except Exception as e:
                log_message(f"Error processing {motion_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            finally:
                if is_temp:
                    os.unlink(motion_file)

    else:
        # Standard motion file playback
        motion_files = get_motion_files(args.motion_files)
        if not motion_files:
            raise ValueError(f"No motion files found in {args.motion_files}")

        log_message(f"Found {len(motion_files)} motion files to process")

        for motion_file in motion_files:
            try:
                motion_name = get_motion_name(motion_file)
                log_message(f"################### PROCESSING {motion_file} ###################")
                benchmark.set_motion(motion_file, motion_name)
                benchmark.run_benchmark()
            except Exception as e:
                log_message(f"Error processing {motion_file}: {str(e)}")
                import traceback

                traceback.print_exc()
                continue

    benchmark.log_joint_properties()


if __name__ == "__main__":
    main()
    simulation_app.close()
