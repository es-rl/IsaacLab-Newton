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
import shutil
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
parser.add_argument("--num-envs", type=int, default=None, help="Number of parallel envs (default: matches motion count)")
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
parser.add_argument(
    "--real-init-pose-sync",
    action="store_true",
    help="Teleport scored joints to row 0 of real state_motor.csv after the buffer phase.",
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

if not args.real_init_pose_sync and "real_init_pose_sync" in _bench_cfg:
    args.real_init_pose_sync = bool(_bench_cfg["real_init_pose_sync"])

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

# Save DISPLAY before AppLauncher: Isaac Sim clears it in headless mode,
# but the Newton viewer (pyglet) needs it for X11 windowing.
_saved_display = os.environ.get("DISPLAY")

# Launch Isaac Lab app (initializes Newton backend)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Restore DISPLAY so the Newton viewer can connect to the X server.
if _saved_display is not None and "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = _saved_display

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
    elif mt == "fmu":
        yaml_file = act_cfg.get("yaml_file")
        if yaml_file:
            actuator_info["yaml_file"] = yaml_file
            try:
                actuator_info["parameters"] = load_actuator_params(yaml_file)
            except Exception:
                pass
        fmu_path = act_cfg.get("fmu_path")
        if fmu_path:
            actuator_info["fmu_path"] = fmu_path
        actuator_info["fmu_step_size"] = act_cfg.get("fmu_step_size", 0.002)
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
        "runtime_options": {
            "real_init_pose_sync": bool(getattr(args, "real_init_pose_sync", False)),
        },
        "cli_overrides": {},
    }

    # Record CLI overrides that differ from config defaults
    for attr in (
        "kp",
        "kd",
        "motor_lag_ms",
        "physics_freq",
        "render_freq",
        "control_freq",
    ):
        val = getattr(args, attr, None)
        if val is not None:
            summary["cli_overrides"][attr] = val

    if not summary["cli_overrides"]:
        del summary["cli_overrides"]

    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    log_message(f"Run summary saved to {summary_path}")


def _detect_motor_csv_joints(csv_path):
    """Detect joint names from a motor CSV header.

    Looks for columns matching ``<joint>_position`` and returns the list of
    joint name prefixes found, in header order.
    """
    with open(csv_path) as f:
        header = [c.strip() for c in f.readline().strip().split(",")]
    joints = []
    for col in header:
        if col.endswith("_position") and col != "commanded_position":
            joints.append(col.removesuffix("_position"))
    return joints


def _load_generic_motor_csv(csv_path, joints):
    """Load a motor CSV with arbitrary joints.

    Returns time_s, commands, positions, velocities, torques — all dicts keyed
    by joint name.
    """
    import csv as csv_mod

    import numpy as np

    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)

    header_keys = set(rows[0].keys()) if rows else set()
    time_s = np.array([float(r["time_s"]) for r in rows])

    positions = {}
    velocities = {}
    torques = {}
    commands = {}

    for joint in joints:
        positions[joint] = np.array([float(r[f"{joint}_position"]) for r in rows])
        velocities[joint] = np.array([float(r[f"{joint}_velocity"]) for r in rows])
        torques[joint] = np.array([float(r[f"{joint}_torque"]) for r in rows])

        if f"{joint}_position_error" in header_keys:
            pos_err = np.array([float(r[f"{joint}_position_error"]) for r in rows])
            commands[joint] = positions[joint] + pos_err
        else:
            commands[joint] = positions[joint].copy()

    return time_s, commands, positions, velocities, torques


def prepare_motor_csv_data(data_dir, output_folder, robot_name, motion_source="custom"):
    """Convert motor CSVs to SAGE real data and motion files for sim.

    Walks data_dir recursively for *_motor.csv files. For each subdirectory
    containing motor CSVs:
    - Converts to SAGE real format in output_folder/real/<robot>/<motion_source>/<motion_name>/
    - Creates a temp motion file (bare CSV of commanded positions) for sim playback

    Auto-detects joint names from CSV headers, so works with any robot
    (single-joint teststand, H1 arm, etc.).

    Returns list of (motion_file_path, motion_name, is_temp) tuples.
    """
    import numpy as np

    from convert_h1_chirp_to_csv import (
        JOINT_MAP,
        find_motor_csvs,
        has_position_error,
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
            # Detect joints from header of first CSV
            joints = _detect_motor_csv_joints(os.path.join(dir_path, csvs[0]))
            if not joints:
                log_message(f"  SKIP {subdir_name}: no <joint>_position columns found")
                continue

            # Map short CSV column prefixes to full robot joint names (e.g. "elbow" → "right_elbow")
            robot_joints = [JOINT_MAP.get(j, j) for j in joints]

            # Per-file: each CSV becomes its own motion
            for fname in csvs:
                path = os.path.join(dir_path, fname)
                motion_name = fname.replace("_motor.csv", "")
                motion_real_dir = os.path.join(real_dir, motion_name)

                time_s, commands, positions, velocities, torques = _load_generic_motor_csv(path, joints)
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
                        cmd_pos = [float(commands[j][k]) for j in joints]
                        state_pos = [float(positions[j][k]) for j in joints]
                        state_vel = [float(velocities[j][k]) for j in joints]
                        state_torque = [float(torques[j][k]) for j in joints]
                        all_cmd_rows.append((time_s[k], cmd_pos))
                        all_state_rows.append((time_s[k], state_pos, state_vel, state_torque))

                    write_sage_output(motion_real_dir, all_cmd_rows, all_state_rows, joint_order=robot_joints)
                    log_message(f"  Converted {motion_name}: {len(time_s)} samples ({1/dt:.0f}Hz)")

                # Create temp motion file for sim playback (bare CSV of commanded positions)
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, prefix=f"motion_{motion_name}_"
                )
                tmp.write(",".join(robot_joints) + "\n")
                for k in range(len(time_s)):
                    cmd_pos = [float(commands[j][k]) for j in joints]
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


def _is_parquet_dir(path):
    """Check if path is a directory containing .parquet files."""
    if not os.path.isdir(path):
        return False
    return any(f.endswith(".parquet") for f in os.listdir(path))


def _convert_parquets_to_motor_csv(parquet_dir, joint_name="elbow"):
    """Auto-convert parquet files in a directory to motor CSVs (in-place).

    Returns the directory path (unchanged) after conversion.
    """
    from convert_benchtop_parquet import convert_parquet_to_motor_csv

    parquets = sorted(
        f for f in os.listdir(parquet_dir) if f.endswith(".parquet")
    )
    log_message(f"Auto-converting {len(parquets)} parquet files to motor CSVs")
    for fname in parquets:
        path = os.path.join(parquet_dir, fname)
        csv_path = convert_parquet_to_motor_csv(path, joint_name, output_dir=parquet_dir)
        log_message(f"  {fname} -> {os.path.basename(csv_path)}")
    return parquet_dir


def _find_real_state_csv_for_motion(motion_file, motion_name, args, bench_cfg):
    """Find the real state_motor.csv used for init-pose sync."""
    candidates = [
        os.path.join(args.output_folder, "real", args.robot_name, args.motion_source, motion_name, "state_motor.csv"),
    ]

    fallback_root = bench_cfg.get("real_data_root")
    if fallback_root:
        candidates.append(os.path.join(os.path.expanduser(fallback_root), motion_name, "state_motor.csv"))

    if args.motion_files:
        motion_root = os.path.expanduser(args.motion_files)
        candidates.extend(
            [
                os.path.join(motion_root, "state_motor.csv"),
                os.path.join(motion_root, motion_name, "state_motor.csv"),
            ]
        )

    if args.real_control_csv:
        candidates.append(os.path.join(os.path.dirname(os.path.expanduser(args.real_control_csv)), "state_motor.csv"))

    motion_dir = os.path.dirname(os.path.expanduser(motion_file))
    candidates.append(os.path.join(motion_dir, "state_motor.csv"))

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _load_first_real_state(state_csv):
    """Read the first STATE_MOTOR row from a SAGE state_motor.csv file."""
    with open(state_csv) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and row[0] == "STATE_MOTOR":
                positions = list(ast.literal_eval(row[2]))
                velocities = list(ast.literal_eval(row[3])) if len(row) > 3 else None
                return positions, velocities
    raise ValueError(f"No STATE_MOTOR row found in {state_csv}")


def _stage_sage_real_motion(source_control_csv, motion_name, args):
    """Stage a SAGE real motion folder into the benchmark output tree."""
    source_dir = os.path.dirname(os.path.abspath(os.path.expanduser(source_control_csv)))
    required = ("control.csv", "state_motor.csv", "joint_list.txt")
    if not all(os.path.isfile(os.path.join(source_dir, name)) for name in required):
        return

    dest_dir = os.path.join(args.output_folder, "real", args.robot_name, args.motion_source, motion_name)
    if os.path.abspath(dest_dir) == source_dir:
        return

    os.makedirs(dest_dir, exist_ok=True)
    for name in required:
        src = os.path.join(source_dir, name)
        dst = os.path.join(dest_dir, name)
        if os.path.exists(dst):
            continue
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    log_message(f"Staged SAGE real data for {motion_name}: {dest_dir}")


def _run_motions(benchmark, motions, temp_files):
    """Run a list of (motion_file, motion_name) pairs, batched or sequential.

    Uses batched parallel execution when there are multiple motions and the
    benchmark was created with matching num_envs. Falls back to sequential
    for single motions or on error.

    Args:
        benchmark: NewtonJointMotionBenchmark instance.
        motions: list of (motion_file, motion_name) tuples.
        temp_files: set of motion_file paths to delete after processing.
    """
    n = len(motions)

    if n > 1 and benchmark._num_envs == n:
        # Batched parallel execution
        log_message(f"################### RUNNING {n} MOTIONS IN PARALLEL ###################")
        try:
            benchmark.run_benchmark_batch(motions)
        except Exception as e:
            log_message(f"Batch execution failed: {e}")
            import traceback
            traceback.print_exc()
            log_message("Falling back to sequential execution...")
            for motion_file, motion_name in motions:
                try:
                    log_message(f"################### PROCESSING {motion_name} ###################")
                    benchmark.set_motion(motion_file, motion_name)
                    benchmark.run_benchmark()
                except Exception as e2:
                    log_message(f"Error processing {motion_name}: {e2}")
                    import traceback
                    traceback.print_exc()
    else:
        # Sequential execution
        for motion_file, motion_name in motions:
            try:
                log_message(f"################### PROCESSING {motion_name} ###################")
                benchmark.set_motion(motion_file, motion_name)
                benchmark.run_benchmark()
            except Exception as e:
                log_message(f"Error processing {motion_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    # Clean up temp files
    for motion_file, _ in motions:
        if motion_file in temp_files:
            os.unlink(motion_file)


def main():
    if args.real_control_csv and args.motion_files:
        raise ValueError("Specify either --motion-files or --real-control-csv, not both")
    if not args.real_control_csv and not args.motion_files:
        raise ValueError("Must specify either --motion-files or --real-control-csv")

    # --- Collect all motions first so we know how many envs to create ---
    motions = []  # list of (motion_file, motion_name)
    temp_files = set()  # motion files to clean up after

    if args.real_control_csv:
        motion_file, detected_freq = control_csv_to_motion_file(args.real_control_csv)

        if detected_freq is not None and args.original_control_freq is None:
            args.original_control_freq = detected_freq
            log_message(f"Auto-set --original-control-freq {detected_freq:.1f} from control.csv timestamps")

        motion_name = args.motion_name or get_motion_name(args.real_control_csv)
        motions.append((motion_file, motion_name))
        if motion_file != args.real_control_csv:
            temp_files.add(motion_file)
        _stage_sage_real_motion(args.real_control_csv, motion_name, args)

    elif _is_parquet_dir(args.motion_files) and not _is_motor_csv_dir(args.motion_files):
        _convert_parquets_to_motor_csv(args.motion_files)
        log_message(f"Detected motor CSVs in {args.motion_files} — auto-converting to SAGE format")
        prepared = prepare_motor_csv_data(args.motion_files, args.output_folder, args.robot_name, args.motion_source)
        if not prepared:
            raise ValueError(f"No convertible motor CSVs found in {args.motion_files}")
        for motion_file, motion_name, is_temp in prepared:
            motions.append((motion_file, motion_name))
            if is_temp:
                temp_files.add(motion_file)

    elif os.path.isfile(args.motion_files) and args.motion_files.endswith("_motor.csv"):
        # Single motor CSV file — treat its parent directory as the source dir
        log_message(f"Detected single motor CSV — auto-converting to SAGE format")
        prepared = prepare_motor_csv_data(
            os.path.dirname(args.motion_files), args.output_folder, args.robot_name, args.motion_source
        )
        basename = os.path.basename(args.motion_files)
        # Filter to only the requested file
        prepared = [(mf, mn, it) for mf, mn, it in prepared if mn == basename.replace("_motor.csv", "")]
        if not prepared:
            raise ValueError(f"No convertible motor CSV found: {args.motion_files}")
        for motion_file, motion_name, is_temp in prepared:
            motions.append((motion_file, motion_name))
            if is_temp:
                temp_files.add(motion_file)

    elif _is_motor_csv_dir(args.motion_files):
        log_message(f"Detected motor CSVs in {args.motion_files} — auto-converting to SAGE format")
        prepared = prepare_motor_csv_data(args.motion_files, args.output_folder, args.robot_name, args.motion_source)
        if not prepared:
            raise ValueError(f"No convertible motor CSVs found in {args.motion_files}")
        for motion_file, motion_name, is_temp in prepared:
            motions.append((motion_file, motion_name))
            if is_temp:
                temp_files.add(motion_file)

    else:
        motion_files_list = get_motion_files(args.motion_files)
        if not motion_files_list:
            raise ValueError(f"No motion files found in {args.motion_files}")
        for motion_file in motion_files_list:
            motion_name = get_motion_name(motion_file)
            motions.append((motion_file, motion_name))
            _stage_sage_real_motion(motion_file, motion_name, args)

    log_message(f"Prepared {len(motions)} motions for benchmark")

    args.real_init_pose = {}
    args.real_init_vel = {}
    if args.real_init_pose_sync:
        init_pose = {}
        init_vel = {}
        for motion_file, motion_name in motions:
            state_csv = _find_real_state_csv_for_motion(motion_file, motion_name, args, _bench_cfg)
            if not state_csv:
                log_message(f"WARNING: --real-init-pose-sync requested but state_motor.csv not found for {motion_name}")
                continue
            try:
                positions, velocities = _load_first_real_state(state_csv)
            except (ValueError, SyntaxError) as exc:
                log_message(f"WARNING: malformed init state in {state_csv}: {exc}")
                continue
            init_pose[motion_name] = positions
            if velocities is not None:
                init_vel[motion_name] = velocities
        args.real_init_pose = init_pose
        args.real_init_vel = init_vel
        log_message(
            f"--real-init-pose-sync: populated init pose for {len(init_pose)}/{len(motions)} motions "
            f"(init velocity for {len(init_vel)})"
        )

    # Create benchmark with num_envs matching motion count for parallel execution
    if args.num_envs is not None:
        num_envs = args.num_envs
    elif args.real_init_pose_sync:
        # Init-pose sync is a per-motion operation; default to one motion per
        # rollout so each motion starts from its own recorded row-0 state.
        num_envs = 1
    else:
        num_envs = len(motions) if len(motions) > 1 else 1
    benchmark = NewtonJointMotionBenchmark(args, num_envs=num_envs)

    _write_run_summary(args.output_folder, args.robot_name, args.motion_source, _run_cfg, args)

    _run_motions(benchmark, motions, temp_files)

    benchmark.log_joint_properties()


if __name__ == "__main__":
    main()
    simulation_app.close()
