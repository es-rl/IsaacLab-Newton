"""Convert H1 right-arm motor CSVs to SAGE/sysid format.

Reads motor CSV files from the H1 right arm sysid experiments and outputs
control.csv + state_motor.csv + joint_list.txt + event.csv in SAGE format.

Supports two input formats:

1. **With position_error columns** (arm_march2026 per-joint recordings):
   Commanded position = position + position_error.  For joints without a
   position_error column, the actual position is used as the command.

2. **Without position_error** (chirp/sine experiments):
   Commanded position is reconstructed from experiment parameters encoded
   in the filename.

Filename format (chirp/sine):
  C{id}_f{f_start}-{f_end}_a{amp}[_{pattern}]_motor.csv   (chirp)
  S{id}_f{freq}_a{amp}_{pattern}_motor.csv                 (sine)

Phase patterns:
  inphase (default for chirps): all joints at 0 deg
  antiphase: elbow at 0, others at 180 deg
  wave: elbow 0, pitch 90, raise 180, yaw 270 deg

Usage (new format with position_error):
    python scripts/sysid/convert_h1_chirp_to_csv.py \\
        --data-dir input/sysid_data/h1/arm_march2026/elbow \\
        --output-dir output/sim2real_benchmark/real/h1/custom/elbow_chirps

Usage (old chirp/sine format):
    python scripts/sysid/convert_h1_chirp_to_csv.py \\
        --data-dir input/sysid_data/h1/right_arm \\
        --output-dir input/sysid_data/h1/right_arm_csv \\
        --ramp-time 8.0

Usage (recursive — convert all subdirs under a data root):
    python scripts/sysid/convert_h1_chirp_to_csv.py \\
        --data-dir input/sysid_data/h1/arm_march2026 \\
        --output-dir output/sim2real_benchmark/real/h1/custom \\
        --recursive
"""

import argparse
import csv
import os
import re
import sys

import numpy as np

# Joint name mapping: CSV column prefix → H1 sim joint name
JOINT_MAP = {
    "pitch": "right_shoulder_pitch",
    "raise": "right_shoulder_roll",
    "yaw": "right_shoulder_yaw",
    "elbow": "right_elbow",
}

# Canonical order for output (matches H1_ARM_JOINT_NAMES right arm subset)
CANONICAL_ORDER = ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"]
CSV_ORDER = ["pitch", "raise", "yaw", "elbow"]

# Phase offsets per pattern (radians) for each joint in CSV_ORDER
PHASE_OFFSETS = {
    "inphase": [0.0, 0.0, 0.0, 0.0],
    "antiphase": [np.pi, np.pi, np.pi, 0.0],  # elbow=0, others=180
    "wave": [np.pi / 2, np.pi, 3 * np.pi / 2, 0.0],  # elbow=0, pitch=90, raise=180, yaw=270
}


def parse_filename(filename):
    """Parse experiment parameters from filename.

    Returns dict with keys: type, f_start, f_end (chirp) or freq (sine),
    amplitude, pattern.
    """
    base = filename.replace("_motor.csv", "")
    params = {}

    if base.startswith("C"):
        params["type"] = "chirp"
        # C{id}_f{start}-{end}_a{amp}[_{pattern}]
        m = re.match(r"C\d+_f(\d+_\d+)-(\d+_\d+)_a(\d+_\d+)(?:_(\w+))?", base)
        if not m:
            raise ValueError(f"Cannot parse chirp filename: {filename}")
        params["f_start"] = float(m.group(1).replace("_", "."))
        params["f_end"] = float(m.group(2).replace("_", "."))
        params["amplitude"] = float(m.group(3).replace("_", "."))
        params["pattern"] = m.group(4) if m.group(4) else "inphase"

    elif base.startswith("S"):
        params["type"] = "sine"
        # S{id}_f{freq}_a{amp}_{pattern}
        m = re.match(r"S\d+_f(\d+_\d+)_a(\d+_\d+)_(\w+)", base)
        if not m:
            raise ValueError(f"Cannot parse sine filename: {filename}")
        params["freq"] = float(m.group(1).replace("_", "."))
        params["amplitude"] = float(m.group(2).replace("_", "."))
        params["pattern"] = m.group(3)

    else:
        raise ValueError(f"Unknown experiment type: {filename}")

    return params


def generate_chirp(t, f_start, f_end, duration):
    """Generate linear chirp signal in [0, 1] range (use sin)."""
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration))
    return np.sin(phase)


def generate_sine(t, freq):
    """Generate sine signal."""
    return np.sin(2 * np.pi * freq * t)


def load_csv(path):
    """Load motor CSV and return time + joint position/velocity/torque arrays."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    time_s = np.array([float(r["time_s"]) for r in rows])

    positions = {}
    velocities = {}
    torques = {}
    for joint in CSV_ORDER:
        positions[joint] = np.array([float(r[f"{joint}_position"]) for r in rows])
        velocities[joint] = np.array([float(r[f"{joint}_velocity"]) for r in rows])
        torques[joint] = np.array([float(r[f"{joint}_torque"]) for r in rows])

    return time_s, positions, velocities, torques


def has_position_error(path):
    """Check if a motor CSV has any position_error columns."""
    with open(path) as f:
        header = f.readline().strip()
    return "position_error" in header


def load_csv_with_commands(path):
    """Load motor CSV with position_error columns.

    For joints with {joint}_position_error, commanded = position + position_error.
    For joints without, commanded = actual position (passive hold).

    Returns time_s, commands, positions, velocities, torques (all dicts by CSV_ORDER keys).
    """
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    header_keys = set(rows[0].keys()) if rows else set()
    time_s = np.array([float(r["time_s"]) for r in rows])

    positions = {}
    velocities = {}
    torques = {}
    commands = {}

    for joint in CSV_ORDER:
        positions[joint] = np.array([float(r[f"{joint}_position"]) for r in rows])
        velocities[joint] = np.array([float(r[f"{joint}_velocity"]) for r in rows])
        torques[joint] = np.array([float(r[f"{joint}_torque"]) for r in rows])

        if f"{joint}_position_error" in header_keys:
            # position_error = target - actual, so target = actual + error
            pos_err = np.array([float(r[f"{joint}_position_error"]) for r in rows])
            commands[joint] = positions[joint] + pos_err
        else:
            # No error column — use actual position as command (passive hold)
            commands[joint] = positions[joint].copy()

    return time_s, commands, positions, velocities, torques


def estimate_centers(time_s, positions, ramp_time):
    """Estimate the oscillation center for each joint from the steady-state portion."""
    mask = time_s >= ramp_time
    centers = {}
    for joint in CSV_ORDER:
        centers[joint] = float(np.mean(positions[joint][mask]))
    return centers


def reconstruct_commands(time_s, centers, params, ramp_time, initial_positions):
    """Reconstruct commanded positions from experiment parameters.

    Returns dict of joint→command_array for the full trajectory.
    During ramp: linear interpolation from initial position to center.
    During oscillation: center + amplitude * signal with phase offset.
    """
    amplitude = params["amplitude"]
    pattern = params["pattern"]
    phase_offsets = PHASE_OFFSETS.get(pattern, PHASE_OFFSETS["inphase"])

    commands = {}
    ramp_mask = time_s < ramp_time
    osc_mask = ~ramp_mask

    # Time relative to oscillation start
    t_osc = time_s[osc_mask] - ramp_time
    osc_duration = t_osc[-1] - t_osc[0] if len(t_osc) > 1 else 1.0

    for i, joint in enumerate(CSV_ORDER):
        cmd = np.zeros_like(time_s)

        # Ramp: linear from initial to center
        if ramp_mask.any():
            t_ramp = time_s[ramp_mask]
            ramp_frac = (t_ramp - t_ramp[0]) / (ramp_time - t_ramp[0]) if ramp_time > t_ramp[0] else np.ones_like(t_ramp)
            cmd[ramp_mask] = initial_positions[joint] + ramp_frac * (centers[joint] - initial_positions[joint])

        # Oscillation: center + amplitude * signal
        if params["type"] == "chirp":
            signal = generate_chirp(t_osc, params["f_start"], params["f_end"], osc_duration)
        else:
            signal = generate_sine(t_osc, params["freq"])

        # Apply phase offset (rotate the signal)
        if phase_offsets[i] != 0:
            if params["type"] == "chirp":
                phase = 2 * np.pi * (
                    params["f_start"] * t_osc
                    + (params["f_end"] - params["f_start"]) * t_osc**2 / (2 * osc_duration)
                )
                signal = np.sin(phase + phase_offsets[i])
            else:
                signal = np.sin(2 * np.pi * params["freq"] * t_osc + phase_offsets[i])

        cmd[osc_mask] = centers[joint] + amplitude * signal
        commands[joint] = cmd

    return commands


def write_sage_output(output_dir, all_cmd_rows, all_state_rows, joint_order=None):
    """Write SAGE-format CSVs to output_dir.

    Args:
        output_dir: Directory to write control.csv, state_motor.csv, joint_list.txt, event.csv.
        all_cmd_rows: List of (time_s, [positions...]) tuples.
        all_state_rows: List of (time_s, [positions...], [velocities...], [torques...]) tuples.
        joint_order: Joint names for joint_list.txt (default: CANONICAL_ORDER).
    """
    if joint_order is None:
        joint_order = CANONICAL_ORDER
    os.makedirs(output_dir, exist_ok=True)

    # control.csv
    ctrl_path = os.path.join(output_dir, "control.csv")
    with open(ctrl_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timestamp", "positions"])
        for t, pos in all_cmd_rows:
            ts_us = t * 1e6
            writer.writerow(["CONTROL", f"{ts_us:.1f}", str(pos)])

    # state_motor.csv
    state_path = os.path.join(output_dir, "state_motor.csv")
    with open(state_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timestamp", "positions", "velocities", "torques"])
        for t, pos, vel, torque in all_state_rows:
            ts_us = t * 1e6
            writer.writerow(["STATE_MOTOR", f"{ts_us:.1f}", str(pos), str(vel), str(torque)])

    # joint_list.txt
    joint_list_path = os.path.join(output_dir, "joint_list.txt")
    with open(joint_list_path, "w") as f:
        for name in joint_order:
            f.write(name + "\n")

    # event.csv
    if all_state_rows:
        first_t = all_state_rows[0][0]
        last_t = all_state_rows[-1][0]
        duration_us = (last_t - first_t) * 1e6
    else:
        duration_us = 0.0
    event_path = os.path.join(output_dir, "event.csv")
    with open(event_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timestamp", "event"])
        writer.writerow(["EVENT", "0.0", "MOTION_START"])
        writer.writerow(["EVENT", f"{duration_us:.1f}", "DISABLE"])

    total_dur = all_state_rows[-1][0] if all_state_rows else 0
    print(f"  Output: {output_dir}")
    print(f"    control.csv:     {len(all_cmd_rows)} rows ({total_dur:.1f}s)")
    print(f"    state_motor.csv: {len(all_state_rows)} rows ({total_dur:.1f}s)")
    print(f"    joint_list.txt:  {len(joint_order)} joints")
    print(f"    event.csv:       MOTION_START=0, DISABLE={duration_us:.0f}us")


def convert_direct(data_dir, output_dir, max_files=None):
    """Convert motor CSVs with position_error columns to SAGE format.

    Each CSV becomes a separate motion folder under output_dir, named after
    the source file (without _motor.csv suffix).

    Returns list of (motion_name, output_path) tuples.
    """
    files = sorted(f for f in os.listdir(data_dir) if f.endswith("_motor.csv"))
    if not files:
        return []

    if max_files is not None:
        files = files[:max_files]

    print(f"Converting {len(files)} files (direct mode) from {data_dir}")
    results = []

    for fname in files:
        path = os.path.join(data_dir, fname)
        motion_name = fname.replace("_motor.csv", "")
        motion_dir = os.path.join(output_dir, motion_name)

        time_s, commands, positions, velocities, torques = load_csv_with_commands(path)

        # Zero-base the time
        t0 = time_s[0]
        time_s = time_s - t0
        dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 0.005
        duration = time_s[-1]

        print(f"  {fname}: {len(time_s)} samples ({1/dt:.0f}Hz), {duration:.1f}s")

        all_cmd_rows = []
        all_state_rows = []
        for k in range(len(time_s)):
            cmd_pos = [float(commands[j][k]) for j in CSV_ORDER]
            state_pos = [float(positions[j][k]) for j in CSV_ORDER]
            state_vel = [float(velocities[j][k]) for j in CSV_ORDER]
            state_torque = [float(torques[j][k]) for j in CSV_ORDER]
            all_cmd_rows.append((time_s[k], cmd_pos))
            all_state_rows.append((time_s[k], state_pos, state_vel, state_torque))

        write_sage_output(motion_dir, all_cmd_rows, all_state_rows)
        results.append((motion_name, motion_dir))

    return results


def convert_direct_concat(data_dir, output_dir, max_files=None):
    """Convert motor CSVs with position_error columns to a single SAGE output.

    All CSVs are concatenated into one trajectory (for sysid). Unlike
    convert_direct() which creates one folder per CSV.
    """
    files = sorted(f for f in os.listdir(data_dir) if f.endswith("_motor.csv"))
    if not files:
        print(f"No *_motor.csv files found in {data_dir}")
        return

    if max_files is not None:
        files = files[:max_files]

    print(f"Converting {len(files)} files (direct concat mode) from {data_dir}")
    os.makedirs(output_dir, exist_ok=True)

    all_cmd_rows = []
    all_state_rows = []
    time_offset = 0.0

    for fname in files:
        path = os.path.join(data_dir, fname)
        time_s, commands, positions, velocities, torques = load_csv_with_commands(path)

        t0 = time_s[0]
        duration = time_s[-1] - t0
        dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else 0.005

        print(f"  {fname}: {len(time_s)} samples ({1/dt:.0f}Hz), {duration:.1f}s")

        for k in range(len(time_s)):
            t_abs = time_s[k] - t0 + time_offset
            cmd_pos = [float(commands[j][k]) for j in CSV_ORDER]
            state_pos = [float(positions[j][k]) for j in CSV_ORDER]
            state_vel = [float(velocities[j][k]) for j in CSV_ORDER]
            state_torque = [float(torques[j][k]) for j in CSV_ORDER]
            all_cmd_rows.append((t_abs, cmd_pos))
            all_state_rows.append((t_abs, state_pos, state_vel, state_torque))

        time_offset += duration + 0.1

    write_sage_output(output_dir, all_cmd_rows, all_state_rows)


def find_motor_csvs(data_dir, recursive=False):
    """Find all *_motor.csv files in a directory (optionally recursive).

    Returns list of (subdir_name, dir_path) tuples, where subdir_name is
    the subdirectory name (or data_dir basename if flat).
    """
    results = []

    # Check for motor CSVs directly in data_dir
    direct_files = [f for f in os.listdir(data_dir) if f.endswith("_motor.csv")]
    if direct_files:
        results.append((os.path.basename(data_dir), data_dir))

    if recursive:
        for root, dirs, files in os.walk(data_dir):
            if root == data_dir and direct_files:
                continue  # Already handled
            motor_files = [f for f in files if f.endswith("_motor.csv")]
            if motor_files:
                rel = os.path.relpath(root, data_dir)
                # Use path components joined with underscore as the name
                name = rel.replace(os.sep, "_")
                results.append((name, root))

    return results


def convert(data_dir, output_dir, ramp_time, max_files=None, skip_ramp=True):
    """Convert all motor CSVs in data_dir to sysid format (chirp/sine filename reconstruction)."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith("_motor.csv"))

    if not files:
        print(f"No *_motor.csv files found in {data_dir}")
        sys.exit(1)

    if max_files is not None:
        files = files[:max_files]

    print(f"Converting {len(files)} files from {data_dir}")
    os.makedirs(output_dir, exist_ok=True)

    all_cmd_rows = []
    all_state_rows = []
    time_offset = 0.0

    for fname in files:
        path = os.path.join(data_dir, fname)
        try:
            params = parse_filename(fname)
        except ValueError as e:
            print(f"  SKIP {fname}: {e}")
            continue

        time_s, positions, velocities, torques = load_csv(path)
        initial_positions = {j: positions[j][0] for j in CSV_ORDER}
        centers = estimate_centers(time_s, positions, ramp_time)
        commands = reconstruct_commands(time_s, centers, params, ramp_time, initial_positions)

        # Optionally skip the ramp portion
        if skip_ramp:
            mask = time_s >= ramp_time
            time_s = time_s[mask]
            for j in CSV_ORDER:
                positions[j] = positions[j][mask]
                velocities[j] = velocities[j][mask]
                torques[j] = torques[j][mask]
                commands[j] = commands[j][mask]

        t0 = time_s[0]
        duration = time_s[-1] - t0
        dt = float(np.median(np.diff(time_s)))

        desc = f"{params['type']}"
        if params["type"] == "chirp":
            desc += f" {params['f_start']}-{params['f_end']}Hz"
        else:
            desc += f" {params['freq']}Hz"
        desc += f" a={params['amplitude']} {params['pattern']}"

        print(f"  {fname}: {len(time_s)} samples ({1/dt:.0f}Hz), {duration:.1f}s — {desc}")

        for k in range(len(time_s)):
            t_abs = time_s[k] - t0 + time_offset

            # Command row: positions in canonical order
            cmd_pos = [commands[j][k] for j in CSV_ORDER]
            all_cmd_rows.append((t_abs, cmd_pos))

            # State row: positions, velocities, torques in canonical order
            state_pos = [positions[j][k] for j in CSV_ORDER]
            state_vel = [velocities[j][k] for j in CSV_ORDER]
            state_torque = [torques[j][k] for j in CSV_ORDER]
            all_state_rows.append((t_abs, state_pos, state_vel, state_torque))

        time_offset += duration + 0.1  # small gap between files

    write_sage_output(output_dir, all_cmd_rows, all_state_rows)


def main():
    parser = argparse.ArgumentParser(description="Convert H1 right-arm motor CSVs to SAGE/sysid format")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with *_motor.csv files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for SAGE files")
    parser.add_argument("--ramp-time", type=float, default=8.0,
                        help="Ramp duration to skip in chirp/sine mode (seconds, default: 8.0)")
    parser.add_argument("--max-files", type=int, default=None, help="Limit to first N files per subdir (default: all)")
    parser.add_argument("--include-ramp", action="store_true", help="Include the ramp phase (chirp/sine mode only)")
    parser.add_argument("--recursive", action="store_true",
                        help="Walk subdirectories recursively. Each subdir with motor CSVs "
                        "becomes a separate motion folder under output-dir.")
    parser.add_argument("--per-file", action="store_true",
                        help="Create a separate SAGE output folder per CSV file (default: concatenate all into one)")
    parser.add_argument("--mode", choices=["auto", "direct", "chirp"], default="auto",
                        help="Conversion mode: 'auto' detects position_error columns, "
                        "'direct' uses position_error, 'chirp' reconstructs from filename (default: auto)")
    args = parser.parse_args()

    if args.recursive:
        subdirs = find_motor_csvs(args.data_dir, recursive=True)
        if not subdirs:
            print(f"No *_motor.csv files found under {args.data_dir}")
            sys.exit(1)
        print(f"Found {len(subdirs)} directories with motor CSVs\n")
        for name, dir_path in subdirs:
            print(f"--- {name} ({dir_path}) ---")
            out = os.path.join(args.output_dir, name)
            _convert_one_dir(dir_path, out, args)
    else:
        _convert_one_dir(args.data_dir, args.output_dir, args)


def _convert_one_dir(data_dir, output_dir, args):
    """Convert a single directory based on mode and flags."""
    # Auto-detect: check first CSV for position_error columns
    mode = args.mode
    if mode == "auto":
        csvs = sorted(f for f in os.listdir(data_dir) if f.endswith("_motor.csv"))
        if csvs and has_position_error(os.path.join(data_dir, csvs[0])):
            mode = "direct"
        else:
            mode = "chirp"

    if mode == "direct":
        if args.per_file:
            convert_direct(data_dir, output_dir, args.max_files)
        else:
            convert_direct_concat(data_dir, output_dir, args.max_files)
    else:
        convert(data_dir, output_dir, args.ramp_time, args.max_files,
                skip_ramp=not args.include_ramp)


if __name__ == "__main__":
    main()
