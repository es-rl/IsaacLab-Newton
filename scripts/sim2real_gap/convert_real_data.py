"""Convert real robot data to SAGE-compatible format.

Takes raw real robot data (state_motor.csv with seconds timestamps,
control.csv with bare CSV values) and converts them to the format
expected by sage.analysis.RobotDataComparator:

- Timestamps in microseconds (SAGE divides by 1e6 internally)
- control.csv: type,timestamp,positions  (CONTROL rows with list strings)
- state_motor.csv: type,timestamp,positions,velocities,torques  (unchanged structure, timestamps scaled)
- event.csv: type,timestamp,event  (MOTION_START and DISABLE markers)

Usage:
    python scripts/sim2real_gap/convert_real_data.py \
        --data-dir output/sim2real_benchmark_h1_left_arm_swing/real/h1/custom/motion_stand_left_arm_swing
"""

import argparse
import csv
import os
import shutil

import numpy as np


def read_state_motor(filepath):
    """Read state_motor.csv and return header + rows with timestamps."""
    rows = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)
    return header, rows


def read_control_bare(filepath):
    """Read control.csv in bare CSV format (joint header + values)."""
    rows = []
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        joint_header = next(reader)  # joint names header
        for row in reader:
            rows.append([float(v) for v in row])
    return joint_header, rows


def convert_state_motor(input_path, output_path):
    """Convert state_motor.csv timestamps from seconds to microseconds.

    Returns (first_timestamp_s, last_timestamp_s) in original seconds.
    """
    header, rows = read_state_motor(input_path)

    if len(rows) == 0:
        raise ValueError("state_motor.csv has no data rows")

    first_ts = float(rows[0][1])
    last_ts = float(rows[-1][1])

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            ts_seconds = float(row[1])
            ts_microseconds = ts_seconds * 1e6
            writer.writerow([row[0], f"{ts_microseconds:.1f}"] + row[2:])

    print(f"  state_motor.csv: {len(rows)} rows, timestamps {first_ts:.4f}s - {last_ts:.4f}s -> microseconds")
    return first_ts, last_ts


def convert_control(input_path, output_path, first_ts, last_ts):
    """Convert control.csv from bare CSV to SAGE format.

    Bare format:  joint_header + rows of comma-separated values (no timestamps)
    SAGE format:  type,timestamp,positions  with CONTROL rows and list strings
    """
    joint_header, rows = read_control_bare(input_path)

    if len(rows) == 0:
        raise ValueError("control.csv has no data rows")

    # Distribute control timestamps evenly across the state_motor time range
    timestamps_s = np.linspace(first_ts, last_ts, len(rows))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timestamp", "positions"])
        for i, row in enumerate(rows):
            ts_us = timestamps_s[i] * 1e6
            positions_str = "[" + ", ".join(f"{v}" for v in row) + "]"
            writer.writerow(["CONTROL", f"{ts_us:.1f}", positions_str])

    dt_s = (last_ts - first_ts) / max(len(rows) - 1, 1)
    print(f"  control.csv: {len(rows)} rows, dt={dt_s*1000:.2f}ms ({1/dt_s:.0f}Hz), timestamps in microseconds")


def create_event(output_path, first_ts, last_ts):
    """Create event.csv with MOTION_START and DISABLE timing markers.

    Timestamps must be RELATIVE (0-based, in microseconds) because SAGE
    computes time_since_zero = timestamp - initial_time (making data 0-based),
    then subtracts event timestamps from time_since_zero in adjust_real_data_timing().
    Using absolute timestamps here would cause a double-subtraction that discards
    the first N seconds of real data.
    """
    duration_us = (last_ts - first_ts) * 1e6

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timestamp", "event"])
        writer.writerow(["EVENT", "0.0", "MOTION_START"])
        writer.writerow(["EVENT", f"{duration_us:.1f}", "DISABLE"])

    print(f"  event.csv: MOTION_START=0.0us, DISABLE={duration_us:.1f}us (duration={last_ts - first_ts:.4f}s)")


def main():
    parser = argparse.ArgumentParser(description="Convert real robot data to SAGE format")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to real data directory")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating .bak backups of original files")
    args = parser.parse_args()

    data_dir = args.data_dir

    state_motor_path = os.path.join(data_dir, "state_motor.csv")
    control_path = os.path.join(data_dir, "control.csv")
    event_path = os.path.join(data_dir, "event.csv")

    # Validate inputs exist
    if not os.path.exists(state_motor_path):
        raise FileNotFoundError(f"state_motor.csv not found in {data_dir}")
    if not os.path.exists(control_path):
        raise FileNotFoundError(f"control.csv not found in {data_dir}")

    # Backup originals
    if not args.no_backup:
        for path in [state_motor_path, control_path]:
            if os.path.exists(path):
                backup = path + ".bak"
                shutil.copy2(path, backup)
                print(f"  Backed up: {os.path.basename(path)} -> {os.path.basename(backup)}")

    print(f"\nConverting real data in: {data_dir}")

    # Step 1: Convert state_motor.csv (timestamps seconds -> microseconds)
    first_ts, last_ts = convert_state_motor(state_motor_path, state_motor_path)

    # Step 2: Convert control.csv (bare CSV -> SAGE format with microsecond timestamps)
    convert_control(control_path, control_path, first_ts, last_ts)

    # Step 3: Create event.csv (timing markers)
    create_event(event_path, first_ts, last_ts)

    # Verify joint_list.txt exists
    joint_list_path = os.path.join(data_dir, "joint_list.txt")
    if os.path.exists(joint_list_path):
        with open(joint_list_path) as f:
            joints = [line.strip() for line in f if line.strip()]
        print(f"  joint_list.txt: {len(joints)} joints (OK)")
    else:
        print(f"  WARNING: joint_list.txt not found in {data_dir}")

    print(f"\nDone. Files ready for SAGE analysis.")


if __name__ == "__main__":
    main()
