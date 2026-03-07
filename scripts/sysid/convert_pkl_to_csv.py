"""Convert UR10e trajectory.pkl files to SAGE-compatible CSVs for sysid.

Reads trajectory.pkl files produced by isaac_manipulator_real_to_sim and
outputs control.csv, state_motor.csv, and joint_list.txt in the format
expected by run_sysid.py --real-data-dir.

Each .pkl contains:
  - joint_state_from_robot: ~495 Hz, 7 joints (6 arm + finger_joint)
  - target_joint_commands:  ~15 Hz, 6 arm joints

Usage:
    python scripts/sysid/convert_pkl_to_csv.py \
        --data-dir input/sysid_data/ur10e \
        --output-dir input/sysid_data/ur10e_csv \
        --episodes 5
"""

import argparse
import csv
import os
import pickle
import sys

# Canonical joint order (matches UR10_CFG / Isaac Lab convention)
CANONICAL_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class _StubUnpickler(pickle.Unpickler):
    """Unpickler that stubs out isaac_manipulator_data_utils classes.

    The .pkl files reference JointState, Timestamp, etc. from
    isaac_manipulator_data_utils which pulls in heavy dependencies (av, etc.).
    We create lightweight stub classes so pickle can reconstruct the objects.
    """

    def find_class(self, module, name):
        if "isaac_manipulator" in module or "sensor_msgs" in module or "builtin_interfaces" in module:
            return type(name, (), {})
        return super().find_class(module, name)


def _ts_seconds(timestamp) -> float:
    """Convert a Timestamp object (seconds + nanoseconds) to float seconds."""
    return timestamp.seconds + timestamp.nanoseconds / 1e9


def _reorder(names, values, canonical):
    """Reorder values from arbitrary joint order to canonical order.

    Returns list of values in canonical order.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    return [values[name_to_idx[c]] for c in canonical]


def load_episode(pkl_path):
    """Load a single trajectory.pkl file.

    Returns:
        state_rows: list of (time_s, positions, velocities, efforts) tuples
        cmd_rows:   list of (time_s, positions) tuples
        Both use canonical joint order with finger_joint filtered out.
    """
    with open(pkl_path, "rb") as f:
        data = _StubUnpickler(f).load()

    states = data["joint_state_from_robot"]
    cmds = data["target_joint_commands"]

    # State data: reorder and filter finger_joint
    state_rows = []
    state_names = list(states[0].names)
    for s in states:
        t = _ts_seconds(s.timestamp)
        pos = _reorder(state_names, list(s.position), CANONICAL_JOINTS)
        vel = _reorder(state_names, list(s.velocity), CANONICAL_JOINTS)
        eff = _reorder(state_names, list(s.effort), CANONICAL_JOINTS)
        state_rows.append((t, pos, vel, eff))

    # Command data: reorder to canonical
    cmd_rows = []
    cmd_names = list(cmds[0].names)
    for c in cmds:
        t = _ts_seconds(c.timestamp)
        pos = _reorder(cmd_names, list(c.position), CANONICAL_JOINTS)
        cmd_rows.append((t, pos))

    return state_rows, cmd_rows


def convert(data_dir, output_dir, max_episodes=None):
    """Convert all episodes in data_dir to CSV output."""
    # Find episode directories (sorted by name = chronological)
    episodes = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, d, "trajectory.pkl"))
    )

    if not episodes:
        print(f"No trajectory.pkl files found in {data_dir}")
        sys.exit(1)

    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    print(f"Converting {len(episodes)} episodes from {data_dir}")

    os.makedirs(output_dir, exist_ok=True)

    all_state_rows = []
    all_cmd_rows = []
    time_offset = 0.0

    for ep_name in episodes:
        pkl_path = os.path.join(data_dir, ep_name, "trajectory.pkl")
        state_rows, cmd_rows = load_episode(pkl_path)

        if not state_rows or not cmd_rows:
            print(f"  Skipping empty episode: {ep_name}")
            continue

        # Make timestamps relative to episode start, then add offset
        ep_t0 = state_rows[0][0]
        ep_duration = state_rows[-1][0] - ep_t0

        for t, pos, vel, eff in state_rows:
            all_state_rows.append((t - ep_t0 + time_offset, pos, vel, eff))

        for t, pos in cmd_rows:
            all_cmd_rows.append((t - ep_t0 + time_offset, pos))

        state_freq = len(state_rows) / ep_duration if ep_duration > 0 else 0
        cmd_freq = len(cmd_rows) / ep_duration if ep_duration > 0 else 0
        print(f"  {ep_name}: {len(state_rows)} states ({state_freq:.0f}Hz), "
              f"{len(cmd_rows)} cmds ({cmd_freq:.0f}Hz), {ep_duration:.1f}s")

        time_offset += ep_duration + 0.1  # small gap between episodes

    # Convert timestamps to microseconds (SAGE format)
    # Write state_motor.csv
    state_path = os.path.join(output_dir, "state_motor.csv")
    with open(state_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timestamp", "positions", "velocities", "torques"])
        for t, pos, vel, eff in all_state_rows:
            ts_us = t * 1e6
            writer.writerow([
                "STATE_MOTOR",
                f"{ts_us:.1f}",
                str(pos),
                str(vel),
                str(eff),
            ])

    # Write control.csv
    ctrl_path = os.path.join(output_dir, "control.csv")
    with open(ctrl_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timestamp", "positions"])
        for t, pos in all_cmd_rows:
            ts_us = t * 1e6
            writer.writerow([
                "CONTROL",
                f"{ts_us:.1f}",
                str(pos),
            ])

    # Write joint_list.txt
    joint_list_path = os.path.join(output_dir, "joint_list.txt")
    with open(joint_list_path, "w") as f:
        for name in CANONICAL_JOINTS:
            f.write(name + "\n")

    total_state_dur = all_state_rows[-1][0] if all_state_rows else 0
    total_cmd_dur = all_cmd_rows[-1][0] if all_cmd_rows else 0
    print(f"\nOutput: {output_dir}")
    print(f"  state_motor.csv: {len(all_state_rows)} rows ({total_state_dur:.1f}s)")
    print(f"  control.csv:     {len(all_cmd_rows)} rows ({total_cmd_dur:.1f}s)")
    print(f"  joint_list.txt:  {len(CANONICAL_JOINTS)} joints")


def main():
    parser = argparse.ArgumentParser(description="Convert UR10e trajectory.pkl files to sysid CSVs")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with episode folders containing trajectory.pkl")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for CSV files")
    parser.add_argument("--episodes", type=int, default=None, help="Limit to first N episodes (default: all)")
    args = parser.parse_args()

    convert(args.data_dir, args.output_dir, args.episodes)


if __name__ == "__main__":
    main()
