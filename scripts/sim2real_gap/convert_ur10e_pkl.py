"""Convert UR10e trajectory pickle to SAGE-compatible format.

Reads a trajectory.pkl from the isaac_manipulator_data_utils recorder
and produces control.csv, state_motor.csv, joint_list.txt, and event.csv
in the SAGE directory layout.

Usage:
    python scripts/sim2real_gap/convert_ur10e_pkl.py \
        --pkl-path output/sim2real_benchmark_ur10e/real/2026-01-26_14-48-58_rosbag2/trajectory.pkl \
        --output-dir output/sim2real_benchmark_ur10e/real/ur10e/custom/rosbag_2026_01_26
"""

import argparse
import csv
import os
import pickle
import sys
import types

import numpy as np

# Canonical UR10e joint order (matches sim joint names, excludes finger_joint)
UR10E_JOINT_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _stub_pickle_modules():
    """Register stub modules so pickle can load isaac_manipulator_data_utils objects."""
    mod = types.ModuleType("isaac_manipulator_data_utils")
    structs = types.ModuleType("isaac_manipulator_data_utils.structures")
    sys.modules["isaac_manipulator_data_utils"] = mod
    sys.modules["isaac_manipulator_data_utils.structures"] = structs
    mod.structures = structs

    class JointState:
        pass

    class Timestamp:
        pass

    class RecordData:
        pass

    structs.JointState = JointState
    structs.Timestamp = Timestamp
    structs.RecordData = RecordData


def _ts_to_us(ts):
    """Convert Timestamp object to microseconds."""
    return ts.seconds * 1e6 + ts.nanoseconds / 1e3


def _reorder(values, src_names, dst_names):
    """Reorder values from src_names order to dst_names order."""
    name_to_idx = {n: i for i, n in enumerate(src_names)}
    return [values[name_to_idx[n]] for n in dst_names]


def convert(pkl_path, output_dir):
    _stub_pickle_modules()

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    joint_states = data["joint_state_from_robot"]
    commands = data["target_joint_commands"]

    print(f"Joint states: {len(joint_states)} samples")
    print(f"Commands: {len(commands)} samples")
    print(f"State joint names: {joint_states[0].names}")
    print(f"Command joint names: {commands[0].names}")

    # Build index maps
    state_names = joint_states[0].names
    cmd_names = commands[0].names

    # Verify all canonical joints exist in both
    for jn in UR10E_JOINT_ORDER:
        if jn not in state_names:
            raise ValueError(f"Joint '{jn}' not found in state names: {state_names}")
        if jn not in cmd_names:
            raise ValueError(f"Joint '{jn}' not found in command names: {cmd_names}")

    os.makedirs(output_dir, exist_ok=True)

    # --- joint_list.txt ---
    jl_path = os.path.join(output_dir, "joint_list.txt")
    with open(jl_path, "w") as f:
        for name in UR10E_JOINT_ORDER:
            f.write(name + "\n")

    # --- state_motor.csv ---
    state_path = os.path.join(output_dir, "state_motor.csv")
    with open(state_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timestamp", "positions", "velocities", "torques"])
        for js in joint_states:
            ts_us = _ts_to_us(js.timestamp)
            pos = _reorder(js.position, state_names, UR10E_JOINT_ORDER)
            vel = _reorder(js.velocity, state_names, UR10E_JOINT_ORDER)
            eff = _reorder(js.effort, state_names, UR10E_JOINT_ORDER)
            # Replace NaN efforts with 0.0
            eff = [0.0 if np.isnan(e) else e for e in eff]
            writer.writerow([
                "STATE_MOTOR",
                f"{ts_us:.1f}",
                str(pos),
                str(vel),
                str(eff),
            ])

    # --- control.csv ---
    ctrl_path = os.path.join(output_dir, "control.csv")
    with open(ctrl_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timestamp", "positions"])
        for cmd in commands:
            ts_us = _ts_to_us(cmd.timestamp)
            pos = _reorder(cmd.position, cmd_names, UR10E_JOINT_ORDER)
            writer.writerow(["CONTROL", f"{ts_us:.1f}", str(pos)])

    # --- event.csv ---
    # Relative timestamps: MOTION_START=0, DISABLE=duration
    state_t0 = _ts_to_us(joint_states[0].timestamp)
    state_t1 = _ts_to_us(joint_states[-1].timestamp)
    duration_us = state_t1 - state_t0

    event_path = os.path.join(output_dir, "event.csv")
    with open(event_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "timestamp", "event"])
        writer.writerow(["EVENT", "0.0", "MOTION_START"])
        writer.writerow(["EVENT", f"{duration_us:.1f}", "DISABLE"])

    # --- Summary ---
    state_dt = np.median(np.diff([
        _ts_to_us(js.timestamp) for js in joint_states[:200]
    ])) / 1e6
    cmd_dt = np.median(np.diff([
        _ts_to_us(cmd.timestamp) for cmd in commands[:50]
    ])) / 1e6

    print(f"\nOutput: {output_dir}")
    print(f"  state_motor.csv: {len(joint_states)} rows ({1/state_dt:.0f}Hz, {duration_us/1e6:.1f}s)")
    print(f"  control.csv:     {len(commands)} rows ({1/cmd_dt:.0f}Hz)")
    print(f"  joint_list.txt:  {len(UR10E_JOINT_ORDER)} joints")
    print(f"  event.csv:       MOTION_START=0, DISABLE={duration_us/1e6:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Convert UR10e trajectory pkl to SAGE format")
    parser.add_argument("--pkl-path", type=str, required=True, help="Path to trajectory.pkl")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for SAGE CSVs")
    args = parser.parse_args()

    convert(args.pkl_path, args.output_dir)


if __name__ == "__main__":
    main()
