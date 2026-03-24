# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate linear chirp motion files for system identification.

Produces frequency-sweep (chirp) trajectories that can be played back in
simulation via ``run_benchmark.py`` or deployed on the real robot.  The
output is a simple ``.txt`` motion file (header + position rows) compatible
with the benchmark loader, plus an optional motor-CSV version for sysid.

Linear chirp formula (PACE convention):
    phase(t) = 2 * pi * (f0 * t + (f1 - f0) / (2 * T) * t^2)
    signal(t) = amplitude * sin(phase(t))

Usage:
    # Single elbow chirp 0.1-5 Hz, 20 s, 0.5 rad amplitude
    python scripts/sysid/generate_chirp.py --joints elbow \
        --f0 0.1 --f1 5.0 --duration 20 --amplitude 0.5

    # All H1 arm joints, in-phase, at 500 Hz control
    python scripts/sysid/generate_chirp.py --joints all_arms \
        --f0 0.1 --f1 10.0 --duration 30 --control-freq 500

    # Per-joint amplitudes and biases
    python scripts/sysid/generate_chirp.py --joints elbow shoulder_pitch \
        --amplitude 0.5 0.3 --bias 0.8 -0.3

    # Teststand single motor
    python scripts/sysid/generate_chirp.py --robot teststand \
        --f0 0.1 --f1 5.0 --duration 20

    # UR10e elbow chirp (shorthand resolved to elbow_joint)
    python scripts/sysid/motion_generation/generate_chirp.py --robot ur10e \
        --joints elbow --f0 0.1 --f1 5.0

    # UR10e all joints simultaneously
    python scripts/sysid/motion_generation/generate_chirp.py --robot ur10e \
        --joints all --f0 0.1 --f1 3.0

    # UR10e all joints sequentially (default, one at a time)
    python scripts/sysid/motion_generation/generate_chirp.py --robot ur10e \
        --joints all --f0 0.1 --f1 5.0

    # UR10e all joints simultaneously (all at once)
    python scripts/sysid/motion_generation/generate_chirp.py --robot ur10e \
        --joints all --simultaneous --f0 0.1 --f1 5.0

    # Output as motor CSV (for sysid replay)
    python scripts/sysid/motion_generation/generate_chirp.py --joints elbow --output-csv
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))

# ---------------------------------------------------------------------------
# Robot joint definitions
# ---------------------------------------------------------------------------

# H1 joints ordered by motor index (matches h1_valid_joints.txt)
H1_ALL_JOINTS = [
    "right_hip_roll", "right_hip_pitch", "right_knee",
    "left_hip_roll", "left_hip_pitch", "left_knee",
    "torso",
    "left_hip_yaw", "right_hip_yaw",
    "left_ankle", "right_ankle",
    "right_shoulder_pitch", "right_shoulder_roll",
    "right_shoulder_yaw", "right_elbow",
    "left_shoulder_pitch", "left_shoulder_roll",
    "left_shoulder_yaw", "left_elbow",
]

# Shorthand joint groups
H1_ARM_JOINTS = [
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
]

TESTSTAND_JOINTS = ["elbow"]

UR10E_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# UR10e shorthand -> full joint name mapping
UR10E_SHORT_TO_FULL = {
    "shoulder_pan": "shoulder_pan_joint",
    "shoulder_lift": "shoulder_lift_joint",
    "elbow": "elbow_joint",
    "wrist_1": "wrist_1_joint",
    "wrist_2": "wrist_2_joint",
    "wrist_3": "wrist_3_joint",
}

# Sequential order: root first, end-effector last (safer for sysid —
# exciting root first keeps end-effector perturbations small)
H1_ARM_ROOT_FIRST = [
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
]
UR10E_ROOT_FIRST = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Default joint limits (approximate safe range in radians)
H1_ARM_LIMITS = {
    "shoulder_pitch": (-2.87, 2.87),
    "shoulder_roll": (-1.34, 2.12),
    "shoulder_yaw": (-1.30, 4.45),
    "elbow": (-1.25, 2.12),
}


# ---------------------------------------------------------------------------
# Chirp generation
# ---------------------------------------------------------------------------

def generate_linear_chirp(
    f0: float,
    f1: float,
    duration: float,
    control_freq: float,
    amplitude: float = 0.5,
    bias: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a linear chirp signal.

    Args:
        f0: Start frequency [Hz].
        f1: End frequency [Hz].
        duration: Total duration [s].
        control_freq: Sample rate [Hz].
        amplitude: Peak amplitude [rad].
        bias: DC offset [rad].

    Returns:
        Tuple of (time_array, signal_array).
    """
    num_steps = int(duration * control_freq)
    t = np.linspace(0, duration, num_steps, endpoint=False)

    # Linear chirp: instantaneous frequency = f0 + (f1 - f0) * t / T
    phase = 2 * math.pi * (f0 * t + (f1 - f0) / (2 * duration) * t ** 2)
    signal = amplitude * np.sin(phase) + bias

    return t, signal


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_motion_txt(
    filepath: str,
    joint_names: list[str],
    all_joints: list[str],
    trajectories: dict[str, np.ndarray],
    num_steps: int,
):
    """Write a simple .txt motion file (benchmark format).

    Args:
        filepath: Output file path.
        joint_names: Joint names to include in the chirp.
        all_joints: Full ordered joint list for the robot.
        trajectories: Dict of joint_name -> position array.
        num_steps: Number of timesteps.
    """
    with open(filepath, "w") as f:
        f.write(",".join(all_joints) + "\n")
        for i in range(num_steps):
            row = []
            for jname in all_joints:
                if jname in trajectories:
                    row.append(f"{trajectories[jname][i]:.6f}")
                else:
                    row.append("0.000000")
            f.write(",".join(row) + "\n")


def write_motor_csv(
    filepath: str,
    t: np.ndarray,
    joint_names: list[str],
    trajectories: dict[str, np.ndarray],
):
    """Write a motor CSV file (sysid format with commanded positions).

    The CSV has columns: time_s, {joint}_position, {joint}_position_error,
    {joint}_velocity, {joint}_torque for each joint.  For generated chirps,
    position = commanded position, position_error = 0, velocity = d(pos)/dt,
    torque = 0.

    Args:
        filepath: Output file path.
        t: Time array [s].
        joint_names: Joint names (short form, e.g. "elbow").
        trajectories: Dict of joint_name -> position array.
    """
    dt = t[1] - t[0] if len(t) > 1 else 1.0

    # Build header
    cols = ["time_s"]
    for jname in joint_names:
        cols.extend([
            f"{jname}_position",
            f"{jname}_position_error",
            f"{jname}_velocity",
            f"{jname}_torque",
        ])

    with open(filepath, "w") as f:
        f.write(",".join(cols) + "\n")
        for i in range(len(t)):
            row = [f"{t[i]:.9f}"]
            for jname in joint_names:
                pos = trajectories[jname][i]
                # Velocity via finite difference
                if i == 0:
                    vel = (trajectories[jname][1] - pos) / dt
                elif i == len(t) - 1:
                    vel = (pos - trajectories[jname][i - 1]) / dt
                else:
                    vel = (trajectories[jname][i + 1]
                           - trajectories[jname][i - 1]) / (2 * dt)
                row.extend([
                    f"{pos:.9f}",
                    "0.000000000",     # position_error = 0 for commanded
                    f"{vel:.9f}",
                    "0.000000000",     # torque = 0 (generated, not measured)
                ])
            f.write(",".join(row) + "\n")


def write_sage_output(
    output_dir: str,
    t: np.ndarray,
    joint_names: list[str],
    trajectories: dict[str, np.ndarray],
):
    """Write SAGE-format output (control.csv, state_motor.csv, etc.).

    This format is directly consumable by run_sysid.py and run_analysis.py.

    Args:
        output_dir: Output directory.
        t: Time array [s].
        joint_names: Ordered joint names.
        trajectories: Dict of joint_name -> position array.
    """
    os.makedirs(output_dir, exist_ok=True)
    dt = t[1] - t[0] if len(t) > 1 else 1.0

    # joint_list.txt
    with open(os.path.join(output_dir, "joint_list.txt"), "w") as f:
        for jname in joint_names:
            f.write(jname + "\n")

    # control.csv — commanded positions in microseconds
    with open(os.path.join(output_dir, "control.csv"), "w") as f:
        f.write("type,timestamp,positions\n")
        for i in range(len(t)):
            ts_us = t[i] * 1e6
            positions = [trajectories[jname][i] for jname in joint_names]
            pos_str = str(positions)
            f.write(f"CONTROL,{ts_us:.1f},\"{pos_str}\"\n")

    # state_motor.csv — "measured" state (same as commanded for generated)
    with open(os.path.join(output_dir, "state_motor.csv"), "w") as f:
        f.write("type,timestamp,positions,velocities,torques\n")
        for i in range(len(t)):
            ts_us = t[i] * 1e6
            positions = [trajectories[jname][i] for jname in joint_names]
            velocities = []
            for jname in joint_names:
                if i == 0:
                    v = (trajectories[jname][1]
                         - trajectories[jname][0]) / dt
                elif i == len(t) - 1:
                    v = (trajectories[jname][-1]
                         - trajectories[jname][-2]) / dt
                else:
                    v = (trajectories[jname][i + 1]
                         - trajectories[jname][i - 1]) / (2 * dt)
                velocities.append(v)
            torques = [0.0] * len(joint_names)
            f.write(
                f"STATE_MOTOR,{ts_us:.1f},"
                f"\"{positions}\","
                f"\"{velocities}\","
                f"\"{torques}\"\n"
            )

    # event.csv
    with open(os.path.join(output_dir, "event.csv"), "w") as f:
        f.write("type,timestamp,event\n")
        f.write("EVENT,0.0,MOTION_START\n")
        f.write(f"EVENT,{t[-1] * 1e6:.1f},DISABLE\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_chirps(
    t: np.ndarray,
    joint_names: list[str],
    trajectories: dict[str, np.ndarray],
    f0: float,
    f1: float,
    save_path: str | None = None,
):
    """Plot generated chirp trajectories."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plot")
        return

    n_joints = len(joint_names)
    fig, axes = plt.subplots(n_joints, 1, figsize=(14, 3 * n_joints),
                             sharex=True, squeeze=False)

    for idx, jname in enumerate(joint_names):
        ax = axes[idx, 0]
        ax.plot(t, trajectories[jname], linewidth=0.8)
        ax.set_ylabel(f"{jname}\n[rad]")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{jname} — chirp {f0}-{f1} Hz")

    axes[-1, 0].set_xlabel("Time [s]")
    fig.suptitle(
        f"Linear Chirp: {f0}-{f1} Hz, {t[-1]:.1f}s",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_full_names(
    jtype: str,
    robot: str,
    all_joints: list[str],
    mirror: bool,
) -> list[str]:
    """Map a chirp joint name to full joint name(s).

    For H1 with mirroring, returns both left_ and right_ variants.
    For other robots, returns the name as-is (already resolved).

    Args:
        jtype: Joint type / name (short for H1, full for UR10e).
        robot: Robot identifier.
        all_joints: Full ordered joint list.
        mirror: Whether to mirror left/right (H1 only).

    Returns:
        List of full joint names.
    """
    if robot == "h1":
        if mirror:
            names = []
            for prefix in ("right_", "left_"):
                full = prefix + jtype
                if full in all_joints:
                    names.append(full)
            return names
        else:
            for candidate in (f"right_{jtype}", jtype):
                if candidate in all_joints:
                    return [candidate]
            return [jtype]
    else:
        return [jtype]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate linear chirp motion files for sysid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single elbow chirp
  python scripts/sysid/generate_chirp.py --joints elbow

  # All H1 arm joints
  python scripts/sysid/generate_chirp.py --joints all_arms

  # Per-joint amplitudes
  python scripts/sysid/generate_chirp.py --joints elbow shoulder_pitch \\
      --amplitude 0.5 0.3 --bias 0.8 -0.3

  # Teststand
  python scripts/sysid/generate_chirp.py --robot teststand

  # Output as motor CSV + SAGE format
  python scripts/sysid/generate_chirp.py --joints elbow --output-csv --output-sage
""",
    )

    # Chirp parameters
    parser.add_argument(
        "--f0", type=float, default=0.1,
        help="Start frequency [Hz] (default: 0.1)",
    )
    parser.add_argument(
        "--f1", type=float, default=5.0,
        help="End frequency [Hz] (default: 5.0)",
    )
    parser.add_argument(
        "--duration", type=float, default=20.0,
        help="Chirp duration [s] (default: 20.0)",
    )
    parser.add_argument(
        "--amplitude", type=float, nargs="+", default=[0.5],
        help="Peak amplitude [rad] per joint (default: 0.5). "
             "Single value = same for all; multiple = per-joint.",
    )
    parser.add_argument(
        "--bias", type=float, nargs="+", default=[0.0],
        help="DC offset [rad] per joint (default: 0.0). "
             "Single value = same for all; multiple = per-joint.",
    )

    # Robot / joint selection
    parser.add_argument(
        "--robot", type=str, default="h1",
        choices=["h1", "teststand", "ur10e"],
        help="Robot name (default: h1)",
    )
    parser.add_argument(
        "--joints", type=str, nargs="+",
        default=["elbow"],
        help="Joints to excite. Use short names (elbow, shoulder_pitch) "
             "or 'all_arms' for all H1 arm joints, 'all' for all UR10e "
             "joints. For H1, both left and right are set (mirrored). "
             "UR10e accepts shorthands: elbow, shoulder_pan, shoulder_lift, "
             "wrist_1, wrist_2, wrist_3. (default: elbow)",
    )
    parser.add_argument(
        "--mirror", action="store_true", default=True,
        help="Mirror left/right for H1 (default: True)",
    )
    parser.add_argument(
        "--no-mirror", action="store_false", dest="mirror",
        help="Disable left/right mirroring",
    )

    # Output options
    parser.add_argument(
        "--control-freq", type=float, default=500.0,
        help="Control frequency [Hz] (default: 500)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: input/motion_files/<robot>/chirp/)",
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Output filename stem (default: auto-generated from params)",
    )
    parser.add_argument(
        "--output-csv", action="store_true",
        help="Also write motor CSV format (for sysid data input)",
    )
    parser.add_argument(
        "--output-sage", action="store_true",
        help="Also write SAGE format (control.csv + state_motor.csv)",
    )
    parser.add_argument(
        "--simultaneous", action="store_true",
        help="Chirp all joints at the same time. Default is "
             "sequential (one joint at a time, root first).",
    )
    parser.add_argument(
        "--rest-time", type=float, default=2.0,
        help="Rest time [s] between sequential joints "
             "(default: 2.0)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip generating the plot",
    )

    args = parser.parse_args()

    # Resolve robot joint configuration
    if args.robot == "h1":
        all_joints = H1_ALL_JOINTS
        if args.joints == ["all_arms"]:
            chirp_joints = list(H1_ARM_JOINTS)
        else:
            chirp_joints = list(args.joints)
    elif args.robot == "teststand":
        all_joints = TESTSTAND_JOINTS
        chirp_joints = ["elbow"]
        args.mirror = False
    elif args.robot == "ur10e":
        all_joints = UR10E_JOINTS
        if args.joints == ["all"]:
            chirp_joints = list(UR10E_JOINTS)
        else:
            # Resolve shorthands (e.g. "elbow" -> "elbow_joint")
            chirp_joints = [
                UR10E_SHORT_TO_FULL.get(j, j) for j in args.joints
            ]
            # Validate joint names
            for j in chirp_joints:
                if j not in all_joints:
                    print(
                        f"ERROR: unknown UR10e joint '{j}'. "
                        f"Valid: {all_joints} or shorthands: "
                        f"{list(UR10E_SHORT_TO_FULL.keys())}"
                    )
                    sys.exit(1)
        args.mirror = False
    else:
        print(f"Unknown robot: {args.robot}")
        sys.exit(1)

    # Expand amplitudes and biases to per-joint
    n_chirp = len(chirp_joints)
    if len(args.amplitude) == 1:
        amplitudes = args.amplitude * n_chirp
    elif len(args.amplitude) == n_chirp:
        amplitudes = args.amplitude
    else:
        print(
            f"ERROR: --amplitude expects 1 or {n_chirp} values, "
            f"got {len(args.amplitude)}"
        )
        sys.exit(1)

    if len(args.bias) == 1:
        biases = args.bias * n_chirp
    elif len(args.bias) == n_chirp:
        biases = args.bias
    else:
        print(
            f"ERROR: --bias expects 1 or {n_chirp} values, "
            f"got {len(args.bias)}"
        )
        sys.exit(1)

    # Reorder for sequential mode: root first, end-effector last
    if not args.simultaneous and len(chirp_joints) > 1:
        if args.robot == "ur10e":
            order = [
                j for j in UR10E_ROOT_FIRST if j in chirp_joints
            ]
            order += [j for j in chirp_joints if j not in order]
            chirp_joints = order
        elif args.robot == "h1":
            order = [
                j for j in H1_ARM_ROOT_FIRST if j in chirp_joints
            ]
            order += [j for j in chirp_joints if j not in order]
            chirp_joints = order

    # Generate chirps
    trajectories: dict[str, np.ndarray] = {}
    t = None

    if not args.simultaneous and len(chirp_joints) > 1:
        # Sequential: one joint at a time, others at bias
        rest_steps = int(args.rest_time * args.control_freq)
        chirp_steps = int(args.duration * args.control_freq)
        segment_steps = chirp_steps + rest_steps
        total_steps = segment_steps * len(chirp_joints)
        t = np.linspace(
            0, total_steps / args.control_freq, total_steps, endpoint=False,
        )

        # Initialize all trajectory arrays to bias
        for i, jtype in enumerate(chirp_joints):
            bias_val = biases[i]
            full_names = _resolve_full_names(
                jtype, args.robot, all_joints, args.mirror,
            )
            for fname in full_names:
                trajectories[fname] = np.full(total_steps, bias_val)

        # Fill in chirp segments one at a time
        for seg_idx, jtype in enumerate(chirp_joints):
            _, signal = generate_linear_chirp(
                f0=args.f0,
                f1=args.f1,
                duration=args.duration,
                control_freq=args.control_freq,
                amplitude=amplitudes[seg_idx],
                bias=biases[seg_idx],
            )
            start = seg_idx * segment_steps
            end = start + chirp_steps
            full_names = _resolve_full_names(
                jtype, args.robot, all_joints, args.mirror,
            )
            for fname in full_names:
                trajectories[fname][start:end] = signal
    else:
        # Simultaneous: all joints chirp at once
        for i, jtype in enumerate(chirp_joints):
            t_arr, signal = generate_linear_chirp(
                f0=args.f0,
                f1=args.f1,
                duration=args.duration,
                control_freq=args.control_freq,
                amplitude=amplitudes[i],
                bias=biases[i],
            )
            if t is None:
                t = t_arr
            full_names = _resolve_full_names(
                jtype, args.robot, all_joints, args.mirror,
            )
            for fname in full_names:
                trajectories[fname] = signal.copy()

    num_steps = len(t)

    # Auto-generate output name
    joints_str = "_".join(chirp_joints)
    if args.name:
        stem = args.name
    else:
        mode = "sim" if args.simultaneous else "seq"
        stem = (
            f"chirp_{joints_str}_f{args.f0}-{args.f1}hz"
            f"_a{amplitudes[0]}_d{args.duration:.0f}s_{mode}"
        )

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            REPO_ROOT, "input", "motion_files", args.robot, "chirp",
        )
    os.makedirs(output_dir, exist_ok=True)

    # Write motion .txt
    txt_path = os.path.join(output_dir, f"{stem}.txt")
    write_motion_txt(txt_path, chirp_joints, all_joints, trajectories,
                     num_steps)
    total_dur = num_steps / args.control_freq
    print(f"Motion file: {txt_path}")
    print(f"  {num_steps} frames at {args.control_freq} Hz "
          f"({total_dur:.1f}s)")
    print(f"  Chirp: {args.f0}-{args.f1} Hz")
    if not args.simultaneous and len(chirp_joints) > 1:
        print(f"  Mode: sequential (root first), "
              f"{args.duration:.0f}s/joint + "
              f"{args.rest_time:.0f}s rest")
        print(f"  Order: {' -> '.join(chirp_joints)}")
    for i, jtype in enumerate(chirp_joints):
        print(f"  {jtype}: amplitude={amplitudes[i]} rad, "
              f"bias={biases[i]} rad")

    # Write motor CSV
    if args.output_csv:
        csv_path = os.path.join(output_dir, f"{stem}_motor.csv")
        # Use short joint names for CSV columns
        write_motor_csv(csv_path, t, chirp_joints, {
            jtype: trajectories.get(
                f"right_{jtype}",
                trajectories.get(jtype, np.zeros(num_steps)),
            )
            for jtype in chirp_joints
        })
        print(f"Motor CSV:   {csv_path}")

    # Write SAGE format
    if args.output_sage:
        sage_dir = os.path.join(output_dir, stem)
        # Use full joint names for SAGE output
        sage_joints = [
            name for name in all_joints if name in trajectories
        ]
        write_sage_output(sage_dir, t, sage_joints, trajectories)
        print(f"SAGE output: {sage_dir}/")

    # Plot
    if not args.no_plot:
        plot_path = os.path.join(output_dir, f"{stem}.png")
        # Use full joint names that have trajectories for plotting
        plot_joints = [
            name for name in all_joints if name in trajectories
        ]
        plot_chirps(t, plot_joints, trajectories,
                    args.f0, args.f1, save_path=plot_path)


if __name__ == "__main__":
    main()
