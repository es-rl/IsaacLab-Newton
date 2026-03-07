"""Generate a sample motion file for H1 robot arm reach motion.

Creates a simple sinusoidal trajectory for the right arm joints,
suitable for testing the SAGE Newton benchmark. All 19 active joints
are included (ordered by real robot motor index).

Usage:
    python scripts/sim2real_gap/generate_sample_motion.py
"""

import math
import os

# H1 joints ordered by motor index (must match h1_valid_joints.txt)
# See JointIndex enum in unitree H1 SDK for reference.
# Motor index 9 (kNotUsedJoint) is excluded.
JOINTS = [
    # Right leg (indices 0-2)
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    # Left leg (indices 3-5)
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    # Torso (index 6)
    "torso",
    # Left/right hip yaw (indices 7-8)
    "left_hip_yaw",
    "right_hip_yaw",
    # Ankles (indices 10-11)
    "left_ankle",
    "right_ankle",
    # Right arm (indices 12-15)
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    # Left arm (indices 16-19)
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
]

# Motion parameters
DURATION = 4.0  # seconds
CONTROL_FREQ = 50  # Hz
NUM_FRAMES = int(DURATION * CONTROL_FREQ)


def generate_arm_reach():
    """Generate a simple right-arm reach motion.

    Right arm sweeps forward and back; legs, left arm, and torso stay at zero.
    """
    rows = []
    for i in range(NUM_FRAMES):
        t = i / CONTROL_FREQ
        phase = 2 * math.pi * t / DURATION  # one full cycle

        # Right arm: smooth sinusoidal reach
        r_shoulder_pitch = 0.5 * math.sin(phase)       # forward/back
        r_shoulder_roll = -0.2 * math.sin(phase)        # slight abduction
        r_shoulder_yaw = 0.3 * math.sin(phase * 0.5)   # slow twist
        r_elbow = -0.4 * (1 - math.cos(phase)) / 2     # bend then straighten

        row = [
            0.0,  # right_hip_roll
            0.0,  # right_hip_pitch
            0.0,  # right_knee
            0.0,  # left_hip_roll
            0.0,  # left_hip_pitch
            0.0,  # left_knee
            0.0,  # torso
            0.0,  # left_hip_yaw
            0.0,  # right_hip_yaw
            0.0,  # left_ankle
            0.0,  # right_ankle
            r_shoulder_pitch,
            r_shoulder_roll,
            r_shoulder_yaw,
            r_elbow,
            0.0,  # left_shoulder_pitch
            0.0,  # left_shoulder_roll
            0.0,  # left_shoulder_yaw
            0.0,  # left_elbow
        ]
        rows.append(row)

    return rows


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "input", "motion_files", "h1", "custom")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "arm_reach.txt")

    rows = generate_arm_reach()

    with open(output_file, "w") as f:
        # Header: joint names
        f.write(",".join(JOINTS) + "\n")
        # Data rows
        for row in rows:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")

    print(f"Generated {len(rows)} frames at {CONTROL_FREQ}Hz ({DURATION}s)")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
