# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert benchtop motor parquet files to *_motor.csv format.

Benchtop parquets have single-motor columns: time, position, velocity,
torque, commanded_position, position_error. This script renames them to
the multi-joint motor CSV convention (<joint>_position, etc.) expected
by the benchmark pipeline.

Usage:
    python convert_benchtop_parquet.py INPUT_DIR [--joint-name elbow] [--output-dir DIR]

INPUT_DIR can be a directory of .parquet files or a single .parquet file.
Output *_motor.csv files are written next to the parquets (or to --output-dir).
"""

import argparse
import os

import pandas as pd


def convert_parquet_to_motor_csv(parquet_path: str, joint_name: str, output_dir: str | None = None) -> str:
    """Convert a single benchtop parquet to motor CSV format.

    Args:
        parquet_path: Path to the parquet file.
        joint_name: Joint name prefix for columns (e.g. "elbow").
        output_dir: Output directory. Defaults to same directory as input.

    Returns:
        Path to the output CSV file.
    """
    df = pd.read_parquet(parquet_path)

    # Normalize timestamps: benchtop parquets use microseconds (large integers)
    time_col = df["time"].values
    if time_col[-1] > 1e6:
        time_s = time_col / 1e6
    else:
        time_s = time_col

    # Build motor CSV with joint-prefixed columns
    out = pd.DataFrame()
    out["time_s"] = time_s - time_s[0]
    out[f"{joint_name}_position"] = df["position"]
    if "position_error" in df.columns:
        out[f"{joint_name}_position_error"] = df["position_error"]
    elif "commanded_position" in df.columns:
        out[f"{joint_name}_position_error"] = df["commanded_position"] - df["position"]
    out[f"{joint_name}_velocity"] = df["velocity"]
    out[f"{joint_name}_torque"] = df["torque"]
    if "temperature" in df.columns:
        out[f"{joint_name}_temperature"] = df["temperature"]
    if "commanded_position" in df.columns:
        # Store as position + position_error for reconstruction
        pass

    # Derive output filename: foo_clean.parquet -> foo_motor.csv
    stem = os.path.splitext(os.path.basename(parquet_path))[0]
    # Strip common suffixes
    for suffix in ("_combined_clean", "_combined", "_clean"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    if output_dir is None:
        output_dir = os.path.dirname(parquet_path)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{stem}_motor.csv")
    out.to_csv(csv_path, index=False)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Convert benchtop parquet to motor CSV")
    parser.add_argument("input", help="Parquet file or directory of parquets")
    parser.add_argument("--joint-name", default="elbow", help="Joint name prefix (default: elbow)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        files = [args.input]
    else:
        files = sorted(
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.endswith(".parquet")
        )

    if not files:
        print(f"No parquet files found in {args.input}")
        return

    for path in files:
        csv_path = convert_parquet_to_motor_csv(path, args.joint_name, args.output_dir)
        print(f"  {os.path.basename(path)} -> {os.path.basename(csv_path)}")

    print(f"Converted {len(files)} files")


if __name__ == "__main__":
    main()
