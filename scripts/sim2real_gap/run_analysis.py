"""Run SAGE analysis comparing Newton sim data with real robot data.

This script wraps sage.analysis to compare CSV output from run_benchmark.py
(Newton simulation) against real robot data collected via SAGE's run_real.py.

Sim output folders include an actuator suffix (e.g. motion_stand_dcmotor),
but real data folders don't. This script auto-creates symlinks in real/ so
that SAGE's strict name matching works, then cleans them up after analysis.

Paths can be set in the run config (input/run_configs/<robot>.yaml) under the
benchmark and analysis sections, or overridden via CLI arguments.

Usage (paths from run config):
    python scripts/sim2real_gap/run_analysis.py --robot-name h1

Usage (CLI override):
    python scripts/sim2real_gap/run_analysis.py \
        --robot-name h1 \
        --result-folder output/sim2real_benchmark \
        --output-dir output/sim2real_analysis \
        --sample-dt 0.005
"""

import argparse
import filecmp
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sage.analysis as _sim2real_analysis
from sage.analysis import RobotDataComparator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "input"))
from actuator_models import load_actuator_params
from run_configs import load_run_cfg

# Known actuator suffixes (must match newton_benchmark.py suffix_map values)
ACTUATOR_SUFFIXES = (
    "_implicit", "_dcmotor",
    "_lstm_perjoint", "_lstm", "_gru_perjoint", "_gru",
    "_fmu", "_actuatornetfmu",
)

# Fallback robot name -> actuator YAML (used if run config has no actuator section)
_ACTUATOR_YAML_MAP = {
    "h1": "h1/h1_arm_implicit.yaml",
    "ur10e": "ur10e/ur10e_implicit.yaml",
}

# Parameters to display: (yaml_key, short_label, unit)
_PARAM_DISPLAY = [
    ("stiffness", "kp", "N-m/rad"),
    ("damping", "kd", "N-m-s/rad"),
    ("armature", "J", "kg-m^2"),
    ("dynamic_friction", "c", "N-m"),
    ("viscous_friction", "b", "N-m-s/rad"),
]

# Module-level storage: loaded once in main(), read by the plot method
_actuator_params = {}
_actuator_cfg = {}  # actuator section from run config (model_type, network_files, etc.)

# Local configs directory (ships with our project)
_LOCAL_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")

# SAGE's expected configs directory (relative to sage/analysis.py)
_SAGE_CONFIGS_DIR = str(Path(_sim2real_analysis.__file__).parent.parent / "configs")


def _load_robot_data_microsecond_safe(self, data_types):
    """Load SAGE robot data while treating benchmark timestamps consistently.

    Some benchmark CSVs are written with microsecond timestamps while upstream
    SAGE assumes seconds. Convert any large timestamp stream to seconds before
    computing the shared analysis window.
    """
    robot_data = {}
    initial_time = float("inf")

    for data_type in data_types:
        data = pd.read_csv(f"{self.file_path}/{data_type}.csv")
        if data["timestamp"].max() > 1000:
            data["timestamp"] = data["timestamp"] / 1e6

        initial_time = min(initial_time, data["timestamp"][0])
        robot_data[data_type] = data

    for data in robot_data.values():
        data["time_since_zero"] = data["timestamp"] - initial_time
        data["time_since_last"] = data["timestamp"].diff()

    dof_command = self._process_dof_data(robot_data["control"], self.joint_list, ["positions"], self.use_radians)
    dof_state = self._process_dof_data(
        robot_data["state_motor"], self.joint_list, ["positions", "velocities", "torques"], self.use_radians
    )

    return {"raw_data": robot_data, "dof_command": dof_command, "dof_state": dof_state}


_sim2real_analysis.RobotDataProcessor._load_robot_data = _load_robot_data_microsecond_safe


def _ensure_sage_joints_config(robot_name):
    """Copy <robot_name>_joints.yaml into SAGE's configs/.

    SAGE loads joint lists from its installed package directory. Refresh the
    installed copy when our local config changes so stale joint ordering cannot
    silently corrupt RMSE calculations.
    """
    sage_yaml = os.path.join(_SAGE_CONFIGS_DIR, f"{robot_name}_joints.yaml")
    local_yaml = os.path.join(_LOCAL_CONFIGS_DIR, f"{robot_name}_joints.yaml")
    if not os.path.isfile(local_yaml):
        print(f"[Analysis] WARNING: No joints config for '{robot_name}' in {_LOCAL_CONFIGS_DIR}")
        return

    if os.path.isfile(sage_yaml) and filecmp.cmp(local_yaml, sage_yaml, shallow=False):
        return

    os.makedirs(_SAGE_CONFIGS_DIR, exist_ok=True)
    action = "Updated" if os.path.isfile(sage_yaml) else "Installed"
    shutil.copy2(local_yaml, sage_yaml)
    print(f"[Analysis] {action} {robot_name}_joints.yaml -> {sage_yaml}")


def _resolve_joint_param(param_value, joint_name):
    """Get the value for a specific joint from a scalar or regex-keyed dict."""
    if isinstance(param_value, (int, float)):
        return param_value
    if isinstance(param_value, dict):
        if joint_name in param_value:
            return param_value[joint_name]
        for pattern, val in param_value.items():
            try:
                if re.fullmatch(pattern, joint_name):
                    return val
            except re.error:
                continue
    return None


def _build_params_text(joint_name):
    """Build a formatted parameter annotation string for a specific joint."""
    model_type = _actuator_cfg.get("model_type", "implicit")

    # Normalize gru -> lstm aliases
    _aliases = {"gru": "lstm", "gru_perjoint": "lstm_perjoint"}
    model_type = _aliases.get(model_type, model_type)

    # LSTM/GRU models: show the .pt file name instead of PD params
    if model_type in ("lstm", "lstm_perjoint"):
        network_file = _actuator_cfg.get("network_file")
        network_files = _actuator_cfg.get("network_files", {})

        if model_type == "lstm" and network_file:
            return f"model: {os.path.basename(network_file)}\n(all joints)"
        elif model_type == "lstm_perjoint" and network_files:
            # Find which .pt file this joint uses by matching joint_type suffix
            for jt, fname in network_files.items():
                if jt in joint_name:
                    return f"model: {os.path.basename(fname)}"
            # No match — show all mappings
            lines = []
            for jt, fname in network_files.items():
                lines.append(f"{jt}: {os.path.basename(fname)}")
            return "\n".join(lines) if lines else None
        return None

    # Implicit / DCMotor: show actuator YAML params
    if not _actuator_params:
        return None

    lines = []
    for yaml_key, label, unit in _PARAM_DISPLAY:
        raw = _actuator_params.get(yaml_key)
        if raw is None:
            continue
        val = _resolve_joint_param(raw, joint_name)
        if val is None:
            continue
        if abs(val) >= 100:
            lines.append(f"{label:<2s} = {val:<.2f}  {unit}")
        elif abs(val) >= 1:
            lines.append(f"{label:<2s} = {val:<.3f}  {unit}")
        else:
            lines.append(f"{label:<2s} = {val:<.4f}  {unit}")

    if not lines:
        return None
    return "\n".join(lines)


def _compute_rmse(sim_time, sim_values, real_time, real_values):
    """Compute RMSE between sim and real signals by interpolating to common time base."""
    t_start = max(sim_time.iloc[0], real_time.iloc[0])
    t_end = min(sim_time.iloc[-1], real_time.iloc[-1])
    if t_end <= t_start:
        return float("nan")
    common_t = np.linspace(t_start, t_end, min(len(sim_time), len(real_time)))
    sim_interp = np.interp(common_t, sim_time, sim_values)
    real_interp = np.interp(common_t, real_time, real_values)
    return np.sqrt(np.mean((sim_interp - real_interp) ** 2))


def _time_to_seconds(series):
    """Convert time_since_zero to seconds.

    SAGE timestamps are in microseconds (µs). If the max value exceeds
    a reasonable duration in seconds (>1000), assume µs and convert.
    """
    if series.max() > 1000:
        return series / 1e6
    return series


def _plot_comparison_data_sim_on_top(self, axes, sim_cmd, sim_state, real_cmd, real_state, joint_name, plot_titles):
    # Convert timestamps from µs to seconds for readable axes
    real_cmd_t = _time_to_seconds(real_cmd["time_since_zero"])
    real_state_t = _time_to_seconds(real_state["time_since_zero"])
    sim_cmd_t = _time_to_seconds(sim_cmd["time_since_zero"])
    sim_state_t = _time_to_seconds(sim_state["time_since_zero"])

    axes[0].plot(
        real_cmd_t, real_cmd[f"positions_{joint_name}"], "#ffcccb", label="Real Command Position"
    )
    axes[0].plot(
        real_state_t, real_state[f"positions_{joint_name}"], "r--", label="Real Position"
    )
    axes[0].plot(
        sim_cmd_t, sim_cmd[f"positions_{joint_name}"], "#add8e6", label="Sim Command Position"
    )
    axes[0].plot(
        sim_state_t, sim_state[f"positions_{joint_name}"], "b--", label="Sim Position"
    )
    pos_rmse = _compute_rmse(
        sim_state_t, sim_state[f"positions_{joint_name}"],
        real_state_t, real_state[f"positions_{joint_name}"],
    )
    axes[0].set_title(f"{plot_titles[0]}  (RMSE: {pos_rmse:.4f} rad)")
    axes[0].legend()
    axes[0].grid(True)

    # Velocity subplot
    axes[1].plot(
        real_state_t, real_state[f"velocities_{joint_name}"], "r--", label="Real Velocity"
    )
    axes[1].plot(
        sim_state_t, sim_state[f"velocities_{joint_name}"], "b--", label="Sim Velocity"
    )
    vel_rmse = _compute_rmse(
        sim_state_t, sim_state[f"velocities_{joint_name}"],
        real_state_t, real_state[f"velocities_{joint_name}"],
    )
    axes[1].set_title(f"{plot_titles[1]}  (RMSE: {vel_rmse:.4f} rad/s)")
    axes[1].legend()
    axes[1].grid(True)

    # Torque subplot
    axes[2].plot(
        real_state_t, real_state[f"torques_{joint_name}"], "r--", label="Real Torque"
    )
    axes[2].plot(
        sim_state_t, sim_state[f"torques_{joint_name}"], "b--", label="Sim Torque"
    )
    torque_rmse = _compute_rmse(
        sim_state_t, sim_state[f"torques_{joint_name}"],
        real_state_t, real_state[f"torques_{joint_name}"],
    )
    axes[2].set_title(f"{plot_titles[2]}  (RMSE: {torque_rmse:.4f} Nm)")
    axes[2].legend()
    axes[2].grid(True)

    # Annotate with actuator parameters from YAML
    params_text = _build_params_text(joint_name)
    if params_text:
        axes[0].text(
            0.01, 0.97, params_text,
            transform=axes[0].transAxes,
            fontsize=8,
            fontfamily="monospace",
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.85),
        )


RobotDataComparator.plot_comparison_data = _plot_comparison_data_sim_on_top


def _strip_actuator_suffix(name):
    """Strip known actuator suffix from a motion name.

    e.g. 'motion_stand_left_arm_swing_dcmotor' -> 'motion_stand_left_arm_swing'
    """
    for suffix in ACTUATOR_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _create_real_symlinks(result_folder, robot_name, motion_source):
    """Create symlinks in real/ so SAGE's strict name matching works.

    Handles two sim output layouts:

    1. **Subfolder layout** (new): sim/.../arm_march2026/implicit/EC01_elbow_chirp/
       Real data is at real/.../arm_march2026/EC01_elbow_chirp/ (no actuator subfolder).
       Creates: real/.../arm_march2026/implicit/ -> .  (symlink to parent)

    2. **Suffix layout** (legacy): sim/.../arm_march2026/EC01_elbow_chirp_implicit/
       Real data is at real/.../arm_march2026/EC01_elbow_chirp/.
       Creates: real/.../EC01_elbow_chirp_implicit -> EC01_elbow_chirp

    Returns list of created symlink paths (for cleanup).
    """
    sim_dir = os.path.join(result_folder, "sim", robot_name, motion_source)
    real_dir = os.path.join(result_folder, "real", robot_name, motion_source)

    if not os.path.isdir(sim_dir):
        return []

    created_links = []

    # Check for subfolder layout: the last component of motion_source is an
    # actuator name, and real data lives one level up without that subfolder.
    parts = motion_source.split("/")
    if len(parts) >= 2:
        actuator_part = parts[-1]
        # Strip the actuator part to get the base motion_source
        base_motion_source = "/".join(parts[:-1])
        real_parent = os.path.join(result_folder, "real", robot_name, base_motion_source)
        # If real/ has the base dir but not the actuator subdir, symlink it
        if os.path.isdir(real_parent) and not os.path.exists(real_dir):
            os.symlink(".", real_dir)
            print(f"[Analysis] Linked real/{robot_name}/{motion_source} -> . (actuator subfolder)")
            created_links.append(real_dir)
            return created_links

    # Legacy suffix layout: sim motion folders have actuator suffix appended
    if not os.path.isdir(real_dir):
        return []

    sim_motions = [d for d in os.listdir(sim_dir) if os.path.isdir(os.path.join(sim_dir, d))]
    for sim_name in sim_motions:
        base_name = _strip_actuator_suffix(sim_name)
        if base_name is None:
            continue

        real_target = os.path.join(real_dir, base_name)
        real_link = os.path.join(real_dir, sim_name)

        if os.path.isdir(real_target) and not os.path.exists(real_link):
            os.symlink(base_name, real_link)
            print(f"[Analysis] Linked real/{sim_name} -> {base_name}")
            created_links.append(real_link)

    return created_links


def _cleanup_symlinks(links):
    """Remove symlinks created by _create_real_symlinks."""
    for link in links:
        if os.path.islink(link):
            os.unlink(link)


def _rename_output_with_suffixes(output_dir, result_folder, robot_name, motion_source):
    """Rename SAGE's output folders to include actuator suffixes.

    SAGE follows symlinks when resolving real data paths, so its output uses
    the base motion name (e.g. motion_stand_left_arm_swing) instead of the
    suffixed sim name (motion_stand_left_arm_swing_lstm). This renames the
    output folders to match the sim folder names.
    """
    sim_dir = os.path.join(result_folder, "sim", robot_name, motion_source)
    if not os.path.isdir(sim_dir):
        return

    # Build mapping: base_name -> suffixed_name from sim/ folders
    base_to_suffixed = {}
    for sim_name in os.listdir(sim_dir):
        if not os.path.isdir(os.path.join(sim_dir, sim_name)):
            continue
        base = _strip_actuator_suffix(sim_name)
        if base is not None:
            base_to_suffixed[base] = sim_name

    if not base_to_suffixed:
        return

    # Rename in each output subdirectory (metrics/, etc.)
    for subdir in os.listdir(output_dir):
        motion_dir = os.path.join(output_dir, subdir, robot_name, motion_source)
        if not os.path.isdir(motion_dir):
            continue
        for folder in os.listdir(motion_dir):
            if folder in base_to_suffixed:
                old_path = os.path.join(motion_dir, folder)
                new_path = os.path.join(motion_dir, base_to_suffixed[folder])
                if os.path.isdir(old_path) and not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"[Analysis] Renamed output {folder} -> {base_to_suffixed[folder]}")


def _is_motion_folder(path):
    """Check if a directory is a SAGE motion folder (has control.csv)."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "control.csv"))


def _discover_motion_sources(result_folder, robot_name, base_motion_source):
    """Discover motion sources, handling flat, nested, and mixed directory structures.

    Flat:     sim/h1/custom/EC01_elbow_chirp_implicit/control.csv
    Nested:   sim/h1/custom/elbow/EC01_elbow_chirp_implicit/control.csv
    Actuator: sim/h1/custom/implicit/EC01_elbow_chirp/control.csv
    Mixed:    combinations of the above

    Returns list of motion_source strings, e.g.:
      flat:     ["custom"]
      nested:   ["custom/elbow", "custom/shoulder_pitch_Config A", ...]
      actuator: ["custom/implicit"]
      mixed:    ["custom", "custom/shoulder_pitch_Config A", ...]
    """
    sim_dir = os.path.join(result_folder, "sim", robot_name, base_motion_source)
    if not os.path.isdir(sim_dir):
        return [base_motion_source]

    has_flat_motions = False
    nested_groups = []

    for entry in sorted(os.listdir(sim_dir)):
        entry_path = os.path.join(sim_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if _is_motion_folder(entry_path):
            has_flat_motions = True
        else:
            # Check if it's a group dir containing motion folders
            for sub in os.listdir(entry_path):
                if _is_motion_folder(os.path.join(entry_path, sub)):
                    nested_groups.append(entry)
                    break

    sources = []
    if has_flat_motions:
        sources.append(base_motion_source)
    for g in nested_groups:
        sources.append(f"{base_motion_source}/{g}")
    return sources if sources else [base_motion_source]


def _group_has_matching_motions(result_folder, robot_name, motion_source, motion_names_arg):
    """Check if a group has any sim motions matching the motion_names pattern.

    Used to skip groups that have no relevant motions (e.g. shoulder_yaw group
    when motion_names="*elbow*").
    """
    import fnmatch

    sim_dir = os.path.join(result_folder, "sim", robot_name, motion_source)
    if not os.path.isdir(sim_dir):
        return False

    sim_motions = [
        d for d in os.listdir(sim_dir)
        if _is_motion_folder(os.path.join(sim_dir, d))
    ]
    if not sim_motions:
        return False

    patterns = [p.strip() for p in motion_names_arg.split(",")]
    for sm in sim_motions:
        base = _strip_actuator_suffix(sm)
        for pat in patterns:
            if "*" in pat or "?" in pat:
                if fnmatch.fnmatch(sm, pat) or (base and fnmatch.fnmatch(base, pat)):
                    return True
            else:
                if sm == pat or base == pat:
                    return True
                # Check with suffix appended
                if any(f"{pat}{s}" == sm for s in ACTUATOR_SUFFIXES):
                    return True
    return False


def _resolve_motion_names(result_folder, robot_name, motion_source, motion_names_arg):
    """Resolve user-supplied motion names to actual sim folder names.

    If the user passes a base name like 'motion_stand_left_arm_swing' but the sim
    folder is 'motion_stand_left_arm_swing_dcmotor', resolve it automatically.
    Handles comma-separated names. '*' resolves to all sim folder names (avoids
    SAGE seeing unsuffixed real folders that have no sim counterpart).

    Only includes directories that are actual motion folders (contain control.csv).

    Returns the resolved motion_names string (comma-separated or '*').
    """
    sim_dir = os.path.join(result_folder, "sim", robot_name, motion_source)
    if not os.path.isdir(sim_dir):
        return motion_names_arg

    # Only consider directories that are actual motion folders (have control.csv)
    sim_motions = sorted(
        d for d in os.listdir(sim_dir)
        if _is_motion_folder(os.path.join(sim_dir, d))
    )

    # For '*', enumerate sim folders explicitly so SAGE doesn't discover
    # unsuffixed real folders that have no sim counterpart.
    if motion_names_arg == "*":
        if sim_motions:
            return ",".join(sim_motions)
        return "*"

    sim_motions_set = set(sim_motions)
    requested = [n.strip() for n in motion_names_arg.split(",")]
    resolved = []

    for name in requested:
        # Glob/wildcard pattern (e.g. "*elbow*", "EC*")
        if "*" in name or "?" in name:
            import fnmatch
            # Match against sim motions (with suffix) and base names (without suffix)
            for sm in sim_motions:
                if fnmatch.fnmatch(sm, name):
                    resolved.append(sm)
                else:
                    base = _strip_actuator_suffix(sm)
                    if base and fnmatch.fnmatch(base, name):
                        resolved.append(sm)
            continue

        if name in sim_motions_set:
            # Exact match — use as-is
            resolved.append(name)
        else:
            # Try appending known suffixes to find a match
            matches = [f"{name}{s}" for s in ACTUATOR_SUFFIXES if f"{name}{s}" in sim_motions_set]
            if matches:
                resolved.extend(matches)
                for m in matches:
                    print(f"[Analysis] Resolved motion '{name}' -> '{m}'")
            else:
                # Pass through and let SAGE report the error
                resolved.append(name)

    if resolved:
        print(f"[Analysis] Matched {len(resolved)} motions: {', '.join(resolved)}")
    return ",".join(resolved)


def main():
    parser = argparse.ArgumentParser(description="SAGE analysis (Newton sim vs real robot)")
    parser.add_argument("--robot-name", type=str, default="h1", help="Robot name")
    parser.add_argument(
        "--motion-names", type=str, default="*",
        help="Motion names to analyze (comma-separated, or '*' for all). "
        "Base names without actuator suffix are auto-resolved.",
    )
    parser.add_argument("--valid-joints-file", type=str, default=None, help="Path to valid joints file")
    parser.add_argument("--result-folder", type=str, default=None, help="Root folder with sim/ and real/ subdirs (default: from run config)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for analysis results (default: from run config)")
    parser.add_argument("--sample-dt", type=float, default=None, help="Sample timestep for comparison (seconds)")
    parser.add_argument("--metrics-file", type=str, default="metrics_summary.xlsx", help="Metrics output filename")
    args = parser.parse_args()

    # Load run config; CLI args override config values
    _rcfg = load_run_cfg(args.robot_name)
    _bench_cfg = _rcfg.get("benchmark", {})
    _analysis_cfg = _rcfg.get("analysis", {})

    # motion_source from config's motion_name (matches benchmark output folder)
    args.motion_source = _bench_cfg.get("motion_name", "custom")

    # None-defaulted args: override only if CLI left them as None
    if args.motion_names == "*":
        cfg_names = _analysis_cfg.get("motion_names")
        if cfg_names:
            # Support both string ("EC01,EC02") and list ([EC01, EC02]) in YAML
            if isinstance(cfg_names, list):
                args.motion_names = ",".join(str(n) for n in cfg_names)
            else:
                args.motion_names = str(cfg_names)
    if args.result_folder is None:
        args.result_folder = _bench_cfg.get("output_folder")  # benchmark output = analysis input
    if args.output_dir is None:
        args.output_dir = _analysis_cfg.get("output_dir")
    if args.sample_dt is None:
        args.sample_dt = _analysis_cfg.get("sample_dt", 0.005)

    # Validate required args after config merge
    if not args.result_folder:
        parser.error("--result-folder is required (set via CLI or run config benchmark.output_folder)")
    if not args.output_dir:
        parser.error("--output-dir is required (set via CLI or run config analysis.output_dir)")

    # Ensure SAGE has the robot's joints config YAML installed
    _ensure_sage_joints_config(args.robot_name)

    # Load actuator config for plot annotations
    global _actuator_params, _actuator_cfg
    _actuator_cfg = _rcfg.get("actuator", {})
    model_type = _actuator_cfg.get("model_type", "implicit")

    # Normalize aliases
    _mt = {"gru": "lstm", "gru_perjoint": "lstm_perjoint"}.get(model_type, model_type)

    if _mt in ("lstm", "lstm_perjoint"):
        # LSTM/GRU: plot annotations will show .pt file names, not PD params
        if _mt == "lstm":
            print(f"[Analysis] Actuator model: LSTM/GRU — {_actuator_cfg.get('network_file', '?')}")
        else:
            nf = _actuator_cfg.get("network_files", {})
            print(f"[Analysis] Actuator model: LSTM/GRU per-joint — {nf}")
    else:
        # Implicit / DCMotor: load actuator YAML for PD param annotations
        actuator_yaml = (_actuator_cfg.get("yaml_file")
                         or _ACTUATOR_YAML_MAP.get(args.robot_name))
        if actuator_yaml:
            try:
                _actuator_params = load_actuator_params(actuator_yaml)
                print(f"[Analysis] Loaded actuator params from {actuator_yaml}")
            except FileNotFoundError:
                print(f"[Analysis] WARNING: Actuator YAML not found: {actuator_yaml}")
        else:
            print(f"[Analysis] No actuator YAML configured for robot '{args.robot_name}'")

    # Discover motion sources — handles both flat and nested directory structures.
    # Flat:   sim/h1/custom/{motion_name}/control.csv    -> motion_source = "custom"
    # Nested: sim/h1/custom/elbow/{motion_name}/control.csv -> motion_source = "custom/elbow"
    motion_sources = _discover_motion_sources(
        args.result_folder, args.robot_name, args.motion_source
    )
    # Filter groups: skip any group where no sim motions match motion_names.
    # e.g. motion_names="*elbow*" will skip shoulder_yaw groups that have no elbow motions.
    if args.motion_names != "*" and len(motion_sources) > 1:
        filtered = []
        for ms in motion_sources:
            if _group_has_matching_motions(args.result_folder, args.robot_name, ms, args.motion_names):
                filtered.append(ms)
        if filtered:
            skipped = len(motion_sources) - len(filtered)
            motion_sources = filtered
            if skipped:
                print(f"[Analysis] motion_names filter '{args.motion_names}': kept {len(filtered)} groups, skipped {skipped}")

    if len(motion_sources) > 1:
        print(f"[Analysis] Detected nested structure with {len(motion_sources)} groups:")
        for ms in motion_sources:
            print(f"  {ms}")

    all_created_links = []

    for motion_source in motion_sources:
        if len(motion_sources) > 1:
            group_label = motion_source.replace(f"{args.motion_source}/", "")
            print(f"\n{'='*60}")
            print(f"[Analysis] Processing group: {group_label}")
            print(f"{'='*60}")

        # Resolve base motion names to suffixed sim folder names
        resolved_names = _resolve_motion_names(
            args.result_folder, args.robot_name, motion_source, args.motion_names
        )

        # Ensure sim directory exists (benchmark should have created it already)
        sim_path = os.path.join(args.result_folder, "sim", args.robot_name, motion_source)
        os.makedirs(sim_path, exist_ok=True)

        # Create symlinks so SAGE finds real data through actuator subfolders.
        # Must run BEFORE creating real dirs, otherwise makedirs blocks symlink creation.
        created_links = _create_real_symlinks(args.result_folder, args.robot_name, motion_source)

        # Ensure real directory exists (only if symlink wasn't created above)
        real_path = os.path.join(args.result_folder, "real", args.robot_name, motion_source)
        if not os.path.exists(real_path):
            os.makedirs(real_path, exist_ok=True)
        all_created_links.extend(created_links)

        # Snapshot existing output folders before analysis so we only report new ones
        plots_base = os.path.join(args.output_dir, "metrics", args.robot_name, motion_source)
        existing_folders = set()
        if os.path.isdir(plots_base):
            existing_folders = set(
                d for d in os.listdir(plots_base) if os.path.isdir(os.path.join(plots_base, d))
            )

        try:
            comparator = RobotDataComparator(
                robot_name=args.robot_name,
                motion_source=motion_source,
                motion_names=resolved_names,
                valid_joints_file=args.valid_joints_file,
                result_folder=args.result_folder,
                sample_dt=args.sample_dt,
            )

            comparator.analyze_all_data(args.output_dir, args.metrics_file)
            comparator.visualize_all_comparisons(args.output_dir)

            # Rename output folders to include actuator suffix (SAGE uses base names)
            _rename_output_with_suffixes(args.output_dir, args.result_folder, args.robot_name, motion_source)

            # Print output locations — only show folders created by this run
            metrics_path = os.path.join(args.output_dir, args.metrics_file)
            print(f"\n[RESULTS] Metrics: {os.path.abspath(metrics_path)}")
            if os.path.isdir(plots_base):
                for motion_dir in sorted(os.listdir(plots_base)):
                    motion_path = os.path.join(plots_base, motion_dir)
                    if os.path.isdir(motion_path) and motion_dir not in existing_folders:
                        num_plots = len([f for f in os.listdir(motion_path) if f.endswith(".png")])
                        print(f"[RESULTS] Plots:   {os.path.abspath(motion_path)}/ ({num_plots} plots)")
        except Exception as e:
            print(f"[Analysis] Error processing {motion_source}: {e}")
            import traceback
            traceback.print_exc()
            continue

    _cleanup_symlinks(all_created_links)


if __name__ == "__main__":
    main()
