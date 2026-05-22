#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay G1 benchmark traces together in one viewer.

This is a visualization-only tool. It reads existing SAGE-format benchmark
outputs and kinematically replays three fixed-base G1 copies:

- measured real trace
- baseline sim trace
- model sim trace

It does not run actuator physics and does not compute benchmark metrics.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay real/baseline/model G1 benchmark traces together.")
parser.add_argument("--benchmark-root", default=None)
parser.add_argument("--motion", default="T_A_01_wave_sine")
parser.add_argument("--slice", choices=["arm", "leg"], default="arm")
parser.add_argument("--baseline-robot", default=None)
parser.add_argument("--model-robot", default=None)
parser.add_argument("--baseline-actuator", default=None)
parser.add_argument("--model-actuator", default=None)
parser.add_argument("--motion-source", default="custom")
parser.add_argument("--layout", choices=["overlay", "spread"], default="overlay")
parser.add_argument(
    "--spacing",
    type=float,
    default=0.45,
    help="Y spacing for --layout spread. Overlay layout ignores this.",
)
parser.add_argument("--render-hz", type=float, default=60.0)
parser.add_argument("--playback-speed", type=float, default=1.0)
parser.add_argument("--start-time", type=float, default=0.0, help="Replay start time in trace seconds.")
parser.add_argument("--duration", type=float, default=None, help="Replay duration in trace seconds.")
parser.add_argument("--no-loop", action="store_true", help="Stop after one replay instead of looping.")
parser.add_argument("--show-real", action="store_true", default=True)
parser.add_argument("--hide-real", action="store_false", dest="show_real")
parser.add_argument("--show-baseline", action="store_true", default=True)
parser.add_argument("--hide-baseline", action="store_false", dest="show_baseline")
parser.add_argument("--show-model", action="store_true", default=True)
parser.add_argument("--hide-model", action="store_false", dest="show_model")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()


# Launch Isaac Lab before importing simulation-dependent modules.
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG  # noqa: E402
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spawn_utils import spawn_from_usd_with_fixed_base  # noqa: E402


CSV_TO_USD_JOINT = {
    "right_shoulder_pitch": "right_shoulder_pitch_joint",
    "right_shoulder_roll": "right_shoulder_roll_joint",
    "right_shoulder_yaw": "right_shoulder_yaw_joint",
    "right_elbow": "right_elbow_pitch_joint",
    "right_hip_pitch": "right_hip_pitch_joint",
    "right_hip_roll": "right_hip_roll_joint",
    "right_hip_yaw": "right_hip_yaw_joint",
    "right_knee": "right_knee_joint",
    "right_ankle_pitch": "right_ankle_pitch_joint",
    "right_ankle_roll": "right_ankle_roll_joint",
}

SLICE_DEFAULTS = {
    "arm": {
        "benchmark_root": "output/g1_arm_repro_full_20260521",
        "baseline_robot": "g1_right_arm_default_pd",
        "model_robot": "g1_right_arm_fulltorque_enriched",
        "baseline_actuator": "g1_arm_implicit",
        "model_actuator": "g1_arm_no_pd",
    },
    "leg": {
        "benchmark_root": "output/g1_leg_repro_full_20260521",
        "baseline_robot": "g1_right_leg_default_pd",
        "model_robot": "g1_right_leg_v2_fixedpd_lag10_sysid",
        "baseline_actuator": "g1_leg_implicit",
        "model_actuator": "g1_leg_v2_fixedpd_lag10_sysid",
    },
}


@dataclass
class Trace:
    label: str
    csv_path: Path
    joint_names: list[str]
    time_s: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray


def _parse_vec(value: str) -> list[float]:
    return [float(v) for v in ast.literal_eval(value)]


def _timestamp_to_seconds(value: str) -> float:
    raw = float(value)
    return raw / 1e6 if abs(raw) > 1000.0 else raw


def _read_joint_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _read_trace(label: str, state_path: Path, joint_list_path: Path) -> Trace:
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    if not joint_list_path.is_file():
        raise FileNotFoundError(joint_list_path)

    times: list[float] = []
    positions: list[list[float]] = []
    velocities: list[list[float]] = []
    with state_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("type") and row["type"] != "STATE_MOTOR":
                continue
            times.append(_timestamp_to_seconds(row["timestamp"]))
            positions.append(_parse_vec(row["positions"]))
            velocities.append(_parse_vec(row.get("velocities", "[]")) or [0.0] * len(positions[-1]))

    if len(times) < 2:
        raise ValueError(f"{state_path} must contain at least two STATE_MOTOR rows")

    time_s = np.asarray(times, dtype=np.float64)
    time_s = time_s - time_s[0]
    return Trace(
        label=label,
        csv_path=state_path,
        joint_names=_read_joint_names(joint_list_path),
        time_s=time_s,
        positions=np.asarray(positions, dtype=np.float64),
        velocities=np.asarray(velocities, dtype=np.float64),
    )


def _sim_state_path(root: Path, robot: str, motion_source: str, actuator: str, motion: str) -> Path:
    return root / "sim" / robot / motion_source / actuator / motion / "state_motor.csv"


def _sim_joint_list_path(root: Path, robot: str, motion_source: str, actuator: str, motion: str) -> Path:
    return root / "sim" / robot / motion_source / actuator / motion / "joint_list.txt"


def _real_state_path(root: Path, robot: str, motion_source: str, motion: str) -> Path:
    return root / "real" / robot / motion_source / motion / "state_motor.csv"


def _real_joint_list_path(root: Path, robot: str, motion_source: str, motion: str) -> Path:
    return root / "real" / robot / motion_source / motion / "joint_list.txt"


def _resolve_trace_paths() -> tuple[Trace, Trace, Trace]:
    defaults = SLICE_DEFAULTS[args.slice]
    root = Path(args.benchmark_root or defaults["benchmark_root"]).expanduser()
    baseline_robot = args.baseline_robot or defaults["baseline_robot"]
    model_robot = args.model_robot or defaults["model_robot"]
    baseline_actuator = args.baseline_actuator or defaults["baseline_actuator"]
    model_actuator = args.model_actuator or defaults["model_actuator"]

    real = _read_trace(
        "real",
        _real_state_path(root, baseline_robot, args.motion_source, args.motion),
        _real_joint_list_path(root, baseline_robot, args.motion_source, args.motion),
    )
    baseline = _read_trace(
        "baseline",
        _sim_state_path(root, baseline_robot, args.motion_source, baseline_actuator, args.motion),
        _sim_joint_list_path(root, baseline_robot, args.motion_source, baseline_actuator, args.motion),
    )
    model = _read_trace(
        "model",
        _sim_state_path(root, model_robot, args.motion_source, model_actuator, args.motion),
        _sim_joint_list_path(root, model_robot, args.motion_source, model_actuator, args.motion),
    )
    return real, baseline, model


def _common_grid(traces: list[Trace], render_hz: float, playback_speed: float) -> np.ndarray:
    base_t0 = max(float(t.time_s[0]) for t in traces)
    base_t1 = min(float(t.time_s[-1]) for t in traces)
    t0 = base_t0 + max(args.start_time, 0.0)
    t1 = base_t1
    if args.duration is not None:
        t1 = min(t1, t0 + args.duration)
    if t1 <= t0:
        raise ValueError(f"No shared time range across traces: start={t0:.6f}, end={t1:.6f}")
    dt = max(playback_speed, 1e-6) / render_hz
    return np.arange(t0, t1 + 1e-12, dt, dtype=np.float64)


def _interp_trace(trace: Trace, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.zeros((len(grid), trace.positions.shape[1]), dtype=np.float32)
    vel = np.zeros((len(grid), trace.velocities.shape[1]), dtype=np.float32)
    for idx in range(trace.positions.shape[1]):
        pos[:, idx] = np.interp(grid, trace.time_s, trace.positions[:, idx])
        vel[:, idx] = np.interp(grid, trace.time_s, trace.velocities[:, idx])
    return pos, vel


def _make_robot_cfg(name: str, pos: tuple[float, float, float], color: tuple[float, float, float]) -> ArticulationCfg:
    spawn = G1_MINIMAL_CFG.spawn.replace(
        func=spawn_from_usd_with_fixed_base,
        activate_contact_sensors=False,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    )
    return G1_MINIMAL_CFG.replace(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos,
            joint_pos={
                ".*_hip_yaw_joint": 0.0,
                ".*_hip_roll_joint": 0.0,
                ".*_hip_pitch_joint": 0.0,
                ".*_knee_joint": 0.0,
                ".*_ankle_pitch_joint": 0.0,
                ".*_ankle_roll_joint": 0.0,
                "torso_joint": 0.0,
                ".*_shoulder_pitch_joint": 0.0,
                ".*_shoulder_roll_joint": 0.0,
                ".*_shoulder_yaw_joint": 0.0,
                ".*_elbow_pitch_joint": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=300,
                stiffness=40.0,
                damping=2.0,
            )
        },
    )


def _apply_display_color(prim_prefix: str, color: tuple[float, float, float]) -> None:
    """Apply a simple USD displayColor override to all mesh/gprim descendants."""
    try:
        import omni.usd
        from pxr import Gf, UsdGeom
    except Exception:
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    color_value = [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
    for prim in stage.TraverseAll():
        prim_path = prim.GetPath().pathString
        if not prim_path.startswith(prim_prefix):
            continue
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Gprim(prim).CreateDisplayColorAttr(color_value)


@configclass
class G1OverlaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    real: ArticulationCfg = _make_robot_cfg("RealTrace", (0.0, 0.0, 1.5), (0.08, 0.08, 0.08))
    baseline: ArticulationCfg = _make_robot_cfg("BaselineSim", (0.0, 0.0, 1.5), (0.95, 0.25, 0.12))
    model: ArticulationCfg = _make_robot_cfg("ModelSim", (0.0, 0.0, 1.5), (0.1, 0.65, 0.2))

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


def _layout_positions() -> dict[str, tuple[float, float, float]]:
    if args.layout == "spread":
        return {
            "real": (0.0, -args.spacing, 1.5),
            "baseline": (0.0, 0.0, 1.5),
            "model": (0.0, args.spacing, 1.5),
        }
    return {
        "real": (0.0, 0.0, 1.5),
        "baseline": (0.0, 0.0, 1.5),
        "model": (0.0, 0.0, 1.5),
    }


def _make_scene() -> tuple[SimulationContext, InteractiveScene]:
    sim_cfg = SimulationCfg(
        dt=1.0 / args.render_hz,
        render_interval=1,
        gravity=(0.0, 0.0, -9.81),
    )
    sim_cfg.physics = NewtonCfg(
        use_cuda_graph=False,
        solver_cfg=MJWarpSolverCfg(integrator="implicitfast", ls_iterations=10, iterations=10),
    )
    sim = SimulationContext(sim_cfg)

    positions = _layout_positions()
    scene_cfg = G1OverlaySceneCfg(num_envs=1, env_spacing=4.0)
    scene_cfg.real = _make_robot_cfg("RealTrace", positions["real"], (0.08, 0.08, 0.08))
    scene_cfg.baseline = _make_robot_cfg("BaselineSim", positions["baseline"], (0.95, 0.25, 0.12))
    scene_cfg.model = _make_robot_cfg("ModelSim", positions["model"], (0.1, 0.65, 0.2))
    scene = InteractiveScene(scene_cfg)
    _apply_display_color("/World/envs/env_0/RealTrace", (0.08, 0.08, 0.08))
    _apply_display_color("/World/envs/env_0/BaselineSim", (0.95, 0.25, 0.12))
    _apply_display_color("/World/envs/env_0/ModelSim", (0.1, 0.65, 0.2))
    sim.reset()
    scene.reset()
    return sim, scene


def _joint_indices(robot, trace: Trace) -> list[int]:
    indices = []
    for csv_name in trace.joint_names:
        sim_name = CSV_TO_USD_JOINT.get(csv_name, csv_name)
        if sim_name not in robot.joint_names:
            raise ValueError(f"Joint '{csv_name}' -> '{sim_name}' not found in G1 USD")
        indices.append(robot.joint_names.index(sim_name))
    return indices


def _write_trace(robot, joint_ids: list[int], pos_row: np.ndarray, vel_row: np.ndarray) -> None:
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    device = joint_pos.device
    joint_pos[:, joint_ids] = torch.tensor(pos_row, dtype=joint_pos.dtype, device=device).unsqueeze(0)
    joint_vel[:, joint_ids] = torch.tensor(vel_row, dtype=joint_vel.dtype, device=device).unsqueeze(0)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)


def main() -> None:
    real_trace, baseline_trace, model_trace = _resolve_trace_paths()
    traces = []
    if args.show_real:
        traces.append(real_trace)
    if args.show_baseline:
        traces.append(baseline_trace)
    if args.show_model:
        traces.append(model_trace)
    if not traces:
        raise ValueError("At least one trace must be visible")

    grid = _common_grid(traces, args.render_hz, args.playback_speed)
    real_pos, real_vel = _interp_trace(real_trace, grid)
    baseline_pos, baseline_vel = _interp_trace(baseline_trace, grid)
    model_pos, model_vel = _interp_trace(model_trace, grid)

    print("G1 overlay replay")
    print(f"  motion: {args.motion}")
    print(f"  layout: {args.layout}")
    print(
        f"  duration: {grid[-1] - grid[0]:.2f}s trace time at "
        f"{args.render_hz:.1f} viewer Hz ({args.playback_speed:.2f}x speed)"
    )
    print(f"  real:     {real_trace.csv_path}")
    print(f"  baseline: {baseline_trace.csv_path}")
    print(f"  model:    {model_trace.csv_path}")

    sim, scene = _make_scene()
    robots = {
        "real": scene["real"],
        "baseline": scene["baseline"],
        "model": scene["model"],
    }
    joint_ids = {
        "real": _joint_indices(robots["real"], real_trace),
        "baseline": _joint_indices(robots["baseline"], baseline_trace),
        "model": _joint_indices(robots["model"], model_trace),
    }

    frame = 0
    wall_next = time.perf_counter()
    frame_dt = 1.0 / args.render_hz
    try:
        while simulation_app.is_running():
            if sim.is_stopped():
                break
            if not sim.is_playing():
                sim.step()
                continue

            idx = frame % len(grid)
            if args.show_real:
                _write_trace(robots["real"], joint_ids["real"], real_pos[idx], real_vel[idx])
            if args.show_baseline:
                _write_trace(robots["baseline"], joint_ids["baseline"], baseline_pos[idx], baseline_vel[idx])
            if args.show_model:
                _write_trace(robots["model"], joint_ids["model"], model_pos[idx], model_vel[idx])

            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(frame_dt)

            frame += 1
            if frame >= len(grid) and args.no_loop:
                break

            wall_next += frame_dt
            sleep_s = wall_next - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
