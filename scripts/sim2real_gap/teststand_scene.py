"""Motor test stand scene — IsaacLab scene config for the elbow benchtop rig.

Loads the motor_teststand USDA (fixed base + bar arm + revolute elbow joint)
via IsaacLab's InteractiveScene + SimulationContext with Newton solver.

Usage (from repo root, inside docker):
    # Sine wave demo
    python scripts/sim2real_gap/teststand_scene.py

    # Custom PD gains
    python scripts/sim2real_gap/teststand_scene.py --kp 131.47 --kd 8.52

    # Replay real CSV data
    python scripts/sim2real_gap/teststand_scene.py --csv path/to/data.csv
"""

import argparse
import math
import os

import numpy as np
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_TESTSTAND_USD = os.path.join(
    _REPO_ROOT, "input", "robot_models", "motor_teststand", "motor_teststand.usda"
)

# ---------------------------------------------------------------------------
# Default PD gains (match Desktop newton-teststand defaults)
# ---------------------------------------------------------------------------
DEFAULT_KP = 60.0
DEFAULT_KD = 1.5
SIM_DT = 1.0 / 200.0  # 200 Hz, matches real motor control rate


# ---------------------------------------------------------------------------
# Scene config
# ---------------------------------------------------------------------------
class TestStandSceneCfg(InteractiveSceneCfg):
    """Scene with the motor test stand articulation and a ground plane."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/TestStand",
        spawn=sim_utils.UsdFileCfg(usd_path=_TESTSTAND_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={"elbow": 0.0},
            joint_vel={"elbow": 0.0},
        ),
        actuators={
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["elbow"],
                effort_limit_sim=300.0,
                stiffness=DEFAULT_KP,
                damping=DEFAULT_KD,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Motor test stand scene")
    parser.add_argument("--kp", type=float, default=DEFAULT_KP, help="PD stiffness")
    parser.add_argument("--kd", type=float, default=DEFAULT_KD, help="PD damping")
    parser.add_argument("--csv", type=str, default=None, help="Real experiment CSV to replay")
    parser.add_argument("--duration", type=float, default=10.0, help="Sine wave duration (s)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Simulation context (Newton solver)
    # ------------------------------------------------------------------
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    sim_cfg = SimulationCfg(
        dt=SIM_DT,
        render_interval=1,
        gravity=(0.0, 0.0, -9.81),
    )
    sim_cfg.physics = NewtonCfg(
        use_cuda_graph=False,
        solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast",
            ls_iterations=10,
            iterations=10,
        ),
    )
    sim = SimulationContext(sim_cfg)

    # ------------------------------------------------------------------
    # Build scene with user-specified gains
    # ------------------------------------------------------------------
    scene_cfg = TestStandSceneCfg(num_envs=1, env_spacing=4.0)
    scene_cfg.robot.actuators["elbow"].stiffness = args.kp
    scene_cfg.robot.actuators["elbow"].damping = args.kd

    scene = InteractiveScene(scene_cfg)

    # Reset sim
    sim.reset()
    scene.reset()

    # ------------------------------------------------------------------
    # Load real data or generate sine wave
    # ------------------------------------------------------------------
    if args.csv:
        import pandas as pd

        print(f"Loading real data: {args.csv}")
        df = pd.read_csv(args.csv)
        df.columns = [c.strip() for c in df.columns]
        real_cmd = df["commanded_position"].to_numpy(dtype=np.float64)
        real_pos = df["position"].to_numpy(dtype=np.float64)
        real_vel = df["velocity"].to_numpy(dtype=np.float64)
        real_torque = df["torque"].to_numpy(dtype=np.float64)
        n_steps = len(real_cmd)
        print(f"  {n_steps} samples")
    else:
        n_steps = int(args.duration / SIM_DT)
        amplitude = math.radians(60.0)
        frequency = 0.5
        real_cmd = np.array([
            amplitude * math.sin(2.0 * math.pi * frequency * i * SIM_DT)
            for i in range(n_steps)
        ])
        real_pos = None
        real_vel = None
        real_torque = None
        print(f"Sine wave: {n_steps} steps, {args.duration}s, amp={math.degrees(amplitude):.0f} deg, freq={frequency} Hz")

    # ------------------------------------------------------------------
    # Logging arrays
    # ------------------------------------------------------------------
    log_time = np.zeros(n_steps)
    log_cmd = np.zeros(n_steps)
    log_sim_pos = np.zeros(n_steps)
    log_sim_vel = np.zeros(n_steps)
    log_sim_torque = np.zeros(n_steps)

    # ------------------------------------------------------------------
    # Set initial joint position
    # ------------------------------------------------------------------
    robot = scene["robot"]
    joint_idx = robot.find_joints("elbow")[0][0]
    init_pos = float(real_cmd[0])

    # Newton uses wp.array — write via wp.to_torch bridge
    default_pos = wp.to_torch(robot.data.default_joint_pos)
    default_pos[0, joint_idx] = init_pos
    robot.write_joint_state_to_sim(
        position=default_pos,
        velocity=wp.to_torch(robot.data.default_joint_vel),
    )
    sim.step()
    scene.update(SIM_DT)

    # ------------------------------------------------------------------
    # Sim loop
    # ------------------------------------------------------------------
    print(f"Running: kp={args.kp}, kd={args.kd}, steps={n_steps}")
    for step in range(n_steps):
        cmd = float(real_cmd[step])

        # Set position target
        target = wp.to_torch(robot.data.joint_pos_target)
        target[0, joint_idx] = cmd
        robot.set_joint_position_target(target)

        # Read state before step (for torque computation)
        pos = wp.to_torch(robot.data.joint_pos)[0, joint_idx].item()
        vel = wp.to_torch(robot.data.joint_vel)[0, joint_idx].item()
        torque = args.kp * (cmd - pos) - args.kd * vel

        # Step
        scene.write_data_to_sim()
        sim.step()
        scene.update(SIM_DT)

        # Log
        log_time[step] = step * SIM_DT
        log_cmd[step] = cmd
        log_sim_pos[step] = wp.to_torch(robot.data.joint_pos)[0, joint_idx].item()
        log_sim_vel[step] = wp.to_torch(robot.data.joint_vel)[0, joint_idx].item()
        log_sim_torque[step] = torque

        if (step + 1) % 200 == 0:
            print(
                f"  Step {step+1}/{n_steps} t={log_time[step]:.2f}s"
                f" | pos={log_sim_pos[step]:.4f} | cmd={cmd:.4f}"
                f" | torque={torque:.2f}"
            )

    print("Done.")

    # ------------------------------------------------------------------
    # Metrics (if CSV provided)
    # ------------------------------------------------------------------
    if real_pos is not None:
        pos_rmse = np.sqrt(np.mean((log_sim_pos - real_pos) ** 2))
        vel_rmse = np.sqrt(np.mean((log_sim_vel - real_vel) ** 2))
        tau_rmse = np.sqrt(np.mean((log_sim_torque - real_torque) ** 2))
        print(f"Position RMSE: {pos_rmse:.4f} rad")
        print(f"Velocity RMSE: {vel_rmse:.4f} rad/s")
        print(f"Torque   RMSE: {tau_rmse:.4f} Nm")

    # Cleanup
    sim.stop()


if __name__ == "__main__":
    main()
