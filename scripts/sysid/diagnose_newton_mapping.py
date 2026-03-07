"""Diagnostic: check what Isaac Lab write_joint_*_to_sim() maps to in Newton.

Run inside Docker:
    python scripts/sysid/diagnose_newton_mapping.py --headless

Prints Newton model properties before/after each write call to determine
which MuJoCo property each Isaac Lab API actually modifies.
"""

import os
import sys

from isaaclab.app import AppLauncher

parser = __import__("argparse").ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from input.actuator_models import load_implicit_actuator_cfg  # noqa: E402

_UR10_USD = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../input/robot_models/ur10/ur10/ur10.usd")
)


@configclass
class DiagSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(10.0, 10.0)),
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=_UR10_USD),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.712,
                "elbow_joint": 1.712,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
            },
        ),
        actuators={
            "arm": load_implicit_actuator_cfg("ur10e/ur10e_implicit.yaml", [".*"]),
        },
    )


def main():
    sim_cfg = SimulationCfg(dt=0.002, render_interval=1, gravity=(0.0, 0.0, -9.81))
    sim_cfg.newton_cfg.solver_cfg.integrator = "implicitfast"
    sim_cfg.newton_cfg.solver_cfg.ls_iterations = 10
    sim_cfg.newton_cfg.solver_cfg.iterations = 10

    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(DiagSceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    scene.reset()

    robot = scene["robot"]
    device = robot.device
    joint_ids = torch.tensor([0], dtype=torch.long, device=device)  # first joint
    env_ids = torch.tensor([0], dtype=torch.long, device=device)

    # --- Try to access Newton's underlying model ---
    print("\n" + "=" * 70)
    print("NEWTON MODEL ACCESS")
    print("=" * 70)

    newton_model = None

    # 1. Try known access paths on sim
    for attr_chain in [
        "sim._world.model",
        "sim._newton_manager._solver._model",
        "sim._newton_manager._solver.model",
        "sim._newton_manager.model",
        "sim._world",
    ]:
        try:
            obj = sim
            for part in attr_chain.split(".")[1:]:
                obj = getattr(obj, part)
            print(f"  {attr_chain} = {type(obj)}")
            if hasattr(obj, "joint_target_ke"):
                newton_model = obj
                print(f"  >>> Found Newton model at {attr_chain}")
        except AttributeError:
            print(f"  {attr_chain} = NOT FOUND")

    # 2. Check robot's view objects
    for attr in ["root_physx_view", "root_view", "_root_view", "root_newton_view"]:
        try:
            view = getattr(robot, attr)
            print(f"  robot.{attr} = {type(view)}")
        except AttributeError:
            print(f"  robot.{attr} = NOT FOUND")

    # 3. Explore robot.root_view (ArticulationView) — this is the Newton view
    view = robot.root_view
    print(f"\nrobot.root_view type: {type(view)}")
    print("  All attributes (incl. private):")
    for attr in sorted(dir(view)):
        if attr.startswith("__"):
            continue
        try:
            val = getattr(view, attr)
            if callable(val):
                print(f"    .{attr}() — callable")
            else:
                print(f"    .{attr} = {type(val).__name__}: {repr(val)[:100]}")
        except Exception as e:
            print(f"    .{attr} — error: {e}")

    # 4. BFS from robot.root_view, 4 levels, looking for joint_target_ke
    print("\nBFS from robot.root_view for 'joint_target_ke' (4 levels):")
    visited = set()
    queue = [("view", view)]
    found_paths = []
    for _depth in range(4):
        next_queue = []
        for path, obj in queue:
            oid = id(obj)
            if oid in visited:
                continue
            visited.add(oid)
            if hasattr(obj, "joint_target_ke"):
                found_paths.append(path)
                print(f"  FOUND: {path} ({type(obj).__name__})")
                newton_model = obj
            for attr in dir(obj):
                if attr.startswith("__"):
                    continue
                try:
                    child = getattr(obj, attr)
                    if not callable(child) and not isinstance(child, (int, float, str, bool, type(None))):
                        next_queue.append((f"{path}.{attr}", child))
                except Exception:
                    pass
        queue = next_queue
    if not found_paths:
        print("  NOT FOUND in 4-level BFS from view")

    # 5. Try newton.World / newton.Model directly
    print("\nNewton module inspection:")
    try:
        import newton
        print(f"  newton version attrs: {[a for a in dir(newton) if not a.startswith('_')]}")
        # Try to find World class and check if it has class method to get instances
        if hasattr(newton, "World"):
            print(f"  newton.World = {newton.World}")
        if hasattr(newton, "Model"):
            print(f"  newton.Model = {newton.Model}")
    except ImportError:
        print("  newton module not importable")

    # --- Snapshot function ---
    def snapshot(label):
        print(f"\n--- {label} ---")
        data = robot.data
        for prop in [
            "joint_stiffness", "joint_damping", "joint_armature",
            "joint_friction", "default_joint_stiffness", "default_joint_damping",
        ]:
            val = getattr(data, prop, None)
            if val is not None:
                print(f"  data.{prop}[0, 0:3] = {val[0, 0:3].list()}")
            else:
                print(f"  data.{prop} = NOT FOUND")

        if newton_model is not None:
            for prop in [
                "joint_target_ke", "joint_target_kd",
                "joint_armature", "joint_friction",
            ]:
                val = getattr(newton_model, prop, None)
                if val is not None:
                    try:
                        arr = val.numpy() if hasattr(val, "numpy") else val
                        print(f"  model.{prop}[0:3] = {arr[:3].tolist()}")
                    except Exception as e:
                        print(f"  model.{prop} = {val} (err: {e})")
                else:
                    print(f"  model.{prop} = NOT FOUND")

            # Check MuJoCo-solver custom attribute: passive viscous damping
            mujoco_ns = getattr(newton_model, "mujoco", None)
            if mujoco_ns is not None and hasattr(mujoco_ns, "dof_passive_damping"):
                val = mujoco_ns.dof_passive_damping
                try:
                    arr = val.numpy() if hasattr(val, "numpy") else val
                    print(f"  model.mujoco.dof_passive_damping[0:3] = {arr[:3].tolist()}")
                except Exception as e:
                    print(f"  model.mujoco.dof_passive_damping = {val} (err: {e})")
            else:
                print(f"  model.mujoco.dof_passive_damping = NOT FOUND")

    # --- Run tests ---
    print("\n" + "=" * 70)
    print("PROPERTY MAPPING TESTS")
    print("=" * 70)

    snapshot("INITIAL STATE")

    # Test 1: write_joint_stiffness_to_sim
    print("\n>>> Calling robot.write_joint_stiffness_to_sim(9999.0)")
    robot.write_joint_stiffness_to_sim(
        torch.tensor([[9999.0]], device=device), joint_ids=joint_ids, env_ids=env_ids
    )
    snapshot("AFTER write_joint_stiffness_to_sim(9999)")

    # Test 2: write_joint_damping_to_sim
    print("\n>>> Calling robot.write_joint_damping_to_sim(7777.0)")
    robot.write_joint_damping_to_sim(
        torch.tensor([[7777.0]], device=device), joint_ids=joint_ids, env_ids=env_ids
    )
    snapshot("AFTER write_joint_damping_to_sim(7777)")

    # Test 3: write_joint_armature_to_sim
    print("\n>>> Calling robot.write_joint_armature_to_sim(5555.0)")
    robot.write_joint_armature_to_sim(
        torch.tensor([[5555.0]], device=device), joint_ids=joint_ids, env_ids=env_ids
    )
    snapshot("AFTER write_joint_armature_to_sim(5555)")

    # Test 4: write_joint_friction_to_sim
    print("\n>>> Calling robot.write_joint_friction_to_sim(3333.0)")
    try:
        robot.write_joint_friction_to_sim(
            torch.tensor([[3333.0]], device=device), joint_ids=joint_ids, env_ids=env_ids
        )
        snapshot("AFTER write_joint_friction_to_sim(3333)")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Test 5: Write directly to mujoco.dof_passive_damping (viscous friction)
    print("\n>>> Writing model.mujoco.dof_passive_damping[0] = 1111.0")
    try:
        mujoco_ns = getattr(newton_model, "mujoco", None)
        if mujoco_ns is not None and hasattr(mujoco_ns, "dof_passive_damping"):
            dof_damping = mujoco_ns.dof_passive_damping
            dof_np = dof_damping.numpy()
            dof_np[0] = 1111.0  # first DOF
            dof_damping.assign(dof_np)
            snapshot("AFTER mujoco.dof_passive_damping[0] = 1111")
        else:
            print("  model.mujoco.dof_passive_damping NOT FOUND — cannot test")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Test 6: Check if everything syncs after sim.step()
    print("\n>>> Running scene.write_data_to_sim() + sim.step() + scene.update()")
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(sim_cfg.dt)
    snapshot("AFTER sim.step()")

    # Test 7: Check the MuJoCo solver model directly
    print("\n" + "=" * 70)
    print("MUJOCO SOLVER MODEL SEARCH")
    print("=" * 70)

    # BFS from newton_model looking for mujoco-specific objects
    if newton_model is not None:
        print(f"\nNewton model type: {type(newton_model)}")
        print("All newton_model attributes:")
        for attr in sorted(dir(newton_model)):
            if attr.startswith("__"):
                continue
            try:
                val = getattr(newton_model, attr)
                if callable(val):
                    print(f"  .{attr}() — callable")
                else:
                    vstr = repr(val)[:100]
                    print(f"  .{attr} = {type(val).__name__}: {vstr}")
            except Exception as e:
                print(f"  .{attr} — error: {e}")

    # Look for solver on the view's world
    print("\nSearching for solver/mujoco model:")
    for attr_chain in [
        "view._world", "view._world.solver", "view._world._solver",
        "view._world.model", "view._articulation._world",
    ]:
        try:
            obj = view
            for part in attr_chain.split(".")[1:]:
                obj = getattr(obj, part)
            print(f"  {attr_chain} = {type(obj)}")
            # Check for MuJoCo model properties
            for prop in ["dof_damping", "dof_armature", "dof_frictionloss",
                         "dof_passive_damping", "joint_target_ke"]:
                if hasattr(obj, prop):
                    print(f"    has .{prop} = {getattr(obj, prop)}")
        except AttributeError:
            print(f"  {attr_chain} = NOT FOUND")

    # Walk the world object if found
    try:
        world = view._world
        print(f"\nview._world type: {type(world)}")
        print("All world attributes:")
        for attr in sorted(dir(world)):
            if attr.startswith("__"):
                continue
            try:
                val = getattr(world, attr)
                if callable(val):
                    print(f"  .{attr}() — callable")
                else:
                    vstr = repr(val)[:100]
                    print(f"  .{attr} = {type(val).__name__}: {vstr}")
            except Exception as e:
                print(f"  .{attr} — error: {e}")
    except AttributeError:
        print("\nview._world = NOT FOUND")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()
