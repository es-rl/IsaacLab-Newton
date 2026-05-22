# SO-101 Simulation Handoff

This is the short handoff for using SO-101 in IsaacLab-Newton when the goal is
to build a task, such as pick-and-place, rather than to run benchmarks.

## What is ready

- Robot USD: `input/robot_models/so101/so101.usd`
- Meshes: `input/robot_models/so101/assets/`
- Runtime config: `input/run_configs/so101/so101.yaml`
- Fitted actuator parameters: `input/actuator_models/so101/so101_implicit.yaml`

`so101_implicit.yaml` is the default fitted actuator file. It contains the
balanced 42-train-motion / 50-generation SysID fit:

- position RMSE improved from `0.045041` to `0.015612` rad on 8 held-out
  validation motions
- torque RMSE improved from `0.615712` to `0.322481` Nm on the same validation
  set

## What the task builder still adds

For pick-and-place, the task builder still needs to add:

- table and object assets
- gripper/object contact setup
- controller, planner, teleop, or policy
- task reset logic
- success metrics, such as object lifted, object moved, or placed in target

The SO-101 branch gives the robot and fitted actuator foundation. It is not a
finished manipulation task.

## Joint names

Use the USD joint names:

```text
Rotation
Pitch
Elbow
Wrist_Pitch
Wrist_Roll
Jaw
```

## If you want to sanity-check the install

Run the no-GPU smoke tests first:

```bash
./isaaclab.sh -p -m pytest \
  scripts/sysid/test/test_so101_smoke.py \
  scripts/sim2real_gap/test/test_so101_benchmark_smoke.py
```

The smoke data zip is optional. It is only for checking the SysID/benchmark
pipeline, not for building a SO-101 task.

## Deeper references

- `docs/so101_onboarding.md`: first-time setup and optional smoke workflow
- `docs/so101_balanced_sysid.md`: SysID provenance and validation numbers
- `scripts/sysid/README.md`: SysID implementation details
- `scripts/sim2real_gap/README.md`: replay/analysis implementation details
