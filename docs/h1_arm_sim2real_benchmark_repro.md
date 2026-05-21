# H1 Arm Sim2Real Benchmark Repro

This file is the source of truth for the current clean H1 right-arm benchmark
on branch `vbhavanantha/h1-arm-sim2real-benchmark-repro`.

It mirrors the G1 cleanup principle: one recorded gantry experiment at a time,
fixed base, measured initial state, documented buffer, documented scoring, and
no untracked WIP branch behavior mixed into the headline number.

## Headline Result

Current test split:

```text
input/sysid_data/h1/multijoint_gripper/test
```

On this machine that folder is a local symlink to:

```text
/home/vbhavanantha/IsaacLab-Newton/input/sysid_data/h1/multijoint_gripper
```

Current output folders:

```text
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_arm_repro_test10_20260521
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_arm_repro_analysis_pd_test10_20260521/metrics_summary.xlsx
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_arm_repro_analysis_lag20_sysid_test10_20260521/metrics_summary.xlsx
```

The table below uses the H1 SAGE analysis convention: arithmetic mean over the
4 scored joints x 10 motions in each `*_rmse_compare` sheet.

| H1 right-arm config | Position RMSE | Velocity RMSE | Torque RMSE |
| --- | ---: | ---: | ---: |
| PD baseline | 0.017179 rad | 0.313199 rad/s | 1.159809 Nm |
| lag20 SysID | 0.011001 rad | 0.153816 rad/s | 0.702555 Nm |
| lag20 SysID reduction | 36.0% | 50.9% | 39.4% |

Lower RMSE is better. A positive reduction means the SysID model has lower
error than the PD baseline.

## What Is Being Compared

Baseline:

```text
h1_arm_pd_baseline_yawclamped_gripperpos
input/actuator_models/h1/h1_arm_implicit.yaml
```

This is implicit PD with:

```text
kp = 60
kd = 1.5
motor_lag_ms = 10
armature = 0
dynamic_friction = 0
viscous_friction = 0
```

Candidate:

```text
h1_arm_lag20_sysid_gripperpos
input/actuator_models/h1/h1_arm_post_velsync_sysid_gen10_lag20.yaml
```

This keeps the same factory-style PD gains but adds identified motor terms:

```text
kp = 60
kd = 1.5
motor_lag_ms = 20
armature, dynamic friction, and viscous friction from SysID gen10
```

This primary H1 benchmark is not using a GRU. There is no learned recurrent
state to warm. The buffer still matters because it settles the fixed-base
physics state and the command-delay path before replay begins.

An experimental residual GRU branch exists separately:

```text
docs/h1_arm_gru_benchmark_test.md
```

That branch keeps this same benchmark method and adds a historical PD+GRU
feed-forward candidate on top of lag20 SysID. It is documented separately so the
simple SysID result and the learned recurrent result do not get mixed together.

## Scored Joints

The benchmark scores only the right arm:

```text
right_elbow
right_shoulder_pitch
right_shoulder_roll
right_shoulder_yaw
```

Those names and their order are recorded in:

```text
scripts/sim2real_gap/configs/h1_arm_pd_baseline_yawclamped_gripperpos_joints.yaml
scripts/sim2real_gap/configs/h1_arm_lag20_sysid_gripperpos_joints.yaml
scripts/sim2real_gap/configs/h1_arm_pd_baseline_yawclamped_gripperpos_valid_joints.txt
scripts/sim2real_gap/configs/h1_arm_lag20_sysid_gripperpos_valid_joints.txt
```

The order matters because SAGE uses these files to map CSV columns to joint
names. The clean branch copies the local joint config into the installed SAGE
package when the package copy is stale.

## Benchmark Method

Each CSV is one independent gantry capture, so the benchmark runs one motion at
a time. It does not concatenate the 10 motions into one long rollout.

For each motion:

1. Convert the raw H1 `*_motor.csv` into SAGE-style `real/.../control.csv`,
   `state_motor.csv`, `event.csv`, and `joint_list.txt`.
2. Spawn H1 with a fixed root using the yaw-clamped gripper-position USD:

   ```text
   /home/vbhavanantha/h1_groundtruth_assets/h1_gripper_groundtruth_yawclamped_gripperpos.usd
   ```

3. Keep the lower body in a neutral fixed-base posture. This matches the
   gantry experiment; it is not a walking or floating-base benchmark.
4. Keep the left arm on factory PD and evaluate only the right-arm actuator
   group. This prevents right-arm actuator changes from accidentally changing
   the unscored left arm.
5. Run the configured 2 second buffer from the initial sim pose to the first
   recorded command.
6. With `--real-init-pose-sync`, write the scored right-arm joints to row 0 of
   the real `state_motor.csv`, including row-0 velocity when available.
7. Replay the recorded right-arm command trajectory.
8. Optionally replay unscored H1 full-body auxiliary commands from:

   ```text
   ~/chiplog/Tools/Postprocess/motion_files
   ```

   These commands are excluded from scoring. For the yaw-clamped USD, left-arm
   aux commands are skipped because that asset has a modified shoulder-yaw
   setup. A C04 smoke test showed the aux replay is not what changes the
   headline H1 numbers.
9. Score sim `state_motor.csv` against real `state_motor.csv`. Command position
   drives replay but is not the scored target.

## Scoring Method

The H1 numbers above come from:

```text
scripts/sim2real_gap/run_analysis.py
```

This script runs the existing SAGE analysis path with two important fixes:

- local H1 joint config files are copied into SAGE if the installed package has
  stale config
- any sim or real timestamp larger than `1000` is treated as microseconds and
  converted to seconds before analysis

SAGE then:

1. uses `event.csv` to align the motion window from `MOTION_START` to
   `DISABLE`
2. resamples sim and real onto a uniform `sample_dt = 0.005` second grid
3. computes RMSE per joint per motion for position, velocity, and torque
4. writes those tables into `metrics_summary.xlsx`

The headline table is the arithmetic mean over all cells in:

```text
positions_rmse_compare
velocities_rmse_compare
torques_rmse_compare
```

This is interpolation onto a fixed time grid. It is not time warping. The
analysis does not slide, stretch, compress, or model-specifically align the sim
trace to lower error. It also does not drop bad motions or bad joints.

## Commands

Use the IsaacLab conda environment:

```bash
cd /home/vbhavanantha/IsaacLab-Newton-g1-repro
source /home/vbhavanantha/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
```

Run the PD baseline:

```bash
./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name h1_arm_pd_baseline_yawclamped_gripperpos \
  --real-init-pose-sync \
  --headless \
  --experience /home/vbhavanantha/IsaacLab-Newton-g1-repro/apps/isaaclab.python.headless.kit
```

Run lag20 SysID:

```bash
./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name h1_arm_lag20_sysid_gripperpos \
  --real-init-pose-sync \
  --headless \
  --experience /home/vbhavanantha/IsaacLab-Newton-g1-repro/apps/isaaclab.python.headless.kit
```

Analyze the PD baseline:

```bash
python scripts/sim2real_gap/run_analysis.py \
  --robot-name h1_arm_pd_baseline_yawclamped_gripperpos \
  --valid-joints-file scripts/sim2real_gap/configs/h1_arm_pd_baseline_yawclamped_gripperpos_valid_joints.txt
```

Analyze lag20 SysID:

```bash
python scripts/sim2real_gap/run_analysis.py \
  --robot-name h1_arm_lag20_sysid_gripperpos \
  --valid-joints-file scripts/sim2real_gap/configs/h1_arm_lag20_sysid_gripperpos_valid_joints.txt
```

## Per-Motion Torque Result

These are motion-wise means over the 4 scored joints from the SAGE
`torques_rmse_compare` sheet.

| Motion | PD tau RMSE | lag20 SysID tau RMSE | Reduction |
| --- | ---: | ---: | ---: |
| `C01_f0_1-0_5_a0_05` | 0.9343 | 0.5882 | 37.0% |
| `C04_f0_1-0_5_a0_30` | 1.2515 | 0.6451 | 48.5% |
| `C08_f0_1-1_0_a0_20` | 1.1836 | 0.6942 | 41.3% |
| `C09_f0_1-1_0_a0_30` | 1.3807 | 0.8641 | 37.4% |
| `C21_f0_1-0_5_a0_15_wave` | 1.0732 | 0.6353 | 40.8% |
| `S01_f0_1_a0_15_inphase` | 1.0364 | 0.6708 | 35.3% |
| `S06_f0_5_a0_05_inphase` | 0.9052 | 0.5435 | 40.0% |
| `S09_f0_5_a0_30_inphase` | 1.2966 | 0.7016 | 45.9% |
| `S14_f1_0_a0_15_wave` | 1.2308 | 0.8055 | 34.6% |
| `S19_f1_0_a0_15_antiphase` | 1.3057 | 0.8772 | 32.8% |

## Plot Outputs

Fresh plots from the same current clean H1 benchmark outputs:

```text
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_plots_current/arm_real_pd_lag20sysid_labeled/h1_arm_real_pd_lag20sysid_labeled_all_motions.pdf
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_plots_current/arm_real_pd_lag20sysid_labeled/h1_arm_pd_lag20sysid_labeled_rmse_summary.png
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_plots_current/arm_real_pd_lag20sysid_labeled/h1_arm_pd_lag20sysid_labeled_plot_metrics.csv
```

There is also one PNG per motion in:

```text
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_plots_current/arm_real_pd_lag20sysid_labeled
```

The trace plots intentionally omit command position and show only real, PD
baseline, and lag20 SysID for the scored position and torque signals. Plot
titles use the same SAGE `metrics_summary.xlsx` per-motion RMSE means as the
tables above.

## Bugs Fixed In The Clean Branch

- Added the H1 yaw-clamped gripper-position aliases to the clean benchmark
  branch.
- Added the H1 lag20 SysID actuator YAML needed by the current result.
- Kept left-arm factory PD separate from the right-arm benchmark actuator
  group.
- Added H1 joint config and valid-joint files in the SAGE/motor CSV order.
- Added SAGE config refresh in `run_analysis.py` so stale installed config does
  not silently score the wrong joints.
- Added timestamp unit normalization in `run_analysis.py` for sim/real CSVs
  written in microseconds.
- Added explicit H1 full-body auxiliary command replay config. It is part of
  the method, but it is not the reason the current result differs from the old
  WIP result.

## Historical WIP Caveat

There is an older H1 run in the original dirty workspace:

```text
/home/vbhavanantha/IsaacLab-Newton/output/sim2real_analysis_pd_yawclamped_gripperpos_test10/metrics_summary.xlsx
/home/vbhavanantha/IsaacLab-Newton/output/sim2real_analysis_lag20_sysid_gripperpos_test10/metrics_summary.xlsx
```

That run was produced on May 16, 2026 from:

```text
/home/vbhavanantha/IsaacLab-Newton
branch: vbhavanantha/h1-multijoint-gripper-sysid
commit: 84b854bb
```

The old WIP result was:

| H1 right-arm config | Position RMSE | Velocity RMSE | Torque RMSE |
| --- | ---: | ---: | ---: |
| old WIP PD baseline | 0.016969 rad | 0.313515 rad/s | 1.147457 Nm |
| old WIP lag20 SysID | 0.013665 rad | 0.165581 rad/s | 1.002965 Nm |

The PD baseline agrees closely between old WIP and current clean branch. The
lag20 SysID output does not: current clean branch gives `0.702555 Nm`, while
the old WIP output gives `1.002965 Nm`.

Things checked and ruled out as the explanation:

- different command CSVs
- different actuator YAML values
- different USD path
- enabling/disabling self-collision on the H1 scene
- syncing row-0 velocity during init-pose sync
- enabling/disabling the new full-body auxiliary command replay

The old WIP branch and the current clean branch diverge across a large IsaacLab
and Newton source update. The old run also did not encode enough source
provenance inside the output folder to replay the exact WIP code path from the
output alone. For reporting, use the current clean-branch result above as the
primary H1 benchmark. Treat the May 16 WIP number as historical context only.

Single-motion C04 smoke check for the aux-replay question, using a raw
interpolation scorer rather than the SAGE headline metric:

| Source | Position RMSE | Velocity RMSE | Torque RMSE |
| --- | ---: | ---: | ---: |
| May 16 old WIP | 0.013748 | 0.178047 | 1.109696 |
| current clean, aux on | 0.010383 | 0.215782 | 0.687284 |
| current clean, aux off | 0.010461 | 0.215299 | 0.687366 |

Aux replay does not explain the old/current gap.

## Verification Already Run

Full benchmark runs completed for:

```text
h1_arm_pd_baseline_yawclamped_gripperpos
h1_arm_lag20_sysid_gripperpos
```

SAGE analysis completed for both outputs and produced:

```text
output/h1_arm_repro_analysis_pd_test10_20260521/metrics_summary.xlsx
output/h1_arm_repro_analysis_lag20_sysid_test10_20260521/metrics_summary.xlsx
```

Python compile check:

```bash
python -m compileall \
  scripts/sim2real_gap/newton_benchmark.py \
  scripts/sim2real_gap/run_benchmark.py \
  scripts/sim2real_gap/run_analysis.py
```

The compile check passed after the H1 benchmark code and documentation updates.
Rerun it after any future benchmark code edit.
