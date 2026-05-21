# G1 current benchmark overview

This file is the short source of truth for the current G1 sim2real benchmark.
It covers the clean right-arm and right-leg results, what is shared between
them, what is different, and what should not be mixed into the headline metric.

Detailed companion docs:

- `docs/g1_arm_sim2real_benchmark_repro.md`
- `docs/g1_leg_benchmark_status.md`
- `docs/g1_leg_gru_benchmark_test.md`

## Headline answer

Yes, the arm and leg are benchmarked with the same current methodology:

1. run one recorded motion at a time
2. convert raw `*_motor.csv` captures into SAGE `control.csv` and
   `state_motor.csv`
3. fix the G1 root/base, matching the gantry experiments rather than walking
4. run with `--real-init-pose-sync`
5. run the configured 5 second buffer before replay
6. for learned recurrent actuators, reset hidden state before the buffer and let
   the buffer warm the recurrent state
7. replay the recorded command trajectory
8. score interpolation RMSE over the shared real/sim time window
9. pool RMSE over the 10 independent motions and the relevant joints

This is the benchmark style we should report as the primary current result.

## What differs between arm and leg

The evaluation method is the same, but the actuator model being evaluated is not.

| Robot slice | Baseline | Current best reported model | Joints | Learned recurrent path |
| --- | --- | --- | ---: | --- |
| right arm | `g1_right_arm_default_pd` | `g1_right_arm_fulltorque_enriched` | 4 | yes, production full-torque GRU |
| right leg | `g1_right_leg_default_pd` plus `g1_right_leg_sdk_pd` reference | `g1_right_leg_v2_fixedpd_lag10_sysid` | 6 | no, fixed-PD lag10 SysID |

The leg GRU has been tested separately on
`vbhavanantha/g1-leg-gru-benchmark-test`. It runs mechanically and gives a tiny
torque improvement over lag10 SysID, but it worsens position and velocity RMSE,
so it is not the primary current leg model.

## Leg baseline gains

The G1 leg baseline is `g1_right_leg_default_pd`, which loads:

```text
input/actuator_models/g1/g1_leg_implicit.yaml
```

Those are stock IsaacLab G1 implicit actuator gains, not SysID gains and not the
real Unitree SDK gains.

| Joint | Baseline kp | Baseline kd |
| --- | ---: | ---: |
| hip pitch | 200 | 5 |
| hip roll | 150 | 5 |
| hip yaw | 150 | 5 |
| knee | 200 | 5 |
| ankle pitch | 40 | 1 |
| ankle roll | 40 | 1 |

The real gantry experiments were commanded with Unitree SDK low-level gains, so
there is also a controller-matched baseline:

```text
input/actuator_models/g1/g1_leg_sdk_pd.yaml
```

This uses the SDK gains only, with no identified armature/friction and no motor
lag. The primary leg SysID model, `g1_right_leg_v2_fixedpd_lag10_sysid`, uses
the same fixed SDK-style PD gains:

```text
kp = [60, 60, 60, 100, 40, 40]
kd = [1, 1, 1, 2, 1, 1]
```

plus identified armature, dynamic friction, viscous friction, and a 10 ms motor
lag.

For presentation, keep these two leg comparisons separate:

- stock IsaacLab PD vs lag10 SysID: "how much better than a fresh IsaacLab G1
  baseline?"
- SDK-PD-only vs lag10 SysID: "how much better after matching the real
  controller gains?"

## Current primary results

Arm run folder:

```text
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/g1_arm_repro_full_20260521
```

| Arm config | Position RMSE | Velocity RMSE | Torque RMSE |
| --- | ---: | ---: | ---: |
| default PD | 0.035610 rad | 0.167895 rad/s | 1.382134 Nm |
| production GRU | 0.023289 rad | 0.136550 rad/s | 0.645107 Nm |
| reduction | 34.6% | 18.7% | 53.3% |

Leg run folder:

```text
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/g1_leg_repro_full_20260521
```

| Leg config | Position RMSE | Velocity RMSE | Torque RMSE |
| --- | ---: | ---: | ---: |
| default PD | 0.042210 rad | 0.126244 rad/s | 1.476786 Nm |
| v2 fixed-PD lag10 SysID | 0.016109 rad | 0.096804 rad/s | 1.013082 Nm |
| reduction | 61.8% | 23.3% | 31.4% |

Controller-matched leg baseline run folder:

```text
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/g1_leg_sdk_pd_full_20260521
```

Same interpolation scorer as the updated leg plots:

| Leg config | Position RMSE | Velocity RMSE | Torque RMSE |
| --- | ---: | ---: | ---: |
| stock IsaacLab PD | 0.042358 rad | 0.127944 rad/s | 1.477597 Nm |
| SDK PD only | 0.022756 rad | 0.137900 rad/s | 1.440068 Nm |
| v2 fixed-PD lag10 SysID | 0.015905 rad | 0.096691 rad/s | 1.014856 Nm |
| SDK PD improvement vs stock | 46.3% | -7.8% | 2.5% |
| lag10 SysID improvement vs SDK PD | 30.1% | 29.9% | 29.5% |

Lower RMSE is better. A positive reduction percentage means the tested model has
lower error than default PD.

## How overall RMSE is computed

The overall numbers are pooled per-motion RMSE numbers, not concatenated
rollouts.

For each recorded motion:

1. load real `state_motor.csv`
2. load sim `state_motor.csv`
3. normalize timestamps and keep the shared real/sim time window
4. interpolate sim position, velocity, and torque onto the real timestamps
5. compute squared error against the real signal for every scored time sample
   and every scored joint

For one signal, the aggregate is:

```text
overall_rmse = sqrt(
  sum_over_motions_time_joints((sim_interp - real)^2)
  / total_number_of_motion_time_joint_samples
)
```

This is equivalent to pooling all squared errors from the 10 independent
experiments, then taking one root mean square. It is not a plain arithmetic mean
of the 10 per-motion RMSE values, and it is not one long concatenated rollout.

The improvement percentage is:

```text
improvement = 100 * (default_pd_rmse - model_rmse) / default_pd_rmse
```

Command position is used to drive the replay and can be shown in plots, but it
is not the scored target for RMSE. The scored target is measured real robot
position, velocity, or torque from `state_motor.csv`.

## What interpolation means here

Interpolation is only a timestamp-resampling step. The real robot and simulator
logs are not guaranteed to write samples at exactly the same timestamps, even
when they replay the same command sequence. To compare them point-by-point, the
sim trace is linearly sampled at the real log timestamps inside the overlapping
time window.

The real and sim logs are both nominally 500 Hz, but nominal frequency is not
the same as identical sample times. The sim log is written on an exact control
grid, usually every `0.002` seconds. The real robot log has timestamp jitter and
occasional short/long intervals from the recording stack. In the current G1
outputs, row counts usually match, but the real `dt` range is not perfectly
constant. For example, checked current runs had real `dt` ranges around
`0.00027-0.0138` seconds while sim `dt` stayed at `0.002` seconds.

That is why timestamp-based scoring is preferred over row-index scoring. A
row-aligned sanity check is still useful, and it lands essentially the same for
the current leg run, but the primary metric should compare signals at the same
time, not merely at the same row number.

This is not time warping. The analysis does not stretch, compress, shift, or
phase-align the sim output to reduce error. It also does not choose a
model-specific offset. The time axes come from the recorded logs after timestamp
normalization, and the same scoring rule is applied to default PD and every
candidate model.

In plain terms:

- allowed: sample the sim curve at the real timestamps
- not allowed: slide the sim curve forward/backward until RMSE is smaller
- not allowed: stretch/compress the sim timeline to match peaks
- not allowed: score against command position instead of measured real position
- not allowed: drop bad motions or bad joints from the headline aggregate

## Why this is not funny business

The benchmark is meant to answer one question: given the same real command
trajectory and the same measured initial state, does the simulator reproduce the
measured real actuator response better than default PD?

The setup choices are there to avoid confounds, not to hide error:

- `--real-init-pose-sync` gives every model the measured row-0 state for that
  experiment. Without it, the metric includes arbitrary cold-start error from
  the sim's home pose.
- the 5 second buffer is applied to every model. It lets the fixed-base physics
  state settle before scoring begins.
- learned recurrent actuators reset before the buffer, then warm during the
  buffer. Resetting after the buffer would score an artificially cold recurrent
  state.
- each motion is run independently because each real CSV is an independent
  gantry capture.
- all 10 motions are included in the aggregate.
- the baseline and candidate models use the same real files, same command
  replay, same init-sync rule, same buffer rule, and same scoring code.

The main caveat is that this is a fixed-base gantry actuator benchmark, not a
free-walking whole-body benchmark. The old concatenated version can still be
reported separately as a long-horizon drift stress test, but it answers a
different question and should not be mixed with the primary per-motion RMSE.

## Plot outputs

Fresh plots from the same current clean benchmark outputs:

```text
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/g1_plots_current/arm_command_real_defaultpd_prodgru/g1_arm_command_real_defaultpd_prodgru_all_motions.pdf
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/g1_plots_current/arm_command_real_defaultpd_prodgru/g1_arm_defaultpd_vs_prodgru_rmse_summary.png
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/g1_plots_current/leg_real_stockpd_sdkpd_lag10sysid/g1_leg_real_stockpd_sdkpd_lag10sysid_all_motions.pdf
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/g1_plots_current/leg_real_stockpd_sdkpd_lag10sysid/g1_leg_stockpd_sdkpd_lag10sysid_rmse_summary.png
```

The arm trace plots show command, real, default PD, and production GRU. The leg
trace plots intentionally omit command position and show real, stock IsaacLab
PD, SDK-PD-only, and lag10 SysID; this keeps the leg figure focused on the
scored real-vs-sim comparison while showing the controller-gain caveat.

The plot script interpolates sim traces onto the real timeline for display and
per-motion plot labels. Because of that interpolation detail, the plot
aggregate can differ slightly from the benchmark summary numbers. The headline
numbers above remain the primary benchmark numbers.

## Why this is the correct primary benchmark

The G1 experiments were collected on a gantry/fixed-torso setup, not during free
walking. A fixed root/base benchmark is therefore the right sim setup for these
captures.

Each real capture is its own experiment. Running one motion at a time with
init-pose sync means every motion starts from its own measured row-0 state. That
removes a PD-gain-dependent cold-start bias and avoids turning the headline
metric into a long-horizon drift metric.

The 5 second buffer is still needed. Init-pose sync sets the scored joints to
the measured initial pose/velocity; the buffer lets the physics state and
learned recurrent actuator state settle before recorded commands begin.

For learned actuators, do not reset recurrent state after the buffer. The reset
happens before the buffer, then the buffer warms the hidden state and previous
torque feature. Resetting after the buffer measures a colder actuator state than
the model expects.

## What not to mix into the headline number

Do not mix these with the primary current per-motion result:

- old April concatenated scoring
- batch `--num-envs 10` scoring for the primary leg number
- runs without `--real-init-pose-sync`
- runs where recurrent state is reset after the buffer
- metrics computed from raw motor CSV timestamps before timestamp normalization

The old concatenated run is not useless. It is a valid long-episode drift stress
test. It answers a different question: "what happens if all motions are stitched
into one long rollout?" The primary current benchmark answers: "how well does
the actuator model reproduce each recorded gantry experiment from its measured
initial state?"

## Re-run commands

Arm:

```bash
cd /home/vbhavanantha/IsaacLab-Newton-g1-repro
source /home/vbhavanantha/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name g1_right_arm_default_pd \
  --motion-files input/sysid_data/g1/multijoint_v2_arm_test10_per_motion \
  --output-folder output/g1_arm_repro_full_20260521 \
  --real-init-pose-sync \
  --num-envs 1 \
  --headless \
  --experience /home/vbhavanantha/IsaacLab-Newton-g1-repro/apps/isaaclab.python.headless.kit

./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name g1_right_arm_fulltorque_enriched \
  --motion-files input/sysid_data/g1/multijoint_v2_arm_test10_per_motion \
  --output-folder output/g1_arm_repro_full_20260521 \
  --real-init-pose-sync \
  --num-envs 1 \
  --headless \
  --experience /home/vbhavanantha/IsaacLab-Newton-g1-repro/apps/isaaclab.python.headless.kit
```

Leg:

```bash
cd /home/vbhavanantha/IsaacLab-Newton-g1-repro
source /home/vbhavanantha/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name g1_right_leg_default_pd \
  --motion-files input/sysid_data/g1/multijoint_v2_leg_test10_per_motion \
  --output-folder output/g1_leg_repro_full_20260521 \
  --real-init-pose-sync \
  --num-envs 1 \
  --headless \
  --experience /home/vbhavanantha/IsaacLab-Newton-g1-repro/apps/isaaclab.python.headless.kit

./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name g1_right_leg_sdk_pd \
  --motion-files input/sysid_data/g1/multijoint_v2_leg_test10_per_motion \
  --output-folder output/g1_leg_sdk_pd_full_20260521 \
  --real-init-pose-sync \
  --num-envs 1 \
  --headless \
  --experience /home/vbhavanantha/IsaacLab-Newton-g1-repro/apps/isaaclab.python.headless.kit

./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name g1_right_leg_v2_fixedpd_lag10_sysid \
  --motion-files input/sysid_data/g1/multijoint_v2_leg_test10_per_motion \
  --output-folder output/g1_leg_repro_full_20260521 \
  --real-init-pose-sync \
  --num-envs 1 \
  --headless \
  --experience /home/vbhavanantha/IsaacLab-Newton-g1-repro/apps/isaaclab.python.headless.kit
```

## Sanity checks already done

- arm default PD and production GRU ran on all 10 arm motions
- leg stock default PD, SDK-PD-only, and lag10 SysID ran on all 10 leg motions
- leg GRU smoke and full run completed on the experimental branch
- raw G1 timestamp normalization is covered by tests
- Python compile and sim2real-gap tests passed after the benchmark fixes
- fresh command/real/default/best plots were generated from the current outputs
