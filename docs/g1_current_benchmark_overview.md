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
| right leg | `g1_right_leg_default_pd` | `g1_right_leg_v2_fixedpd_lag10_sysid` | 6 | no, fixed-PD lag10 SysID |

The leg GRU has been tested separately on
`vbhavanantha/g1-leg-gru-benchmark-test`. It runs mechanically and gives a tiny
torque improvement over lag10 SysID, but it worsens position and velocity RMSE,
so it is not the primary current leg model.

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

Lower RMSE is better. A positive reduction percentage means the tested model has
lower error than default PD.

## Plot outputs

Fresh plots from the same current clean benchmark outputs:

```text
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/g1_plots_current/arm_command_real_defaultpd_prodgru/g1_arm_command_real_defaultpd_prodgru_all_motions.pdf
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/g1_plots_current/arm_command_real_defaultpd_prodgru/g1_arm_defaultpd_vs_prodgru_rmse_summary.png
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/g1_plots_current/leg_command_real_defaultpd_lag10sysid/g1_leg_command_real_defaultpd_lag10sysid_all_motions.pdf
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/g1_plots_current/leg_command_real_defaultpd_lag10sysid/g1_leg_defaultpd_vs_lag10sysid_rmse_summary.png
```

The plot script interpolates sim and command traces onto the real timeline for
display and per-motion plot labels. Because of that interpolation detail, the
plot aggregate can differ slightly from the benchmark summary numbers. The
headline numbers above remain the primary benchmark numbers.

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
- leg default PD and lag10 SysID ran on all 10 leg motions
- leg GRU smoke and full run completed on the experimental branch
- raw G1 timestamp normalization is covered by tests
- Python compile and sim2real-gap tests passed after the benchmark fixes
- fresh command/real/default/best plots were generated from the current outputs
