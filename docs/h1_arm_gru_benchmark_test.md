# H1 Arm GRU Benchmark Test

This file documents the experimental H1 GRU branch:

```text
vbhavanantha/h1-arm-gru-benchmark-test
```

It starts from the clean H1 arm benchmark branch and adds one historical GRU
candidate on top of the same current per-motion benchmark methodology.

## Bottom Line

The tested GRU is stable and gives a small improvement over the current lag20
SysID model, but it is not a dramatic replacement for the simple SysID path.

| H1 right-arm config | Position RMSE | Velocity RMSE | Torque RMSE |
| --- | ---: | ---: | ---: |
| PD baseline | 0.017179 rad | 0.313199 rad/s | 1.159809 Nm |
| lag20 SysID | 0.011001 rad | 0.153816 rad/s | 0.702555 Nm |
| v28 lag20 SysID + GRU | 0.010921 rad | 0.149824 rad/s | 0.678022 Nm |
| v28 GRU reduction vs PD | 36.4% | 52.2% | 41.5% |
| v28 GRU improvement vs lag20 SysID | 0.7% | 2.6% | 3.5% |

Lower RMSE is better. The last row means the GRU branch is lower-error than the
current lag20 SysID by that percent.

## What Was Tested

Run config:

```text
input/run_configs/h1_arm_v28_lag20_gripperpos_s033/h1_arm_v28_lag20_gripperpos_s033.yaml
```

Actuator stack:

```text
h1_arm_post_velsync_sysid_gen10_lag20.yaml
+ h1_arm_tier1_enriched_v28_yawclip10_script.pt
```

This is not GRU-only. It keeps implicit PD and the lag20 SysID motor terms, then
adds the GRU as a residual feed-forward torque.

The GRU feature contract is the historical 24-feature H1 arm contract:

```text
[q, position_error, velocity, PD_hint, qfrc_bias, previous_torque]
```

The branch uses:

```text
pd_plus_gru: true
include_qfrc_bias: true
include_prev_torque: true
output_scale: [0.33, 0.33, 0.33, 0.33]
```

Old full-torque / GRU-only H1 paths were not used for the headline test because
the old records showed collapse-style torque RMSE. The practical candidate was
the residual PD+GRU path.

## Benchmark Method

This uses the same method as `docs/h1_arm_sim2real_benchmark_repro.md`:

- fixed-base H1 gantry benchmark
- yaw-clamped gripper-position USD
- 10 independent gripper test motions
- one motion per rollout, not one concatenated rollout
- `--real-init-pose-sync`
- 2 second buffer before replay
- recurrent GRU hidden state reset before each motion, then warmed during the
  buffer
- SAGE scoring over the same 4 right-arm joints

The scored joints are:

```text
right_elbow
right_shoulder_pitch
right_shoulder_roll
right_shoulder_yaw
```

The GRU internally uses the training order:

```text
right_shoulder_pitch
right_shoulder_roll
right_shoulder_yaw
right_elbow
```

That order difference is intentional. The benchmark scoring files define the
SAGE column order; the GRU patcher uses the network's feature order.

## Commands

Run the benchmark:

```bash
cd /home/vbhavanantha/IsaacLab-Newton-g1-repro
source /home/vbhavanantha/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name h1_arm_v28_lag20_gripperpos_s033 \
  --real-init-pose-sync \
  --headless \
  --experience /home/vbhavanantha/IsaacLab-Newton-g1-repro/apps/isaaclab.python.headless.kit
```

Run analysis:

```bash
python scripts/sim2real_gap/run_analysis.py \
  --robot-name h1_arm_v28_lag20_gripperpos_s033 \
  --valid-joints-file scripts/sim2real_gap/configs/h1_arm_v28_lag20_gripperpos_s033_valid_joints.txt
```

Generate the comparison plots:

```bash
python scripts/sim2real_gap/plot_h1_arm_gru_comparison.py
```

## Output Artifacts

Benchmark output:

```text
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_arm_gru_v28_test10_20260521
```

Analysis output:

```text
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_arm_gru_v28_analysis_test10_20260521/metrics_summary.xlsx
```

SAGE plots:

```text
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_arm_gru_v28_analysis_test10_20260521
```

Custom comparison plots, showing real, PD baseline, lag20 SysID, and v28
PD+GRU on the same axes:

```text
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_plots_current/arm_real_pd_lag20sysid_v28gru_labeled/h1_arm_real_pd_lag20sysid_v28gru_labeled_all_motions.pdf
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_plots_current/arm_real_pd_lag20sysid_v28gru_labeled/h1_arm_pd_lag20sysid_v28gru_labeled_rmse_summary.png
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_plots_current/arm_real_pd_lag20sysid_v28gru_labeled/h1_arm_pd_lag20sysid_v28gru_labeled_plot_metrics.csv
```

There is also one PNG per motion in:

```text
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_plots_current/arm_real_pd_lag20sysid_v28gru_labeled
```

The plot PDF has 10 pages, one per motion. The trace plots omit commanded
position and show the scored real-vs-sim position and torque signals. The page
titles use the SAGE RMSE means for that motion, including the GRU reduction
relative to PD and the GRU improvement relative to lag20 SysID.

## Per-Motion Torque RMSE

These are means over the 4 scored joints for each motion.

| Motion | PD tau RMSE | lag20 SysID tau RMSE | v28 GRU tau RMSE | v28 vs PD | v28 vs lag20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `C01_f0_1-0_5_a0_05` | 0.934289 | 0.588242 | 0.567512 | 39.3% | 3.5% |
| `C04_f0_1-0_5_a0_30` | 1.251539 | 0.645100 | 0.620972 | 50.4% | 3.7% |
| `C08_f0_1-1_0_a0_20` | 1.183580 | 0.694216 | 0.672086 | 43.2% | 3.2% |
| `C09_f0_1-1_0_a0_30` | 1.380733 | 0.864064 | 0.806129 | 41.6% | 6.7% |
| `C21_f0_1-0_5_a0_15_wave` | 1.073161 | 0.635325 | 0.637141 | 40.6% | -0.3% |
| `S01_f0_1_a0_15_inphase` | 1.036400 | 0.670757 | 0.650527 | 37.2% | 3.0% |
| `S06_f0_5_a0_05_inphase` | 0.905232 | 0.543534 | 0.541597 | 40.2% | 0.4% |
| `S09_f0_5_a0_30_inphase` | 1.296625 | 0.701603 | 0.700665 | 46.0% | 0.1% |
| `S14_f1_0_a0_15_wave` | 1.230804 | 0.805478 | 0.726498 | 41.0% | 9.8% |
| `S19_f1_0_a0_15_antiphase` | 1.305727 | 0.877231 | 0.857097 | 34.4% | 2.3% |

## Caveats

The GRU gain is small relative to the added complexity:

- extra TorchScript artifact
- extra stats JSON
- extra H1 MJCF dependency for online `qfrc_bias`
- recurrent hidden-state behavior to warm and reset correctly

Because of that, the current clean lag20 SysID branch remains the simpler
primary H1 result. This branch is useful if we want to show that the historical
H1 residual GRU path can be reproduced cleanly and gives a modest additional
torque improvement.
