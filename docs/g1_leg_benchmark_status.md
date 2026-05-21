# G1 Leg Sim2Real Benchmark Repro

This branch adds the lean G1 right-leg benchmark surface on top of the clean G1
arm repro branch. It is intentionally stacked separately from the arm PR so the
arm reproduction can stay small and reviewable.

Branch:

`vbhavanantha/g1-leg-sim2real-benchmark-repro`

Base branch:

`vbhavanantha/g1-sim2real-benchmark-repro`

## What Was Ported

Minimal leg inputs copied from the older working repo:

- `input/run_configs/g1_right_leg_default_pd/g1_right_leg_default_pd.yaml`
- `input/run_configs/g1_right_leg_v2_fixedpd_lag10_sysid/g1_right_leg_v2_fixedpd_lag10_sysid.yaml`
- `input/actuator_models/g1/g1_leg_implicit.yaml`
- `input/actuator_models/g1/g1_leg_v2_fixedpd_lag10_sysid.yaml`
- leg joint/valid-joint configs under `scripts/sim2real_gap/configs/`
- G1 leg alias and actuator wiring in `scripts/sim2real_gap/newton_benchmark.py`

Only the two configs needed for the clean current comparison were ported:

- baseline: `g1_right_leg_default_pd` (stock IsaacLab PD gains)
- best current leg model: `g1_right_leg_v2_fixedpd_lag10_sysid`

## Current Benchmark Rule

Use the same primary benchmark rule as the clean G1 arm repro:

1. one motion at a time, not concatenated historical scoring
2. `--real-init-pose-sync`
3. 5 second buffer settle from the configured benchmark default
4. raw G1 motor CSV timestamps normalized by `scripts/sim2real_gap/motor_csv.py`
5. interpolation-based RMSE over the shared real/sim time window

Do not use `--num-envs 10` for the primary current leg number. The batch path
runs all motions in parallel, but it does not take the same per-motion
init-sync path as the sequential runner. It is useful for speed checks, not for
the current reported metric.

The raw leg `*_motor.csv` captures have the same mixed timestamp convention as
the arm data: the early rows are microsecond-scale and later rows are
second-scale. The converter normalizes those values, then writes SAGE CSV
timestamps in microseconds.

The raw leg CSV fixture folder is intentionally not tracked in this GitHub
branch. It was used for verification from the local ignored path:

`/home/vbhavanantha/IsaacLab-Newton-g1-repro/input/sysid_data/g1/multijoint_v2_leg_test10_per_motion`

## Full Clean-Branch Run

Output folder:

`/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/g1_leg_repro_full_20260521`

Commands:

```bash
cd /home/vbhavanantha/IsaacLab-Newton-g1-repro
source /home/vbhavanantha/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name g1_right_leg_default_pd \
  --motion-files input/sysid_data/g1/multijoint_v2_leg_test10_per_motion \
  --output-folder output/g1_leg_repro_full_20260521 \
  --real-init-pose-sync \
  --headless \
  --experience /home/vbhavanantha/IsaacLab-Newton-g1-repro/apps/isaaclab.python.headless.kit

./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name g1_right_leg_v2_fixedpd_lag10_sysid \
  --motion-files input/sysid_data/g1/multijoint_v2_leg_test10_per_motion \
  --output-folder output/g1_leg_repro_full_20260521 \
  --real-init-pose-sync \
  --headless \
  --experience /home/vbhavanantha/IsaacLab-Newton-g1-repro/apps/isaaclab.python.headless.kit
```

Both runs completed on 10/10 motions. Each run populated init pose and init
velocity for 10/10 motions.

## Full-Run Metrics

Primary metric: interpolation RMSE over the shared real/sim time window, pooled
over all 10 motions and all six right-leg joints.

| Config | Position RMSE | Velocity RMSE | Torque RMSE | Position change vs default | Velocity change vs default | Torque change vs default |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default PD | 0.042210 rad | 0.126244 rad/s | 1.476786 Nm | baseline | baseline | baseline |
| v2 fixed-PD lag10 SysID | 0.016109 rad | 0.096804 rad/s | 1.013082 Nm | 61.8% lower | 23.3% lower | 31.4% lower |

Row-aligned scoring is a sanity check and lands essentially the same:

| Config | Position RMSE | Velocity RMSE | Torque RMSE | Position change vs default | Velocity change vs default | Torque change vs default |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default PD | 0.042364 rad | 0.127992 rad/s | 1.477572 Nm | baseline | baseline | baseline |
| v2 fixed-PD lag10 SysID | 0.015898 rad | 0.096671 rad/s | 1.014909 Nm | 62.5% lower | 24.5% lower | 31.3% lower |

## Per-Motion Breakdown

Interpolation RMSE:

| Motion | Default pos | SysID pos | Pos change | Default torque | SysID torque | Torque change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `H_C_01_wave_sine` | 0.034517 | 0.015637 | 54.7% lower | 1.241570 | 0.837389 | 32.6% lower |
| `H_C_06_multi_chirp` | 0.032012 | 0.016764 | 47.6% lower | 1.287860 | 1.125565 | 12.6% lower |
| `H_E_06_multi_chirp` | 0.035505 | 0.015090 | 57.5% lower | 1.614009 | 1.087023 | 32.7% lower |
| `H_F_06_multi_chirp` | 0.043917 | 0.017761 | 59.6% lower | 1.612198 | 1.192518 | 26.0% lower |
| `T_A_01_wave_sine` | 0.019082 | 0.016416 | 14.0% lower | 1.292502 | 0.895649 | 30.7% lower |
| `T_A_02_multi_freq` | 0.019539 | 0.014013 | 28.3% lower | 1.483670 | 0.806790 | 45.6% lower |
| `T_B_02_multi_freq` | 0.045256 | 0.012979 | 71.3% lower | 1.563462 | 0.789957 | 49.5% lower |
| `T_B_05_slow_wave` | 0.044682 | 0.015636 | 65.0% lower | 1.572626 | 0.944067 | 40.0% lower |
| `T_D_02_multi_freq` | 0.065289 | 0.020108 | 69.2% lower | 1.602094 | 1.302766 | 18.7% lower |
| `T_D_04_multisine` | 0.057951 | 0.015597 | 73.1% lower | 1.431284 | 1.012884 | 29.2% lower |

## Interpretation

The current clean-branch leg result is stronger than the earlier copied
working-repo note:

- position improves by about 62%
- torque improves by about 31%
- velocity improves by about 23%

This is now a clean current-repo reproduction, not a historical April-style
concatenated run. The old concatenated setup can still be useful as a long
episode stress test, but it should not be mixed with this current per-motion
metric when reporting improvement percentages.
