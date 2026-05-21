# G1 Arm Sim2Real Benchmark Repro

This branch is the lean repro path for the G1 right-arm benchmark from current
`origin/develop`. It intentionally brings over only the pieces needed to run:

- default SDK PD baseline: `g1_right_arm_default_pd`
- production full-torque GRU: `g1_right_arm_fulltorque_enriched`

The raw per-motion G1 arm CSVs are intentionally not tracked in this branch.
They were used for verification, but should be supplied locally when rerunning
the benchmark.

## Benchmark Method

Use the current per-motion benchmark, not the old concatenated April run.

1. Convert each `*_motor.csv` motion into SAGE `real/.../{control.csv,state_motor.csv,joint_list.txt}`.
2. Start the sim from the G1 fixed-base benchmark scene.
3. Run the 5 second buffer.
4. With `--real-init-pose-sync`, teleport scored joints to row 0 of the real
   `state_motor.csv`, including row-0 velocity when available.
5. Do not reset learned actuator recurrent state after the buffer. The GRU state
   is reset before the buffer, then the buffer warms it, then replay begins.
6. Replay the recorded real commands and score sim `state_motor.csv` against real
   `state_motor.csv`.

This avoids the two main old confounds:

- buffer-only start leaves a PD-gain-dependent initial pose error
- concatenating motions turns the metric into a long-horizon drift test

## Files Added

Runtime:

- `scripts/sim2real_gap/run_benchmark.py`
- `scripts/sim2real_gap/newton_benchmark.py`
- `scripts/sim2real_gap/motor_csv.py`
- `scripts/sim2real_gap/test/test_motor_csv.py`

G1 benchmark configs/assets:

- `input/run_configs/g1_right_arm_default_pd/g1_right_arm_default_pd.yaml`
- `input/run_configs/g1_right_arm_fulltorque_enriched/g1_right_arm_fulltorque_enriched.yaml`
- `input/actuator_models/g1/g1_arm_implicit.yaml`
- `input/actuator_models/g1/g1_arm_no_pd.yaml`
- `input/actuator_models/g1/g1_fulltorque_enriched_warmstart_script.pt`
- `input/actuator_models/g1/g1_fulltorque_enriched_warmstart_stats.json`
- `input/run_configs/g1_right_arm_full_sysid/g1_right_arm_full_sysid.yaml`
- `input/actuator_models/g1/g1_arm_sysid_full.yaml`
- `scripts/sim2real_gap/configs/g1_right_arm_default_pd_joints.yaml`
- `scripts/sim2real_gap/configs/g1_right_arm_default_pd_valid_joints.txt`
- `scripts/sim2real_gap/configs/g1_right_arm_fulltorque_enriched_joints.yaml`
- `scripts/sim2real_gap/configs/g1_right_arm_fulltorque_enriched_valid_joints.txt`

The `full_sysid` run config and actuator YAML are kept because the production
GRU qfrc-bias feature uses the same SysID mass/armature/friction parameters.

External data used for verification:

- `input/sysid_data/g1/multijoint_v2_arm_test10_per_motion`

The CSV fixture folder is ignored by the repo's `.gitignore` and is not part of
the GitHub branch. On the test machine it lives at:

`/home/vbhavanantha/IsaacLab-Newton-g1-repro/input/sysid_data/g1/multijoint_v2_arm_test10_per_motion`

The production GRU checkpoint is tracked via Git LFS.

## Commands

Use the IsaacLab conda environment. On this machine the clean worktree also
needs an explicit experience file path because current IsaacLab resolves Isaac
Sim 5 apps under `apps/isaacsim_5/`, while this repo has the kit file at
`apps/`.

```bash
cd /home/vbhavanantha/IsaacLab-Newton-g1-repro
source /home/vbhavanantha/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
```

Single-motion smoke:

```bash
./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name g1_right_arm_fulltorque_enriched \
  --motion-files input/sysid_data/g1/multijoint_v2_arm_test10_per_motion/T_A_01_wave_sine/T_A_01_wave_sine_motor.csv \
  --output-folder output/g1_repro_smoke \
  --real-init-pose-sync \
  --num-envs 1 \
  --headless \
  --experience /home/vbhavanantha/IsaacLab-Newton-g1-repro/apps/isaaclab.python.headless.kit
```

Full per-motion baseline:

```bash
./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name g1_right_arm_default_pd \
  --motion-files input/sysid_data/g1/multijoint_v2_arm_test10_per_motion \
  --output-folder output/g1_arm_repro_current \
  --real-init-pose-sync \
  --num-envs 1 \
  --headless \
  --experience /home/vbhavanantha/IsaacLab-Newton-g1-repro/apps/isaaclab.python.headless.kit
```

Full per-motion production GRU:

```bash
./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name g1_right_arm_fulltorque_enriched \
  --motion-files input/sysid_data/g1/multijoint_v2_arm_test10_per_motion \
  --output-folder output/g1_arm_repro_current \
  --real-init-pose-sync \
  --num-envs 1 \
  --headless \
  --experience /home/vbhavanantha/IsaacLab-Newton-g1-repro/apps/isaaclab.python.headless.kit
```

## Smoke Result

Single motion: `T_A_01_wave_sine`

| Config | Position RMSE | Torque RMSE |
| --- | ---: | ---: |
| default PD | 0.020480 rad | 0.693302 Nm |
| production GRU | 0.006954 rad | 0.292445 Nm |
| GRU reduction | 66.0% | 57.8% |

This is only a smoke check. The reported paper/standup numbers should come from
the full per-motion suite.

## Full 10-Motion Result

Run folder on this machine:

`/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/g1_arm_repro_full_20260521`

Scoring method: per-motion, 10 recorded arm motions, 4 right-arm joints,
`--real-init-pose-sync`, `--num-envs 1`, 5 second buffer, recurrent actuator
state warmed through the buffer.

| Config | Position RMSE | Velocity RMSE | Torque RMSE |
| --- | ---: | ---: | ---: |
| default PD | 0.035610 rad | 0.167895 rad/s | 1.382134 Nm |
| production GRU | 0.023289 rad | 0.136550 rad/s | 0.645107 Nm |
| GRU reduction | 34.6% | 18.7% | 53.3% |

Per-motion torque reductions:

| Motion | default PD τ RMSE | production GRU τ RMSE | Reduction |
| --- | ---: | ---: | ---: |
| `H_C_01_wave_sine` | 1.7201 | 0.9052 | 47.4% |
| `H_C_06_multi_chirp` | 1.6340 | 0.9665 | 40.9% |
| `H_E_06_multi_chirp` | 1.3387 | 0.6015 | 55.1% |
| `H_F_06_multi_chirp` | 1.1673 | 0.4358 | 62.7% |
| `T_A_01_wave_sine` | 0.7093 | 0.3038 | 57.2% |
| `T_A_02_multi_freq` | 0.7224 | 0.3528 | 51.2% |
| `T_B_02_multi_freq` | 1.3322 | 0.5326 | 60.0% |
| `T_B_05_slow_wave` | 1.3195 | 0.5229 | 60.4% |
| `T_D_02_multi_freq` | 1.5253 | 0.6997 | 54.1% |
| `T_D_04_multisine` | 1.6621 | 0.6562 | 60.5% |

## Timestamp Caveat

The raw historical G1 motor CSVs have a mixed-unit `time_s` column in the first
roughly 1 second of each capture: early rows are in microseconds, later rows are
in seconds. If those rows are converted literally, `run_analysis.py` sees a
non-monotonic/incorrect real timestamp series and interpolation-based RMSE can
be inflated.

This branch fixes that in `scripts/sim2real_gap/motor_csv.py` by normalizing the
raw motor CSV timestamps before writing SAGE `real/.../state_motor.csv` and
`control.csv`. The physics replay itself is not changed by this bug because the
sim motion file uses row order, but plots and analysis metrics need the fixed
timestamps.

The same timestamp caveat applies to the G1 leg raw per-motion CSVs. Leg-specific
docs live on the stacked leg branch.

Verification run after the fix:

```bash
python -m py_compile scripts/sim2real_gap/run_benchmark.py scripts/sim2real_gap/newton_benchmark.py scripts/sim2real_gap/motor_csv.py
python -m pytest scripts/sim2real_gap/test/test_motor_csv.py scripts/sim2real_gap/test/test_actuator_compat.py scripts/sim2real_gap/test/test_so101_benchmark_smoke.py -q
```

Result: `9 passed`.
