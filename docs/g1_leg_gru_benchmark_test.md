# G1 leg GRU benchmark test

For the cross-arm/leg primary benchmark source of truth, see
`docs/g1_current_benchmark_overview.md`.

Experimental branch: `vbhavanantha/g1-leg-gru-benchmark-test`

This branch tests the old G1 right-leg full-torque GRU on the same fixed-base/gantry benchmark setup used by the clean leg SysID branch. This is not walking. The torso/root is fixed, matching the physical gantry experiments.

## What was added

The clean leg branch only had default PD and `g1_right_leg_v2_fixedpd_lag10_sysid`. This test branch adds one GRU config:

`g1_right_leg_fulltorque_enriched_v2_iter1_lag10`

Files added:

- `input/run_configs/g1_right_leg_fulltorque_enriched_v2_iter1_lag10/g1_right_leg_fulltorque_enriched_v2_iter1_lag10.yaml`
- `input/actuator_models/g1/g1_leg_v2_fixedpd_lag10_no_pd.yaml`
- `input/actuator_models/g1/g1_leg_fulltorque_enriched_warmstart_v2_iter1_lag10_script.pt`
- `input/actuator_models/g1/g1_leg_fulltorque_enriched_warmstart_v2_iter1_lag10_stats.json`
- `scripts/sim2real_gap/configs/g1_right_leg_fulltorque_enriched_v2_iter1_lag10_valid_joints.txt`

The benchmark code now supports this leg GRU alias by patching the `legs` actuator group after simulation reset.

## Actuator path

The leg GRU is a full-torque model. It replaces solver-side leg PD, so the actuator YAML intentionally sets leg `kp=0` and `kd=0`.

Feature order:

```text
[q, position_error, velocity, SDK_PD_hint, qfrc_bias, previous_torque]
```

For six leg joints this is 36 inputs. The `SDK_PD_hint` block uses the fixed SDK leg gains:

```text
kp = [60, 60, 60, 100, 40, 40]
kd = [1, 1, 1, 2, 1, 1]
```

The deployed GRU uses online MuJoCo `qfrc_bias` from the G1 MJCF at the current simulated `(q, v)`. The recurrent hidden state and previous-torque feature are reset before each motion's 5 second buffer, then the buffer warms the recurrent state before the recorded command replay starts.

## Bugs caught while wiring this

Two wiring issues were caught by the one-motion smoke test:

- The new valid-joints file initially used Isaac joint names like `right_hip_pitch_joint`, but the converted G1 motor CSVs use short names like `right_hip_pitch`. This caused `No valid joints found`.
- The first patch condition also patched the arm group when running the leg GRU alias. That would have loaded a 6-output leg model into the arm path. The condition now patches arms only for non-leg aliases.

## Commands run

Use the IsaacLab conda env. Running from base Python fails because base does not have `warp`.

```bash
conda activate env_isaaclab
```

One-motion smoke:

```bash
./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name g1_right_leg_fulltorque_enriched_v2_iter1_lag10 \
  --motion-files input/sysid_data/g1/multijoint_v2_leg_test10_per_motion/T_A_01_wave_sine/T_A_01_wave_sine_motor.csv \
  --output-folder output/g1_leg_gru_smoke_20260520_v2 \
  --real-init-pose-sync \
  --num-envs 1 \
  --headless \
  --experience /home/vbhavanantha/IsaacLab-Newton-g1-repro/apps/isaaclab.python.headless.kit
```

Full 10-motion run:

```bash
./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name g1_right_leg_fulltorque_enriched_v2_iter1_lag10 \
  --motion-files input/sysid_data/g1/multijoint_v2_leg_test10_per_motion \
  --output-folder output/g1_leg_gru_full_20260520 \
  --real-init-pose-sync \
  --headless \
  --experience /home/vbhavanantha/IsaacLab-Newton-g1-repro/apps/isaaclab.python.headless.kit
```

## Results

Metric method: per-motion interpolation RMSE, reported below as aggregate RMSE over the 10 independent motions. This is the same current benchmark style as the clean leg branch, not a concatenated long episode.

| Config | Position RMSE rad | Velocity RMSE rad/s | Torque RMSE Nm | Position improvement vs default | Torque improvement vs default |
| --- | ---: | ---: | ---: | ---: | ---: |
| default PD | 0.042212 | 0.126663 | 1.475396 | baseline | baseline |
| lag10 SysID | 0.016110 | 0.097357 | 1.013385 | 61.8% | 31.3% |
| lag10 full-torque GRU | 0.018001 | 0.102239 | 0.992895 | 57.4% | 32.7% |

Compared directly against lag10 SysID:

| Comparison | Position | Velocity | Torque |
| --- | ---: | ---: | ---: |
| GRU vs lag10 SysID | 11.7% worse | 5.0% worse | 2.0% better |

Per-motion position and torque RMSE:

| Motion | default pos | default tau | lag10 SysID pos | lag10 SysID tau | GRU pos | GRU tau |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T_B_05_slow_wave | 0.044684 | 1.570362 | 0.015638 | 0.944435 | 0.018885 | 0.920638 |
| T_D_02_multi_freq | 0.065291 | 1.599715 | 0.020108 | 1.302985 | 0.021374 | 1.279036 |
| H_C_06_multi_chirp | 0.032013 | 1.287287 | 0.016764 | 1.125777 | 0.018752 | 1.086905 |
| H_E_06_multi_chirp | 0.035506 | 1.612502 | 0.015091 | 1.087195 | 0.017183 | 1.056369 |
| T_A_02_multi_freq | 0.019540 | 1.483848 | 0.014016 | 0.807275 | 0.015422 | 0.820484 |
| H_C_01_wave_sine | 0.034519 | 1.240787 | 0.015638 | 0.837814 | 0.016489 | 0.829521 |
| H_F_06_multi_chirp | 0.043919 | 1.610871 | 0.017761 | 1.192721 | 0.019758 | 1.160442 |
| T_B_02_multi_freq | 0.045258 | 1.561152 | 0.012981 | 0.790439 | 0.016782 | 0.792095 |
| T_A_01_wave_sine | 0.019083 | 1.292803 | 0.016417 | 0.896067 | 0.017785 | 0.878247 |
| T_D_04_multisine | 0.057954 | 1.428544 | 0.015598 | 1.013139 | 0.016791 | 0.985377 |

## Interpretation

The leg GRU works mechanically in the current benchmark path: it runs all 10 motions, uses the intended no-PD actuator path, and does not explode.

It is not a clear production improvement over lag10 SysID. It slightly improves aggregate torque RMSE, but it worsens position and velocity tracking. For the current G1 leg benchmark, the lean primary result should remain `g1_right_leg_v2_fixedpd_lag10_sysid` unless the goal is specifically torque RMSE and the position regression is acceptable.

Raw outputs are local only and intentionally not tracked in git:

```text
/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/g1_leg_gru_full_20260520
```
