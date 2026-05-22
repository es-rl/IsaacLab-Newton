# SO-101 Balanced SysID Notes

This note documents the per-experiment balanced pooled SysID trial for SO-101.
It is separate from sim2real benchmarking: SysID fits actuator parameters on
training motions; benchmark/analysis must still be run on held-out motions.

For a first-time setup path, start with `docs/so101_onboarding.md`.

## What changed

The old SO-101 training corpus was effectively treated as one long concatenated
recording. That is workable for a first pass, but it has two problems:

- Longer recordings get more weight because they have more rows.
- The sim state can drift across artificial motion boundaries, so a later
  recording is scored partly on error accumulated during earlier recordings.

Balanced mode keeps one global parameter vector, but evaluates it motion by
motion. Every training motion starts from its own first measured real pose,
uses the normal settle buffer, then replays commands and produces one motion
loss. The optimizer score is the mean of those per-motion losses.

## Exact scoring

For each CMA-ES candidate and each motion:

1. Load `control.csv` and `state_motor.csv`.
2. Interpolate command position and measured position onto the requested
   control timestep over their shared timestamp range.
3. Reset joint position to measured row 0 and velocity to zero.
4. Hold the PD target at measured row 0 for `buffer_time`.
5. Replay recorded command positions.
6. Accumulate squared position error over joints and timesteps.
7. Divide by that motion's timestep count.

The candidate score is the mean of those motion losses. This is not cheating:
all candidates see the same train motions, same interpolation grid, same reset
rule, and same buffer rule. The reset is there to make each real experiment an
independent episode instead of pretending the separate recordings are one
continuous robot run.

## Important caveats

- Standard `--real-data-dir` still expects row-aligned CSVs. If control/state
  row counts differ by more than one row, it raises. The timestamp-aligned
  loader is used by `--balanced-real-data-root`.
- Use `--epsilon 0` for pilots. The balanced objective can have a small first
  generation score spread, and the default convergence threshold can stop too
  early.
- The current clean-repo SO-101 bounds optimize armature, Coulomb friction, and
  viscous friction only. The historical Desktop SO-101 fit used wider bounds
  that also optimized `stiffness` and `damping`, plus the upstream Feetech base
  actuator. Do not compare those two setups as if they were the same model.
- Armature, Coulomb friction, stiffness, and damping should be written through
  the Isaac Lab/Newton public APIs. Viscous friction still writes directly to
  `model.mujoco.dof_passive_damping` because there is no public writer for that
  MuJoCo-solver field.

## Smoke result

A two-motion diagnostic was run with the historical SO-101 bounds and upstream
Feetech base actuator:

- Motions: `actuator_bandwidth`, `boundary_exploration`
- Max replay length: 2500 samples per motion at 500 Hz
- Population: 8 candidates
- Generations: 1
- Output:
  `/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/sysid/so101/balanced_historical_bounds_sensitivity_2motion_20260521`

Result from `optimization_log.csv`:

- Best candidate MSE: `0.0006674961`
- Population mean MSE: `0.0016280088`
- Ratio mean/best: `2.44x`

That is the key validation signal: after fixing the parameter writer path,
different candidate parameters no longer produce nearly identical losses.

## Overnight fit

The 42-train-motion balanced run completed on 2026-05-22:

- Output:
  `/home/vbhavanantha/IsaacLab-Newton/output/sysid/so101/balanced_historical_bounds_42train_50gen_20260521`
- Split: seed 42, 42 train motions, 8 held-out validation motions
- Generations: `50/50`
- Runtime: `9h 0m 55s`
- Best train MSE: `0.0006280504`

The fitted actuator YAML was generated from `best_params.yaml` with the
upstream Feetech `effort_limit_sim=3.35` and `velocity_limit_sim=30.0`.

## Held-out validation result

The overnight fit was benchmarked against the upstream Feetech baseline on the
8 held-out motions:

`backlash_detection`, `coupled_joints`, `custom_motion`, `diagonal_sweep`,
`frequency_sweep`, `friction_gravity`, `hold_under_gravity`, `square_pattern`.

Validation method:

1. Convert each held-out SAGE folder into benchmark-ready per-motion inputs.
2. Timestamp-align command and measured state onto a 500 Hz grid over their
   shared time range.
3. Use fixed-base Newton replay with `--real-init-pose-sync`.
4. Replay each held-out motion independently, batched only for speed.
5. Compute pooled RMSE over all validation motions and all 6 SO-101 joints,
   sampled at `0.005 s`.

Important implementation detail: the handoff branch now keeps the balanced
42-train / 50-gen fit in `input/actuator_models/so101/so101_implicit.yaml` so
new SO-101 simulations use the fitted values by default. During the validation
run we temporarily swapped that file between the upstream baseline and the
balanced fit; the benchmark output folders are the durable record of which
parameters were loaded in each run.

Overall held-out results:

| Model | Position RMSE (rad) | Torque RMSE (Nm) |
|---|---:|---:|
| Upstream Feetech baseline | 0.045041 | 0.615712 |
| Balanced 42-train / 50-gen fit | 0.015612 | 0.322481 |
| Improvement | 65.34% reduction | 47.62% reduction |

Mean-of-motion RMSE, matching the balanced-loss intuition:

| Model | Position RMSE (rad) | Torque RMSE (Nm) |
|---|---:|---:|
| Upstream Feetech baseline | 0.036417 | 0.554951 |
| Balanced 42-train / 50-gen fit | 0.014782 | 0.285157 |
| Improvement | 59.41% reduction | 48.62% reduction |

Per-motion pooled RMSE:

| Motion | Upstream pos | Fit pos | Pos reduction | Upstream torque | Fit torque | Torque reduction |
|---|---:|---:|---:|---:|---:|---:|
| `backlash_detection` | 0.015182 | 0.006675 | 56.04% | 0.412617 | 0.543355 | -31.68% |
| `coupled_joints` | 0.050003 | 0.013781 | 72.44% | 0.834585 | 0.199206 | 76.13% |
| `custom_motion` | 0.039058 | 0.013807 | 64.65% | 0.621167 | 0.199879 | 67.82% |
| `diagonal_sweep` | 0.028576 | 0.016133 | 43.54% | 0.445325 | 0.261461 | 41.29% |
| `frequency_sweep` | 0.071783 | 0.021462 | 70.10% | 0.761400 | 0.374944 | 50.76% |
| `friction_gravity` | 0.025718 | 0.014020 | 45.48% | 0.401656 | 0.193365 | 51.86% |
| `hold_under_gravity` | 0.032655 | 0.017035 | 47.83% | 0.546685 | 0.244650 | 55.25% |
| `square_pattern` | 0.028362 | 0.015343 | 45.90% | 0.416174 | 0.264398 | 36.47% |

Files:

- Validation summary:
  `/home/vbhavanantha/IsaacLab-Newton/output/sysid/so101/balanced_historical_bounds_42train_50gen_20260521/validation_val8_summary.md`
- Validation per-motion metrics:
  `/home/vbhavanantha/IsaacLab-Newton/output/sysid/so101/balanced_historical_bounds_42train_50gen_20260521/validation_val8_metrics.csv`
- Upstream baseline benchmark:
  `/home/vbhavanantha/IsaacLab-Newton/output/sim2real_benchmark_so101_upstream_val8_20260522`
- Balanced fit benchmark:
  `/home/vbhavanantha/IsaacLab-Newton/output/sim2real_benchmark_so101_balanced_42train_val8_20260522`

## Held-out validation plots

Generated on 2026-05-22 with:

```bash
cd /home/vbhavanantha/IsaacLab-Newton-g1-repro
python scripts/sim2real_gap/plot_so101_validation_comparison.py \
    --baseline-root /home/vbhavanantha/IsaacLab-Newton/output/sim2real_benchmark_so101_upstream_val8_20260522 \
    --fit-root /home/vbhavanantha/IsaacLab-Newton/output/sim2real_benchmark_so101_balanced_42train_val8_20260522 \
    --out-dir /home/vbhavanantha/IsaacLab-Newton/output/so101_plots_val8_20260522
```

Main plot PDF:

`/home/vbhavanantha/IsaacLab-Newton/output/so101_plots_val8_20260522/so101_val8_real_vs_upstream_vs_balanced.pdf`

That PDF has one page per held-out motion. Each page overlays:

- real measured position/torque
- upstream Feetech baseline replay
- balanced SysID replay

Per-motion PNGs and plot-side metrics are in:

`/home/vbhavanantha/IsaacLab-Newton/output/so101_plots_val8_20260522`

The plotting script lives at:

`/home/vbhavanantha/IsaacLab-Newton-g1-repro/scripts/sim2real_gap/plot_so101_validation_comparison.py`

Torque caveat: the validation staging uses the source `torques_nm` column as
the staged SAGE `torques` field when available. That keeps torque RMSE in Nm.
The fit was position-loss only, so torque improvement is a validation signal,
not a directly optimized objective. `backlash_detection` improved position but
worsened torque; the overall torque RMSE still improved substantially.

## Production run shape

For a real SO-101 refit, use all train motions, historical or chosen final
bounds, the intended base actuator YAML, `--epsilon 0`, and enough generations
to see convergence. Then benchmark the resulting actuator YAML on held-out
motions. Do not treat the one-generation smoke output above as a fitted model.
