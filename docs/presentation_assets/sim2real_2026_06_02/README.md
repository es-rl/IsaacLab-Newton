# Sim2Real Presentation Plot Assets - 2026-06-02

Slide-ready plots for the actuator SysID / GRU sim2real presentation.

This folder intentionally contains only curated presentation assets:

- PNG plots used for slides or slide transitions.
- Small JSON/CSV metric files used as provenance for plotted values.
- No raw robot logs, no simulation output folders, no videos, no PDFs, and no experimental model dumps.

## Folder Map

| Folder | Contents |
|---|---|
| `g1/` | G1 29-DOF tri-hand right-arm plots: aggregate RMSE, per-joint RMSE, representative trajectory reveals, and FK end-effector trajectory reveals. |
| `h1/` | H1 right-arm plots: aggregate RMSE, per-joint RMSE, representative trajectory reveals, and FK end-effector trajectory reveals. |
| `so101/` | SO-101 held-out validation and HF vial plots, including episode 204 end-effector slide reveal. |
| `teststand/` | Newton teststand before/after SysID plot and the zoomed current/voltage chirp response plot. |

## Main Slide Assets

Use these for platform-specific slides:

- `g1/g1_29dof_trihand_per_joint_position_rmse.png`
- `g1/g1_29dof_trihand_per_joint_torque_rmse.png`
- `g1/g1_H_F_06_multi_chirp_ee_03_real_baseline_fulltorque_gru.png`
- `h1/h1_arm_per_joint_position_rmse.png`
- `h1/h1_arm_per_joint_torque_rmse.png`
- `h1/h1_C04_f0_1-0_5_a0_30_ee_03_real_baseline_sysid_gru.png`
- `so101/so101_per_joint_position_rmse_deg_presentation.png`
- `so101/so101_hf_vials_ee_rmse_tri_presentation.png`
- `teststand/teststand_sysid_pos_torque_before_after_presentation.png`

## Methodology Summary

All benchmark stats are per-motion, not one long concatenated episode.

For each motion:

1. The sim starts from the recorded real initial pose when that robot benchmark supports init-pose sync.
2. The recorded command trajectory is replayed in Newton.
3. Sim traces are interpolated onto the real robot timestamps.
4. RMSE is computed against real robot measurements for the relevant scored joints.
5. Aggregate rows pool or average the held-out validation motions as recorded in the metric JSON/CSV files next to each plot.

End-effector plots are forward-kinematics-derived from logged joint positions. They are useful for visualization and task-space interpretation, but they are not independent mocap or camera measurements.

## Headline Values

| Platform | Model | Position RMSE Reduction | Torque RMSE Reduction | Validation Set |
|---|---|---:|---:|---|
| G1 arm | SysID + GRU | 41.6% | 34.5% | 10 held-out G1 29-DOF tri-hand right-arm motions |
| H1 arm | SysID + GRU | 36.4% | 41.5% | 10 held-out H1 right-arm motions |
| SO-101 | SysID | 65.3% | 47.6% | 8 held-out custom validation motions |
| Newton teststand | SysID | 16.0% | 43.7% | 10 held-out chirp motions, first 10 s trimmed |

The exact plotted values live in the adjacent `*_values.json`, `*_metrics.json`, and `*_values.csv` files.

## Local Provenance

These assets were copied from the local presentation/doc folders on 2026-06-02:

- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/`
- `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/`
- `/home/vbhavanantha/Desktop/Lerobot SysID/docs/`
- `/home/vbhavanantha/Desktop/newton-teststand/benchmark_results/`
