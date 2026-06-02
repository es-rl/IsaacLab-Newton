# H1 Presentation Assets - 2026-05-29

These files are slide-ready assets for the H1 arm sim2real section. RMSE bar plots use baseline PD in orange and the current model in blue. Trajectory reveals match the G1/SO101 style: real robot in black, baseline PD in red/orange, and SysID + GRU in blue. The current slide assets are PNG-first; this repo copy intentionally omits duplicate PDF exports.

## Headline H1 stats

| Config | Position RMSE | Torque RMSE |
|---|---:|---:|
| Baseline PD | 0.017179 rad | 1.159809 Nm |
| SysID + GRU | 0.010921 rad | 0.678022 Nm |
| Reduction vs baseline PD | 36.43% | 41.54% |

Use this plot for the H1 stats slide:

`/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_before_after_pos_torque_redgreen.png`

Per-joint RMSE, mean across the 10 held-out validation motions:

- `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_arm_per_joint_position_rmse.png`
- `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_arm_per_joint_position_rmse_values.csv`
- `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_arm_per_joint_position_rmse_values.json`
- `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_arm_per_joint_torque_rmse.png`
- `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_arm_per_joint_torque_rmse_values.csv`
- `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_arm_per_joint_torque_rmse_values.json`

## Trajectory reveal

Representative held-out motion: `C04_f0_1-0_5_a0_30`, titled in plots as "Right-Arm Chirp Sweep."

Use these in order for a slide animation:

1. `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_C04_f0_1-0_5_a0_30_01_real_only.png`
2. `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_C04_f0_1-0_5_a0_30_02_real_baseline.png`
3. `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_C04_f0_1-0_5_a0_30_03_real_baseline_model.png`

## Metric convention

The benchmark interpolates sim traces onto the real-data timestamps, computes
RMSE per joint, then averages the per-joint RMSEs. This matches the current H1
benchmark summary CSV:

`/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_plots_current/arm_real_pd_lag20sysid_v28gru_labeled/h1_arm_pd_lag20sysid_v28gru_labeled_plot_metrics.csv`

Source paths and recomputed per-joint numbers are recorded in:

`/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_presentation_assets_manifest.json`

The per-joint plots are arithmetic means of each joint's RMSE across the 10 held-out validation motions. That is useful for seeing where the model improves most, but it is separate from the headline overall row.

## Torque reveal

Use these in order for a torque slide animation:

1. `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_C04_f0_1-0_5_a0_30_torque_01_real_only.png`
2. `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_C04_f0_1-0_5_a0_30_torque_02_real_baseline.png`
3. `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_C04_f0_1-0_5_a0_30_torque_03_real_baseline_model.png`

The title shows the canonical torque RMSE reduction for this held-out motion: `1.252 -> 0.621 Nm`, `50.4%` reduction.

## H1 end-effector position reveal

The H1 EE plots use MuJoCo forward kinematics on `right_gripper_link` with the logged joint positions, plotted in mm. This is the gripper-link path implied by the joint logs, not an independent camera or motion-capture measurement.

Use these 3D trajectory slides in order:

1. `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_C04_f0_1-0_5_a0_30_ee_01_real_only.png`
2. `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_C04_f0_1-0_5_a0_30_ee_02_real_baseline.png`
3. `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_C04_f0_1-0_5_a0_30_ee_03_real_baseline_sysid_gru.png`

Backup XYZ time-trace slides:

1. `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_C04_f0_1-0_5_a0_30_ee_xyz_01_real_only.png`
2. `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_C04_f0_1-0_5_a0_30_ee_xyz_02_real_baseline.png`
3. `/home/vbhavanantha/Desktop/UnitreeH1/docs/presentation_assets_20260529/h1_C04_f0_1-0_5_a0_30_ee_xyz_03_real_baseline_sysid_gru.png`

For this motion, FK-derived EE RMSE is `22.0 -> 11.5 mm`, `47.5%` reduction.

## Source Files

- Plot generator: `/home/vbhavanantha/IsaacLab-Newton-g1-viewer/scripts/sim2real_gap/plot_h1_so101_style_assets.py`
- Metrics CSV: `/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_plots_current/arm_real_pd_lag20sysid_v28gru_labeled/h1_arm_pd_lag20sysid_v28gru_labeled_plot_metrics.csv`
- Real robot CSV: `/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_arm_repro_test10_20260521/real/h1_arm_pd_baseline_yawclamped_gripperpos/multijoint_gripper_test10/C04_f0_1-0_5_a0_30/state_motor.csv`
- Baseline PD sim CSV: `/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_arm_repro_test10_20260521/sim/h1_arm_pd_baseline_yawclamped_gripperpos/multijoint_gripper_test10/h1_arm_implicit/C04_f0_1-0_5_a0_30/state_motor.csv`
- SysID + GRU sim CSV: `/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_arm_gru_v28_test10_20260521/sim/h1_arm_v28_lag20_gripperpos_s033/multijoint_gripper_test10/h1_arm_post_velsync_sysid_gen10_lag20/C04_f0_1-0_5_a0_30/state_motor.csv`
- Joint list: `/home/vbhavanantha/IsaacLab-Newton-g1-repro/output/h1_arm_repro_test10_20260521/real/h1_arm_pd_baseline_yawclamped_gripperpos/multijoint_gripper_test10/C04_f0_1-0_5_a0_30/joint_list.txt`
- FK MuJoCo XML: `/home/vbhavanantha/IsaacLab-Newton/input/robot_models/h1_with_gripper/h1_with_gripper_groundtruth_yawclamped.xml`
