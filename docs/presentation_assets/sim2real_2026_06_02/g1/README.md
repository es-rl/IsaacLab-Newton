# G1 Presentation Assets - 2026-05-29

Slide-ready plots for the current correct-robot G1 right-arm sim2real benchmark. RMSE bar plots use baseline PD in orange and the current model in blue. End-effector trajectory reveals match the SO101 style: real robot in black, baseline PD in red/orange, and the current model in blue.

## Headline Result

Aggregate over the 10 current G1 29-DOF tri-hand benchmark motions. Sim traces are interpolated to real robot timestamps per motion, RMSE is computed on the right 4 scored arm joints, then the overall row pools the squared error over all validation samples.

| Metric | Baseline PD | SysID + GRU | Reduction |
|:--|--:|--:|--:|
| Position RMSE | 0.0204 rad | 0.0119 rad | 41.6% |
| Torque RMSE | 0.742 Nm | 0.486 Nm | 34.5% |

Main plot:

- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_arm_before_after_pos_torque_redgreen.png`

Per-joint RMSE, mean across the 10 held-out validation motions:

- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_29dof_trihand_per_joint_position_rmse.png`
- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_29dof_trihand_per_joint_position_rmse_values.csv`
- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_29dof_trihand_per_joint_position_rmse_values.json`
- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_29dof_trihand_per_joint_torque_rmse.png`
- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_29dof_trihand_per_joint_torque_rmse_values.csv`
- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_29dof_trihand_per_joint_torque_rmse_values.json`

## Representative Motion

Representative motion: `H_F_06_multi_chirp`

| Metric | Baseline PD | SysID + GRU | Reduction |
|:--|--:|--:|--:|
| Position RMSE | 0.0175 rad | 0.0108 rad | 38.1% |
| Torque RMSE | 0.656 Nm | 0.439 Nm | 33.1% |
| FK end-effector RMSE | 12.6 mm | 8.1 mm | 36.2% |

Position reveal:

- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_H_F_06_multi_chirp_01_real_only.png`
- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_H_F_06_multi_chirp_02_real_baseline.png`
- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_H_F_06_multi_chirp_03_real_baseline_fulltorque_gru.png`

Torque reveal:

- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_H_F_06_multi_chirp_torque_01_real_only.png`
- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_H_F_06_multi_chirp_torque_02_real_baseline.png`
- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_H_F_06_multi_chirp_torque_03_real_baseline_fulltorque_gru.png`

End-effector 3D reveal:

- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_H_F_06_multi_chirp_ee_01_real_only.png`
- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_H_F_06_multi_chirp_ee_02_real_baseline.png`
- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_H_F_06_multi_chirp_ee_03_real_baseline_fulltorque_gru.png`
- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_H_F_06_multi_chirp_ee_metrics_so101_style.json`

End-effector XYZ backup:

- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_H_F_06_multi_chirp_ee_xyz_01_real_only.png`
- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_H_F_06_multi_chirp_ee_xyz_02_real_baseline.png`
- `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_H_F_06_multi_chirp_ee_xyz_03_real_baseline_fulltorque_gru.png`

The current slide assets above are PNG-first. This repo copy intentionally omits duplicate PDF exports.

## Source Files

- Metrics CSV: `/home/vbhavanantha/IsaacLab-Newton-g1-viewer/output/g1_29dof_trihand_10motion_results/g1_29dof_trihand_10motion_summary.csv`
- Per-joint metrics CSV: `/home/vbhavanantha/IsaacLab-Newton-g1-viewer/output/g1_29dof_trihand_10motion_results/g1_29dof_trihand_10motion_joint_metrics.csv`
- Real robot CSV: `/home/vbhavanantha/Desktop/UnitreeG1/data/experiments/presentation_H_F_06_multi_chirp_botharms_20260527_141707/state_motor.csv`
- Baseline PD sim CSV: `/home/vbhavanantha/IsaacLab-Newton-g1-viewer/output/g1_29dof_trihand_10motion_default_pd/sim/g1_29dof_right_arm_trihand_benchmark_default_pd/custom/g1_29dof_arm_implicit/H_F_06_multi_chirp/state_motor.csv`
- SysID + GRU sim CSV: `/home/vbhavanantha/IsaacLab-Newton-g1-viewer/output/g1_29dof_trihand_10motion_seed48/sim/g1_29dof_right_arm_trihand_benchmark_fulltorque_optuna_v2_row10_seed48/custom/g1_29dof_arm_no_pd/H_F_06_multi_chirp/state_motor.csv`
- Joint list: `/home/vbhavanantha/Desktop/UnitreeG1/data/experiments/presentation_H_F_06_multi_chirp_botharms_20260527_141707/joint_list.txt`
- FK USD: `/home/vbhavanantha/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0_tri_hand_closed.usda`
- Plot generator: `/home/vbhavanantha/IsaacLab-Newton-g1-viewer/scripts/sim2real_gap/plot_g1_29dof_trihand_so101_style_assets.py`
- Asset manifest: `/home/vbhavanantha/Desktop/UnitreeG1/docs/presentation_assets_20260529/g1_presentation_assets_manifest.json`

## Caveats

- The end-effector plot is FK-derived from logged 29-DOF tri-hand arm joints through the G1 29-DOF tri-hand USD kinematic chain. It is not an independent external mocap or camera measurement.
- EE FK targets `right_palm_link` on the fixed tri-hand. The right 4 shoulder/elbow joints are the scored benchmark slice; wrist joints are included in the FK trajectory from the 29-DOF log.
- The per-joint position/torque RMSE plots are arithmetic means of each joint's RMSE across the 10 held-out validation motions. That makes it easy to see which joint improved most, but it is not the same aggregation as the headline pooled overall RMSE.
- These are current per-motion benchmark stats, not one long concatenated episode.
