# Newton Benchmark & Analysis

Sim-to-real actuator gap estimation with the Newton physics backend. Two-stage workflow:

- **Benchmark** (`run_benchmark.py`): Replays joint motion trajectories in Newton simulation across multiple actuator models (implicit PD, DC motor, LSTM/GRU) and records sim joint states as CSVs.
- **Analysis** (`run_analysis.py`): Compares sim vs real joint data — computes per-joint RMSE, correlation, cosine similarity, and generates comparison plots and metrics.

Based on: [SAGE](https://github.com/isaac-sim2real/sage)

## Overview

Plays back motion trajectories in simulation, records joint states as CSV, and compares them against real robot data. This integration adapts the upstream simulation module for the Newton physics engine (replacing PhysX) with some improvements to the process.

## Quick Start (Teststand)

Minimal single-motor environment (base cylinder + arm bar + revolute elbow joint) for actuator model validation. Uses the same M8010-6.333 motor as the H1 elbow.

### 1. Configure data paths

Edit `input/run_configs/teststand/teststand.yaml`:

```yaml
benchmark:
  motion_files: input/motion_files/h1/motor_benchtop/SysID Position 0/chirp_type3
  motion_name: chirp_type3_pos0

actuator:
  model_type: fmu
  yaml_file: teststand/teststand_implicit.yaml
  fmu_path: h1/nvidia-motor.fmu
  fmu_step_size: 0.002
```

The teststand actuator YAML uses scalar values (no per-joint regex patterns) since there is only one joint (`elbow`).

### 2. Run the benchmark

Input data can be benchtop parquet files or motor CSVs — parquet files are auto-detected and converted.

```bash
python scripts/sim2real_gap/run_benchmark.py --robot-name teststand --headless
```

### 3. Run analysis

```bash
python scripts/sim2real_gap/run_analysis.py --robot-name teststand
```

## Quick Start (H1)

Most settings (motion files, output paths, actuator config) are in `input/run_configs/h1/h1.yaml`. Set them once and run scripts with minimal CLI args.

### 1. Configure data paths

Edit `input/run_configs/h1/h1.yaml`:

```yaml
benchmark:
  motion_files: input/motion_files/h1/arm_march2026  # motor CSVs or motion .txt files
  output_folder: output/sim2real_benchmark

analysis:
  output_dir: output/sim2real_analysis
  sample_dt: 0.005
```

---
### 2. Run the benchmark

The script auto-detects `.csv` vs `.txt` input and converts sysid `.csv` recordings automatically. Run one of the following commands:

```bash
# from config (motion_files set in YAML)
python scripts/sim2real_gap/run_benchmark.py --robot-name h1 --headless

# explicit motion file
python scripts/sim2real_gap/run_benchmark.py \
    --robot-name h1 \
    --motion-files input/motion_files/h1/custom/arm_reach.txt \
    --output-folder output/sim2real_benchmark \
    --original-control-freq 50 \
    --headless

# replay exact real-robot commands
python scripts/sim2real_gap/run_benchmark.py \
    --robot-name h1 \
    --motion-name motion_stand_left_arm_swing \
    --real-control-csv output/sim2real_benchmark/real/h1/custom/motion_stand_left_arm_swing/control.csv.bak \
    --original-control-freq 100 \
    --headless
```

Drop `--headless` and add `--visualizer newton` to visualize.

### Manually convert data

For raw H1 data not auto-converted (seconds timestamps, bare CSV control):

```bash
python scripts/sim2real_gap/convert_real_data.py \
    --data-dir output/sim2real_benchmark/real/h1/custom/motion_name
```

Converts `state_motor.csv` (s→μs), `control.csv` (bare→SAGE format), and creates `event.csv`. Originals backed up as `.bak`.

For sysid motor CSVs, use `convert_h1_chirp_to_csv.py` (see [sysid README](../sysid/README.md)).

---

### 3. Run analysis (sim vs real)

```bash
# from config (paths set in YAML)
python scripts/sim2real_gap/run_analysis.py --robot-name h1

# override paths
python scripts/sim2real_gap/run_analysis.py \
    --robot-name h1 \
    --result-folder output/sim2real_benchmark \
    --output-dir output/sim2real_analysis \
    --sample-dt 0.005

# filter by motion name (glob patterns supported)
python scripts/sim2real_gap/run_analysis.py --robot-name h1 --motion-names "*elbow*"
```

`--motion-names` accepts comma-separated names or glob patterns. Base names auto-resolve to actuator-suffixed sim folders. Can also be set in the run config YAML (`analysis.motion_names`); pass `--motion-names "*"` to override and analyze all.

## Quick Start (UR10e)

### 1. Convert real data from pickle

UR10e real data is recorded as a trajectory pickle from `isaac_manipulator_data_utils`. Convert to the expected format:

```bash
python scripts/sim2real_gap/convert_ur10e_pkl.py \
    --pkl-path output/sim2real_benchmark_ur10e/real/2026-01-26_14-48-58_rosbag2/trajectory.pkl \
    --output-dir output/sim2real_benchmark_ur10e/real/ur10e/custom/rosbag_2026_01_26
```

This produces `control.csv`, `state_motor.csv`, `joint_list.txt`, and `event.csv` with all 6 UR10e joints in canonical order.

### 2. Run benchmark & analysis

Same workflow as H1 — just pass `--robot-name ur10e`. Frequencies default to 500Hz from `input/run_configs/ur10e/ur10e.yaml`.

```bash
# from config (paths set in ur10e.yaml)
python scripts/sim2real_gap/run_benchmark.py --robot-name ur10e --headless

# analysis
python scripts/sim2real_gap/run_analysis.py --robot-name ur10e
```

Analysis produces per-joint comparison plots (with RMSE), boxplots, and a metrics Excel file (`metrics_summary.xlsx`). Output folders are suffixed with the actuator model type.

---

## Quick Start (SO-101)

> **Note:** SO-101 benchmark dispatch (the `So101BenchmarkSceneCfg` scene
> and its `_BENCHMARK_ROBOT_CONFIGS["so101"]` entry) is added by a
> companion change to `newton_benchmark.py`. If `--robot-name so101`
> raises `ValueError: Unknown robot 'so101'`, that change has not yet
> landed on your branch — pull `develop` and try again. SO-101 sysid
> (`scripts/sysid/README.md#so-101`) is unaffected and works
> independently.

Validate SO-101 sysid output against held-out real motor data. Assumes you
have already produced fitted parameters via
[`scripts/sysid/README.md`](../sysid/README.md#so-101); the benchmark
replays a real-robot recording in Newton sim and the analysis step
computes per-joint sim-vs-real metrics.

### 1. Configure data paths

Edit `input/run_configs/so101/so101.yaml` and replace the `<motion_dir>`
placeholders:

```yaml
benchmark:
  # Path can be anywhere on disk — the shipped placeholder lives under
  # input/motion_files/so101/<motion_dir>; this example uses sysid_data/
  # to reuse the same recordings produced for sysid.
  motion_files: input/sysid_data/so101/my_holdout_motion   # was <motion_dir>
  motion_name: my_holdout_motion                           # was customer_motion
  output_folder: output/sim2real_benchmark

analysis:
  output_dir: output/sim2real_analysis
  sample_dt: 0.005
  motion_names: "*"

actuator:
  model_type: implicit
  yaml_file: so101/so101_implicit.yaml         # the YAML you populated from sysid
```

`motion_files` should be a SAGE-format directory (`control.csv`,
`state_motor.csv`, `joint_list.txt`) — the same format consumed by sysid.

### 2. Run the benchmark

```bash
# from config (paths set in YAML)
python scripts/sim2real_gap/run_benchmark.py --robot-name so101 --headless

# explicit motion dir
python scripts/sim2real_gap/run_benchmark.py \
    --robot-name so101 \
    --motion-files input/sysid_data/so101/my_holdout_motion \
    --motion-name my_holdout_motion \
    --output-folder output/sim2real_benchmark \
    --headless
```

Drop `--headless` and add `--visualizer newton` to view the playback in
the Newton viewer (requires `pip install isaaclab_visualizers[newton]`).

The benchmark writes sim joint trajectories to
`output/sim2real_benchmark/sim/so101/<motion_name>_implicit/` and stages
the real recording at
`output/sim2real_benchmark/real/so101/<motion_name>/` (auto-converted to
SAGE if needed).

### 3. Run analysis

```bash
# from config (paths set in YAML)
python scripts/sim2real_gap/run_analysis.py --robot-name so101

# override paths
python scripts/sim2real_gap/run_analysis.py \
    --robot-name so101 \
    --result-folder output/sim2real_benchmark \
    --output-dir output/sim2real_analysis \
    --sample-dt 0.005
```

Output goes to `output/sim2real_analysis/metrics/so101/<motion_name>/`:
per-joint position/velocity/torque comparison plots, a
`metrics_summary.xlsx` with RMSE / cosine similarity / correlation per
joint, and a boxplot summary. Compare against the same metrics produced
from the unfit STS3215 template to quantify the sim-to-real improvement
delivered by your CMA-ES fit.

### Notes

- `scripts/sim2real_gap/configs/so101_joints.yaml` and
  `so101_valid_joints.txt` enumerate the six USD joint names
  (`Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw`) for SAGE's
  strict joint-name matching. They ship with the toolbox; you do not
  need to edit them.
- The SO-101 USD is fixed-base, so `--fix-root` is the default and there
  is no torso/pelvis joint to suppress.
- Motor torque in `state_motor.csv` must be in N·m; if your driver
  reports current (mA) you must convert before benchmarking, otherwise
  the torque-RMSE column is meaningless.

---

## Sim-in-the-loop vs. Open-loop (Real Data) Evaluation

`sim_vs_nosim_benchmark.py` compares open-loop performance of model by feeding real data directly through an actuator model without running Newton physics. This isolates the actuator model's accuracy from physics simulation effects (contact, integrator drift, etc.).

Supported model types (same as the sim benchmark):

| Model type | Description |
|---|---|
| `implicit` | Pure PD control: `kp * error - kd * vel` |
| `dcmotor` | DC motor model (PD + saturation/effort limits) |
| `lstm` / `gru` | Stateful LSTM/GRU neural network (single model for all joints) |
| `lstm_perjoint` / `gru_perjoint` | Per-joint LSTM/GRU (separate model per joint type) |
| `fmu` | FMI 2.0 CoSimulation FMU (Ansys Twin Builder) |

### Basic usage

```bash
# Uses model type from run config (default: teststand)
python scripts/sim2real_gap/sim_vs_nosim_benchmark.py --robot-name teststand

# Load config directly by path
python scripts/sim2real_gap/sim_vs_nosim_benchmark.py \
    --config input/run_configs/h1/h1.yaml \
    --data-path input/motion_files/h1/motor_benchtop/SysID\ Position\ 0/chirp_type3

# Explicit data path and model type
python scripts/sim2real_gap/sim_vs_nosim_benchmark.py \
    --robot-name teststand \
    --data-dir input/motion_files/h1/motor_benchtop/SysID\ Position\ 0/chirp_type3 \
    --model-type fmu

# LSTM/GRU model
python scripts/sim2real_gap/sim_vs_nosim_benchmark.py \
    --config input/run_configs/h1/h1.yaml \
    --model-type lstm \
    --network-file h1/gru_fullarm_elbow_stateful_best.pt

# Override PD gains
python scripts/sim2real_gap/sim_vs_nosim_benchmark.py --robot-name teststand --kp 60 --kd 1.5
```

### Three-way comparison (real vs no-sim vs sim-in-the-loop)

Run the benchmark first, then pass `--sim-results` to overlay sim-in-the-loop results:

```bash
# Step 1: run benchmark (sim-in-the-loop)
python scripts/sim2real_gap/run_benchmark.py --robot-name teststand --headless

# Step 2: run no-sim eval with sim overlay
python scripts/sim2real_gap/sim_vs_nosim_benchmark.py \
    --robot-name teststand \
    --sim-results output/sim2real_benchmark
```

### Output

For each run, the script produces:

- **Per-file PNG plots** — three-panel comparison (torque overlay, error, scatter) for each data file
- **`summary.csv`** — RMSE metrics per file (no-sim, PD baseline, sim-in-the-loop if available)
- **`report.pdf`** — PDF report containing:
  - Title page with actuator config, simulation parameters, and software versions (Isaac Lab, Newton, Isaac Sim, fmpy, PyTorch, CUDA)
  - Summary RMSE table with improvement percentages (color-coded)
  - All per-file comparison plots

Accepts both parquet files (benchtop motor data) and motor CSVs. When both formats exist for the same motion, parquets are preferred to avoid duplicate processing.

---

## Supported Robots

| Robot | Scene Config | USD | Actuator YAML | Valid Joints |
|---|---|---|---|---|
| `h1` | `H1BenchmarkSceneCfg` | `input/robot_models/h1_minimal/h1_minimal.usda` | `h1/h1_arm_implicit.yaml` | `configs/h1_valid_joints.txt` (19 joints) |
| `ur10e` | `Ur10eBenchmarkSceneCfg` | `input/robot_models/ur10/ur10/ur10.usd` | `ur10e/ur10e_implicit.yaml` | `configs/ur10e_valid_joints.txt` (6 joints) |
| `so101` | `So101BenchmarkSceneCfg` | `input/robot_models/so101/so101.usd` | `so101/so101_implicit.yaml` | `configs/so101_valid_joints.txt` (6 joints) |
| `teststand` | `TestStandBenchmarkSceneCfg` | `input/robot_models/teststand/teststand.usda` | `teststand/teststand_implicit.yaml` | `configs/teststand_valid_joints.txt` (1 joint) |

Robot selection is via `--robot-name`. Each robot has its own scene config class in `newton_benchmark.py`, selected at runtime via the `_BENCHMARK_ROBOT_CONFIGS` dict (same pattern as `_ROBOT_CONFIGS` in `run_sysid.py`). See [Adding a New Robot](#adding-a-new-robot) for how to add support for a new robot.

## Scripts

| Script | Description |
|---|---|
| `run_benchmark.py` | Run joint motion playback (or replay real commands) under Newton physics, producing benchmark-compatible CSVs. Auto-converts motor CSVs when `motion_files` points to a sysid data directory. |
| `run_analysis.py` | Compare sim vs real CSV data using the analysis tools |
| `convert_real_data.py` | Convert raw H1 real robot data (seconds timestamps) to the expected format |
| `convert_ur10e_pkl.py` | Convert UR10e trajectory pickle to the expected format |
| `../sysid/convert_h1_chirp_to_csv.py` | Convert H1 motor CSVs to benchmark/sysid format (supports position_error columns and chirp/sine filename reconstruction) |
| `generate_sample_motion.py` | Generate a sample H1 motion file for testing |
| `sim_vs_nosim_benchmark.py` | No-sim actuator model evaluation — feeds real data directly through any actuator model (implicit, dcmotor, lstm, fmu) without physics. Supports three-way comparison (real vs no-sim vs sim-in-the-loop) and generates a PDF report. |
| `convert_benchtop_parquet.py` | Convert benchtop parquet files (single-motor) to `*_motor.csv` format with joint-prefixed columns |
| `benchmark_fmu.py` | Standalone FMU torque benchmark — compares Ansys Twin Builder output vs real torque from parquet data. No Isaac Lab required. Supports pytwin and fmpy backends. |
| `newton_benchmark.py` | Core `NewtonJointMotionBenchmark` class with per-robot scene configs |

## Runtime Configuration

Per-robot settings live in `input/run_configs/<robot>/<robot>.yaml`. These YAML files centralize simulation parameters, benchmark settings, actuator model selection, and sysid bounds so you don't need to hunt through Python files or remember long CLI commands.

```
input/run_configs/
├── __init__.py              # load_run_cfg() loader
├── h1/
│   ├── h1.yaml              # H1 run config
│   ├── h1_arms_sysid_bounds.yaml       # sysid bounds (mirrored)
│   └── h1_right_arm_sysid_bounds.yaml  # sysid bounds (right arm only)
├── teststand/
│   └── teststand.yaml
└── ur10e/
    ├── ur10e.yaml
    └── ur10e_sysid_bounds.yaml
```

**Override priority:** CLI argument > run config YAML > hardcoded default

If a CLI arg is explicitly provided, it wins. Otherwise the run config value is used. If neither is set, the original hardcoded default applies. Deleting the YAML file reverts all behavior to defaults.

### Configuration Sections

**`simulation`** — Physics engine settings:

| Key | Default | Description |
|---|---|---|
| `physics_freq` | 200 (H1), 500 (UR10e) | Physics timestep frequency (Hz) |
| `control_freq` | 200 / 500 | Control frequency (Hz) |
| `render_freq` | 200 / 500 | Render timestep frequency (Hz) |
| `integrator` | `implicitfast` | Newton solver integrator |
| `solver_iterations` | 10 | Newton solver iterations |
| `ls_iterations` | 10 | Newton line-search iterations |

**`benchmark`** — Benchmark settings:

| Key | Default | Description |
|---|---|---|
| `fix_root` | `true` | Fix robot root joint |
| `buffer_time` | `5.0` | Settle time before motion playback (seconds) |
| `motor_lag_ms` | `null` | Motor command delay (ms). `null` = read from actuator YAML |
| `kp` | `null` | Override joint stiffness. `null` = use actuator YAML |
| `kd` | `null` | Override joint damping. `null` = use actuator YAML |
| `motion_files` | `null` | Path to motion file(s), directory of `.txt` files, or directory of motor CSVs. Motor CSVs are auto-converted to the expected format. |
| `real_control_csv` | `null` | Path to real robot control.csv for replay in sim |
| `original_control_freq` | `null` | Control frequency of real data (Hz). Auto-detected from CSV timestamps if null. |
| `motion_name` | `null` | Motion name for output directory |
| `output_folder` | `output/sim2real_benchmark` | Root output folder (contains `sim/` and `real/` subdirs) |

**`analysis`** — Analysis settings:

| Key | Default | Description |
|---|---|---|
| `output_dir` | `output/sim2real_analysis` | Output directory for analysis plots and metrics |
| `sample_dt` | `0.005` | Comparison timestep (seconds) |
| `motion_names` | `"*"` (all) | Which motions to analyze. Accepts a list (`[EC01_elbow_chirp, EC02_elbow_chirp]`), comma-separated string, or glob pattern (`"*elbow*"`). Base names auto-resolve to actuator-suffixed sim folders. |

**`actuator`** — Actuator model selection (benchmark only):

| Key | Description |
|---|---|
| `model_type` | `implicit`, `dcmotor`, `lstm`, `lstm_perjoint`, or `fmu` (`gru`/`gru_perjoint` also accepted) |
| `yaml_file` | Actuator parameter YAML (relative to `input/actuator_models/`) |
| `network_file` | `.pt` model file for `lstm` (relative to `input/actuator_models/`) |
| `network_files` | Dict of joint_type -> `.pt` file for `lstm_perjoint` (relative to `input/actuator_models/`) |
| `fmu_path` | FMU directory or `.fmu` archive for `fmu` (relative to `input/actuator_models/`) |
| `fmu_step_size` | Communication step size [s] for `fmu` (default: 0.002, should match physics dt) |

**`sysid`** — System identification settings:

| Key | Default | Description |
|---|---|---|
| `bounds_yaml` | auto | Sysid parameter bounds YAML (path relative to `input/run_configs/`) |
| `real_data_dir` | — | Motor CSV or benchmark directory. Motor CSVs auto-converted at startup. |
| `joints` | `full` | Joint types to optimize: `full` (all) or list (e.g., `[elbow]`, `[elbow, shoulder_pitch]`). Without `left_`/`right_` prefix, mirrors both sides. |
| `num_envs` | 64 | Parallel environments (CMA-ES population) |
| `max_iter` | 200 | Maximum CMA-ES generations |
| `sigma` | 0.5 | CMA-ES initial step size |
| `epsilon` | 0.01 | Convergence threshold |
| `buffer_time` | 2.0 | Settle time before replay (seconds) |

### Example: Switching Actuator Models

To switch H1 from implicit PD to a single LSTM, edit `input/run_configs/h1/h1.yaml`:

```yaml
actuator:
  model_type: lstm
  network_file: h1/gru_fullarm_elbow_stateful_best.pt
```

For per-joint LSTM (separate model per joint type, left/right share):

```yaml
actuator:
  model_type: lstm_perjoint
  network_files:
    shoulder_pitch: h1/shoulder_pitch.pt
    shoulder_roll: h1/shoulder_roll.pt
    shoulder_yaw: h1/shoulder_yaw.pt
    elbow: h1/elbow.pt
```

For DC motor:

```yaml
actuator:
  model_type: dcmotor
  yaml_file: h1/h1_arm_dcmotor.yaml
```

For FMU (Ansys Twin Builder CoSimulation model):

```yaml
actuator:
  model_type: fmu
  yaml_file: h1/h1_arm_implicit.yaml           # PD gains for torque_pred input
  fmu_path: h1/nvidia-motor.fmu                # CoSimulation FMU (.fmu archive)
  fmu_step_size: 0.002                          # should match physics dt
```

No Python code changes needed — just edit the YAML and rerun.

## Data Format

The analysis module expects a specific directory layout and CSV format for both sim and real data.

### Directory Structure

```
result_folder/
├── sim/{robot_name}/{motion_source}/{motion_name}_{actuator}/
│   ├── control.csv
│   ├── state_motor.csv
│   └── joint_list.txt
└── real/{robot_name}/{motion_source}/{motion_name}/
    ├── control.csv
    ├── state_motor.csv
    ├── joint_list.txt
    └── event.csv            # real data only
```

The benchmark also writes a `run_summary.yaml` (timestamp, simulation settings, actuator config, and run parameters) alongside the sim output for reproducibility.

Sim output folders are suffixed with the actuator model type (e.g., `motion_stand_left_arm_swing_lstm`, `motion_stand_left_arm_swing_dcmotor`, `motion_stand_left_arm_swing_implicit`). Real data folders use the base motion name without suffix. The analysis script auto-creates symlinks in `real/` to match suffixed sim names, then cleans them up after analysis.

The analysis script auto-discovers **nested directory structures** (e.g., when motor CSVs are organized by joint type). For example:

```
result_folder/
├── sim/h1/custom/
│   ├── EC01_elbow_chirp_implicit/       # flat motion
│   └── shoulder_pitch/                  # nested group
│       └── SP01_chirp_implicit/
└── real/h1/custom/
    ├── EC01_elbow_chirp/
    └── shoulder_pitch/
        └── SP01_chirp/
```

Both flat and nested structures are handled automatically — no configuration needed. The script identifies valid motion folders by checking for `control.csv` inside each directory.

### control.csv

Commanded joint position targets. Timestamps are in microseconds for both sim and real data.

```
type,timestamp,positions
CONTROL,0.0,"[0.1, 0.2, 0.3, ...]"
CONTROL,5000.0,"[0.11, 0.21, 0.31, ...]"
```

### state_motor.csv

Actual measured/simulated joint state. Timestamps are in microseconds for both sim and real data.

```
type,timestamp,positions,velocities,torques
STATE_MOTOR,0.0,"[0.1, 0.2, ...]","[0.01, 0.02, ...]","[1.5, 2.3, ...]"
```

### event.csv (real data only)

Timing markers used to align real data with sim data. Timestamps must be **relative** (0-based, in microseconds) because the analysis module internally computes `time_since_zero = timestamp - initial_time`, then subtracts event timestamps from `time_since_zero`. Using absolute timestamps would cause a double-subtraction that discards data.

```
type,timestamp,event
EVENT,0.0,MOTION_START
EVENT,21247830.0,DISABLE
```

### joint_list.txt

One joint name per line, defining the column order for position/velocity/torque arrays.

### Timestamp Convention

- **Both sim and real data** write timestamps in **microseconds** (e.g., `5335070.0`). The benchmark converts sim time (seconds) to microseconds when writing CSVs.
- The analysis script auto-detects microsecond timestamps (values > 1000) and converts to seconds for plotting via `_time_to_seconds()`.
- Position/velocity/torque values are stored as Python list strings, parsed with `ast.literal_eval()`.
- The analysis module resamples both sim and real data to a common timestep (`--sample-dt`), aligns real data using `MOTION_START`/`DISABLE` from `event.csv`, then computes metrics per joint.

## Joint Configuration

### H1

The H1 robot has 20 motor slots with 19 active joints (motor index 9 is unused). Joints are defined in `configs/h1_valid_joints.txt`, ordered by motor index:

| Motor Index | Joint Name |
|---|---|
| 0 | right_hip_roll |
| 1 | right_hip_pitch |
| 2 | right_knee |
| 3 | left_hip_roll |
| 4 | left_hip_pitch |
| 5 | left_knee |
| 6 | torso |
| 7 | left_hip_yaw |
| 8 | right_hip_yaw |
| 9 | *(unused)* |
| 10 | left_ankle |
| 11 | right_ankle |
| 12 | right_shoulder_pitch |
| 13 | right_shoulder_roll |
| 14 | right_shoulder_yaw |
| 15 | right_elbow |
| 16 | left_shoulder_pitch |
| 17 | left_shoulder_roll |
| 18 | left_shoulder_yaw |
| 19 | left_elbow |

### Teststand

1 joint defined in `configs/teststand_valid_joints.txt`:

| Joint Name |
|---|
| elbow |

### UR10e

6 joints defined in `configs/ur10e_valid_joints.txt`:

| Joint Name |
|---|
| shoulder_pan_joint |
| shoulder_lift_joint |
| elbow_joint |
| wrist_1_joint |
| wrist_2_joint |
| wrist_3_joint |

## Actuator Model

### Newton Property Mapping

All actuator parameters map to specific Newton (MuJoCo Warp) solver properties:

| Parameter | Newton Property | Description |
|---|---|---|
| armature | `model.joint_armature` | Rotor inertia [kg·m²] |
| dynamic_friction | `model.joint_friction` | Coulomb/dry friction [N·m] |
| viscous_friction | `model.mujoco.dof_passive_damping` | Passive viscous damping F = -b·ω [N·m·s/rad] |
| stiffness (kp) | `model.joint_target_ke` | PD position gain [N·m/rad] |
| damping (kd) | `model.joint_target_kd` | PD velocity gain [N·m·s/rad] |

Note: `viscous_friction` uses a custom writer (no Isaac Lab API exists for `mujoco.dof_passive_damping`). This is a MuJoCo-solver custom attribute, separate from PD gains. See `scripts/sysid/README.md` for details.

### H1

The benchmark supports three actuator models for H1 arm joints, selected via `model_type` in `input/run_configs/h1/h1.yaml` (see [Runtime Configuration](#runtime-configuration)):

**Option A: `model_type: implicit`** — Physics-engine PD control with motor mechanical parameters from `input/actuator_models/h1/h1_arm_implicit.yaml`. Simple and fast, useful for baseline comparison. Parameters can be scalars (same for all joints) or per-joint dicts with regex keys.
- stiffness (kp): 60.0, damping (kd): 1.5
- Per-joint mechanical params from CMA-ES sysid (right arm chirp experiments):

| Parameter | shoulder_pitch | shoulder_roll | shoulder_yaw | elbow |
|---|---|---|---|---|
| armature (J) kg-m^2 | 0.001867 | 0.019726 | 0.026347 | 0.021208 |
| dynamic_friction (c) Nm | 0.948609 | 0.8105 | 0.606761 | 0.858461 |
| viscous_friction (b) Nm-s/rad | 1.582401 | 1.878386 | 1.502171 | 1.607976 |

- motor_lag_ms: 35.6 ms (applied as software command buffer, configurable via `--motor-lag-ms`)

**Option B: `model_type: dcmotor`** — Velocity-dependent torque saturation with real H1 motor parameters (loaded from `input/actuator_models/h1/h1_arm_dcmotor.yaml`):

| Parameter | Value | Description |
|---|---|---|
| K_t | 1.89 N-m/A | Torque constant |
| I_peak | 100 A | Peak current |
| saturation_effort | 189.0 N-m | K_t x I_peak |
| armature (J) | 0.041212 kg-m^2 | Rotor inertia |
| viscous_friction (b) | 2.099477 N-m-s/rad | Viscous damping |
| friction (c) | 0.000003 N-m | Coulomb friction |
| stiffness (kp) | 40.0 | PD proportional gain |
| damping (kd) | 10.0 | PD derivative gain |

**Option C: `model_type: lstm` or `lstm_perjoint`** — LSTM/GRU actuator net trained on real H1 data. Two modes:

- `lstm` — One LSTM/GRU for all 8 arm joints. Set `network_file` in the run config.
- `lstm_perjoint` — Separate model per joint type (left/right share same model). Set `network_files` dict in the run config. Creates 4 actuator groups: `arms_shoulder_pitch`, `arms_shoulder_roll`, `arms_shoulder_yaw`, `arms_elbow`.

```yaml
# Single model (input/run_configs/h1/h1.yaml):
actuator:
  model_type: lstm
  network_file: h1/gru_fullarm_elbow_stateful_best.pt

# Per-joint models:
actuator:
  model_type: lstm_perjoint
  network_files:
    shoulder_pitch: h1/shoulder_pitch.pt
    shoulder_roll: h1/shoulder_roll.pt
    shoulder_yaw: h1/shoulder_yaw.pt
    elbow: h1/elbow.pt
```

The `_ensure_torchscript()` wrapper auto-converts a regular `torch.save(state_dict)` checkpoint to TorchScript on first use (cached as `*_scripted.pt`). The converted model wraps the raw LSTM+head with Isaac Lab's expected `forward(input, (hidden, cell)) -> (output, (hidden, cell))` signature.

Normalization stats are loaded per model from a sidecar JSON file (`<model_name>_stats.json`) if present, otherwise falls back to the global `_SYSID_STATS` in `newton_benchmark.py`. This allows each per-joint model to use its own training statistics.

Required files in `input/actuator_models/` (e.g., `h1/model.pt`):
- Model `.pt` file(s) — auto-converted to TorchScript if needed
- Optional `<model_name>_stats.json` sidecar files for per-model normalization stats

**Option D: `model_type: fmu`** — Ansys Twin Builder FMU (Functional Mock-up Unit) hybrid actuator. The FMU is a dynamic ROM exported as FMI 2.0 CoSimulation. It takes `(position, velocity, torque_pred)` as inputs and outputs a corrected `torque_true`.

```yaml
actuator:
  model_type: fmu
  yaml_file: h1/h1_arm_implicit.yaml           # PD gains for torque_pred
  fmu_path: h1/nvidia-motor.fmu                # CoSimulation FMU (.fmu archive)
  fmu_step_size: 0.002                          # must match physics dt
```

The FMU actuator (`actuator_fmu.py`) follows the FMI 2.0 CoSimulation protocol:
1. Computes PD torque from the YAML's kp/kd gains
2. Sets `(position, velocity, torque_pred)` as FMU inputs
3. Calls `doStep()` to advance the FMU's internal solver
4. Reads `torque_true` output

Requires `fmpy` (`pip install fmpy`). One FMU instance is created per (env, joint) pair.

#### Standalone FMU Benchmark

`benchmark_fmu.py` benchmarks the FMU torque prediction against real data from parquet files, without requiring Isaac Lab or Newton. This is useful for validating the FMU model independently of the physics simulation.

```bash
# default paths (parquet data + FMU from input/)
python scripts/sim2real_gap/benchmark_fmu.py

# custom PD gains and data directory
python scripts/sim2real_gap/benchmark_fmu.py --kp 60 --kd 1.5 \
    --data-dir input/sysid_data/h1/motor/chirp_data

# force fmpy backend (default: auto-selects)
python scripts/sim2real_gap/benchmark_fmu.py --backend fmpy
```

Backends: `pytwin` (preferred, matches Twin Builder exactly) or `fmpy` (fallback, uses `completedIntegratorStep` protocol). Outputs per-file comparison plots (real vs FMU vs PD torque) and a summary CSV with RMSE/MAE metrics.

Leg and feet joints always use `ImplicitActuatorCfg` (physics-engine PD control).

### UR10e

UR10e uses a single `ImplicitActuatorCfg` covering all 6 joints (configured in `Ur10eBenchmarkSceneCfg`). Parameters from `input/actuator_models/ur10e/ur10e_implicit.yaml`:

- stiffness (kp): 800.0, damping (kd): 40.0
- effort_limit: 87.0 N-m, velocity_limit: 100.0 rad/s
- Per-joint mechanical params from CMA-ES sysid:

| Parameter | shoulder_pan | shoulder_lift | elbow | wrist_1 | wrist_2 | wrist_3 |
|---|---|---|---|---|---|---|
| armature (J) kg-m^2 | 0.486 | 0.044 | 0.155 | 0.151 | 0.410 | 0.424 |
| dynamic_friction (c) | 3.427 | 4.365 | 4.719 | 1.793 | 4.393 | 2.090 |
| viscous_friction (b) | 2.606 | 4.720 | 4.840 | 3.377 | 1.298 | 0.997 |

The benchmark logs the actual instantiated actuator type at startup — check for `Actuator 'arms': ActuatorNetLSTM` vs `ImplicitActuator` vs `DCMotor` to verify. Per-joint LSTM mode shows `arms_shoulder_pitch`, `arms_elbow`, etc. Output folder suffix is `lstm_perjoint` for per-joint mode vs `lstm` for single-model mode.

### Newton ActuatorNetLSTM Bug Fix

Newton's `ActuatorNetLSTM.compute()` has a joint indexing bug in multi-actuator setups (e.g., H1 with separate legs/feet/arms groups). The shared data arrays contain all 19 robot joints, but the LSTM's `sea_input` buffer is sized for the actuator's joints only. Newton's other actuators (ImplicitActuator, DCMotor) handle this correctly using Warp kernels with boolean joint masks, but ActuatorNetLSTM uses raw PyTorch operations and skips the masking.

The benchmark automatically patches all LSTM/GRU actuators' `compute()` method after `sim.reset()` (via `_patch_lstm_compute()` in `newton_benchmark.py`). This works for both single-model and per-joint configurations. The patch:
1. **Indexes inputs** using `self._joint_indices` to extract only this actuator's joints from full-robot data arrays
2. **Scatters outputs** to the correct positions in the full-robot `_computed_effort` array
3. **Fixes kernel dimensions** using `self._num_joints` (19, total) instead of `self.num_joints` (actuator count) for the final Warp masking kernel
4. **Auto-detects model type** from network input_size: 3-input (sysid) uses per-model stats with percentile clipping; 2-input uses raw inputs without normalization
5. **Loads per-model stats** from `<model_name>_stats.json` sidecar file if present, otherwise falls back to global `_SYSID_STATS`
6. **Resizes hidden/cell state** if the network's num_layers differs from Newton's default allocation

This patch is applied transparently — no changes to Isaac Lab or Newton source code are needed.

## AMASS Motion Data

The benchmark supports motion files from the [AMASS](https://amass.is.tue.mpg.de/) dataset, but these require a retargeting step to convert human mocap to robot joint trajectories. The upstream repo references [Human2Humanoid](https://github.com/LeCAR-Lab/human2humanoid) as one approach.

The upstream SAGE repo provides pre-retargeted AMASS motions for H1-2, G1, and WR75S, but **not for H1**. To use AMASS with H1:

1. Run a retargeting pipeline (e.g. Human2Humanoid) targeting the H1 URDF/USD
2. Place the output files in `input/motion_files/h1/amass/`
3. Set `motion_files` in the run config or pass via `--motion-files`

## Collecting Real Robot Data

To complete the sim-to-real comparison, collect real data and place it under `result_folder/real/{robot_name}/{motion_source}/{motion_name}/`. You need:

1. **state_motor.csv** — joint positions, velocities, and torques recorded from the real robot
2. **control.csv** — commanded joint position targets sent to the robot
3. **joint_list.txt** — joint names matching the column order in the CSV arrays
4. **event.csv** — timing markers (`MOTION_START` and `DISABLE`) with relative timestamps in microseconds

### H1

If your H1 data is in raw format (seconds timestamps, bare CSV), use `convert_real_data.py` to convert it.

The upstream SAGE repo provides a Unitree data collector (`sage.real_unitree.unitree_collector`) for H1-2 and G1. For H1, you can adapt this collector or write your own using the Unitree SDK.

### UR10e

UR10e data is recorded as a `trajectory.pkl` via `isaac_manipulator_data_utils`. Convert with `convert_ur10e_pkl.py` (see [Quick Start (UR10e)](#quick-start-ur10e) step 1).

## Adding a New Robot

To add a new robot to the benchmark:

1. **USD model** — Place in `input/robot_models/<robot>/`
2. **Actuator YAML** — Create `input/actuator_models/<robot>/<robot>_implicit.yaml` with kp, kd, effort/velocity limits, and per-joint mechanical params (regex keys matching joint names)
3. **Run config** — Create `input/run_configs/<robot>/<robot>.yaml` with simulation, benchmark, actuator, and sysid sections (copy `ur10e/ur10e.yaml` as a template). Place sysid bounds YAML alongside.
4. **Scene config** — Add a `@configclass` class in `newton_benchmark.py` (see `Ur10eBenchmarkSceneCfg` as template)
5. **Robot config entry** — Add to `_BENCHMARK_ROBOT_CONFIGS` and `_ROBOT_ARM_CFG` dicts in `newton_benchmark.py`
6. **Valid joints** — Create `scripts/sim2real_gap/configs/<robot>_valid_joints.txt`
7. **Data converter** — If needed, create a conversion script for the robot's real data format
