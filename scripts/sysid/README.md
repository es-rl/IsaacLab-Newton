# System Identification (CMA-ES)

Automated motor parameter optimization for robot actuators.
Runs N parallel sim environments with different candidate parameters,
replays real robot data, and minimizes position MSE vs real measurements.

Based on: [PACE Sim2Real](https://github.com/leggedrobotics/pace-sim2real) (ETH Zurich)

## Motor Model

The DC motor dynamics at each joint:

    tau = J * alpha + b * omega + c * sign(omega)

where:
- **J** (`armature`) — rotor inertia reflected at joint (kg-m^2)
- **b** (`viscous_friction`) — passive viscous damping (Nm-s/rad)
- **c** (`dynamic_friction`) — Coulomb friction torque (Nm)

### Stribeck Friction (optional)

When `stribeck_velocity` is included in the bounds YAML, Coulomb friction is
replaced with a smooth Stribeck model applied as an external torque each step:

    tau_friction = -c * tanh(v / v_s)

where **v_s** (`stribeck_velocity`) controls the velocity-dependent transition.
At `|v| >> v_s` this converges to standard Coulomb friction `±c`. Near `v ≈ 0`
friction smoothly goes to zero instead of jumping discontinuously.

When active, Newton's built-in `joint_friction` is zeroed and all Coulomb
friction is handled externally. Viscous friction (`dof_passive_damping`) remains
in the solver. When `stribeck_velocity` is commented out, the standard Coulomb
model is used unchanged.

### Newton Property Mapping

The simulator uses Newton physics with the MuJoCo Warp solver. Each sysid
parameter maps to a specific Newton model property:

| Sysid Parameter | Newton Model Property | Isaac Lab API | Description |
|---|---|---|---|
| `armature` | `model.joint_armature` | `write_joint_armature_to_sim()` | Rotor inertia [kg·m²] |
| `dynamic_friction` | `model.joint_friction` | `write_joint_friction_to_sim()` | Coulomb/dry friction [N·m] |
| `viscous_friction` | `model.mujoco.dof_passive_damping` | *(custom writer)* | Passive viscous damping F = -b·ω [N·m·s/rad] |
| `stiffness` | `model.joint_target_ke` | `write_joint_stiffness_to_sim()` | PD position gain kp [N·m/rad] |
| `damping` | `model.joint_target_kd` | `write_joint_damping_to_sim()` | PD velocity gain kd [N·m·s/rad] |
| `stribeck_velocity` | *(external torque via `joint_f`)* | *(custom writer)* | Stribeck transition velocity [rad/s] |

Notes:
- `viscous_friction` writes directly to Newton's `mujoco.dof_passive_damping` custom attribute (no Isaac Lab API exists). This is a MuJoCo-solver-only property (default 0.0), separate from PD gains.
- `stiffness` and `damping` are PD controller gains, not passive mechanical properties. They can be optimized alongside `viscous_friction`.
- `stribeck_velocity` replaces Newton's built-in Coulomb friction with a smooth `tanh(v/v_s)` model applied as an external torque each sim step. When active, `joint_friction` is zeroed.
- All 6 parameters can be used together in any combination.

Current per-joint values from CMA-ES sysid on right arm chirp data (h1_arm_implicit.yaml):

| Parameter | shoulder_pitch | shoulder_roll | shoulder_yaw | elbow |
|---|---|---|---|---|
| J (armature) kg-m^2 | 0.001867 | 0.019726 | 0.026347 | 0.021208 |
| c (dynamic_friction) Nm | 0.948609 | 0.8105 | 0.606761 | 0.858461 |
| b (viscous_friction) Nm-s/rad | 1.582401 | 1.878386 | 1.502171 | 1.607976 |

CMA-ES optimizes J, b, c (and optionally kp, kd) per joint type to minimize sim-to-real position error.

## Dependencies

All dependencies are installed in the Docker image (`docker/Dockerfile.sysid`). Build all images with:

```bash
./docker/build-docker.sh            # uses cache (fast if already built)
./docker/build-docker.sh --no-cache  # force full rebuild (e.g. after Dockerfile changes)
```

- `cmaes` — CMA-ES optimizer for system identification
- `pandas`, `pyarrow` — data loading (parquet chirp files, CSVs)
- `optuna` — hyperparameter search for GRU/hybrid model training
- `pyyaml` — YAML config loading
- `torch` — GRU model training and inference

## Workflow

1. Place raw motor data in `input/motion_files/<robot>/` (e.g., `input/motion_files/h1/arm_march2026/elbow/`)
2. Configure `input/run_configs/<robot>/<robot>.yaml`:
   ```yaml
   sysid:
     output_dir: elbow                                    # → output/sysid/h1/elbow/
     bounds_yaml: h1/h1_arms_sysid_bounds.yaml
     real_data_dir: input/motion_files/h1/arm_march2026/elbow
     joints: [elbow]

   benchmark:
     motion_files: input/motion_files/h1/arm_march2026/elbow
     motion_name: arm_march2026
     output_folder: output/sim2real_benchmark
   ```
3. Run sysid (motor CSVs are auto-converted):
   ```bash
   python scripts/sysid/run_sysid.py --robot-name h1 --headless
   ```
   Or override data/joints via CLI:
   ```bash
   python scripts/sysid/run_sysid.py \
       --robot-name h1 \
       --real-data-dir "input/motion_files/h1/arm_march2026/shoulder_pitch" \
       --joints shoulder_pitch \
       --output-dir output/sysid/h1/shoulder_pitch \
       --headless
   ```
4. Copy `best_params.yaml` per-joint values into the robot's actuator YAML:
   - H1: `input/actuator_models/h1/h1_arm_implicit.yaml`
   - UR10e: `input/actuator_models/ur10e/ur10e_implicit.yaml`
5. *(Optional)* Train a GRU actuator model per joint:
   ```bash
   # Standard GRU (full torque prediction)
   python scripts/sysid/train_model/gru_model_train.py \
       --joint-type elbow \
       --data-dirs "/path/to/training/data" \
       --skip-optuna

   # Hybrid residual GRU (better closed-loop sim accuracy)
   python scripts/sysid/train_model/hybrid_model_train.py \
       --implicit-yaml input/actuator_models/h1/h1_arm_implicit.yaml \
       --joint-type elbow \
       --data-dirs "/path/to/training/data" \
       --noise-std 0.02 \
       --skip-optuna
   ```
6. Validate: run benchmark + analysis (auto-converts motor CSVs and populates `real/` folder):
   ```bash
   python scripts/sim2real_gap/run_benchmark.py --robot-name h1 --headless
   python scripts/sim2real_gap/run_analysis.py --robot-name h1
   ```

## Input Modes

### Single-joint (parquet chirp data)

For sysid experiments where one joint is excited with a chirp signal.
Each `.parquet` file has columns: `time`, `position`, `commanded_position`, `velocity`, `torque`, ...

```bash
python scripts/sysid/run_sysid.py \
    --data-dir "input/motion_files/h1/motor/chirp_type3" \
    --joint-name left_elbow \
    --output-dir output/sysid/h1/motor \
    --headless
```

Use `--data-files` to select specific files instead of a whole directory:

```bash
python scripts/sysid/run_sysid.py \
    --data-files "input/motion_files/.../sysid_chirp_0p5_2p0hz_large_0_type3_combined_clean.parquet" \
    --joint-name left_elbow \
    --output-dir output/sysid/h1/motor \
    --headless
```

### All-joints (CSV or motor CSV data)

For multi-joint motion data. Accepts either benchmark-format directories (control.csv + state_motor.csv + joint_list.txt) or raw `*_motor.csv` directories — motor CSVs are **auto-converted** to the expected format at startup.

When specifying joint types without a `left_`/`right_` prefix (e.g., `elbow`), the optimizer automatically mirrors parameters to both sides. Scoring uses only the joints present in the real data, but the optimized parameters are written to both left and right joints.

```bash
# Config-driven (reads everything from h1.yaml sysid section):
python scripts/sysid/run_sysid.py --robot-name h1 --headless

# Override data/joints via CLI:
python scripts/sysid/run_sysid.py \
    --robot-name h1 \
    --real-data-dir input/motion_files/h1/arm_march2026/elbow \
    --joints elbow \
    --output-dir output/sysid/h1/elbow \
    --headless
```

Auto-conversion detects the format: if the directory has `control.csv`, it's used directly. If it has `*_motor.csv` files (flat or in subdirs), they're concatenated into one an output directory at `output_dir/converted_sage/`.

### UR10e

UR10e uses a local USD at `input/robot_models/ur10/ur10/ur10.usd` (no URDF conversion needed). Frequencies default to 500Hz from `input/run_configs/ur10e/ur10e.yaml`:

```bash
python scripts/sysid/run_sysid.py \
    --robot-name ur10e \
    --real-data-dir input/motion_files/ur10e_csv \
    --max-iter 100 \
    --output-dir output/sysid/ur10e \
    --headless
```

Optimized parameters are written to `best_params.yaml`. Copy per-joint values into `input/actuator_models/ur10e/ur10e_implicit.yaml` (keys must match joint names, e.g. `shoulder_pan_joint`, `elbow_joint`).

## Runtime Configuration

Per-robot settings live in `input/run_configs/<robot>/<robot>.yaml`. This means you don't need to pass `--physics-freq`, `--num-envs`, etc. on every run — just set them once in the YAML. Bounds YAML files live alongside the run config.

```
input/run_configs/
├── __init__.py              # load_run_cfg() loader
├── h1/
│   ├── h1.yaml              # H1 run config
│   ├── h1_arms_sysid_bounds.yaml       # mirrored (both arms)
│   └── h1_right_arm_sysid_bounds.yaml  # right arm only
└── ur10e/
    ├── ur10e.yaml
    └── ur10e_sysid_bounds.yaml
```

**Override priority:** CLI argument > run config YAML > hardcoded default

For example, with `input/run_configs/h1/h1.yaml`:

```yaml
simulation:
  physics_freq: 500
  control_freq: 500
  integrator: implicitfast

sysid:
  output_dir: elbow                                    # → output/sysid/h1/elbow/
  bounds_yaml: h1/h1_arms_sysid_bounds.yaml            # path relative to input/run_configs/
  real_data_dir: input/motion_files/h1/arm_march2026/elbow  # motor CSVs or benchmark directory
  joints: [elbow]                                      # "full" = all, or list
  num_envs: 64
  max_iter: 200
  sigma: 0.5
  epsilon: 0.01
  buffer_time: 2.0
```

Running `--robot-name h1` picks up all sysid settings from the YAML. Use CLI args only to override:

```bash
# Config-driven (everything from h1.yaml):
python scripts/sysid/run_sysid.py --robot-name h1 --headless

# Override to optimize with different data and more iterations:
python scripts/sysid/run_sysid.py \
    --robot-name h1 \
    --real-data-dir input/motion_files/h1/arm_march2026/shoulder_pitch \
    --joints shoulder_pitch \
    --max-iter 300 \
    --headless

# Use a different bounds YAML:
python scripts/sysid/run_sysid.py \
    --robot-name h1 \
    --config input/run_configs/h1/h1_right_arm_sysid_bounds.yaml \
    --headless
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| **Input (choose one):** | | |
| `--data-dir` | — | Directory with parquet chirp files (single-joint mode) |
| `--data-files` | — | Comma-separated parquet paths (single-joint mode) |
| `--real-data-dir` | —* | CSV or motor CSV directory (all-joints mode). Motor CSVs auto-converted. |
| `--joint-name` | `left_elbow` | Sim joint for single-joint mode |
| `--robot-name` | `h1` | Robot variant: `h1` (mirrored arms), `h1_right_arm` (right arm only), or `ur10e` |
| **Optimization:** | | |
| `--joints` | `full`* | Joint types to optimize: `full` (all) or comma-separated list (e.g., `elbow` or `elbow,shoulder_pitch`). Without `left_`/`right_` prefix, mirrors both sides. |
| `--num-envs` | 64* | Parallel environments = CMA-ES population size |
| `--max-iter` | 200* | Maximum CMA-ES generations |
| `--sigma` | 0.5* | CMA-ES initial step size |
| `--epsilon` | 0.01* | Convergence threshold |
| `--config` | auto* | Parameter bounds YAML (default: from `sysid.bounds_yaml` in run config) |
| **Simulation:** | | |
| `--physics-freq` | 200* | Physics frequency (Hz) |
| `--control-freq` | 200* | Control replay frequency (Hz) |
| `--buffer-time` | 2.0* | Settle time before replay (s) |
| `--max-trajectory-len` | all | Truncate trajectory to N steps |
| `--output-dir` | auto* | Results directory (`output/sysid/{robot}/{output_dir or joints}`) |
| `--headless` | off | Run without rendering |

\* Defaults marked with \* are overridden by `input/run_configs/<robot>.yaml` when present. CLI args always take highest priority.

## What Gets Optimized

Up to 6 motor properties per joint type (configured in bounds YAML):

| Symbol | YAML key | H1 Bounds | Description | Newton Property |
|---|---|---|---|---|
| J | `armature` | [0.001, 0.05] | Rotor inertia (kg·m²) | `joint_armature` |
| c | `dynamic_friction` | [0.0, 1.0] | Coulomb friction (N·m) | `joint_friction` |
| b | `viscous_friction` | [0.0, 2.0] | Passive viscous damping (N·m·s/rad) | `mujoco.dof_passive_damping` |
| v_s | `stribeck_velocity` | *(commented)* [0.01, 2.0] | Stribeck transition velocity (rad/s) | external torque via `joint_f` |
| kp | `stiffness` | *(commented)* | PD position gain (N·m/rad) | `joint_target_ke` |
| kd | `damping` | *(commented)* | PD velocity gain (N·m·s/rad) | `joint_target_kd` |

By default, J, c, b are optimized while v_s, kp, and kd are commented out. Uncomment them in the bounds YAML to include in the optimization. When `stribeck_velocity` is active, Coulomb friction is applied externally as `τ = -c * tanh(v / v_s)` instead of through Newton's solver.

Three bounds configs are available (in `input/run_configs/`):

- **`h1/h1_arms_sysid_bounds.yaml`** — 4 joint types x N properties, left/right mirrored. Joint types: `shoulder_pitch`, `shoulder_roll`, `shoulder_yaw`, `elbow`.
- **`h1/h1_right_arm_sysid_bounds.yaml`** — 4 right-arm joints x N properties, no mirroring (`mirror: false`). Joint types: `right_shoulder_pitch`, `right_shoulder_roll`, `right_shoulder_yaw`, `right_elbow`.
- **`ur10e/ur10e_sysid_bounds.yaml`** — 6 joints x N properties, no mirroring (`mirror: false`). Joint types: `shoulder_pan`, `shoulder_lift`, `elbow`, `wrist_1`, `wrist_2`, `wrist_3`.

Select which bounds YAML to use via `sysid.bounds_yaml` in the run config or `--config` CLI arg. Edit the bounds YAML to change bounds, move parameters between optimized/fixed, or adjust fixed values.

## Output

Results are written to `output/sysid/{robot_name}/{output_dir}/`. The `output_dir` is set via `sysid.output_dir` in the run config YAML, or defaults to the joint names (e.g., `elbow` or `elbow_shoulder_pitch`).

| File | Description |
|---|---|
| `run_summary.yaml` | Timestamp, simulation settings, sysid config, bounds, and actuator model parameters |
| `best_params.yaml` | Best J, b, c per joint type. Copy per-joint values into `input/actuator_models/h1/h1_arm_implicit.yaml` (supports per-joint dicts with regex keys). |
| `optimization_log.csv` | Per-generation: best/mean MSE, all parameter values |

## Actuator Model Training

Training scripts live in `scripts/sysid/train_model/`. Both are standalone — no Newton, Isaac Lab, or MuJoCo required. Runs on any machine with PyTorch and a GPU.

Two model types are available:

| Script | Model Type | Predicts | Use Case |
|---|---|---|---|
| `gru_model_train.py` | Standard GRU | Full torque | Best open-loop accuracy |
| `hybrid_model_train.py` | Hybrid residual GRU | `tau_real - (kp * pos_error - kd * vel)` | Better closed-loop sim accuracy |

Both use the same architecture (TorqueGRU: GRU + linear head, 3 inputs -> 1 output), stateful TBPTT training, and Optuna hyperparameter search.

### Standard GRU

Trains on full motor torque directly from [position, position_error, velocity] inputs.

```bash
# Train for elbow (quick, no Optuna)
python scripts/sysid/train_model/gru_model_train.py \
    --joint-type elbow \
    --data-dirs "/path/to/SysID Position 0" "/path/to/SysID Position 3" \
    --skip-optuna

# Train for shoulder pitch with Optuna search
python scripts/sysid/train_model/gru_model_train.py \
    --joint-type shoulder_pitch \
    --data-dirs "/path/to/Config A" "/path/to/Config B" "/path/to/Config C" \
    --trials 30 --final-epochs 150

# Generic column names (no joint prefix in CSVs)
python scripts/sysid/train_model/gru_model_train.py \
    --data-dirs "/path/to/data" \
    --skip-optuna
```

Output: `gru_{joint}_stateful_best.pt` + `gru_{joint}_stateful_stats.json`

### Hybrid Residual GRU

Instead of predicting full torque, the GRU learns only what the physics model cannot explain:

    tau_residual = tau_real - (kp * pos_error - kd * velocity) - c * sign(vel) - b * vel

The physics model (from the implicit actuator YAML) handles:
- **PD torque**: `kp * pos_error - kd * velocity`
- **Coulomb friction**: `dynamic_friction * sign(velocity)` (if non-zero in YAML)
- **Viscous friction**: `viscous_friction * velocity` (if non-zero in YAML)

Since the residual is much smaller than full torque, prediction errors compound far less during closed-loop simulation. The residual captures gravity, inertia effects, sensor bias, and other unmodeled dynamics.

When using the sysid-identified YAML (`h1_arm_sysid_implicit.yaml`), friction is automatically subtracted from the residual. With the default YAML (`h1_arm_implicit.yaml` where friction values are zero), only PD torque is subtracted (original behavior).

The residual's mean (stored in normalization stats) acts as an implicit **torque bias correction** — any constant offset between sim and real torque sensors is captured here and applied during inference.

```bash
# Train with sysid params (subtracts PD + friction from residual)
python scripts/sysid/train_model/hybrid_model_train.py \
    --implicit-yaml input/actuator_models/h1/h1_arm_sysid_implicit.yaml \
    --joint-type elbow \
    --data-dirs "/path/to/SysID Position 0" "/path/to/SysID Position 3" \
    --noise-std 0.02 \
    --skip-optuna

# Train with default params (subtracts PD only, friction=0)
python scripts/sysid/train_model/hybrid_model_train.py \
    --implicit-yaml input/actuator_models/h1/h1_arm_implicit.yaml \
    --joint-type elbow \
    --data-dirs "/path/to/SysID Position 0" "/path/to/SysID Position 3" \
    --noise-std 0.02 \
    --skip-optuna

# Train for shoulder pitch with Optuna search
python scripts/sysid/train_model/hybrid_model_train.py \
    --implicit-yaml input/actuator_models/h1/h1_arm_sysid_implicit.yaml \
    --joint-type shoulder_pitch \
    --data-dirs "/path/to/Config A" "/path/to/Config B" "/path/to/Config C" \
    --noise-std 0.01 \
    --trials 30 --final-epochs 150
```

Output: `hybrid_residual_{joint}_best.pt` + `hybrid_residual_{joint}_stats.json`

The stats JSON is tagged with `"model_type": "hybrid_residual"` and includes:
- `physics_params`: kp, kd, armature, dynamic_friction, viscous_friction used during training
- `subtract_friction`: whether friction terms were subtracted from the residual
- `normalization`: per-column mean/std/p1/p99 (residual mean = torque bias)

### How Training Works

1. **Two-pass data loading**: Pass 1 loads all CSVs raw and computes normalization stats (mean, std, p1, p99). For the hybrid model, the residual is also computed in this pass. Pass 2 clip-normalizes all columns and uploads to GPU as BF16 tensors.
2. **Stateful TBPTT**: N parallel CSV streams (default 64), each carrying GRU hidden state across chunks within a CSV. Hidden state is reset only at CSV boundaries. Gradients are detached every chunk_len steps.
3. **Noise injection** (`--noise-std`): Adds Gaussian noise to normalized inputs during training. Makes the model robust to sim state drift. Recommended: 0.01-0.05.
4. **Optuna search**: Tunes hidden_dim, num_layers, chunk_len, lr, dropout, huber_delta, weight_decay. Use `--skip-optuna` to train with default params.

### Shared CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data-dirs` | required | Directories containing training CSV files |
| `--joint-type` | None | Joint type suffix (e.g., `elbow`, `shoulder_pitch`). Sets CSV column prefix |
| `--noise-std` | 0.0 | Gaussian noise std on normalized inputs |
| `--save-dir` | `input/actuator_models/h1/` | Output directory for model and stats files |
| `--trials` | 30 | Optuna trials |
| `--trial-epochs` | 30 | Epochs per Optuna trial |
| `--final-epochs` | 150 | Epochs for final retrain |
| `--n-streams` | 64 | Parallel TBPTT streams |
| `--skip-optuna` | off | Skip Optuna; train with default hyperparameters |
| `--study-name` | varies | Optuna study name |
| `--storage` | None | Optuna DB URL for distributed search |

The hybrid script has one additional required argument: `--implicit-yaml` (path to the implicit actuator YAML with kp, kd values).

### Column Names

If `--joint-type` is set (e.g., `--joint-type elbow`), the scripts look for CSV columns with that prefix: `elbow_position`, `elbow_position_error`, `elbow_velocity`, `elbow_torque`.

If `--joint-type` is not set, they use generic names: `position`, `position_error`, `velocity`, `torque`.

## Motion Generation

`motion_generation/generate_chirp.py` generates linear chirp (frequency-sweep) motion files for system identification. Chirps sweep from a low frequency to a high frequency over a set duration, exciting a range of dynamics useful for parameter fitting.

### Basic Usage

```bash
# Single elbow chirp (0.1-5 Hz, 20s, 0.5 rad amplitude)
python scripts/sysid/motion_generation/generate_chirp.py \
    --joints elbow --f0 0.1 --f1 5.0 --duration 20

# All H1 arm joints sequentially (root first: pitch -> roll -> yaw -> elbow)
python scripts/sysid/motion_generation/generate_chirp.py \
    --joints all_arms --f0 0.1 --f1 10.0 --duration 30

# UR10e elbow (shorthand resolved to elbow_joint)
python scripts/sysid/motion_generation/generate_chirp.py \
    --robot ur10e --joints elbow --f0 0.1 --f1 5.0

# UR10e all joints sequentially (root first)
python scripts/sysid/motion_generation/generate_chirp.py \
    --robot ur10e --joints all --f0 0.1 --f1 5.0

# Teststand
python scripts/sysid/motion_generation/generate_chirp.py \
    --robot teststand --f0 0.1 --f1 5.0
```

### Sequential vs Simultaneous

When multiple joints are specified, the default is **sequential** mode: each joint is chirped one at a time (root first, end-effector last) while all other joints hold at their bias position. This is safer for sysid because exciting the root first keeps end-effector perturbations small.

- **Sequential** (default): Total duration = `(duration + rest_time) * num_joints`. Order: root -> end-effector.
- **Simultaneous** (`--simultaneous`): All joints chirp at the same time. Total duration = `duration`.

```bash
# Sequential (default) — one joint at a time with 2s rest between
python scripts/sysid/motion_generation/generate_chirp.py \
    --joints all_arms --rest-time 2.0

# Simultaneous — all joints move at once
python scripts/sysid/motion_generation/generate_chirp.py \
    --joints all_arms --simultaneous
```

Sequential ordering per robot:
- **H1 arms**: shoulder_pitch -> shoulder_roll -> shoulder_yaw -> elbow
- **UR10e**: shoulder_pan -> shoulder_lift -> elbow -> wrist_1 -> wrist_2 -> wrist_3

### Per-Joint Amplitudes and Biases

```bash
# Different amplitude and bias per joint
python scripts/sysid/motion_generation/generate_chirp.py \
    --joints elbow shoulder_pitch \
    --amplitude 0.5 0.3 --bias 0.8 -0.3
```

### Output Formats

By default, a `.txt` motion file is written (compatible with `run_benchmark.py`). Additional formats:

```bash
# Also write motor CSV (for sysid data input)
python scripts/sysid/motion_generation/generate_chirp.py \
    --joints elbow --output-csv

# Also write SAGE format (control.csv + state_motor.csv)
python scripts/sysid/motion_generation/generate_chirp.py \
    --joints elbow --output-sage
```

Output goes to `input/motion_files/<robot>/chirp/` by default. Override with `--output-dir`.

The auto-generated filename encodes all parameters (joints, frequency, amplitude, duration, mode) which can get long for multi-joint runs. Use `--name` to set a short custom stem:

```bash
# Short custom name (produces ur10e_all_chirp.txt + ur10e_all_chirp.png)
python scripts/sysid/motion_generation/generate_chirp.py \
    --robot ur10e --joints all --f0 0.1 --f1 5.0 --name ur10e_all_chirp
```

### Visualizing in Simulation

Play back the generated motion in Newton sim:

```bash
python scripts/sim2real_gap/run_benchmark.py \
    --robot-name ur10e \
    --motion-files input/motion_files/ur10e/chirp/ur10e_all_chirp.txt \
    --headless
```

Add `--visualizer newton` instead of `--headless` if `isaaclab_visualizers` is installed (`pip install isaaclab_visualizers[newton]`).

### Supported Robots

| Robot | Joint Shorthands | Groups |
|---|---|---|
| `h1` | `elbow`, `shoulder_pitch`, `shoulder_roll`, `shoulder_yaw` | `all_arms` |
| `ur10e` | `elbow`, `shoulder_pan`, `shoulder_lift`, `wrist_1`, `wrist_2`, `wrist_3` | `all` |
| `teststand` | `elbow` | — |

For H1, left/right mirroring is on by default (`--no-mirror` to disable). UR10e and teststand have no mirroring.

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--robot` | `h1` | Robot: `h1`, `ur10e`, `teststand` |
| `--joints` | `elbow` | Joints to excite (short names or groups) |
| `--f0` | `0.1` | Start frequency [Hz] |
| `--f1` | `5.0` | End frequency [Hz] |
| `--duration` | `20.0` | Chirp duration per joint [s] |
| `--amplitude` | `0.5` | Peak amplitude [rad] (1 or per-joint) |
| `--bias` | `0.0` | DC offset [rad] (1 or per-joint) |
| `--control-freq` | `500` | Sample rate [Hz] |
| `--simultaneous` | off | All joints at once (default: sequential) |
| `--rest-time` | `2.0` | Rest between sequential joints [s] |
| `--mirror` / `--no-mirror` | on | Left/right mirroring (H1 only) |
| `--output-csv` | off | Also write motor CSV format |
| `--output-sage` | off | Also write SAGE format |
| `--output-dir` | auto | Output directory |
| `--name` | auto | Output filename stem |
| `--no-plot` | off | Skip PNG plot generation |

## H1 Motor CSV Converter

Raw H1 motor CSVs (4 arm joints: elbow, pitch/shoulder_pitch, raise/shoulder_roll, yaw/shoulder_yaw) are auto-converted when passed to `--real-data-dir`. You can also convert manually using `convert_h1_chirp_to_csv.py`.

The converter supports two input formats:

1. **With `position_error` columns** (e.g., `arm_march2026/` per-joint recordings): commanded = position + position_error. Auto-detected.
2. **Without `position_error`** (old chirp/sine experiments): commanded positions reconstructed from filename-encoded experiment parameters.

```bash
# Manual conversion — recursive (all subdirs):
python scripts/sysid/convert_h1_chirp_to_csv.py \
    --data-dir input/motion_files/h1/arm_march2026 \
    --output-dir output/sysid/h1_arm_march2026 \
    --recursive

# Per-file mode (one output folder per CSV):
python scripts/sysid/convert_h1_chirp_to_csv.py \
    --data-dir input/motion_files/h1/arm_march2026/elbow \
    --output-dir output/sim2real_benchmark/real/h1/custom/elbow \
    --per-file

# Old format (chirp/sine) — reconstructs from filename params:
python scripts/sysid/convert_h1_chirp_to_csv.py \
    --data-dir input/motion_files/h1/right_arm \
    --output-dir input/motion_files/h1/right_arm_csv \
    --ramp-time 8.0
```

### Converter CLI Flags

| Flag | Description |
|---|---|
| `--recursive` | Walk subdirectories; each subdir with motor CSVs becomes a separate output folder |
| `--per-file` | One output folder per CSV file (default: concatenate all into one) |
| `--mode auto\|direct\|chirp` | Force conversion mode (`auto` detects `position_error` columns) |
| `--ramp-time` | Ramp duration to skip in chirp/sine mode (default: 8.0s) |
| `--max-files` | Limit to first N files per subdir |

### Old Chirp/Sine Filename Format

- Chirp: `C{id}_f{start}-{end}_a{amp}[_{pattern}]_motor.csv`
- Sine: `S{id}_f{freq}_a{amp}_{pattern}_motor.csv`
- Phase patterns: `inphase` (all at 0°), `antiphase` (elbow vs others at 180°), `wave` (0°, 90°, 180°, 270°)
