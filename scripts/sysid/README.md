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

Notes:
- `viscous_friction` writes directly to Newton's `mujoco.dof_passive_damping` custom attribute (no Isaac Lab API exists). This is a MuJoCo-solver-only property (default 0.0), separate from PD gains.
- `stiffness` and `damping` are PD controller gains, not passive mechanical properties. They can be optimized alongside `viscous_friction`.
- All 5 parameters can be used together in any combination.

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

Up to 5 motor properties per joint type (configured in bounds YAML):

| Symbol | YAML key | H1 Bounds | Description | Newton Property |
|---|---|---|---|---|
| J | `armature` | [0.001, 0.05] | Rotor inertia (kg·m²) | `joint_armature` |
| c | `dynamic_friction` | [0.0, 1.0] | Coulomb friction (N·m) | `joint_friction` |
| b | `viscous_friction` | [0.0, 2.0] | Passive viscous damping (N·m·s/rad) | `mujoco.dof_passive_damping` |
| kp | `stiffness` | *(commented)* | PD position gain (N·m/rad) | `joint_target_ke` |
| kd | `damping` | *(commented)* | PD velocity gain (N·m·s/rad) | `joint_target_kd` |

By default, J, c, b are optimized while kp and kd are fixed. Uncomment `stiffness` and/or `damping` in the bounds YAML to include PD gains in the optimization.

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

Instead of predicting full torque, the GRU learns only what the PD model cannot explain:

    tau_residual = tau_real - (kp * pos_error - kd * velocity)

The PD model (kp, kd from the implicit actuator YAML) handles the bulk of the dynamics. Since the residual is much smaller than full torque, prediction errors compound far less during closed-loop simulation.

```bash
# Train hybrid residual model for elbow (quick, no Optuna)
python scripts/sysid/train_model/hybrid_model_train.py \
    --implicit-yaml input/actuator_models/h1/h1_arm_implicit.yaml \
    --joint-type elbow \
    --data-dirs "/path/to/SysID Position 0" "/path/to/SysID Position 3" \
    --noise-std 0.02 \
    --skip-optuna

# Train for shoulder pitch with Optuna search
python scripts/sysid/train_model/hybrid_model_train.py \
    --implicit-yaml input/actuator_models/h1/h1_arm_implicit.yaml \
    --joint-type shoulder_pitch \
    --data-dirs "/path/to/Config A" "/path/to/Config B" "/path/to/Config C" \
    --noise-std 0.01 \
    --trials 30 --final-epochs 150
```

Output: `hybrid_residual_{joint}_best.pt` + `hybrid_residual_{joint}_stats.json`

The stats JSON is tagged with `"model_type": "hybrid_residual"` and includes the physics params (kp, kd) used to compute the residual, so downstream inference knows to add PD torque back.

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
