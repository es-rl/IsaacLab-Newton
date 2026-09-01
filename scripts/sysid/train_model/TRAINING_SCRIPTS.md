# GRU Training Scripts — Variations

> **Models and assets:** the pretrained checkpoints, normalization stats, and the
> motion datasets / `qfrc_bias` caches referenced below are hosted on Hugging Face:
> **<https://huggingface.co/datasets/nvidia/Anchor-Lab>**. Download them from there
> to deploy the trained models or to supply the inputs the enriched trainers expect
> (none of these binaries are committed to this repo).

This folder holds two families of GRU torque-model trainers. They differ in the
input features they consume, the target they predict, the model class, and how
the resulting checkpoint is deployed. This document summarizes those differences
so you can pick the right one (and know what each produced checkpoint expects).

## At a glance

| Script | Family | Inputs | Target | Model | Deploy `model_type` |
|---|---|---|---|---|---|
| `train_gru_simple.py` | Simple (3-feat) | `[position, position_error, velocity]` | full `torque` | `TorqueGRU` (GRU + linear head, 1 out) | `gru` / `lstm` |
| `train_gru_simple_residual.py` | Simple (3-feat) | `[position, position_error, velocity]` | `tau_real - (kp·pos_err - kd·vel)` residual | `TorqueGRU` | `hybrid_residual` |
| `train_gru_enriched_g1.py` | Enriched (24-feat) | 6×4 block (below) | full `torque` | `ForceResidualGRU` (GRU + tanh-bounded 4-joint head) | `full_torque_enriched` |
| `train_gru_enriched_h1.py` | Enriched (20-feat) | 5×4 deployable block (below) | `tau_real - (kp·pos_err - kd·vel)` residual | `ForceResidualGRU` | `enriched_residual` (PD + configurable `GRU`, default scale `0.33`) |

The **24-feature enriched layout** is six 4-joint blocks concatenated:

```text
[q, position_error, velocity, PD_hint, qfrc_bias, previous_torque]   # ×4 joints = 24
```

G1 retains that layout. H1 now defaults to a **deployable 20-feature layout**
that removes the model-dependent `qfrc_bias` block:

```text
[q, position_error, velocity, PD_hint, previous_torque]   # ×4 joints = 20
```

The old H1 24-input layout remains available as `--feature-layout legacy24` for
provenance only. It is not accepted by the `enriched_residual` runtime because
this repository does not have matching H1 runtime bias-force wiring.

## Simple vs. enriched

|  | Simple (3-feat) | Enriched (20- or 24-feat) |
|---|---|---|
| Model class | `TorqueGRU` — GRU + `Linear(hidden, 1)`, one joint at a time | `ForceResidualGRU` — GRU + `Linear(hidden, 4)`, `tanh`-bounded to `±force_bound`, all 4 arm joints jointly |
| Inputs | 3 scalars | 20 deployable H1 scalars, or 24 G1/legacy H1 scalars with `qfrc_bias` |
| Temporal context | GRU hidden state via **stateful TBPTT** across long CSV streams | GRU hidden state over fixed 200-step windows (`prev_torque` also feeds history explicitly) |
| Normalization | per-column clip to p1/p99 + z-score, stats saved with mean/std/p1/p99 | per-feature z-score (mean/std only), 20- or 24-vector sidecar plus deployable H1 contract metadata |
| Loss | Huber (`delta` tuned) | MSE, target clamped to `±force_bound` |
| Hyperparameter search | Optuna (TPE + Hyperband) | fixed, pre-tuned hparams (Optuna winner baked in) |
| Runnable in this repo | **Yes** — standalone, PyTorch only | Needs the raw motion datasets; only the G1/legacy H1 24-input layout also needs the matching arm MJCF or bias caches |

## Full-torque vs. residual (applies to both families)

- **Full-torque** (`train_gru_simple.py`, `train_gru_enriched_g1.py`): the network
  predicts the entire motor torque. At deploy the solver PD is disabled
  (`kp=0, kd=0`) and the network output *is* the applied torque.
- **Residual** (`train_gru_simple_residual.py`, `train_gru_enriched_h1.py`): the
  network predicts only what the PD model cannot explain,
  `tau_real - (kp·pos_err - kd·vel)`. At deploy the implicit/SysID PD actuator runs
  and the GRU residual is added on top. The enriched H1 path defaults to a `0.33`
  runtime scale (`applied = pd_torque + 0.33·GRU`), configurable as
  `actuator.residual_scale`. The previous-*applied-total*-torque feature makes this
  a closed-loop recurrent model; its state is reset with the environment.

## Enriched trainers — dependencies & provenance

`train_gru_enriched_g1.py` and `train_gru_enriched_h1.py`:

- **G1** (`train_gru_enriched_g1.py`): hidden 192, 2 layers, 25 epochs, `force_bound=15`,
  target = full torque. Uses the 32-motion `enriched_split_g1.json` pool.
- **H1** (`train_gru_enriched_h1.py`): hidden 128, 2 layers, deployable 20-input
  layout, and residual target by default. The constrained v28-style recipe uses
  `--force-bound 5 --tau-clip 10`. It uses `enriched_split_h1_combined.json`;
  `enriched_split_h1.json` is the trainer's default manifest.

Shared modules in this folder:

| File | Role |
|---|---|
| `enriched_model.py` | `ForceResidualGRU` — the shared 20/24-in, 4-out bounded GRU |
| `enriched_contract.py` | dimension inference, H1 feature ordering/metadata validation, and closed-loop residual state |
| `enriched_normalization.py` | per-feature z-score fit/apply/save/load, including optional contract metadata |
| `enriched_data_g1.py` | G1 per-experiment CSV loader (resamples to 500 Hz) |
| `enriched_data_h1.py` | H1 SAGE-format loader (alphabetical → canonical joint reorder) |
| `enriched_split_*.json` | train/val/test motion manifests |
| `precompute_qfrc_bias.py` | regenerate the `qfrc_bias` feature caches (below) |

### Setting the data directory

Each enriched trainer reads its per-motion trajectories
(positions/velocities/torques/targets) from a data directory you point it at; the
motion names come from the split manifest.

- **G1** (`train_gru_enriched_g1.py`): `--data-root <dir>` — a directory of
  per-experiment folders, each named exactly as the entries in
  `enriched_split_g1.json` (e.g. `T_A_01_wave_sine_20260407_115544/`) and containing
  `state_motor.csv` + `control.csv`. Default: `~/data/g1/experiments`.
- **H1** (`train_gru_enriched_h1.py`): `--sage-root <dir>` — a directory of
  per-motion SAGE folders, each named as the `enriched_split_h1*.json` entries (with
  the `_motor.csv` suffix stripped) and containing `state_motor.csv` +
  `control.csv` + `joint_list.txt`. Default: `~/data/h1/sage`.

### `qfrc_bias`: G1 and legacy H1 only

The `qfrc_bias` feature block is MuJoCo's bias force `C(q, q̇)·q̇ + g(q)`
(Coriolis/centrifugal + gravity). It is not in the raw robot logs — it is derived
from each motion's positions/velocities with a MuJoCo model via
`mujoco.mj_forward → data.qfrc_bias`.

The deployable H1 default deliberately omits this feature. For G1 or H1 with
`--feature-layout legacy24`, pass `--model-xml` (the arm MJCF) and the trainer
computes `qfrc_bias` on the fly from the `--data-root`/`--sage-root`
trajectories, or provide a matching cache:

```bash
./isaaclab.sh -p scripts/sysid/train_model/train_gru_enriched_g1.py \
    --data-root ~/data/g1/experiments \
    --model-xml /path/to/g1_right_arm.xml \
    --save-qfrc-cache          # optional: cache results to --qfrc-bias-root
```

Per motion, the bias is resolved in this order: (1) unless `--recompute-qfrc` is
set, load `--qfrc-bias-root/<motion>_qfrc_bias.npy` if it exists (its length is
validated against the trajectory); (2) else compute it with `--model-xml`;
(3) else error. So if you already have caches (e.g. the extracted `qfrc_bias_*`
sets under `~/data/{g1,h1}/qfrc_bias`), the trainer uses them and `--model-xml` is
optional. To regenerate stale caches after changing the model, pass
`--recompute-qfrc --save-qfrc-cache`. Pass the **same** MJCF used to train/deploy
the checkpoint (`qfrc_bias` is entirely model-dependent), and a fixed-base arm-only
model so non-arm joints don't perturb the result (the provider warns if the model
has extra DOFs). Override `--joint-names` if your MJCF's joint names differ from the
canonical `[shoulder_pitch, shoulder_roll, shoulder_yaw, elbow]`.

> **G1 column order.** The G1 loader has no per-motion joint metadata, so it trusts
> that the data columns are already in `--joint-names` order
> (`[shoulder_pitch, shoulder_roll, shoulder_yaw, elbow]`). If your G1 CSVs are in a
> different order, reorder the columns or the bias will be silently wrong. (H1 SAGE
> motions carry `joint_list.txt`, so the H1 path reorders automatically.)

**Or precompute once, standalone**, with the same logic exposed as a CLI helper
(`precompute_qfrc_bias.py`), then point `--qfrc-bias-root` at its `--out-dir`:

```bash
./isaaclab.sh -p scripts/sysid/train_model/precompute_qfrc_bias.py \
    --robot g1 --model-xml /path/to/g1_right_arm.xml \
    --data-root ~/data/g1/experiments \
    --split-json enriched_split_g1.json --roles train test \
    --out-dir ~/data/g1/qfrc_bias
```

### Training-vs-deploy note for enriched models

During training, `previous_torque[t]` is the *measured* real torque at `t-1`
(teacher forcing). During closed-loop Newton deployment it is the *previous
predicted/applied* torque. This is one reason the enriched deploy path needs an explicit
recurrent multi-actuator contract.

### Train and deploy the H1 enriched residual

The default command produces a 20-input residual model and embeds the feature
layout, joint order, PD gains, target definition, and recommended runtime scale
in both the checkpoint and stats sidecar:

```bash
./isaaclab.sh -p scripts/sysid/train_model/train_gru_enriched_h1.py \
    --sage-root /path/to/h1/sage \
    --split-json scripts/sysid/train_model/enriched_split_h1_combined.json \
    --force-bound 5 \
    --tau-clip 10 \
    --out output/train_gru_enriched_h1/h1_arm_enriched_residual_deployable20.pt \
    --stats-out output/train_gru_enriched_h1/h1_arm_enriched_residual_deployable20_stats.json

cp output/train_gru_enriched_h1/h1_arm_enriched_residual_deployable20_script.pt \
    input/actuator_models/h1/
cp output/train_gru_enriched_h1/h1_arm_enriched_residual_deployable20_stats.json \
    input/actuator_models/h1/

./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
    --robot-name h1 \
    --run-config input/run_configs/h1/h1_enriched_residual.yaml \
    --headless
```

Deployment rejects missing or mismatched metadata, normalization lengths,
feature counts, joint order, and PD gains before the benchmark starts. A legacy
24-input H1 checkpoint therefore cannot be silently paired with the 20-input
runtime.

## Selecting a trained model

Model selection is config-driven — there is no CLI `--model` flag. Point the
`actuator:` section of `input/run_configs/<robot>/<robot>.yaml` at the checkpoint
and set the matching `model_type`:

G1 full torque:

```yaml
actuator:
  model_type: full_torque_enriched
  yaml_file: g1/g1_arm_no_pd.yaml
  network_file: g1/<your_model>_script.pt
  stats_file: g1/<your_model>_stats.json
```

H1 residual (the complete example is
`input/run_configs/h1/h1_enriched_residual.yaml`):

```yaml
actuator:
  model_type: enriched_residual
  yaml_file: h1/h1_arm_implicit.yaml
  network_file: h1/h1_arm_enriched_residual_deployable20_script.pt
  stats_file: h1/h1_arm_enriched_residual_deployable20_stats.json
  residual_scale: 0.33
```
