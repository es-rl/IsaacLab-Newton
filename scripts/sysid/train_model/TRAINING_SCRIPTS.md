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
| `train_gru_enriched_h1.py` | Enriched (24-feat) | 6×4 block (below) | `tau_real - (kp·pos_err - kd·vel)` residual | `ForceResidualGRU` | `hybrid_residual` (PD + `0.33·GRU`) |

The **24-feature enriched layout** is six 4-joint blocks concatenated:

```text
[q, position_error, velocity, PD_hint, qfrc_bias, previous_torque]   # ×4 joints = 24
```

## Simple vs. enriched

|  | Simple (3-feat) | Enriched (24-feat) |
|---|---|---|
| Model class | `TorqueGRU` — GRU + `Linear(hidden, 1)`, one joint at a time | `ForceResidualGRU` — GRU + `Linear(hidden, 4)`, `tanh`-bounded to `±force_bound`, all 4 arm joints jointly |
| Inputs | 3 scalars | 24 scalars (adds PD hint, `qfrc_bias` gravity/bias term, and previous torque) |
| Temporal context | GRU hidden state via **stateful TBPTT** across long CSV streams | GRU hidden state over fixed 200-step windows (`prev_torque` also feeds history explicitly) |
| Normalization | per-column clip to p1/p99 + z-score, stats saved with mean/std/p1/p99 | per-feature z-score (mean/std only), 24-vector sidecar |
| Loss | Huber (`delta` tuned) | MSE, target clamped to `±force_bound` |
| Hyperparameter search | Optuna (TPE + Hyperband) | fixed, pre-tuned hparams (Optuna winner baked in) |
| Runnable in this repo | **Yes** — standalone, PyTorch only | Needs the raw motion datasets + arm MJCF (see below) |

## Full-torque vs. residual (applies to both families)

- **Full-torque** (`train_gru_simple.py`, `train_gru_enriched_g1.py`): the network
  predicts the entire motor torque. At deploy the solver PD is disabled
  (`kp=0, kd=0`) and the network output *is* the applied torque.
- **Residual** (`train_gru_simple_residual.py`, `train_gru_enriched_h1.py`): the
  network predicts only what the PD model cannot explain,
  `tau_real - (kp·pos_err - kd·vel)`. At deploy the implicit/SysID PD actuator runs
  and the GRU residual is added on top. The enriched H1 path scales the residual by
  `0.33` at runtime (`applied = pd_torque + 0.33·GRU`). Because the residual is much
  smaller than full torque, prediction errors compound less in closed-loop sim.

## Enriched trainers — dependencies & provenance

`train_gru_enriched_g1.py` and `train_gru_enriched_h1.py`:

- **G1** (`train_gru_enriched_g1.py`): hidden 192, 2 layers, 25 epochs, `force_bound=15`,
  target = full torque. Uses the 32-motion `enriched_split_g1.json` pool.
- **H1** (`train_gru_enriched_h1.py`): hidden 128, 2 layers, `force_bound=5` (via
  `--force-bound`), `--residual-target`, `tau_clip=10`. Uses
  `enriched_split_h1_combined.json` (v28 recipe); `enriched_split_h1.json` is the
  trainer's default manifest.

Shared modules in this folder:

| File | Role |
|---|---|
| `enriched_model.py` | `ForceResidualGRU` — the shared 24-in / 4-out bounded GRU |
| `enriched_normalization.py` | per-feature z-score fit/apply/save/load |
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

### `qfrc_bias`: computed from the data directory

The `qfrc_bias` feature block is MuJoCo's bias force `C(q, q̇)·q̇ + g(q)`
(Coriolis/centrifugal + gravity). It is not in the raw robot logs — it is derived
from each motion's positions/velocities with a MuJoCo model via
`mujoco.mj_forward → data.qfrc_bias`.

**The trainer computes it for you.** Pass `--model-xml` (the arm MJCF) and the
trainer computes `qfrc_bias` on the fly from the `--data-root`/`--sage-root`
trajectories — no separate step:

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

## Selecting a trained model

Model selection is config-driven — there is no CLI `--model` flag. Point the
`actuator:` section of `input/run_configs/<robot>/<robot>.yaml` at the checkpoint
and set the matching `model_type`:

```yaml
actuator:
  model_type: full_torque_enriched      # or: gru | hybrid_residual
  yaml_file: g1/g1_arm_no_pd.yaml
  network_file: g1/<your_model>_script.pt
  stats_file: g1/<your_model>_stats.json
```
