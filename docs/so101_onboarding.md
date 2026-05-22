# SO-101 Newton onboarding

This is the short path for a coworker who wants to get started with the SO-101
SysID and sim2real benchmark workflow in IsaacLab-Newton.

Use this as the entry point before reading the deeper investigation note in
`docs/so101_balanced_sysid.md`.

## What this workflow does

There are two separate steps:

1. **SysID training**: fit one global SO-101 actuator parameter set from
   training motions.
2. **Sim2real benchmark**: replay held-out real motions in Newton and compare
   simulated joint position/torque against measured real data.

Do not train and report on the same motions. Use train motions for fitting and
held-out motions for validation.

## Branch

Use the existing IsaacLab-Newton repo. Do not fork for onboarding.

```bash
git clone git@github.com:es-rl/IsaacLab-Newton.git
cd IsaacLab-Newton
git checkout vbhavanantha/so101-balanced-sysid
```

On Vaibhav's machine the working checkout is:

```text
/home/vbhavanantha/IsaacLab-Newton-g1-repro
```

## Environment

Use the IsaacLab conda environment:

```bash
cd /path/to/IsaacLab-Newton
source /home/vbhavanantha/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
```

If running on a different machine, replace the conda path and `--experience`
path in the commands below with that machine's IsaacLab-Newton checkout.

## Starter data bundle

The starter bundle should be unzipped into the repo root so this folder exists:

```text
input/sysid_data/so101/smoke_per_motion/
```

Expected contents:

```text
input/sysid_data/so101/smoke_per_motion/
├── split_manifest.json
├── actuator_bandwidth/
│   ├── control.csv
│   ├── event.csv
│   ├── joint_list.txt
│   └── state_motor.csv
├── boundary_exploration/
│   ├── control.csv
│   ├── event.csv
│   ├── joint_list.txt
│   └── state_motor.csv
└── backlash_detection/
    ├── control.csv
    ├── event.csv
    ├── joint_list.txt
    └── state_motor.csv
```

`actuator_bandwidth` and `boundary_exploration` are the smoke train motions.
`backlash_detection` is the smoke held-out validation motion.

## Smoke balanced SysID

This checks that the balanced per-motion training path runs. It is not a final
fit and should not be reported as a model result.

```bash
./isaaclab.sh -p scripts/sysid/run_sysid.py \
  --robot-name so101 \
  --balanced-real-data-root input/sysid_data/so101/smoke_per_motion \
  --balanced-manifest input/sysid_data/so101/smoke_per_motion/split_manifest.json \
  --balanced-split train \
  --balanced-max-motions 2 \
  --physics-freq 500 \
  --control-freq 500 \
  --num-envs 8 \
  --max-iter 1 \
  --epsilon 0 \
  --output-dir output/sysid/so101/smoke_balanced \
  --headless \
  --experience /path/to/IsaacLab-Newton/apps/isaaclab.python.headless.kit
```

Expected output:

```text
output/sysid/so101/smoke_balanced/
├── best_params.yaml
├── optimization_log.csv
└── run_summary.yaml
```

## Smoke benchmark

This checks the held-out benchmark replay path using the default
`input/actuator_models/so101/so101_implicit.yaml` actuator.

```bash
./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name so101 \
  --motion-files input/sysid_data/so101/smoke_per_motion/backlash_detection \
  --motion-name smoke_holdout \
  --output-folder output/sim2real_benchmark_so101_smoke \
  --real-init-pose-sync \
  --headless \
  --experience /path/to/IsaacLab-Newton/apps/isaaclab.python.headless.kit
```

Then run analysis:

```bash
./isaaclab.sh -p scripts/sim2real_gap/run_analysis.py \
  --robot-name so101 \
  --result-folder output/sim2real_benchmark_so101_smoke \
  --output-dir output/sim2real_analysis_so101_smoke \
  --sample-dt 0.005 \
  --headless \
  --experience /path/to/IsaacLab-Newton/apps/isaaclab.python.headless.kit
```

Expected benchmark/analysis outputs:

```text
output/sim2real_benchmark_so101_smoke/
output/sim2real_analysis_so101_smoke/
```

## Simulating your own SO-101 recording

After the smoke run works, replace the smoke motion with your own recorded
SO-101 motion. The branch is ready for this workflow; the key requirement is
that the recording uses the expected SAGE folder format.

Place one recorded motion here:

```text
input/sysid_data/so101/my_motion/
├── control.csv
├── state_motor.csv
├── joint_list.txt
└── event.csv              # optional, but useful when available
```

`joint_list.txt` must use these USD joint names in the same order as the
position arrays in `control.csv` and `state_motor.csv`:

```text
Rotation
Pitch
Elbow
Wrist_Pitch
Wrist_Roll
Jaw
```

The CSVs should look like the smoke data:

```text
control.csv:     type,timestamp,positions
state_motor.csv: type,timestamp,positions,velocities,torques
```

Timestamps are in microseconds. Joint positions are in radians. Torques should
be in N m if you want torque RMSE to be meaningful; if the robot logger reports
servo current, convert current to torque before treating the values as N m.

Run the benchmark on your motion:

```bash
./isaaclab.sh -p scripts/sim2real_gap/run_benchmark.py \
  --robot-name so101 \
  --motion-files input/sysid_data/so101/my_motion \
  --motion-name my_motion \
  --output-folder output/sim2real_benchmark_so101_my_motion \
  --real-init-pose-sync \
  --headless \
  --experience /path/to/IsaacLab-Newton/apps/isaaclab.python.headless.kit
```

Then analyze it:

```bash
./isaaclab.sh -p scripts/sim2real_gap/run_analysis.py \
  --robot-name so101 \
  --result-folder output/sim2real_benchmark_so101_my_motion \
  --output-dir output/sim2real_analysis_so101_my_motion \
  --sample-dt 0.005 \
  --headless \
  --experience /path/to/IsaacLab-Newton/apps/isaaclab.python.headless.kit
```

For a brand-new IsaacLab-Newton user, do this in order:

1. Clone the repo and switch to the SO-101 branch.
2. Activate the IsaacLab environment.
3. Unzip and run the smoke bundle exactly as written above.
4. Copy the smoke folder shape for their own recording.
5. Run the benchmark and analysis on their own held-out motion.
6. Only then start fitting SysID parameters.

## Methodology rules

- Primary validation numbers should use held-out motions, not training motions.
- Balanced SysID keeps one global parameter vector but scores each motion as an
  independent episode.
- Each balanced training motion starts from measured row 0, uses the settle
  buffer, replays recorded commands, then contributes one motion loss.
- Timestamp alignment samples command/state onto a common grid. It is not
  time-warping, phase-shifting, or peak matching.
- Torque RMSE only means something if `state_motor.csv` torques are in N m.
- SO-101 torque improvement is a validation signal here; the balanced fit uses
  position loss unless the objective is explicitly changed.

## Deeper docs

Read these after the smoke run works:

```text
docs/so101_balanced_sysid.md
scripts/sysid/README.md
scripts/sim2real_gap/README.md
```

The current held-out validation result and caveats are documented in
`docs/so101_balanced_sysid.md`.
