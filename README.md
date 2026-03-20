![Isaac Lab](docs/source/_static/isaaclab.jpg)

---

# Isaac Lab (with Newton SysID toolbox)

[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


This branch is a development branch for Isaac Sim 6.0, which is currently only available through the Isaac Sim [GitHub repo](https://github.com/isaac-sim/IsaacSim).
For installation, please refer to the Isaac Sim GitHub repo to build the latest Isaac Sim branch, and follow the binary installation method in the
Isaac Lab documentation for Isaac Lab installation.

Note that this branch is currently under active development and may experience breaking changes or error messages.
Performance issues and regressions may also be observed in some use cases.


**Isaac Lab** is a GPU-accelerated, open-source framework designed to unify and simplify robotics research workflows,
such as reinforcement learning, imitation learning, and motion planning. Built on [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html),
it combines fast and accurate physics and sensor simulation, making it an ideal choice for sim-to-real
transfer in robotics.

Isaac Lab provides developers with a range of essential features for accurate sensor simulation, such as RTX-based
cameras, LIDAR, or contact sensors. The framework's GPU acceleration enables users to run complex simulations and
computations faster, which is key for iterative processes like reinforcement learning and data-intensive tasks.
Moreover, Isaac Lab can run locally or be distributed across the cloud, offering flexibility for large-scale deployments.

A detailed description of Isaac Lab can be found in our [arXiv paper](https://arxiv.org/abs/2511.04831).

## Version Info

| Component | Version |
|---|---|
| Isaac Lab | [3.0.0 (develop branch)](https://github.com/isaac-sim/IsaacLab/tree/develop) |
| Isaac Sim | [6.0.0](https://github.com/isaac-sim/IsaacSim) |
| Newton | [release-1.0](https://github.com/newton-physics/newton/tree/release-1.0) |
| MuJoCo | >= 3.5.0 |
| MuJoCo Warp | >= 3.5.0 |

> **Note:** Isaac Lab 3.0 is on the [`develop` branch](https://github.com/isaac-sim/IsaacLab/tree/develop), not `main`. It is currently only available on Ubuntu. Windows support and pip wheels are not yet available.

## Key Features

Isaac Lab offers a comprehensive set of tools and environments designed to facilitate robot learning:

- **Robots**: A diverse collection of robots, from manipulators, quadrupeds, to humanoids, with more than 16 commonly available models.
- **Environments**: Ready-to-train implementations of more than 30 environments, which can be trained with popular reinforcement learning frameworks such as RSL RL, SKRL, RL Games, or Stable Baselines. We also support multi-agent reinforcement learning.
- **Physics**: Rigid bodies, articulated systems, deformable objects
- **Sensors**: RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, ray casters.

## System Identification & Sim2Real Gap Estimation

This repo includes scripts for **system identification (sysid)** and **sim-to-real actuator gap estimation**, targeting the Newton physics backend with the MuJoCo Warp solver.

### SysID (`scripts/sysid/`)

CMA-ES optimization of robot actuator parameters (armature, friction, viscous damping, PD gains). Replays real robot data in N parallel sim environments and minimizes position MSE vs measured response.

- **Supported robots**: H1 (mirrored left/right arms), UR10e
- **Input modes**: Single-joint parquet chirp files, multi-joint CSVs (control.csv + state_motor.csv), raw motor CSVs (auto-converted)
- **GRU model training**: Standard GRU (full torque) and hybrid residual GRU (learns what PD model can't explain), with Optuna hyperparameter search

See [`scripts/sysid/README.md`](scripts/sysid/README.md) for full documentation. Based on: [PACE Sim2Real](https://github.com/leggedrobotics/pace-sim2real) (ETH Zurich)


### Sim2Real Gap Estimation (`scripts/sim2real_gap/`)

Sim-to-real actuator gap estimation with the Newton physics backend. Two-stage workflow: **benchmark** replays joint motion trajectories in Newton simulation and records sim joint states; **analysis** compares sim vs real data to quantify the actuator gap. 

- **Benchmark**: Runs motions through multiple actuator models (implicit PD, DC motor, LSTM/GRU) and outputs sim joint states
- **Analysis**: Per-joint RMSE, correlation, cosine similarity plots and metrics comparing sim vs real
- **Auto-conversion**: Motor CSVs from sysid experiments are auto-converted to the expected format

See [`scripts/sim2real_gap/README.md`](scripts/sim2real_gap/README.md) for full documentation. Based on: [SAGE](https://github.com/isaac-sim2real/sage)


### Prerequisites

This repo uses **Git LFS** for large binary assets (USD robot models, trained `.pt` models, mesh files). Install and pull LFS objects before building:

```bash
# Install Git LFS (if not already installed)
sudo apt install git-lfs   # Ubuntu/Debian
git lfs install

# Pull LFS objects (required after cloning)
git lfs pull
```

Without `git lfs pull`, robot model files in `input/robot_models/` will be LFS pointer files instead of actual data, causing runtime errors.

### Quick Start

```bash
# Build all Docker images (only needed once, each step builds on the previous)
./docker/build-docker.sh

# Or build a specific target (automatically builds dependencies)
./docker/build-docker.sh sysid              # isaacsim → base → sysid
./docker/build-docker.sh base               # isaacsim → base
./docker/build-docker.sh sysid --no-cache   # rebuild sysid only (keeps cached deps)
./docker/build-docker.sh --no-cache         # rebuild everything from scratch

# Launch an interactive shell (with GUI/X11 forwarding)
./docker/run-gui.sh

# Headless mode (no X11)
./docker/run-headless.sh python scripts/sysid/run_sysid.py --robot-name h1 --headless

# Key commands to run inside Docker container
# NOTE: modify the config yaml in input/run_configs for each robot
python scripts/sysid/run_sysid.py --robot-name h1 --headless
python scripts/sim2real_gap/run_benchmark.py --robot-name h1 --headless
python scripts/sim2real_gap/run_analysis.py --robot-name h1
```

The `run-gui.sh` script sets up X11 forwarding for GUI apps and reuses an existing container across invocations.

### Directory Structure

```
input/                           # Data and configs (bind-mounted in Docker)
├── actuator_models/             # Actuator parameter YAMLs and trained .pt models
│   ├── h1/                      # H1 implicit, dcmotor, GRU/LSTM models
│   └── ur10e/                   # UR10e implicit actuator params
├── run_configs/                 # Per-robot runtime configs (simulation, sysid, benchmark)
│   ├── h1/h1.yaml
│   └── ur10e/ur10e.yaml
├── robot_models/                # USD/URDF robot assets
└── motion_files/                # Motion trajectory files (e.g. sysid data)

scripts/
├── sysid/                       # System identification scripts
│   ├── run_sysid.py             # CMA-ES optimizer driver
│   ├── optimizer.py             # CMA-ES wrapper
│   ├── convert_h1_chirp_to_csv.py
│   ├── convert_pkl_to_csv.py
│   ├── convert_ur10_urdf.py
│   ├── diagnose_newton_mapping.py
│   └── train_model/             # GRU/hybrid model training
│       ├── gru_model_train.py
│       └── hybrid_model_train.py
└── sim2real_gap/                # Sim2real gap benchmark & analysis
    ├── run_benchmark.py         # Newton sim playback
    ├── run_analysis.py          # Sim vs real comparison
    ├── newton_benchmark.py      # Core benchmark implementation
    ├── convert_real_data.py
    ├── convert_ur10e_pkl.py
    ├── generate_sample_motion.py
    └── configs/                 # Joint configs for sim2real gap
```

## Getting Started

### Documentation

Our [documentation page](https://isaac-sim.github.io/IsaacLab) provides everything you need to get started, including
detailed tutorials and step-by-step guides. Follow these links to learn more about:

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)


## Troubleshooting

Please see the [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html) section for
common fixes or [submit an issue](https://github.com/isaac-sim/IsaacLab/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/isaac-sim/IsaacLab/discussions) for discussing ideas,
  asking questions, and requests for new features.
* Github [Issues](https://github.com/isaac-sim/IsaacLab/issues) should only be used to track executable pieces of
  work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features,
  or general updates.

## Connect with the NVIDIA Omniverse Community

Do you have a project or resource you'd like to share more widely? We'd love to hear from you!
Reach out to the NVIDIA Omniverse Community team at OmniverseCommunity@nvidia.com to explore opportunities
to spotlight your work.

You can also join the conversation on the [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) to
connect with other developers, share your projects, and help grow a vibrant, collaborative ecosystem
where creativity and technology intersect. Your contributions can make a meaningful impact on the Isaac Lab
community and beyond!

## License

The Isaac Lab framework is released under [BSD-3 License](LICENSE). The `isaaclab_mimic` extension and its
corresponding standalone scripts are released under [Apache 2.0](LICENSE-mimic). The license files of its
dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

Note that Isaac Lab requires Isaac Sim, which includes components under proprietary licensing terms. Please see the [Isaac Sim license](docs/licenses/dependencies/isaacsim-license.txt) for information on Isaac Sim licensing.

Note that the `isaaclab_mimic` extension requires cuRobo, which has proprietary licensing terms that can be found in [`docs/licenses/dependencies/cuRobo-license.txt`](docs/licenses/dependencies/cuRobo-license.txt).


## Citation

If you use Isaac Lab in your research, please cite the technical report:

```
@article{mittal2025isaaclab,
  title={Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning},
  author={Mayank Mittal and Pascal Roth and James Tigue and Antoine Richard and Octi Zhang and Peter Du and Antonio Serrano-Muñoz and Xinjie Yao and René Zurbrügg and Nikita Rudin and Lukasz Wawrzyniak and Milad Rakhsha and Alain Denzler and Eric Heiden and Ales Borovicka and Ossama Ahmed and Iretiayo Akinola and Abrar Anwar and Mark T. Carlson and Ji Yuan Feng and Animesh Garg and Renato Gasoto and Lionel Gulich and Yijie Guo and M. Gussert and Alex Hansen and Mihir Kulkarni and Chenran Li and Wei Liu and Viktor Makoviychuk and Grzegorz Malczyk and Hammad Mazhar and Masoud Moghani and Adithyavairavan Murali and Michael Noseworthy and Alexander Poddubny and Nathan Ratliff and Welf Rehberg and Clemens Schwarke and Ritvik Singh and James Latham Smith and Bingjie Tang and Ruchik Thaker and Matthew Trepte and Karl Van Wyk and Fangzhou Yu and Alex Millane and Vikram Ramasamy and Remo Steiner and Sangeeta Subramanian and Clemens Volk and CY Chen and Neel Jawale and Ashwin Varghese Kuruttukulam and Michael A. Lin and Ajay Mandlekar and Karsten Patzwaldt and John Welsh and Huihua Zhao and Fatima Anes and Jean-Francois Lafleche and Nicolas Moënne-Loccoz and Soowan Park and Rob Stepinski and Dirk Van Gelder and Chris Amevor and Jan Carius and Jumyung Chang and Anka He Chen and Pablo de Heras Ciechomski and Gilles Daviet and Mohammad Mohajerani and Julia von Muralt and Viktor Reutskyy and Michael Sauter and Simon Schirm and Eric L. Shi and Pierre Terdiman and Kenny Vilella and Tobias Widmer and Gordon Yeoman and Tiffany Chen and Sergey Grizan and Cathy Li and Lotus Li and Connor Smith and Rafael Wiltz and Kostas Alexis and Yan Chang and David Chu and Linxi "Jim" Fan and Farbod Farshidian and Ankur Handa and Spencer Huang and Marco Hutter and Yashraj Narang and Soha Pouya and Shiwei Sheng and Yuke Zhu and Miles Macklin and Adam Moravanszky and Philipp Reist and Yunrong Guo and David Hoeller and Gavriel State},
  journal={arXiv preprint arXiv:2511.04831},
  year={2025},
  url={https://arxiv.org/abs/2511.04831}
}
```

If you use the system identification or sim-to-real gap estimation tooling, please also cite PACE Sim2Real:

```
@article{bjelonic2025towards,
  title         = {Towards Bridging the Gap: Systematic Sim-to-Real Transfer for Diverse Legged Robots},
  author        = {Bjelonic, Filip and Tischhauser, Fabian and Hutter, Marco},
  journal       = {arXiv preprint arXiv:2509.06342},
  year          = {2025},
  eprint        = {2509.06342},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
}
```


## Acknowledgement

Isaac Lab development initiated from the [Orbit](https://isaac-orbit.github.io/) framework.
We gratefully acknowledge the authors of Orbit for their foundational contributions.

The system identification and sim-to-real workflows in this repository build upon the following projects:

* [PACE](https://github.com/leggedrobotics/pace-sim2real): Systematic sim-to-real transfer framework for legged robots, identifying actuator and joint dynamics with standard joint encoders.
* [SAGE](https://github.com/isaac-sim2real/sage): Sim2Real Actuator Gap Estimator for benchmarking actuator model fidelity against real robot data.
