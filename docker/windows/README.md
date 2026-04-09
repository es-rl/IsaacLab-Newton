# Windows Docker Setup for Isaac Lab Sysid

Run the `isaac-lab-sysid` container on Windows with GPU acceleration and
optional GUI (Newton visualizer).

This guide assumes you are running **Docker Engine inside WSL 2** — no Docker
Desktop required.

## Prerequisites

1. **Windows 10/11** with WSL 2 enabled and an Ubuntu distro installed.
2. **NVIDIA GPU drivers** installed on the Windows host (Game Ready or Studio).
   The driver is automatically visible inside WSL 2 — do **not** install a
   separate Linux driver inside WSL.
3. **(GUI only) VcXsrv** — free X server for Windows:
   <https://sourceforge.net/projects/vcxsrv/>

## One-time WSL 2 setup

All commands below run inside your WSL 2 Ubuntu shell.

### 1. Install Docker Engine

```bash
# Remove any old installs
sudo apt remove docker docker-engine docker.io containerd runc

# Install prerequisites
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key and repo
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list

# Install Docker Engine and Compose plugin
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

# Add your user to the docker group (avoids needing sudo for every command)
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Auto-start Docker on WSL launch

WSL 2 does not run systemd by default, so add this to your `~/.bashrc` or
`~/.profile`:

```bash
if service docker status 2>&1 | grep -q "not running"; then
    sudo service docker start
fi
```

### 3. Install NVIDIA Container Toolkit (GPU support)

```bash
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo service docker restart
```

Verify GPU access:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

### 4. Build the images

```bash
# From the repo root inside WSL 2:
cd docker

# Build the base image (only needed once, or when Dockerfile.base changes)
docker compose --env-file .env.base --profile base build isaac-lab-base

# Build the sysid image
docker compose --env-file .env.base --profile sysid build isaac-lab-sysid
```

## Running

### Headless (no GUI)

```bash
# Interactive shell
HEADLESS=1 bash docker/windows/run-gui.sh

# Run a command
HEADLESS=1 bash docker/windows/run-gui.sh \
    python scripts/sim2real_gap/run_benchmark.py --robot-name ur10e --headless
```

### With GUI (Newton visualizer)

1. Launch **XLaunch** (VcXsrv) on Windows with these settings:
   - Display number: `0`
   - Start no client
   - Check **"Disable access control"**
   - Uncheck "Native opengl" (let the container use its own GL)

2. Find your Windows host IP (run in PowerShell):
   ```powershell
   (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "vEthernet (WSL)").IPAddress
   ```

3. Inside WSL 2, set `DISPLAY` and run the container:
   ```bash
   export DISPLAY=<windows-host-ip>:0.0
   bash docker/windows/run-gui.sh
   ```

4. Inside the container, run with `--visualizer newton`:
   ```bash
   python scripts/sim2real_gap/run_benchmark.py \
       --robot-name ur10e \
       --motion-files input/motion_files/ur10e/chirp/*.txt \
       --visualizer newton
   ```

## How it works

- `run-gui.sh` — runs inside WSL 2.
- `x11.yaml` — Docker Compose overlay that sets `DISPLAY` and switches from
  `network_mode: host` (not supported on Windows) to bridge mode.
- `Dockerfile.sysid` is platform-agnostic and shared with Linux.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `docker: command not found` | Run `sudo service docker start` — the daemon is not running. Add the auto-start snippet to `~/.bashrc`. |
| `permission denied /var/run/docker.sock` | Run `sudo usermod -aG docker $USER && newgrp docker`. |
| `Cannot connect to X server` | Make sure VcXsrv is running with "Disable access control" checked, and `DISPLAY` points to the correct Windows host IP. |
| Black screen in Newton viewer | Uncheck "Native opengl" in XLaunch settings. |
| `docker: Error response ... GPU` | Verify NVIDIA Container Toolkit is installed and `nvidia-smi` works inside WSL 2 (no Docker). |
| `WSL environment detected but no adapters were found` | WSL 2 doesn't support cgroups v1. Fix: `sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups=true --in-place && sudo service docker restart`. If it still fails, use CDI mode: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` then run with `--device nvidia.com/gpu=all` instead of `--gpus all`. |
| `nvidia-smi` fails in WSL 2 (no Docker) | Your Windows NVIDIA driver is too old or doesn't support WSL 2 GPU passthrough. Update to Game Ready 515+ or Studio driver. |
| `network_mode host not supported` | Expected on Windows. The `x11.yaml` overlay switches to bridge mode automatically. |
