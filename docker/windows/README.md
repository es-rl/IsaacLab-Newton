# Windows Docker Setup for Isaac Lab Sysid

Run the `isaac-lab-sysid` container on Windows with GPU acceleration and
optional GUI (Newton visualizer).

## Prerequisites

1. **Windows 10/11** with WSL2 enabled.
2. **Docker Desktop** with the WSL2 backend and
   [NVIDIA GPU support](https://docs.docker.com/desktop/gpu/) enabled.
3. **NVIDIA GPU drivers** installed on the Windows host (Game Ready or Studio).
4. **(GUI only) VcXsrv** — free X server for Windows:
   <https://sourceforge.net/projects/vcxsrv/>

## One-time setup

```powershell
# 1. Build the base image (only needed once, or when Dockerfile.base changes)
cd docker
docker compose --env-file .env.base --profile base build isaac-lab-base

# 2. Build the sysid image
docker compose --env-file .env.base --profile sysid build isaac-lab-sysid
```

## Running

### Headless (no GUI)

```powershell
# Interactive shell
$env:HEADLESS=1; .\docker\windows\run-gui.ps1

# Run a command
$env:HEADLESS=1; .\docker\windows\run-gui.ps1 python scripts/sim2real_gap/run_benchmark.py --robot-name ur10e --headless
```

### With GUI (Newton visualizer)

1. Launch **XLaunch** (VcXsrv) with these settings:
   - Display number: `0`
   - Start no client
   - Check **"Disable access control"**
   - Uncheck "Native opengl" (let the container use its own GL)

2. Run the container:

```powershell
.\docker\windows\run-gui.ps1
```

3. Inside the container, run with `--visualizer newton`:

```bash
python scripts/sim2real_gap/run_benchmark.py \
    --robot-name ur10e \
    --motion-files input/motion_files/ur10e/chirp/*.txt \
    --visualizer newton
```

## How it works

- `run-gui.ps1` — PowerShell equivalent of the Linux `run-gui.sh`. Reuses an
  existing container or creates a new one via `docker compose run`.
- `x11.yaml` — Docker Compose overlay that sets `DISPLAY=host.docker.internal:0.0`
  and switches from `network_mode: host` (Linux-only) to `bridge`.
- The `Dockerfile.sysid` is platform-agnostic and shared with Linux.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot connect to X server` | Make sure VcXsrv is running with "Disable access control" checked. |
| Black screen in Newton viewer | Uncheck "Native opengl" in XLaunch settings. The container needs to use its own GPU-accelerated GL, not the X server's. |
| `docker: Error response ... GPU` | Ensure Docker Desktop has GPU support enabled (Settings > Resources > GPU). Requires NVIDIA drivers on the host. |
| `network_mode host not supported` | This is expected on Windows. The `x11.yaml` overlay switches to bridge mode automatically. |
