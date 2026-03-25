# PowerShell wrapper to run commands in the isaac-lab-sysid container on Windows.
# Reuses an existing container if one exists, otherwise creates one.
# Usage:
#   .\docker\windows\run-gui.ps1                           # interactive shell
#   .\docker\windows\run-gui.ps1 python script.py --arg    # run a command
# Set $env:HEADLESS=1 to run without GUI/X11 forwarding.
#
# Prerequisites (for GUI mode):
#   1. Install VcXsrv (https://sourceforge.net/projects/vcxsrv/)
#   2. Launch XLaunch with "Disable access control" checked
#   3. Docker Desktop with WSL2 backend + NVIDIA GPU support enabled

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DockerDir = Split-Path -Parent $ScriptDir
$RepoRoot = Split-Path -Parent $DockerDir
$ContainerName = "isaac-lab-sysid"
$ServiceName = "isaac-lab-sysid"
$Profile = "sysid"
$EnvFile = ".env.base"

# Determine X11 overlay
if ($env:HEADLESS) {
    Write-Host "[INFO] Running in headless mode (no GUI)"
    $X11Overlay = @()
} else {
    Write-Host "[INFO] Using Windows X11 overlay (VcXsrv on host.docker.internal:0.0)"
    $X11Overlay = @("--file", "windows/x11.yaml")
}

# Ensure bash history file exists (bind mount requires it)
$HistoryFile = Join-Path $DockerDir ".isaac-lab-docker-history"
if (-not (Test-Path $HistoryFile)) {
    New-Item -ItemType File -Path $HistoryFile -Force | Out-Null
}

# Check if the container already exists
$Existing = docker ps -aq --filter "name=^${ContainerName}$" 2>$null

if ($Existing) {
    # Container exists - start it if stopped, then exec into it
    $Running = docker ps -q --filter "name=^${ContainerName}$" 2>$null
    if (-not $Running) {
        Write-Host "[INFO] Starting existing container '$ContainerName'..."
        docker start $ContainerName | Out-Null
    } else {
        Write-Host "[INFO] Container '$ContainerName' already running"
    }

    # Build env flags for X11 forwarding into existing container
    $X11EnvFlags = @()
    if (-not $env:HEADLESS) {
        $X11EnvFlags = @(
            "-e", "DISPLAY=host.docker.internal:0.0",
            "-e", "QT_X11_NO_MITSHM=1",
            "-e", "LIBGL_ALWAYS_INDIRECT=0"
        )
    }

    if ($args.Count -eq 0) {
        Write-Host "[INFO] Attaching to container..."
        Write-Host ""
        docker exec -it @X11EnvFlags $ContainerName bash
    } else {
        $Cmd = $args -join " "
        Write-Host "[INFO] Running command in existing container: $Cmd"
        Write-Host ""
        docker exec -it @X11EnvFlags $ContainerName bash -c $Cmd
    }
} else {
    # Container doesn't exist - create it with docker compose run
    Write-Host "[INFO] Creating new container '$ContainerName'..."
    Push-Location $DockerDir
    try {
        if ($args.Count -eq 0) {
            Write-Host "[INFO] Running container with interactive shell..."
            Write-Host ""
            docker compose `
                --file docker-compose.yaml `
                @X11Overlay `
                --env-file $EnvFile `
                --profile $Profile `
                run --name $ContainerName `
                $ServiceName
        } else {
            $Cmd = $args -join " "
            Write-Host "[INFO] Running command in container: $Cmd"
            Write-Host ""
            docker compose `
                --file docker-compose.yaml `
                @X11Overlay `
                --env-file $EnvFile `
                --profile $Profile `
                run --name $ContainerName `
                $ServiceName `
                bash -c $Cmd
        }
    } finally {
        Pop-Location
    }
}
