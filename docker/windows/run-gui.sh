#!/usr/bin/env bash
# Run commands in the isaac-lab-sysid container from WSL 2.
# Reuses an existing container if one exists, otherwise creates one.
#
# Usage:
#   bash docker/windows/run-gui.sh                           # interactive shell
#   bash docker/windows/run-gui.sh python script.py --arg   # run a command
#   HEADLESS=1 bash docker/windows/run-gui.sh               # no GUI/X11
#
# Prerequisites (for GUI mode):
#   1. Install VcXsrv on Windows (https://sourceforge.net/projects/vcxsrv/)
#   2. Launch XLaunch with "Disable access control" checked
#   3. Set DISPLAY to the Windows host IP before running this script:
#        export DISPLAY=$(ip route show default | awk '{print $3}'):0.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(dirname "$SCRIPT_DIR")"
CONTAINER_NAME="isaac-lab-sysid"
SERVICE_NAME="isaac-lab-sysid"
PROFILE="sysid"
ENV_FILE=".env.base"

# ---------------------------------------------------------------------------
# X11 / display setup
# ---------------------------------------------------------------------------
X11_OVERLAY=()
X11_ENV_FLAGS=()

if [[ -n "${HEADLESS:-}" ]]; then
    echo "[INFO] Running in headless mode (no GUI)"
else
    # Auto-detect Windows host IP via the WSL default route if DISPLAY not set
    if [[ -z "${DISPLAY:-}" ]]; then
        HOST_IP=$(ip route show default 2>/dev/null | awk '{print $3; exit}')
        if [[ -z "$HOST_IP" ]]; then
            echo "[ERROR] Could not detect Windows host IP. Set DISPLAY manually:" >&2
            echo "  export DISPLAY=<windows-host-ip>:0.0" >&2
            exit 1
        fi
        export DISPLAY="${HOST_IP}:0.0"
        echo "[INFO] Auto-detected DISPLAY=${DISPLAY}"
    else
        echo "[INFO] Using DISPLAY=${DISPLAY}"
    fi
    X11_OVERLAY=("--file" "windows/x11.yaml")
    X11_ENV_FLAGS=(
        "-e" "DISPLAY=${DISPLAY}"
        "-e" "QT_X11_NO_MITSHM=1"
        "-e" "LIBGL_ALWAYS_INDIRECT=0"
    )
fi

# ---------------------------------------------------------------------------
# Ensure bash history bind-mount file exists
# ---------------------------------------------------------------------------
HISTORY_FILE="${DOCKER_DIR}/.isaac-lab-docker-history"
touch "$HISTORY_FILE"

# ---------------------------------------------------------------------------
# Run or attach
# ---------------------------------------------------------------------------
EXISTING=$(docker ps -aq --filter "name=^${CONTAINER_NAME}$" 2>/dev/null || true)

if [[ -n "$EXISTING" ]]; then
    RUNNING=$(docker ps -q --filter "name=^${CONTAINER_NAME}$" 2>/dev/null || true)
    if [[ -z "$RUNNING" ]]; then
        echo "[INFO] Starting existing container '${CONTAINER_NAME}'..."
        docker start "$CONTAINER_NAME" > /dev/null
    else
        echo "[INFO] Container '${CONTAINER_NAME}' already running"
    fi

    if [[ $# -eq 0 ]]; then
        echo "[INFO] Attaching to container..."
        echo ""
        docker exec -it "${X11_ENV_FLAGS[@]}" "$CONTAINER_NAME" bash
    else
        echo "[INFO] Running command in existing container: $*"
        echo ""
        docker exec -it "${X11_ENV_FLAGS[@]}" "$CONTAINER_NAME" bash -c "$*"
    fi
else
    echo "[INFO] Creating new container '${CONTAINER_NAME}'..."
    cd "$DOCKER_DIR"

    if [[ $# -eq 0 ]]; then
        echo "[INFO] Running container with interactive shell..."
        echo ""
        docker compose \
            --file docker-compose.yaml \
            "${X11_OVERLAY[@]}" \
            --env-file "$ENV_FILE" \
            --profile "$PROFILE" \
            run --name "$CONTAINER_NAME" \
            "$SERVICE_NAME"
    else
        echo "[INFO] Running command in container: $*"
        echo ""
        docker compose \
            --file docker-compose.yaml \
            "${X11_OVERLAY[@]}" \
            --env-file "$ENV_FILE" \
            --profile "$PROFILE" \
            run --name "$CONTAINER_NAME" \
            "$SERVICE_NAME" \
            bash -c "$*"
    fi
fi
