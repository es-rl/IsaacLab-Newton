#!/bin/bash
# Wrapper script to run commands in the isaac-lab-sysid container
# Reuses an existing container if one exists, otherwise creates one.
# Usage: ./docker/run-gui.sh [command] [args...]
# Example: ./docker/run-gui.sh python scripts/sysid/run_sysid.py --robot-name h1 --headless
# No args: opens an interactive bash shell
# Set HEADLESS=1 to run without GUI/X11 forwarding

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOFTWARE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
CONTAINER_NAME="isaac-lab-sysid"
SERVICE_NAME="isaac-lab-sysid"
PROFILE="sysid"
ENV_FILE=".env.base"

# Setup X11 forwarding if not headless
TMP_DIR=""
if [ -n "$HEADLESS" ]; then
    echo "[INFO] Running in headless mode (no GUI)"
    X11_OVERLAY=""
else
    echo "[INFO] Setting up X11 forwarding..."
    TMP_DIR=$(mktemp -d)
    TMP_XAUTH=$(mktemp --suffix=.xauth --tmpdir="$TMP_DIR")

    XAUTH_COOKIE=$(xauth nlist "$DISPLAY" 2>/dev/null || true)
    if [ -n "$XAUTH_COOKIE" ]; then
        echo "$XAUTH_COOKIE" | sed 's/ffff//' | xauth -f "$TMP_XAUTH" nmerge -
        echo "[INFO] X11 forwarding enabled"
    else
        echo "[WARN] No X11 display found, running in headless mode"
    fi

    export __ISAACLAB_TMP_XAUTH="$TMP_XAUTH"
    export __ISAACLAB_TMP_DIR="$TMP_DIR"
    X11_OVERLAY="--file x11.yaml"
fi

# Ensure bash history file exists (bind mount requires it)
touch "$SCRIPT_DIR/.isaac-lab-docker-history"

# Check if the container already exists
EXISTING=$(docker ps -aq --filter "name=^${CONTAINER_NAME}$" 2>/dev/null)

if [ -n "$EXISTING" ]; then
    # Container exists — start it if stopped, then exec into it
    RUNNING=$(docker ps -q --filter "name=^${CONTAINER_NAME}$" 2>/dev/null)
    if [ -z "$RUNNING" ]; then
        echo "[INFO] Starting existing container '$CONTAINER_NAME'..."
        docker start "$CONTAINER_NAME" >/dev/null
    else
        echo "[INFO] Container '$CONTAINER_NAME' already running"
    fi

    # Build env flags for X11 forwarding into existing container
    X11_ENV_FLAGS=""
    if [ -z "$HEADLESS" ] && [ -n "$DISPLAY" ]; then
        X11_ENV_FLAGS="-e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1"
        if [ -n "$TMP_XAUTH" ] && [ -f "$TMP_XAUTH" ]; then
            X11_ENV_FLAGS="$X11_ENV_FLAGS -e XAUTHORITY=$TMP_XAUTH"
        fi
    fi

    if [ $# -eq 0 ]; then
        echo "[INFO] Attaching to container..."
        echo ""
        docker exec -it $X11_ENV_FLAGS "$CONTAINER_NAME" bash
    else
        echo "[INFO] Running command in existing container: $*"
        echo ""
        docker exec -it $X11_ENV_FLAGS "$CONTAINER_NAME" bash -c "$*"
    fi
else
    # Container doesn't exist — create it with docker compose run
    echo "[INFO] Creating new container '$CONTAINER_NAME'..."
    cd "$SCRIPT_DIR"
    if [ $# -eq 0 ]; then
        echo "[INFO] Running container with interactive shell..."
        echo ""
        docker compose \
            --file docker-compose.yaml \
            $X11_OVERLAY \
            --env-file "$ENV_FILE" \
            --profile "$PROFILE" \
            run --name "$CONTAINER_NAME" \
            "$SERVICE_NAME"
    else
        echo "[INFO] Running command in container: $*"
        echo ""
        docker compose \
            --file docker-compose.yaml \
            $X11_OVERLAY \
            --env-file "$ENV_FILE" \
            --profile "$PROFILE" \
            run --name "$CONTAINER_NAME" \
            "$SERVICE_NAME" \
            bash -c "$*"
    fi
fi

EXIT_CODE=$?

# Cleanup X11 temp files
if [ -n "$TMP_DIR" ]; then
    echo ""
    echo "[INFO] Cleaning up X11 temp files..."
    rm -rf "$TMP_DIR"
fi

exit $EXIT_CODE
