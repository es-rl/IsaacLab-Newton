#!/bin/bash
# Build Docker images for the IsaacLab-Newton toolbox.
# Usage: ./docker/build-docker.sh [TARGET] [--no-cache]
#
# TARGET can be: isaacsim, base, sysid, ros2, or all (default: all)
# Each target automatically builds its dependencies first.

set -e

BUILD_ARGS=""
TARGET="all"
for arg in "$@"; do
    case "$arg" in
        --no-cache) BUILD_ARGS="$BUILD_ARGS --no-cache" ;;
        isaacsim|base|sysid|ros2|all) TARGET="$arg" ;;
        *) echo "Unknown option: $arg"; echo "Usage: $0 [isaacsim|base|sysid|ros2|all] [--no-cache]"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "$TARGET" in
    isaacsim)
        echo "[INFO] Building Isaac Sim 6.0.0 from source..."
        docker compose --env-file .env.base --profile isaacsim build $BUILD_ARGS
        ;;
    base)
        echo "[INFO] Ensuring Isaac Sim image is up to date..."
        docker compose --env-file .env.base --profile isaacsim build
        echo "[INFO] Building Isaac Lab base image..."
        docker compose --env-file .env.base --profile base build $BUILD_ARGS
        ;;
    sysid)
        echo "[INFO] Ensuring Isaac Sim + base images are up to date..."
        docker compose --env-file .env.base --profile isaacsim build
        docker compose --env-file .env.base --profile base build
        echo "[INFO] Building sysid image (newton, cmaes, optuna, sage)..."
        docker compose --env-file .env.base --profile sysid build $BUILD_ARGS
        ;;
    ros2)
        echo "[INFO] Ensuring Isaac Sim + base images are up to date..."
        docker compose --env-file .env.base --profile isaacsim build
        docker compose --env-file .env.base --profile base build
        echo "[INFO] Building ROS2 image..."
        docker compose --env-file .env.base --profile ros2 build $BUILD_ARGS
        ;;
    all)
        echo "[INFO] Building Isaac Sim 6.0.0 from source..."
        docker compose --env-file .env.base --profile isaacsim build $BUILD_ARGS
        echo "[INFO] Building Isaac Lab base image..."
        docker compose --env-file .env.base --profile base build $BUILD_ARGS
        echo "[INFO] Building sysid image (newton, cmaes, optuna, sage)..."
        docker compose --env-file .env.base --profile sysid build $BUILD_ARGS
        ;;
esac

echo "[INFO] Done."
