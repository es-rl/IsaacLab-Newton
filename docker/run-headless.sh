#!/bin/bash
# Run the isaac-lab-sysid container in headless mode (no GUI)
# Usage: ./docker/run-headless.sh [command] [args...]
# Example: ./docker/run-headless.sh python scripts/sysid/run_sysid.py --robot-name h1 --headless

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set HEADLESS environment variable and call run-gui.sh
export HEADLESS=1
exec "$SCRIPT_DIR/run-gui.sh" "$@"
