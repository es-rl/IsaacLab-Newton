#!/bin/bash
# Symlink SAGE configs from bind-mounted scripts so new configs are picked up
# without rebuilding the image
SAGE_CONFIGS_SRC="${ISAACLAB_PATH}/scripts/sim2real_gap/configs"
SAGE_CONFIGS_DST="${ISAACSIM_ROOT_PATH}/kit/python/lib/python3.11/site-packages/configs"
if [ -d "$SAGE_CONFIGS_SRC" ]; then
    rm -rf "$SAGE_CONFIGS_DST"
    ln -sf "$SAGE_CONFIGS_SRC" "$SAGE_CONFIGS_DST"
fi

# If arguments are provided, execute them; otherwise drop into bash shell
if [ $# -eq 0 ]; then
    exec bash
else
    exec "$@"
fi
