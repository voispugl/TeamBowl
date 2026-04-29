#!/bin/bash
# Start the container with a bash shell instead of the ROS stack.
# Useful for running one-off commands (trtexec, pip installs, diagnostics, etc.)
# before launching the full stack with ./launch.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo ""
echo "=============================="
echo "      SHELL"
echo "=============================="
echo "[shell] Starting container with bash (ROS stack NOT launched)..."
echo ""
docker compose run --rm teambowl bash
