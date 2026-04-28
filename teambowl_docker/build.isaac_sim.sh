#!/bin/bash
# Build and start the Isaac Sim container for full robot simulation.
#
# Usage:
#   ./build.isaac_sim.sh            # build (cached) + start
#   ./build.isaac_sim.sh --clean    # full rebuild + wipe workspace install
#   ./build.isaac_sim.sh --help
#
# Once running:
#   WebRTC UI:  http://localhost:8211
#   Foxglove:   ws://localhost:8765
#
# Override workspace path:
#   TEAMBOWL_WS=/custom/path ./build.isaac_sim.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAMBOWL_WS="${TEAMBOWL_WS:-${HOME}/TeamBowl/teambowl_ws}"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.isaac_sim.yml"
CLEAN=0

for arg in "$@"; do
    case "$arg" in
        --clean|-c) CLEAN=1 ;;
        --help|-h)
            echo "Usage: $0 [--clean]"
            echo "  --clean  Stop containers, wipe colcon install, rebuild Docker --no-cache"
            exit 0
            ;;
    esac
done

cd "${SCRIPT_DIR}"

if [ $CLEAN -eq 1 ]; then
    echo "[build.isaac_sim] Stopping containers..."
    docker compose -f "${COMPOSE_FILE}" down --remove-orphans 2>/dev/null || true

    echo "[build.isaac_sim] Wiping colcon install for Isaac Sim..."
    rm -f "${TEAMBOWL_WS}/install/.colcon_isaac_sim_build_complete"

    echo "[build.isaac_sim] Rebuilding Docker image (--no-cache)..."
    echo "[build.isaac_sim] WARNING: First build pulls 22 GB Isaac Sim base + ~45 min nvblox compile."
    docker build --no-cache -f Dockerfile.isaac_sim -t teambowl:isaac_sim .
else
    echo "[build.isaac_sim] Building Docker image (cached)..."
    docker build -f Dockerfile.isaac_sim -t teambowl:isaac_sim .
fi

echo "[build.isaac_sim] Starting Isaac Sim container..."
echo "[build.isaac_sim] WebRTC UI will be available at: http://localhost:8211"
echo "[build.isaac_sim] Foxglove bridge at: ws://localhost:8765"
TEAMBOWL_WS="${TEAMBOWL_WS}" docker compose -f "${COMPOSE_FILE}" up
