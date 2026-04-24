#!/bin/bash
# Build the TeamBowl simulation Docker image.
#
# Builds teambowl:laptop first (base layer), then teambowl:sim (adds mujoco pip).
# The laptop image build compiles the workspace inside the container on first run.
#
# Usage:
#   ./build.sim.sh             # build images (cached) then start sim
#   ./build.sim.sh --clean     # full clean rebuild from scratch (~15 min)
#   ./build.sim.sh --help
#
# Set TEAMBOWL_ROOT to override the repo root (default: ~/TeamBowl):
#   TEAMBOWL_ROOT=/my/path ./build.sim.sh
#
# Set TEAMBOWL_WS to override workspace source:
#   TEAMBOWL_WS=/my/path/teambowl_ws ./build.sim.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/.." 2>/dev/null && pwd || echo "${HOME}/TeamBowl")"
DEFAULT_WS="${DEFAULT_ROOT}/teambowl_ws"

export TEAMBOWL_ROOT="${TEAMBOWL_ROOT:-${DEFAULT_ROOT}}"
export TEAMBOWL_WS="${TEAMBOWL_WS:-${DEFAULT_WS}}"

LAPTOP_COMPOSE="${SCRIPT_DIR}/docker-compose.laptop.yml"
SIM_COMPOSE="${SCRIPT_DIR}/docker-compose.sim.yml"
CLEAN=false

usage() {
    echo ""
    echo "Usage: ./build.sim.sh [OPTIONS]"
    echo ""
    echo "  (no flags)       Build Docker images (cached) then start sim"
    echo "  -c, --clean      Full clean rebuild:"
    echo "                     - Stop containers"
    echo "                     - Wipe colcon build/install/log"
    echo "                     - Rebuild Docker images with --no-cache"
    echo "  -h, --help       Show this help"
    echo ""
    echo "Repo root:         ${TEAMBOWL_ROOT}"
    echo "Workspace source:  ${TEAMBOWL_WS}"
    echo "Override root:     TEAMBOWL_ROOT=/your/path ./build.sim.sh"
    echo "Override ws:       TEAMBOWL_WS=/your/path ./build.sim.sh"
    echo ""
    echo "Examples:"
    echo "  ./build.sim.sh              # fast rebuild + start sim"
    echo "  ./build.sim.sh --clean      # full clean rebuild from scratch"
    echo ""
}

for arg in "$@"; do
    case $arg in
        -c|--clean)
            CLEAN=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            usage
            exit 1
            ;;
    esac
done

cd "${SCRIPT_DIR}"

if [ "${CLEAN}" = true ]; then
    echo ""
    echo "=============================="
    echo "  CLEAN BUILD (sim)"
    echo "=============================="

    echo "[build] Stopping containers..."
    docker compose -f "${SIM_COMPOSE}" down || true
    docker compose -f "${LAPTOP_COMPOSE}" down || true

    echo "[build] Removing colcon build artifacts (build/ install/ log/)..."
    rm -rf "${TEAMBOWL_WS}/build" "${TEAMBOWL_WS}/install" "${TEAMBOWL_WS}/log"
    echo "[build] Colcon artifacts removed."

    echo "[build] Building teambowl:laptop (no cache, base layer)..."
    docker compose -f "${LAPTOP_COMPOSE}" build --no-cache

    echo "[build] Building teambowl:sim (no cache)..."
    docker compose -f "${SIM_COMPOSE}" build --no-cache
else
    echo ""
    echo "=============================="
    echo "  NORMAL BUILD (sim)"
    echo "=============================="
    echo "[build] Repo root:  ${TEAMBOWL_ROOT}"
    echo "[build] Workspace:  ${TEAMBOWL_WS}"

    echo "[build] Building teambowl:laptop (cached, base layer)..."
    docker compose -f "${LAPTOP_COMPOSE}" build

    echo "[build] Building teambowl:sim (cached)..."
    docker compose -f "${SIM_COMPOSE}" build
fi

echo ""
echo "[build] Starting sim container..."
echo "[build] Colcon will build the workspace on first run (~10 min, depthai skipped)."
echo "[build] Sim launches automatically: ros2 launch bringup sim.launch.py"
echo "[build] Foxglove bridge: ws://\$(hostname -I | awk '{print \$1}'):8765"
echo "[build]"
echo "[build] To tune driving mode instead:"
echo "[build]   docker compose -f docker-compose.sim.yml run --rm teambowl-sim \\"
echo "[build]     ros2 launch bringup sim.launch.py velocity_controller:=driving"
echo "[build]"
echo "[build] To reset sim: ros2 service call /sim_reset std_srvs/srv/Trigger {}"
echo "[build] Type Ctrl+C to stop."
echo ""
docker compose -f "${SIM_COMPOSE}" up
