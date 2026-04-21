#!/bin/bash
# Build the TeamBowl laptop Docker image (x86_64, no GPU, skips depthai packages).
#
# Usage:
#   ./build.laptop.sh             # build image (cached) then drop to bash
#   ./build.laptop.sh --clean     # full clean rebuild from scratch (~10 min)
#   ./build.laptop.sh --help
#
# Set TEAMBOWL_WS to override the workspace source path:
#   TEAMBOWL_WS=/my/path/teambowl_ws ./build.laptop.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default workspace location — resolve relative to teambowl_docker/../teambowl_ws
DEFAULT_WS="$(cd "${SCRIPT_DIR}/../teambowl_ws" 2>/dev/null && pwd || echo "${HOME}/TeamBowl/teambowl_ws")"
export TEAMBOWL_WS="${TEAMBOWL_WS:-${DEFAULT_WS}}"

COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.laptop.yml"
CLEAN=false

usage() {
    echo ""
    echo "Usage: ./build.laptop.sh [OPTIONS]"
    echo ""
    echo "  (no flags)       Build Docker image (cached) then start interactive shell"
    echo "  -c, --clean      Full clean rebuild:"
    echo "                     - Stop containers"
    echo "                     - Wipe colcon build/install/log"
    echo "                     - Rebuild Docker image with --no-cache"
    echo "  -h, --help       Show this help"
    echo ""
    echo "Workspace source: ${TEAMBOWL_WS}"
    echo "Override:         TEAMBOWL_WS=/your/path ./build.laptop.sh"
    echo ""
    echo "Examples:"
    echo "  ./build.laptop.sh             # fast rebuild"
    echo "  ./build.laptop.sh --clean     # full clean rebuild from scratch"
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
    echo "  CLEAN BUILD (laptop)"
    echo "=============================="

    echo "[build] Stopping containers..."
    docker compose -f "${COMPOSE_FILE}" down || true

    echo "[build] Removing colcon build artifacts (build/ install/ log/)..."
    rm -rf "${TEAMBOWL_WS}/build" "${TEAMBOWL_WS}/install" "${TEAMBOWL_WS}/log"
    echo "[build] Colcon artifacts removed."

    echo "[build] Building Docker image (no cache)..."
    docker compose -f "${COMPOSE_FILE}" build --no-cache
else
    echo ""
    echo "=============================="
    echo "  NORMAL BUILD (laptop)"
    echo "=============================="
    echo "[build] Workspace: ${TEAMBOWL_WS}"

    echo "[build] Building Docker image (cached)..."
    docker compose -f "${COMPOSE_FILE}" build
fi

echo ""
echo "[build] Starting laptop container (interactive shell)..."
echo "[build] Colcon will build the workspace on first run (~10 min, depthai skipped)."
echo "[build] Type 'exit' to stop."
echo ""
docker compose -f "${COMPOSE_FILE}" run --rm teambowl-laptop bash
