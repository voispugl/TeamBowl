#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="${SCRIPT_DIR}/../teambowl_ws"

CLEAN=false

usage() {
    echo ""
    echo "Usage: ./build.sh [OPTIONS]"
    echo ""
    echo "  (no flags)       Rebuild Docker image (cached)"
    echo "  -c, --clean      Full clean rebuild:"
    echo "                     - Stop containers"
    echo "                     - Wipe colcon build/install/log"
    echo "                     - Rebuild Docker image with --no-cache"
    echo "  -h, --help       Show this help"
    echo ""
    echo "To start the container after building, run: ./launch.sh"
    echo ""
    echo "Examples:"
    echo "  ./build.sh             # fast rebuild using Docker layer cache"
    echo "  ./build.sh --clean     # full clean rebuild from scratch (~20 min)"
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
    echo "      CLEAN BUILD"
    echo "=============================="

    echo "[build] Stopping containers..."
    docker compose down || true

    echo "[build] Removing colcon build artifacts (build/ install/ log/)..."
    rm -rf "${WS_DIR}/build" "${WS_DIR}/install" "${WS_DIR}/log"
    echo "[build] Colcon artifacts removed."

    echo "[build] Building Docker image (no cache — this will take a while)..."
    docker compose build --no-cache
else
    echo ""
    echo "=============================="
    echo "      NORMAL BUILD"
    echo "=============================="

    echo "[build] Building Docker image (cached)..."
    docker compose build
fi

echo ""
echo "[build] Image ready. Run ./launch.sh to start the container."
echo ""
