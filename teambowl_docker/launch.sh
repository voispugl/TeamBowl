#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${SCRIPT_DIR}"

echo ""
echo "=============================="
echo "      LAUNCH"
echo "=============================="
echo "[launch] Starting container (Ctrl+C to stop)..."
echo "[launch] Workspace colcon build runs on first start (~15-20 min)."
echo ""
docker compose up
