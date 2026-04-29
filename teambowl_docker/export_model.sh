#!/bin/bash
# Export the YOLO26m TRT FP16 engine for this Jetson (device-specific, ~15-20 min).
# Run once after a clean build or whenever the model variant changes.
# Output: ~/TeamBowl/models/yolo26m.engine  (volume-mounted, persists across rebuilds)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo ""
echo "=============================="
echo "      EXPORT MODEL"
echo "=============================="
echo "[export] Running yolo26m TRT FP16 export inside container (~15-20 min)..."
echo ""
docker compose run --rm teambowl python3 /workspaces/teambowl_ws/src/perception/scripts/export_yolo26.py --out /home/box/TeamBowl/models/yolo26m.engine
