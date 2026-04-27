#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export TEAMBOWL_ROOT="${TEAMBOWL_ROOT:-${DEFAULT_ROOT}}"
export TEAMBOWL_WS="${TEAMBOWL_WS:-${TEAMBOWL_ROOT}/teambowl_ws}"

cd "${SCRIPT_DIR}"

echo ""
echo "=============================="
echo "    LAUNCH SIM"
echo "=============================="
echo "[launch] Foxglove: ws://$(hostname -I | awk '{print $1}'):8765"
echo "[launch] Reset sim: ros2 service call /sim_reset std_srvs/srv/Trigger {}"
echo "[launch] Ctrl+C to stop."
echo ""
docker compose -f docker-compose.sim.yml up
