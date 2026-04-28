#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAMBOWL_WS="${TEAMBOWL_WS:-${HOME}/TeamBowl/teambowl_ws}"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.isaac_sim.yml"

cd "${SCRIPT_DIR}"

echo ""
echo "=============================="
echo "   LAUNCH ISAAC SIM"
echo "=============================="
echo "[launch] WebRTC UI:  http://localhost:8211"
echo "[launch] Foxglove:   ws://localhost:8765"
echo "[launch] Ctrl+C to stop."
echo ""
TEAMBOWL_WS="${TEAMBOWL_WS}" docker compose -f "${COMPOSE_FILE}" up
