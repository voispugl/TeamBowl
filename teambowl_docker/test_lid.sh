#!/bin/bash
# ==============================================================================
# Lid Testing & Tuning Utility
# ==============================================================================
# This script helps test and tune the RS05 cargo bay lid angles.
# It requires the teambowl_dev container to be running.
#
# Usage:
#   ./test_lid.sh
#
# Workflow:
#   1. Edit teambowl_ws/src/locomotion/config/lid_controller.yaml (host or container)
#   2. Run this script to restart the controller and test the new angles.
# ==============================================================================

set -e

CONTAINER="teambowl_dev"
WS="/workspaces/teambowl_ws"
LID_CONFIG="${WS}/src/locomotion/config/lid_controller.yaml"

# --------------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------------- #
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "[lid_test] ERROR: Container '${CONTAINER}' is not running."
    echo "[lid_test] Start it first with:  cd teambowl_docker && ./build.sh"
    exit 1
fi

# --------------------------------------------------------------------------- #
# 1. Bring up CAN interfaces (lid is on can1)
# --------------------------------------------------------------------------- #
echo "[lid_test] Ensuring CAN interfaces are up..."
docker exec "${CONTAINER}" bash -c "
    ip link set can1 up type can bitrate 1000000 2>/dev/null || true
    echo '[lid_test] CAN interfaces ready.'
"

# --------------------------------------------------------------------------- #
# 2. Restart Lid Controller
# --------------------------------------------------------------------------- #
# If bringup.launch.py is running, we need to kill the existing lid_controller
# so we can launch a fresh one with updated parameters for tuning.
echo "[lid_test] Restarting lid_controller node to pick up fresh parameters..."
docker exec "${CONTAINER}" bash -c "pkill -f lid_controller || true"
sleep 1

# Launch in background inside container
docker exec -d "${CONTAINER}" bash -c "
    source /opt/ros/humble/setup.bash
    source ${WS}/install/setup.bash
    ros2 run locomotion lid_controller --ros-args --params-file ${LID_CONFIG} \
        > /tmp/lid_controller_test.log 2>&1
"

echo "[lid_test] Lid controller started with config: teambowl_ws/src/locomotion/config/lid_controller.yaml"
echo "[lid_test] Log available at: /tmp/lid_controller_test.log inside container"
sleep 2

# --------------------------------------------------------------------------- #
# 3. Interactive Menu
# --------------------------------------------------------------------------- #
echo ""
echo "=========================================="
echo "  LID CONTROL ACTIVE"
echo "=========================================="
echo "  Commands:"
echo "    [o] Open Lid"
echo "    [c] Close Lid"
echo "    [t] Toggle Lid"
echo "    [s] Show /lid_state"
echo "    [q] Quit (kills test node)"
echo "=========================================="
echo ""

while true; do
    read -p "Enter command: " cmd
    case $cmd in
        [oO]*)
            echo "[lid_test] Sending OPEN command..."
            docker exec "${CONTAINER}" bash -c "source /opt/ros/humble/setup.bash && ros2 topic pub /lid_command std_msgs/msg/String '{data: \"open\"}' --once" > /dev/null
            ;;
        [cC]*)
            echo "[lid_test] Sending CLOSE command..."
            docker exec "${CONTAINER}" bash -c "source /opt/ros/humble/setup.bash && ros2 topic pub /lid_command std_msgs/msg/String '{data: \"close\"}' --once" > /dev/null
            ;;
        [tT]*)
            echo "[lid_test] Sending TOGGLE command..."
            docker exec "${CONTAINER}" bash -c "source /opt/ros/humble/setup.bash && ros2 topic pub /lid_command std_msgs/msg/String '{data: \"toggle\"}' --once" > /dev/null
            ;;
        [sS]*)
            echo "[lid_test] Current Lid State:"
            docker exec "${CONTAINER}" bash -c "source /opt/ros/humble/setup.bash && ros2 topic echo /lid_state --once"
            ;;
        [qQ]*)
            echo "[lid_test] Stopping test node..."
            docker exec "${CONTAINER}" bash -c "pkill -f lid_controller || true"
            echo "[lid_test] Done."
            exit 0
            ;;
        *)
            echo "[lid_test] Unknown command. Use o, c, t, s, or q."
            ;;
    esac
done
