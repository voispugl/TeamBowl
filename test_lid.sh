#!/bin/bash
# ==============================================================================
# Lid Testing & Tuning Utility (Host Version)
# ==============================================================================
# This script helps test and tune the RS05 cargo bay lid angles.
#
# Usage:
#   ./test_lid.sh
#
# Workflow:
#   1. Edit teambowl_ws/src/locomotion/config/lid_controller.yaml
#   2. Run this script to restart the controller and test the new angles.
# ==============================================================================

set -e

# Define paths relative to this script's location
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WS="${PROJECT_ROOT}/teambowl_ws"
LID_CONFIG="${WS}/src/locomotion/config/lid_controller.yaml"

cd "${WS}"

# --------------------------------------------------------------------------- #
# 1. Bring up CAN interfaces (lid is on can1)
# --------------------------------------------------------------------------- #
echo "[lid_test] Ensuring CAN1 is up at 1 Mbps..."
sudo modprobe mttcan 2>/dev/null || true
sudo ip link set can1 up type can bitrate 1000000 2>/dev/null || true
echo "[lid_test] CAN1 ready."

# --------------------------------------------------------------------------- #
# 2. Source ROS and Workspace
# --------------------------------------------------------------------------- #
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
else
    echo "[lid_test] ERROR: ROS Humble setup not found at /opt/ros/humble/setup.bash"
    exit 1
fi

if [ -f "install/setup.bash" ]; then
    source install/setup.bash
else
    echo "[lid_test] ERROR: Workspace not built. Run ./full_rebuild.sh first."
    exit 1
fi

# --------------------------------------------------------------------------- #
# 3. Restart Lid Controller
# --------------------------------------------------------------------------- #
# Kill existing lid_controller (either from launch.sh or previous test run)
echo "[lid_test] Restarting lid_controller node to pick up fresh parameters..."
pkill -f lid_controller || true
sleep 1

# Launch in background
ros2 run locomotion lid_controller --ros-args --params-file ${LID_CONFIG} > /tmp/lid_controller_test.log 2>&1 &
LID_PID=$!

echo "[lid_test] Lid controller started (PID: ${LID_PID})"
echo "[lid_test] Config: teambowl_ws/src/locomotion/config/lid_controller.yaml"
echo "[lid_test] Log:    tail -f /tmp/lid_controller_test.log"
sleep 2

# --------------------------------------------------------------------------- #
# 4. Interactive Menu
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
echo "    [m] Monitor Position/Effort (Debug)"
echo "    [f] Check Motor Faults/Mode (Debug)"
echo "    [e] Re-send Enable Motors Service"
echo "    [q] Quit (kills test node)"
echo "=========================================="
echo ""

# Handle cleanup on exit
cleanup() {
    echo -e "\n[lid_test] Stopping test node..."
    kill ${LID_PID} 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

while true; do
    read -p "Enter command: " cmd
    case $cmd in
        [oO]*)
            echo "[lid_test] Sending OPEN command..."
            ros2 topic pub /lid_command std_msgs/msg/String '{data: "open"}' --once > /dev/null
            ;;
        [cC]*)
            echo "[lid_test] Sending CLOSE command..."
            ros2 topic pub /lid_command std_msgs/msg/String '{data: "close"}' --once > /dev/null
            ;;
        [tT]*)
            echo "[lid_test] Sending TOGGLE command..."
            ros2 topic pub /lid_command std_msgs/msg/String '{data: "toggle"}' --once > /dev/null
            ;;
        [sS]*)
            echo "[lid_test] Current Lid State:"
            ros2 topic echo /lid_state --once
            ;;
        [mM]*)
            echo "[lid_test] Monitoring joint_rs05_1 (Ctrl+C to stop)..."
            # Use a python script to filter joint states for just the lid
            ros2 topic echo /joint_states --once | grep -A 20 "joint_rs05_1" || echo "No joint_rs05_1 found in /joint_states"
            echo "--- (Showing one snapshot. Use 'ros2 topic echo /joint_states' for live stream) ---"
            ;;
        [fF]*)
            echo "[lid_test] Checking /motor_faults for joint_rs05_1..."
            ros2 topic echo /motor_faults --once | grep -A 15 "joint_rs05_1" || echo "No fault data for joint_rs05_1"
            ;;
        [eE]*)
            echo "[lid_test] Calling /enable_motors service..."
            ros2 service call /enable_motors std_srvs/srv/Trigger {}
            ;;
        [qQ]*)
            cleanup
            ;;
        *)
            echo "[lid_test] Unknown command. Use o, c, t, s, m, f, e, or q."
            ;;
    esac
done
