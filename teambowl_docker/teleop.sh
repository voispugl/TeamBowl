#!/bin/bash
# Driving-leg teleop helper.
#
# Requires the teambowl_dev container to already be running (./build.sh).
# Run from the host:
#   ./teleop.sh
#
# What this script does:
#   1. Brings up CAN interfaces (can0 / can1) inside the container.
#   2. Launches the robstride motor driver in the background (inside the container).
#   3. Sets robot mode → "teleop" so the driving_leg_controller holds positions.
#   4. Opens an interactive teleop_twist_keyboard session publishing to /cmd_vel_teleop.
#
# The bringup (vel_cmd_mux, collision_guard, driving_leg_controller, etc.) is
# assumed to already be running via the container's default command.
#
# Press Ctrl+C in this terminal to stop the teleop keyboard.
# To stop the motor driver:  docker exec teambowl_dev pkill -f driver_node

set -e

CONTAINER="teambowl_dev"
WS="/workspaces/teambowl_ws"

# --------------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------------- #
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "[teleop] ERROR: Container '${CONTAINER}' is not running."
    echo "[teleop] Start it first with:  cd teambowl_docker && ./build.sh"
    exit 1
fi

# --------------------------------------------------------------------------- #
# 1. Bring up CAN interfaces
# --------------------------------------------------------------------------- #
echo "[teleop] Bringing up CAN interfaces (can0, can1) at 1 Mbps..."
docker exec "${CONTAINER}" bash -c "
    ip link set can0 up type can bitrate 1000000 2>/dev/null || true
    ip link set can1 up type can bitrate 1000000 2>/dev/null || true
    echo '[teleop] CAN interfaces ready.'
"

# --------------------------------------------------------------------------- #
# 2. Ensure teleop_twist_keyboard is installed
# --------------------------------------------------------------------------- #
echo "[teleop] Checking for teleop_twist_keyboard..."
docker exec "${CONTAINER}" bash -c "
    source /opt/ros/humble/setup.bash
    if ! ros2 pkg list 2>/dev/null | grep -q teleop_twist_keyboard; then
        echo '[teleop] Installing teleop_twist_keyboard...'
        apt-get install -y -q ros-humble-teleop-twist-keyboard
    fi
"

# --------------------------------------------------------------------------- #
# 3. Launch robstride motor driver in the background (inside the container)
# --------------------------------------------------------------------------- #
echo "[teleop] Starting robstride motor driver..."
docker exec -d "${CONTAINER}" bash -c "
    source /opt/ros/humble/setup.bash
    source ${WS}/install/setup.bash
    ros2 launch robstride_can_driver driver.launch.py \
        > /tmp/robstride_driver.log 2>&1
"

# Give the driver a moment to come up and enable motors.
echo "[teleop] Waiting 4 s for driver startup..."
sleep 4

# --------------------------------------------------------------------------- #
# 4. Set robot mode → teleop
# --------------------------------------------------------------------------- #
echo "[teleop] Setting robot mode to 'teleop'..."
docker exec "${CONTAINER}" bash -c "
    source /opt/ros/humble/setup.bash
    source ${WS}/install/setup.bash
    ros2 topic pub /robot_mode_set std_msgs/msg/String \
        '{data: \"teleop\"}' --once
"

echo ""
echo "=========================================="
echo "  TELEOP ACTIVE — keyboard controls:"
echo "    u i o       — forward diagonal / straight"
echo "    j k l       — rotate left / stop / rotate right"
echo "    m , .       — backward diagonal / straight"
echo "  Speed:  q/z raise/lower linear, e/c angular"
echo "  Publishing → /cmd_vel_teleop"
echo "  Press Ctrl+C to quit."
echo "=========================================="
echo ""

# --------------------------------------------------------------------------- #
# 5. Interactive teleop keyboard (foreground — blocks until Ctrl+C)
# --------------------------------------------------------------------------- #
docker exec -it "${CONTAINER}" bash -c "
    source /opt/ros/humble/setup.bash
    source ${WS}/install/setup.bash
    ros2 run teleop_twist_keyboard teleop_twist_keyboard \
        --ros-args --remap cmd_vel:=/cmd_vel_teleop
"

# --------------------------------------------------------------------------- #
# Cleanup: set mode back to off when user exits
# --------------------------------------------------------------------------- #
echo ""
echo "[teleop] Keyboard closed — setting robot mode to 'off'..."
docker exec "${CONTAINER}" bash -c "
    source /opt/ros/humble/setup.bash
    source ${WS}/install/setup.bash
    ros2 topic pub /robot_mode_set std_msgs/msg/String '{data: \"off\"}' --once
" 2>/dev/null || true

echo "[teleop] Done."
