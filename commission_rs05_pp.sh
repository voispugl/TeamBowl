#!/usr/bin/env bash
# Flash RS05 (joint_rs05_1) to PP position mode permanently.
# Run once with robstride_can_driver running.
# After this, the motor always boots in PP mode — no motor param writes at startup needed.
set -eo pipefail

JOINT="joint_rs05_1"
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WS="${PROJECT_ROOT}/teambowl_ws"

if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
else
    echo "ERROR: ROS Humble not found"; exit 1
fi

if [ -f "${WS}/install/setup.bash" ]; then
    source "${WS}/install/setup.bash"
else
    echo "ERROR: Workspace not built — run full_rebuild.sh first"; exit 1
fi

write_param() {
    local index="$1" value="$2" type="${3:-float}"
    echo "  write_param index=${index} value=${value} type=${type}"
    ros2 service call /write_motor_param \
        robstride_can_interfaces/srv/WriteMotorParam \
        "{joint_name: '${JOINT}', param_index: ${index}, value: ${value}, value_type: '${type}'}" \
        2>/dev/null | grep -E "success|message" || true
}

echo "[rs05] Checking for robstride_can_driver..."
if ! ros2 service list 2>/dev/null | grep -q "/write_motor_param"; then
    echo "ERROR: /write_motor_param not found. Is robstride_can_driver running?"
    exit 1
fi
echo "[rs05] ✓ robstride_can_driver is running"

echo "[rs05] Enabling motors..."
ros2 service call /enable_motors std_srvs/srv/Trigger {} 2>/dev/null | grep -E "success|message" || true
sleep 1

echo "[rs05] Setting run_mode=1 (PP position mode)..."
write_param 28677 1 uint8          # 0x7005 run_mode

echo "[rs05] Writing PP gains..."
write_param 28702 40.0 float       # 0x701E loc_kp
write_param 28703  6.0 float       # 0x701F spd_kp
write_param 28704  0.02 float      # 0x7020 spd_ki
write_param 28695  2.0 float       # 0x7017 limit_spd
write_param 28696  2.0 float       # 0x7018 limit_cur

echo "[rs05] Saving all motor params to flash..."
ros2 service call /save_motor_params std_srvs/srv/Trigger {} 2>/dev/null | grep -E "success|message" || true

echo ""
echo "[rs05] Done. RS05 will boot in PP position mode on every power cycle."
echo "       Run:  bash ~/TeamBowl/test_lid.sh"
