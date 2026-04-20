#!/usr/bin/env bash
# ==============================================================================
# Lid Tuning Utility — RS05 cargo bay lid (PP position mode)
# ==============================================================================
#
# PREREQUISITES — must be running in a separate terminal first:
#   source /opt/ros/humble/setup.bash
#   source ~/TeamBowl/teambowl_ws/install/setup.bash
#   ros2 launch robstride_can_driver driver.launch.py
#
# Usage:
#   bash ~/TeamBowl/test_lid.sh
#
# Workflow to calibrate from scratch:
#   1. Physically move lid to fully closed position.
#   2. Press [z] to set mechanical zero.
#   3. Press [o] to open — observe position readout.
#   4. Press [sopen] to save open_position_rad to YAML.
#   5. Press [sclosed] to save closed_position_rad (should be ~0.0).
#   6. Press [r] to restart the lid_controller with new YAML.
#
# Gain tuning:
#   [kp/vp/vi] write to motor live — no restart needed.
#   Save final values back to lid_controller.yaml manually, or press [r]
#   after editing the YAML to reload.
# ==============================================================================

set -eo pipefail

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WS="${PROJECT_ROOT}/teambowl_ws"
LID_CONFIG="${WS}/src/locomotion/config/lid_controller.yaml"
JOINT="joint_rs05_1"

# Param indices (decimal) for ros2 service calls
PARAM_RUN_MODE=28677    # 0x7005 uint8
PARAM_LOC_REF=28694     # 0x7016 float  — position target
PARAM_LIMIT_SPD=28695   # 0x7017 float
PARAM_LIMIT_CUR=28696   # 0x7018 float
PARAM_LOC_KP=28702      # 0x701E float  — position P gain
PARAM_SPD_KP=28703      # 0x701F float  — velocity P gain
PARAM_SPD_KI=28704      # 0x7020 float  — velocity I gain

cd "${WS}"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

write_param() {
    local index="$1" value="$2" type="${3:-float}"
    ros2 service call /write_motor_param \
        robstride_can_interfaces/srv/WriteMotorParam \
        "{joint_name: '${JOINT}', param_index: ${index}, value: ${value}, value_type: '${type}'}" \
        2>/dev/null | grep -E "success|message" || true
}

read_param() {
    local index="$1"
    ros2 service call /read_motor_param \
        robstride_can_interfaces/srv/ReadMotorParam \
        "{joint_name: '${JOINT}', param_index: ${index}}" \
        2>/dev/null | grep "value\|message" || true
}

get_position() {
    ros2 topic echo /joint_states --once 2>/dev/null \
        | python3 -c "
import sys, re
data = sys.stdin.read()
name_section = data.split('position:')[0] if 'position:' in data else ''
names = re.findall(r'^-\s+(\S+)', name_section, re.MULTILINE)
vals  = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?', data.split('position:')[1].split('velocity:')[0]) \
        if 'position:' in data else []
try:
    idx = names.index('${JOINT}')
    print(f'{float(vals[idx]):.4f} rad')
except:
    print('NOT FOUND in joint_states')
" 2>/dev/null || echo "(could not read /joint_states)"
}

yaml_get() {
    python3 -c "
import yaml
with open('${LID_CONFIG}') as f:
    cfg = yaml.safe_load(f)
p = cfg['lid_controller']['ros__parameters']
print(p.get('$1', 'NOT SET'))
" 2>/dev/null
}

yaml_set() {
    python3 - "$1" "$2" <<'PYEOF'
import sys, re
key, val = sys.argv[1], sys.argv[2]
with open('${LID_CONFIG}') as f:
    text = f.read()
# Replace the value on the line matching the key
pattern = rf'^(\s*{re.escape(key)}\s*:\s*)[\S]+'
replacement = rf'\g<1>{val}'
new = re.sub(pattern, replacement, text, flags=re.MULTILINE)
if new == text:
    print(f"  WARNING: key '{key}' not found in YAML — not updated")
else:
    with open('${LID_CONFIG}', 'w') as f:
        f.write(new)
    print(f"  Saved {key}: {val}")
PYEOF
}

restart_lid() {
    echo "[lid] Restarting lid_controller..."
    pkill -f lid_controller 2>/dev/null || true
    sleep 1
    ros2 run locomotion lid_controller \
        --ros-args --params-file "${LID_CONFIG}" \
        > /tmp/lid_controller_test.log 2>&1 &
    LID_PID=$!
    echo "[lid] Started PID=${LID_PID}  log: tail -f /tmp/lid_controller_test.log"
    sleep 2
}

# --------------------------------------------------------------------------- #
# Source ROS
# --------------------------------------------------------------------------- #
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
else
    echo "ERROR: ROS Humble not found"; exit 1
fi

if [ -f install/setup.bash ]; then
    source install/setup.bash
else
    echo "ERROR: Workspace not built — run full_rebuild.sh first"; exit 1
fi

# --------------------------------------------------------------------------- #
# CAN
# --------------------------------------------------------------------------- #
echo "[lid] Ensuring CAN1 is up..."
sudo modprobe mttcan 2>/dev/null || true
sudo ip link set can1 up type can bitrate 1000000 2>/dev/null || true

# --------------------------------------------------------------------------- #
# Pre-flight check
# --------------------------------------------------------------------------- #
echo "[lid] Checking for robstride_can_driver..."
if ! ros2 service list 2>/dev/null | grep -q "/write_motor_param"; then
    echo ""
    echo "  WARNING: /write_motor_param not found."
    echo "  Run in another terminal:"
    echo "    source /opt/ros/humble/setup.bash"
    echo "    source ${WS}/install/setup.bash"
    echo "    ros2 launch robstride_can_driver driver.launch.py"
    echo ""
    read -rp "  Continue anyway? (y/N): " yn
    [[ "${yn,,}" == "y" ]] || exit 1
else
    echo "[lid] ✓ robstride_can_driver is running"
fi

# --------------------------------------------------------------------------- #
# Start lid controller
# --------------------------------------------------------------------------- #
LID_PID=""
restart_lid

# --------------------------------------------------------------------------- #
# Menu
# --------------------------------------------------------------------------- #
cleanup() {
    echo ""
    echo "[lid] Stopping lid_controller..."
    kill "${LID_PID}" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

print_menu() {
    echo ""
    echo "════════════════════════════════════════════"
    echo "  LID TUNING MENU"
    echo "════════════════════════════════════════════"
    echo "  Position"
    echo "    p          — print current position"
    echo "    m          — monitor position live (Ctrl+C to stop)"
    echo "    o          — open lid (open_position_rad)"
    echo "    c          — close lid (closed_position_rad)"
    echo "    t <rad>    — move to arbitrary position"
    echo ""
    echo "  Calibration"
    echo "    z          — set mechanical zero (lid must be physically closed)"
    echo "    sopen      — save current position as open_position_rad in YAML"
    echo "    sclosed    — save current position as closed_position_rad in YAML"
    echo ""
    echo "  Gains (written to motor live, volatile)"
    echo "    g          — show current YAML config"
    echo "    kp <val>   — set loc_kp  (position P, default 40.0)"
    echo "    vp <val>   — set spd_kp  (velocity P, default 6.0)"
    echo "    vi <val>   — set spd_ki  (velocity I, default 0.02)"
    echo "    spd <val>  — set limit_spd (rad/s, default 2.0)"
    echo "    cur <val>  — set limit_cur (A, default 2.0)"
    echo ""
    echo "  Other"
    echo "    s          — show /lid_state"
    echo "    r          — restart lid_controller with current YAML"
    echo "    ?          — show this menu"
    echo "    q          — quit"
    echo "════════════════════════════════════════════"
    echo ""
}

print_menu

while true; do
    read -rp "cmd> " input
    cmd="${input%% *}"
    arg="${input#* }"
    [[ "$arg" == "$cmd" ]] && arg=""  # no argument provided

    case "$cmd" in

        p)
            echo -n "  Current position: "
            get_position
            ;;

        m)
            echo "  Monitoring /joint_states for ${JOINT} (Ctrl+C to stop)..."
            ros2 topic echo /joint_states 2>/dev/null | python3 -u -c "
import sys, re
buf = ''
for line in sys.stdin:
    buf += line
    if 'velocity:' in buf and 'position:' in buf:
        name_section = buf.split('position:')[0]
        names = re.findall(r'^-\s+(\S+)', name_section, re.MULTILINE)
        vals  = re.findall(r'[-+]?[0-9]*\.?[0-9]+(?:e[-+]?[0-9]+)?',
                            buf.split('position:')[1].split('velocity:')[0])
        try:
            idx = names.index('${JOINT}')
            print(f'\r  pos: {float(vals[idx]):+.4f} rad    ', end='', flush=True)
        except:
            pass
        buf = ''
" || true
            echo ""
            ;;

        o)
            echo "  Sending OPEN..."
            ros2 topic pub /lid_command std_msgs/msg/String '{data: "open"}' --once > /dev/null
            ;;

        c)
            echo "  Sending CLOSE..."
            ros2 topic pub /lid_command std_msgs/msg/String '{data: "close"}' --once > /dev/null
            ;;

        t)
            if [[ -z "$arg" ]]; then
                echo "  Usage: t <radians>  e.g.  t 0.785"
            else
                echo "  Moving to ${arg} rad via loc_ref..."
                write_param "${PARAM_LOC_REF}" "${arg}" "float"
            fi
            ;;

        z)
            echo "  Setting mechanical zero for ${JOINT}..."
            echo "  (Lid should be physically at the CLOSED position now)"
            ros2 service call /set_zero \
                robstride_can_interfaces/srv/SetZero \
                "{joint_name: '${JOINT}'}" 2>/dev/null | grep -E "success|message" || true
            echo "  Zero set. closed_position_rad should now be ~0.0"
            ;;

        sopen)
            echo -n "  Reading position... "
            pos="$(get_position | awk '{print $1}')"
            echo "${pos}"
            yaml_set "open_position_rad" "${pos}"
            ;;

        sclosed)
            echo -n "  Reading position... "
            pos="$(get_position | awk '{print $1}')"
            echo "${pos}"
            yaml_set "closed_position_rad" "${pos}"
            ;;

        g)
            echo "  ── lid_controller.yaml ──────────────────────"
            cat "${LID_CONFIG}"
            echo "  ─────────────────────────────────────────────"
            ;;

        kp)
            if [[ -z "$arg" ]]; then echo "  Usage: kp <value>"; else
                echo "  Setting loc_kp = ${arg}"
                write_param "${PARAM_LOC_KP}" "${arg}" "float"
            fi
            ;;

        vp)
            if [[ -z "$arg" ]]; then echo "  Usage: vp <value>"; else
                echo "  Setting spd_kp = ${arg}"
                write_param "${PARAM_SPD_KP}" "${arg}" "float"
            fi
            ;;

        vi)
            if [[ -z "$arg" ]]; then echo "  Usage: vi <value>"; else
                echo "  Setting spd_ki = ${arg}"
                write_param "${PARAM_SPD_KI}" "${arg}" "float"
            fi
            ;;

        spd)
            if [[ -z "$arg" ]]; then echo "  Usage: spd <rad/s>"; else
                echo "  Setting limit_spd = ${arg} rad/s"
                write_param "${PARAM_LIMIT_SPD}" "${arg}" "float"
            fi
            ;;

        cur)
            if [[ -z "$arg" ]]; then echo "  Usage: cur <amps>"; else
                echo "  Setting limit_cur = ${arg} A"
                write_param "${PARAM_LIMIT_CUR}" "${arg}" "float"
            fi
            ;;

        s)
            echo "  /lid_state:"
            ros2 topic echo /lid_state --once 2>/dev/null || echo "  (no data)"
            ;;

        r)
            restart_lid
            ;;

        \?|help)
            print_menu
            ;;

        q|quit)
            cleanup
            ;;

        "")
            ;;

        *)
            echo "  Unknown command '${cmd}'. Press ? for help."
            ;;
    esac
done
