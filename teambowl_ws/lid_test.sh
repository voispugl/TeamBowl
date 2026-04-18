#!/bin/bash
# =============================================================================
# lid_test.sh — Lid controller test & tuning session
#
# Starts a tmux session with 6 panes:
#
#   +---------------------------+---------------------------+
#   |  [DRIVER] robstride_can_  |  [LID STATE] /lid_state  |
#   |  driver + lid_controller  |  + /estop                 |
#   +---------------------------+---------------------------+
#   |  [POSITION] /joint_states |  [COMMANDS] /joint_cmds  |
#   |  joint_rs05_1 feedback    |  joint_rs05_1 targets    |
#   +---------------------------+---------------------------+
#   |  [PARAMS] current gains   |  [CONTROL] send commands |
#   +---------------------------+---------------------------+
#
# Usage:
#   ./lid_test.sh              # start session
#   ./lid_test.sh stop         # kill session
#
# Tuning cheat sheet (run in the CONTROL pane):
#   open       → lid_open
#   close      → lid_close
#   toggle     → lid_toggle
#   see gains  → lid_params
#   set gain   → ros2 param set /lid_controller kp 80.0
#                (NOTE: gains take effect only after restart)
#   restart    → lid_restart   (re-reads all params from YAML)
#
# Key parameters in: teambowl_ws/src/locomotion/config/lid_controller.yaml
#   open_position_rad    — where the lid goes on "open"  (tune first)
#   closed_position_rad  — where the lid goes on "close" (tune first)
#   kp                   — MIT mode Kp (stiffness)
#   kd                   — MIT mode Kd (damping, prevents oscillation)
#   torque_ff            — feedforward torque while moving (prevents stalling)
#   move_timeout_sec     — seconds before declaring "arrived" regardless of pos
#   position_tolerance_rad — how close is "close enough" to declare arrived
#
# IMPORTANT: joint_rs05_1 is disabled by default in motors.yaml.
#   Set `enabled: true` in:
#   teambowl_ws/src/drivers/robstride_can_driver/config/motors.yaml
#   then rebuild: colcon build --packages-select robstride_can_driver
# =============================================================================

set -e

SESSION="lid_test"
WS="$HOME/TeamBowl/teambowl_ws"
ROS_SETUP="/opt/ros/humble/setup.bash"

# ── Handle stop ──────────────────────────────────────────────────────────────
if [[ "$1" == "stop" ]]; then
    echo "Killing tmux session '$SESSION'..."
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "Done." || echo "Session not found."
    exit 0
fi

# ── Pre-flight checks ─────────────────────────────────────────────────────────
MOTORS_YAML="$WS/src/drivers/robstride_can_driver/config/motors.yaml"

if grep -A2 "joint_rs05_1:" "$MOTORS_YAML" | grep -q "enabled: false"; then
    echo ""
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║  WARNING: joint_rs05_1 is DISABLED in motors.yaml            ║"
    echo "  ║                                                                ║"
    echo "  ║  Edit: $MOTORS_YAML"
    echo "  ║  Change: enabled: false  →  enabled: true                     ║"
    echo "  ║  Then:   colcon build --packages-select robstride_can_driver   ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
    echo ""
    read -rp "  Continue anyway? (y/N) " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists. Attaching..."
    tmux attach-session -t "$SESSION"
    exit 0
fi

# ── CAN setup ─────────────────────────────────────────────────────────────────
echo "[lid_test] Setting up CAN interfaces..."
sudo modprobe mttcan 2>/dev/null || true
sudo ip link set can0 down 2>/dev/null || true
sudo ip link set can1 down 2>/dev/null || true
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can0 up
sudo ip link set can1 up
echo "[lid_test] CAN up."

# ── Shared preamble ───────────────────────────────────────────────────────────
# Each tmux pane sources ROS + workspace, then runs its command.
PREAMBLE="source $ROS_SETUP && source $WS/install/setup.bash && cd $WS"

# Convenience aliases injected into the CONTROL pane
ALIASES="
alias lid_open='ros2 topic pub --once /lid_command std_msgs/msg/String \"{data: open}\"'
alias lid_close='ros2 topic pub --once /lid_command std_msgs/msg/String \"{data: close}\"'
alias lid_toggle='ros2 topic pub --once /lid_command std_msgs/msg/String \"{data: toggle}\"'
alias lid_params='ros2 param list /lid_controller && echo --- && ros2 param get /lid_controller open_position_rad && ros2 param get /lid_controller closed_position_rad && ros2 param get /lid_controller kp && ros2 param get /lid_controller kd && ros2 param get /lid_controller torque_ff && ros2 param get /lid_controller move_timeout_sec && ros2 param get /lid_controller position_tolerance_rad'
alias lid_restart='ros2 lifecycle set /lid_controller shutdown 2>/dev/null; pkill -f lid_controller 2>/dev/null; sleep 0.5; ros2 run locomotion lid_controller &'
alias lid_zero='ros2 service call /set_zero std_srvs/srv/Trigger {}  # zero RS05 at current position'
alias lid_enable='ros2 service call /enable_motors std_srvs/srv/Trigger {}'
alias lid_stop_motors='ros2 service call /stop_motors std_srvs/srv/Trigger {}'
alias estop_on='ros2 topic pub --once /estop std_msgs/msg/Bool \"{data: true}\"'
alias estop_off='ros2 topic pub --once /estop std_msgs/msg/Bool \"{data: false}\"'
"

# ── Create tmux session ───────────────────────────────────────────────────────
tmux new-session  -d -s "$SESSION" -x 220 -y 50

# ─ Layout: 3 rows × 2 columns ─────────────────────────────────────────────
# Row 0 (top)
tmux rename-window -t "$SESSION:0" "lid"

# Split into 2 columns (top row)
tmux split-window  -t "$SESSION:0" -h          # pane 0=left, 1=right

# Row 1 (middle) — split both columns vertically
tmux select-pane   -t "$SESSION:0.0"
tmux split-window  -t "$SESSION:0.0" -v        # pane 2 below pane 0

tmux select-pane   -t "$SESSION:0.1"
tmux split-window  -t "$SESSION:0.1" -v        # pane 3 below pane 1

# Row 2 (bottom) — split both middle panes vertically
tmux select-pane   -t "$SESSION:0.2"
tmux split-window  -t "$SESSION:0.2" -v        # pane 4 below pane 2

tmux select-pane   -t "$SESSION:0.3"
tmux split-window  -t "$SESSION:0.3" -v        # pane 5 below pane 3

# ── Pane 0 — DRIVER ──────────────────────────────────────────────────────────
# Runs robstride_can_driver + lid_controller. robstride must start first.
tmux select-pane   -t "$SESSION:0.0"
tmux send-keys     -t "$SESSION:0.0" "$PREAMBLE && \
  echo '=== DRIVER: robstride_can_driver ===' && \
  ros2 launch robstride_can_driver driver.launch.py" Enter

# ── Pane 1 — LID STATE ───────────────────────────────────────────────────────
# /lid_state shows current FSM state; /estop shows safety status
tmux select-pane   -t "$SESSION:0.1"
tmux send-keys     -t "$SESSION:0.1" "$PREAMBLE && \
  echo '=== LID STATE + ESTOP ===' && \
  sleep 3 && \
  ( ros2 topic echo /lid_state & ros2 topic echo /estop & wait )" Enter

# ── Pane 2 — POSITION (joint_states for joint_rs05_1) ────────────────────────
# Position feedback from the motor encoder — primary tuning signal.
# Shows actual position vs target to judge if kp/kd/torque_ff are adequate.
tmux select-pane   -t "$SESSION:0.2"
tmux send-keys     -t "$SESSION:0.2" "$PREAMBLE && \
  echo '=== /joint_states  (joint_rs05_1 position/velocity/effort) ===' && \
  sleep 3 && \
  ros2 topic echo /joint_states | grep -A 10 'joint_rs05_1'" Enter

# ── Pane 3 — COMMANDS (joint_commands targeting joint_rs05_1) ────────────────
# What position + effort the lid_controller is commanding.
# Lets you see the target_pos, torque_ff, and when it goes to zero (holding).
tmux select-pane   -t "$SESSION:0.3"
tmux send-keys     -t "$SESSION:0.3" "$PREAMBLE && \
  echo '=== /joint_commands  (what lid_controller is sending) ===' && \
  sleep 3 && \
  ros2 topic echo /joint_commands | grep -A 10 'joint_rs05_1'" Enter

# ── Pane 4 — PARAMS ──────────────────────────────────────────────────────────
# Shows all current lid_controller parameter values at a glance.
# Run `lid_params` any time to refresh. Use `ros2 param set` to change live.
# NOTE: params only take effect on restart (node reads them at init time).
tmux select-pane   -t "$SESSION:0.4"
tmux send-keys     -t "$SESSION:0.4" "$PREAMBLE && \
  echo '=== LID CONTROLLER PARAMS ===' && \
  echo 'NOTE: ros2 param set takes effect only after lid_restart' && \
  echo 'Edit YAML: src/locomotion/config/lid_controller.yaml' && \
  sleep 5 && \
  ros2 param list /lid_controller 2>/dev/null || echo '(waiting for node...)'" Enter

# ── Pane 5 — CONTROL ─────────────────────────────────────────────────────────
# Interactive shell with aliases for all common commands.
# Run `lid_controller` to see current node, or use aliases below.
tmux select-pane   -t "$SESSION:0.5"
tmux send-keys     -t "$SESSION:0.5" "$PREAMBLE && \
  sleep 2 && ros2 run locomotion lid_controller & \
  sleep 0.5" Enter
# Inject aliases and print help
tmux send-keys     -t "$SESSION:0.5" "$ALIASES" Enter
tmux send-keys     -t "$SESSION:0.5" "echo ''" Enter
tmux send-keys     -t "$SESSION:0.5" "echo '============================== LID TEST CONTROL =============================='" Enter
tmux send-keys     -t "$SESSION:0.5" "echo '  lid_open          — send open command'" Enter
tmux send-keys     -t "$SESSION:0.5" "echo '  lid_close         — send close command'" Enter
tmux send-keys     -t "$SESSION:0.5" "echo '  lid_toggle        — toggle current state'" Enter
tmux send-keys     -t "$SESSION:0.5" "echo '  lid_params        — print all current gain values'" Enter
tmux send-keys     -t "$SESSION:0.5" "echo '  lid_restart       — restart lid_controller (re-reads YAML gains)'" Enter
tmux send-keys     -t "$SESSION:0.5" "echo '  lid_zero          — zero RS05 at current position (careful!)'" Enter
tmux send-keys     -t "$SESSION:0.5" "echo '  lid_enable        — call /enable_motors'" Enter
tmux send-keys     -t "$SESSION:0.5" "echo '  estop_on/off      — toggle estop'" Enter
tmux send-keys     -t "$SESSION:0.5" "echo ''" Enter
tmux send-keys     -t "$SESSION:0.5" "echo '  ros2 param set /lid_controller open_position_rad 1.57'" Enter
tmux send-keys     -t "$SESSION:0.5" "echo '  ros2 param set /lid_controller kp 80.0'" Enter
tmux send-keys     -t "$SESSION:0.5" "echo '  ros2 param set /lid_controller torque_ff 0.8'" Enter
tmux send-keys     -t "$SESSION:0.5" "echo '================================================================================'" Enter

# ── Pane titles ──────────────────────────────────────────────────────────────
tmux select-pane   -t "$SESSION:0.0" -T "DRIVER"
tmux select-pane   -t "$SESSION:0.1" -T "LID STATE"
tmux select-pane   -t "$SESSION:0.2" -T "POSITION (joint_rs05_1)"
tmux select-pane   -t "$SESSION:0.3" -T "COMMANDS (joint_rs05_1)"
tmux select-pane   -t "$SESSION:0.4" -T "PARAMS"
tmux select-pane   -t "$SESSION:0.5" -T "CONTROL"

# ── Focus on the control pane and attach ─────────────────────────────────────
tmux select-pane   -t "$SESSION:0.5"
tmux attach-session -t "$SESSION"
