#!/bin/bash
# Dry-run smoke test for the Isaac Sim desktop image (teambowl:isaac_sim).
# Must be run on the desktop (x86_64 + RTX GPU) — NOT on the Jetson.
#
# Usage: ./test_dry_run_isaac_sim.sh
# Exit code: 0 = all pass, 1 = one or more failures

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEAMBOWL_WS="${TEAMBOWL_WS:-${HOME}/TeamBowl/teambowl_ws}"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.isaac_sim.yml"
cd "${SCRIPT_DIR}"

echo ""
echo "=============================="
echo " Isaac Sim Dry-Run Test"
echo "=============================="

# ── Pre-flight ─────────────────────────────────────────────────────────────────
if ! docker image inspect teambowl:isaac_sim > /dev/null 2>&1; then
    echo "FAIL  teambowl:isaac_sim image not found — run ./build.isaac_sim.sh first"
    exit 1
fi
echo "  PASS  teambowl:isaac_sim image exists"

# ── All checks inside one container ───────────────────────────────────────────
TEAMBOWL_WS="${TEAMBOWL_WS}" docker compose -f "${COMPOSE_FILE}" run --rm --no-TTY \
    teambowl_isaac_sim bash -s << 'EOF'
set +e

PASS=0
FAIL=0

ok()   { echo "  PASS  $1"; ((PASS++)); }
fail() { echo "  FAIL  $1"; ((FAIL++)); }

check() {
    local desc="$1"; shift
    if "$@" > /dev/null 2>&1; then ok "$desc"; else fail "$desc"; fi
}

source /opt/ros/humble/setup.bash
if [ -f /opt/isaac_ros_ws/install/setup.bash ]; then
    source /opt/isaac_ros_ws/install/setup.bash
fi
source /workspaces/teambowl_ws/install/setup.bash 2>/dev/null || true

# ── Workspace packages (software-only — hardware drivers skipped) ──────────────
echo ""
echo "-- Workspace packages --"
EXPECTED_PKGS=(
    bringup
    locomotion
    management
    safety
    perception
    planning
    state_estimation
    simulation
    steamdeck_teleop
)
for pkg in "${EXPECTED_PKGS[@]}"; do
    check "pkg: $pkg" bash -c "ros2 pkg list | grep -qw $pkg"
done

# ── Executables ────────────────────────────────────────────────────────────────
echo ""
echo "-- Executables --"
declare -A EXECUTABLES=(
    [simulation]="mujoco_bridge"
    [locomotion]="balance_controller driving_controller vel_cmd_mux collision_guard"
    [management]="mode_manager"
    [safety]="pico_bridge stuck_detector heartbeat_publisher system_health"
    [planning]="nav_cloud_filter trajectory_test follow_goal follow_executor"
    [state_estimation]="diff_drive_odom"
    [perception]="cam_ops"
)
for pkg in "${!EXECUTABLES[@]}"; do
    for exe in ${EXECUTABLES[$pkg]}; do
        check "exec: $pkg/$exe" bash -c "ros2 pkg executables $pkg | grep -qw $exe"
    done
done

# ── Config files ───────────────────────────────────────────────────────────────
echo ""
echo "-- Config files --"
CONFIG_FILES=(
    "$(ros2 pkg prefix locomotion 2>/dev/null)/share/locomotion/config/locomotion.yaml"
    "$(ros2 pkg prefix locomotion 2>/dev/null)/share/locomotion/config/balance_controller.yaml"
    "$(ros2 pkg prefix locomotion 2>/dev/null)/share/locomotion/config/driving_controller.yaml"
    "$(ros2 pkg prefix management 2>/dev/null)/share/management/config/management.yaml"
    "$(ros2 pkg prefix simulation 2>/dev/null)/share/simulation/config/mujoco_bridge.yaml"
    "$(ros2 pkg prefix bringup 2>/dev/null)/share/bringup/config/diagnostics.yaml"
    "$(ros2 pkg prefix bringup 2>/dev/null)/share/bringup/robot_description/bowl.urdf"
)
for f in "${CONFIG_FILES[@]}"; do
    check "file: $(basename $f)" test -f "$f"
done

# ── Launch files ───────────────────────────────────────────────────────────────
echo ""
echo "-- Launch files --"
check "sim.launch.py parses" \
    bash -c "ros2 launch bringup sim.launch.py --show-args 2>&1 | grep -q 'Arguments'"
check "bringup.launch.py parses" \
    bash -c "ros2 launch bringup bringup.launch.py --show-args 2>&1 | grep -q 'Arguments'"

# ── Isaac Sim scene script ─────────────────────────────────────────────────────
echo ""
echo "-- Isaac Sim --"
check "setup_scene.py exists" \
    test -f /workspaces/teambowl_ws/src/simulation/isaac_sim/setup_scene.py
check "Isaac Sim runapp.sh exists" test -f /isaac-sim/runapp.sh

# ── ROS2 environment ───────────────────────────────────────────────────────────
echo ""
echo "-- ROS2 environment --"
check "ros2 doctor" bash -c "ros2 doctor 2>&1 | grep -qE 'All .* passed|1/1'"

# ── Python imports ─────────────────────────────────────────────────────────────
echo ""
echo "-- Python imports --"
check "import pyvesc"     python3 -c "import pyvesc"
check "import can"        python3 -c "import can"
check "import websockets" python3 -c "import websockets"
check "import cv2"        python3 -c "import cv2"

# GPU checks (require RTX on desktop)
if python3 -c "import torch; assert torch.cuda.is_available()" > /dev/null 2>&1; then
    ok  "torch CUDA available (GPU present)"
else
    echo "  SKIP  torch CUDA (no GPU or torch not installed — OK if Isaac Sim provides it)"
fi

# ── nvblox interface packages ──────────────────────────────────────────────────
echo ""
echo "-- nvblox (Isaac ROS overlay) --"
check "pkg: nvblox_msgs"      bash -c "ros2 pkg list | grep -q nvblox_msgs"
check "pkg: nvblox_ros_common" bash -c "ros2 pkg list | grep -q nvblox_ros_common"

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo "  $PASS passed   $FAIL failed"
echo "=============================="
[ $FAIL -eq 0 ] && exit 0 || exit 1
EOF

EXIT=$?
echo ""
[ $EXIT -eq 0 ] && echo "All checks passed." || echo "Some checks failed — see above."
exit $EXIT
