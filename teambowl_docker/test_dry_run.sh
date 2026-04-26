#!/bin/bash
# Dry-run smoke test — validates the Docker image and workspace without any hardware.
# Runs all checks inside a single container invocation and reports pass/fail.
#
# Usage: ./test_dry_run.sh
# Exit code: 0 = all pass, 1 = one or more failures

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo ""
echo "=============================="
echo "   TeamBowl Dry-Run Test"
echo "=============================="

# ── Pre-flight: image must exist ──────────────────────────────────────────────
if ! docker image inspect teambowl:dev > /dev/null 2>&1; then
    echo "FAIL  teambowl:dev image not found — run ./build.sh first"
    exit 1
fi
echo "  PASS  teambowl:dev image exists"

# ── All remaining checks run inside one container ─────────────────────────────
docker compose run --rm --no-TTY teambowl bash -s << 'EOF'
set +e   # don't abort on first failure — we want to collect all results

PASS=0
FAIL=0

ok()   { echo "  PASS  $1"; ((PASS++)); }
fail() { echo "  FAIL  $1"; ((FAIL++)); }

check() {
    local desc="$1"; shift
    if "$@" > /dev/null 2>&1; then ok "$desc"; else fail "$desc"; fi
}

# ── Source overlays ───────────────────────────────────────────────────────────
source /opt/ros/humble/setup.bash
if [ -f /opt/isaac_ros_ws/install/setup.bash ]; then
    source /opt/isaac_ros_ws/install/setup.bash
fi
source /workspaces/teambowl_ws/install/setup.bash

# ── Workspace packages ────────────────────────────────────────────────────────
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
    vesc_driver
    robstride_can_driver
    robstride_can_interfaces
    steamdeck_teleop
    depthai_ros_driver
    depthai_ros_msgs
    simulation
)
for pkg in "${EXPECTED_PKGS[@]}"; do
    check "pkg: $pkg" bash -c "ros2 pkg list | grep -qw $pkg"
done

# ── Key executables ───────────────────────────────────────────────────────────
echo ""
echo "-- Executables --"
declare -A EXECUTABLES=(
    [locomotion]="driving_leg_controller hold_position_controller lid_controller jump_controller balance_controller driving_controller wheel_odom fall_recovery_controller vel_cmd_mux collision_guard"
    [management]="mode_manager"
    [safety]="pico_bridge stuck_detector heartbeat_publisher system_health"
    [perception]="cam_ops"
    [planning]="nav_cloud_filter trajectory_test follow_goal follow_executor"
    [state_estimation]="diff_drive_odom"
    [vesc_driver]="cmd_vel_to_vesc"
    [robstride_can_driver]="robstride_can_driver"
)
for pkg in "${!EXECUTABLES[@]}"; do
    for exe in ${EXECUTABLES[$pkg]}; do
        check "exec: $pkg/$exe" bash -c "ros2 pkg executables $pkg | grep -qw $exe"
    done
done

# ── Config files reachable ────────────────────────────────────────────────────
echo ""
echo "-- Config files --"
CONFIG_FILES=(
    "$(ros2 pkg prefix management 2>/dev/null)/share/management/config/management.yaml"
    "$(ros2 pkg prefix safety 2>/dev/null)/share/safety/config/safety.yaml"
    "$(ros2 pkg prefix locomotion 2>/dev/null)/share/locomotion/config/locomotion.yaml"
    "$(ros2 pkg prefix locomotion 2>/dev/null)/share/locomotion/config/balance_controller.yaml"
    "$(ros2 pkg prefix locomotion 2>/dev/null)/share/locomotion/config/driving_controller.yaml"
    "$(ros2 pkg prefix locomotion 2>/dev/null)/share/locomotion/config/lid_controller.yaml"
    "$(ros2 pkg prefix locomotion 2>/dev/null)/share/locomotion/config/jump_controller.yaml"
    "$(ros2 pkg prefix vesc_driver 2>/dev/null)/share/vesc_driver/config/vesc_driver.yaml"
    "$(ros2 pkg prefix perception 2>/dev/null)/share/perception/config/perception.yaml"
    "$(ros2 pkg prefix planning 2>/dev/null)/share/planning/config/planning.yaml"
    "$(ros2 pkg prefix state_estimation 2>/dev/null)/share/state_estimation/config/state_estimation.yaml"
    "$(ros2 pkg prefix bringup 2>/dev/null)/share/bringup/robot_description/bowl.urdf"
    "$(ros2 pkg prefix bringup 2>/dev/null)/share/bringup/config/oak_cam.yaml"
)
for f in "${CONFIG_FILES[@]}"; do
    check "file: $(basename $f)" test -f "$f"
done

# ── Launch file parses ────────────────────────────────────────────────────────
echo ""
echo "-- Launch file --"
check "bringup.launch.py parses (--show-args)" \
    bash -c "ros2 launch bringup bringup.launch.py --show-args 2>&1 | grep -q 'Arguments'"

# ── Python imports ────────────────────────────────────────────────────────────
echo ""
echo "-- Python imports --"
check "import pyvesc"       python3 -c "import pyvesc"
check "import can"          python3 -c "import can"
check "import websockets"   python3 -c "import websockets"
check "import cv2"          python3 -c "import cv2"
check "cv2 built with CUDA" python3 -c "import cv2; info = cv2.getBuildInformation(); assert 'CUDA' in info and 'YES' in info[info.index('CUDA'):info.index('CUDA')+200]"
check "import torch"        python3 -c "import torch"
# GPU checks: only pass when Jetson GPU is accessible (not required for dry run)
if python3 -c "import torch; torch.cuda.is_available()" > /dev/null 2>&1 && \
   python3 -c "import torch; assert torch.cuda.is_available()" > /dev/null 2>&1; then
    ok  "torch CUDA available (GPU present)"
else
    echo "  SKIP  torch CUDA available (no GPU in this environment — OK for dry run)"
fi
if python3 -c "import cv2; assert cv2.cuda.getCudaEnabledDeviceCount() > 0" > /dev/null 2>&1; then
    ok  "cv2 CUDA device count > 0 (GPU present)"
else
    echo "  SKIP  cv2 CUDA device (no GPU in this environment — OK for dry run)"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
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
