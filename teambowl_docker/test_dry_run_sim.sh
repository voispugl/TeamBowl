#!/bin/bash
# Dry-run smoke test for the MuJoCo sim image (teambowl:sim).
# Verifies the image, workspace, packages, and sim-specific setup
# without any hardware or physical robot.
#
# Usage: ./test_dry_run_sim.sh
# Exit code: 0 = all pass, 1 = one or more failures

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export TEAMBOWL_ROOT="${TEAMBOWL_ROOT:-${DEFAULT_ROOT}}"
export TEAMBOWL_WS="${TEAMBOWL_WS:-${TEAMBOWL_ROOT}/teambowl_ws}"
cd "${SCRIPT_DIR}"

echo ""
echo "=============================="
echo " TeamBowl Sim Dry-Run Test"
echo "=============================="

# ── Pre-flight ─────────────────────────────────────────────────────────────────
for img in teambowl:laptop teambowl:sim; do
    if ! docker image inspect "$img" > /dev/null 2>&1; then
        echo "FAIL  $img image not found — run ./build.sim.sh first"
        exit 1
    fi
    echo "  PASS  $img image exists"
done

# ── All checks inside one container ───────────────────────────────────────────
docker compose -f docker-compose.sim.yml run --rm --no-TTY teambowl-sim bash -s << 'EOF'
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
source /workspaces/teambowl_ws/install/setup.bash

# ── Workspace packages ─────────────────────────────────────────────────────────
echo ""
echo "-- Workspace packages --"
EXPECTED_PKGS=(
    bringup
    locomotion
    management
    safety
    simulation
    planning
    state_estimation
    steamdeck_teleop
)
for pkg in "${EXPECTED_PKGS[@]}"; do
    check "pkg: $pkg" bash -c "ros2 pkg list | grep -qw $pkg"
done

# ── Sim-specific executables ───────────────────────────────────────────────────
echo ""
echo "-- Executables --"
declare -A EXECUTABLES=(
    [simulation]="mujoco_bridge"
    [locomotion]="balance_controller driving_controller vel_cmd_mux collision_guard"
    [management]="mode_manager"
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
)
for f in "${CONFIG_FILES[@]}"; do
    check "file: $(basename $f)" test -f "$f"
done

# ── Launch file ────────────────────────────────────────────────────────────────
echo ""
echo "-- Launch file --"
check "sim.launch.py parses (--show-args)" \
    bash -c "ros2 launch bringup sim.launch.py --show-args 2>&1 | grep -q 'Arguments'"

# ── ROS2 environment ───────────────────────────────────────────────────────────
echo ""
echo "-- ROS2 environment --"
check "ros2 doctor" bash -c "ros2 doctor 2>&1 | grep -qE 'All .* passed|1/1'"

# ── Python imports ─────────────────────────────────────────────────────────────
echo ""
echo "-- Python imports --"
check "import mujoco"    python3 -c "import mujoco; print(mujoco.__version__)"
check "import pyvesc"    python3 -c "import pyvesc"
check "import can"       python3 -c "import can"
check "import websockets" python3 -c "import websockets"
check "import cv2"       python3 -c "import cv2"

# ── MuJoCo model reachable ─────────────────────────────────────────────────────
echo ""
echo "-- MuJoCo model --"
MODEL=/workspaces/teambowl_mjlab/teambowl_mjlab.xml
if [ -f "$MODEL" ]; then
    check "MuJoCo model loads (teambowl_mjlab.xml)" \
        python3 -c "import mujoco; mujoco.MjModel.from_xml_path('$MODEL')"
else
    echo "  SKIP  MuJoCo model (teambowl_mjlab not mounted — run ./launch.sim.sh to use it)"
fi

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
