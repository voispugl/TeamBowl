#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

# Source Isaac ROS overlay (built into Docker image at /opt/isaac_ros_ws).
# Must be sourced before robot workspace colcon build so packages can find
# isaac_ros_visual_slam and nvblox_ros as ament dependencies.
if [ -f /opt/isaac_ros_ws/install/setup.bash ]; then
    source /opt/isaac_ros_ws/install/setup.bash
fi

WS=/workspaces/teambowl_ws
BUILD_MARKER="${WS}/install/.colcon_build_complete"

# Auto-build the full workspace on first run.
# The marker file persists on the volume so subsequent container starts skip this.
if [ ! -f "${BUILD_MARKER}" ]; then
    echo "[entrypoint] First run — building full workspace (this will take several minutes)..."
    cd "${WS}"

    # Always ignore: depthai_filters (requires opencv_contrib/ximgproc — not present in
    # the Jetson CUDA OpenCV or standard Ubuntu opencv). depthai_examples is skipped
    # because depthai_ros_driver no longer declares a dependency on it.
    # SKIP_DEPTHAI=1: also ignore all depthai source packages (laptop builds — the
    # OAK-D camera is never attached to a laptop, so the C++ SDK is not installed).
    IGNORE_PKGS="depthai_filters depthai_examples"
    if [ "${SKIP_DEPTHAI:-0}" = "1" ]; then
        echo "[entrypoint] SKIP_DEPTHAI=1 — skipping all depthai-ros source packages"
        IGNORE_PKGS="$IGNORE_PKGS depthai_bridge depthai_ros_driver depthai_ros_msgs depthai_descriptions"
    fi

    colcon build \
        --symlink-install \
        --packages-ignore ${IGNORE_PKGS} \
        --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=OFF \
        2>&1 | tee "${WS}/colcon_build.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[entrypoint] ERROR: colcon build failed. Check colcon_build.log in the workspace."
        exit 1
    fi

    touch "${BUILD_MARKER}"
    echo "[entrypoint] Build complete."
fi

# Source the built workspace overlay
if [ -f "${WS}/install/setup.bash" ]; then
    source "${WS}/install/setup.bash"
fi

exec "$@"
