#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

WS=/workspaces/teambowl_ws
BUILD_MARKER="${WS}/install/.colcon_build_complete"

# Auto-build the full workspace on first run.
# The marker file persists on the volume so subsequent container starts skip this.
if [ ! -f "${BUILD_MARKER}" ]; then
    echo "[entrypoint] First run — building full workspace (this will take several minutes)..."
    cd "${WS}"

    # depthai_filters requires opencv_contrib (ximgproc) which is NOT included in
    # the Jetson CUDA OpenCV. depthai_examples is skipped because depthai_ros_driver
    # no longer declares a dependency on it. Neither package is used by bringup.launch.py.
    colcon build \
        --symlink-install \
        --packages-ignore depthai_filters depthai_examples \
        --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=OFF \
        2>&1 | tee /tmp/colcon_build.log

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[entrypoint] ERROR: colcon build failed. Check /tmp/colcon_build.log for details."
        echo "[entrypoint] Dropping to bash for debugging."
        exec bash
    fi

    touch "${BUILD_MARKER}"
    echo "[entrypoint] Build complete."
fi

# Source the built workspace overlay
if [ -f "${WS}/install/setup.bash" ]; then
    source "${WS}/install/setup.bash"
fi

exec "$@"
