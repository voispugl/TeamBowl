#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

# Source nvblox overlay built into the image
if [ -f /opt/isaac_ros_ws/install/setup.bash ]; then
    source /opt/isaac_ros_ws/install/setup.bash
fi

WS=/workspaces/teambowl_ws
BUILD_MARKER="${WS}/install/.colcon_isaac_sim_build_complete"

# Build robot workspace on first run (software packages only — no hardware drivers).
# Separate marker from the robot image marker (.colcon_build_complete) so both containers
# can share the same teambowl_ws volume without conflicting build state.
if [ ! -f "${BUILD_MARKER}" ]; then
    echo "[isaac_sim] First run — building software packages (skip hardware drivers)..."
    cd "${WS}"

    # Hardware drivers not present or needed on the desktop:
    #   depthai-ros       — OAK-D camera (PoE, not attached to desktop)
    #   robstride_*       — CAN bus actuators
    #   vesc_driver       — wheel motor controllers
    #   xsens_mti_*       — IMU (simulated by Isaac Sim)
    SKIP="depthai_filters depthai_examples depthai_bridge depthai_ros_driver \
          depthai_ros_msgs depthai_descriptions \
          robstride_can_driver robstride_can_interfaces \
          vesc_driver \
          xsens_mti_ros2_driver"

    colcon build \
        --packages-ignore ${SKIP} \
        --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DBUILD_TESTING=OFF \
        2>&1 | tee "${WS}/colcon_build.log"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[isaac_sim] ERROR: colcon build failed. Check colcon_build.log in the workspace."
        exit 1
    fi

    touch "${BUILD_MARKER}"
    echo "[isaac_sim] Build complete."
fi

# Source the built workspace overlay
if [ -f "${WS}/install/setup.bash" ]; then
    source "${WS}/install/setup.bash"
fi

# Launch Isaac Sim with the scene setup script.
# The script imports the robot URDF, places obstacles, and starts the ROS2 bridge.
SCENE_SCRIPT="${WS}/src/simulation/isaac_sim/setup_scene.py"
if [ -f "${SCENE_SCRIPT}" ]; then
    echo "[isaac_sim] Starting Isaac Sim..."
    echo "[isaac_sim] WebRTC UI: http://localhost:8211"
    echo "[isaac_sim] Foxglove:  ws://localhost:8765"
    exec /isaac-sim/runapp.sh \
        --/app/window/title="TeamBowl Isaac Sim" \
        --enable omni.isaac.ros2_bridge \
        --enable omni.kit.livestream.webrtc \
        --exec "${SCENE_SCRIPT}" \
        "$@"
else
    echo "[isaac_sim] Scene script not found at ${SCENE_SCRIPT}"
    echo "[isaac_sim] Dropping to bash. Run setup_scene.py manually or rebuild workspace."
    exec bash
fi
