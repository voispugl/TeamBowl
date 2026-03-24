#!/bin/bash
set -e

cd ~/TeamBowl/teambowl_ws

echo "[build] Cleaning previous build artifacts..."
rm -rf build install log

echo "[build] Sourcing ROS Humble..."
source /opt/ros/humble/setup.bash

echo "[build] Building workspace packages..."
colcon build --symlink-install --packages-select \
    depthai_ros_msgs \
    depthai_bridge \
    depthai_descriptions \
    depthai_ros_driver \
    robstride_can_interfaces \
    robstride_can_driver \
    bringup \
    locomotion \
    management \
    safety \
    perception \
    planning \
    vesc_driver \
    state_estimation 

echo "[build] Sourcing workspace overlay..."
source install/setup.bash

echo "[build] Launching bringup..."
ros2 launch bringup bringup.launch.py
