#!/bin/bash
set -e

cd ~/TeamBowl/teambowl_ws

echo "[build] Setting can network up..."
# sudo modprobe mttcan
# sudo ip link set can0 type can bitrate 1000000
# sudo ip link set can1 type can bitrate 1000000
# sudo ip link set can0 up
# sudo ip link set can1 up

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
    state_estimation \

echo "[build] Sourcing workspace overlay..."
source install/setup.bash