#!/bin/bash
set -e

cd ~/TeamBowl/teambowl_ws

echo "[ekf_test] Sourcing ROS Humble..."
source /opt/ros/humble/setup.bash

echo "[ekf_test] Sourcing workspace overlay..."
source install/setup.bash

echo "[ekf_test] Launching EKF test stack (Xsens + Oak-D IMU + EKF + Foxglove)..."
echo "[ekf_test] Connect Foxglove to ws://$(hostname -I | awk '{print $1}'):8765"
ros2 launch bringup ekf_test.launch.py
