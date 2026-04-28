#!/bin/bash
set -e

cd ~/TeamBowl/teambowl_ws

echo "[build] Setting can network up..."
sudo modprobe mttcan
sudo ip link set can0 down
sudo ip link set can1 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can0 up
sudo ip link set can1 up

echo "[build] Sourcing ROS Humble..."
source /opt/ros/humble/setup.bash

echo "[build] Sourcing workspace overlay..."
source install/setup.bash

export RCUTILS_COLORIZE_OUTPUT=1
export RCUTILS_LOGGING_USE_STDOUT=1

echo "[build] Launching bringup with full steamdeck web UI..."
ros2 launch bringup bringup.launch.py steamdeck_ui:=full velocity_controller:=driving
