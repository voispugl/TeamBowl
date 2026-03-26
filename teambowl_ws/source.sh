#!/bin/bash
set -e

cd ~/TeamBowl/teambowl_ws

echo "[build] Sourcing ROS Humble..."
source /opt/ros/humble/setup.bash

echo "[build] Sourcing workspace overlay..."
source install/setup.bash