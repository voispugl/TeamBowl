#!/bin/bash
# Diagnose why the wheels don't move when "Go" is pressed in the steamdeck web UI.
# Run while the robot stack is up: bash ~/TeamBowl/debug_wheels.sh

source /opt/ros/humble/setup.bash
source ~/TeamBowl/teambowl_ws/install/setup.bash

python3 ~/TeamBowl/debug_wheels.py
