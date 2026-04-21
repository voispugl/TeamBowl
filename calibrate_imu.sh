#!/bin/bash
set -e

cd ~/TeamBowl/teambowl_ws

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║          IMU Allan Variance Calibration              ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║  Place the robot on a flat, stable surface.          ║"
echo "║  Do NOT move it during recording.                    ║"
echo "║  Minimum recording time: 30 minutes.                 ║"
echo "║  Press Ctrl+C to stop recording when done.           ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

BAG_NAME=~/TeamBowl/imu_calib_$(date +%Y%m%d_%H%M%S)

echo "[calibrate_imu] Launching IMU drivers..."
ros2 launch bringup ekf_test.launch.py &
LAUNCH_PID=$!

echo "[calibrate_imu] Waiting 8 s for IMU drivers to start..."
sleep 8

echo "[calibrate_imu] Recording to: $BAG_NAME"
echo "[calibrate_imu] Topics: /imu/data  /oak/imu/data"
echo ""
ros2 bag record /imu/data /oak/imu/data -o "$BAG_NAME"

kill $LAUNCH_PID 2>/dev/null || true

echo ""
echo "[calibrate_imu] Recording complete: $BAG_NAME"
echo "[calibrate_imu] Now run:"
echo "  python3 ~/TeamBowl/calibrate_imu.py $BAG_NAME"
