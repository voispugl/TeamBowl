#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo ""
echo "=============================="
echo "  CAMERA DEBUG LAUNCH"
echo "=============================="
echo "Launching: OAK-D camera + VSLAM + nvblox + Foxglove"
echo "No motors, CAN, Nav2, or robot hardware."
echo ""
echo "Foxglove: ws://$(hostname -I | awk '{print $1}'):8765"
echo "Topics to check:"
echo "  /oak/rgb/image_raw           ~10 Hz"
echo "  /oak/left/image_rect         ~30 Hz"
echo "  /oak/imu/data                turned off"
echo "  /visual_slam/tracking/odometry"
echo "  /nvblox/mesh"
echo ""
if [ $# -gt 0 ]; then
    echo "Args: $@"
    echo "=============================="
else
    echo "Optional args: vslam_debug:=true  use_nvblox:=false"
    echo "=============================="
fi
echo ""

docker compose run --rm teambowl \
    ros2 launch bringup isaac_ros_test.launch.py "$@"
