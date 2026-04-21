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

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# --- rosbag ---
BAG_DIR=~/TeamBowl/bags/$(date +%Y%m%d_%H%M%S)
mkdir -p ~/TeamBowl/bags
echo "[bag] Recording to $BAG_DIR (excluding raw images + pointclouds)"
ros2 bag record --all \
  --exclude "/oak/rgb/image_raw|/oak/rgb/image_rect|/oak/stereo/image_raw|/oak/left/image_raw|/oak/right/image_raw|/oak/points|/robot/debug/cam_ops_image" \
  --output "$BAG_DIR" &
BAG_PID=$!

BRINGUP_PID=""
cleanup() {
    if [ -n "$BRINGUP_PID" ]; then
        kill "$BRINGUP_PID" 2>/dev/null || true
    fi
    if [ -n "$BAG_PID" ]; then
        echo "[bag] Stopping rosbag recorder (PID $BAG_PID)..."
        kill "$BAG_PID" 2>/dev/null || true
        wait "$BAG_PID" 2>/dev/null || true
        echo "[bag] Bag saved to $BAG_DIR"
    fi
    if [ -n "$PITCH_PID" ]; then
        kill "$PITCH_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# pitch/fallover monitor (starts after 8s so ROS is up)
python3 - 2>/dev/null <<'PYEOF' &
import json, time, signal, sys
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
time.sleep(8)
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
RED='\033[1;91m'; ORANGE='\033[1;33m'; YELLOW='\033[0;93m'; RESET='\033[0m'
class PitchMonitor(Node):
    def __init__(self):
        super().__init__('_pitch_monitor')
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.FATAL)
        self.create_subscription(String, '/driving_gains_echo', self._cb, 10)
    def _cb(self, msg):
        try:
            d = json.loads(msg.data); t = d.get('_theta_deg')
            if t is None: return
            a = abs(float(t))
            if a > 20: print(f'{RED}[FALLOVER RISK] pitch={t}°  ← estop imminent{RESET}', flush=True)
            elif a > 15: print(f'{ORANGE}[PITCH WARN] pitch={t}°{RESET}', flush=True)
            elif a > 8: print(f'{YELLOW}[pitch] {t}°{RESET}', flush=True)
        except Exception: pass
rclpy.init()
rclpy.spin(PitchMonitor())
PYEOF
PITCH_PID=$!

echo "[build] Launching bringup..."
set +e
ros2 launch bringup bringup.launch.py &
BRINGUP_PID=$!
wait "$BRINGUP_PID"
