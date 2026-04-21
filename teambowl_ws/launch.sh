#!/bin/bash
set -e

cd ~/TeamBowl/teambowl_ws

echo "[launch] Setting CAN network up..."
sudo modprobe mttcan
sudo ip link set can0 down
sudo ip link set can1 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can0 up
sudo ip link set can1 up

echo "[launch] Sourcing ROS Humble..."
source /opt/ros/humble/setup.bash

echo "[launch] Sourcing workspace overlay..."
source install/setup.bash

export RCUTILS_COLORIZE_OUTPUT=1
export RCUTILS_LOGGING_USE_STDOUT=1

# ── Pitch / fallover monitor ──────────────────────────────────────────────────
# Starts 8 s after launch (time for driving_gains_echo to appear).
# Prints colored lines to stdout alongside normal bringup output:
#   yellow  >8°   [pitch] 9.2°
#   orange  >15°  [PITCH WARN] 16.1°
#   red     >20°  [FALLOVER RISK] 21.3°  (balance_controller estop imminent)
python3 - 2>/dev/null <<'PYEOF' &
import json, time, signal, sys
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
time.sleep(8)

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

RED    = '\033[1;91m'
ORANGE = '\033[1;33m'
YELLOW = '\033[0;93m'
RESET  = '\033[0m'

class PitchMonitor(Node):
    def __init__(self):
        super().__init__('_pitch_monitor')
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.FATAL)
        self.create_subscription(String, '/driving_gains_echo', self._cb, 10)

    def _cb(self, msg):
        try:
            d = json.loads(msg.data)
            t = d.get('_theta_deg')
            if t is None:
                return
            a = abs(float(t))
            if a > 20:
                print(f'{RED}[FALLOVER RISK] pitch={t}°  ← estop imminent{RESET}', flush=True)
            elif a > 15:
                print(f'{ORANGE}[PITCH WARN] pitch={t}°{RESET}', flush=True)
            elif a > 8:
                print(f'{YELLOW}[pitch] {t}°{RESET}', flush=True)
        except Exception:
            pass

rclpy.init()
rclpy.spin(PitchMonitor())
PYEOF
PITCH_PID=$!

cleanup() {
    kill "$PITCH_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[launch] Launching bringup..."
ros2 launch bringup bringup.launch.py
