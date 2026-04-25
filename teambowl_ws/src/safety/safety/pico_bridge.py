#!/usr/bin/env python3
"""
pico_bridge — USB-serial bridge between the Pico and ROS2.

Responsibilities:
  1. LED control: maps robot state to Pico LED commands (highest priority wins).
  2. Kill-switch / lid button: GP15 on the Pico sends K1/K0 lines over USB serial.
       • While moving  → K1 asserts /kill_switch (e-stop)
       • While stopped → K1 toggles the cargo-bay lid via /lid_command "toggle"

LED priority (highest first):
  estop          → red solid          (0x00)
  turning right  → orange wave right  (0x20)
  turning left   → orange wave left   (0x21)
  moving fwd/rev → yellow solid       (0x02)
  stuck          → purple blink       (0x40 0x80 0x00 0x80)
  teleop idle    → blue solid         (0x10 0x00 0x00 0xFF)
  default/alive  → green solid        (0x01)

Serial protocol (ROS → Pico):
  0x00          red static
  0x01          green static
  0x02          yellow static
  0x20          wave right (orange — set beforehand via 0x10 0xFF 0x78 0x00)
  0x21          wave left
  0x40 R G B    blink with RGB color

Serial protocol (Pico → ROS):
  K1\\n          button pressed
  K0\\n          button released
"""

import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from geometry_msgs.msg import Twist

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False


# Pre-built command bytes for each LED state
_CMD_RED          = bytes([0x00])
_CMD_GREEN        = bytes([0x01])
_CMD_YELLOW       = bytes([0x02])
_CMD_BLUE         = bytes([0x10, 0x00, 0x00, 0xFF])         # teleop idle
_CMD_WAVE_RIGHT   = bytes([0x10, 0xFF, 0x78, 0x00, 0x20])  # set orange then wave right
_CMD_WAVE_LEFT    = bytes([0x10, 0xFF, 0x78, 0x00, 0x21])  # set orange then wave left
_CMD_PURPLE_BLINK = bytes([0x40, 0x80, 0x00, 0x80])


class PicoBridge(Node):
    def __init__(self):
        super().__init__('pico_bridge')

        self.declare_parameter('serial_port', '/dev/serial/by-id/usb-Raspberry_Pi_Pico_2_XXXX-if00')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('cmd_vel_linear_threshold', 0.05)
        self.declare_parameter('cmd_vel_angular_threshold', 0.1)

        port     = self.get_parameter('serial_port').value
        baud     = self.get_parameter('baud_rate').value
        self._lin_thresh = float(self.get_parameter('cmd_vel_linear_threshold').value)
        self._ang_thresh = float(self.get_parameter('cmd_vel_angular_threshold').value)

        # Robot state
        self._estop      = False
        self._stuck      = False
        self._robot_mode = 'off'
        self._lid_state  = 'unknown'
        self._linear_x   = 0.0
        self._angular_z  = 0.0
        self._last_led_cmd: bytes = b''

        # Serial port
        self._ser = None
        if not HAS_SERIAL:
            self.get_logger().error('pyserial not installed — Pico bridge disabled.')
        else:
            try:
                self._ser = serial.Serial(port, baud, timeout=0)
                self.get_logger().info(f'Pico connected on {port}')
            except Exception as e:
                self.get_logger().error(f'Failed to open {port}: {e}')

        # Publications
        self._kill_switch_pub = self.create_publisher(Bool, '/kill_switch', 10)
        self._lid_cmd_pub     = self.create_publisher(String, '/lid_command', 10)

        # Subscriptions
        self.create_subscription(Bool,   '/estop',       self._estop_cb,   10)
        self.create_subscription(String, '/lid_state',   self._lid_cb,     10)
        self.create_subscription(String, '/robot_mode',  self._mode_cb,    10)
        self.create_subscription(Twist,  '/cmd_vel',     self._vel_cb,     10)
        self.create_subscription(Bool,   '/robot_stuck', self._stuck_cb,   10)

        # Serial read thread
        if self._ser:
            t = threading.Thread(target=self._serial_reader, daemon=True)
            t.start()

        # LED update timer at 10 Hz
        self.create_timer(0.1, self._update_leds)

    # ── Subscription callbacks ─────────────────────────────────────────────────

    def _estop_cb(self, msg: Bool):
        self._estop = msg.data

    def _lid_cb(self, msg: String):
        self._lid_state = msg.data

    def _mode_cb(self, msg: String):
        self._robot_mode = msg.data

    def _vel_cb(self, msg: Twist):
        self._linear_x  = msg.linear.x
        self._angular_z = msg.angular.z

    def _stuck_cb(self, msg: Bool):
        self._stuck = msg.data

    # ── LED state machine ──────────────────────────────────────────────────────

    def _desired_led_cmd(self) -> bytes:
        if self._estop:
            return _CMD_RED

        turning_right = self._angular_z < -self._ang_thresh
        turning_left  = self._angular_z >  self._ang_thresh
        moving        = abs(self._linear_x) > self._lin_thresh

        if turning_right:
            return _CMD_WAVE_RIGHT
        if turning_left:
            return _CMD_WAVE_LEFT
        if moving:
            return _CMD_YELLOW
        if self._stuck:
            return _CMD_PURPLE_BLINK
        if self._robot_mode == 'teleop':
            return _CMD_BLUE
        return _CMD_GREEN

    def _update_leds(self):
        cmd = self._desired_led_cmd()
        if cmd != self._last_led_cmd:
            self._send(cmd)
            self._last_led_cmd = cmd

    # ── Kill switch / lid button logic ─────────────────────────────────────────

    def _handle_button_press(self):
        """GP15 pressed: kill switch while moving, lid toggle while stopped."""
        moving = (abs(self._linear_x)  > self._lin_thresh or
                  abs(self._angular_z) > self._ang_thresh)
        if moving:
            msg = Bool()
            msg.data = True
            self._kill_switch_pub.publish(msg)
            self.get_logger().warn('Kill switch pressed — asserting e-stop.')
        else:
            msg = String()
            msg.data = 'toggle'
            self._lid_cmd_pub.publish(msg)
            self.get_logger().info('Button pressed while stopped — toggling lid.')

    def _handle_button_release(self):
        """GP15 released: clear virtual kill switch."""
        msg = Bool()
        msg.data = False
        self._kill_switch_pub.publish(msg)

    # ── Serial I/O ─────────────────────────────────────────────────────────────

    def _send(self, data: bytes):
        if self._ser and self._ser.is_open:
            try:
                self._ser.write(data)
            except Exception as e:
                self.get_logger().error(f'Serial write error: {e}')

    def _serial_reader(self):
        buf = b''
        while rclpy.ok():
            try:
                chunk = self._ser.read(64)
            except Exception as e:
                self.get_logger().error(f'Serial read error: {e}')
                break
            if not chunk:
                continue
            buf += chunk
            while b'\n' in buf:
                line, buf = buf.split(b'\n', 1)
                self._parse_line(line.strip())

    def _parse_line(self, line: bytes):
        if line == b'K1':
            self._handle_button_press()
        elif line == b'K0':
            self._handle_button_release()
        # Other lines (debug prints from BOOTSEL, etc.) are silently ignored.


def main(args=None):
    rclpy.init(args=args)
    node = PicoBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
