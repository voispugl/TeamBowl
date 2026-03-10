#!/usr/bin/env python3
import math
import serial
import pyvesc

from pyvesc.VESC.messages import SetRPM, SetCurrent

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class CmdVelToVescNode(Node):
    """
    Differential-drive motor driver for two USB VESCs.

    Subscribes:
      - /cmd_vel         geometry_msgs/Twist
      - /estop           std_msgs/Bool

    Behavior:
      - converts linear.x and angular.z into left/right wheel RPM
      - converts wheel RPM into VESC ERPM using erpm_per_wheel_rpm
      - sends SetRPM(...) to left and right VESCs
      - sends zero current on estop, timeout, and shutdown
    """

    def __init__(self):
        super().__init__('cmd_vel_to_vesc')

        # Topics
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('estop_topic', '/estop')

        # Robot geometry
        self.declare_parameter('wheel_radius_m', 0.10)
        self.declare_parameter('track_width_m', 0.45)

        # Conversion / limits
        self.declare_parameter('erpm_per_wheel_rpm', 1.0)
        self.declare_parameter('max_erpm', 30000)

        # Command timeout
        self.declare_parameter('cmd_timeout_s', 0.5)

        # Serial ports
        self.declare_parameter('left_port', '/dev/ttyACM0')
        self.declare_parameter('right_port', '/dev/ttyACM1')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('serial_timeout_s', 0.05)

        # Wheel sign convention
        # Set one of these to -1 if that motor is mounted reversed
        self.declare_parameter('left_sign', 1)
        self.declare_parameter('right_sign', 1)

        # Read parameters
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.estop_topic = self.get_parameter('estop_topic').value

        self.wheel_radius_m = float(self.get_parameter('wheel_radius_m').value)
        self.track_width_m = float(self.get_parameter('track_width_m').value)

        self.erpm_per_wheel_rpm = float(self.get_parameter('erpm_per_wheel_rpm').value)
        self.max_erpm = int(self.get_parameter('max_erpm').value)

        self.cmd_timeout = Duration(seconds=float(self.get_parameter('cmd_timeout_s').value))

        self.left_port = self.get_parameter('left_port').value
        self.right_port = self.get_parameter('right_port').value
        self.baud = int(self.get_parameter('baud').value)
        self.serial_timeout_s = float(self.get_parameter('serial_timeout_s').value)

        self.left_sign = int(self.get_parameter('left_sign').value)
        self.right_sign = int(self.get_parameter('right_sign').value)

        # State
        self.estop = False
        self.last_cmd_time = None
        self.left_ser = None
        self.right_ser = None
        self.last_left_erpm = None
        self.last_right_erpm = None

        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribers
        self.sub_cmd = self.create_subscription(Twist, self.cmd_vel_topic, self._cmd_reader, qos)
        self.sub_estop = self.create_subscription(Bool, self.estop_topic, self._estop_reader, qos)

        # Timer for timeout supervision
        self.timer = self.create_timer(0.05, self._tick)

        # Open serial ports
        self._open_ports()

        self.get_logger().info(
            f'CmdVelToVesc up. cmd_vel={self.cmd_vel_topic}, estop={self.estop_topic}, '
            f'left_port={self.left_port}, right_port={self.right_port}, '
            f'wheel_radius={self.wheel_radius_m}, track_width={self.track_width_m}, '
            f'erpm_per_wheel_rpm={self.erpm_per_wheel_rpm}, max_erpm={self.max_erpm}'
        )

    def _open_ports(self):
        try:
            self.left_ser = serial.Serial(
                self.left_port,
                baudrate=self.baud,
                timeout=self.serial_timeout_s
            )
            self.get_logger().info(f'Opened left VESC on {self.left_port}')
        except Exception as e:
            self.left_ser = None
            self.get_logger().error(f'Failed to open left VESC port {self.left_port}: {e}')

        try:
            self.right_ser = serial.Serial(
                self.right_port,
                baudrate=self.baud,
                timeout=self.serial_timeout_s
            )
            self.get_logger().info(f'Opened right VESC on {self.right_port}')
        except Exception as e:
            self.right_ser = None
            self.get_logger().error(f'Failed to open right VESC port {self.right_port}: {e}')

    def _estop_reader(self, msg: Bool):
        new_estop = bool(msg.data)

        if new_estop and not self.estop:
            self.get_logger().warn('E-stop asserted. Stopping motors.')
            self._send_stop()

        self.estop = new_estop

    def _cmd_reader(self, msg: Twist):
        self.last_cmd_time = self.get_clock().now()

        if self.estop:
            self._send_stop()
            return

        v = msg.linear.x
        w = msg.angular.z

        # Differential-drive wheel linear velocities [m/s]
        v_left = v - 0.5 * self.track_width_m * w
        v_right = v + 0.5 * self.track_width_m * w

        # Wheel angular velocity [rad/s]
        omega_left = v_left / self.wheel_radius_m
        omega_right = v_right / self.wheel_radius_m

        # Wheel RPM
        wheel_rpm_left = omega_left * 60.0 / (2.0 * math.pi)
        wheel_rpm_right = omega_right * 60.0 / (2.0 * math.pi)

        # Convert wheel RPM -> VESC ERPM
        erpm_left = int(round(wheel_rpm_left * self.erpm_per_wheel_rpm * self.left_sign))
        erpm_right = int(round(wheel_rpm_right * self.erpm_per_wheel_rpm * self.right_sign))

        # Clamp
        erpm_left = max(-self.max_erpm, min(self.max_erpm, erpm_left))
        erpm_right = max(-self.max_erpm, min(self.max_erpm, erpm_right))

        self._send_erpm(erpm_left, erpm_right)

    def _tick(self):
        if self.estop:
            return

        if self.last_cmd_time is None:
            return

        if (self.get_clock().now() - self.last_cmd_time) > self.cmd_timeout:
            self.get_logger().warn('cmd_vel timeout. Stopping motors.')
            self._send_stop()
            self.last_cmd_time = None

    def _write_erpm(self, ser, erpm: int, side: str):
        if ser is None:
            return

        try:
            ser.write(pyvesc.encode(SetRPM(erpm)))
        except Exception as e:
            self.get_logger().error(f'Failed sending SetRPM to {side} VESC: {e}')

    def _write_stop(self, ser, side: str):
        if ser is None:
            return

        try:
            ser.write(pyvesc.encode(SetCurrent(0)))
        except Exception as e:
            self.get_logger().error(f'Failed sending stop to {side} VESC: {e}')

    def _send_erpm(self, left_erpm: int, right_erpm: int):
        self._write_erpm(self.left_ser, left_erpm, 'left')
        self._write_erpm(self.right_ser, right_erpm, 'right')

        self.last_left_erpm = left_erpm
        self.last_right_erpm = right_erpm

    def _send_stop(self):
        self._write_stop(self.left_ser, 'left')
        self._write_stop(self.right_ser, 'right')

        self.last_left_erpm = 0
        self.last_right_erpm = 0

    def destroy_node(self):
        # Stop motors before shutting down
        try:
            self._send_stop()
        except Exception:
            pass

        try:
            if self.left_ser is not None and self.left_ser.is_open:
                self.left_ser.close()
        except Exception:
            pass

        try:
            if self.right_ser is not None and self.right_ser.is_open:
                self.right_ser.close()
        except Exception:
            pass

        super().destroy_node()


def main():
    rclpy.init()
    node = CmdVelToVescNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
