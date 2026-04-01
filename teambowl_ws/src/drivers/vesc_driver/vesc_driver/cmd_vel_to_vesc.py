#!/usr/bin/env python3
import math
import serial
import struct
import pyvesc

from pyvesc.VESC.messages import SetRPM, SetCurrent, SetDutyCycle

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64


COMM_GET_VALUES = 4


def crc16_ccitt(data: bytes, poly: int = 0x1021, init: int = 0x0000) -> int:
    crc = init
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def vesc_packet(payload: bytes) -> bytes:
    length = len(payload)
    if length < 256:
        header = bytes([2, length])
    else:
        header = bytes([3, (length >> 8) & 0xFF, length & 0xFF])
    crc = crc16_ccitt(payload)
    return header + payload + bytes([(crc >> 8) & 0xFF, crc & 0xFF, 3])


def get_values_packet() -> bytes:
    return vesc_packet(bytes([COMM_GET_VALUES]))


def decode_rpm_from_values_payload(payload: bytes) -> int:
    if len(payload) < 27:
        raise ValueError(f'payload too short for COMM_GET_VALUES: {len(payload)} bytes')
    if payload[0] != COMM_GET_VALUES:
        raise ValueError(f'unexpected response id: {payload[0]}')
    return struct.unpack_from('>i', payload, 23)[0]


class CmdVelToVescNode(Node):
    """
    Differential-drive motor driver for two USB VESCs.

    Subscribes to:
      - /cmd_vel         geometry_msgs/Twist
      - /estop           std_msgs/Bool

    Behavior:
      - converts linear.x and angular.z into left/right wheel RPM
      - converts wheel RPM into VESC ERPM using erpm_per_wheel_rpm
      - sends SetRPM(...) to left and right VESCs
      - sends zero on estop, timeout, and shutdown
    """

    def __init__(self):
        super().__init__('cmd_vel_to_vesc')

        # Topics
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('estop_topic', '/estop')

        # Robot geometry
        self.declare_parameter('wheel_radius_m', 0.307975)
        self.declare_parameter('track_width_m', 0.5588)

        # Conversion / limits
        self.declare_parameter('erpm_per_wheel_rpm', 99.0)
        self.declare_parameter('max_erpm_step_per_tick', 200)
        self.declare_parameter('max_erpm', 2560)

        # Command timeout
        self.declare_parameter('cmd_timeout_s', 0.5)

        # Serial ports
        self.declare_parameter('left_port', '/dev/ttyACM0')
        self.declare_parameter('right_port', '/dev/ttyACM1')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('serial_timeout_s', 0.05)

        # Wheel sign convention
        self.declare_parameter('left_sign', 1)
        self.declare_parameter('right_sign', -1)

        # Debugging options
        self.declare_parameter('print_RPM_cmds', False)
        self.declare_parameter('left_wheel_vel_topic', '/wheel_vel_left')
        self.declare_parameter('right_wheel_vel_topic', '/wheel_vel_right')
        self.declare_parameter('feedback_poll_rate_hz', 20.0)
        self.declare_parameter('publish_wheel_feedback', True)

        # Read parameters
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.estop_topic = self.get_parameter('estop_topic').value

        self.wheel_radius_m = float(self.get_parameter('wheel_radius_m').value)
        self.track_width_m = float(self.get_parameter('track_width_m').value)

        self.erpm_per_wheel_rpm = float(self.get_parameter('erpm_per_wheel_rpm').value)
        self.max_erpm_step_per_tick = int(self.get_parameter('max_erpm_step_per_tick').value)
        self.max_erpm = int(self.get_parameter('max_erpm').value)

        self.cmd_timeout = Duration(seconds=float(self.get_parameter('cmd_timeout_s').value))

        self.left_port = self.get_parameter('left_port').value
        self.right_port = self.get_parameter('right_port').value
        self.baud = int(self.get_parameter('baud').value)
        self.serial_timeout_s = float(self.get_parameter('serial_timeout_s').value)

        self.left_sign = int(self.get_parameter('left_sign').value)
        self.right_sign = int(self.get_parameter('right_sign').value)

        self.print_RPM_cmds = bool(self.get_parameter('print_RPM_cmds').value)
        self.left_wheel_vel_topic = self.get_parameter('left_wheel_vel_topic').value
        self.right_wheel_vel_topic = self.get_parameter('right_wheel_vel_topic').value
        self.feedback_poll_rate_hz = float(self.get_parameter('feedback_poll_rate_hz').value)
        self.publish_wheel_feedback = bool(self.get_parameter('publish_wheel_feedback').value)

        # State
        self.estop = False
        self.last_cmd_time = None
        self.left_ser = None
        self.right_ser = None
        self.last_left_erpm = None
        self.last_right_erpm = None

        self.target_left_erpm = 0
        self.target_right_erpm = 0
        self.cmd_left_erpm = 0
        self.cmd_right_erpm = 0
        self.left_measured_rad_s = 0.0
        self.right_measured_rad_s = 0.0

        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers
        self.sub_cmd = self.create_subscription(Twist, self.cmd_vel_topic, self._cmd_reader, qos)
        self.sub_estop = self.create_subscription(Bool, self.estop_topic, self._estop_reader, qos)

        self.left_wheel_vel_pub = self.create_publisher(Float64, self.left_wheel_vel_topic, 10)
        self.right_wheel_vel_pub = self.create_publisher(Float64, self.right_wheel_vel_topic, 10)

        # Timer for timeout supervision
        self.timer = self.create_timer(0.05, self._tick)
        feedback_period = 1.0 / max(self.feedback_poll_rate_hz, 1.0)
        self.feedback_timer = self.create_timer(feedback_period, self._poll_feedback)

        # Open serial ports
        self._open_ports()

        self.get_logger().info(
            f'CmdVelToVesc up. cmd_vel={self.cmd_vel_topic}, estop={self.estop_topic}, '
            f'left_port={self.left_port}, right_port={self.right_port}, '
            f'wheel_radius={self.wheel_radius_m}, track_width={self.track_width_m}, '
            f'erpm_per_wheel_rpm={self.erpm_per_wheel_rpm}, max_erpm={self.max_erpm}, '
            f'wheel_feedback={self.publish_wheel_feedback}'
        )

    def _slew(self, current: int, target: int, step: int) -> int:
        if target > current + step:
            return current + step
        if target < current - step:
            return current - step
        return target

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
            self.target_left_erpm = 0
            self.target_right_erpm = 0
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

        # Clamp to the configured wheel ERPM safety cap.
        erpm_left = max(-self.max_erpm, min(self.max_erpm, erpm_left))
        erpm_right = max(-self.max_erpm, min(self.max_erpm, erpm_right))

        self.target_left_erpm = erpm_left
        self.target_right_erpm = erpm_right

    def _tick(self):
        if self.estop:
            self._send_stop()
            return

        if self.last_cmd_time is None:
            self.target_left_erpm = 0
            self.target_right_erpm = 0
            self._send_erpm(0, 0)
            return

        if (self.get_clock().now() - self.last_cmd_time) > self.cmd_timeout:
            self.get_logger().warn('cmd_vel timeout. Stopping motors.')
            self._send_stop()
            return
        
        self.cmd_left_erpm = self._slew(
            self.cmd_left_erpm,
            self.target_left_erpm,
            self.max_erpm_step_per_tick
        )
        self.cmd_right_erpm = self._slew(
            self.cmd_right_erpm,
            self.target_right_erpm,
            self.max_erpm_step_per_tick
        )

        self._send_erpm(self.cmd_left_erpm, self.cmd_right_erpm)

    def _write_erpm(self, ser, erpm: int, side: str):
        if ser is None:
            return

        try:
            if abs(erpm) < 300:
                ser.write(pyvesc.encode(SetDutyCycle(0)))
            else:
                ser.write(pyvesc.encode(SetRPM(erpm)))
            
        except Exception as e:
            self.get_logger().error(f'Failed sending SetRPM to {side} VESC: {e}')

    def _write_stop(self, ser, side: str):
        if ser is None:
            return

        try:
            ser.write(pyvesc.encode(SetRPM(0)))
        except Exception as e:
            self.get_logger().error(f'Failed sending stop to {side} VESC: {e}')

    def _read_vesc_packet(self, ser) -> bytes | None:
        if ser is None:
            return None

        start = ser.read(1)
        if not start:
            return None

        start_byte = start[0]
        if start_byte == 2:
            length_bytes = ser.read(1)
            if len(length_bytes) != 1:
                return None
            payload_len = length_bytes[0]
        elif start_byte == 3:
            length_bytes = ser.read(2)
            if len(length_bytes) != 2:
                return None
            payload_len = (length_bytes[0] << 8) | length_bytes[1]
        else:
            return None

        payload = ser.read(payload_len)
        crc = ser.read(2)
        end = ser.read(1)
        if len(payload) != payload_len or len(crc) != 2 or len(end) != 1:
            return None
        if end[0] != 3:
            return None

        expected_crc = crc16_ccitt(payload)
        actual_crc = (crc[0] << 8) | crc[1]
        if expected_crc != actual_crc:
            raise ValueError(
                f'CRC mismatch: expected 0x{expected_crc:04X}, got 0x{actual_crc:04X}'
            )
        return payload

    def _read_erpm(self, ser, side: str) -> int | None:
        if ser is None:
            return None

        try:
            ser.reset_input_buffer()
            ser.write(get_values_packet())
            payload = self._read_vesc_packet(ser)
            if payload is None:
                return None
            return decode_rpm_from_values_payload(payload)
        except Exception as e:
            self.get_logger().error(f'Failed reading feedback from {side} VESC: {e}')
            return None

    def _publish_wheel_velocity(self, publisher, value_rad_s: float):
        msg = Float64()
        msg.data = value_rad_s
        publisher.publish(msg)

    def _poll_feedback(self):
        if not self.publish_wheel_feedback:
            return

        left_erpm = self._read_erpm(self.left_ser, 'left')
        right_erpm = self._read_erpm(self.right_ser, 'right')

        if left_erpm is not None:
            left_wheel_rpm = left_erpm / self.erpm_per_wheel_rpm
            left_wheel_rpm *= self.left_sign
            self.left_measured_rad_s = left_wheel_rpm * (2.0 * math.pi / 60.0)
            self._publish_wheel_velocity(self.left_wheel_vel_pub, self.left_measured_rad_s)

        if right_erpm is not None:
            right_wheel_rpm = right_erpm / self.erpm_per_wheel_rpm
            right_wheel_rpm *= self.right_sign
            self.right_measured_rad_s = right_wheel_rpm * (2.0 * math.pi / 60.0)
            self._publish_wheel_velocity(self.right_wheel_vel_pub, self.right_measured_rad_s)

    def _send_erpm(self, left_erpm: int, right_erpm: int):
        if self.print_RPM_cmds:
            self.get_logger().info(f"send left={left_erpm} right={right_erpm}")
        self._write_erpm(self.left_ser, left_erpm, 'left')
        self._write_erpm(self.right_ser, right_erpm, 'right')

        self.last_left_erpm = left_erpm
        self.last_right_erpm = right_erpm

    def _send_stop(self):
        self.target_left_erpm = 0
        self.target_right_erpm = 0
        self.cmd_left_erpm = 0
        self.cmd_right_erpm = 0
        self._write_erpm(self.left_ser, 0, 'left')
        self._write_erpm(self.right_ser, 0, 'right')

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
