#!/usr/bin/env python3
import json
import math
import serial
import struct
import threading
import os
import time
import pyvesc

from pyvesc.VESC.messages import SetRPM, SetCurrent, SetDutyCycle

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64, String


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


def decode_voltage_from_values_payload(payload: bytes) -> float:
    """Extract input voltage (V) from COMM_GET_VALUES response. Byte 27: int16 /10."""
    if len(payload) < 29:
        raise ValueError(f'payload too short for voltage decode: {len(payload)} bytes')
    if payload[0] != COMM_GET_VALUES:
        raise ValueError(f'unexpected response id: {payload[0]}')
    return struct.unpack_from('>H', payload, 27)[0] / 10.0


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
        self.declare_parameter('erpm_per_wheel_rpm', 500.0)
        self.declare_parameter('max_erpm_step_per_tick', 2000)
        self.declare_parameter('max_erpm', 20000)

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

        # Velocity / yaw PI gains (default 0.0 = open-loop, existing behaviour)
        self.declare_parameter('kp_v', 0.0)
        self.declare_parameter('ki_v', 0.0)
        self.declare_parameter('kp_w', 0.0)
        self.declare_parameter('ki_w', 0.0)
        self.declare_parameter('vesc_integral_max', 2.0)  # m/s integral clamp

        # Gain echo / update topics
        self.declare_parameter('vesc_gains_echo_topic', '/vesc_gains_echo')
        self.declare_parameter('vesc_gains_topic',      '/vesc_gains')
        self.declare_parameter('robot_mode_topic',      '/robot_mode')

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

        # PI gains (live-tunable via /vesc_gains)
        self._kp_v = float(self.get_parameter('kp_v').value)
        self._ki_v = float(self.get_parameter('ki_v').value)
        self._kp_w = float(self.get_parameter('kp_w').value)
        self._ki_w = float(self.get_parameter('ki_w').value)
        self._integral_max = float(self.get_parameter('vesc_integral_max').value)

        # State
        self.estop = False
        self._coasting = False
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
        self.left_voltage = None
        self.right_voltage = None

        # PI state
        self._v_cmd = 0.0
        self._w_cmd = 0.0
        self._integral_v = 0.0
        self._integral_w = 0.0

        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        transient_local = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Subscribers
        self.sub_cmd = self.create_subscription(Twist, self.cmd_vel_topic, self._cmd_reader, qos)
        self.sub_estop = self.create_subscription(Bool, self.estop_topic, self._estop_reader, qos)
        self.create_subscription(String, self.get_parameter('robot_mode_topic').value,
                                 self._mode_cb, transient_local)

        self.declare_parameter('battery_voltage_topic', '/vesc/battery_voltage')
        self.left_wheel_vel_pub = self.create_publisher(Float64, self.left_wheel_vel_topic, 10)
        self.right_wheel_vel_pub = self.create_publisher(Float64, self.right_wheel_vel_topic, 10)
        self.battery_voltage_pub = self.create_publisher(Float64, self.get_parameter('battery_voltage_topic').value, 10)
        self._vesc_gains_echo_pub = self.create_publisher(String, self.get_parameter('vesc_gains_echo_topic').value, 10)
        self.create_subscription(String, self.get_parameter('vesc_gains_topic').value, self._on_vesc_gains, 10)
        self.create_timer(0.5, self._publish_vesc_gains_echo)

        self._shutdown = False
        self._error_since = {'left': None, 'right': None}  # tracks first error time per side

        # Timer for timeout supervision
        self.timer = self.create_timer(0.05, self._tick)

        # Open serial ports before starting feedback threads
        self._open_ports()

        # Feedback: serial I/O in daemon threads; ROS timer only publishes shared state
        feedback_period = 1.0 / max(self.feedback_poll_rate_hz, 1.0)
        if self.publish_wheel_feedback:
            for side in ('left', 'right'):
                t = threading.Thread(
                    target=self._feedback_loop, args=(side,), daemon=True
                )
                t.start()
        self.feedback_timer = self.create_timer(feedback_period, self._publish_feedback)

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

    def _mode_cb(self, msg: String):
        mode = msg.data.strip().lower()
        if mode == 'off':
            if not self._coasting:
                self.get_logger().info('Robot mode → off: coasting wheel motors.')
            self._coasting = True
            self._send_stop()
        else:
            if self._coasting:
                self.get_logger().info(f'Robot mode → {mode}: resuming motor control.')
            self._coasting = False

    def _cmd_reader(self, msg: Twist):
        self.last_cmd_time = self.get_clock().now()
        if self.estop:
            self._v_cmd = 0.0
            self._w_cmd = 0.0
            return
        self._v_cmd = msg.linear.x
        self._w_cmd = msg.angular.z

    def _cmd_to_erpm(self, v: float, w: float) -> tuple:
        """Convert (v m/s, w rad/s) → (left_erpm, right_erpm)."""
        v_left  = v - 0.5 * self.track_width_m * w
        v_right = v + 0.5 * self.track_width_m * w
        wheel_rpm_left  = (v_left  / self.wheel_radius_m) * 60.0 / (2.0 * math.pi)
        wheel_rpm_right = (v_right / self.wheel_radius_m) * 60.0 / (2.0 * math.pi)
        erpm_left  = int(round(wheel_rpm_left  * self.erpm_per_wheel_rpm * self.left_sign))
        erpm_right = int(round(wheel_rpm_right * self.erpm_per_wheel_rpm * self.right_sign))
        erpm_left  = max(-self.max_erpm, min(self.max_erpm, erpm_left))
        erpm_right = max(-self.max_erpm, min(self.max_erpm, erpm_right))
        return erpm_left, erpm_right

    def _tick(self):
        if self._coasting:
            self._send_stop()
            return

        if self.estop:
            self._send_stop()
            self._integral_v = 0.0
            self._integral_w = 0.0
            return

        if self.last_cmd_time is None:
            self._send_erpm(0, 0)
            return

        if (self.get_clock().now() - self.last_cmd_time) > self.cmd_timeout:
            self.get_logger().warn('cmd_vel timeout. Stopping motors.')
            self._send_stop()
            self._integral_v = 0.0
            self._integral_w = 0.0
            return

        # Measured velocities from wheel feedback (already sign-corrected in _feedback_loop)
        v_measured = (self.left_measured_rad_s + self.right_measured_rad_s) * 0.5 * self.wheel_radius_m
        w_measured = (self.right_measured_rad_s - self.left_measured_rad_s) * self.wheel_radius_m / self.track_width_m

        # Velocity PI
        dt = 0.05
        v_err = self._v_cmd - v_measured
        w_err = self._w_cmd - w_measured
        self._integral_v = max(-self._integral_max, min(self._integral_max, self._integral_v + v_err * dt))
        self._integral_w = max(-self._integral_max, min(self._integral_max, self._integral_w + w_err * dt))
        v_eff = self._v_cmd + self._kp_v * v_err + self._ki_v * self._integral_v
        w_eff = self._w_cmd + self._kp_w * w_err + self._ki_w * self._integral_w

        target_left, target_right = self._cmd_to_erpm(v_eff, w_eff)

        self.cmd_left_erpm  = self._slew(self.cmd_left_erpm,  target_left,  self.max_erpm_step_per_tick)
        self.cmd_right_erpm = self._slew(self.cmd_right_erpm, target_right, self.max_erpm_step_per_tick)

        self._send_erpm(self.cmd_left_erpm, self.cmd_right_erpm)

    def _publish_vesc_gains_echo(self):
        v_measured = (self.left_measured_rad_s + self.right_measured_rad_s) * 0.5 * self.wheel_radius_m
        w_measured = (self.right_measured_rad_s - self.left_measured_rad_s) * self.wheel_radius_m / self.track_width_m
        msg = String()
        msg.data = json.dumps({
            'kp_v':          round(self._kp_v, 6),
            'ki_v':          round(self._ki_v, 6),
            'kp_w':          round(self._kp_w, 6),
            'ki_w':          round(self._ki_w, 6),
            'integral_max':  round(self._integral_max, 4),
            '_v_measured':   round(v_measured, 4),
            '_w_measured':   round(w_measured, 4),
        })
        self._vesc_gains_echo_pub.publish(msg)

    def _on_vesc_gains(self, msg: String):
        try:
            gains = json.loads(msg.data)
        except (json.JSONDecodeError, ValueError):
            self.get_logger().warn('vesc_gains: invalid JSON')
            return
        _KEYS = {'kp_v', 'ki_v', 'kp_w', 'ki_w', 'integral_max'}
        for k, v in gains.items():
            if k not in _KEYS:
                continue
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            if k == 'kp_v':         self._kp_v = val
            elif k == 'ki_v':       self._ki_v = val; self._integral_v = 0.0
            elif k == 'kp_w':       self._kp_w = val
            elif k == 'ki_w':       self._ki_w = val; self._integral_w = 0.0
            elif k == 'integral_max': self._integral_max = val
        self.get_logger().info(
            f'vesc_gains updated: kp_v={self._kp_v} ki_v={self._ki_v} '
            f'kp_w={self._kp_w} ki_w={self._ki_w}'
        )

    def _write_erpm(self, ser, erpm: int, side: str):
        if ser is None:
            return

        try:
            if abs(erpm) < 300:
                ser.write(pyvesc.encode(SetDutyCycle(0)))
            else:
                ser.write(pyvesc.encode(SetRPM(erpm)))
            
        except Exception as e:
            self.get_logger().error(
                f'Failed sending SetRPM to {side} VESC: {e}',
                throttle_duration_sec=5.0)

    def _write_stop(self, ser, side: str):
        if ser is None:
            return

        try:
            ser.write(pyvesc.encode(SetRPM(0)))
        except Exception as e:
            self.get_logger().error(
                f'Failed sending stop to {side} VESC: {e}',
                throttle_duration_sec=5.0)

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

    def _read_erpm(self, ser, side: str) -> tuple[int, float] | tuple[None, None]:
        """Returns (erpm, voltage_V) or (None, None) on failure."""
        if ser is None:
            return None, None

        try:
            ser.reset_input_buffer()
            ser.write(get_values_packet())
            payload = self._read_vesc_packet(ser)
            if payload is None:
                return None, None
            erpm = decode_rpm_from_values_payload(payload)
            voltage = decode_voltage_from_values_payload(payload)
            return erpm, voltage
        except Exception as e:
            self.get_logger().error(
                f'Failed reading feedback from {side} VESC: {e}',
                throttle_duration_sec=5.0)
            return None, None

    def _publish_wheel_velocity(self, publisher, value_rad_s: float):
        msg = Float64()
        msg.data = value_rad_s
        publisher.publish(msg)

    def _feedback_loop(self, side: str):
        """Background daemon thread: poll one VESC for ERPM and update shared state."""
        interval = 1.0 / max(self.feedback_poll_rate_hz, 1.0)
        sign = self.left_sign if side == 'left' else self.right_sign
        while not self._shutdown:
            ser = self.left_ser if side == 'left' else self.right_ser
            if ser is not None:
                erpm, voltage = self._read_erpm(ser, side)
                if erpm is not None:
                    self._error_since[side] = None  # clear error streak on success
                    rad_s = (erpm / self.erpm_per_wheel_rpm) * sign * (2.0 * math.pi / 60.0)
                    if side == 'left':
                        self.left_measured_rad_s = rad_s
                        self.left_voltage = voltage
                    else:
                        self.right_measured_rad_s = rad_s
                        self.right_voltage = voltage
                    time.sleep(interval)
                else:
                    now = time.monotonic()
                    if self._error_since[side] is None:
                        self._error_since[side] = now
                    elif now - self._error_since[side] > 30.0:
                        self.get_logger().fatal(
                            f'{side} VESC unreachable for 30 s — shutting down node')
                        os._exit(1)
                    time.sleep(1.0)  # back off after failure — don't spam at poll rate
            else:
                time.sleep(interval)

    def _publish_feedback(self):
        """ROS timer callback: publish pre-computed wheel velocities (no serial I/O)."""
        if not self.publish_wheel_feedback:
            return
        self._publish_wheel_velocity(self.left_wheel_vel_pub, self.left_measured_rad_s)
        self._publish_wheel_velocity(self.right_wheel_vel_pub, self.right_measured_rad_s)

        voltages = [v for v in (self.left_voltage, self.right_voltage) if v is not None]
        if voltages:
            msg = Float64()
            msg.data = min(voltages)  # use minimum (conservative)
            self.battery_voltage_pub.publish(msg)

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
        self._shutdown = True
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
