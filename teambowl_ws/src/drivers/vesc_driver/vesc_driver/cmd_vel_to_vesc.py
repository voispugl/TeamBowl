#!/usr/bin/env python3
import can
import json
import math
import struct
import threading
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64, String

# VESC CAN packet types
_CAN_PACKET_SET_CURRENT = 1
_CAN_PACKET_SET_RPM     = 3
_CAN_PACKET_STATUS      = 9   # auto-broadcast: RPM, current, duty
_CAN_PACKET_STATUS_5    = 27  # auto-broadcast: tachometer, voltage


class CmdVelToVescNode(Node):
    """
    Differential-drive motor driver for two VESC controllers on a shared SocketCAN bus.

    Subscribes to:
      - /cmd_vel         geometry_msgs/Twist
      - /estop           std_msgs/Bool

    Behavior:
      - converts linear.x and angular.z into left/right wheel ERPM
      - sends SET_RPM CAN frames to left and right VESCs
      - sends SET_CURRENT(0) on estop, coast, timeout, and low-speed deadband
        (true zero-torque free-spin, unlike SET_DUTY(0) which regeneratively brakes)
      - reads STATUS_1 and STATUS_5 auto-broadcast frames for RPM and voltage feedback
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
        self.declare_parameter('erpm_per_wheel_rpm', 101.5)
        self.declare_parameter('max_erpm_step_per_tick', 200)
        self.declare_parameter('max_erpm', 2560)

        # Command timeout
        self.declare_parameter('cmd_timeout_s', 0.5)

        # CAN bus
        self.declare_parameter('can_interface', 'can1')
        self.declare_parameter('left_can_id',   14)
        self.declare_parameter('right_can_id',  24)

        # Wheel sign convention
        self.declare_parameter('left_sign', 1)
        self.declare_parameter('right_sign', -1)

        # Velocity / yaw PI gains (default 0.0 = open-loop, existing behaviour)
        self.declare_parameter('kp_v', 0.0)
        self.declare_parameter('ki_v', 0.0)
        self.declare_parameter('kp_w', 0.0)
        self.declare_parameter('ki_w', 0.0)
        self.declare_parameter('vesc_integral_max', 2.0)

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

        self.can_interface = self.get_parameter('can_interface').value
        self.left_can_id   = int(self.get_parameter('left_can_id').value)
        self.right_can_id  = int(self.get_parameter('right_can_id').value)

        self.left_sign  = int(self.get_parameter('left_sign').value)
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
        self.can_bus = None
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
        self.battery_voltage_pub = self.create_publisher(
            Float64, self.get_parameter('battery_voltage_topic').value, 10)
        self._vesc_gains_echo_pub = self.create_publisher(
            String, self.get_parameter('vesc_gains_echo_topic').value, 10)
        self.create_subscription(
            String, self.get_parameter('vesc_gains_topic').value, self._on_vesc_gains, 10)
        self.create_timer(0.5, self._publish_vesc_gains_echo)

        self._shutdown = False
        self._error_since = {'left': None, 'right': None}

        # Timer for timeout supervision
        self.timer = self.create_timer(0.05, self._tick)

        # Open CAN bus before starting feedback threads
        self._open_can()

        # Feedback: CAN receive in daemon threads; ROS timer only publishes shared state
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
            f'can={self.can_interface}, left_id={self.left_can_id}, right_id={self.right_can_id}, '
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

    def _open_can(self):
        try:
            self.can_bus = can.Bus(
                interface='socketcan',
                channel=self.can_interface,
                bitrate=1000000,
            )
            self.get_logger().info(f'Opened CAN bus {self.can_interface}')
        except Exception as e:
            self.can_bus = None
            self.get_logger().error(f'Failed to open CAN bus {self.can_interface}: {e}')

    def _can_send_rpm(self, unit_id: int, erpm: int):
        if self.can_bus is None:
            return
        arb_id = (_CAN_PACKET_SET_RPM << 8) | unit_id
        data = struct.pack('>i', erpm)
        msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=True)
        try:
            self.can_bus.send(msg)
        except Exception as e:
            self.get_logger().error(
                f'CAN send SET_RPM to id {unit_id} failed: {e}',
                throttle_duration_sec=5.0)

    def _can_send_current(self, unit_id: int, milliamps: int):
        """Send SET_CURRENT. Use milliamps=0 for true free-spin coast (zero torque)."""
        if self.can_bus is None:
            return
        arb_id = (_CAN_PACKET_SET_CURRENT << 8) | unit_id
        data = struct.pack('>i', milliamps)
        msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=True)
        try:
            self.can_bus.send(msg)
        except Exception as e:
            self.get_logger().error(
                f'CAN send SET_CURRENT to id {unit_id} failed: {e}',
                throttle_duration_sec=5.0)

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

        v_measured = (self.left_measured_rad_s + self.right_measured_rad_s) * 0.5 * self.wheel_radius_m
        w_measured = (self.right_measured_rad_s - self.left_measured_rad_s) * self.wheel_radius_m / self.track_width_m

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
            if k == 'kp_v':           self._kp_v = val
            elif k == 'ki_v':         self._ki_v = val; self._integral_v = 0.0
            elif k == 'kp_w':         self._kp_w = val
            elif k == 'ki_w':         self._ki_w = val; self._integral_w = 0.0
            elif k == 'integral_max': self._integral_max = val
        self.get_logger().info(
            f'vesc_gains updated: kp_v={self._kp_v} ki_v={self._ki_v} '
            f'kp_w={self._kp_w} ki_w={self._ki_w}'
        )

    def _send_erpm(self, left_erpm: int, right_erpm: int):
        if self.print_RPM_cmds:
            self.get_logger().info(f'send left={left_erpm} right={right_erpm}')

        for unit_id, erpm in ((self.left_can_id, left_erpm), (self.right_can_id, right_erpm)):
            if abs(erpm) < 300:
                # Zero torque (free coast) in the low-speed deadband
                self._can_send_current(unit_id, 0)
            else:
                self._can_send_rpm(unit_id, erpm)

        self.last_left_erpm = left_erpm
        self.last_right_erpm = right_erpm

    def _send_stop(self):
        self.target_left_erpm = 0
        self.target_right_erpm = 0
        self.cmd_left_erpm = 0
        self.cmd_right_erpm = 0
        # SET_CURRENT(0) = zero torque / free coast
        self._can_send_current(self.left_can_id, 0)
        self._can_send_current(self.right_can_id, 0)

    def _feedback_loop(self, side: str):
        """
        Background daemon thread: receive VESC CAN status frames and update shared state.

        The VESC auto-broadcasts STATUS_1 (RPM, current, duty) and STATUS_5 (tachometer,
        voltage) at a rate configured in VESC Tool. This thread filters for frames from
        the relevant unit ID; all other CAN frames (e.g. RobStride) are ignored.
        """
        unit_id = self.left_can_id if side == 'left' else self.right_can_id
        sign = self.left_sign if side == 'left' else self.right_sign

        while not self._shutdown:
            if self.can_bus is None:
                time.sleep(1.0)
                continue

            try:
                msg = self.can_bus.recv(timeout=0.1)
            except Exception as e:
                self.get_logger().error(
                    f'{side} CAN recv error: {e}', throttle_duration_sec=5.0)
                time.sleep(0.1)
                continue

            if msg is None:
                continue  # timeout with no frame

            arb_id = msg.arbitration_id
            frame_unit_id = arb_id & 0xFF
            frame_cmd     = (arb_id >> 8) & 0xFF

            if frame_unit_id != unit_id:
                continue  # not our VESC

            if frame_cmd == _CAN_PACKET_STATUS and len(msg.data) >= 8:
                # bytes 0-3: RPM (int32), bytes 4-5: current*10 (int16), bytes 6-7: duty*1000 (int16)
                erpm = struct.unpack_from('>i', msg.data, 0)[0]
                rad_s = (erpm / self.erpm_per_wheel_rpm) * sign * (2.0 * math.pi / 60.0)
                if side == 'left':
                    self.left_measured_rad_s = rad_s
                else:
                    self.right_measured_rad_s = rad_s
                self._error_since[side] = None

            elif frame_cmd == _CAN_PACKET_STATUS_5 and len(msg.data) >= 6:
                # bytes 0-3: tachometer (int32), bytes 4-5: voltage*10 (int16)
                voltage = struct.unpack_from('>H', msg.data, 4)[0] / 10.0
                if side == 'left':
                    self.left_voltage = voltage
                else:
                    self.right_voltage = voltage

    def _publish_wheel_velocity(self, publisher, value_rad_s: float):
        msg = Float64()
        msg.data = value_rad_s
        publisher.publish(msg)

    def _publish_feedback(self):
        """ROS timer callback: publish pre-computed wheel velocities (no CAN I/O)."""
        if not self.publish_wheel_feedback:
            return
        self._publish_wheel_velocity(self.left_wheel_vel_pub, self.left_measured_rad_s)
        self._publish_wheel_velocity(self.right_wheel_vel_pub, self.right_measured_rad_s)

        voltages = [v for v in (self.left_voltage, self.right_voltage) if v is not None]
        if voltages:
            msg = Float64()
            msg.data = min(voltages)  # conservative: use minimum
            self.battery_voltage_pub.publish(msg)

    def destroy_node(self):
        self._shutdown = True
        try:
            self._send_stop()
        except Exception:
            pass

        try:
            if self.can_bus is not None:
                self.can_bus.shutdown()
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
