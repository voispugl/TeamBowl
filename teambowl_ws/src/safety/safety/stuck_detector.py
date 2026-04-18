#!/usr/bin/env python3
"""
stuck_detector — detects two failure modes and publishes /robot_stuck (Bool).

Stall:     wheels commanded to move but actual wheel velocity is too low
           (robot blocked by obstacle, motor can't turn).

Free-spin: wheels ARE spinning at commanded speed but IMU shows no body
           acceleration (robot lifted off ground, wheels on ice/cart).

Both are "stuck" — the robot is going nowhere useful.
"""

import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu


class StuckDetector(Node):
    def __init__(self):
        super().__init__('stuck_detector')

        self.declare_parameter('wheel_radius_m',      0.153988)
        self.declare_parameter('cmd_threshold_ms',    0.05)   # m/s — "trying to move"
        self.declare_parameter('slip_ratio',          0.2)    # stall: measured < 20% of cmd
        self.declare_parameter('accel_threshold_ms2', 0.3)    # m/s² — free-spin threshold
        self.declare_parameter('stuck_timeout_s',     2.0)
        self.declare_parameter('publish_rate_hz',     10.0)

        self._wheel_radius   = self.get_parameter('wheel_radius_m').value
        self._cmd_thresh     = self.get_parameter('cmd_threshold_ms').value
        self._slip_ratio     = self.get_parameter('slip_ratio').value
        self._accel_thresh   = self.get_parameter('accel_threshold_ms2').value
        self._timeout        = self.get_parameter('stuck_timeout_s').value
        rate                 = self.get_parameter('publish_rate_hz').value

        # State
        self._cmd_linear_x   = 0.0
        self._cmd_angular_z  = 0.0
        self._wheel_vel_left  = 0.0   # rad/s
        self._wheel_vel_right = 0.0   # rad/s
        self._imu_accel_x    = 0.0   # m/s²
        self._estop          = False
        self._stuck_accum    = 0.0   # seconds the stuck condition has been active

        self._pub = self.create_publisher(Bool, '/robot_stuck', 10)

        self.create_subscription(Twist,   '/cmd_vel',          self._cmd_cb,        10)
        self.create_subscription(Float64, '/wheel_vel_left',   self._wvl_cb,        10)
        self.create_subscription(Float64, '/wheel_vel_right',  self._wvr_cb,        10)
        self.create_subscription(Imu,     '/imu/data',         self._imu_cb,        10)
        self.create_subscription(Bool,    '/estop',            self._estop_cb,      10)

        self._dt = 1.0 / max(rate, 1.0)
        self.create_timer(self._dt, self._tick)

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def _cmd_cb(self, msg: Twist):
        self._cmd_linear_x  = msg.linear.x
        self._cmd_angular_z = msg.angular.z

    def _wvl_cb(self, msg: Float64):
        self._wheel_vel_left = msg.data

    def _wvr_cb(self, msg: Float64):
        self._wheel_vel_right = msg.data

    def _imu_cb(self, msg: Imu):
        self._imu_accel_x = msg.linear_acceleration.x

    def _estop_cb(self, msg: Bool):
        self._estop = msg.data

    # ── Detection logic ────────────────────────────────────────────────────────

    def _tick(self):
        # Stopped ≠ stuck — clear immediately on estop
        if self._estop:
            self._stuck_accum = 0.0
            self._publish(False)
            return

        # Convert wheel velocity (rad/s) to expected body speed (m/s)
        v_meas = (abs(self._wheel_vel_left) + abs(self._wheel_vel_right)) / 2.0 \
                 * self._wheel_radius

        # Commanded body speed magnitude
        v_cmd = abs(self._cmd_linear_x)

        trying_to_move = v_cmd > self._cmd_thresh

        # Stall: commanded but wheels barely turning
        stall = trying_to_move and (v_meas < v_cmd * self._slip_ratio)

        # Free-spin: wheels turning but body not accelerating
        freespin = (v_meas > self._cmd_thresh) and \
                   (abs(self._imu_accel_x) < self._accel_thresh)

        if stall or freespin:
            self._stuck_accum += self._dt
        else:
            self._stuck_accum = 0.0

        stuck = self._stuck_accum >= self._timeout
        self._publish(stuck)

    def _publish(self, stuck: bool):
        msg = Bool()
        msg.data = stuck
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = StuckDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
