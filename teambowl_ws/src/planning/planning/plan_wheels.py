#!/usr/bin/env python3

import math

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PointStamped, Twist
from std_msgs.msg import Bool


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def zero_twist() -> Twist:
    msg = Twist()
    msg.linear.x = 0.0
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0
    return msg


class PlanWheels(Node):
    """
    Simple person-follow planner.

    Inputs:
      - /user_pos   (PointStamped in camera optical frame)
      - /user_valid (Bool)

    Output:
      - /cmd_vel_auto (Twist)

    Control:
      - linear.x from forward distance error (z - follow_distance)
      - angular.z from lateral offset x
    """

    def __init__(self):
        super().__init__('plan_wheels')

        # Topics
        self.declare_parameter('target_topic', '/user_pos')
        self.declare_parameter('target_valid_topic', '/user_valid')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_auto')

        # Timing
        self.declare_parameter('target_timeout_s', 0.5)
        self.declare_parameter('publish_rate_hz', 20.0)

        # Follow geometry
        self.declare_parameter('follow_distance_m', 1.5)
        self.declare_parameter('distance_deadband_m', 0.15)
        self.declare_parameter('lateral_deadband_m', 0.10)

        # Control gains
        self.declare_parameter('k_linear', 0.8)
        self.declare_parameter('k_angular', 1.8)

        # Limits
        self.declare_parameter('max_linear_x', 0.2)
        self.declare_parameter('max_angular_z', 0.8)

        # Optional reverse behavior
        self.declare_parameter('allow_reverse', False)
        self.declare_parameter('max_reverse_x', 0.2)

        # Turn-in-place behavior
        self.declare_parameter('turn_in_place_angle_only', False)
        self.declare_parameter('turn_only_lateral_threshold_m', 0.5)

        # Read params
        self.target_topic = self.get_parameter('target_topic').value
        self.target_valid_topic = self.get_parameter('target_valid_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.target_timeout = Duration(
            seconds=float(self.get_parameter('target_timeout_s').value)
        )
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)

        self.follow_distance_m = float(self.get_parameter('follow_distance_m').value)
        self.distance_deadband_m = float(self.get_parameter('distance_deadband_m').value)
        self.lateral_deadband_m = float(self.get_parameter('lateral_deadband_m').value)

        self.k_linear = float(self.get_parameter('k_linear').value)
        self.k_angular = float(self.get_parameter('k_angular').value)

        self.max_linear_x = float(self.get_parameter('max_linear_x').value)
        self.max_angular_z = float(self.get_parameter('max_angular_z').value)

        self.allow_reverse = bool(self.get_parameter('allow_reverse').value)
        self.max_reverse_x = float(self.get_parameter('max_reverse_x').value)

        self.turn_in_place_angle_only = bool(
            self.get_parameter('turn_in_place_angle_only').value
        )
        self.turn_only_lateral_threshold_m = float(
            self.get_parameter('turn_only_lateral_threshold_m').value
        )

        # State
        self.target_valid = False
        self.last_target_time = None
        self.last_target_x = 0.0
        self.last_target_z = 0.0

        # Subs/pubs
        self.target_sub = self.create_subscription(
            PointStamped,
            self.target_topic,
            self._target_cb,
            qos_profile_sensor_data
        )
        self.valid_sub = self.create_subscription(
            Bool,
            self.target_valid_topic,
            self._valid_cb,
            qos_profile_sensor_data
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        # Timer
        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f'plan_wheels started | target={self.target_topic} valid={self.target_valid_topic} '
            f'cmd_out={self.cmd_vel_topic}'
        )

    def _target_cb(self, msg: PointStamped):
        self.last_target_x = float(msg.point.x)
        self.last_target_z = float(msg.point.z)
        self.last_target_time = self.get_clock().now()

    def _valid_cb(self, msg: Bool):
        self.target_valid = bool(msg.data)

    def _target_fresh(self) -> bool:
        if self.last_target_time is None:
            return False
        return (self.get_clock().now() - self.last_target_time) <= self.target_timeout

    def _compute_cmd(self) -> Twist:
        cmd = zero_twist()

        if not self.target_valid:
            return cmd

        if not self._target_fresh():
            return cmd

        x = self.last_target_x
        z = self.last_target_z

        # Safety: nonsense depth
        if z <= 0.0 or not math.isfinite(z) or not math.isfinite(x):
            return cmd

        # Errors
        distance_error = z - self.follow_distance_m
        lateral_error = x

        # Deadbands
        if abs(distance_error) < self.distance_deadband_m:
            distance_error = 0.0
        if abs(lateral_error) < self.lateral_deadband_m:
            lateral_error = 0.0

        # Angular command:
        # If target x > 0 (person is to camera-right), turn right => angular.z negative
        angular_z = -self.k_angular * lateral_error
        angular_z = clamp(angular_z, -self.max_angular_z, self.max_angular_z)

        # Linear command
        linear_x = self.k_linear * distance_error

        if not self.allow_reverse and linear_x < 0.0:
            linear_x = 0.0
        else:
            linear_x = clamp(linear_x, -self.max_reverse_x, self.max_linear_x)

        # Optional: if target is far off-center, turn first before driving much
        if self.turn_in_place_angle_only and abs(lateral_error) > self.turn_only_lateral_threshold_m:
            linear_x = 0.0

        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        return cmd

    def _tick(self):
        cmd = self._compute_cmd()
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = PlanWheels()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
