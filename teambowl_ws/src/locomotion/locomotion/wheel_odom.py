#!/usr/bin/env python3
"""
Wheel Odometry Node

Converts /cmd_vel (the wheel velocity command sent to VESCs) into a
nav_msgs/Odometry message on /odom_wheels. This is used as the wheel
odometry input to the robot_localization EKF.

Note: This is command-based dead reckoning (not encoder feedback), which
is adequate for EKF fusion when combined with IMU data. The EKF will
weight this against the IMU orientation and produce /odometry/filtered.

Subscribes:
  /cmd_vel  (geometry_msgs/Twist) — velocity command sent to VESCs

Publishes:
  /odom_wheels  (nav_msgs/Odometry) — dead-reckoning odometry from cmd_vel
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import tf2_ros


class WheelOdomNode(Node):
    """
    Dead-reckoning odometry from wheel velocity commands.

    Integrates cmd_vel to produce pose + velocity estimates.
    Publishes nav_msgs/Odometry on /odom_wheels for EKF fusion.
    """

    def __init__(self):
        super().__init__('wheel_odom')

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom_wheels')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('cmd_timeout_s', 0.5)
        self.declare_parameter('publish_tf', False)  # EKF publishes tf; avoid duplicate

        self._cmd_topic = self.get_parameter('cmd_vel_topic').value
        self._odom_topic = self.get_parameter('odom_topic').value
        self._odom_frame = self.get_parameter('odom_frame').value
        self._base_frame = self.get_parameter('base_frame').value
        self._cmd_timeout = Duration(
            seconds=float(self.get_parameter('cmd_timeout_s').value)
        )
        self._publish_tf = self.get_parameter('publish_tf').value

        # Integrated pose state
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0

        # Last velocity command
        self._v = 0.0
        self._omega = 0.0
        self._last_cmd_time = None
        self._last_update_time = self.get_clock().now()

        # Publisher
        self._odom_pub = self.create_publisher(Odometry, self._odom_topic, 10)

        if self._publish_tf:
            self._tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Subscriber
        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Twist, self._cmd_topic, self._on_cmd_vel, best_effort)

        # Update timer at 50 Hz
        self.create_timer(0.02, self._update)

        self.get_logger().info(
            f'WheelOdom up. cmd_vel={self._cmd_topic} → odom={self._odom_topic}'
        )

    def _on_cmd_vel(self, msg: Twist):
        self._v = msg.linear.x
        self._omega = msg.angular.z
        self._last_cmd_time = self.get_clock().now()

    def _update(self):
        now = self.get_clock().now()

        # Zero velocity on timeout
        if self._last_cmd_time is not None:
            if (now - self._last_cmd_time) > self._cmd_timeout:
                self._v = 0.0
                self._omega = 0.0

        # Compute dt
        dt = (now - self._last_update_time).nanoseconds * 1e-9
        self._last_update_time = now

        if dt <= 0.0 or dt > 1.0:
            return

        # Integrate pose (midpoint integration)
        d = self._v * dt
        self._x += d * math.cos(self._yaw + 0.5 * self._omega * dt)
        self._y += d * math.sin(self._yaw + 0.5 * self._omega * dt)
        self._yaw += self._omega * dt

        # Wrap yaw to [-pi, pi]
        self._yaw = math.atan2(math.sin(self._yaw), math.cos(self._yaw))

        # Build quaternion from yaw (2D: roll=pitch=0)
        qw = math.cos(self._yaw * 0.5)
        qx = 0.0
        qy = 0.0
        qz = math.sin(self._yaw * 0.5)

        # Publish odometry
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame

        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        # Pose covariance (diagonal, 6x6 row-major)
        # Conservative: position uncertainty grows with motion
        pos_cov = 0.05 ** 2  # 5 cm std dev
        yaw_cov = 0.05 ** 2  # ~3 deg std dev
        odom.pose.covariance[0] = pos_cov   # x
        odom.pose.covariance[7] = pos_cov   # y
        odom.pose.covariance[14] = 9999.0   # z (not observable)
        odom.pose.covariance[21] = 9999.0   # roll (not observable)
        odom.pose.covariance[28] = 9999.0   # pitch (not observable)
        odom.pose.covariance[35] = yaw_cov  # yaw

        odom.twist.twist.linear.x = self._v
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.linear.z = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = self._omega

        # Twist covariance
        vel_cov = 0.1 ** 2   # 10 cm/s std dev
        omega_cov = 0.05 ** 2
        odom.twist.covariance[0] = vel_cov     # vx
        odom.twist.covariance[7] = 9999.0      # vy (not observable)
        odom.twist.covariance[14] = 9999.0     # vz
        odom.twist.covariance[21] = 9999.0     # roll rate
        odom.twist.covariance[28] = 9999.0     # pitch rate
        odom.twist.covariance[35] = omega_cov  # yaw rate

        self._odom_pub.publish(odom)

        if self._publish_tf:
            tf = TransformStamped()
            tf.header.stamp = now.to_msg()
            tf.header.frame_id = self._odom_frame
            tf.child_frame_id = self._base_frame
            tf.transform.translation.x = self._x
            tf.transform.translation.y = self._y
            tf.transform.translation.z = 0.0
            tf.transform.rotation.x = qx
            tf.transform.rotation.y = qy
            tf.transform.rotation.z = qz
            tf.transform.rotation.w = qw
            self._tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = WheelOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
