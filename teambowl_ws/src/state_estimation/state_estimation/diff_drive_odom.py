#!/usr/bin/env python3

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster


def yaw_to_quaternion(yaw: float) -> Quaternion:
    msg = Quaternion()
    msg.x = 0.0
    msg.y = 0.0
    msg.z = math.sin(0.5 * yaw)
    msg.w = math.cos(0.5 * yaw)
    return msg


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class DiffDriveOdomNode(Node):
    """Local differential-drive odometry with optional IMU yaw-rate fusion."""

    def __init__(self):
        super().__init__('diff_drive_odom')

        self.declare_parameter('left_wheel_vel_topic', '/wheel_vel_left')
        self.declare_parameter('right_wheel_vel_topic', '/wheel_vel_right')
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('cmd_vel_fallback_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('wheel_radius_m', 0.307975)
        self.declare_parameter('track_width_m', 0.5588)
        self.declare_parameter('left_sign', 1.0)
        self.declare_parameter('right_sign', -1.0)
        self.declare_parameter('use_imu_yaw_rate', True)
        self.declare_parameter('imu_timeout_s', 0.25)
        self.declare_parameter('wheel_timeout_s', 0.25)
        self.declare_parameter('use_cmd_vel_fallback', True)
        self.declare_parameter('cmd_vel_timeout_s', 0.25)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter(
            'pose_covariance_diagonal',
            [0.05, 0.05, 99999.0, 99999.0, 99999.0, 0.1],
        )
        self.declare_parameter(
            'twist_covariance_diagonal',
            [0.02, 0.02, 99999.0, 99999.0, 99999.0, 0.05],
        )

        self.left_wheel_vel_topic = self.get_parameter('left_wheel_vel_topic').value
        self.right_wheel_vel_topic = self.get_parameter('right_wheel_vel_topic').value
        self.imu_topic = self.get_parameter('imu_topic').value
        self.cmd_vel_fallback_topic = self.get_parameter('cmd_vel_fallback_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.wheel_radius_m = float(self.get_parameter('wheel_radius_m').value)
        self.track_width_m = float(self.get_parameter('track_width_m').value)
        self.left_sign = float(self.get_parameter('left_sign').value)
        self.right_sign = float(self.get_parameter('right_sign').value)
        self.use_imu_yaw_rate = bool(self.get_parameter('use_imu_yaw_rate').value)
        self.imu_timeout = Duration(seconds=float(self.get_parameter('imu_timeout_s').value))
        self.wheel_timeout = Duration(seconds=float(self.get_parameter('wheel_timeout_s').value))
        self.use_cmd_vel_fallback = bool(self.get_parameter('use_cmd_vel_fallback').value)
        self.cmd_vel_timeout = Duration(
            seconds=float(self.get_parameter('cmd_vel_timeout_s').value)
        )
        self.publish_tf = bool(self.get_parameter('publish_tf').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)

        pose_diag = list(self.get_parameter('pose_covariance_diagonal').value)
        twist_diag = list(self.get_parameter('twist_covariance_diagonal').value)
        self.pose_covariance = [0.0] * 36
        self.twist_covariance = [0.0] * 36
        for i, value in enumerate(pose_diag[:6]):
            self.pose_covariance[i * 6 + i] = float(value)
        for i, value in enumerate(twist_diag[:6]):
            self.twist_covariance[i * 6 + i] = float(value)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_update_time = self.get_clock().now()

        self.left_wheel_rad_s = 0.0
        self.right_wheel_rad_s = 0.0
        self.last_left_time: Optional[rclpy.time.Time] = None
        self.last_right_time: Optional[rclpy.time.Time] = None

        self.imu_yaw_rate = 0.0
        self.last_imu_time: Optional[rclpy.time.Time] = None

        self.cmd_fallback_vx = 0.0
        self.cmd_fallback_wz = 0.0
        self.last_cmd_time: Optional[rclpy.time.Time] = None

        self.create_subscription(Float64, self.left_wheel_vel_topic, self._on_left_wheel, 10)
        self.create_subscription(Float64, self.right_wheel_vel_topic, self._on_right_wheel, 10)
        self.create_subscription(Imu, self.imu_topic, self._on_imu, 20)
        self.create_subscription(Twist, self.cmd_vel_fallback_topic, self._on_cmd_vel, 10)

        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            'diff_drive_odom up. '
            f'left={self.left_wheel_vel_topic}, right={self.right_wheel_vel_topic}, '
            f'imu={self.imu_topic}, fallback={self.cmd_vel_fallback_topic}, '
            f'odom={self.odom_topic}, frames={self.odom_frame}->{self.base_frame}'
        )

    def _on_left_wheel(self, msg: Float64):
        self.left_wheel_rad_s = float(msg.data) * self.left_sign
        self.last_left_time = self.get_clock().now()

    def _on_right_wheel(self, msg: Float64):
        self.right_wheel_rad_s = float(msg.data) * self.right_sign
        self.last_right_time = self.get_clock().now()

    def _on_imu(self, msg: Imu):
        self.imu_yaw_rate = float(msg.angular_velocity.z)
        self.last_imu_time = self.get_clock().now()

    def _on_cmd_vel(self, msg: Twist):
        self.cmd_fallback_vx = float(msg.linear.x)
        self.cmd_fallback_wz = float(msg.angular.z)
        self.last_cmd_time = self.get_clock().now()

    def _fresh(self, stamp, timeout: Duration) -> bool:
        if stamp is None:
            return False
        return (self.get_clock().now() - stamp) <= timeout

    def _have_wheel_feedback(self) -> bool:
        return self._fresh(self.last_left_time, self.wheel_timeout) and self._fresh(
            self.last_right_time, self.wheel_timeout
        )

    def _have_imu(self) -> bool:
        return self.use_imu_yaw_rate and self._fresh(self.last_imu_time, self.imu_timeout)

    def _have_cmd_fallback(self) -> bool:
        return self.use_cmd_vel_fallback and self._fresh(self.last_cmd_time, self.cmd_vel_timeout)

    def _compute_body_twist(self) -> tuple[float, float]:
        if self._have_wheel_feedback():
            v_left = self.left_wheel_rad_s * self.wheel_radius_m
            v_right = self.right_wheel_rad_s * self.wheel_radius_m
            linear_x = 0.5 * (v_left + v_right)
            angular_z = (v_right - v_left) / self.track_width_m
            if self._have_imu():
                angular_z = self.imu_yaw_rate
            return linear_x, angular_z

        if self._have_cmd_fallback():
            return self.cmd_fallback_vx, self.cmd_fallback_wz

        return 0.0, 0.0

    def _tick(self):
        now = self.get_clock().now()
        dt = (now - self.last_update_time).nanoseconds / 1e9
        self.last_update_time = now

        if dt <= 0.0:
            return

        linear_x, angular_z = self._compute_body_twist()

        self.yaw = normalize_angle(self.yaw + angular_z * dt)
        self.x += linear_x * math.cos(self.yaw) * dt
        self.y += linear_x * math.sin(self.yaw) * dt

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = yaw_to_quaternion(self.yaw)
        odom.pose.covariance = self.pose_covariance
        odom.twist.twist.linear.x = linear_x
        odom.twist.twist.angular.z = angular_z
        odom.twist.covariance = self.twist_covariance
        self.odom_pub.publish(odom)

        if self.tf_broadcaster is not None:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = odom.header.stamp
            tf_msg.header.frame_id = self.odom_frame
            tf_msg.child_frame_id = self.base_frame
            tf_msg.transform.translation.x = self.x
            tf_msg.transform.translation.y = self.y
            tf_msg.transform.translation.z = 0.0
            tf_msg.transform.rotation = odom.pose.pose.orientation
            self.tf_broadcaster.sendTransform(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DiffDriveOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
