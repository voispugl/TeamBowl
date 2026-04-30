#!/usr/bin/env python3

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener


def _quat_to_rot_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class FollowGoal(Node):
    def __init__(self):
        super().__init__('follow_goal')

        self.declare_parameter('target_topic', '/user_pos')
        self.declare_parameter('target_valid_topic', '/user_valid')
        self.declare_parameter('target_timeout_s', 0.5)
        self.declare_parameter('valid_holdout_s', 1.0)

        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('transform_timeout_s', 0.1)

        self.declare_parameter('follow_distance_m', 1.5)
        self.declare_parameter('min_goal_distance_m', 0.25)
        self.declare_parameter('goal_lateral_deadband_m', 0.05)
        self.declare_parameter('goal_update_rate_hz', 10.0)
        self.declare_parameter('goal_smoothing_alpha', 0.35)

        self.declare_parameter('search_timeout_s', 3.0)
        self.declare_parameter('search_offset_rad', math.pi / 2)

        self.declare_parameter('user_base_topic', '/user_pos_base_link')
        self.declare_parameter('user_world_topic', '/user_pos_odom')
        self.declare_parameter('goal_topic', '/follow_goal')

        self.target_topic = str(self.get_parameter('target_topic').value)
        self.target_valid_topic = str(self.get_parameter('target_valid_topic').value)
        self.target_timeout = Duration(seconds=float(self.get_parameter('target_timeout_s').value))
        self._valid_holdout = Duration(seconds=float(self.get_parameter('valid_holdout_s').value))

        self.base_frame = str(self.get_parameter('base_frame').value)
        self.world_frame = str(self.get_parameter('world_frame').value)
        self.transform_timeout = Duration(
            seconds=float(self.get_parameter('transform_timeout_s').value)
        )

        self.follow_distance_m = float(self.get_parameter('follow_distance_m').value)
        self.min_goal_distance_m = float(self.get_parameter('min_goal_distance_m').value)
        self.goal_lateral_deadband_m = float(self.get_parameter('goal_lateral_deadband_m').value)
        self.goal_update_rate_hz = float(self.get_parameter('goal_update_rate_hz').value)
        self.goal_smoothing_alpha = float(self.get_parameter('goal_smoothing_alpha').value)
        self._search_timeout = Duration(
            seconds=float(self.get_parameter('search_timeout_s').value)
        )
        self._search_offset_rad = float(self.get_parameter('search_offset_rad').value)

        self.user_base_topic = str(self.get_parameter('user_base_topic').value)
        self.user_world_topic = str(self.get_parameter('user_world_topic').value)
        self.goal_topic = str(self.get_parameter('goal_topic').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.target_valid = False
        self._last_valid_true_time = None  # last time user_valid=True arrived
        self.last_target_msg = None
        self.last_target_time = None
        self.last_smoothed_goal = None
        self._last_tf_warn_ns = 0
        self._search_active = False

        self.target_sub = self.create_subscription(
            PointStamped,
            self.target_topic,
            self._target_cb,
            qos_profile_sensor_data,
        )
        self.valid_sub = self.create_subscription(
            Bool,
            self.target_valid_topic,
            self._valid_cb,
            qos_profile_sensor_data,
        )

        self.user_base_pub = self.create_publisher(PointStamped, self.user_base_topic, 10)
        self.user_world_pub = self.create_publisher(PointStamped, self.user_world_topic, 10)
        self.goal_pub = self.create_publisher(PoseStamped, self.goal_topic, 10)

        period = 1.0 / max(self.goal_update_rate_hz, 1.0)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f'follow_goal started | target={self.target_topic} base={self.base_frame} '
            f'world={self.world_frame} goal={self.goal_topic}'
        )

    def _target_cb(self, msg: PointStamped):
        self.last_target_msg = msg
        self.last_target_time = self.get_clock().now()

    def _valid_cb(self, msg: Bool):
        self.target_valid = bool(msg.data)
        if msg.data:
            self._last_valid_true_time = self.get_clock().now()

    def _target_fresh(self) -> bool:
        if self.last_target_time is None:
            return False
        return (self.get_clock().now() - self.last_target_time) <= self.target_timeout

    def _lookup_transform(self, target_frame: str, source_frame: str, stamp) -> object | None:
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                timeout=self.transform_timeout,
            )
        except TransformException as exc:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_tf_warn_ns > int(2e9):
                self.get_logger().warn(
                    f'Failed transform {source_frame} -> {target_frame}: {exc}'
                )
                self._last_tf_warn_ns = now_ns
            return None

    def _transform_xyz(self, xyz: np.ndarray, transform) -> np.ndarray:
        rot = transform.transform.rotation
        trans = transform.transform.translation
        rot_matrix = _quat_to_rot_matrix(rot.x, rot.y, rot.z, rot.w)
        result = rot_matrix @ xyz
        result[0] += trans.x
        result[1] += trans.y
        result[2] += trans.z
        return result

    def _transform_point_msg(self, msg: PointStamped, target_frame: str) -> PointStamped | None:
        transform = self._lookup_transform(target_frame, msg.header.frame_id, Time())
        if transform is None:
            return None
        xyz = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float64)
        xyz_out = self._transform_xyz(xyz, transform)

        out = PointStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = target_frame
        out.point.x = float(xyz_out[0])
        out.point.y = float(xyz_out[1])
        out.point.z = float(xyz_out[2])
        return out

    def _compute_goal_in_base(self, user_base: PointStamped) -> np.ndarray | None:
        user_xy = np.array([user_base.point.x, user_base.point.y], dtype=np.float64)
        distance = float(np.linalg.norm(user_xy))
        if not math.isfinite(distance) or distance < 0.05:
            return None

        goal_xy = user_xy * (distance - self.follow_distance_m) / distance

        if abs(goal_xy[1]) < self.goal_lateral_deadband_m:
            goal_xy[1] = 0.0

        return np.array([goal_xy[0], goal_xy[1], 0.0], dtype=np.float64)

    def _smooth_goal(self, goal_xy_world: np.ndarray) -> np.ndarray:
        if self.last_smoothed_goal is None:
            self.last_smoothed_goal = goal_xy_world.copy()
            return goal_xy_world

        alpha = min(1.0, max(0.0, self.goal_smoothing_alpha))
        self.last_smoothed_goal = (
            alpha * goal_xy_world + (1.0 - alpha) * self.last_smoothed_goal
        )
        return self.last_smoothed_goal

    def _robot_yaw(self, base_to_world) -> float:
        q = base_to_world.transform.rotation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _tick(self):
        valid_recent = (
            self._last_valid_true_time is not None and
            (self.get_clock().now() - self._last_valid_true_time) <= self._valid_holdout
        )
        target_live = valid_recent and self._target_fresh() and self.last_target_msg is not None

        if not target_live:
            lost_elapsed = (
                (self.get_clock().now() - self._last_valid_true_time)
                if self._last_valid_true_time is not None
                else Duration(seconds=999)
            )
            if lost_elapsed < self._search_timeout:
                return

            base_to_world = self._lookup_transform(self.world_frame, self.base_frame, Time())
            if base_to_world is None:
                return

            self._search_active = True
            search_yaw = self._robot_yaw(base_to_world) + self._search_offset_rad
            dist = self.follow_distance_m
            qx, qy, qz, qw = _yaw_to_quat(search_yaw)

            goal_msg = PoseStamped()
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.header.frame_id = self.world_frame
            goal_msg.pose.position.x = (
                base_to_world.transform.translation.x + dist * math.cos(search_yaw)
            )
            goal_msg.pose.position.y = (
                base_to_world.transform.translation.y + dist * math.sin(search_yaw)
            )
            goal_msg.pose.position.z = 0.0
            goal_msg.pose.orientation.x = qx
            goal_msg.pose.orientation.y = qy
            goal_msg.pose.orientation.z = qz
            goal_msg.pose.orientation.w = qw
            self.goal_pub.publish(goal_msg)
            return

        if self._search_active:
            self._search_active = False
            self.last_smoothed_goal = None

        user_base = self._transform_point_msg(self.last_target_msg, self.base_frame)
        if user_base is None:
            return
        self.user_base_pub.publish(user_base)

        goal_base_xyz = self._compute_goal_in_base(user_base)
        if goal_base_xyz is None:
            return

        base_to_world = self._lookup_transform(
            self.world_frame,
            self.base_frame,
            Time(),
        )
        if base_to_world is None:
            return

        user_world = self._transform_point_msg(self.last_target_msg, self.world_frame)
        if user_world is None:
            return
        self.user_world_pub.publish(user_world)

        goal_world_xyz = self._transform_xyz(goal_base_xyz, base_to_world)
        goal_xy_world = self._smooth_goal(goal_world_xyz[:2])

        yaw = math.atan2(
            user_world.point.y - goal_xy_world[1],
            user_world.point.x - goal_xy_world[0],
        )
        qx, qy, qz, qw = _yaw_to_quat(yaw)

        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = self.world_frame
        goal_msg.pose.position.x = float(goal_xy_world[0])
        goal_msg.pose.position.y = float(goal_xy_world[1])
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.x = qx
        goal_msg.pose.orientation.y = qy
        goal_msg.pose.orientation.z = qz
        goal_msg.pose.orientation.w = qw
        self.goal_pub.publish(goal_msg)


def main():
    rclpy.init()
    node = FollowGoal()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
