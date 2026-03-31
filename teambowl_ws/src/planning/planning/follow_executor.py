#!/usr/bin/env python3

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class FollowExecutor(Node):
    def __init__(self):
        super().__init__('follow_executor')

        self.declare_parameter('goal_topic', '/follow_goal')
        self.declare_parameter('goal_timeout_s', 0.75)
        self.declare_parameter('replan_rate_hz', 2.0)
        self.declare_parameter('min_goal_change_m', 0.10)
        self.declare_parameter('min_goal_change_rad', 0.20)
        self.declare_parameter('planner_action_name', '/compute_path_to_pose')
        self.declare_parameter('controller_action_name', '/follow_path')
        self.declare_parameter('planner_id', 'GridBased')
        self.declare_parameter('controller_id', 'FollowPath')
        self.declare_parameter('goal_checker_id', 'goal_checker')
        self.declare_parameter('debug_path_topic', '/follow_path')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_auto')
        self.declare_parameter('mode_topic', '/robot_mode')
        self.declare_parameter('autonomous_mode_name', 'auton')

        self.goal_topic = str(self.get_parameter('goal_topic').value)
        self.goal_timeout = Duration(seconds=float(self.get_parameter('goal_timeout_s').value))
        self.replan_rate_hz = float(self.get_parameter('replan_rate_hz').value)
        self.min_goal_change_m = float(self.get_parameter('min_goal_change_m').value)
        self.min_goal_change_rad = float(self.get_parameter('min_goal_change_rad').value)
        self.planner_action_name = str(self.get_parameter('planner_action_name').value)
        self.controller_action_name = str(self.get_parameter('controller_action_name').value)
        self.planner_id = str(self.get_parameter('planner_id').value)
        self.controller_id = str(self.get_parameter('controller_id').value)
        self.goal_checker_id = str(self.get_parameter('goal_checker_id').value)
        self.debug_path_topic = str(self.get_parameter('debug_path_topic').value)
        self.cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        self.mode_topic = str(self.get_parameter('mode_topic').value)
        self.autonomous_mode_name = str(self.get_parameter('autonomous_mode_name').value)

        self.goal_sub = self.create_subscription(PoseStamped, self.goal_topic, self._goal_cb, 10)
        self.path_pub = self.create_publisher(Path, self.debug_path_topic, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        mode_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.mode_sub = self.create_subscription(
            String,
            self.mode_topic,
            self._mode_cb,
            mode_qos,
        )

        self.planner_client = ActionClient(self, ComputePathToPose, self.planner_action_name)
        self.controller_client = ActionClient(self, FollowPath, self.controller_action_name)

        self.latest_goal: PoseStamped | None = None
        self.latest_goal_time = None
        self.last_planned_goal: PoseStamped | None = None
        self.robot_mode = 'off'

        self.planner_goal_handle = None
        self.controller_goal_handle = None
        self.planner_request_in_flight = False
        self.controller_cancel_in_flight = False

        period = 1.0 / max(self.replan_rate_hz, 0.1)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f'follow_executor started | goal={self.goal_topic} '
            f'planner={self.planner_action_name} controller={self.controller_action_name}'
        )

    def _goal_cb(self, msg: PoseStamped):
        self.latest_goal = msg
        self.latest_goal_time = self.get_clock().now()

    def _mode_cb(self, msg: String):
        new_mode = msg.data.strip().lower()
        if new_mode == self.robot_mode:
            return

        self.robot_mode = new_mode
        if self.robot_mode != self.autonomous_mode_name:
            self._cancel_controller_if_needed()
            self._publish_zero_cmd()

    def _goal_fresh(self) -> bool:
        if self.latest_goal is None or self.latest_goal_time is None:
            return False
        return (self.get_clock().now() - self.latest_goal_time) <= self.goal_timeout

    def _tick(self):
        if not self._goal_fresh():
            self._cancel_controller_if_needed()
            self._publish_zero_cmd()
            return

        if self.planner_request_in_flight or self.controller_cancel_in_flight:
            return

        if self.latest_goal is None:
            return

        if self.robot_mode != self.autonomous_mode_name:
            if self.last_planned_goal is None or self._goal_changed_enough(
                self.last_planned_goal, self.latest_goal
            ):
                self._request_path(self.latest_goal)
            return

        if not self.planner_client.server_is_ready():
            if not self.planner_client.wait_for_server(timeout_sec=0.0):
                self.get_logger().warn('Planner action server not ready yet')
                return

        if not self.controller_client.server_is_ready():
            if not self.controller_client.wait_for_server(timeout_sec=0.0):
                self.get_logger().warn('Controller action server not ready yet')
                return

        if self.last_planned_goal is not None and not self._goal_changed_enough(
            self.last_planned_goal, self.latest_goal
        ):
            return

        self._request_path(self.latest_goal)

    def _goal_changed_enough(self, previous: PoseStamped, current: PoseStamped) -> bool:
        dx = current.pose.position.x - previous.pose.position.x
        dy = current.pose.position.y - previous.pose.position.y
        distance = math.hypot(dx, dy)

        prev_yaw = self._quat_to_yaw(previous)
        curr_yaw = self._quat_to_yaw(current)
        yaw_delta = math.atan2(math.sin(curr_yaw - prev_yaw), math.cos(curr_yaw - prev_yaw))

        return distance >= self.min_goal_change_m or abs(yaw_delta) >= self.min_goal_change_rad

    def _quat_to_yaw(self, pose: PoseStamped) -> float:
        q = pose.pose.orientation
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

    def _request_path(self, goal_pose: PoseStamped):
        goal = ComputePathToPose.Goal()
        goal.goal = goal_pose
        goal.planner_id = self.planner_id
        goal.use_start = False

        self.planner_request_in_flight = True
        self.get_logger().info(
            f'Requesting path to '
            f'({goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f}) '
            f'in {goal_pose.header.frame_id}'
        )
        future = self.planner_client.send_goal_async(goal)
        future.add_done_callback(self._on_planner_goal_response)

    def _on_planner_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.planner_request_in_flight = False
            self.get_logger().error(f'Failed to send planner goal: {exc}')
            return

        if not goal_handle.accepted:
            self.planner_request_in_flight = False
            self.get_logger().warn('Planner rejected follow goal')
            return

        self.planner_goal_handle = goal_handle
        self.get_logger().info('Planner accepted follow goal')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_planner_result)

    def _on_planner_result(self, future):
        self.planner_request_in_flight = False
        try:
            wrapped_result = future.result()
        except Exception as exc:
            self.get_logger().error(f'Planner result failed: {exc}')
            return

        result = wrapped_result.result
        path = result.path
        if len(path.poses) == 0:
            self.get_logger().warn('Planner returned an empty path')
            return

        self.get_logger().info(f'Planner returned path with {len(path.poses)} poses')
        self.path_pub.publish(path)
        self.last_planned_goal = self.latest_goal

        if self.robot_mode != self.autonomous_mode_name:
            self._cancel_controller_if_needed()
            self._publish_zero_cmd()
            return

        if self.controller_goal_handle is not None:
            self._cancel_controller_if_needed(next_path=path)
        else:
            self._send_follow_path(path)

    def _cancel_controller_if_needed(self, next_path: Path | None = None):
        if self.controller_goal_handle is None:
            if next_path is not None:
                self._send_follow_path(next_path)
            return

        if self.controller_cancel_in_flight:
            return

        self.controller_cancel_in_flight = True
        cancel_future = self.controller_goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(
            lambda future: self._on_controller_cancelled(future, next_path)
        )

    def _on_controller_cancelled(self, future, next_path: Path | None):
        self.controller_cancel_in_flight = False
        try:
            future.result()
        except Exception as exc:
            self.get_logger().warn(f'Controller cancel failed: {exc}')

        self.controller_goal_handle = None
        if next_path is not None:
            self._send_follow_path(next_path)

    def _send_follow_path(self, path: Path):
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = self.controller_id
        goal.goal_checker_id = self.goal_checker_id

        future = self.controller_client.send_goal_async(goal)
        future.add_done_callback(self._on_controller_goal_response)

    def _on_controller_goal_response(self, future):
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'Failed to send controller goal: {exc}')
            return

        if not goal_handle.accepted:
            self.get_logger().warn('Controller rejected follow path')
            return

        self.controller_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_controller_result)

    def _on_controller_result(self, future):
        try:
            future.result()
        except Exception as exc:
            self.get_logger().warn(f'Controller result error: {exc}')
        finally:
            self.controller_goal_handle = None

    def _publish_zero_cmd(self):
        msg = Twist()
        self.cmd_pub.publish(msg)


def main():
    rclpy.init()
    node = FollowExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
