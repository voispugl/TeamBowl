#!/usr/bin/env python3
"""
trajectory_test — Foxglove-driven trajectory test node.

Accepts a JSON goal from Foxglove, converts it to an odom-frame PoseStamped,
then continuously replans + executes via the nav2 stack (ComputePathToPose +
FollowPath actions) at replan_rate_hz while in the RUNNING state.

Foxglove usage:
  1. Publish goal: /trajectory_goal  {"data": "{\"x\": 2.0, \"y\": 0.0, \"theta\": 0.0, \"relative\": true}"}
  2. Send command: /trajectory_cmd   {"data": "go"}
  3. Watch path:   /trajectory_path  (nav_msgs/Path)
  4. Read status:  /trajectory_status (std_msgs/String JSON)
  5. Stop:         /trajectory_cmd   {"data": "stop"}
"""

from __future__ import annotations

import json
import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String


def _yaw_from_quat(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def _yaw_to_quat(yaw: float):
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.z = math.sin(yaw / 2.0)
    return q


def _wrap(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class TrajectoryTestNode(Node):

    IDLE = 'IDLE'
    RUNNING = 'RUNNING'

    def __init__(self):
        super().__init__('trajectory_test')

        # --- Parameters ---
        self.declare_parameter('planner_action_name', '/compute_path_to_pose')
        self.declare_parameter('controller_action_name', '/follow_path')
        self.declare_parameter('planner_id', 'GridBased')
        self.declare_parameter('controller_id', 'FollowPath')
        self.declare_parameter('goal_checker_id', 'goal_checker')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('mode_topic', '/robot_mode')
        self.declare_parameter('estop_topic', '/estop')
        self.declare_parameter('path_topic', '/trajectory_path')
        self.declare_parameter('status_topic', '/trajectory_status')
        self.declare_parameter('goal_topic', '/trajectory_goal')
        self.declare_parameter('cmd_topic', '/trajectory_cmd')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_auto')
        self.declare_parameter('replan_rate_hz', 2.0)
        self.declare_parameter('min_goal_change_m', 0.10)
        self.declare_parameter('min_goal_change_rad', 0.20)
        self.declare_parameter('odom_timeout_s', 0.5)

        p = self.get_parameter
        self._planner_action  = str(p('planner_action_name').value)
        self._controller_action = str(p('controller_action_name').value)
        self._planner_id      = str(p('planner_id').value)
        self._controller_id   = str(p('controller_id').value)
        self._goal_checker_id = str(p('goal_checker_id').value)
        self._replan_rate     = float(p('replan_rate_hz').value)
        self._min_dist        = float(p('min_goal_change_m').value)
        self._min_rad         = float(p('min_goal_change_rad').value)
        self._odom_timeout    = Duration(seconds=float(p('odom_timeout_s').value))

        # --- State ---
        self._state = self.IDLE
        self._robot_mode = 'off'
        self._estop = False
        self._latest_goal: PoseStamped | None = None     # pending goal (JSON parsed)
        self._active_goal: PoseStamped | None = None     # goal currently being executed
        self._last_planned_goal: PoseStamped | None = None
        self._latest_odom: Odometry | None = None
        self._latest_odom_time = None

        self._planner_request_in_flight = False
        self._controller_cancel_in_flight = False
        self._planner_goal_handle = None
        self._controller_goal_handle = None

        # --- QoS ---
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        mode_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # --- Subscribers ---
        self.create_subscription(String, p('goal_topic').value, self._goal_cb, 10)
        self.create_subscription(String, p('cmd_topic').value, self._cmd_cb, 10)
        self.create_subscription(Odometry, p('odom_topic').value, self._odom_cb, best_effort_qos)
        self.create_subscription(String, p('mode_topic').value, self._mode_cb, mode_qos)
        self.create_subscription(Bool, p('estop_topic').value, self._estop_cb, best_effort_qos)

        # --- Publishers ---
        self._path_pub = self.create_publisher(Path, p('path_topic').value, 10)
        self._status_pub = self.create_publisher(String, p('status_topic').value, 10)
        self._cmd_vel_pub = self.create_publisher(Twist, p('cmd_vel_topic').value, 10)

        # --- Action clients ---
        self._planner_client = ActionClient(self, ComputePathToPose, self._planner_action)
        self._controller_client = ActionClient(self, FollowPath, self._controller_action)

        # --- Timers ---
        period = 1.0 / max(self._replan_rate, 0.1)
        self._replan_timer = self.create_timer(period, self._replan_tick)
        self._status_timer = self.create_timer(0.5, self._publish_status)

        self.get_logger().info(
            f'trajectory_test ready | planner={self._planner_action} '
            f'controller={self._controller_action} replan={self._replan_rate} Hz'
        )

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _goal_cb(self, msg: String):
        """Parse JSON goal from Foxglove. Does NOT start execution."""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f'Invalid JSON on /trajectory_goal: {exc}')
            return

        x = float(data.get('x', 0.0))
        y = float(data.get('y', 0.0))
        theta = float(data.get('theta', 0.0))
        relative = bool(data.get('relative', False))

        if relative:
            odom = self._latest_odom
            if odom is None:
                self.get_logger().error('Cannot resolve relative goal — no odom yet')
                return
            rx = odom.pose.pose.position.x
            ry = odom.pose.pose.position.y
            yaw = _yaw_from_quat(odom.pose.pose.orientation)
            gx = rx + math.cos(yaw) * x - math.sin(yaw) * y
            gy = ry + math.sin(yaw) * x + math.cos(yaw) * y
            gtheta = _wrap(yaw + theta)
        else:
            gx, gy, gtheta = x, y, theta

        pose = PoseStamped()
        pose.header.frame_id = 'odom'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = gx
        pose.pose.position.y = gy
        pose.pose.orientation = _yaw_to_quat(gtheta)

        self._latest_goal = pose
        self.get_logger().info(
            f'Goal set: ({gx:.2f}, {gy:.2f}, θ={math.degrees(gtheta):.1f}°) '
            f'[{"relative" if relative else "absolute"}]'
        )

    def _cmd_cb(self, msg: String):
        cmd = msg.data.strip().lower()

        if cmd == 'go':
            if self._latest_goal is None:
                self.get_logger().warn('"go" received but no goal set — publish to /trajectory_goal first')
                return
            if self._estop:
                self.get_logger().warn('"go" received but estop is active')
                return
            if self._robot_mode != 'driving':
                self.get_logger().warn(f'"go" received but mode is "{self._robot_mode}" (need "driving")')
                return
            self._active_goal = self._latest_goal
            self._last_planned_goal = None  # force replan on next tick
            self._state = self.RUNNING
            self.get_logger().info('Trajectory: IDLE → RUNNING')

        elif cmd == 'stop':
            if self._state == self.RUNNING:
                self.get_logger().info('Trajectory: RUNNING → IDLE (stop command)')
                self._stop()

        elif cmd == 'reset':
            self.get_logger().info('Trajectory: reset — clearing goal and path')
            self._stop()
            self._latest_goal = None
            self._active_goal = None
            self._publish_empty_path()

        else:
            self.get_logger().warn(f'Unknown trajectory_cmd: "{cmd}" (valid: go, stop, reset)')

    def _odom_cb(self, msg: Odometry):
        self._latest_odom = msg
        self._latest_odom_time = self.get_clock().now()

    def _mode_cb(self, msg: String):
        new_mode = msg.data.strip().lower()
        if new_mode == self._robot_mode:
            return
        prev = self._robot_mode
        self._robot_mode = new_mode
        self.get_logger().info(f'robot_mode: {prev} → {new_mode}')
        if self._state == self.RUNNING and new_mode != 'driving':
            self.get_logger().warn('Mode left "driving" — stopping trajectory')
            self._stop()

    def _estop_cb(self, msg: Bool):
        new_estop = bool(msg.data)
        if new_estop and not self._estop:
            self.get_logger().warn('Estop active — stopping trajectory')
            self._stop()
        self._estop = new_estop

    # ------------------------------------------------------------------
    # Replan tick (runs at replan_rate_hz while RUNNING)
    # ------------------------------------------------------------------

    def _replan_tick(self):
        if self._state != self.RUNNING:
            return
        if self._active_goal is None:
            return
        if self._planner_request_in_flight or self._controller_cancel_in_flight:
            return

        # Skip replan if goal hasn't changed enough and controller is still running
        if (
            self._last_planned_goal is not None
            and self._controller_goal_handle is not None
            and not self._goal_changed_enough(self._last_planned_goal, self._active_goal)
        ):
            return

        # Check action servers are available
        if not self._planner_client.server_is_ready():
            self.get_logger().warn('Planner not ready yet', throttle_duration_sec=5.0)
            return
        if not self._controller_client.server_is_ready():
            self.get_logger().warn('Controller not ready yet', throttle_duration_sec=5.0)
            return

        self._request_path(self._active_goal)

    def _goal_changed_enough(self, prev: PoseStamped, curr: PoseStamped) -> bool:
        dx = curr.pose.position.x - prev.pose.position.x
        dy = curr.pose.position.y - prev.pose.position.y
        dist = math.hypot(dx, dy)
        yaw_prev = _yaw_from_quat(prev.pose.orientation)
        yaw_curr = _yaw_from_quat(curr.pose.orientation)
        dyaw = abs(_wrap(yaw_curr - yaw_prev))
        return dist >= self._min_dist or dyaw >= self._min_rad

    # ------------------------------------------------------------------
    # Action client chain (mirrors follow_executor.py)
    # ------------------------------------------------------------------

    def _request_path(self, goal_pose: PoseStamped):
        goal = ComputePathToPose.Goal()
        goal.goal = goal_pose
        goal.planner_id = self._planner_id
        goal.use_start = False

        self._planner_request_in_flight = True
        future = self._planner_client.send_goal_async(goal)
        future.add_done_callback(self._on_planner_goal_response)

    def _on_planner_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as exc:
            self._planner_request_in_flight = False
            self.get_logger().error(f'Failed to send planner goal: {exc}')
            return

        if not handle.accepted:
            self._planner_request_in_flight = False
            self.get_logger().warn('Planner rejected goal')
            return

        self._planner_goal_handle = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._on_planner_result)

    def _on_planner_result(self, future):
        self._planner_request_in_flight = False
        try:
            wrapped = future.result()
        except Exception as exc:
            self.get_logger().error(f'Planner result error: {exc}')
            return

        path: Path = wrapped.result.path
        if len(path.poses) == 0:
            self.get_logger().warn('Planner returned empty path')
            return

        self._path_pub.publish(path)
        self._last_planned_goal = self._active_goal

        if self._state != self.RUNNING:
            # State changed while planner was running — cancel controller
            self._cancel_controller_if_needed()
            self._publish_zero_cmd()
            return

        # Cancel old FollowPath, then send the new path
        if self._controller_goal_handle is not None:
            self._cancel_controller_if_needed(next_path=path)
        else:
            self._send_follow_path(path)

    def _cancel_controller_if_needed(self, next_path: Path | None = None):
        if self._controller_goal_handle is None:
            if next_path is not None:
                self._send_follow_path(next_path)
            return
        if self._controller_cancel_in_flight:
            return
        self._controller_cancel_in_flight = True
        cancel_future = self._controller_goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(
            lambda f: self._on_controller_cancelled(f, next_path)
        )

    def _on_controller_cancelled(self, future, next_path: Path | None):
        self._controller_cancel_in_flight = False
        try:
            future.result()
        except Exception as exc:
            self.get_logger().warn(f'Controller cancel error: {exc}')
        self._controller_goal_handle = None
        if next_path is not None and self._state == self.RUNNING:
            self._send_follow_path(next_path)

    def _send_follow_path(self, path: Path):
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = self._controller_id
        goal.goal_checker_id = self._goal_checker_id

        future = self._controller_client.send_goal_async(goal)
        future.add_done_callback(self._on_controller_goal_response)

    def _on_controller_goal_response(self, future):
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'Failed to send controller goal: {exc}')
            return
        if not handle.accepted:
            self.get_logger().warn('Controller rejected follow path')
            return
        self._controller_goal_handle = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._on_controller_result)

    def _on_controller_result(self, future):
        try:
            future.result()
            self.get_logger().info('Controller finished path — awaiting replan')
        except Exception as exc:
            self.get_logger().warn(f'Controller result error: {exc}')
        finally:
            self._controller_goal_handle = None

    # ------------------------------------------------------------------
    # Stop helpers
    # ------------------------------------------------------------------

    def _stop(self):
        """Cancel active actions, publish zero vel, go to IDLE."""
        self._state = self.IDLE
        self._active_goal = None
        self._last_planned_goal = None
        self._cancel_controller_if_needed()
        self._publish_zero_cmd()

    def _publish_zero_cmd(self):
        self._cmd_vel_pub.publish(Twist())

    def _publish_empty_path(self):
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = self.get_clock().now().to_msg()
        self._path_pub.publish(path)

    # ------------------------------------------------------------------
    # Status publisher
    # ------------------------------------------------------------------

    def _publish_status(self):
        status: dict = {'state': self._state, 'mode': self._robot_mode}
        if self._active_goal is not None:
            p = self._active_goal.pose.position
            yaw = _yaw_from_quat(self._active_goal.pose.orientation)
            status['goal_x'] = round(p.x, 3)
            status['goal_y'] = round(p.y, 3)
            status['goal_theta_deg'] = round(math.degrees(yaw), 1)
        status['planner_in_flight'] = self._planner_request_in_flight
        status['controller_active'] = self._controller_goal_handle is not None
        msg = String()
        msg.data = json.dumps(status)
        self._status_pub.publish(msg)


def main():
    rclpy.init()
    node = TrajectoryTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
