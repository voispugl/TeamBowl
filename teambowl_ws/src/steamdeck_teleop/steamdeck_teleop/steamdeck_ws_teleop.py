#!/usr/bin/env python3
"""
steamdeck_ws_teleop — Steam Deck browser gamepad → Nav2 goal sender.

Runs a WebSocket server on the robot. The Steam Deck opens a browser to
http://ROBOT_IP:8888 to get the control page, which reads the Steam Deck
gamepad via the Web Gamepad API and streams state over WebSocket.

Controls:
  Hold RT (right trigger)   arm goal accumulation
  Left stick                position goal (forward/back/strafe)
  Right stick X             rotate goal heading
  Release RT                reset goal accumulator to origin
  A button                  send goal to Nav2
  B button                  cancel active navigation / reset goal
  Menu button               E-stop
"""

from __future__ import annotations

import asyncio
import http
import json
import math
import threading

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

import websockets

# ---------------------------------------------------------------------------
# Embedded control page — served at http://ROBOT_IP:8888/
# ---------------------------------------------------------------------------

_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TeamBowl Teleop</title>
<style>
  body { font-family: monospace; background: #111; color: #eee; padding: 20px; font-size: 16px; }
  h2 { color: #4af; margin: 0 0 16px; }
  .box { background: #222; border-radius: 8px; padding: 14px; margin: 10px 0; }
  .label { color: #888; font-size: 13px; }
  .val { font-size: 22px; font-weight: bold; }
  .armed { color: #4f4; }
  .idle  { color: #f84; }
  .run   { color: #4af; }
  .estop { color: #f44; }
  #help  { color: #aaa; font-size: 13px; line-height: 1.7; }
</style>
</head>
<body>
<h2>TeamBowl Teleop</h2>

<div class="box">
  <div class="label">WebSocket</div>
  <div class="val" id="ws-status">Connecting…</div>
</div>

<div class="box">
  <div class="label">Gamepad</div>
  <div class="val" id="gp-status">Press any button to activate</div>
</div>

<div class="box">
  <div class="label">Dead-man (RT)</div>
  <div class="val idle" id="dm-status">DISARMED</div>
</div>

<div class="box">
  <div class="label">Goal preview (robot frame)</div>
  <div class="val" id="goal-preview">x=0.00  y=0.00  θ=0°</div>
</div>

<div class="box">
  <div class="label">Robot state</div>
  <div class="val" id="robot-state">—</div>
</div>

<div class="box" id="help">
  <b>Controls</b><br>
  Hold RT → arm goal | Left stick → position | Right stick X → heading<br>
  Release RT → reset goal | A → send goal | B → cancel / reset | Menu → E-STOP
</div>

<script>
const wsUrl = 'ws://' + location.host + '/ws';
let ws = null;
let gpIndex = null;
let sendInterval = null;
let prevDm = false;

function connect() {
  ws = new WebSocket(wsUrl);
  ws.onopen = () => {
    document.getElementById('ws-status').textContent = 'Connected';
    document.getElementById('ws-status').className = 'val run';
    startSending();
  };
  ws.onclose = () => {
    document.getElementById('ws-status').textContent = 'Disconnected — retrying…';
    document.getElementById('ws-status').className = 'val idle';
    stopSending();
    setTimeout(connect, 2000);
  };
  ws.onerror = () => {};
  ws.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data);
      const stateEl = document.getElementById('robot-state');
      stateEl.textContent =
        'state=' + d.state + '  mode=' + d.mode +
        '  goal=(' + d.goal_x + ', ' + d.goal_y + ', ' + d.goal_theta_deg + '°)';
      stateEl.className = 'val ' + (d.state === 'RUNNING' ? 'run' : 'idle');
    } catch (_) {}
  };
}

window.addEventListener('gamepadconnected', (e) => {
  gpIndex = e.gamepad.index;
  document.getElementById('gp-status').textContent = e.gamepad.id;
  document.getElementById('gp-status').className = 'val armed';
});
window.addEventListener('gamepaddisconnected', () => {
  gpIndex = null;
  document.getElementById('gp-status').textContent = 'Disconnected';
  document.getElementById('gp-status').className = 'val idle';
});

function startSending() {
  if (sendInterval) return;
  sendInterval = setInterval(sendState, 50);
}
function stopSending() {
  if (sendInterval) { clearInterval(sendInterval); sendInterval = null; }
}

function sendState() {
  if (gpIndex === null) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const gp = navigator.getGamepads()[gpIndex];
  if (!gp) return;

  const axes = Array.from(gp.axes);
  const buttons = gp.buttons.map(b => b.pressed ? 1 : 0);
  const dmAxis = axes.length > 5 ? axes[5] : 0;
  const dm = dmAxis > 0.5;

  const dmEl = document.getElementById('dm-status');
  if (dm !== prevDm) {
    dmEl.textContent = dm ? 'ARMED' : 'DISARMED';
    dmEl.className = 'val ' + (dm ? 'armed' : 'idle');
    prevDm = dm;
  }

  ws.send(JSON.stringify({ axes, buttons }));
}

connect();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Helpers (copied from trajectory_test.py)
# ---------------------------------------------------------------------------

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


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class SteamDeckWSTeleop(Node):

    IDLE = 'IDLE'
    RUNNING = 'RUNNING'

    def __init__(self):
        super().__init__('steamdeck_ws_teleop')

        # --- Parameters ---
        self.declare_parameter('ws_host', '0.0.0.0')
        self.declare_parameter('ws_port', 8888)
        self.declare_parameter('joy_rate_hz', 20.0)
        self.declare_parameter('goal_scale_m_per_tick', 0.02)
        self.declare_parameter('yaw_scale_rad_per_tick', 0.03)
        self.declare_parameter('max_goal_dist_m', 3.0)
        self.declare_parameter('dead_man_axis', 5)
        self.declare_parameter('dead_man_threshold', 0.5)
        self.declare_parameter('forward_axis', 1)
        self.declare_parameter('strafe_axis', 0)
        self.declare_parameter('yaw_axis', 3)
        self.declare_parameter('confirm_button', 0)
        self.declare_parameter('cancel_button', 1)
        self.declare_parameter('estop_button', 8)
        self.declare_parameter('planner_action_name', '/compute_path_to_pose')
        self.declare_parameter('controller_action_name', '/follow_path')
        self.declare_parameter('planner_id', 'GridBased')
        self.declare_parameter('controller_id', 'FollowPath')
        self.declare_parameter('goal_checker_id', 'goal_checker')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('mode_topic', '/robot_mode')
        self.declare_parameter('mode_set_topic', '/robot_mode_set')
        self.declare_parameter('estop_topic', '/estop')
        self.declare_parameter('preview_topic', '/steamdeck/goal_preview')
        self.declare_parameter('auto_set_driving_mode', True)
        self.declare_parameter('driving_mode_delay_s', 5.0)

        p = self.get_parameter
        self._ws_host         = str(p('ws_host').value)
        self._ws_port         = int(p('ws_port').value)
        self._goal_scale      = float(p('goal_scale_m_per_tick').value)
        self._yaw_scale       = float(p('yaw_scale_rad_per_tick').value)
        self._max_dist        = float(p('max_goal_dist_m').value)
        self._dm_axis         = int(p('dead_man_axis').value)
        self._dm_thresh       = float(p('dead_man_threshold').value)
        self._fwd_axis        = int(p('forward_axis').value)
        self._str_axis        = int(p('strafe_axis').value)
        self._yaw_axis        = int(p('yaw_axis').value)
        self._confirm_btn     = int(p('confirm_button').value)
        self._cancel_btn      = int(p('cancel_button').value)
        self._estop_btn       = int(p('estop_button').value)
        self._planner_action  = str(p('planner_action_name').value)
        self._controller_action = str(p('controller_action_name').value)
        self._planner_id      = str(p('planner_id').value)
        self._controller_id   = str(p('controller_id').value)
        self._goal_checker_id = str(p('goal_checker_id').value)

        # --- State ---
        self._state = self.IDLE
        self._robot_mode = 'off'
        self._estop = False
        self._latest_odom: Odometry | None = None

        self._goal_x = 0.0
        self._goal_y = 0.0
        self._goal_theta = 0.0
        self._dead_man_prev = False
        self._prev_confirm = 0
        self._prev_cancel  = 0
        self._prev_estop   = 0

        self._active_goal: PoseStamped | None = None
        self._planner_request_in_flight = False
        self._controller_cancel_in_flight = False
        self._planner_goal_handle = None
        self._controller_goal_handle = None

        # Shared joy state (written by WS thread, read by rclpy timer)
        self._lock = threading.Lock()
        self._joy_state = {'axes': [0.0] * 10, 'buttons': [0] * 12}

        # --- QoS ---
        best_effort = QoSProfile(
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

        # --- Subscriptions ---
        self.create_subscription(String, p('mode_topic').value, self._mode_cb, mode_qos)
        self.create_subscription(Odometry, p('odom_topic').value, self._odom_cb, best_effort)
        self.create_subscription(Bool, p('estop_topic').value, self._estop_cb, best_effort)

        # --- Publishers ---
        self._preview_pub   = self.create_publisher(PoseStamped, p('preview_topic').value, 10)
        self._mode_set_pub  = self.create_publisher(String, p('mode_set_topic').value, 10)
        self._estop_pub     = self.create_publisher(Bool, p('estop_topic').value, 10)

        # --- Action clients ---
        self._planner_client    = ActionClient(self, ComputePathToPose, self._planner_action)
        self._controller_client = ActionClient(self, FollowPath, self._controller_action)

        # --- Timers ---
        rate = float(p('joy_rate_hz').value)
        self.create_timer(1.0 / max(rate, 1.0), self._joy_tick)

        if p('auto_set_driving_mode').value:
            delay = float(p('driving_mode_delay_s').value)
            self._mode_timer = self.create_timer(delay, self._auto_set_driving_mode)

        # --- WebSocket server in background thread ---
        self._ws_thread = threading.Thread(target=self._run_ws_server, daemon=True)
        self._ws_thread.start()

        self.get_logger().info(
            f'steamdeck_ws_teleop ready | ws://0.0.0.0:{self._ws_port} | '
            f'open http://ROBOT_IP:{self._ws_port} in Steam Deck browser'
        )

    # ------------------------------------------------------------------
    # WebSocket server (runs in background daemon thread)
    # ------------------------------------------------------------------

    def _run_ws_server(self):
        asyncio.run(self._ws_main())

    async def _ws_main(self):
        async def process_request(path, request_headers):
            # Serve the HTML page for plain HTTP GET requests (non-WebSocket)
            upgrade = request_headers.get('Upgrade', '').lower()
            if upgrade != 'websocket':
                return http.HTTPStatus.OK, [('Content-Type', 'text/html')], _HTML_PAGE.encode()

        async with websockets.serve(
            self._ws_handler,
            self._ws_host,
            self._ws_port,
            process_request=process_request,
        ):
            self.get_logger().info(f'WebSocket server listening on port {self._ws_port}')
            await asyncio.Future()  # run forever

    async def _ws_handler(self, websocket, path):
        remote = websocket.remote_address
        self.get_logger().info(f'Steam Deck connected from {remote}')
        try:
            async for raw in websocket:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                with self._lock:
                    self._joy_state = data
                # Send status back to the browser
                status = {
                    'state':         self._state,
                    'mode':          self._robot_mode,
                    'goal_x':        round(self._goal_x, 2),
                    'goal_y':        round(self._goal_y, 2),
                    'goal_theta_deg': round(math.degrees(self._goal_theta), 1),
                }
                await websocket.send(json.dumps(status))
        except websockets.ConnectionClosed:
            pass
        self.get_logger().info(f'Steam Deck disconnected from {remote}')
        # Reset joy state to zero so the robot doesn't hold a stale input
        with self._lock:
            self._joy_state = {'axes': [0.0] * 10, 'buttons': [0] * 12}

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _mode_cb(self, msg: String):
        new_mode = msg.data.strip().lower()
        if new_mode == self._robot_mode:
            return
        prev = self._robot_mode
        self._robot_mode = new_mode
        self.get_logger().info(f'robot_mode: {prev} → {new_mode}')
        if self._state == self.RUNNING and new_mode != 'driving':
            self.get_logger().warn('Mode left "driving" — stopping navigation')
            self._stop()

    def _odom_cb(self, msg: Odometry):
        self._latest_odom = msg

    def _estop_cb(self, msg: Bool):
        new_val = bool(msg.data)
        if new_val and not self._estop:
            self.get_logger().warn('Estop received — stopping navigation')
            self._stop()
        self._estop = new_val

    # ------------------------------------------------------------------
    # Joy tick — runs at joy_rate_hz (20 Hz) via rclpy timer
    # ------------------------------------------------------------------

    def _joy_tick(self):
        with self._lock:
            joy = dict(self._joy_state)

        axes    = joy.get('axes', [])
        buttons = joy.get('buttons', [])

        def _axis(i: int) -> float:
            return axes[i] if i < len(axes) else 0.0

        def _btn(i: int) -> int:
            return buttons[i] if i < len(buttons) else 0

        # --- Dead-man (right trigger) ---
        dead_man = _axis(self._dm_axis) > self._dm_thresh

        # Falling edge: trigger released → reset goal accumulator
        if self._dead_man_prev and not dead_man:
            self._goal_x = self._goal_y = self._goal_theta = 0.0

        self._dead_man_prev = dead_man

        # --- Accumulate goal (only while armed) ---
        if dead_man:
            self._goal_x = _clamp(
                self._goal_x + _axis(self._fwd_axis) * self._goal_scale,
                -self._max_dist, self._max_dist,
            )
            self._goal_y = _clamp(
                self._goal_y + _axis(self._str_axis) * self._goal_scale,
                -self._max_dist, self._max_dist,
            )
            self._goal_theta = _wrap(
                self._goal_theta + _axis(self._yaw_axis) * self._yaw_scale
            )

        # --- Button edge detection ---
        confirm = _btn(self._confirm_btn)
        cancel  = _btn(self._cancel_btn)
        estop   = _btn(self._estop_btn)

        if confirm == 1 and self._prev_confirm == 0:
            self._on_confirm_pressed()
        if cancel == 1 and self._prev_cancel == 0:
            self._on_cancel_pressed()
        if estop == 1 and self._prev_estop == 0:
            self._on_estop_pressed()

        self._prev_confirm = confirm
        self._prev_cancel  = cancel
        self._prev_estop   = estop

        # --- Publish goal preview at tick rate ---
        if self._latest_odom is not None:
            self._preview_pub.publish(self._compute_odom_goal())

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_confirm_pressed(self):
        if self._robot_mode != 'driving':
            self.get_logger().warn(
                f'A pressed but mode is "{self._robot_mode}" — need "driving"'
            )
            return
        if self._estop:
            self.get_logger().warn('A pressed but estop is active')
            return
        if self._latest_odom is None:
            self.get_logger().error('No odometry yet — cannot send goal')
            return

        goal_pose = self._compute_odom_goal()
        self.get_logger().info(
            f'Sending goal: ({goal_pose.pose.position.x:.2f}, '
            f'{goal_pose.pose.position.y:.2f})'
        )

        if self._state == self.RUNNING:
            # Cancel current execution, then send new goal
            self._active_goal = goal_pose
            self._last_active_goal = goal_pose
            self._cancel_controller_if_needed(next_path=None)
        else:
            self._active_goal = goal_pose
            self._state = self.RUNNING
            self._request_path(goal_pose)

    def _on_cancel_pressed(self):
        if self._state == self.RUNNING:
            self.get_logger().info('B pressed — cancelling navigation')
            self._stop()
        else:
            self._goal_x = self._goal_y = self._goal_theta = 0.0
            self.get_logger().info('B pressed (idle) — goal accumulator reset')

    def _on_estop_pressed(self):
        self.get_logger().warn('Menu pressed — E-STOP')
        msg = Bool()
        msg.data = True
        self._estop_pub.publish(msg)
        mode_msg = String()
        mode_msg.data = 'off'
        self._mode_set_pub.publish(mode_msg)
        self._stop()

    # ------------------------------------------------------------------
    # Frame math
    # ------------------------------------------------------------------

    def _compute_odom_goal(self) -> PoseStamped:
        odom = self._latest_odom
        rx = odom.pose.pose.position.x
        ry = odom.pose.pose.position.y
        yaw = _yaw_from_quat(odom.pose.pose.orientation)

        gx = rx + math.cos(yaw) * self._goal_x - math.sin(yaw) * self._goal_y
        gy = ry + math.sin(yaw) * self._goal_x + math.cos(yaw) * self._goal_y
        gtheta = _wrap(yaw + self._goal_theta)

        pose = PoseStamped()
        pose.header.frame_id = 'odom'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = gx
        pose.pose.position.y = gy
        pose.pose.orientation = _yaw_to_quat(gtheta)
        return pose

    # ------------------------------------------------------------------
    # Auto mode setter
    # ------------------------------------------------------------------

    def _auto_set_driving_mode(self):
        self._mode_timer.cancel()
        msg = String()
        msg.data = 'driving'
        self._mode_set_pub.publish(msg)
        self.get_logger().info('Auto-set robot mode to "driving"')

    # ------------------------------------------------------------------
    # Nav2 action chain (mirrors trajectory_test.py verbatim)
    # ------------------------------------------------------------------

    def _request_path(self, goal_pose: PoseStamped):
        if not self._planner_client.server_is_ready():
            self.get_logger().warn('Planner not ready yet')
            self._state = self.IDLE
            return
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
            self.get_logger().error(f'Planner goal send failed: {exc}')
            self._state = self.IDLE
            return

        if not handle.accepted:
            self._planner_request_in_flight = False
            self.get_logger().warn('Planner rejected goal')
            self._state = self.IDLE
            return

        self._planner_goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_planner_result)

    def _on_planner_result(self, future):
        self._planner_request_in_flight = False
        try:
            wrapped = future.result()
        except Exception as exc:
            self.get_logger().error(f'Planner result error: {exc}')
            self._state = self.IDLE
            return

        path: Path = wrapped.result.path
        if len(path.poses) == 0:
            self.get_logger().warn('Planner returned empty path')
            self._state = self.IDLE
            return

        if self._state != self.RUNNING:
            self._cancel_controller_if_needed()
            return

        if self._controller_goal_handle is not None:
            self._cancel_controller_if_needed(next_path=path)
        else:
            self._send_follow_path(path)

    def _cancel_controller_if_needed(self, next_path: Path | None = None):
        if self._controller_goal_handle is None:
            if next_path is not None:
                self._send_follow_path(next_path)
            elif self._state == self.RUNNING and self._active_goal is not None:
                # New goal arrived while controller was idle — start fresh
                self._request_path(self._active_goal)
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
        elif self._state == self.RUNNING and self._active_goal is not None:
            self._request_path(self._active_goal)

    def _send_follow_path(self, path: Path):
        if not self._controller_client.server_is_ready():
            self.get_logger().warn('Controller not ready yet')
            self._state = self.IDLE
            return
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
            self.get_logger().error(f'Controller goal send failed: {exc}')
            self._state = self.IDLE
            return
        if not handle.accepted:
            self.get_logger().warn('Controller rejected path')
            self._state = self.IDLE
            return
        self._controller_goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_controller_result)

    def _on_controller_result(self, future):
        try:
            future.result()
            self.get_logger().info('Navigation goal reached')
        except Exception as exc:
            self.get_logger().warn(f'Controller result error: {exc}')
        finally:
            self._controller_goal_handle = None
            self._state = self.IDLE

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    def _stop(self):
        self._state = self.IDLE
        self._active_goal = None
        self._cancel_controller_if_needed()


def main():
    rclpy.init()
    node = SteamDeckWSTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
