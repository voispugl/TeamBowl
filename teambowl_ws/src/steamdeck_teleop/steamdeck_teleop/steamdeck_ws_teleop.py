#!/usr/bin/env python3
"""
steamdeck_ws_teleop — Steam Deck browser gamepad → Nav2 goal sender + diagnostics + nav map.

Runs a WebSocket server on the robot. The Steam Deck opens a browser to
http://ROBOT_IP:8888 to get the control page, which reads the Steam Deck
gamepad via the Web Gamepad API and streams state over WebSocket.

Two WS message types from server → browser:
  type=status  sent in reply to every joy message (20 Hz)
  type=push    sent on a timer (default 2 Hz): diagnostics + coarse nav map

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

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav_msgs.msg import OccupancyGrid, Odometry, Path
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
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:monospace;background:#111;color:#ddd;padding:12px;font-size:14px}
h2{color:#4af;margin-bottom:10px;font-size:18px}
.hdr{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}
.ws-dot{width:10px;height:10px;border-radius:50%;background:#f44;display:inline-block;margin-right:6px}
.ws-dot.ok{background:#4f4}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
.box{background:#1e1e1e;border:1px solid #333;border-radius:6px;padding:10px}
.box h3{color:#888;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.row{display:flex;justify-content:space-between;align-items:center;padding:3px 0;border-bottom:1px solid #2a2a2a}
.row:last-child{border-bottom:none}
.key{color:#888;font-size:12px}
.val{font-weight:bold;font-size:13px}
.ok  {color:#4f4}
.warn{color:#fa0}
.err {color:#f44}
.info{color:#4af}
.dim {color:#666}
#map-canvas{display:block;border:1px solid #444;image-rendering:pixelated;width:240px;height:240px}
.map-wrap{display:flex;flex-direction:column;align-items:center;gap:6px}
.map-legend{display:flex;gap:10px;font-size:11px;color:#888}
.swatch{width:12px;height:12px;display:inline-block;vertical-align:middle;margin-right:3px}
.footer{background:#1e1e1e;border:1px solid #333;border-radius:6px;padding:8px 10px;font-size:12px;color:#888;margin-top:10px}
</style>
</head>
<body>

<div class="hdr">
  <h2>TeamBowl Teleop</h2>
  <span><span class="ws-dot" id="ws-dot"></span><span id="ws-label">Connecting…</span></span>
</div>

<div class="grid">

  <!-- Left: diagnostics -->
  <div class="box">
    <h3>Diagnostics</h3>
    <div class="row"><span class="key">Gamepad</span><span class="val dim" id="gp-val">—</span></div>
    <div class="row"><span class="key">Dead-man RT</span><span class="val warn" id="dm-val">DISARMED</span></div>
    <div class="row"><span class="key">Mode</span><span class="val info" id="mode-val">—</span></div>
    <div class="row"><span class="key">Nav state</span><span class="val dim" id="nav-val">—</span></div>
    <div class="row"><span class="key">Goal (rel)</span><span class="val dim" id="goal-val">—</span></div>
    <div class="row"><span class="key">E-stop</span><span class="val ok" id="estop-val">NO</span></div>
    <div class="row"><span class="key">Stuck</span><span class="val ok" id="stuck-val">NO</span></div>
    <div class="row"><span class="key">Kill switch</span><span class="val ok" id="kill-val">NO</span></div>
    <div class="row"><span class="key">Lid</span><span class="val dim" id="lid-val">—</span></div>
    <div class="row"><span class="key">Odom XY</span><span class="val dim" id="odom-xy">—</span></div>
    <div class="row"><span class="key">Odom θ</span><span class="val dim" id="odom-th">—</span></div>
  </div>

  <!-- Right: nav map -->
  <div class="box">
    <h3>Nav Map (coarse)</h3>
    <div class="map-wrap">
      <canvas id="map-canvas" width="240" height="240"></canvas>
      <div class="map-legend">
        <span><span class="swatch" style="background:#e8e8e8"></span>free</span>
        <span><span class="swatch" style="background:#555"></span>unknown</span>
        <span><span class="swatch" style="background:#222;border:1px solid #555"></span>occupied</span>
        <span><span class="swatch" style="background:#e44;border-radius:50%"></span>robot</span>
        <span><span class="swatch" style="background:#4af"></span>goal</span>
      </div>
      <div id="map-age" style="font-size:11px;color:#666">No map yet</div>
    </div>
  </div>

</div>

<div class="footer">
  RT=arm &nbsp;|&nbsp; L-stick=position &nbsp;|&nbsp; R-stick X=heading &nbsp;|&nbsp;
  A=send goal &nbsp;|&nbsp; B=cancel &nbsp;|&nbsp; Menu=E-STOP
</div>

<script>
const wsUrl = 'ws://' + location.host + '/ws';
const N = 20;
let ws = null, gpIndex = null, sendInterval = null, prevDm = false;
let lastPush = null;

// ---- WebSocket ----
function connect() {
  ws = new WebSocket(wsUrl);
  ws.onopen  = () => { setWs(true); startSending(); };
  ws.onclose = () => { setWs(false); stopSending(); setTimeout(connect, 2000); };
  ws.onerror = () => {};
  ws.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data);
      if (d.type === 'push')   handlePush(d);
      if (d.type === 'status') handleStatus(d);
    } catch(_) {}
  };
}

function setWs(ok) {
  document.getElementById('ws-dot').className   = 'ws-dot' + (ok ? ' ok' : '');
  document.getElementById('ws-label').textContent = ok ? 'Connected' : 'Disconnected — retrying…';
}

// ---- Gamepad ----
window.addEventListener('gamepadconnected',    (e) => { gpIndex = e.gamepad.index; set('gp-val', e.gamepad.id.slice(0,30), 'ok'); });
window.addEventListener('gamepaddisconnected', ()  => { gpIndex = null; set('gp-val', 'Disconnected', 'err'); });

function startSending() { if (!sendInterval) sendInterval = setInterval(sendState, 50); }
function stopSending()  { if (sendInterval) { clearInterval(sendInterval); sendInterval = null; } }

function sendState() {
  if (gpIndex === null || !ws || ws.readyState !== WebSocket.OPEN) return;
  const gp = navigator.getGamepads()[gpIndex];
  if (!gp) return;
  const axes    = Array.from(gp.axes);
  const buttons = gp.buttons.map(b => b.pressed ? 1 : 0);
  const dm = (axes.length > 5 ? axes[5] : 0) > 0.5;
  if (dm !== prevDm) {
    set('dm-val', dm ? 'ARMED' : 'DISARMED', dm ? 'ok' : 'warn');
    prevDm = dm;
  }
  ws.send(JSON.stringify({axes, buttons}));
}

// ---- Message handlers ----
function handleStatus(d) {
  set('nav-val',  d.state,  d.state === 'RUNNING' ? 'info' : 'dim');
  set('goal-val', 'x=' + d.goal_x + ' y=' + d.goal_y + ' θ=' + d.goal_theta_deg + '°', 'dim');
}

function handlePush(d) {
  lastPush = d;
  set('mode-val',  d.mode  || '—', d.mode === 'driving' ? 'info' : 'warn');
  set('nav-val',   d.state || '—', d.state === 'RUNNING' ? 'info' : 'dim');
  setBool('estop-val', d.estop,       true);
  setBool('stuck-val', d.stuck,       true);
  setBool('kill-val',  d.kill_switch, true);
  set('lid-val',   d.lid  || '—', 'dim');
  if (d.odom_x != null) {
    set('odom-xy', '(' + d.odom_x + ', ' + d.odom_y + ')', 'dim');
    set('odom-th', d.odom_theta_deg + '°', 'dim');
  }
  if (d.map) drawMap(d);
  document.getElementById('map-age').textContent = 'Updated ' + new Date().toLocaleTimeString();
}

function set(id, text, cls) {
  const el = document.getElementById(id);
  el.textContent = text;
  el.className = 'val ' + (cls || 'dim');
}
function setBool(id, val, alarmOnTrue) {
  // alarmOnTrue=true means val=true is bad (estop, stuck, kill)
  set(id, val ? 'YES' : 'NO', val ? (alarmOnTrue ? 'err' : 'ok') : (alarmOnTrue ? 'ok' : 'warn'));
}

// ---- Map canvas ----
function drawMap(d) {
  const canvas = document.getElementById('map-canvas');
  const ctx    = canvas.getContext('2d');
  const sz     = canvas.width / N;   // pixels per coarse cell (12px)

  // Background
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Grid cells
  for (let row = 0; row < N; row++) {
    for (let col = 0; col < N; col++) {
      const c = d.map[row * N + col];
      ctx.fillStyle = c === '0' ? '#e8e8e8'   // free
                    : c === '1' ? '#1a1a1a'   // occupied
                    :             '#505050';   // unknown
      // Row 0 in OccupancyGrid = bottom of map (y+ up in ROS) → flip
      ctx.fillRect(col * sz, (N - 1 - row) * sz, sz, sz);
    }
  }

  // Grid lines (faint)
  ctx.strokeStyle = '#2a2a2a';
  ctx.lineWidth = 0.5;
  for (let i = 1; i < N; i++) {
    ctx.beginPath(); ctx.moveTo(i*sz, 0); ctx.lineTo(i*sz, canvas.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i*sz); ctx.lineTo(canvas.width, i*sz); ctx.stroke();
  }

  if (d.map_cell_m == null || d.odom_x == null) return;

  const ox  = d.map_origin_x, oy = d.map_origin_y, cm = d.map_cell_m;

  // Helper: odom meters → canvas pixels
  function toCanvas(wx, wy) {
    return [
      ((wx - ox) / cm) * sz,
      (N - (wy - oy) / cm) * sz,
    ];
  }

  // Goal marker (blue cross) — only when nav is running or goal is non-zero
  if (d.goal_odom_x != null) {
    const [gx, gy] = toCanvas(d.goal_odom_x, d.goal_odom_y);
    const r = sz * 0.7;
    ctx.strokeStyle = '#4af';
    ctx.lineWidth   = 2;
    ctx.beginPath(); ctx.moveTo(gx-r, gy); ctx.lineTo(gx+r, gy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(gx, gy-r); ctx.lineTo(gx, gy+r); ctx.stroke();
  }

  // Robot dot (red) + heading line
  const [rx, ry] = toCanvas(d.odom_x, d.odom_y);
  const rr = Math.max(sz * 0.55, 5);
  ctx.fillStyle = '#e44';
  ctx.beginPath(); ctx.arc(rx, ry, rr, 0, Math.PI*2); ctx.fill();

  if (d.odom_theta_deg != null) {
    const th = d.odom_theta_deg * Math.PI / 180;
    // In canvas: x right, y down. ROS: x forward=right, y left=up → canvas y flipped
    const hx = rx + Math.cos(th)  * rr * 2.5;
    const hy = ry - Math.sin(th)  * rr * 2.5;
    ctx.strokeStyle = '#fa0';
    ctx.lineWidth   = 2;
    ctx.beginPath(); ctx.moveTo(rx, ry); ctx.lineTo(hx, hy); ctx.stroke();
  }
}

connect();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Helpers (from trajectory_test.py)
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
        self.declare_parameter('map_push_rate_hz', 2.0)
        self.declare_parameter('coarse_map_size', 20)
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
        self.declare_parameter('costmap_topic', '/global_costmap/costmap')
        self.declare_parameter('stuck_topic', '/robot_stuck')
        self.declare_parameter('kill_switch_topic', '/kill_switch')
        self.declare_parameter('lid_state_topic', '/lid_state')
        self.declare_parameter('preview_topic', '/steamdeck/goal_preview')
        self.declare_parameter('auto_set_driving_mode', True)
        self.declare_parameter('driving_mode_delay_s', 5.0)

        p = self.get_parameter
        self._ws_host           = str(p('ws_host').value)
        self._ws_port           = int(p('ws_port').value)
        self._map_push_rate     = float(p('map_push_rate_hz').value)
        self._coarse_map_size   = int(p('coarse_map_size').value)
        self._goal_scale        = float(p('goal_scale_m_per_tick').value)
        self._yaw_scale         = float(p('yaw_scale_rad_per_tick').value)
        self._max_dist          = float(p('max_goal_dist_m').value)
        self._dm_axis           = int(p('dead_man_axis').value)
        self._dm_thresh         = float(p('dead_man_threshold').value)
        self._fwd_axis          = int(p('forward_axis').value)
        self._str_axis          = int(p('strafe_axis').value)
        self._yaw_axis          = int(p('yaw_axis').value)
        self._confirm_btn       = int(p('confirm_button').value)
        self._cancel_btn        = int(p('cancel_button').value)
        self._estop_btn         = int(p('estop_button').value)
        self._planner_action    = str(p('planner_action_name').value)
        self._controller_action = str(p('controller_action_name').value)
        self._planner_id        = str(p('planner_id').value)
        self._controller_id     = str(p('controller_id').value)
        self._goal_checker_id   = str(p('goal_checker_id').value)

        # --- State ---
        self._state = self.IDLE
        self._robot_mode = 'off'
        self._estop = False
        self._robot_stuck = False
        self._kill_switch = False
        self._lid_state = 'unknown'
        self._latest_odom: Odometry | None = None
        self._latest_costmap: OccupancyGrid | None = None

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
        reliable_tl = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # --- Subscriptions ---
        self.create_subscription(String,        p('mode_topic').value,        self._mode_cb,        reliable_tl)
        self.create_subscription(Odometry,      p('odom_topic').value,        self._odom_cb,        best_effort)
        self.create_subscription(Bool,          p('estop_topic').value,       self._estop_cb,       best_effort)
        self.create_subscription(OccupancyGrid, p('costmap_topic').value,     self._costmap_cb,     reliable_tl)
        self.create_subscription(Bool,          p('stuck_topic').value,       self._stuck_cb,       best_effort)
        self.create_subscription(Bool,          p('kill_switch_topic').value, self._kill_switch_cb, best_effort)
        self.create_subscription(String,        p('lid_state_topic').value,   self._lid_cb,         reliable)

        # --- Publishers ---
        self._preview_pub  = self.create_publisher(PoseStamped, p('preview_topic').value,  10)
        self._mode_set_pub = self.create_publisher(String,      p('mode_set_topic').value, 10)
        self._estop_pub    = self.create_publisher(Bool,        p('estop_topic').value,    10)

        # --- Action clients ---
        self._planner_client    = ActionClient(self, ComputePathToPose, self._planner_action)
        self._controller_client = ActionClient(self, FollowPath, self._controller_action)

        # --- Timers ---
        rate = float(p('joy_rate_hz').value)
        self.create_timer(1.0 / max(rate, 1.0), self._joy_tick)

        if p('auto_set_driving_mode').value:
            delay = float(p('driving_mode_delay_s').value)
            self._mode_timer = self.create_timer(delay, self._auto_set_driving_mode)

        # --- WebSocket server in background daemon thread ---
        self._ws_thread = threading.Thread(target=self._run_ws_server, daemon=True)
        self._ws_thread.start()

        self.get_logger().info(
            f'steamdeck_ws_teleop ready | ws://0.0.0.0:{self._ws_port} | '
            f'open http://ROBOT_IP:{self._ws_port} in Steam Deck browser'
        )

    # ------------------------------------------------------------------
    # WebSocket server (background daemon thread)
    # ------------------------------------------------------------------

    def _run_ws_server(self):
        asyncio.run(self._ws_main())

    async def _ws_main(self):
        async def process_request(path, request_headers):
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
        push_task = asyncio.create_task(self._push_loop(websocket))
        try:
            async for raw in websocket:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                with self._lock:
                    self._joy_state = data
                status = {
                    'type':          'status',
                    'state':         self._state,
                    'mode':          self._robot_mode,
                    'goal_x':        round(self._goal_x, 2),
                    'goal_y':        round(self._goal_y, 2),
                    'goal_theta_deg': round(math.degrees(self._goal_theta), 1),
                }
                await websocket.send(json.dumps(status))
        except websockets.ConnectionClosed:
            pass
        finally:
            push_task.cancel()
        self.get_logger().info(f'Steam Deck disconnected from {remote}')
        with self._lock:
            self._joy_state = {'axes': [0.0] * 10, 'buttons': [0] * 12}

    async def _push_loop(self, websocket):
        interval = 1.0 / max(self._map_push_rate, 0.1)
        while True:
            await asyncio.sleep(interval)
            try:
                await websocket.send(json.dumps(self._build_push_msg()))
            except websockets.ConnectionClosed:
                break

    def _build_push_msg(self) -> dict:
        odom = self._latest_odom
        odom_x = odom_y = odom_theta = None
        if odom is not None:
            odom_x     = round(odom.pose.pose.position.x, 2)
            odom_y     = round(odom.pose.pose.position.y, 2)
            odom_theta = round(math.degrees(_yaw_from_quat(odom.pose.pose.orientation)), 1)

        # Goal in odom frame (for map marker)
        goal_odom_x = goal_odom_y = None
        if odom is not None and (self._goal_x != 0.0 or self._goal_y != 0.0):
            g = self._compute_odom_goal()
            goal_odom_x = round(g.pose.position.x, 2)
            goal_odom_y = round(g.pose.position.y, 2)

        msg: dict = {
            'type':        'push',
            'mode':        self._robot_mode,
            'state':       self._state,
            'estop':       self._estop,
            'stuck':       self._robot_stuck,
            'kill_switch': self._kill_switch,
            'lid':         self._lid_state,
            'odom_x':      odom_x,
            'odom_y':      odom_y,
            'odom_theta_deg': odom_theta,
            'goal_odom_x': goal_odom_x,
            'goal_odom_y': goal_odom_y,
            'map':         self._encode_coarse_map(),
        }

        g = self._latest_costmap
        if g is not None:
            N = self._coarse_map_size
            msg['map_origin_x'] = round(g.info.origin.position.x, 3)
            msg['map_origin_y'] = round(g.info.origin.position.y, 3)
            msg['map_cell_m']   = round(g.info.resolution * g.info.width / N, 3)

        return msg

    def _encode_coarse_map(self) -> str | None:
        grid = self._latest_costmap
        if grid is None:
            return None
        N = self._coarse_map_size
        w, h = grid.info.width, grid.info.height
        if w < N or h < N:
            return None
        block_w = w // N
        block_h = h // N
        data = np.frombuffer(bytes(grid.data), dtype=np.int8).reshape((h, w))
        # Crop to exact multiple, then max-pool to NxN
        sub = data[:block_h * N, :block_w * N]
        coarse = sub.reshape(N, block_h, N, block_w).max(axis=(1, 3))
        # Encode: unknown=-1→'?', free=0→'0', occupied>0→'1'
        flat = coarse.ravel()
        result = bytearray(N * N)
        for i, v in enumerate(flat):
            result[i] = ord('?') if v < 0 else ord('0') if v == 0 else ord('1')
        return result.decode('ascii')

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

    def _costmap_cb(self, msg: OccupancyGrid):
        self._latest_costmap = msg

    def _stuck_cb(self, msg: Bool):
        self._robot_stuck = bool(msg.data)

    def _kill_switch_cb(self, msg: Bool):
        self._kill_switch = bool(msg.data)

    def _lid_cb(self, msg: String):
        self._lid_state = msg.data.strip()

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

        # Dead-man (right trigger)
        dead_man = _axis(self._dm_axis) > self._dm_thresh

        # Falling edge: trigger released → reset goal accumulator
        if self._dead_man_prev and not dead_man:
            self._goal_x = self._goal_y = self._goal_theta = 0.0
        self._dead_man_prev = dead_man

        # Accumulate goal while armed
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

        # Button edge detection
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

        # Publish goal preview for Foxglove
        if self._latest_odom is not None:
            self._preview_pub.publish(self._compute_odom_goal())

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_confirm_pressed(self):
        if self._robot_mode != 'driving':
            self.get_logger().warn(f'A pressed but mode is "{self._robot_mode}" — need "driving"')
            return
        if self._estop:
            self.get_logger().warn('A pressed but estop is active')
            return
        if self._latest_odom is None:
            self.get_logger().error('No odometry yet — cannot send goal')
            return

        goal_pose = self._compute_odom_goal()
        self.get_logger().info(
            f'Sending goal: ({goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f})'
        )

        if self._state == self.RUNNING:
            self._active_goal = goal_pose
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
        estop_msg = Bool(); estop_msg.data = True
        self._estop_pub.publish(estop_msg)
        mode_msg = String(); mode_msg.data = 'off'
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

        gx     = rx + math.cos(yaw) * self._goal_x - math.sin(yaw) * self._goal_y
        gy     = ry + math.sin(yaw) * self._goal_x + math.cos(yaw) * self._goal_y
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
        msg = String(); msg.data = 'driving'
        self._mode_set_pub.publish(msg)
        self.get_logger().info('Auto-set robot mode to "driving"')

    # ------------------------------------------------------------------
    # Nav2 action chain (mirrors trajectory_test.py)
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
        self._planner_client.send_goal_async(goal).add_done_callback(
            self._on_planner_goal_response
        )

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
                self._request_path(self._active_goal)
            return
        if self._controller_cancel_in_flight:
            return
        self._controller_cancel_in_flight = True
        self._controller_goal_handle.cancel_goal_async().add_done_callback(
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
        self._controller_client.send_goal_async(goal).add_done_callback(
            self._on_controller_goal_response
        )

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
