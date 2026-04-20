# steamdeck_teleop

## 2026-04-20 — Phone UI + ui_mode parameter; estop panel command

**`steamdeck_ws_teleop.py`**:
- Added `ui_mode` parameter (default `'phone'`). Selects between `_HTML_PHONE` and `_HTML_FULL` at startup.
- `_HTML_PHONE`: phone-first page — 3 full-width buttons (ENABLE green / OPEN LID blue / KILL red, `min-height:18vh`, `font-size:clamp(2.5rem,10vw,5rem)`) + diagnostics panel (mode, estop, stuck, kill_switch, lid, battery, legs, planner). No gamepad, trajectory, or gains.
- `_HTML_FULL`: existing full UI (renamed from `_HTML_PAGE`, aliased as `_HTML_PAGE = _HTML_FULL` for compatibility).
- Added `type:'estop'` branch in `_handle_panel_cmd`: publishes `Bool(True)` to `/estop` and `String('off')` to `/robot_mode_set`. Used by KILL button in phone UI.
- `_ws_main` now serves `self._html` (set from `ui_mode` in `__init__`) instead of hardcoded `_HTML_PAGE`.

**Usage:**
- `launch.sh` (bringup default) → phone UI at `http://ROBOT_IP:8888`
- `launch_debug.sh` (bringup with `steamdeck_ui:=full`) → full UI at same address

## Overview

ROS2 Python package. Runs a WebSocket server on the robot so the Steam Deck (or any device)
can drive Nav2 navigation goals from a browser tab — no ROS2 installation on the Steam Deck.

Mirrors how Foxglove Bridge works (WebSocket over TCP) but for gamepad input instead of
topic visualization.

## Node: `steamdeck_ws_teleop`

**File:** `steamdeck_teleop/steamdeck_ws_teleop.py`

**Port:** 8888 (Foxglove is 8765 — no conflict)

**Usage:**
1. Start robot: `ros2 launch bringup trajectory_test.launch.py`
2. Start this node: `ros2 launch steamdeck_teleop steamdeck_ws.launch.py`
3. On Steam Deck browser: navigate to `http://ROBOT_IP:8888`
4. Press any button on the Steam Deck to activate the gamepad in the browser

**Controls (Web Gamepad API, Xbox/Steam Deck layout):**

| Control | Action |
|---------|--------|
| Hold RT (right trigger) | Arm goal accumulation |
| Left stick | Move goal position (forward/back/strafe) |
| Right stick X | Rotate goal heading |
| Release RT | Reset goal accumulator to origin |
| A button | Send current goal to Nav2 |
| B button | Cancel active navigation (or reset goal if idle) |
| Menu button | E-stop + set mode "off" |

## Architecture

The node runs a `websockets` Python server in a background daemon thread. A 20 Hz rclpy timer
(`_joy_tick`) reads the latest gamepad state (written by the WS thread) via a `threading.Lock`,
accumulates the goal position, and drives the Nav2 action chain.

Nav2 uses `ComputePathToPose` + `FollowPath` directly (not `NavigateToPose`) because
`bt_navigator` is not in the bringup lifecycle manager. Action chain mirrors `trajectory_test.py`.

The WebSocket server also serves the HTML control page on plain HTTP GET requests (same port 8888),
so no separate HTTP server is needed.

## Dependencies

- `websockets` Python library (pip) — added to `Dockerfile` and `Dockerfile.laptop`
- `nav2_msgs` (ROS2) — for `ComputePathToPose` and `FollowPath` actions

## Config

All parameters in `config/steamdeck_teleop.yaml`. Key tuning params:
- `goal_scale_m_per_tick`: meters moved per 20 Hz tick (default 0.02 → 0.4 m/s max)
- `yaw_scale_rad_per_tick`: heading change per tick (default 0.03 → ~0.6 rad/s max)
- `dead_man_axis/threshold`: which axis is the dead-man and its threshold
- `confirm_button`, `cancel_button`, `estop_button`: button indices (verify with browser console)

## Button index verification

If buttons don't match, open `http://ROBOT_IP:8888` in browser, open DevTools console,
and watch the axis/button arrays printed by the gamepad polling loop. Update
`steamdeck_teleop.yaml` to match.

## 2026-04-19 — Fixed websockets 16 API incompatibility

websockets 16 changed `process_request(path, request_headers)` → `process_request(connection, request)`
where `request` is a `Request` object (use `request.headers.get()`). Return type changed from tuple
to `WsResponse(status_int, reason_str, WsHeaders([...]), body_bytes)`. Handler signature changed from
`_ws_handler(websocket, path)` → `_ws_handler(websocket)`. Switched `websockets.serve` → `ws_serve`
from `websockets.asyncio.server`.

## 2026-04-19 — Removed duplicate Nav2 action clients; routed A-button through trajectory_test

**`steamdeck_ws_teleop.py`**: Removed the entire Nav2 action chain (`_planner_client`, `_controller_client`, `_request_path`, `_on_planner_*`, `_on_controller_*`, etc.). Two action clients on the same server caused "Ignoring unexpected goal response" cross-talk. Now:
- A-button (`_on_confirm_pressed`) publishes the absolute odom goal to `/trajectory_goal` then `/trajectory_cmd go`
- B-button stop publishes to `/trajectory_cmd stop`
- Planner readiness check uses `get_service_names_and_types()` (no action client needed)
- `auto_set_driving_mode: false` — mode must be set manually via web UI buttons (prevents balance controller from auto-starting)

## 2026-04-19 — Added Planner/Legs status indicators to web UI

**`steamdeck_teleop/steamdeck_ws_teleop.py`**:
- Added `_leg_running` state + subscription to `/leg_controller_running` (Bool, TRANSIENT_LOCAL)
- `_build_push_msg()` now includes `planner_ready` (from `_planner_client.server_is_ready()`) and `legs_running`
- HTML Diagnostics box: two new rows "Planner" and "Legs" — green when ready/running, red otherwise
- `handlePush()` JS: calls `setBool('planner-val', ...)` and `setBool('legs-val', ...)`

New param: `leg_running_topic` (default `/leg_controller_running`).

**`locomotion/driving_leg_controller.py`**: publishes `Bool` on `/leg_controller_running` at 2 Hz
(TRANSIENT_LOCAL) reflecting `self._running`. Imported `rclpy.qos` for TRANSIENT_LOCAL profile.

## 2026-04-19 — Extended web UI with full robot control panel

**`steamdeck_ws_teleop.py`** — added four control panels to the browser UI:
- **Mode panel**: Driving / Balance / Auton / Off buttons → publishes to `/robot_mode_set`
- **Lid panel**: Open / Close / Toggle buttons → publishes to `/lid_command`
- **Trajectory Goal panel**: x/y/θ inputs + relative checkbox + Go/Stop/Reset buttons
  → Go sends `traj_goal` then `traj_cmd go`; Stop/Reset send `traj_cmd stop/reset`
- **Balance Gains panel**: 12 editable gain inputs pre-filled from `/balance_gains_echo` push,
  Apply button → publishes JSON dict to `/balance_gains`; shows live θ and v_actual

New publishers: `_traj_goal_pub` (`/trajectory_goal`), `_traj_cmd_pub` (`/trajectory_cmd`),
`_lid_cmd_pub` (`/lid_command`), `_balance_gains_pub` (`/balance_gains`).

New subscriptions: `/balance_gains_echo` → `self._balance_gains_echo`,
`/trajectory_status` → `self._traj_status` (both included in 2 Hz push to browser).

New `_handle_panel_cmd(data)` routes `type` field: `set_mode`, `traj_goal`, `traj_cmd`,
`lid_cmd`, `balance_gains`.

**`config/steamdeck_teleop.yaml`** — added six new topic name params:
`trajectory_goal_topic`, `trajectory_cmd_topic`, `lid_command_topic`,
`balance_gains_topic`, `balance_gains_echo_topic`, `trajectory_status_topic`.

## 2026-04-17 — Initial implementation

- `steamdeck_ws_teleop.py`: WebSocket server + Nav2 goal sender. Replaces the
  original ROS2-on-Steam-Deck approach (which required CycloneDDS unicast config
  on CMU mesh). Browser-based gamepad via Web Gamepad API — no software on Steam Deck.
- `config/steamdeck_teleop.yaml`: All parameters with comments.
- `launch/steamdeck_ws.launch.py`: Single-node launch for the robot.
- `Dockerfile` + `Dockerfile.laptop`: Added `websockets` pip install.
