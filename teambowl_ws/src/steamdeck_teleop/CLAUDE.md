# steamdeck_teleop

## 2026-04-28 — Added SHUTDOWN button (mega kill) to phone and full UIs

**`steamdeck_ws_teleop.py`**: Added `⏻ SHUTDOWN` button to `_HTML_PHONE` (below KILL) and `_HTML_FULL` (Robot Mode panel). Button sends `{type:'mega_kill'}` with a browser confirm dialog. Handler (`_mega_kill`) runs in a daemon thread: publishes `/estop True` + mode `off`, waits 300ms, brings down `can0`/`can1` via `ip link set`, then sends `SIGINT` to PID 1 (ros2 launch, via `exec` in entrypoint). This triggers graceful ROS shutdown — camera driver and CAN driver exit cleanly — and the Docker container exits naturally. Container must be `privileged: true` for CAN shutdown to work.

## 2026-04-28 — Virtual joystick replacing D-pad in Phone UI

- **`steamdeck_ws_teleop.py`** (`_HTML_PHONE` only): Replaced the 4-button D-pad with a drag virtual joystick. `#joystick` (circle, min(80vw,300px)) contains `#joystick-knob` (34% inner circle). Pointer events handle drag: `vx = -dy/maxR * 0.3`, `wz = -dx/maxR * 0.8`. Knob follows finger, snaps back on release. Sends `teleop_vel` at 100 ms intervals while held. Rescue and full UIs unchanged.

## 2026-04-28 — Fixed person detection topic; added Target ID row

- **`config/steamdeck_teleop.yaml`**: Fixed `user_valid_topic` from `/user_valid` → `/yolo26/user_valid` (wrong topic was why Person always showed "—"). Added `target_id_topic: /yolo26/target_id`.
- **`steamdeck_ws_teleop.py`**: Added `Int32` import. Added `target_id_topic` parameter, `_target_id` state (default -1), `_target_id_cb` subscription. `_build_push_msg()` now includes `target_id`. Both `_HTML_PHONE` and `_HTML_FULL` diagnostics panels now have a **Target ID** row — shows the track ID (blue) when locked, "—" (dim) when no target.

## 2026-04-21 — ENABLE button now properly clears kill-switch latch

- **`steamdeck_ws_teleop.py`**: Added `_clear_estop_pub` publisher on `/clear_estop`. On `clear_estop` web message, publishes `Bool(True)` to `/clear_estop` (in addition to existing `/estop false`). This signals `system_health` to clear its kill-switch latch so the heartbeat tick can no longer silently override it.

## 2026-04-21 — Added D-pad and Teleop button to Phone UI

**`steamdeck_teleop/steamdeck_ws_teleop.py`** (`_HTML_PHONE` only):
- Added TELEOP button (blue, `set_mode:driving`) side-by-side with AUTON in a 2-column grid.
- Added D-pad (4-direction, identical velocities to rescue UI: FWD=0.3, BACK=-0.15 m/s, LEFT/RIGHT ±0.6 rad/s) below KILL button, above diagnostics.
- Added `startDrive(vx, wz, el)` / `stopDrive(el)` JS + `_driveTimer`/`_curVx`/`_curWz` state.
- Added `.dpad`, `.dpad-btn`, `.dpad-empty`, `.btn-teleop` CSS classes.
- Added `touch-action:manipulation` to body to prevent scroll interference.
- No Python changes — `teleop_vel` handler and publisher already existed.

## 2026-04-21 — Added Rescue Teleop UI (ui_mode='rescue')

**`steamdeck_teleop/steamdeck_ws_teleop.py`**:
- Added `_HTML_RESCUE`: minimal phone-first page with ENABLE+KILL buttons and a 4-direction D-pad (↑FWD / ←LEFT | →RIGHT / ↓BACK). ENABLE sends both `clear_estop` and `set_mode:driving`. Hold-to-move via pointer events — `pointerdown` starts 100 ms `setInterval` sending `{type:'teleop_vel', vx, wz}`; `pointerup`/`pointercancel` clears interval and sends zero. Velocities: FWD=0.3, BACK=-0.15 m/s, LEFT/RIGHT ±0.6 rad/s.
- Added `teleop_vel_topic` parameter (default `/cmd_vel_auto`).
- Added `_teleop_vel_pub` publisher (`Twist`).
- `_handle_panel_cmd` handles `type:'teleop_vel'`: publishes `Twist(linear.x=vx, angular.z=wz)`.
- `ui_mode='rescue'` selects `_HTML_RESCUE`; fallback chain: rescue → phone → full.

**`config/steamdeck_teleop.yaml`**: added `teleop_vel_topic: /cmd_vel_auto`.

**Usage**: launch with `steamdeck_ui:=rescue` or set `ui_mode: rescue` in YAML. Robot must be in `driving` mode for motion (vel goes through vel_cmd_mux → driving_controller).

## 2026-04-20 — ENABLE button now clears e-stop instead of setting driving mode

**`steamdeck_ws_teleop.py`**: Phone UI ENABLE button changed from `{type:'set_mode', mode:'driving'}` to `{type:'clear_estop'}`. Added `clear_estop` handler in `_handle_panel_cmd`: publishes `Bool(False)` to `/estop`. Previously there was no way to clear the estop from the web UI after hitting KILL — the robot stayed stopped even after re-setting a mode.

## 2026-04-20 — Added Person detection indicator to diagnostics panel

**`steamdeck_ws_teleop.py`**:
- Added `user_valid_topic` parameter (default `/user_valid`).
- Added `_user_valid` state + `_user_valid_cb` subscription (BEST_EFFORT Bool).
- `_build_push_msg()` includes `user_valid`.
- Both `_HTML_PHONE` and `_HTML_FULL` diagnostics: added **Person** row — green "YES" when detected, red "NO" when not.
- Both `handlePush` JS blocks: `setBool('person-val', d.user_valid, false)`.

**`config/steamdeck_teleop.yaml`**: added `user_valid_topic: /user_valid`.

## 2026-04-20 — Phone UI: changed OPEN LID → TOGGLE LID

**`steamdeck_ws_teleop.py`** `_HTML_PHONE`: changed lid button from `cmd:'open'` to
`cmd:'toggle'` and label from `OPEN LID` to `TOGGLE LID`.

## 2026-04-20 — Pitch fallover warning banner in both UIs

**`steamdeck_teleop/steamdeck_ws_teleop.py`**:
- Added `<div id="pitch-warn">` banner (hidden by default) to both `_HTML_PHONE` and `_HTML_FULL`.
- `handlePush`: when `|theta_deg| > 15°` shows orange banner "⚠ PITCH X° — NEAR FALLOVER"; turns dark red when `> 20°`; hidden when safe.
- Banner appears at the top of the page above all controls for maximum visibility.

## 2026-04-20 — Added Reset Odom button to both UIs

**`steamdeck_teleop/steamdeck_ws_teleop.py`**:
- Added `set_pose_topic` parameter (default `/set_pose`).
- Added `_set_pose_pub` publisher (`PoseWithCovarianceStamped`).
- `_handle_panel_cmd` handles `type: 'reset_odom'`: publishes a zero-pose `PoseWithCovarianceStamped` to `/set_pose` (resets the `robot_localization` EKF to the odom origin).
- `_HTML_PHONE`: added **RESET ODOM** button (amber/brown, full-width, between OPEN LID and KILL).
- `_HTML_FULL`: added **⟳ Reset Odom** button in the Robot Mode panel.
- Imported `PoseWithCovarianceStamped` from geometry_msgs.

**`config/steamdeck_teleop.yaml`**: added `set_pose_topic: /set_pose`.

## 2026-04-20 — Replaced Balance/Driver/VESC gains panels with Driving Gains panel

**`steamdeck_teleop/steamdeck_ws_teleop.py`**:
- Removed Balance Gains, Motor Driver Gains, and VESC Gains panels from both `_HTML_PHONE` and `_HTML_FULL`.
- Added **Driving Gains** panel to both UIs: fields for `kp_vel, ki_vel, kd_vel, kp_pitch, kd_pitch, ki_pitch, kp_yaw, ki_yaw, kd_yaw, kff_decel, theta_eq_offset`. Live readout: θ, v, ω from `/driving_gains_echo`. Receive + Send buttons.
- Added subscription to `/driving_gains_echo` → `self._driving_gains_echo`.
- Added publisher to `/driving_gains`.
- `_build_push_msg()` includes `driving_gains`.
- `_handle_panel_cmd()` handles `type: 'driving_gains'`.
- Added `driving_gains_topic` / `driving_gains_echo_topic` parameters.

**`config/steamdeck_teleop.yaml`**: added `driving_gains_topic` and `driving_gains_echo_topic`.



## 2026-04-20 — Added Balance, Driver, and VESC gains panels to both UIs

**`steamdeck_teleop/steamdeck_ws_teleop.py`**:
- Added subscriptions to `/driver_gains_echo` and `/vesc_gains_echo`; state vars `_driver_gains_echo`, `_vesc_gains_echo`.
- Added publishers `_driver_gains_pub` (`/driver_gains`) and `_vesc_gains_pub` (`/vesc_gains`).
- `_build_push_msg()` includes `driver_gains` and `vesc_gains` fields.
- `_handle_panel_cmd()` handles `type: 'driver_gains'` and `type: 'vesc_gains'`.
- **Balance Gains panel** (both UIs): added `ki_yaw` field; renamed "Apply Gains" → "Send"; added "Receive" button (fills from last push).
- **Driver Gains panel** (NEW, both UIs): auto-populated table of per-joint kp/kd with Receive + Send buttons. Table is built dynamically from the first `/driver_gains_echo` push.
- **VESC Gains panel** (NEW, both UIs): kp_v, ki_v, kp_w, ki_w, integral_max inputs with architecture info text, Receive + Send buttons. Shows live `v_measured` and `w_measured` from `/vesc_gains_echo`.
- All three panels always visible; phone UI now includes all three panels.

**`config/steamdeck_teleop.yaml`**: added `driver_gains_echo_topic`, `driver_gains_topic`, `vesc_gains_echo_topic`, `vesc_gains_topic` params.

## 2026-04-20 — Added motor cmd + controller input display to web UI

**`steamdeck_ws_teleop.py`**:
- Subscribes to `/cmd_vel` (motor output) and `/cmd_vel_safe` (balance controller input), both BEST_EFFORT.
- `_build_push_msg()` includes `ctrl_in_vx`, `ctrl_in_wz` (from `/cmd_vel_safe`) and `motor_vx`, `motor_wz` (from `/cmd_vel`).
- Both `_HTML_FULL` and `_HTML_PHONE` show "Ctrl in v/ω" and "Motor cmd v/ω" rows in the diagnostics panel.
- `Twist` added to geometry_msgs imports.

**`config/steamdeck_teleop.yaml`**: added `cmd_vel_topic` and `cmd_vel_safe_topic` params.

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
