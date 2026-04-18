# steamdeck_teleop

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

## 2026-04-17 — Initial implementation

- `steamdeck_ws_teleop.py`: WebSocket server + Nav2 goal sender. Replaces the
  original ROS2-on-Steam-Deck approach (which required CycloneDDS unicast config
  on CMU mesh). Browser-based gamepad via Web Gamepad API — no software on Steam Deck.
- `config/steamdeck_teleop.yaml`: All parameters with comments.
- `launch/steamdeck_ws.launch.py`: Single-node launch for the robot.
- `Dockerfile` + `Dockerfile.laptop`: Added `websockets` pip install.
