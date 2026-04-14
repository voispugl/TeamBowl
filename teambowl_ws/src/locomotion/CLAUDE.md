# locomotion

## 2026-04-14 — Upgraded inner pitch loop to PID with 2-second sliding window integral

**`balance_controller`** — `locomotion/balance_controller.py`, **`config/balance_controller.yaml`**
- Added `ki_pitch` parameter (default 0.0, live-tunable via `/balance_gains`)
- Inner loop is now PID: `u = -(kp*err + ki*integral + kd*theta_dot)`
- Integral uses a `collections.deque` of `(ros_time_sec, contribution)` pairs pruned to
  the last 2.0 seconds — prevents windup from long-term drift
- Deque is cleared on mode exit (leaving balance mode) and on estop

## 2026-04-14 — Rewrote balance_controller from LQR → cascaded PID

**`balance_controller`** — `locomotion/balance_controller.py`
Full rewrite. Replaced LQR gain matrix with a simpler cascaded PID architecture.
- Outer PI (50 Hz): velocity error → `theta_ref` (lean angle setpoint)
- Inner PD (50 Hz): `(theta - theta_ref)` + `kd_pitch * theta_dot` → wheel velocity cmd (upgraded to PID same day)
- Removed: LQR gain matrix, mass_mode scheduling (light/nominal/heavy), `_get_gains()`
- Added: `kp_pitch`, `kd_pitch`, `ki_pitch` parameters (all live-tunable via `/balance_gains` JSON)
- All other logic unchanged: mode switching, estop, fallover detection, Foxglove echo

**`config/balance_controller.yaml`** — updated gain names:
- Removed: `k_theta`, `k_theta_dot`, `k_v`, `mass_mode`, `k_*_light`, `k_*_heavy`
- Added: `kp_pitch: 60.0`, `kd_pitch: 8.0`
- Outer PI (`kp_vel`, `ki_vel`) and safety limits unchanged

Tune starting from `kp_pitch=60, kd_pitch=8`. Raise `kp_pitch` until oscillation
then back off 30%; raise `kd_pitch` to damp. Use Foxglove `/balance_gains` JSON
or `ros2 param set` for live adjustments without restarting.

---

## 2026-04-13 — Added balance controller, wheel odometry, EKF config

### New nodes

**`balance_controller`** — `locomotion/balance_controller.py`
~~LQR inner loop + PI velocity outer loop~~ (replaced 2026-04-14 with PID — see above).
- In `balance` mode: uses LQR on `[theta, theta_dot, v_error]` → symmetric wheel
  velocity cmd. Outer PI generates `theta_ref` from velocity error.
- In all other modes: passthrough (`/cmd_vel_safe` → `/cmd_vel` unchanged).
- Subscribes: `/odometry/filtered`, `/imu/data`, `/cmd_vel_safe`, `/robot_mode`, `/estop`
- Publishes: `/cmd_vel`, `/estop` (on fallover)
- All gains are live-tunable via `ros2 param set`.
- Gain scheduling for three mass setpoints: `light` / `nominal` / `heavy`.

**`wheel_odom`** — `locomotion/wheel_odom.py`
Dead-reckoning odometry from `/cmd_vel` → `/odom_wheels` (nav_msgs/Odometry).
Used as wheel odometry input to the EKF.

### New config files

- **`config/balance_controller.yaml`**: All gains for `balance_controller` and
  `wheel_odom` nodes. Tune `k_theta`, `k_theta_dot`, `k_v` from `tune_lqr.py` output.
- **`config/ekf.yaml`**: `robot_localization` EKF config. Fuses `/imu/data` +
  `/odom_wheels` → `/odometry/filtered` at 50 Hz.

### Pipeline change: collision_guard output renamed

`collision_guard` now publishes to `/cmd_vel_safe` (was `/cmd_vel`). The
`balance_controller` sits downstream and publishes to `/cmd_vel`. This places
`balance_controller` in-path:

```
collision_guard → /cmd_vel_safe → balance_controller → /cmd_vel → cmd_vel_to_vesc
```

### New design doc

See `BALANCE_CONTROLLER.md` for full LQR theory, tuning workflow, and topic list.

### Live gain tuner

`scripts/tune_gains.py` — interactive terminal UI and one-liner helper for
adjusting all LQR/PI gains on the running node via `ros2 param set`.
Changes take effect within one control tick (≤20ms), no restart needed.



## 2026-04-13 — Added lid_controller (RS05 cargo bay lid)

**`lid_controller`** — `locomotion/lid_controller.py`
Drives the RS05 motor (`joint_rs05_1`, can1, 0x1E) between open and closed positions.
- Triggered from Foxglove: Publish panel → `/lid_command` (std_msgs/String)
  - `"open"` / `"close"` / `"toggle"`
- Reports state on `/lid_state` (std_msgs/String): `open`, `closed`, `moving_open`, `moving_closed`, `unknown`
- Publishes `/joint_commands` at 50Hz; hold torque=0 when idle, `torque_ff` while moving
- Declares arrived when position error < `position_tolerance_rad` or after `move_timeout_sec`
- No `/robot_mode` dependency — commandable any time the stack is up
- Config: `config/lid_controller.yaml` — tune `open_position_rad` after zeroing the motor

**Foxglove setup:**
1. Publish panel → `/lid_command`, message `{"data": "open"}` → Open Lid button
2. Publish panel → `/lid_command`, message `{"data": "close"}` → Close Lid button
3. Raw Messages panel → `/lid_state` to read current state

## Package overview

ROS2 Python package containing locomotion-layer nodes: velocity command muxing,
collision guarding, and driving-leg position control.

## Nodes

### `vel_cmd_mux` — `locomotion/vel_cmd_mux.py`
Selects between teleop (`/cmd_vel_teleop`) and autonomous (`/cmd_vel_auto`) velocity
commands based on `/robot_mode` and command freshness. Publishes `/cmd_vel_selected`.

### `collision_guard` — `locomotion/collision_guard.py`
Safety layer that clamps `/cmd_vel_selected` to configured linear/angular limits
and enforces e-stop by publishing zero twist on `/cmd_vel`.

### `driving_leg_controller` — `locomotion/driving_leg_controller.py`
Holds the 6 RS04 leg joints at their calibrated driving positions via MIT mode
(Type 1 CAN, through the robstride_can_driver's `/joint_commands` topic).

**Behaviour:**
- `RUNNING` (mode ≠ "off" AND no e-stop): publishes `/joint_commands` at 50 Hz
  for the RS04 joints with `torque_ff = 1.0 Nm`, `velocity = 0.0`.
  Calls `/enable_motors` on first entry.
- `STOPPED` (mode = "off" OR e-stop): calls `/stop_motors`. No commands published.

**RS00 coast mode** (setup once, ~3 s after startup):
  Calls `/set_gains` (kp=0, kd=0) and `/write_motor_param` (damper=0x702A=1)
  for each RS00 joint so the driver sends zero-torque Type 1 frames → freewheel.

**RS05 is unplugged** — this node sends no commands to it.

**Config:** `locomotion/driving_leg_pos.yaml` — maps RS04 joint names to target
positions in radians. Installed to share/locomotion at build time.

### `hold_position_controller` — `locomotion/hold_position_controller.py`
Identical state machine to `driving_leg_controller` but instead of commanding
the YAML positions, it snapshots `/joint_states` at the moment of enable and
holds those live positions. Use when you want the robot to stay wherever it
physically is rather than snap to the calibrated driving pose.

## Required Startup Order

The leg controllers depend on the robstride driver's ROS2 services being available.
Things must come up in this sequence:

```
1. CAN interfaces up          sudo ip link set can0 up type can bitrate 1000000
                              sudo ip link set can1 up type can bitrate 1000000

2. Source ROS2 + workspace    source /opt/ros/humble/setup.bash
                              source /workspaces/teambowl_ws/install/setup.bash

3. robstride_can_driver       ros2 launch robstride_can_driver driver.launch.py
   (must be running first)    → enables all motors, starts /joint_states at 100 Hz
                              → exposes /enable_motors, /stop_motors, /set_gains, etc.

4. Leg controller             bringup.launch.py launches hold_position_controller
   (default via bringup)    OR ros2 run locomotion hold_position_controller
                            OR ros2 run locomotion driving_leg_controller
                              → waits up to 5 s for driver services (coast setup)
                              → stays STOPPED until robot mode is set

5. Set robot mode             ros2 topic pub /robot_mode_set std_msgs/msg/String \
                                  '{data: "teleop"}' --once
                              → controller transitions to RUNNING, calls /enable_motors
```

**`hold_position_controller` is what bringup.launch.py launches for teleop** (switched
from `driving_leg_controller` on 2026-03-17). `teleop.sh` (in `teambowl_docker/`)
handles the robstride driver startup + mode set inside the container.

bringup.launch.py NOW includes the robstride driver (added 2026-03-17).

## 2026-03-18 — Moved parameters to config/locomotion.yaml

- **`config/locomotion.yaml`**: New file. Contains parameters for all three nodes
  (`hold_position_controller`, `vel_cmd_mux`, `collision_guard`) in standard ROS2
  YAML format. `bringup.launch.py` passes this single file to all three nodes.
- **`setup.py`**: Added `config/locomotion.yaml` to `data_files`.

## 2026-03-17 — Added per-joint status print to hold_position_controller

- **`hold_position_controller.py`**: Added `_status_timer` (2 s) and `_print_status`.
  Prints `[HOLD]  joint_name: +0.1234  ...` to stdout when active, or
  `[HOLD]  joint_name: X  ...` per-joint when not enabled or no joint_states.

## 2026-03-16 — Added hold_position_controller + movement preview

- **`hold_position_controller.py`**: New node. Reads joint names from the same
  YAML but snapshots positions from `/joint_states` at enable time. Same
  RS00 coast setup, same enable/disable logic.
- **`driving_leg_controller.py`**: Added `/joint_states` subscription. Before
  the first enable, logs a per-joint movement preview (current → target, Δ rad).
  Any joint requiring > 0.3 rad of movement is logged at WARN level.
- **`setup.py`**: Added `hold_position_controller` entry point.

## 2026-03-16 — Added driving_leg_controller

- **`driving_leg_pos.yaml`**: Reformatted from ad-hoc whitespace format to proper
  YAML. Removed CAN ID and "Radians" columns; only joint name → position (rad).
- **`driving_leg_controller.py`**: New node that reads the YAML and holds RS04
  joints at those positions via MIT mode. Enables motors on mode transition to
  active; stops all motors on transition to "off"/e-stop. Sets RS00 joints to
  coast mode (zero gains + damper disabled) once at startup.
- **`setup.py`**: Added `driving_leg_pos.yaml` to `data_files` and
  `driving_leg_controller` entry point.
