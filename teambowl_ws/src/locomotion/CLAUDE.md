# locomotion

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

## 2026-03-24 — driving_leg_controller: auto_start on launch

- **`locomotion/driving_leg_controller.py`**: Added `auto_start` (default `true`) and
  `auto_start_delay_s` (default `2.0`) parameters. When `auto_start=true`, a one-shot
  timer fires after the delay and calls `_transition_to_running()` automatically —
  no need to publish to `/robot_mode` to enable the legs. E-stop and mode `"off"` still
  stop the controller normally. Auto-start does not re-enable after a stop.

## 2026-03-24 — driving_leg_controller: removed RS00 coast setup; added status print

- **`locomotion/driving_leg_controller.py`**: Removed all RS00 coast-mode code (service
  clients for set_gains/read_motor_param/write_motor_param, _coast_timer, _setup_coast_mode,
  rs00_joints parameter). RS00 freewheeling is now handled by motors.yaml (kp=0, kd=0).
  Added `_status_timer` (5 s) printing target, actual, and error per RS04 joint.
- **`drivers/robstride_can_driver/config/motors.yaml`**: RS00 default_kp and default_kd set
  to 0.0 so wheel motors freewheel from driver startup without any controller intervention.

## 2026-03-24 — driving_leg_controller now default; torque_ff = 0; can1 resilience

- **`launch/bringup.launch.py`**: `leg_controller` default changed to `driving`.
- **`config/locomotion.yaml`**: Added `driving_leg_controller` section with `torque_ff: 0.0`
  (MIT mode hold relies on Kp/Kd only, no feedforward torque).
- **`driver_node.py`**: CAN bus open failure no longer crashes the node; motors on an
  unavailable bus are silently skipped everywhere.

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
