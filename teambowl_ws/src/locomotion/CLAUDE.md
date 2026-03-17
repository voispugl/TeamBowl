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

4. Leg controller             ros2 run locomotion driving_leg_controller
   (either one)            OR ros2 run locomotion hold_position_controller
                              → waits up to 5 s for driver services (coast setup)
                              → stays STOPPED until robot mode is set

5. Set robot mode             ros2 topic pub /robot_mode_set std_msgs/msg/String \
                                  '{data: "teleop"}' --once
                              → controller transitions to RUNNING, calls /enable_motors
```

`teleop.sh` in the repo root handles steps 1–5 automatically (native, no Docker).

The bringup.launch.py does NOT launch the robstride driver — it must be started separately.

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
