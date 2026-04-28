# locomotion

## 2026-04-20 — Driving controller zeros output immediately on mode change

**`locomotion/driving_controller.py`**: `_on_mode` now calls `_publish_cmd(0.0, 0.0)` immediately when mode changes away from `driving`, eliminating a brief window where stale velocity was published to `/cmd_vel` before the next tick.

## 2026-04-20 — Driving controller ki windows reduced from 2.0 s → 0.5 s

**`locomotion/driving_controller.py`**: All three integral sliding windows (velocity, pitch, yaw) now prune samples older than 0.5 s (was 2.0 s). Faster windup decay; reduces overshoot from sustained error in any axis.

## 2026-04-20 — Driving controller redesigned: parallel velocity + pitch + yaw PIDs

**`locomotion/driving_controller.py`**:
- Removed two-timer cascade (50 Hz outer + 100 Hz inner). Replaced with a **single `_tick` at `control_rate_hz` (default 100 Hz)**.
- Three PIDs run in parallel each tick:
  - **Velocity PID**: `u_vel = kp_vel*v_err + ki_vel*∫ + kd_vel*dv_err/dt`
  - **Pitch PID** (nose-dive correction, 3-sample FIR derivative): `u_pitch = kp_pitch*pitch_err + kd_pitch*d_pitch + ki_pitch*∫`
  - **Yaw PID** (new): `u_yaw = omega_cmd + kp_yaw*yaw_err + ki_yaw*∫ + kd_yaw*d_yaw` — additive, gains=0 → passthrough
- `v_out = clamp(u_vel + u_pitch, -v_max, v_max)`, `omega_out = u_yaw`
- Added `_yaw_dot` (from `angular_velocity.z`), `_yaw_err_prev`, `_yaw_i_window` state vars.
- `/driving_gains_echo` now includes `kp_yaw`, `ki_yaw`, `kd_yaw`, `_yaw_dot`.
- `/driving_gains` accepts `kp_yaw`, `ki_yaw`, `kd_yaw` for live tuning.

**`config/driving_controller.yaml`**:
- Replaced `outer_rate_hz` + `inner_rate_hz` with `control_rate_hz: 100.0`.
- Added `kp_yaw: 0.0`, `ki_yaw: 0.0`, `kd_yaw: 0.0` with tuning comments.



## 2026-04-20 — Yaw PD upgraded to PID (added ki_yaw)

**`locomotion/balance_controller.py`**:
- Added `ki_yaw` parameter (default 0.0, live-tunable via `/balance_gains`).
- `_inner_tick`: yaw output is now `kp_yaw * yaw_err + ki_yaw * integral + (-kd_yaw * yaw_dot)` where integral uses a 1-second sliding window `_yaw_i_window` (same pattern as pitch integral).
- `_yaw_i_window` cleared on mode exit (leaving balance mode) and on estop.
- `/balance_gains_echo` and `_on_gains` updated to include `ki_yaw`.

**`config/balance_controller.yaml`**: added `ki_yaw: 0.0`.

## 2026-04-19 — driving_leg_controller publishes /leg_controller_running

**`locomotion/driving_leg_controller.py`**: Added a 2 Hz TRANSIENT_LOCAL Bool publisher on
`/leg_controller_running` that reflects `self._running`. Used by the steamdeck_teleop web UI
to show a green/red "Legs" status box without any extra subscription overhead.

## 2026-04-19 — Balance mode now accepts Nav2 autonomous goals

**`locomotion/vel_cmd_mux.py`** — balance mode routing updated.
Previously: balance mode only routed `/cmd_vel_teleop`.
Now: balance mode routes `/cmd_vel_auto` if fresh, falls back to `/cmd_vel_teleop` if fresh, else zero.
Nav2 goals can now drive the self-balancing controller. Teleop still works as a fallback.

## 2026-04-19 — Disabled compass in EKF / IMU config

**`config/ekf.yaml`** — set IMU yaw fusion to `false` (VRU mode has no magnetometer
reference so yaw drifts; wheel odometry provides yaw instead). Roll/pitch still fused.

**`src/drivers/xsens_mti_ros2_driver/param/xsens_mti_node.yaml`** — set
`enable_filter_config: true`, `mti_filter_option: 4` (vru_general — no compass),
`pub_mag: false`. Built xsens_mti_ros2_driver package for the first time.

## 2026-04-20 — Reduced pitch integral window from 2.0s → 0.5s

**`locomotion/balance_controller.py`** — `_inner_tick()` sliding-window pruning threshold changed from 2.0 → 0.5 seconds. Faster windup decay; reduces overshoot from sustained pitch error.

## 2026-04-19 — Added balance controller tuning guide

**`BALANCE_TUNING.md`**: Step-by-step tuning document for the cascaded PID balance
controller. Covers architecture, live tuning via Foxglove `/balance_gains`, gain ordering
(theta_eq_offset → kp_pitch → kd_pitch → ki_pitch → outer PI → yaw), symptom tables,
safety limits, and Foxglove panel setup.

## 2026-04-19 — Reverted lid_controller from PP mode back to MIT mode

PP position mode was broken in practice. Reverted both files to the pre-PP-mode git HEAD.

**`locomotion/lid_controller.py`** — restored MIT mode: 50 Hz `_control_tick` publishing `/joint_commands`, `kp`/`kd`/`torque_ff` params, removed `WriteMotorParam` client.
**`config/lid_controller.yaml`** — restored MIT params: `kp: 60.0`, `kd: 1.0`, `torque_ff: 0.5`, `publish_rate_hz: 50.0`.

## 2026-04-19 — Added verbose flag to silence periodic debug output

**`locomotion/lid_controller.py`** and **`locomotion/driving_leg_controller.py`**
- Added `verbose` parameter (default `false`) to both nodes.
- `lid_controller._debug_status()` (2 s timer) now returns early unless `verbose=true`.
- `driving_leg_controller._print_status()` (5 s timer) now returns early unless `verbose=true`.
- All state-change logs (moves, arrivals, mode transitions, errors) are unaffected and always logged.

**`config/lid_controller.yaml`** — added `verbose: false`
**`config/locomotion.yaml`** — added `verbose: false` under `driving_leg_controller`

Enable at launch time:
```
ros2 launch bringup bringup.launch.py verbose_controllers:=true
```

## 2026-04-18 — Switched lid_controller to RS05 built-in PP position mode

PP mode is flashed permanently to the motor via `~/TeamBowl/commission_rs05_pp.sh`
(run once). The motor boots in PP mode every time — no startup sequencing needed.

**`locomotion/lid_controller.py`** — complete rewrite
- Replaced MIT mode (50 Hz Type 1 streaming) with **PP position mode**: the motor runs
  its own cascade controller (Position P → Velocity PI → Current PI). The ROS node only
  writes `loc_ref` (param 0x7016, dec 28694) once per move command via `/write_motor_param`.
- On estop: writes `loc_ref = current_pos` to hold in place (motor holds autonomously).
- Removed: `_joint_pub`, 50 Hz control timer, `_publish_joint()`, MIT-mode params
  (kp, kd, torque_ff, publish_rate_hz), `/set_gains` client, `/lid_gains` subscriber.
- Added: `_write_param_client` (WriteMotorParam), `_command_position()`,
  10 Hz lightweight monitor timer for arrival/timeout detection.
- Kept: `/enable_motors` (wakes motor from standby), all `[LID DEBUG]` status logging.

**`config/lid_controller.yaml`** — updated params
- Removed: `kp`, `kd`, `torque_ff`, `publish_rate_hz` (all MIT-mode params)
- PP gains (loc_kp, spd_kp, spd_ki, limit_spd, limit_cur) live on the motor flash, not YAML.

**`~/TeamBowl/commission_rs05_pp.sh`** — one-shot motor flash script
- NEW file. Enables motor, writes run_mode=1 + PP gains, calls `/save_motor_params`.
- Run once before first use. Gains survive power cycles.

**`~/TeamBowl/test_lid.sh`** — rewritten as full tuning script
- Live position readout (`p`), continuous monitor (`m`)
- Move to arbitrary position (`t <rad>`) via loc_ref write, open/close presets
- Set mechanical zero (`z`) via `/set_zero` service
- Live volatile gain writes: `kp`, `vp`, `vi`, `spd`, `cur` → `/write_motor_param`
- Save calibrated positions to YAML: `sopen`, `sclosed`
- Restart lid_controller: `r`

## 2026-04-19 — Added leg IK + jump controller

**`leg_kinematics`** — `locomotion/leg_kinematics.py`
Standalone (no-ROS) FK/IK module for the parallel 5-bar legs.
- Models each leg as equivalent 2R arm: Motor A controls knee pivot position
  (via Thigh, L≈0.297 m), Motor B controls Calf direction via the parallel link
  (L≈0.297 m). Total reach ≈ 0.593 m.
- `leg_fk_urdf(θ_A, θ_B)` — foot position in Hip frame given URDF-convention angles
- `LegCalibration` — fits encoder zero offsets from the known driving position +
  measured foot height; call `calibrate_from_driving_pos()` at startup
- `compute_jump_waypoints(cal_l, cal_r, crouch_depth)` — returns crouch + extend
  joint dicts ready for the jump controller; falls back to heuristic delta if IK fails

Key constants: `L_THIGH≈0.297 m`, `L_CALF≈0.297 m`, `L_MAX≈0.593 m`
Geometry from `bringup/robot_description/bowl.urdf` joints `dof_calf1_0`,
`dof_driver1_0`, `closing_knee1_0`, `dof_ankle1_0`.

**`jump_controller`** — `locomotion/jump_controller.py`
ROS2 node: IDLE → CROUCH → EXTEND → RETURN → IDLE.
- **CROUCH**: legs retract to `crouch_depth × L_MAX`, normal gains, hold `crouch_hold_s`
- **EXTEND**: legs slam to 95% max extension, `extend_kd_override` Kd, `extend_torque_ff`
  feedforward, hold `extend_hold_s`
- **RETURN**: command driving positions, release suspend; IDLE when settled or timeout
- Publishes `True` on `/balance_suspend` during CROUCH+EXTEND (balance_controller zeroes wheels)
- Publishes `/joint_commands` at 100 Hz during jump (preempts 50 Hz driving controller)
- Trigger: `ros2 topic pub /jump_command std_msgs/msg/String '{data: "jump"}' --once`
- Config: `config/jump_controller.yaml` — tune `foot_height_driving_m` first (physical measurement)

**`balance_controller`** — added `/balance_suspend` (std_msgs/Bool) subscription.
When True: `_inner_tick` publishes zero wheel command without touching integrators.

**Calibration note**: `foot_height_driving_m` in jump_controller.yaml controls IK
zero-fitting. Default −0.28 m is an estimate; measure actual foot-to-hip distance
on the physical robot and update before hardware testing.

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

## 2026-03-31 — Added trick mode to driving_leg_controller

- **`locomotion/trick_leg_offsets.yaml`**: New file. Per-joint offset values (rad) added
  to base `driving_leg_pos.yaml` positions when in trick mode. Installed to `share/locomotion/`.
- **`setup.py`**: Added `trick_leg_offsets.yaml` to `data_files`.
- **`driving_leg_controller.py`**: Added `_trick_offsets` dict (joint → float). Subscribes
  to `/trick_leg_offsets` (JointState from keyboard_operator). In `_publish_commands`,
  adds offsets to base positions when `self._mode == 'trick'`; otherwise uses base only.
  `_print_status` also shows effective (offset-adjusted) target in trick mode.

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
