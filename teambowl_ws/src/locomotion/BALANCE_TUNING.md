# Balance Controller Tuning Guide

## Running the Stack for Tuning

```bash
ros2 launch bringup trajectory_test.launch.py
```

Launches the full stack in driving mode (Nav2 + driving legs + VESCs). Robot mode is set to `"driving"` automatically after 3 seconds. Use Foxglove to publish gains to `/balance_gains` and monitor `/balance_gains_echo`.

---

## Sending Position Goals (Driving Mode)

Goals are sent via Foxglove to the `trajectory_test` node, which calls Nav2 to plan and execute the path.

**Step 1 — Set a goal** (publish to `/trajectory_goal`, type `std_msgs/String`):
```json
{"data": "{\"x\": 2.0, \"y\": 0.0, \"theta\": 0.0, \"relative\": true}"}
```
- `x`, `y` — target position in metres
- `theta` — target heading in radians
- `relative: true` — goal is in the robot's current frame; `false` = odom frame

**Step 2 — Execute** (publish to `/trajectory_cmd`, type `std_msgs/String`):
```json
{"data": "go"}
```

Other commands: `stop` (cancel, stay idle), `reset` (clear stored goal).

**Useful monitoring topics:**

| Topic | Content |
|-------|---------|
| `/trajectory_status` | JSON: state, active goal, errors |
| `/trajectory_path` | Planned path (visualise in Foxglove 3D panel) |
| `/balance_gains_echo` | Current gains + `_theta_deg`, `_v_actual` |

The planner (SmacPlanner2D) replans at 2 Hz. Goal tolerance: 0.25 m / 0.35 rad.

---

## Architecture

```
/cmd_vel_safe ──► Outer PI (50 Hz) ──► theta_ref ──► Inner PID (150 Hz) ──► /cmd_vel
                  velocity error           lean setpoint   pitch error
                  → lean setpoint                          → wheel speed
/imu/data ──────────────────────────────────────────────►
/odometry/filtered ──────────────────────────────────────►
```

**Outer PI** (`_outer_tick`, 50 Hz): `theta_ref = kp_vel * v_err + ki_vel * ∫v_err`
Limits theta_ref to ±`theta_max_cmd` (0.25 rad).

**Inner PID** (`_inner_tick`, 150 Hz): `u = -(kp_pitch * (θ - theta_ref) + ki_pitch * ∫pitch_err + kd_pitch * θ_dot)`
Integral uses a 2-second sliding window to prevent windup.

**Yaw PD**: `u_yaw = kp_yaw * (ω_cmd - ω_actual) - kd_yaw * ω_actual`

---

## Live Tuning

All gains update without restart. Two methods:

**Foxglove** — publish JSON to `/balance_gains` (std_msgs/String). Only keys present are updated:
```json
{"kp_pitch": 94.97, "kd_pitch": 262.39, "ki_pitch": 239.92, "kp_yaw": 5.0, "kd_yaw": 0.5, "kp_vel": 1.4, "ki_vel": 19.48, "kff_pitch": 0.23, "theta_max_cmd": 0.25, "theta_max_fallover": 0.50, "theta_eq_offset": 0.0, "l_com": 0.45}
```
Read current state from `/balance_gains_echo` (published every 2 s, includes `_theta_deg` and `_v_actual`).

**Terminal** (one param at a time):
```bash
ros2 param set /balance_controller kp_pitch 95.0
```

---

## Tuning Order

### 1. Find the balance point — `theta_eq_offset`

With the robot upright and motors running, read `_theta_deg` from `/balance_gains_echo`.
If the robot leans to one side at rest, that angle is the offset. Set:
```json
{"theta_eq_offset": 0.03}
```
Sign convention: positive = leaning forward. Adjust until the robot can stand without drifting.

### 2. Inner proportional gain — `kp_pitch`

Start with `ki_pitch=0`, `ki_vel=0`. Set `kp_vel` just high enough to create a small lean
(~0.05 rad) that will drive tuning. Raise `kp_pitch` until the robot oscillates (buzzes),
then back off 25–30%.

| Symptom | Action |
|---------|--------|
| Falls over slowly | kp_pitch too low |
| High-frequency oscillation / buzzing | kp_pitch too high — back off |

Current value: **94.97**

### 3. Inner derivative — `kd_pitch`

Raise `kd_pitch` to damp the oscillation found in step 2. Too much → sluggish, stiff
response; too little → overshoot and ringing.

| Symptom | Action |
|---------|--------|
| Oscillation after disturbance | kd_pitch too low |
| Robot feels stiff, slow to recover | kd_pitch too high |

Current value: **262.39**

### 4. Inner integral — `ki_pitch`

Leave at 0 until kp/kd are settled. `ki_pitch` corrects steady-state lean error
(e.g. payload off-centre). Raise slowly — too much causes slow oscillation or windup.
The 2-second window limits windup but doesn't eliminate it.

Current value: **239.92** (high — watch for slow oscillation with payloads)

### 5. Outer velocity proportional — `kp_vel`

With the inner loop stable, raise `kp_vel` to improve velocity tracking. Too high
causes the robot to oscillate forward/backward at the outer loop frequency (50 Hz = slow,
visible pumping motion).

| Symptom | Action |
|---------|--------|
| Slow to reach commanded speed | kp_vel too low |
| Forward/backward pumping | kp_vel too high |

Current value: **1.40**

### 6. Outer velocity integral — `ki_vel`

Raise to eliminate steady-state velocity error. At high values causes slow drift oscillation.

Current value: **19.48**

### 7. Feed-forward lean — `kff_pitch`

`kff_pitch` adds a lean proportional to `v_cmd` directly, bypassing PI lag. Useful for
reducing the lean transient when accelerating from rest. Raise until the step response
is crisp without overshoot.

Current value: **0.23**

### 8. Yaw gains — `kp_yaw`, `kd_yaw`

Tune while spinning in place. Raise `kp_yaw` until turns track commands, then raise
`kd_yaw` to damp yaw overshoot.

Current values: kp_yaw **5.0**, kd_yaw **0.5**

---

## Safety Limits

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `theta_max_cmd` | 0.25 rad (14°) | Max lean setpoint the outer PI can request |
| `theta_max_fallover` | 0.50 rad (28°) | Estop threshold — motors cut if exceeded |
| `theta_eq_offset` | 0.00 rad | Trim to true standing-vertical |

Do not raise `theta_max_fallover` above 0.6 rad — the robot's CoM height (`l_com=0.45 m`)
means recovery is unlikely past ~35°.

---

## Saving Tuned Gains

Once stable, write the new values into `config/balance_controller.yaml` and rebuild:
```bash
cd ~/TeamBowl/teambowl_ws
colcon build --packages-select locomotion
```

---

## Foxglove Panel Setup

| Panel | Config |
|-------|--------|
| Publish → `/balance_gains` | type `std_msgs/String`, message `{"data": "{\"kp_pitch\": 95.0}"}` |
| Raw Messages → `/balance_gains_echo` | shows gains + `_theta_deg`, `_v_actual`, mode, estop |
| Plot → `/odometry/filtered` `.twist.twist.linear.x` | actual velocity vs time |
| Plot → `/imu/data` `.angular_velocity.y` | pitch rate (proxy for oscillation) |
