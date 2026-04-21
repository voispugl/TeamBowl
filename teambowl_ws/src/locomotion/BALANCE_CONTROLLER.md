# TeamBowl Balance Controller Design

## Overview

The balance controller turns TeamBowl into a two-wheel self-balancing robot. When
active, the legs are stowed in their calibrated positions (locked via MIT-mode impedance
on the RS04s) and the two drive wheels act as an inverted-pendulum balance platform.

The controller has two nested loops:

```
/cmd_vel_safe ──► [Outer PI: velocity] ──► θ_ref ──► [Inner LQR: balance] ──► /cmd_vel ──► VESCs
     ▲                                                       ▲
/odometry/filtered (v)                         /imu/data (θ, θ̇)
```

### State estimator (robot_localization EKF)

Raw IMU data (`/imu/data`) and wheel odometry (`/odom_wheels`) are fused by
`robot_localization`'s `ekf_node` into `/odometry/filtered`. This provides:

- `pose.orientation` quaternion → pitch angle θ (via ZYX Euler extraction, around Y axis)
- `twist.linear.x` → filtered body velocity v

The balance controller primarily uses:
- θ and θ̇ from `/imu/data` directly (lower latency, no EKF lag for the inner loop)
- v from `/odometry/filtered` for the outer PI loop

---

## System Model

### Assumptions

1. Leg movement has **negligible effect on overall CoG** — legs are stowed in a fixed
   position during balance mode and the CoM shift from leg joints is small relative to
   the chassis mass.
2. Variable mass (cargo bay payload) acts as a **point mass at a fixed height** on top
   of the chassis. The effective CoM height `l_com` changes with cargo mass:
   ```
   l_eff = (m_chassis * l_chassis + m_cargo * l_cargo) / (m_chassis + m_cargo)
   ```
3. The robot moves in a **2D plane** (no terrain variation).

### Linearized inverted pendulum on wheels

State vector: **x** = [θ, θ̇, v]

- θ: pitch angle from vertical (positive = forward lean), radians
- θ̇: pitch rate, rad/s (from IMU gyro Y axis)
- v: forward body velocity, m/s (from EKF odometry)

Control input: **u** = v_wheel_cmd (symmetric wheel velocity command, m/s)

Linearized equations of motion about θ = 0:

```
θ̈  =  (g / l_eff) · θ  −  (1 / l_eff) · v_wheel_cmd
v̇  ≈  v_wheel_cmd  (direct velocity control, VESC inner loop assumed fast)
```

Compact state-space (continuous time):
```
ẋ = A·x + B·u

A = [ 0           1     0  ]      B = [  0     ]
    [ g/l_eff     0     0  ]          [ -1/l_eff]
    [ 0           0     0  ]          [  1     ]

C = I_3x3   (full state observable via IMU + odometry)
```

where:
- g = 9.81 m/s²
- l_eff = effective CoM height from wheel axle (meters)

### LQR gain computation

The discrete-time gain matrix **K** is computed offline using `python-control`:

```python
import control
import numpy as np

# Physical parameters
M = 25.0    # total mass, kg (chassis + nominal cargo)
l = 0.45    # CoM height from wheel axle, m
g = 9.81
dt = 0.02   # 50 Hz control loop

A = np.array([[0, 1, 0],
              [g/l, 0, 0],
              [0, 0, 0]])
B = np.array([[0], [-1/l], [1]])

# LQR cost weights (tune these)
Q = np.diag([100.0, 10.0, 1.0])   # penalize: theta, theta_dot, velocity
R = np.array([[0.1]])               # penalize: wheel velocity command

# Compute LQR gain
K, _, _ = control.lqr(A, B, Q, R)
# K is shape (1, 3): [k_theta, k_theta_dot, k_v]
```

See `sim/mujoco/tune_lqr.py` for the full solver with mass gain scheduling.

### Gain scheduling for variable mass

The gain matrix K depends on mass (through l_eff). Three mass setpoints are defined:

| Mode | Total mass | `l_eff` | K values |
|------|-----------|---------|---------|
| `light` | ~20 kg | 0.42 m | from tune_lqr.py |
| `nominal` | ~28 kg | 0.45 m | from tune_lqr.py |
| `heavy` | ~38 kg | 0.48 m | from tune_lqr.py |

Switch via `ros2 param set /balance_controller mass_mode light|nominal|heavy`.

---

## Control Law

### Outer loop: velocity PI controller

```
e_v    = v_cmd − v_actual
θ_ref  = clip(kp_vel · e_v + ki_vel · Σe_v·dt, −θ_max_cmd, +θ_max_cmd)
```

- `v_cmd` comes from `/cmd_vel_safe` (linear.x)
- `v_actual` comes from `/odometry/filtered` (twist.linear.x)
- `θ_ref` is the tilt angle setpoint: lean forward to accelerate, back to decelerate
- `θ_max_cmd = 0.25 rad` (~14°) — safety limit

### Inner loop: LQR balance

```
e = [ θ − θ_ref,  θ̇,  v_actual − v_cmd ]

u_balance = −K · e  (m/s wheel velocity correction)
```

The K matrix is [k_theta, k_theta_dot, k_v]. The LQR drives tilt error to zero while
simultaneously managing velocity error in the cost function.

### Final command assembly

```
cmd_vel.linear.x  = u_balance          # symmetric wheel command
cmd_vel.angular.z = ω_cmd              # from /cmd_vel_safe, passed through
```

This goes to `/cmd_vel` → `cmd_vel_to_vesc`, which converts to ERPM via existing
differential kinematics. The existing ERPM slew rate (2000 ERPM/tick) provides an
additional low-pass filter on the wheel acceleration.

### Non-balance passthrough

When `robot_mode ≠ "balance"`, the controller simply passes `/cmd_vel_safe` through
to `/cmd_vel` unchanged, keeping the existing drive pipeline intact.

---

## Safety

| Condition | Action |
|-----------|--------|
| `|θ| > θ_max_fallover` (default 0.5 rad / ~28°) | Publish zero twist, raise `/estop` |
| EKF odometry timeout > 0.3s | Revert to raw IMU velocity estimate |
| `/estop` received | Zero all outputs immediately |
| Mode transitions out of balance | Reset PI integrator, transition smoothly |

---

## Topics

### Subscribed

| Topic | Type | Use |
|-------|------|-----|
| `/odometry/filtered` | `nav_msgs/Odometry` | Body velocity v (outer PI) |
| `/imu/data` | `sensor_msgs/Imu` | θ (quat pitch), θ̇ (gyro.y) |
| `/cmd_vel_safe` | `geometry_msgs/Twist` | Desired v_cmd, ω_cmd |
| `/robot_mode` | `std_msgs/String` | Mode switching |
| `/estop` | `std_msgs/Bool` | Emergency stop |

### Published

| Topic | Type | Use |
|-------|------|-----|
| `/cmd_vel` | `geometry_msgs/Twist` | Wheel velocity → cmd_vel_to_vesc |
| `/estop` | `std_msgs/Bool` | Triggers if fallover detected |

---

## Parameters (balance_controller.yaml)

All parameters are live-tunable via `ros2 param set /balance_controller <param> <value>`.

```yaml
balance_controller:
  ros__parameters:
    # -- LQR gains (computed by tune_lqr.py for nominal mass) --
    k_theta:       30.0   # rad → m/s wheel velocity
    k_theta_dot:    5.0   # (rad/s) → m/s
    k_v:            1.0   # (m/s) → m/s

    # -- Outer PI gains --
    kp_vel:         0.3   # (m/s error) → rad lean setpoint
    ki_vel:         0.05  # integrator gain

    # -- Mass gain schedule (override K at each setpoint) --
    mass_mode:      nominal      # light | nominal | heavy
    k_theta_light:  35.0
    k_theta_dot_light: 6.0
    k_v_light:      1.2
    k_theta_heavy:  25.0
    k_theta_dot_heavy: 4.5
    k_v_heavy:      0.8

    # -- Safety limits --
    theta_max_cmd:      0.25   # max lean setpoint magnitude (rad)
    theta_max_fallover: 0.50   # triggers estop (rad)

    # -- Robot geometry --
    l_com:          0.45   # CoM height above wheel axle (m)

    # -- Misc --
    control_rate_hz: 50.0
    odom_timeout_s:  0.3
```

---

## Tuning Workflow

1. Run `python sim/mujoco/tune_lqr.py` to compute K for each mass setpoint. Update
   `k_theta`, `k_theta_dot`, `k_v` in `balance_controller.yaml`.

2. In sim: run `python sim/mujoco/run_sim.py` to verify balance for ≥30s with
   ±0.1 rad perturbations across all mass setpoints.

3. On hardware:
   - Start in `nominal` mass mode, robot unloaded.
   - Increase `k_theta` until oscillation appears, then back off 20%.
   - Adjust `kp_vel` until velocity tracking is responsive without overshoot.
   - Use `ros2 param set /balance_controller k_theta 35.0` for live adjustment (no restart).

4. Log with `ros2 bag record /imu/data /odometry/filtered /cmd_vel /robot_mode`
   for post-hoc analysis.

---

## Node Pipeline (balance mode active)

```
/cmd_vel_safe  ──────────────────────────────────────────────────┐
                                                                   ▼
/imu/data ──► [balance_controller] ──► /cmd_vel ──► [cmd_vel_to_vesc] ──► VESC L/R
/odometry/filtered ─────────────────►

/imu/data ──► [ekf_node] ──► /odometry/filtered
/odom_wheels ──►

/cmd_vel ──► [wheel_odom] ──► /odom_wheels
```

---

## Related Files

| File | Purpose |
|------|---------|
| `locomotion/balance_controller.py` | This node |
| `locomotion/wheel_odom.py` | cmd_vel → odometry for EKF input |
| `config/balance_controller.yaml` | Tunable parameters |
| `config/ekf.yaml` | robot_localization EKF configuration |
| `sim/mujoco/tune_lqr.py` | Offline LQR gain solver |
| `sim/mujoco/balance_env.py` | MuJoCo Gymnasium env |
| `sim/mujoco/run_sim.py` | Simulation runner |
| `sim/mujoco/SIM_TO_REAL.md` | Sim-to-real gap checklist |
