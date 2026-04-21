# TeamBowl MuJoCo Balance Sim

Standalone Python simulation for the two-wheeled self-balancing robot.
Tune the cascaded PID controller here before deploying gains to hardware.

## Quick start

```bash
pip install mujoco        # one-time
cd sim/mujoco
python sim.py
```

Opens the MuJoCo passive viewer. Type gain commands in the **same terminal** while it runs.

---

## Control architecture

```
Physics timestep : 0.002 s  (500 Hz)
Inner PID        : 150 Hz  — pitch error → wheel torque (Nm)
Outer PI         :  40 Hz  — velocity error → target lean angle (θ_ref)
Yaw PD           : 150 Hz  — yaw rate error → differential torque
```

### Inner PID loop

```
u_torque = -(kp_pitch * pitch_err
           + ki_pitch * pitch_integral   ← 2-second sliding window
           + kd_pitch * gyro_pitch)
```

`pitch_err = θ - θ_ref`

The I term accumulates only the **last 2 seconds** of pitch error (rolling deque).
It corrects steady-state lean without winding up from long-term drift.

### Outer PI loop

```
θ_ref = kp_vel * v_err + ki_vel * ∫v_err dt
```

`v_err = v_cmd - v_actual`  (ground-truth velocity from `gt_linvel`)

θ_ref is clamped to ±`THETA_MAX` (default 0.35 rad, ~20°).

### Yaw PD

```
u_yaw = kp_yaw * (ω_cmd - ω_actual) - kd_yaw * ω_actual
ctrl[0] = torque - u_yaw   (left wheel)
ctrl[1] = torque + u_yaw   (right wheel)
```

---

## Live gain tuning — terminal commands

Type these while the sim is running and press Enter:

| Command | Gain | Units |
|---|---|---|
| `kp=1600` | `kp_pitch` — inner pitch P | Nm/rad |
| `kd=20` | `kd_pitch` — inner pitch D | Nm·s/rad |
| `ki=5` | `ki_pitch` — inner pitch I (2-s window) | Nm/(rad·s) |
| `kpv=0.5` | `kp_vel` — outer velocity P | rad/(m/s) |
| `kiv=2.0` | `ki_vel` — outer velocity I | rad/(m·s) |
| `kyp=5` | `kp_yaw` — yaw rate P | Nm/(rad/s) |
| `kyd=0.5` | `kd_yaw` — yaw rate D | Nm·s²/rad |
| `v=0.3` | target forward velocity | m/s |
| `omega=0.5` | target yaw rate | rad/s |
| `sign` | flip pitch sign (if robot falls wrong way) | |
| `yawsign` | flip yaw sign (if yaw corrects wrong way) | |
| `gains` | print all current values | |
| `reset` | zero v_cmd + ω_cmd, clear integrators | |

---

## Tuning workflow

### Step 1 — Get it balancing (inner P only)

1. Set `kpv=0` and `kiv=0` (open-loop velocity, robot holds still).
2. Start with `kp=800, kd=0, ki=0`.
3. If the robot falls immediately in the wrong direction: type `sign` to flip.
4. Raise `kp` until the robot oscillates steadily (e.g. `kp=1600`).
5. Back off `kp` by ~30%.
6. Raise `kd` until oscillation is damped (try `kd=10`, then `kd=20`).
7. Robot should now balance indefinitely.

### Step 2 — Add pitch I (trim steady-state lean)

Only needed if the robot balances but leans slightly forward or backward at rest.

1. Type `ki=5`. Watch whether the lean bias reduces within ~2 seconds.
2. If it oscillates, reduce `ki` or increase `kd`.
3. Typical working range: `ki=0–20`.

### Step 3 — Velocity control (outer PI)

1. Type `v=0.3` to command forward motion.
2. Raise `kpv` until the robot accelerates toward the target speed (try `kpv=0.5`).
3. Raise `kiv` to eliminate steady-state velocity error (try `kiv=2`).
4. If the robot oscillates in speed: reduce `kpv`, then raise `kd` or `ki`.

### Step 4 — Yaw control

1. Type `omega=0.5` to command a turn.
2. Raise `kyp` until it turns (try `kyp=5`).
3. If yaw overcorrects: type `yawsign` to flip. Raise `kyd` to damp.

---

## Telemetry — terminal printout (every 1 s)

```
[t= 12.0s]  φ=0.1°  θ=-1.2°  ψ=5.4°  θ_t=-0.8°  x_t=0.31m  y_t=0.01m  ψ̇=0.3°/s
```

| Symbol | Meaning |
|---|---|
| φ (phi) | Roll (ground truth) |
| θ (theta) | Pitch — positive = leaning forward |
| ψ (psi) | Yaw |
| θ_t | Target pitch (θ_ref from outer PI) |
| x_t | Ground-truth x position |
| y_t | Ground-truth y position |
| ψ̇ | Yaw rate |

---

## Gain comparison: sim vs real robot

The sim outputs **torque (Nm)** to the MuJoCo motor actuator.  
The real controller outputs **velocity (m/s)** to the VESC driver.  
**The gains are NOT directly transferable** — use the sim for qualitative tuning and sign/structure, then re-tune on hardware.

| Gain | Sim default | Real (`balance_controller.yaml`) |
|---|---|---|
| `kp_pitch` | 1600 Nm/rad | 200 (m/s)/rad |
| `kd_pitch` | 0 | 8 |
| `ki_pitch` | 0 | 0 |
| `kp_vel` | 0 | 0.30 |
| `ki_vel` | 0 | 0.05 |
| `kp_yaw` | 0 | 5.0 |
| `kd_yaw` | 0 | 0.5 |

---

## Files

| File | Description |
|---|---|
| `sim.py` | Standalone simulation — runs with `python sim.py` |
| `teambowl_balance.xml` | MuJoCo model (motor actuators, friction, sensors) |
| `meshes/` | Binary STL files for robot geometry |
| `teambowl_sim_log.csv` | Auto-generated CSV log (every run overwrites) |

### CSV log columns

`t, pitch_cf, pitch_gt, v_actual, v_cmd, theta_ref, torque, x, y, yaw`

- `pitch_cf` — complementary filter estimate (what controller sees)
- `pitch_gt` — ground truth from MuJoCo sensor (for filter validation)
