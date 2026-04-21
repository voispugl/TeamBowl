# sim/mujoco — MuJoCo Balance Simulation

## 2026-04-14 — Initial standalone Python simulation

Standalone Python simulation for the TeamBowl two-wheeled self-balancing robot.
Lets you test and tune the cascaded PID controller before deploying to hardware.

### Files

| File | Description |
|---|---|
| `teambowl_balance.xml` | MuJoCo model — motor (torque) actuators, friction params, global sensors |
| `sim.py` | Standalone Python sim — cascaded PID + viewer + CSV logging |
| `meshes/` | STL mesh files referenced by the XML |

### Running

```bash
pip install mujoco        # one-time install
cd sim/mujoco
python sim.py
```

Opens the MuJoCo passive viewer. Type gain commands in the terminal while it runs.

### Control architecture

```
Physics   : 500 Hz (timestep=0.002 s)
Inner PD  : 150 Hz — complementary-filter pitch → torque (Nm)
Outer PI  :  40 Hz — velocity error → target pitch angle (theta_ref)
```

### Live gain tuning (terminal commands while sim is running)

| Command | Effect |
|---|---|
| `kp=80` | Set KP_PITCH (inner PD proportional) |
| `kd=10` | Set KD_PITCH (inner PD derivative) |
| `kpv=0.4` | Set KP_VEL (outer PI proportional) |
| `kiv=0.06` | Set KI_VEL (outer PI integral) |
| `v=0.5` | Set target forward velocity (m/s) |
| `gains` | Print current gain values |
| `reset` | Zero v_cmd, clear velocity integrator |

### Sensor overview

**Body-frame IMU** (moves with robot):
- `imu_gyro`   — angular velocity [wx, wy, wz]
- `imu_accel`  — linear acceleration [ax, ay, az]

**Global-frame ground truth** (world-anchored, "perfect EKF"):
- `gt_pos`     — Frame body world position [x, y, z]
- `gt_quat`    — Frame body world orientation [w, x, y, z]
- `gt_linvel`  — Frame body world linear velocity
- `gt_angvel`  — Frame body world angular velocity

The sim logs `pitch_cf` (complementary filter) vs `pitch_gt` (ground truth) every
tick so you can measure how well the filter is tracking in `teambowl_sim_log.csv`.

### Tuning workflow

1. Start with default gains (`kp=60, kd=8`). Robot should balance but may oscillate.
2. If robot falls forward/backward immediately: flip `PITCH_SIGN = -1` in `sim.py`.
3. Raise `kp` until steady oscillation, then back off ~30%.
4. Raise `kd` to damp the oscillation.
5. Test nonzero velocity: `v=0.3`, then tune `kpv` / `kiv`.
6. Copy final gains into `balance_controller.yaml` for the real robot.

### XML changes from original

- **Actuators**: Changed from `velocity` to `motor` (torque, Nm). `ctrlrange="-30 30"`.
- **Physics**: Added explicit `timestep="0.002"`, `gravity="0 0 -9.81"`.
- **Friction**: Added `<default class="wheel">` with realistic friction/solref/solimp.
  Floor geom also gets matching friction parameters.
- **Global sensors**: Added `gt_pos`, `gt_quat`, `gt_linvel`, `gt_angvel` sensors
  (framepos/framequat/framelinvel/frameangvel on the Frame body) for ground-truth
  world-frame state — the "dummy global IMU" for EKF validation.
