# simulation

## 2026-04-23 — Added Isaac Sim bridge (isaac_sim/ subdirectory)

**`isaac_sim/setup_scene.py`**: Python script run inside the Isaac Sim Docker container via `runapp.sh --exec`.
- Imports `bowl.urdf` via `omni.isaac.urdf` with `convex_decomp=True` (simplified collision, full STL visual)
- Publishes same ROS2 topic interface as `mujoco_bridge` plus rendered camera topics
- Ground-truth VSLAM substitute: publishes ground-truth pose to `/visual_slam/tracking/odometry` so EKF `odom1` is fed on desktop (real VSLAM runs on Jetson only — VPI is aarch64-only)
- Does NOT publish `/odometry/filtered` — that is produced by `ekf_filter_node`

**Topic interface (published by Isaac Sim, consumed by `isaac_sim.launch.py`):**

| Topic | Rate | Notes |
|-------|------|-------|
| `/imu/data` | 100 Hz | Simulated IMU → EKF imu0 |
| `/wheel/odometry` | 100 Hz | Simulated wheel encoders → EKF odom0 |
| `/visual_slam/tracking/odometry` | 100 Hz | Ground-truth → EKF odom1 (VSLAM substitute) |
| `/joint_states` | 50 Hz | All 23 joints |
| `/oak/rgb/image_raw` | 30 Hz | Rendered RGB for YOLO26/cam_ops |
| `/oak/stereo/image_raw` | 30 Hz | Rendered depth for nvblox |

**Obstacle layout:** 7 static objects in a navigable test course (~10×10 m floor). Human mesh loaded from Isaac Nucleus People library; falls back to box placeholder if Nucleus is unavailable.



MuJoCo simulation bridge ROS2 package. Runs `teambowl_mjlab.xml` at 500 Hz in a
background thread and bridges sensor data to ROS2 topics so real controllers can
be tuned without hardware.

## Key Files

| File | Purpose |
|------|---------|
| `simulation/mujoco_bridge.py` | Main bridge node — physics loop, sensor publish, cmd_vel subscriber, /sim_reset service |
| `config/mujoco_bridge.yaml` | All parameters: model_path, spawn_z, gear ratios, ctrl_sign, frame IDs |

## Node: `mujoco_bridge`

**Publishes (from 500 Hz physics thread):**
- `/imu/data` at 100 Hz — orientation from `gt_quat`, angular velocity from `imu_gyro`, accel from `imu_accel`
- `/odometry/filtered` at 100 Hz — pose from `gt_pos`/`gt_quat`, twist from `gt_linvel`/`gt_angvel`
- `/joint_states` at 50 Hz — wheel velocities from `left_wheel_vel`, `right_wheel_vel`

**Subscribes:**
- `/cmd_vel` → motor ctrl via: `ctrl = sign * N * (v_wheel / wheel_radius)` (clipped to ctrlrange)
- `/estop` → zeroes ctrl
- `/robot_mode` → zeroes ctrl when "off"

**Service:** `/sim_reset` (std_srvs/Trigger) → resets to upright spawn pose (z=−0.090)

## Spawn Height Note

The MJCF floor is at `z=−0.3`. The Frame body spawn height for wheels-touching-floor:
```
spawn_z = floor_z + wheel_radius + |wheel_local_z|
        = -0.3    + 0.154        + 0.056195       = -0.090 m
```
This differs from `robot_constants.py`'s `_SPAWN_Z=0.210`, which uses the RL
terrain importer floor at `z=0`.

## cmd_vel → Motor Control

```python
v_left  = vx - omega * track_width / 2
v_right = vx + omega * track_width / 2
ctrl_left  = clip(ctrl_sign_left  * gear_left  * v_left  / wheel_radius, -40, 40)
ctrl_right = clip(ctrl_sign_right * gear_right * v_right / wheel_radius, -48, 48)
```
`ctrl_sign = -1.0` (default) because gear constraint flips direction: `wheel_vel = -(1/N) * motor_vel`.
Flip to `+1.0` if robot drives backward on first test.

## Frame Convention

Wheels rotate about the robot X axis → robot drives along Y.
Lean forward/back = rotation about X → `angular_velocity.x` is pitch rate.
The `balance_controller` reads `angular_velocity.y` as pitch rate. If it's
not responding correctly, check this axis mapping.

## 2026-04-16 — Created simulation package

New ROS2 package for MuJoCo-based controller tuning on Ubuntu 24.04 VM.
Launch with: `ros2 launch bringup sim.launch.py`
