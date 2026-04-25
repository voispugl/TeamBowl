# state_estimation

## 2026-04-23 — Added VSLAM as third EKF odometry source (odom1)

**`config/state_estimation.yaml`**: Added `odom1: /visual_slam/tracking/odometry` as a third EKF fusion source alongside `imu0` (Xsens Sirius 300) and `odom0` (wheel encoders).

- **Fused from VSLAM**: x, y position; yaw; vx; vyaw (`two_d_mode` projects out z/roll/pitch).
- **Covariance**: reported dynamically by Isaac VSLAM from tracking quality — no manual value needed.
- **Rejection thresholds**: `pose: 2.0`, `twist: 2.0` — guards against pose jumps on VSLAM re-localization.
- **Safe when VSLAM is off**: if `use_vslam:=false`, no messages arrive on `odom1` topic and the EKF silently uses only `imu0` + `odom0`. No parameter change needed to disable.
- **Enabled by**: launching with `use_vslam:=true` (Isaac ROS VSLAM node must be running and Docker rebuilt with Isaac ROS).

## IMU Calibration

Run `bash ~/TeamBowl/calibrate_imu.sh` with the robot stationary for 30+ minutes to record a bag, then `python3 ~/TeamBowl/calibrate_imu.py <bag>` to compute Allan variance and get ready-to-paste YAML for `process_noise_covariance` and `xsens_mti_node.yaml` stddev values.

## 2026-04-19 — IMU Mahalanobis rejection (single subscription)

**`config/state_estimation.yaml`**: Single `imu0` subscription with `imu0_mahalanobis_threshold: 3.0`.
Rejects large IMU jumps (e.g. Xsens yaw settling from 0° to 170° after filter convergence).
The split imu0/imu1 approach was reverted — it caused EKF instability in practice.

## 2026-04-19 — Added wheel speed sanity clamp to diff_drive_odom

**`state_estimation/diff_drive_odom.py`** — added `max_wheel_speed_m_s` parameter (default 3.0 m/s).
If either wheel reports a speed beyond this limit (e.g. garbage ERPM on VESC serial startup), the tick
discards that reading and outputs zero instead of integrating an insane position into the EKF.
**`config/state_estimation.yaml`** — added `max_wheel_speed_m_s: 3.0` under `diff_drive_odom`.

## 2026-04-19 — Applied Allan variance calibration to process_noise_covariance

**`config/state_estimation.yaml`** — replaced datasheet-estimated `process_noise_covariance` with empirically calibrated diagonal values from 74.1-minute stationary bag (`imu_calib_20260419_194758`).

Key diagonal values (Xsens Sirius 300, ARW=0.526 °/√hr):
- roll/pitch/yaw: 1e-08 (clamped to minimum — orientation noise is very low)
- vroll/vpitch/vyaw: 1.170e-06 (from gyro bias instability)
- vx/vy: 1.563e-05 (from accel VRW at 50 Hz)
- ax/ay/az: 1.563e-05 (same)

**`../drivers/xsens_mti_ros2_driver/param/xsens_mti_node.yaml`** — updated stddev values to calibrated ARW and VRW values (see that package's CLAUDE.md).

## 2026-04-19 — EKF config fixes: odom0 yaw conflict + IMU acceleration fusion

**`config/state_estimation.yaml`** is the ACTUAL config loaded by bringup (NOT `locomotion/config/ekf.yaml` which is unused).

### odom0_config fix (yaw conflict)
`/wheel/odometry` starts at yaw=0 on every boot. Fusing that yaw alongside the IMU yaw (64° on first boot) caused the EKF to deadlock at all zeros. Fixed by removing position and yaw from `odom0_config` — now only fuses vx and vyaw (velocities, no absolute pose).

### IMU acceleration fusion (position estimation)
Enabled `ax` and `ay` in `imu0_config` (indices 12, 13 = true) so the EKF integrates IMU linear acceleration into position estimates. Set `imu0_remove_gravitational_acceleration: true` to strip gravity before integration (otherwise gravity along any tilted axis corrupts the estimate).

Final sensor fusion matrix:
- `odom0` (`/wheel/odometry`): vx, vyaw only
- `imu0` (`/imu/data`): yaw, vyaw, ax, ay
- `two_d_mode: true` — z/roll/pitch suppressed

**Note:** IMU acceleration integration is noisy over long distances. Position will drift when the robot is stationary (accelerometer bias). If this causes issues, disable ax/ay and rely solely on wheel odometry for position.
