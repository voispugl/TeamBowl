# bringup

## 2026-04-17 — Replaced led_controller with pico_bridge

- **`launch/bringup.launch.py`**: Replaced `management/led_controller` node with `safety/pico_bridge`. The pico_bridge node handles both LED state signaling (via Pico USB-serial) and the physical kill switch / lid toggle button. Uses `safety_config` (safety.yaml) for parameters.

## 2026-04-16 — Added sim.launch.py for MuJoCo simulation tuning

- **`launch/sim.launch.py`**: Minimal launch stack for controller tuning on Ubuntu VM.
  Launches `mujoco_bridge` (simulation pkg) + `mode_manager` + `vel_cmd_mux` +
  `collision_guard` + `balance_controller` or `driving_controller` (conditional) +
  `foxglove_bridge`. Does NOT launch hardware drivers (robstride, vesc, xsens, depthai),
  EKF, or nav2 stack.
  - `velocity_controller` arg: `balance` (default) or `driving`
  - Static TF: `base_link→imu_link` (identity, sim IMU = body frame)
  - Static TF: `map→odom` (identity, no SLAM in sim)
  - Usage: `ros2 launch bringup sim.launch.py`

## 2026-04-16 — Added trajectory_test node + trajectory_test.launch.py

- **`launch/bringup.launch.py`**: Added `trajectory_test` node (planning pkg,
  `planning.yaml` config). Always launched; idle outside `"driving"` mode.
- **`launch/trajectory_test.launch.py`**: New one-command launch file for
  trajectory tuning sessions. Includes full bringup with
  `velocity_controller:=driving leg_controller:=driving` and auto-sets
  robot mode to `"driving"` after 3 s via `TimerAction` + `ExecuteProcess`.
  Usage: `ros2 launch bringup trajectory_test.launch.py`

## 2026-04-13 — Added lid_controller to bringup

- **`launch/bringup.launch.py`**: Added `lid_controller` node (locomotion pkg, `lid_controller.yaml`).
  Drives the RS05 motor on the cargo bay lid. Subscribes to `/lid_command` (std_msgs/String).
  Commandable from Foxglove Publish panel without `/robot_mode` dependency.


## 2026-04-13 — Added foxglove_bridge, Xsens IMU, EKF, balance_controller, wheel_odom to bringup

- **`launch/bringup.launch.py`**:
  - Added `xsens_imu` include launch for `xsens_mti_ros2_driver/xsens_mti_node.launch.py`
  - Added `balance_controller` node (locomotion pkg, `balance_controller.yaml`)
  - Added `wheel_odom` node (locomotion pkg, `balance_controller.yaml`)
  - Added `ekf_filter_node` from `robot_localization` pkg (`ekf.yaml`)
  - `collision_guard` now outputs to `/cmd_vel_safe` (changed in `locomotion.yaml`);
    `balance_controller` receives this and publishes to `/cmd_vel` for VESCs.
  - Added `foxglove_bridge` node (port 8765). Disabled if package not installed.
    Disable at launch time: `ros2 launch bringup bringup.launch.py foxglove:=false`
    Install: `sudo apt install ros-humble-foxglove-bridge`



## 2026-03-18 — Pass oak_d_pro_w.yaml to camera launch

- **`launch/bringup.launch.py`**: Added `params_file` pointing to
  `depthai_ros_driver/config/oak_d_pro_w.yaml` in the `oak_camera` launch arguments.
  Previously the camera defaulted to `camera.yaml`, which doesn't set `i_resolution: '720'`
  for the OAK-D-W — causing the camera to run at full/default resolution. Now 720p is
  correctly enforced.

## 2026-03-24 — Switched default leg controller to driving_leg_controller

- **`launch/bringup.launch.py`**: Changed `leg_controller` default from `hold` to `driving`.
  Robot now moves to and holds the YAML target positions on enable instead of freezing at
  current positions.

## 2026-03-18 — Added leg_controller launch argument + README

- **`launch/bringup.launch.py`**: Added `leg_controller` launch argument (`hold`
  default, options: `hold`, `driving`, `none`). Uses `IfCondition`/`PythonExpression`
  to conditionally launch the selected controller. `hold` and `driving` cannot run
  simultaneously. Pass as `ros2 launch bringup bringup.launch.py leg_controller:=driving`.
- **`README.md`**: Created. Documents all launch arguments, node inventory, mode
  setting, config file locations, and tuning workflow.

## 2026-03-18 — Moved all inline node parameters to per-package YAML config files

- **`launch/bringup.launch.py`**: Replaced all inline `parameters=[{...}]` dicts
  with `parameters=[path_to_yaml]` loading from each package's installed
  `config/<pkg>.yaml`. Uses `get_package_share_directory` to resolve paths.
  No functional change — same parameter values, now editable per-package without
  touching the launch file.

## 2026-03-17 — Switched default leg controller to hold_position_controller

- **`launch/bringup.launch.py`**: Replaced `driving_leg_controller` with `hold_position_controller` as the default. Robot freezes at current joint positions on enable instead of snapping to calibrated YAML positions.

## 2026-03-17 — Added robstride_can_driver to bringup

- **`launch/bringup.launch.py`**: Added `IncludeLaunchDescription` for `robstride_can_driver/driver.launch.py` so the CAN motor driver starts automatically with the rest of the stack. Uses default `motors.yaml` config.

## 2026-03-17 — Fixed OAK-D-W camera crash + cam_ops topic

## 2026-03-16 — Added driving_leg_controller node

- **`launch/bringup.launch.py`**: Added `driving_leg_controller` node from the
  `locomotion` package. It holds RS04 leg joints at driving positions via MIT
  mode, enables motors on mode transitions, and stops motors when mode is "off"
  or e-stop fires. RS00 joints are coasted (zero gains) and RS05 is ignored
  (unplugged). Node tolerates the robstride driver starting before/after via
  `wait_for_service`.

---

### What changed
- **`launch/bringup.launch.py`**:
  - Added `rectify_rgb: 'false'` to the `oak_camera` launch arguments. The `image_proc::RectifyNode` was crashing with an OpenCV remap assertion because the OAK-D-W wide-angle camera's calibration maps don't match the image dimensions. Disabling the rectify node prevents the crash.
  - Changed `cam_ops` `image_topic` from `/oak/rgb/image_rect` to `/oak/rgb/image_raw`. Since the rectify node is disabled, `image_rect` is no longer published. Color-based person tracking does not require lens rectification.
