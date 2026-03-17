# bringup

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
