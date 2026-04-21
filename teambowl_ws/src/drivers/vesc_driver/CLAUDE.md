# vesc_driver

## 2026-04-20 — Coast wheel motors when robot mode is 'off'

**`vesc_driver/cmd_vel_to_vesc.py`**:
- Added `robot_mode_topic` parameter (default `/robot_mode`) with TRANSIENT_LOCAL subscription.
- Added `self._coasting` flag (bool). Set `True` when mode → `'off'`; cleared when mode → anything else.
- New `_mode_cb`: on mode `'off'`, calls `_send_stop()` immediately (sends `SetDutyCycle(0)` = coast) and sets `_coasting=True`.
- `_tick` returns early if `_coasting`, resending `SetDutyCycle(0)` at 20 Hz to hold coast.
- Added `DurabilityPolicy` to rclpy.qos imports.

**`config/vesc_driver.yaml`**: added `robot_mode_topic: /robot_mode`.

## 2026-04-20 — Added velocity/yaw closed-loop PI

**`vesc_driver/cmd_vel_to_vesc.py`**:
- Added `kp_v`, `ki_v`, `kp_w`, `ki_w`, `vesc_integral_max` params (default 0.0 — open-loop; existing behaviour unchanged).
- Moved ERPM computation from `_cmd_reader` (which now just stores `_v_cmd`, `_w_cmd`) into `_tick` via new `_cmd_to_erpm(v, w)` helper.
- `_tick` computes `v_measured`/`w_measured` from `left_measured_rad_s`/`right_measured_rad_s`, runs PI corrections, then calls `_cmd_to_erpm(v_eff, w_eff)`.
- Integrals reset to 0 on estop and cmd_vel timeout.
- Added `_publish_vesc_gains_echo()` (2 Hz) → `/vesc_gains_echo` (JSON: gains + `_v_measured`, `_w_measured`).
- Added `_on_vesc_gains()` subscriber on `/vesc_gains` — updates gains live, resets integrals when ki changes.
- Added `import json`; added `String` to std_msgs imports.

**`config/vesc_driver.yaml`**: added `kp_v`, `ki_v`, `kp_w`, `ki_w`, `vesc_integral_max`, `vesc_gains_echo_topic`, `vesc_gains_topic`.

## Package overview

ROS2 Python package that converts `/cmd_vel` Twist messages into ERPM commands
sent over serial to left and right VESC motor controllers.

## Nodes

### `cmd_vel_to_vesc` — `vesc_driver/cmd_vel_to_vesc.py`
Subscribes to `/cmd_vel` and `/estop`. Converts linear/angular velocity to
per-wheel ERPM using wheel radius and track width. Sends commands via serial to
two VESC controllers.

## Config

Parameters live in `config/vesc_driver.yaml` (installed to `share/vesc_driver/config/`).
Loaded by `bringup.launch.py` via native ROS2 YAML parameter loading.

## 2026-04-19 — Battery voltage from COMM_GET_VALUES

**`vesc_driver/cmd_vel_to_vesc.py`**: The existing `COMM_GET_VALUES` poll already returns input
voltage at payload byte 27 (int16 unsigned, /10 = V). Added `decode_voltage_from_values_payload()`,
store `self.left_voltage` and `self.right_voltage` in `_feedback_loop`, publish `min(voltages)` on
`/vesc/battery_voltage` (Float64) in `_publish_feedback`. New param `battery_voltage_topic`
(default `/vesc/battery_voltage`). The web UI shows green/yellow/red coloring at >44 V / 42–44 V / <42 V.

## 2026-04-19 — Stable udev symlinks for VESC serial ports

Both VESCs have identical USB serial numbers (`304`) so ttyACMx order is
non-deterministic. Created `/etc/udev/rules.d/99-vesc.rules` using physical
USB port paths (confirmed by spin test) to create stable symlinks:

- `/dev/vesc_left`  → USB port `1-4.2` (Jetson hub port 2)
- `/dev/vesc_right` → USB port `1-4.1` (Jetson hub port 1)

Updated `config/vesc_driver.yaml` to use `left_port: /dev/vesc_left` and
`right_port: /dev/vesc_right`. Symlinks survive reboot and USB replug as long
as the VESCs stay in the same physical hub ports.

If VESCs are moved to different ports, re-run `udevadm info -n /dev/ttyACMx`
to get the new `KERNELS` path and update `/etc/udev/rules.d/99-vesc.rules`.

## 2026-04-19 — Fixed feedback threads always reading zero (ports opened after threads started)

**`vesc_driver/cmd_vel_to_vesc.py`**
- Bug: feedback threads were started before `_open_ports()`, so they received `ser=None` and
  skipped every poll iteration. Result: `/wheel_vel_left` and `/wheel_vel_right` always
  published 0.0 regardless of actual wheel motion → EKF position never updated.
- Fix: moved `_open_ports()` before thread creation. Changed `_feedback_loop(ser, side)` to
  `_feedback_loop(side)` — reads `self.left_ser`/`self.right_ser` directly each iteration
  so it always has the live serial object.

## 2026-04-19 — Moved feedback serial I/O to background daemon threads

**`vesc_driver/cmd_vel_to_vesc.py`**
- `_poll_feedback` (ROS timer callback) was doing blocking `ser.read()` inside the executor,
  causing slow shutdown (5-10 s SIGINT drain). Fix: split into two layers:
  - `_feedback_loop(ser, side)` — daemon thread per VESC; does blocking serial request/response,
    writes to `left_measured_rad_s` / `right_measured_rad_s`. Exits when `_shutdown=True`.
  - `_publish_feedback()` — ROS timer; reads shared floats and publishes, no serial I/O.
- Added `self._shutdown = False`; set to `True` at top of `destroy_node()` so threads exit
  promptly on teardown.
- Added `import threading`, `import time`.

## 2026-03-18 — Moved parameters to config/vesc_driver.yaml

- **`config/vesc_driver.yaml`**: New file. Contains all `cmd_vel_to_vesc` parameters
  (serial ports, wheel geometry, ERPM limits, baud rate, signs).
- **`setup.py`**: Added `config/vesc_driver.yaml` to `data_files`.
