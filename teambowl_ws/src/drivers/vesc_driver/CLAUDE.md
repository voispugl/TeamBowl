# vesc_driver

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

## 2026-03-18 — Moved parameters to config/vesc_driver.yaml

- **`config/vesc_driver.yaml`**: New file. Contains all `cmd_vel_to_vesc` parameters
  (serial ports, wheel geometry, ERPM limits, baud rate, signs).
- **`setup.py`**: Added `config/vesc_driver.yaml` to `data_files`.
