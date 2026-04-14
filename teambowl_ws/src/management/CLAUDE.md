# management

## 2026-04-13 — Added "balance" to valid modes

- **`management/mode_manager.py`**: Added `"balance"` to `VALID_MODES`. Set via:
  `ros2 topic pub /robot_mode_set std_msgs/msg/String '{data: "balance"}' --once`



## Package overview

ROS2 Python package containing robot mode management and keyboard operator nodes.

## Nodes

### `mode_manager` — `management/mode_manager.py`
Manages the robot's operational mode. Subscribes to `/robot_mode_set` and publishes
the current mode on `/robot_mode` at `publish_rate_hz`.

### `keyboard_operator` — `management/keyboard_operator.py`
Keyboard-based operator interface.

## Config

Parameters live in `config/management.yaml` (installed to `share/management/config/`).
Loaded by `bringup.launch.py` via native ROS2 YAML parameter loading.

## 2026-03-18 — Moved parameters to config/management.yaml

- **`config/management.yaml`**: New file. Contains all `mode_manager` parameters
  (`mode_topic`, `mode_set_topic`, `start_mode`, `publish_rate_hz`).
- **`setup.py`**: Added `config/management.yaml` to `data_files` so it installs
  to `share/management/config/`.
