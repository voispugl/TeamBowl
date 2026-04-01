# management

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

## 2026-03-31 — Added trick mode

- **`mode_manager.py`**: Added `'trick'` to `VALID_MODES`. No other logic changes needed.
- **`keyboard_operator.py`**: Added trick mode (key `4`). Loads `trick_leg_offsets.yaml`
  from `share/locomotion/` at startup. Tracks `_trick_pose_active` (bool) and
  `_requested_mode` (str). Key `j` sets all joints to YAML offsets; key `n` returns
  all joints to base (stays in trick mode). Publishes `/trick_leg_offsets` (JointState)
  every tick; `driving_leg_controller` only applies offsets when mode == `'trick'`.
  Pressing `1`/`2`/`3` resets `_trick_pose_active` to False.

## 2026-03-18 — Moved parameters to config/management.yaml

- **`config/management.yaml`**: New file. Contains all `mode_manager` parameters
  (`mode_topic`, `mode_set_topic`, `start_mode`, `publish_rate_hz`).
- **`setup.py`**: Added `config/management.yaml` to `data_files` so it installs
  to `share/management/config/`.
