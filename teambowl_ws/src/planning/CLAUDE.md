# planning

## Package overview

ROS2 Python package for autonomous following behavior.

## Nodes

### `plan_wheels` — `planning/plan_wheels.py`
Subscribes to `/user_pos` and `/user_valid`, publishes velocity commands on
`/cmd_vel_auto` to follow the detected user at a target distance.

## Config

Parameters live in `config/planning.yaml` (installed to `share/planning/config/`).
Loaded by `bringup.launch.py` via native ROS2 YAML parameter loading.

## 2026-03-18 — Moved parameters to config/planning.yaml

- **`config/planning.yaml`**: New file. Contains all `plan_wheels` parameters
  (topics, gains, distance thresholds, speed limits, reverse settings).
- **`setup.py`**: Added `config/planning.yaml` to `data_files`.
