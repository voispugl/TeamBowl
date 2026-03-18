# perception

## Package overview

ROS2 Python package for camera-based perception. Detects the user (pink target)
and estimates their 3D position from RGB + depth frames.

## Nodes

### `cam_ops` — `perception/cam_ops.py`
Subscribes to RGB image, depth image, and camera info. Detects pink-colored regions,
estimates depth, and publishes target position on `/user_pos` and validity on `/user_valid`.

## Config

Parameters live in `config/perception.yaml` (installed to `share/perception/config/`).
Loaded by `bringup.launch.py` via native ROS2 YAML parameter loading.

## 2026-03-18 — Moved parameters to config/perception.yaml

- **`config/perception.yaml`**: New file. Contains all `cam_ops_node` parameters
  (topics, thresholds, resize settings, depth limits).
- **`setup.py`**: Added `config/perception.yaml` to `data_files`.
