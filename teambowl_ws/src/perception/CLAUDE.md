# perception

## 2026-04-21 — Fixed ApproximateTimeSynchronizer never firing + 5 Hz OAK-D rate

**Root cause**: The ATS included `info_sub` (CameraInfo) in the synchronizer alongside RGB and depth. CameraInfo timestamps don't match image frame timestamps in depthai_ros_driver, so the ATS queue never found a valid triple and never called `synchronized_callback`. Additionally, the OAK-D was running at ~8/13 Hz with variable jitter, making slop-based matching unreliable.

**Fixes in `cam_ops.py`**:
- CameraInfo is now subscribed separately via `_camera_info_cb` which caches `fx/fy/cx/cy`. Removed from ATS.
- ATS now syncs only RGB + depth (`queue_size=5`, `slop=0.1`). At 5 Hz, 100ms slop is correct.
- `pixel_to_3d` uses cached intrinsics instead of taking `info_msg`.
- `synchronized_callback` signature reduced to `(rgb_msg, depth_msg)`.
- Added intrinsics-ready guard at top of `synchronized_callback` (publishes debug overlay while waiting).
- Default topics corrected to `/oak/rgb/image_raw` and `/oak/stereo/image_raw`.

**New file `bringup/config/oak_cam.yaml`**: Sets OAK-D RGB and stereo to 5 Hz, RGB resolution 720p, `i_align_depth: true`, `i_subpixel: true`. Bringup now passes this file as `params_file` instead of `depthai_ros_driver/config/rgbd.yaml`.


## 2026-04-21 — Removed enable_debug_image gate; debug images always published

Removed `enable_debug_image` parameter and all its guards from `cam_ops.py` and `perception.yaml`.
Debug images now always publish to `/robot/debug/cam_ops_image`.

## 2026-04-21 — Removed timestamp sync check from _try_process

**Root cause**: OAK-D driver occasionally publishes RGB and depth with a 600ms inter-frame gap
(confirmed by live diagnostic). The `sync_slop_s` guard was silently blocking `synchronized_callback`,
so `_last_processed_stamp` never updated and the watchdog fired despite the camera being live.

**Fix**: Removed `sync_slop_s` parameter and the timestamp check from `_try_process`. The node now
calls `synchronized_callback` whenever both latest frames are non-None — the half-rate sampling
already throttles processing, and a slightly stale depth reading (person at 2-8m scale) is irrelevant
for the use case. Watchdog still catches true camera dropout (no callbacks at all).

Files changed: `cam_ops.py` (removed `sync_slop_s` declare/read/check), `perception.yaml` (removed `sync_slop_s` key).

## 2026-04-21 — Robustness hardening + half-rate sampling

**`perception/cam_ops.py`**:
- `_try_process` wrapped in try/except — malformed ROS2 headers no longer crash the node
- `enable_debug_image` parameter (default `false`) — skips frame copy, cv2 draws, and image encoding/publishing when not needed; set to `true` in Foxglove sessions
- Watchdog timer (2s period) — logs a warning if no frame has been processed recently, making silent OAK-D dropout visible in the logs
- Frame counter for half-rate processing: `_frame_count % 2 == 0` in `_rgb_cb` halves CPU load with no config change

## 2026-04-20 — Replaced ApproximateTimeSynchronizer with manual latest-frame sync

**`perception/cam_ops.py`**: Added `_frame_count` counter in `_rgb_cb`. Only calls `_try_process()` on every other RGB frame (`_frame_count % 2 == 0`), halving CPU load with no config change needed.

## 2026-04-20 — Replaced ApproximateTimeSynchronizer with manual latest-frame sync

**`perception/cam_ops.py`**: Removed `message_filters.ApproximateTimeSynchronizer`. It was fragile — if either camera stream had a brief gap (dropped frame, OAK-D hiccup), the synchronizer's internal queue got desynchronized and never recovered without a node restart. Replaced with:
- `_rgb_cb` / `_depth_cb`: store latest frame of each type in `self._latest_rgb` / `self._latest_depth`, then call `_try_process()`
- `_try_process()`: if both frames are non-None and timestamps are within `sync_slop_s`, calls `synchronized_callback(rgb, depth)`
- Self-healing: recovers automatically on the next frame pair, no stateful queue to corrupt

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
