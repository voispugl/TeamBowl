# perception

## 2026-04-30 — Faster relock and looser track matching for large robot movements

**`config/perception.yaml`**:
- `target_lost_timeout_s: 120.0 → 3.0` — primary fix. After BoT-SORT drops a track and assigns a new ID to the same person, `yolo26_node` was waiting 2 minutes before triggering ReID relock. Now resets in 3s and relocks via OSNet appearance embedding.
- `reid_threshold: 0.3 → 0.25` — slightly more permissive appearance match on relock, since person appearance can shift during large/fast movements.

**`config/botsort.yaml`**:
- `match_thresh: 0.5 → 0.3` — lower IoU threshold keeps BoT-SORT track alive through larger bounding-box shifts caused by camera motion. GMC (sparseOptFlow) compensates for camera movement but can fail on fast/blurry motion; this adds tolerance.

## 2026-04-29 — Swapped model from yolo26l → yolo26m

**`config/perception.yaml`**: `model_path` updated to `yolo26m.engine`. Medium variant trades ~3% mAP for faster inference on the Jetson — better real-time tracking cadence.

**`scripts/export_yolo26.py`**: `--model` and `--out` defaults updated to `yolo26m`. Re-run `python3 export_yolo26.py` to generate the new engine (~15-20 min).

**`teambowl_docker/export_model.sh`**: Comments and `--out` flag updated to reference `yolo26m.engine`.

## 2026-04-29 — Upgraded to yolo26l + custom OSNet ReID for crowded environments

**`config/perception.yaml`**: `model_path` updated to `yolo26l.engine`. yolo26l gives significantly better mAP (~55 vs ~48) over yolo26s, which matters for detecting partially occluded persons in crowds. New parameters: `reid_weights`, `reid_threshold` (0.65), `reid_update_interval_s` (1.0).

**`perception/yolo26_node.py`**: Added custom appearance ReID using boxmot's OSNet (`osnet_x0_25_msmt17.pt`, ~3 MB, ~2ms/crop on GPU). Three new helpers (`_get_reid_encoder`, `_get_embedding`, `_cosine_sim`). Appearance embeddings are updated via EMA at most once per second while the target is visible and confirmed by depth. When re-locking after loss, all detected candidates are compared against the saved embedding; best match above `reid_threshold` wins, otherwise falls back to largest person. This prevents the robot from accidentally locking onto a bystander in a crowded scene.

**`scripts/export_yolo26.py`**: Default model changed `yolo26s → yolo26l`. The `.pt` file is now preserved in `models/` after export (needed for future use; OSNet weights are separate at `models/osnet_x0_25_msmt17.pt`).

**One-time setup required**: Run `python3 export_yolo26.py` to generate `yolo26l.engine` (~15-20 min). `osnet_x0_25_msmt17.pt` is already in `models/`.

## 2026-04-29 — Enabled BoT-SORT appearance ReID via boxmot

**`config/botsort.yaml`**: Enabled appearance ReID (`with_reid: True`, `model: osnet_x0_25_msmt17`). BoT-SORT now maintains a lost-tracks pool and re-assigns the original track ID when a person re-enters the frame, matching appearance via OSNet cosine similarity (`appearance_thresh: 0.25`). OSNet runs on GPU via PyTorch (torchvision already installed); `boxmot` must be installed once with `pip install boxmot`. First launch downloads the ~6MB model to `~/.cache/`.

`track_buffer: 150 → 30` — 6s at 5 Hz. The longer buffer was left over from before the ghost-depth fix and is no longer needed; BoT-SORT's ReID pool handles longer-term re-association.

**`config/perception.yaml`**: `target_lost_timeout_s: 3600.0 → 120.0` — 1 hour was never intentional; 2 minutes is generous and allows the relock timer to eventually fire if ReID fails to re-associate.

**Tuning `appearance_thresh`**: If the wrong person gets re-locked, lower toward 0.15 (stricter). If the same person fails to re-lock, raise toward 0.35 (looser).

## 2026-04-28 — Fixed user_valid staying True after person leaves

**`config/bytetrack.yaml`**: `track_buffer: 150 → 5` — 150 frames at 5 Hz = 30s of Kalman prediction, which kept the bounding box alive at the last known position. Depth there was often valid (wall/floor), so `target_xyz` stayed non-None → `user_valid=True` → lights stayed green. 5 frames ≈ 1s is enough for brief occlusion recovery.

**`perception/yolo26_node.py`**: Moved `_target_last_seen = now_s` from "any frame where target is in tracker list" to inside `if z is not None:` (only when depth is confirmed). Previously, ByteTrack predictions kept `_target_last_seen` fresh even without a real 3D detection, preventing the 15-second relock timer from ever firing.

## 2026-04-21 — Added ByteTrack identity tracking to yolo26_node

**`perception/yolo26_node.py`**: Replaced `model()` inference with `model.track(persist=True, tracker='bytetrack.yaml')`. ByteTrack assigns a persistent integer ID to each person across frames. Node auto-locks onto the largest person on first detection and follows that track ID. If the target is absent for `target_lost_timeout_s` (default 3s), the lock resets and re-locks on the next detection. New topic `/yolo26/target_id` (Int32, -1 = no lock) published every frame. Debug image marks the locked target in green (thick) and others in gray (thin).

## 2026-04-21 — Added YOLO26 standalone node + TensorRT export pipeline

**Goal**: ML-based person detection running on the Jetson GPU via TensorRT, alongside the existing pink HSV tracker (cam_ops unchanged).

**New files:**
- `perception/yolo26_node.py` — ROS2 node that loads a `yolo26n.engine` TensorRT model and runs YOLO26 inference on each synced RGB+depth frame. Publishes:
  - `/yolo26/detections` (vision_msgs/Detection2DArray) — all person bounding boxes + confidence
  - `/yolo26/user_pos` (geometry_msgs/PointStamped) — 3D position of largest (closest) person
  - `/yolo26/user_valid` (std_msgs/Bool) — detection validity
  - `/yolo26/debug_image` (sensor_msgs/Image) — annotated BGR view
- `scripts/export_yolo26.py` — one-shot script to download `yolo26n.pt` and export a TRT FP16 engine to `~/TeamBowl/models/yolo26n.engine`. Run once on the Jetson before launching the node.

**Modified files:**
- `setup.py` — added `yolo26_node = perception.yolo26_node:main` entry point
- `config/perception.yaml` — added `yolo26_node` parameter block (`model_path`, `min_confidence`, depth limits)

**bringup.launch.py** — added `use_yolo26` launch arg (default `false`) and conditional TimerAction(10s) for `yolo26_node`. Enable with: `ros2 launch bringup bringup.launch.py use_yolo26:=true`

**ML stack installed (system-wide pip):**
- `torch 2.10.0` + `torchvision 0.25.0` — Ultralytics Jetson ARM64 wheels (replace the broken nv24.8 / PyPI pairing)
- `ultralytics 8.4.40` — supports YOLO26 models
- `onnxruntime-gpu 1.23.0` — Jetson ARM64 wheel for ONNX export path
- `cudss 0.7.1` — installed via local .deb for TensorRT solver support

**To export the engine (first time):**
```bash
python3 ~/TeamBowl/teambowl_ws/src/perception/scripts/export_yolo26.py
```
Takes ~10 min. Engine is saved to `~/TeamBowl/models/yolo26n.engine`.

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
