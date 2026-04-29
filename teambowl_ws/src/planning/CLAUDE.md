# planning

## 2026-04-28 — controller_frequency 20 → 10 Hz; nav_cloud_filter retired

**`config/planning.yaml`**: `controller_frequency: 20.0 → 10.0` — MPPI was missing its 50ms budget; 100ms is achievable and sufficient for walking-speed person following.

**`nav_cloud_filter` node** is no longer launched (removed from bringup). Its config block remains in `planning.yaml` but is unused. Obstacle detection now goes directly depth image → depthimage_to_laserscan → `/oak/nav_scan`.

## 2026-04-21 — Reduced planner/controller load (person-following tuning)

**`config/planning.yaml`**:
- `max_planning_time: 5.0 → 0.5` — fail fast and replan next tick rather than blocking 5s
- `lookup_table_size: 20.0 → 10.0` — halves startup precompute for SmacPlannerHybrid
- `controller_frequency: 20.0 → 10.0` — halves MPPI cost (112K → 56K trajectory evals/step)
- `batch_size: 2000 → 1000` — halves MPPI trajectories per step


## 2026-04-20 — Fixed follow_goal TF timestamp lookup causing silent goal drop

**`planning/follow_goal.py`**: `_transform_point_msg` was using `Time.from_msg(msg.header.stamp)` (the camera image timestamp) for TF lookup. TF2 can fail to find a transform at that exact historical time if the buffer doesn't go back far enough, causing silent `TransformException` and no `/follow_goal` output. Changed to `Time()` (latest available transform), which always works for the static `oak_rgb_camera_optical_frame → base_link` transform.

## 2026-04-20 — Smoothed auton person-following: replan rate, min goal change, max velocity

**`config/planning.yaml`**:
- `follow_executor.replan_rate_hz`: 2.0 → 1.0 — fewer cancel/replan interruptions per second
- `follow_executor.min_goal_change_m`: 0.10 → 0.25 — only replan when goal moves ≥ 25 cm
- `FollowPath.vx_max`: 0.5 → 0.3 m/s — reduces aggressive acceleration during following

## 2026-04-20 — Fixed follow_executor cross-talk bug; re-added to bringup

**`planning/follow_executor.py`**: `_tick()` was erroneously calling `_request_path()` even
when `robot_mode != autonomous_mode_name` (e.g., during `driving`-mode trajectory tests).
This caused competing Nav2 action goals → "Planner rejected goal" cross-talk with
`trajectory_test`. Fixed to `return` immediately when not in auton mode. Mode-exit
cancellation is already handled by `_mode_cb`.

**`bringup/launch/bringup.launch.py`**: Re-added `follow_goal` and `follow_executor` nodes
(removed 2026-04-19). Now safe to coexist with `trajectory_test` because the two nodes
use separate robot modes: `follow_executor` only sends Nav2 goals in `auton` mode,
`trajectory_test` only sends goals in `driving`/`balance` mode.

## Package overview

ROS2 Python package for autonomous following behavior and trajectory testing.

## Nodes

### `trajectory_test` — `planning/trajectory_test.py`
Foxglove-driven trajectory test node for tuning `driving_controller`.
Active only in `"driving"` mode. Accepts a JSON goal, converts it to an
odom-frame PoseStamped, then live-replans + executes via nav2 at 2 Hz.

**Foxglove usage:**
1. `ros2 launch bringup trajectory_test.launch.py` — starts stack in driving mode
2. Publish to `/trajectory_goal`: `{"data": "{\"x\": 2.0, \"y\": 0.0, \"theta\": 0.0, \"relative\": true}"}`
3. Publish to `/trajectory_cmd`: `{"data": "go"}` — starts live execution
4. Watch `/trajectory_path` (nav_msgs/Path) in Foxglove 3D panel
5. Read `/trajectory_status` for `{"state": "RUNNING", "goal_x": ..., ...}`
6. Stop: `/trajectory_cmd` → `{"data": "stop"}`

**State machine:** `IDLE ↔ RUNNING` (replans at 2 Hz while RUNNING)
**Pipeline:** trajectory_test → ComputePathToPose → nav2_planner → FollowPath →
nav2_controller → /cmd_vel_auto → vel_cmd_mux → driving_controller → wheels

### `follow_executor` — `planning/follow_executor.py`
Nav2 action client: sends `ComputePathToPose` + `FollowPath` for person-following.
Active in `auton` mode. Subscribes to `/follow_goal` (PoseStamped from follow_goal node).

### `follow_goal` — `planning/follow_goal.py`
Converts `/user_pos` (person detection) to `/follow_goal` (PoseStamped in odom frame).

### `nav_cloud_filter` — `planning/nav_cloud_filter.py`
Filters `/oak/points` → `/oak/nav_points` for nav2 obstacle detection.

### `plan_wheels` — `planning/plan_wheels.py`
Legacy reactive PD person-follower. Publishes `/cmd_vel_auto` directly.

## Config

Parameters live in `config/planning.yaml` (installed to `share/planning/config/`).
Loaded by `bringup.launch.py` via native ROS2 YAML parameter loading.

## 2026-04-20 — Swapped RPP → MPPI controller (try-mppi branch)

**`config/planning.yaml`** — `FollowPath` controller replaced:
- Plugin: `nav2_mppi_controller::MPPIController`
- `vx_max: 0.5`, `vx_min: -0.5`, `wz_max: 1.9` — velocity bounds enforced by controller
- `batch_size: 2000`, `time_steps: 56`, `model_dt: 0.05` — 2.8s horizon, 2000 samples
- `motion_model: "DiffDrive"` — matches robot kinematics
- Critics: Constraint, Cost, Goal, GoalAngle, PathAlign, PathFollow, PathAngle, PreferForward
- Roll back: `git checkout git-how -- teambowl_ws/src/planning/config/planning.yaml`

## 2026-04-19 — Faster Nav2 startup: reduced lookup table + increased bond timeout

**`config/planning.yaml`**:
- `lifecycle_manager_navigation.bond_timeout: 20.0` (was default 4.0) — prevents lifecycle manager
  from declaring planner_server "unconfigured" before SmacPlannerHybrid finishes its precomputation.
- `angle_quantization_bins: 36` (was 72, 10° vs 5° resolution) — halves the lookup table entries.
- `lookup_table_size: 10.0` (was 20.0) — halves the range, further reducing precompute time.

These reduce startup from ~10–15s to ~3–5s with acceptable path quality loss for a robot-radius corridor.

## 2026-04-20 — Enabled use_rotate_to_heading in RPP controller

**`config/planning.yaml`** — `FollowPath` (RegulatedPurePursuitController):
- `use_rotate_to_heading: true` — robot turns in place to face goal heading near XY tolerance
- `allow_reversing: false` — required; RPP disallows both flags simultaneously

## 2026-04-20 — Reduced minimum_turning_radius + enabled plan_wheels reverse

**`config/planning.yaml`**
- `minimum_turning_radius: 0.4` m (was 1.0) — allows tighter Reeds-Shepp arcs so the planner can find reverse paths in constrained spaces.
- `allow_reverse: true` in `plan_wheels` — reactive follower will now back up when the target is closer than `follow_distance_m`.

## 2026-04-19 — Enabled reversing: SmacPlannerHybrid + RPP allow_reversing

**`config/planning.yaml`**
- Planner switched from `SmacPlanner2D` → `SmacPlannerHybrid` with `motion_model_for_search: "REEDS_SHEPP"` — planner now generates reverse arcs as part of the path.
- `minimum_turning_radius: 0.40` m (tune if robot turns tighter/wider).
- `reverse_penalty: 1.5` — robot prefers forward but will reverse when shorter.
- Controller `allow_reversing: true` — RPP now executes reverse segments from the planner.

## 2026-04-16 — Added trajectory_test node

- **`planning/trajectory_test.py`**: New node. Foxglove-driven test tool for
  tuning `driving_controller`. Accepts JSON goals, calls nav2 ComputePathToPose +
  FollowPath actions, live-replans at 2 Hz. Active only in `"driving"` mode.
- **`config/planning.yaml`**: Added `trajectory_test` section with all parameters.
- **`setup.py`**: Added `trajectory_test` console script entry point.
- **`bringup/launch/trajectory_test.launch.py`**: New auto-start launch file.
  Starts full bringup with `velocity_controller:=driving leg_controller:=driving`,
  then auto-sets mode to `"driving"` after 3 s.

## 2026-03-18 — Moved parameters to config/planning.yaml

- **`config/planning.yaml`**: New file. Contains all `plan_wheels` parameters
  (topics, gains, distance thresholds, speed limits, reverse settings).
- **`setup.py`**: Added `config/planning.yaml` to `data_files`.
