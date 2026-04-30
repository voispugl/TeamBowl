# planning

## 2026-04-30 — Added search circle when user is lost for >3 s

**`planning/follow_goal.py`**: When `target_live` is False and the user has been lost for more than `search_timeout_s` (3 s), `_tick()` now publishes a search goal to `/follow_goal` instead of returning silently. The search goal is placed at `follow_distance_m` in the direction 90° left of the robot's current heading (via `base_link→odom` TF). Since the goal direction is always relative to the robot's current yaw, as MPPI steers the robot left, the goal rotates with it — the robot circles continuously left.

When the person is reacquired, `_search_active` is cleared and `last_smoothed_goal` is reset so the first person-tracking goal isn't smoothed against the last search goal.

**`config/planning.yaml`** and **`config/planning_nvblox.yaml`**: Added `search_timeout_s: 3.0` and `search_offset_rad: 1.5708` (π/2) to the `follow_goal` block.

## 2026-04-30 — Fixed follow_goal backup behavior when person is too close

**`planning/follow_goal.py`** `_compute_goal_in_base()`: Removed the `if distance <= follow_distance_m` branch which was placing the goal *toward* the person even when they were already within the follow distance. The unified formula `user_xy * (distance - follow_distance_m) / distance` now handles all cases: positive (follow), zero (hold), and negative (back up). Also changed the early-return guard from `distance < min_goal_distance_m` to `distance < 0.05` (pure numerical safety). Previously the `max(0.0, ...)` clamp explicitly prevented backup motion, causing the robot to advance into the person.

## 2026-04-30 — Added person_scan_filter: excludes person from obstacle costmap

**`planning/person_scan_filter.py`** (new node): subscribes to `/oak/nav_scan` (LaserScan in base_link) and `/user_pos_base_link` (PointStamped from follow_goal). For each scan, any ray whose endpoint is within `exclusion_radius_m: 1.5` of the person's 2D position in base_link is set to inf. Publishes filtered scan to `/oak/nav_scan_filtered`. Passes scan through unmodified if person position is stale (>`person_timeout_s: 1.0`).

**`config/planning.yaml`**: added `person_scan_filter` params block; local costmap `oak_scan` topic changed from `/oak/nav_scan` → `/oak/nav_scan_filtered`.

**`setup.py`**: added `person_scan_filter` console script entry.

**`bringup/launch/bringup.launch.py`**: added `person_scan_filter` node before `follow_goal`.

## 2026-04-30 — CostCritic tuning: critical_cost fixed, footprint off, observation_persistence added

**`config/planning.yaml`**:
- `CostCritic.critical_cost: 300.0 → 200.0` — costmap max is 254; 300 was unreachable so the critical penalty path never triggered. 200 catches the inflation zone.
- `CostCritic.consider_footprint: false` — kept off. With footprint on, MPPI trajectories that overshoot past the 1m follow_goal extend into the person's inflation zone at their footprint edge, exceeding critical_cost → collision_cost=1000000 → robot refuses to approach the person. Center-only checking: cost at follow_goal is ~33 (well below critical). Real lethal obstacles (cost=254) still block the robot at their inscribed radius.
- `obstacle_layer.observation_persistence: 0.0 → 0.5` — raytrace clearing handles forward movement naturally (scan ray passes through old person position to reach new one, clearing it). Lateral movement leaves orphan marks no ray ever clears; 0.5s expiry ages them out, keeping the costmap clean.

These changes depend on the bringup fix (output_frame: base_link) that enables obstacle detection.

## 2026-04-29 — MPPI tuning: reduce wobble, increase speed

**`config/planning.yaml`** MPPI changes:
- `wz_std: 0.4 → 0.2` — less angular velocity noise in trajectory sampling; was the primary source of yaw wobble
- `iteration_count: 1 → 2` — second refinement pass selects smoother/faster trajectories from the same 2000 samples
- `GoalCritic.threshold_to_consider: 1.4 → 0.5` — robot was decelerating from 1.4m out (nearly always inside this zone when following at 1m distance); now only slows within 0.5m of goal
- `PathAlignCritic.cost_weight: 14.0 → 7.0` — was 3× heavier than any other critic; caused aggressive curve-chasing on Reeds-Shepp paths, producing oscillation
- `PathAlignCritic.use_path_orientations: true` — MPPI now uses path heading hints to know which way to face along curves, reducing the need for angular correction

## 2026-04-29 — Reduced minimum_turning_radius for tighter in-place turns

**`config/planning.yaml`**: `minimum_turning_radius: 0.2 → 0.05` m. The robot is a differential drive and can turn with near-zero radius. The previous 0.2m value caused the planner to produce wide arcs when the person moved to the side, making auton following look sluggish at reorienting. At 0.05m the Reeds-Shepp planner finds tight arc paths close to in-place rotation.

## 2026-04-29 — Fixed YAML integer/float type mismatches crashing follow_executor

**`config/planning.yaml`**: Two bare integers were causing `InvalidParameterTypeException` crashes on node startup:
- `follow_executor.replan_rate_hz: 2` → `2.0` — node declares as DOUBLE, YAML integer is rejected
- `plan_wheels.max_linear_x: 1` → `1.0` — same issue

`follow_executor` was crashing in `__init__` before ever starting, which is why `/follow_path` had no messages and `/cmd_vel_auto` was empty. ROS2 does not coerce YAML integer → DOUBLE parameters.

## 2026-04-29 — Fixed MPPI controller_frequency and local costmap inflation_radius

**`config/planning.yaml`**:
- `controller_frequency: 15.0 → 10.0` — MPPI `model_dt: 0.10` means the controller must run at 1/0.10 = 10 Hz. Running at 15 Hz caused MPPI to roll out trajectories with the wrong timestep, producing near-zero velocity output. Nav2 logged "Controller period is less than model dt".
- `local_costmap.inflation_layer.inflation_radius: 0.3 → 0.4` — robot inscribed radius is 0.324 m; inflation must be ≥ inscribed radius or Nav2 logs `[ERROR]` and MPPI may see the robot as inside an obstacle (zero-velocity trap).

## 2026-04-29 — Removed ghost mode from follow_goal

**`planning/follow_goal.py`**: Removed ghost mode entirely. Ghost mode was republishing the last known goal with a fresh timestamp when the person was lost, to prevent `follow_executor`'s `goal_timeout_s` from expiring. In practice it caused the robot to stop dead: if the robot reached the static ghost position before the person was re-acquired, Nav2's FollowPath action completed and nothing restarted it. Ghost mode was also redundant — `follow_executor` already keeps Nav2 running for `goal_timeout_s: 10.0` seconds after the last received goal without needing refreshed messages. Now: when `target_live=False`, `_tick()` returns immediately; `follow_executor`'s timeout handles the rest.

Removed: `ghost_timeout_s` param/state, `_last_ghost_goal`, `_user_lost_time`, `ghost_pub` (`/follow_goal_ghost_active`), `_publish_ghost()`, all ghost-mode logic in `_tick()`.

**`config/planning.yaml`**: Removed `ghost_timeout_s: 8.0` from `follow_goal` section.

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

## 2026-04-30 — Fixed nvblox plugin class name and local inflation radius in planning_nvblox.yaml

**`config/planning_nvblox.yaml`**: Two fixes for `use_nvblox:=true` launch:
- `nvblox_layer.plugin`: `"nvblox::NvbloxCostmapLayer"` → `"nvblox::nav2::NvbloxCostmapLayer"` (both global and local costmap). The old name is not registered; nav2 logs `does not exist` and the costmap lifecycle configure transition fails.
- `local_costmap.inflation_layer.inflation_radius`: `0.3` → `0.4`. Must be ≥ robot inscribed radius (0.32 m) or Nav2 SMAC planner logs an ERROR and runs degraded. Matches the non-nvblox `planning.yaml` value.

## 2026-04-30 — Added planning_nvblox.yaml to setup.py data_files

`setup.py` `data_files` only listed `planning.yaml`; `planning_nvblox.yaml` was never copied to the install tree, causing "not a file" warnings when launching with `use_nvblox:=true`. Added `'config/planning_nvblox.yaml'` to the same install glob.

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
