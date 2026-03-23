# bringup

ROS2 launch package for the TeamBowl robot. A single launch file starts the full
robot stack. Node parameters are defined in per-package YAML config files rather
than inline in the launch file.

---

## Launch File

**`launch/bringup.launch.py`**

Starts all robot nodes. Accepts the following launch arguments:

### `leg_controller`

Selects which leg controller node to start. `hold` and `driving` cannot run
simultaneously — they both send commands to the same RS04 joints.

| Value | Default | Description |
|-------|---------|-------------|
| `hold` | yes | Freezes legs at their current joint positions at enable time. Safe default — no snap to a calibrated pose. |
| `driving` | no | Snaps RS04 joints to the calibrated driving positions defined in `locomotion/config/locomotion.yaml`. Verify legs are near the target pose before enabling. |
| `none` | no | No leg controller is launched. Useful for wheel-only testing or when the CAN motor driver is not running. |

---

## Node Inventory

Nodes that start unconditionally:

| Node | Package | Config file |
|------|---------|-------------|
| `mode_manager` | management | `management/config/management.yaml` |
| `heartbeat_publisher` | safety | `safety/config/safety.yaml` |
| `system_health` | safety | `safety/config/safety.yaml` |
| `vel_cmd_mux` | locomotion | `locomotion/config/locomotion.yaml` |
| `collision_guard` | locomotion | `locomotion/config/locomotion.yaml` |
| `cmd_vel_to_vesc` | vesc_driver | `vesc_driver/config/vesc_driver.yaml` |
| `diff_drive_odom` | state_estimation | `state_estimation/config/state_estimation.yaml` |
| `ekf_filter_node` | robot_localization | `state_estimation/config/state_estimation.yaml` |
| `cam_ops_node` | perception | `perception/config/perception.yaml` |
| `plan_wheels` | planning | `planning/config/planning.yaml` |

Conditional nodes (controlled by `leg_controller` argument):

| Node | Condition |
|------|-----------|
| `hold_position_controller` | `leg_controller:=hold` |
| `driving_leg_controller` | `leg_controller:=driving` |

Included via nested launch files:

| Launch file | Purpose |
|-------------|---------|
| `depthai_ros_driver/camera.launch.py` | OAK-D camera driver (rectification disabled) |
| `robstride_can_driver/driver.launch.py` | CAN motor driver (uses `robstride_can_driver/config/motors.yaml`) |

---

## Config File Locations

All tunable parameters live in per-package YAML files. Changes require rebuilding
the relevant package.

| Package | Config file |
|---------|------------|
| management | `src/management/config/management.yaml` |
| safety | `src/safety/config/safety.yaml` |
| locomotion | `src/locomotion/config/locomotion.yaml` |
| vesc_driver | `src/drivers/vesc_driver/config/vesc_driver.yaml` |
| state_estimation | `src/state_estimation/config/state_estimation.yaml` |
| perception | `src/perception/config/perception.yaml` |
| planning | `src/planning/config/planning.yaml` |
| robstride motors | `src/drivers/robstride_can_driver/config/motors.yaml` |

---

## Robot Mode

The robot starts in `"off"` mode (motors stopped). Mode is controlled at runtime
via the `/robot_mode_set` topic and published on `/robot_mode` by the `mode_manager`
node. Valid modes: `teleop`, `auto`, `off`. In `teleop` mode `vel_cmd_mux` passes
`/cmd_vel_teleop` through; in `auto` mode it passes `/cmd_vel_auto`.
