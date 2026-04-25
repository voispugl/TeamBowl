# TeamBowl

ROS2 Humble robot with 6 × RS04 leg joints (CAN bus), wheel drive (VESC), and an OAK-D-W depth camera.
Runs natively on a Jetson AGX Orin.

---

## Repository Layout

```
TeamBowl/
├── teambowl_docker/        Docker image and compose file (optional)
└── teambowl_ws/            ROS2 workspace
    ├── full_rebuild.sh     Clean build
    ├── source.sh           Source environment
    ├── launch.sh           Source environment + launch full stack
    └── src/
        ├── bringup/        System-wide launch file (see bringup/README.md)
        ├── locomotion/     Leg controllers, velocity mux, collision guard
        ├── management/     Robot mode manager, keyboard operator
        ├── drivers/        robstride_can_driver, vesc_driver, depthai-ros, xsens
        ├── safety/         Heartbeat / system health / e-stop
        ├── perception/     Camera ops
        └── planning/       Autonomous planning
```

For the full node inventory, launch arguments, and config file locations see
[`teambowl_ws/src/bringup/README.md`](teambowl_ws/src/bringup/README.md).

---

## Prerequisites

```bash
pip install aenum          # required by python-can (system package missing this dep)
```

### CAN Interface Setup (Jetson AGX Orin)

The AGX Orin has two built-in SocketCAN interfaces (`can0`, `can1`) driven by the `mttcan` kernel module.
They must be up **on the host** before running ROS or the Docker container.

#### One-shot (current session only)

```bash
sudo modprobe mttcan
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can1 type can bitrate 1000000
sudo ip link set can0 up
sudo ip link set can1 up
```

#### Persistent (survives reboot) — systemd-networkd

```bash
sudo tee /etc/systemd/network/80-can0.network << 'EOF'
[Match]
Name=can0

[CAN]
BitRate=1000000
EOF

sudo tee /etc/systemd/network/80-can1.network << 'EOF'
[Match]
Name=can1

[CAN]
BitRate=1000000
EOF

sudo systemctl enable systemd-networkd
sudo systemctl restart systemd-networkd
```

#### Verify

```bash
ip link show can0    # should show state UP
ip link show can1    # should show state UP
candump can0         # live frames if motors are powered
```

---

## Quick Start (Native — Jetson)

### Terminal 1 — Build and launch full stack

```bash
cd ~/TeamBowl/teambowl_ws
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select \
    robstride_can_interfaces robstride_can_driver \
    bringup locomotion management safety perception planning vesc_driver
source install/setup.bash
ros2 launch bringup bringup.launch.py
```

Or use the convenience script (does the above automatically):

```bash
bash ~/TeamBowl/teambowl_ws/build.sh
```

### Terminal 2 — Keyboard operator

```bash
cd ~/TeamBowl/teambowl_ws
source /opt/ros/humble/setup.bash
source ~/TeamBowl/teambowl_ws/install/setup.bash
ros2 run management keyboard_operator
```

**Keys:**

| Key | Action |
|-----|--------|
| `1` | OFF mode |
| `2` | TELEOP mode |
| `3` | AUTON mode |
| `4` | TRICK mode (leg pose control) |
| `w` / `s` | Forward / back |
| `a` / `d` | Turn left / right |
| `q` / `e` | Forward-left / forward-right |
| `z` / `c` | Back-left / back-right |
| `space` / `x` | Stop |
| `[` / `]` | Decrease / increase linear speed |
| `;` / `'` | Decrease / increase angular speed |
| `j` | (trick mode) All joints → trick offsets |
| `n` | (trick mode) All joints → base driving positions |

---

## Trajectory Testing

Use `trajectory_test.launch.py` to test Nav2 path planning and execution without any
additional setup. It launches the full stack in driving mode automatically.

### Terminal 1 — Launch

```bash
ros2 launch bringup trajectory_test.launch.py
```

Brings up bringup + Nav2 (planner + controller) + driving leg controller, then
auto-sets robot mode to `"driving"` after 3 seconds.

### Foxglove panels

| Panel | Topic | Message |
|-------|-------|---------|
| Publish | `/trajectory_goal` | `{"data": "{\"x\": 2.0, \"y\": 0.0, \"theta\": 0.0, \"relative\": true}"}` |
| Publish | `/trajectory_cmd` | `{"data": "go"}` |
| Raw Messages | `/trajectory_status` | Current state, active goal, errors |
| 3D / Map | `/trajectory_path` | Planned path visualization |

**Workflow:**
1. Publish a goal to `/trajectory_goal` (`relative: true` = robot frame, `false` = odom frame).
2. Publish `"go"` to `/trajectory_cmd` to start execution.
3. The node calls Nav2 `ComputePathToPose` → `FollowPath` at 2 Hz, replanning if the goal moves.

**Other `/trajectory_cmd` values:** `stop` (cancel, stay idle), `reset` (clear goal).

**Planner:** Nav2 SmacPlanner2D (GridBased), goal tolerance 0.25 m / 0.35 rad.
**Controller:** RegulatedPurePursuit, desired speed 0.5 m/s.

---

## Steam Deck Web UI

A browser-based control panel runs at `http://ROBOT_IP:8888` (served by the `steamdeck_ws_teleop` node).

```bash
ros2 launch steamdeck_teleop steamdeck_ws.launch.py
```

Navigate to `http://ROBOT_IP:8888` from any browser (Steam Deck, laptop, phone). Three UI modes selectable via `steamdeck_ui` launch arg:

| `steamdeck_ui` | Description |
|---|---|
| `phone` (default) | ENABLE / TOGGLE LID / KILL + diagnostics. For normal operation. |
| `rescue` | ENABLE + KILL + 4-direction D-pad (↑↓←→). Hold buttons to drive out of tight spots. Publishes to `/cmd_vel_auto`; robot must be in `driving` mode. |
| `full` | Gamepad goals, mode/lid/trajectory/gains panels, nav map. For development sessions. |

No ROS2 or software installation needed on the client device.

> **Note — E-stop bypass:** `disable_estop: true` is set in `steamdeck_teleop.yaml`. This makes
> the web UI ignore incoming `/estop` messages (the hardware e-stop is not wired up yet).
> Set `disable_estop: false` once the `/estop` topic is properly connected.

---

## Incremental Rebuild (after code changes)

No need to clean rebuild if only Python files changed and `--symlink-install` was used:

```bash
cd ~/TeamBowl/teambowl_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select <changed_package>
source install/setup.bash
```

Common packages to rebuild after edits:

| Package | When to rebuild |
|---------|----------------|
| `robstride_can_interfaces` | After changing service/msg definitions |
| `robstride_can_driver` | After editing driver_node.py or motors.yaml |
| `locomotion` | After editing any leg controller |
| `management` | After editing keyboard_operator or mode_manager |

---

## Fall Recovery

The `fall_recovery_controller` node detects fallover and automatically runs a kip-up manoeuvre:

1. **Trigger**: `|pitch| > 0.45 rad` (≈26°) from `/imu/data`
2. **EXTENDING** (2.5 s): slowly splay `joint_rs04_2/3/5/6` outward by 30°
3. **RETRACTING** (0.35 s): fast snap back to driving position (Kd reduced to 2.0 for speed)
4. **SETTLING** (1.5 s): smooth return to exact YAML positions (Kd restored to 15.0)

The node sets robot mode to `"recovery"` during the manoeuvre (stopping the driving leg controller)
and returns to `"driving"` when done. Wheels are locked at zero throughout.

---

## Joint Layout

- **RS04** (`joint_rs04_1` … `joint_rs04_6`, `can0`): actively controlled by leg controllers.
- **RS00** (`joint_rs00_1`, `joint_rs00_2`, `can1`): coast mode (zero gains, damper disabled).
- **RS05** (`joint_rs05_1`, `can1`): cargo bay lid motor. Controlled by `lid_controller`
  node in MIT mode (50 Hz `/joint_commands`). Commands via `/lid_command` (`open` / `close` / `toggle`).

---

## Status LEDs (Pico 2)

The Pico 2 drives a WS2812 NeoPixel strip on GPIO 28. The `pico_bridge` ROS2 node maps robot state to LED colors automatically.

| State | Color | Pattern |
|-------|-------|---------|
| E-stopped | Red | Solid |
| Turning right | Orange | Wave right |
| Turning left | Orange | Wave left |
| Moving forward/back | Yellow | Solid |
| Stuck (`/robot_stuck true`) | Purple | Blink ~3 Hz |
| Teleop, idle | Blue | Solid |
| Alive / default | Green | Solid |

**Build:** `cd pico/status_leds && mkdir build && cd build && cmake .. -DPICO_BOARD=pico2 -DPICO_SDK_PATH=$HOME/pico-sdk && make -j$(nproc)`  
**Flash:** Hold BOOTSEL, plug USB, mount `/dev/sda1`, copy `build/status_leds.uf2`.  
**Serial port:** `/dev/serial/by-id/usb-Raspberry_Pi_Pico_1B8494CFA7EDCDA1-if00` (configured in `safety/config/safety.yaml`).

---

## TODO

- [ ] **YOLO perception** — replace pink-blob detector with YOLOv8 + ByteTrack + Re-ID.
  Requires Docker (Jetson PyTorch CUDA deps are complex to install natively). See branch `try-yolo-perception`.

- [ ] **Dockerize** — containerize the full robot stack so bringup, drivers, and
  dependencies are fully encapsulated and reproducible across machines.

- [ ] **CAN auto-launch on Jetson AGX Orin** — configure CAN interfaces to come
  up automatically on boot. The persistent systemd-networkd setup is documented
  in the [CAN Interface Setup](#can-interface-setup-jetson-agx-orin) section above.
  Steps remaining: confirm interface names on the specific carrier board, enable
  `mttcan` module loading at boot (`/etc/modules-load.d/mttcan.conf`), and verify
  after a cold reboot.

