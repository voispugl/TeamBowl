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
| `2` | Activate teleop mode |
| `1` | Return to off |
| `3` | Autonomous mode |
| `w` / `s` | Forward / back |
| `a` / `d` | Turn left / right |
| `q` / `e` | Forward-left / forward-right |
| `z` / `c` | Back-left / back-right |
| `space` / `x` | Stop |
| `[` / `]` | Decrease / increase linear speed |
| `;` / `'` | Decrease / increase angular speed |

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

## Joint Layout

- **RS04** (`joint_rs04_1` … `joint_rs04_6`, `can0`): actively controlled by leg controllers.
- **RS00** (`joint_rs00_1`, `joint_rs00_2`, `can1`): coast mode (zero gains, damper disabled).
- **RS05** (`joint_rs05_1`, `can1`): currently unplugged — ignored.

---

## TODO

- [ ] **Dockerize** — containerize the full robot stack so bringup, drivers, and
  dependencies are fully encapsulated and reproducible across machines.

- [ ] **CAN auto-launch on Jetson AGX Orin** — configure CAN interfaces to come
  up automatically on boot. The persistent systemd-networkd setup is documented
  in the [CAN Interface Setup](#can-interface-setup-jetson-agx-orin) section above.
  Steps remaining: confirm interface names on the specific carrier board, enable
  `mttcan` module loading at boot (`/etc/modules-load.d/mttcan.conf`), and verify
  after a cold reboot.

