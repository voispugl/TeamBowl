## Package: robstride_can_driver

ROS2 Humble Python package for controlling 9 RobStride actuators (6×RS04 on can0, 2×RS00 + 1×RS05 on can1) on a Jetson AGX Orin 64GB. Uses the Private Protocol over CAN 2.0, 1 Mbps, 29-bit extended frames via SocketCAN.

**Location:** `teambowl_ws/src/drivers/robstride_can_driver/`

### Build
```bash
cd ~/teambowl_ws
colcon build --packages-select robstride_can_driver
source install/setup.bash
```

### Run
```bash
# Bring up CAN interfaces first (Jetson built-in CAN)
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000

# Launch driver
ros2 launch robstride_can_driver driver.launch.py

# Or with startup_home mode
ros2 launch robstride_can_driver driver.launch.py startup_mode:=startup_home
```

### Key design decisions
- **Kp/Kd are node-side only** — gains in `config/motors.yaml` are used in the Type 1 CAN frame but are NOT written to actuator flash. Use `/write_motor_param` + `/save_motor_params` if you want to persist gains on the motor.
- **No auto-PID setup** — `setup_pids_on_startup: false` in motors.yaml by default.
- **Startup modes**: `startup_safe` holds current position on enable (no snap); `startup_home` moves to `home_position_rad` per joint.
- **Three zero methods**: `/set_zero` (Type 6), `/shift_zero` (iterative add_offset), `/set_zero_offset` (absolute add_offset).

### Module overview
| File | Purpose |
|---|---|
| `robstride_can_driver/can_protocol.py` | Pure CAN frame encode/decode, no ROS2/python-can |
| `robstride_can_driver/motor_config.py` | Dataclasses + YAML loader |
| `robstride_can_driver/driver_node.py` | ROS2 node, CAN threads, control loop, services |
| `config/motors.yaml` | **Edit this** — motor IDs, joint names, gains, home positions |
| `config/commands_reference.yaml` | Machine-readable protocol reference |
| `tools/commissioning.py` | Standalone CLI for motor setup (no ROS2 needed) |
| `docs/COMMANDS.md` | Full human-readable reference |
| `test/test_can_protocol.py` | Unit tests (no hardware needed): `pytest test/` |
