---
## 2026-04-20 — Added gains echo publisher and gains subscriber

**`driver_node.py`**:
- Added `/driver_gains_echo` String publisher (2 Hz): JSON `{"joint_name": {"kp": x, "kd": y}, ...}` for all enabled motors.
- Added `/driver_gains` String subscriber: parse JSON, update `motor.current_kp/kd`. Key `"all"` updates every motor; any other key must match a joint name (unknown names logged as WARN).
- Added `import json`; added `String` to std_msgs imports.

## Module: robstride_can_driver/

### motor_config.py
Dataclasses (`MotorTypeSpec`, `MotorConfig`, `BusConfig`, `DriverConfig`) and a `load_config(yaml_path)` loader. No ROS2 imports — only Python stdlib + PyYAML. The loader reads `config/motors.yaml`, parses hex strings via `int(str(val), 0)`, validates all required fields and section keys, and raises `ValueError` with a descriptive message on any missing/invalid entry. `DriverConfig` provides helper methods `motors_on_bus(bus_name)` and `get_spec(motor_type)`. `MotorConfig` carries both YAML-sourced defaults and mutable runtime state (`current_kp/kd`, `commanded_position/velocity`) initialised in `__post_init__`.

### can_protocol.py
Pure stateless encode/decode for the RobStride Private Protocol (CAN 2.0, 1 Mbps, 29-bit extended frames). No ROS2, no python-can imports — only Python stdlib. All CAN ID builder functions return an int suitable for use as python-can `arbitration_id` with `is_extended_id=True`. Scaling helpers map physical values (rad, rad/s, Nm) to/from 16-bit raw integers [0, 65535]. Motor-type-specific ranges (velocity, torque) are passed as arguments — they live in motor_config.py, not here.

**Key encoding rule:** Type 1 torque feedforward occupies bits 23–8 of the CAN ID (not the data bytes). Type 2/24 feedback has mode status and fault flags in bits 23–16 of the received CAN ID.

### driver_node.py (2026-03-24 changes)
- **Bus open failure**: no longer raises — logs an error and continues. Motors on the missing
  bus are silently skipped in `_send`, `_startup_safe`, and `_startup_home`. Prevents the node
  from crashing and eliminates log spam when can1 is not connected.
- `_send`: uses `self._buses.get(bus_name)` and returns immediately if `None`. Also tracks
  consecutive TX failures per bus (`_bus_fail_counts`); after 5 consecutive failures the bus
  is silently disabled with a one-time error log. Fixes ENOBUFS spam when a bus is up but
  no motors are ACKing (TX queue saturation).

**motor_config.py (2026-03-24):** Motor entries in motors.yaml can include `enabled: false`
to exclude a motor from the driver entirely (no CAN frames sent, not in /joint_states).
Omitting `enabled` or setting it to `true` behaves as before. RS05 is currently disabled.

### driver_node.py
ROS2 Humble node `RobstrideCanDriverNode`. Loads `motors.yaml` at startup via `load_config()`. Opens one `can.Bus` per CAN bus (socketcan, 1 Mbps). Runs one RX thread per bus that decodes Type 2 / Type 24 active report frames into `_motor_states` (lock-protected). A 100 Hz timer sends Type 1 Operation Control frames for all motors using their current commanded position/velocity and current Kp/Kd (node state only — not written to motor). Services for enable, stop, gains, CAN ID change, zeroing (3 methods), param read/write, and flash save. Two startup modes: `startup_safe` (read mechPos → hold) and `startup_home` (go to YAML home position). Emergency stop via `/e_stop` Bool topic.

**Service "all" shorthand:** `/set_gains`, `/set_zero`, `/shift_zero`, `/set_zero_offset` all accept `joint_name: "all"` to operate on every motor simultaneously.
