# RobStride CAN Motor Driver — Command Reference

## Overview

This driver controls **9 RobStride actuators** over two CAN buses on a **Jetson AGX Orin**:

| Bus   | Motors                          | Count |
|-------|---------------------------------|-------|
| can0  | RS04 (joints 1–6)               | 6     |
| can1  | RS00 (joints 1–2) + RS05 (joint 1) | 3  |

**Protocol:** RobStride Private Protocol
**Frame format:** 29-bit extended CAN frames (CAN 2.0B)
**Bus speed:** 1 Mbps
**Host CAN ID:** `0xFD` (configurable in `config/motors.yaml`)

Two tools are available:

- **ROS2 driver node** — real-time motion control at 100 Hz, publishes `/joint_states`, subscribes to `/joint_commands`
- **Standalone commissioning CLI** (`tools/commissioning.py`) — no ROS2 required; used for one-time setup, CAN ID assignment, and calibration

---

## CAN Protocol Quick Reference

All Private Protocol messages use a 29-bit extended CAN arbitration ID split into three fields:

| Field | Bits | Description |
|-------|------|-------------|
| comm_type | 28–24 | Message type identifier |
| Data Area 2 | 23–8 | Host CAN ID (bits 15–8) and/or secondary data (e.g., torque ff or new ID) |
| Destination Address | 7–0 | Target motor CAN ID |
| Data Area 1 | Byte 0–7 | 8-byte payload |

### Communication Types

| Type ID | Name | Direction | Purpose | Data Format Summary |
|---------|------|-----------|---------|---------------------|
| `0x00` | `get_device_id` | host → motor | Query 64-bit MCU UID; discover motors on bus | Data: 8 zero bytes; reply contains 64-bit UID in Byte 0–7 |
| `0x01` | `operation_ctrl` | host → motor | Main MIT-style motion command (pos + vel + Kp + Kd + torque_ff) | ID bits 23–8 = torque_ff raw; Bytes 0–1 angle, 2–3 vel, 4–5 Kp, 6–7 Kd (big-endian uint16) |
| `0x02` | `motor_feedback` | motor → host | Live position/velocity/torque/temperature; sent in reply to most commands | Bytes 0–1 angle, 2–3 vel, 4–5 torque, 6–7 temp; fault flags in ID bits 21–16 |
| `0x03` | `motor_enable` | host → motor | Enable motor torque output | Data: 8 zero bytes |
| `0x04` | `motor_stop` | host → motor | Stop motor; optionally clear fault | Byte 0: `0x00` = stop, `0x01` = stop + clear fault |
| `0x06` | `set_mech_zero` | host → motor | Set current encoder position as mechanical zero | Byte 0: `0x01`; not available in PP mode |
| `0x07` | `set_can_id` | host → motor | Change motor CAN ID immediately (no reboot) | ID bits 23–16 = new CAN ID; reply is Type 0 broadcast |
| `0x11` | `read_param` | host → motor | Read one parameter by index from motor's parameter table | Bytes 0–1: index (LE); Bytes 4–7 of reply: value (LE float or int) |
| `0x12` | `write_param` | host → motor | Write one parameter (volatile — lost on power-off unless saved) | Bytes 0–1: index (LE); Bytes 4–7: value (LE float or int) |
| `0x15` | `fault_feedback` | motor → host | Active fault and warning flags | Bytes 0–3: fault bitmask; Bytes 4–7: warning bitmask |
| `0x16` | `save_params` | host → motor | Persist all Type 18 writes to non-volatile flash | Data: `01 02 03 04 05 06 07 08` (fixed magic) |
| `0x17` | `set_baud` | host → motor | Change CAN baud rate (takes effect after power cycle) | Bytes 0–5: `01 02 03 04 05 06`; Byte 6: baud code |
| `0x18` | `active_report` / `active_report_response` | host → motor / motor → host | Enable/disable periodic unsolicited Type 2 feedback | Bytes 0–5: magic; Byte 6: `0x00`=off, `0x01`=on; response same layout as Type 2 |
| `0x19` | `set_protocol` | host → motor | Switch protocol (Private / CANopen / MIT); takes effect after power cycle | Bytes 0–5: magic; Byte 6: `0x00`=private, `0x01`=CANopen, `0x02`=MIT |

---

## Type 1 Operation Control Frame Detail

Type 1 is the primary real-time motion command. The motor must be enabled (Type 3) before Type 1 frames have effect. The driver sends these at 100 Hz from the `/joint_commands` topic.

### 29-bit CAN ID Layout

```
Bit:  28  27  26  25  24 | 23  22  21  20  19  18  17  16 | 15  14  13  12  11  10   9   8 |  7   6   5   4   3   2   1   0
      [  comm_type=0x01  ] [  torque_ff_raw[15:8] (high)  ] [   torque_ff_raw[7:0]  (low)  ] [        motor_id              ]
```

| Bits  | Field | Value |
|-------|-------|-------|
| 28–24 | `comm_type` | `0x01` (fixed) |
| 23–8 | `torque_feedforward_raw` | 16-bit unsigned integer, scaled from Nm per motor type table below |
| 7–0 | `motor_id` | Target motor CAN ID |

### Data Bytes (all big-endian uint16, range 0–65535)

| Bytes | Raw Field | Physical Quantity | Range |
|-------|-----------|-------------------|-------|
| 0–1 | `angle_raw` | Position (rad) | −4π ~ +4π (all motor types) |
| 2–3 | `velocity_raw` | Velocity (rad/s) | Motor-type-specific (see table below) |
| 4–5 | `kp_raw` | Kp gain | Motor-type-specific |
| 6–7 | `kd_raw` | Kd gain | Motor-type-specific |

### Motor-Specific Ranges for Type 1

| Parameter | RS00 | RS04 | RS05 |
|-----------|------|------|------|
| Torque feedforward (ID bits 23–8) | −14 ~ +14 Nm | −120 ~ +120 Nm | −5.5 ~ +5.5 Nm |
| Position (Bytes 0–1) | −4π ~ +4π rad | −4π ~ +4π rad | −4π ~ +4π rad |
| Velocity (Bytes 2–3) | −33 ~ +33 rad/s | −15 ~ +15 rad/s | −50 ~ +50 rad/s |
| Kp (Bytes 4–5) | 0 ~ 500 | 0 ~ 5000 | 0 ~ 500 |
| Kd (Bytes 6–7) | 0 ~ 5 | 0 ~ 100 | 0 ~ 5 |

> **Important:** Kp and Kd in the Type 1 frame are the **node's software gains** set via `config/motors.yaml` (`default_kp`, `default_kd`) or the `/set_gains` service. They are packed into each outgoing motion frame but **do not modify the motor's stored internal PID registers** (`loc_kp`, `spd_kp`, etc.). The motor's stored registers are only changed by an explicit Type 18 (`write_param`) command targeting those indices.

---

## Startup Modes

Configured via `startup_mode` in `config/motors.yaml`. Applied to all motors at node startup after enable.

### `startup_safe` (default)

```
1. Send Type 3 (motor_enable) to all motors
2. Send Type 17 (read_param) for mechPos (index 0x7019) to each motor
3. Immediately command that read-back position as the Type 1 target
```

The motor holds its current position without snapping to a commanded home. Use this mode whenever the robot's physical pose at startup is unknown or varies.

### `startup_home`

```
1. Send Type 3 (motor_enable) to all motors
2. Send Type 1 with target = home_position_rad defined per joint in motors.yaml
```

Commands each joint to a known home pose on startup. Use only when the robot is physically placed in the home configuration before launch, and `home_position_rad` values are correctly calibrated.

---

## Zero-Setting Methods

All three methods write to volatile RAM. Changes are **lost on power-off** unless followed by `save` (Type 22 / `save` CLI command / `/save_motor_params` service).

### Method 1: `set-zero` (Type 6)

Hardware command. Sets the motor's current encoder position as the new mechanical zero origin. The position reading (`mechPos`) will read `0.0` rad from this point forward.

**CLI:**
```bash
python commissioning.py --bus can0 set-zero 0x03
```

**ROS2 service:**
```bash
ros2 service call /set_zero robstride_can_driver/srv/SetZero "{motor_id: 3}"
```

### Method 2: `shift-zero` (Type 18, index `0x702B` += delta)

Iterative fine-tuning. Reads the current `add_offset` value from the motor, adds `delta_rad` to it, and writes the result back. Safe for small repeated adjustments during calibration.

**CLI:**
```bash
# Add 0.05 rad to the current offset
python commissioning.py --bus can0 shift-zero 0x03 0.05

# Subtract 0.1 rad
python commissioning.py --bus can0 shift-zero 0x03 -- -0.1
```

**ROS2 service:**
```bash
ros2 service call /shift_zero robstride_can_driver/srv/ShiftZero "{motor_id: 3, delta_rad: 0.05}"
```

### Method 3: `set-offset` (Type 18, index `0x702B` = absolute value)

Writes an exact absolute offset directly to `add_offset`. Use this in automated calibration pipelines where the desired offset is computed externally.

**CLI:**
```bash
# Set add_offset to exactly π/2 rad
python commissioning.py --bus can0 set-offset 0x03 1.5708
```

**ROS2 service:**
```bash
ros2 service call /set_zero_offset robstride_can_driver/srv/SetZeroOffset "{motor_id: 3, offset_rad: 1.5708}"
```

> After any zero-setting operation, call `save` to persist the change to flash.

---

## Parameter Index Table

Used with Type 17 (`read_param`) and Type 18 (`write_param`). All indices are 16-bit and sent **little-endian** in Bytes 0–1 of the frame. Parameter values in Bytes 4–7 are also **little-endian** (IEEE-754 float or integer depending on type).

| Index | Name | Type | R/W | RS00 Range / Default | RS04 Range / Default | RS05 Range / Default | Description |
|-------|------|------|-----|----------------------|----------------------|----------------------|-------------|
| `0x7005` | `run_mode` | uint8 | W/R | 0–5 | 0–5 | 0–5 | Operating mode: `0`=Operation ctrl (Type 1), `1`=PP position, `2`=Velocity, `3`=Current, `5`=CSP position |
| `0x7006` | `iq_ref` | float | W/R | −16 ~ +16 A | −90 ~ +90 A | −11 ~ +11 A | Current mode Iq command (used in run_mode=3) |
| `0x700A` | `spd_ref` | float | W/R | −33 ~ +33 rad/s | −20 ~ +20 rad/s | −50 ~ +50 rad/s | Velocity mode speed command (used in run_mode=2) |
| `0x700B` | `limit_torque` | float | W/R | 0 ~ 14 Nm | 0 ~ 120 Nm | 0 ~ 5.5 Nm | Output torque limit (all modes) |
| `0x7010` | `cur_kp` | float | W/R | default 0.17 | default 0.17 | default 0.17 | Current loop proportional gain |
| `0x7011` | `cur_ki` | float | W/R | default 0.012 | default 0.012 | default 0.012 | Current loop integral gain |
| `0x7014` | `cur_filt_gain` | float | W/R | 0 ~ 1.0, def 0.1 | 0 ~ 1.0, def 0.1 | 0 ~ 1.0, def 0.1 | Current loop low-pass filter coefficient |
| `0x7016` | `loc_ref` | float | W/R | rad | rad | rad | Position mode angle command (run_mode=1 or 5) |
| `0x7017` | `limit_spd` | float | W/R | 0 ~ 33 rad/s | 0 ~ 20 rad/s | 0 ~ 50 rad/s | Speed limit in CSP position mode (run_mode=5) |
| `0x7018` | `limit_cur` | float | W/R | 0 ~ 16 A | 0 ~ 90 A | 0 ~ 11 A | Current limit in velocity and position modes |
| `0x7019` | `mechPos` | float | R | rad | rad | rad | Mechanical angle of the load shaft (read-only) |
| `0x701A` | `iqf` | float | R | −16 ~ +16 A | −90 ~ +90 A | −11 ~ +11 A | Filtered Iq feedback (read-only) |
| `0x701B` | `mechVel` | float | R | −33 ~ +33 rad/s | −15 ~ +15 rad/s | −50 ~ +50 rad/s | Load shaft velocity feedback (read-only) |
| `0x701C` | `VBUS` | float | R | V | V | V | Bus voltage (read-only) |
| `0x701E` | `loc_kp` | float | W/R | default 40 | default 60 | default 40 | Position loop proportional gain (stored in motor flash) |
| `0x701F` | `spd_kp` | float | W/R | default 6 | default 6 | default 6 | Velocity loop proportional gain |
| `0x7020` | `spd_ki` | float | W/R | default 0.02 | default 0.02 | default 0.02 | Velocity loop integral gain |
| `0x7021` | `spd_filt_gain` | float | W/R | default 0.1 | default 0.1 | W only, def 0.1 | Velocity loop filter coefficient |
| `0x7022` | `acc_rad` | float | W/R | 20 rad/s² | 15 rad/s² | W only, 20 rad/s² | Velocity mode acceleration |
| `0x7024` | `vel_max` | float | W/R | default 10 rad/s | default 10 rad/s | W only, 10 rad/s | PP position mode maximum speed |
| `0x7025` | `acc_set` | float | W/R | default 10 rad/s² | default 10 rad/s² | W only, 10 rad/s² | PP position mode acceleration |
| `0x7026` | `EPScan_time` | uint16 | W/R | default 1 | default 1 | W only | Active report interval: 1=10 ms; each additional count adds 5 ms |
| `0x7028` | `canTimeout` | uint32 | W/R | default 0 (off) | default 0 (off) | W only | CAN timeout threshold: 20000 = 1 s; 0 = disabled |
| `0x7029` | `zero_sta` | uint8 | W | `0`=0~2π range | `0`=0~2π range | `0`=0~2π range | Zero range flag: `0`=0~2π, `1`=−π~+π |
| `0x702A` | `damper` | uint8 | W/R | `0`=enabled (def) | `0`=enabled (def) | not supported | Post-power-off anti-backdrive damping: `0`=on, `1`=off (RS00/RS04 only) |
| `0x702B` | `add_offset` | float | W/R | 0.0 rad | 0.0 rad | 0.0 rad | Position zero offset in radians — shifts the zero point by this value |

---

## Fault Bit Reference

### Fault Bits (Type 21 frame Bytes 0–3, and reflected in Type 2 CAN ID bits 21–16)

| Bit | Name | Description |
|-----|------|-------------|
| 16 | `overcurrent_a` | A-phase current sampling overcurrent |
| 14 | `stall_overload` | Motor stall / I²t overload protection triggered |
| 9 | `pos_init` | Position initialization fault |
| 8 | `hw_id` | Hardware identification fault |
| 7 | `encoder_uncal` | Encoder uncalibrated |
| 5 | `overcurrent_c` | C-phase current sampling overcurrent |
| 4 | `overcurrent_b` | B-phase current sampling overcurrent |
| 3 | `overvoltage` | Bus overvoltage fault |
| 2 | `undervoltage` | Bus undervoltage fault |
| 1 | `driver_chip` | Driver chip fault |
| 0 | `overtemp` | Motor overtemperature (RS00/RS05 threshold: 135°C; RS04: 145°C) |

### Warning Bits (Type 21 frame Bytes 4–7)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | `overtemp_warning` | Motor temperature approaching overtemp threshold (default 135°C) |

> The `stop-clear` CLI command (Type 4, Byte 0 = `0x01`) clears active faults. The `/stop_motors` ROS2 service sends a normal stop (Byte 0 = `0x00`); call `/stop_motors` then `stop-clear` manually if faults need clearing.

---

## ROS2 Topics

| Topic | Message Type | Direction | Description |
|-------|-------------|-----------|-------------|
| `/joint_commands` | `sensor_msgs/JointState` | Subscribe | Send motion commands: `position` (rad), `velocity` (rad/s), `effort` (torque feedforward, Nm). `name` field must match joint names in `motors.yaml`. |
| `/joint_states` | `sensor_msgs/JointState` | Publish | Live motor feedback at 100 Hz: position (rad), velocity (rad/s), effort (torque Nm). One entry per motor. |
| `/e_stop` | `std_msgs/Bool` | Subscribe | Emergency stop. Publishing `True` sends Type 4 stop to **all** motors immediately. |
| `/motor_faults` | `diagnostic_msgs/DiagnosticArray` | Publish | Fault flags per motor. One `DiagnosticStatus` per joint; `level` is `OK` (0), `WARN` (1), or `ERROR` (2) based on fault/warning bits. |

---

## ROS2 Services

| Service | Type | CAN Command | Description |
|---------|------|-------------|-------------|
| `/enable_motors` | `std_srvs/Trigger` | Type 3 (all motors) | Enable torque output on all configured motors. Applies startup_mode logic (see Startup Modes). |
| `/stop_motors` | `std_srvs/Trigger` | Type 4 (all motors) | Stop all motors without clearing faults. Safe stop for normal shutdown. |
| `/set_gains` | `robstride_can_driver/srv/SetGains` | None (no CAN write) | Update Kp/Kd in the node's in-memory state for a specific joint. Values take effect in the next Type 1 frame. Does NOT write to motor flash. |
| `/set_motor_id` | `robstride_can_driver/srv/SetMotorId` | Type 7 | Change motor CAN ID. New ID takes effect immediately; update `motors.yaml` and restart the node to persist. |
| `/set_zero` | `robstride_can_driver/srv/SetZero` | Type 6 | Set current encoder position as mechanical zero for a motor. Volatile — follow with `/save_motor_params`. |
| `/shift_zero` | `robstride_can_driver/srv/ShiftZero` | Type 18 (`0x702B` += delta) | Add delta_rad to the current `add_offset`. Reads current value first, then writes new value. Volatile. |
| `/set_zero_offset` | `robstride_can_driver/srv/SetZeroOffset` | Type 18 (`0x702B` = value) | Write an absolute offset in radians to `add_offset`. Volatile. |
| `/read_motor_param` | `robstride_can_driver/srv/ReadMotorParam` | Type 17 | Read one parameter by index from a motor. Returns both float and uint32 interpretations of the raw bytes. |
| `/write_motor_param` | `robstride_can_driver/srv/WriteMotorParam` | Type 18 (volatile) | Write one parameter by index to a motor. Change is volatile until saved. |
| `/save_motor_params` | `std_srvs/Trigger` | Type 22 (all motors) | Save all parameters to non-volatile flash on all motors. |

---

## Commissioning CLI Reference

`tools/commissioning.py` — standalone, no ROS2 required. Requires `python-can` and `pyyaml`.

**Global flags (apply to all commands):**

| Flag | Default | Description |
|------|---------|-------------|
| `--bus <name>` | `can0` | SocketCAN interface name |
| `--host-id <id>` | `0xFD` | Host node ID embedded in outgoing frames |
| `--bitrate <bps>` | `1000000` | CAN bus bitrate |
| `--dry-run` | off | Print the CAN frame(s) that would be sent without opening the bus |

**Bring up the CAN interface before any command:**
```bash
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
```

---

### `scan`

Scan bus for all motors by sending Type 0 (`get_device_id`) to IDs `0x01`–`0x20`. Prints a table of responding motor IDs and their MCU UIDs.

```bash
python commissioning.py --bus can0 scan
```

---

### `get-id`

Read the 64-bit MCU unique identifier from a specific motor (Type 0).

```bash
python commissioning.py --bus can0 get-id 0x03
# Output: MCU Unique ID (64-bit): 0x0123456789ABCDEF
```

---

### `enable`

Enable motor torque output (Type 3).

```bash
python commissioning.py --bus can0 enable 0x03
```

---

### `stop`

Stop the motor and hold position (Type 4, Byte 0 = `0x00`). Does not clear faults.

```bash
python commissioning.py --bus can0 stop 0x03
```

---

### `stop-clear`

Stop the motor and clear any active fault (Type 4, Byte 0 = `0x01`).

```bash
python commissioning.py --bus can0 stop-clear 0x03
```

---

### `set-id`

Change the motor's CAN ID immediately (Type 7). New ID takes effect without a reboot.

```bash
# Change motor at ID 0x01 to ID 0x03
python commissioning.py --bus can0 set-id 0x01 0x03
```

---

### `set-zero`

Set the current encoder position as the new mechanical zero (Type 6). Not available in PP position mode.

```bash
python commissioning.py --bus can0 set-zero 0x03
```

---

### `shift-zero`

Read the current `add_offset` (index `0x702B`), add `delta_rad`, and write the result back (two Type 18 operations). For iterative fine-tuning.

```bash
# Add 0.05 rad to offset
python commissioning.py --bus can0 shift-zero 0x03 0.05

# Subtract 0.1 rad
python commissioning.py --bus can0 shift-zero 0x03 -- -0.1
```

---

### `set-offset`

Write an exact absolute value to `add_offset` (index `0x702B`) via Type 18. For use in calibration pipelines.

```bash
# Set add_offset to exactly π/2
python commissioning.py --bus can0 set-offset 0x03 1.5708
```

---

### `read`

Read one parameter by index using Type 17. Prints the value as both float and uint32.

```bash
# Read mechPos (current position)
python commissioning.py --bus can0 read 0x03 0x7019

# Read bus voltage
python commissioning.py --bus can0 read 0x03 0x701C
```

---

### `write`

Write a float parameter by index using Type 18 (volatile).

```bash
# Set add_offset to 1.5708 rad
python commissioning.py --bus can0 write 0x03 0x702B 1.5708

# Set torque limit to 80 Nm (RS04)
python commissioning.py --bus can0 write 0x03 0x700B 80.0
```

---

### `write-int`

Write an integer parameter by index using Type 18 (volatile). Use this for uint8/uint16/uint32 parameters like `run_mode`.

```bash
# Set run_mode to velocity mode (2)
python commissioning.py --bus can0 write-int 0x03 0x7005 2

# Disable damper (RS04/RS00 only)
python commissioning.py --bus can0 write-int 0x03 0x702A 1
```

---

### `save`

Persist all parameters modified via Type 18 to non-volatile flash (Type 22).

```bash
python commissioning.py --bus can0 save 0x03
```

---

### `set-baud`

Change the motor's CAN baud rate (Type 23). Takes effect after power cycle.

```bash
# Options: 1M, 500K, 250K, 125K
python commissioning.py --bus can0 set-baud 0x03 1M
```

---

### `set-protocol`

Switch the motor's communication protocol (Type 25). Takes effect after power cycle. The driver requires `private` protocol.

```bash
# Options: private, mit, canopen
python commissioning.py --bus can0 set-protocol 0x03 private
```

---

### `active-report`

Enable or disable periodic unsolicited Type 2 feedback from the motor (Type 24). The default interval is 10 ms (configurable via `EPScan_time`, index `0x7026`).

```bash
python commissioning.py --bus can0 active-report 0x03 on
python commissioning.py --bus can0 active-report 0x03 off
```

---

### `fault-read`

Read and display the raw fault/status frame for a motor.

```bash
python commissioning.py --bus can0 fault-read 0x03
```

---

### `version`

Read the firmware version from the motor (Type 4 with Byte 1 = `0xC4`).

```bash
python commissioning.py --bus can0 version 0x03
```

---

### `--dry-run`

Print the CAN frame(s) that would be sent without opening the bus. Works with any command. Useful for verifying frame construction without connected hardware.

```bash
python commissioning.py --bus can0 --dry-run enable 0x03
# Output: [TX] CAN ID: 0x030000FD03  DLC: 8  Data: 00 00 00 00 00 00 00 00
```

---

## Typical Workflows

### 1. Initial Motor Commissioning (brand new motor)

A new motor ships with CAN ID `0x01`. This workflow assigns a unique ID, sets the zero position, and saves.

```
1. Connect the motor to the CAN bus. Only one new motor at a time.
2. Bring up the interface:
      sudo ip link set can0 up type can bitrate 1000000

3. Scan to confirm the motor is detected:
      python commissioning.py --bus can0 scan

4. Assign a new CAN ID (e.g., to 0x04):
      python commissioning.py --bus can0 set-id 0x01 0x04

5. Enable the motor and physically move it to the desired zero position:
      python commissioning.py --bus can0 enable 0x04

6. Set current position as mechanical zero:
      python commissioning.py --bus can0 set-zero 0x04

7. Save to flash:
      python commissioning.py --bus can0 save 0x04

8. Stop the motor:
      python commissioning.py --bus can0 stop 0x04

9. Update config/motors.yaml with the new can_id and joint name.
```

---

### 2. Safe Robot Startup

```
1. Bring up both CAN interfaces:
      sudo ip link set can0 up type can bitrate 1000000
      sudo ip link set can1 up type can bitrate 1000000

2. Ensure startup_mode in config/motors.yaml is set to "startup_safe".

3. Launch the ROS2 driver node:
      ros2 launch robstride_can_driver driver.launch.py

4. Enable motors via service:
      ros2 service call /enable_motors std_srvs/srv/Trigger {}

5. Verify feedback is flowing:
      ros2 topic echo /joint_states --once

6. Confirm positions are reasonable (should match the robot's physical pose).
   If any joint shows an unexpected value, stop immediately and check wiring/IDs:
      ros2 service call /stop_motors std_srvs/srv/Trigger {}
```

---

### 3. Setting Motor Zeros on a Running Robot (via ROS2)

```
1. Enable motors and command all joints to the desired zero pose
   by publishing to /joint_commands.

2. Once the robot is holding the correct zero pose, set zeros one joint at a time:
      ros2 service call /set_zero robstride_can_driver/srv/SetZero "{motor_id: 1}"
      ros2 service call /set_zero robstride_can_driver/srv/SetZero "{motor_id: 2}"
      # ... repeat for each motor

3. Fine-tune if needed with shift_zero:
      ros2 service call /shift_zero robstride_can_driver/srv/ShiftZero "{motor_id: 1, delta_rad: 0.01}"

4. Save all to flash:
      ros2 service call /save_motor_params std_srvs/srv/Trigger {}

5. Verify by stopping and re-enabling, then checking /joint_states reads near 0.0 rad.
```

---

### 4. Tuning Gains

Node Kp/Kd (affect Type 1 frames only — no CAN write):

```
1. Update gains in node memory via service:
      ros2 service call /set_gains robstride_can_driver/srv/SetGains \
        "{joint_name: 'joint_rs04_1', kp: 120.0, kd: 3.0}"

2. Test motion by publishing to /joint_commands and observing /joint_states.

3. If gains are satisfactory, update config/motors.yaml (default_kp, default_kd)
   so they persist across node restarts.
```

Motor internal PID registers (stored in motor flash via Type 18):

```
1. Write a new value to the motor's internal loc_kp register:
      ros2 service call /write_motor_param robstride_can_driver/srv/WriteMotorParam \
        "{motor_id: 1, index: 0x701E, value: 55.0}"

2. Test — if acceptable, save to flash:
      ros2 service call /save_motor_params std_srvs/srv/Trigger {}

3. If not acceptable, write the old value back before saving.
```

---

### 5. Saving Parameters to Flash

Type 18 (`write_param`) writes are volatile and lost on power-off. Always save after any parameter change you want to persist.

**Save one motor via CLI:**
```bash
python commissioning.py --bus can0 save 0x03
```

**Save all motors via ROS2:**
```bash
ros2 service call /save_motor_params std_srvs/srv/Trigger {}
```

> The save command sends the fixed magic payload `01 02 03 04 05 06 07 08` (Type 22). The motor replies with a Type 2 feedback frame to confirm. Allow ~100 ms after saving before power-cycling.

---

## Motor Specification Summary

| Parameter | RS00 | RS04 | RS05 |
|-----------|------|------|------|
| Peak torque | 14 Nm | 120 Nm | 5.5 Nm |
| Max speed | 33 rad/s | 15 rad/s | 50 rad/s |
| Max phase current | 16 A | 90 A | 11 A |
| Overtemperature fault threshold | 135°C | 145°C | 135°C |
| CAN baud rate | 1 Mbps | 1 Mbps | 1 Mbps |
| CAN frame type (private protocol) | 29-bit extended | 29-bit extended | 29-bit extended |
| `damper` parameter (`0x702A`) support | Yes | Yes | No |
| Bus on this robot | can1 | can0 | can1 |
| Count on this robot | 2 | 6 | 1 |
