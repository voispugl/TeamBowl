## Directory: tools/

### commissioning.py
Standalone CLI for one-time motor commissioning. No ROS2 required — only `python-can` and `pyyaml`.
Run this before the ROS2 driver to set motor CAN IDs, mechanical zeros, and zero offsets.

**Typical commissioning workflow:**
1. Connect motor, bring up CAN interface: `sudo ip link set can0 up type can bitrate 1000000`
2. Scan the bus: `python commissioning.py --bus can0 scan`
3. Set CAN IDs to match motors.yaml: `python commissioning.py --bus can0 set-id 0x01 0x03`
4. Enable motor and set zero: `python commissioning.py --bus can0 enable 0x03` then `set-zero 0x03`
5. Save: `python commissioning.py --bus can0 save 0x03`

All CAN frames use Extended 29-bit IDs (is_extended_id=True) at 1 Mbps.
The `shift-zero` command reads the current add_offset, adds the delta, then writes back.

### diagnose.py
Standalone pre-flight diagnostic tool. No ROS2 required — only `python-can`, `pyyaml`, and stdlib.
Run this before launching the ROS2 driver to confirm all motors are reachable and healthy.
Exits with code 0 on success, code 1 if any check FAILs.

**Checks performed (11 total):**
1. CAN bus accessible
2. Motor scan — count of responding motors
3. Duplicate motor IDs on the same bus (FAIL)
4. Duplicate UIDs at different motor IDs (FAIL)
5. Expected motors present per motors.yaml (FAIL; requires --config)
6. Unexpected motors not in config (WARN; requires --config)
7. Hardware fault bits on enabled motor (FAIL)
8. Encoder uncalibrated flag (FAIL)
9. Motor temperature exceeds --temp-warn threshold (WARN)
10. Same CAN ID on multiple buses (WARN; requires multiple --bus args)
11. CAN error frames observed during scan (WARN)

**Typical usage:**
```bash
python diagnose.py --bus can0 --bus can1 --config config/motors.yaml
python diagnose.py --bus can0 --no-enable   # skip enable/feedback if motors already running
```

**Implementation notes:**
- All CAN protocol logic (frame builders, feedback decoder) is inlined; no imports from the package.
- The scan loop sends Type 0 to each ID and waits 50 ms per motor.
- The duplicate-ID detection does a second pass over the bus to count responses per ID.
- The health check sends Type 3 (enable), waits up to 200 ms for Type 2 feedback, then immediately sends Type 4 (stop).
- Cross-bus results are appended to every bus's result list so they are counted in the summary.

### monitor.py
Standalone live CAN sensor monitor. No ROS2 required — only `python-can`, `pyyaml`, and stdlib.
Passively listens on one or more CAN buses and displays an auto-refreshing table of all RobStride
motor feedback decoded from Type 2 and Type 24 active-report frames.

**CLI flags:**
| Flag | Default | Description |
|---|---|---|
| `--bus IFACE` | `can0` | CAN interface to listen on (repeatable for multiple buses) |
| `--config PATH` | — | motors.yaml for joint names and per-motor type/range info |
| `--bitrate BPS` | `1000000` | CAN bus bitrate |
| `--enable-reporting` | off | Send Type 24 active-report enable to IDs 0x01–0x20 before loop |
| `--interval SEC` | `0.1` | Display refresh interval in seconds |

**Typical usage:**
```bash
# Basic single-bus monitor
python monitor.py

# Two buses with joint names from config
python monitor.py --bus can0 --bus can1 --config config/motors.yaml

# Trigger active reporting first (motors that don't broadcast by default)
python monitor.py --bus can0 --enable-reporting --interval 0.05
```

**Display:**
- Uses curses for an in-place refreshing table when stdout is a TTY; falls back to plain text otherwise.
- Columns: Bus, CAN ID, Joint, Pos (rad), Vel (r/s), Torque (Nm), Temp (°C), Mode, Faults.
- Rows with no frame received for >2 s show `---` (STALE_TIMEOUT).
- Fault rows are highlighted red in curses mode; temperature ≥ 100 °C is highlighted yellow.
- Press `q` or Ctrl+C to quit.

**Implementation notes:**
- All CAN protocol logic (decode_feedback, raw_to_value) is inlined; no imports from the package.
- One daemon thread per bus calls `bus.recv(timeout=0.2)` in a loop and updates shared state under a lock.
- Motor type (RS04/RS00/RS05) determines velocity and torque decode ranges; defaults to RS04.
- The `--enable-reporting` path sends `(0x18 << 24) | (HOST_ID << 8) | motor_id` with payload `\x01\x02\x03\x04\x05\x06\x01\x00` to IDs 1–32.
