# safety

## 2026-04-17 — pico_bridge node + kill switch in system_health

- **`safety/pico_bridge.py`**: New ROS2 node. Opens Pico USB-CDC serial port and:
  - Routes robot state → LED commands to Pico (priority: estop→red, turning→orange wave, moving→yellow, stuck→purple blink, default→green)
  - On button press (GP15, `K1\n` from Pico): if robot moving → publishes `/kill_switch true`; if stopped → publishes `"toggle"` to `/lid_command`
  - On button release (`K0\n`): publishes `/kill_switch false`
  - Subscribes: `/estop`, `/robot_mode`, `/cmd_vel`, `/lid_state`, `/robot_stuck`
  - Publishes: `/kill_switch` (Bool), `/lid_command` (String)
- **`safety/system_health.py`**:
  - Uncommented `self.pub.publish(msg)` (estop was silently not publishing)
  - Added `/kill_switch` subscription — physical button press immediately asserts estop (latch only, no auto-clear)
- **`setup.py`**: Added `pico_bridge = safety.pico_bridge:main` entry point
- **`config/safety.yaml`**: Added `pico_bridge` section with `serial_port`, `baud_rate`, and threshold params

## 2026-03-18 — Moved parameters to config/safety.yaml

- **`config/safety.yaml`**: New file. Contains all `heartbeat_publisher` and
  `system_health` parameters in standard ROS2 YAML format.
- **`setup.py`**: Added `config/safety.yaml` to `data_files`.

---

# safety

## 2026-03-17 — Created heartbeat_publisher.py

### What changed
- **`safety/heartbeat_publisher.py`**: Created — the file was missing entirely despite being declared in `setup.py` entry points.
  - Publishes `std_msgs/Empty` at `publish_rate_hz` Hz on `heartbeat_topic`
  - Parameters: `heartbeat_topic` (default `/heartbeat`), `publish_rate_hz` (default `10.0`)
  - `system_health.py` subscribes to this topic and triggers estop if no heartbeat is received within `timeout_s`
