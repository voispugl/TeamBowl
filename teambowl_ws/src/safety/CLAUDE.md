# safety

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
