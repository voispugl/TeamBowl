## Directory: config/

### motors.yaml
Primary configuration file loaded at node startup. Edit this file to:
- Set the actual CAN IDs for each motor (replace 0x01–0x06 placeholders)
- Rename joint keys to match your robot's URDF joint names
- Adjust `default_kp` / `default_kd` — these affect the Type 1 command frame only; NOT written to motor flash
- Set `home_position_rad` per joint (used only in `startup_home` mode)
- Choose `startup_mode: "startup_safe"` (hold current pos on enable) or `"startup_home"` (go to home_position_rad)
- `setup_pids_on_startup: false` means the node will NOT write gains to actuator registers; set to true only if you want to configure the motor's internal PID registers

### commands_reference.yaml
Machine-readable table of all CAN command types (0x00–0x19) and all parameter indices (0x7005–0x702B). Used by `tools/commissioning.py` for validation and help text. Do not edit unless the RobStride protocol changes.
