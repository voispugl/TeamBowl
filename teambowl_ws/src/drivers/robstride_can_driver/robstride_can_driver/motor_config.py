"""
Motor configuration dataclasses and YAML loader for robstride_can_driver.
All physical parameters and motor identities are read from motors.yaml at startup.
"""
from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class MotorTypeSpec:
    """Physical limits for one motor model — used only for scaling raw CAN values."""
    torque_min: float
    torque_max: float
    velocity_min: float
    velocity_max: float
    kp_min: float
    kp_max: float
    kd_min: float
    kd_max: float
    position_min: float   # always -4π
    position_max: float   # always +4π


@dataclass
class MotorConfig:
    """Configuration for a single motor joint."""
    joint_name: str
    motor_type: str          # "RS04", "RS00", or "RS05"
    bus: str                 # "can0" or "can1"
    can_id: int
    default_kp: float
    default_kd: float
    default_torque_ff: float
    home_position_rad: float
    # Runtime state (mutable, not from YAML):
    current_kp: float = field(init=False)
    current_kd: float = field(init=False)
    current_torque_ff: float = field(init=False)
    commanded_position: Optional[float] = field(default=None, init=False)
    commanded_velocity: Optional[float] = field(default=None, init=False)

    def __post_init__(self):
        self.current_kp = self.default_kp
        self.current_kd = self.default_kd
        self.current_torque_ff = self.default_torque_ff


@dataclass
class BusConfig:
    name: str       # "can0" / "can1"
    interface: str  # socketcan interface name


@dataclass
class DriverConfig:
    host_can_id: int
    buses: Dict[str, BusConfig]             # key = bus name
    motors: Dict[str, MotorConfig]          # key = joint_name
    motor_specs: Dict[str, MotorTypeSpec]   # key = "RS04" / "RS00" / "RS05"
    loop_rate_hz: float
    active_reporting_interval: int
    setup_pids_on_startup: bool
    startup_mode: str                       # "startup_safe" | "startup_home"

    def motors_on_bus(self, bus_name: str) -> List[MotorConfig]:
        return [m for m in self.motors.values() if m.bus == bus_name]

    def get_spec(self, motor_type: str) -> MotorTypeSpec:
        return self.motor_specs[motor_type]


def load_config(yaml_path: str | Path) -> DriverConfig:
    """Load and validate motors.yaml, returning a fully populated DriverConfig."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise ValueError(f"Configuration file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Configuration file is empty: {yaml_path}")

    # --- host_can_id ---
    if "host_can_id" not in raw:
        raise ValueError("Missing required field: host_can_id")
    host_can_id_raw = raw["host_can_id"]
    if isinstance(host_can_id_raw, str):
        host_can_id = int(host_can_id_raw, 0)
    else:
        host_can_id = int(host_can_id_raw)

    # --- can_buses ---
    if "can_buses" not in raw:
        raise ValueError("Missing required section: can_buses")
    buses: Dict[str, BusConfig] = {}
    for bus_name, bus_data in raw["can_buses"].items():
        if "interface" not in bus_data:
            raise ValueError(f"Bus '{bus_name}' is missing required field: interface")
        buses[bus_name] = BusConfig(name=bus_name, interface=bus_data["interface"])

    # --- motor_specs ---
    if "motor_specs" not in raw:
        raise ValueError("Missing required section: motor_specs")
    motor_specs: Dict[str, MotorTypeSpec] = {}
    for spec_name, spec_data in raw["motor_specs"].items():
        for key in ("torque", "velocity", "kp", "kd", "position"):
            if key not in spec_data:
                raise ValueError(
                    f"motor_specs['{spec_name}'] is missing required field: {key}"
                )
            if len(spec_data[key]) != 2:
                raise ValueError(
                    f"motor_specs['{spec_name}']['{key}'] must be a list of [min, max]"
                )
        motor_specs[spec_name] = MotorTypeSpec(
            torque_min=float(spec_data["torque"][0]),
            torque_max=float(spec_data["torque"][1]),
            velocity_min=float(spec_data["velocity"][0]),
            velocity_max=float(spec_data["velocity"][1]),
            kp_min=float(spec_data["kp"][0]),
            kp_max=float(spec_data["kp"][1]),
            kd_min=float(spec_data["kd"][0]),
            kd_max=float(spec_data["kd"][1]),
            position_min=float(spec_data["position"][0]),
            position_max=float(spec_data["position"][1]),
        )

    # --- motors ---
    if "motors" not in raw:
        raise ValueError("Missing required section: motors")
    motors: Dict[str, MotorConfig] = {}
    required_motor_fields = (
        "type", "bus", "can_id", "default_kp", "default_kd",
        "default_torque_ff", "home_position_rad",
    )
    for joint_name, motor_data in raw["motors"].items():
        for field_name in required_motor_fields:
            if field_name not in motor_data:
                raise ValueError(
                    f"motors['{joint_name}'] is missing required field: {field_name}"
                )
        motor_type = motor_data["type"]
        if motor_type not in motor_specs:
            raise ValueError(
                f"motors['{joint_name}'] references unknown motor_type '{motor_type}'. "
                f"Known types: {list(motor_specs.keys())}"
            )
        motor_bus = motor_data["bus"]
        if motor_bus not in buses:
            raise ValueError(
                f"motors['{joint_name}'] references unknown bus '{motor_bus}'. "
                f"Known buses: {list(buses.keys())}"
            )
        can_id_raw = motor_data["can_id"]
        if isinstance(can_id_raw, str):
            can_id = int(can_id_raw, 0)
        else:
            can_id = int(can_id_raw)

        motors[joint_name] = MotorConfig(
            joint_name=joint_name,
            motor_type=motor_type,
            bus=motor_bus,
            can_id=can_id,
            default_kp=float(motor_data["default_kp"]),
            default_kd=float(motor_data["default_kd"]),
            default_torque_ff=float(motor_data["default_torque_ff"]),
            home_position_rad=float(motor_data["home_position_rad"]),
        )

    # --- control ---
    if "control" not in raw:
        raise ValueError("Missing required section: control")
    ctrl = raw["control"]
    for field_name in ("loop_rate_hz", "active_reporting_interval",
                       "setup_pids_on_startup", "startup_mode"):
        if field_name not in ctrl:
            raise ValueError(
                f"control section is missing required field: {field_name}"
            )
    startup_mode = ctrl["startup_mode"]
    valid_startup_modes = ("startup_safe", "startup_home")
    if startup_mode not in valid_startup_modes:
        raise ValueError(
            f"control.startup_mode must be one of {valid_startup_modes}, "
            f"got '{startup_mode}'"
        )

    return DriverConfig(
        host_can_id=host_can_id,
        buses=buses,
        motors=motors,
        motor_specs=motor_specs,
        loop_rate_hz=float(ctrl["loop_rate_hz"]),
        active_reporting_interval=int(ctrl["active_reporting_interval"]),
        setup_pids_on_startup=bool(ctrl["setup_pids_on_startup"]),
        startup_mode=startup_mode,
    )
