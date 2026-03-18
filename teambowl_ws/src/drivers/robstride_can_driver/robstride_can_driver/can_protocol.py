    """
can_protocol.py

Pure stateless encode/decode module for the RobStride Private CAN Protocol
(CAN 2.0, 1 Mbps, 29-bit extended frames).

No ROS2 or python-can dependencies — only Python stdlib.

CAN ID layout (29-bit extended):
    Bits 28–24 (5 bits) : Communication Type
    Bits 23–8  (16 bits): Data Area 2 (usage varies per command type)
    Bits 7–0   (8 bits) : Destination Address (motor CAN ID)
"""

import math
import struct
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Dataclasses for decoded feedback
# ---------------------------------------------------------------------------

@dataclass
class MotorFeedback:
    motor_id: int
    host_id: int
    position: float       # rad
    velocity: float       # rad/s
    torque: float         # Nm
    temperature: float    # °C
    mode: int             # 0=Reset, 1=Cali, 2=Run
    fault_uncalibrated: bool
    fault_overload: bool
    fault_encoder: bool
    fault_overtemp: bool
    fault_overcurrent: bool
    fault_undervoltage: bool


@dataclass
class ParamReply:
    motor_id: int
    param_index: int
    value_bytes: bytes    # raw 4 bytes — caller interprets as float or int
    success: bool         # True if bits 23–16 of CAN ID == 0x00


@dataclass
class FaultFrame:
    motor_id: int
    fault_bits: int       # raw 32-bit fault word
    warning_bits: int     # raw 32-bit warning word
    # Named fault properties derived from fault_bits
    fault_overcurrent_a: bool   # bit 16
    fault_stall: bool           # bit 14
    fault_pos_init: bool        # bit 9
    fault_hw_id: bool           # bit 8
    fault_encoder_uncal: bool   # bit 7
    fault_overcurrent_c: bool   # bit 5
    fault_overcurrent_b: bool   # bit 4
    fault_overvoltage: bool     # bit 3
    fault_undervoltage: bool    # bit 2
    fault_driver: bool          # bit 1
    fault_overtemp: bool        # bit 0
    warning_overtemp: bool      # warning bit 0


# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------

def scale_to_raw(value: float, min_val: float, max_val: float, bits: int = 16) -> int:
    """
    Clamp value to [min_val, max_val], then map to [0, 2^bits - 1].

    raw = int((clamp(value, min_val, max_val) - min_val)
              / (max_val - min_val) * (2**bits - 1))
    """
    clamped = max(min_val, min(max_val, value))
    return int((clamped - min_val) / (max_val - min_val) * (2 ** bits - 1))


def raw_to_value(raw: int, min_val: float, max_val: float, bits: int = 16) -> float:
    """
    Map [0, 2^bits - 1] back to [min_val, max_val].
    """
    return min_val + raw / (2 ** bits - 1) * (max_val - min_val)


# ---------------------------------------------------------------------------
# Convenience: extract comm_type from a received CAN ID
# ---------------------------------------------------------------------------

def get_comm_type(can_id: int) -> int:
    """Extract the 5-bit communication type from bits 28–24 of the CAN ID."""
    return (can_id >> 24) & 0x1F


# ---------------------------------------------------------------------------
# CAN ID builders
# ---------------------------------------------------------------------------

def build_type1_id(motor_id: int, torque_raw: int) -> int:
    """
    Type 1 — Motion control command.

    torque_raw is a 16-bit value [0, 65535] placed in bits 23–8 of the CAN ID.
    The 8 data bytes carry angle, velocity, Kp, Kd.
    """
    return (0x01 << 24) | (torque_raw << 8) | motor_id


def build_type3_id(host_id: int, motor_id: int) -> int:
    """Type 3 — Motor enable."""
    return (0x03 << 24) | (host_id << 8) | motor_id


def build_type4_id(host_id: int, motor_id: int, clear_fault: bool = False) -> int:
    """
    Type 4 — Motor stop / clear fault.

    The clear_fault flag affects data byte 0 (0x01 if True, 0x00 otherwise).
    This function returns only the CAN ID; the caller must build the data frame
    with build_type4_data().
    """
    return (0x04 << 24) | (host_id << 8) | motor_id


def build_type6_id(host_id: int, motor_id: int) -> int:
    """Type 6 — Set motor origin."""
    return (0x06 << 24) | (host_id << 8) | motor_id


def build_type7_id(host_id: int, motor_id: int, new_id: int) -> int:
    """
    Type 7 — Change motor CAN ID.

    new_id occupies bits 23–16 of the CAN ID.
    """
    return (0x07 << 24) | (new_id << 16) | (host_id << 8) | motor_id


def build_type17_id(host_id: int, motor_id: int) -> int:
    """Type 17 (0x11) — Read parameter."""
    return (0x11 << 24) | (host_id << 8) | motor_id


def build_type18_id(host_id: int, motor_id: int) -> int:
    """Type 18 (0x12) — Write parameter."""
    return (0x12 << 24) | (host_id << 8) | motor_id


def build_type22_id(host_id: int, motor_id: int) -> int:
    """Type 22 (0x16) — Communication heartbeat / keep-alive."""
    return (0x16 << 24) | (host_id << 8) | motor_id


def build_type24_id(host_id: int, motor_id: int) -> int:
    """Type 24 (0x18) — Enable/disable active feedback reporting."""
    return (0x18 << 24) | (host_id << 8) | motor_id


# ---------------------------------------------------------------------------
# Data frame builders (all return bytes of length 8)
# ---------------------------------------------------------------------------

def build_type1_data(angle_raw: int, vel_raw: int, kp_raw: int, kd_raw: int) -> bytes:
    """
    Type 1 — Motion control data payload.

    Big-endian uint16 packing of angle, velocity, Kp, Kd.
    Torque feedforward is encoded in the CAN ID (bits 23–8), not here.
    """
    return struct.pack('>HHHH', angle_raw, vel_raw, kp_raw, kd_raw)


def build_type3_data() -> bytes:
    """Type 3 — Enable motor: 8 zero bytes."""
    return bytes(8)


def build_type4_data(clear_fault: bool = False) -> bytes:
    """
    Type 4 — Stop motor / clear fault.

    byte0 = 0x01 if clear_fault else 0x00; bytes 1–7 are zero.
    """
    byte0 = 0x01 if clear_fault else 0x00
    return bytes([byte0]) + bytes(7)


def build_type6_data() -> bytes:
    """Type 6 — Set origin: byte0 = 0x01, bytes 1–7 are zero."""
    return bytes([0x01]) + bytes(7)


def build_type17_data(param_index: int) -> bytes:
    """
    Type 17 — Read parameter request.

    param_index is encoded as a little-endian uint16 in bytes 0–1;
    bytes 2–7 are zero.
    """
    return struct.pack('<H', param_index) + bytes(6)


def build_type18_data(param_index: int, value: float, value_type: str = 'float') -> bytes:
    """
    Type 18 — Write parameter request.

    Layout: bytes 0–1 = param_index (LE uint16), bytes 2–3 = 0x00,
            bytes 4–7 = value encoded as little-endian (4 bytes).

    value_type must be one of: 'float', 'uint8', 'uint16', 'uint32'.
    Integer types are padded with zero bytes to fill 4 bytes.
    """
    index_bytes = struct.pack('<H', param_index)
    padding = b'\x00\x00'

    if value_type == 'float':
        value_bytes = struct.pack('<f', value)
    elif value_type == 'uint8':
        value_bytes = struct.pack('<B', int(value)) + bytes(3)
    elif value_type == 'uint16':
        value_bytes = struct.pack('<H', int(value)) + bytes(2)
    elif value_type == 'uint32':
        value_bytes = struct.pack('<I', int(value))
    else:
        raise ValueError(f"Unsupported value_type '{value_type}'. "
                         "Use 'float', 'uint8', 'uint16', or 'uint32'.")

    return index_bytes + padding + value_bytes


def build_type22_data() -> bytes:
    """Type 22 — Communication heartbeat: fixed payload 01 02 03 04 05 06 07 08."""
    return b'\x01\x02\x03\x04\x05\x06\x07\x08'


def build_type24_data(enable: bool = True) -> bytes:
    """
    Type 24 — Enable/disable active feedback reporting.

    bytes 0–5 = 01 02 03 04 05 06 (fixed),
    byte  6   = 0x01 if enable else 0x00,
    byte  7   = 0x00.
    """
    byte6 = 0x01 if enable else 0x00
    return b'\x01\x02\x03\x04\x05\x06' + bytes([byte6, 0x00])


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

def decode_type2_frame(
    can_id: int,
    data: bytes,
    vel_min: float,
    vel_max: float,
    torque_min: float,
    torque_max: float,
) -> MotorFeedback:
    """
    Decode a Type 2 motor feedback frame.

    CAN ID field extraction:
        motor_id           = bits 15–8
        host_id            = bits 7–0
        mode               = bits 23–22 (2 bits)
        fault_uncalibrated = bit 21
        fault_overload     = bit 20
        fault_encoder      = bit 19
        fault_overtemp     = bit 18
        fault_overcurrent  = bit 17
        fault_undervoltage = bit 16

    Data bytes (big-endian uint16):
        bytes 0–1: angle_raw  → position mapped to [-4π, +4π] rad
        bytes 2–3: vel_raw    → velocity mapped to [vel_min, vel_max] rad/s
        bytes 4–5: torque_raw → torque  mapped to [torque_min, torque_max] Nm
        bytes 6–7: temp_raw   → temperature = temp_raw / 10.0 °C

    vel_min/vel_max and torque_min/torque_max are motor-type-specific;
    they are defined in motor_config.py and passed by the caller.
    """
    motor_id = (can_id >> 8) & 0xFF
    host_id  = can_id & 0xFF
    mode               = (can_id >> 22) & 0x3
    fault_uncalibrated = bool((can_id >> 21) & 1)
    fault_overload     = bool((can_id >> 20) & 1)
    fault_encoder      = bool((can_id >> 19) & 1)
    fault_overtemp     = bool((can_id >> 18) & 1)
    fault_overcurrent  = bool((can_id >> 17) & 1)
    fault_undervoltage = bool((can_id >> 16) & 1)

    angle_raw, vel_raw, torque_raw = struct.unpack('>HHH', data[0:6])
    temp_raw = struct.unpack('>H', data[6:8])[0]

    position    = raw_to_value(angle_raw, -4 * math.pi, 4 * math.pi)
    velocity    = raw_to_value(vel_raw, vel_min, vel_max)
    torque      = raw_to_value(torque_raw, torque_min, torque_max)
    temperature = temp_raw / 10.0

    return MotorFeedback(
        motor_id=motor_id,
        host_id=host_id,
        position=position,
        velocity=velocity,
        torque=torque,
        temperature=temperature,
        mode=mode,
        fault_uncalibrated=fault_uncalibrated,
        fault_overload=fault_overload,
        fault_encoder=fault_encoder,
        fault_overtemp=fault_overtemp,
        fault_overcurrent=fault_overcurrent,
        fault_undervoltage=fault_undervoltage,
    )


def decode_type2_active(
    can_id: int,
    data: bytes,
    vel_min: float,
    vel_max: float,
    torque_min: float,
    torque_max: float,
) -> MotorFeedback:
    """
    Decode a Type 24 active report frame (comm_type = 0x18).

    The data layout is identical to Type 2; only the comm_type field in the
    CAN ID differs.  motor_id and host_id positions are the same.
    """
    return decode_type2_frame(can_id, data, vel_min, vel_max, torque_min, torque_max)


def decode_type17_reply(can_id: int, data: bytes) -> ParamReply:
    """
    Decode a Type 17 parameter read reply.

    success     = True if bits 23–16 of the CAN ID are 0x00
    motor_id    = bits 15–8 of the CAN ID
    param_index = little-endian uint16 from data bytes 0–1
    value_bytes = data bytes 4–7 (raw 4 bytes; caller interprets as float or int)
    """
    success     = ((can_id >> 16) & 0xFF) == 0x00
    motor_id    = (can_id >> 8) & 0xFF
    param_index = struct.unpack('<H', data[0:2])[0]
    value_bytes = data[4:8]

    return ParamReply(
        motor_id=motor_id,
        param_index=param_index,
        value_bytes=bytes(value_bytes),
        success=success,
    )


def decode_type21_frame(can_id: int, data: bytes) -> FaultFrame:
    """
    Decode a Type 21 fault status frame.

    motor_id     = bits 15–8 of the CAN ID
    fault_bits   = little-endian uint32 from data bytes 0–3
    warning_bits = little-endian uint32 from data bytes 4–7

    Named fault bits (from fault_bits):
        bit 16 : fault_overcurrent_a
        bit 14 : fault_stall
        bit  9 : fault_pos_init
        bit  8 : fault_hw_id
        bit  7 : fault_encoder_uncal
        bit  5 : fault_overcurrent_c
        bit  4 : fault_overcurrent_b
        bit  3 : fault_overvoltage
        bit  2 : fault_undervoltage
        bit  1 : fault_driver
        bit  0 : fault_overtemp

    Named warning bits (from warning_bits):
        bit  0 : warning_overtemp
    """
    motor_id     = (can_id >> 8) & 0xFF
    fault_bits   = struct.unpack('<I', data[0:4])[0]
    warning_bits = struct.unpack('<I', data[4:8])[0]

    return FaultFrame(
        motor_id=motor_id,
        fault_bits=fault_bits,
        warning_bits=warning_bits,
        fault_overcurrent_a  = bool((fault_bits >> 16) & 1),
        fault_stall          = bool((fault_bits >> 14) & 1),
        fault_pos_init       = bool((fault_bits >>  9) & 1),
        fault_hw_id          = bool((fault_bits >>  8) & 1),
        fault_encoder_uncal  = bool((fault_bits >>  7) & 1),
        fault_overcurrent_c  = bool((fault_bits >>  5) & 1),
        fault_overcurrent_b  = bool((fault_bits >>  4) & 1),
        fault_overvoltage    = bool((fault_bits >>  3) & 1),
        fault_undervoltage   = bool((fault_bits >>  2) & 1),
        fault_driver         = bool((fault_bits >>  1) & 1),
        fault_overtemp       = bool((fault_bits >>  0) & 1),
        warning_overtemp     = bool((warning_bits >> 0) & 1),
    )
