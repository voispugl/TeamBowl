"""Unit tests for can_protocol.py — no ROS2 or hardware required.

Run with: pytest test/test_can_protocol.py -v
"""
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from robstride_can_driver.can_protocol import (
    build_type1_id, build_type1_data,
    build_type3_id, build_type3_data,
    build_type4_id, build_type4_data,
    build_type6_id, build_type6_data,
    build_type7_id,
    build_type17_id, build_type17_data,
    build_type18_id, build_type18_data,
    build_type22_id, build_type22_data,
    build_type24_id, build_type24_data,
    decode_type2_frame, decode_type17_reply, decode_type21_frame,
    scale_to_raw, raw_to_value, get_comm_type,
)

# ---------------------------------------------------------------------------
# Scaling tests
# ---------------------------------------------------------------------------

def test_scale_roundtrip_rs04_torque():
    """Scale -60 Nm in RS04 torque range and back."""
    raw = scale_to_raw(-60.0, -120.0, 120.0)
    recovered = raw_to_value(raw, -120.0, 120.0)
    assert abs(recovered - (-60.0)) < 0.01

def test_scale_roundtrip_rs00_velocity():
    """Scale 15.0 rad/s in RS00 velocity range and back."""
    raw = scale_to_raw(15.0, -33.0, 33.0)
    recovered = raw_to_value(raw, -33.0, 33.0)
    assert abs(recovered - 15.0) < 0.01

def test_scale_clamp_max():
    raw = scale_to_raw(999.0, -120.0, 120.0)
    assert raw == 65535

def test_scale_clamp_min():
    raw = scale_to_raw(-999.0, -120.0, 120.0)
    assert raw == 0

def test_scale_midpoint():
    """Zero should map to mid-range raw value (32767 or 32768)."""
    raw = scale_to_raw(0.0, -120.0, 120.0)
    assert 32760 <= raw <= 32775

# ---------------------------------------------------------------------------
# CAN ID builder tests
# ---------------------------------------------------------------------------

def test_type1_id_fields():
    torque_raw = 32767
    motor_id = 0x01
    can_id = build_type1_id(motor_id, torque_raw)
    assert get_comm_type(can_id) == 0x01
    assert (can_id & 0xFF) == motor_id
    assert ((can_id >> 8) & 0xFFFF) == torque_raw

def test_type1_data_big_endian():
    data = build_type1_data(1000, 2000, 3000, 4000)
    assert len(data) == 8
    unpacked = struct.unpack('>HHHH', data)
    assert unpacked == (1000, 2000, 3000, 4000)

def test_type3_id():
    can_id = build_type3_id(host_id=0xFD, motor_id=0x02)
    assert get_comm_type(can_id) == 0x03
    assert ((can_id >> 8) & 0xFF) == 0xFD
    assert (can_id & 0xFF) == 0x02

def test_type3_data():
    assert build_type3_data() == bytes(8)

def test_type4_data_normal():
    data = build_type4_data(clear_fault=False)
    assert data[0] == 0x00
    assert data[1:] == bytes(7)

def test_type4_data_clear_fault():
    data = build_type4_data(clear_fault=True)
    assert data[0] == 0x01

def test_type6_data():
    data = build_type6_data()
    assert data[0] == 0x01
    assert data[1:] == bytes(7)

def test_type7_id():
    can_id = build_type7_id(host_id=0xFD, motor_id=0x01, new_id=0x05)
    assert get_comm_type(can_id) == 0x07
    assert ((can_id >> 16) & 0xFF) == 0x05   # new_id in bits 23-16
    assert ((can_id >> 8) & 0xFF) == 0xFD    # host_id in bits 15-8
    assert (can_id & 0xFF) == 0x01           # motor_id in bits 7-0

def test_type17_data_little_endian():
    data = build_type17_data(0x701C)
    assert data[0:2] == struct.pack('<H', 0x701C)
    assert data[2:] == bytes(6)

def test_type18_data_float():
    data = build_type18_data(0x702B, 1.5, 'float')
    assert data[0:2] == struct.pack('<H', 0x702B)
    assert data[2:4] == b'\x00\x00'
    assert data[4:8] == struct.pack('<f', 1.5)

def test_type18_data_uint8():
    data = build_type18_data(0x7005, 3.0, 'uint8')
    assert data[4] == 3
    assert data[5:8] == bytes(3)

def test_type22_data():
    assert build_type22_data() == b'\x01\x02\x03\x04\x05\x06\x07\x08'

def test_type24_data_enable():
    data = build_type24_data(enable=True)
    assert data[0:6] == b'\x01\x02\x03\x04\x05\x06'
    assert data[6] == 0x01
    assert data[7] == 0x00

def test_type24_data_disable():
    data = build_type24_data(enable=False)
    assert data[6] == 0x00

# ---------------------------------------------------------------------------
# Decoder tests
# ---------------------------------------------------------------------------

def test_decode_type2_rs04_center():
    """Center values (raw=32767~32768) should decode near zero for RS04."""
    motor_id = 0x01
    host_id = 0xFD
    mode = 2  # Run
    # CAN ID: comm_type=0x02, mode=2 in bits 23-22, motor_id in bits 15-8, host_id in bits 7-0
    can_id = (0x02 << 24) | (mode << 22) | (motor_id << 8) | host_id
    # Center raw values
    center = 32768
    temp_raw = 250  # 25.0 °C
    data = struct.pack('>HHHHHH', center, center, center, center)[0:6] + struct.pack('>H', temp_raw)
    # RS04 ranges
    fb = decode_type2_frame(can_id, data, vel_min=-15.0, vel_max=15.0, torque_min=-120.0, torque_max=120.0)
    assert fb.motor_id == motor_id
    assert fb.mode == mode
    assert abs(fb.temperature - 25.0) < 0.2
    assert not fb.fault_overtemp
    assert not fb.fault_undervoltage

def test_decode_type2_faults():
    """Fault bits in CAN ID should parse correctly."""
    motor_id = 0x03
    host_id = 0xFD
    # Set fault_overtemp (bit 18) and fault_undervoltage (bit 16)
    can_id = (0x02 << 24) | (1 << 18) | (1 << 16) | (motor_id << 8) | host_id
    data = bytes(8)
    fb = decode_type2_frame(can_id, data, -15.0, 15.0, -120.0, 120.0)
    assert fb.fault_overtemp is True
    assert fb.fault_undervoltage is True
    assert fb.fault_overcurrent is False

def test_decode_type17_reply_success():
    motor_id = 0x01
    host_id = 0xFD
    # success: bits 23-16 == 0x00
    can_id = (0x11 << 24) | (0x00 << 16) | (motor_id << 8) | host_id
    value = struct.pack('<f', 3.14)
    data = struct.pack('<H', 0x701C) + b'\x00\x00' + value
    reply = decode_type17_reply(can_id, data)
    assert reply.success is True
    assert reply.motor_id == motor_id
    assert reply.param_index == 0x701C
    assert abs(struct.unpack('<f', reply.value_bytes)[0] - 3.14) < 0.001

def test_decode_type17_reply_failure():
    motor_id = 0x02
    host_id = 0xFD
    # failure: bits 23-16 == 0x01
    can_id = (0x11 << 24) | (0x01 << 16) | (motor_id << 8) | host_id
    data = struct.pack('<H', 0x7005) + b'\x00\x00' + b'\x00' * 4
    reply = decode_type17_reply(can_id, data)
    assert reply.success is False

def test_decode_type21_faults():
    motor_id = 0x01
    host_id = 0xFD
    can_id = (0x15 << 24) | (motor_id << 8) | host_id
    fault_bits = (1 << 14) | (1 << 2)   # stall + undervoltage
    data = struct.pack('<I', fault_bits) + struct.pack('<I', 0)
    ff = decode_type21_frame(can_id, data)
    assert ff.fault_stall is True
    assert ff.fault_undervoltage is True
    assert ff.fault_overtemp is False
    assert ff.warning_overtemp is False

def test_get_comm_type():
    assert get_comm_type(0x01000001) == 0x01
    assert get_comm_type(0x02000000) == 0x02
    assert get_comm_type(0x18FDFD01) == 0x18
