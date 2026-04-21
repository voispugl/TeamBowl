#!/usr/bin/env python3
"""
Standalone VESC serial test — no ROS needed.
Tests each ttyACM port: can we get a response, and can we send RPM?

Usage:
    python3 ~/TeamBowl/test_vesc_serial.py [--spin PORT --rpm VALUE]

    Default: scan all ttyACM ports, query GET_VALUES, report.
    --spin /dev/ttyACMx --rpm 500   Send a SetRPM command for 2 s then stop.
"""
import sys
import struct
import time
import glob
import argparse


# ── Packet helpers (same as cmd_vel_to_vesc.py) ──────────────────────────────

COMM_GET_VALUES = 4
COMM_SET_RPM    = 8
COMM_SET_DUTY   = 5

def crc16(data):
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc

def make_packet(payload):
    n = len(payload)
    hdr = bytes([2, n]) if n < 256 else bytes([3, n >> 8, n & 0xFF])
    c = crc16(payload)
    return hdr + payload + bytes([c >> 8, c & 0xFF, 3])

def get_values_pkt():
    return make_packet(bytes([COMM_GET_VALUES]))

def set_rpm_pkt(erpm):
    return make_packet(bytes([COMM_SET_RPM]) + struct.pack('>i', erpm))

def set_duty_pkt(duty_1e5):  # duty in 1e5 units, e.g. 0 = coast
    return make_packet(bytes([COMM_SET_DUTY]) + struct.pack('>i', duty_1e5))

def read_packet(ser, timeout=0.2):
    ser.timeout = timeout
    start = ser.read(1)
    if not start:
        return None
    if start[0] == 2:
        lb = ser.read(1)
        if not lb:
            return None
        n = lb[0]
    elif start[0] == 3:
        lb = ser.read(2)
        if len(lb) < 2:
            return None
        n = (lb[0] << 8) | lb[1]
    else:
        return None
    payload = ser.read(n)
    crc_bytes = ser.read(2)
    end = ser.read(1)
    if len(payload) != n or len(crc_bytes) != 2 or not end:
        return None
    if end[0] != 3:
        return None
    expected = crc16(payload)
    got = (crc_bytes[0] << 8) | crc_bytes[1]
    if expected != got:
        print(f"  CRC mismatch: expected 0x{expected:04X} got 0x{got:04X}")
        return None
    return payload

def parse_get_values(payload):
    if len(payload) < 27 or payload[0] != COMM_GET_VALUES:
        return None
    # Offsets per VESC protocol (firmware 3.x/4.x common layout)
    try:
        temp_fet   = struct.unpack_from('>h', payload, 1)[0] / 10.0
        temp_mot   = struct.unpack_from('>h', payload, 3)[0] / 10.0
        avg_motor_current = struct.unpack_from('>i', payload, 5)[0] / 100.0
        avg_input_current = struct.unpack_from('>i', payload, 9)[0] / 100.0
        duty_now   = struct.unpack_from('>h', payload, 17)[0] / 1000.0
        rpm        = struct.unpack_from('>i', payload, 23)[0]
        v_in       = struct.unpack_from('>h', payload, 27)[0] / 10.0
        return dict(temp_fet=temp_fet, temp_mot=temp_mot,
                    motor_A=avg_motor_current, input_A=avg_input_current,
                    duty=duty_now, rpm=rpm, v_in=v_in)
    except Exception as e:
        return {'parse_error': str(e), 'payload_hex': payload.hex()}


# ── Main ──────────────────────────────────────────────────────────────────────

def probe_port(port):
    import serial
    print(f"\n{'─'*50}")
    print(f"  Port: {port}")
    try:
        ser = serial.Serial(port, baudrate=115200, timeout=0.3)
    except Exception as e:
        print(f"  OPEN FAILED: {e}")
        return False

    print(f"  Opened OK")

    # Flush any stale data
    ser.reset_input_buffer()
    time.sleep(0.05)

    # Send GET_VALUES
    ser.write(get_values_pkt())
    payload = read_packet(ser)

    if payload is None:
        print(f"  GET_VALUES: NO RESPONSE (not a VESC, or wrong port)")
        ser.close()
        return False

    vals = parse_get_values(payload)
    if vals is None:
        print(f"  GET_VALUES: response too short ({len(payload)} bytes)")
        ser.close()
        return False

    if 'parse_error' in vals:
        print(f"  GET_VALUES: parse error — {vals}")
    else:
        print(f"  GET_VALUES OK:")
        print(f"    Temp FET={vals['temp_fet']}°C  Motor={vals['temp_mot']}°C")
        print(f"    Input voltage={vals['v_in']} V")
        print(f"    RPM={vals['rpm']}  Duty={vals['duty']:.3f}")
        print(f"    Motor current={vals['motor_A']} A  Input={vals['input_A']} A")
        if vals['v_in'] < 5.0:
            print(f"  ⚠ LOW VOLTAGE — VESC may not be powered from battery")

    ser.close()
    return True


def spin_test(port, erpm):
    import serial
    print(f"\nSpin test: {port} at {erpm} ERPM for 2 s")
    try:
        ser = serial.Serial(port, baudrate=115200, timeout=0.3)
    except Exception as e:
        print(f"  OPEN FAILED: {e}")
        return

    # Query before
    ser.reset_input_buffer()
    ser.write(get_values_pkt())
    payload = read_packet(ser)
    if payload:
        vals = parse_get_values(payload)
        if vals and 'v_in' in vals:
            print(f"  Before: RPM={vals['rpm']}, V_in={vals['v_in']} V")

    print(f"  Sending SetRPM({erpm})...")
    for _ in range(20):   # 20 × 100 ms = 2 s
        ser.write(set_rpm_pkt(erpm))
        time.sleep(0.1)

    print(f"  Sending stop (SetRPM 0)...")
    ser.write(set_rpm_pkt(0))
    time.sleep(0.2)

    # Query after
    ser.reset_input_buffer()
    ser.write(get_values_pkt())
    payload = read_packet(ser)
    if payload:
        vals = parse_get_values(payload)
        if vals and 'v_in' in vals:
            print(f"  After:  RPM={vals['rpm']}, Duty={vals['duty']:.3f}, Motor A={vals['motor_A']}")

    ser.close()
    print("  Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spin', metavar='PORT', help='Port to send SetRPM to')
    parser.add_argument('--rpm', type=int, default=1000, help='ERPM to command (default 1000)')
    args = parser.parse_args()

    if args.spin:
        spin_test(args.spin, args.rpm)
        return

    ports = sorted(glob.glob('/dev/ttyACM*'))
    if not ports:
        print("No /dev/ttyACM* devices found — check USB cables")
        return

    print(f"Found ports: {ports}")
    print("Probing each for VESC GET_VALUES response...")

    vesc_ports = []
    for p in ports:
        if probe_port(p):
            vesc_ports.append(p)

    print(f"\n{'='*50}")
    print(f"VESC ports found: {vesc_ports if vesc_ports else 'NONE'}")
    if not vesc_ports:
        print("No VESC responses — check power and USB connections")
    else:
        print(f"\nTo spin-test a wheel:")
        print(f"  python3 ~/TeamBowl/test_vesc_serial.py --spin {vesc_ports[0]} --rpm 1000")
        print(f"  (negative rpm reverses direction)")


if __name__ == '__main__':
    main()
