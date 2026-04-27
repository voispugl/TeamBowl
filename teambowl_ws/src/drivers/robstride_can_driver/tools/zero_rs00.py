"""
zero_rs00.py — Move both RS00 wheel actuators to position 0.0 rad.

No ROS2 required. Requires python-can and can1 to be up at 1 Mbps.

Usage:
    python3 zero_rs00.py [--kp 80] [--kd 2] [--timeout 8]

    sudo ip link set can1 up type can bitrate 1000000
"""

import argparse
import struct
import time
import math

HOST_ID  = 0x01
RS00_IDS = [0x0D, 0x17]   # left wheel = 0x0D (13), right wheel = 0x17 (23)

# RS00 physical ranges (from motor spec / can_protocol.py)
POS_MIN,  POS_MAX  = -4 * math.pi,  4 * math.pi   # rad
VEL_MIN,  VEL_MAX  = -33.0,         33.0           # rad/s
KP_MIN,   KP_MAX   = 0.0,           500.0
KD_MIN,   KD_MAX   = 0.0,           5.0
TRQ_MIN,  TRQ_MAX  = -14.0,         14.0           # Nm

AT_ZERO_THRESHOLD_RAD = 0.05   # consider "at zero" when within this many rad


def scale(value, lo, hi, bits=16):
    clamped = max(lo, min(hi, value))
    return int((clamped - lo) / (hi - lo) * (2**bits - 1))


def raw_to_value(raw, lo, hi, bits=16):
    return lo + raw / (2**bits - 1) * (hi - lo)


# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------

def frame_enable(motor_id):
    arb_id = (0x03 << 24) | (HOST_ID << 8) | motor_id
    return {'arbitration_id': arb_id, 'data': bytes(8), 'is_extended_id': True}


def frame_stop(motor_id):
    arb_id = (0x04 << 24) | (HOST_ID << 8) | motor_id
    return {'arbitration_id': arb_id, 'data': bytes(8), 'is_extended_id': True}


def frame_motion(motor_id, pos_rad, vel_rad_s, kp, kd, torque_nm=0.0):
    """Type 1 — motion control frame targeting (pos_rad, vel_rad_s) with gains kp, kd."""
    angle_raw  = scale(pos_rad,   POS_MIN, POS_MAX)
    vel_raw    = scale(vel_rad_s, VEL_MIN, VEL_MAX)
    kp_raw     = scale(kp,        KP_MIN,  KP_MAX)
    kd_raw     = scale(kd,        KD_MIN,  KD_MAX)
    torque_raw = scale(torque_nm, TRQ_MIN, TRQ_MAX)

    arb_id = (0x01 << 24) | (torque_raw << 8) | motor_id
    data   = struct.pack('>HHHH', angle_raw, vel_raw, kp_raw, kd_raw)
    return {'arbitration_id': arb_id, 'data': data, 'is_extended_id': True}


# ---------------------------------------------------------------------------
# Feedback decode
# ---------------------------------------------------------------------------

def decode_feedback(msg):
    """
    Decode a Type 2 feedback frame. Returns (motor_id, pos_rad) or None if not Type 2.
    Type 2 comm_type = 0x02; motor_id is in bits 15–8 of the CAN ID.
    """
    comm_type = (msg.arbitration_id >> 24) & 0x1F
    if comm_type != 0x02:
        return None
    if len(msg.data) < 8:
        return None

    motor_id  = (msg.arbitration_id >> 8) & 0xFF
    angle_raw = struct.unpack_from('>H', msg.data, 0)[0]
    pos_rad   = raw_to_value(angle_raw, POS_MIN, POS_MAX)
    return motor_id, pos_rad


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Move RS00 wheel actuators to zero position')
    parser.add_argument('--interface', default='can1')
    parser.add_argument('--kp',      type=float, default=80.0,  help='Position gain (default 80)')
    parser.add_argument('--kd',      type=float, default=2.0,   help='Damping gain (default 2)')
    parser.add_argument('--timeout', type=float, default=8.0,   help='Max seconds to wait (default 8)')
    args = parser.parse_args()

    import can

    print(f'Opening {args.interface}...')
    try:
        bus = can.Bus(interface='socketcan', channel=args.interface, bitrate=1000000)
    except Exception as e:
        print(f'ERROR: {e}')
        print(f'  sudo ip link set {args.interface} up type can bitrate 1000000')
        return

    positions = {mid: None for mid in RS00_IDS}
    labels    = {0x0D: 'left (0x0D)', 0x17: 'right (0x17)'}

    # --- Enable ---
    print('Enabling RS00 motors...')
    for mid in RS00_IDS:
        f = frame_enable(mid)
        bus.send(can.Message(**f))
        time.sleep(0.02)

    time.sleep(0.1)  # let motors come up

    print(f'Moving to 0.0 rad  (kp={args.kp}, kd={args.kd}, timeout={args.timeout}s)\n')

    deadline = time.monotonic() + args.timeout
    last_print = 0.0

    while time.monotonic() < deadline:
        # Send motion commands to both motors
        for mid in RS00_IDS:
            f = frame_motion(mid, pos_rad=0.0, vel_rad_s=0.0, kp=args.kp, kd=args.kd)
            bus.send(can.Message(**f))

        # Drain feedback frames for up to 20 ms
        poll_end = time.monotonic() + 0.02
        while time.monotonic() < poll_end:
            msg = bus.recv(timeout=max(0.0, poll_end - time.monotonic()))
            if msg is None:
                break
            result = decode_feedback(msg)
            if result is not None:
                mid, pos = result
                if mid in positions:
                    positions[mid] = pos

        # Print status at ~4 Hz
        now = time.monotonic()
        if now - last_print >= 0.25:
            last_print = now
            parts = []
            for mid in RS00_IDS:
                p = positions[mid]
                if p is None:
                    parts.append(f'{labels[mid]}: --')
                else:
                    marker = ' ✓' if abs(p) <= AT_ZERO_THRESHOLD_RAD else ''
                    parts.append(f'{labels[mid]}: {p:+.3f} rad{marker}')
            print('  ' + '    |    '.join(parts))

        # Stop early if both are at zero
        all_done = all(
            p is not None and abs(p) <= AT_ZERO_THRESHOLD_RAD
            for p in positions.values()
        )
        if all_done:
            print('\nBoth motors at zero.')
            break
    else:
        print('\nTimeout reached.')
        for mid in RS00_IDS:
            p = positions[mid]
            if p is None:
                print(f'  {labels[mid]}: no feedback received')
            else:
                print(f'  {labels[mid]}: final position = {p:+.3f} rad')

    # --- Stop ---
    print('Sending stop...')
    for mid in RS00_IDS:
        f = frame_stop(mid)
        bus.send(can.Message(**f))
        time.sleep(0.01)

    bus.shutdown()
    print('Done.')


if __name__ == '__main__':
    main()
