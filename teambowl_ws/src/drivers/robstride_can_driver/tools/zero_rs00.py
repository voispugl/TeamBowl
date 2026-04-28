"""
zero_rs00.py — Interactive zero-setter for RS00 foot actuators.

Displays live position of both RS00 motors, then lets you set the current
position as mechanical zero (Type 6) per-motor or both at once.

No ROS2 required. Requires python-can and can1 up at 1 Mbps.

Usage:
    python3 zero_rs00.py [--interface can1]

    sudo ip link set can1 up type can bitrate 1000000
"""

import argparse
import struct
import time
import math
import sys

HOST_ID  = 0x01
RS00_IDS = [0x0D, 0x17]
LABELS   = {0x0D: 'Left Foot  (0x0D)', 0x17: 'Right Foot (0x17)'}

# RS00 physical ranges
POS_MIN, POS_MAX = -4 * math.pi, 4 * math.pi   # rad
VEL_MIN, VEL_MAX = -33.0, 33.0                  # rad/s
TRQ_MIN, TRQ_MAX = -14.0, 14.0                  # Nm

FEED_TIMEOUT_S = 2.0   # seconds before a motor is shown as STALE


# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------

def _arb(comm_type, data_area2, motor_id):
    return (comm_type << 24) | (data_area2 << 8) | motor_id


def frame_enable(motor_id):
    return dict(arbitration_id=_arb(0x03, HOST_ID, motor_id),
                data=bytes(8), is_extended_id=True)


def frame_stop(motor_id):
    return dict(arbitration_id=_arb(0x04, HOST_ID, motor_id),
                data=bytes(8), is_extended_id=True)


def frame_set_zero(motor_id):
    """Type 6 — set current mechanical position as the new zero."""
    return dict(arbitration_id=_arb(0x06, HOST_ID, motor_id),
                data=b'\x01' + bytes(7), is_extended_id=True)


def frame_save(motor_id):
    """Type 22 — save params to flash."""
    return dict(arbitration_id=_arb(0x16, HOST_ID, motor_id),
                data=b'\x01\x02\x03\x04\x05\x06\x07\x08', is_extended_id=True)


# ---------------------------------------------------------------------------
# Feedback decode
# ---------------------------------------------------------------------------

def _raw_to(raw, lo, hi, bits=16):
    return lo + raw / (2 ** bits - 1) * (hi - lo)


def decode_feedback(msg):
    """Return (motor_id, pos_rad, vel_rad_s, torque_nm) from a Type 2 frame, or None."""
    if (msg.arbitration_id >> 24) & 0x1F != 0x02:
        return None
    if len(msg.data) < 8:
        return None
    motor_id  = (msg.arbitration_id >> 8) & 0xFF
    pos   = _raw_to(struct.unpack_from('>H', msg.data, 0)[0], POS_MIN, POS_MAX)
    vel   = _raw_to(struct.unpack_from('>H', msg.data, 2)[0], VEL_MIN, VEL_MAX)
    torq  = _raw_to(struct.unpack_from('>H', msg.data, 4)[0], TRQ_MIN, TRQ_MAX)
    return motor_id, pos, vel, torq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def send(bus, frame):
    import can
    bus.send(can.Message(**frame))


def drain(bus, states, poll_s=0.05):
    """Drain incoming frames for poll_s seconds, updating states dict."""
    end = time.monotonic() + poll_s
    while time.monotonic() < end:
        msg = bus.recv(timeout=max(0.0, end - time.monotonic()))
        if msg is None:
            break
        r = decode_feedback(msg)
        if r and r[0] in states:
            states[r[0]] = (r[1], r[2], r[3], time.monotonic())


def print_status(states):
    now = time.monotonic()
    print()
    for mid in RS00_IDS:
        entry = states[mid]
        if entry is None or (now - entry[3]) > FEED_TIMEOUT_S:
            print(f'  {LABELS[mid]}:  -- no feedback --')
        else:
            pos, vel, torq, _ = entry
            print(f'  {LABELS[mid]}:  pos={pos:+.4f} rad ({math.degrees(pos):+.2f}°)'
                  f'  vel={vel:+.3f} r/s  torq={torq:+.3f} Nm')
    print()


# ---------------------------------------------------------------------------
# Interactive zero workflow
# ---------------------------------------------------------------------------

def set_zero_motor(bus, states, motor_id, save):
    entry = states[motor_id]
    if entry is None:
        print(f'  WARNING: no feedback from {LABELS[motor_id]} — zeroing anyway.')
    else:
        pos = entry[0]
        print(f'  {LABELS[motor_id]}: current pos = {pos:+.4f} rad ({math.degrees(pos):+.2f}°)')

    send(bus, frame_set_zero(motor_id))
    time.sleep(0.05)

    if save:
        send(bus, frame_save(motor_id))
        time.sleep(0.1)
        print(f'  {LABELS[motor_id]}: zero set and saved to flash.')
    else:
        print(f'  {LABELS[motor_id]}: zero set (NOT saved — power cycle will revert).')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Interactive zero-setter for RS00 wheel actuators')
    parser.add_argument('--interface', default='can1')
    args = parser.parse_args()

    import can

    print(f'\nOpening {args.interface}...')
    try:
        bus = can.Bus(interface='socketcan', channel=args.interface, bitrate=1000000)
    except Exception as e:
        print(f'ERROR: {e}')
        print(f'  sudo ip link set {args.interface} up type can bitrate 1000000')
        return

    # states: motor_id → (pos, vel, torq, timestamp) or None
    states = {mid: None for mid in RS00_IDS}

    # Enable motors
    print('Enabling RS00 motors...')
    for mid in RS00_IDS:
        send(bus, frame_enable(mid))
        time.sleep(0.02)

    # Collect ~0.5 s of feedback to get stable readings
    print('Reading positions...')
    drain(bus, states, poll_s=0.5)

    # ── Display current positions ────────────────────────────────────────────
    print('\n' + '─' * 55)
    print('  RS00 Foot Positions')
    print('─' * 55)
    print_status(states)

    # ── Menu ────────────────────────────────────────────────────────────────
    print('  Commands:')
    print('    l  — set zero on Left Foot  (0x0D)')
    print('    r  — set zero on Right Foot (0x17)')
    print('    b  — set zero on Both motors')
    print('    s  — refresh position readout')
    print('    q  — quit (no changes)')
    print()

    save_prompt_done = False
    save_to_flash = False

    try:
        while True:
            try:
                cmd = input('  > ').strip().lower()
            except EOFError:
                break

            if cmd == 'q':
                break

            elif cmd == 's':
                drain(bus, states, poll_s=0.3)
                print_status(states)

            elif cmd in ('l', 'r', 'b'):
                # Ask once whether to save to flash
                if not save_prompt_done:
                    ans = input('  Save zero to flash? Motor will remember after power cycle. [y/N] ').strip().lower()
                    save_to_flash = ans == 'y'
                    save_prompt_done = True

                targets = RS00_IDS if cmd == 'b' else [0x0D if cmd == 'l' else 0x17]

                # Refresh readings right before zeroing
                drain(bus, states, poll_s=0.2)

                print()
                for mid in targets:
                    set_zero_motor(bus, states, mid, save=save_to_flash)

                # Read back new positions
                time.sleep(0.1)
                drain(bus, states, poll_s=0.3)
                print('\n  Positions after zero:')
                print_status(states)

            else:
                print('  Unknown command. Use l / r / b / s / q.')

    except KeyboardInterrupt:
        print()

    # Stop motors
    print('Sending stop...')
    for mid in RS00_IDS:
        send(bus, frame_stop(mid))
        time.sleep(0.01)

    bus.shutdown()
    print('Done.')


if __name__ == '__main__':
    main()
