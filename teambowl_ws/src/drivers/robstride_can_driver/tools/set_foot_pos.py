"""
set_foot_pos.py — Interactive foot position controller for RS00 actuators.

Zero = hardstop (feet fully retracted upward).
Positive extension moves feet downward. Because the feet are mechanically
mirrored, commanding extension E means:
    left foot  (0x0D) = +E
    right foot (0x17) = -E
This ensures both feet reach the same height when on the same surface.

No ROS2 required. Requires python-can and can1 up at 1 Mbps.

Usage:
    python3 set_foot_pos.py [--interface can1] [--kp 40] [--kd 2]

    sudo ip link set can1 up type can bitrate 1000000
"""

import argparse
import struct
import time
import math

HOST_ID  = 0x01
LEFT_ID  = 0x0D
RIGHT_ID = 0x17
RS00_IDS = [LEFT_ID, RIGHT_ID]

LABELS = {LEFT_ID: 'Left Foot  (0x0D)', RIGHT_ID: 'Right Foot (0x17)'}

# RS00 physical ranges
POS_MIN, POS_MAX = -4 * math.pi,  4 * math.pi   # rad
VEL_MIN, VEL_MAX = -33.0,         33.0           # rad/s
KP_MIN,  KP_MAX  =   0.0,         500.0
KD_MIN,  KD_MAX  =   0.0,           5.0
TRQ_MIN, TRQ_MAX = -14.0,          14.0          # Nm

FEED_TIMEOUT_S   = 2.0    # s before motor shown as STALE
AT_TARGET_RAD    = 0.03   # rad — "close enough" threshold for arrival marker
CONTROL_HZ       = 50     # Hz — command rate while holding position


# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------

def _arb(comm_type, data_area2, motor_id):
    return (comm_type << 24) | (data_area2 << 8) | motor_id


def _scale(value, lo, hi, bits=16):
    clamped = max(lo, min(hi, value))
    return int((clamped - lo) / (hi - lo) * (2 ** bits - 1))


def frame_enable(motor_id):
    return dict(arbitration_id=_arb(0x03, HOST_ID, motor_id),
                data=bytes(8), is_extended_id=True)


def frame_stop(motor_id):
    return dict(arbitration_id=_arb(0x04, HOST_ID, motor_id),
                data=bytes(8), is_extended_id=True)


def frame_motion(motor_id, pos_rad, kp, kd, torque_ff=0.0):
    """Type 1 — position hold at pos_rad with gains kp, kd."""
    torque_raw = _scale(torque_ff, TRQ_MIN, TRQ_MAX)
    arb_id = _arb(0x01, torque_raw, motor_id)
    data = struct.pack('>HHHH',
                       _scale(pos_rad,  POS_MIN, POS_MAX),
                       _scale(0.0,      VEL_MIN, VEL_MAX),
                       _scale(kp,       KP_MIN,  KP_MAX),
                       _scale(kd,       KD_MIN,  KD_MAX))
    return dict(arbitration_id=arb_id, data=data, is_extended_id=True)


# ---------------------------------------------------------------------------
# Feedback decode
# ---------------------------------------------------------------------------

def _raw_to(raw, lo, hi, bits=16):
    return lo + raw / (2 ** bits - 1) * (hi - lo)


def decode_feedback(msg):
    """Return (motor_id, pos, vel, torq) from a Type 2 frame, or None."""
    if (msg.arbitration_id >> 24) & 0x1F != 0x02:
        return None
    if len(msg.data) < 8:
        return None
    mid  = (msg.arbitration_id >> 8) & 0xFF
    pos  = _raw_to(struct.unpack_from('>H', msg.data, 0)[0], POS_MIN, POS_MAX)
    vel  = _raw_to(struct.unpack_from('>H', msg.data, 2)[0], VEL_MIN, VEL_MAX)
    torq = _raw_to(struct.unpack_from('>H', msg.data, 4)[0], TRQ_MIN, TRQ_MAX)
    return mid, pos, vel, torq


# ---------------------------------------------------------------------------
# Bus helpers
# ---------------------------------------------------------------------------

def send(bus, frame):
    import can
    bus.send(can.Message(**frame))


def drain(bus, states, poll_s=0.05):
    """Read incoming frames for poll_s seconds, update states."""
    end = time.monotonic() + poll_s
    while time.monotonic() < end:
        msg = bus.recv(timeout=max(0.0, end - time.monotonic()))
        if msg is None:
            break
        r = decode_feedback(msg)
        if r and r[0] in states:
            states[r[0]] = (r[1], r[2], r[3], time.monotonic())


def send_targets(bus, targets, kp, kd):
    """Send Type 1 motion command to each motor using its current target."""
    for mid, pos_rad in targets.items():
        send(bus, frame_motion(mid, pos_rad, kp, kd))


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_status(states, targets):
    """Print a two-line status block for both feet."""
    now = time.monotonic()
    print()
    for mid in RS00_IDS:
        entry  = states[mid]
        target = targets[mid]
        stale  = entry is None or (now - entry[3]) > FEED_TIMEOUT_S

        if stale:
            actual_str = '  -- no feedback --'
        else:
            pos = entry[0]
            err = pos - target
            marker = ' ✓' if abs(err) <= AT_TARGET_RAD else f'  Δ={err:+.3f} rad'
            actual_str = (f'  pos={pos:+.4f} rad ({math.degrees(pos):+.1f}°)'
                          f'  vel={entry[1]:+.3f} r/s{marker}')

        print(f'  {LABELS[mid]}:{actual_str}')
        print(f'    target = {target:+.4f} rad ({math.degrees(target):+.1f}°)')
    print()


def mirror_target(extension_rad):
    """
    Given an extension value (positive = down from hardstop):
        left foot  = +extension
        right foot = -extension
    """
    return {LEFT_ID: +extension_rad, RIGHT_ID: -extension_rad}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Interactive foot position controller for RS00 actuators')
    parser.add_argument('--interface', default='can1')
    parser.add_argument('--kp', type=float, default=40.0,
                        help='Position gain (default 40)')
    parser.add_argument('--kd', type=float, default=2.0,
                        help='Damping gain (default 2)')
    args = parser.parse_args()

    import can

    print(f'\nOpening {args.interface}...')
    try:
        bus = can.Bus(interface='socketcan', channel=args.interface, bitrate=1000000)
    except Exception as e:
        print(f'ERROR: {e}')
        print(f'  sudo ip link set {args.interface} up type can bitrate 1000000')
        return

    # Initial state and targets (hardstop = 0)
    states  = {mid: None for mid in RS00_IDS}
    targets = {LEFT_ID: 0.0, RIGHT_ID: 0.0}

    # Enable
    print('Enabling RS00 foot motors...')
    for mid in RS00_IDS:
        send(bus, frame_enable(mid))
        time.sleep(0.02)

    # Collect initial feedback
    print('Reading positions...')
    drain(bus, states, poll_s=0.5)

    # ── Header ──────────────────────────────────────────────────────────────
    print('\n' + '─' * 60)
    print('  RS00 Foot Position Control')
    print('  Zero = hardstop (fully retracted). Positive = extended down.')
    print('  Mirrored: left=+E, right=-E for equal height on flat surface.')
    print('─' * 60)
    print_status(states, targets)

    print('  Commands:')
    print('    e <deg>   — extend both feet (left=+deg, right=−deg)')
    print('    l <deg>   — set left foot only  (independent)')
    print('    r <deg>   — set right foot only (independent)')
    print('    h         — return both feet to hardstop (0°)')
    print('    s         — refresh position readout')
    print('    q         — quit and stop motors')
    print()
    print(f'  Gains: kp={args.kp}  kd={args.kd}  '
          f'(override: --kp <val> --kd <val>)')
    print()

    # Send initial hold at 0
    send_targets(bus, targets, args.kp, args.kd)

    try:
        while True:
            try:
                raw = input('  > ').strip().lower()
            except EOFError:
                break

            if not raw:
                continue

            parts = raw.split()
            cmd   = parts[0]

            # ── quit ────────────────────────────────────────────────────────
            if cmd == 'q':
                break

            # ── refresh ─────────────────────────────────────────────────────
            elif cmd == 's':
                drain(bus, states, poll_s=0.3)
                print_status(states, targets)

            # ── hardstop ────────────────────────────────────────────────────
            elif cmd == 'h':
                targets = {LEFT_ID: 0.0, RIGHT_ID: 0.0}
                print('  → Returning to hardstop (0.0 rad)...')
                _move_to(bus, states, targets, args.kp, args.kd)
                print_status(states, targets)

            # ── extend both (mirrored) ───────────────────────────────────────
            elif cmd == 'e':
                if len(parts) < 2:
                    print('  Usage: e <degrees>')
                    continue
                try:
                    deg = float(parts[1])
                except ValueError:
                    print('  ERROR: expected a number in degrees')
                    continue
                ext = math.radians(deg)
                targets = mirror_target(ext)
                print(f'  → Left={math.degrees(targets[LEFT_ID]):+.1f}°  '
                      f'Right={math.degrees(targets[RIGHT_ID]):+.1f}°')
                _move_to(bus, states, targets, args.kp, args.kd)
                print_status(states, targets)

            # ── individual left ──────────────────────────────────────────────
            elif cmd == 'l':
                if len(parts) < 2:
                    print('  Usage: l <degrees>')
                    continue
                try:
                    deg = float(parts[1])
                except ValueError:
                    print('  ERROR: expected a number in degrees')
                    continue
                targets[LEFT_ID] = math.radians(deg)
                print(f'  → Left foot → {deg:+.1f}°')
                _move_to(bus, states, targets, args.kp, args.kd)
                print_status(states, targets)

            # ── individual right ─────────────────────────────────────────────
            elif cmd == 'r':
                if len(parts) < 2:
                    print('  Usage: r <degrees>')
                    continue
                try:
                    deg = float(parts[1])
                except ValueError:
                    print('  ERROR: expected a number in degrees')
                    continue
                targets[RIGHT_ID] = math.radians(deg)
                print(f'  → Right foot → {deg:+.1f}°')
                _move_to(bus, states, targets, args.kp, args.kd)
                print_status(states, targets)

            else:
                print('  Unknown command. Use e / l / r / h / s / q.')

    except KeyboardInterrupt:
        print()

    # Stop
    print('Sending stop...')
    for mid in RS00_IDS:
        send(bus, frame_stop(mid))
        time.sleep(0.01)

    bus.shutdown()
    print('Done.')


def _move_to(bus, states, targets, kp, kd, timeout=5.0, print_hz=4):
    """
    Command motors toward targets, printing progress until both arrive
    or timeout is reached. Keeps sending Type 1 commands at CONTROL_HZ.
    """
    deadline   = time.monotonic() + timeout
    interval   = 1.0 / CONTROL_HZ
    print_dt   = 1.0 / print_hz
    last_print = 0.0

    while time.monotonic() < deadline:
        t0 = time.monotonic()

        send_targets(bus, targets, kp, kd)
        drain(bus, states, poll_s=interval * 0.8)

        now = time.monotonic()
        if now - last_print >= print_dt:
            last_print = now
            parts = []
            for mid in RS00_IDS:
                entry = states[mid]
                if entry is None:
                    parts.append(f'{LABELS[mid]}: --')
                else:
                    pos = entry[0]
                    tgt = targets[mid]
                    marker = '✓' if abs(pos - tgt) <= AT_TARGET_RAD else ' '
                    parts.append(f'{LABELS[mid]}: {pos:+.3f} rad ({math.degrees(pos):+.1f}°) [{marker}]')
            print('  ' + '    '.join(parts))

        # Done when both are within threshold
        all_arrived = all(
            states[mid] is not None
            and abs(states[mid][0] - targets[mid]) <= AT_TARGET_RAD
            for mid in RS00_IDS
        )
        if all_arrived:
            print('  Both feet at target.')
            return

        # Pace the loop
        elapsed = time.monotonic() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    print('  Timeout — stopped commanding. Check motor feedback.')


if __name__ == '__main__':
    main()
