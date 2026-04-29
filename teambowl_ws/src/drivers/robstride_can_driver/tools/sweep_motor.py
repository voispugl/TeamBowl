"""
sweep_motor.py — Slow motor range sweep for finding physical limits.

Moves a single motor from its current position toward a target in small steps,
printing position / velocity / torque / fault at each step. Stops immediately
on any hardware fault or on Ctrl+C. Useful for finding the usable range of a
motor before hitting the hardstop and triggering an overcurrent fault.

No ROS2 required. Requires python-can and the CAN interface up at 1 Mbps.

Usage:
    python3 sweep_motor.py --bus can1 --motor-id 0x17 --motor-type RS00 \\
                           --target -2.1817 --step 0.05 --delay 0.3

    sudo ip link set can1 up type can bitrate 1000000

Motor type ranges:
    RS04: pos ±4π, vel ±15, kp 0–5000, kd 0–100, torque ±120 Nm
    RS00: pos ±4π, vel ±33, kp 0–500,  kd 0–5,   torque ±14  Nm
    RS05: pos ±4π, vel ±50, kp 0–500,  kd 0–5,   torque ±5.5 Nm
"""

import argparse
import math
import struct
import time

HOST_ID = 0x01

# Physical ranges per motor type
MOTOR_SPECS = {
    'RS04': dict(pos=(-4*math.pi, 4*math.pi), vel=(-15.0, 15.0),
                 kp=(0.0, 5000.0), kd=(0.0, 100.0), torque=(-120.0, 120.0)),
    'RS00': dict(pos=(-4*math.pi, 4*math.pi), vel=(-33.0, 33.0),
                 kp=(0.0, 500.0),  kd=(0.0, 5.0),   torque=(-14.0, 14.0)),
    'RS05': dict(pos=(-4*math.pi, 4*math.pi), vel=(-50.0, 50.0),
                 kp=(0.0, 500.0),  kd=(0.0, 5.0),   torque=(-5.5, 5.5)),
}

PARAM_MECH_POS = 0x7019


# ---------------------------------------------------------------------------
# Frame builders (inlined — no package imports needed)
# ---------------------------------------------------------------------------

def _scale(value, lo, hi, bits=16):
    clamped = max(lo, min(hi, value))
    return int((clamped - lo) / (hi - lo) * (2 ** bits - 1))


def _arb(comm_type, data_area2, motor_id):
    return (comm_type << 24) | (data_area2 << 8) | motor_id


def frame_enable(motor_id):
    return dict(arbitration_id=_arb(0x03, HOST_ID, motor_id),
                data=bytes(8), is_extended_id=True)


def frame_stop(motor_id, clear_fault=False):
    flag = 1 if clear_fault else 0
    return dict(arbitration_id=_arb(0x04, (HOST_ID << 8) | flag, motor_id),
                data=bytes(8), is_extended_id=True)


def frame_motion(motor_id, pos_rad, kp, kd, torque_ff, spec):
    torque_raw = _scale(torque_ff, *spec['torque'])
    arb_id = _arb(0x01, torque_raw, motor_id)
    data = struct.pack('>HHHH',
                       _scale(pos_rad, *spec['pos']),
                       _scale(0.0,     *spec['vel']),
                       _scale(kp,      *spec['kp']),
                       _scale(kd,      *spec['kd']))
    return dict(arbitration_id=arb_id, data=data, is_extended_id=True)


def frame_param_read(motor_id, param_index):
    arb_id = _arb(0x11, HOST_ID, motor_id)
    data = struct.pack('<HH', param_index, 0)
    return dict(arbitration_id=arb_id, data=data + bytes(4), is_extended_id=True)


# ---------------------------------------------------------------------------
# Feedback decode
# ---------------------------------------------------------------------------

def _raw_to(raw, lo, hi, bits=16):
    return lo + raw / (2 ** bits - 1) * (hi - lo)


def decode_feedback(msg, spec):
    """Decode Type 2 feedback. Returns dict or None."""
    comm = (msg.arbitration_id >> 24) & 0x1F
    if comm not in (0x02, 0x18):
        return None
    if len(msg.data) < 8:
        return None

    motor_id = (msg.arbitration_id >> 8) & 0xFF
    fault_bits = (msg.arbitration_id >> 16) & 0x3F

    pos  = _raw_to(struct.unpack_from('>H', msg.data, 0)[0], *spec['pos'])
    vel  = _raw_to(struct.unpack_from('>H', msg.data, 2)[0], *spec['vel'])
    torq = _raw_to(struct.unpack_from('>H', msg.data, 4)[0], *spec['torque'])

    faults = {
        'uncalibrated': bool(fault_bits & (1 << 0)),
        'overload':      bool(fault_bits & (1 << 3)),
        'encoder':       bool(fault_bits & (1 << 4)),
        'overtemp':      bool(fault_bits & (1 << 5)),
        'overcurrent':   bool(fault_bits & (1 << 1)),
        'undervoltage':  bool(fault_bits & (1 << 2)),
    }
    any_fault = any(faults.values())

    return dict(motor_id=motor_id, pos=pos, vel=vel, torque=torq,
                faults=faults, any_fault=any_fault)


def decode_param_reply(msg):
    """Decode Type 17 param reply. Returns (motor_id, value_bytes) or None."""
    if (msg.arbitration_id >> 24) & 0x1F != 0x11:
        return None
    motor_id = (msg.arbitration_id >> 8) & 0xFF
    if len(msg.data) < 8:
        return None
    value_bytes = msg.data[4:8]
    return motor_id, value_bytes


# ---------------------------------------------------------------------------
# Bus helpers
# ---------------------------------------------------------------------------

def send(bus, frame):
    import can
    bus.send(can.Message(**frame))


def recv_feedback(bus, motor_id, spec, timeout=0.5):
    """Block until feedback from motor_id arrives or timeout. Returns decoded dict or None."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        msg = bus.recv(timeout=max(0.0, remaining))
        if msg is None:
            break
        fb = decode_feedback(msg, spec)
        if fb and fb['motor_id'] == motor_id:
            return fb
    return None


def read_mech_pos(bus, motor_id, spec, retries=3):
    """Read current mechanical position via Type 17 param read. Returns float or None."""
    import can
    for _ in range(retries):
        send(bus, frame_param_read(motor_id, PARAM_MECH_POS))
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            msg = bus.recv(timeout=max(0.0, remaining))
            if msg is None:
                break
            r = decode_param_reply(msg)
            if r and r[0] == motor_id:
                return struct.unpack('<f', r[1])[0]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Slow motor sweep to find physical range limits')
    parser.add_argument('--bus',        default='can1',
                        help='CAN interface (default: can1)')
    parser.add_argument('--motor-id',   required=True,
                        help='Motor CAN ID in hex (e.g. 0x17)')
    parser.add_argument('--motor-type', default='RS00', choices=list(MOTOR_SPECS),
                        help='Motor type for scaling ranges (default: RS00)')
    parser.add_argument('--target',     type=float, required=True,
                        help='Target position in radians')
    parser.add_argument('--step',       type=float, default=0.05,
                        help='Step size in radians per iteration (default: 0.05)')
    parser.add_argument('--delay',      type=float, default=0.3,
                        help='Seconds to wait and observe after each step (default: 0.3)')
    parser.add_argument('--kp',         type=float, default=20.0,
                        help='Position gain (default: 20)')
    parser.add_argument('--kd',         type=float, default=0.3,
                        help='Damping gain (default: 0.3)')
    args = parser.parse_args()

    motor_id = int(args.motor_id, 0)
    spec = MOTOR_SPECS[args.motor_type]

    import can

    print(f'\nOpening {args.bus}...')
    try:
        bus = can.Bus(interface='socketcan', channel=args.bus, bitrate=1000000)
    except Exception as e:
        print(f'ERROR: {e}')
        print(f'  sudo ip link set {args.bus} up type can bitrate 1000000')
        return

    print(f'Motor: {args.motor_type} CAN ID 0x{motor_id:02X} on {args.bus}')
    print(f'Target: {args.target:.4f} rad ({math.degrees(args.target):.1f}°)')
    print(f'Step: {args.step:.3f} rad  Delay: {args.delay:.2f}s  kp={args.kp}  kd={args.kd}')

    # Enable motor
    print('\nEnabling motor...')
    send(bus, frame_enable(motor_id))
    time.sleep(0.1)

    # Read current position
    print('Reading current position...')
    current = read_mech_pos(bus, motor_id, spec)
    if current is None:
        fb = recv_feedback(bus, motor_id, spec, timeout=1.0)
        current = fb['pos'] if fb else 0.0
        print(f'  (mechPos read failed — using feedback pos: {current:.4f} rad)')
    else:
        print(f'  mechPos = {current:.4f} rad ({math.degrees(current):.1f}°)')

    direction = 1.0 if args.target > current else -1.0
    step = abs(args.step) * direction

    print('\n' + '─' * 72)
    print(f'  {"Step":>4}  {"Target":>10}  {"Actual":>10}  {"Err":>8}  '
          f'{"Torque":>8}  {"Vel":>7}  Faults')
    print('─' * 72)

    commanded = current
    step_num = 0
    stopped_at = None

    try:
        while True:
            # Advance commanded position one step
            remaining = args.target - commanded
            if abs(remaining) <= abs(step):
                commanded = args.target
                at_target = True
            else:
                commanded += step
                at_target = False

            step_num += 1

            # Send command
            send(bus, frame_motion(motor_id, commanded, args.kp, args.kd, 0.0, spec))

            # Wait and collect feedback
            time.sleep(args.delay)

            fb = recv_feedback(bus, motor_id, spec, timeout=0.3)
            if fb is None:
                print(f'  {step_num:>4}  {commanded:>+10.4f}  {"NO FEEDBACK":>10}')
            else:
                pos   = fb['pos']
                torq  = fb['torque']
                vel   = fb['vel']
                err   = pos - commanded
                fault_names = [k for k, v in fb['faults'].items() if v]
                fault_str = ', '.join(fault_names).upper() if fault_names else 'OK'

                print(f'  {step_num:>4}  {commanded:>+10.4f}  {pos:>+10.4f}  '
                      f'{err:>+8.4f}  {torq:>+8.3f}N  {vel:>+7.3f}  {fault_str}')

                if fb['any_fault']:
                    stopped_at = pos
                    print(f'\n  *** FAULT at {pos:.4f} rad ({math.degrees(pos):.1f}°) ***')
                    print(f'      Faults: {fault_str}')
                    print(f'      Last safe position was ~{commanded - step:.4f} rad '
                          f'({math.degrees(commanded - step):.1f}°)')
                    break

            if at_target:
                stopped_at = commanded
                print(f'\n  Reached target {args.target:.4f} rad ({math.degrees(args.target):.1f}°).')
                break

    except KeyboardInterrupt:
        print('\n  Interrupted by user.')
        if fb:
            stopped_at = fb['pos']

    print('─' * 72)

    if stopped_at is not None:
        print(f'\nStopped at: {stopped_at:.4f} rad ({math.degrees(stopped_at):.1f}°)')

    # Stop motor
    print('Sending stop...')
    send(bus, frame_stop(motor_id))
    time.sleep(0.05)
    bus.shutdown()
    print('Done.')


if __name__ == '__main__':
    main()
