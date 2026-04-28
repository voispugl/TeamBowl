#!/usr/bin/env python3
"""
check_vesc_can.py

Pings VESC controllers on a SocketCAN bus and reports which ones respond.

Usage:
    python3 check_vesc_can.py [--interface can1] [--ids 14 24]

Requires: python-can  (pip install python-can  or  apt install python3-can)
Can bus must already be up:
    sudo ip link set can1 type can bitrate 1000000
    sudo ip link set can1 up
"""

import argparse
import struct
import time
import can

CAN_PACKET_PING = 17
CAN_PACKET_PONG = 18
PING_TIMEOUT_S  = 0.25   # wait up to 250 ms per ping for a pong


def ping_vesc(bus: can.Bus, unit_id: int) -> bool:
    """Send a PING to unit_id and return True if a PONG arrives within PING_TIMEOUT_S."""
    ping_id = (CAN_PACKET_PING << 8) | unit_id
    pong_id = (CAN_PACKET_PONG << 8) | unit_id

    bus.send(can.Message(arbitration_id=ping_id, data=b'', is_extended_id=True))

    deadline = time.monotonic() + PING_TIMEOUT_S
    while time.monotonic() < deadline:
        msg = bus.recv(timeout=max(0.0, deadline - time.monotonic()))
        if msg is None:
            break
        if msg.arbitration_id == pong_id:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Ping VESCs on a CAN bus')
    parser.add_argument('--interface', default='can1', help='SocketCAN interface (default: can1)')
    parser.add_argument('--ids', nargs='+', type=int, default=[14, 24],
                        help='VESC unit IDs to ping (default: 14 24)')
    args = parser.parse_args()

    labels = {14: 'left', 24: 'right'}

    print(f'Opening {args.interface}...')
    try:
        bus = can.Bus(interface='socketcan', channel=args.interface, bitrate=1000000)
    except Exception as e:
        print(f'ERROR: could not open {args.interface}: {e}')
        print('Make sure the bus is up:')
        print(f'  sudo ip link set {args.interface} type can bitrate 1000000')
        print(f'  sudo ip link set {args.interface} up')
        return

    print(f'Pinging VESC IDs: {args.ids}\n')
    results = {}
    for uid in args.ids:
        label = labels.get(uid, f'id{uid}')
        responded = ping_vesc(bus, uid)
        results[uid] = responded
        status = 'OK  ✓' if responded else 'NO RESPONSE'
        print(f'  VESC {uid:3d} ({label:>6s})  {status}')

    bus.shutdown()

    print()
    if all(results.values()):
        print('All VESCs responding.')
    elif any(results.values()):
        missing = [uid for uid, ok in results.items() if not ok]
        print(f'WARNING: no response from VESC(s): {missing}')
        print('Check: CAN ID set in VESC Tool, CAN baud = 1000 kbps, VESC powered.')
    else:
        print('No VESCs responded. Check CAN wiring and VESC Tool configuration.')
        print('See VESC_CAN_SETUP.md for setup steps.')


if __name__ == '__main__':
    main()
