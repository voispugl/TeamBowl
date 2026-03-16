"""
RobStride CAN Motor Monitor
===========================
Standalone live CAN sensor monitor. No ROS2. Only python-can, pyyaml, and stdlib.

Passively listens on one or more CAN buses and displays a live auto-refreshing
table of all RobStride motor feedback (Type 2 and Type 24 active-report frames).

Usage examples:
  # Monitor can0 with defaults
  python monitor.py

  # Monitor two buses with a config file
  python monitor.py --bus can0 --bus can1 --config config/motors.yaml

  # Send active-reporting enable before monitoring, refresh every 50 ms
  python monitor.py --bus can0 --enable-reporting --interval 0.05

  # Non-interactive pipeline use (plain output)
  python monitor.py --bus can0 | tee motor_log.txt

Bring up the CAN interface first:
  sudo ip link set can0 up type can bitrate 1000000
"""

import math
import struct
import threading
import time
import sys
import argparse
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import can

HOST_ID = 0xFD
POS_MIN, POS_MAX = -4 * math.pi, 4 * math.pi
MOTOR_RANGES = {
    'RS04': {'vel': (-15.0, 15.0),   'torque': (-120.0, 120.0)},
    'RS00': {'vel': (-33.0, 33.0),   'torque': (-14.0,   14.0)},
    'RS05': {'vel': (-50.0, 50.0),   'torque': (-5.5,    5.5)},
}
DEFAULT_RANGES = MOTOR_RANGES['RS04']
MODE_STRINGS = {0: 'Reset', 1: 'Cali', 2: 'Run', 3: '???'}
STALE_TIMEOUT = 2.0  # seconds before showing --- in table


@dataclass
class MotorRow:
    bus: str
    can_id: int
    joint: str = '?'
    motor_type: str = 'RS04'
    position: float = 0.0
    velocity: float = 0.0
    torque: float = 0.0
    temperature: float = 0.0
    mode: int = 0
    fault_uncalibrated: bool = False
    fault_overload: bool = False
    fault_encoder: bool = False
    fault_overtemp: bool = False
    fault_overcurrent: bool = False
    fault_undervoltage: bool = False
    last_seen: float = field(default_factory=time.monotonic)


def load_config(path: str) -> Dict[str, Dict[int, dict]]:
    """Load motors.yaml. Returns {bus_name: {can_id: {'joint': str, 'type': str}}}."""
    import yaml
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
        result = {}
        for joint_name, m in raw.get('motors', {}).items():
            bus = m['bus']
            cid_raw = m['can_id']
            cid = int(str(cid_raw), 0) if isinstance(cid_raw, str) else int(cid_raw)
            result.setdefault(bus, {})[cid] = {'joint': joint_name, 'type': m.get('type', 'RS04')}
        return result
    except Exception as e:
        print(f"[warn] Could not load config '{path}': {e}", file=sys.stderr)
        return {}


def raw_to_value(raw: int, lo: float, hi: float) -> float:
    return lo + raw / 65535.0 * (hi - lo)


def decode_feedback(can_id: int, data: bytes, vel_range: Tuple, torque_range: Tuple) -> dict:
    """Decode a Type 2 or Type 24 active-report frame into a dict of fields."""
    motor_id           = (can_id >> 8) & 0xFF
    mode               = (can_id >> 22) & 0x3
    fault_uncalibrated = bool((can_id >> 21) & 1)
    fault_overload     = bool((can_id >> 20) & 1)
    fault_encoder      = bool((can_id >> 19) & 1)
    fault_overtemp     = bool((can_id >> 18) & 1)
    fault_overcurrent  = bool((can_id >> 17) & 1)
    fault_undervoltage = bool((can_id >> 16) & 1)
    angle_raw, vel_raw, torque_raw = struct.unpack('>HHH', data[0:6])
    temp_raw = struct.unpack('>H', data[6:8])[0]
    return {
        'motor_id': motor_id,
        'position': raw_to_value(angle_raw, POS_MIN, POS_MAX),
        'velocity': raw_to_value(vel_raw, *vel_range),
        'torque':   raw_to_value(torque_raw, *torque_range),
        'temperature': temp_raw / 10.0,
        'mode': mode,
        'fault_uncalibrated': fault_uncalibrated,
        'fault_overload':     fault_overload,
        'fault_encoder':      fault_encoder,
        'fault_overtemp':     fault_overtemp,
        'fault_overcurrent':  fault_overcurrent,
        'fault_undervoltage': fault_undervoltage,
    }


def fault_str(row: MotorRow) -> str:
    faults = []
    if row.fault_uncalibrated: faults.append('UNCAL')
    if row.fault_overload:     faults.append('OVERLOAD')
    if row.fault_encoder:      faults.append('ENCODER')
    if row.fault_overtemp:     faults.append('OVERTEMP')
    if row.fault_overcurrent:  faults.append('OVERCURR')
    if row.fault_undervoltage: faults.append('UNDERVOLT')
    return ' '.join(faults) if faults else 'OK'


def has_fault(row: MotorRow) -> bool:
    return any([row.fault_uncalibrated, row.fault_overload, row.fault_encoder,
                row.fault_overtemp, row.fault_overcurrent, row.fault_undervoltage])


def send_enable_reporting(buses: dict):
    """Send Type 24 enable-active-reporting to IDs 0x01–0x20 on all buses."""
    for bus_name, bus in buses.items():
        for motor_id in range(1, 33):
            arb_id = (0x18 << 24) | (HOST_ID << 8) | motor_id
            data = b'\x01\x02\x03\x04\x05\x06\x01\x00'
            try:
                bus.send(can.Message(arbitration_id=arb_id, data=data, is_extended_id=True))
            except Exception:
                pass
        time.sleep(0.05)


def bus_listener(bus_name: str, bus: can.Bus, state: dict, lock: threading.Lock,
                 motor_info: dict, running_flag: list):
    """
    Daemon thread. Reads frames, decodes Type 2 (0x02) and Type 24 (0x18).
    Updates state[(bus_name, can_id)] = MotorRow.
    motor_info: {can_id: {'joint': str, 'type': str}} for this bus.
    running_flag: [True] — set to False to stop.
    """
    while running_flag[0]:
        try:
            msg = bus.recv(timeout=0.2)
        except Exception:
            break
        if msg is None or not msg.is_extended_id or len(msg.data) < 8:
            continue
        comm_type = (msg.arbitration_id >> 24) & 0x1F
        if comm_type not in (0x02, 0x18):
            continue
        # Get per-type ranges
        motor_id = (msg.arbitration_id >> 8) & 0xFF
        info = motor_info.get(motor_id, {})
        mtype = info.get('type', 'RS04')
        ranges = MOTOR_RANGES.get(mtype, DEFAULT_RANGES)
        decoded = decode_feedback(msg.arbitration_id, bytes(msg.data),
                                  ranges['vel'], ranges['torque'])
        key = (bus_name, motor_id)
        with lock:
            if key not in state:
                state[key] = MotorRow(bus=bus_name, can_id=motor_id,
                                      joint=info.get('joint', '?'),
                                      motor_type=mtype)
            row = state[key]
            row.position    = decoded['position']
            row.velocity    = decoded['velocity']
            row.torque      = decoded['torque']
            row.temperature = decoded['temperature']
            row.mode        = decoded['mode']
            row.fault_uncalibrated = decoded['fault_uncalibrated']
            row.fault_overload     = decoded['fault_overload']
            row.fault_encoder      = decoded['fault_encoder']
            row.fault_overtemp     = decoded['fault_overtemp']
            row.fault_overcurrent  = decoded['fault_overcurrent']
            row.fault_undervoltage = decoded['fault_undervoltage']
            row.last_seen   = time.monotonic()


def format_row(row: MotorRow, now: float) -> str:
    stale = (now - row.last_seen) > STALE_TIMEOUT
    if stale:
        return (f" {row.bus:<5} │ 0x{row.can_id:02X} │ {row.joint:<18} │"
                f"  {'---':>8} │ {'---':>8} │ {'---':>10} │ {'---':>8} │ {'---':<5} │ ---")
    temp_str = f"{row.temperature:>6.1f}"
    if row.temperature >= 100.0:
        temp_str += '!'
    else:
        temp_str += ' '
    return (f" {row.bus:<5} │ 0x{row.can_id:02X} │ {row.joint:<18} │"
            f" {row.position:>+8.4f} │ {row.velocity:>+8.3f} │ {row.torque:>+10.2f} │"
            f" {temp_str:<8} │ {MODE_STRINGS.get(row.mode,'?'):<5} │ {fault_str(row)}")


def run_curses(state: dict, lock: threading.Lock, interval: float, running_flag: list):
    import curses

    def draw(stdscr):
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)     # fault
        curses.init_pair(2, curses.COLOR_YELLOW, -1)  # warn
        curses.init_pair(3, curses.COLOR_GREEN, -1)   # ok
        stdscr.nodelay(True)
        header = (" Bus   │ ID   │ Joint              │ Pos(rad) │ Vel(r/s) │"
                  " Torque(Nm) │ Temp(°C) │ Mode  │ Faults")
        sep    = ("═══════╪══════╪════════════════════╪══════════╪══════════╪"
                  "════════════╪══════════╪═══════╪════════════")
        while running_flag[0]:
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            title = " RobStride CAN Monitor "
            stdscr.addstr(0, max(0, (w - len(title))//2), title, curses.A_BOLD)
            stdscr.addstr(1, 0, sep[:w-1])
            stdscr.addstr(2, 0, header[:w-1])
            stdscr.addstr(3, 0, sep[:w-1])
            now = time.monotonic()
            with lock:
                rows = sorted(state.values(), key=lambda r: (r.bus, r.can_id))
            line = 4
            last_update = 0.0
            for row in rows:
                if line >= h - 2:
                    break
                age = now - row.last_seen
                last_update = max(last_update, row.last_seen)
                text = format_row(row, now)[:w-1]
                attr = curses.A_NORMAL
                if has_fault(row):
                    attr = curses.color_pair(1) | curses.A_BOLD
                elif row.temperature >= 100.0:
                    attr = curses.color_pair(2)
                stdscr.addstr(line, 0, text, attr)
                line += 1
            stdscr.addstr(min(line+1, h-1), 0,
                f" Listening on {', '.join(sorted(set(r.bus for r in rows)))}   "
                f"Last frame: {now - last_update:.3f}s ago   Ctrl+C to quit"
            )
            stdscr.refresh()
            time.sleep(interval)
            key = stdscr.getch()
            if key == ord('q'):
                running_flag[0] = False

    try:
        curses.wrapper(draw)
    except KeyboardInterrupt:
        running_flag[0] = False


def run_plain(state: dict, lock: threading.Lock, interval: float, running_flag: list):
    header = (" Bus   │ ID   │ Joint              │ Pos(rad) │ Vel(r/s) │"
              " Torque(Nm) │ Temp(°C) │ Mode  │ Faults")
    sep    = ("─" * len(header))
    try:
        while running_flag[0]:
            now = time.monotonic()
            if sys.stdout.isatty():
                print('\033[2J\033[H', end='')  # clear screen
            print(sep)
            print(header)
            print(sep)
            with lock:
                rows = sorted(state.values(), key=lambda r: (r.bus, r.can_id))
            for row in rows:
                print(format_row(row, now))
            print(sep)
            bus_names = sorted(set(r.bus for r in rows)) if rows else ['—']
            print(f" Listening on: {', '.join(bus_names)}   Ctrl+C to quit")
            time.sleep(interval)
    except KeyboardInterrupt:
        running_flag[0] = False


def main():
    parser = argparse.ArgumentParser(
        description='Live RobStride CAN motor monitor.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--bus', action='append', default=[], metavar='IFACE',
                        help='CAN interface (repeatable, default: can0)')
    parser.add_argument('--config', metavar='PATH',
                        help='Path to motors.yaml for joint names and motor types')
    parser.add_argument('--bitrate', type=int, default=1000000, metavar='BPS')
    parser.add_argument('--enable-reporting', action='store_true',
                        help='Send Type 24 active-report enable to IDs 0x01-0x20 before monitoring')
    parser.add_argument('--interval', type=float, default=0.1, metavar='SEC',
                        help='Display refresh interval (default: 0.1)')
    args = parser.parse_args()

    bus_names = args.bus if args.bus else ['can0']

    # Load config
    motor_info_all = {}
    if args.config:
        motor_info_all = load_config(args.config)

    # Open buses
    buses = {}
    for name in bus_names:
        try:
            buses[name] = can.Bus(interface='socketcan', channel=name, bitrate=args.bitrate)
            print(f"Opened {name}")
        except Exception as e:
            print(f"[warn] Cannot open {name}: {e}")
            print(f"       Try: sudo ip link set {name} up type can bitrate {args.bitrate}")

    if not buses:
        print("No CAN buses available. Exiting.")
        sys.exit(1)

    if args.enable_reporting:
        print("Sending active-reporting enable to all motors...")
        send_enable_reporting(buses)

    state: dict = {}
    lock = threading.Lock()
    running_flag = [True]

    # Start listener threads
    threads = []
    for name, bus in buses.items():
        motor_info = motor_info_all.get(name, {})
        t = threading.Thread(target=bus_listener,
                             args=(name, bus, state, lock, motor_info, running_flag),
                             daemon=True)
        t.start()
        threads.append(t)

    print(f"Monitoring {', '.join(buses.keys())}... (Ctrl+C to quit)")
    time.sleep(0.3)  # brief pause to collect initial frames

    # Try curses display, fall back to plain
    use_curses = sys.stdout.isatty()
    if use_curses:
        try:
            import curses as _curses  # noqa: just checking import
            run_curses(state, lock, args.interval, running_flag)
        except Exception:
            use_curses = False

    if not use_curses:
        try:
            run_plain(state, lock, args.interval, running_flag)
        except KeyboardInterrupt:
            pass

    running_flag[0] = False
    for bus in buses.values():
        try:
            bus.shutdown()
        except Exception:
            pass
    print("\nMonitor stopped.")


if __name__ == '__main__':
    main()
