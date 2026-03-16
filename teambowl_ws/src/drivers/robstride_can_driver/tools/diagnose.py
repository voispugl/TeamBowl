"""
diagnose.py — Pre-flight diagnostic tool for RobStride motors.

Runs a series of automated checks before launching the ROS2 driver and prints a
structured PASS/WARN/FAIL report. Exits with code 1 if any check FAILs.

No ROS2 required — only `python-can`, `pyyaml`, and stdlib.

Usage examples:
  # Bring up CAN interfaces first:
  #   sudo ip link set can0 up type can bitrate 1000000
  #   sudo ip link set can1 up type can bitrate 1000000

  python diagnose.py
  python diagnose.py --bus can0
  python diagnose.py --bus can0 --bus can1 --config config/motors.yaml
  python diagnose.py --bus can0 --config config/motors.yaml --no-enable
  python diagnose.py --bus can0 --scan-range 1-16 --temp-warn 100.0
  python diagnose.py --bus can0 --bitrate 500000

Checks performed:
  Check 1  — CAN bus accessible (one per bus)
  Check 2  — Motor scan: count of responding motors
  Check 3  — Duplicate motor IDs on the same bus
  Check 4  — Duplicate UIDs at different motor IDs
  Check 5  — Expected motors present (requires --config)
  Check 6  — Unexpected motors found (requires --config)
  Check 7  — Hardware fault bits on enabled motor
  Check 8  — Encoder uncalibrated flag
  Check 9  — Motor temperature warning threshold
  Check 10 — Same CAN ID appears on multiple buses (requires multiple --bus args)
  Check 11 — CAN error frames observed during scan
"""

import argparse
import math
import struct
import sys
from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Color / output helpers
# ---------------------------------------------------------------------------

GREEN  = '\033[32m'  if sys.stdout.isatty() else ''
YELLOW = '\033[33m'  if sys.stdout.isatty() else ''
RED    = '\033[1;31m' if sys.stdout.isatty() else ''
RESET  = '\033[0m'   if sys.stdout.isatty() else ''

ICONS: dict  # forward declaration — filled after Status is defined


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class Status(str, Enum):
    PASS = 'PASS'
    WARN = 'WARN'
    FAIL = 'FAIL'


ICONS = {
    Status.PASS: '[PASS]',
    Status.WARN: '[WARN]',
    Status.FAIL: '[FAIL]',
}


@dataclass
class CheckResult:
    status: Status
    message: str


# ---------------------------------------------------------------------------
# Protocol constants  (inline — do NOT import from the package)
# ---------------------------------------------------------------------------

HOST_ID = 0xFD

FAULT_NAMES = {
    21: 'UNCALIBRATED',
    20: 'OVERLOAD',
    19: 'ENCODER_FAULT',
    18: 'OVERTEMP',
    17: 'OVERCURRENT',
    16: 'UNDERVOLTAGE',
}


def _build_can_id(msg_type: int, motor_id: int) -> int:
    return (msg_type << 24) | (HOST_ID << 8) | motor_id


def _make_msg(bus, arb_id: int, data: bytes):
    """Return a can.Message ready to send."""
    import can
    return can.Message(arbitration_id=arb_id, data=data, is_extended_id=True)


def _send_type0(bus, motor_id: int):
    """Type 0 — get_device_id / scan ping."""
    arb_id = _build_can_id(0x00, motor_id)
    bus.send(_make_msg(bus, arb_id, bytes(8)))


def _send_type3(bus, motor_id: int):
    """Type 3 — enable."""
    arb_id = _build_can_id(0x03, motor_id)
    bus.send(_make_msg(bus, arb_id, bytes(8)))


def _send_type4(bus, motor_id: int):
    """Type 4 — stop (Byte0=0x00, hold position, no fault clear)."""
    arb_id = _build_can_id(0x04, motor_id)
    bus.send(_make_msg(bus, arb_id, bytes([0x00]) + bytes(7)))


def _decode_type2_feedback(msg):
    """
    Decode a Type 2 feedback frame.

    Returns a dict with keys:
      comm_type, motor_id, mode,
      fault_uncalibrated, fault_overload, fault_encoder,
      fault_overtemp, fault_overcurrent, fault_undervoltage,
      temperature, position
    or None if the frame is not a valid Type 2 reply.
    """
    can_id = msg.arbitration_id
    comm_type = (can_id >> 24) & 0x1F
    if comm_type != 0x02:
        return None
    if len(msg.data) < 8:
        return None

    motor_id           = (can_id >> 8)  & 0xFF
    mode               = (can_id >> 22) & 0x3
    fault_uncalibrated = bool((can_id >> 21) & 1)
    fault_overload     = bool((can_id >> 20) & 1)
    fault_encoder      = bool((can_id >> 19) & 1)
    fault_overtemp     = bool((can_id >> 18) & 1)
    fault_overcurrent  = bool((can_id >> 17) & 1)
    fault_undervoltage = bool((can_id >> 16) & 1)

    angle_raw, vel_raw, torque_raw = struct.unpack('>HHH', bytes(msg.data[0:6]))
    temp_raw = struct.unpack('>H', bytes(msg.data[6:8]))[0]
    temperature = temp_raw / 10.0
    position = -4 * math.pi + angle_raw / 65535.0 * 8 * math.pi

    return dict(
        comm_type=comm_type,
        motor_id=motor_id,
        mode=mode,
        fault_uncalibrated=fault_uncalibrated,
        fault_overload=fault_overload,
        fault_encoder=fault_encoder,
        fault_overtemp=fault_overtemp,
        fault_overcurrent=fault_overcurrent,
        fault_undervoltage=fault_undervoltage,
        temperature=temperature,
        position=position,
    )


# ---------------------------------------------------------------------------
# Check 1: Bus accessible
# ---------------------------------------------------------------------------

def check_bus_open(bus_name: str, bitrate: int):
    """
    Try to open the named SocketCAN interface.

    Returns (bus_or_None, CheckResult).
    """
    try:
        import can
        bus = can.Bus(interface='socketcan', channel=bus_name, bitrate=bitrate)
        return bus, CheckResult(Status.PASS, f"Bus {bus_name} accessible")
    except Exception as e:
        hint = f"sudo ip link set {bus_name} up type can bitrate {bitrate}"
        return None, CheckResult(
            Status.FAIL,
            f"Cannot open {bus_name}: {e}. Try: {hint}",
        )


# ---------------------------------------------------------------------------
# Checks 2, 3, 4, 11: Motor scan + duplicate detection + error frames
# ---------------------------------------------------------------------------

def check_scan(bus, bus_name: str, scan_ids):
    """
    Scan the bus for responding motors.

    Sends Type 0 (get_device_id) to each ID in scan_ids and collects replies.

    Returns:
      found         — dict mapping motor_id (int) -> uid_bytes (bytes)
      results       — list[CheckResult] covering checks 2, 3, 4
      error_count   — int, number of CAN error frames observed (for check 11)
    """
    found: dict[int, bytes] = {}   # motor_id -> uid bytes
    uid_map: dict[bytes, int] = {} # uid bytes -> first motor_id that reported it
    error_count = 0

    for motor_id in scan_ids:
        try:
            _send_type0(bus, motor_id)
        except Exception:
            continue

        msg = bus.recv(timeout=0.05)
        if msg is None:
            continue

        if msg.is_error_frame:
            error_count += 1
            continue

        can_id = msg.arbitration_id
        # Type 0 reply: bits 7-0 == 0xFE (broadcast reply marker)
        if (can_id & 0xFF) == 0xFE:
            replied_motor_id = (can_id >> 8) & 0xFF
            uid_bytes = bytes(msg.data[0:8])
            if replied_motor_id in found:
                # duplicate ID — will be caught in check 3
                pass
            found[replied_motor_id] = uid_bytes
            uid_map.setdefault(uid_bytes, replied_motor_id)

    results: list[CheckResult] = []

    # Check 2: found count
    if found:
        results.append(CheckResult(
            Status.PASS,
            f"{bus_name}: {len(found)} motor(s) found: "
            + ", ".join(f"0x{mid:02X}" for mid in sorted(found)),
        ))
    else:
        results.append(CheckResult(
            Status.WARN,
            f"{bus_name}: No motors responded — check power and cabling",
        ))

    # Check 3: duplicate motor IDs
    # (If the same motor_id replied twice during the sweep it overwrote the
    # earlier entry — we detect this by re-scanning for collisions differently.
    # The dict naturally keeps only the last, so we re-scan with a list.)
    seen_ids: list[int] = []
    dup_ids: list[int] = []

    # Re-run a lightweight duplicate pass using a fresh list
    id_counts: dict[int, int] = {}
    for motor_id in scan_ids:
        try:
            _send_type0(bus, motor_id)
        except Exception:
            continue
        msg = bus.recv(timeout=0.05)
        if msg is None:
            continue
        if msg.is_error_frame:
            error_count += 1
            continue
        can_id = msg.arbitration_id
        if (can_id & 0xFF) == 0xFE:
            replied_motor_id = (can_id >> 8) & 0xFF
            id_counts[replied_motor_id] = id_counts.get(replied_motor_id, 0) + 1
            # Update found with latest data
            found[replied_motor_id] = bytes(msg.data[0:8])

    dup_ids = [mid for mid, cnt in id_counts.items() if cnt > 1]
    if dup_ids:
        dup_list = ", ".join(f"0x{mid:02X}" for mid in sorted(dup_ids))
        results.append(CheckResult(
            Status.FAIL,
            f"{bus_name}: Duplicate motor ID(s) detected: {dup_list} — "
            "two motors sharing a CAN ID will cause communication errors",
        ))
    else:
        if found:
            results.append(CheckResult(
                Status.PASS,
                f"{bus_name}: No duplicate motor IDs",
            ))

    # Check 4: same UID at two different motor IDs
    uid_to_ids: dict[bytes, list[int]] = {}
    for mid, uid in found.items():
        uid_to_ids.setdefault(uid, []).append(mid)
    uid_dups = {uid: ids for uid, ids in uid_to_ids.items() if len(ids) > 1}
    if uid_dups:
        for uid, ids in uid_dups.items():
            id_list = ", ".join(f"0x{mid:02X}" for mid in sorted(ids))
            uid_hex = uid.hex().upper()
            results.append(CheckResult(
                Status.FAIL,
                f"{bus_name}: Same UID {uid_hex} seen at motor IDs {id_list} — "
                "likely a physically moved motor that was not re-flashed",
            ))
    else:
        if found:
            results.append(CheckResult(
                Status.PASS,
                f"{bus_name}: No duplicate UIDs across motor IDs",
            ))

    return found, results, error_count


# ---------------------------------------------------------------------------
# Checks 5, 6: Expected vs found (requires config)
# ---------------------------------------------------------------------------

def check_expected_motors(
    found_ids: set,
    bus_name: str,
    config: dict,
) -> list[CheckResult]:
    """
    Compare the set of responding motor IDs against the motors.yaml config.

    config: {bus_name: {can_id: joint_name}}

    Check 5: missing motors   → FAIL
    Check 6: unexpected motors → WARN
    """
    results: list[CheckResult] = []
    bus_cfg = config.get(bus_name, {})
    expected = set(bus_cfg.keys())

    missing = expected - found_ids
    unexpected = found_ids - expected

    # Check 5
    if missing:
        names = ", ".join(
            f"0x{mid:02X} ({bus_cfg[mid]})" for mid in sorted(missing)
        )
        results.append(CheckResult(
            Status.FAIL,
            f"{bus_name}: Expected motor(s) did not respond: {names}",
        ))
    else:
        if expected:
            results.append(CheckResult(
                Status.PASS,
                f"{bus_name}: All {len(expected)} expected motor(s) responded",
            ))

    # Check 6
    if unexpected:
        ids = ", ".join(f"0x{mid:02X}" for mid in sorted(unexpected))
        results.append(CheckResult(
            Status.WARN,
            f"{bus_name}: Unexpected motor(s) responded (not in config): {ids} "
            "— may be a newly added motor or wrong bus",
        ))

    return results


# ---------------------------------------------------------------------------
# Checks 7, 8, 9: Enable and read feedback
# ---------------------------------------------------------------------------

def check_motor_health(
    bus,
    bus_name: str,
    found_ids: list,
    temp_warn_threshold: float,
) -> list[CheckResult]:
    """
    Enable each found motor, read feedback, then immediately stop it.

    Check 7: any hardware fault bits → FAIL
    Check 8: encoder uncalibrated   → FAIL
    Check 9: temperature too high   → WARN
    """
    results: list[CheckResult] = []

    for motor_id in sorted(found_ids):
        # Enable
        try:
            _send_type3(bus, motor_id)
        except Exception as e:
            results.append(CheckResult(
                Status.FAIL,
                f"{bus_name} motor 0x{motor_id:02X}: Failed to send enable: {e}",
            ))
            continue

        # Wait for Type 2 feedback (up to 200 ms)
        fb = None
        deadline = 0.20
        elapsed = 0.0
        step = 0.01
        while elapsed < deadline:
            msg = bus.recv(timeout=step)
            elapsed += step
            if msg is None:
                continue
            if msg.is_error_frame:
                continue
            decoded = _decode_type2_feedback(msg)
            if decoded and decoded['motor_id'] == motor_id:
                fb = decoded
                break

        # Always stop the motor
        try:
            _send_type4(bus, motor_id)
        except Exception:
            pass

        if fb is None:
            results.append(CheckResult(
                Status.WARN,
                f"{bus_name} motor 0x{motor_id:02X}: No Type 2 feedback received after enable",
            ))
            continue

        # Check 7 + 8: fault bits
        active_faults = []
        if fb['fault_overload']:
            active_faults.append('OVERLOAD')
        if fb['fault_encoder']:
            active_faults.append('ENCODER_FAULT')
        if fb['fault_overtemp']:
            active_faults.append('OVERTEMP')
        if fb['fault_overcurrent']:
            active_faults.append('OVERCURRENT')
        if fb['fault_undervoltage']:
            active_faults.append('UNDERVOLTAGE')

        if fb['fault_uncalibrated']:
            results.append(CheckResult(
                Status.FAIL,
                f"{bus_name} motor 0x{motor_id:02X}: Encoder uncalibrated — "
                "run commissioning.py set-zero before operating",
            ))
        elif active_faults:
            results.append(CheckResult(
                Status.FAIL,
                f"{bus_name} motor 0x{motor_id:02X}: Hardware fault(s): "
                + ", ".join(active_faults),
            ))
        else:
            results.append(CheckResult(
                Status.PASS,
                f"{bus_name} motor 0x{motor_id:02X}: No faults "
                f"(temp={fb['temperature']:.1f}\u00b0C, "
                f"pos={fb['position']:.3f} rad)",
            ))

        # Check 9: temperature
        if fb['temperature'] > temp_warn_threshold:
            results.append(CheckResult(
                Status.WARN,
                f"{bus_name} motor 0x{motor_id:02X}: Temperature {fb['temperature']:.1f}\u00b0C "
                f"exceeds warning threshold {temp_warn_threshold:.1f}\u00b0C",
            ))

    return results


# ---------------------------------------------------------------------------
# Check 10: Cross-bus ID shadowing
# ---------------------------------------------------------------------------

def check_cross_bus_ids(found_per_bus: dict) -> list[CheckResult]:
    """
    Detect the same CAN ID appearing on more than one bus.

    This is technically valid (separate buses are independent) but worth
    flagging in case the user mis-connected a motor.

    found_per_bus: {bus_name: set_of_found_ids}
    """
    results: list[CheckResult] = []
    bus_names = list(found_per_bus.keys())

    # Collect all IDs and which buses they appear on
    id_to_buses: dict[int, list[str]] = {}
    for bus_name, ids in found_per_bus.items():
        for mid in ids:
            id_to_buses.setdefault(mid, []).append(bus_name)

    shared = {mid: buses for mid, buses in id_to_buses.items() if len(buses) > 1}
    if shared:
        for mid, buses in sorted(shared.items()):
            bus_list = ", ".join(buses)
            results.append(CheckResult(
                Status.WARN,
                f"Motor ID 0x{mid:02X} found on multiple buses: {bus_list} — "
                "verify correct physical connections (separate buses are independent "
                "but the same ID on both may indicate a mis-connected motor)",
            ))
    else:
        results.append(CheckResult(
            Status.PASS,
            "No CAN IDs shared across buses",
        ))

    return results


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_motor_config(path: str):
    """
    Load motors.yaml and return a nested dict: {bus_name: {can_id: joint_name}}.

    Returns None on failure (prints a warning and continues).
    """
    try:
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        result: dict[str, dict[int, str]] = {}
        for joint_name, m in raw.get('motors', {}).items():
            bus = m['bus']
            can_id_raw = m['can_id']
            can_id = (
                int(str(can_id_raw), 0)
                if isinstance(can_id_raw, str)
                else int(can_id_raw)
            )
            result.setdefault(bus, {})[can_id] = joint_name
        return result
    except Exception as e:
        print(f"[WARN] Could not load config '{path}': {e}")
        return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_scan_range(range_str: str) -> range:
    """Convert '1-32' to range(1, 33)."""
    try:
        parts = range_str.split('-')
        if len(parts) != 2:
            raise ValueError
        lo, hi = int(parts[0]), int(parts[1])
        return range(lo, hi + 1)
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid scan range '{range_str}' — expected format: 1-32"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        prog='diagnose.py',
        description=(
            'Pre-flight diagnostic tool for RobStride motors.\n'
            'Runs automated checks and exits with code 1 if any check FAILs.\n'
            'No ROS2 required — only python-can, pyyaml, and stdlib.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--bus', dest='bus', action='append', metavar='INTERFACE',
        help='CAN interface(s) to check (repeatable, default: can0 can1)',
    )
    parser.add_argument(
        '--config', metavar='PATH',
        help='Path to motors.yaml; enables expected/unexpected motor checks',
    )
    parser.add_argument(
        '--bitrate', type=int, default=1_000_000, metavar='BPS',
        help='CAN bus bitrate in bps (default: 1000000)',
    )
    parser.add_argument(
        '--scan-range', default='1-32', metavar='LO-HI',
        help='Range of CAN IDs to scan (default: 1-32)',
    )
    parser.add_argument(
        '--temp-warn', type=float, default=115.0, metavar='CELSIUS',
        help='Temperature warning threshold in \u00b0C (default: 115.0)',
    )
    parser.add_argument(
        '--no-enable', action='store_true',
        help='Skip enable/feedback health checks (useful if motors are already running)',
    )

    args = parser.parse_args()

    # Apply default buses if none supplied
    if not args.bus:
        args.bus = ['can0', 'can1']

    # Validate scan range eagerly
    args.scan_ids = parse_scan_range(args.scan_range)

    return args


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _color(status: Status, text: str) -> str:
    if status == Status.PASS:
        return f"{GREEN}{text}{RESET}"
    if status == Status.WARN:
        return f"{YELLOW}{text}{RESET}"
    return f"{RED}{text}{RESET}"


def print_result(result: CheckResult) -> None:
    icon = ICONS[result.status]
    line = f"  {_color(result.status, icon)} {result.message}"
    print(line)


def print_summary(all_results: dict) -> None:
    failures = 0
    warnings = 0
    for results in all_results.values():
        for r in results:
            if r.status == Status.FAIL:
                failures += 1
            elif r.status == Status.WARN:
                warnings += 1

    summary = f"Summary: {failures} failure(s), {warnings} warning(s)"
    if failures:
        color = RED
    elif warnings:
        color = YELLOW
    else:
        color = GREEN

    print(f"\n{color}{summary}{RESET}")


def any_fail(all_results: dict) -> bool:
    return any(
        r.status == Status.FAIL
        for results in all_results.values()
        for r in results
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    config = load_motor_config(args.config) if args.config else None
    scan_ids = args.scan_ids

    all_results: dict[str, list[CheckResult]] = {}
    found_per_bus: dict[str, set] = {}

    width = 60
    print('=' * width)
    print('  RobStride Pre-flight Diagnostics')
    print(f'  Buses      : {", ".join(args.bus)}')
    print(f'  Bitrate    : {args.bitrate} bps')
    print(f'  Scan range : {args.scan_range}')
    print(f'  Config     : {args.config or "(none)"}')
    print(f'  Temp warn  : {args.temp_warn}\u00b0C')
    print(f'  No-enable  : {args.no_enable}')
    print('=' * width)

    for bus_name in args.bus:
        print(f"\nBus: {bus_name}")
        results: list[CheckResult] = []

        bus, r = check_bus_open(bus_name, args.bitrate)
        results.append(r)
        print_result(r)

        if bus is None:
            all_results[bus_name] = results
            continue

        try:
            found, scan_results, error_count = check_scan(bus, bus_name, scan_ids)
            results.extend(scan_results)
            for r in scan_results:
                print_result(r)

            found_per_bus[bus_name] = set(found.keys())

            if config is not None:
                expected_results = check_expected_motors(
                    set(found.keys()), bus_name, config
                )
                results.extend(expected_results)
                for r in expected_results:
                    print_result(r)

            # Check 11: error frames
            if error_count > 0:
                r = CheckResult(
                    Status.WARN,
                    f"{bus_name}: {error_count} CAN error frame(s) observed "
                    "— check wiring/termination",
                )
                results.append(r)
                print_result(r)

            # Checks 7-9: enable + health (skip if --no-enable or no motors found)
            if not args.no_enable and found:
                health_results = check_motor_health(
                    bus, bus_name, list(found.keys()), args.temp_warn
                )
                results.extend(health_results)
                for r in health_results:
                    print_result(r)

        finally:
            bus.shutdown()

        all_results[bus_name] = results

    # Check 10: cross-bus ID shadowing (only if we saw more than one bus)
    if len(found_per_bus) > 1:
        print("\nCross-bus checks:")
        cross_results = check_cross_bus_ids(found_per_bus)
        for r in cross_results:
            print_result(r)
        # Attach cross-bus results to each bus's result list for summary counting
        for bus_name in all_results:
            all_results[bus_name].extend(cross_results)

    print_summary(all_results)
    sys.exit(1 if any_fail(all_results) else 0)


if __name__ == '__main__':
    main()
