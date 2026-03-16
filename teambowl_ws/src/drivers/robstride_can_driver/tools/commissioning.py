"""
commissioning.py — Standalone CLI for RobStride motor commissioning.

Supported motors: RS04, RS00, RS05
Protocol: RobStride Private Protocol over CAN 2.0, 1 Mbps, Extended 29-bit frames.
No ROS2 required — only `python-can` and `pyyaml` (plus stdlib).

Use this tool before or outside of ROS2 to:
  - Discover motors on the bus
  - Assign / change CAN IDs
  - Set mechanical zero positions
  - Adjust zero offsets
  - Read / write arbitrary parameters
  - Save parameters to flash
  - Change baud rate or switch protocol

Basic usage examples:
  # Bring up the CAN interface first:
  #   sudo ip link set can0 up type can bitrate 1000000

  python commissioning.py --bus can0 scan
  python commissioning.py --bus can0 get-id 0x01
  python commissioning.py --bus can0 set-id 0x01 0x03
  python commissioning.py --bus can0 enable 0x03
  python commissioning.py --bus can0 set-zero 0x03
  python commissioning.py --bus can0 shift-zero 0x03 0.05
  python commissioning.py --bus can0 set-offset 0x03 1.5708
  python commissioning.py --bus can0 read 0x03 0x702B
  python commissioning.py --bus can0 write 0x03 0x702B 1.5708
  python commissioning.py --bus can0 write-int 0x03 0x7005 2
  python commissioning.py --bus can0 save 0x03
  python commissioning.py --bus can0 set-baud 0x03 1M
  python commissioning.py --bus can0 set-protocol 0x03 private
  python commissioning.py --bus can0 active-report 0x03 on
  python commissioning.py --bus can0 fault-read 0x03
  python commissioning.py --bus can0 version 0x03

  # Dry-run (print frame without opening bus):
  python commissioning.py --bus can0 --dry-run enable 0x03
"""

import argparse
import struct
import sys

# ---------------------------------------------------------------------------
# CAN frame construction helpers
# ---------------------------------------------------------------------------

def build_can_id(msg_type: int, host_id: int, motor_id: int, new_id: int = 0) -> int:
    """Compose a 29-bit CAN arbitration ID from message type and node IDs."""
    if msg_type == 0x07:  # set_id embeds new_id in bits [23:16]
        return (msg_type << 24) | (new_id << 16) | (host_id << 8) | motor_id
    return (msg_type << 24) | (host_id << 8) | motor_id


def make_frame(msg_type: int, host_id: int, motor_id: int,
               data: bytes, new_id: int = 0):
    """Return a dict suitable for constructing a can.Message."""
    arb_id = build_can_id(msg_type, host_id, motor_id, new_id)
    return {"arbitration_id": arb_id, "data": data, "is_extended_id": True}


# ---------------------------------------------------------------------------
# Frame builders for every message type
# ---------------------------------------------------------------------------

BAUD_CODES = {"1M": 0x01, "500K": 0x02, "250K": 0x03, "125K": 0x04}
PROTO_CODES = {"private": 0x00, "canopen": 0x01, "mit": 0x02}

_MAGIC_HEADER = b'\x01\x02\x03\x04\x05\x06'


def frame_get_device_id(host_id, motor_id):
    return make_frame(0x00, host_id, motor_id, bytes(8))


def frame_enable(host_id, motor_id):
    return make_frame(0x03, host_id, motor_id, bytes(8))


def frame_stop(host_id, motor_id, clear_fault=False):
    data = bytes([0x01 if clear_fault else 0x00]) + bytes(7)
    return make_frame(0x04, host_id, motor_id, data)


def frame_version(host_id, motor_id):
    """Type 4 with Byte1=0xC4 reads firmware version."""
    data = b'\x00\xC4' + bytes(6)
    return make_frame(0x04, host_id, motor_id, data)


def frame_set_zero(host_id, motor_id):
    data = b'\x01' + bytes(7)
    return make_frame(0x06, host_id, motor_id, data)


def frame_set_id(host_id, motor_id, new_id):
    return make_frame(0x07, host_id, motor_id, bytes(8), new_id=new_id)


def frame_read_param(host_id, motor_id, index: int):
    data = struct.pack('<H', index) + bytes(6)
    return make_frame(0x11, host_id, motor_id, data)


def frame_write_param_float(host_id, motor_id, index: int, value: float):
    value_bytes = struct.pack('<f', value)
    data = struct.pack('<H', index) + b'\x00\x00' + value_bytes
    return make_frame(0x12, host_id, motor_id, data)


def frame_write_param_int(host_id, motor_id, index: int, value: int):
    value_bytes = struct.pack('<I', value & 0xFFFFFFFF)
    data = struct.pack('<H', index) + b'\x00\x00' + value_bytes
    return make_frame(0x12, host_id, motor_id, data)


def frame_save(host_id, motor_id):
    data = b'\x01\x02\x03\x04\x05\x06\x07\x08'
    return make_frame(0x16, host_id, motor_id, data)


def frame_set_baud(host_id, motor_id, baud_str: str):
    code = BAUD_CODES[baud_str]
    data = _MAGIC_HEADER + bytes([code]) + b'\x00'
    return make_frame(0x17, host_id, motor_id, data)


def frame_active_report(host_id, motor_id, enable: bool):
    data = _MAGIC_HEADER + bytes([0x01 if enable else 0x00]) + b'\x00'
    return make_frame(0x18, host_id, motor_id, data)


def frame_set_protocol(host_id, motor_id, proto_str: str):
    code = PROTO_CODES[proto_str]
    data = _MAGIC_HEADER + bytes([code]) + b'\x00'
    return make_frame(0x19, host_id, motor_id, data)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_frame(frame: dict, label: str = "TX"):
    arb_id = frame["arbitration_id"]
    data = frame["data"]
    hex_data = " ".join(f"{b:02X}" for b in data)
    print(f"[{label}] CAN ID: 0x{arb_id:08X}  DLC: {len(data)}  Data: {hex_data}")


def print_response(msg, label: str = "RX"):
    if msg is None:
        print(f"[{label}] <no response>")
        return
    hex_data = " ".join(f"{b:02X}" for b in msg.data)
    print(f"[{label}] CAN ID: 0x{msg.arbitration_id:08X}  DLC: {len(msg.data)}  Data: {hex_data}")


def parse_reply_success(can_id: int) -> bool:
    return ((can_id >> 16) & 0xFF) == 0x00


def parse_read_param_reply(msg):
    """Return (success, float_val, uint_val) from a Type 17 reply."""
    if msg is None or len(msg.data) < 8:
        return False, None, None
    success = parse_reply_success(msg.arbitration_id)
    raw = bytes(msg.data[4:8])
    float_val = struct.unpack('<f', raw)[0]
    uint_val = struct.unpack('<I', raw)[0]
    return success, float_val, uint_val


def print_table(rows, headers):
    """Simple fixed-width table printer."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


# ---------------------------------------------------------------------------
# Bus helpers
# ---------------------------------------------------------------------------

def send_and_recv(bus, frame: dict, timeout: float = 0.5):
    """Send a CAN frame and wait for a response, retrying once on None."""
    import can
    msg = can.Message(
        arbitration_id=frame["arbitration_id"],
        data=frame["data"],
        is_extended_id=frame["is_extended_id"],
    )
    bus.send(msg)
    response = bus.recv(timeout=timeout)
    if response is None:
        response = bus.recv(timeout=timeout)
    return response


def open_bus(bus_name: str, bitrate: int):
    """Open a SocketCAN bus, printing a clear error on failure."""
    try:
        import can
        return can.Bus(interface="socketcan", channel=bus_name, bitrate=bitrate)
    except Exception as exc:
        print(f"ERROR: Failed to open CAN bus '{bus_name}': {exc}", file=sys.stderr)
        print("  Make sure the interface is up, e.g.:", file=sys.stderr)
        print(f"    sudo ip link set {bus_name} up type can bitrate {bitrate}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_scan(args):
    host_id = args.host_id
    found = []

    if args.dry_run:
        print("Dry-run: would scan motor IDs 0x01–0x20")
        for mid in range(0x01, 0x21):
            frame = frame_get_device_id(host_id, mid)
            print_frame(frame, label=f"TX id=0x{mid:02X}")
        return

    bus = open_bus(args.bus, args.bitrate)
    try:
        import can
        print(f"Scanning bus '{args.bus}' for motors (IDs 0x01–0x20) ...")
        for mid in range(0x01, 0x21):
            frame = frame_get_device_id(host_id, mid)
            msg = can.Message(
                arbitration_id=frame["arbitration_id"],
                data=frame["data"],
                is_extended_id=True,
            )
            bus.send(msg)
            resp = bus.recv(timeout=0.05)
            if resp is not None:
                found.append((f"0x{mid:02X}", f"0x{resp.arbitration_id:08X}",
                               " ".join(f"{b:02X}" for b in resp.data)))
        if found:
            print()
            print_table(found, ["Motor ID", "Reply CAN ID", "Data"])
        else:
            print("No motors found.")
    finally:
        bus.shutdown()


def cmd_get_id(args):
    motor_id = args.motor_id
    frame = frame_get_device_id(args.host_id, motor_id)
    print_frame(frame)
    if args.dry_run:
        return
    bus = open_bus(args.bus, args.bitrate)
    try:
        resp = send_and_recv(bus, frame)
        print_response(resp)
        if resp and len(resp.data) >= 8:
            uid = int.from_bytes(resp.data[:8], "little")
            print(f"  MCU Unique ID (64-bit): 0x{uid:016X}")
    finally:
        bus.shutdown()


def cmd_enable(args):
    motor_id = args.motor_id
    frame = frame_enable(args.host_id, motor_id)
    print_frame(frame)
    if args.dry_run:
        return
    bus = open_bus(args.bus, args.bitrate)
    try:
        resp = send_and_recv(bus, frame)
        print_response(resp)
        if resp:
            ok = parse_reply_success(resp.arbitration_id)
            print(f"  Status: {'OK' if ok else 'FAULT'}")
    finally:
        bus.shutdown()


def cmd_stop(args):
    motor_id = args.motor_id
    clear = getattr(args, "clear_fault", False)
    frame = frame_stop(args.host_id, motor_id, clear_fault=clear)
    print_frame(frame)
    if args.dry_run:
        return
    bus = open_bus(args.bus, args.bitrate)
    try:
        resp = send_and_recv(bus, frame)
        print_response(resp)
        if resp:
            ok = parse_reply_success(resp.arbitration_id)
            print(f"  Status: {'OK' if ok else 'FAULT'}")
    finally:
        bus.shutdown()


def cmd_set_id(args):
    current_id = args.current_id
    new_id = args.new_id
    frame = frame_set_id(args.host_id, current_id, new_id)
    print_frame(frame)
    if args.dry_run:
        return
    bus = open_bus(args.bus, args.bitrate)
    try:
        resp = send_and_recv(bus, frame)
        print_response(resp)
        if resp:
            ok = parse_reply_success(resp.arbitration_id)
            print(f"  Status: {'OK — new ID 0x{new_id:02X} takes effect immediately' if ok else 'FAULT'}")
    finally:
        bus.shutdown()


def cmd_set_zero(args):
    motor_id = args.motor_id
    frame = frame_set_zero(args.host_id, motor_id)
    print_frame(frame)
    if args.dry_run:
        return
    bus = open_bus(args.bus, args.bitrate)
    try:
        resp = send_and_recv(bus, frame)
        print_response(resp)
        if resp:
            ok = parse_reply_success(resp.arbitration_id)
            print(f"  Status: {'OK' if ok else 'FAULT'}")
    finally:
        bus.shutdown()


def cmd_shift_zero(args):
    motor_id = args.motor_id
    delta = args.delta_rad

    read_frame = frame_read_param(args.host_id, motor_id, 0x702B)
    print_frame(read_frame, label="TX (read add_offset)")

    if args.dry_run:
        print(f"  Would read 0x702B, add delta={delta} rad, then write back.")
        write_frame = frame_write_param_float(args.host_id, motor_id, 0x702B, 0.0 + delta)
        print_frame(write_frame, label="TX (write add_offset, example with current=0.0)")
        return

    bus = open_bus(args.bus, args.bitrate)
    try:
        resp = send_and_recv(bus, read_frame)
        print_response(resp, label="RX (add_offset)")
        ok, current_val, _ = parse_read_param_reply(resp)
        if not ok or current_val is None:
            print("ERROR: Could not read current add_offset (0x702B).", file=sys.stderr)
            sys.exit(1)
        print(f"  Current add_offset: {current_val:.6f} rad")
        new_val = current_val + delta
        print(f"  New add_offset:     {new_val:.6f} rad  (delta={delta:+.6f})")
        write_frame = frame_write_param_float(args.host_id, motor_id, 0x702B, new_val)
        print_frame(write_frame, label="TX (write add_offset)")
        resp2 = send_and_recv(bus, write_frame)
        print_response(resp2, label="RX (write reply)")
        if resp2:
            ok2 = parse_reply_success(resp2.arbitration_id)
            print(f"  Status: {'OK' if ok2 else 'FAULT'}")
    finally:
        bus.shutdown()


def cmd_set_offset(args):
    motor_id = args.motor_id
    value = args.rad
    frame = frame_write_param_float(args.host_id, motor_id, 0x702B, value)
    print_frame(frame)
    if args.dry_run:
        return
    bus = open_bus(args.bus, args.bitrate)
    try:
        resp = send_and_recv(bus, frame)
        print_response(resp)
        if resp:
            ok = parse_reply_success(resp.arbitration_id)
            print(f"  Status: {'OK' if ok else 'FAULT'}")
    finally:
        bus.shutdown()


def cmd_read(args):
    motor_id = args.motor_id
    index = args.index
    frame = frame_read_param(args.host_id, motor_id, index)
    print_frame(frame)
    if args.dry_run:
        return
    bus = open_bus(args.bus, args.bitrate)
    try:
        resp = send_and_recv(bus, frame)
        print_response(resp)
        ok, float_val, uint_val = parse_read_param_reply(resp)
        if resp and len(resp.data) >= 8:
            print()
            print_table(
                [
                    ["Index",        f"0x{index:04X}"],
                    ["Success",      str(ok)],
                    ["As float",     f"{float_val}"],
                    ["As uint32",    f"{uint_val}  (0x{uint_val:08X})"],
                    ["Raw bytes",    " ".join(f"{b:02X}" for b in resp.data[4:8])],
                ],
                ["Field", "Value"],
            )
    finally:
        bus.shutdown()


def cmd_write(args):
    motor_id = args.motor_id
    index = args.index
    value = args.value
    frame = frame_write_param_float(args.host_id, motor_id, index, value)
    print_frame(frame)
    if args.dry_run:
        return
    bus = open_bus(args.bus, args.bitrate)
    try:
        resp = send_and_recv(bus, frame)
        print_response(resp)
        if resp:
            ok = parse_reply_success(resp.arbitration_id)
            print(f"  Status: {'OK' if ok else 'FAULT'}")
    finally:
        bus.shutdown()


def cmd_write_int(args):
    motor_id = args.motor_id
    index = args.index
    value = args.value
    frame = frame_write_param_int(args.host_id, motor_id, index, value)
    print_frame(frame)
    if args.dry_run:
        return
    bus = open_bus(args.bus, args.bitrate)
    try:
        resp = send_and_recv(bus, frame)
        print_response(resp)
        if resp:
            ok = parse_reply_success(resp.arbitration_id)
            print(f"  Status: {'OK' if ok else 'FAULT'}")
    finally:
        bus.shutdown()


def cmd_save(args):
    motor_id = args.motor_id
    frame = frame_save(args.host_id, motor_id)
    print_frame(frame)
    if args.dry_run:
        return
    bus = open_bus(args.bus, args.bitrate)
    try:
        resp = send_and_recv(bus, frame)
        print_response(resp)
        if resp:
            ok = parse_reply_success(resp.arbitration_id)
            print(f"  Status: {'OK — parameters saved to flash' if ok else 'FAULT'}")
    finally:
        bus.shutdown()


def cmd_set_baud(args):
    motor_id = args.motor_id
    baud_str = args.baud.upper()
    if baud_str not in BAUD_CODES:
        print(f"ERROR: Unknown baud rate '{baud_str}'. Choose from: {', '.join(BAUD_CODES)}", file=sys.stderr)
        sys.exit(1)
    frame = frame_set_baud(args.host_id, motor_id, baud_str)
    print_frame(frame)
    if args.dry_run:
        return
    bus = open_bus(args.bus, args.bitrate)
    try:
        resp = send_and_recv(bus, frame)
        print_response(resp)
        if resp:
            ok = parse_reply_success(resp.arbitration_id)
            print(f"  Status: {'OK — power cycle required to apply new baud rate' if ok else 'FAULT'}")
    finally:
        bus.shutdown()


def cmd_set_protocol(args):
    motor_id = args.motor_id
    proto = args.protocol.lower()
    if proto not in PROTO_CODES:
        print(f"ERROR: Unknown protocol '{proto}'. Choose from: {', '.join(PROTO_CODES)}", file=sys.stderr)
        sys.exit(1)
    frame = frame_set_protocol(args.host_id, motor_id, proto)
    print_frame(frame)
    if args.dry_run:
        return
    bus = open_bus(args.bus, args.bitrate)
    try:
        resp = send_and_recv(bus, frame)
        print_response(resp)
        if resp:
            ok = parse_reply_success(resp.arbitration_id)
            print(f"  Status: {'OK — power cycle required to apply new protocol' if ok else 'FAULT'}")
    finally:
        bus.shutdown()


def cmd_active_report(args):
    motor_id = args.motor_id
    enable = args.state.lower() == "on"
    frame = frame_active_report(args.host_id, motor_id, enable)
    print_frame(frame)
    if args.dry_run:
        return
    bus = open_bus(args.bus, args.bitrate)
    try:
        resp = send_and_recv(bus, frame)
        print_response(resp)
        if resp:
            ok = parse_reply_success(resp.arbitration_id)
            state_str = "enabled" if enable else "disabled"
            print(f"  Status: {'OK — active report ' + state_str if ok else 'FAULT'}")
    finally:
        bus.shutdown()


def cmd_fault_read(args):
    motor_id = args.motor_id
    frame = frame_get_device_id(args.host_id, motor_id)
    print_frame(frame, label="TX (fault/status query)")
    if args.dry_run:
        return
    bus = open_bus(args.bus, args.bitrate)
    try:
        resp = send_and_recv(bus, frame)
        print_response(resp, label="RX (fault frame)")
        if resp and len(resp.data) >= 8:
            ok = parse_reply_success(resp.arbitration_id)
            raw_bytes = " ".join(f"{b:02X}" for b in resp.data)
            print()
            print_table(
                [
                    ["CAN ID",    f"0x{resp.arbitration_id:08X}"],
                    ["No Fault",  str(ok)],
                    ["Raw Data",  raw_bytes],
                ],
                ["Field", "Value"],
            )
    finally:
        bus.shutdown()


def cmd_version(args):
    motor_id = args.motor_id
    frame = frame_version(args.host_id, motor_id)
    print_frame(frame)
    if args.dry_run:
        return
    bus = open_bus(args.bus, args.bitrate)
    try:
        resp = send_and_recv(bus, frame)
        print_response(resp)
        if resp and len(resp.data) >= 8:
            raw_bytes = " ".join(f"{b:02X}" for b in resp.data)
            print(f"  Raw version bytes: {raw_bytes}")
    finally:
        bus.shutdown()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def auto_int(x):
    """Accept hex (0x..) or decimal integers from CLI."""
    return int(x, 0)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="commissioning.py",
        description=(
            "Standalone CLI for commissioning RobStride motors (RS04, RS00, RS05).\n"
            "No ROS2 required — only python-can and pyyaml.\n"
            "All frames use Extended 29-bit CAN IDs at 1 Mbps."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--bus", default="can0",
        help="SocketCAN interface name (default: can0)",
    )
    parser.add_argument(
        "--host-id", type=auto_int, default=0xFD, metavar="HOST_ID",
        help="Host node ID embedded in outgoing frames (default: 0xFD)",
    )
    parser.add_argument(
        "--bitrate", type=int, default=1000000,
        help="CAN bus bitrate in bps (default: 1000000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the CAN frame(s) that would be sent without opening the bus",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    # scan
    sub.add_parser(
        "scan",
        help="Scan bus for motors by sending Type 0 to IDs 0x01–0x20",
    )

    # get-id
    p = sub.add_parser("get-id", help="Read 64-bit MCU unique ID (Type 0)")
    p.add_argument("motor_id", type=auto_int, help="Motor CAN ID (hex or decimal)")

    # enable
    p = sub.add_parser("enable", help="Enable motor torque output (Type 3)")
    p.add_argument("motor_id", type=auto_int, help="Motor CAN ID")

    # stop
    p = sub.add_parser("stop", help="Stop motor and hold position (Type 4, Byte0=0x00)")
    p.add_argument("motor_id", type=auto_int, help="Motor CAN ID")

    # stop-clear
    p = sub.add_parser("stop-clear", help="Stop motor and clear faults (Type 4, Byte0=0x01)")
    p.add_argument("motor_id", type=auto_int, help="Motor CAN ID")

    # set-id
    p = sub.add_parser("set-id", help="Change motor CAN ID immediately (Type 7)")
    p.add_argument("current_id", type=auto_int, help="Current motor CAN ID")
    p.add_argument("new_id", type=auto_int, help="New motor CAN ID")

    # set-zero
    p = sub.add_parser("set-zero", help="Set current position as mechanical zero (Type 6)")
    p.add_argument("motor_id", type=auto_int, help="Motor CAN ID")

    # shift-zero
    p = sub.add_parser(
        "shift-zero",
        help="Add delta_rad to current add_offset (reads 0x702B then writes 0x702B += delta)",
    )
    p.add_argument("motor_id", type=auto_int, help="Motor CAN ID")
    p.add_argument("delta_rad", type=float, help="Offset delta in radians (can be negative)")

    # set-offset
    p = sub.add_parser("set-offset", help="Write absolute zero offset in radians (Type 18, index 0x702B)")
    p.add_argument("motor_id", type=auto_int, help="Motor CAN ID")
    p.add_argument("rad", type=float, help="Absolute offset value in radians")

    # read
    p = sub.add_parser(
        "read",
        help="Read parameter by hex index (Type 17); prints both float and uint32 interpretations",
    )
    p.add_argument("motor_id", type=auto_int, help="Motor CAN ID")
    p.add_argument("index", type=auto_int, help="Parameter index e.g. 0x701C")

    # write
    p = sub.add_parser("write", help="Write float parameter (Type 18, volatile — use save to persist)")
    p.add_argument("motor_id", type=auto_int, help="Motor CAN ID")
    p.add_argument("index", type=auto_int, help="Parameter index e.g. 0x702B")
    p.add_argument("value", type=float, help="Float value to write")

    # write-int
    p = sub.add_parser("write-int", help="Write integer parameter (Type 18, volatile — use save to persist)")
    p.add_argument("motor_id", type=auto_int, help="Motor CAN ID")
    p.add_argument("index", type=auto_int, help="Parameter index e.g. 0x7005")
    p.add_argument("value", type=auto_int, help="Integer value to write (hex or decimal)")

    # save
    p = sub.add_parser("save", help="Save all parameters to flash (Type 22)")
    p.add_argument("motor_id", type=auto_int, help="Motor CAN ID")

    # set-baud
    p = sub.add_parser(
        "set-baud",
        help="Change CAN baud rate (Type 23); power cycle required to apply",
    )
    p.add_argument("motor_id", type=auto_int, help="Motor CAN ID")
    p.add_argument("baud", choices=["1M", "500K", "250K", "125K"], help="Target baud rate")

    # set-protocol
    p = sub.add_parser(
        "set-protocol",
        help="Switch communication protocol (Type 25); power cycle required to apply",
    )
    p.add_argument("motor_id", type=auto_int, help="Motor CAN ID")
    p.add_argument("protocol", choices=["private", "mit", "canopen"], help="Target protocol")

    # active-report
    p = sub.add_parser(
        "active-report",
        help="Enable or disable periodic motor feedback (Type 24)",
    )
    p.add_argument("motor_id", type=auto_int, help="Motor CAN ID")
    p.add_argument("state", choices=["on", "off"], help="on = enable, off = disable")

    # fault-read
    p = sub.add_parser("fault-read", help="Read and decode fault/status frame (Type 0, parse response)")
    p.add_argument("motor_id", type=auto_int, help="Motor CAN ID")

    # version
    p = sub.add_parser("version", help="Read firmware version (Type 4, Byte1=0xC4)")
    p.add_argument("motor_id", type=auto_int, help="Motor CAN ID")

    return parser


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

COMMANDS = {
    "scan":           cmd_scan,
    "get-id":         cmd_get_id,
    "enable":         cmd_enable,
    "stop":           cmd_stop,
    "stop-clear":     lambda args: (setattr(args, "clear_fault", True), cmd_stop(args)),
    "set-id":         cmd_set_id,
    "set-zero":       cmd_set_zero,
    "shift-zero":     cmd_shift_zero,
    "set-offset":     cmd_set_offset,
    "read":           cmd_read,
    "write":          cmd_write,
    "write-int":      cmd_write_int,
    "save":           cmd_save,
    "set-baud":       cmd_set_baud,
    "set-protocol":   cmd_set_protocol,
    "active-report":  cmd_active_report,
    "fault-read":     cmd_fault_read,
    "version":        cmd_version,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")

    try:
        handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()
