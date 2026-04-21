#!/usr/bin/env python3
"""
Diagnose why the wheels don't move when "Go" is pressed in the steamdeck web UI.
Run while the robot stack is up:
    python3 ~/TeamBowl/debug_wheels.py
"""
import os
import sys
import json
import time
import subprocess
import threading

# Force CycloneDDS to match the running stack
os.environ.setdefault('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import Twist

RED   = '\033[0;31m'
GRN   = '\033[0;32m'
YEL   = '\033[1;33m'
NC    = '\033[0m'

failures = []

def pad(s, w=42):
    return s.ljust(w)

def ok(label, detail=''):
    print(f"{pad(label)}{GRN}PASS{NC}" + (f" ({detail})" if detail else ''))

def fail(label, detail='', note=''):
    print(f"{pad(label)}{RED}FAIL{NC}" + (f" ({detail})" if detail else ''))
    if note:
        print(f"      → {note}")
    failures.append(f"{label}: {detail}")

def warn(label, detail=''):
    print(f"{pad(label)}{YEL}WARN{NC}" + (f" ({detail})" if detail else ''))

def info(label, detail=''):
    print(f"{pad(label)}{YEL}INFO{NC}" + (f" ({detail})" if detail else ''))


def receive_once(node, msg_type, topic, timeout_s=3.0, qos=None):
    """Subscribe and return the first message received, or None on timeout."""
    received = threading.Event()
    result = [None]

    if qos is None:
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

    def cb(msg):
        result[0] = msg
        received.set()

    sub = node.create_subscription(msg_type, topic, cb, qos)
    deadline = time.time() + timeout_s
    while not received.is_set() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return result[0]


def main():
    rclpy.init()
    node = Node('debug_wheels')

    # Best-effort QoS for most topics
    be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST)
    # Transient-local for latched topics (/robot_mode, /estop)
    tl = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    history=HistoryPolicy.KEEP_LAST)

    print()
    print("========================================")
    print("  TeamBowl wheel command diagnostics")
    print("========================================")
    print()

    # ── Check 1 — Robot mode ────────────────────────────────────────────────
    msg = receive_once(node, String, '/robot_mode', timeout_s=2.0, qos=tl)
    mode = msg.data if msg else None
    if mode is None:
        fail("[1] Robot mode", "no data", "is the stack running?")
    elif mode == 'driving':
        ok("[1] Robot mode", mode)
    elif mode == 'off':
        fail("[1] Robot mode", mode, "set to 'driving' first — Nav2 cmd_vel blocked by vel_cmd_mux")
    else:
        warn("[1] Robot mode", f"{mode} — Nav2 routes via 'driving' or 'auton'")

    # ── Check 2 — E-stop ────────────────────────────────────────────────────
    msg = receive_once(node, Bool, '/estop', timeout_s=2.0, qos=tl)
    if msg is None:
        # try /e_stop too
        msg = receive_once(node, Bool, '/e_stop', timeout_s=1.0, qos=tl)
    if msg is None:
        warn("[2] E-stop", "no data — assumed False")
    elif not msg.data:
        ok("[2] E-stop", str(msg.data))
    else:
        fail("[2] E-stop", str(msg.data), "e-stop active — all motion blocked")

    # ── Check 3 — trajectory_status ─────────────────────────────────────────
    msg = receive_once(node, String, '/trajectory_status', timeout_s=2.0, qos=be)
    if msg is None:
        warn("[3] trajectory_status", "no data — trajectory_test node not running?")
    else:
        try:
            state = json.loads(msg.data).get('state', '?')
        except Exception:
            state = msg.data
        info("[3] trajectory_status", f"state={state}")

    # ── Checks 4–6 need an active goal ──────────────────────────────────────
    print()
    print("Checks 4–6 need an active goal. Press Go in the browser, then press Enter...")
    input()

    # ── Check 4 — /cmd_vel_auto ─────────────────────────────────────────────
    msg = receive_once(node, Twist, '/cmd_vel_auto', timeout_s=3.0, qos=be)
    if msg is None:
        fail("[4] /cmd_vel_auto (Nav2 output)", "no data in 3 s",
             "Nav2 not publishing — is trajectory_test running? Nav2 planner/controller up?")
    else:
        ok("[4] /cmd_vel_auto (Nav2 output)", f"linear.x={msg.linear.x:.3f}")

    # ── Check 5 — /cmd_vel_selected ─────────────────────────────────────────
    msg = receive_once(node, Twist, '/cmd_vel_selected', timeout_s=3.0, qos=be)
    if msg is None:
        fail("[5] /cmd_vel_selected (mux out)", "no data in 3 s",
             "vel_cmd_mux blocking — check mode (need 'driving') and estop")
    else:
        ok("[5] /cmd_vel_selected (mux out)", f"linear.x={msg.linear.x:.3f}")

    # ── Check 6 — /cmd_vel ──────────────────────────────────────────────────
    msg = receive_once(node, Twist, '/cmd_vel', timeout_s=3.0, qos=be)
    if msg is None:
        fail("[6] /cmd_vel (balance_ctrl out)", "no data in 3 s",
             "balance_controller not publishing — check node is running")
    else:
        ok("[6] /cmd_vel (balance_ctrl out)", f"linear.x={msg.linear.x:.3f}")

    # ── Check 7 — VESC serial ports ─────────────────────────────────────────
    print()
    ports = [p for p in ['/dev/ttyACM0', '/dev/ttyACM1'] if os.path.exists(p)]
    if not ports:
        fail("[7] VESC serial ports", "none found", "check USB cables")
    elif len(ports) == 2:
        ok("[7] VESC serial ports", " ".join(ports))
    else:
        warn("[7] VESC serial ports", f"only found {ports}, expected ttyACM0+ttyACM1")

    # ── Check 8 — VESC node alive ───────────────────────────────────────────
    node_names = node.get_node_names()
    vesc_nodes = [n for n in node_names if 'vesc' in n.lower()]
    if not vesc_nodes:
        fail("[8] VESC node running", "not found in node list")
    else:
        ok("[8] VESC node running", str(vesc_nodes))
        # Quick rate check on /wheel_vel_left
        counts = [0]
        def count_cb(msg):
            counts[0] += 1
        sub = node.create_subscription(Twist, '/wheel_vel_left', count_cb,
                                       QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        t0 = time.time()
        while time.time() - t0 < 3.0:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_subscription(sub)
        hz = counts[0] / 3.0
        if counts[0] == 0:
            warn("    /wheel_vel_left rate", "no data — VESCs may not be responding")
        else:
            info("    /wheel_vel_left rate", f"~{hz:.1f} Hz")

    # ── Check 9 — Direct /cmd_vel injection ─────────────────────────────────
    print()
    print("────────────────────────────────────────")
    print("[9] Direct /cmd_vel injection test")
    print("    Bypasses Nav2/mux/guard/balance — isolates VESC/serial faults.")
    print()
    answer = input("    Inject 0.1 m/s forward for 1 s? (y/N) ").strip().lower()
    if answer == 'y':
        pub = node.create_publisher(Twist, '/cmd_vel', 10)
        t_msg = Twist()
        t_msg.linear.x = 0.1
        print("    Sending...")
        t0 = time.time()
        while time.time() - t0 < 1.0:
            pub.publish(t_msg)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.05)
        stop = Twist()
        pub.publish(stop)
        rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_publisher(pub)
        print()
        moved = input("    Did the wheels move? (y/N) ").strip().lower()
        if moved == 'y':
            print(f"    {GRN}PASS{NC} — wheels respond to direct /cmd_vel")
            print("    → Fault is upstream of cmd_vel_to_vesc (Nav2/mux/mode/estop)")
        else:
            fail("[9] Direct /cmd_vel injection", "wheels did NOT move",
                 "VESC/serial/hardware fault")

    # ── Summary ─────────────────────────────────────────────────────────────
    node.destroy_node()
    rclpy.shutdown()

    print()
    print("========================================")
    print("  SUMMARY")
    print("========================================")
    if not failures:
        print(f"  {GRN}All checks passed.{NC}")
        print("  If wheels still don't move, check collision_guard velocity cap")
        print("  (max 0.15 m/s) in src/locomotion/config/locomotion.yaml.")
    else:
        print(f"  {RED}{len(failures)} issue(s) found:{NC}")
        for f in failures:
            print(f"  ✗ {f}")
    print()


if __name__ == '__main__':
    main()
