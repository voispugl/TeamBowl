#!/usr/bin/env python3
"""
Live Balance Controller Gain Tuner

Interactive terminal UI for adjusting LQR gains on the running
balance_controller node via ros2 param set — no node restart needed.

Usage (from inside the Docker container, after sourcing ROS2):
    python3 src/locomotion/scripts/tune_gains.py

Requirements: ROS2 (ros2 CLI on PATH), running balance_controller node.

Controls:
    Arrow keys / wasd  — navigate params and change values
    +/-                — fine adjust (±10%)
    Enter              — confirm and send to node
    q                  — quit

Or skip the UI and use ros2 param set directly:
    ros2 param set /balance_controller k_theta 35.0
    ros2 param set /balance_controller mass_mode heavy
    ros2 param dump /balance_controller      # see all current values
    ros2 param list /balance_controller      # list param names
"""

import subprocess
import sys
import os

# ---- Params to expose in the tuner ----
# (name, description, step_size)
PARAMS = [
    # -- LQR inner loop --
    ('k_theta',          'LQR: pitch angle gain (rad→m/s)',     2.0),
    ('k_theta_dot',      'LQR: pitch rate gain ((rad/s)→m/s)',  0.5),
    ('k_v',              'LQR: velocity error gain',            0.1),
    # -- Velocity PI outer loop --
    ('kp_vel',           'PI: proportional vel gain',           0.05),
    ('ki_vel',           'PI: integrator vel gain',             0.01),
    # -- Safety / geometry --
    ('theta_max_cmd',    'Max lean setpoint (rad)',              0.02),
    ('theta_eq_offset',  'Balance offset (rad), ID on hardware',0.01),
    ('l_com',            'CoM height above axle (m)',           0.01),
    # -- Mass gain schedule --
    ('k_theta_light',    'LQR theta gain — light mass',         2.0),
    ('k_theta_heavy',    'LQR theta gain — heavy mass',         2.0),
]

NODE = '/balance_controller'


def get_param(name: str) -> float:
    result = subprocess.run(
        ['ros2', 'param', 'get', NODE, name],
        capture_output=True, text=True
    )
    # Output format: "Double value is: 30.0"
    for line in result.stdout.splitlines():
        if 'value is:' in line:
            try:
                return float(line.split('value is:')[-1].strip())
            except ValueError:
                pass
    return 0.0


def set_param(name: str, value: float):
    subprocess.run(
        ['ros2', 'param', 'set', NODE, name, str(value)],
        capture_output=True
    )


def dump_all():
    """Print current state of all tunable params."""
    print(f"\n{'─'*60}")
    print(f"  Current gains on {NODE}:")
    print(f"{'─'*60}")
    for name, desc, _ in PARAMS:
        val = get_param(name)
        print(f"  {name:<22} = {val:>8.4f}   ({desc})")
    print(f"{'─'*60}\n")


def interactive_tune():
    """Simple non-curses interactive tuner using numbered menu."""
    print(__doc__)
    print("\nFetching current values from node...")

    # Check node is running
    result = subprocess.run(
        ['ros2', 'param', 'list', NODE],
        capture_output=True, text=True
    )
    if result.returncode != 0 or NODE[1:] not in result.stdout:
        print(f"ERROR: Node {NODE} not running or not found.")
        print("Start the full bringup first, then run this script.")
        sys.exit(1)

    while True:
        dump_all()
        print("Options:")
        for i, (name, desc, step) in enumerate(PARAMS):
            print(f"  [{i:2d}] {name}")
        print("  [r ] Refresh / dump all params")
        print("  [q ] Quit")
        print()
        choice = input("Select param number (or r/q): ").strip().lower()

        if choice == 'q':
            break
        if choice == 'r':
            continue

        try:
            idx = int(choice)
            name, desc, step = PARAMS[idx]
        except (ValueError, IndexError):
            print("Invalid choice.")
            continue

        current = get_param(name)
        print(f"\n  {name} = {current:.4f}   ({desc})")
        print(f"  Enter new value, or +/-N for relative change (e.g. +5, -2.5, 35.0):")
        val_str = input("  > ").strip()

        if val_str.startswith('+'):
            new_val = current + float(val_str[1:])
        elif val_str.startswith('-') and len(val_str) > 1:
            new_val = current - float(val_str[1:])
        else:
            try:
                new_val = float(val_str)
            except ValueError:
                print("  Invalid value, skipping.")
                continue

        print(f"  Setting {name} = {new_val:.4f} ... ", end='', flush=True)
        set_param(name, new_val)
        print("done.")


def main():
    # If args given, do a quick set and exit
    # Usage: tune_gains.py k_theta 35.0
    if len(sys.argv) == 3:
        name, value = sys.argv[1], float(sys.argv[2])
        print(f"Setting {NODE} {name} = {value}")
        set_param(name, value)
        print(f"Current: {get_param(name)}")
        return

    if len(sys.argv) == 2 and sys.argv[1] == 'dump':
        dump_all()
        return

    interactive_tune()


if __name__ == '__main__':
    main()
