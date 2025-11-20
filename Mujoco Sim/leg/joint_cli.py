#!/usr/bin/env python3
"""Interactive MuJoCo driver that lets you command each actuator from the CLI."""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Dict

import mujoco
import mujoco.viewer

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")


def _build_actuator_map(model: mujoco.MjModel) -> Dict[str, int]:
    """Return a mapping of actuator names to ids."""
    actuators: Dict[str, int] = {}
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if name:
            actuators[name] = actuator_id
    return actuators


def _print_help(actuator_names: list[str]) -> None:
    """Display available commands."""
    print(
        "Commands:\n"
        "  <actuator> <value>  Set target command for an actuator (position/torque depending on model).\n"
        "  list                Show the actuators that can be controlled.\n"
        "  zero                Reset all actuator commands to 0.\n"
        "  help                Show this help message.\n"
        "  quit/exit           Stop the simulation.\n"
    )
    print("Actuators:", ", ".join(actuator_names))


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    actuator_map = _build_actuator_map(model)
    if not actuator_map:
        raise SystemExit("No actuators defined in the model.")

    actuator_names = sorted(actuator_map)
    ctrl_values = {actuator_id: 0.0 for actuator_id in actuator_map.values()}
    ctrl_lock = threading.Lock()
    stop_event = threading.Event()

    def command_loop() -> None:
        """Read user commands from stdin and update the control targets."""
        _print_help(actuator_names)
        while not stop_event.is_set():
            try:
                raw = input("> ").strip()
            except EOFError:
                stop_event.set()
                break
            if not raw:
                continue

            lower = raw.lower()
            if lower in {"quit", "exit"}:
                stop_event.set()
                break
            if lower == "help":
                _print_help(actuator_names)
                continue
            if lower == "list":
                print("Actuators:", ", ".join(actuator_names))
                continue
            if lower == "zero":
                with ctrl_lock:
                    for actuator_id in ctrl_values:
                        ctrl_values[actuator_id] = 0.0
                print("All actuator commands reset to 0.")
                continue

            parts = raw.split()
            if len(parts) != 2:
                print("Invalid command. Use '<actuator> <value>' or type 'help'.")
                continue

            name, value_str = parts
            actuator_id = actuator_map.get(name)
            if actuator_id is None:
                print(f"Unknown actuator '{name}'. Type 'list' to see valid names.")
                continue
            try:
                value = float(value_str)
            except ValueError:
                print(f"Could not parse value '{value_str}' as a number.")
                continue

            with ctrl_lock:
                ctrl_values[actuator_id] = value
            print(f"Set {name} command to {value}")

    command_thread = threading.Thread(target=command_loop, name="cli-input", daemon=True)
    command_thread.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while not stop_event.is_set():
                now = data.time
                with ctrl_lock:
                    for actuator_id, value in ctrl_values.items():
                        data.ctrl[actuator_id] = value
                mujoco.mj_step(model, data)
                viewer.sync()
                # Small sleep keeps CPU usage reasonable when paused.
                # Use a minimum timestep if the sim is not advancing quickly.
                if data.time == now:
                    time.sleep(0.002)
        except KeyboardInterrupt:
            stop_event.set()
            command_thread.join()


    stop_event.set()
    command_thread.join()


if __name__ == "__main__":
    main()
