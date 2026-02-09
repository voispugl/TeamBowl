#!/usr/bin/env python3
"""Minimal viewer that holds actuators at editable position setpoints."""

from __future__ import annotations

import os

import mujoco
import mujoco.viewer
import time

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")

# Edit the values below to change the commanded position targets (radians).
# The "(1)" actuators mirror their counterparts automatically so you only
# need to tweak one side (hip-surge/sway, knee, ankle).
ACTUATOR_SETPOINTS = [
    ("hip-surge", 0),
    ("hip-sway", 0.0),
    ("knee", -0),
    ("ankle", -0.2),
]


def _build_setpoints(model: mujoco.MjModel) -> list[tuple[int, float]]:
    """Map actuator names to ids and duplicate the bilateral actuator pairs."""
    setpoints = {name: value for name, value in ACTUATOR_SETPOINTS}
    mirror_names = ["hip-surge", "hip-sway", "knee", "ankle"]
    for base in mirror_names:
        alt = f"{base} (1)"
        if base in setpoints:
            setpoints[alt] = setpoints[base]
        elif alt in setpoints:
            setpoints[base] = setpoints[alt]

    mapped: list[tuple[int, float]] = []
    for name, value in setpoints.items():
        try:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        except ValueError:
            continue
        mapped.append((actuator_id, value))
    return mapped


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    targets = _build_setpoints(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while True:
                for actuator_id, value in targets:
                    data.ctrl[actuator_id] = value
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
