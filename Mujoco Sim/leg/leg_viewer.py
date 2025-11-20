#!/usr/bin/env python3
"""Minimal viewer that keeps the MuJoCo scene running indefinitely."""

from __future__ import annotations

import os
import math

import mujoco
import mujoco.viewer
import time

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")
ACTUATOR_WAVES = [
    ("passive3", 0.35, 0.4),
    ("passive4", 0.35, 0.55),
    ("passive2", 0.35, 0.7),
    ("hip-surge", 0.25, 0.6),
    ("hip-sway", 0.25, 0.75),
    ("knee", 0.5, 0.65),
    ("ankle", 0.5, 0.9),
]


def _configure_sine_drivers(model: mujoco.MjModel) -> list[tuple[int, float, float, float]]:
    """Map actuator names to ids and attach a phase offset to each sine driver."""
    drivers: list[tuple[int, float, float, float]] = []
    for index, (name, amplitude, frequency_hz) in enumerate(ACTUATOR_WAVES):
        try:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        except ValueError:
            continue
        phase = index * math.pi / 5.0
        drivers.append((actuator_id, amplitude, frequency_hz, phase))
    return drivers


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    drivers = _configure_sine_drivers(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while True:
                t = data.time
                for actuator_id, amplitude, frequency_hz, phase in drivers:
                    data.ctrl[actuator_id] = amplitude * math.sin(
                        2.0 * math.pi * frequency_hz * t + phase
                    )
                mujoco.mj_step(model, data)
                viewer.sync()
                # time.sleep(0.01)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
