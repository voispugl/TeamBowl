#!/usr/bin/env python3
"""Keep the weight IMU level with a tunable PID on the hip joints."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")

# Base pose targets (radians). The "(1)" joints mirror automatically.
BASE_SETPOINTS = [
    ("hip-surge", 0.0),
    ("hip-sway", 0.0),
    ("knee", 0.0),
    ("ankle", -0.1),
]

# PID gains for roll (hip-surge) and pitch (hip-sway). Tune these as needed.
KP_ROLL = 80.0
KI_ROLL = 5.0
KD_ROLL = 6.0

KP_PITCH = 80.0
KI_PITCH = 5.0
KD_PITCH = 6.0

INTEGRAL_CLAMP = 0.4  # limit integral windup (radians)
CTRL_LIMIT = 1.5      # clamp commanded joint targets to this magnitude


@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    integral: float = 0.0

    def step(self, error: float, rate: float, dt: float) -> float:
        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -INTEGRAL_CLAMP, INTEGRAL_CLAMP))
        return self.kp * error + self.ki * self.integral - self.kd * rate


def _mirror_setpoints(base: Iterable[Tuple[str, float]]) -> Dict[str, float]:
    """Mirror left/right joints so you only provide one side."""
    setpoints: Dict[str, float] = {name: value for name, value in base}
    for joint in ("hip-surge", "hip-sway", "knee", "ankle"):
        mirror = f"{joint} (1)"
        if joint in setpoints:
            setpoints[mirror] = setpoints[joint]
        elif mirror in setpoints:
            setpoints[joint] = setpoints[mirror]
    return setpoints


def _build_actuator_map(model: mujoco.MjModel) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            mapping[name] = i
    return mapping


def _sensor_span(model: mujoco.MjModel, name: str) -> Tuple[int, int]:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    start = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return start, dim


def _level_angles(mat: np.ndarray) -> Tuple[float, float]:
    """Return roll and pitch (radians) from a 3x3 rotation matrix."""
    roll = math.atan2(mat[2, 1], mat[2, 2])
    pitch = math.atan2(-mat[2, 0], math.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
    return roll, pitch


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    actuator_map = _build_actuator_map(model)
    setpoints = _mirror_setpoints(BASE_SETPOINTS)
    imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_weight")
    gyro_start, gyro_dim = _sensor_span(model, "imu_gyro")
    if gyro_dim < 3:
        raise SystemExit("imu_gyro must provide at least 3 axes.")

    roll_pid = PID(KP_ROLL, KI_ROLL, KD_ROLL)
    pitch_pid = PID(KP_PITCH, KI_PITCH, KD_PITCH)

    # Initialize the pose.
    for name, target in setpoints.items():
        act_id = actuator_map.get(name)
        if act_id is not None:
            data.ctrl[act_id] = target

    last_time = data.time
    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while viewer.is_running():
                now = data.time
                dt = max(model.opt.timestep, now - last_time)
                last_time = now

                imu_mat = data.site_xmat[imu_site_id].reshape(3, 3)
                roll, pitch = _level_angles(imu_mat)
                gyro = data.sensor[gyro_start:gyro_start + gyro_dim]
                roll_rate = float(gyro[0])
                pitch_rate = float(gyro[1])

                roll_correction = roll_pid.step(-roll, roll_rate, dt)
                pitch_correction = pitch_pid.step(-pitch, pitch_rate, dt)

                for name, base in setpoints.items():
                    act_id = actuator_map.get(name)
                    if act_id is None:
                        continue
                    cmd = base
                    if "hip-surge" in name:
                        cmd += roll_correction
                    elif "hip-sway" in name:
                        cmd += pitch_correction
                    cmd = float(np.clip(cmd, -CTRL_LIMIT, CTRL_LIMIT))
                    data.ctrl[act_id] = cmd

                mujoco.mj_step(model, data)
                viewer.sync()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
