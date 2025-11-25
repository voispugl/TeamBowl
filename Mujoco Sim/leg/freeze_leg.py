#!/usr/bin/env python3
"""Hold the leg in the air at configurable joint angles with gravity off."""

from __future__ import annotations

import os
import time
from typing import Dict

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")

# Desired joint angles in radians. Adjust these to change the starting pose.
TARGET_ANGLES: Dict[str, float] = {
    "hip-sway": 0,
    "hip-sway (1)": 0,
    "hip-surge": -0.8,
    "hip-surge (1)": -0.8,
    "knee": -0.8,
    "knee (1)": -0.8,
    "ankle": 0.2,
    "ankle (1)": 0.2,
}

KP = 120.0  # proportional gain
KD = 4.0    # derivative gain
ANKLE_TORQUE_LIMIT = 10.0
OTHER_TORQUE_LIMIT = 120.0
LOCK_BASE = True  # keep freejoint pose fixed to stop spinning


def _set_joint_angles(model: mujoco.MjModel, data: mujoco.MjData, targets: Dict[str, float]) -> None:
    """Directly write target hinge angles into qpos and zero the velocities."""
    for name, angle in targets.items():
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = model.jnt_qposadr[jnt_id]
        data.qpos[adr] = angle
        data.qvel[adr] = 0.0
    mujoco.mj_forward(model, data)


def _apply_pd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    targets: Dict[str, float],
    actuator_map: Dict[str, int],
) -> None:
    """PD control to hold each joint at its target."""
    for name, target in targets.items():
        act_id = actuator_map.get(name)
        if act_id is None:
            continue
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = model.jnt_qposadr[jnt_id]
        q = data.qpos[adr]
        qd = data.qvel[adr]
        torque = KP * (target - q) - KD * qd
        limit = ANKLE_TORQUE_LIMIT if "ankle" in name else OTHER_TORQUE_LIMIT
        data.ctrl[act_id] = float(np.clip(torque, -limit, limit))


def _build_actuator_map(model: mujoco.MjModel) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            mapping[name] = i
    return mapping


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # Disable gravity to keep the leg floating while you tune angles.
    model.opt.gravity[:] = 0.0
    base_qpos0 = np.copy(data.qpos[:7])

    actuator_map = _build_actuator_map(model)
    _set_joint_angles(model, data, TARGET_ANGLES)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while viewer.is_running():
                if LOCK_BASE:
                    # Freeze the freejoint pose and velocity to stop drift/spin.
                    data.qpos[:7] = base_qpos0
                    data.qvel[:6] = 0.0
                _apply_pd(model, data, TARGET_ANGLES, actuator_map)
                mujoco.mj_step(model, data)
                time.sleep(0.005)
                viewer.sync()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
