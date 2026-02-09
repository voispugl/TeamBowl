#!/usr/bin/env python3
"""Sweep hip-sway and knee joints while locking the base to vertical motion only."""

from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")
BASE_DEGREES = 0  # sweep start angle in degrees
PEAK_DEGREES = 45.0  # sweep end angle in degrees
CYCLE_DURATION = 6.0  # seconds for 0 -> peak -> 0 sweep
OUTPUT_PNG = os.path.join(os.path.dirname(__file__), "planar_torque_sweep.png")
OUTPUT_XLSX = os.path.join(os.path.dirname(__file__), "planar_torque_sweep.xlsx")
HEADLESS = False

# Hold the other joints steady so the motion is dominated by the knees/hip-sway.
# Hip-surge joints are left free so only hip-sway/knees drive motion; ankles will mirror knees oppositely.
STATIC_POS_TARGETS: Dict[str, float] = {}


class _HeadlessViewer:
    """Stand-in for mujoco.viewer when running headless."""

    def __enter__(self) -> "_HeadlessViewer":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        return None


def _build_actuator_map(model: mujoco.MjModel) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            mapping[name] = i
    return mapping


def _joint_torques(model: mujoco.MjModel, data: mujoco.MjData, joints: Iterable[str]) -> Dict[str, float]:
    """Return actuator torques per joint using qfrc_actuator (works without explicit sensors)."""
    torques: Dict[str, float] = {}
    for name in joints:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = model.jnt_dofadr[jnt_id]
        jtype = model.jnt_type[jnt_id]
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            dof_count = 6
        elif jtype == mujoco.mjtJoint.mjJNT_BALL:
            dof_count = 3
        else:
            dof_count = 1  # hinge or slide
        torques[name] = float(np.sum(data.qfrc_actuator[adr:adr + dof_count]))
    return torques


def _set_joint_angles(model: mujoco.MjModel, data: mujoco.MjData, targets: Dict[str, float]) -> None:
    """Directly set joint angles for initialization."""
    for name, angle in targets.items():
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        adr = model.jnt_qposadr[jnt_id]
        data.qpos[adr] = angle
        data.qvel[adr] = 0.0
    mujoco.mj_forward(model, data)


def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")  # ensure plotting runs on main thread without macOS GUI windows
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "matplotlib and pandas are required. Install with "
            "'python3 -m pip install matplotlib pandas' (with PYTHONHOME/PYTHONPATH cleared if needed)."
        ) from exc

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    actuator_map = _build_actuator_map(model)
    required_actuators = [
        "hip-sway-pos",
        "hip-sway (1)-pos",
        "hip-surge-pos",
        "hip-surge (1)-pos",
        "knee-pos",
        "knee (1)-pos",
        "ankle-pos",
        "ankle (1)-pos",
    ]
    for name in required_actuators:
        if name not in actuator_map:
            raise SystemExit(f"Actuator '{name}' not found in the model.")

    base_rad = np.deg2rad(BASE_DEGREES)
    peak_rad = np.deg2rad(PEAK_DEGREES)

    # Initial pose: hip-sway/ankles at 0, hip-surge offset by base, knees offset opposite base.
    _set_joint_angles(
        model,
        data,
        {
            "hip-sway": 0.0,
            "hip-sway (1)": 0.0,
            "hip-surge": -0.8 + base_rad,
            "hip-surge (1)": -0.8 + base_rad,
            "knee": 0.5 - base_rad,
            "knee (1)": 0.5 - base_rad,
            "ankle": 0.0,
            "ankle (1)": 0.0,
        },
    )

    joints_to_log = ["hip-sway", "hip-sway (1)", "knee", "knee (1)", "ankle", "ankle (1)"]
    half_steps = int((CYCLE_DURATION / 2.0) / model.opt.timestep)
    if half_steps < 2:
        raise SystemExit("Cycle duration too short for the simulation timestep.")
    trajectory = np.concatenate(
        [
            np.linspace(base_rad, peak_rad, half_steps, endpoint=False),
            np.linspace(peak_rad, base_rad, half_steps, endpoint=True),
        ]
    )

    times: list[float] = []
    torque_log: Dict[str, list[float]] = {name: [] for name in joints_to_log}

    viewer_ctx = _HeadlessViewer() if HEADLESS else mujoco.viewer.launch_passive(model, data)
    with viewer_ctx as viewer:
        for target_angle in trajectory:
            if not viewer.is_running():
                break

            data.ctrl[:] = 0.0
            data.ctrl[actuator_map["hip-sway-pos"]] = 0.0
            data.ctrl[actuator_map["hip-sway (1)-pos"]] = 0.0
            data.ctrl[actuator_map["hip-surge-pos"]] = -0.8 
            data.ctrl[actuator_map["hip-surge (1)-pos"]] = -0.8 
            data.ctrl[actuator_map["knee-pos"]] = 0.5
            data.ctrl[actuator_map["knee (1)-pos"]] = 0.5
            data.ctrl[actuator_map["ankle-pos"]] = 0.0
            data.ctrl[actuator_map["ankle (1)-pos"]] = 0.0
            print(target_angle)
            for static_name, static_target in STATIC_POS_TARGETS.items():
                data.ctrl[actuator_map[static_name]] = static_target

            mujoco.mj_step(model, data)
            viewer.sync()

            times.append(data.time)
            torques = _joint_torques(model, data, joints_to_log)
            for name, value in torques.items():
                torque_log[name].append(value)

    if not times:
        print("No data recorded; viewer may have been closed immediately.")
        return

    plt.figure(figsize=(10, 6))
    for name in joints_to_log:
        plt.plot(times, torque_log[name], label=name.replace("_", " "))
    plt.xlabel("Time [s]")
    plt.ylabel("Measured actuator torque")
    plt.title("Torques during planar hip-sway/knee sweep")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)
    print(f"Saved torque plot to {OUTPUT_PNG}")

    df = pd.DataFrame({"time": times})
    for name in joints_to_log:
        df[name] = torque_log[name]
    df.to_excel(OUTPUT_XLSX, index=False)
    print(f"Saved torque data to {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
