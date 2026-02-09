from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import mujoco
import numpy as np
from mujoco import viewer

ACTION_DELAY_ALPHA = 0.2


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def mlp_forward(x: np.ndarray, params: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    for w, b in params[:-1]:
        x = x @ w + b
        x = silu(x)
    w, b = params[-1]
    return x @ w + b


def scale_action(action: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    ctrlrange = model.actuator_ctrlrange.copy()
    jnt_range = model.jnt_range[model.actuator_trnid[:, 0]].copy()

    ctrl_lo = ctrlrange[:, 0]
    ctrl_hi = ctrlrange[:, 1]
    ctrl_span = ctrl_hi - ctrl_lo
    use_ctrl = ctrl_span > 0.0
    ctrl = 0.5 * (action + 1.0) * ctrl_span + ctrl_lo

    jnt_lo = jnt_range[:, 0]
    jnt_hi = jnt_range[:, 1]
    jnt_span = jnt_hi - jnt_lo
    use_jnt = jnt_span > 0.0
    jnt_ctrl = 0.5 * (action + 1.0) * jnt_span + jnt_lo

    return np.where(use_ctrl, ctrl, np.where(use_jnt, jnt_ctrl, action))


def local_target(model: mujoco.MjModel, data: mujoco.MjData, target_xy: np.ndarray) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "frame")
    pos_xy = data.xpos[body_id, :2]
    v = target_xy - pos_xy

    # xmat is row-major; column 0 is the body x-axis in world coordinates.
    xmat = data.xmat[body_id]
    yaw = np.arctan2(xmat[3], xmat[0])

    # Rotate global vector by -yaw to get body-frame target coordinates.
    c = np.cos(yaw)
    s = np.sin(yaw)
    x_local = c * v[0] + s * v[1]
    y_local = -s * v[0] + c * v[1]
    return np.array([x_local, y_local])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="robot/scene.xml")
    parser.add_argument("--policy", default="robot/policy.pkl")
    parser.add_argument("--target_x", type=float, default=3.0)
    parser.add_argument("--target_y", type=float, default=2.0)
    args = parser.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        xml_path = Path(__file__).resolve().parent / xml_path.name
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    policy_path = Path(args.policy)
    if not policy_path.exists():
        policy_path = Path(__file__).resolve().parent / policy_path.name
    with open(policy_path, "rb") as f:
        payload = pickle.load(f)

    params = payload["params"]
    obs_mean = np.array(payload["mean"])
    obs_std = np.array(payload["std"])

    action_size = model.nu

    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    accel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
    gyro_slice = slice(model.sensor_adr[gyro_id], model.sensor_adr[gyro_id] + model.sensor_dim[gyro_id])
    accel_slice = slice(model.sensor_adr[accel_id], model.sensor_adr[accel_id] + model.sensor_dim[accel_id])

    target_xy = np.array([args.target_x, args.target_y], dtype=np.float64)
    prev_action = np.zeros((action_size,), dtype=np.float64)

    with viewer.launch_passive(model, data) as v:
        geom = mujoco.MjvGeom()
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.08, 0.0, 0.0]),
            np.array([target_xy[0], target_xy[1], 0.05]),
            np.eye(3).flatten(),
            np.array([1.0, 0.0, 0.0, 1.0]),
        )
        v.user_scn.geoms[0] = geom
        v.user_scn.ngeom = 1

        while v.is_running():
            qpos = data.qpos.copy()
            qvel = data.qvel.copy()
            gyro = data.sensordata[gyro_slice].copy()
            accel = data.sensordata[accel_slice].copy()
            local_xy = local_target(model, data, target_xy)

            obs = np.concatenate([qpos, qvel, gyro, accel, local_xy])
            obs = (obs - obs_mean) / obs_std

            raw_out = mlp_forward(obs, params)
            mean = raw_out[:action_size]
            action = np.tanh(mean)

            delayed = ACTION_DELAY_ALPHA * action + (1.0 - ACTION_DELAY_ALPHA) * prev_action
            delayed = np.clip(delayed, -1.0, 1.0)
            ctrl = scale_action(delayed, model)
            data.ctrl[:] = ctrl

            mujoco.mj_step(model, data)
            v.user_scn.geoms[0].pos = np.array([target_xy[0], target_xy[1], 0.05])
            v.sync()

            prev_action = action

            # Real-time pacing based on the model timestep.
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
