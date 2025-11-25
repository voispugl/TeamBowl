#!/usr/bin/env python3
"""Episodic ARS-style learner that balances the base IMU with motor torques."""

from __future__ import annotations

import os
import time
from typing import Tuple

import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = os.path.join(os.path.dirname(__file__), "scene.xml")
MAX_TORQUE = 120.0
STATUS_PERIOD = 0.5
FLIP_RESET_DOT_THRESHOLD = -0.2  # negative z-axis alignment triggers reset
ANKLE_TORQUE_LIMIT = 10.0  # keep ankle torques gentle
EPISODE_STEPS = 800
EXPLORE_STD = 0.4
STEP_SIZE = 0.08
NUM_DIRECTIONS = 6
TOP_DIRECTIONS = 4
DEATH_PENALTY = 50.0


def _sensor_span(model: mujoco.MjModel, name: str) -> Tuple[int, int]:
    """Return (start, length) slice info for a named sensor."""
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    start = model.sensor_adr[sensor_id]
    length = model.sensor_dim[sensor_id]
    return start, length


def _imu_cost(model: mujoco.MjModel, data: mujoco.MjData, accel_span: Tuple[int, int],
              gyro_span: Tuple[int, int]) -> float:
    """Squared error between measured and target gravity / angular rate."""
    start_accel, len_accel = accel_span
    start_gyro, len_gyro = gyro_span
    accel = data.sensordata[start_accel:start_accel + len_accel]
    gyro = data.sensordata[start_gyro:start_gyro + len_gyro]
    gravity_mag = float(np.linalg.norm(model.opt.gravity))
    target_accel = np.array([0.0, 0.0, -gravity_mag])
    tilt_error = accel - target_accel
    return 0.5 * (tilt_error @ tilt_error + 0.1 * (gyro @ gyro))


def _get_obs(data: mujoco.MjData, accel_span: Tuple[int, int], gyro_span: Tuple[int, int]) -> np.ndarray:
    start_accel, len_accel = accel_span
    start_gyro, len_gyro = gyro_span
    accel = data.sensordata[start_accel:start_accel + len_accel]
    gyro = data.sensordata[start_gyro:start_gyro + len_gyro]
    return np.concatenate([accel, gyro])


def _apply_symmetry_and_limits(ctrl: np.ndarray, hip_sway_id: int, hip_sway_2_id: int, hip_surge_id: int,
                               hip_surge_2_id: int, ankle_id: int, ankle_2_id: int) -> None:
    """Force hips to be inverses and bound ankle torques."""
    sway_avg = 0.5 * (ctrl[hip_sway_id] - ctrl[hip_sway_2_id])
    ctrl[hip_sway_id] = sway_avg
    ctrl[hip_sway_2_id] = -sway_avg
    surge_avg = 0.5 * (ctrl[hip_surge_id] - ctrl[hip_surge_2_id])
    ctrl[hip_surge_id] = surge_avg
    ctrl[hip_surge_2_id] = -surge_avg
    np.clip(ctrl, -MAX_TORQUE, MAX_TORQUE, out=ctrl)
    ctrl[ankle_id] = np.clip(ctrl[ankle_id], -ANKLE_TORQUE_LIMIT, ANKLE_TORQUE_LIMIT)
    ctrl[ankle_2_id] = np.clip(ctrl[ankle_2_id], -ANKLE_TORQUE_LIMIT, ANKLE_TORQUE_LIMIT)


def _is_flipped(data: mujoco.MjData, imu_site_id: int) -> bool:
    imu_mat = data.site_xmat[imu_site_id].reshape(3, 3)
    z_up_dot = imu_mat[2, 2]
    return z_up_dot < FLIP_RESET_DOT_THRESHOLD


def _rollout(model: mujoco.MjModel, data: mujoco.MjData, policy: np.ndarray, accel_span: Tuple[int, int],
             gyro_span: Tuple[int, int], imu_site_id: int, hip_ids: tuple[int, int, int, int],
             ankle_ids: tuple[int, int], viewer: mujoco.viewer.Handle, base_qpos: np.ndarray,
             base_qvel: np.ndarray) -> float:
    """Run one episode and return cumulative reward."""
    mujoco.mj_resetData(model, data)
    data.qpos[:] = base_qpos
    data.qvel[:] = base_qvel
    mujoco.mj_forward(model, data)

    hip_sway_id, hip_sway_2_id, hip_surge_id, hip_surge_2_id = hip_ids
    ankle_id, ankle_2_id = ankle_ids

    total_reward = 0.0
    for _ in range(EPISODE_STEPS):
        obs = _get_obs(data, accel_span, gyro_span)
        ctrl = policy @ obs
        _apply_symmetry_and_limits(ctrl, hip_sway_id, hip_sway_2_id, hip_surge_id, hip_surge_2_id, ankle_id, ankle_2_id)
        data.ctrl[:] = ctrl

        mujoco.mj_step(model, data)
        viewer.sync()

        if _is_flipped(data, imu_site_id):
            total_reward -= DEATH_PENALTY
            break

        total_reward -= _imu_cost(model, data, accel_span, gyro_span)

    return total_reward


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    accel_span = _sensor_span(model, "imu_accel")
    gyro_span = _sensor_span(model, "imu_gyro")
    imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_weight")

    obs_dim = _get_obs(data, accel_span, gyro_span).shape[0]
    policy = np.zeros((model.nu, obs_dim), dtype=float)
    last_status = time.time()
    qpos0 = np.copy(data.qpos)
    qvel0 = np.copy(data.qvel)

    # Actuator ids for symmetry constraints and ankle limiting.
    hip_sway_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip-sway")
    hip_sway_2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip-sway (1)")
    hip_surge_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip-surge")
    hip_surge_2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip-surge (1)")
    ankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ankle")
    ankle_2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ankle (1)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            iteration = 0
            while viewer.is_running():
                rollouts: list[tuple[float, float, np.ndarray]] = []
                for _ in range(NUM_DIRECTIONS):
                    if not viewer.is_running():
                        break
                    noise = np.random.randn(*policy.shape) * EXPLORE_STD
                    r_pos = _rollout(
                        model, data, policy + noise, accel_span, gyro_span, imu_site_id,
                        (hip_sway_id, hip_sway_2_id, hip_surge_id, hip_surge_2_id),
                        (ankle_id, ankle_2_id), viewer, qpos0, qvel0,
                    )
                    if not viewer.is_running():
                        break
                    r_neg = _rollout(
                        model, data, policy - noise, accel_span, gyro_span, imu_site_id,
                        (hip_sway_id, hip_sway_2_id, hip_surge_id, hip_surge_2_id),
                        (ankle_id, ankle_2_id), viewer, qpos0, qvel0,
                    )
                    rollouts.append((r_pos, r_neg, noise))

                if not rollouts or not viewer.is_running():
                    break

                rollouts.sort(key=lambda r: max(r[0], r[1]), reverse=True)
                rollouts = rollouts[:TOP_DIRECTIONS]

                all_rewards = [r for pair in rollouts for r in pair[:2]]
                reward_std = np.std(all_rewards) + 1e-6

                update = sum((r_pos - r_neg) * noise for r_pos, r_neg, noise in rollouts)
                policy += (STEP_SIZE / (TOP_DIRECTIONS * reward_std)) * update

                iteration += 1
                now = time.time()
                if now - last_status > STATUS_PERIOD:
                    best = max(max(r[0], r[1]) for r in rollouts)
                    print(f"iter={iteration} best_reward={best:7.2f} "
                          f"reward_std={reward_std:6.3f}")
                    last_status = now
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
