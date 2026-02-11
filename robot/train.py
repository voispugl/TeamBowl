from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Tuple

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.training import checkpoints
from mujoco import mjx

ROOT = Path(__file__).resolve().parent

HIDDEN_SIZES = (256, 256, 256)
LEARNING_RATE = 3e-4
ENTROPY_COST = 0.01
DISCOUNTING = 0.99
GAE_LAMBDA = 0.95
PPO_CLIP_EPS = 0.2
VALUE_LOSS_COEF = 0.5
GRAD_CLIP = 10.0
NUM_MINIBATCHES = 32
NUM_UPDATES_PER_BATCH = 4
UNROLL_LENGTH = 10
NUM_ENVS = 128
NUM_TIMESTEPS = 5_000_000
MAX_EPISODE_STEPS = 100000
OBS_NORM_EPS = 1e-6

UPRIGHT_BONUS_COEF = 1.0
CTRL_COST_COEF = 1.0e-3
ACTION_DELAY_ALPHA = 0.2
DEFAULT_UNLIMITED_ACT_RANGE = 6.28

QPOS_NOISE_STD = 1.0e-3
QVEL_NOISE_STD = 5.0e-2
GYRO_NOISE_STD = 2.0e-2
GYRO_BIAS_STD = 2.0e-2

TARGET_RADIUS = 5.0
FALL_HEIGHT = 0.25
TILT_RAD = 0.78


class PolicyMLP(nn.Module):
    hidden_sizes: Tuple[int, ...]
    action_size: int

    @nn.compact
    def __call__(self, x: jp.ndarray) -> jp.ndarray:
        for i, size in enumerate(self.hidden_sizes):
            x = nn.Dense(size, name=f"hidden_{i}")(x)
            x = nn.silu(x)
        return nn.Dense(self.action_size, name="out")(x)


class ValueMLP(nn.Module):
    hidden_sizes: Tuple[int, ...]

    @nn.compact
    def __call__(self, x: jp.ndarray) -> jp.ndarray:
        for i, size in enumerate(self.hidden_sizes):
            x = nn.Dense(size, name=f"hidden_{i}")(x)
            x = nn.silu(x)
        value = nn.Dense(1, name="out")(x)
        return jp.squeeze(value, axis=-1)


@struct.dataclass
class EnvState:
    data: Any
    target_pos: jp.ndarray
    prev_action: jp.ndarray
    gyro_bias: jp.ndarray
    steps: jp.ndarray


@struct.dataclass
class ObsStats:
    count: jp.ndarray
    mean: jp.ndarray
    var: jp.ndarray


@struct.dataclass
class RolloutBatch:
    obs_norm: jp.ndarray
    obs_raw: jp.ndarray
    actions: jp.ndarray
    log_probs: jp.ndarray
    rewards: jp.ndarray
    dones: jp.ndarray
    values: jp.ndarray


class MJXNavEnv:
    def __init__(self, xml_path: str):
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self._model = mjx.put_model(self._mj_model)
        self._data_template = mjx.make_data(self._mj_model)

        self._torso_body_id = int(
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "frame")
        )
        self._floor_geom_id = int(
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        )
        contact_body_ids = [self._torso_body_id]
        geom_bodyid = np.asarray(self._mj_model.geom_bodyid)
        target_geom_ids = np.where(np.isin(geom_bodyid, contact_body_ids))[0]
        self._contact_geom_ids = jp.array(target_geom_ids, dtype=jp.int32)
        geom_mask = np.zeros((self._mj_model.ngeom,), dtype=bool)
        geom_mask[target_geom_ids] = True
        self._contact_geom_mask = jp.array(geom_mask)
        self._imu_site_id = int(
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        )

        gyro_id = int(
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        )
        accel_id = int(
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        )
        self._gyro_slice = slice(
            int(self._mj_model.sensor_adr[gyro_id]),
            int(self._mj_model.sensor_adr[gyro_id] + self._mj_model.sensor_dim[gyro_id]),
        )
        self._accel_slice = slice(
            int(self._mj_model.sensor_adr[accel_id]),
            int(self._mj_model.sensor_adr[accel_id] + self._mj_model.sensor_dim[accel_id]),
        )

        self._action_size = int(self._mj_model.nu)
        self._obs_size = int(self._mj_model.nq + self._mj_model.nv + 6 + 2)

        root_qpos = []
        root_dofs = []
        for jnt_id, jnt_type in enumerate(self._mj_model.jnt_type):
            if int(jnt_type) == int(mujoco.mjtJoint.mjJNT_FREE):
                qadr = int(self._mj_model.jnt_qposadr[jnt_id])
                dadr = int(self._mj_model.jnt_dofadr[jnt_id])
                root_qpos.extend(range(qadr, qadr + 7))
                root_dofs.extend(range(dadr, dadr + 6))

        qpos_mask = jp.ones((self._mj_model.nq,), dtype=jp.float32)
        if root_qpos:
            qpos_mask = qpos_mask.at[jp.array(root_qpos, dtype=jp.int32)].set(0.0)
        self._non_root_qpos_mask = qpos_mask

        self._qpos0 = jp.array(self._mj_model.qpos0, dtype=jp.float32)
        self._qvel0 = jp.zeros((self._mj_model.nv,), dtype=jp.float32)

        ctrlrange = jp.array(self._mj_model.actuator_ctrlrange, dtype=jp.float32)
        trnid = self._mj_model.actuator_trnid[:, 0]
        jnt_range = jp.array(self._mj_model.jnt_range[trnid], dtype=jp.float32)
        ctrl_lo = ctrlrange[:, 0]
        ctrl_hi = ctrlrange[:, 1]
        ctrl_span = ctrl_hi - ctrl_lo
        jnt_lo = jnt_range[:, 0]
        jnt_hi = jnt_range[:, 1]
        jnt_span = jnt_hi - jnt_lo
        use_ctrl = ctrl_span > 0.0
        use_jnt = (~use_ctrl) & (jnt_span > 0.0)
        scale = jp.where(
            use_ctrl,
            0.5 * ctrl_span,
            jp.where(use_jnt, 0.5 * jnt_span, DEFAULT_UNLIMITED_ACT_RANGE),
        )
        bias = jp.where(
            use_ctrl,
            0.5 * (ctrl_hi + ctrl_lo),
            jp.where(use_jnt, 0.5 * (jnt_hi + jnt_lo), 0.0),
        )
        self._actuator_scale = scale
        self._actuator_bias = bias

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def obs_size(self) -> int:
        return self._obs_size

    def _scale_action(self, action: jp.ndarray) -> jp.ndarray:
        return action * self._actuator_scale + self._actuator_bias

    def _upright(self, data: Any) -> jp.ndarray:
        xmat = data.site_xmat[self._imu_site_id]
        return xmat[2, 2]

    def _ground_contact(self, data: Any) -> jp.ndarray:
        contact = data._impl.contact
        geoms = contact.geom
        g1 = geoms[:, 0]
        g2 = geoms[:, 1]
        valid = (g1 >= 0) & (g2 >= 0)
        floor = self._floor_geom_id
        floor_contact = (g1 == floor) | (g2 == floor)
        in_contact = contact.dist <= 0.0
        max_geom = self._contact_geom_mask.shape[0] - 1
        g1_clamped = jp.clip(g1, 0, max_geom)
        g2_clamped = jp.clip(g2, 0, max_geom)
        g1_target = self._contact_geom_mask[g1_clamped] & (g1 >= 0)
        g2_target = self._contact_geom_mask[g2_clamped] & (g2 >= 0)
        target_contact = g1_target | g2_target
        return jp.any(valid & in_contact & floor_contact & target_contact)

    def _local_target(self, data: Any, target_pos: jp.ndarray) -> jp.ndarray:
        pos = data.xpos[self._torso_body_id][:2]
        v = target_pos - pos
        xmat = data.xmat[self._torso_body_id]
        yaw = jp.arctan2(xmat[1, 0], xmat[0, 0])
        c = jp.cos(yaw)
        s = jp.sin(yaw)
        x_local = c * v[0] + s * v[1]
        y_local = -s * v[0] + c * v[1]
        return jp.array([x_local, y_local], dtype=jp.float32)

    def _get_obs(
        self,
        data: Any,
        target_pos: jp.ndarray,
        gyro_bias: jp.ndarray,
        rng: jp.ndarray,
    ) -> jp.ndarray:
        rng_qpos, rng_qvel, rng_gyro = jax.random.split(rng, 3)

        qpos = data.qpos + jax.random.normal(rng_qpos, data.qpos.shape) * QPOS_NOISE_STD
        qpos = self._qpos0 + (qpos - self._qpos0) * self._non_root_qpos_mask

        qvel = data.qvel + jax.random.normal(rng_qvel, data.qvel.shape) * QVEL_NOISE_STD

        sensordata = data.sensordata
        gyro = sensordata[self._gyro_slice] + gyro_bias
        gyro = gyro + jax.random.normal(rng_gyro, gyro.shape) * GYRO_NOISE_STD
        accel = sensordata[self._accel_slice]

        local_target = self._local_target(data, target_pos)
        return jp.concatenate([qpos, qvel, gyro, accel, local_target], axis=0)

    def reset(self, rng: jp.ndarray) -> tuple[EnvState, jp.ndarray]:
        rng_qpos, rng_qvel, rng_goal, rng_bias, rng_obs = jax.random.split(rng, 5)

        qpos_noise = jax.random.normal(rng_qpos, self._qpos0.shape) * 1.0e-2
        qvel_noise = jax.random.normal(rng_qvel, self._qvel0.shape) * 1.0e-2

        qpos = self._qpos0 + qpos_noise * self._non_root_qpos_mask
        qvel = self._qvel0 + qvel_noise

        data = self._data_template.replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=jp.zeros((self._action_size,), dtype=jp.float32),
        )
        data = mjx.forward(self._model, data)

        rng_theta, rng_radius = jax.random.split(rng_goal)
        theta = jax.random.uniform(rng_theta, (), minval=0.0, maxval=2.0 * jp.pi)
        radius = jax.random.uniform(rng_radius, (), minval=0.0, maxval=TARGET_RADIUS)
        target_pos = jp.array([radius * jp.cos(theta), radius * jp.sin(theta)], dtype=jp.float32)

        gyro_bias = jax.random.normal(rng_bias, (3,), dtype=jp.float32) * GYRO_BIAS_STD

        obs = self._get_obs(data, target_pos, gyro_bias, rng_obs)

        state = EnvState(
            data=data,
            target_pos=target_pos,
            prev_action=jp.zeros((self._action_size,), dtype=jp.float32),
            gyro_bias=gyro_bias,
            steps=jp.array(0, dtype=jp.int32),
        )
        return state, obs

    def step(
        self,
        state: EnvState,
        action: jp.ndarray,
        rng: jp.ndarray,
    ) -> tuple[EnvState, jp.ndarray, jp.ndarray, jp.ndarray]:
        delayed = ACTION_DELAY_ALPHA * action + (1.0 - ACTION_DELAY_ALPHA) * state.prev_action
        delayed = jp.clip(delayed, -1.0, 1.0)
        ctrl = self._scale_action(delayed)

        data = state.data.replace(ctrl=ctrl)
        data = mjx.step(self._model, data)

        obs = self._get_obs(data, state.target_pos, state.gyro_bias, rng)

        prev_pos = state.data.xpos[self._torso_body_id][:2]
        new_pos = data.xpos[self._torso_body_id][:2]
        old_dist = jp.linalg.norm(state.target_pos - prev_pos)
        new_dist = jp.linalg.norm(state.target_pos - new_pos)

        upright = self._upright(data)
        upright_bonus = UPRIGHT_BONUS_COEF * upright
        ctrl_cost = CTRL_COST_COEF * jp.sum(jp.square(delayed))
        reward = (old_dist - new_dist) + upright_bonus - ctrl_cost

        ground_contact = self._ground_contact(data)
        nan_fail = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))

        steps = state.steps + 1
        timeout = steps >= MAX_EPISODE_STEPS
        done = ground_contact | nan_fail | timeout

        next_state = EnvState(
            data=data,
            target_pos=state.target_pos,
            prev_action=action,
            gyro_bias=state.gyro_bias,
            steps=steps,
        )

        return next_state, obs, reward, done.astype(jp.float32)


def _tree_select(cond: jp.ndarray, new_tree: Any, old_tree: Any) -> Any:
    def select(new_leaf: jp.ndarray, old_leaf: jp.ndarray) -> jp.ndarray:
        cond_leaf = cond
        while cond_leaf.ndim < new_leaf.ndim:
            cond_leaf = cond_leaf[..., None]
        return jp.where(cond_leaf, new_leaf, old_leaf)

    return jax.tree_util.tree_map(select, new_tree, old_tree)


def _init_obs_stats(obs_size: int) -> ObsStats:
    return ObsStats(
        count=jp.array(1e-4, dtype=jp.float32),
        mean=jp.zeros((obs_size,), dtype=jp.float32),
        var=jp.ones((obs_size,), dtype=jp.float32),
    )


def _update_obs_stats(stats: ObsStats, obs_batch: jp.ndarray) -> ObsStats:
    batch = obs_batch.reshape((-1, obs_batch.shape[-1]))
    batch_count = jp.array(batch.shape[0], dtype=jp.float32)
    batch_mean = jp.mean(batch, axis=0)
    batch_var = jp.var(batch, axis=0)

    delta = batch_mean - stats.mean
    total_count = stats.count + batch_count

    new_mean = stats.mean + delta * (batch_count / total_count)
    m_a = stats.var * stats.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jp.square(delta) * (stats.count * batch_count / total_count)
    new_var = jp.maximum(m2 / total_count, 1e-8)

    return ObsStats(count=total_count, mean=new_mean, var=new_var)


def _normalize_obs(obs_raw: jp.ndarray, stats: ObsStats) -> jp.ndarray:
    std = jp.sqrt(stats.var + OBS_NORM_EPS)
    return (obs_raw - stats.mean) / std


def _gaussian_log_prob(pre_tanh: jp.ndarray, mean: jp.ndarray, log_std: jp.ndarray) -> jp.ndarray:
    inv_std = jp.exp(-log_std)
    return -0.5 * jp.sum(
        jp.square((pre_tanh - mean) * inv_std) + 2.0 * log_std + jp.log(2.0 * jp.pi),
        axis=-1,
    )


def _tanh_log_prob_from_pre_tanh(
    pre_tanh: jp.ndarray,
    action: jp.ndarray,
    mean: jp.ndarray,
    log_std: jp.ndarray,
) -> jp.ndarray:
    log_prob = _gaussian_log_prob(pre_tanh, mean, log_std)
    correction = jp.sum(jp.log(1.0 - jp.square(action) + 1e-6), axis=-1)
    return log_prob - correction


def _tanh_log_prob(action: jp.ndarray, mean: jp.ndarray, log_std: jp.ndarray) -> jp.ndarray:
    clipped = jp.clip(action, -0.999999, 0.999999)
    pre_tanh = jp.arctanh(clipped)
    return _tanh_log_prob_from_pre_tanh(pre_tanh, clipped, mean, log_std)


def _extract_policy_mlp(params: Any) -> list[tuple[Any, Any]]:
    if isinstance(params, Mapping) and "params" in params:
        params = params["params"]

    layers: list[tuple[str, Any, Any]] = []

    def visit(prefix: str, node: Any) -> None:
        if isinstance(node, Mapping) and "kernel" in node and "bias" in node:
            layers.append((prefix, node["kernel"], node["bias"]))
            return
        if isinstance(node, Mapping):
            for key, value in node.items():
                name = f"{prefix}/{key}" if prefix else key
                visit(name, value)

    visit("", params)
    if not layers:
        raise ValueError("No Dense layers found in policy params.")

    def sort_key(name: str) -> tuple[int, int]:
        tokens = name.replace("/", "_").split("_")
        for token in reversed(tokens):
            if token.isdigit():
                return (0, int(token))
        if "out" in name:
            return (2, 0)
        return (1, 0)

    layers.sort(key=lambda item: sort_key(item[0]))
    return [(w, b) for _name, w, b in layers]


def _render_scene_only(
    xml_path: Path,
    target_xy: Tuple[float, float],
    max_steps: int,
) -> None:
    import time

    import numpy as np
    from mujoco import viewer

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    target_xy_arr = np.array(target_xy, dtype=np.float64)

    with viewer.launch_passive(model, data) as v:
        geom = v.user_scn.geoms[0]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.08, 0.0, 0.0]),
            np.array([target_xy_arr[0], target_xy_arr[1], 0.05]),
            np.eye(3).flatten(),
            np.array([1.0, 0.0, 0.0, 1.0]),
        )
        v.user_scn.ngeom = 1

        steps = 0
        while v.is_running() and (max_steps <= 0 or steps < max_steps):
            if data.ctrl.size:
                data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            v.user_scn.geoms[0].pos = np.array([target_xy_arr[0], target_xy_arr[1], 0.05])
            v.sync()
            steps += 1
            time.sleep(model.opt.timestep)


def _render_policy(
    xml_path: Path,
    policy_params: Any,
    obs_mean: Any,
    obs_std: Any,
    target_xy: Tuple[float, float],
    max_steps: int,
) -> None:
    import time

    import numpy as np
    from mujoco import viewer

    def silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))

    def mlp_forward(x: np.ndarray, weights: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        for w, b in weights[:-1]:
            x = x @ w + b
            x = silu(x)
        w, b = weights[-1]
        return x @ w + b

    def scale_action(action: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
        ctrlrange = model.actuator_ctrlrange.copy()
        jnt_range = model.jnt_range[model.actuator_trnid[:, 0]].copy()

        ctrl_lo = ctrlrange[:, 0]
        ctrl_hi = ctrlrange[:, 1]
        ctrl_span = ctrl_hi - ctrl_lo
        jnt_lo = jnt_range[:, 0]
        jnt_hi = jnt_range[:, 1]
        jnt_span = jnt_hi - jnt_lo
        use_ctrl = ctrl_span > 0.0
        use_jnt = (~use_ctrl) & (jnt_span > 0.0)
        scale = np.where(
            use_ctrl,
            0.5 * ctrl_span,
            np.where(use_jnt, 0.5 * jnt_span, DEFAULT_UNLIMITED_ACT_RANGE),
        )
        bias = np.where(
            use_ctrl,
            0.5 * (ctrl_hi + ctrl_lo),
            np.where(use_jnt, 0.5 * (jnt_hi + jnt_lo), 0.0),
        )
        return action * scale + bias

    def local_target(model: mujoco.MjModel, data: mujoco.MjData, target_xy_arr: np.ndarray) -> np.ndarray:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "frame")
        pos_xy = data.xpos[body_id, :2]
        v = target_xy_arr - pos_xy
        xmat = data.xmat[body_id]
        yaw = np.arctan2(xmat[3], xmat[0])
        c = np.cos(yaw)
        s = np.sin(yaw)
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])

    weights = _extract_policy_mlp(policy_params)
    weights = [(np.array(w), np.array(b)) for w, b in weights]
    mean = np.array(obs_mean)
    std = np.array(obs_std)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    accel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
    gyro_slice = slice(model.sensor_adr[gyro_id], model.sensor_adr[gyro_id] + model.sensor_dim[gyro_id])
    accel_slice = slice(model.sensor_adr[accel_id], model.sensor_adr[accel_id] + model.sensor_dim[accel_id])

    target_xy_arr = np.array(target_xy, dtype=np.float64)
    prev_action = np.zeros((model.nu,), dtype=np.float64)

    with viewer.launch_passive(model, data) as v:
        geom = v.user_scn.geoms[0]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.08, 0.0, 0.0]),
            np.array([target_xy_arr[0], target_xy_arr[1], 0.05]),
            np.eye(3).flatten(),
            np.array([1.0, 0.0, 0.0, 1.0]),
        )
        v.user_scn.ngeom = 1

        steps = 0
        while v.is_running() and (max_steps <= 0 or steps < max_steps):
            qpos = data.qpos.copy()
            qvel = data.qvel.copy()
            gyro = data.sensordata[gyro_slice].copy()
            accel = data.sensordata[accel_slice].copy()
            local_xy = local_target(model, data, target_xy_arr)

            obs = np.concatenate([qpos, qvel, gyro, accel, local_xy])
            obs = (obs - mean) / std

            mean_action = mlp_forward(obs, weights)
            action = np.tanh(mean_action)

            delayed = 0.2 * action + 0.8 * prev_action
            delayed = np.clip(delayed, -1.0, 1.0)
            data.ctrl[:] = scale_action(delayed, model)

            mujoco.mj_step(model, data)
            v.user_scn.geoms[0].pos = np.array([target_xy_arr[0], target_xy_arr[1], 0.05])
            v.sync()

            prev_action = action
            steps += 1
            time.sleep(model.opt.timestep)


@jax.jit
def _compute_gae(
    rewards: jp.ndarray,
    dones: jp.ndarray,
    values: jp.ndarray,
    last_value: jp.ndarray,
) -> tuple[jp.ndarray, jp.ndarray]:
    def scan_fn(carry: tuple[jp.ndarray, jp.ndarray], xs: tuple[jp.ndarray, jp.ndarray, jp.ndarray]):
        gae, next_value = carry
        reward_t, done_t, value_t = xs
        delta = reward_t + DISCOUNTING * (1.0 - done_t) * next_value - value_t
        gae = delta + DISCOUNTING * GAE_LAMBDA * (1.0 - done_t) * gae
        return (gae, value_t), gae

    (_, _), advantages_rev = jax.lax.scan(
        scan_fn,
        (jp.zeros_like(last_value), last_value),
        (rewards[::-1], dones[::-1], values[::-1]),
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


def _flatten_time_env(x: jp.ndarray) -> jp.ndarray:
    return x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:])


def _format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    total = int(seconds + 0.5)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _debug_done_conditions(env: "MJXNavEnv", seed: int, samples: int = 4) -> None:
    rng = jax.random.PRNGKey(seed)
    heights: list[float] = []
    uprights: list[float] = []
    for _ in range(samples):
        rng, key = jax.random.split(rng)
        state, _ = env.reset(key)
        heights.append(float(state.data.xpos[env._torso_body_id][2]))
        uprights.append(float(env._upright(state.data)))

    height_min = min(heights)
    height_max = max(heights)
    height_mean = sum(heights) / len(heights)
    upright_min = min(uprights)
    upright_max = max(uprights)
    upright_mean = sum(uprights) / len(uprights)
    tilt_threshold = float(jp.cos(TILT_RAD))

    print("Done-condition debug (at reset):", flush=True)
    print(
        f"  height: mean={height_mean:.3f} min={height_min:.3f} max={height_max:.3f} "
        f"(FALL_HEIGHT={FALL_HEIGHT:.3f})",
        flush=True,
    )
    print(
        f"  upright: mean={upright_mean:.3f} min={upright_min:.3f} max={upright_max:.3f} "
        f"(cos(TILT_RAD)={tilt_threshold:.3f})",
        flush=True,
    )
    print(
        f"  fall_fail={height_mean < FALL_HEIGHT} tilt_fail={upright_mean < tilt_threshold}",
        flush=True,
    )
    print("  done uses frame ground_contact | nan | timeout", flush=True)
    ground_contact = env._ground_contact(env.reset(jax.random.PRNGKey(seed + 1))[0].data)
    print(f"  ground_contact_at_reset={bool(ground_contact)}", flush=True)


def _debug_rollout(env: "MJXNavEnv", seed: int, steps: int = 32) -> None:
    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    state, obs = env.reset(key)

    counts = {
        "done": 0,
        "ground": 0,
        "nan": 0,
        "timeout": 0,
    }

    for _ in range(max(1, steps)):
        rng, action_key, step_key, reset_key = jax.random.split(rng, 4)
        action = jax.random.uniform(
            action_key,
            (env.action_size,),
            minval=-1.0,
            maxval=1.0,
        )
        next_state, next_obs, _reward, done = env.step(state, action, step_key)

        ground_contact = bool(env._ground_contact(next_state.data))
        nan_fail = bool(
            jp.any(jp.isnan(next_state.data.qpos)) | jp.any(jp.isnan(next_state.data.qvel))
        )
        timeout = int(next_state.steps) >= MAX_EPISODE_STEPS

        if float(done) > 0.5:
            counts["done"] += 1
            counts["ground"] += int(ground_contact)
            counts["nan"] += int(nan_fail)
            counts["timeout"] += int(timeout)
            state, obs = env.reset(reset_key)
        else:
            state, obs = next_state, next_obs

    print("Done-condition debug (random rollout):", flush=True)
    print(f"  steps={steps} done={counts['done']}", flush=True)
    print(
        f"  ground={counts['ground']} nan={counts['nan']} timeout={counts['timeout']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default=str(ROOT / "scene.xml"))
    parser.add_argument("--checkpoint_dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_timesteps", type=int, default=NUM_TIMESTEPS)
    parser.add_argument("--num_envs", type=int, default=NUM_ENVS)
    parser.add_argument("--unroll_length", type=int, default=UNROLL_LENGTH)
    parser.add_argument("--num_minibatches", type=int, default=NUM_MINIBATCHES)
    parser.add_argument("--num_updates_per_batch", type=int, default=NUM_UPDATES_PER_BATCH)
    parser.add_argument(
        "--viewer_only",
        action="store_true",
        help="Open a MuJoCo viewer and exit without training.",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--render_on_start",
        action="store_true",
        help="Open a viewer immediately after initialization, before training updates.",
    )
    parser.add_argument(
        "--render_every_updates",
        type=int,
        default=0,
        help="If > 0, open a viewer every N training updates using the current policy.",
    )
    parser.add_argument("--render_steps", type=int, default=1000)
    parser.add_argument(
        "--checkpoint_every_updates",
        type=int,
        default=0,
        help="Save a checkpoint every N updates (0 disables).",
    )
    parser.add_argument(
        "--checkpoint_every_seconds",
        type=float,
        default=0.0,
        help="Save a checkpoint every N seconds (0 disables).",
    )
    parser.add_argument(
        "--log_compiles",
        action="store_true",
        help="Enable JAX compile logging (very verbose).",
    )
    parser.add_argument(
        "--compile_timing",
        action="store_true",
        help="Print timing for first-time JAX compilations.",
    )
    parser.add_argument(
        "--quiet_compiles",
        action="store_true",
        help="Reduce JAX compile log spam while keeping timing output.",
    )
    parser.add_argument("--render_target_x", type=float, default=3.0)
    parser.add_argument("--render_target_y", type=float, default=2.0)
    parser.add_argument(
        "--debug_dones",
        action="store_true",
        help="Print reset-time height/upright stats to debug done conditions.",
    )
    parser.add_argument(
        "--debug_rollout",
        action="store_true",
        help="Run a short random-action rollout and print done reasons.",
    )
    parser.add_argument(
        "--debug_rollout_steps",
        type=int,
        default=32,
        help="Number of steps for --debug_rollout.",
    )
    args = parser.parse_args()

    xml_path = Path(args.xml).expanduser()
    if not xml_path.exists():
        xml_path = ROOT / xml_path.name

    if args.viewer_only:
        print("Opening viewer without training ...", flush=True)
        _render_scene_only(
            xml_path=xml_path,
            target_xy=(args.render_target_x, args.render_target_y),
            max_steps=args.render_steps,
        )
        return

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = Path.cwd() / checkpoint_dir

    if args.quiet_compiles:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        logging.getLogger("jax").setLevel(logging.ERROR)
        logging.getLogger("jax._src.dispatch").setLevel(logging.ERROR)
        logging.getLogger("jax._src.interpreters.pxla").setLevel(logging.ERROR)

    if args.log_compiles:
        jax.config.update("jax_log_compiles", True)
        jax.config.update("jax_explain_cache_misses", True)

    print(f"Loading MJX model from {xml_path} ...", flush=True)
    env = MJXNavEnv(str(xml_path))
    print(f"MJX model loaded: obs_size={env.obs_size}, action_size={env.action_size}", flush=True)
    if args.debug_dones:
        _debug_done_conditions(env, args.seed)
    if args.debug_rollout:
        _debug_rollout(env, args.seed, steps=args.debug_rollout_steps)
    num_envs = int(args.num_envs)
    unroll_length = int(args.unroll_length)

    if num_envs <= 0:
        raise ValueError("--num_envs must be > 0")
    if unroll_length <= 0:
        raise ValueError("--unroll_length must be > 0")
    if args.render_every_updates < 0:
        raise ValueError("--render_every_updates must be >= 0")

    num_updates = max(1, args.num_timesteps // (num_envs * unroll_length))
    steps_per_update = num_envs * unroll_length
    print(
        f"Starting MJX PPO: envs={num_envs}, unroll={unroll_length}, "
        f"updates={num_updates}, total_steps≈{num_updates * num_envs * unroll_length}",
        flush=True,
    )
    start_time = time.perf_counter()
    compile_start: float | None = None
    first_rollout_timed = False
    first_minibatch_timed = False
    last_checkpoint_time = start_time

    policy_model = PolicyMLP(hidden_sizes=HIDDEN_SIZES, action_size=env.action_size)
    value_model = ValueMLP(hidden_sizes=HIDDEN_SIZES)

    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.adam(LEARNING_RATE),
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, policy_key, value_key, env_key = jax.random.split(rng, 4)

    dummy_obs = jp.zeros((env.obs_size,), dtype=jp.float32)
    params: dict[str, Any] = {
        "policy": policy_model.init(policy_key, dummy_obs),
        "value": value_model.init(value_key, dummy_obs),
        "log_std": jp.zeros((env.action_size,), dtype=jp.float32),
    }
    opt_state = optimizer.init(params)

    env_keys = jax.random.split(env_key, num_envs)
    env_state, obs_raw = jax.vmap(env.reset)(env_keys)

    obs_stats = _init_obs_stats(env.obs_size)
    obs_stats = _update_obs_stats(obs_stats, obs_raw)

    if args.render_on_start:
        print("Opening initial viewer before training ...", flush=True)
        obs_std_start = jp.sqrt(obs_stats.var + OBS_NORM_EPS)
        _render_policy(
            xml_path=xml_path,
            policy_params=params["policy"],
            obs_mean=obs_stats.mean,
            obs_std=obs_std_start,
            target_xy=(args.render_target_x, args.render_target_y),
            max_steps=args.render_steps,
        )

    @jax.jit
    def collect_rollout(
        params_in: dict[str, Any],
        obs_stats_in: ObsStats,
        env_state_in: EnvState,
        obs_raw_in: jp.ndarray,
        rng_in: jp.ndarray,
    ) -> tuple[EnvState, jp.ndarray, jp.ndarray, RolloutBatch]:
        log_std = params_in["log_std"]

        def step_fn(
            carry: tuple[EnvState, jp.ndarray, jp.ndarray],
            _unused: Any,
        ) -> tuple[tuple[EnvState, jp.ndarray, jp.ndarray], RolloutBatch]:
            env_state_t, obs_raw_t, rng_t = carry
            rng_t, action_key, step_key, reset_key = jax.random.split(rng_t, 4)

            obs_norm_t = _normalize_obs(obs_raw_t, obs_stats_in)
            mean_action = policy_model.apply(params_in["policy"], obs_norm_t)
            value_t = value_model.apply(params_in["value"], obs_norm_t)

            noise = jax.random.normal(action_key, mean_action.shape)
            pre_tanh = mean_action + jp.exp(log_std) * noise
            action_t = jp.tanh(pre_tanh)
            log_prob_t = _tanh_log_prob_from_pre_tanh(pre_tanh, action_t, mean_action, log_std)

            step_keys = jax.random.split(step_key, num_envs)
            next_state, next_obs_raw, reward_t, done_t = jax.vmap(env.step)(
                env_state_t,
                action_t,
                step_keys,
            )

            reset_keys = jax.random.split(reset_key, num_envs)
            reset_state, reset_obs_raw = jax.vmap(env.reset)(reset_keys)

            done_bool = done_t.astype(jp.bool_)
            next_state = _tree_select(done_bool, reset_state, next_state)
            next_obs_raw = _tree_select(done_bool, reset_obs_raw, next_obs_raw)

            transition = RolloutBatch(
                obs_norm=obs_norm_t,
                obs_raw=obs_raw_t,
                actions=action_t,
                log_probs=log_prob_t,
                rewards=reward_t,
                dones=done_t,
                values=value_t,
            )
            return (next_state, next_obs_raw, rng_t), transition

        (env_state_out, obs_raw_out, rng_out), rollout = jax.lax.scan(
            step_fn,
            (env_state_in, obs_raw_in, rng_in),
            xs=None,
            length=unroll_length,
        )
        return env_state_out, obs_raw_out, rng_out, rollout

    @jax.jit
    def train_minibatch(
        params_in: dict[str, Any],
        opt_state_in: Any,
        obs_mb: jp.ndarray,
        actions_mb: jp.ndarray,
        old_logp_mb: jp.ndarray,
        adv_mb: jp.ndarray,
        returns_mb: jp.ndarray,
    ) -> tuple[dict[str, Any], Any, dict[str, jp.ndarray]]:
        def loss_fn(p: dict[str, Any]) -> tuple[jp.ndarray, dict[str, jp.ndarray]]:
            mean_action = policy_model.apply(p["policy"], obs_mb)
            value = value_model.apply(p["value"], obs_mb)
            log_std = p["log_std"]

            new_logp = _tanh_log_prob(actions_mb, mean_action, log_std)
            ratio = jp.exp(new_logp - old_logp_mb)
            clipped_ratio = jp.clip(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS)

            policy_loss = -jp.mean(jp.minimum(ratio * adv_mb, clipped_ratio * adv_mb))
            value_loss = 0.5 * jp.mean(jp.square(returns_mb - value))
            entropy = jp.sum(log_std + 0.5 * (1.0 + jp.log(2.0 * jp.pi)))

            total_loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COST * entropy
            metrics = {
                "loss": total_loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
            }
            return total_loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_in)
        del loss
        updates, opt_state_out = optimizer.update(grads, opt_state_in, params_in)
        params_out = optax.apply_updates(params_in, updates)
        return params_out, opt_state_out, metrics

    last_metrics = {
        "loss": jp.array(0.0),
        "policy_loss": jp.array(0.0),
        "value_loss": jp.array(0.0),
        "entropy": jp.array(0.0),
    }

    total_steps = 0
    for update in range(num_updates):
        if update == 0:
            print("Compiling JAX (first update can take a few minutes)...", flush=True)
            compile_start = time.perf_counter()
        rollout_start = time.perf_counter()
        env_state, obs_raw, rng, rollout = collect_rollout(params, obs_stats, env_state, obs_raw, rng)
        if args.compile_timing and not first_rollout_timed:
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                rollout,
            )
            elapsed = time.perf_counter() - rollout_start
            print(f"JAX compile+run collect_rollout: {elapsed:.1f}s", flush=True)
            first_rollout_timed = True

        next_obs_norm = _normalize_obs(obs_raw, obs_stats)
        next_values = value_model.apply(params["value"], next_obs_norm)

        advantages, returns = _compute_gae(
            rollout.rewards,
            rollout.dones,
            rollout.values,
            next_values,
        )
        advantages = (advantages - jp.mean(advantages)) / (jp.std(advantages) + 1e-8)

        obs_batch = _flatten_time_env(rollout.obs_norm)
        actions_batch = _flatten_time_env(rollout.actions)
        old_logp_batch = _flatten_time_env(rollout.log_probs)
        adv_batch = _flatten_time_env(advantages)
        returns_batch = _flatten_time_env(returns)

        batch_size = int(obs_batch.shape[0])
        num_minibatches = max(1, min(int(args.num_minibatches), batch_size))
        minibatch_size = batch_size // num_minibatches
        trim = minibatch_size * num_minibatches

        obs_batch = obs_batch[:trim]
        actions_batch = actions_batch[:trim]
        old_logp_batch = old_logp_batch[:trim]
        adv_batch = adv_batch[:trim]
        returns_batch = returns_batch[:trim]

        for _epoch in range(int(args.num_updates_per_batch)):
            rng, perm_key = jax.random.split(rng)
            perm = jax.random.permutation(perm_key, trim)

            obs_perm = obs_batch[perm].reshape((num_minibatches, minibatch_size, -1))
            actions_perm = actions_batch[perm].reshape((num_minibatches, minibatch_size, -1))
            old_logp_perm = old_logp_batch[perm].reshape((num_minibatches, minibatch_size))
            adv_perm = adv_batch[perm].reshape((num_minibatches, minibatch_size))
            returns_perm = returns_batch[perm].reshape((num_minibatches, minibatch_size))

            for mb in range(num_minibatches):
                if args.compile_timing and not first_minibatch_timed:
                    t0 = time.perf_counter()
                    params, opt_state, last_metrics = train_minibatch(
                        params,
                        opt_state,
                        obs_perm[mb],
                        actions_perm[mb],
                        old_logp_perm[mb],
                        adv_perm[mb],
                        returns_perm[mb],
                    )
                    jax.tree_util.tree_map(
                        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                        last_metrics,
                    )
                    elapsed = time.perf_counter() - t0
                    print(f"JAX compile+run train_minibatch: {elapsed:.1f}s", flush=True)
                    first_minibatch_timed = True
                else:
                    params, opt_state, last_metrics = train_minibatch(
                        params,
                        opt_state,
                        obs_perm[mb],
                        actions_perm[mb],
                        old_logp_perm[mb],
                        adv_perm[mb],
                        returns_perm[mb],
                    )

        obs_for_stats = jp.concatenate([_flatten_time_env(rollout.obs_raw), obs_raw], axis=0)
        obs_stats = _update_obs_stats(obs_stats, obs_for_stats)

        total_steps += steps_per_update

        elapsed = time.perf_counter() - start_time
        sps = (total_steps / elapsed) if elapsed > 0 else 0.0
        remaining_steps = (num_updates - (update + 1)) * steps_per_update
        eta = (remaining_steps / sps) if sps > 0 else float("inf")
        status_line = (
            f"progress {update + 1}/{num_updates} "
            f"steps={total_steps} "
            f"sps={sps:,.0f} "
            f"eta={_format_eta(eta)}"
        )
        print(status_line.ljust(120), end="\r", flush=True)

        if update == 0 or (update + 1) % 10 == 0 or update + 1 == num_updates:
            mean_reward = float(jp.mean(rollout.rewards))
            done_rate = float(jp.mean(rollout.dones))
            print(flush=True)
            print(
                f"update {update + 1}/{num_updates} "
                f"steps={total_steps} "
                f"reward={mean_reward:.4f} "
                f"done_rate={done_rate:.4f} "
                f"loss={float(last_metrics['loss']):.4f}"
            , flush=True)
        if update == 0 and compile_start is not None:
            jax.tree_util.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                last_metrics,
            )
            elapsed = time.perf_counter() - compile_start
            print(f"JAX compile+first update completed in {elapsed:.1f}s.", flush=True)
            compile_start = None

        should_ckpt_update = (
            args.checkpoint_every_updates > 0
            and (update + 1) % args.checkpoint_every_updates == 0
        )
        now = time.perf_counter()
        should_ckpt_time = (
            args.checkpoint_every_seconds > 0.0
            and (now - last_checkpoint_time) >= args.checkpoint_every_seconds
        )
        if should_ckpt_update or should_ckpt_time:
            obs_std_now = jp.sqrt(obs_stats.var + OBS_NORM_EPS)
            payload = {
                "params": {
                    "policy": params["policy"],
                    "value": params["value"],
                    "log_std": params["log_std"],
                },
                "normalizer": {
                    "mean": obs_stats.mean,
                    "std": obs_std_now,
                },
            }
            checkpoints.save_checkpoint(
                str(checkpoint_dir),
                payload,
                step=total_steps,
                overwrite=True,
            )
            last_checkpoint_time = now

        should_render_periodic = (
            args.render_every_updates > 0
            and (update + 1) % args.render_every_updates == 0
            and not (args.render and (update + 1) == num_updates)
        )
        if should_render_periodic:
            obs_std_now = jp.sqrt(obs_stats.var + OBS_NORM_EPS)
            print(
                f"Opening eval viewer at update {update + 1}/{num_updates} ...",
                flush=True,
            )
            _render_policy(
                xml_path=xml_path,
                policy_params=params["policy"],
                obs_mean=obs_stats.mean,
                obs_std=obs_std_now,
                target_xy=(args.render_target_x, args.render_target_y),
                max_steps=args.render_steps,
            )

    obs_std = jp.sqrt(obs_stats.var + OBS_NORM_EPS)
    payload: dict[str, Any] = {
        "params": {
            "policy": params["policy"],
            "value": params["value"],
            "log_std": params["log_std"],
        },
        "normalizer": {
            "mean": obs_stats.mean,
            "std": obs_std,
        },
    }

    checkpoints.save_checkpoint(
        str(checkpoint_dir),
        payload,
        step=total_steps,
        overwrite=True,
    )

    total_time = time.perf_counter() - start_time
    avg_sps = (total_steps / total_time) if total_time > 0 else 0.0
    print(
        f"Training complete in {total_time:.1f}s "
        f"({avg_sps:,.0f} steps/sec avg).",
        flush=True,
    )

    if args.render:
        _render_policy(
            xml_path=xml_path,
            policy_params=params["policy"],
            obs_mean=obs_stats.mean,
            obs_std=obs_std,
            target_xy=(args.render_target_x, args.render_target_y),
            max_steps=args.render_steps,
        )


if __name__ == "__main__":
    main()
