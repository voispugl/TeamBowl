from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.mjx import pipeline

# Environment constants (tune as needed).
CTRL_COST_COEF = 1.0e-3
ACTION_DELAY_ALPHA = 0.2
DEFAULT_UNLIMITED_ACT_RANGE = 6.28
DEFAULT_UNLIMITED_MOTOR_RANGE = 1.0
ALIVE_REWARD = 0.001

QPOS_NOISE_STD = 1.0e-3
QVEL_NOISE_STD = 5.0e-2
GYRO_NOISE_STD = 2.0e-2
GYRO_BIAS_STD = 2.0e-2

TARGET_RADIUS = 5.0
FALL_HEIGHT = 0.6
TILT_RAD = 0.78
MAX_EPISODE_STEPS = 1000


class CustomNavEnv(PipelineEnv):
    """Goal-conditioned navigation environment with MJX physics and noise."""

    def __init__(self, xml_path: str = "robot/scene.xml", **kwargs: Any):
        # Keep a MuJoCo model for name/id lookups and sensor metadata,
        # and a Brax system for MJX stepping/training.
        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        sys = mjcf.load(xml_path)

        backend = kwargs.pop("backend", "mjx")
        n_frames = kwargs.pop("n_frames", 1)
        debug = kwargs.pop("debug", False)
        super().__init__(sys=sys, backend=backend, n_frames=n_frames, debug=debug)
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unknown}")

        self._torso_body_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "frame"
        )
        self._imu_site_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu"
        )
        self._floor_geom_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        contact_body_ids = [self._torso_body_id]
        for name in ("wheel", "wheel_2"):
            body_id = int(
                mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            )
            if body_id >= 0:
                contact_body_ids.append(body_id)
        geom_bodyid = np.asarray(self._mj_model.geom_bodyid)
        target_geom_ids = np.where(np.isin(geom_bodyid, contact_body_ids))[0]
        geom_mask = np.zeros((self._mj_model.ngeom,), dtype=bool)
        geom_mask[target_geom_ids] = True
        self._contact_geom_mask = jp.array(geom_mask)

        self._action_size = int(self.sys.nu)
        self._obs_size = int(self.sys.nq + self.sys.nv + 6 + 2)

        # Sensor slices for IMU.
        gyro_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro"
        )
        accel_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel"
        )
        self._gyro_slice = slice(
            int(self._mj_model.sensor_adr[gyro_id]),
            int(self._mj_model.sensor_adr[gyro_id] + self._mj_model.sensor_dim[gyro_id]),
        )
        self._accel_slice = slice(
            int(self._mj_model.sensor_adr[accel_id]),
            int(self._mj_model.sensor_adr[accel_id] + self._mj_model.sensor_dim[accel_id]),
        )

        # Root joint qpos indices (for noise masking).
        root_qpos = []
        for jnt_id, jnt_type in enumerate(self._mj_model.jnt_type):
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                adr = int(self._mj_model.jnt_qposadr[jnt_id])
                qnum = 7
                root_qpos.extend(range(adr, adr + qnum))
        root_qpos = jp.array(root_qpos, dtype=jp.int32)
        qpos_mask = jp.ones((self.sys.nq,), dtype=jp.bool_)
        if root_qpos.size > 0:
            qpos_mask = qpos_mask.at[root_qpos].set(False)
        self._non_root_qpos_mask = qpos_mask

        # Non-root dof mask for damping randomization.
        root_dofs = []
        dof_per_joint = {
            mujoco.mjtJoint.mjJNT_FREE: 6,
            mujoco.mjtJoint.mjJNT_BALL: 3,
            mujoco.mjtJoint.mjJNT_HINGE: 1,
            mujoco.mjtJoint.mjJNT_SLIDE: 1,
        }
        for jnt_id, jnt_type in enumerate(self._mj_model.jnt_type):
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                adr = int(self._mj_model.jnt_dofadr[jnt_id])
                dofnum = int(dof_per_joint.get(jnt_type, 0))
                root_dofs.extend(range(adr, adr + dofnum))
        root_dofs = jp.array(root_dofs, dtype=jp.int32)
        dof_mask = jp.ones((self.sys.nv,), dtype=jp.bool_)
        if root_dofs.size > 0:
            dof_mask = dof_mask.at[root_dofs].set(False)
        self._non_root_dof_mask = dof_mask

        # Action scaling references.
        ctrlrange = self._mj_model.actuator_ctrlrange
        self._actuator_ctrlrange = jp.array(ctrlrange, dtype=jp.float32)
        bias_type = self._mj_model.actuator_biastype
        self._actuator_biastype = jp.array(bias_type, dtype=jp.int32)
        trnid = self._mj_model.actuator_trnid[:, 0]
        jnt_range = self._mj_model.jnt_range[trnid]
        self._actuator_jnt_range = jp.array(jnt_range, dtype=jp.float32)

    @property
    def observation_size(self) -> int:  # type: ignore[override]
        return self._obs_size

    @property
    def action_size(self) -> int:  # type: ignore[override]
        return self._action_size

    def reset(self, rng: jp.ndarray) -> State:
        rng, rng_mass, rng_fric, rng_damp, rng_goal, rng_bias, rng_qpos, rng_qvel = (
            jax.random.split(rng, 8)
        )

        # Domain randomization.
        mass_scale = jax.random.uniform(rng_mass, (), minval=0.9, maxval=1.1)
        body_mass = self.sys.body_mass.at[self._torso_body_id].multiply(mass_scale)

        fric_scale = jax.random.uniform(rng_fric, (), minval=0.8, maxval=1.2)
        geom_friction = self.sys.geom_friction.at[self._floor_geom_id].multiply(
            fric_scale
        )

        log_low = jp.log(0.9)
        log_high = jp.log(1.1)
        log_scale = jax.random.uniform(
            rng_damp, (self.sys.nv,), minval=log_low, maxval=log_high
        )
        dof_scale = jp.exp(log_scale)
        dof_damping = jp.where(self._non_root_dof_mask, self.sys.dof_damping * dof_scale, self.sys.dof_damping)

        rand_sys = self.sys.replace(
            body_mass=body_mass,
            geom_friction=geom_friction,
            dof_damping=dof_damping,
        )

        # Initial state.
        qpos0 = getattr(self.sys, "qpos0", jp.zeros((self.sys.nq,)))
        qvel0 = jp.zeros((self.sys.nv,))
        qpos_noise = jax.random.normal(rng_qpos, qpos0.shape) * 1.0e-2
        qvel_noise = jax.random.normal(rng_qvel, qvel0.shape) * 1.0e-2
        qpos = qpos0 + qpos_noise * self._non_root_qpos_mask
        qvel = qvel0 + qvel_noise

        data = pipeline.init(rand_sys, qpos, qvel)

        # Random target in XY plane.
        rng_goal, rng_theta, rng_radius = jax.random.split(rng_goal, 3)
        theta = jax.random.uniform(rng_theta, (), minval=0.0, maxval=2.0 * jp.pi)
        radius = jax.random.uniform(rng_radius, (), minval=0.0, maxval=TARGET_RADIUS)
        target_pos = jp.array([radius * jp.cos(theta), radius * jp.sin(theta)])

        gyro_bias = jax.random.normal(rng_bias, (3,)) * GYRO_BIAS_STD

        rng, rng_obs = jax.random.split(rng)
        obs = self._get_obs(data, target_pos, gyro_bias, rng_obs)

        info = {
            "rng": rng,
            "target_pos": target_pos,
            "prev_action": jp.zeros((self.action_size,)),
            "gyro_bias": gyro_bias,
            "sys": rand_sys,
            "steps": jp.array(0, dtype=jp.int32),
        }
        metrics = {"distance": jp.linalg.norm(target_pos - data.xpos[self._torso_body_id][:2])}

        return State(
            pipeline_state=data,
            obs=obs,
            reward=jp.array(0.0),
            done=jp.array(0.0),
            metrics=metrics,
            info=info,
        )

    def step(self, state: State, action: jp.ndarray) -> State:
        rng = state.info["rng"]
        rng, rng_obs = jax.random.split(rng)

        prev_action = state.info["prev_action"]
        delayed = ACTION_DELAY_ALPHA * action + (1.0 - ACTION_DELAY_ALPHA) * prev_action
        delayed = jp.clip(delayed, -1.0, 1.0)
        ctrl = self._scale_action(delayed)

        sys = state.info["sys"]
        data = pipeline.step(sys, state.pipeline_state, ctrl)

        target_pos = state.info["target_pos"]
        gyro_bias = state.info["gyro_bias"]
        obs = self._get_obs(data, target_pos, gyro_bias, rng_obs)

        prev_pos = state.pipeline_state.xpos[self._torso_body_id][:2]
        new_pos = data.xpos[self._torso_body_id][:2]
        old_dist = jp.linalg.norm(target_pos - prev_pos)
        new_dist = jp.linalg.norm(target_pos - new_pos)

        ctrl_cost = CTRL_COST_COEF * jp.sum(jp.square(delayed))
        reward = (old_dist - new_dist) - ctrl_cost + ALIVE_REWARD

        upright = self._upright(data)

        ground_contact = self._ground_contact(data)
        nan_fail = jp.any(jp.isnan(data.qpos)) | jp.any(jp.isnan(data.qvel))

        steps = state.info["steps"] + 1
        timeout = steps >= MAX_EPISODE_STEPS
        done = ground_contact | nan_fail | timeout

        info = dict(state.info)
        info.update(
            {
                "rng": rng,
                "prev_action": action,
                "steps": steps,
            }
        )

        metrics = {
            "distance": new_dist,
            "upright": upright,
            "height": data.xpos[self._torso_body_id][2],
        }

        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done.astype(jp.float32),
            metrics=metrics,
            info=info,
        )

    def _upright(self, data: Any) -> jp.ndarray:
        # IMU site orientation; xmat is row-major rotation from site to world.
        xmat = data.site_xmat[self._imu_site_id]
        z_axis_world = jp.array([xmat[2], xmat[5], xmat[8]])
        return z_axis_world[2]

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
        # Global-to-body transform: rotate the global target vector by -yaw.
        pos = data.xpos[self._torso_body_id][:2]
        v = target_pos - pos
        xmat = data.xmat[self._torso_body_id]
        # xmat is row-major; column 0 is the body x-axis in world coordinates.
        yaw = jp.arctan2(xmat[3], xmat[0])
        c = jp.cos(yaw)
        s = jp.sin(yaw)
        x_local = c * v[0] + s * v[1]
        y_local = -s * v[0] + c * v[1]
        return jp.array([x_local, y_local])

    def _get_obs(
        self,
        data: Any,
        target_pos: jp.ndarray,
        gyro_bias: jp.ndarray,
        rng: jp.ndarray,
    ) -> jp.ndarray:
        rng, rng_qpos, rng_qvel, rng_gyro = jax.random.split(rng, 4)

        qpos = data.qpos
        qvel = data.qvel
        qpos = qpos + jax.random.normal(rng_qpos, qpos.shape) * QPOS_NOISE_STD
        qvel = qvel + jax.random.normal(rng_qvel, qvel.shape) * QVEL_NOISE_STD

        sensordata = data.sensordata
        gyro = sensordata[self._gyro_slice] + gyro_bias
        gyro = gyro + jax.random.normal(rng_gyro, gyro.shape) * GYRO_NOISE_STD
        accel = sensordata[self._accel_slice]

        local_target = self._local_target(data, target_pos)

        return jp.concatenate([qpos, qvel, gyro, accel, local_target])

    def _scale_action(self, action: jp.ndarray) -> jp.ndarray:
        ctrlrange = self._actuator_ctrlrange
        bias_type = self._actuator_biastype
        jnt_range = self._actuator_jnt_range

        ctrl_lo = ctrlrange[:, 0]
        ctrl_hi = ctrlrange[:, 1]
        ctrl_span = ctrl_hi - ctrl_lo
        jnt_lo = jnt_range[:, 0]
        jnt_hi = jnt_range[:, 1]
        jnt_span = jnt_hi - jnt_lo

        use_ctrl = ctrl_span > 0.0
        is_position = bias_type != 0
        use_jnt = (~use_ctrl) & is_position & (jnt_span > 0.0)

        scale = jp.where(
            use_ctrl,
            0.5 * ctrl_span,
            jp.where(
                use_jnt,
                0.5 * jnt_span,
                jp.where(is_position, DEFAULT_UNLIMITED_ACT_RANGE, DEFAULT_UNLIMITED_MOTOR_RANGE),
            ),
        )
        bias = jp.where(
            use_ctrl,
            0.5 * (ctrl_hi + ctrl_lo),
            jp.where(use_jnt, 0.5 * (jnt_hi + jnt_lo), 0.0),
        )
        return action * scale + bias


def create_env(xml_path: str = "robot/scene.xml", **kwargs: Any) -> CustomNavEnv:
    return CustomNavEnv(xml_path=xml_path, **kwargs)
