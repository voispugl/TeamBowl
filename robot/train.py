from __future__ import annotations

import argparse
import inspect
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jp
from flax import linen as nn
from flax.training import checkpoints

from brax.training import distribution
from brax.training import networks as brax_networks
from brax.training.agents import ppo

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from custom_nav_env import CustomNavEnv

HIDDEN_SIZES = (256, 256, 256)
LEARNING_RATE = 3e-4
ENTROPY_COST = 0.01
DISCOUNTING = 0.99
GRAD_CLIP = 10.0
BATCH_SIZE = 2048
NUM_MINIBATCHES = 32
NUM_UPDATES_PER_BATCH = 4
UNROLL_LENGTH = 10
NUM_ENVS = 2048
NUM_TIMESTEPS = 50_000_000
EPISODE_LENGTH = 1000


class MLP(nn.Module):
    layer_sizes: Tuple[int, ...]

    @nn.compact
    def __call__(self, x: jp.ndarray) -> jp.ndarray:
        for i, size in enumerate(self.layer_sizes[:-1]):
            x = nn.Dense(size, name=f"hidden_{i}")(x)
            x = nn.silu(x)
        x = nn.Dense(self.layer_sizes[-1], name="out")(x)
        return x


def _make_networks(obs_size: int, action_size: int):
    param_dist = distribution.NormalTanhDistribution(event_size=action_size)
    policy_sizes = (*HIDDEN_SIZES, param_dist.param_size)
    value_sizes = (*HIDDEN_SIZES, 1)

    policy_module = MLP(policy_sizes)
    value_module = MLP(value_sizes)

    def policy_init(rng: jp.ndarray) -> Dict[str, Any]:
        return policy_module.init(rng, jp.zeros((obs_size,)))

    def policy_apply(params: Dict[str, Any], obs: jp.ndarray) -> jp.ndarray:
        return policy_module.apply(params, obs)

    def value_init(rng: jp.ndarray) -> Dict[str, Any]:
        return value_module.init(rng, jp.zeros((obs_size,)))

    def value_apply(params: Dict[str, Any], obs: jp.ndarray) -> jp.ndarray:
        return value_module.apply(params, obs)

    policy_network = brax_networks.FeedForwardNetwork(policy_init, policy_apply)
    value_network = brax_networks.FeedForwardNetwork(value_init, value_apply)

    try:
        from brax.training.agents.ppo import networks as ppo_networks

        return ppo_networks.PPONetworks(policy_network, value_network, param_dist)
    except Exception:
        return brax_networks.PPONetworks(policy_network, value_network, param_dist)


def _extract_normalizer(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        for key in ("normalizer", "obs_normalizer"):
            if key in obj:
                return obj[key]
    for key in ("normalizer", "obs_normalizer"):
        if hasattr(obj, key):
            return getattr(obj, key)
    return None


def _get_normalizer_mean_std(normalizer: Any) -> Tuple[jp.ndarray, jp.ndarray] | None:
    if normalizer is None:
        return None
    if isinstance(normalizer, Mapping):
        mean = normalizer.get("mean")
        std = normalizer.get("std")
        var = normalizer.get("var")
    else:
        mean = getattr(normalizer, "mean", None)
        std = getattr(normalizer, "std", None)
        var = getattr(normalizer, "var", None)
    if std is None and var is not None:
        std = jp.sqrt(var + 1e-8)
    if mean is None or std is None:
        return None
    return mean, std


def _flatten_dense_layers(params: Any) -> list[tuple[str, Any, Any]]:
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
    return layers


def _layer_sort_key(name: str) -> Tuple[int, int]:
    tokens = name.replace("/", "_").split("_")
    for token in reversed(tokens):
        if token.isdigit():
            return (0, int(token))
    if "out" in name or "mean" in name:
        return (2, 0)
    return (1, 0)


def _extract_policy_mlp(params: Any) -> list[tuple[Any, Any]]:
    policy = params
    if isinstance(params, Mapping):
        for key in ("policy", "actor"):
            if key in params:
                policy = params[key]
                break

    layers = _flatten_dense_layers(policy)
    if not layers:
        raise ValueError("No Dense layers found in policy params.")

    layers.sort(key=lambda item: _layer_sort_key(item[0]))
    return [(w, b) for _name, w, b in layers]


def _render_policy(
    xml_path: Path,
    params: Any,
    obs_mean: Any,
    obs_std: Any,
    target_xy: Tuple[float, float],
    max_steps: int,
) -> None:
    import time

    import mujoco
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

    weights = _extract_policy_mlp(params)
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
        geom = mujoco.MjvGeom()
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.08, 0.0, 0.0]),
            np.array([target_xy_arr[0], target_xy_arr[1], 0.05]),
            np.eye(3).flatten(),
            np.array([1.0, 0.0, 0.0, 1.0]),
        )
        v.user_scn.geoms[0] = geom
        v.user_scn.ngeom = 1

        steps = 0
        while v.is_running() and steps < max_steps:
            qpos = data.qpos.copy()
            qvel = data.qvel.copy()
            gyro = data.sensordata[gyro_slice].copy()
            accel = data.sensordata[accel_slice].copy()
            local_xy = local_target(model, data, target_xy_arr)

            obs = np.concatenate([qpos, qvel, gyro, accel, local_xy])
            obs = (obs - mean) / std

            raw_out = mlp_forward(obs, weights)
            mean_action = raw_out[: model.nu]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="robot/scene.xml")
    parser.add_argument("--checkpoint_dir", default="robot/checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_steps", type=int, default=1000)
    parser.add_argument("--render_target_x", type=float, default=3.0)
    parser.add_argument("--render_target_y", type=float, default=2.0)
    args = parser.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        xml_path = ROOT / xml_path.name
    env = CustomNavEnv(xml_path=str(xml_path))

    train_fn = ppo.train
    sig = inspect.signature(train_fn)
    kwargs: Dict[str, Any] = {
        "environment": env,
        "num_timesteps": NUM_TIMESTEPS,
        "num_envs": NUM_ENVS,
        "episode_length": EPISODE_LENGTH,
        "learning_rate": LEARNING_RATE,
        "entropy_cost": ENTROPY_COST,
        "discounting": DISCOUNTING,
        "batch_size": BATCH_SIZE,
        "num_minibatches": NUM_MINIBATCHES,
        "num_updates_per_batch": NUM_UPDATES_PER_BATCH,
        "unroll_length": UNROLL_LENGTH,
        "seed": args.seed,
    }

    if "num_eval_envs" in sig.parameters:
        kwargs["num_eval_envs"] = max(1, NUM_ENVS // 16)
    if "normalize_observations" in sig.parameters:
        kwargs["normalize_observations"] = True
    if "grad_clip" in sig.parameters:
        kwargs["grad_clip"] = GRAD_CLIP
    if "grad_clipping" in sig.parameters:
        kwargs["grad_clipping"] = GRAD_CLIP
    if "max_grad_norm" in sig.parameters:
        kwargs["max_grad_norm"] = GRAD_CLIP

    if "network_factory" in sig.parameters:
        kwargs["network_factory"] = _make_networks
    else:
        networks = _make_networks(env.observation_size, env.action_size)
        if "policy_network" in sig.parameters:
            kwargs["policy_network"] = networks.policy_network
        if "value_network" in sig.parameters:
            kwargs["value_network"] = networks.value_network
        if "parametric_action_distribution" in sig.parameters:
            kwargs["parametric_action_distribution"] = networks.parametric_action_distribution

    for ckpt_key in ("save_checkpoint_path", "checkpoint_path", "checkpoint_dir"):
        if ckpt_key in sig.parameters:
            kwargs[ckpt_key] = args.checkpoint_dir
            break

    result = train_fn(**kwargs)

    make_inference_fn = None
    params = None
    metrics = None
    training_state = None

    if isinstance(result, tuple):
        if len(result) >= 1:
            make_inference_fn = result[0]
        if len(result) >= 2:
            params = result[1]
        if len(result) >= 3:
            metrics = result[2]
        if len(result) >= 4:
            training_state = result[3]

    normalizer = _extract_normalizer(training_state) or _extract_normalizer(metrics)
    mean_std = _get_normalizer_mean_std(normalizer)

    if params is None:
        raise RuntimeError("PPO training did not return policy params.")

    payload: Dict[str, Any] = {"params": params}
    if mean_std is not None:
        payload["normalizer"] = {"mean": mean_std[0], "std": mean_std[1]}

    checkpoints.save_checkpoint(
        args.checkpoint_dir, payload, step=NUM_TIMESTEPS, overwrite=True
    )

    if args.render:
        if mean_std is None:
            raise RuntimeError(
                "Cannot render without observation normalizer stats. "
                "Ensure normalize_observations=True is supported by your Brax version."
            )
        _render_policy(
            xml_path=xml_path,
            params=params,
            obs_mean=mean_std[0],
            obs_std=mean_std[1],
            target_xy=(args.render_target_x, args.render_target_y),
            max_steps=args.render_steps,
        )


if __name__ == "__main__":
    main()
