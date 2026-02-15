#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import mujoco
from flax.training import checkpoints
from mujoco import viewer

ACTION_DELAY_ALPHA = 0.2
DEFAULT_UNLIMITED_ACT_RANGE = 6.28


def _extract_policy_mlp(params: Any) -> list[tuple[np.ndarray, np.ndarray]]:
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
    return [(np.array(w), np.array(b)) for _name, w, b in layers]


def _mlp_forward(x: np.ndarray, weights: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    for w, b in weights[:-1]:
        x = x @ w + b
        x = x / (1.0 + np.exp(-x))
    w, b = weights[-1]
    return x @ w + b


def _scale_action(action: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
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


def _local_target(model: mujoco.MjModel, data: mujoco.MjData, target_xy: np.ndarray) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "frame")
    pos_xy = data.xpos[body_id, :2]
    v = target_xy - pos_xy
    xmat = data.xmat[body_id]
    yaw = np.arctan2(xmat[3], xmat[0])
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])


def _load_checkpoint(
    checkpoint_dir: Path, step: int | None
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray, str]:
    ckpt = checkpoints.restore_checkpoint(str(checkpoint_dir), target=None, step=step)
    if not ckpt or "params" not in ckpt or "normalizer" not in ckpt:
        raise FileNotFoundError(f"No valid checkpoint found in {checkpoint_dir} (step={step}).")
    weights = _extract_policy_mlp(ckpt["params"]["policy"])
    mean = np.array(ckpt["normalizer"]["mean"])
    std = np.array(ckpt["normalizer"]["std"])
    return weights, mean, std, "ok"


def _latest_checkpoint_path(checkpoint_dir: Path) -> str | None:
    return checkpoints.latest_checkpoint(str(checkpoint_dir))


def _parse_step_from_path(path: str | None) -> int:
    if not path:
        return -1
    match = re.search(r"checkpoint_(\\d+)", path)
    if not match:
        return -1
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Live viewer that reloads latest checkpoint.")
    parser.add_argument("--xml", default=str(Path(__file__).resolve().parent / "scene.xml"))
    parser.add_argument("--checkpoint_dir", default=str(Path(__file__).resolve().parent / "checkpoints"))
    parser.add_argument("--render_steps", type=int, default=0, help="0 means run until window closes.")
    parser.add_argument("--poll_seconds", type=float, default=2.0)
    parser.add_argument("--checkpoint_step", type=int, default=-1, help="-1 to follow latest.")
    parser.add_argument("--render_target_x", type=float, default=3.0)
    parser.add_argument("--render_target_y", type=float, default=2.0)
    args = parser.parse_args()

    xml_path = Path(args.xml).expanduser()
    if not xml_path.exists():
        xml_path = Path(__file__).resolve().parent / xml_path.name

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = Path.cwd() / checkpoint_dir

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_gyro")
    accel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
    gyro_slice = slice(model.sensor_adr[gyro_id], model.sensor_adr[gyro_id] + model.sensor_dim[gyro_id])
    accel_slice = slice(model.sensor_adr[accel_id], model.sensor_adr[accel_id] + model.sensor_dim[accel_id])

    target_xy = np.array([args.render_target_x, args.render_target_y], dtype=np.float64)
    prev_action = np.zeros((model.nu,), dtype=np.float64)

    weights: list[tuple[np.ndarray, np.ndarray]] | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    last_ckpt_path: str | None = None
    last_poll = 0.0

    def maybe_reload(now: float) -> None:
        nonlocal weights, mean, std, last_ckpt_path, last_poll
        if (now - last_poll) < args.poll_seconds:
            return
        last_poll = now
        if args.checkpoint_step >= 0:
            step = args.checkpoint_step
            if last_ckpt_path is None:
                try:
                    weights, mean, std, _ = _load_checkpoint(checkpoint_dir, step=step)
                except (ValueError, FileNotFoundError):
                    return
                last_ckpt_path = f"checkpoint_{step}"
                print(f"Loaded checkpoint step {step}", flush=True)
            return

        ckpt_path = _latest_checkpoint_path(checkpoint_dir)
        if ckpt_path and ckpt_path != last_ckpt_path:
            step = _parse_step_from_path(ckpt_path)
            try:
                if step < 0:
                    weights, mean, std, _ = _load_checkpoint(checkpoint_dir, step=None)
                else:
                    weights, mean, std, _ = _load_checkpoint(checkpoint_dir, step=step)
            except (ValueError, FileNotFoundError):
                return
            last_ckpt_path = ckpt_path
            print(f"Loaded checkpoint {ckpt_path}", flush=True)

    with viewer.launch_passive(model, data) as v:
        geom = v.user_scn.geoms[0]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.08, 0.0, 0.0]),
            np.array([target_xy[0], target_xy[1], 0.05]),
            np.eye(3).flatten(),
            np.array([1.0, 0.0, 0.0, 1.0]),
        )
        v.user_scn.ngeom = 1

        steps = 0
        while v.is_running() and (args.render_steps <= 0 or steps < args.render_steps):
            now = time.perf_counter()
            maybe_reload(now)

            if weights is not None and mean is not None and std is not None:
                qpos = data.qpos.copy()
                qvel = data.qvel.copy()
                gyro = data.sensordata[gyro_slice].copy()
                accel = data.sensordata[accel_slice].copy()
                local_xy = _local_target(model, data, target_xy)

                obs = np.concatenate([qpos, qvel, gyro, accel, local_xy])
                obs = (obs - mean) / std

                mean_action = _mlp_forward(obs, weights)
                action = np.tanh(mean_action)

                delayed = ACTION_DELAY_ALPHA * action + (1.0 - ACTION_DELAY_ALPHA) * prev_action
                delayed = np.clip(delayed, -1.0, 1.0)
                data.ctrl[:] = _scale_action(delayed, model)
                prev_action[:] = action
            else:
                if data.ctrl.size:
                    data.ctrl[:] = 0.0

            mujoco.mj_step(model, data)
            v.user_scn.geoms[0].pos = np.array([target_xy[0], target_xy[1], 0.05])
            v.sync()
            steps += 1
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
