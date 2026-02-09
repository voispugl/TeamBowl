from __future__ import annotations

import argparse
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from flax.training import checkpoints


def _find_attr(obj: Any, names: Tuple[str, ...]) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _extract_training_state(ckpt: Any) -> Any:
    if isinstance(ckpt, Mapping):
        for key in ("training_state", "state"):
            if key in ckpt:
                return ckpt[key]
    return ckpt


def _extract_params_and_normalizer(state: Any) -> Tuple[Any, Any]:
    params = _find_attr(state, ("params", "policy_params", "actor_params"))
    normalizer = _find_attr(state, ("normalizer", "obs_normalizer"))

    if isinstance(state, Mapping):
        if params is None and "params" in state:
            params = state["params"]
        if normalizer is None and "normalizer" in state:
            normalizer = state["normalizer"]

    return params, normalizer


def _normalizer_mean_std(normalizer: Any) -> Tuple[np.ndarray, np.ndarray] | None:
    if normalizer is None:
        return None
    mean = _find_attr(normalizer, ("mean",))
    std = _find_attr(normalizer, ("std",))
    var = _find_attr(normalizer, ("var",))
    if std is None and var is not None:
        std = np.sqrt(np.array(var) + 1e-8)
    if mean is None or std is None:
        return None
    return np.array(mean), np.array(std)


def _flatten_dense_layers(params: Any) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    if isinstance(params, Mapping) and "params" in params:
        params = params["params"]

    layers: List[Tuple[str, np.ndarray, np.ndarray]] = []

    def visit(prefix: str, node: Any) -> None:
        if isinstance(node, Mapping) and "kernel" in node and "bias" in node:
            layers.append((prefix, np.array(node["kernel"]), np.array(node["bias"])))
            return
        if isinstance(node, Mapping):
            for key, value in node.items():
                name = f"{prefix}/{key}" if prefix else key
                visit(name, value)

    visit("", params)
    return layers


def _layer_sort_key(name: str) -> Tuple[int, int]:
    # Extract layer index from common naming conventions.
    tokens = name.replace("/", "_").split("_")
    for token in reversed(tokens):
        if token.isdigit():
            return (0, int(token))
    if "out" in name or "mean" in name:
        return (2, 0)
    return (1, 0)


def _extract_policy_mlp(params: Any) -> List[Tuple[np.ndarray, np.ndarray]]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="robot/checkpoints")
    parser.add_argument("--output", default="robot/policy.pkl")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        ckpt_dir = Path(__file__).resolve().parent / ckpt_dir.name
    ckpt = checkpoints.restore_checkpoint(str(ckpt_dir), target=None)
    state = _extract_training_state(ckpt)
    params, normalizer = _extract_params_and_normalizer(state)

    if params is None:
        raise RuntimeError("No params found in checkpoint. Cannot export policy.")

    mean_std = _normalizer_mean_std(normalizer)
    if mean_std is None:
        raise RuntimeError(
            "No observation normalizer found in checkpoint. "
            "Ensure PPO was run with normalize_observations=True and checkpoints saved."
        )

    weights = _extract_policy_mlp(params)

    payload = {
        "params": weights,
        "mean": mean_std[0],
        "std": mean_std[1],
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)


if __name__ == "__main__":
    main()
