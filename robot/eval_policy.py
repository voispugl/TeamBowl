#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from flax.training import checkpoints

from train import _render_policy

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a trained policy checkpoint.")
    parser.add_argument("--xml", default=str(ROOT / "scene.xml"))
    parser.add_argument("--checkpoint_dir", default=str(ROOT / "checkpoints"))
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=-1,
        help="Checkpoint step to load (-1 uses latest).",
    )
    parser.add_argument("--render_steps", type=int, default=1000)
    parser.add_argument("--render_target_x", type=float, default=3.0)
    parser.add_argument("--render_target_y", type=float, default=2.0)
    args = parser.parse_args()

    xml_path = Path(args.xml).expanduser()
    if not xml_path.exists():
        xml_path = ROOT / xml_path.name

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = Path.cwd() / checkpoint_dir

    step = None if args.checkpoint_step < 0 else int(args.checkpoint_step)
    ckpt = checkpoints.restore_checkpoint(str(checkpoint_dir), target=None, step=step)
    if not ckpt or "params" not in ckpt or "normalizer" not in ckpt:
        raise FileNotFoundError(
            f"No valid checkpoint found in {checkpoint_dir} (step={args.checkpoint_step})."
        )

    print(f"Loaded checkpoint from {checkpoint_dir}", flush=True)
    _render_policy(
        xml_path=xml_path,
        policy_params=ckpt["params"]["policy"],
        obs_mean=ckpt["normalizer"]["mean"],
        obs_std=ckpt["normalizer"]["std"],
        target_xy=(args.render_target_x, args.render_target_y),
        max_steps=args.render_steps,
    )


if __name__ == "__main__":
    main()
