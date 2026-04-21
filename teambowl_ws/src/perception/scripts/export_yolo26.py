#!/usr/bin/env python3
"""
One-shot script: download yolo26n.pt and export a TensorRT FP16 engine for this Jetson.
Run once on the Jetson — the .engine file is device-specific and cannot be transferred.

Usage:
    python3 export_yolo26.py [--model yolo26n] [--out ~/TeamBowl/models/yolo26n.engine]
"""

import argparse
import os
import shutil

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26s", help="Ultralytics model name (yolo26n/yolo26s/...)")
    parser.add_argument("--out", default=os.path.expanduser("~/TeamBowl/models/yolo26s.onnx"),
                        help="Destination path for the .engine file")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT builder RAM in GiB")
    args = parser.parse_args()

    print(f"[export_yolo26] Loading {args.model}.pt ...")
    model = YOLO(f"{args.model}.pt")

    print(f"[export_yolo26] Exporting to TensorRT FP16 (imgsz={args.imgsz}, workspace={args.workspace} GiB) ...")
    print("[export_yolo26] This takes 5-15 minutes on first run.")
    engine_path = model.export(
        format="engine",
        half=True,
        device=0,
        imgsz=args.imgsz,
        workspace=args.workspace,
        verbose=False,
    )

    dest = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if str(engine_path) != dest:
        shutil.move(str(engine_path), dest)
    print(f"[export_yolo26] Engine saved to: {dest}")
    print("[export_yolo26] Done. Pass model_path:={dest} to yolo26_node.")


if __name__ == "__main__":
    main()
