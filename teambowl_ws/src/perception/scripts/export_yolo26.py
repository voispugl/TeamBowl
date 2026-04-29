#!/usr/bin/env python3
"""
One-shot script: download a YOLO26 .pt and export a TensorRT FP16 engine for this Jetson.
Run once on the Jetson — the .engine file is device-specific and cannot be transferred.
The downloaded .pt file is kept in models/ and used as the OSNet ReID backbone.

Usage:
    python3 export_yolo26.py [--model yolo26m] [--out ~/TeamBowl/models/yolo26m.engine]
"""

import argparse
import os
import shutil

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26m", help="Ultralytics model name (yolo26n/yolo26s/yolo26m/yolo26l/...)")
    parser.add_argument("--out", default="/home/box/TeamBowl/models/yolo26m.engine",
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
    # Keep the .pt file in models/ — yolo26_node uses it as the OSNet ReID backbone fallback
    pt_src = f"{args.model}.pt"
    pt_dest = os.path.join(os.path.dirname(dest), f"{args.model}.pt")
    if os.path.exists(pt_src) and pt_src != pt_dest:
        shutil.copy2(pt_src, pt_dest)
        print(f"[export_yolo26] .pt weights kept at: {pt_dest}")

    print(f"[export_yolo26] Engine saved to: {dest}")
    print("[export_yolo26] Done. Pass model_path:={dest} to yolo26_node.")


if __name__ == "__main__":
    main()
