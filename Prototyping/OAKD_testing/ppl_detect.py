#!/usr/bin/env python3

from pathlib import Path
import time
import cv2
import depthai as dai
import numpy as np


# ---------- Pink detection ----------
def find_pink_center_bgr(frame_bgr, min_area_px=300):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower_pink = np.array([140, 150, 120], dtype=np.uint8)
    upper_pink = np.array([175, 255, 220], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0, None, mask

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area_px:
        return None, None, 0, None, mask

    x, y, w, h = cv2.boundingRect(c)
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy, area, (x, y, w, h), mask


def get_depth_m(depth_frame_mm, u, v, window_radius=2, min_depth_m=0.2, max_depth_m=8.0):
    h, w = depth_frame_mm.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return None

    u0 = max(0, u - window_radius)
    u1 = min(w, u + window_radius + 1)
    v0 = max(0, v - window_radius)
    v1 = min(h, v + window_radius + 1)

    patch = depth_frame_mm[v0:v1, u0:u1]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None

    depth_mm = float(np.median(valid))
    depth_m = depth_mm / 1000.0

    if depth_m < min_depth_m or depth_m > max_depth_m:
        return None

    return depth_m


def pixel_to_3d(u, v, z_m, fx, fy, cx, cy):
    x_m = (float(u) - cx) * z_m / fx
    y_m = (float(v) - cy) * z_m / fy
    return np.array([x_m, y_m, z_m], dtype=np.float64)


def dist3(a, b):
    return float(np.linalg.norm(a - b))


# ---------- Build pipeline ----------
pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatial_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_depth = pipeline.create(dai.node.XLinkOut)
xout_det = pipeline.create(dai.node.XLinkOut)

xout_rgb.setStreamName("rgb")
xout_depth.setStreamName("depth")
xout_det.setStreamName("detections")

# Camera settings
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setCamera("left")
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setCamera("right")

# Stereo settings
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align depth to RGB
stereo.setOutputSize(1280, 720)

# NN settings
# Change this path if needed
nn_blob = str(Path("./mobilenet-ssd_openvino_2021.4_5shave.blob").resolve())
spatial_nn.setBlobPath(nn_blob)
spatial_nn.setConfidenceThreshold(0.5)
spatial_nn.input.setBlocking(False)
spatial_nn.setBoundingBoxScaleFactor(0.5)
spatial_nn.setDepthLowerThreshold(100)
spatial_nn.setDepthUpperThreshold(8000)

# Link pipeline
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

cam_rgb.video.link(xout_rgb.input)
cam_rgb.preview.link(spatial_nn.input)
stereo.depth.link(spatial_nn.inputDepth)

spatial_nn.out.link(xout_det.input)
stereo.depth.link(xout_depth.input)

label_map = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# ---------- Run ----------
with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
    q_det = device.getOutputQueue("detections", maxSize=4, blocking=False)

    calib = device.readCalibration()
    intrinsics = calib.getCameraIntrinsics(
        dai.CameraBoardSocket.CAM_A,
        1280, 720
    )
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]

    print(f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")

    start_time = time.monotonic()
    counter = 0
    fps = 0.0

    last_target_xyz = None

    while True:
        in_rgb = q_rgb.get()
        in_depth = q_depth.get()
        in_det = q_det.get()

        frame = in_rgb.getCvFrame()
        depth_frame = in_depth.getFrame()  # uint16 mm
        detections = in_det.detections

        counter += 1
        now = time.monotonic()
        if (now - start_time) > 1.0:
            fps = counter / (now - start_time)
            counter = 0
            start_time = now

        # Pink detection
        px, py, parea, pbbox, mask = find_pink_center_bgr(frame)
        pink_xyz = None
        if px is not None and py is not None:
            z_m = get_depth_m(depth_frame, px, py)
            if z_m is not None:
                pink_xyz = pixel_to_3d(px, py, z_m, fx, fy, cx, cy)

        # Draw pink blob
        if pbbox is not None:
            x, y, w, h = pbbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.circle(frame, (px, py), 5, (255, 255, 255), -1)
            cv2.putText(frame, f"pink area={int(parea)}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if pink_xyz is not None:
            cv2.putText(frame,
                        f"pink xyz=({pink_xyz[0]:.2f},{pink_xyz[1]:.2f},{pink_xyz[2]:.2f})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Choose person nearest to pink 3D point
        chosen_idx = None
        chosen_dist = None

        for i, det in enumerate(detections):
            label = label_map[det.label] if 0 <= det.label < len(label_map) else str(det.label)
            if label != "person":
                continue

            roi = det.boundingBoxMapping.roi
            roi = roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            det_xyz = np.array([
                det.spatialCoordinates.x / 1000.0,
                det.spatialCoordinates.y / 1000.0,
                det.spatialCoordinates.z / 1000.0
            ], dtype=np.float64)

            score_text = f"{label} {det.confidence:.2f}"
            xyz_text = f"xyz=({det_xyz[0]:.2f},{det_xyz[1]:.2f},{det_xyz[2]:.2f})"

            color = (255, 0, 0)
            thickness = 1

            if pink_xyz is not None:
                d = dist3(det_xyz, pink_xyz)
                if chosen_dist is None or d < chosen_dist:
                    chosen_dist = d
                    chosen_idx = i
                cv2.putText(frame, f"d_pink={d:.2f}m", (x1 + 5, y1 + 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, score_text, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, xyz_text, (x1 + 5, y1 + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Highlight chosen target
        if chosen_idx is not None:
            det = detections[chosen_idx]
            roi = det.boundingBoxMapping.roi
            roi = roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            det_xyz = np.array([
                det.spatialCoordinates.x / 1000.0,
                det.spatialCoordinates.y / 1000.0,
                det.spatialCoordinates.z / 1000.0
            ], dtype=np.float64)
            last_target_xyz = det_xyz

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, f"TARGET d={chosen_dist:.2f}m", (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        status = "TARGET: none"
        if last_target_xyz is not None:
            status = f"TARGET xyz=({last_target_xyz[0]:.2f},{last_target_xyz[1]:.2f},{last_target_xyz[2]:.2f})"

        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Depth visualization
        depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        if px is not None and py is not None:
            cv2.circle(depth_vis, (px, py), 5, (255, 255, 255), -1)

        cv2.imshow("rgb_debug", frame)
        cv2.imshow("pink_mask", mask)
        cv2.imshow("depth_debug", depth_vis)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()