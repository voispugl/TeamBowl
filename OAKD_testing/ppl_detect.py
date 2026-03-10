#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
import argparse

def find_pink_center_bgr(frame_bgr):
    """
    Returns (cx, cy, area, bbox) for the largest 'pink' blob, or (None, None, 0, None) if not found.
    bbox is (x,y,w,h) in pixels.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Pink/magenta usually lives near hue ~160-179 and also wraps near ~0-10 for some cameras.
    # We'll use two ranges and combine. You MUST tune these for your lighting/vest.
    lower1 = np.array([150, 0, 130], dtype=np.uint8)
    upper1 = np.array([179, 120, 255], dtype=np.uint8)
    lower2 = np.array([0,   80, 80], dtype=np.uint8)
    upper2 = np.array([10, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = mask1 #cv2.bitwise_or(mask1, mask2)

    # Clean up mask (reduce noise)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find largest blob
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0, None, mask

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 300:  # minimum blob area in px; tune
        return None, None, 0, None, mask

    x, y, w, h = cv2.boundingRect(c)
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy, area, (x, y, w, h), mask


def point_in_roi(px, py, roi_xyxy):
    x1, y1, x2, y2 = roi_xyxy
    return (x1 <= px <= x2) and (y1 <= py <= y2)

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

nnPathDefault = str((Path(__file__).parent / Path('./depthai-python/examples/models/mobilenet-ssd_openvino_2021.4_5shave.blob')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
parser.add_argument('-ff', '--full_frame', action="store_true", help="Perform tracking on full RGB frame", default=False)

args = parser.parse_args()

fullFrameTracking = args.full_frame

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
objectTracker = pipeline.create(dai.node.ObjectTracker)

xoutRgb = pipeline.create(dai.node.XLinkOut)
trackerOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("preview")
trackerOut.setStreamName("tracklets")

# Properties
camRgb.setPreviewSize(300, 300)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setCamera("left")
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setCamera("right")

# setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

spatialDetectionNetwork.setBlobPath(args.nnPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

objectTracker.setDetectionLabelsToTrack([15])  # track only person
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
objectTracker.passthroughTrackerFrame.link(xoutRgb.input)
objectTracker.out.link(trackerOut.input)

if fullFrameTracking:
    camRgb.setPreviewKeepAspectRatio(False)
    camRgb.video.link(objectTracker.inputTrackerFrame)
    objectTracker.inputTrackerFrame.setBlocking(False)
    # do not block the pipeline if it's too slow on full frame
    objectTracker.inputTrackerFrame.setQueueSize(2)
else:
    spatialDetectionNetwork.passthrough.link(objectTracker.inputTrackerFrame)

spatialDetectionNetwork.passthrough.link(objectTracker.inputDetectionFrame)
spatialDetectionNetwork.out.link(objectTracker.inputDetections)
stereo.depth.link(spatialDetectionNetwork.inputDepth)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    preview = device.getOutputQueue("preview", 4, False)
    tracklets = device.getOutputQueue("tracklets", 4, False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)
    target_id = None
    lost_frames = 0
    LOST_MAX = 20          # how long we tolerate losing the target tracklet
    RELOCK_COOLDOWN = 0    # optional: frames to wait before allowing relock
    cooldown = 0


    while(True):
        imgFrame = preview.get()
        track = tracklets.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time
        
        frame = imgFrame.getCvFrame()
        trackletsData = track.tracklets

        # --- Detect pink patch in the RGB frame ---
        px, py, area, pbbox, mask = find_pink_center_bgr(frame)

        # Visualize the mask if you want (super helpful while tuning):
        # cv2.imshow("pink_mask", mask)

        if cooldown > 0:
            cooldown -= 1

        # --- If we see the pink patch, choose the person tracklet that contains it ---
        if px is not None and py is not None and cooldown == 0:
            best_id = None
            best_area = None

            for t in trackletsData:
                if t.label != 15:
                    continue
                if t.status.name not in ("TRACKED", "NEW"):
                    continue

                roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                x1 = int(roi.topLeft().x)
                y1 = int(roi.topLeft().y)
                x2 = int(roi.bottomRight().x)
                y2 = int(roi.bottomRight().y)

                if point_in_roi(px, py, (x1, y1, x2, y2)):
                    # If multiple people contain the point (rare), pick the smallest bbox area
                    a = (x2 - x1) * (y2 - y1)
                    if best_area is None or a < best_area:
                        best_area = a
                        best_id = int(t.id)

            if best_id is not None:
                target_id = best_id
                lost_frames = 0
                cooldown = RELOCK_COOLDOWN

        # --- Decide if target is present this frame ---
        target_tracklet = None
        if target_id is not None:
            for t in trackletsData:
                if int(t.id) == int(target_id) and t.status.name in ("TRACKED", "NEW"):
                    target_tracklet = t
                    break

            if target_tracklet is None:
                lost_frames += 1
                if lost_frames >= LOST_MAX:
                    target_id = None
                    lost_frames = 0
        else:
            lost_frames = 0

        # --- Draw debug: pink patch box/center ---
        if pbbox is not None:
            x, y, w, h = pbbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.circle(frame, (px, py), 5, (255, 255, 255), -1)
            cv2.putText(frame, f"pink area={int(area)}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # --- Draw people: target in green, others in blue (optional) ---
        for t in trackletsData:
            if t.label != 15:
                continue
            if t.status.name not in ("TRACKED", "NEW"):
                continue

            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            is_target = (target_id is not None and int(t.id) == int(target_id))
            box_color = (0, 255, 0) if is_target else (255, 0, 0)
            thick = 3 if is_target else 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thick)
            cv2.putText(frame, f"ID:{int(t.id)} {t.status.name}", (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            if is_target:
                cv2.putText(frame, f"X:{int(t.spatialCoordinates.x)} Y:{int(t.spatialCoordinates.y)} Z:{int(t.spatialCoordinates.z)} mm",
                            (x1 + 5, y1 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # --- Show target state ---
        if target_id is None:
            cv2.putText(frame, "TARGET: none", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            cv2.putText(frame, f"TARGET: {target_id} (lost={lost_frames})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)


        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        cv2.imshow("tracker", frame)

        if cv2.waitKey(1) == ord('q'):
            break
