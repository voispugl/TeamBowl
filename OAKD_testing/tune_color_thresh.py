#!/usr/bin/env python3

import cv2
import numpy as np

# =========================
# CONFIG
# =========================
USE_OAKD = True   # set True if you want to use OAK-D via DepthAI
CAMERA_INDEX = 0  # webcam index if USE_OAKD = False

MIN_AREA = 200    # minimum blob area (px) to consider valid

# =========================
# Trackbar helpers
# =========================
def nothing(x):
    pass

cv2.namedWindow("controls", cv2.WINDOW_NORMAL)
cv2.resizeWindow("controls", 400, 300)

# HSV lower / upper
cv2.createTrackbar("H min", "controls", 160, 179, nothing)
cv2.createTrackbar("H max", "controls", 179, 179, nothing)
cv2.createTrackbar("S min", "controls", 80, 255, nothing)
cv2.createTrackbar("S max", "controls", 255, 255, nothing)
cv2.createTrackbar("V min", "controls", 80, 255, nothing)
cv2.createTrackbar("V max", "controls", 255, 255, nothing)

# Morphology
cv2.createTrackbar("Open", "controls", 1, 5, nothing)
cv2.createTrackbar("Close", "controls", 2, 5, nothing)

# =========================
# Video source
# =========================
if USE_OAKD:
    import depthai as dai

    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640, 480)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)

    device = dai.Device(pipeline)
    q = device.getOutputQueue("rgb", 4, False)

    def get_frame():
        return q.get().getCvFrame()
else:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    def get_frame():
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Camera frame grab failed")
        return frame

# =========================
# Main loop
# =========================
while True:
    frame = get_frame()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Read HSV bounds
    hmin = cv2.getTrackbarPos("H min", "controls")
    hmax = cv2.getTrackbarPos("H max", "controls")
    smin = cv2.getTrackbarPos("S min", "controls")
    smax = cv2.getTrackbarPos("S max", "controls")
    vmin = cv2.getTrackbarPos("V min", "controls")
    vmax = cv2.getTrackbarPos("V max", "controls")

    lower = np.array([hmin, smin, vmin])
    upper = np.array([hmax, smax, vmax])

    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup
    open_k = cv2.getTrackbarPos("Open", "controls")
    close_k = cv2.getTrackbarPos("Close", "controls")

    if open_k > 0:
        kernel = np.ones((open_k, open_k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if close_k > 0:
        kernel = np.ones((close_k, close_k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find largest blob
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    display = frame.copy()

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if area >= MIN_AREA:
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            cy = y + h // 2

            cv2.rectangle(display, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.circle(display, (cx, cy), 5, (0,255,0), -1)
            cv2.putText(display, f"area={int(area)}",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Show
    cv2.imshow("frame", display)
    cv2.imshow("mask", mask)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        print(f"HSV LOWER = [{hmin}, {smin}, {vmin}]")
        print(f"HSV UPPER = [{hmax}, {smax}, {vmax}]")
        print("-"*40)

cv2.destroyAllWindows()
