#!/usr/bin/env python3

import cv2
import message_filters
import numpy as np
import rclpy

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool


class CamOpsNode(Node):
    """
    Pink-only target detector.

    Subscribes to:
      - image_topic        (RGB image)
      - depth_topic        (16UC1 depth image, mm)
      - camera_info_topic  (intrinsics for RGB image)

    Publishes:
      - target_topic       (PointStamped, meters, camera optical frame)
      - target_valid_topic (Bool)
      - debug_image_topic  (Image)

    Behavior:
      - Detect pink blob in RGB image
      - Optionally run detection on resized RGB image for speed
      - Map detected center back to original image coordinates
      - Sample depth at original pixel
      - Convert to XYZ using camera intrinsics
    """

    def __init__(self):
        super().__init__('cam_ops_node')
        self.bridge = CvBridge()

        # Topics
        self.declare_parameter('image_topic', '/oak/rgb/image_rect')
        self.declare_parameter('depth_topic', '/oak/stereo/image_raw')
        self.declare_parameter('camera_info_topic', '/oak/rgb/camera_info')
        self.declare_parameter('target_topic', '/robot/target_person_pos')
        self.declare_parameter('target_valid_topic', '/robot/target_valid')
        self.declare_parameter('debug_image_topic', '/robot/debug/cam_ops_image')

        # Sync
        self.declare_parameter('sync_slop_s', 0.2)

        # Pink detection
        self.declare_parameter('min_pink_area_px', 300)

        # Optional resize for faster detection
        self.declare_parameter('enable_resize', False)
        self.declare_parameter('resize_scale', 0.5)

        # Depth filtering
        self.declare_parameter('min_depth_m', 0.2)
        self.declare_parameter('max_depth_m', 8.0)
        self.declare_parameter('depth_window_radius_px', 2)

        # Read parameters
        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.target_topic = self.get_parameter('target_topic').value
        self.target_valid_topic = self.get_parameter('target_valid_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value

        self.sync_slop_s = float(self.get_parameter('sync_slop_s').value)
        self.min_pink_area_px = int(self.get_parameter('min_pink_area_px').value)

        self.enable_resize = bool(self.get_parameter('enable_resize').value)
        self.resize_scale = float(self.get_parameter('resize_scale').value)

        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.depth_window_radius_px = int(self.get_parameter('depth_window_radius_px').value)

        if self.enable_resize and not (0.0 < self.resize_scale <= 1.0):
            self.get_logger().warn(
                f'Invalid resize_scale={self.resize_scale}. Resetting to 0.5'
            )
            self.resize_scale = 0.5

        # Pink HSV thresholds
        self.lower_pink = np.array([140, 150, 120], dtype=np.uint8)
        self.upper_pink = np.array([175, 255, 220], dtype=np.uint8)

        # Camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Publishers
        self.target_pub = self.create_publisher(PointStamped, self.target_topic, 10)
        self.target_valid_pub = self.create_publisher(Bool, self.target_valid_topic, 10)
        self.debug_image_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        # Camera info
        self.info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        # Sync RGB + depth
        self.image_sub = message_filters.Subscriber(
            self, Image, self.image_topic, qos_profile=qos_profile_sensor_data
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub],
            queue_size=10,
            slop=self.sync_slop_s
        )
        self.ts.registerCallback(self.synchronized_callback)

        self.get_logger().info(
            f'cam_ops_node started | resize={self.enable_resize} scale={self.resize_scale:.3f}'
        )

    def camera_info_callback(self, info_msg: CameraInfo):
        self.fx = float(info_msg.k[0])
        self.fy = float(info_msg.k[4])
        self.cx = float(info_msg.k[2])
        self.cy = float(info_msg.k[5])

    def publish_target_valid(self, valid: bool):
        msg = Bool()
        msg.data = valid
        self.target_valid_pub.publish(msg)

    def publish_target_position(self, pos_xyz_m, header):
        msg = PointStamped()
        msg.header = header
        msg.point.x = float(pos_xyz_m[0])
        msg.point.y = float(pos_xyz_m[1])
        msg.point.z = float(pos_xyz_m[2])
        self.target_pub.publish(msg)

        self.publish_target_valid(True)

    def publish_debug_image(self, frame_bgr, header):
        msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding='bgr8')
        msg.header = header
        self.debug_image_pub.publish(msg)

    def detect_pink(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_pink, self.upper_pink)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.min_pink_area_px:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        u = x + w // 2
        v = y + h // 2

        return {
            'u': u,
            'v': v,
            'bbox': (x, y, w, h),
            'area': area,
        }

    def maybe_resize_for_detection(self, frame_bgr):
        """
        Returns:
          detect_frame: frame used for pink detection
          scale_used: factor such that original = detected / scale_used
        """
        if not self.enable_resize:
            return frame_bgr, 1.0

        detect_frame = cv2.resize(
            frame_bgr,
            None,
            fx=self.resize_scale,
            fy=self.resize_scale,
            interpolation=cv2.INTER_AREA
        )
        return detect_frame, self.resize_scale

    def scale_detection_to_original(self, det, scale_used, orig_shape):
        """
        Convert detection result from resized image coordinates back to original image.
        """
        if det is None:
            return None

        if scale_used == 1.0:
            return det

        orig_h, orig_w = orig_shape[:2]

        u = int(round(det['u'] / scale_used))
        v = int(round(det['v'] / scale_used))

        x, y, w, h = det['bbox']
        x = int(round(x / scale_used))
        y = int(round(y / scale_used))
        w = int(round(w / scale_used))
        h = int(round(h / scale_used))

        x = max(0, min(orig_w - 1, x))
        y = max(0, min(orig_h - 1, y))
        w = max(1, min(w, orig_w - x))
        h = max(1, min(h, orig_h - y))
        u = max(0, min(orig_w - 1, u))
        v = max(0, min(orig_h - 1, v))

        return {
            'u': u,
            'v': v,
            'bbox': (x, y, w, h),
            'area': det['area'],
        }

    def get_depth_m(self, depth_img, u, v):
        h, w = depth_img.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None

        r = self.depth_window_radius_px
        u0 = max(0, u - r)
        u1 = min(w, u + r + 1)
        v0 = max(0, v - r)
        v1 = min(h, v + r + 1)

        patch = depth_img[v0:v1, u0:u1]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None

        depth_m = float(np.median(valid)) / 1000.0
        if depth_m < self.min_depth_m or depth_m > self.max_depth_m:
            return None

        return depth_m

    def pixel_to_3d(self, u, v, z_m):
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            return None

        x_m = (float(u) - self.cx) * z_m / self.fx
        y_m = (float(v) - self.cy) * z_m / self.fy
        return np.array([x_m, y_m, z_m], dtype=np.float64)

    def synchronized_callback(self, img_msg, depth_msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            debug_frame = frame.copy()

            if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
                cv2.putText(
                    debug_frame,
                    'Waiting for camera intrinsics...',
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
                self.publish_target_valid(False)
                self.publish_debug_image(debug_frame, img_msg.header)
                return

            if depth_msg.encoding != '16UC1':
                cv2.putText(
                    debug_frame,
                    f'Bad depth encoding: {depth_msg.encoding}',
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
                self.publish_target_valid(False)
                self.publish_debug_image(debug_frame, img_msg.header)
                return

            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            detect_frame, scale_used = self.maybe_resize_for_detection(frame)
            det_small = self.detect_pink(detect_frame)
            det = self.scale_detection_to_original(det_small, scale_used, frame.shape)

            if det is None:
                cv2.putText(
                    debug_frame,
                    'TARGET: none',
                    (20, debug_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                self.publish_target_valid(False)
                self.publish_debug_image(debug_frame, img_msg.header)
                return

            u = det['u']
            v = det['v']
            x, y, w, h = det['bbox']
            area = det['area']

            z_m = self.get_depth_m(depth_img, u, v)
            xyz_m = None
            if z_m is not None:
                xyz_m = self.pixel_to_3d(u, v, z_m)

            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(debug_frame, (u, v), 5, (0, 255, 0), -1)

            cv2.putText(
                debug_frame,
                f'pink area={int(area)}',
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                debug_frame,
                f'resize={self.enable_resize} scale={scale_used:.3f}',
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                debug_frame,
                f'pixel=({u}, {v})',
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            if xyz_m is not None:
                self.publish_target_position(xyz_m, img_msg.header)

                cv2.putText(
                    debug_frame,
                    f'XYZ=({xyz_m[0]:.2f}, {xyz_m[1]:.2f}, {xyz_m[2]:.2f})',
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
            else:
                self.publish_target_valid(False)
                cv2.putText(
                    debug_frame,
                    'Pink found but no valid depth',
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            self.publish_debug_image(debug_frame, img_msg.header)

        except Exception as e:
            self.get_logger().error(f'synchronized_callback failed: {e}')


def main():
    rclpy.init()
    node = CamOpsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()