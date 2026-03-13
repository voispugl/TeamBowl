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
from vision_msgs.msg import Detection2DArray


class CamOpsNode(Node):
    """
    Subscribes to:
      - /oak/nn/passthrough/image_raw      (for overlay + pink detection)
      - /oak/nn/passthrough/camera_info    (intrinsics for passthrough image)
      - /oak/nn/detections                 (Detection2DArray)
      - /oak/stereo/image_raw              (cached latest depth frame; NOT time-synced)

    Publishes:
      - /robot/target_person_pos   (PointStamped, meters, camera optical frame)
      - /robot/target_valid        (Bool)
      - /robot/debug/cam_ops_image (Image)

    Behavior:
      - Finds pink pants blob in passthrough image
      - Finds all person detections in the same passthrough image
      - Chooses the person nearest the pink blob in image space
      - If no pink match, tries to reacquire nearest to previously chosen target center
      - Uses latest cached depth frame to estimate XYZ at chosen detection center
    """

    def __init__(self):
        super().__init__('cam_ops_node')
        self.bridge = CvBridge()

        # Topics
        self.declare_parameter('image_topic', '/oak/nn/passthrough/image_raw')
        self.declare_parameter('depth_topic', '/oak/stereo/image_raw')
        self.declare_parameter('camera_info_topic', '/oak/nn/passthrough/camera_info')
        self.declare_parameter('detections_topic', '/oak/nn/detections')
        self.declare_parameter('target_topic', '/robot/target_person_pos')
        self.declare_parameter('target_valid_topic', '/robot/target_valid')
        self.declare_parameter('debug_image_topic', '/robot/debug/cam_ops_image')

        # Behavior
        self.declare_parameter('sync_slop_s', 0.25)
        self.declare_parameter('min_pink_area_px', 300)
        self.declare_parameter('lost_max', 20)

        # Depth filtering
        self.declare_parameter('min_depth_m', 0.2)
        self.declare_parameter('max_depth_m', 8.0)
        self.declare_parameter('depth_window_radius_px', 2)

        # Matching
        self.declare_parameter('pink_match_max_dist_px', 120.0)
        self.declare_parameter('reacquire_max_dist_px', 140.0)

        # Read parameters
        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.detections_topic = self.get_parameter('detections_topic').value
        self.target_topic = self.get_parameter('target_topic').value
        self.target_valid_topic = self.get_parameter('target_valid_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value

        self.sync_slop_s = float(self.get_parameter('sync_slop_s').value)
        self.min_pink_area_px = int(self.get_parameter('min_pink_area_px').value)
        self.lost_max = int(self.get_parameter('lost_max').value)

        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.depth_window_radius_px = int(self.get_parameter('depth_window_radius_px').value)

        self.pink_match_max_dist_px = float(self.get_parameter('pink_match_max_dist_px').value)
        self.reacquire_max_dist_px = float(self.get_parameter('reacquire_max_dist_px').value)

        # Publishers
        self.target_pub = self.create_publisher(PointStamped, self.target_topic, 10)
        self.target_valid_pub = self.create_publisher(Bool, self.target_valid_topic, 10)
        self.debug_image_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        # Pink HSV thresholds
        self.lower_pink = np.array([140, 150, 120], dtype=np.uint8)
        self.upper_pink = np.array([175, 255, 220], dtype=np.uint8)

        # Camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Cached latest depth
        self.latest_depth_img = None
        self.latest_depth_header = None

        # Tracking state
        self.target_center_uv = None
        self.target_pos_xyz = None
        self.lost_frames = 0

        # Camera info
        self.info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        # Depth cached separately: not in message_filters sync
        self.depth_plain_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data
        )

        # Only sync image + detections
        self.image_sub = message_filters.Subscriber(
            self, Image, self.image_topic, qos_profile=qos_profile_sensor_data
        )
        self.det_sub = message_filters.Subscriber(
            self, Detection2DArray, self.detections_topic, qos_profile=qos_profile_sensor_data
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.det_sub],
            queue_size=20,
            slop=self.sync_slop_s
        )
        self.ts.registerCallback(self.synchronized_callback)

        self.get_logger().info(
            'cam_ops_node started | syncing passthrough image + detections, caching depth separately'
        )

    def camera_info_callback(self, info_msg: CameraInfo):
        self.fx = float(info_msg.k[0])
        self.fy = float(info_msg.k[4])
        self.cx = float(info_msg.k[2])
        self.cy = float(info_msg.k[5])

    def depth_callback(self, msg: Image):
        try:
            if msg.encoding != '16UC1':
                return
            self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth_header = msg.header
        except Exception as e:
            self.get_logger().error(f'depth_callback failed: {e}')

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

        self.target_pos_xyz = np.array(pos_xyz_m, dtype=np.float64)
        self.publish_target_valid(True)

    def publish_debug_image(self, frame_bgr, header):
        msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding='bgr8')
        msg.header = header
        self.debug_image_pub.publish(msg)

    def find_pink_center(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_pink, self.upper_pink)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, 0, None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.min_pink_area_px:
            return None, None, area, None

        x, y, w, h = cv2.boundingRect(contour)
        u = x + w // 2
        v = y + h // 2
        return u, v, area, (x, y, w, h)

    def get_depth_m(self, depth_img, u, v):
        if depth_img is None:
            return None

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

    def detection_is_person(self, det):
        if not det.results:
            return False

        class_id = str(det.results[0].hypothesis.class_id).strip().lower()
        return class_id in ('person', '15')

    def detection_score(self, det):
        if not det.results:
            return 0.0
        return float(det.results[0].hypothesis.score)

    def detection_center_uv(self, det):
        try:
            u = int(round(float(det.bbox.center.position.x)))
            v = int(round(float(det.bbox.center.position.y)))
            return (u, v)
        except Exception:
            return None

    def detection_rect(self, det, img_shape):
        try:
            u = float(det.bbox.center.position.x)
            v = float(det.bbox.center.position.y)
            w = float(det.bbox.size_x)
            h = float(det.bbox.size_y)
        except Exception:
            return None

        img_h, img_w = img_shape[:2]

        x = int(round(u - w / 2.0))
        y = int(round(v - h / 2.0))
        w = int(round(w))
        h = int(round(h))

        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        return x, y, w, h

    def dist2(self, a_uv, b_uv):
        a = np.array(a_uv, dtype=np.float64)
        b = np.array(b_uv, dtype=np.float64)
        return float(np.linalg.norm(a - b))

    def synchronized_callback(self, img_msg, det_msg):
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

            depth_img = self.latest_depth_img
            has_depth = depth_img is not None

            # Pink detection in passthrough image frame
            pink_u, pink_v, pink_area, pink_bbox = self.find_pink_center(frame)
            pink_uv = (pink_u, pink_v) if pink_u is not None and pink_v is not None else None

            if pink_bbox is not None:
                x, y, w, h = pink_bbox
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.circle(debug_frame, pink_uv, 5, (255, 255, 255), -1)
                cv2.putText(
                    debug_frame,
                    f'pink area={int(pink_area)}',
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            else:
                cv2.putText(
                    debug_frame,
                    'pink: none',
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            # Build list of person detections
            persons = []
            for i, det in enumerate(det_msg.detections):
                if not self.detection_is_person(det):
                    continue

                uv = self.detection_center_uv(det)
                rect = self.detection_rect(det, frame.shape)
                if uv is None or rect is None:
                    continue

                z_m = None
                pos_xyz = None
                if has_depth:
                    z_m = self.get_depth_m(depth_img, uv[0], uv[1])
                    if z_m is not None:
                        pos_xyz = self.pixel_to_3d(uv[0], uv[1], z_m)

                score = self.detection_score(det)

                persons.append({
                    'index': i,
                    'uv': uv,
                    'rect': rect,
                    'score': score,
                    'z_m': z_m,
                    'pos_xyz': pos_xyz,
                })

            # Draw all persons
            for j, p in enumerate(persons):
                x, y, w, h = p['rect']
                u, v = p['uv']

                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.circle(debug_frame, (u, v), 5, (255, 0, 0), -1)

                label = f'person {p["index"]} s={p["score"]:.2f}'
                if p['pos_xyz'] is not None:
                    px, py, pz = p['pos_xyz']
                    label += f' xyz=({px:.2f},{py:.2f},{pz:.2f})'
                elif p['z_m'] is not None:
                    label += f' z={p["z_m"]:.2f}m'
                else:
                    label += ' no_depth'

                cv2.putText(
                    debug_frame,
                    label,
                    (20, 70 + 30 * j),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 0),
                    2
                )

                if pink_uv is not None:
                    dpx = self.dist2(pink_uv, p['uv'])
                    cv2.putText(
                        debug_frame,
                        f'dpx={dpx:.1f}',
                        (max(0, x), max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2
                    )

            chosen = None
            chosen_dpx = None

            # Primary match: nearest person to pink blob in image space
            if pink_uv is not None:
                best = None
                best_d = None
                for p in persons:
                    d = self.dist2(pink_uv, p['uv'])
                    if d > self.pink_match_max_dist_px:
                        continue
                    if best is None or d < best_d:
                        best = p
                        best_d = d
                chosen = best
                chosen_dpx = best_d

            # Fallback reacquire: nearest to previous target center
            if chosen is None and self.target_center_uv is not None:
                best = None
                best_d = None
                for p in persons:
                    d = self.dist2(self.target_center_uv, p['uv'])
                    if d > self.reacquire_max_dist_px:
                        continue
                    if best is None or d < best_d:
                        best = p
                        best_d = d
                chosen = best
                chosen_dpx = best_d

            # Publish chosen target
            if chosen is not None:
                self.target_center_uv = chosen['uv']
                self.lost_frames = 0

                x, y, w, h = chosen['rect']
                u, v = chosen['uv']

                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.circle(debug_frame, (u, v), 6, (0, 255, 0), -1)

                if chosen['pos_xyz'] is not None:
                    self.publish_target_position(chosen['pos_xyz'], img_msg.header)
                    px, py, pz = chosen['pos_xyz']
                    cv2.putText(
                        debug_frame,
                        f'TARGET xyz=({px:.2f}, {py:.2f}, {pz:.2f})',
                        (20, debug_frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                else:
                    self.publish_target_valid(False)
                    cv2.putText(
                        debug_frame,
                        'TARGET chosen but no valid depth',
                        (20, debug_frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

                if chosen_dpx is not None:
                    cv2.putText(
                        debug_frame,
                        f'target dist to pink = {chosen_dpx:.1f} px',
                        (20, debug_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                self.publish_debug_image(debug_frame, img_msg.header)
                return

            # No target found
            if self.target_center_uv is not None:
                self.lost_frames += 1
                if self.lost_frames >= self.lost_max:
                    self.target_center_uv = None
                    self.target_pos_xyz = None
                    self.lost_frames = 0

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