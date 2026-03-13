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
from vision_msgs.msg import Detection3DArray


class CamOpsNode(Node):
    """
    Subscribes to:
      - /oak/rgb/image_rect
      - /oak/stereo/image_raw        (16UC1 depth, mm)
      - /oak/rgb/camera_info
      - /oak/nn/spatial_detections   (Detection3DArray; bbox center/size assumed in NN image coords)

    Publishes:
      - /robot/target_person_pos   (PointStamped, meters, camera optical frame)
      - /robot/target_valid        (Bool)
      - /robot/debug/cam_ops_image (Image)
    """

    def __init__(self):
        super().__init__('cam_ops_node')
        self.bridge = CvBridge()

        # Topics
        self.declare_parameter('image_topic', '/oak/rgb/image_rect')
        self.declare_parameter('depth_topic', '/oak/stereo/image_raw')
        self.declare_parameter('camera_info_topic', '/oak/rgb/camera_info')
        self.declare_parameter('detections_topic', '/oak/nn/spatial_detections')
        self.declare_parameter('target_topic', '/robot/target_person_pos')
        self.declare_parameter('target_valid_topic', '/robot/target_valid')
        self.declare_parameter('debug_image_topic', '/robot/debug/cam_ops_image')

        # Timing / behavior
        self.declare_parameter('sync_slop_s', 0.2)
        self.declare_parameter('min_pink_area_px', 300)
        self.declare_parameter('lost_max', 20)
        self.declare_parameter('relock_cooldown', 0)

        # Depth filtering
        self.declare_parameter('min_depth_m', 0.2)
        self.declare_parameter('max_depth_m', 8.0)
        self.declare_parameter('depth_window_radius_px', 2)

        # Matching thresholds
        self.declare_parameter('pink_match_max_dist_m', 1.0)
        self.declare_parameter('reacquire_max_dist_m', 1.0)

        # NN geometry / mapping
        self.declare_parameter('nn_width', 300)
        self.declare_parameter('nn_height', 300)
        self.declare_parameter('nn_mapping_mode', 'letterbox')  # 'letterbox' or 'stretch'
        self.declare_parameter('bbox_shrink_x', 0.8)
        self.declare_parameter('bbox_shrink_y', 0.8)

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
        self.relock_cooldown = int(self.get_parameter('relock_cooldown').value)

        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.depth_window_radius_px = int(self.get_parameter('depth_window_radius_px').value)

        self.pink_match_max_dist_m = float(self.get_parameter('pink_match_max_dist_m').value)
        self.reacquire_max_dist_m = float(self.get_parameter('reacquire_max_dist_m').value)

        self.nn_width = int(self.get_parameter('nn_width').value)
        self.nn_height = int(self.get_parameter('nn_height').value)
        self.nn_mapping_mode = str(self.get_parameter('nn_mapping_mode').value).strip().lower()
        self.bbox_shrink_x = float(self.get_parameter('bbox_shrink_x').value)
        self.bbox_shrink_y = float(self.get_parameter('bbox_shrink_y').value)

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

        # Tracking state
        self.target_id = None
        self.target_pos = None
        self.lost_frames = 0
        self.cooldown = 0

        # Camera info cached separately
        self.info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        # Synced subscribers: RGB + depth + detections
        self.image_sub = message_filters.Subscriber(
            self, Image, self.image_topic, qos_profile=qos_profile_sensor_data
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data
        )
        self.det_sub = message_filters.Subscriber(
            self, Detection3DArray, self.detections_topic, qos_profile=qos_profile_sensor_data
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub, self.det_sub],
            queue_size=10,
            slop=self.sync_slop_s
        )
        self.ts.registerCallback(self.synchronized_callback)

        self.get_logger().info(
            f'cam_ops_node started | nn={self.nn_width}x{self.nn_height} | '
            f'mapping={self.nn_mapping_mode} | shrink=({self.bbox_shrink_x:.2f}, {self.bbox_shrink_y:.2f})'
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

    def publish_target_position(self, pos_xyz_m, header, det_id=None):
        msg = PointStamped()
        msg.header = header
        msg.point.x = float(pos_xyz_m[0])
        msg.point.y = float(pos_xyz_m[1])
        msg.point.z = float(pos_xyz_m[2])
        self.target_pub.publish(msg)

        self.target_pos = np.array(pos_xyz_m, dtype=np.float64)
        self.target_id = det_id
        self.publish_target_valid(True)

    def publish_debug_image(self, frame_bgr, header):
        msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding='bgr8')
        msg.header = header
        self.debug_image_pub.publish(msg)

    def find_pink_center(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_pink, self.upper_pink)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, 0, None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.min_pink_area_px:
            return None, None, 0, None

        x, y, w, h = cv2.boundingRect(contour)
        u = x + w // 2
        v = y + h // 2
        return u, v, area, (x, y, w, h)

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

    def detection_is_person(self, det):
        if not det.results:
            return False
        class_id = str(det.results[0].hypothesis.class_id).strip().lower()
        return class_id in ('person', '15')

    def detection_score(self, det):
        if not det.results:
            return 0.0
        return float(det.results[0].hypothesis.score)

    def map_nn_to_rgb(self, u_nn, v_nn, rgb_shape):
        rgb_h, rgb_w = rgb_shape[:2]

        if self.nn_mapping_mode == 'stretch':
            u_rgb = u_nn * rgb_w / self.nn_width
            v_rgb = v_nn * rgb_h / self.nn_height
        else:
            # letterbox inverse mapping
            scale = min(self.nn_width / rgb_w, self.nn_height / rgb_h)
            resized_w = rgb_w * scale
            resized_h = rgb_h * scale

            pad_x = (self.nn_width - resized_w) / 2.0
            pad_y = (self.nn_height - resized_h) / 2.0

            u_rgb = (u_nn - pad_x) / scale
            v_rgb = (v_nn - pad_y) / scale

        u_rgb = int(round(max(0, min(rgb_w - 1, u_rgb))))
        v_rgb = int(round(max(0, min(rgb_h - 1, v_rgb))))
        return u_rgb, v_rgb

    def detection_center_uv(self, det, rgb_shape):
        try:
            u_nn = float(det.bbox.center.position.x)
            v_nn = float(det.bbox.center.position.y)
        except Exception:
            return None

        return self.map_nn_to_rgb(u_nn, v_nn, rgb_shape)

    def detection_rect(self, det, rgb_shape):
        try:
            u_nn = float(det.bbox.center.position.x)
            v_nn = float(det.bbox.center.position.y)
            w_nn = float(det.bbox.size.x)
            h_nn = float(det.bbox.size.y)
        except Exception:
            return None

        rgb_h, rgb_w = rgb_shape[:2]

        if self.nn_mapping_mode == 'stretch':
            u = u_nn * rgb_w / self.nn_width
            v = v_nn * rgb_h / self.nn_height
            w = w_nn * rgb_w / self.nn_width
            h = h_nn * rgb_h / self.nn_height
        else:
            scale = min(self.nn_width / rgb_w, self.nn_height / rgb_h)
            resized_w = rgb_w * scale
            resized_h = rgb_h * scale

            pad_x = (self.nn_width - resized_w) / 2.0
            pad_y = (self.nn_height - resized_h) / 2.0

            u = (u_nn - pad_x) / scale
            v = (v_nn - pad_y) / scale
            w = w_nn / scale
            h = h_nn / scale

        w *= self.bbox_shrink_x
        h *= self.bbox_shrink_y

        u = int(round(u))
        v = int(round(v))
        w = int(round(w))
        h = int(round(h))

        x = u - w // 2
        y = v - h // 2

        x = max(0, x)
        y = max(0, y)
        w = min(w, rgb_w - x)
        h = min(h, rgb_h - y)

        return x, y, w, h

    def detection_position_from_bbox(self, det, depth_img, rgb_shape):
        uv = self.detection_center_uv(det, rgb_shape)
        if uv is None:
            return None

        u, v = uv
        z_m = self.get_depth_m(depth_img, u, v)
        if z_m is None:
            return None

        return self.pixel_to_3d(u, v, z_m)

    def dist3(self, a, b):
        return float(np.linalg.norm(a - b))

    def synchronized_callback(self, img_msg, depth_msg, det_msg):
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

            # Pink detection in RGB frame
            pink_u, pink_v, _, pink_bbox = self.find_pink_center(frame)
            pink_xyz_m = None
            if pink_u is not None and pink_v is not None:
                pink_z_m = self.get_depth_m(depth_img, pink_u, pink_v)
                if pink_z_m is not None:
                    pink_xyz_m = self.pixel_to_3d(pink_u, pink_v, pink_z_m)

            if pink_bbox is not None:
                x, y, w, h = pink_bbox
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.circle(debug_frame, (pink_u, pink_v), 5, (255, 255, 255), -1)
                if pink_xyz_m is not None:
                    cv2.putText(
                        debug_frame,
                        f'pink xyz=({pink_xyz_m[0]:.2f}, {pink_xyz_m[1]:.2f}, {pink_xyz_m[2]:.2f})',
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )

            if self.cooldown > 0:
                self.cooldown -= 1

            detections_3d = []
            for i, det in enumerate(det_msg.detections):
                if not self.detection_is_person(det):
                    continue

                det_pos = self.detection_position_from_bbox(det, depth_img, frame.shape)
                if det_pos is None:
                    continue

                rect = self.detection_rect(det, frame.shape)
                uv = self.detection_center_uv(det, frame.shape)
                score = self.detection_score(det)

                detections_3d.append({
                    'index': i,
                    'det': det,
                    'pos': det_pos,
                    'rect': rect,
                    'uv': uv,
                    'score': score,
                })

            # Draw person detections
            for draw_idx, item in enumerate(detections_3d):
                rect = item['rect']
                uv = item['uv']
                pos = item['pos']
                score = item['score']

                if rect is not None:
                    x, y, w, h = rect
                    cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                if uv is not None:
                    cv2.circle(debug_frame, uv, 5, (255, 0, 0), -1)

                cv2.putText(
                    debug_frame,
                    f'person {item["index"]} s={score:.2f} xyz=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})',
                    (20, 70 + 30 * draw_idx),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 0),
                    2
                )

                if pink_xyz_m is not None:
                    d = self.dist3(pink_xyz_m, pos)
                    cv2.putText(
                        debug_frame,
                        f'd={d:.2f}m',
                        (1150, 70 + 30 * draw_idx),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 0, 0),
                        2
                    )

            chosen_item = None
            chosen_dist = None

            # Primary lock: nearest person to pink point
            if pink_xyz_m is not None and self.cooldown == 0:
                best_item = None
                best_dist = None
                for item in detections_3d:
                    d = self.dist3(pink_xyz_m, item['pos'])
                    if d > self.pink_match_max_dist_m:
                        continue
                    if best_item is None or d < best_dist:
                        best_item = item
                        best_dist = d
                chosen_item = best_item
                chosen_dist = best_dist
                if chosen_item is not None:
                    self.lost_frames = 0
                    self.cooldown = self.relock_cooldown

            # Fallback reacquire: nearest to previous target
            if chosen_item is None and self.target_pos is not None:
                best_item = None
                best_dist = None
                for item in detections_3d:
                    d = self.dist3(self.target_pos, item['pos'])
                    if d > self.reacquire_max_dist_m:
                        continue
                    if best_item is None or d < best_dist:
                        best_item = item
                        best_dist = d
                chosen_item = best_item
                chosen_dist = best_dist

            # Publish chosen target
            if chosen_item is not None:
                det = chosen_item['det']
                det_pos = chosen_item['pos']

                det_id = None
                if hasattr(det, 'id'):
                    try:
                        det_id = det.id if det.id != '' else None
                    except Exception:
                        det_id = None

                self.lost_frames = 0
                self.publish_target_position(det_pos, det_msg.header, det_id=det_id)

                if chosen_item['rect'] is not None:
                    x, y, w, h = chosen_item['rect']
                    cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                if chosen_item['uv'] is not None:
                    cv2.circle(debug_frame, chosen_item['uv'], 6, (0, 255, 0), -1)

                cv2.putText(
                    debug_frame,
                    f'TARGET xyz=({det_pos[0]:.2f}, {det_pos[1]:.2f}, {det_pos[2]:.2f})',
                    (20, debug_frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                if chosen_dist is not None:
                    cv2.putText(
                        debug_frame,
                        f'target dist to pink = {chosen_dist:.2f} m',
                        (20, debug_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                self.publish_debug_image(debug_frame, img_msg.header)
                return

            # No target chosen
            if self.target_pos is not None or self.target_id is not None:
                self.lost_frames += 1
                if self.lost_frames >= self.lost_max:
                    self.target_pos = None
                    self.target_id = None
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
