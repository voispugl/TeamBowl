#!/usr/bin/env python3

import math

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool
from vision_msgs.msg import Detection3DArray


class CamOpsNode(Node):
    """
    Subscribes to:
      - /oak/rgb/image_raw
      - /oak/stereo/image_raw        (aligned depth, 16UC1, mm)
      - /oak/stereo/camera_info
      - /oak/nn/spatial_detections   (vision_msgs/Detection3DArray)

    Publishes:
      - /robot/target_person_pos   (PointStamped, meters)
      - /robot/target_valid        (Bool)
    """

    def __init__(self):
        super().__init__('cam_ops_node')
        self.bridge = CvBridge()

        # Topics
        self.declare_parameter('image_topic', '/oak/rgb/image_raw')
        self.declare_parameter('depth_topic', '/oak/stereo/image_raw')
        self.declare_parameter('camera_info_topic', '/oak/stereo/camera_info')
        self.declare_parameter('detections_topic', '/oak/nn/spatial_detections')
        self.declare_parameter('target_topic', '/robot/target_person_pos')
        self.declare_parameter('target_valid_topic', '/robot/target_valid')

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

        # Debugging
        self.declare_parameter('debug_image_topic', '/robot/debug/cam_ops_image')
        
        # Read parameters
        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.detections_topic = self.get_parameter('detections_topic').value
        self.target_topic = self.get_parameter('target_topic').value
        self.target_valid_topic = self.get_parameter('target_valid_topic').value

        self.sync_slop_s = float(self.get_parameter('sync_slop_s').value)
        self.min_pink_area_px = int(self.get_parameter('min_pink_area_px').value)
        self.lost_max = int(self.get_parameter('lost_max').value)
        self.relock_cooldown = int(self.get_parameter('relock_cooldown').value)

        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)
        self.depth_window_radius_px = int(self.get_parameter('depth_window_radius_px').value)

        self.pink_match_max_dist_m = float(self.get_parameter('pink_match_max_dist_m').value)
        self.reacquire_max_dist_m = float(self.get_parameter('reacquire_max_dist_m').value)

        self.debug_image_topic = self.get_parameter('debug_image_topic').value

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
        self.target_pos = None   # np.array([x,y,z]) in meters
        self.lost_frames = 0
        self.cooldown = 0

        # Subscribers
        self.image_sub = message_filters.Subscriber(self, Image, self.image_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, self.camera_info_topic)
        self.det_sub = message_filters.Subscriber(self, Detection3DArray, self.detections_topic)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub, self.info_sub, self.det_sub],
            queue_size=10,
            slop=self.sync_slop_s
        )
        self.ts.registerCallback(self.synchronized_callback)

        self.get_logger().info('cam_ops_node started')

    def find_pink_center(self, frame_bgr):
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_pink, self.upper_pink)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, 0, None

        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < self.min_pink_area_px:
            return None, None, 0, None

        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2
        cy = y + h // 2

        # [debugging] pink detection
        # self.get_logger().info(f"Pink Area: {area} px")
        
        return cx, cy, area, (x, y, w, h)
    
    def publish_debug_image(self, frame_bgr, header):
        msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding='bgr8')
        msg.header = header
        self.debug_image_pub.publish(msg)

    def update_intrinsics(self, info_msg: CameraInfo):
        self.fx = float(info_msg.k[0])
        self.fy = float(info_msg.k[4])
        self.cx = float(info_msg.k[2])
        self.cy = float(info_msg.k[5])

    def get_depth_m(self, depth_img, u, v):
        """
        depth_img is expected to be 16UC1 in mm.
        Use a small window median around the pink center to reduce bad pixels.
        """
        h, w = depth_img.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None

        r = self.depth_window_radius_px
        u0 = max(0, u - r)
        u1 = min(w, u + r + 1)
        v0 = max(0, v - r)
        v1 = min(h, v + r + 1)

        patch = depth_img[v0:v1, u0:u1]

        # 16UC1 depth in mm; ignore zeros
        valid = patch[patch > 0]
        if valid.size == 0:
            return None

        depth_mm = float(np.median(valid))
        depth_m = depth_mm / 1000.0

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
        """
        Keep this tolerant because class IDs can vary by config.
        For many setups the class_id may literally be 'person'.
        """
        if not det.results:
            return False

        best = det.results[0]
        class_id = str(best.hypothesis.class_id).strip().lower()

        # Common possibilities
        if class_id in ('person', '15'):
            return True

        return False

    def detection_score(self, det):
        if not det.results:
            return 0.0
        return float(det.results[0].hypothesis.score)

    def detection_position(self, det):
        """
        Use the first / best hypothesis pose position.
        Assumes meters, which is standard for Detection3D pose.
        """
        if not det.results:
            return None

        p = det.results[0].pose.pose.position
        return np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)

    def dist3(self, a, b):
        return float(np.linalg.norm(a - b))

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

    def choose_detection_from_pink(self, pink_xyz_m, detections):
        best_det = None
        best_dist = None

        for det in detections:
            if not self.detection_is_person(det):
                continue

            det_pos = self.detection_position(det)
            if det_pos is None:
                continue

            d = self.dist3(pink_xyz_m, det_pos)
            if d > self.pink_match_max_dist_m:
                continue

            if best_det is None or d < best_dist:
                best_det = det
                best_dist = d

        return best_det

    def choose_detection_from_previous_target(self, detections):
        if self.target_pos is None:
            return None

        best_det = None
        best_dist = None

        for det in detections:
            if not self.detection_is_person(det):
                continue

            det_pos = self.detection_position(det)
            if det_pos is None:
                continue

            d = self.dist3(self.target_pos, det_pos)
            if d > self.reacquire_max_dist_m:
                continue

            if best_det is None or d < best_dist:
                best_det = det
                best_dist = d

        return best_det

    def synchronized_callback(self, img_msg, depth_msg, info_msg, det_msg):
        # Update camera intrinsics
        self.update_intrinsics(info_msg)

        # Convert images
        frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        debug_frame = frame.copy()

        if depth_msg.encoding != '16UC1':
            self.get_logger().warn(
                f'Unexpected depth encoding {depth_msg.encoding}; expected 16UC1'
            )
            cv2.putText(debug_frame, f'Bad depth encoding: {depth_msg.encoding}',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.publish_debug_image(debug_frame, img_msg.header)
            self.publish_target_valid(False)
            return

        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        # Pink detection
        u, v, area, bbox = self.find_pink_center(frame)
        pink_xyz_m = None

        if u is not None and v is not None:
            z_m = self.get_depth_m(depth_img, u, v)
            if z_m is not None:
                pink_xyz_m = self.pixel_to_3d(u, v, z_m)

        ## START DEBUGGING ##
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.circle(debug_frame, (u, v), 5, (255, 255, 255), -1)
            cv2.putText(debug_frame, f'pink area={int(area)}',
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if pink_xyz_m is not None:
            cv2.putText(debug_frame,
                        f'pink xyz=({pink_xyz_m[0]:.2f}, {pink_xyz_m[1]:.2f}, {pink_xyz_m[2]:.2f})',
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(debug_frame, 'pink xyz: none',
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.get_logger().info(f'Num dets: {len(det_msg.detections)}')
        for det in det_msg.detections:
            if not det.results:
                continue

            cls = str(det.results[0].hypothesis.class_id)
            score = float(det.results[0].hypothesis.score)
            pos = det.results[0].pose.pose.position
            self.get_logger().info(
                f'det id={det.id} class={cls} score={score:.3f} '
                f'xyz=({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})'
            )

            if pink_xyz_m is not None:
                self.get_logger().info(
                    f'pink xyz: ({pink_xyz_m[0]:.3f}, {pink_xyz_m[1]:.3f}, {pink_xyz_m[2]:.3f})'
                )
                det_pos = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
                d = np.linalg.norm(det_pos - pink_xyz_m)
                self.get_logger().info(f'  dist_to_pink={d:.3f} m')
        ## END DEBUGGING ##

        if self.cooldown > 0:
            self.cooldown -= 1

        chosen_det = None
        chosen_idx = -1
        chosen_dist = None

        # Draw detections
        for i, det in enumerate(det_msg.detections):
            if not det.results:
                continue

            cls = str(det.results[0].hypothesis.class_id).strip().lower()
            score = float(det.results[0].hypothesis.score)
            pos = det.results[0].pose.pose.position
            det_pos = np.array([float(pos.x), float(pos.y), float(pos.z)], dtype=np.float64)

            # Only care about people
            is_person = cls in ('person', '15')
            if not is_person:
                continue

            # We do not have 2D boxes from Detection3DArray, so just print detections in a list
            y_text = 70 + 30 * i
            cv2.putText(debug_frame,
                        f'det {i}: cls={cls} score={score:.2f} xyz=({det_pos[0]:.2f}, {det_pos[1]:.2f}, {det_pos[2]:.2f})',
                        (20, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

            if pink_xyz_m is not None:
                d = self.dist3(pink_xyz_m, det_pos)
                cv2.putText(debug_frame,
                            f'd={d:.2f}m',
                            (900, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        # Primary lock source
        if pink_xyz_m is not None and self.cooldown == 0:
            best_det = None
            best_dist = None
            best_idx = -1

            for i, det in enumerate(det_msg.detections):
                if not self.detection_is_person(det):
                    continue

                det_pos = self.detection_position(det)
                if det_pos is None:
                    continue

                d = self.dist3(pink_xyz_m, det_pos)
                if d > self.pink_match_max_dist_m:
                    continue

                if best_det is None or d < best_dist:
                    best_det = det
                    best_dist = d
                    best_idx = i

            chosen_det = best_det
            chosen_idx = best_idx
            chosen_dist = best_dist

            if chosen_det is not None:
                self.lost_frames = 0
                self.cooldown = self.relock_cooldown

        # Fallback reacquire
        if chosen_det is None and self.target_pos is not None:
            best_det = None
            best_dist = None
            best_idx = -1

            for i, det in enumerate(det_msg.detections):
                if not self.detection_is_person(det):
                    continue

                det_pos = self.detection_position(det)
                if det_pos is None:
                    continue

                d = self.dist3(self.target_pos, det_pos)
                if d > self.reacquire_max_dist_m:
                    continue

                if best_det is None or d < best_dist:
                    best_det = det
                    best_dist = d
                    best_idx = i

            chosen_det = best_det
            chosen_idx = best_idx
            chosen_dist = best_dist

        # Draw chosen target
        if chosen_det is not None:
            det_pos = self.detection_position(chosen_det)
            det_id = chosen_det.id if chosen_det.id != '' else None
            self.lost_frames = 0
            self.publish_target_position(det_pos, det_msg.header, det_id=det_id)

            cv2.putText(debug_frame,
                        f'TARGET det {chosen_idx} xyz=({det_pos[0]:.2f}, {det_pos[1]:.2f}, {det_pos[2]:.2f})',
                        (20, debug_frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if chosen_dist is not None:
                cv2.putText(debug_frame,
                            f'target dist to pink = {chosen_dist:.2f} m',
                            (20, debug_frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            self.publish_debug_image(debug_frame, img_msg.header)
            return

        # No target chosen
        if self.target_pos is not None or self.target_id is not None:
            self.lost_frames += 1
            if self.lost_frames >= self.lost_max:
                self.target_pos = None
                self.target_id = None
                self.lost_frames = 0

        cv2.putText(debug_frame, 'TARGET: none',
                    (20, debug_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.publish_target_valid(False)
        self.publish_debug_image(debug_frame, img_msg.header)


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
