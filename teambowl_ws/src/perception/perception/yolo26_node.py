#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
import message_filters

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Int32
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from ultralytics import YOLO


class Yolo26Node(Node):

    def __init__(self):
        super().__init__('yolo26_node')
        self.bridge = CvBridge()

        self.declare_parameter('image_topic', '/oak/rgb/image_raw')
        self.declare_parameter('depth_topic', '/oak/stereo/image_raw')
        self.declare_parameter('camera_info_topic', '/oak/rgb/camera_info')
        self.declare_parameter('model_path', '/home/box/TeamBowl/models/yolo26n.engine')
        self.declare_parameter('tracker_config', '')
        self.declare_parameter('min_confidence', 0.5)
        self.declare_parameter('min_depth_m', 0.2)
        self.declare_parameter('max_depth_m', 8.0)
        self.declare_parameter('depth_window_radius_px', 2)
        self.declare_parameter('target_lost_timeout_s', 15.0)
        self.declare_parameter('reid_weights', '')
        self.declare_parameter('reid_threshold', 0.65)
        self.declare_parameter('reid_update_interval_s', 1.0)

        image_topic       = self.get_parameter('image_topic').value
        depth_topic       = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        model_path        = self.get_parameter('model_path').value
        self.min_conf     = float(self.get_parameter('min_confidence').value)
        self.min_depth    = float(self.get_parameter('min_depth_m').value)
        self.max_depth    = float(self.get_parameter('max_depth_m').value)
        self.depth_r       = int(self.get_parameter('depth_window_radius_px').value)
        self.lost_timeout  = float(self.get_parameter('target_lost_timeout_s').value)
        self._tracker_cfg  = self.get_parameter('tracker_config').value or 'bytetrack.yaml'
        self._reid_weights_path = str(self.get_parameter('reid_weights').value)
        self.reid_threshold = float(self.get_parameter('reid_threshold').value)
        self.reid_update_interval_s = float(self.get_parameter('reid_update_interval_s').value)

        self.fx = self.fy = self.cx = self.cy = None

        # Track ID we're locked onto (None = waiting for first detection)
        self._target_id: int | None = None
        self._target_last_seen: float | None = None  # ROS time in seconds

        # ReID state
        self._reid_encoder = None
        self._target_embedding: np.ndarray | None = None  # EMA of target appearance
        self._last_reid_update_s: float | None = None

        self.get_logger().info(f'Loading YOLO26 model from {model_path} ...')
        self.yolo = YOLO(model_path, task='detect')
        self.get_logger().info('YOLO26 model loaded.')

        self.det_pub    = self.create_publisher(Detection2DArray, '/yolo26/detections', 10)
        self.pos_pub    = self.create_publisher(PointStamped,     '/yolo26/user_pos', 10)
        self.valid_pub  = self.create_publisher(Bool,             '/yolo26/user_valid', 10)
        self.id_pub     = self.create_publisher(Int32,            '/yolo26/target_id', 10)
        self.dbg_pub    = self.create_publisher(Image,            '/yolo26/debug_image', qos_profile_sensor_data)

        self.create_subscription(Bool, '/yolo26/relock', self._relock_cb, 10)

        self.cb_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(
            CameraInfo, camera_info_topic,
            self._camera_info_cb, qos_profile_sensor_data)

        self.rgb_sub = message_filters.Subscriber(
            self, Image, image_topic,
            qos_profile=qos_profile_sensor_data, callback_group=self.cb_group)
        self.depth_sub = message_filters.Subscriber(
            self, Image, depth_topic,
            qos_profile=qos_profile_sensor_data, callback_group=self.cb_group)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.3)
        self.ts.registerCallback(self._callback)

        self._last_stamp = self.get_clock().now()
        self.create_timer(2.0, self._watchdog_cb, callback_group=self.cb_group)

        self.get_logger().info('yolo26_node ready — will lock onto first detected person.')

    def _relock_cb(self, msg: Bool):
        self._target_id = None
        self._target_last_seen = None
        self._target_embedding = None
        self._last_reid_update_s = None
        self.get_logger().info('Manual relock triggered — waiting for next person.')

    def _get_reid_encoder(self):
        if self._reid_encoder is None and self._reid_weights_path:
            try:
                import onnxruntime as ort
                sess = ort.InferenceSession(
                    self._reid_weights_path,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                )
                self._reid_encoder = sess
                self.get_logger().info(f'ReID encoder loaded from {self._reid_weights_path}')
            except Exception as e:
                self.get_logger().warn(f'ReID encoder load failed: {e} — relocking by size only')
                self._reid_weights_path = ''  # stop retrying
        return self._reid_encoder

    def _get_embedding(self, frame, x1, y1, x2, y2):
        sess = self._get_reid_encoder()
        if sess is None:
            return None
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        try:
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, (128, 256))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            crop = (crop - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            inp = crop.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
            emb = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0]
            norm = np.linalg.norm(emb)
            return emb / (norm + 1e-8)
        except Exception:
            return None

    def _cosine_sim(self, a, b):
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b))  # both already unit-norm from boxmot

    def _camera_info_cb(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def _watchdog_cb(self):
        age = (self.get_clock().now() - self._last_stamp).nanoseconds / 1e9
        if age > 2.0:
            self.get_logger().warn(f'No synced frames in {age:.1f}s')

    def _get_depth_m(self, depth_img, u, v):
        h, w = depth_img.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None
        r = self.depth_r
        patch = depth_img[max(0, v-r):min(h, v+r+1), max(0, u-r):min(w, u+r+1)]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        z = float(np.median(valid)) / 1000.0
        return z if self.min_depth <= z <= self.max_depth else None

    def _pixel_to_3d(self, u, v, z):
        if self.fx is None:
            return None
        return np.array([
            (float(u) - self.cx) * z / self.fx,
            (float(v) - self.cy) * z / self.fy,
            z,
        ], dtype=np.float64)

    def _callback(self, rgb_msg, depth_msg):
        try:
            self._last_stamp = self.get_clock().now()
            now_s = self._last_stamp.nanoseconds / 1e9

            frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

            if depth_msg.encoding != '16UC1':
                self._publish_valid(False)
                return

            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            results = self.yolo.track(
                frame,
                classes=[0],
                conf=self.min_conf,
                persist=True,        # maintain tracker state across calls
                tracker=self._tracker_cfg,
                verbose=False,
            )
            boxes = results[0].boxes

            rgb_h, rgb_w = frame.shape[:2]
            depth_h, depth_w = depth_img.shape[:2]

            # Build a list of (track_id, x1, y1, x2, y2, conf) for valid tracked boxes
            tracked = []
            for box in boxes:
                tid = int(box.id[0]) if box.id is not None else None
                if tid is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                tracked.append((tid, x1, y1, x2, y2, float(box.conf[0])))

            # Publish all detections
            det_array = Detection2DArray()
            det_array.header = rgb_msg.header
            for tid, x1, y1, x2, y2, conf in tracked:
                d = Detection2D()
                d.header = rgb_msg.header
                d.bbox.center.position.x = float((x1 + x2) / 2)
                d.bbox.center.position.y = float((y1 + y2) / 2)
                d.bbox.size_x = float(x2 - x1)
                d.bbox.size_y = float(y2 - y1)
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(tid)
                hyp.hypothesis.score = conf
                d.results.append(hyp)
                det_array.detections.append(d)
            self.det_pub.publish(det_array)

            dbg = frame.copy()

            # Auto-relock if target lost for too long
            if self._target_id is not None and self._target_last_seen is not None:
                if now_s - self._target_last_seen > self.lost_timeout:
                    self.get_logger().info(
                        f'Target ID {self._target_id} lost for >{self.lost_timeout:.0f}s — re-locking on next detection.')
                    self._target_id = None
                    self._target_last_seen = None

            # Lock onto a person if we have no target yet — use ReID if we have a prior appearance
            if self._target_id is None and tracked:
                if self._target_embedding is not None and self._get_reid_encoder() is not None:
                    best_tid, best_sim = None, 0.0
                    for tid, x1, y1, x2, y2, conf in tracked:
                        emb = self._get_embedding(frame, x1, y1, x2, y2)
                        sim = self._cosine_sim(emb, self._target_embedding)
                        if sim > best_sim:
                            best_sim, best_tid = sim, tid
                    if best_sim >= self.reid_threshold:
                        self._target_id = best_tid
                        self.get_logger().info(
                            f'ReID: re-locked onto ID {self._target_id} (similarity={best_sim:.2f})')
                    else:
                        best = max(tracked, key=lambda t: (t[3] - t[1]) * (t[4] - t[2]))
                        self._target_id = best[0]
                        self.get_logger().info(
                            f'ReID: no match (best={best_sim:.2f} < {self.reid_threshold:.2f}) '
                            f'— locked onto largest (ID {self._target_id})')
                else:
                    best = max(tracked, key=lambda t: (t[3] - t[1]) * (t[4] - t[2]))
                    self._target_id = best[0]
                    self.get_logger().info(f'Locked onto person with track ID {self._target_id}.')
                self._target_last_seen = now_s

            # Find the locked target in this frame
            target_xyz = None
            for tid, x1, y1, x2, y2, conf in tracked:
                is_target = (tid == self._target_id)
                color = (0, 255, 0) if is_target else (128, 128, 128)
                thickness = 3 if is_target else 1
                cv2.rectangle(dbg, (x1, y1), (x2, y2), color, thickness)
                label = f'ID{tid} {conf:.2f}' + (' [TARGET]' if is_target else '')
                cv2.putText(dbg, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, thickness)

                if is_target:
                    cx_px = (x1 + x2) // 2
                    cy_px = (y1 + y2) // 2
                    du = int(cx_px * depth_w / rgb_w)
                    dv = int(cy_px * depth_h / rgb_h)
                    z = self._get_depth_m(depth_img, du, dv)
                    if z is not None:
                        target_xyz = self._pixel_to_3d(cx_px, cy_px, z)
                        self._target_last_seen = now_s
                        # Update appearance embedding at most once per reid_update_interval_s
                        if (self._last_reid_update_s is None or
                                now_s - self._last_reid_update_s >= self.reid_update_interval_s):
                            emb = self._get_embedding(frame, x1, y1, x2, y2)
                            if emb is not None:
                                if self._target_embedding is None:
                                    self._target_embedding = emb
                                else:
                                    self._target_embedding = (
                                        0.9 * self._target_embedding + 0.1 * emb
                                    )
                                self._last_reid_update_s = now_s

            if target_xyz is not None:
                pos = PointStamped()
                pos.header = rgb_msg.header
                pos.point.x, pos.point.y, pos.point.z = target_xyz.tolist()
                self.pos_pub.publish(pos)
                self._publish_valid(True)
                cv2.putText(dbg, f'XYZ=({target_xyz[0]:.2f},{target_xyz[1]:.2f},{target_xyz[2]:.2f})',
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                self._publish_valid(False)
                status = f'target ID {self._target_id} not visible' if self._target_id else 'no person detected'
                cv2.putText(dbg, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Always publish current target ID (-1 = none)
            id_msg = Int32()
            id_msg.data = self._target_id if self._target_id is not None else -1
            self.id_pub.publish(id_msg)

            dbg_msg = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
            dbg_msg.header = rgb_msg.header
            self.dbg_pub.publish(dbg_msg)

        except Exception as e:
            self.get_logger().error(f'_callback failed: {e}')

    def _publish_valid(self, valid: bool):
        msg = Bool()
        msg.data = valid
        self.valid_pub.publish(msg)


def main():
    rclpy.init()
    node = Yolo26Node()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
