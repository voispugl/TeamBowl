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

        # Depth limits
        self.declare_parameter('min_depth_m', 0.2)
        self.declare_parameter('max_depth_m', 8.0)

        # Pink detection
        self.declare_parameter('min_pink_area_px', 300)

        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value

        self.target_topic = self.get_parameter('target_topic').value
        self.target_valid_topic = self.get_parameter('target_valid_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value

        self.min_depth_m = float(self.get_parameter('min_depth_m').value)
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)

        self.min_pink_area_px = int(self.get_parameter('min_pink_area_px').value)

        # HSV range for pink pants
        self.lower_pink = np.array([140,150,120], dtype=np.uint8)
        self.upper_pink = np.array([175,255,220], dtype=np.uint8)

        # Camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Publishers
        self.target_pub = self.create_publisher(PointStamped, self.target_topic, 10)
        self.valid_pub = self.create_publisher(Bool, self.target_valid_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        # Camera info
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data
        )

        # Sync RGB + depth
        img_sub = message_filters.Subscriber(
            self, Image, self.image_topic, qos_profile=qos_profile_sensor_data)

        depth_sub = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)

        ts = message_filters.ApproximateTimeSynchronizer(
            [img_sub, depth_sub], 10, 0.2)

        ts.registerCallback(self.callback)

        self.get_logger().info("cam_ops_node started (pink tracking mode)")


    # -------------------------------------------------
    # Camera intrinsics
    # -------------------------------------------------

    def camera_info_callback(self, msg):

        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]


    # -------------------------------------------------
    # Pink detection
    # -------------------------------------------------

    def detect_pink(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower_pink, self.upper_pink)

        kernel = np.ones((5,5), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(contour) < self.min_pink_area_px:
            return None

        x,y,w,h = cv2.boundingRect(contour)

        u = x + w//2
        v = y + h//2

        return (u,v,(x,y,w,h))


    # -------------------------------------------------
    # Depth helpers
    # -------------------------------------------------

    def depth_at_pixel(self, depth_img, u, v):

        h,w = depth_img.shape

        if not (0<=u<w and 0<=v<h):
            return None

        patch = depth_img[v-2:v+3, u-2:u+3]

        valid = patch[patch>0]

        if valid.size==0:
            return None

        depth_m = np.median(valid)/1000.0

        if depth_m<self.min_depth_m or depth_m>self.max_depth_m:
            return None

        return depth_m


    def pixel_to_xyz(self, u, v, z):

        x = (u-self.cx)*z/self.fx
        y = (v-self.cy)*z/self.fy

        return np.array([x,y,z])


    # -------------------------------------------------
    # Main callback
    # -------------------------------------------------

    def callback(self, img_msg, depth_msg):

        frame = self.bridge.imgmsg_to_cv2(img_msg,'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg,'passthrough')

        debug = frame.copy()

        if self.fx is None:
            return

        result = self.detect_pink(frame)

        if result is None:

            msg = Bool()
            msg.data = False
            self.valid_pub.publish(msg)

            self.publish_debug(debug,img_msg.header)

            return

        u,v,bbox = result

        x,y,w,h = bbox

        z = self.depth_at_pixel(depth,u,v)

        if z is None:

            msg = Bool()
            msg.data = False
            self.valid_pub.publish(msg)

            self.publish_debug(debug,img_msg.header)

            return

        xyz = self.pixel_to_xyz(u,v,z)

        # Publish target
        pt = PointStamped()
        pt.header = img_msg.header
        pt.point.x = float(xyz[0])
        pt.point.y = float(xyz[1])
        pt.point.z = float(xyz[2])

        self.target_pub.publish(pt)

        valid = Bool()
        valid.data = True
        self.valid_pub.publish(valid)

        # Debug drawing
        cv2.rectangle(debug,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.circle(debug,(u,v),5,(0,255,0),-1)

        cv2.putText(debug,
            f"XYZ: {xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,(0,255,0),2)

        self.publish_debug(debug,img_msg.header)


    def publish_debug(self,img,header):

        msg = self.bridge.cv2_to_imgmsg(img,'bgr8')
        msg.header = header
        self.debug_pub.publish(msg)


def main():

    rclpy.init()

    node = CamOpsNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__=="__main__":
    main()