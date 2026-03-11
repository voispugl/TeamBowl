#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from depthai_ros_msgs.msg import TrackletArray
from cv_bridge import CvBridge
import message_filters

class CamOpsNode(Node):
    """
    Subscribes to:
      - /oak/rgb/image_raw
      - /oak/nn/tracklets

    Publishes: 
      - /robot/target_person_pos
    """
        
    def __init__(self):
        super().__init__('cam_ops_node')
        self.bridge = CvBridge()

        # 1. Setup Publisher for the person's location
        # Using PointStamped includes a header with time and frame information
        self.target_pub = self.create_publisher(PointStamped, '/robot/target_person_pos', 10)

        # 2. Tracking State (from your ppl_detect.py)
        self.lower_pink = np.array([150, 0, 130])
        self.upper_pink = np.array([179, 120, 255])
        self.target_id = None

        # 3. Synchronized Subscribers
        self.image_sub = message_filters.Subscriber(self, Image, '/oak/rgb/image_raw')
        self.track_sub = message_filters.Subscriber(self, TrackletArray, '/oak/nn/tracklets')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.track_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.synchronized_callback)

    def find_pink_center(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_pink, self.upper_pink)
        
        # Cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 300:
                x, y, w, h = cv2.boundingRect(c)
                return (x + w // 2, y + h // 2)
        return None

    def synchronized_callback(self, img_msg, track_msg):
        frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        pink_center = self.find_pink_center(frame) # Logic from your original script

        for t in track_msg.tracklets:
            # Logic: If pink is inside this tracklet's ROI, lock ID
            if pink_center and self.is_point_in_roi(pink_center, t.roi):
                self.target_id = t.id

            # If this is our locked target, publish its position
            if self.target_id is not None and t.id == self.target_id:
                self.publish_target_position(t, img_msg.header.frame_id)

    def is_point_in_roi(self, pt, roi):
        # Maps pink pixel to the tracklet bounding box
        return (roi.x_offset <= pt[0] <= roi.x_offset + roi.width and
                roi.y_offset <= pt[1] <= roi.y_offset + roi.height)
    
    def publish_target_position(self, tracklet, frame_id):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        
        # Pulling spatial data from the tracklet
        msg.point.x = tracklet.position.x
        msg.point.y = tracklet.position.y
        msg.point.z = tracklet.position.z
        
        self.target_pub.publish(msg)

def main():
    rclpy.init()
    rclpy.spin(CamOpsNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()