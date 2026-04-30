#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class PersonScanFilter(Node):
    def __init__(self):
        super().__init__('person_scan_filter')
        self.declare_parameter('exclusion_radius_m', 1.5)
        self.declare_parameter('person_timeout_s', 1.0)
        self.declare_parameter('input_scan_topic', '/oak/nav_scan')
        self.declare_parameter('output_scan_topic', '/oak/nav_scan_filtered')
        self.declare_parameter('person_pos_topic', '/user_pos_base_link')

        self._r2 = self.get_parameter('exclusion_radius_m').value ** 2
        self._timeout = self.get_parameter('person_timeout_s').value

        self._px: float | None = None
        self._py: float | None = None
        self._person_stamp = None

        self.create_subscription(
            PointStamped,
            self.get_parameter('person_pos_topic').value,
            self._person_cb,
            10,
        )
        self.create_subscription(
            LaserScan,
            self.get_parameter('input_scan_topic').value,
            self._scan_cb,
            qos_profile_sensor_data,
        )
        self._pub = self.create_publisher(
            LaserScan,
            self.get_parameter('output_scan_topic').value,
            qos_profile_sensor_data,
        )

    def _person_cb(self, msg: PointStamped) -> None:
        self._px = msg.point.x
        self._py = msg.point.y
        self._person_stamp = self.get_clock().now()

    def _scan_cb(self, msg: LaserScan) -> None:
        # Pass through unmodified when no fresh person position.
        if self._person_stamp is None or self._px is None:
            self._pub.publish(msg)
            return
        age = (self.get_clock().now() - self._person_stamp).nanoseconds * 1e-9
        if age > self._timeout:
            self._pub.publish(msg)
            return

        px, py, r2 = self._px, self._py, self._r2
        new_ranges = list(msg.ranges)
        angle = msg.angle_min
        for i in range(len(new_ranges)):
            r = new_ranges[i]
            if math.isfinite(r):
                sx = r * math.cos(angle)
                sy = r * math.sin(angle)
                if (sx - px) ** 2 + (sy - py) ** 2 < r2:
                    new_ranges[i] = float('inf')
            angle += msg.angle_increment

        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min
        out.angle_max = msg.angle_max
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = new_ranges
        out.intensities = msg.intensities
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PersonScanFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
