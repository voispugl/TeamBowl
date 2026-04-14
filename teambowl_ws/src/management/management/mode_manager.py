#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# ROS Node
class ModeManagerNode(Node):
    """
    Runtime mode manager

    Subscribes to:
      - /robot_mode_set (String): "off", "teleop", "auton"

    Publishes:
      - /robot_mode (String): "off", "teleop", "auton"
    """

    VALID_MODES = {'off', 'teleop', 'auton', 'balance'}

    def __init__(self):
        super().__init__('mode_manager')

        # Declare parameters for mode management
        self.declare_parameter('mode_topic', '/robot_mode')
        self.declare_parameter('mode_set_topic', '/robot_mode_set')
        self.declare_parameter('start_mode', 'off')
        self.declare_parameter('publish_rate_hz', 5.0)

        # Read/store all values from input parameters
        self.mode_topic = self.get_parameter('mode_topic').value
        self.mode_set_topic = self.get_parameter('mode_set_topic').value
        self.mode = str(self.get_parameter('start_mode').value).strip().lower()
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)

        # Safety fallback in case of invalid mode
        if self.mode not in self.VALID_MODES:
            self.get_logger().warn(
                f'Invalid start_mode="{self.mode}". Falling back to "off".'
            )
            self.mode = 'off'

        # QoS Setup
        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Subscribe to mode set topic (sent from terminal)
        self.sub = self.create_subscription(String, self.mode_set_topic, self._mode_cmd_reader, sub_qos)

        # Publish current mode topic
        self.pub = self.create_publisher(String, self.mode_topic, pub_qos)

        # Create timer to tick every period
        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f'ModeManager up. mode_topic={self.mode_topic}, '
            f'mode_set_topic={self.mode_set_topic}, start_mode={self.mode}'
        )

        self._publish_mode()

    def _publish_mode(self):
        msg = String()
        msg.data = self.mode
        self.pub.publish(msg)

    def _mode_cmd_reader(self, msg: String):
        # Requested mode change has arrived
        requested = msg.data.strip().lower()

        # If not a valid mode, reject
        if requested not in self.VALID_MODES:
            self.get_logger().warn(
                f'Ignoring invalid mode request "{msg.data}". '
                f'Valid modes: {sorted(self.VALID_MODES)}'
            )
            return
        
        # Ignore request for current mode
        if requested == self.mode:
            return

        # Valid mode change request, apply it
        self.mode = requested
        self.get_logger().info(f'robot_mode set -> {self.mode}')
        self._publish_mode()
        
    def _tick(self):
        # Publish current mode
        self._publish_mode()

def main():
    rclpy.init()
    node = ModeManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
