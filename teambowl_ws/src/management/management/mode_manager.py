#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ROS Node
class ModeManagerNode(Node):
    """
    Simple mode manager:
      - publishes /teleop_enable (Bool)
      - optionally listens to /teleop_enable_set (Bool) to change it

    Subscribes to:
      - /teleop_enable_set

    Publishes:
      - /teleop_enable
    """

    def __init__(self):
        super().__init__('mode_manager')

        # Topics to help manage operation mode
        self.declare_parameter('teleop_enable_topic', '/teleop_enable')
        self.declare_parameter('teleop_enable_set_topic', '/teleop_enable_set')
        self.declare_parameter('start_teleop_enabled', True)
        self.declare_parameter('publish_rate_hz', 5.0)

        # Read all topics
        self.enable_topic = self.get_parameter('teleop_enable_topic').value
        self.enable_set_topic = self.get_parameter('teleop_enable_set_topic').value
        self.enabled = bool(self.get_parameter('start_teleop_enabled').value)
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)

        # QoS Setup
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribe to teleop enable (#NOTE sent from terminal for now)
        self.sub = self.create_subscription(Bool, self.enable_set_topic, self._mode_cmd_reader, qos)

        # Publish enable topic
        self.pub = self.create_publisher(Bool, self.enable_topic, qos)

        # Create timer to tick every period
        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f'ModeManager up. enable={self.enable_topic} (start={self.enabled}), set={self.enable_set_topic}'
        )

    def _mode_cmd_reader(self, msg: Bool):
        # Mode cmd arrived -> set teleop mode
        self.enabled = bool(msg.data)
        self.get_logger().info(f'teleop_enabled set -> {self.enabled}')

    def _tick(self):
        # Tick through time and publish current mode
        msg = Bool()
        msg.data = self.enabled
        self.pub.publish(msg)

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
