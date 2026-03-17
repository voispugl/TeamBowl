import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty


class HeartbeatPublisher(Node):
    def __init__(self):
        super().__init__('heartbeat_publisher')
        self.declare_parameter('heartbeat_topic', '/heartbeat')
        self.declare_parameter('publish_rate_hz', 10.0)
        topic = self.get_parameter('heartbeat_topic').value
        rate = self.get_parameter('publish_rate_hz').value
        self.pub = self.create_publisher(Empty, topic, 10)
        self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info(f'HeartbeatPublisher up. topic={topic}, rate={rate}Hz')

    def _publish(self):
        self.pub.publish(Empty())


def main(args=None):
    rclpy.init(args=args)
    node = HeartbeatPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
