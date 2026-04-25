#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration


class SystemHealthNode(Node):
    """
    Watchdog:
      - subscribes /heartbeat (std_msgs/Empty)
      - publishes /estop (Bool)
      - if heartbeat not received within timeout -> estop true
    """

    def __init__(self):
        super().__init__('system_health')

        # Topics to help system health monitoring
        self.declare_parameter('heartbeat_topic', '/heartbeat')
        self.declare_parameter('estop_topic', '/estop')
        self.declare_parameter('timeout_s', 1.0)
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('start_estop_true', False)

        # Read all topics
        self.heartbeat_topic = self.get_parameter('heartbeat_topic').value
        self.estop_topic = self.get_parameter('estop_topic').value
        self.timeout = Duration(seconds=float(self.get_parameter('timeout_s').value))
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)

        # State variables
        self.estop = bool(self.get_parameter('start_estop_true').value)
        self.last_heartbeat_time = None
        self._kill_switch_latched = False

        # QoS Setup
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribe to heartbeat
        self.sub = self.create_subscription(Empty, self.heartbeat_topic, self._hb_reader, qos)

        # Subscribe to physical kill switch (published by pico_bridge)
        self.create_subscription(Bool, '/kill_switch', self._kill_switch_cb, qos)

        # Subscribe to clear signal — ENABLE button on phone app
        self.create_subscription(Bool, '/clear_estop', self._clear_estop_cb, qos)

        # Publish estop state
        self.pub = self.create_publisher(Bool, self.estop_topic, qos)

        # Create timer to tick every period
        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self._tick)

        # Logging for debugging
        self.get_logger().info(
            f'Watchdog up. heartbeat={self.heartbeat_topic}, estop={self.estop_topic}, timeout={self.timeout.nanoseconds/1e9:.2f}s'
        )

    def _hb_reader(self, _msg: Empty):
        # Heartbeat arrived -> update time
        self.last_heartbeat_time = self.get_clock().now()

    def _kill_switch_cb(self, msg: Bool):
        if msg.data:
            self._kill_switch_latched = True
            self.estop = True
            self.get_logger().warn('Kill switch latched — press ENABLE to clear.')

    def _clear_estop_cb(self, msg: Bool):
        if msg.data:
            self._kill_switch_latched = False
            self.get_logger().info('Kill switch latch cleared via ENABLE.')

    def _tick(self):
        now = self.get_clock().now()
        if self.last_heartbeat_time is None:
            pass
        else:
            timed_out = (now - self.last_heartbeat_time) > self.timeout
            if timed_out:
                self.estop = True
            elif not self._kill_switch_latched:
                self.estop = False

        # Publish estop state
        msg = Bool()
        msg.data = self.estop
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = SystemHealthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
