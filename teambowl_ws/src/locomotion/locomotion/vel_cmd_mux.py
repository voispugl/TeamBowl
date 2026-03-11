#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from rclpy.duration import Duration

# Helpers
def zero_twist() -> Twist:
    msg = Twist() # inits lin.xyz and ang.xyz as zero
    msg.linear.x = 0.
    msg.linear.y = 0.
    msg.linear.z = 0.
    msg.angular.x = 0.
    msg.angular.y = 0.
    msg.angular.z = 0.
    return msg

# ROS Node
class VelCmdMuxNode(Node):
    """
    Publishes /cmd_vel_selected based on:
      - /estop (Bool): forces zero output
      - /teleop_enable (Bool): if true, choose teleop vel cmd
      - freshness of /cmd_vel_teleop and /cmd_vel_auto

    Subscribes to:
      - /teleop_enable
      - /cmd_vel_teleop
      - /cmd_vel_auto
      - /estop

    Publishes:
      - /cmd_vel_selected (teleop cmd, auto cmd, or zero cmd)
    """

    def __init__(self):
        super().__init__('vel_cmd_mux')

        # Topics for velocity selection
        self.declare_parameter('teleop_enable_topic', '/teleop_enable')
        self.declare_parameter('teleop_topic', '/cmd_vel_teleop')
        self.declare_parameter('auto_topic', '/cmd_vel_auto')
        self.declare_parameter('estop_topic', '/estop')
        self.declare_parameter('output_topic', '/cmd_vel_selected')
        # Cmd timeout topics for safety
        self.declare_parameter('teleop_timeout_s', 0.5)
        self.declare_parameter('auto_timeout_s', 0.5)
        self.declare_parameter('publish_rate_hz', 30.)

        # Read all topics
        self.teleop_enable_topic = self.get_parameter('teleop_enable_topic').value
        self.teleop_topic = self.get_parameter('teleop_topic').value
        self.auto_topic = self.get_parameter('auto_topic').value
        self.estop_topic = self.get_parameter('estop_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.teleop_timeout = Duration(seconds=float(self.get_parameter('teleop_timeout_s').value))
        self.auto_timeout = Duration(seconds=float(self.get_parameter('auto_timeout_s').value))
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)

        # State variabels
        self.teleop_enabled = None
        self.estop = False
        self.last_teleop = zero_twist()
        self.last_auto = zero_twist()
        self.last_teleop_time = None
        self.last_auto_time = None

        # QoS Setup
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribe to topics
        self.sub_teleop = self.create_subscription(Twist, self.teleop_topic, self._teleop_reader, qos)
        self.sub_auto = self.create_subscription(Twist, self.auto_topic, self._auto_reader, qos)
        self.sub_enable = self.create_subscription(Bool, self.teleop_enable_topic, self._enable_reader, qos)
        self.sub_estop = self.create_subscription(Bool, self.estop_topic, self._estop_reader, qos)

        # Publish selected velocity
        self.pub_out = self.create_publisher(Twist, self.output_topic, qos)

        # Create internal clock which executes _tick every period
        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self._tick)

        # Logging for debugging
        self.get_logger().info(
            f'CmdMux up. teleop={self.teleop_topic}, auto={self.auto_topic}, out={self.output_topic}, '
            f'enable={self.teleop_enable_topic}, estop={self.estop_topic}'
        )

    def _publish_selected(self, msg: Twist):
        if self.estop:
            self.pub_out.publish(zero_twist())
            return
        self.pub_out.publish(msg)

    def _teleop_reader(self, msg: Twist):
        # Teleop msg arrived -> update twist, get time
        self.last_teleop = msg
        self.last_teleop_time = self.get_clock().now()

        if self.teleop_enabled is True:
            self._publish_selected(self.last_teleop)

    def _auto_reader(self, msg: Twist):
        # Auto msg arrived -> update twist, get time
        self.last_auto = msg
        self.last_auto_time = self.get_clock().now()

        if self.teleop_enabled is False:
            self._publish_selected(self.last_auto)

    def _enable_reader(self, msg: Bool):
	# Get teleop state
	new_enabled = bool(msg.data)

	# Ignore repeated messages with no state change
	if self.teleop_enabled is not None and new_enabled == self.teleop_enabled:
	    return

        # Update teleop state (on/off)
        self.teleop_enabled = new_enabled

        if self.teleop_enabled:
            if self._fresh(self.last_teleop_time, self.teleop_timeout):
                self._publish_selected(self.last_teleop)
            else:
                self._publish_selected(zero_twist())
        else:
            if self._fresh(self.last_auto_time, self.auto_timeout):
                self._publish_selected(self.last_auto)
            else:
                self._publish_selected(zero_twist())

    def _estop_reader(self, msg: Bool):
        # Update estop state (on/off)
        new_estop = bool(msg.data)
        if new_estop and not self.estop:
            self._publish_selected(zero_twist())
        self.estop = new_estop

    def _fresh(self, last_time, timeout: Duration) -> bool:
        # Check how fresh a certain vel cmd is
        if last_time is None:
            return False
        return (self.get_clock().now() - last_time) <= timeout

    def _tick(self):
        if self.estop:
            self._publish_selected(zero_twist())
            return

        if self.teleop_enabled is True:
            if not self._fresh(self.last_teleop_time, self.teleop_timeout):
                self._publish_selected(zero_twist())
        elif self.teleop_enabled is False:
            if not self._fresh(self.last_auto_time, self.auto_timeout):
                self._publish_selected(zero_twist())

def main():
    rclpy.init()
    node = VelCmdMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
