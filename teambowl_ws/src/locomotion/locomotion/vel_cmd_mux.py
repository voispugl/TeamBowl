#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
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
      - /robot_mode (String): "off", "teleop", "auton"
      - freshness of /cmd_vel_teleop and /cmd_vel_auto

    Subscribes to:
      - /robot_mode
      - /cmd_vel_teleop
      - /cmd_vel_auto
      - /estop

    Publishes:
      - /cmd_vel_selected (teleop cmd, auto cmd, or zero cmd)
    """

    VALID_MODES = {'off', 'teleop', 'auton', 'driving', 'balance'}

    def __init__(self):
        super().__init__('vel_cmd_mux')

        # Declare parameters for velocity selection
        self.declare_parameter('mode_topic', '/robot_mode')
        self.declare_parameter('teleop_topic', '/cmd_vel_teleop')
        self.declare_parameter('auto_topic', '/cmd_vel_auto')
        self.declare_parameter('estop_topic', '/estop')
        self.declare_parameter('output_topic', '/cmd_vel_selected')

        # Declare parameters for freshness checks & debugging
        self.declare_parameter('teleop_timeout_s', 0.5)
        self.declare_parameter('auto_timeout_s', 0.5)
        self.declare_parameter('publish_rate_hz', 30.0)
        self.declare_parameter('debug', False)

        # Read/store all values from input parameters
        self.mode_topic = self.get_parameter('mode_topic').value
        self.teleop_topic = self.get_parameter('teleop_topic').value
        self.auto_topic = self.get_parameter('auto_topic').value
        self.estop_topic = self.get_parameter('estop_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.teleop_timeout = Duration(seconds=float(self.get_parameter('teleop_timeout_s').value))
        self.auto_timeout = Duration(seconds=float(self.get_parameter('auto_timeout_s').value))
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.debug = bool(self.get_parameter('debug').value)

        # Init state variabels
        self.robot_mode = None
        self.estop = False
        self.last_teleop = zero_twist()
        self.last_auto = zero_twist()
        self.last_teleop_time = None
        self.last_auto_time = None

        # QoS Setup
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        mode_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # Subscribe to input topics
        self.sub_teleop = self.create_subscription(Twist, self.teleop_topic, self._teleop_reader, qos)
        self.sub_auto = self.create_subscription(Twist, self.auto_topic, self._auto_reader, qos)
        self.sub_mode = self.create_subscription(String, self.mode_topic, self._mode_reader, qos)
        self.sub_estop = self.create_subscription(Bool, self.estop_topic, self._estop_reader, qos)

        # Publish selected velocity
        self.pub_out = self.create_publisher(Twist, self.output_topic, qos)

        # Create internal clock which executes _tick every period
        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self._tick)

        # Logging for debugging
        self.get_logger().info(
            f'CmdMux up. mode={self.mode_topic}, teleop={self.teleop_topic}, '
            f'auto={self.auto_topic}, out={self.output_topic}, estop={self.estop_topic}'
        )

    def _teleop_reader(self, msg: Twist):
        # Teleop msg arrived -> update twist, get time
        self.last_teleop = msg
        self.last_teleop_time = self.get_clock().now()

    def _auto_reader(self, msg: Twist):
        # Auto msg arrived -> update twist, get time
        self.last_auto = msg
        self.last_auto_time = self.get_clock().now()

    def _mode_reader(self, msg: String):
        # Get new mode message
        new_mode = msg.data.strip().lower()

        # Ignore invalid mode commands
        if new_mode not in self.VALID_MODES:
            self.get_logger().warn(f'Ignoring invalid robot_mode "{msg.data}"')
            return

        # Ignore same mode commands
        if self.robot_mode == new_mode:
            return
        
        # Update new mode
        self.robot_mode = new_mode
        self.get_logger().info(f'robot_mode -> {self.robot_mode}')

        # For safety, zero once immediately on mode switch
        self.pub_out.publish(zero_twist())

    def _estop_reader(self, msg: Bool):
        # Update estop state (on/off)
        new_estop = bool(msg.data)
        if new_estop and not self.estop:
            self.pub_out.publish(zero_twist())
        self.estop = new_estop

    def _fresh(self, last_time, timeout: Duration) -> bool:
        # Check how fresh a certain vel cmd is
        if last_time is None:
            return False
        return (self.get_clock().now() - last_time) <= timeout

    def _tick(self):
        # Zero output if estop
        if self.estop:
            if self.debug: self.get_logger().info('MUX: estop active -> zero')
            self.pub_out.publish(zero_twist())
            return
        
        # Zero output if off
        if self.robot_mode == 'off':
            if self.debug: self.get_logger().info('MUX: mode=off -> zero')
            self.pub_out.publish(zero_twist())
            return

        # Handle teleop mode
        if self.robot_mode == 'teleop':
            if self._fresh(self.last_teleop_time, self.teleop_timeout):
                if self.debug: self.get_logger().info(f'MUX: mode=teleop, fresh -> cmd {self.last_teleop}')
                self.pub_out.publish(self.last_teleop)
            else:
                if self.debug: self.get_logger().info('MUX: mode=teleop, stale -> zero')
                self.pub_out.publish(zero_twist())
            return

        # Hande auton mode
        if self.robot_mode == 'auton':
            if self._fresh(self.last_auto_time, self.auto_timeout):
                if self.debug: self.get_logger().info(f'MUX: mode=auton, fresh -> cmd {self.last_auto}')
                self.pub_out.publish(self.last_auto)
            else:
                if self.debug: self.get_logger().info('MUX: mode=auton, auto stale -> zero')
                self.pub_out.publish(zero_twist())
            return

        # Handle balance mode — auto if fresh, else teleop if fresh, else zero
        if self.robot_mode == 'balance':
            if self._fresh(self.last_auto_time, self.auto_timeout):
                if self.debug: self.get_logger().info(f'MUX: mode=balance, fresh auto -> cmd {self.last_auto}')
                self.pub_out.publish(self.last_auto)
            elif self._fresh(self.last_teleop_time, self.teleop_timeout):
                if self.debug: self.get_logger().info(f'MUX: mode=balance, fresh teleop -> cmd {self.last_teleop}')
                self.pub_out.publish(self.last_teleop)
            else:
                if self.debug: self.get_logger().info('MUX: mode=balance, both stale -> zero')
                self.pub_out.publish(zero_twist())
            return

        # Handle driving mode (autonomous nav with locked legs — same routing as auton)
        if self.robot_mode == 'driving':
            if self._fresh(self.last_auto_time, self.auto_timeout):
                if self.debug: self.get_logger().info(f'MUX: mode=driving, fresh -> cmd {self.last_auto}')
                self.pub_out.publish(self.last_auto)
            else:
                if self.debug: self.get_logger().info('MUX: mode=driving, auto stale -> zero')
                self.pub_out.publish(zero_twist())
            return

        # No valid mode -> zero output
        if self.debug: self.get_logger().info('MUX: no valid mode yet -> zero')
        self.pub_out.publish(zero_twist())

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