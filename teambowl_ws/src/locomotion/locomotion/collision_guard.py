#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

# Helpers
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def zero_twist() -> Twist:
    msg = Twist() # inits lin.xyz and ang.xyz as zero
    return msg

# ROS Node
class CollisionGuardNode(Node):
    """
      - #TODO implement reactive collision avoidance for safety here
      - clamps linear.x and angular.z
      - zeros output if /estop is true
      - forwards to output topic (/cmd_vel by default)

    Subscribes to:
      - /cmd_vel_selected
      - /estop

    Publishes: 
      - /cmd_vel (Output twist command)
    """

    def __init__(self):
        super().__init__('collision_guard')

        # Topics for velocity clamping
        self.declare_parameter('input_topic', '/cmd_vel_selected')
        self.declare_parameter('max_linear_x', 0.5)     # m/s
        self.declare_parameter('max_angular_z', 1.0)    # rad/s
        self.declare_parameter('estop_topic', '/estop')
        self.declare_parameter('output_topic', '/cmd_vel')

        # Read all topics
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.estop_topic = self.get_parameter('estop_topic').value
        self.max_linear_x = float(self.get_parameter('max_linear_x').value)
        self.max_angular_z = float(self.get_parameter('max_angular_z').value)
        
        # State variables
        self.estop = False

        # QoS Setup
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Subscribe to topics
        self.sub_cmd = self.create_subscription(Twist, self.input_topic, self._cmd_reader, qos)
        self.sub_estop = self.create_subscription(Bool, self.estop_topic, self._estop_reader, qos)

        # Publish clamped velocity command
        self.pub = self.create_publisher(Twist, self.output_topic, qos)

        # Logging for debugging
        self.get_logger().info(
            f'CollisionGuard up. in={self.input_topic}, out={self.output_topic}, estop={self.estop_topic}, '
            f'max_v={self.max_linear_x}, max_w={self.max_angular_z}'
        )

    def _estop_reader(self, msg: Bool):
        # Update estop state (on/off)
        new_estop = bool(msg.data)
        if new_estop and not self.estop:
            self.pub.publish(zero_twist())
        self.estop = new_estop

    def _cmd_reader(self, msg: Twist):
        # Vel cmd arrived

        # Estop triggered -> No vel cmd
        if self.estop:
            self.pub.publish(zero_twist())
            return
        
        #TODO Imminent collision, stop the robot

        # Init output twist and clamp
        out = Twist()
        out.linear.x = clamp(msg.linear.x, -self.max_linear_x, self.max_linear_x)
        out.angular.z = clamp(msg.angular.z, -self.max_angular_z, self.max_angular_z)

        # Pass through other fields untouched (usually unused)
        out.linear.y = msg.linear.y
        out.linear.z = msg.linear.z
        out.angular.x = msg.angular.x
        out.angular.y = msg.angular.y

        # Publish twist
        self.pub.publish(out)

def main():
    rclpy.init()
    node = CollisionGuardNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()