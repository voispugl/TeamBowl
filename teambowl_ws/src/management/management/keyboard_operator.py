#!/usr/bin/env python3
import sys
import select
import termios
import tty

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import String


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def zero_twist() -> Twist:
    msg = Twist()
    msg.linear.x = 0.0
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0
    return msg


class KeyboardOperatorNode(Node):
    """
    Single-terminal keyboard operator.

    Publishes:
      - /cmd_vel_teleop   (geometry_msgs/Twist)
      - /robot_mode_set   (std_msgs/String)

    Mode keys:
      1 -> off
      2 -> teleop
      3 -> auton

    Motion keys:
      w/s : forward/back
      a/d : left/right turn
      q/e : forward-left / forward-right
      z/c : backward-left / backward-right
      space or x : stop teleop command

    Speed tuning:
      [ / ] : decrease / increase linear speed
      ; / ' : decrease / increase angular speed

    Misc:
      h : print help
      Ctrl-C : quit
    """

    def __init__(self):
        super().__init__('keyboard_operator')

        self.declare_parameter('teleop_topic', '/cmd_vel_teleop')
        self.declare_parameter('mode_set_topic', '/robot_mode_set')
        self.declare_parameter('publish_rate_hz', 20.0)

        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('angular_speed', 0.40)

        self.declare_parameter('linear_speed_step', 0.05)
        self.declare_parameter('angular_speed_step', 0.10)

        self.declare_parameter('linear_speed_min', 0.0)
        self.declare_parameter('linear_speed_max', 1.0)
        self.declare_parameter('angular_speed_min', 0.0)
        self.declare_parameter('angular_speed_max', 2.5)

        self.teleop_topic = self.get_parameter('teleop_topic').value
        self.mode_set_topic = self.get_parameter('mode_set_topic').value
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)

        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.angular_speed = float(self.get_parameter('angular_speed').value)

        self.linear_speed_step = float(self.get_parameter('linear_speed_step').value)
        self.angular_speed_step = float(self.get_parameter('angular_speed_step').value)

        self.linear_speed_min = float(self.get_parameter('linear_speed_min').value)
        self.linear_speed_max = float(self.get_parameter('linear_speed_max').value)
        self.angular_speed_min = float(self.get_parameter('angular_speed_min').value)
        self.angular_speed_max = float(self.get_parameter('angular_speed_max').value)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub_cmd = self.create_publisher(Twist, self.teleop_topic, qos)
        self.pub_mode = self.create_publisher(String, self.mode_set_topic, qos)

        self.current_twist = zero_twist()

        # Put terminal into cbreak mode so we can read single keypresses.
        self.stdin_fd = sys.stdin.fileno()
        self.old_term_settings = termios.tcgetattr(self.stdin_fd)
        tty.setcbreak(self.stdin_fd)

        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f'KeyboardOperator up. teleop_topic={self.teleop_topic}, '
            f'mode_set_topic={self.mode_set_topic}'
        )
        self._print_help()
        self._log_speeds()

    def destroy_node(self):
        try:
            termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, self.old_term_settings)
        except Exception:
            pass
        super().destroy_node()

    def _print_help(self):
        help_text = """
Keyboard Operator Controls
--------------------------
Modes:
  1 -> OFF
  2 -> TELEOP
  3 -> AUTON

Teleop motion:
  w -> forward
  s -> backward
  a -> turn left
  d -> turn right
  q -> forward-left
  e -> forward-right
  z -> backward-left
  c -> backward-right
  space or x -> stop teleop command

Speed tuning:
  [ -> decrease linear speed
  ] -> increase linear speed
  ; -> decrease angular speed
  ' -> increase angular speed

Misc:
  h -> print this help
  Ctrl-C -> quit
"""
        for line in help_text.strip('\n').splitlines():
            self.get_logger().info(line)

    def _log_speeds(self):
        self.get_logger().info(
            f'linear_speed={self.linear_speed:.2f} m/s, '
            f'angular_speed={self.angular_speed:.2f} rad/s'
        )

    def _publish_mode(self, mode: str):
        msg = String()
        msg.data = mode
        self.pub_mode.publish(msg)
        self.get_logger().info(f'mode request -> {mode}')

    def _set_twist(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.current_twist = msg

    def _handle_key(self, key: str):
        # Modes
        if key == '1':
            self._publish_mode('off')
            self._set_twist(0.0, 0.0)
            return
        if key == '2':
            self._publish_mode('teleop')
            return
        if key == '3':
            self._publish_mode('auton')
            self._set_twist(0.0, 0.0)
            return

        # Motion
        if key == 'w':
            self._set_twist(+self.linear_speed, 0.0)
            return
        if key == 's':
            self._set_twist(-self.linear_speed, 0.0)
            return
        if key == 'a':
            self._set_twist(0.0, +self.angular_speed)
            return
        if key == 'd':
            self._set_twist(0.0, -self.angular_speed)
            return
        if key == 'q':
            self._set_twist(+self.linear_speed, +self.angular_speed)
            return
        if key == 'e':
            self._set_twist(+self.linear_speed, -self.angular_speed)
            return
        if key == 'z':
            self._set_twist(-self.linear_speed, +self.angular_speed)
            return
        if key == 'c':
            self._set_twist(-self.linear_speed, -self.angular_speed)
            return
        if key == ' ' or key == 'x':
            self._set_twist(0.0, 0.0)
            return

        # Speed tuning
        if key == '[':
            self.linear_speed = clamp(
                self.linear_speed - self.linear_speed_step,
                self.linear_speed_min,
                self.linear_speed_max,
            )
            self._log_speeds()
            return
        if key == ']':
            self.linear_speed = clamp(
                self.linear_speed + self.linear_speed_step,
                self.linear_speed_min,
                self.linear_speed_max,
            )
            self._log_speeds()
            return
        if key == ';':
            self.angular_speed = clamp(
                self.angular_speed - self.angular_speed_step,
                self.angular_speed_min,
                self.angular_speed_max,
            )
            self._log_speeds()
            return
        if key == "'":
            self.angular_speed = clamp(
                self.angular_speed + self.angular_speed_step,
                self.angular_speed_min,
                self.angular_speed_max,
            )
            self._log_speeds()
            return

        if key == 'h':
            self._print_help()
            return

    def _tick(self):
        # Non-blocking single-char read
        if select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
            self._handle_key(key)

        # Keep publishing current teleop command so mux freshness stays alive.
        self.pub_cmd.publish(self.current_twist)


def main():
    rclpy.init()
    node = KeyboardOperatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()