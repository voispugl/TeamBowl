#!/usr/bin/env python3
import os
import sys
import select
import termios
import time
import tty

import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String


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
      - /estop            (std_msgs/Bool)

    Mode keys:
      1 -> off
      2 -> teleop
      3 -> auton
      4 -> trick
      0 -> estop on
      9 -> estop off

    Motion keys:
      w/s : forward/back
      a/d : left/right turn
      q/e : forward-left / forward-right
      z/c : backward-left / backward-right
      space or x : stop teleop command

    Speed tuning:
      [ / ] : decrease / increase linear speed
      ; / ' : decrease / increase angular speed

    Trick mode (key 4):
      j : move all leg joints to trick offsets + drive forward for 2 s
      n : return all leg joints to base driving positions

  Misc:
      h : print help
      Ctrl-C : quit
    """

    def __init__(self):
        super().__init__('keyboard_operator')

        self.declare_parameter('teleop_topic', '/cmd_vel_teleop')
        self.declare_parameter('mode_set_topic', '/robot_mode_set')
        self.declare_parameter('estop_topic', '/estop')
        self.declare_parameter('publish_rate_hz', 20.0)
        self.declare_parameter('teleop_hold_timeout_s', 0.20)

        self.declare_parameter('linear_speed', 0.20)
        self.declare_parameter('angular_speed', 0.80)

        self.declare_parameter('linear_speed_step', 0.05)
        self.declare_parameter('angular_speed_step', 0.10)

        self.declare_parameter('linear_speed_min', 0.0)
        self.declare_parameter('linear_speed_max', 0.20)
        self.declare_parameter('angular_speed_min', 0.0)
        self.declare_parameter('angular_speed_max', 0.80)

        self.teleop_topic = self.get_parameter('teleop_topic').value
        self.mode_set_topic = self.get_parameter('mode_set_topic').value
        self.estop_topic = self.get_parameter('estop_topic').value
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.teleop_hold_timeout_s = float(
            self.get_parameter('teleop_hold_timeout_s').value
        )

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
        self.pub_estop = self.create_publisher(Bool, self.estop_topic, qos)
        self.pub_trick_offsets = self.create_publisher(JointState, '/trick_leg_offsets', qos)

        self.current_twist = zero_twist()
        self.last_motion_command_time = 0.0

        # Trick mode state
        self._trick_drive_until: float = 0.0   # monotonic deadline for trick forward drive
        share_dir = get_package_share_directory('locomotion')
        default_trick_path = os.path.join(share_dir, 'trick_leg_offsets.yaml')
        self.declare_parameter('trick_offsets_path', default_trick_path)
        trick_path = self.get_parameter('trick_offsets_path').value

        self._trick_targets: dict = {}   # joint_name -> offset value from YAML
        self._trick_pose_active: bool = False  # True = offsets applied
        self._requested_mode: str = 'off'      # last mode key pressed locally

        try:
            with open(trick_path, 'r') as f:
                data = yaml.safe_load(f)
            self._trick_targets = {k: float(v) for k, v in data.get('joints', {}).items()}
            self.get_logger().info(f'Trick offsets loaded from {trick_path}')
        except Exception as e:
            self.get_logger().warn(f'Could not load trick offsets: {e}')

        # Put terminal into cbreak mode so we can read single keypresses.
        self.stdin_fd = sys.stdin.fileno()
        self.old_term_settings = termios.tcgetattr(self.stdin_fd)
        tty.setcbreak(self.stdin_fd)

        period = 1.0 / max(self.publish_rate_hz, 1.0)
        self.timer = self.create_timer(period, self._tick)

        self.get_logger().info(
            f'KeyboardOperator up. teleop_topic={self.teleop_topic}, '
            f'mode_set_topic={self.mode_set_topic}, estop_topic={self.estop_topic}'
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
  4 -> TRICK
  0 -> ESTOP ON
  9 -> ESTOP OFF

Teleop motion:
  Hold the key to keep moving
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

Trick mode (press 4 first):
  j     -> move all leg joints to trick offsets + drive forward for 2 s
  n     -> return all leg joints to base driving positions (stay in trick mode)
  space -> ESTOP ON (disables all joints and wheels)

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

    def _publish_estop(self, asserted: bool):
        msg = Bool()
        msg.data = asserted
        self.pub_estop.publish(msg)
        state = 'ON' if asserted else 'OFF'
        self.get_logger().warn(f'estop -> {state}')

    def _set_twist(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.current_twist = msg

    def _set_motion_twist(self, linear_x: float, angular_z: float):
        self._set_twist(linear_x, angular_z)
        self.last_motion_command_time = time.monotonic()

    def _handle_key(self, key: str):
        # Modes
        if key == '1':
            self._requested_mode = 'off'
            self._trick_pose_active = False
            self._publish_mode('off')
            self._set_twist(0.0, 0.0)
            return
        if key == '2':
            self._requested_mode = 'teleop'
            self._trick_pose_active = False
            self._publish_mode('teleop')
            return
        if key == '3':
            self._requested_mode = 'auton'
            self._trick_pose_active = False
            self._publish_mode('auton')
            self._set_twist(0.0, 0.0)
            return
        if key == '4':
            self._requested_mode = 'trick'
            self._publish_mode('trick')
            return
        if key == '0':
            self._publish_estop(True)
            self._set_twist(0.0, 0.0)
            return
        if key == '9':
            self._publish_estop(False)
            return

        # Motion
        if key == 'w':
            self._set_motion_twist(+self.linear_speed, 0.0)
            return
        if key == 's':
            self._set_motion_twist(-self.linear_speed, 0.0)
            return
        if key == 'a':
            self._set_motion_twist(0.0, +self.angular_speed)
            return
        if key == 'd':
            self._set_motion_twist(0.0, -self.angular_speed)
            return
        if key == 'q':
            self._set_motion_twist(+self.linear_speed, +self.angular_speed)
            return
        if key == 'e':
            self._set_motion_twist(+self.linear_speed, -self.angular_speed)
            return
        if key == 'z':
            self._set_motion_twist(-self.linear_speed, +self.angular_speed)
            return
        if key == 'c':
            self._set_motion_twist(-self.linear_speed, -self.angular_speed)
            return
        if key == ' ' or key == 'x':
            if self._requested_mode == 'trick':
                self._publish_estop(True)
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

        # Trick mode joint controls (only active when trick mode was requested)
        if self._requested_mode == 'trick':
            if key == 'j':
                self._trick_pose_active = True
                self._trick_drive_until = time.monotonic() + 2.0
                self.get_logger().info('trick pose -> ON (driving forward for 2 s)')
                return
            if key == 'n':
                self._trick_pose_active = False
                self.get_logger().info('trick pose -> OFF (base)')
                return

        if key == 'h':
            self._print_help()
            return

    def _publish_trick_offsets(self):
        if not self._trick_targets:
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self._trick_targets.keys())
        msg.position = [
            self._trick_targets[j] if self._trick_pose_active else 0.0
            for j in msg.name
        ]
        msg.velocity = [0.0] * len(msg.name)
        msg.effort = [0.0] * len(msg.name)
        self.pub_trick_offsets.publish(msg)

    def _tick(self):
        # Non-blocking single-char read
        if select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
            self._handle_key(key)

        now = time.monotonic()
        if (
            (self.current_twist.linear.x != 0.0 or self.current_twist.angular.z != 0.0)
            and (now - self.last_motion_command_time) > self.teleop_hold_timeout_s
        ):
            self._set_twist(0.0, 0.0)

        # Keep publishing current teleop command so mux freshness stays alive.
        cmd = self.current_twist
        if self._trick_pose_active and time.monotonic() < self._trick_drive_until:
            trick_cmd = Twist()
            trick_cmd.linear.x = self.linear_speed
            cmd = trick_cmd
        self.pub_cmd.publish(cmd)
        # Always publish trick offsets; driving_leg_controller only applies them in trick mode.
        self._publish_trick_offsets()


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
