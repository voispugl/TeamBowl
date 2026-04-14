#!/usr/bin/env python3
"""
Balance Controller Node
========================
Cascaded PID balance controller for two-wheel self-balancing.

In "balance" mode:
  - Outer PI loop (50 Hz): velocity error → target pitch angle (theta_ref)
  - Inner PD loop (50 Hz): pitch error → symmetric wheel velocity command
  - Angular (turn) command passes through unchanged from /cmd_vel_safe

In all other modes:
  - Passthrough: /cmd_vel_safe → /cmd_vel unchanged

Topics
------
Subscribes:
  /odometry/filtered  nav_msgs/Odometry   — filtered body velocity (from EKF)
  /imu/data           sensor_msgs/Imu     — pitch angle, pitch rate
  /cmd_vel_safe       geometry_msgs/Twist — desired v_cmd, omega_cmd
  /robot_mode         std_msgs/String     — mode switching
  /estop              std_msgs/Bool       — emergency stop
  /balance_gains      std_msgs/String     — JSON gain updates from Foxglove

Publishes:
  /cmd_vel            geometry_msgs/Twist — wheel command to cmd_vel_to_vesc
  /estop              std_msgs/Bool       — asserted if robot falls over
  /balance_gains_echo std_msgs/String     — JSON echo of current gains

Foxglove live gain tuning
--------------------------
Publish to /balance_gains with a JSON string containing any subset of params:
  {"kp_pitch": 70.0, "kd_pitch": 10.0, "kp_vel": 0.25}
Only keys present are updated. All others unchanged.
Read back current gains from /balance_gains_echo (published every 2 s).

Foxglove setup:
  1. Publish panel → /balance_gains, type: std_msgs/String
     Message template: {"data": "{\"kp_pitch\": 70.0}"}
  2. Raw Messages panel → /balance_gains_echo to read current values
"""

import json
import math
from collections import deque
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, String


def _quat_to_pitch(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract pitch (rotation about Y) from quaternion — ZYX convention.

    Positive = leaning forward. Matches real IMU mounting convention
    where angular_velocity.y is the pitch rate.
    """
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    return math.asin(sinp)


class BalanceController(Node):
    """
    Cascaded PID balance controller.

    Outer PI:  velocity error → theta_ref  (lean angle setpoint)
    Inner PD:  (theta - theta_ref) + kd * theta_dot → wheel velocity cmd

    Output is a symmetric wheel velocity (linear.x on /cmd_vel).
    The VESC driver converts this to motor torque internally.
    """

    BALANCE_MODE = 'balance'

    def __init__(self):
        super().__init__('balance_controller')

        # ------------------------------------------------------------------ #
        # Parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter('cmd_vel_safe_topic', '/cmd_vel_safe')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('mode_topic', '/robot_mode')
        self.declare_parameter('estop_topic', '/estop')
        self.declare_parameter('cmd_vel_out_topic', '/cmd_vel')

        # Inner PD gains
        self.declare_parameter('kp_pitch', 60.0)   # [cmd_vel / rad]
        self.declare_parameter('kd_pitch', 8.0)
        self.declare_parameter('ki_pitch', 0.0)    # [cmd_vel / (rad·s)] sliding-window I
        self.declare_parameter('kp_yaw', 5.0)
        self.declare_parameter('kd_yaw', 0.5)    # [cmd_vel·s / rad]

        # Outer PI gains
        self.declare_parameter('kp_vel', 0.30)     # [rad / (m/s)]
        self.declare_parameter('ki_vel', 0.05)     # [rad / (m·s)]
        self.declare_parameter('kff_pitch', 0.0)   # feed-forward lean  [rad / (m/s)]

        # Safety limits
        self.declare_parameter('theta_max_cmd', 0.25)        # [rad]
        self.declare_parameter('theta_max_fallover', 0.50)   # [rad]
        self.declare_parameter('theta_eq_offset', 0.00)      # [rad] lean trim

        # Misc
        self.declare_parameter('l_com', 0.45)
        self.declare_parameter('control_rate_hz', 50.0)
        self.declare_parameter('inner_rate_hz', 150.0)
        self.declare_parameter('odom_timeout_s', 0.30)

        # ------------------------------------------------------------------ #
        # Read topics
        # ------------------------------------------------------------------ #
        safe_topic  = self.get_parameter('cmd_vel_safe_topic').value
        odom_topic  = self.get_parameter('odom_topic').value
        imu_topic   = self.get_parameter('imu_topic').value
        mode_topic  = self.get_parameter('mode_topic').value
        estop_topic = self.get_parameter('estop_topic').value
        out_topic   = self.get_parameter('cmd_vel_out_topic').value
        rate_hz       = float(self.get_parameter('control_rate_hz').value)
        inner_rate_hz = float(self.get_parameter('inner_rate_hz').value)

        self._odom_timeout = Duration(
            seconds=float(self.get_parameter('odom_timeout_s').value)
        )

        # ------------------------------------------------------------------ #
        # State
        # ------------------------------------------------------------------ #
        self._mode   = 'off'
        self._estop  = False

        self._theta      = 0.0
        self._theta_dot  = 0.0
        self._yaw_dot    = 0.0
        self._v_actual   = 0.0
        self._last_odom_time = None
        self._last_imu_time  = None

        self._v_cmd    = 0.0
        self._omega_cmd = 0.0
        self._last_cmd_time = None

        # Outer PI integrator + shared theta_ref
        self._v_integral = 0.0
        self._theta_ref  = 0.0

        # Inner pitch I: 2-second sliding window
        self._pitch_i_window = deque()  # (ros_time_sec, contribution) pairs

        # ------------------------------------------------------------------ #
        # QoS
        # ------------------------------------------------------------------ #
        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        transient = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ------------------------------------------------------------------ #
        # Publishers
        # ------------------------------------------------------------------ #
        self._cmd_pub        = self.create_publisher(Twist,  out_topic,            best_effort)
        self._estop_pub      = self.create_publisher(Bool,   estop_topic,          reliable)
        self._gains_echo_pub = self.create_publisher(String, '/balance_gains_echo', reliable)

        # ------------------------------------------------------------------ #
        # Subscribers
        # ------------------------------------------------------------------ #
        self.create_subscription(Imu,      imu_topic,   self._on_imu,         best_effort)
        self.create_subscription(Odometry, odom_topic,  self._on_odom,        best_effort)
        self.create_subscription(Twist,    safe_topic,  self._on_cmd_vel_safe, best_effort)
        self.create_subscription(String,   mode_topic,  self._on_mode,        transient)
        self.create_subscription(Bool,     estop_topic, self._on_estop,       reliable)
        self.create_subscription(String,   '/balance_gains', self._on_gains,  reliable)

        # ------------------------------------------------------------------ #
        # Timers
        # ------------------------------------------------------------------ #
        self._dt_outer = 1.0 / rate_hz
        self._dt_inner = 1.0 / inner_rate_hz
        self.create_timer(self._dt_outer, self._outer_tick)
        self.create_timer(self._dt_inner, self._inner_tick)
        self.create_timer(2.0,            self._publish_gains_echo)

        self.get_logger().info(
            f'BalanceController (PID) up. Passthrough until mode="{self.BALANCE_MODE}". '
            f'safe_in={safe_topic} → cmd_out={out_topic}. '
            f'Outer PI {rate_hz:.0f} Hz, inner PID {inner_rate_hz:.0f} Hz. '
            f'Foxglove: pub JSON to /balance_gains, read /balance_gains_echo'
        )

    # ---------------------------------------------------------------------- #
    # Subscribers
    # ---------------------------------------------------------------------- #

    def _on_imu(self, msg: Imu):
        q = msg.orientation
        self._theta     = _quat_to_pitch(q.x, q.y, q.z, q.w)
        self._theta_dot = msg.angular_velocity.y   # pitch rate, Y-axis convention
        self._yaw_dot   = msg.angular_velocity.z   # yaw rate, Z-axis
        self._last_imu_time = self.get_clock().now()

    def _on_odom(self, msg: Odometry):
        self._v_actual = msg.twist.twist.linear.x
        self._last_odom_time = self.get_clock().now()

    def _on_cmd_vel_safe(self, msg: Twist):
        self._v_cmd     = msg.linear.x
        self._omega_cmd = msg.angular.z
        self._last_cmd_time = self.get_clock().now()

    def _on_mode(self, msg: String):
        new_mode = msg.data.strip().lower()
        if new_mode == self._mode:
            return
        prev = self._mode
        self._mode = new_mode
        self.get_logger().info(f'Mode {prev} → {new_mode}')
        if new_mode != self.BALANCE_MODE:
            self._v_integral = 0.0
            self._pitch_i_window.clear()

    def _on_estop(self, msg: Bool):
        self._estop = msg.data
        if self._estop:
            self._v_integral = 0.0
            self._pitch_i_window.clear()

    def _on_gains(self, msg: String):
        """
        Foxglove / live gain update.
        Accepts a JSON dict with any subset of tunable params, e.g.:
          {"kp_pitch": 70.0, "kd_pitch": 10.0, "kp_vel": 0.25}
        """
        FLOAT_PARAMS = {
            'kp_pitch', 'kd_pitch', 'ki_pitch',
            'kp_yaw', 'kd_yaw',
            'kp_vel', 'ki_vel', 'kff_pitch',
            'theta_max_cmd', 'theta_max_fallover', 'theta_eq_offset', 'l_com',
        }

        try:
            updates = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'/balance_gains: invalid JSON — {e}')
            return

        if not isinstance(updates, dict):
            self.get_logger().error('/balance_gains: expected a JSON object {}')
            return

        applied = []
        for key, value in updates.items():
            if key in FLOAT_PARAMS:
                try:
                    self.set_parameters([
                        rclpy.parameter.Parameter(
                            key, rclpy.Parameter.Type.DOUBLE, float(value)
                        )
                    ])
                    applied.append(f'{key}={float(value):.4f}')
                except Exception as e:
                    self.get_logger().warn(f'/balance_gains: failed to set {key}: {e}')
            else:
                self.get_logger().warn(f'/balance_gains: unknown param "{key}" ignored')

        if applied:
            self.get_logger().info(f'Gains updated: {", ".join(applied)}')
            self._v_integral = 0.0   # clear integrator to avoid windup transient

    # ---------------------------------------------------------------------- #
    # Gains echo (Foxglove readback)
    # ---------------------------------------------------------------------- #

    def _publish_gains_echo(self):
        gains = {
            'kp_pitch':           self.get_parameter('kp_pitch').value,
            'kd_pitch':           self.get_parameter('kd_pitch').value,
            'ki_pitch':           self.get_parameter('ki_pitch').value,
            'kp_yaw':             self.get_parameter('kp_yaw').value,
            'kd_yaw':             self.get_parameter('kd_yaw').value,
            'kp_vel':             self.get_parameter('kp_vel').value,
            'ki_vel':             self.get_parameter('ki_vel').value,
            'kff_pitch':          self.get_parameter('kff_pitch').value,
            'theta_max_cmd':      self.get_parameter('theta_max_cmd').value,
            'theta_eq_offset':    self.get_parameter('theta_eq_offset').value,
            # Runtime state
            '_mode':      self._mode,
            '_estop':     self._estop,
            '_theta_deg': round(math.degrees(self._theta), 2),
            '_v_actual':  round(self._v_actual, 3),
        }
        msg = String()
        msg.data = json.dumps(gains, indent=2)
        self._gains_echo_pub.publish(msg)

    # ---------------------------------------------------------------------- #
    # Control loops
    # ---------------------------------------------------------------------- #

    def _outer_tick(self):
        """50 Hz — velocity PI → updates theta_ref. Also checks fallover."""
        if self._estop or self._mode != self.BALANCE_MODE:
            return

        now = self.get_clock().now()

        theta_fallover = float(self.get_parameter('theta_max_fallover').value)
        theta_eq       = float(self.get_parameter('theta_eq_offset').value)
        theta          = self._theta - theta_eq

        # Fallover detection
        if abs(theta) > theta_fallover:
            self.get_logger().error(
                f'FALLOVER: |theta|={abs(theta):.3f} rad > {theta_fallover:.3f} rad. '
                f'Triggering estop.'
            )
            self._publish_cmd(0.0, 0.0)
            self._trigger_estop()
            return

        # Use EKF velocity if fresh, else fall back to 0
        if self._last_odom_time is None or (now - self._last_odom_time) > self._odom_timeout:
            v_actual = 0.0
        else:
            v_actual = self._v_actual

        kp_vel    = float(self.get_parameter('kp_vel').value)
        ki_vel    = float(self.get_parameter('ki_vel').value)
        kff_pitch = float(self.get_parameter('kff_pitch').value)
        theta_max = float(self.get_parameter('theta_max_cmd').value)

        v_error = self._v_cmd - v_actual
        self._v_integral += v_error * self._dt_outer

        max_integral = theta_max / max(ki_vel, 1e-6)
        self._v_integral = max(-max_integral, min(max_integral, self._v_integral))

        theta_ref = (kp_vel * v_error
                   + ki_vel * self._v_integral
                   + kff_pitch * self._v_cmd)  # feed-forward lean
        self._theta_ref = max(-theta_max, min(theta_max, theta_ref))

    def _inner_tick(self):
        """150 Hz — inner PID pitch → cmd_vel. Passthrough + estop handled here."""
        # Safety: always zero on estop
        if self._estop:
            self._publish_cmd(0.0, 0.0)
            return

        # Passthrough when not in balance mode
        if self._mode != self.BALANCE_MODE:
            if self._last_cmd_time is not None:
                self._publish_cmd(self._v_cmd, self._omega_cmd)
            else:
                self._publish_cmd(0.0, 0.0)
            return

        now = self.get_clock().now()

        theta_eq  = float(self.get_parameter('theta_eq_offset').value)
        theta     = self._theta - theta_eq
        pitch_err = theta - self._theta_ref

        kp_pitch = float(self.get_parameter('kp_pitch').value)
        kd_pitch = float(self.get_parameter('kd_pitch').value)
        ki_pitch = float(self.get_parameter('ki_pitch').value)

        # Rolling 2-second pitch integral
        now_sec = now.nanoseconds * 1e-9
        self._pitch_i_window.append((now_sec, pitch_err * self._dt_inner))
        while self._pitch_i_window and now_sec - self._pitch_i_window[0][0] > 2.0:
            self._pitch_i_window.popleft()
        pitch_integral = sum(v for _, v in self._pitch_i_window)

        u_balance = -(kp_pitch * pitch_err
                    + ki_pitch * pitch_integral
                    + kd_pitch * self._theta_dot)

        kp_yaw  = float(self.get_parameter('kp_yaw').value)
        kd_yaw  = float(self.get_parameter('kd_yaw').value)
        yaw_err = self._omega_cmd - self._yaw_dot
        yaw_out = kp_yaw * yaw_err - kd_yaw * self._yaw_dot

        self._publish_cmd(u_balance, yaw_out)

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _publish_cmd(self, v: float, omega: float):
        msg = Twist()
        msg.linear.x  = float(v)
        msg.angular.z = float(omega)
        self._cmd_pub.publish(msg)

    def _trigger_estop(self):
        msg = Bool()
        msg.data = True
        self._estop_pub.publish(msg)
        self._estop = True


def main(args=None):
    rclpy.init(args=args)
    node = BalanceController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
