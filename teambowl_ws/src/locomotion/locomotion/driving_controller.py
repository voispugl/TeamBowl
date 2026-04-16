#!/usr/bin/env python3
"""
Driving Controller Node
=======================
Velocity-tracking PID with pitch correction for autonomous driving with
legs locked in place and rear legs dragging.

The robot in this configuration is mechanically stable (not an inverted
pendulum), but hard deceleration pitches the nose down due to inertia.
This controller tracks commanded velocity with a fast PID and adds a pitch
correction term that reduces effective braking when the nose starts to dip.

Control architecture
--------------------
Outer loop (50 Hz) — Velocity PID:
  v_err  = v_cmd - v_actual
  u_vel  = kp_vel * v_err + ki_vel * integral(v_err) + kd_vel * dv_err/dt
  Written to shared _u_vel, read by inner loop.

Inner loop (100 Hz) — Pitch PID + output:
  theta_ref = theta_eq_offset + kff_decel * (v_cmd - v_cmd_prev) / dt
  pitch_err = theta - theta_ref
  Derivative: 3-sample weighted FIR (weights [0.5, 0.25, 0.25]) to reduce
              IMU noise without significant lag (~15 ms at 100 Hz).
  u_pitch   = kp_pitch * pitch_err + kd_pitch * d_pitch + ki_pitch * integral
  v_out     = clamp(u_vel + u_pitch, -v_max, +v_max)

Sign convention:
  theta > 0  — nose down (forward lean), per _quat_to_pitch ZYX convention
  theta_dot  — angular_velocity.y from IMU
  linear.x > 0 — forward wheel velocity

Pitch limit:
  |theta| > theta_max_pitch → WARN log only. No estop. The robot is
  already contacting the ground at that angle; zeroing velocity mid-contact
  risks frame damage. Let the pitch loop continue and let navigation respond.

Topics
------
Subscribes:
  /cmd_vel_safe        geometry_msgs/Twist  — desired v_cmd, omega_cmd
  /imu/data            sensor_msgs/Imu     — pitch angle, pitch rate
  /odometry/filtered   nav_msgs/Odometry   — filtered body velocity (EKF)
  /robot_mode          std_msgs/String     — mode switching
  /estop               std_msgs/Bool       — emergency stop
  /driving_gains       std_msgs/String     — JSON gain updates (Foxglove)

Publishes:
  /cmd_vel             geometry_msgs/Twist — wheel command to cmd_vel_to_vesc
  /driving_gains_echo  std_msgs/String     — JSON echo of current gains (2 Hz)

Active mode: "driving"  (passthrough in all other modes)

Foxglove live gain tuning
--------------------------
Publish JSON to /driving_gains with any subset of tunable params:
  {"kp_pitch": 8.0, "kd_pitch": 1.5, "kp_vel": 3.0}
Read back state + current gains from /driving_gains_echo (2 Hz).
"""

import json
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
)
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, String


def _quat_to_pitch(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract pitch (rotation about Y) from quaternion — ZYX convention.

    Positive = nose down (leaning forward). Matches Xsens IMU mounting
    convention where angular_velocity.y is the pitch rate.
    """
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    return math.asin(sinp)


class DrivingController(Node):
    """
    Velocity-tracking PID with pitch correction for locked-leg driving.

    Outer PI  (50 Hz): velocity error → u_vel (shared with inner loop)
    Inner PID (100 Hz): pitch correction → u_pitch; v_out = u_vel + u_pitch
    """

    DRIVING_MODE = 'driving'

    def __init__(self):
        super().__init__('driving_controller')

        # ------------------------------------------------------------------ #
        # Parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter('cmd_vel_safe_topic', '/cmd_vel_safe')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('mode_topic', '/robot_mode')
        self.declare_parameter('estop_topic', '/estop')
        self.declare_parameter('cmd_vel_out_topic', '/cmd_vel')

        # Outer velocity PID gains
        self.declare_parameter('kp_vel', 2.0)
        self.declare_parameter('ki_vel', 0.5)
        self.declare_parameter('kd_vel', 0.05)

        # Inner pitch PID gains
        self.declare_parameter('kp_pitch', 5.0)
        self.declare_parameter('kd_pitch', 1.0)
        self.declare_parameter('ki_pitch', 0.0)

        # Decel feedforward: anticipate pitch from commanded velocity change
        self.declare_parameter('kff_decel', 0.0)

        # Static pitch trim (positive = nose-down offset to trim)
        self.declare_parameter('theta_eq_offset', 0.0)

        # Pitch limit: WARN only, no estop (robot is already grounded at this angle)
        self.declare_parameter('theta_max_pitch', 0.35)

        # Output velocity clamp [m/s]
        self.declare_parameter('v_max', 3.0)

        # Control rates
        self.declare_parameter('outer_rate_hz', 50.0)
        self.declare_parameter('inner_rate_hz', 100.0)

        # Odometry staleness timeout
        self.declare_parameter('odom_timeout_s', 0.30)

        # ------------------------------------------------------------------ #
        # Read topics + rates
        # ------------------------------------------------------------------ #
        safe_topic  = self.get_parameter('cmd_vel_safe_topic').value
        odom_topic  = self.get_parameter('odom_topic').value
        imu_topic   = self.get_parameter('imu_topic').value
        mode_topic  = self.get_parameter('mode_topic').value
        estop_topic = self.get_parameter('estop_topic').value
        out_topic   = self.get_parameter('cmd_vel_out_topic').value

        outer_hz = float(self.get_parameter('outer_rate_hz').value)
        inner_hz = float(self.get_parameter('inner_rate_hz').value)

        self._dt_outer = 1.0 / outer_hz
        self._dt_inner = 1.0 / inner_hz

        self._odom_timeout = Duration(
            seconds=float(self.get_parameter('odom_timeout_s').value)
        )

        # ------------------------------------------------------------------ #
        # State — sensors
        # ------------------------------------------------------------------ #
        self._mode  = 'off'
        self._estop = False

        self._theta     = 0.0   # pitch angle [rad]
        self._theta_dot = 0.0   # pitch rate from gyro [rad/s]
        self._v_actual  = 0.0   # EKF-filtered forward velocity [m/s]

        self._last_odom_time = None
        self._last_imu_time  = None

        # ------------------------------------------------------------------ #
        # State — command inputs
        # ------------------------------------------------------------------ #
        self._v_cmd      = 0.0
        self._v_cmd_prev = 0.0   # for decel feedforward (inner tick)
        self._omega_cmd  = 0.0
        self._last_cmd_time = None

        # ------------------------------------------------------------------ #
        # State — outer velocity PID
        # ------------------------------------------------------------------ #
        self._v_err_prev = 0.0
        self._v_i_window: deque = deque()   # (ros_time_sec, contribution)
        self._u_vel = 0.0                   # shared: outer writes, inner reads

        # ------------------------------------------------------------------ #
        # State — inner pitch PID
        # ------------------------------------------------------------------ #
        # 3-sample ring buffer for weighted derivative [k, k-1, k-2]
        self._pitch_err_buf = [0.0, 0.0, 0.0]
        self._pitch_i_window: deque = deque()  # 2-s sliding window integral

        # ------------------------------------------------------------------ #
        # QoS profiles
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
        self._cmd_pub        = self.create_publisher(Twist,  out_topic,              best_effort)
        self._gains_echo_pub = self.create_publisher(String, '/driving_gains_echo',  reliable)
        # NOTE: This node intentionally does NOT publish /estop.
        # If pitch exceeds theta_max_pitch the robot is already grounded;
        # zeroing velocity mid-contact risks frame damage.

        # ------------------------------------------------------------------ #
        # Subscribers
        # ------------------------------------------------------------------ #
        self.create_subscription(Imu,      imu_topic,   self._on_imu,          best_effort)
        self.create_subscription(Odometry, odom_topic,  self._on_odom,         best_effort)
        self.create_subscription(Twist,    safe_topic,  self._on_cmd_vel_safe,  best_effort)
        self.create_subscription(String,   mode_topic,  self._on_mode,         transient)
        self.create_subscription(Bool,     estop_topic, self._on_estop,        reliable)
        self.create_subscription(String,   '/driving_gains', self._on_gains,   reliable)

        # ------------------------------------------------------------------ #
        # Timers
        # ------------------------------------------------------------------ #
        self.create_timer(self._dt_outer, self._outer_tick)
        self.create_timer(self._dt_inner, self._inner_tick)
        self.create_timer(2.0,            self._publish_gains_echo)

        self.get_logger().info(
            f'DrivingController up. Passthrough until mode="{self.DRIVING_MODE}". '
            f'Outer velocity PID {outer_hz:.0f} Hz, inner pitch PID {inner_hz:.0f} Hz. '
            f'safe_in={safe_topic} → cmd_out={out_topic}. '
            f'Foxglove: pub JSON to /driving_gains, read /driving_gains_echo'
        )

    # ---------------------------------------------------------------------- #
    # Subscribers
    # ---------------------------------------------------------------------- #

    def _on_imu(self, msg: Imu):
        q = msg.orientation
        self._theta     = _quat_to_pitch(q.x, q.y, q.z, q.w)
        self._theta_dot = msg.angular_velocity.y   # pitch rate, Y-axis
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
        if new_mode != self.DRIVING_MODE:
            self._reset_integrators()

    def _on_estop(self, msg: Bool):
        self._estop = msg.data
        if self._estop:
            self._reset_integrators()

    def _on_gains(self, msg: String):
        """
        Live gain update from Foxglove / CLI.
        Accepts JSON dict with any subset of tunable params, e.g.:
          {"kp_pitch": 8.0, "kd_pitch": 1.5, "kp_vel": 3.0}
        """
        FLOAT_PARAMS = {
            'kp_vel', 'ki_vel', 'kd_vel',
            'kp_pitch', 'kd_pitch', 'ki_pitch',
            'kff_decel', 'theta_eq_offset', 'theta_max_pitch', 'v_max',
        }

        try:
            updates = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f'/driving_gains: invalid JSON — {e}')
            return

        if not isinstance(updates, dict):
            self.get_logger().error('/driving_gains: expected a JSON object {}')
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
                    self.get_logger().warn(f'/driving_gains: failed to set {key}: {e}')
            else:
                self.get_logger().warn(f'/driving_gains: unknown param "{key}" ignored')

        if applied:
            self.get_logger().info(f'Gains updated: {", ".join(applied)}')
            self._reset_integrators()

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _reset_integrators(self):
        """Clear all integrators and derivative history on mode/estop transitions."""
        self._v_err_prev = 0.0
        self._v_i_window.clear()
        self._u_vel = 0.0
        self._pitch_err_buf = [0.0, 0.0, 0.0]
        self._pitch_i_window.clear()

    def _publish_cmd(self, v: float, omega: float):
        msg = Twist()
        msg.linear.x  = float(v)
        msg.angular.z = float(omega)
        self._cmd_pub.publish(msg)

    # ---------------------------------------------------------------------- #
    # Gains echo (Foxglove readback)
    # ---------------------------------------------------------------------- #

    def _publish_gains_echo(self):
        gains = {
            'kp_vel':           self.get_parameter('kp_vel').value,
            'ki_vel':           self.get_parameter('ki_vel').value,
            'kd_vel':           self.get_parameter('kd_vel').value,
            'kp_pitch':         self.get_parameter('kp_pitch').value,
            'kd_pitch':         self.get_parameter('kd_pitch').value,
            'ki_pitch':         self.get_parameter('ki_pitch').value,
            'kff_decel':        self.get_parameter('kff_decel').value,
            'theta_eq_offset':  self.get_parameter('theta_eq_offset').value,
            'theta_max_pitch':  self.get_parameter('theta_max_pitch').value,
            'v_max':            self.get_parameter('v_max').value,
            # Runtime state
            '_mode':      self._mode,
            '_estop':     self._estop,
            '_theta_deg': round(math.degrees(self._theta), 2),
            '_v_actual':  round(self._v_actual, 3),
            '_v_cmd':     round(self._v_cmd, 3),
            '_u_vel':     round(self._u_vel, 4),
        }
        msg = String()
        msg.data = json.dumps(gains, indent=2)
        self._gains_echo_pub.publish(msg)

    # ---------------------------------------------------------------------- #
    # Control loops
    # ---------------------------------------------------------------------- #

    def _outer_tick(self):
        """50 Hz — velocity PID → updates shared _u_vel."""
        if self._estop or self._mode != self.DRIVING_MODE:
            self._u_vel = 0.0
            return

        now = self.get_clock().now()

        # Use EKF velocity if fresh, else fall back to 0
        if self._last_odom_time is None or (now - self._last_odom_time) > self._odom_timeout:
            v_actual = 0.0
        else:
            v_actual = self._v_actual

        kp_vel = float(self.get_parameter('kp_vel').value)
        ki_vel = float(self.get_parameter('ki_vel').value)
        kd_vel = float(self.get_parameter('kd_vel').value)

        v_err  = self._v_cmd - v_actual
        dv_err = (v_err - self._v_err_prev) / self._dt_outer
        self._v_err_prev = v_err

        # Sliding 2-second window integral (prevents windup)
        now_sec = now.nanoseconds * 1e-9
        self._v_i_window.append((now_sec, v_err * self._dt_outer))
        while self._v_i_window and now_sec - self._v_i_window[0][0] > 2.0:
            self._v_i_window.popleft()
        v_integral = sum(c for _, c in self._v_i_window)

        self._u_vel = (kp_vel * v_err
                     + ki_vel * v_integral
                     + kd_vel * dv_err)

    def _inner_tick(self):
        """100 Hz — pitch PID + combined output → /cmd_vel."""
        # Always zero on estop
        if self._estop:
            self._publish_cmd(0.0, 0.0)
            return

        # Passthrough when not in driving mode
        if self._mode != self.DRIVING_MODE:
            if self._last_cmd_time is not None:
                self._publish_cmd(self._v_cmd, self._omega_cmd)
            else:
                self._publish_cmd(0.0, 0.0)
            return

        now = self.get_clock().now()

        theta         = self._theta
        theta_eq      = float(self.get_parameter('theta_eq_offset').value)
        theta_max     = float(self.get_parameter('theta_max_pitch').value)
        kff_decel     = float(self.get_parameter('kff_decel').value)

        # Warn if pitch is extreme (robot likely contacting ground — do NOT estop)
        if abs(theta) > theta_max:
            self.get_logger().warn(
                f'Pitch |theta|={abs(theta):.3f} rad > {theta_max:.3f} rad limit. '
                f'Robot may be contacting ground. Continuing pitch correction.',
                throttle_duration_sec=1.0
            )

        # Desired pitch: static trim + decel feedforward
        # kff_decel * (v_cmd_dot): when braking hard, lean theta_ref forward
        # slightly so the pitch correction loop pre-compensates
        v_cmd_dot  = (self._v_cmd - self._v_cmd_prev) / self._dt_inner
        self._v_cmd_prev = self._v_cmd
        theta_ref  = theta_eq + kff_decel * v_cmd_dot

        # Pitch error
        pitch_err = theta - theta_ref

        # 3-sample weighted FIR derivative (shift ring buffer: [k, k-1, k-2])
        e0, e1, e2 = self._pitch_err_buf
        self._pitch_err_buf = [pitch_err, e0, e1]
        # weights [0.5, 0.25, 0.25] applied to successive differences
        d_pitch = (0.5 * (pitch_err - e0) + 0.25 * (e0 - e1)) / self._dt_inner

        kp_pitch = float(self.get_parameter('kp_pitch').value)
        kd_pitch = float(self.get_parameter('kd_pitch').value)
        ki_pitch = float(self.get_parameter('ki_pitch').value)

        # Sliding 2-second window integral for pitch
        now_sec = now.nanoseconds * 1e-9
        self._pitch_i_window.append((now_sec, pitch_err * self._dt_inner))
        while self._pitch_i_window and now_sec - self._pitch_i_window[0][0] > 2.0:
            self._pitch_i_window.popleft()
        pitch_integral = sum(c for _, c in self._pitch_i_window)

        # Pitch correction:
        # pitch_err > 0  → nose down during decel → u_pitch > 0 → reduces braking
        u_pitch = (kp_pitch * pitch_err
                 + kd_pitch * d_pitch
                 + ki_pitch * pitch_integral)

        # Combined output
        v_max = float(self.get_parameter('v_max').value)
        v_out = max(-v_max, min(v_max, self._u_vel + u_pitch))

        self._publish_cmd(v_out, self._omega_cmd)


def main(args=None):
    rclpy.init(args=args)
    node = DrivingController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
