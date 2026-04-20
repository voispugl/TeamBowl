#!/usr/bin/env python3
"""
Driving Controller Node
=======================
Velocity-tracking PID with parallel pitch and yaw correction for autonomous
driving with legs locked in place and rear legs dragging.

Control architecture
--------------------
All three PIDs run in a single loop at `control_rate_hz` (default 100 Hz):

  Velocity PID:
    v_err    = v_cmd - v_actual
    u_vel    = kp_vel*v_err + ki_vel*∫v_err + kd_vel*dv_err/dt

  Pitch PID (parallel — corrects nose-dive during deceleration):
    theta_ref = theta_eq_offset + kff_decel * v_cmd_dot
    pitch_err = theta - theta_ref
    u_pitch   = kp_pitch*pitch_err + kd_pitch*d_pitch + ki_pitch*∫pitch_err
    Derivative uses 3-sample weighted FIR to reduce IMU noise (~15 ms lag).

  Yaw PID (parallel — corrects angular velocity error):
    yaw_err  = omega_cmd - yaw_dot
    u_yaw    = omega_cmd + kp_yaw*yaw_err + ki_yaw*∫yaw_err + kd_yaw*d_yaw_err
    Additive: gains=0.0 → pure passthrough on omega.

  Combined output:
    v_out     = clamp(u_vel + u_pitch, -v_max, +v_max)
    omega_out = u_yaw

Sign convention:
  theta > 0  — nose down (forward lean), per _quat_to_pitch ZYX convention
  theta_dot  — angular_velocity.y from IMU
  yaw_dot    — angular_velocity.z from IMU
  linear.x > 0 — forward wheel velocity

Topics
------
Subscribes:
  /cmd_vel_safe        geometry_msgs/Twist  — desired v_cmd, omega_cmd
  /imu/data            sensor_msgs/Imu     — pitch angle, pitch rate, yaw rate
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
  {"kp_vel": 3.0, "kp_pitch": 8.0, "kp_yaw": 2.0}
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
    Velocity-tracking PID with parallel pitch and yaw correction.

    Single control loop at control_rate_hz:
      Velocity PID  → u_vel
      Pitch PID     → u_pitch   (parallel, corrects nose-dive)
      Yaw PID       → u_yaw     (parallel, additive correction on omega)

    v_out = clamp(u_vel + u_pitch, -v_max, v_max)
    omega_out = u_yaw  (= omega_cmd when kp/ki/kd_yaw = 0)
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

        # Velocity PID gains
        self.declare_parameter('kp_vel', 2.0)
        self.declare_parameter('ki_vel', 0.5)
        self.declare_parameter('kd_vel', 0.05)

        # Pitch PID gains
        self.declare_parameter('kp_pitch', 5.0)
        self.declare_parameter('kd_pitch', 1.0)
        self.declare_parameter('ki_pitch', 0.0)

        # Yaw PID gains (additive — 0.0 = passthrough)
        self.declare_parameter('kp_yaw', 0.0)
        self.declare_parameter('ki_yaw', 0.0)
        self.declare_parameter('kd_yaw', 0.0)

        # Decel feedforward: anticipate pitch from commanded velocity change
        self.declare_parameter('kff_decel', 0.0)

        # Static pitch trim
        self.declare_parameter('theta_eq_offset', 0.0)

        # Pitch limit: WARN only, no estop (robot is already grounded at this angle)
        self.declare_parameter('theta_max_pitch', 0.35)

        # Output velocity clamp [m/s]
        self.declare_parameter('v_max', 3.0)

        # Single control rate
        self.declare_parameter('control_rate_hz', 100.0)

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

        self._dt = 1.0 / float(self.get_parameter('control_rate_hz').value)

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
        self._yaw_dot   = 0.0   # yaw rate from gyro [rad/s]
        self._v_actual  = 0.0   # EKF-filtered forward velocity [m/s]

        self._last_odom_time = None
        self._last_imu_time  = None

        # ------------------------------------------------------------------ #
        # State — command inputs
        # ------------------------------------------------------------------ #
        self._v_cmd      = 0.0
        self._v_cmd_prev = 0.0
        self._omega_cmd  = 0.0
        self._last_cmd_time = None

        # ------------------------------------------------------------------ #
        # State — velocity PID
        # ------------------------------------------------------------------ #
        self._v_err_prev  = 0.0
        self._v_i_window: deque = deque()

        # ------------------------------------------------------------------ #
        # State — pitch PID
        # ------------------------------------------------------------------ #
        self._pitch_err_buf  = [0.0, 0.0, 0.0]   # ring buffer [k, k-1, k-2]
        self._pitch_i_window: deque = deque()

        # ------------------------------------------------------------------ #
        # State — yaw PID
        # ------------------------------------------------------------------ #
        self._yaw_err_prev  = 0.0
        self._yaw_i_window: deque = deque()

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
        self._cmd_pub        = self.create_publisher(Twist,  out_topic,             best_effort)
        self._gains_echo_pub = self.create_publisher(String, '/driving_gains_echo', reliable)

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
        self.create_timer(self._dt, self._tick)
        self.create_timer(2.0,      self._publish_gains_echo)

        self.get_logger().info(
            f'DrivingController up. Passthrough until mode="{self.DRIVING_MODE}". '
            f'Parallel velocity + pitch + yaw PIDs at {1.0/self._dt:.0f} Hz. '
            f'safe_in={safe_topic} → cmd_out={out_topic}. '
            f'Foxglove: pub JSON to /driving_gains, read /driving_gains_echo'
        )

    # ---------------------------------------------------------------------- #
    # Subscribers
    # ---------------------------------------------------------------------- #

    def _on_imu(self, msg: Imu):
        q = msg.orientation
        self._theta     = _quat_to_pitch(q.x, q.y, q.z, q.w)
        self._theta_dot = msg.angular_velocity.y
        self._yaw_dot   = msg.angular_velocity.z
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
        """Live gain update from Foxglove / web UI. JSON dict with any subset of tunable params."""
        FLOAT_PARAMS = {
            'kp_vel', 'ki_vel', 'kd_vel',
            'kp_pitch', 'kd_pitch', 'ki_pitch',
            'kp_yaw', 'ki_yaw', 'kd_yaw',
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
        self._v_err_prev    = 0.0
        self._v_i_window.clear()
        self._pitch_err_buf = [0.0, 0.0, 0.0]
        self._pitch_i_window.clear()
        self._yaw_err_prev  = 0.0
        self._yaw_i_window.clear()

    def _publish_cmd(self, v: float, omega: float):
        msg = Twist()
        msg.linear.x  = float(v)
        msg.angular.z = float(omega)
        self._cmd_pub.publish(msg)

    # ---------------------------------------------------------------------- #
    # Gains echo (Foxglove / web UI readback)
    # ---------------------------------------------------------------------- #

    def _publish_gains_echo(self):
        gains = {
            'kp_vel':          self.get_parameter('kp_vel').value,
            'ki_vel':          self.get_parameter('ki_vel').value,
            'kd_vel':          self.get_parameter('kd_vel').value,
            'kp_pitch':        self.get_parameter('kp_pitch').value,
            'kd_pitch':        self.get_parameter('kd_pitch').value,
            'ki_pitch':        self.get_parameter('ki_pitch').value,
            'kp_yaw':          self.get_parameter('kp_yaw').value,
            'ki_yaw':          self.get_parameter('ki_yaw').value,
            'kd_yaw':          self.get_parameter('kd_yaw').value,
            'kff_decel':       self.get_parameter('kff_decel').value,
            'theta_eq_offset': self.get_parameter('theta_eq_offset').value,
            'theta_max_pitch': self.get_parameter('theta_max_pitch').value,
            'v_max':           self.get_parameter('v_max').value,
            # Runtime state
            '_mode':      self._mode,
            '_estop':     self._estop,
            '_theta_deg': round(math.degrees(self._theta), 2),
            '_yaw_dot':   round(self._yaw_dot, 4),
            '_v_actual':  round(self._v_actual, 3),
            '_v_cmd':     round(self._v_cmd, 3),
        }
        msg = String()
        msg.data = json.dumps(gains, indent=2)
        self._gains_echo_pub.publish(msg)

    # ---------------------------------------------------------------------- #
    # Control loop — single timer, three parallel PIDs
    # ---------------------------------------------------------------------- #

    def _tick(self):
        """100 Hz — velocity PID + pitch PID + yaw PID in parallel → /cmd_vel."""
        if self._estop:
            self._publish_cmd(0.0, 0.0)
            return

        if self._mode != self.DRIVING_MODE:
            if self._last_cmd_time is not None:
                self._publish_cmd(self._v_cmd, self._omega_cmd)
            else:
                self._publish_cmd(0.0, 0.0)
            return

        now     = self.get_clock().now()
        now_sec = now.nanoseconds * 1e-9

        # Read params once per tick
        kp_vel   = float(self.get_parameter('kp_vel').value)
        ki_vel   = float(self.get_parameter('ki_vel').value)
        kd_vel   = float(self.get_parameter('kd_vel').value)
        kp_pitch = float(self.get_parameter('kp_pitch').value)
        kd_pitch = float(self.get_parameter('kd_pitch').value)
        ki_pitch = float(self.get_parameter('ki_pitch').value)
        kp_yaw   = float(self.get_parameter('kp_yaw').value)
        ki_yaw   = float(self.get_parameter('ki_yaw').value)
        kd_yaw   = float(self.get_parameter('kd_yaw').value)
        kff_decel  = float(self.get_parameter('kff_decel').value)
        theta_eq   = float(self.get_parameter('theta_eq_offset').value)
        theta_max  = float(self.get_parameter('theta_max_pitch').value)
        v_max      = float(self.get_parameter('v_max').value)

        # ── Velocity PID ──────────────────────────────────────────────────────
        v_actual = self._v_actual if (
            self._last_odom_time is not None
            and (now - self._last_odom_time) <= self._odom_timeout
        ) else 0.0

        v_err  = self._v_cmd - v_actual
        dv_err = (v_err - self._v_err_prev) / self._dt
        self._v_err_prev = v_err

        self._v_i_window.append((now_sec, v_err * self._dt))
        while self._v_i_window and now_sec - self._v_i_window[0][0] > 0.5:  # 0.5 s window
            self._v_i_window.popleft()
        v_integral = sum(c for _, c in self._v_i_window)

        u_vel = self._v_cmd + kp_vel * v_err + ki_vel * v_integral + kd_vel * dv_err

        # ── Pitch PID ─────────────────────────────────────────────────────────
        if abs(self._theta) > theta_max:
            self.get_logger().warn(
                f'Pitch |theta|={abs(self._theta):.3f} rad > {theta_max:.3f} rad limit.',
                throttle_duration_sec=1.0
            )

        v_cmd_dot        = (self._v_cmd - self._v_cmd_prev) / self._dt
        self._v_cmd_prev = self._v_cmd
        theta_ref        = theta_eq + kff_decel * v_cmd_dot
        pitch_err        = self._theta - theta_ref

        e0, e1, _ = self._pitch_err_buf
        self._pitch_err_buf = [pitch_err, e0, e1]
        d_pitch = (0.5 * (pitch_err - e0) + 0.25 * (e0 - e1)) / self._dt

        self._pitch_i_window.append((now_sec, pitch_err * self._dt))
        while self._pitch_i_window and now_sec - self._pitch_i_window[0][0] > 0.5:  # 0.5 s window
            self._pitch_i_window.popleft()
        pitch_integral = sum(c for _, c in self._pitch_i_window)

        u_pitch = kp_pitch * pitch_err + kd_pitch * d_pitch + ki_pitch * pitch_integral

        # ── Yaw PID ───────────────────────────────────────────────────────────
        yaw_err = self._omega_cmd - self._yaw_dot
        d_yaw   = (yaw_err - self._yaw_err_prev) / self._dt
        self._yaw_err_prev = yaw_err

        self._yaw_i_window.append((now_sec, yaw_err * self._dt))
        while self._yaw_i_window and now_sec - self._yaw_i_window[0][0] > 0.5:  # 0.5 s window
            self._yaw_i_window.popleft()
        yaw_integral = sum(c for _, c in self._yaw_i_window)

        # Additive: gains=0 → omega_out = omega_cmd (pure passthrough)
        u_yaw = self._omega_cmd + kp_yaw * yaw_err + ki_yaw * yaw_integral + kd_yaw * d_yaw

        # ── Combined output ───────────────────────────────────────────────────
        v_out = max(-v_max, min(v_max, u_vel + u_pitch))
        self._publish_cmd(v_out, u_yaw)


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
