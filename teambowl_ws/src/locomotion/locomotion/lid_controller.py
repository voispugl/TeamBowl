#!/usr/bin/env python3
"""
Lid Controller — RS05 motor on the cargo bay lid.

Drives the lid to open or closed position via MIT mode (Type 1 CAN frames
through the robstride_can_driver's /joint_commands topic).

Commands arrive on /lid_command (std_msgs/String):
  "open"   — move to open_position_rad
  "close"  — move to closed_position_rad
  "toggle" — switch to the opposite of the current target

State is reported on /lid_state (std_msgs/String):
  "open", "closed", "moving_open", "moving_closed", "unknown"

Does NOT depend on /robot_mode — the lid is a payload actuator and should
be commandable whenever the stack is up. E-stop zeros torque output.

Foxglove setup:
  1. Add a Publish panel → topic /lid_command, type std_msgs/String
     Pre-fill message: {"data": "open"} — click to open
  2. Add a second Publish panel, message: {"data": "close"} — click to close
  3. Add a Raw Messages panel → /lid_state to read current state
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger


# States
_UNKNOWN = 'unknown'
_OPEN = 'open'
_CLOSED = 'closed'
_MOVING_OPEN = 'moving_open'
_MOVING_CLOSED = 'moving_closed'


class LidController(Node):
    """Drives the RS05 lid motor between open and closed positions."""

    def __init__(self):
        super().__init__('lid_controller')

        # ------------------------------------------------------------------ #
        # Parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter('joint_name', 'joint_rs05_1')
        self.declare_parameter('open_position_rad', 1.57)
        self.declare_parameter('closed_position_rad', 0.0)
        self.declare_parameter('kp', 60.0)
        self.declare_parameter('kd', 1.0)
        self.declare_parameter('torque_ff', 0.5)
        self.declare_parameter('move_timeout_sec', 3.0)
        self.declare_parameter('position_tolerance_rad', 0.05)
        self.declare_parameter('publish_rate_hz', 50.0)

        self._joint_name = self.get_parameter('joint_name').value
        self._open_pos = self.get_parameter('open_position_rad').value
        self._closed_pos = self.get_parameter('closed_position_rad').value
        self._kp = self.get_parameter('kp').value
        self._kd = self.get_parameter('kd').value
        self._torque_ff = self.get_parameter('torque_ff').value
        self._move_timeout = self.get_parameter('move_timeout_sec').value
        self._tolerance = self.get_parameter('position_tolerance_rad').value
        rate_hz = self.get_parameter('publish_rate_hz').value

        # ------------------------------------------------------------------ #
        # State
        # ------------------------------------------------------------------ #
        self._target_pos = self._closed_pos   # start closed
        self._current_pos = None              # from /joint_states feedback
        self._state = _UNKNOWN
        self._estop = False
        self._move_start_time = None          # set when MOVING starts

        # ------------------------------------------------------------------ #
        # TRANSIENT_LOCAL QoS for topics that latch last value
        # ------------------------------------------------------------------ #
        transient = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        # ------------------------------------------------------------------ #
        # Subscribers
        # ------------------------------------------------------------------ #
        self.create_subscription(String, '/lid_command', self._on_command, 10)
        self.create_subscription(Bool, '/estop', self._on_estop, transient)
        self.create_subscription(
            JointState, '/joint_states', self._on_joint_states, 10)

        # ------------------------------------------------------------------ #
        # Publishers
        # ------------------------------------------------------------------ #
        self._joint_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self._state_pub = self.create_publisher(String, '/lid_state', transient)

        # ------------------------------------------------------------------ #
        # Control timer
        # ------------------------------------------------------------------ #
        period = 1.0 / rate_hz
        self.create_timer(period, self._control_tick)

        # ------------------------------------------------------------------ #
        # Enable motors (one-shot on startup)
        # ------------------------------------------------------------------ #
        self._enable_client = self.create_client(Trigger, '/enable_motors')
        self._motors_enabled = False
        self.create_timer(0.5, self._try_enable_motors)

        self.get_logger().info(
            f'LidController ready — joint: {self._joint_name}  '
            f'open: {self._open_pos:.3f} rad  closed: {self._closed_pos:.3f} rad'
        )

    # ------------------------------------------------------------------ #
    # Startup — enable motors once
    # ------------------------------------------------------------------ #

    def _try_enable_motors(self):
        if self._motors_enabled:
            return
        if not self._enable_client.wait_for_service(timeout_sec=0.0):
            return  # driver not up yet, will retry on next tick
        req = Trigger.Request()
        future = self._enable_client.call_async(req)
        future.add_done_callback(self._on_enable_done)
        self._motors_enabled = True  # mark so we don't call again

    def _on_enable_done(self, future):
        try:
            resp = future.result()
            if resp.success:
                self.get_logger().info('Motors enabled.')
            else:
                self.get_logger().warn(f'enable_motors returned: {resp.message}')
        except Exception as e:
            self.get_logger().error(f'enable_motors call failed: {e}')
            self._motors_enabled = False  # allow retry

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def _on_command(self, msg: String):
        cmd = msg.data.strip().lower()
        if cmd == 'open':
            self._begin_move(_MOVING_OPEN, self._open_pos)
        elif cmd == 'close':
            self._begin_move(_MOVING_CLOSED, self._closed_pos)
        elif cmd == 'toggle':
            if self._state in (_OPEN, _MOVING_OPEN):
                self._begin_move(_MOVING_CLOSED, self._closed_pos)
            else:
                self._begin_move(_MOVING_OPEN, self._open_pos)
        else:
            self.get_logger().warn(f'Unknown lid command: "{msg.data}" — use open/close/toggle')

    def _begin_move(self, new_state, target_pos):
        self._target_pos = target_pos
        self._state = new_state
        self._move_start_time = self.get_clock().now()
        direction = 'open' if new_state == _MOVING_OPEN else 'close'
        self.get_logger().info(
            f'Lid → {direction}  target={target_pos:.3f} rad'
        )

    def _on_estop(self, msg: Bool):
        self._estop = msg.data
        if self._estop:
            self._state = _UNKNOWN
            self.get_logger().warn('E-stop received — lid holding zero torque')

    def _on_joint_states(self, msg: JointState):
        try:
            idx = msg.name.index(self._joint_name)
            self._current_pos = msg.position[idx]
        except (ValueError, IndexError):
            pass

    # ------------------------------------------------------------------ #
    # Control loop
    # ------------------------------------------------------------------ #

    def _control_tick(self):
        if self._estop:
            self._publish_joint(self._target_pos, torque=0.0)
            self._publish_state()
            return

        # Check if we've arrived or timed out while moving
        if self._state in (_MOVING_OPEN, _MOVING_CLOSED):
            # Arrival check using /joint_states feedback
            if self._current_pos is not None:
                err = abs(self._current_pos - self._target_pos)
                if err < self._tolerance:
                    self._state = _OPEN if self._state == _MOVING_OPEN else _CLOSED
                    self.get_logger().info(
                        f'Lid arrived at {"open" if self._state == _OPEN else "closed"} '
                        f'(pos={self._current_pos:.3f} rad)'
                    )

            # Timeout check
            if self._move_start_time is not None:
                elapsed = (self.get_clock().now() - self._move_start_time).nanoseconds * 1e-9
                if elapsed > self._move_timeout:
                    prev = self._state
                    self._state = _OPEN if prev == _MOVING_OPEN else _CLOSED
                    self.get_logger().warn(
                        f'Lid move timed out after {elapsed:.1f}s — '
                        f'declaring {"open" if self._state == _OPEN else "closed"}'
                    )
                    self._move_start_time = None

        # Publish command
        moving = self._state in (_MOVING_OPEN, _MOVING_CLOSED)
        torque = self._torque_ff if moving else 0.0
        self._publish_joint(self._target_pos, torque=torque)
        self._publish_state()

    def _publish_joint(self, position: float, torque: float):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [self._joint_name]
        msg.position = [position]
        msg.velocity = [0.0]
        msg.effort = [torque]
        # Embed kp/kd in header frame_id as "kp:XX kd:XX" convention used by driver
        # The robstride driver reads these from the JointState directly, not header.
        # Gains are set per-joint via /set_gains service at startup; here we just
        # send the position command and rely on the driver's stored gains.
        self._joint_pub.publish(msg)

    def _publish_state(self):
        msg = String()
        msg.data = self._state
        self._state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
