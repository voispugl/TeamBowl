#!/usr/bin/env python3
"""
Jump Controller Node
====================
Executes a jump by rapidly extending both legs into the ground.

Sequence:
  IDLE → CROUCH → EXTEND → RETURN → IDLE

  CROUCH  Legs retract to `crouch_depth` × max extension. Normal gains.
          Held for `crouch_hold_s` seconds before extending.
  EXTEND  Legs slam to ~95% max extension at high Kd + torque feedforward.
          Held for `extend_hold_s` (impulse duration).
  RETURN  Legs commanded back to driving positions. Normal gains.
          Transitions to IDLE once joints settle or timeout.

Balance suspension
-------------------
During CROUCH and EXTEND the node publishes True on /balance_suspend so the
balance_controller zeroes its wheel output and doesn't fight the impulse.

Topics
------
Sub  /jump_command        std_msgs/String  — publish "jump" to trigger
Sub  /joint_states        sensor_msgs/JointState
Sub  /robot_mode          std_msgs/String
Sub  /estop               std_msgs/Bool
Pub  /joint_commands      sensor_msgs/JointState  — 100 Hz during jump phases
Pub  /balance_suspend     std_msgs/Bool           — True while jumping
"""

import os
import math
import numpy as np

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

from locomotion.leg_kinematics import (
    LegCalibration,
    DRIVING_POS,
    JOINT_NAMES,
    compute_jump_waypoints,
    L_MAX,
)

# ── States ─────────────────────────────────────────────────────────────────── #
_IDLE   = 'IDLE'
_CROUCH = 'CROUCH'
_EXTEND = 'EXTEND'
_RETURN = 'RETURN'

# All six RS04 joints in command order
_ALL_JOINTS = [
    'joint_rs04_1', 'joint_rs04_2', 'joint_rs04_3',
    'joint_rs04_4', 'joint_rs04_5', 'joint_rs04_6',
]

# Driving positions as flat dict
_DRIVING_FLAT = {
    JOINT_NAMES['left']['hip']: DRIVING_POS['left']['hip'],
    JOINT_NAMES['left']['A']:   DRIVING_POS['left']['A'],
    JOINT_NAMES['left']['B']:   DRIVING_POS['left']['B'],
    JOINT_NAMES['right']['hip']: DRIVING_POS['right']['hip'],
    JOINT_NAMES['right']['A']:   DRIVING_POS['right']['A'],
    JOINT_NAMES['right']['B']:   DRIVING_POS['right']['B'],
}


class JumpController(Node):
    """4-state jump state machine for TeamBowl parallel 5-bar legs."""

    def __init__(self):
        super().__init__('jump_controller')

        # ── Parameters ──────────────────────────────────────────────────── #
        self.declare_parameter('crouch_depth', 0.6)
        self.declare_parameter('crouch_hold_s', 0.30)
        self.declare_parameter('extend_hold_s', 0.12)
        self.declare_parameter('extend_torque_ff', 10.0)
        self.declare_parameter('extend_kd_override', 60.0)
        self.declare_parameter('return_timeout_s', 1.5)
        self.declare_parameter('publish_rate_hz', 100.0)
        self.declare_parameter('joint_settled_threshold', 0.05)
        self.declare_parameter('foot_height_driving_m', -0.28)

        rate_hz   = float(self.get_parameter('publish_rate_hz').value)
        foot_h    = float(self.get_parameter('foot_height_driving_m').value)
        crouch_d  = float(self.get_parameter('crouch_depth').value)

        # ── Calibrate kinematics ─────────────────────────────────────────── #
        self._cal_left  = LegCalibration()
        self._cal_right = LegCalibration()
        self._cal_left.calibrate_from_driving_pos(
            DRIVING_POS['left']['A'], DRIVING_POS['left']['B'], foot_h)
        self._cal_right.calibrate_from_driving_pos(
            DRIVING_POS['right']['A'], DRIVING_POS['right']['B'], foot_h)

        self._waypoints = compute_jump_waypoints(
            self._cal_left, self._cal_right, crouch_depth=crouch_d)

        self.get_logger().info(
            f'IK crouch:  {self._waypoints["crouch"]}\n'
            f'IK extend:  {self._waypoints["extend"]}'
        )

        # ── State ────────────────────────────────────────────────────────── #
        self._state: str = _IDLE
        self._state_start: float = 0.0
        self._mode: str = 'off'
        self._estop: bool = False
        self._joint_pos: dict = {}

        # ── Pubs ─────────────────────────────────────────────────────────── #
        self._cmd_pub     = self.create_publisher(JointState, '/joint_commands', 10)
        self._suspend_pub = self.create_publisher(Bool, '/balance_suspend', 10)

        # ── Subs ─────────────────────────────────────────────────────────── #
        self.create_subscription(String,     '/jump_command', self._on_jump_cmd,    10)
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10)
        self.create_subscription(String,     '/robot_mode',   self._on_mode,        10)
        self.create_subscription(Bool,       '/estop',        self._on_estop,       10)

        # ── Timer ────────────────────────────────────────────────────────── #
        self._timer = self.create_timer(1.0 / rate_hz, self._tick)

        self.get_logger().info('JumpController ready — publish "jump" to /jump_command.')

    # ── Callbacks ─────────────────────────────────────────────────────────── #

    def _on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._joint_pos[name] = pos

    def _on_mode(self, msg: String):
        self._mode = msg.data.strip().lower()
        if self._mode == 'off' and self._state != _IDLE:
            self._abort()

    def _on_estop(self, msg: Bool):
        self._estop = msg.data
        if self._estop and self._state != _IDLE:
            self._abort()

    def _on_jump_cmd(self, msg: String):
        if msg.data.strip().lower() != 'jump':
            return
        if self._estop:
            self.get_logger().warn('Jump rejected: estop active.')
            return
        if self._mode == 'off':
            self.get_logger().warn('Jump rejected: robot mode is off.')
            return
        if self._state != _IDLE:
            self.get_logger().warn(f'Jump rejected: already in state {self._state}.')
            return
        self._enter_state(_CROUCH)

    # ── State machine tick ────────────────────────────────────────────────── #

    def _tick(self):
        if self._state == _IDLE:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now - self._state_start

        crouch_hold = float(self.get_parameter('crouch_hold_s').value)
        extend_hold = float(self.get_parameter('extend_hold_s').value)
        return_to   = float(self.get_parameter('return_timeout_s').value)
        thr         = float(self.get_parameter('joint_settled_threshold').value)

        if self._state == _CROUCH:
            self._publish_joints(self._waypoints['crouch'], torque_ff=0.0, kd_override=None)
            if elapsed >= crouch_hold:
                self._enter_state(_EXTEND)

        elif self._state == _EXTEND:
            ff  = float(self.get_parameter('extend_torque_ff').value)
            kd  = float(self.get_parameter('extend_kd_override').value)
            self._publish_joints(self._waypoints['extend'], torque_ff=ff, kd_override=kd)
            if elapsed >= extend_hold:
                self._enter_state(_RETURN)

        elif self._state == _RETURN:
            self._publish_joints(_DRIVING_FLAT, torque_ff=0.0, kd_override=None)
            settled = self._joints_settled(_DRIVING_FLAT, thr)
            if settled or elapsed >= return_to:
                self._enter_state(_IDLE)

    def _enter_state(self, new_state: str):
        self._state = new_state
        self._state_start = self.get_clock().now().nanoseconds * 1e-9

        suspend = new_state in (_CROUCH, _EXTEND)
        msg = Bool()
        msg.data = suspend
        self._suspend_pub.publish(msg)

        if new_state == _IDLE:
            self.get_logger().info('Jump complete — IDLE.')
        else:
            self.get_logger().info(f'Jump state → {new_state}')

    def _abort(self):
        self.get_logger().warn('Jump aborted — returning to driving positions.')
        self._publish_joints(_DRIVING_FLAT, torque_ff=0.0, kd_override=None)
        msg = Bool()
        msg.data = False
        self._suspend_pub.publish(msg)
        self._state = _IDLE

    # ── Publishing helpers ────────────────────────────────────────────────── #

    def _publish_joints(self, positions: dict,
                        torque_ff: float,
                        kd_override: float | None):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        names, pos, vel, effort = [], [], [], []
        for name in _ALL_JOINTS:
            if name not in positions:
                continue
            names.append(name)
            pos.append(float(positions[name]))
            # Velocity field repurposed as Kd override when non-zero
            vel.append(float(kd_override) if kd_override is not None else 0.0)
            effort.append(float(torque_ff))
        msg.name     = names
        msg.position = pos
        msg.velocity = vel
        msg.effort   = effort
        self._cmd_pub.publish(msg)

    def _joints_settled(self, targets: dict, threshold: float) -> bool:
        for name, target in targets.items():
            actual = self._joint_pos.get(name)
            if actual is None:
                return False
            if abs(actual - target) > threshold:
                return False
        return True


def main(args=None):
    rclpy.init(args=args)
    node = JumpController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
