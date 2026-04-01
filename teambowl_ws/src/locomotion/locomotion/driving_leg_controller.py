#!/usr/bin/env python3
"""
Driving Leg Position Controller

Holds the 6 RS04 leg joints at their calibrated driving positions using MIT
mode (Type 1 CAN frames via the robstride_can_driver).

Behaviour:
- RUNNING (mode != "off" AND NOT estop):
    Publishes /joint_commands at publish_rate_hz for the RS04 joints only.
    Positions come from driving_leg_pos.yaml; velocity = 0, torque_ff = 0 Nm.
    Calls /enable_motors on first transition into RUNNING.
- STOPPED (mode == "off" OR estop active):
    Calls /stop_motors and stops publishing.

RS00 and RS05 are not controlled by this node. RS00 gains are set to zero
in motors.yaml so they freewheel by default.
"""

import os

import yaml
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from robstride_can_interfaces.srv import SetGains


class DrivingLegController(Node):
    """Holds RS04 leg joints at driving positions via MIT mode."""

    def __init__(self):
        super().__init__('driving_leg_controller')

        # ------------------------------------------------------------------ #
        # Parameters
        # ------------------------------------------------------------------ #
        share_dir = get_package_share_directory('locomotion')
        default_config = os.path.join(share_dir, 'driving_leg_pos.yaml')

        self.declare_parameter('config_path', default_config)
        self.declare_parameter('mode_topic', '/robot_mode')
        self.declare_parameter('estop_topic', '/estop')
        self.declare_parameter('joint_commands_topic', '/joint_commands')
        self.declare_parameter('torque_ff', 0.0)
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('auto_start', True)
        self.declare_parameter('auto_start_delay_s', 8.0)
        self.declare_parameter('rs00_joints', ['joint_rs00_1', 'joint_rs00_2'])
        self.declare_parameter('trick_rs00_kp', 10.0)
        self.declare_parameter('trick_rs00_kd', 5.0)

        config_path = self.get_parameter('config_path').value
        mode_topic = self.get_parameter('mode_topic').value
        estop_topic = self.get_parameter('estop_topic').value
        joint_cmds_topic = self.get_parameter('joint_commands_topic').value
        self._torque_ff = self.get_parameter('torque_ff').value
        publish_rate_hz = self.get_parameter('publish_rate_hz').value
        auto_start = self.get_parameter('auto_start').value
        auto_start_delay_s = self.get_parameter('auto_start_delay_s').value
        self._rs00_joints = list(self.get_parameter('rs00_joints').value)
        self._trick_rs00_kp = float(self.get_parameter('trick_rs00_kp').value)
        self._trick_rs00_kd = float(self.get_parameter('trick_rs00_kd').value)

        # ------------------------------------------------------------------ #
        # Load joint positions from YAML
        # ------------------------------------------------------------------ #
        self._joint_names, self._joint_positions = self._load_config(config_path)
        self.get_logger().info(
            f'Loaded {len(self._joint_names)} joints from {config_path}'
        )

        # ------------------------------------------------------------------ #
        # State
        # ------------------------------------------------------------------ #
        self._mode = 'off'
        self._estop = False
        self._running = False
        self._current_positions: dict = {}   # joint_name → float, from /joint_states
        self._trick_offsets: dict = {}       # joint_name → offset (rad), from /trick_leg_offsets

        # ------------------------------------------------------------------ #
        # Publisher
        # ------------------------------------------------------------------ #
        self._cmd_pub = self.create_publisher(JointState, joint_cmds_topic, 10)

        # ------------------------------------------------------------------ #
        # Subscriptions
        # ------------------------------------------------------------------ #
        self.create_subscription(String, mode_topic, self._on_mode, 10)
        self.create_subscription(Bool, estop_topic, self._on_estop, 10)
        self.create_subscription(JointState, '/joint_states', self._on_joint_states, 10)
        self.create_subscription(JointState, '/trick_leg_offsets', self._on_trick_offsets, 10)

        # ------------------------------------------------------------------ #
        # Service clients
        # ------------------------------------------------------------------ #
        self._enable_client = self.create_client(Trigger, '/enable_motors')
        self._stop_client = self.create_client(Trigger, '/stop_motors')
        self._set_gains_client = self.create_client(SetGains, '/set_gains')

        # ------------------------------------------------------------------ #
        # Timers
        # ------------------------------------------------------------------ #
        period = 1.0 / publish_rate_hz
        self._publish_timer = self.create_timer(period, self._publish_commands)
        self._status_timer = self.create_timer(5.0, self._print_status)

        if auto_start:
            self._auto_start_timer = self.create_timer(
                auto_start_delay_s, self._auto_start_callback
            )

        self.get_logger().info('DrivingLegController ready.')

    # ---------------------------------------------------------------------- #
    # Config loading
    # ---------------------------------------------------------------------- #

    def _load_config(self, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        joints_map = data.get('joints', {})
        names = list(joints_map.keys())
        positions = [float(joints_map[n]) for n in names]
        return names, positions

    # ---------------------------------------------------------------------- #
    # Subscriptions
    # ---------------------------------------------------------------------- #

    def _on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._current_positions[name] = pos

    def _on_trick_offsets(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._trick_offsets[name] = pos

    def _on_mode(self, msg: String):
        new_mode = msg.data
        if new_mode == self._mode:
            return
        old_mode = self._mode
        self._mode = new_mode
        self.get_logger().info(f'Mode → {new_mode}')
        if new_mode == 'trick':
            self._lock_rs00_wheels()
        elif old_mode == 'trick':
            self._release_rs00_wheels()
        self._update_state()

    def _on_estop(self, msg: Bool):
        if msg.data == self._estop:
            return
        self._estop = msg.data
        if self._estop:
            self.get_logger().warn('E-stop active')
        self._update_state()

    # ---------------------------------------------------------------------- #
    # State machine
    # ---------------------------------------------------------------------- #

    def _auto_start_callback(self):
        self._auto_start_timer.cancel()
        if not self._running:
            self.get_logger().info('auto_start: enabling leg controller.')
            self._transition_to_running()

    def _should_run(self) -> bool:
        return self._mode != 'off' and not self._estop

    def _update_state(self):
        want_running = self._should_run()
        if want_running and not self._running:
            self._transition_to_running()
        elif not want_running and self._running:
            self._transition_to_stopped()

    def _log_movement_preview(self):
        """Log current → target delta for each RS04 joint before enabling."""
        log = self.get_logger()
        if not self._current_positions:
            log.warn('Movement preview: no /joint_states received yet — cannot compare.')
            return

        log.info('--- Movement preview (current → target) ---')
        max_delta = 0.0
        max_joint = ''
        for name, target in zip(self._joint_names, self._joint_positions):
            if name not in self._current_positions:
                log.warn(f'  {name}: no current position in /joint_states')
                continue
            current = self._current_positions[name]
            delta = abs(target - current)
            line = f'  {name}: {current:+.4f} rad → {target:+.4f} rad  (Δ = {delta:.4f} rad)'
            if delta > 0.3:
                log.warn(line + '  <-- LARGE MOVE')
            else:
                log.info(line)
            if delta > max_delta:
                max_delta = delta
                max_joint = name
        if max_joint:
            log.info(f'  Max movement: {max_delta:.4f} rad on {max_joint}')

    def _transition_to_running(self):
        self._log_movement_preview()
        self.get_logger().info('Enabling motors and starting position hold.')
        self._running = True
        self._call_trigger(self._enable_client, '/enable_motors')

    def _transition_to_stopped(self):
        self.get_logger().info('Stopping motors.')
        self._running = False
        self._call_trigger(self._stop_client, '/stop_motors')

    # ---------------------------------------------------------------------- #
    # Publish loop
    # ---------------------------------------------------------------------- #

    def _publish_commands(self):
        if not self._running:
            return

        in_trick = (self._mode == 'trick')
        positions = [
            base + self._trick_offsets.get(name, 0.0) if in_trick else base
            for name, base in zip(self._joint_names, self._joint_positions)
        ]

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self._joint_names
        msg.position = positions
        msg.velocity = [0.0] * len(self._joint_names)
        msg.effort = [self._torque_ff] * len(self._joint_names)
        self._cmd_pub.publish(msg)

    # ---------------------------------------------------------------------- #
    # Status print
    # ---------------------------------------------------------------------- #

    def _print_status(self):
        state = 'RUNNING' if self._running else 'STOPPED'
        in_trick = (self._mode == 'trick')
        for name, base in zip(self._joint_names, self._joint_positions):
            target = base + self._trick_offsets.get(name, 0.0) if in_trick else base
            actual = self._current_positions.get(name)
            if actual is not None:
                err = target - actual
                print(
                    f'[DRIVE/{state}]  {name}: target={target:+.4f}  '
                    f'actual={actual:+.4f}  err={err:+.4f}',
                    flush=True,
                )
            else:
                print(
                    f'[DRIVE/{state}]  {name}: target={target:+.4f}  actual=no_data',
                    flush=True,
                )

    # ---------------------------------------------------------------------- #
    # RS00 wheel lock / release
    # ---------------------------------------------------------------------- #

    def _lock_rs00_wheels(self):
        positions = [self._current_positions.get(n, 0.0) for n in self._rs00_joints]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self._rs00_joints)
        msg.position = positions
        msg.velocity = [0.0] * len(self._rs00_joints)
        msg.effort = [0.0] * len(self._rs00_joints)
        self._cmd_pub.publish(msg)
        self._set_rs00_gains(self._trick_rs00_kp, self._trick_rs00_kd)
        self.get_logger().info(
            f'RS00 wheels locked at {[f"{p:.4f}" for p in positions]} '
            f'(kp={self._trick_rs00_kp}, kd={self._trick_rs00_kd})'
        )

    def _release_rs00_wheels(self):
        self._set_rs00_gains(0.0, 0.0)
        self.get_logger().info('RS00 wheels released to coast.')

    def _set_rs00_gains(self, kp: float, kd: float):
        if not self._set_gains_client.service_is_ready():
            self.get_logger().warn('/set_gains not ready — skipping RS00 gain change')
            return
        for name in self._rs00_joints:
            req = SetGains.Request()
            req.joint_name = name
            req.kp = kp
            req.kd = kd
            future = self._set_gains_client.call_async(req)
            future.add_done_callback(
                lambda f, n=name: self._log_set_gains_result(f, n)
            )

    def _log_set_gains_result(self, future, joint_name: str):
        try:
            result = future.result()
            if not result.success:
                self.get_logger().warn(f'/set_gains {joint_name} → {result.message}')
        except Exception as e:
            self.get_logger().error(f'/set_gains {joint_name} failed: {e}')

    # ---------------------------------------------------------------------- #
    # Service helper
    # ---------------------------------------------------------------------- #

    def _call_trigger(self, client, name: str):
        if not client.service_is_ready():
            self.get_logger().warn(f'{name} not ready — skipping call.')
            return
        future = client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda f: self._log_trigger_result(f, name)
        )

    def _log_trigger_result(self, future, name: str):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f'{name} → OK')
            else:
                self.get_logger().warn(f'{name} → {result.message}')
        except Exception as e:
            self.get_logger().error(f'{name} call failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = DrivingLegController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
