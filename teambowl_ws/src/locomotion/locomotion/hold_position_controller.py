#!/usr/bin/env python3
"""
Hold Position Controller

Holds the RS04 leg joints at whatever position they were at the moment the
robot transitions from "off" to an active mode.  Unlike driving_leg_controller,
this does NOT command the calibrated YAML positions — it snapshots the current
/joint_states at enable time and holds those positions.

RS00 ankle motors are held at the fixed positions defined in leg_positions.yaml
(instead of freewheeling).  Kp/Kd gains for the ankles are set via the
rs00_kp and rs00_kd parameters in locomotion.yaml.
"""

import math
import os

import yaml
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from robstride_can_interfaces.srv import SetGains


class HoldPositionController(Node):
    """Holds RS04 joints at the positions they were in at enable time."""

    def __init__(self):
        super().__init__('hold_position_controller')

        # ------------------------------------------------------------------ #
        # Parameters
        # ------------------------------------------------------------------ #
        share_dir = get_package_share_directory('locomotion')
        default_config = os.path.join(share_dir, 'driving_leg_pos.yaml')
        default_leg_pos = os.path.join(share_dir, 'leg_positions.yaml')

        self.declare_parameter('config_path', default_config)
        self.declare_parameter('leg_positions_path', default_leg_pos)
        self.declare_parameter('mode_topic', '/robot_mode')
        self.declare_parameter('estop_topic', '/estop')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('joint_commands_topic', '/joint_commands')
        self.declare_parameter('torque_ff', 1.0)
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('rs00_joints', ['joint_rs00_1', 'joint_rs00_2'])
        self.declare_parameter('rs00_kp', 20.0)
        self.declare_parameter('rs00_kd', 0.3)
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('leveling_enabled', True)
        self.declare_parameter('leveling_kp', 0.5)
        self.declare_parameter('leveling_max_offset_rad', 0.2)
        self.declare_parameter('leveling_alpha', 0.85)

        config_path = self.get_parameter('config_path').value
        leg_positions_path = self.get_parameter('leg_positions_path').value
        mode_topic = self.get_parameter('mode_topic').value
        estop_topic = self.get_parameter('estop_topic').value
        joint_states_topic = self.get_parameter('joint_states_topic').value
        joint_cmds_topic = self.get_parameter('joint_commands_topic').value
        self._torque_ff = self.get_parameter('torque_ff').value
        publish_rate_hz = self.get_parameter('publish_rate_hz').value
        self._rs00_joints = list(self.get_parameter('rs00_joints').value)
        self._rs00_kp = self.get_parameter('rs00_kp').value
        self._rs00_kd = self.get_parameter('rs00_kd').value
        imu_topic = self.get_parameter('imu_topic').value
        self._leveling_enabled = self.get_parameter('leveling_enabled').value
        self._leveling_kp = self.get_parameter('leveling_kp').value
        self._leveling_max_offset = self.get_parameter('leveling_max_offset_rad').value
        self._leveling_alpha = self.get_parameter('leveling_alpha').value

        # RS04: names only — positions snapshotted from /joint_states at enable time.
        rs04_names = self._load_joint_names(config_path)
        # RS00: fixed ankle positions from leg_positions.yaml.
        self._ankle_positions = self._load_ankle_positions(leg_positions_path)
        # Leveling: per-joint sign factors from leg_positions.yaml.
        self._roll_signs = self._load_joint_signs(leg_positions_path, 'roll_signs')
        self._pitch_signs = self._load_joint_signs(leg_positions_path, 'pitch_signs')
        # Command all RS04 first, then RS00 (ankles) at the end.
        self._joint_names = rs04_names + list(self._ankle_positions.keys())
        self.get_logger().info(f'RS04 joints (snapshotted): {rs04_names}')
        self.get_logger().info(
            f'RS00 ankle joints (fixed): {list(self._ankle_positions.items())}'
        )
        if self._leveling_enabled:
            self.get_logger().info(
                f'Leveling ON  kp={self._leveling_kp}  '
                f'max_offset={self._leveling_max_offset} rad  alpha={self._leveling_alpha}'
            )

        # ------------------------------------------------------------------ #
        # State
        # ------------------------------------------------------------------ #
        self._mode = 'off'
        self._estop = False
        self._running = False
        self._coast_setup_done = False
        self._current_positions: dict = {}   # joint_name → float, updated continuously
        self._held_positions: list = []      # snapshot taken at enable time
        self._robot_roll = 0.0              # filtered roll estimate (rad)
        self._robot_pitch = 0.0             # filtered pitch estimate (rad)

        # ------------------------------------------------------------------ #
        # Publisher
        # ------------------------------------------------------------------ #
        self._cmd_pub = self.create_publisher(JointState, joint_cmds_topic, 10)

        # ------------------------------------------------------------------ #
        # Subscriptions
        # ------------------------------------------------------------------ #
        self.create_subscription(String, mode_topic, self._on_mode, 10)
        self.create_subscription(Bool, estop_topic, self._on_estop, 10)
        self.create_subscription(JointState, joint_states_topic, self._on_joint_states, 10)
        if self._leveling_enabled:
            self.create_subscription(Imu, imu_topic, self._on_imu, 10)

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

        # Deferred ankle hold setup: wait 3 s then apply RS00 Kp/Kd
        self._coast_timer = self.create_timer(3.0, self._setup_ankle_hold)

        # Periodic status print: per-joint position or X if not enabled
        self._status_timer = self.create_timer(2.0, self._print_status)

        self.get_logger().info('HoldPositionController ready.')

    # ---------------------------------------------------------------------- #
    # Config loading
    # ---------------------------------------------------------------------- #

    def _load_joint_names(self, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return list(data.get('joints', {}).keys())

    def _load_ankle_positions(self, path: str) -> dict:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        positions = data.get('ankle_positions', {})
        if not positions:
            self.get_logger().warn(f'No ankle_positions found in {path}')
        return {str(k): float(v) for k, v in positions.items()}

    def _load_joint_signs(self, path: str, key: str) -> dict:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        signs = data.get(key, {})
        return {str(k): float(v) for k, v in signs.items()}

    # ---------------------------------------------------------------------- #
    # Subscriptions
    # ---------------------------------------------------------------------- #

    def _on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._current_positions[name] = pos

    def _on_imu(self, msg: Imu):
        # IMU is mounted 90° clockwise from above:
        #   IMU +X → robot RIGHT (-robot Y)
        #   IMU +Y → robot FORWARD (+robot X)
        #   IMU +Z → robot UP (unchanged)
        #
        # Extract gravity-based roll and pitch in robot frame.
        # robot_roll  > 0 → robot leans LEFT
        # robot_pitch > 0 → robot leans FORWARD (nose down)
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z

        roll_raw = math.atan2(-ax, az)
        pitch_raw = math.atan2(-ay, az)

        a = self._leveling_alpha
        self._robot_roll = a * self._robot_roll + (1.0 - a) * roll_raw
        self._robot_pitch = a * self._robot_pitch + (1.0 - a) * pitch_raw

    def _on_mode(self, msg: String):
        new_mode = msg.data
        if new_mode == self._mode:
            return
        self._mode = new_mode
        self.get_logger().info(f'Mode → {new_mode}')
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

    def _should_run(self) -> bool:
        return self._mode != 'off' and not self._estop

    def _update_state(self):
        want_running = self._should_run()
        if want_running and not self._running:
            self._transition_to_running()
        elif not want_running and self._running:
            self._transition_to_stopped()

    def _transition_to_running(self):
        # RS04: snapshot current positions from /joint_states.
        rs04_names = [n for n in self._joint_names if n not in self._ankle_positions]
        missing = [n for n in rs04_names if n not in self._current_positions]
        if missing:
            self.get_logger().warn(
                f'No /joint_states yet for: {missing} — holding 0.0 rad for those joints.'
            )

        rs04_positions = [self._current_positions.get(n, 0.0) for n in rs04_names]

        # RS00: use fixed positions from leg_positions.yaml.
        ankle_names = list(self._ankle_positions.keys())
        ankle_pos = list(self._ankle_positions.values())

        self._held_positions = rs04_positions + ankle_pos

        self.get_logger().info('Holding positions:')
        for name, pos in zip(self._joint_names, self._held_positions):
            self.get_logger().info(f'  {name}: {pos:+.4f} rad')

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
        if not self._running or not self._held_positions:
            return

        positions = list(self._held_positions)

        if self._leveling_enabled:
            for i, name in enumerate(self._joint_names):
                roll_sign = self._roll_signs.get(name, 0.0)
                pitch_sign = self._pitch_signs.get(name, 0.0)
                if roll_sign == 0.0 and pitch_sign == 0.0:
                    continue
                offset = self._leveling_kp * (
                    roll_sign * self._robot_roll +
                    pitch_sign * self._robot_pitch
                )
                offset = max(-self._leveling_max_offset,
                             min(self._leveling_max_offset, offset))
                positions[i] += offset

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self._joint_names
        msg.position = positions
        msg.velocity = [0.0] * len(self._joint_names)
        msg.effort = [self._torque_ff] * len(self._joint_names)
        self._cmd_pub.publish(msg)

    # ---------------------------------------------------------------------- #
    # RS00 ankle hold setup (one-shot, deferred)
    # ---------------------------------------------------------------------- #

    def _setup_ankle_hold(self):
        self._coast_timer.cancel()

        if not self._rs00_joints:
            return

        self.get_logger().info(
            f'Setting RS00 ankle hold gains: kp={self._rs00_kp}, kd={self._rs00_kd} '
            f'for {self._rs00_joints}'
        )

        if not self._set_gains_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                '/set_gains not available — RS00 ankle hold gains not applied.'
            )
            return

        for joint in self._rs00_joints:
            req = SetGains.Request()
            req.joint_name = joint
            req.kp = self._rs00_kp
            req.kd = self._rs00_kd
            future = self._set_gains_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if future.done() and future.result() and future.result().success:
                self.get_logger().info(
                    f'  {joint}: kp={self._rs00_kp}, kd={self._rs00_kd}'
                )
            else:
                self.get_logger().warn(f'  {joint}: /set_gains call failed')

        self._coast_setup_done = True

    # ---------------------------------------------------------------------- #
    # Status print
    # ---------------------------------------------------------------------- #

    def _print_status(self):
        parts = []
        for i, name in enumerate(self._joint_names):
            if self._running and self._held_positions:
                val = f'{self._held_positions[i]:+.4f}'
            else:
                val = 'X'
            parts.append(f'{name}: {val}')
        status = f'[HOLD]  {"  ".join(parts)}'
        if self._leveling_enabled:
            status += (
                f'  |  roll={math.degrees(self._robot_roll):+.1f}°'
                f'  pitch={math.degrees(self._robot_pitch):+.1f}°'
            )
        print(status, flush=True)

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
    node = HoldPositionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
