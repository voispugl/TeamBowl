#!/usr/bin/env python3
"""
Hold Position Controller

Holds the RS04 leg joints at whatever position they were at the moment the
robot transitions from "off" to an active mode.  Unlike driving_leg_controller,
this does NOT command the calibrated YAML positions — it snapshots the current
/joint_states at enable time and holds those positions.

Use this when you want the joints to stay where they physically are (e.g. after
manually repositioning the robot) rather than snap back to the driving
calibration.

RS00 coast setup and RS05 exclusion are identical to driving_leg_controller.
"""

import os

import yaml
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from robstride_can_interfaces.srv import SetGains, ReadMotorParam, WriteMotorParam

# RS00 anti-backdrive damper parameter index.  Value 1 = damping disabled.
_PARAM_DAMPER = 0x702A


class HoldPositionController(Node):
    """Holds RS04 joints at the positions they were in at enable time."""

    def __init__(self):
        super().__init__('hold_position_controller')

        # ------------------------------------------------------------------ #
        # Parameters
        # ------------------------------------------------------------------ #
        share_dir = get_package_share_directory('locomotion')
        default_config = os.path.join(share_dir, 'driving_leg_pos.yaml')

        self.declare_parameter('config_path', default_config)
        self.declare_parameter('mode_topic', '/robot_mode')
        self.declare_parameter('estop_topic', '/estop')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('joint_commands_topic', '/joint_commands')
        self.declare_parameter('torque_ff', 1.0)
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('rs00_joints', ['joint_rs00_1', 'joint_rs00_2'])

        config_path = self.get_parameter('config_path').value
        mode_topic = self.get_parameter('mode_topic').value
        estop_topic = self.get_parameter('estop_topic').value
        joint_states_topic = self.get_parameter('joint_states_topic').value
        joint_cmds_topic = self.get_parameter('joint_commands_topic').value
        self._torque_ff = self.get_parameter('torque_ff').value
        publish_rate_hz = self.get_parameter('publish_rate_hz').value
        self._rs00_joints = list(self.get_parameter('rs00_joints').value)

        # Load joint names only from the shared YAML — positions come from /joint_states.
        self._joint_names = self._load_joint_names(config_path)
        self.get_logger().info(f'Joints to hold: {self._joint_names}')

        # ------------------------------------------------------------------ #
        # State
        # ------------------------------------------------------------------ #
        self._mode = 'off'
        self._estop = False
        self._running = False
        self._coast_setup_done = False
        self._current_positions: dict = {}   # joint_name → float, updated continuously
        self._held_positions: list = []      # snapshot taken at enable time

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

        # ------------------------------------------------------------------ #
        # Service clients
        # ------------------------------------------------------------------ #
        self._enable_client = self.create_client(Trigger, '/enable_motors')
        self._stop_client = self.create_client(Trigger, '/stop_motors')
        self._set_gains_client = self.create_client(SetGains, '/set_gains')
        self._read_param_client = self.create_client(ReadMotorParam, '/read_motor_param')
        self._write_param_client = self.create_client(
            WriteMotorParam, '/write_motor_param'
        )

        # ------------------------------------------------------------------ #
        # Timers
        # ------------------------------------------------------------------ #
        period = 1.0 / publish_rate_hz
        self._publish_timer = self.create_timer(period, self._publish_commands)

        # Deferred coast setup: wait 3 s then configure RS00 gains
        self._coast_timer = self.create_timer(3.0, self._setup_coast_mode)

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

    # ---------------------------------------------------------------------- #
    # Subscriptions
    # ---------------------------------------------------------------------- #

    def _on_joint_states(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._current_positions[name] = pos

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
        # Snapshot current positions for the joints we control.
        missing = [n for n in self._joint_names if n not in self._current_positions]
        if missing:
            self.get_logger().warn(
                f'No /joint_states yet for: {missing} — holding 0.0 rad for those joints.'
            )

        self._held_positions = [
            self._current_positions.get(n, 0.0) for n in self._joint_names
        ]

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

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self._joint_names
        msg.position = self._held_positions
        msg.velocity = [0.0] * len(self._joint_names)
        msg.effort = [self._torque_ff] * len(self._joint_names)
        self._cmd_pub.publish(msg)

    # ---------------------------------------------------------------------- #
    # RS00 coast setup (one-shot, deferred) — identical to driving_leg_controller
    # ---------------------------------------------------------------------- #

    def _setup_coast_mode(self):
        self._coast_timer.cancel()

        if not self._rs00_joints:
            return

        self.get_logger().info(
            f'Configuring RS00 coast mode for: {self._rs00_joints}'
        )

        for client, name in [
            (self._set_gains_client, '/set_gains'),
            (self._read_param_client, '/read_motor_param'),
            (self._write_param_client, '/write_motor_param'),
        ]:
            if not client.wait_for_service(timeout_sec=5.0):
                self.get_logger().warn(
                    f'Service {name} not available — RS00 coast setup skipped.'
                )
                return

        for joint in self._rs00_joints:
            gains_req = SetGains.Request()
            gains_req.joint_name = joint
            gains_req.kp = 0.0
            gains_req.kd = 0.0
            future = self._set_gains_client.call_async(gains_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if future.done() and future.result() and future.result().success:
                self.get_logger().info(f'  {joint}: gains set to kp=0, kd=0')
            else:
                self.get_logger().warn(f'  {joint}: /set_gains call failed')

            read_req = ReadMotorParam.Request()
            read_req.joint_name = joint
            read_req.param_index = _PARAM_DAMPER
            future = self._read_param_client.call_async(read_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

            if not (future.done() and future.result() and future.result().success):
                self.get_logger().warn(
                    f'  {joint}: could not read damper param — skipping write'
                )
                continue

            current_damper = int(round(future.result().value_float))
            if current_damper == 1:
                self.get_logger().info(
                    f'  {joint}: damper already disabled (0x702A=1) — skipping write'
                )
                continue

            self.get_logger().info(
                f'  {joint}: damper currently {current_damper} — disabling (0x702A→1)'
            )
            param_req = WriteMotorParam.Request()
            param_req.joint_name = joint
            param_req.param_index = _PARAM_DAMPER
            param_req.value = 1.0
            param_req.value_type = 'float'
            future = self._write_param_client.call_async(param_req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
            if future.done() and future.result() and future.result().success:
                self.get_logger().info(f'  {joint}: damper disabled (0x702A=1)')
            else:
                self.get_logger().warn(
                    f'  {joint}: /write_motor_param damper call failed'
                )

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
        print(f'[HOLD]  {"  ".join(parts)}', flush=True)

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
