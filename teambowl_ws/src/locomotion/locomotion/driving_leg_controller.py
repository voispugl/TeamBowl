#!/usr/bin/env python3
"""
Driving Leg Position Controller

Holds RS04 leg joints at calibrated driving positions using MIT mode
(Type 1 CAN frames via the robstride_can_driver).

Behaviour:
- RUNNING (mode != "off" AND NOT estop):
    Publishes /joint_commands at publish_rate_hz.
    RS04 joints: YAML positions with torque_ff from a PI loop on measured joint
      effort, keeping rollers pressed against the ground while moving.
    RS00 joints: YAML tiptoe (roller) positions while /cmd_vel is non-zero;
      brake positions (tiptoe + rs00_brake_offsets ≈ ±π/2) when stopped for
      brake_debounce_s seconds, so the sandpaper sole contacts the ground.
- STOPPED (mode == "off" OR estop active):
    Calls /stop_motors and stops publishing.

Ground-force PI: reads RS04 effort from /joint_states each tick and adjusts
torque_ff so mean effort converges to target_ground_torque (Nm).
"""

import math
import os

import yaml
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


class DrivingLegController(Node):
    """Holds RS04 leg joints at driving positions; manages RS00 tiptoe/brake."""

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
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('verbose', False)
        self.declare_parameter('auto_start', True)
        self.declare_parameter('auto_start_delay_s', 8.0)

        # Brake mode
        self.declare_parameter('rs00_joints', ['joint_rs00_1', 'joint_rs00_2'])
        self.declare_parameter('rs00_brake_offsets', [-1.5708, 1.5708])
        self.declare_parameter('vel_stopped_threshold', 0.05)
        self.declare_parameter('brake_debounce_s', 0.3)

        # Ground-force PI
        self.declare_parameter('target_ground_torque', 2.0)
        self.declare_parameter('kp_gnd', 0.5)
        self.declare_parameter('ki_gnd', 0.1)
        self.declare_parameter('max_torque_ff_nm', 5.0)

        # RS00 sole leveling (brake mode)
        self.declare_parameter('kp_rs00_level', 0.05)
        self.declare_parameter('max_rs00_level_rad', 0.5)

        config_path        = self.get_parameter('config_path').value
        mode_topic         = self.get_parameter('mode_topic').value
        estop_topic        = self.get_parameter('estop_topic').value
        joint_cmds_topic   = self.get_parameter('joint_commands_topic').value
        cmd_vel_topic      = self.get_parameter('cmd_vel_topic').value
        publish_rate_hz    = self.get_parameter('publish_rate_hz').value
        self._verbose      = self.get_parameter('verbose').value
        auto_start         = self.get_parameter('auto_start').value
        auto_start_delay_s = self.get_parameter('auto_start_delay_s').value

        rs00_joints            = self.get_parameter('rs00_joints').value
        rs00_brake_offsets     = self.get_parameter('rs00_brake_offsets').value
        self._vel_threshold    = self.get_parameter('vel_stopped_threshold').value
        self._brake_debounce_s = self.get_parameter('brake_debounce_s').value

        self._target_ground_torque = self.get_parameter('target_ground_torque').value
        self._kp_gnd               = self.get_parameter('kp_gnd').value
        self._ki_gnd               = self.get_parameter('ki_gnd').value
        self._max_torque_ff        = self.get_parameter('max_torque_ff_nm').value
        self._kp_rs00_level        = self.get_parameter('kp_rs00_level').value
        self._max_rs00_level       = self.get_parameter('max_rs00_level_rad').value

        # ------------------------------------------------------------------ #
        # Load joint positions from YAML
        # ------------------------------------------------------------------ #
        self._joint_names, self._joint_positions = self._load_config(config_path)
        self.get_logger().info(
            f'Loaded {len(self._joint_names)} joints from {config_path}'
        )

        # Compute brake positions: tiptoe YAML pos + per-joint offset
        joints_map = dict(zip(self._joint_names, self._joint_positions))
        self._rs00_set = set(rs00_joints)
        self._brake_positions = {
            name: joints_map[name] + offset
            for name, offset in zip(rs00_joints, rs00_brake_offsets)
            if name in joints_map
        }
        for name, pos in self._brake_positions.items():
            tiptoe = joints_map[name]
            self.get_logger().info(
                f'Brake position: {name} = {pos:.4f} rad '
                f'(tiptoe {tiptoe:.4f} + {pos - tiptoe:.4f})'
            )

        # ------------------------------------------------------------------ #
        # State
        # ------------------------------------------------------------------ #
        self._mode    = 'off'
        self._estop   = False
        self._running = False

        self._current_positions: dict = {}
        self._current_efforts:   dict = {}
        self._trick_offsets:     dict = {}
        self._glitch_warned:     set  = set()

        # Ground-force PI
        self._ground_torque_ff = 0.0
        self._gnd_i_accum      = 0.0
        self._last_tick_time   = None

        # Brake-mode
        self._is_moving          = False  # start in brake mode until cmd_vel arrives
        self._stopped_since      = None
        self._rs00_level_offsets = {name: 0.0 for name in rs00_joints}

        # ------------------------------------------------------------------ #
        # Publishers
        # ------------------------------------------------------------------ #
        self._cmd_pub = self.create_publisher(JointState, joint_cmds_topic, 10)
        transient = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self._running_pub = self.create_publisher(Bool, '/leg_controller_running', transient)

        # ------------------------------------------------------------------ #
        # Subscriptions
        # ------------------------------------------------------------------ #
        self.create_subscription(String,     mode_topic,          self._on_mode,          10)
        self.create_subscription(Bool,       estop_topic,         self._on_estop,         10)
        self.create_subscription(JointState, '/joint_states',     self._on_joint_states,  10)
        self.create_subscription(JointState, '/trick_leg_offsets', self._on_trick_offsets, 10)
        best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10,
        )
        self.create_subscription(Twist,      cmd_vel_topic,       self._on_cmd_vel,       best_effort)

        # ------------------------------------------------------------------ #
        # Service clients
        # ------------------------------------------------------------------ #
        self._enable_client = self.create_client(Trigger, '/enable_motors')
        self._stop_client   = self.create_client(Trigger, '/stop_motors')

        # ------------------------------------------------------------------ #
        # Timers
        # ------------------------------------------------------------------ #
        period = 1.0 / publish_rate_hz
        self._publish_timer = self.create_timer(period, self._publish_commands)
        self._status_timer  = self.create_timer(5.0,   self._print_status)
        self.create_timer(0.5, self._publish_running_state)

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
        for i, name in enumerate(msg.name):
            self._current_positions[name] = msg.position[i]
            if i < len(msg.effort):
                self._current_efforts[name] = msg.effort[i]

    def _on_trick_offsets(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._trick_offsets[name] = pos

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

    def _on_cmd_vel(self, msg: Twist):
        moving = (abs(msg.linear.x) > self._vel_threshold or
                  abs(msg.angular.z) > self._vel_threshold)
        if moving:
            if not self._is_moving:
                self.get_logger().info('Moving → roller mode (RS00 tiptoe)')
                for name in self._rs00_set:
                    self._rs00_level_offsets[name] = 0.0
            self._is_moving     = True
            self._stopped_since = None
        else:
            if self._stopped_since is None:
                self._stopped_since = self.get_clock().now()
            else:
                elapsed = (self.get_clock().now() - self._stopped_since).nanoseconds * 1e-9
                if elapsed >= self._brake_debounce_s and self._is_moving:
                    self._is_moving = False
                    self.get_logger().info('Stopped → brake mode (RS00 flat sole)')

    # ---------------------------------------------------------------------- #
    # State machine
    # ---------------------------------------------------------------------- #

    def _auto_start_callback(self):
        self._auto_start_timer.cancel()
        if not self._running:
            self.get_logger().info('auto_start: enabling leg controller.')
            self._transition_to_running()

    def _should_run(self) -> bool:
        return self._mode not in ('off', 'recovery') and not self._estop

    def _update_state(self):
        want_running = self._should_run()
        if want_running and not self._running:
            self._transition_to_running()
        elif not want_running and self._running:
            self._transition_to_stopped()

    def _log_movement_preview(self):
        log = self.get_logger()
        if not self._current_positions:
            log.warn('Movement preview: no /joint_states received yet — cannot compare.')
            return
        log.info('--- Movement preview (current → target) ---')
        max_delta, max_joint = 0.0, ''
        for name, target in zip(self._joint_names, self._joint_positions):
            if name not in self._current_positions:
                log.warn(f'  {name}: no current position in /joint_states')
                continue
            current = self._current_positions[name]
            delta   = abs(target - current)
            line    = f'  {name}: {current:+.4f} rad → {target:+.4f} rad  (Δ = {delta:.4f} rad)'
            if delta > 0.3:
                log.warn(line + '  <-- LARGE MOVE')
            else:
                log.info(line)
            if delta > max_delta:
                max_delta, max_joint = delta, name
        if max_joint:
            log.info(f'  Max movement: {max_delta:.4f} rad on {max_joint}')

    def _transition_to_running(self):
        self._log_movement_preview()
        self.get_logger().info('Enabling motors and starting position hold.')
        self._running = True
        self._call_trigger(self._enable_client, '/enable_motors')

    def _transition_to_stopped(self):
        self.get_logger().info('Stopping motors.')
        self._running          = False
        self._ground_torque_ff = 0.0
        self._gnd_i_accum      = 0.0
        self._last_tick_time   = None
        self._call_trigger(self._stop_client, '/stop_motors')

    # ---------------------------------------------------------------------- #
    # Ground-force PI
    # ---------------------------------------------------------------------- #

    def _update_ground_torque_pi(self, dt: float):
        efforts = [
            self._current_efforts[n]
            for n in self._joint_names
            if n.startswith('joint_rs04')
            and n not in ('joint_rs04_1', 'joint_rs04_4')
            and n in self._current_efforts
        ]
        if not efforts:
            return
        mean_eff = sum(efforts) / len(efforts)
        err = self._target_ground_torque - mean_eff
        self._gnd_i_accum += err * dt
        adjustment = self._kp_gnd * err + self._ki_gnd * self._gnd_i_accum
        self._ground_torque_ff = max(
            0.0,
            min(self._max_torque_ff, self._ground_torque_ff + adjustment)
        )

    # ---------------------------------------------------------------------- #
    # RS00 sole leveling (brake mode)
    # ---------------------------------------------------------------------- #

    def _update_rs00_leveling(self, dt: float):
        """Integrate RS00 torque toward zero so the flat sole lies flush on ground."""
        for name in self._rs00_set:
            torque = self._current_efforts.get(name, 0.0)
            self._rs00_level_offsets[name] -= self._kp_rs00_level * torque * dt
            self._rs00_level_offsets[name] = max(
                -self._max_rs00_level,
                min(self._max_rs00_level, self._rs00_level_offsets[name])
            )

    # ---------------------------------------------------------------------- #
    # Publish loop
    # ---------------------------------------------------------------------- #

    def _publish_commands(self):
        if not self._running:
            return

        now = self.get_clock().now()
        dt  = (
            min((now - self._last_tick_time).nanoseconds * 1e-9, 0.1)
            if self._last_tick_time else 0.02
        )
        self._last_tick_time = now

        self._update_ground_torque_pi(dt)
        if not self._is_moving:
            self._update_rs00_leveling(dt)

        in_trick = (self._mode == 'trick')
        positions, efforts = [], []

        for name, base in zip(self._joint_names, self._joint_positions):
            if name in self._rs00_set:
                if not self._is_moving:
                    target = (self._brake_positions.get(name, base)
                              + self._rs00_level_offsets.get(name, 0.0))
                else:
                    target = base
                efforts.append(0.0)
            else:
                target = base + self._trick_offsets.get(name, 0.0) if in_trick else base
                efforts.append(
                    self._ground_torque_ff
                    if name.startswith('joint_rs04')
                    and name not in ('joint_rs04_1', 'joint_rs04_4')
                    else 0.0
                )

            # Glitch guard: don't command more than π rad from current position
            current = self._current_positions.get(name)
            if current is not None and abs(target - current) > math.pi:
                if name not in self._glitch_warned:
                    self.get_logger().warn(
                        f'Joint {name}: target {target:.3f} rad is >{math.pi:.3f} rad '
                        f'from current {current:.3f} rad — holding position.'
                    )
                    self._glitch_warned.add(name)
                target = current
            else:
                self._glitch_warned.discard(name)

            positions.append(target)

        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name     = self._joint_names
        msg.position = positions
        msg.velocity = [0.0] * len(self._joint_names)
        msg.effort   = efforts
        self._cmd_pub.publish(msg)

    # ---------------------------------------------------------------------- #
    # Status print
    # ---------------------------------------------------------------------- #

    def _print_status(self):
        if not self._verbose:
            return
        state    = 'RUNNING' if self._running else 'STOPPED'
        in_trick = (self._mode == 'trick')
        mode_str = 'BRAKE' if not self._is_moving else 'ROLL'
        for name, base in zip(self._joint_names, self._joint_positions):
            if name in self._rs00_set:
                target = self._brake_positions.get(name, base) if not self._is_moving else base
            elif in_trick:
                target = base + self._trick_offsets.get(name, 0.0)
            else:
                target = base
            actual = self._current_positions.get(name)
            effort = self._current_efforts.get(name)
            if actual is not None:
                err     = target - actual
                eff_str = f'  eff={effort:+.2f}' if effort is not None else ''
                if name in self._rs00_set and not self._is_moving:
                    offset = self._rs00_level_offsets.get(name, 0.0)
                    eff_str += f'  level_offset={offset:+.4f}'
                print(
                    f'[DRIVE/{state}/{mode_str}]  {name}: target={target:+.4f}  '
                    f'actual={actual:+.4f}  err={err:+.4f}{eff_str}',
                    flush=True,
                )
            else:
                print(
                    f'[DRIVE/{state}/{mode_str}]  {name}: target={target:+.4f}  actual=no_data',
                    flush=True,
                )
        print(
            f'[DRIVE/{state}/{mode_str}]  ground_torque_ff={self._ground_torque_ff:.3f} Nm',
            flush=True,
        )

    def _publish_running_state(self):
        msg = Bool()
        msg.data = self._running
        self._running_pub.publish(msg)

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
