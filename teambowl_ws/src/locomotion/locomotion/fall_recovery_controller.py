#!/usr/bin/env python3
import enum
import math
import os
import yaml

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from robstride_can_interfaces.srv import SetGains


class Phase(enum.Enum):
    IDLE = 0
    GROUND_SETTLE = 1  # wait for robot to come to rest before kip-up
    EXTENDING = 2
    RETRACTING = 3     # single direct setpoint at full gain
    SETTLING = 4
    DRIVE_BACK = 5


# Joints that move during recovery: sign = direction of extension from YAML
_MOVING_JOINTS = {
    'joint_rs04_2': -1,
    'joint_rs04_3': +1,
    'joint_rs04_5': -1,
    'joint_rs04_6': +1,
}

_DEFAULT_KD = 15.0
_DEFAULT_KP = 80.0


class FallRecoveryController(Node):
    """
    Automatic fall recovery for the RS04-legged robot.

    Detects fallover via IMU roll and runs a kip-up manoeuvre:
      GROUND_SETTLE — wait for robot to come fully to rest
      EXTENDING     — slowly ramp legs ~30° outward (ramp, safe speed)
      RETRACTING    — single direct setpoint snap back (high Kp, low Kd)
      SETTLING      — ramp to exact YAML positions (default gains restored)
      DRIVE_BACK    — reverse 0.5 m to clear the fallen position
    """

    def __init__(self):
        super().__init__('fall_recovery_controller')

        share_dir = get_package_share_directory('locomotion')
        default_pos_path = os.path.join(share_dir, 'driving_leg_pos.yaml')

        self.declare_parameter('driving_leg_pos_path', default_pos_path)
        self.declare_parameter('pitch_trigger_rad', 0.45)
        self.declare_parameter('extend_rad', 0.524)
        self.declare_parameter('extend_time_s', 2.5)
        self.declare_parameter('retract_time_s', 0.5)
        self.declare_parameter('retract_undershoot_rad', 0.1)
        self.declare_parameter('recovery_kp', 200.0)
        self.declare_parameter('retract_kd', 2.0)
        self.declare_parameter('settle_time_s', 1.5)
        self.declare_parameter('control_rate_hz', 50.0)
        self.declare_parameter('mode_topic', '/robot_mode')
        self.declare_parameter('estop_topic', '/estop')
        self.declare_parameter('ground_settle_s', 1.5)
        self.declare_parameter('cooldown_s', 3.0)
        self.declare_parameter('drive_back_speed_m_s', 0.3)
        self.declare_parameter('drive_back_dist_m', 0.5)

        pos_path = self.get_parameter('driving_leg_pos_path').value
        self._tilt_trigger = self.get_parameter('pitch_trigger_rad').value
        self._extend_rad = self.get_parameter('extend_rad').value
        self._extend_time_s = self.get_parameter('extend_time_s').value
        self._retract_time_s = self.get_parameter('retract_time_s').value
        self._undershoot = self.get_parameter('retract_undershoot_rad').value
        self._recovery_kp = self.get_parameter('recovery_kp').value
        self._retract_kd = self.get_parameter('retract_kd').value
        self._settle_time_s = self.get_parameter('settle_time_s').value
        self._rate_hz = self.get_parameter('control_rate_hz').value
        mode_topic = self.get_parameter('mode_topic').value
        estop_topic = self.get_parameter('estop_topic').value
        ground_settle_s = self.get_parameter('ground_settle_s').value
        cooldown_s = self.get_parameter('cooldown_s').value
        drive_back_speed = self.get_parameter('drive_back_speed_m_s').value
        drive_back_dist = self.get_parameter('drive_back_dist_m').value

        self._yaml_pos = self._load_yaml_positions(pos_path)
        self._ground_settle_ticks = max(1, int(ground_settle_s * self._rate_hz))
        self._cooldown_ticks_reset = int(cooldown_s * self._rate_hz)
        self._drive_back_ticks = max(1, int(drive_back_dist / drive_back_speed * self._rate_hz))
        self._drive_back_speed = drive_back_speed

        # State
        self._phase = Phase.IDLE
        self._phase_tick = 0
        self._cooldown_ticks = 0
        self._tilt = 0.0
        self._current_pos: dict = {}
        self._extend_start: dict = {}  # snapshot at end of ground settle

        # Publishers
        self._joint_pub = self.create_publisher(JointState, '/joint_commands', 10)
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self._mode_set_pub = self.create_publisher(String, '/robot_mode_set', 10)
        self._estop_pub = self.create_publisher(Bool, estop_topic, 10)
        self._clear_estop_pub = self.create_publisher(Bool, '/clear_estop', 10)

        # Subscriptions
        self.create_subscription(Imu, '/imu/data', self._imu_cb, 10)
        self.create_subscription(JointState, '/joint_states', self._joint_states_cb, 10)
        self.create_subscription(String, mode_topic, self._mode_cb, 10)

        # Service clients
        self._enable_client = self.create_client(Trigger, '/enable_motors')
        self._set_gains_client = self.create_client(SetGains, '/set_gains')

        self._timer = self.create_timer(1.0 / self._rate_hz, self._tick)

        self.get_logger().info(
            f'FallRecoveryController ready. trigger={self._tilt_trigger:.3f} rad, '
            f'extend={math.degrees(self._extend_rad):.1f}°, '
            f'recovery_kp={self._recovery_kp}, ground_settle={ground_settle_s:.1f}s'
        )

    def _load_yaml_positions(self, path: str) -> dict:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return {k: float(v) for k, v in data.get('joints', {}).items()}

    def _imu_cb(self, msg: Imu):
        q = msg.orientation
        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        self._tilt = math.atan2(sinr, cosr)

    def _joint_states_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self._current_pos[name] = pos

    def _mode_cb(self, msg: String):
        pass  # monitor-only; recovery is phase-driven, not mode-driven

    def _set_mode(self, mode: str):
        msg = String()
        msg.data = mode
        self._mode_set_pub.publish(msg)

    def _set_gains_async(self, kp: float, kd: float):
        for joint_name in _MOVING_JOINTS:
            req = SetGains.Request()
            req.joint_name = joint_name
            req.kp = kp
            req.kd = kd
            self._set_gains_client.call_async(req)

    def _publish_joints(self, positions: dict):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        for name, pos in positions.items():
            msg.name.append(name)
            msg.position.append(float(pos))
            msg.velocity.append(0.0)
            msg.effort.append(0.0)
        self._joint_pub.publish(msg)

    @staticmethod
    def _interp(start: float, end: float, frac: float) -> float:
        return start + (end - start) * max(0.0, min(1.0, frac))

    def _tick(self):
        if self._phase == Phase.IDLE:
            if self._cooldown_ticks > 0:
                self._cooldown_ticks -= 1
                return
            if abs(self._tilt) > self._tilt_trigger:
                self._start_recovery()
            return

        if self._phase == Phase.DRIVE_BACK:
            self._tick_drive_back()
            return

        # All leg phases lock the wheels
        self._cmd_vel_pub.publish(Twist())

        if self._phase == Phase.GROUND_SETTLE:
            self._tick_ground_settle()
        elif self._phase == Phase.EXTENDING:
            self._tick_extending()
        elif self._phase == Phase.RETRACTING:
            self._tick_retracting()
        elif self._phase == Phase.SETTLING:
            self._tick_settling()

    def _start_recovery(self):
        self.get_logger().warn(
            f'Fall detected (roll={math.degrees(self._tilt):.1f}°) — waiting for robot to settle.')
        self._phase = Phase.GROUND_SETTLE
        self._phase_tick = 0

        clear_msg = Bool()
        clear_msg.data = True
        self._clear_estop_pub.publish(clear_msg)
        estop_msg = Bool()
        estop_msg.data = False
        self._estop_pub.publish(estop_msg)
        self._set_mode('recovery')

        if self._enable_client.service_is_ready():
            self._enable_client.call_async(Trigger.Request())
        else:
            self.get_logger().warn('enable_motors not ready at recovery start.')

    def _tick_ground_settle(self):
        self._phase_tick += 1
        if self._phase_tick >= self._ground_settle_ticks:
            self._extend_start = dict(self._current_pos)
            self._set_gains_async(self._recovery_kp, _DEFAULT_KD)
            self._phase = Phase.EXTENDING
            self._phase_tick = 0
            self.get_logger().info(
                'Ground settle done — EXTENDING (Kp boosted to %.0f).' % self._recovery_kp)

    def _tick_extending(self):
        total_ticks = max(1, int(self._extend_time_s * self._rate_hz))
        frac = self._phase_tick / total_ticks

        cmds = {}
        for name, yaml_pos in self._yaml_pos.items():
            if name in _MOVING_JOINTS:
                sign = _MOVING_JOINTS[name]
                extend_target = yaml_pos + sign * self._extend_rad
                start = self._extend_start.get(name, yaml_pos)
                cmds[name] = self._interp(start, extend_target, frac)
            else:
                cmds[name] = yaml_pos

        self._publish_joints(cmds)
        self._phase_tick += 1

        if self._phase_tick >= total_ticks:
            self._set_gains_async(self._recovery_kp, self._retract_kd)
            self._phase = Phase.RETRACTING
            self._phase_tick = 0
            self.get_logger().info(
                'EXTENDING done — RETRACTING (Kd reduced to %.1f).' % self._retract_kd)

    def _tick_retracting(self):
        # Single direct setpoint — motor snaps there at full speed
        cmds = {}
        for name, yaml_pos in self._yaml_pos.items():
            if name in _MOVING_JOINTS:
                sign = _MOVING_JOINTS[name]
                cmds[name] = yaml_pos + sign * self._undershoot
            else:
                cmds[name] = yaml_pos

        self._publish_joints(cmds)
        self._phase_tick += 1

        if self._phase_tick >= max(1, int(self._retract_time_s * self._rate_hz)):
            self._set_gains_async(_DEFAULT_KP, _DEFAULT_KD)
            self._phase = Phase.SETTLING
            self._phase_tick = 0
            self.get_logger().info('RETRACTING done — SETTLING (gains restored).')

    def _tick_settling(self):
        total_ticks = max(1, int(self._settle_time_s * self._rate_hz))
        frac = self._phase_tick / total_ticks

        cmds = {}
        for name, yaml_pos in self._yaml_pos.items():
            if name in _MOVING_JOINTS:
                sign = _MOVING_JOINTS[name]
                retract_target = yaml_pos + sign * self._undershoot
                cmds[name] = self._interp(retract_target, yaml_pos, frac)
            else:
                cmds[name] = yaml_pos

        self._publish_joints(cmds)
        self._phase_tick += 1

        if self._phase_tick >= total_ticks:
            self._phase = Phase.DRIVE_BACK
            self._phase_tick = 0
            self.get_logger().info('SETTLING done — driving back %.2f m.' % (
                self._drive_back_speed * self._drive_back_ticks / self._rate_hz))

    def _tick_drive_back(self):
        twist = Twist()
        twist.linear.x = -self._drive_back_speed
        self._cmd_vel_pub.publish(twist)
        self._phase_tick += 1

        if self._phase_tick >= self._drive_back_ticks:
            self._cmd_vel_pub.publish(Twist())  # stop
            self._phase = Phase.IDLE
            self._phase_tick = 0
            self._cooldown_ticks = self._cooldown_ticks_reset
            self._set_mode('driving')
            self.get_logger().info('Recovery complete — back to IDLE, mode → driving.')


def main():
    rclpy.init()
    node = FallRecoveryController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
