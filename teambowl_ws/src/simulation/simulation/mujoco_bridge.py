#!/usr/bin/env python3
"""
MuJoCo Bridge Node
==================
Runs the teambowl_mjlab.xml simulation at 500 Hz in a background thread
and bridges it to ROS2 topics so the real controllers (balance_controller,
driving_controller, vel_cmd_mux, etc.) see a live robot without any hardware.

Topics published
----------------
  /imu/data           sensor_msgs/Imu         100 Hz — orientation, angular_vel, lin_accel
  /odometry/filtered  nav_msgs/Odometry       100 Hz — pose + twist from ground-truth sensors
  /joint_states       sensor_msgs/JointState   50 Hz — wheel joint velocities

Topics subscribed
-----------------
  /cmd_vel            geometry_msgs/Twist — converted to motor velocity commands
  /estop              std_msgs/Bool       — zeros ctrl when True
  /robot_mode         std_msgs/String     — zeros ctrl when "off"

Services
--------
  /sim_reset          std_srvs/srv/Trigger — reset to upright spawn pose

Physics thread
--------------
Runs at sim_hz (500 Hz) in a daemon thread. Publishes IMU + odom every
pub_imu_decimation steps (default 5 → 100 Hz) and joint_states every
pub_joint_decimation steps (default 10 → 50 Hz).

cmd_vel → motor control
-----------------------
  v_left  = vx - omega * track_width / 2
  v_right = vx + omega * track_width / 2
  omega_wheel = v_wheel / wheel_radius
  ctrl = clip(ctrl_sign * gear_ratio * omega_wheel, -ctrl_max, +ctrl_max)

  ctrl_sign=-1.0 by default (gear flips direction; see XML equality constraints).
  Flip to +1.0 if the robot drives backward on first test.

Frame convention (MuJoCo world frame)
--------------------------------------
  Wheels rotate about the X axis → robot drives along Y.
  Lean forward/back = rotation about X.
  IMU angular_velocity.x is the pitch rate seen by balance_controller.
  If balance_controller was tuned expecting .y as pitch rate, set
  imu_pitch_axis: 1 (0=x, 1=y, 2=z) in the YAML to swap axes.
"""

import math
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
)

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

try:
    import mujoco
except ImportError as exc:
    raise ImportError(
        "mujoco Python package not found. "
        "Install it with: pip3 install mujoco"
    ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sensor_slice(model, name):
    """Return (adr, adr+dim) for a named sensor."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    return adr, adr + dim


def _wrap_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class MujocoBridgeNode(Node):

    def __init__(self):
        super().__init__('mujoco_bridge')

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter('model_path',
                               '/workspaces/teambowl_mjlab/teambowl_mjlab.xml')
        self.declare_parameter('sim_hz', 500.0)
        self.declare_parameter('pub_imu_decimation', 5)    # 500/5 = 100 Hz
        self.declare_parameter('pub_joint_decimation', 10) # 500/10 = 50 Hz
        self.declare_parameter('imu_frame_id', 'imu_link')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('wheel_radius', 0.154)
        self.declare_parameter('track_width', 0.5588)
        self.declare_parameter('gear_ratio_left', 145.0 / 12.0)
        self.declare_parameter('gear_ratio_right', 174.0 / 12.0)
        self.declare_parameter('ctrl_sign_left', -1.0)
        self.declare_parameter('ctrl_sign_right', -1.0)
        self.declare_parameter('ctrl_max_left', 40.0)
        self.declare_parameter('ctrl_max_right', 48.0)
        # z-position of Frame body at spawn (wheels touching floor at z=-0.3).
        # floor_z=-0.3, wheel_radius=0.154, wheel_local_z=0.056195 → -0.0898
        self.declare_parameter('spawn_z', -0.090)

        model_path = self.get_parameter('model_path').value
        self._sim_hz = float(self.get_parameter('sim_hz').value)
        self._imu_dec = int(self.get_parameter('pub_imu_decimation').value)
        self._jnt_dec = int(self.get_parameter('pub_joint_decimation').value)
        self._imu_frame = self.get_parameter('imu_frame_id').value
        self._odom_frame = self.get_parameter('odom_frame_id').value
        self._base_frame = self.get_parameter('base_frame_id').value

        self._wheel_radius = float(self.get_parameter('wheel_radius').value)
        self._track_width  = float(self.get_parameter('track_width').value)
        self._gear_left    = float(self.get_parameter('gear_ratio_left').value)
        self._gear_right   = float(self.get_parameter('gear_ratio_right').value)
        self._sign_left    = float(self.get_parameter('ctrl_sign_left').value)
        self._sign_right   = float(self.get_parameter('ctrl_sign_right').value)
        self._max_left     = float(self.get_parameter('ctrl_max_left').value)
        self._max_right    = float(self.get_parameter('ctrl_max_right').value)
        self._spawn_z      = float(self.get_parameter('spawn_z').value)

        # ------------------------------------------------------------------
        # Load MuJoCo model
        # ------------------------------------------------------------------
        self.get_logger().info(f'Loading MuJoCo model: {model_path}')
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data  = mujoco.MjData(self._model)
        self._reset_to_spawn()

        # Pre-compute sensor address slices (avoids per-step name lookups)
        self._s_gyro    = _sensor_slice(self._model, 'imu_gyro')
        self._s_accel   = _sensor_slice(self._model, 'imu_accel')
        self._s_pos     = _sensor_slice(self._model, 'gt_pos')
        self._s_quat    = _sensor_slice(self._model, 'gt_quat')
        self._s_linvel  = _sensor_slice(self._model, 'gt_linvel')
        self._s_angvel  = _sensor_slice(self._model, 'gt_angvel')
        self._s_lwvel   = _sensor_slice(self._model, 'left_wheel_vel')
        self._s_rwvel   = _sensor_slice(self._model, 'right_wheel_vel')

        self.get_logger().info(
            f'Model loaded: nq={self._model.nq} nv={self._model.nv} '
            f'nu={self._model.nu} neq={self._model.neq}'
        )

        # ------------------------------------------------------------------
        # Shared state (physics thread ↔ ROS callbacks)
        # ------------------------------------------------------------------
        self._lock      = threading.Lock()
        self._ctrl      = np.zeros(self._model.nu, dtype=np.float64)
        self._estop     = False
        self._mode      = 'off'
        self._reset_req = False

        # ------------------------------------------------------------------
        # Publishers
        # ------------------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        latching_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self._pub_imu   = self.create_publisher(Imu,        '/imu/data',          sensor_qos)
        self._pub_odom  = self.create_publisher(Odometry,   '/odometry/filtered', sensor_qos)
        self._pub_joint = self.create_publisher(JointState, '/joint_states',      sensor_qos)

        # ------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------
        self.create_subscription(Twist,  '/cmd_vel',    self._on_cmd_vel,   10)
        self.create_subscription(Bool,   '/estop',      self._on_estop,     10)
        self.create_subscription(String, '/robot_mode', self._on_mode,      latching_qos)

        # ------------------------------------------------------------------
        # Services
        # ------------------------------------------------------------------
        self.create_service(Trigger, '/sim_reset', self._on_reset)

        # ------------------------------------------------------------------
        # Start physics thread
        # ------------------------------------------------------------------
        self._running = True
        self._phys_thread = threading.Thread(
            target=self._physics_loop, daemon=True, name='mujoco_physics'
        )
        self._phys_thread.start()
        self.get_logger().info(
            f'Simulation running at {self._sim_hz:.0f} Hz. '
            f'IMU/odom @ {self._sim_hz/self._imu_dec:.0f} Hz, '
            f'joint_states @ {self._sim_hz/self._jnt_dec:.0f} Hz.'
        )

    # ------------------------------------------------------------------
    # Destructor
    # ------------------------------------------------------------------

    def destroy_node(self):
        self._running = False
        self._phys_thread.join(timeout=2.0)
        super().destroy_node()

    # ------------------------------------------------------------------
    # ROS callbacks (called from executor thread)
    # ------------------------------------------------------------------

    def _on_cmd_vel(self, msg: Twist):
        vx    = msg.linear.x
        omega = msg.angular.z

        v_left  = vx - omega * self._track_width / 2.0
        v_right = vx + omega * self._track_width / 2.0

        ow_left  = v_left  / self._wheel_radius
        ow_right = v_right / self._wheel_radius

        ctrl_l = float(np.clip(
            self._sign_left  * self._gear_left  * ow_left,
            -self._max_left,  self._max_left
        ))
        ctrl_r = float(np.clip(
            self._sign_right * self._gear_right * ow_right,
            -self._max_right, self._max_right
        ))

        with self._lock:
            self._ctrl[0] = ctrl_l
            self._ctrl[1] = ctrl_r

    def _on_estop(self, msg: Bool):
        with self._lock:
            self._estop = msg.data
            if self._estop:
                self._ctrl[:] = 0.0

    def _on_mode(self, msg: String):
        with self._lock:
            self._mode = msg.data
            if self._mode == 'off':
                self._ctrl[:] = 0.0

    def _on_reset(self, request, response):
        with self._lock:
            self._reset_req = True
        # Wait for physics thread to execute the reset
        for _ in range(100):
            time.sleep(0.01)
            with self._lock:
                if not self._reset_req:
                    break
        response.success = True
        response.message = f'Sim reset to spawn pose (z={self._spawn_z:.3f})'
        self.get_logger().info('Simulation reset to spawn pose.')
        return response

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_to_spawn(self):
        """Reset simulation to upright spawn pose."""
        mujoco.mj_resetData(self._model, self._data)
        # Free joint qpos layout: [x, y, z, qw, qx, qy, qz, ...]
        self._data.qpos[0] = 0.0           # x
        self._data.qpos[1] = 0.0           # y
        self._data.qpos[2] = self._spawn_z # z (wheels touching floor)
        self._data.qpos[3] = 1.0           # qw (identity → upright)
        self._data.qpos[4] = 0.0           # qx
        self._data.qpos[5] = 0.0           # qy
        self._data.qpos[6] = 0.0           # qz
        mujoco.mj_forward(self._model, self._data)

    # ------------------------------------------------------------------
    # Physics loop (background thread at sim_hz)
    # ------------------------------------------------------------------

    def _physics_loop(self):
        dt      = 1.0 / self._sim_hz
        step    = 0
        t_next  = time.monotonic()

        while self._running:
            t_now = time.monotonic()
            sleep_s = t_next - t_now
            if sleep_s > 0:
                time.sleep(sleep_s)
            t_next += dt

            with self._lock:
                # Handle reset request
                if self._reset_req:
                    self._reset_to_spawn()
                    self._ctrl[:] = 0.0
                    self._reset_req = False
                    step = 0

                # Apply ctrl
                if self._estop or self._mode == 'off':
                    self._data.ctrl[:] = 0.0
                else:
                    self._data.ctrl[:] = self._ctrl

                # Step physics
                mujoco.mj_step(self._model, self._data)

                # Snapshot sensor data for publishing (copy to avoid races)
                sd   = self._data.sensordata.copy()
                time_sec = self._data.time

            step += 1

            # Publish IMU + odometry at imu decimation rate
            if step % self._imu_dec == 0:
                self._publish_imu(sd, time_sec)
                self._publish_odom(sd, time_sec)

            # Publish joint_states at joint decimation rate
            if step % self._jnt_dec == 0:
                self._publish_joint_states(sd, time_sec)

    # ------------------------------------------------------------------
    # Publishers (called from physics thread — rclpy publish is thread-safe)
    # ------------------------------------------------------------------

    def _ros_stamp(self, sim_time_sec: float):
        """Convert sim time (float s) to ROS2 Time stamp."""
        sec  = int(sim_time_sec)
        nsec = int((sim_time_sec - sec) * 1e9)
        t = rclpy.time.Time(seconds=sec, nanoseconds=nsec)
        return t.to_msg()

    def _publish_imu(self, sd: np.ndarray, sim_time: float):
        msg = Imu()
        msg.header.stamp    = self._ros_stamp(sim_time)
        msg.header.frame_id = self._imu_frame

        # Orientation: gt_quat sensor outputs (w, x, y, z)
        qw, qx, qy, qz = sd[self._s_quat[0]:self._s_quat[1]]
        msg.orientation.w = float(qw)
        msg.orientation.x = float(qx)
        msg.orientation.y = float(qy)
        msg.orientation.z = float(qz)
        # Unknown orientation covariance
        msg.orientation_covariance[0] = -1.0

        # Angular velocity (body frame from gyro sensor)
        gx, gy, gz = sd[self._s_gyro[0]:self._s_gyro[1]]
        msg.angular_velocity.x = float(gx)
        msg.angular_velocity.y = float(gy)
        msg.angular_velocity.z = float(gz)
        msg.angular_velocity_covariance[0] = -1.0

        # Linear acceleration (body frame from accelerometer)
        ax, ay, az = sd[self._s_accel[0]:self._s_accel[1]]
        msg.linear_acceleration.x = float(ax)
        msg.linear_acceleration.y = float(ay)
        msg.linear_acceleration.z = float(az)
        msg.linear_acceleration_covariance[0] = -1.0

        self._pub_imu.publish(msg)

    def _publish_odom(self, sd: np.ndarray, sim_time: float):
        msg = Odometry()
        msg.header.stamp    = self._ros_stamp(sim_time)
        msg.header.frame_id = self._odom_frame
        msg.child_frame_id  = self._base_frame

        # Pose from ground-truth sensors
        px, py, pz = sd[self._s_pos[0]:self._s_pos[1]]
        msg.pose.pose.position.x = float(px)
        msg.pose.pose.position.y = float(py)
        msg.pose.pose.position.z = float(pz)

        qw, qx, qy, qz = sd[self._s_quat[0]:self._s_quat[1]]
        msg.pose.pose.orientation.w = float(qw)
        msg.pose.pose.orientation.x = float(qx)
        msg.pose.pose.orientation.y = float(qy)
        msg.pose.pose.orientation.z = float(qz)
        msg.pose.covariance[0] = -1.0  # unknown

        # Twist from ground-truth linear + angular velocity
        lvx, lvy, lvz = sd[self._s_linvel[0]:self._s_linvel[1]]
        msg.twist.twist.linear.x = float(lvx)
        msg.twist.twist.linear.y = float(lvy)
        msg.twist.twist.linear.z = float(lvz)

        avx, avy, avz = sd[self._s_angvel[0]:self._s_angvel[1]]
        msg.twist.twist.angular.x = float(avx)
        msg.twist.twist.angular.y = float(avy)
        msg.twist.twist.angular.z = float(avz)
        msg.twist.covariance[0] = -1.0  # unknown

        self._pub_odom.publish(msg)

    def _publish_joint_states(self, sd: np.ndarray, sim_time: float):
        msg = JointState()
        msg.header.stamp = self._ros_stamp(sim_time)

        lw_vel = float(sd[self._s_lwvel[0]])
        rw_vel = float(sd[self._s_rwvel[0]])

        msg.name     = ['left_wheel_0',  'right_wheel_0']
        msg.position = [0.0, 0.0]   # position not tracked (not needed for control)
        msg.velocity = [lw_vel, rw_vel]
        msg.effort   = [0.0, 0.0]

        self._pub_joint.publish(msg)


# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = MujocoBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
