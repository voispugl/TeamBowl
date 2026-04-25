"""
Isaac Sim scene setup script for TeamBowl robot simulation.

Run automatically by entrypoint_isaac_sim.sh via:
    /isaac-sim/runapp.sh --exec setup_scene.py

Responsibilities:
  - Import bowl.urdf with convex-hull collision meshes (full STL visuals)
  - Configure ROS2 bridge: IMU, odometry, joint states, RGB + depth cameras
  - Publish /visual_slam/tracking/odometry from ground-truth pose (Jetson VPI substitute)
  - Subscribe /cmd_vel → articulation controller
  - Place pre-built test course (6-8 obstacles + static human mesh)
  - Enable WebRTC streaming on port 8211

URDF joint name note: Isaac Sim sanitizes USD prim names (spaces → underscores).
Joint names in /joint_states use the original URDF names for ROS2 compatibility.
"""

import asyncio
import math
import os

import carb
import numpy as np
import omni
import omni.kit.app
from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrim, GeometryPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.urdf import _urdf

# ── ROS2 bridge imports ────────────────────────────────────────────────────────
from omni.isaac.ros2_bridge import SimulationContext
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState, Image, CameraInfo
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped

# ── Paths ──────────────────────────────────────────────────────────────────────
WS = "/workspaces/teambowl_ws"
URDF_PATH = os.path.join(WS, "src/bringup/robot_description/bowl.urdf")
MESH_DIR  = os.path.join(WS, "src/bringup/robot_description/meshes")

# Physics: match MuJoCo sim (500 Hz)
PHYSICS_HZ = 500.0
PHYSICS_DT = 1.0 / PHYSICS_HZ

# Camera simulation rate
CAMERA_HZ = 30.0

# Robot spawn position
SPAWN_POS = (0.0, 0.0, 0.35)   # z slightly above floor

# ── Test course obstacle layout ────────────────────────────────────────────────
# Pre-built navigation course: 6 boxes + 1 wall + 1 human placeholder.
# Coordinate frame: x-forward, y-left, z-up from spawn point.
OBSTACLES = [
    {"pos": ( 2.0,  0.0, 0.25), "size": (0.50, 0.50, 0.50)},  # center blocker
    {"pos": ( 0.0,  1.5, 0.50), "size": (0.30, 1.50, 1.00)},  # left wall segment
    {"pos": ( 3.0, -1.0, 0.30), "size": (0.60, 0.60, 0.60)},  # right obstacle
    {"pos": (-1.5,  2.0, 0.40), "size": (0.40, 0.40, 0.80)},  # rear-left obstacle
    {"pos": ( 1.0, -2.5, 0.30), "size": (1.00, 0.30, 0.60)},  # wall fragment
    {"pos": ( 4.0,  1.0, 0.25), "size": (0.50, 0.50, 0.50)},  # far right
    {"pos": (-2.0, -1.0, 0.50), "size": (0.30, 0.30, 1.00)},  # narrow pillar
]
# Human mesh position — ~3 m in front of spawn, facing origin (for YOLO26 testing)
HUMAN_POS = (3.0, 0.0, 0.0)


class TeamBowlSimulation:
    """Manages the Isaac Sim scene, physics loop, and ROS2 bridge."""

    def __init__(self):
        self._world = World(physics_dt=PHYSICS_DT, rendering_dt=1.0 / 30.0)
        self._robot_prim_path = "/World/TeamBowl"
        self._articulation = None
        self._cmd_vel = (0.0, 0.0)       # (linear_x, angular_z)
        self._estop = False
        self._mode = "off"
        self._step_count = 0

        # ROS2 node (one node for all topics)
        rclpy.init()
        self._node = rclpy.create_node("isaac_sim_bridge")
        self._setup_ros2()

    def _setup_ros2(self):
        n = self._node

        # Publishers
        # /imu/data          — simulated IMU → EKF imu0
        # /wheel/odometry    — simulated wheel encoders → EKF odom0
        # /visual_slam/…     — ground-truth pose as desktop VSLAM substitute → EKF odom1
        # /odometry/filtered is NOT published here — the ekf_filter_node produces it
        self._imu_pub    = n.create_publisher(Imu,        "/imu/data",                      10)
        self._wheel_pub  = n.create_publisher(Odometry,   "/wheel/odometry",                10)
        self._jstate_pub = n.create_publisher(JointState, "/joint_states",                  10)
        self._vslam_pub  = n.create_publisher(Odometry,   "/visual_slam/tracking/odometry", 10)
        self._rgb_pub    = n.create_publisher(Image,      "/oak/rgb/image_raw",             10)
        self._rgb_info_pub = n.create_publisher(CameraInfo, "/oak/rgb/camera_info",         10)
        self._depth_pub  = n.create_publisher(Image,      "/oak/stereo/image_raw",          10)
        self._depth_info_pub = n.create_publisher(CameraInfo, "/oak/stereo/camera_info",    10)

        # Subscribers
        n.create_subscription(Twist,  "/cmd_vel",    self._on_cmd_vel,   10)
        n.create_subscription(Bool,   "/estop",      self._on_estop,     10)
        n.create_subscription(String, "/robot_mode", self._on_mode,      10)

        # Publish at 50 Hz (every 10 physics steps)
        n.create_timer(1.0 / 50.0, self._spin_ros2)

    def _on_cmd_vel(self, msg: Twist):
        if self._estop or self._mode == "off":
            self._cmd_vel = (0.0, 0.0)
        else:
            self._cmd_vel = (msg.linear.x, msg.angular.z)

    def _on_estop(self, msg: Bool):
        self._estop = msg.data
        if self._estop:
            self._cmd_vel = (0.0, 0.0)

    def _on_mode(self, msg: String):
        self._mode = msg.data
        if self._mode == "off":
            self._cmd_vel = (0.0, 0.0)

    def _spin_ros2(self):
        rclpy.spin_once(self._node, timeout_sec=0.0)

    def setup_scene(self):
        """Load URDF, place obstacles, add cameras. Called before simulation loop."""
        world = self._world
        world.scene.add_default_ground_plane()

        # Import robot URDF
        carb.log_info(f"[TeamBowl] Importing URDF: {URDF_PATH}")
        urdf_interface = _urdf.acquire_urdf_interface()
        import_config = _urdf.ImportConfig()
        import_config.merge_fixed_joints = False    # keep all 23 joints
        import_config.convex_decomp = True          # simplified collision, full visual
        import_config.fix_base = False              # robot is free-standing
        import_config.import_inertia_tensor = True
        import_config.default_drive_strength = 1e5
        import_config.default_position_drive_damping = 1e3
        import_config.distance_scale = 1.0

        robot_path = urdf_interface.import_robot(URDF_PATH, import_config)
        if not robot_path:
            carb.log_error("[TeamBowl] URDF import failed — check URDF_PATH and mesh files")
            return False

        # Move robot to spawn position
        from omni.isaac.core.utils.transformations import set_transform_helper
        prim = self._world.stage.GetPrimAtPath(robot_path)
        from pxr import UsdGeom, Gf
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*SPAWN_POS))

        # Articulation (for reading joint states)
        from omni.isaac.core.articulations import Articulation
        self._articulation = world.scene.add(
            Articulation(prim_path=robot_path, name="teambowl"))

        # Obstacles — static boxes
        for i, obs in enumerate(OBSTACLES):
            from omni.isaac.core.objects import FixedCuboid
            world.scene.add(FixedCuboid(
                prim_path=f"/World/Obstacle_{i}",
                name=f"obstacle_{i}",
                position=np.array(obs["pos"]),
                scale=np.array(obs["size"]),
            ))
            carb.log_info(f"[TeamBowl] Placed obstacle_{i} at {obs['pos']}")

        # Human placeholder mesh — try Isaac Nucleus People asset, fall back to simple box
        human_placed = self._place_human()
        if not human_placed:
            carb.log_warn("[TeamBowl] Human mesh unavailable — placing box placeholder")
            from omni.isaac.core.objects import FixedCuboid
            world.scene.add(FixedCuboid(
                prim_path="/World/HumanPlaceholder",
                name="human_placeholder",
                position=np.array([*HUMAN_POS[:2], 0.9]),
                scale=np.array([0.5, 0.4, 1.8]),
            ))

        # Camera sensor at OAK-D position (rgb_cam_0 frame from URDF)
        self._setup_cameras(robot_path)

        carb.log_info("[TeamBowl] Scene setup complete.")
        return True

    def _place_human(self) -> bool:
        """Try to place a human mesh from Isaac Nucleus. Returns True on success."""
        try:
            assets_root = get_assets_root_path()
            if assets_root is None:
                return False
            # Isaac Sim 4.x Nucleus path for a standing human
            human_usd = f"{assets_root}/Isaac/People/Characters/standing_person/standing_person.usd"
            add_reference_to_stage(human_usd, "/World/Human")
            from pxr import UsdGeom, Gf
            prim = self._world.stage.GetPrimAtPath("/World/Human")
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*HUMAN_POS))
            carb.log_info(f"[TeamBowl] Placed human mesh at {HUMAN_POS}")
            return True
        except Exception as e:
            carb.log_warn(f"[TeamBowl] Could not load human mesh: {e}")
            return False

    def _setup_cameras(self, robot_path: str):
        """Add RGB and depth camera sensors at the OAK-D position."""
        # OAK-D position relative to robot: from URDF rgb_cam_0 joint (~0.15 m forward, ~0.3 m up)
        # These will be attached to the Frame link which is the robot body.
        try:
            from omni.isaac.sensor import Camera
            from omni.isaac.core.utils.stage import get_current_stage
            stage = get_current_stage()

            # RGB camera — 720p, 30 Hz, matches oak_cam.yaml
            self._rgb_camera = Camera(
                prim_path=f"{robot_path}/Frame/oak_rgb_camera",
                name="oak_rgb",
                frequency=CAMERA_HZ,
                resolution=(1280, 720),
            )
            self._world.scene.add(self._rgb_camera)

            # Depth camera — matches stereo depth output
            self._depth_camera = Camera(
                prim_path=f"{robot_path}/Frame/oak_depth_camera",
                name="oak_depth",
                frequency=CAMERA_HZ,
                resolution=(640, 400),
            )
            self._world.scene.add(self._depth_camera)

            # Start cameras
            self._rgb_camera.initialize()
            self._depth_camera.initialize()
            carb.log_info("[TeamBowl] Cameras initialized.")
        except Exception as e:
            carb.log_warn(f"[TeamBowl] Camera setup failed: {e}")
            self._rgb_camera = None
            self._depth_camera = None

    def _publish_sensors(self):
        """Called every physics step — publishes ROS2 sensor topics."""
        now = self._node.get_clock().now().to_msg()
        art = self._articulation

        if art is None:
            return

        # Robot pose from Isaac Sim ground truth
        pos  = art.get_world_pose()[0]   # (x, y, z)
        quat = art.get_world_pose()[1]   # (w, x, y, z)
        vel  = art.get_linear_velocity()
        ang  = art.get_angular_velocity()

        # ── IMU ───────────────────────────────────────────────────────────────
        if self._step_count % 5 == 0:   # 100 Hz (500 Hz / 5)
            imu = Imu()
            imu.header.stamp = now
            imu.header.frame_id = "imu_link"
            imu.orientation.w, imu.orientation.x = float(quat[0]), float(quat[1])
            imu.orientation.y, imu.orientation.z = float(quat[2]), float(quat[3])
            imu.angular_velocity.x = float(ang[0])
            imu.angular_velocity.y = float(ang[1])
            imu.angular_velocity.z = float(ang[2])
            imu.linear_acceleration.x = 0.0  # simplified — no gravity vector
            imu.linear_acceleration.y = 0.0
            imu.linear_acceleration.z = 0.0
            self._imu_pub.publish(imu)

        # ── Wheel odometry (simulated encoders → EKF odom0) ──────────────────
        # Publishes velocity-only odometry (no absolute position) matching
        # what diff_drive_odom produces from VESC encoders on real hardware.
        if self._step_count % 5 == 0:   # 100 Hz
            wheel_odom = Odometry()
            wheel_odom.header.stamp = now
            wheel_odom.header.frame_id = "odom"
            wheel_odom.child_frame_id  = "base_link"
            wheel_odom.twist.twist.linear.x  = float(vel[0])
            wheel_odom.twist.twist.angular.z = float(ang[2])
            self._wheel_pub.publish(wheel_odom)

        # ── VSLAM substitute (ground-truth pose → EKF odom1) ─────────────────
        # Published as /visual_slam/tracking/odometry so the EKF odom1 config
        # (which expects this topic) gets full 6-DOF pose from Isaac Sim.
        # On the Jetson this topic comes from isaac_ros_visual_slam VIO.
        if self._step_count % 5 == 0:
            gt_odom = Odometry()
            gt_odom.header.stamp = now
            gt_odom.header.frame_id = "odom"
            gt_odom.child_frame_id  = "base_link"
            gt_odom.pose.pose.position.x = float(pos[0])
            gt_odom.pose.pose.position.y = float(pos[1])
            gt_odom.pose.pose.position.z = float(pos[2])
            gt_odom.pose.pose.orientation.w = float(quat[0])
            gt_odom.pose.pose.orientation.x = float(quat[1])
            gt_odom.pose.pose.orientation.y = float(quat[2])
            gt_odom.pose.pose.orientation.z = float(quat[3])
            gt_odom.twist.twist.linear.x  = float(vel[0])
            gt_odom.twist.twist.angular.z = float(ang[2])
            self._vslam_pub.publish(gt_odom)

        # ── Joint states ──────────────────────────────────────────────────────
        if self._step_count % 10 == 0:  # 50 Hz
            try:
                dof_names  = art.dof_names
                dof_pos    = art.get_joint_positions()
                dof_vel    = art.get_joint_velocities()
                js = JointState()
                js.header.stamp = now
                js.name     = list(dof_names)
                js.position = [float(p) for p in dof_pos]
                js.velocity = [float(v) for v in dof_vel]
                self._jstate_pub.publish(js)
            except Exception:
                pass

    def _apply_controls(self):
        """Convert /cmd_vel to wheel velocity commands via differential drive."""
        if self._articulation is None:
            return

        vx, omega = self._cmd_vel
        if self._estop or self._mode == "off":
            vx, omega = 0.0, 0.0

        # Differential drive → motor velocities (from mujoco_bridge.py)
        TRACK_WIDTH  = 0.5588  # m
        WHEEL_RADIUS = 0.154   # m
        GEAR_LEFT    = 145.0 / 12.0  # 12.083
        GEAR_RIGHT   = 174.0 / 12.0  # 14.5

        v_left  = (vx - omega * TRACK_WIDTH / 2.0)
        v_right = (vx + omega * TRACK_WIDTH / 2.0)
        ctrl_left  = np.clip(-GEAR_LEFT  * v_left  / WHEEL_RADIUS, -40.0,  40.0)
        ctrl_right = np.clip(-GEAR_RIGHT * v_right / WHEEL_RADIUS, -48.0,  48.0)

        try:
            # Find left/right motor DOF indices by name
            dof_names = list(self._articulation.dof_names)
            ctrl = np.zeros(len(dof_names))
            for i, name in enumerate(dof_names):
                if "left_motor" in name:
                    ctrl[i] = ctrl_left
                elif "right_motor" in name:
                    ctrl[i] = ctrl_right
            self._articulation.set_joint_velocities(ctrl)
        except Exception:
            pass

    async def run(self):
        """Main simulation loop — called by Isaac Sim app."""
        await self._world.initialize_simulation_context_async()
        self._world.reset()

        if not self.setup_scene():
            carb.log_error("[TeamBowl] Scene setup failed, aborting.")
            return

        await self._world.play_async()
        carb.log_info("[TeamBowl] Simulation running. WebRTC: http://localhost:8211")

        while self._world.is_playing():
            await self._world.step_async(render=True)
            self._step_count += 1
            self._apply_controls()
            self._publish_sensors()
            rclpy.spin_once(self._node, timeout_sec=0.0)

        rclpy.shutdown()


# ── Entry point called by Isaac Sim --exec flag ────────────────────────────────
sim = TeamBowlSimulation()
asyncio.ensure_future(sim.run())
