"""
isaac_sim.launch.py — Full robot stack connected to Isaac Sim.

Isaac Sim (running in Docker) publishes:
  /imu/data                        — simulated IMU  → EKF imu0
  /wheel/odometry                  — simulated wheel encoders → EKF odom0
  /visual_slam/tracking/odometry   — ground-truth pose substitute → EKF odom1
  /oak/rgb/image_raw               — rendered RGB camera (for YOLO26 / cam_ops)
  /oak/stereo/image_raw            — rendered depth (for nvblox)
  /joint_states                    — all 23 joint positions + velocities
  /clock                           — simulation time

This launch connects those topics to the full control + navigation stack.
Hardware drivers (OAK-D, Xsens, VESC, CAN) are not launched — Isaac Sim simulates them.

Launch arguments
----------------
  velocity_controller   driving (default) | balance
  use_nvblox            true (default) | false   — nvblox on RTX 5080 via CUDA
  use_yolo26            false (default) | true    — ML person detection on simulated camera
  foxglove              true (default) | false

Usage
-----
  ros2 launch bringup isaac_sim.launch.py
  ros2 launch bringup isaac_sim.launch.py use_yolo26:=true
  ros2 launch bringup isaac_sim.launch.py use_nvblox:=false velocity_controller:=balance

Then in browser: http://localhost:8211  (Isaac Sim WebRTC)
     Foxglove:  ws://localhost:8765
     Set mode:  ros2 topic pub /robot_mode_set std_msgs/msg/String '{data: "driving"}' --once
"""

import os

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():

    # ── Config paths ───────────────────────────────────────────────────────────
    locomotion_config     = os.path.join(
        get_package_share_directory('locomotion'), 'config', 'locomotion.yaml')
    balance_config        = os.path.join(
        get_package_share_directory('locomotion'), 'config', 'balance_controller.yaml')
    driving_config        = os.path.join(
        get_package_share_directory('locomotion'), 'config', 'driving_controller.yaml')
    management_config     = os.path.join(
        get_package_share_directory('management'), 'config', 'management.yaml')
    state_est_config      = os.path.join(
        get_package_share_directory('state_estimation'), 'config', 'state_estimation.yaml')
    perception_config     = os.path.join(
        get_package_share_directory('perception'), 'config', 'perception.yaml')

    _planning_default = os.path.join(
        get_package_share_directory('planning'), 'config', 'planning.yaml')
    _planning_nvblox  = os.path.join(
        get_package_share_directory('planning'), 'config', 'planning_nvblox.yaml')

    # ── Package availability checks ────────────────────────────────────────────
    try:
        get_package_share_directory('foxglove_bridge')
        _foxglove_available = True
    except PackageNotFoundError:
        _foxglove_available = False

    try:
        get_package_share_directory('nvblox_ros')
        _nvblox_available = True
    except PackageNotFoundError:
        _nvblox_available = False

    # ── Launch arguments ───────────────────────────────────────────────────────
    foxglove_arg = DeclareLaunchArgument(
        'foxglove',
        default_value='true' if _foxglove_available else 'false',
        description='Launch foxglove_bridge for remote visualization.')
    use_foxglove = LaunchConfiguration('foxglove')

    velocity_controller_arg = DeclareLaunchArgument(
        'velocity_controller',
        default_value='driving',
        description='driving (default, velocity PID + pitch) or balance (cascaded PID self-balancing).')
    vel_ctrl = LaunchConfiguration('velocity_controller')

    use_nvblox_arg = DeclareLaunchArgument(
        'use_nvblox',
        default_value='true',
        description='Enable nvblox 3D TSDF costmap (CUDA on RTX 5080). '
                    'Uses /oak/stereo/image_raw from Isaac Sim rendered depth.')
    use_nvblox = LaunchConfiguration('use_nvblox')

    use_yolo26_arg = DeclareLaunchArgument(
        'use_yolo26',
        default_value='false',
        description='Enable YOLO26 person detection on Isaac Sim rendered RGB camera. '
                    'Requires ~/TeamBowl/models/yolo26n.engine (export_yolo26.py).')
    use_yolo26 = LaunchConfiguration('use_yolo26')

    return LaunchDescription([
        foxglove_arg,
        velocity_controller_arg,
        use_nvblox_arg,
        use_yolo26_arg,

        # ── Static TFs (hardware TF tree comes from Isaac Sim) ─────────────────
        # base_link → imu_link: identity (Isaac Sim IMU IS body frame in sim)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_imu_tf',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'imu_link'],
        ),

        # map → odom: identity (no SLAM/localization — Isaac Sim is ground truth)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_tf',
            output='screen',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        ),

        # nvblox_camera frame — dedicated TF for nvblox, at approximate OAK-D position
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='nvblox_camera_tf',
            output='screen',
            # Approximate OAK-D position: 0.15m forward, 0.30m up from base_link
            arguments=['0.15', '0', '0.30', '0', '0', '0', 'base_link', 'nvblox_camera'],
            condition=IfCondition(use_nvblox),
        ),

        # ── Mode + safety ──────────────────────────────────────────────────────
        Node(
            package='management',
            executable='mode_manager',
            name='mode_manager',
            output='screen',
            parameters=[management_config],
        ),

        Node(
            package='locomotion',
            executable='vel_cmd_mux',
            name='vel_cmd_mux',
            output='screen',
            parameters=[locomotion_config],
        ),

        Node(
            package='locomotion',
            executable='collision_guard',
            name='collision_guard',
            output='screen',
            parameters=[locomotion_config],
        ),

        # ── Velocity controllers (one runs at a time) ──────────────────────────
        Node(
            package='locomotion',
            executable='balance_controller',
            name='balance_controller',
            output='screen',
            parameters=[balance_config],
            condition=IfCondition(PythonExpression(["'", vel_ctrl, "' == 'balance'"])),
        ),

        Node(
            package='locomotion',
            executable='driving_controller',
            name='driving_controller',
            output='screen',
            parameters=[driving_config],
            condition=IfCondition(PythonExpression(["'", vel_ctrl, "' == 'driving'"])),
        ),

        # wheel_odom: dead-reckoning /cmd_vel → /odom_wheels as backup odometry
        Node(
            package='locomotion',
            executable='wheel_odom',
            name='wheel_odom',
            output='screen',
            parameters=[balance_config],
        ),

        # ── State estimation ───────────────────────────────────────────────────
        # EKF fuses:
        #   imu0  ← /imu/data                       (Isaac Sim simulated IMU)
        #   odom0 ← /wheel/odometry                 (Isaac Sim simulated wheel encoders)
        #   odom1 ← /visual_slam/tracking/odometry  (Isaac Sim ground-truth → VSLAM substitute)
        # Output: /odometry/filtered → nav2 controller + planning
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[state_est_config, {'use_sim_time': True}],
        ),

        # ── Nav2 planning stack ────────────────────────────────────────────────
        # use_sim_time:=true — Isaac Sim publishes /clock
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[PythonExpression([
                '"', _planning_nvblox, '" if "', use_nvblox, '" == "true" else "',
                _planning_default, '"',
            ]), {'use_sim_time': True}],
        ),

        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[PythonExpression([
                '"', _planning_nvblox, '" if "', use_nvblox, '" == "true" else "',
                _planning_default, '"',
            ]), {'use_sim_time': True}],
            remappings=[('/cmd_vel', '/cmd_vel_auto')],
        ),

        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[PythonExpression([
                '"', _planning_nvblox, '" if "', use_nvblox, '" == "true" else "',
                _planning_default, '"',
            ]), {'use_sim_time': True}],
        ),

        # CPU obstacle pipeline — disabled when nvblox is active
        Node(
            package='planning',
            executable='nav_cloud_filter',
            name='nav_cloud_filter',
            output='screen',
            parameters=[_planning_default],
            condition=UnlessCondition(use_nvblox),
        ),

        Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='nav_cloud_to_scan',
            output='screen',
            remappings=[
                ('cloud_in', '/oak/nav_points'),
                ('scan', '/oak/nav_scan'),
            ],
            parameters=[{
                'target_frame': 'base_link',
                'transform_tolerance': 0.1,
                'min_height': -0.10,
                'max_height': 1.20,
                'angle_min': -1.5708,
                'angle_max': 1.5708,
                'angle_increment': 0.00872665,
                'scan_time': 0.1,
                'range_min': 0.15,
                'range_max': 2.50,
                'use_inf': True,
            }],
            condition=UnlessCondition(use_nvblox),
        ),

        # ── nvblox 3D TSDF — CUDA on RTX 5080 ─────────────────────────────────
        *([Node(
            package='nvblox_ros',
            executable='nvblox_node',
            name='nvblox',
            output='screen',
            parameters=[{
                'voxel_size': 0.05,
                'max_depth_m': 3.0,
                'min_depth_m': 0.1,
                'esdf_2d_min_height': 0.0,
                'esdf_2d_max_height': 1.2,
                'esdf_slice_height': 0.15,
                'map_clearing_radius_m': 5.0,
                'global_frame': 'odom',
                'pose_frame': 'base_link',
                'use_sim_time': True,
            }],
            remappings=[
                ('depth/image',       '/oak/stereo/image_raw'),
                ('depth/camera_info', '/oak/stereo/camera_info'),
            ],
            condition=IfCondition(use_nvblox),
        )] if _nvblox_available else []),

        # ── Person following ───────────────────────────────────────────────────
        Node(
            package='planning',
            executable='follow_goal',
            name='follow_goal',
            output='screen',
            parameters=[_planning_default],
        ),

        Node(
            package='planning',
            executable='follow_executor',
            name='follow_executor',
            output='screen',
            parameters=[_planning_default, {'use_sim_time': True}],
        ),

        # ── Perception — simulated OAK-D camera ───────────────────────────────
        # cam_ops: HSV pink blob on /oak/rgb/image_raw from Isaac Sim renderer
        Node(
            package='perception',
            executable='cam_ops',
            name='cam_ops_node',
            output='screen',
            parameters=[perception_config],
        ),

        # yolo26: ML person detection on simulated RGB (optional, requires .engine file)
        Node(
            condition=IfCondition(use_yolo26),
            package='perception',
            executable='yolo26_node',
            name='yolo26_node',
            output='screen',
            parameters=[perception_config],
        ),

        # ── Foxglove ───────────────────────────────────────────────────────────
        *([Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output='screen',
            parameters=[{
                'port': 8765,
                'address': '0.0.0.0',
                'tls': False,
                'topic_whitelist': ['.*'],
                'param_whitelist': ['.*'],
                'max_qos_depth': 1,
            }],
            condition=IfCondition(use_foxglove),
        )] if _foxglove_available else []),
    ])
