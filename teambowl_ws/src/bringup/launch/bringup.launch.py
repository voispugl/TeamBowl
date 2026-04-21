from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
import os
import math
import xml.etree.ElementTree as ET
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError


def _parse_xyz(text):
    return [float(value) for value in text.split()]


def _compute_base_to_imu_tf(urdf_path):
    root = ET.parse(urdf_path).getroot()
    wheel_positions = {}

    for joint in root.findall('joint'):
        name = joint.get('name')
        if name not in {'left_wheel_0', 'right_wheel_0'}:
            continue
        origin = joint.find('origin')
        if origin is None or origin.get('xyz') is None:
            raise RuntimeError(f'Joint {name} is missing an origin xyz in {urdf_path}')
        wheel_positions[name] = _parse_xyz(origin.get('xyz'))

    missing = {'left_wheel_0', 'right_wheel_0'} - set(wheel_positions)
    if missing:
        raise RuntimeError(
            f'URDF {urdf_path} is missing required wheel joints: {sorted(missing)}'
        )

    base_in_imu = [
        (wheel_positions['left_wheel_0'][index] + wheel_positions['right_wheel_0'][index]) / 2.0
        for index in range(3)
    ]

    # IMU axes are x-right, y-forward, z-up. base_link is x-forward, y-left, z-up.
    yaw_base_to_imu = -math.pi / 2.0
    cos_yaw = math.cos(yaw_base_to_imu)
    sin_yaw = math.sin(yaw_base_to_imu)
    imu_in_base = [
        -(cos_yaw * base_in_imu[0] - sin_yaw * base_in_imu[1]),
        -sin_yaw * base_in_imu[0] - cos_yaw * base_in_imu[1],
        -base_in_imu[2],
    ]

    return imu_in_base, yaw_base_to_imu


def _compute_base_to_rgb_camera_tf(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joint_origins = {}

    for joint in root.findall('joint'):
        name = joint.get('name')
        if name not in {'left_wheel_0', 'right_wheel_0', 'rgb_cam_0'}:
            continue
        origin = joint.find('origin')
        if origin is None or origin.get('xyz') is None:
            raise RuntimeError(f'Joint {name} is missing an origin xyz in {urdf_path}')
        joint_origins[name] = {
            'xyz': _parse_xyz(origin.get('xyz')),
            'rpy': _parse_xyz(origin.get('rpy', '0 0 0')),
        }

    missing = {'left_wheel_0', 'right_wheel_0', 'rgb_cam_0'} - set(joint_origins)
    if missing:
        raise RuntimeError(
            f'URDF {urdf_path} is missing required joints: {sorted(missing)}'
        )

    base_in_imu = [
        (joint_origins['left_wheel_0']['xyz'][index] + joint_origins['right_wheel_0']['xyz'][index]) / 2.0
        for index in range(3)
    ]
    cam_in_imu = joint_origins['rgb_cam_0']['xyz']
    dx_i = cam_in_imu[0] - base_in_imu[0]
    dy_i = cam_in_imu[1] - base_in_imu[1]
    dz_i = cam_in_imu[2] - base_in_imu[2]

    # URDF Frame/IMU axes: x-right, y-forward, z-up. base_link: x-forward, y-left, z-up.
    cam_pos_in_base = [
        dy_i,
        -dx_i,
        dz_i,
    ]

    # camera.launch.py creates the optical-frame rotation internally. We only want
    # the physical mount of the camera base frame relative to base_link here.
    return cam_pos_in_base, [0.0, 0.0, 0.0]


def generate_launch_description():

    robot_urdf = os.path.join(
        get_package_share_directory('bringup'),
        'robot_description',
        'bowl.urdf',
    )
    imu_translation, imu_yaw = _compute_base_to_imu_tf(robot_urdf)
    cam_translation, cam_rpy = _compute_base_to_rgb_camera_tf(robot_urdf)

    oak_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('depthai_ros_driver'),
                'launch',
                'camera.launch.py'
            )
        ),
        launch_arguments={
            'name': 'oak',
            'rectify_rgb': 'true',
            'pointcloud.enable': 'true',
            'params_file': os.path.join(
                get_package_share_directory('bringup'),
                'config',
                'oak_cam.yaml',
            ),
            'parent_frame': 'base_link',
            'cam_pos_x': str(cam_translation[0]),
            'cam_pos_y': str(cam_translation[1]),
            'cam_pos_z': str(cam_translation[2]),
            'cam_roll': str(cam_rpy[0]),
            'cam_pitch': str(cam_rpy[1]),
            'cam_yaw': str(cam_rpy[2]),
        }.items()
    )

    robstride_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('robstride_can_driver'),
                'launch',
                'driver.launch.py'
            )
        ),
    )

    try:
        _xsens_launch = os.path.join(
            get_package_share_directory('xsens_mti_ros2_driver'),
            'launch',
            'xsens_mti_node.launch.py',
        )
        xsens_imu = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(_xsens_launch)
        )
        _xsens_available = True
    except PackageNotFoundError:
        xsens_imu = None
        _xsens_available = False

    management_config = os.path.join(
        get_package_share_directory('management'), 'config', 'management.yaml')
    safety_config = os.path.join(
        get_package_share_directory('safety'), 'config', 'safety.yaml')
    locomotion_config = os.path.join(
        get_package_share_directory('locomotion'), 'config', 'locomotion.yaml')
    balance_config = os.path.join(
        get_package_share_directory('locomotion'), 'config', 'balance_controller.yaml')
    driving_config = os.path.join(
        get_package_share_directory('locomotion'), 'config', 'driving_controller.yaml')
    vesc_config = os.path.join(
        get_package_share_directory('vesc_driver'), 'config', 'vesc_driver.yaml')
    perception_config = os.path.join(
        get_package_share_directory('perception'), 'config', 'perception.yaml')
    planning_config = os.path.join(
        get_package_share_directory('planning'), 'config', 'planning.yaml')
    lid_config = os.path.join(
        get_package_share_directory('locomotion'), 'config', 'lid_controller.yaml')
    state_estimation_config = os.path.join(
        get_package_share_directory('state_estimation'), 'config', 'state_estimation.yaml')

    # Foxglove bridge — allows Foxglove Studio to connect for visualization and
    # live gain tuning via /balance_gains, /driving_gains topics.
    # Requires ros-humble-foxglove-bridge.
    # Install: sudo apt install ros-humble-foxglove-bridge
    # Connect: open Foxglove Studio → Open Connection → Rosbridge (ws://robot-ip:8765)
    try:
        _steamdeck_config = os.path.join(
            get_package_share_directory('steamdeck_teleop'),
            'config', 'steamdeck_teleop.yaml')
        _steamdeck_available = True
    except PackageNotFoundError:
        _steamdeck_config = None
        _steamdeck_available = False

    try:
        get_package_share_directory('foxglove_bridge')
        _foxglove_available = True
    except PackageNotFoundError:
        _foxglove_available = False

    steamdeck_ui_arg = DeclareLaunchArgument(
        'steamdeck_ui',
        default_value='phone',
        description='steamdeck web UI mode: phone (default, 3 big buttons) or full (trajectory/gains/map)')
    steamdeck_ui = LaunchConfiguration('steamdeck_ui')

    foxglove_arg = DeclareLaunchArgument(
        'foxglove',
        default_value='true' if _foxglove_available else 'false',
        description='Launch foxglove_bridge for remote visualization (default: true if installed)'
    )
    use_foxglove = LaunchConfiguration('foxglove')

    leg_controller_arg = DeclareLaunchArgument(
        'leg_controller',
        default_value='driving',
        description='Leg controller to launch: hold (default), driving, or none. '
                    'hold and driving cannot run simultaneously.')
    leg_ctrl = LaunchConfiguration('leg_controller')

    velocity_controller_arg = DeclareLaunchArgument(
        'velocity_controller',
        default_value='driving',
        description='Velocity controller: driving (default) or balance. '
                    'driving runs the locked-leg velocity+pitch+yaw PID for autonomous nav. '
                    'balance runs the self-balancing cascaded PID. '
                    'driving and balance cannot run simultaneously.')
    vel_ctrl = LaunchConfiguration('velocity_controller')

    verbose_controllers_arg = DeclareLaunchArgument(
        'verbose_controllers',
        default_value='false',
        description='Enable periodic status logging for lid_controller and '
                    'driving_leg_controller (2 s and 5 s intervals respectively). '
                    'Off by default to reduce console noise.')
    verbose_controllers = LaunchConfiguration('verbose_controllers')

    return LaunchDescription([

        steamdeck_ui_arg,
        foxglove_arg,
        leg_controller_arg,
        velocity_controller_arg,
        verbose_controllers_arg,

        oak_camera,
        robstride_driver,
        *([xsens_imu] if _xsens_available else []),

        # TF: base_link → imu_link (computed from URDF wheel positions)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_imu_tf',
            output='screen',
            arguments=[
                str(imu_translation[0]),
                str(imu_translation[1]),
                str(imu_translation[2]),
                str(imu_yaw),
                '0',
                '0',
                'base_link',
                'imu_link',
            ],
        ),

        Node(
            package='management',
            executable='mode_manager',
            name='mode_manager',
            output='screen',
            parameters=[management_config],
        ),

        Node(
            package='safety',
            executable='pico_bridge',
            name='pico_bridge',
            output='screen',
            parameters=[safety_config],
        ),

        Node(
            package='safety',
            executable='stuck_detector',
            name='stuck_detector',
            output='screen',
            parameters=[safety_config],
        ),

        Node(
            package='safety',
            executable='heartbeat_publisher',
            name='heartbeat_publisher',
            output='screen',
            parameters=[safety_config],
        ),

        Node(
            package='safety',
            executable='system_health',
            name='system_health',
            output='screen',
            parameters=[safety_config],
        ),

        # driving and hold cannot run at the same time.
        # Verify legs are in the correct position on startup if using driving mode.
        Node(
            package='locomotion',
            executable='hold_position_controller',
            name='hold_position_controller',
            output='screen',
            parameters=[locomotion_config],
            condition=IfCondition(PythonExpression(["'", leg_ctrl, "' == 'hold'"])),
        ),

        Node(
            package='locomotion',
            executable='driving_leg_controller',
            name='driving_leg_controller',
            output='screen',
            parameters=[locomotion_config, {'verbose': verbose_controllers}],
            condition=IfCondition(PythonExpression(["'", leg_ctrl, "' == 'driving'"])),
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

        # Velocity controller: sits between collision_guard (/cmd_vel_safe) and
        # cmd_vel_to_vesc (/cmd_vel). Only one may run at a time.
        #   balance (default) — cascaded PID self-balancing (mode="balance")
        #   driving           — velocity PID + pitch correction for locked-leg nav (mode="driving")
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

        # Wheel odometry: integrates /cmd_vel into /odom_wheels for EKF fusion.
        Node(
            package='locomotion',
            executable='wheel_odom',
            name='wheel_odom',
            output='screen',
            parameters=[balance_config],
        ),

        # State estimation: differential drive odometry from VESC wheel encoders.
        Node(
            package='state_estimation',
            executable='diff_drive_odom',
            name='diff_drive_odom',
            output='screen',
            parameters=[state_estimation_config],
        ),

        # EKF: fuses /imu/data + wheel odometry → /odometry/filtered
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[state_estimation_config],
        ),

        Node(
            package='vesc_driver',
            executable='cmd_vel_to_vesc',
            name='cmd_vel_to_vesc',
            output='screen',
            parameters=[vesc_config],
        ),

        # Nav2 planning stack
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[planning_config],
        ),

        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[planning_config],
            remappings=[
                ('/cmd_vel', '/cmd_vel_auto'),
            ],
        ),

        Node(
            package='planning',
            executable='nav_cloud_filter',
            name='nav_cloud_filter',
            output='screen',
            parameters=[planning_config],
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
                'inf_epsilon': 1.0,
            }],
        ),

        # Trajectory test — Foxglove-driven goal → nav2 plan + execute
        # Idle outside "driving" mode. Publish JSON goal to /trajectory_goal,
        # then "go" to /trajectory_cmd to start live-replanning execution.
        Node(
            package='planning',
            executable='trajectory_test',
            name='trajectory_test',
            output='screen',
            parameters=[planning_config],
        ),

        Node(
            package='planning',
            executable='follow_goal',
            name='follow_goal',
            output='screen',
            parameters=[planning_config],
        ),

        Node(
            package='planning',
            executable='follow_executor',
            name='follow_executor',
            output='screen',
            parameters=[planning_config],
        ),

        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[planning_config],
        ),

        TimerAction(
            period=10.0,
            actions=[
                Node(
                    package='perception',
                    executable='cam_ops',
                    name='cam_ops_node',
                    output='screen',
                    parameters=[perception_config],
                    respawn=True,
                    respawn_delay=3.0,
                ),
            ]
        ),

        # Lid controller: drives RS05 motor (cargo bay lid) between open/close.
        # Trigger from Foxglove: Publish panel → /lid_command (std_msgs/String)
        # Messages: {"data": "open"}, {"data": "close"}, {"data": "toggle"}
        Node(
            package='locomotion',
            executable='lid_controller',
            name='lid_controller',
            output='screen',
            parameters=[lid_config, {'verbose': verbose_controllers}],
        ),

        # Steam Deck / phone web UI — port 8888
        # phone mode (default): 3 big buttons (ENABLE / OPEN LID / KILL) + diagnostics
        # full mode: trajectory goals, mode buttons, balance gains, nav map
        # Override: ros2 launch bringup bringup.launch.py steamdeck_ui:=full
        *([Node(
            package='steamdeck_teleop',
            executable='steamdeck_ws_teleop',
            name='steamdeck_ws_teleop',
            output='screen',
            parameters=[_steamdeck_config, {'ui_mode': steamdeck_ui}],
        )] if _steamdeck_available else []),

        # Foxglove bridge — remote visualization + gain tuning topics
        # Disable with: ros2 launch bringup bringup.launch.py foxglove:=false
        Node(
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
        ),
    ])
