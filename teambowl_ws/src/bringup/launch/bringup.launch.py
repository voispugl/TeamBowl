from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
import os
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError


def generate_launch_description():
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
            'rectify_rgb': 'false',
            'params_file': os.path.join(
                get_package_share_directory('depthai_ros_driver'),
                'config', 'oak_d_pro_w.yaml'
            ),
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

    xsens_imu = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('xsens_mti_ros2_driver'),
                'launch',
                'xsens_mti_node.launch.py'
            )
        ),
    )

    management_config = os.path.join(
        get_package_share_directory('management'), 'config', 'management.yaml')
    safety_config = os.path.join(
        get_package_share_directory('safety'), 'config', 'safety.yaml')
    locomotion_config = os.path.join(
        get_package_share_directory('locomotion'), 'config', 'locomotion.yaml')
    balance_config = os.path.join(
        get_package_share_directory('locomotion'), 'config', 'balance_controller.yaml')
    ekf_config = os.path.join(
        get_package_share_directory('locomotion'), 'config', 'ekf.yaml')
    vesc_config = os.path.join(
        get_package_share_directory('vesc_driver'), 'config', 'vesc_driver.yaml')
    perception_config = os.path.join(
        get_package_share_directory('perception'), 'config', 'perception.yaml')
    planning_config = os.path.join(
        get_package_share_directory('planning'), 'config', 'planning.yaml')
    lid_config = os.path.join(
        get_package_share_directory('locomotion'), 'config', 'lid_controller.yaml')

    # Foxglove bridge — allows Foxglove Studio to connect for visualization and
    # gain tuning via /balance_gains topic. Requires ros-humble-foxglove-bridge.
    # Install: sudo apt install ros-humble-foxglove-bridge
    # Connect: open Foxglove Studio → Open Connection → Rosbridge (ws://robot-ip:8765)
    try:
        get_package_share_directory('foxglove_bridge')
        _foxglove_available = True
    except PackageNotFoundError:
        _foxglove_available = False

    foxglove_arg = DeclareLaunchArgument(
        'foxglove',
        default_value='true' if _foxglove_available else 'false',
        description='Launch foxglove_bridge for remote visualization (default: true if installed)'
    )
    use_foxglove = LaunchConfiguration('foxglove')

    leg_controller_arg = DeclareLaunchArgument(
        'leg_controller',
        default_value='hold',
        description='Leg controller to launch: hold (default), driving, or none. '
                    'hold and driving cannot run simultaneously.')
    leg_ctrl = LaunchConfiguration('leg_controller')

    return LaunchDescription([

        foxglove_arg,
        leg_controller_arg,

        oak_camera,
        robstride_driver,
        xsens_imu,

        Node(
            package='management',
            executable='mode_manager',
            name='mode_manager',
            output='screen',
            parameters=[management_config],
        ),

        Node(
            package='management',
            executable='led_controller',
            name='led_controller',
            output='screen',
            parameters=[management_config],
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
            parameters=[locomotion_config],
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

        # Balance controller: passthrough in non-balance modes,
        # LQR+PI balance in "balance" mode.
        # Sits between collision_guard (/cmd_vel_safe) and cmd_vel_to_vesc (/cmd_vel).
        Node(
            package='locomotion',
            executable='balance_controller',
            name='balance_controller',
            output='screen',
            parameters=[balance_config],
        ),

        # Wheel odometry: integrates /cmd_vel into /odom_wheels for EKF fusion.
        Node(
            package='locomotion',
            executable='wheel_odom',
            name='wheel_odom',
            output='screen',
            parameters=[balance_config],
        ),

        # EKF: fuses /imu/data + /odom_wheels → /odometry/filtered
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[ekf_config],
        ),

        Node(
            package='vesc_driver',
            executable='cmd_vel_to_vesc',
            name='cmd_vel_to_vesc',
            output='screen',
            parameters=[vesc_config],
        ),

        Node(
            package='perception',
            executable='cam_ops',
            name='cam_ops_node',
            output='screen',
            parameters=[perception_config],
        ),

        Node(
            package='planning',
            executable='plan_wheels',
            name='plan_wheels',
            output='screen',
            parameters=[planning_config],
        ),

        # Foxglove bridge — remote visualization + /balance_gains topic tuning
        # Disable with: ros2 launch bringup bringup.launch.py foxglove:=false
        # Lid controller: drives RS05 motor (cargo bay lid) between open/close.
        # Trigger from Foxglove: Publish panel → /lid_command (std_msgs/String)
        # Messages: {"data": "open"}, {"data": "close"}, {"data": "toggle"}
        Node(
            package='locomotion',
            executable='lid_controller',
            name='lid_controller',
            output='screen',
            parameters=[lid_config],
        ),

        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output='screen',
            parameters=[{
                'port': 8765,
                'address': '0.0.0.0',
                'tls': False,
                'topic_whitelist': ['.*'],  # expose all topics
                'param_whitelist': ['.*'],   # expose all parameters
                'max_qos_depth': 1,
            }],
            condition=IfCondition(use_foxglove),
        ),
    ])
