from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
import os
from ament_index_python.packages import get_package_share_directory


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
            'rectify_rgb': 'true',
            'pointcloud.enable': 'false',
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

    management_config = os.path.join(
        get_package_share_directory('management'), 'config', 'management.yaml')
    safety_config = os.path.join(
        get_package_share_directory('safety'), 'config', 'safety.yaml')
    locomotion_config = os.path.join(
        get_package_share_directory('locomotion'), 'config', 'locomotion.yaml')
    vesc_config = os.path.join(
        get_package_share_directory('vesc_driver'), 'config', 'vesc_driver.yaml')
    perception_config = os.path.join(
        get_package_share_directory('perception'), 'config', 'perception.yaml')
    planning_config = os.path.join(
        get_package_share_directory('planning'), 'config', 'planning.yaml')

    leg_controller_arg = DeclareLaunchArgument(
        'leg_controller',
        default_value='hold',
        description='Leg controller to launch: hold (default), driving, or none. '
                    'hold and driving cannot run simultaneously.')
    leg_ctrl = LaunchConfiguration('leg_controller')

    return LaunchDescription([

        leg_controller_arg,

        oak_camera,
        robstride_driver,

        Node(
            package='management',
            executable='mode_manager',
            name='mode_manager',
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
    ])
