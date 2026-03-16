"""
Launch file for the RobStride CAN Motor Driver.

Usage:
    ros2 launch robstride_can_driver driver.launch.py
    ros2 launch robstride_can_driver driver.launch.py config_file:=/path/to/motors.yaml
    ros2 launch robstride_can_driver driver.launch.py startup_mode:=startup_home
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('robstride_can_driver')
    default_config = os.path.join(pkg_share, 'config', 'motors.yaml')

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Absolute path to the motors.yaml configuration file',
    )

    startup_mode_arg = DeclareLaunchArgument(
        'startup_mode',
        default_value='',
        description=(
            'Override startup_mode from YAML. '
            'Options: startup_safe, startup_home. '
            'Leave empty to use the value in motors.yaml.'
        ),
    )

    driver_node = Node(
        package='robstride_can_driver',
        executable='driver_node',
        name='robstride_can_driver',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'config_file': LaunchConfiguration('config_file'),
            'startup_mode_override': LaunchConfiguration('startup_mode'),
        }],
    )

    return LaunchDescription([
        config_file_arg,
        startup_mode_arg,
        driver_node,
    ])
