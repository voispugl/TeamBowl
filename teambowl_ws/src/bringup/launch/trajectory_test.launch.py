"""
trajectory_test.launch.py — one-command launch for driving-mode trajectory testing.

Starts the full bringup stack with:
  - velocity_controller := driving  (driving_controller active — pitch + velocity PID)
  - leg_controller      := driving  (legs locked at driving positions)

Then auto-sets robot mode to "driving" after a 3-second delay so the
trajectory_test node and driving_controller are both active without any
manual Foxglove steps.

Usage:
  ros2 launch bringup trajectory_test.launch.py

Optional overrides:
  ros2 launch bringup trajectory_test.launch.py mode_delay_s:=5.0
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
    ExecuteProcess,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    bringup_dir = get_package_share_directory('bringup')

    mode_delay_arg = DeclareLaunchArgument(
        'mode_delay_s',
        default_value='3.0',
        description='Seconds to wait after launch before setting mode to "driving". '
                    'Increase if mode_manager is slow to start on your hardware.',
    )
    mode_delay = LaunchConfiguration('mode_delay_s')

    # Full bringup with driving controllers
    full_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_dir, 'launch', 'bringup.launch.py')
        ),
        launch_arguments={
            'velocity_controller': 'driving',
            'leg_controller': 'driving',
        }.items(),
    )

    # Auto-set mode to "driving" after mode_delay_s seconds
    set_driving_mode = TimerAction(
        period=mode_delay,
        actions=[
            ExecuteProcess(
                cmd=[
                    'ros2', 'topic', 'pub', '--once',
                    '/robot_mode_set',
                    'std_msgs/msg/String',
                    '{data: driving}',
                ],
                output='screen',
            )
        ],
    )

    return LaunchDescription([
        mode_delay_arg,
        full_bringup,
        set_driving_mode,
    ])
