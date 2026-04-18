"""
steamdeck_ws.launch.py — Launch the Steam Deck WebSocket teleop server on the robot.

Run alongside bringup:
  ros2 launch bringup trajectory_test.launch.py &
  ros2 launch steamdeck_teleop steamdeck_ws.launch.py

Then on the Steam Deck: open browser → http://ROBOT_IP:8888
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('steamdeck_teleop'),
        'config',
        'steamdeck_teleop.yaml',
    )

    return LaunchDescription([
        Node(
            package='steamdeck_teleop',
            executable='steamdeck_ws_teleop',
            name='steamdeck_ws_teleop',
            output='screen',
            parameters=[config],
        ),
    ])
