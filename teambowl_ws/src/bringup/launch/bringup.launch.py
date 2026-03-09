from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='teambowl_mode_manager',
            executable='mode_manager',
            name='mode_manager',
            output='screen',
            parameters=[
                {'teleop_enable_topic': '/teleop_enable'},
                {'teleop_enable_set_topic': '/teleop_enable_set'},
                {'start_teleop_enabled': True},
                {'publish_rate_hz': 5.0},
            ],
        ),

        Node(
            package='teambowl_safety_ops',
            executable='heartbeat_publisher',
            name='heartbeat_publisher',
            output='screen',
            parameters=[
                {'heartbeat_topic': '/heartbeat'},
                {'publish_rate_hz': 10.0},
            ],
        ),

        Node(
            package='teambowl_safety_ops',
            executable='watchdog',
            name='watchdog',
            output='screen',
            parameters=[
                {'heartbeat_topic': '/heartbeat'},
                {'estop_topic': '/estop'},
                {'timeout_s': 1.0},
                {'publish_rate_hz': 10.0},
                {'start_estop_true': True},
            ],
        ),

        Node(
            package='teambowl_motion',
            executable='vel_cmd_mux',
            name='vel_cmd_mux',
            output='screen',
            parameters=[
                {'teleop_enable_topic': '/teleop_enable'},
                {'teleop_topic': '/cmd_vel_teleop'},
                {'auto_topic': '/cmd_vel_auto'},
                {'estop_topic': '/estop'},
                {'output_topic': '/cmd_vel_selected'},
                {'teleop_timeout_s': 0.35},
                {'auto_timeout_s': 0.35},
                {'publish_rate_hz': 30.0},
            ],
        ),

        Node(
            package='teambowl_motion',
            executable='collision_guard',
            name='collision_guard',
            output='screen',
            parameters=[
                {'input_topic': '/cmd_vel_selected'},
                {'estop_topic': '/estop'},
                {'output_topic': '/cmd_vel'},
                {'max_linear_x': 0.5},
                {'max_angular_z': 1.0},
            ],
        ),

        Node(
            package='teambowl_vesc_driver',
            executable='cmd_vel_to_vesc',
            name='cmd_vel_to_vesc',
            output='screen',
            parameters=[
                {'cmd_vel_topic': '/cmd_vel'},
                {'wheel_radius_m': 0.10},
                {'track_width_m': 0.45},
                {'max_wheel_rpm': 300.0},
                {'left_port': '/dev/vesc_left'},
                {'right_port': '/dev/vesc_right'},
            ],
        ),
    ])
