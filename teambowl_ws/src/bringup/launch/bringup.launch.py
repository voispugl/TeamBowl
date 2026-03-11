from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='management',
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
            package='safety',
            executable='heartbeat_publisher',
            name='heartbeat_publisher',
            output='screen',
            parameters=[
                {'heartbeat_topic': '/heartbeat'},
                {'publish_rate_hz': 10.0},
            ],
        ),

        Node(
            package='safety',
            executable='system_health',
            name='system_health',
            output='screen',
            parameters=[
                {'heartbeat_topic': '/heartbeat'},
                {'estop_topic': '/estop'},
                {'timeout_s': 1.0},
                {'publish_rate_hz': 10.0},
                {'start_estop_true': False},
            ],
        ),

        Node(
            package='locomotion',
            executable='vel_cmd_mux',
            name='vel_cmd_mux',
            output='screen',
            parameters=[
                {'teleop_enable_topic': '/teleop_enable'},
                {'teleop_topic': '/cmd_vel_teleop'},
                {'auto_topic': '/cmd_vel_auto'},
                {'estop_topic': '/estop'},
                {'output_topic': '/cmd_vel_selected'},
                {'teleop_timeout_s': 0.5},
                {'auto_timeout_s': 0.5},
                {'publish_rate_hz': 30.0},
            ],
        ),

        Node(
            package='locomotion',
            executable='collision_guard',
            name='collision_guard',
            output='screen',
            parameters=[
                {'input_topic': '/cmd_vel_selected'},
                {'estop_topic': '/estop'},
                {'output_topic': '/cmd_vel'},
                {'max_linear_x': 2.0},
                {'max_angular_z': 4.0},
            ],
        ),

        Node(
            package='vesc_driver',
            executable='cmd_vel_to_vesc',
            name='cmd_vel_to_vesc',
            output='screen',
            parameters=[
                {'cmd_vel_topic': '/cmd_vel'},
                {'left_port': '/dev/ttyACM0'},
                {'right_port': '/dev/ttyACM1'},
                {'estop_topic': '/estop'},
                {'wheel_radius_m': 0.307975},
                {'track_width_m': 0.5588},
                {'erpm_per_wheel_rpm': 500.0},
                {'max_erpm': 20000},
                {'cmd_timeout_s': 0.5},
                {'baud': 115200},
                {'serial_timeout_s': 0.05},
                {'left_sign': 1},
                {'right_sign': -1},
            ],
        ),
    ])
