from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
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
            'name': 'oak'
        }.items()
    )
    
    return LaunchDescription([

        oak_camera,

        Node(
            package='management',
            executable='mode_manager',
            name='mode_manager',
            output='screen',
            parameters=[
                {'teleop_enable_topic': '/teleop_enable'},
                {'teleop_enable_set_topic': '/teleop_enable_set'},
                {'start_teleop_enabled': False},
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
                {'max_erpm_step_per_tick': 2000},
                {'max_erpm': 20000},
                {'cmd_timeout_s': 0.5},
                {'baud': 115200},
                {'serial_timeout_s': 0.05},
                {'left_sign': 1},
                {'right_sign': -1},
                {'print_RPM_cmds': False},
            ],
        ),

        Node(
            package='perception',
            executable='cam_ops',
            name='cam_ops_node',
            output='screen',
            parameters=[
                {'image_topic': '/oak/rgb/image_rect'},
                {'depth_topic': '/oak/stereo/image_raw'},
                {'camera_info_topic': '/oak/rgb/camera_info'},
                {'target_topic': '/user_pos'},
                {'target_valid_topic': '/user_valid'},
                {'debug_image_topic': '/robot/debug/cam_ops_image'},
                {'sync_slop_s': 0.2},
                {'min_pink_area_px': 300},
                {'enable_resize': True},
                {'resize_scale': 0.5},
                {'min_depth_m': 0.2},
                {'max_depth_m': 8.0},
                {'depth_window_radius_px': 2},
            ],
        ), 

        Node(
            package='planning',
            executable='plan_wheels',
            name='plan_wheels',
            output='screen',
            parameters=[
                {'target_topic': '/user_pos'},
                {'target_valid_topic': '/user_valid'},
                {'cmd_vel_topic': '/cmd_vel_auto'},
                {'target_timeout_s': 0.5},
                {'publish_rate_hz': 20.0},
                {'follow_distance_m': 1.5},
                {'distance_deadband_m': 0.15},
                {'lateral_deadband_m': 0.10},
                {'k_linear': 0.8},
                {'k_angular': 1.8},
                {'max_linear_x': 0.8},
                {'max_angular_z': 1.2},
                {'allow_reverse': False},
                {'max_reverse_x': 0.25},
                {'turn_in_place_angle_only': True},
                {'turn_only_lateral_threshold_m': 0.5},
            ],
        ),
    ])
