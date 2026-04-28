"""
Standalone test launch for Isaac ROS Visual SLAM + nvblox.

Use this to validate VSLAM and nvblox independently of the full robot stack
(no motors, no CAN, no VESC, no Nav2 planner). All you need is the OAK-D PoE W
connected and the Docker image rebuilt with Isaac ROS.

Usage:
  ros2 launch bringup isaac_ros_test.launch.py
  ros2 launch bringup isaac_ros_test.launch.py vslam_debug:=true
  ros2 launch bringup isaac_ros_test.launch.py use_nvblox:=false

What to check in Foxglove (ws://robot-ip:8765):
  /visual_slam/tracking/odometry    — VSLAM pose, should update as robot moves
  /oak/imu/data                     — camera IMU at ~200 Hz
  /oak/left/image_rect              — rectified stereo at ~90 Hz
  /nvblox/mesh                      — 3D TSDF mesh (if use_nvblox:=true)
  /tf                               — verify odom→base_link chain is publishing
"""

import os
import math
import xml.etree.ElementTree as ET

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError


def _parse_xyz(text):
    return [float(v) for v in text.split()]


def _compute_cam_tf(urdf_path):
    root = ET.parse(urdf_path).getroot()
    origins = {}
    for joint in root.findall('joint'):
        name = joint.get('name')
        if name not in {'left_wheel_0', 'right_wheel_0', 'rgb_cam_0'}:
            continue
        origin = joint.find('origin')
        if origin is None:
            continue
        origins[name] = {
            'xyz': _parse_xyz(origin.get('xyz', '0 0 0')),
            'rpy': _parse_xyz(origin.get('rpy', '0 0 0')),
        }
    base_in_imu = [
        (origins['left_wheel_0']['xyz'][i] + origins['right_wheel_0']['xyz'][i]) / 2.0
        for i in range(3)
    ]
    cam_in_imu = origins['rgb_cam_0']['xyz']
    cam_pos = [
        cam_in_imu[1] - base_in_imu[1],
        -(cam_in_imu[0] - base_in_imu[0]),
        cam_in_imu[2] - base_in_imu[2],
    ]
    return cam_pos, origins['rgb_cam_0']['rpy']


def generate_launch_description():

    robot_urdf = os.path.join(
        get_package_share_directory('bringup'),
        'robot_description', 'bowl.urdf')
    cam_translation, cam_rpy = _compute_cam_tf(robot_urdf)

    oak_params = os.path.join(
        get_package_share_directory('bringup'), 'config', 'oak_cam.yaml')

    try:
        get_package_share_directory('isaac_ros_visual_slam')
        _vslam_available = True
    except PackageNotFoundError:
        _vslam_available = False

    try:
        get_package_share_directory('nvblox_ros')
        _nvblox_available = True
    except PackageNotFoundError:
        _nvblox_available = False

    try:
        get_package_share_directory('foxglove_bridge')
        _foxglove_available = True
    except PackageNotFoundError:
        _foxglove_available = False

    use_nvblox_arg = DeclareLaunchArgument(
        'use_nvblox', default_value='true',
        description='Launch nvblox 3D TSDF node alongside VSLAM.')
    use_nvblox = LaunchConfiguration('use_nvblox')

    vslam_debug_arg = DeclareLaunchArgument(
        'vslam_debug', default_value='false',
        description='Enable VSLAM visualization topics (observations, landmarks, path).')
    vslam_debug = LaunchConfiguration('vslam_debug')

    return LaunchDescription([
        use_nvblox_arg,
        vslam_debug_arg,

        # OAK-D PoE W — full VSLAM config (IMU, 90 Hz stereo, 15 Hz H.264 RGB)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('depthai_ros_driver'),
                'launch', 'camera.launch.py')),
            launch_arguments={
                'name': 'oak',
                'rectify_rgb': 'false',
                'pointcloud.enable': 'true',
                'params_file': oak_params,
                'parent_frame': 'base_link',
                'cam_pos_x': str(cam_translation[0]),
                'cam_pos_y': str(cam_translation[1]),
                'cam_pos_z': str(cam_translation[2]),
                'cam_roll':  str(cam_rpy[0]),
                'cam_pitch': str(cam_rpy[1]),
                'cam_yaw':   str(cam_rpy[2]),
            }.items()
        ),

        # Minimal TF tree for testing: odom → base_link (identity) and base_link → imu_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='odom_to_base_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link'],
        ),

        # nvblox_camera dedicated TF — same position as OAK-D
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='nvblox_camera_tf',
            arguments=[
                str(cam_translation[0]), str(cam_translation[1]), str(cam_translation[2]),
                str(cam_rpy[2]), str(cam_rpy[1]), str(cam_rpy[0]),
                'base_link', 'nvblox_camera',
            ],
        ),

        # Isaac ROS Visual SLAM
        *([OpaqueFunction(function=lambda context: [Node(
            package='isaac_ros_visual_slam',
            executable='isaac_ros_visual_slam',
            name='visual_slam',
            output='screen',
            parameters=[{
                'use_sim_time': False,
                'denoise_input_images': False,
                'rectified_images': True,
                'enable_imu_fusion': True,
                'base_frame': 'base_link',
                'imu_frame': 'oak_imu_frame',
                'fixed_frame': 'odom',
                'enable_slam_visualization':
                    context.perform_substitution(LaunchConfiguration('vslam_debug')) == 'true',
                'enable_landmarks_view':
                    context.perform_substitution(LaunchConfiguration('vslam_debug')) == 'true',
                'enable_observations_view':
                    context.perform_substitution(LaunchConfiguration('vslam_debug')) == 'true',
            }],
            remappings=[
                ('visual_slam/image_0',       '/oak/left/image_rect'),
                ('visual_slam/camera_info_0', '/oak/left/camera_info'),
                ('visual_slam/image_1',       '/oak/right/image_rect'),
                ('visual_slam/camera_info_1', '/oak/right/camera_info'),
                ('visual_slam/imu',           '/oak/imu/data'),
            ],
        )])] if _vslam_available else []),

        # nvblox 3D TSDF node
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
            }],
            remappings=[
                ('depth/image',       '/oak/stereo/image_raw'),
                ('depth/camera_info', '/oak/stereo/camera_info'),
            ],
            condition=IfCondition(use_nvblox),
        )] if _nvblox_available else []),

        # Foxglove bridge for visualization
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
            ros_arguments=['--log-level', 'foxglove_bridge:=warn'],
        )] if _foxglove_available else []),
    ])
