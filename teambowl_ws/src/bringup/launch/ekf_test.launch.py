from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
import math
import xml.etree.ElementTree as ET
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError


def _parse_xyz(text):
    return [float(v) for v in text.split()]


def _compute_base_to_imu_tf(urdf_path):
    root = ET.parse(urdf_path).getroot()
    wheel_positions = {}
    for joint in root.findall('joint'):
        name = joint.get('name')
        if name in {'left_wheel_0', 'right_wheel_0'}:
            origin = joint.find('origin')
            wheel_positions[name] = _parse_xyz(origin.get('xyz'))
    base_in_imu = [
        (wheel_positions['left_wheel_0'][i] + wheel_positions['right_wheel_0'][i]) / 2.0
        for i in range(3)
    ]
    yaw = -math.pi / 2.0
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    imu_in_base = [
        -(cos_y * base_in_imu[0] - sin_y * base_in_imu[1]),
        -sin_y * base_in_imu[0] - cos_y * base_in_imu[1],
        -base_in_imu[2],
    ]
    return imu_in_base, yaw


def _compute_base_to_rgb_camera_tf(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joint_origins = {}
    for joint in root.findall('joint'):
        name = joint.get('name')
        if name in {'left_wheel_0', 'right_wheel_0', 'rgb_cam_0'}:
            origin = joint.find('origin')
            joint_origins[name] = {
                'xyz': _parse_xyz(origin.get('xyz')),
                'rpy': _parse_xyz(origin.get('rpy', '0 0 0')),
            }
    base_in_imu = [
        (joint_origins['left_wheel_0']['xyz'][i] + joint_origins['right_wheel_0']['xyz'][i]) / 2.0
        for i in range(3)
    ]
    cam_in_imu = joint_origins['rgb_cam_0']['xyz']
    dx_i = cam_in_imu[0] - base_in_imu[0]
    dy_i = cam_in_imu[1] - base_in_imu[1]
    dz_i = cam_in_imu[2] - base_in_imu[2]
    return [dy_i, -dx_i, dz_i], [0.0, 0.0, 0.0]


def generate_launch_description():

    robot_urdf = os.path.join(
        get_package_share_directory('bringup'),
        'robot_description',
        'bowl.urdf',
    )
    imu_translation, imu_yaw = _compute_base_to_imu_tf(robot_urdf)
    cam_translation, cam_rpy = _compute_base_to_rgb_camera_tf(robot_urdf)

    state_estimation_config = os.path.join(
        get_package_share_directory('state_estimation'), 'config', 'state_estimation.yaml')

    try:
        _xsens_launch = os.path.join(
            get_package_share_directory('xsens_mti_ros2_driver'),
            'launch',
            'xsens_mti_node.launch.py',
        )
        xsens_imu = IncludeLaunchDescription(PythonLaunchDescriptionSource(_xsens_launch))
        _xsens_available = True
    except PackageNotFoundError:
        xsens_imu = None
        _xsens_available = False

    oak_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('depthai_ros_driver'), 'launch', 'camera.launch.py')
        ),
        launch_arguments={
            'name': 'oak',
            'rectify_rgb': 'true',
            'pointcloud.enable': 'false',
            'params_file': os.path.join(
                get_package_share_directory('depthai_ros_driver'), 'config', 'rgbd.yaml'),
            'parent_frame': 'base_link',
            'cam_pos_x': str(cam_translation[0]),
            'cam_pos_y': str(cam_translation[1]),
            'cam_pos_z': str(cam_translation[2]),
            'cam_roll': str(cam_rpy[0]),
            'cam_pitch': str(cam_rpy[1]),
            'cam_yaw': str(cam_rpy[2]),
        }.items()
    )

    return LaunchDescription([

        *([xsens_imu] if _xsens_available else []),
        oak_camera,

        # TF: base_link → imu_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_imu_tf',
            output='screen',
            arguments=[
                str(imu_translation[0]), str(imu_translation[1]), str(imu_translation[2]),
                str(imu_yaw), '0', '0',
                'base_link', 'imu_link',
            ],
        ),

        # diff_drive_odom: wheel encoder velocities → /wheel/odometry
        Node(
            package='state_estimation',
            executable='diff_drive_odom',
            name='diff_drive_odom',
            output='screen',
            parameters=[state_estimation_config],
        ),

        # EKF: fuses /imu/data + /oak/imu/data + /wheel/odometry → /odometry/filtered
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[state_estimation_config],
        ),

        # Foxglove bridge for visualization
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
        ),
    ])
