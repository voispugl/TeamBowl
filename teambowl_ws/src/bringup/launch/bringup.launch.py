from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
import os
import math
import xml.etree.ElementTree as ET
from ament_index_python.packages import get_package_share_directory

def _parse_xyz(text):
    return [float(value) for value in text.split()]


def _compute_base_to_imu_tf(urdf_path):
    root = ET.parse(urdf_path).getroot()
    wheel_positions = {}

    for joint in root.findall('joint'):
        name = joint.get('name')
        if name not in {'left_wheel_0', 'right_wheel_0'}:
            continue
        origin = joint.find('origin')
        if origin is None or origin.get('xyz') is None:
            raise RuntimeError(f'Joint {name} is missing an origin xyz in {urdf_path}')
        wheel_positions[name] = _parse_xyz(origin.get('xyz'))

    missing = {'left_wheel_0', 'right_wheel_0'} - set(wheel_positions)
    if missing:
        raise RuntimeError(
            f'URDF {urdf_path} is missing required wheel joints: {sorted(missing)}'
        )

    base_in_imu = [
        (wheel_positions['left_wheel_0'][index] + wheel_positions['right_wheel_0'][index]) / 2.0
        for index in range(3)
    ]

    # IMU axes are x-right, y-forward, z-up. base_link is x-forward, y-left, z-up.
    yaw_base_to_imu = -math.pi / 2.0
    cos_yaw = math.cos(yaw_base_to_imu)
    sin_yaw = math.sin(yaw_base_to_imu)
    imu_in_base = [
        -(cos_yaw * base_in_imu[0] - sin_yaw * base_in_imu[1]),
        -sin_yaw * base_in_imu[0] - cos_yaw * base_in_imu[1],
        -base_in_imu[2],
    ]

    return imu_in_base, yaw_base_to_imu


def _compute_base_to_rgb_camera_tf(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joint_origins = {}

    for joint in root.findall('joint'):
        name = joint.get('name')
        if name not in {'left_wheel_0', 'right_wheel_0', 'rgb_cam_0'}:
            continue
        origin = joint.find('origin')
        if origin is None or origin.get('xyz') is None:
            raise RuntimeError(f'Joint {name} is missing an origin xyz in {urdf_path}')
        joint_origins[name] = {
            'xyz': _parse_xyz(origin.get('xyz')),
            'rpy': _parse_xyz(origin.get('rpy', '0 0 0')),
        }

    missing = {'left_wheel_0', 'right_wheel_0', 'rgb_cam_0'} - set(joint_origins)
    if missing:
        raise RuntimeError(
            f'URDF {urdf_path} is missing required joints: {sorted(missing)}'
        )

    base_in_imu = [
        (joint_origins['left_wheel_0']['xyz'][index] + joint_origins['right_wheel_0']['xyz'][index]) / 2.0
        for index in range(3)
    ]
    cam_in_imu = joint_origins['rgb_cam_0']['xyz']
    cam_rpy_in_imu = joint_origins['rgb_cam_0']['rpy']

    dx_i = cam_in_imu[0] - base_in_imu[0]
    dy_i = cam_in_imu[1] - base_in_imu[1]
    dz_i = cam_in_imu[2] - base_in_imu[2]

    # URDF Frame/IMU axes: x-right, y-forward, z-up. base_link: x-forward, y-left, z-up.
    cam_pos_in_base = [
        dy_i,
        -dx_i,
        dz_i,
    ]

    # Convert camera orientation into base_link coordinates. The rgb_cam_0 joint is
    # modeled as a -90 deg roll in the IMU frame; base_link itself is rotated -90 deg
    # yaw relative to that IMU frame.
    cam_roll_in_base = cam_rpy_in_imu[0]
    cam_pitch_in_base = cam_rpy_in_imu[1]
    cam_yaw_in_base = -math.pi / 2.0

    return cam_pos_in_base, [cam_roll_in_base, cam_pitch_in_base, cam_yaw_in_base]


def generate_launch_description():

    robot_urdf = os.path.join(
        get_package_share_directory('bringup'),
        'robot_description',
        'bowl.urdf',
    )
    imu_translation, imu_yaw = _compute_base_to_imu_tf(robot_urdf)
    cam_translation, cam_rpy = _compute_base_to_rgb_camera_tf(robot_urdf)

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
            'pointcloud.enable': 'true',
            'params_file': os.path.join(
                get_package_share_directory('depthai_ros_driver'),
                'config', 'rgbd.yaml'),
            'parent_frame': 'base_link',
            'cam_pos_x': str(cam_translation[0]),
            'cam_pos_y': str(cam_translation[1]),
            'cam_pos_z': str(cam_translation[2]),
            'cam_roll': str(cam_rpy[0]),
            'cam_pitch': str(cam_rpy[1]),
            'cam_yaw': str(cam_rpy[2]),
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
    state_estimation_config = os.path.join(
        get_package_share_directory('state_estimation'), 'config', 'state_estimation.yaml')

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
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_to_imu_tf',
            output='screen',
            arguments=[
                str(imu_translation[0]),
                str(imu_translation[1]),
                str(imu_translation[2]),
                str(imu_yaw),
                '0',
                '0',
                'base_link',
                'imu_link',
            ],
        ),

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
            package='state_estimation',
            executable='diff_drive_odom',
            name='diff_drive_odom',
            output='screen',
            parameters=[state_estimation_config],
        ),

        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[state_estimation_config],
        ),

        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[planning_config],
        ),

        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[planning_config],
            remappings=[
                ('/cmd_vel', '/cmd_vel_auto'),
            ],
        ),

        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[planning_config],
        ),

        Node(
            package='perception',
            executable='cam_ops',
            name='cam_ops_node',
            output='screen',
            parameters=[perception_config],
        ),

        # Node(
        #     package='planning',
        #     executable='plan_wheels',
        #     name='plan_wheels',
        #     output='screen',
        #     parameters=[planning_config],
        # ),
    ])
