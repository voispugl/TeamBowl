"""
sim.launch.py — Minimal stack for MuJoCo simulation-based controller tuning.

Launches the mujoco_bridge node (which runs teambowl_mjlab.xml at 500 Hz and
publishes /imu/data, /odometry/filtered, /joint_states) plus the controller
pipeline, so balance_controller or driving_controller can be tuned without
any physical hardware.

NOT launched (hardware/vision):
  - robstride_can_driver, vesc_driver, cmd_vel_to_vesc
  - xsens_mti_ros2_driver (IMU comes from mujoco_bridge)
  - depthai_ros_driver, cam_ops, pointcloud_to_laserscan
  - robot_localization EKF, state_estimation/diff_drive_odom
  - nav2 planning stack

Launch arguments
----------------
  velocity_controller  balance (default) | driving
  foxglove             true (default if installed) | false

Usage
-----
  ros2 launch bringup sim.launch.py
  ros2 launch bringup sim.launch.py velocity_controller:=driving

Then in Foxglove (ws://<vm-ip>:8765):
  Set mode:  /robot_mode_set → {"data": "balance"}
  Send cmd:  /cmd_vel_teleop → {"linear": {"x": 0.1}, "angular": {"z": 0.0}}
  Tune:      ros2 param set /balance_controller kp_pitch 70.0
  Reset sim: ros2 service call /sim_reset std_srvs/srv/Trigger {}
"""

import os

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():

    # ------------------------------------------------------------------ #
    # Config file paths
    # ------------------------------------------------------------------ #
    locomotion_config = os.path.join(
        get_package_share_directory('locomotion'), 'config', 'locomotion.yaml')
    balance_config = os.path.join(
        get_package_share_directory('locomotion'), 'config', 'balance_controller.yaml')
    driving_config = os.path.join(
        get_package_share_directory('locomotion'), 'config', 'driving_controller.yaml')
    management_config = os.path.join(
        get_package_share_directory('management'), 'config', 'management.yaml')
    sim_config = os.path.join(
        get_package_share_directory('simulation'), 'config', 'mujoco_bridge.yaml')

    # ------------------------------------------------------------------ #
    # Launch arguments
    # ------------------------------------------------------------------ #
    try:
        get_package_share_directory('foxglove_bridge')
        _foxglove_available = True
    except PackageNotFoundError:
        _foxglove_available = False

    foxglove_arg = DeclareLaunchArgument(
        'foxglove',
        default_value='true' if _foxglove_available else 'false',
        description='Launch foxglove_bridge for remote visualization (default: true if installed)',
    )
    use_foxglove = LaunchConfiguration('foxglove')

    velocity_controller_arg = DeclareLaunchArgument(
        'velocity_controller',
        default_value='balance',
        description='balance (default, cascaded PID) or driving (locked-leg velocity PID)',
    )
    vel_ctrl = LaunchConfiguration('velocity_controller')

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    # Sim bridge: runs MuJoCo at 500 Hz, publishes /imu/data + /odometry/filtered + /joint_states
    mujoco_bridge = Node(
        package='simulation',
        executable='mujoco_bridge',
        name='mujoco_bridge',
        output='screen',
        parameters=[sim_config],
    )

    # Static TF: base_link → imu_link (identity — sim IMU IS the body frame)
    base_to_imu_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_imu_tf',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'imu_link'],
    )

    # Static TF: odom → map (identity — no SLAM in sim)
    odom_to_map_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='odom_to_map_tf',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
    )

    # Mode manager: handles /robot_mode_set → /robot_mode state machine
    mode_manager = Node(
        package='management',
        executable='mode_manager',
        name='mode_manager',
        output='screen',
        parameters=[management_config],
    )

    # vel_cmd_mux: routes teleop/auto commands based on mode
    vel_cmd_mux = Node(
        package='locomotion',
        executable='vel_cmd_mux',
        name='vel_cmd_mux',
        output='screen',
        parameters=[locomotion_config],
    )

    # collision_guard: clamps velocities, enforces e-stop
    collision_guard = Node(
        package='locomotion',
        executable='collision_guard',
        name='collision_guard',
        output='screen',
        parameters=[locomotion_config],
    )

    # balance_controller: cascaded PID self-balancing (active in "balance" mode)
    balance_controller = Node(
        package='locomotion',
        executable='balance_controller',
        name='balance_controller',
        output='screen',
        parameters=[balance_config],
        condition=IfCondition(PythonExpression(["'", vel_ctrl, "' == 'balance'"])),
    )

    # driving_controller: velocity PID + pitch correction (active in "driving" mode)
    driving_controller = Node(
        package='locomotion',
        executable='driving_controller',
        name='driving_controller',
        output='screen',
        parameters=[driving_config],
        condition=IfCondition(PythonExpression(["'", vel_ctrl, "' == 'driving'"])),
    )

    # Foxglove bridge — remote visualization + live gain tuning
    foxglove_bridge = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        output='screen',
        parameters=[{
            'port': 8765,
            'address': '0.0.0.0',
        }],
        condition=IfCondition(use_foxglove),
    )

    return LaunchDescription([
        foxglove_arg,
        velocity_controller_arg,

        mujoco_bridge,
        base_to_imu_tf,
        odom_to_map_tf,
        mode_manager,
        vel_cmd_mux,
        collision_guard,
        balance_controller,
        driving_controller,
        foxglove_bridge,
    ])
