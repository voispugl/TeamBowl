from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    depthai_driver_path = os.path.join(
        get_package_share_directory('depthai_ros_driver'), 'launch', 'camera.launch.py'
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(depthai_driver_path),
            launch_arguments={
                'name': 'oak',
                'parent_frame': 'base_link',
                # Pipeline Configuration
                'i_enable_nn': 'True',             # Replaces pipeline.create(NN)
                'i_nn_type': 'spatial',            # Enables 3D XYZ data
                'i_enable_tracker': 'True',        # Replaces pipeline.create(Tracker)
                'i_tracker_type': 'ZERO_TERM_COLOR_HISTOGRAM',
                # Setting the detection label (15 = person)
                'i_conf_threshold': '0.5',
            }.items()
        )
    ])