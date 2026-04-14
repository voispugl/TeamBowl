from setuptools import find_packages, setup

package_name = 'locomotion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'config/locomotion.yaml',
            'config/balance_controller.yaml',
            'config/ekf.yaml',
            'config/lid_controller.yaml',
        ]),
        ('share/' + package_name, ['locomotion/driving_leg_pos.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='box',
    maintainer_email='jvoisin@andrew.cmu.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vel_cmd_mux = locomotion.vel_cmd_mux:main',
            'collision_guard = locomotion.collision_guard:main',
            'driving_leg_controller = locomotion.driving_leg_controller:main',
            'hold_position_controller = locomotion.hold_position_controller:main',
            'balance_controller = locomotion.balance_controller:main',
            'wheel_odom = locomotion.wheel_odom:main',
            'lid_controller = locomotion.lid_controller:main',
        ],
    },
)
