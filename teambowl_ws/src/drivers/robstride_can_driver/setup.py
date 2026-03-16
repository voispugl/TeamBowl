from setuptools import find_packages, setup

package_name = 'robstride_can_driver'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/driver.launch.py']),
        ('share/' + package_name + '/config', [
            'config/motors.yaml',
            'config/commands_reference.yaml',
        ]),
    ],
    install_requires=['setuptools', 'python-can', 'pyyaml'],
    zip_safe=True,
    maintainer='TeamBowl',
    maintainer_email='placeholder@example.com',
    description='ROS2 Humble driver for RobStride CAN actuators',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'driver_node = robstride_can_driver.driver_node:main',
        ],
    },
)
