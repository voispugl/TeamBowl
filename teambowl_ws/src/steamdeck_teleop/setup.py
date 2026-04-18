from setuptools import find_packages, setup

package_name = 'steamdeck_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/steamdeck_teleop.yaml']),
        ('share/' + package_name + '/launch', ['launch/steamdeck_ws.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='teambowl',
    maintainer_email='cherber@andrew.cmu.edu',
    description='Steam Deck joystick → Nav2 navigation goal sender',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'steamdeck_ws_teleop = steamdeck_teleop.steamdeck_ws_teleop:main',
        ],
    },
)
