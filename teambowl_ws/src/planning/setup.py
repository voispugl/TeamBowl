from setuptools import find_packages, setup

package_name = 'planning'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/planning.yaml']),
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
            'plan_wheels = planning.plan_wheels:main',
            'nav_cloud_filter = planning.nav_cloud_filter:main',
            'follow_goal = planning.follow_goal:main',
            'follow_executor = planning.follow_executor:main',
            'trajectory_test = planning.trajectory_test:main',
        ],
    },
)
