## Directory: launch/

### driver.launch.py
ROS2 launch file. Accepts two launch arguments:
- `config_file`: path to motors.yaml (defaults to the installed package config)
- `startup_mode`: override the YAML startup_mode at launch time (leave empty to use YAML value)

Usage:
  ros2 launch robstride_can_driver driver.launch.py
  ros2 launch robstride_can_driver driver.launch.py startup_mode:=startup_home
  ros2 launch robstride_can_driver driver.launch.py config_file:=/my_robot/motors.yaml
