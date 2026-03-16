cmake_minimum_required(VERSION 3.8)
project(robstride_can_driver)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Generate custom service interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetGains.srv"
  "srv/SetMotorId.srv"
  "srv/SetZero.srv"
  "srv/ShiftZero.srv"
  "srv/SetZeroOffset.srv"
  "srv/ReadMotorParam.srv"
  "srv/WriteMotorParam.srv"
  DEPENDENCIES std_msgs
)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install docs
install(DIRECTORY docs/
  DESTINATION share/${PROJECT_NAME}/docs
)

# Install tools
install(PROGRAMS tools/commissioning.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
