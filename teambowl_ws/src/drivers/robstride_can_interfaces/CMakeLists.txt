cmake_minimum_required(VERSION 3.8)
project(robstride_can_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

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

ament_export_dependencies(rosidl_default_runtime)
ament_package()
