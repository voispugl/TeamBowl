cmake_minimum_required(VERSION 3.8)
project(robstride_can_driver)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY docs/
  DESTINATION share/${PROJECT_NAME}/docs
)

install(PROGRAMS
  tools/commissioning.py
  tools/diagnose.py
  tools/monitor.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
