cmake_minimum_required(VERSION 3.8)
project(td3)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY
    launch
    models
    scripts
    urdf
    worlds
  DESTINATION
    share/${PROJECT_NAME}/
)

install(
  PROGRAMS
    scripts/replay_buffer.py
    scripts/test_robot.py
    scripts/train_robot.py
    scripts/point_cloud2.py
    scripts/ros2_env.py
    scripts/TD3.py
    launch/gz_robot.launch.py
    launch/robot_state_publisher.launch.py
    launch/test_td3.launch.py    
    launch/train_td3.launch.py
    launch/pioneer3dx.rviz
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
