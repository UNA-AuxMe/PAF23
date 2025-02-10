cmake_minimum_required(VERSION 3.0.2)
project(localization)

find_package(catkin REQUIRED)

## System dependencies are found with CMake's conventions
find_package(catkin REQUIRED COMPONENTS
  rosout
  rospy
  std_msgs
  message_generation
  dynamic_reconfigure
)

catkin_python_setup()


generate_messages(
  DEPENDENCIES
  std_msgs
)

## Generate dynamic reconfigure parameters in the 'cfg' folder
generate_dynamic_reconfigure_options(
  config/odometry_covariance.cfg
  config/sensor_covariance.cfg
)

catkin_package()


include_directories()