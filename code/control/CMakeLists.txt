cmake_minimum_required(VERSION 3.0.2)
project(control)


find_package(catkin REQUIRED)

catkin_python_setup()

find_package(catkin REQUIRED COMPONENTS
    rospy
    std_msgs
    dynamic_reconfigure
)

generate_dynamic_reconfigure_options(
  config/Controller.cfg
  config/PurePursuit.cfg
  config/Velocity.cfg
  config/Stanley.cfg
)

catkin_package()



include_directories(
)

