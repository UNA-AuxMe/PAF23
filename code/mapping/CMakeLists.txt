cmake_minimum_required(VERSION 3.0.2)
project(mapping)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  message_generation
  uuid_msgs
  geometry_msgs
)


## This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
## Note that the mapping common_package is install separately below
catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

# Generate messages in the 'msg' folder
add_message_files(
  FILES
  Entity.msg
  Flags.msg
  Map.msg
  Motion2D.msg
  Shape2D.msg
  TrackingInfo.msg
  Transform2D.msg
  TypeCar.msg
  TypeLanemarking.msg
  TypeTrafficLight.msg
  ClusteredPointsArray.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

# Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  uuid_msgs
  geometry_msgs
  sensor_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
LIBRARIES mapping mapping_common
CATKIN_DEPENDS message_runtime uuid_msgs std_msgs geometry_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

# Cython build for mapping_common
# based on 
# https://answers.ros.org/question/9731/how-to-use-python-setuptools-to-build-a-python-extension/
# http://wiki.ros.org/catkin/CMakeLists.txt#catkin_package.28.29
#
# This is a hack and should probably not be done this way. 
# Unfortunately using catkin_python_setup() does not work, because it does not install external modules -> cython.
set(PAF_MAPPING_COMMON_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/mapping_common_setup)
set(PAF_DEVEL_PYTHON_PACKAGES_DIR ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}/../)
add_custom_command(OUTPUT ${PAF_MAPPING_COMMON_BUILD_DIR}/pip-install.stamp
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH="${PAF_DEVEL_PYTHON_PACKAGES_DIR}:$ENV{PYTHONPATH}" ${PYTHON_EXECUTABLE} cmake_setup.py --base_dir "${CATKIN_DEVEL_PREFIX}" --packages_dir "${PAF_DEVEL_PYTHON_PACKAGES_DIR}"
  COMMAND mkdir -p ${PAF_MAPPING_COMMON_BUILD_DIR} && ${CMAKE_COMMAND} -E touch ${PAF_MAPPING_COMMON_BUILD_DIR}/pip-install.stamp
  DEPENDS 
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/*.py
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/mapping_common/*.py
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/mapping_common/*.pxd
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/.debug_enabled
  BYPRODUCTS
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/mapping_common/__init__.c
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/mapping_common/entity.c
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/mapping_common/filter.c
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/mapping_common/map.c
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/mapping_common/shape.c
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/mapping_common/transform.c
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/mapping_common/__init__.html
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/mapping_common/entity.html
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/mapping_common/filter.html
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/mapping_common/map.html
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/mapping_common/shape.html
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/mapping_common/transform.html
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/build 
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/dist 
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/mapping_common.egg-info
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules/mapping_common/mapping_common.egg-info
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ext_modules
)

add_custom_target(mapping_common ALL
  DEPENDS ${PAF_MAPPING_COMMON_BUILD_DIR}/pip-install.stamp
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

# Install currently not supported

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  add_rostest(tests/mapping_common.test)
endif()
