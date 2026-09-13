cmake_minimum_required(VERSION 3.0.2)
project(custom_autorace_description)

find_package(catkin REQUIRED COMPONENTS
  gazebo_msgs
  geometry_msgs
  nav_msgs
  rospy
  std_srvs
  urdf
  visualization_msgs
  xacro
)

catkin_package(CATKIN_DEPENDS
  gazebo_msgs geometry_msgs nav_msgs rospy std_srvs urdf visualization_msgs xacro
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_gazebo_course_server.py)
endif()

catkin_install_python(PROGRAMS
  nodes/encoder_covariance_adapter.py
  nodes/gazebo_course_server.py
  nodes/trajectory_comparison.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch meshes urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES README.md HARDWARE_PARAMETERS.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
