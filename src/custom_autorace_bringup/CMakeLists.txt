cmake_minimum_required(VERSION 3.0.2)
project(custom_autorace_bringup)

catkin_python_setup()

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  message_generation
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf2_ros
  turtlebot3_autorace_msgs
  visualization_msgs
)

add_message_files(
  FILES
  TrafficSign.msg
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
  std_msgs
)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_autorace_image_adapter.py)
  catkin_add_nosetests(test/test_analyze_integrated_bag.py)
  catkin_add_nosetests(test/test_detector_rate_configuration.py)
  catkin_add_nosetests(test/test_gazebo_launch_wiring.py)
  catkin_add_nosetests(test/test_hardware_perception_launch.py)
  catkin_add_nosetests(test/test_intersection_controller.py)
  catkin_add_nosetests(test/test_intersection_symmetry.py)
  catkin_add_nosetests(test/test_lane_lookahead_geometry.py)
  catkin_add_nosetests(test/test_lane_benchmark_monitor.py)
  catkin_add_nosetests(test/test_lane_centerline_detection.py)
  catkin_add_nosetests(test/test_lane_lookahead_path.py)
  catkin_add_nosetests(test/test_level_crossing_controller.py)
  catkin_add_nosetests(test/test_level_crossing_geometry.py)
  catkin_add_nosetests(test/test_local_registration.py)
  catkin_add_nosetests(test/test_mission_zone.py)
  catkin_add_nosetests(test/test_mission_zone_manager.py)
  catkin_add_nosetests(test/test_obstacle_planner.py)
  catkin_add_nosetests(test/test_obstacle_controller.py)
  catkin_add_nosetests(test/test_path_following.py)
  catkin_add_nosetests(test/test_parking_controller.py)
  catkin_add_nosetests(test/test_parking_local_registration.py)
  catkin_add_nosetests(test/test_parking_route_geometry.py)
  catkin_add_nosetests(test/test_safe_lane_controller.py)
  catkin_add_nosetests(test/test_sign_detector.py)
  catkin_add_nosetests(test/test_tunnel_costmap.py)
  catkin_add_nosetests(test/test_tunnel_controller.py)
  catkin_add_nosetests(test/test_tunnel_registration.py)
  catkin_add_nosetests(test/test_tunnel_planner.py)
  catkin_add_nosetests(test/test_zigzag_path.py)
endif()

catkin_package(
  CATKIN_DEPENDS cv_bridge geometry_msgs message_runtime nav_msgs rospy sensor_msgs std_msgs std_srvs tf2_ros turtlebot3_autorace_msgs visualization_msgs
)

catkin_install_python(PROGRAMS
  tools/analyze_integrated_bag.py
  nodes/amcl_startup_update.py
  nodes/autorace_image_adapter.py
  nodes/intersection_mission_controller.py
  nodes/lane_benchmark_monitor.py
  nodes/level_crossing_lidar_controller.py
  nodes/mission_zone_manager.py
  nodes/obstacle_mission_controller.py
  nodes/parking_mission_controller.py
  nodes/sign_detector.py
  nodes/safe_lane_controller.py
  nodes/tunnel_mission_controller.py
  nodes/zigzag_mission_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch maps
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES README.md PATH_FOLLOWING.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
