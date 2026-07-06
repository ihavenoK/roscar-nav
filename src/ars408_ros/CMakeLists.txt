cmake_minimum_required(VERSION 3.0.2)
project(ars408_ros)
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  visualization_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
  message_filters
  cv_bridge
  can_msgs
)
catkin_package()
catkin_install_python(PROGRAMS
  src/ars408_node.py
  src/fusion_node.py
  src/fusion_person_detect.py
  src/pedestrian_to_scan.py
  src/scan_test.py
  src/tf_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
