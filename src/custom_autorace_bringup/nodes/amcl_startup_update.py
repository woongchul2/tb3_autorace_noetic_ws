#!/usr/bin/env python3
"""Force AMCL's first scan update so map -> odom exists before motion."""

import rospy
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty


if __name__ == "__main__":
    rospy.init_node("amcl_startup_update")
    scan_topic = rospy.get_param("~scan_topic", "/scan_mid360_raw")
    service_name = rospy.get_param(
        "~request_nomotion_update_service", "/request_nomotion_update"
    )
    timeout = float(rospy.get_param("~timeout", 15.0))
    try:
        rospy.wait_for_message(scan_topic, LaserScan, timeout=timeout)
        rospy.wait_for_service(service_name, timeout=timeout)
        rospy.ServiceProxy(service_name, Empty)()
        rospy.loginfo("Requested initial AMCL scan update")
    except (rospy.ROSException, rospy.ServiceException) as error:
        rospy.logerr("Could not request initial AMCL scan update: %s", error)
