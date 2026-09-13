#!/usr/bin/env python3
"""Add an explicit twist covariance to Gazebo encoder odometry."""

import copy

import rospy
from nav_msgs.msg import Odometry


class EncoderCovarianceAdapter:
    def __init__(self):
        input_topic = rospy.get_param("~input_topic", "/odom")
        output_topic = rospy.get_param(
            "~output_topic", "/odom/encoder_with_covariance"
        )
        linear_stddev = float(rospy.get_param("~linear_velocity_stddev", 0.02))
        angular_stddev = float(rospy.get_param("~angular_velocity_stddev", 0.05))

        self.linear_variance = linear_stddev * linear_stddev
        self.angular_variance = angular_stddev * angular_stddev
        self.publisher = rospy.Publisher(
            output_topic, Odometry, queue_size=10
        )
        rospy.Subscriber(input_topic, Odometry, self.callback, queue_size=10)

    def callback(self, message):
        output = copy.deepcopy(message)
        # vx, vy, vz, vroll, vpitch, vyaw diagonal. The EKF configuration only
        # selects vx. Finite values on the unused axes are still required because
        # robot_localization rotates the complete covariance matrix between frames.
        covariance = [0.0] * 36
        covariance[0] = self.linear_variance
        covariance[7] = self.linear_variance
        covariance[14] = self.linear_variance
        covariance[21] = self.angular_variance
        covariance[28] = self.angular_variance
        covariance[35] = self.angular_variance
        output.twist.covariance = covariance
        self.publisher.publish(output)


if __name__ == "__main__":
    rospy.init_node("encoder_covariance_adapter")
    EncoderCovarianceAdapter()
    rospy.spin()
