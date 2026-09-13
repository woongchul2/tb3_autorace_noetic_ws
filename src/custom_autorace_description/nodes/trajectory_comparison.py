#!/usr/bin/env python3
"""Compare actual simulated positions with the EKF trajectory."""

import math
import threading

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point, Pose, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_srvs.srv import Empty, EmptyResponse
from visualization_msgs.msg import Marker, MarkerArray


def yaw_from_quaternion(quaternion):
    sin_yaw = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y
    )
    cos_yaw = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z
    )
    return math.atan2(sin_yaw, cos_yaw)


def quaternion_from_yaw(yaw):
    pose = Pose()
    pose.orientation.z = math.sin(yaw * 0.5)
    pose.orientation.w = math.cos(yaw * 0.5)
    return pose.orientation


def angular_distance(first, second):
    return abs(math.atan2(math.sin(second - first), math.cos(second - first)))


class TrajectoryComparison:
    def __init__(self):
        self.model_name = rospy.get_param("~model_name", "custom_autorace")
        self.frame_id = rospy.get_param("~frame_id", "odom")
        self.max_points = int(rospy.get_param("~max_points", 20000))
        self.filtered_topic = rospy.get_param(
            "~filtered_topic", "/odometry/filtered"
        )
        self.min_translation = float(rospy.get_param("~min_translation", 0.005))
        self.min_rotation = float(rospy.get_param("~min_rotation", 0.01))
        self.line_width = float(rospy.get_param("~line_width", 0.012))

        self.lock = threading.Lock()
        self.initial_ground_truth = None
        self.initial_filtered = None
        self.latest_ground_truth = None
        self.filtered_pose_covariance = None
        self.ground_truth_path = Path()
        self.filtered_path = Path()
        self.ground_truth_path.header.frame_id = self.frame_id
        self.filtered_path.header.frame_id = self.frame_id

        self.ground_truth_publisher = rospy.Publisher(
            "/ground_truth/path", Path, queue_size=1, latch=True
        )
        self.filtered_publisher = rospy.Publisher(
            "/filtered/path", Path, queue_size=1, latch=True
        )
        self.marker_publisher = rospy.Publisher(
            "/trajectory/comparison", MarkerArray, queue_size=1, latch=True
        )

        rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.model_states_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.filtered_topic, Odometry, self.filtered_callback, queue_size=1
        )
        rospy.Service("/trajectory/reset", Empty, self.reset_callback)

    def reset_callback(self, _request):
        with self.lock:
            self.initial_ground_truth = None
            self.initial_filtered = None
            self.latest_ground_truth = None
            self.filtered_pose_covariance = None
            self.ground_truth_path.poses = []
            self.filtered_path.poses = []
            self.publish_paths(rospy.Time.now())
        return EmptyResponse()

    def model_states_callback(self, message):
        try:
            index = message.name.index(self.model_name)
        except ValueError:
            rospy.logwarn_throttle(
                5.0, "Model '%s' is not present in /gazebo/model_states",
                self.model_name,
            )
            return

        stamp = rospy.Time.now()
        with self.lock:
            self.latest_ground_truth = message.pose[index]
            if self.initial_ground_truth is None:
                self.initial_ground_truth = message.pose[index]
            if self.initial_filtered is None:
                return

            aligned = self.align_ground_truth(message.pose[index])
            if self.append_pose(self.ground_truth_path, aligned, stamp):
                self.publish_paths(stamp)

    def filtered_callback(self, message):
        stamp = message.header.stamp
        if stamp == rospy.Time():
            stamp = rospy.Time.now()

        with self.lock:
            if self.initial_filtered is None:
                self.initial_filtered = message.pose.pose
            self.filtered_pose_covariance = list(message.pose.covariance)
            changed = self.append_pose(
                self.filtered_path, message.pose.pose, stamp
            )
            if (
                self.initial_ground_truth is not None
                and not self.ground_truth_path.poses
                and self.latest_ground_truth is not None
            ):
                aligned = self.align_ground_truth(self.latest_ground_truth)
                changed = self.append_pose(
                    self.ground_truth_path, aligned, stamp
                ) or changed
            if changed:
                self.publish_paths(stamp)

    def align_ground_truth(self, pose):
        ground_yaw = yaw_from_quaternion(self.initial_ground_truth.orientation)
        filtered_yaw = yaw_from_quaternion(self.initial_filtered.orientation)

        delta_x = pose.position.x - self.initial_ground_truth.position.x
        delta_y = pose.position.y - self.initial_ground_truth.position.y

        local_x = math.cos(ground_yaw) * delta_x + math.sin(ground_yaw) * delta_y
        local_y = -math.sin(ground_yaw) * delta_x + math.cos(ground_yaw) * delta_y

        aligned = Pose()
        aligned.position.x = (
            self.initial_filtered.position.x
            + math.cos(filtered_yaw) * local_x
            - math.sin(filtered_yaw) * local_y
        )
        aligned.position.y = (
            self.initial_filtered.position.y
            + math.sin(filtered_yaw) * local_x
            + math.cos(filtered_yaw) * local_y
        )
        aligned.position.z = (
            self.initial_filtered.position.z
            + pose.position.z
            - self.initial_ground_truth.position.z
        )

        relative_yaw = yaw_from_quaternion(pose.orientation) - ground_yaw
        aligned.orientation = quaternion_from_yaw(filtered_yaw + relative_yaw)
        return aligned

    def append_pose(self, path, pose, stamp):
        if path.poses:
            previous = path.poses[-1].pose
            distance = math.hypot(
                pose.position.x - previous.position.x,
                pose.position.y - previous.position.y,
            )
            rotation = angular_distance(
                yaw_from_quaternion(previous.orientation),
                yaw_from_quaternion(pose.orientation),
            )
            if distance < self.min_translation and rotation < self.min_rotation:
                return False

        stamped = PoseStamped()
        stamped.header.stamp = stamp
        stamped.header.frame_id = self.frame_id
        stamped.pose = pose
        path.poses.append(stamped)
        if self.max_points > 0 and len(path.poses) > self.max_points:
            del path.poses[:len(path.poses) - self.max_points]
        return True

    def publish_paths(self, stamp):
        self.ground_truth_path.header.stamp = stamp
        self.filtered_path.header.stamp = stamp
        self.ground_truth_publisher.publish(self.ground_truth_path)
        self.filtered_publisher.publish(self.filtered_path)
        self.marker_publisher.publish(self.make_markers(stamp))

    def make_markers(self, stamp):
        marker_array = MarkerArray()
        marker_array.markers = [
            self.make_position_markers(
                0, "actual_map_positions", self.ground_truth_path, stamp,
                red=1.0, green=0.1, blue=0.1,
            ),
            self.make_marker(
                1, "encoder_imu_ekf", self.filtered_path, stamp,
                red=0.1, green=1.0, blue=0.2,
            ),
            self.make_covariance_marker(stamp),
        ]
        return marker_array

    def make_position_markers(
        self, marker_id, namespace, path, stamp, red, green, blue
    ):
        """Draw independently sampled true positions as dots on the map."""
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.025
        marker.scale.y = 0.025
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = 1.0
        marker.points = [
            Point(x=item.pose.position.x, y=item.pose.position.y,
                  z=item.pose.position.z + 0.01)
            for item in path.poses
        ]
        return marker

    def make_covariance_marker(self, stamp):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = "ekf_position_covariance_2sigma"
        marker.id = 2

        if self.filtered_pose_covariance is None or not self.filtered_path.poses:
            marker.action = Marker.DELETE
            return marker

        covariance = self.filtered_pose_covariance
        covariance_xx = max(0.0, covariance[0])
        covariance_xy = 0.5 * (covariance[1] + covariance[6])
        covariance_yy = max(0.0, covariance[7])
        trace = covariance_xx + covariance_yy
        difference = covariance_xx - covariance_yy
        discriminant = math.sqrt(
            max(0.0, difference * difference + 4.0 * covariance_xy ** 2)
        )
        major_variance = max(0.0, 0.5 * (trace + discriminant))
        minor_variance = max(0.0, 0.5 * (trace - discriminant))
        major_radius = 2.0 * math.sqrt(major_variance)
        minor_radius = 2.0 * math.sqrt(minor_variance)
        ellipse_yaw = 0.5 * math.atan2(2.0 * covariance_xy, difference)

        center = self.filtered_path.poses[-1].pose.position
        cos_yaw = math.cos(ellipse_yaw)
        sin_yaw = math.sin(ellipse_yaw)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.006
        marker.color.r = 1.0
        marker.color.g = 0.85
        marker.color.b = 0.1
        marker.color.a = 0.9
        marker.points = []
        for index in range(49):
            angle = 2.0 * math.pi * index / 48.0
            local_x = major_radius * math.cos(angle)
            local_y = minor_radius * math.sin(angle)
            marker.points.append(Point(
                x=center.x + cos_yaw * local_x - sin_yaw * local_y,
                y=center.y + sin_yaw * local_x + cos_yaw * local_y,
                z=center.z + 0.015,
            ))
        return marker

    def make_marker(self, marker_id, namespace, path, stamp, red, green, blue):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        # RViz rejects LINE_STRIP markers containing fewer than two points.
        marker.action = Marker.ADD if len(path.poses) >= 2 else Marker.DELETE
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.line_width
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = 1.0
        marker.points = [
            Point(x=item.pose.position.x, y=item.pose.position.y,
                  z=item.pose.position.z + 0.01)
            for item in path.poses
        ]
        return marker


if __name__ == "__main__":
    rospy.init_node("trajectory_comparison")
    TrajectoryComparison()
    rospy.spin()
