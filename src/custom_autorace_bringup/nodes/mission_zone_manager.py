#!/usr/bin/env python3
"""Arm ordered missions and open their gate only from fresh readiness."""

import math
import threading
import time

import rospy
import tf2_ros
from geometry_msgs.msg import Point, PoseStamped
from std_msgs.msg import Bool, Header, String, UInt8
from visualization_msgs.msg import Marker, MarkerArray

from custom_autorace_bringup.mission_zone import (
    MissionZoneSequence,
    is_inside_with_margin,
    signed_polygon_distances,
)


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2),
    )


class MissionZoneManager:
    @staticmethod
    def _wait_for_initial_ros_time():
        """Return a non-zero startup stamp before publishing a sim arm.

        Under ``/use_sim_time`` ROS time remains zero until Gazebo publishes
        the first ``/clock`` sample.  Publishing the latched arm before that
        point can permanently strand a controller which rejects a zero stamp
        and de-duplicates later messages by generation.  Wall-clock sleep is
        intentional here because ``rospy.sleep`` also waits on simulated time.
        """
        now = rospy.Time.now()
        if now != rospy.Time() or not bool(
            rospy.get_param("/use_sim_time", False)
        ):
            return now

        rospy.loginfo("Waiting for the first valid simulated-time sample")
        while not rospy.is_shutdown():
            time.sleep(0.01)
            now = rospy.Time.now()
            if now != rospy.Time():
                return now
        raise rospy.ROSInterruptException(
            "shutdown while waiting for valid simulated time"
        )

    def __init__(self):
        get = rospy.get_param
        activation_mode = str(get("~activation/mode", "ready")).strip().lower()
        if activation_mode != "ready":
            raise rospy.ROSInitException(
                "mission activation/mode must be 'ready'"
            )
        self.ready_max_age = float(get("~activation/ready_max_age", 0.50))
        self.ready_future_tolerance = float(
            get("~activation/ready_future_tolerance", 0.05)
        )
        self.ready_before_arm_tolerance = float(
            get("~activation/ready_before_arm_tolerance", 0.0)
        )
        if (
            not math.isfinite(self.ready_max_age)
            or self.ready_max_age <= 0.0
            or not math.isfinite(self.ready_future_tolerance)
            or self.ready_future_tolerance < 0.0
            or not math.isfinite(self.ready_before_arm_tolerance)
            or self.ready_before_arm_tolerance < 0.0
        ):
            raise rospy.ROSInitException(
                "mission readiness timing parameters are invalid"
            )
        self.polygon_diagnostics_enabled = bool(
            get("~diagnostics/polygons_enabled", False)
        )
        self.map_frame = str(get("~pose/map_frame", "map"))
        self.base_frame = str(get("~pose/base_frame", "base_footprint"))
        self.transform_timeout = max(
            0.0, float(get("~pose/transform_timeout", 0.03))
        )
        self.enter_margin = max(0.0, float(get("~zone/enter_margin", 0.03)))
        self.signal_period = max(
            0.05, float(get("~zone/signal_period", 0.20))
        )

        mission_names = [str(name) for name in get("~sequence", [])]
        configured = get("~missions", {})
        self.missions = []
        seen = set()
        for name in mission_names:
            if name in seen:
                raise rospy.ROSInitException(
                    "mission '%s' occurs more than once in sequence" % name
                )
            seen.add(name)
            values = configured.get(name, {})
            polygon = (
                self._parse_polygon(name, values.get("polygon", []))
                if self.polygon_diagnostics_enabled
                else []
            )
            completion_topic = str(values.get("completion_topic", "")).strip()
            if not completion_topic:
                raise rospy.ROSInitException(
                    "mission '%s' needs completion_topic" % name
                )
            self.missions.append(
                {
                    "name": name,
                    "polygon": polygon,
                    "arm_topic": str(
                        values.get("arm_topic", "/mission/arm/" + name)
                    ),
                    "ready_topic": str(
                        values.get("ready_topic", "/mission/ready/" + name)
                    ),
                    "gate_topic": str(
                        values.get("gate_topic", "/mission/enable/" + name)
                    ),
                    "completion_topic": completion_topic,
                    "completion_values": [
                        str(value)
                        for value in values.get("completion_values", ["COMPLETE"])
                    ],
                    # Each mission sets the outside distance required at its
                    # known exit pose and footprint orientation.
                    "clearance_margin": max(
                        0.0,
                        float(values.get("clearance_margin", 0.15)),
                    ),
                    "clearance_topic": str(
                        values.get(
                            "clearance_topic", "/mission/clear/" + name
                        )
                    ),
                    "inside_topic": str(
                        values.get(
                            "inside_topic", "/mission/inside/" + name
                        )
                    ),
                }
            )

        configured_regions = (
            get("~regions", {}) if self.polygon_diagnostics_enabled else {}
        )
        if configured_regions is None:
            configured_regions = {}
        if not isinstance(configured_regions, dict):
            raise rospy.ROSInitException("regions must be a YAML mapping")
        self.regions = []
        for raw_name, values in configured_regions.items():
            name = str(raw_name)
            if not isinstance(values, dict):
                raise rospy.ROSInitException(
                    "region '%s' must be a YAML mapping" % name
                )
            polygon = self._parse_polygon(
                name, values.get("polygon", []), kind="region"
            )
            raw_inside_topic = values.get("inside_topic")
            if raw_inside_topic is None or not str(raw_inside_topic).strip():
                raise rospy.ROSInitException(
                    "region '%s' needs inside_topic" % name
                )
            inside_topic = str(raw_inside_topic).strip()
            self.regions.append(
                {
                    "name": name,
                    "polygon": polygon,
                    "inside_margin": max(
                        0.0, float(values.get("inside_margin", 0.0))
                    ),
                    "inside_topic": inside_topic,
                }
            )

        self.polygon_entries = [
            (("mission", mission["name"]), mission["polygon"])
            for mission in self.missions
        ] + [
            (("region", region["name"]), region["polygon"])
            for region in self.regions
        ]

        now = self._wait_for_initial_ros_time().to_sec()
        # A per-process nonce protects against a latched readiness message from
        # a manager that restarted within the configured freshness window.
        initial_generation = int(time.monotonic_ns() & 0xFFFFFFFF) or 1
        self.sequence = MissionZoneSequence(
            self.missions,
            armed_at=now,
            initial_generation=initial_generation,
        )
        self.lock = threading.RLock()
        self.last_reported_zone = None
        self.tf_buffer = None
        self.tf_listener = None
        self.signal_timer = None
        if self.polygon_diagnostics_enabled:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.state_pub = rospy.Publisher(
            # Preserve the immediate READY -> ACTIVE pair for diagnostics.
            "/mission/state", String, queue_size=4, latch=True
        )
        self.current_pub = rospy.Publisher(
            "/mission/current", String, queue_size=1, latch=True
        )
        self.index_pub = rospy.Publisher(
            "/mission/sequence_index", UInt8, queue_size=1, latch=True
        )
        self.detected_zone_pub = None
        self.pose_pub = None
        self.marker_pub = None
        if self.polygon_diagnostics_enabled:
            self.detected_zone_pub = rospy.Publisher(
                "/mission/detected_zone", String, queue_size=1, latch=True
            )
            self.pose_pub = rospy.Publisher(
                "/mission/map_pose", PoseStamped, queue_size=1
            )
            self.marker_pub = rospy.Publisher(
                "/mission/zones", MarkerArray, queue_size=1, latch=True
            )
        self.arm_publishers = {
            mission["name"]: rospy.Publisher(
                mission["arm_topic"], Header, queue_size=1, latch=True
            )
            for mission in self.missions
        }
        self.gate_publishers = {
            mission["name"]: rospy.Publisher(
                mission["gate_topic"], Bool, queue_size=1, latch=True
            )
            for mission in self.missions
        }
        self.clearance_publishers = {}
        self.inside_publishers = {}
        self.region_inside_publishers = {}
        if self.polygon_diagnostics_enabled:
            self.clearance_publishers = {
                mission["name"]: rospy.Publisher(
                    mission["clearance_topic"], Bool, queue_size=1
                )
                for mission in self.missions
            }
            self.inside_publishers = {
                mission["name"]: rospy.Publisher(
                    mission["inside_topic"], Bool, queue_size=1
                )
                for mission in self.missions
            }
            self.region_inside_publishers = {
                region["name"]: rospy.Publisher(
                    region["inside_topic"], Bool, queue_size=1
                )
                for region in self.regions
            }

        for mission in self.missions:
            rospy.Subscriber(
                mission["ready_topic"],
                Header,
                self.ready_callback,
                callback_args=mission,
                queue_size=1,
            )
            rospy.Subscriber(
                mission["completion_topic"],
                String,
                self.completion_callback,
                callback_args=mission,
                queue_size=1,
            )

        if self.polygon_diagnostics_enabled:
            self.signal_timer = rospy.Timer(
                rospy.Duration(self.signal_period), self.signal_timer_callback
            )
        self._publish_status()
        for publisher in self.inside_publishers.values():
            publisher.publish(Bool(data=False))
        for publisher in self.region_inside_publishers.values():
            publisher.publish(Bool(data=False))
        rospy.loginfo(
            "Mission readiness sequence armed: %s",
            " -> ".join(mission_names),
        )
        if self.polygon_diagnostics_enabled:
            self._publish_markers()
            rospy.loginfo(
                "Optional polygon diagnostics use AMCL TF=%s -> %s",
                self.map_frame,
                self.base_frame,
            )
        if self.polygon_diagnostics_enabled and self.regions:
            rospy.loginfo(
                "Auxiliary map regions ready: %s",
                ", ".join(region["name"] for region in self.regions),
            )

    @staticmethod
    def _parse_polygon(name, raw_polygon, kind="mission"):
        polygon = []
        for vertex in raw_polygon:
            if not isinstance(vertex, (list, tuple)) or len(vertex) < 2:
                raise rospy.ROSInitException(
                    "%s '%s' has an invalid polygon vertex" % (kind, name)
                )
            polygon.append((float(vertex[0]), float(vertex[1])))
        if len(polygon) < 3:
            raise rospy.ROSInitException(
                "%s '%s' polygon needs at least three vertices" % (kind, name)
            )
        return polygon

    def signal_timer_callback(self, _event):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rospy.Time(0),
                rospy.Duration(self.transform_timeout),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as error:
            rospy.logwarn_throttle(
                2.0,
                "Waiting for AMCL %s -> %s transform: %s",
                self.map_frame,
                self.base_frame,
                error,
            )
            return
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        stamp = (
            transform.header.stamp
            if transform.header.stamp != rospy.Time()
            else rospy.Time.now()
        )
        with self.lock:
            self._process_map_pose(
                float(translation.x),
                float(translation.y),
                yaw_from_quaternion(rotation),
                stamp,
            )

    def _process_map_pose(self, map_x, map_y, map_yaw, stamp):
        all_distances = signed_polygon_distances(
            (map_x, map_y), self.polygon_entries
        )
        distances = {
            mission["name"]: all_distances[("mission", mission["name"])]
            for mission in self.missions
        }
        region_distances = {
            region["name"]: all_distances[("region", region["name"])]
            for region in self.regions
        }
        detected_zone = ",".join(
            mission["name"]
            for mission in self.missions
            if distances[mission["name"]] >= 0.0
        )
        if detected_zone != self.last_reported_zone:
            self.last_reported_zone = detected_zone
            self.detected_zone_pub.publish(String(data=detected_zone))
            rospy.loginfo(
                "Map pose entered physical zone: %s",
                detected_zone or "lane",
            )

        # Publish the pose first so consumers that combine a region heartbeat
        # with map heading use the same localization sample, not the previous
        # timer cycle.
        self._publish_pose(stamp, map_x, map_y, map_yaw)
        self._publish_zone_signals(distances)
        self._publish_region_signals(region_distances)

    def _publish_zone_signals(self, distances):
        for mission in self.missions:
            self.inside_publishers[mission["name"]].publish(
                Bool(
                    data=is_inside_with_margin(
                        distances[mission["name"]], self.enter_margin
                    )
                )
            )
            self.clearance_publishers[mission["name"]].publish(
                Bool(
                    data=(
                        distances[mission["name"]]
                        <= -mission["clearance_margin"]
                    )
                )
            )

    def _publish_region_signals(self, distances):
        for region in self.regions:
            self.region_inside_publishers[region["name"]].publish(
                Bool(
                    data=is_inside_with_margin(
                        distances[region["name"]], region["inside_margin"]
                    )
                )
            )

    def ready_callback(self, message, mission):
        """Accept only a fresh readiness sample for the current arm token."""
        received_at = rospy.Time.now().to_sec()
        ready_at = message.stamp.to_sec()
        with self.lock:
            previous_state = self.sequence.state
            previous_name = self.sequence.current_name
            problem = self.sequence.readiness_problem(
                mission["name"],
                message.seq,
                ready_at,
                received_at,
                self.ready_max_age,
                self.ready_future_tolerance,
                self.ready_before_arm_tolerance,
            )
            if problem is not None:
                rospy.logwarn_throttle(
                    1.0,
                    "Rejected mission readiness on %s: %s",
                    mission["ready_topic"],
                    problem,
                )
                return
            if message.frame_id != mission["name"]:
                rospy.logwarn_throttle(
                    1.0,
                    "Rejected readiness frame '%s' on %s; expected '%s'",
                    message.frame_id,
                    mission["ready_topic"],
                    mission["name"],
                )
                return
            if not self.sequence.mark_ready(
                mission["name"],
                message.seq,
                ready_at,
                received_at,
                self.ready_max_age,
                self.ready_future_tolerance,
                self.ready_before_arm_tolerance,
            ):
                return
            self._log_transition(previous_state, previous_name)
            # Publish READY with every enable false before the only legal
            # transition that may open the current mission gate.
            self._publish_status()
            previous_state = self.sequence.state
            previous_name = self.sequence.current_name
            if not self.sequence.activate_current(received_at):
                rospy.logfatal(
                    "Fresh readiness could not activate mission '%s'",
                    mission["name"],
                )
                return
            self._log_transition(previous_state, previous_name)
            self._publish_status()
            self._publish_markers()

    def completion_callback(self, message, mission):
        if message.data not in mission["completion_values"]:
            return
        with self.lock:
            previous_state = self.sequence.state
            previous_name = self.sequence.current_name
            if not self.sequence.complete_current(
                mission["name"], rospy.Time.now().to_sec()
            ):
                return
            self._log_transition(previous_state, previous_name)
            self._publish_status()
            self._publish_markers()

    def _log_transition(self, previous_state, previous_name):
        rospy.loginfo(
            "Mission sequence: %s/%s -> %s/%s",
            previous_state,
            previous_name or "none",
            self.sequence.state,
            self.sequence.current_name or "none",
        )

    def _publish_status(self):
        current = self.sequence.current_name
        now = rospy.Time.now()
        self.state_pub.publish(String(data=self.sequence.state))
        self.current_pub.publish(String(data=current))
        self.index_pub.publish(UInt8(data=min(255, self.sequence.index)))
        armed_states = (
            MissionZoneSequence.ARMED,
            MissionZoneSequence.READY,
            MissionZoneSequence.ACTIVE,
        )
        for mission in self.missions:
            name = mission["name"]
            armed = name == current and self.sequence.state in armed_states
            arm = Header()
            arm.seq = self.sequence.generation if armed else 0
            arm.stamp = (
                rospy.Time.from_sec(self.sequence.armed_at)
                if armed and self.sequence.armed_at is not None
                else now
            )
            arm.frame_id = name
            self.arm_publishers[name].publish(arm)
            self.gate_publishers[name].publish(
                Bool(
                    data=(
                        self.sequence.state == MissionZoneSequence.ACTIVE
                        and name == current
                    )
                )
            )

    def _publish_pose(self, stamp, map_x, map_y, map_yaw):
        message = PoseStamped()
        message.header.stamp = stamp if stamp != rospy.Time() else rospy.Time.now()
        message.header.frame_id = self.map_frame
        message.pose.position.x = map_x
        message.pose.position.y = map_y
        message.pose.orientation.z = math.sin(0.5 * map_yaw)
        message.pose.orientation.w = math.cos(0.5 * map_yaw)
        self.pose_pub.publish(message)

    def _publish_markers(self):
        if not self.polygon_diagnostics_enabled:
            return
        now = rospy.Time.now()
        markers = MarkerArray()
        active_name = self.sequence.current_name
        for index, mission in enumerate(self.missions):
            line = Marker()
            line.header.stamp = now
            line.header.frame_id = self.map_frame
            line.ns = "mission_zone_boundaries"
            line.id = 2 * index
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.025
            line.pose.orientation.w = 1.0
            line.color.a = 0.9
            if mission["name"] in self.sequence.completed:
                line.color.r, line.color.g, line.color.b = 0.3, 0.3, 0.3
            elif mission["name"] == active_name:
                line.color.r, line.color.g, line.color.b = 0.1, 1.0, 0.2
            else:
                line.color.r, line.color.g, line.color.b = 1.0, 0.65, 0.0
            vertices = mission["polygon"] + [mission["polygon"][0]]
            for map_x, map_y in vertices:
                line.points.append(Point(x=map_x, y=map_y, z=0.035))
            markers.markers.append(line)

            label = Marker()
            label.header = line.header
            label.ns = "mission_zone_labels"
            label.id = 2 * index + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.scale.z = 0.11
            label.color = line.color
            label.pose.orientation.w = 1.0
            center_x = sum(point[0] for point in mission["polygon"]) / len(mission["polygon"])
            center_y = sum(point[1] for point in mission["polygon"]) / len(mission["polygon"])
            label.pose.position.x, label.pose.position.y = center_x, center_y
            label.pose.position.z = 0.12
            label.text = "%d. %s" % (index + 1, mission["name"])
            markers.markers.append(label)

        first_region_marker_id = 2 * len(self.missions)
        for index, region in enumerate(self.regions):
            line = Marker()
            line.header.stamp = now
            line.header.frame_id = self.map_frame
            line.ns = "mission_region_boundaries"
            line.id = first_region_marker_id + 2 * index
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.018
            line.pose.orientation.w = 1.0
            line.color.r, line.color.g, line.color.b, line.color.a = (
                0.1,
                0.75,
                1.0,
                0.9,
            )
            vertices = region["polygon"] + [region["polygon"][0]]
            for map_x, map_y in vertices:
                line.points.append(Point(x=map_x, y=map_y, z=0.045))
            markers.markers.append(line)

            label = Marker()
            label.header = line.header
            label.ns = "mission_region_labels"
            label.id = first_region_marker_id + 2 * index + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.scale.z = 0.09
            label.color = line.color
            label.pose.orientation.w = 1.0
            center_x = sum(point[0] for point in region["polygon"]) / len(
                region["polygon"]
            )
            center_y = sum(point[1] for point in region["polygon"]) / len(
                region["polygon"]
            )
            label.pose.position.x, label.pose.position.y = center_x, center_y
            label.pose.position.z = 0.13
            label.text = "region: %s" % region["name"]
            markers.markers.append(label)
        self.marker_pub.publish(markers)


if __name__ == "__main__":
    rospy.init_node("mission_zone_manager")
    MissionZoneManager()
    rospy.spin()
