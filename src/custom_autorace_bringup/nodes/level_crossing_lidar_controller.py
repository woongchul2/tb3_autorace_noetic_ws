#!/usr/bin/env python3
"""Locally register, stop for and clear a LiDAR level crossing."""

from collections import deque
import math
import threading

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64MultiArray, Header, String
from std_srvs.srv import SetBool

from custom_autorace_bringup.level_crossing import (
    CrossingFrame,
    HorizontalBarrierConfig,
    compose_pose_2d,
    crossing_progress,
    crossing_rear_clearance,
    detect_horizontal_barrier,
    normalize_angle,
)


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2),
    )


class LevelCrossingLidarController:
    """Own ``cmd_vel`` while the crossing is stopped or fail-safe."""

    WAIT_GATE = "WAIT_GATE"
    APPROACH = "APPROACH"
    STOPPED = "STOPPED"
    PASSING = "PASSING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

    def __init__(self):
        get = rospy.get_param
        prefix = "~level_crossing/"

        self.scan_topic = str(
            get(prefix + "topics/scan", "/scan_mid360_raw")
        )
        self.odom_topic = str(
            get(prefix + "topics/odometry", "/odometry/filtered")
        )
        self.gate_topic = str(
            get(
                prefix + "topics/zone_gate",
                "/mission/enable/level_crossing",
            )
        )
        self.arm_topic = str(
            get(
                prefix + "topics/arm",
                "/mission/arm/level_crossing",
            )
        )
        self.ready_topic = str(
            get(
                prefix + "topics/ready",
                "/mission/ready/level_crossing",
            )
        )
        self.landmark_pose_topic = str(
            get(
                prefix + "topics/landmark_pose",
                "/level_crossing/landmark_pose",
            )
        )
        self.manual_stop_topic = str(
            get(prefix + "topics/manual_stop", "/control/manual_stop")
        )
        self.cmd_vel_topic = str(
            get(prefix + "topics/cmd_vel", "/cmd_vel")
        )
        self.lane_service_name = str(
            get(
                prefix + "topics/lane_control_service",
                "/control/lane_mission_handoff",
            )
        )
        self.lane_stop_service_name = str(
            get(
                prefix + "topics/lane_stop_service",
                "/control/lane_following",
            )
        )

        self.base_frame = str(
            get(prefix + "registration/base_frame", "base_footprint")
        ).lstrip("/")
        self.scan_frame = str(
            get(prefix + "registration/scan_frame", "base_scan")
        ).lstrip("/")
        if not self.base_frame or not self.scan_frame:
            raise rospy.ROSInitException(
                "level-crossing registration frames cannot be empty"
            )
        scan_sensor_pose = tuple(
            float(value)
            for value in get(
                prefix + "registration/scan_sensor_pose",
                [-0.033073, 0.0, 0.0],
            )
        )
        if len(scan_sensor_pose) != 3 or not all(
            math.isfinite(value) for value in scan_sensor_pose
        ):
            raise rospy.ROSInitException(
                "level-crossing scan_sensor_pose needs finite [x, y, yaw_deg]"
            )
        self.scan_sensor_pose = (
            scan_sensor_pose[0],
            scan_sensor_pose[1],
            math.radians(scan_sensor_pose[2]),
        )
        self.allow_barrier_anchor_fallback = bool(
            get(
                prefix + "registration/allow_barrier_anchor_fallback",
                False,
            )
        )
        self.registration_samples = max(
            1, int(get(prefix + "registration/confirmation_samples", 3))
        )
        self.registration_maximum_gap = max(
            0.01,
            float(get(prefix + "registration/maximum_sample_gap", 0.25)),
        )
        self.registration_maximum_position_delta = max(
            0.001,
            float(
                get(prefix + "registration/maximum_position_delta", 0.05)
            ),
        )
        self.registration_maximum_heading_delta = math.radians(
            abs(
                float(
                    get(
                        prefix + "registration/maximum_heading_delta_deg",
                        6.0,
                    )
                )
            )
        )
        self.registration_minimum_forward_distance = max(
            0.0,
            float(
                get(prefix + "registration/minimum_forward_distance", 0.05)
            ),
        )
        self.registration_maximum_forward_distance = max(
            self.registration_minimum_forward_distance + 0.01,
            float(
                get(prefix + "registration/maximum_forward_distance", 1.20)
            ),
        )
        self.registration_pose_stamp_skew = max(
            0.0,
            float(get(prefix + "registration/maximum_pose_stamp_skew", 0.06)),
        )
        self.maximum_future_stamp = max(
            0.0,
            float(get(prefix + "registration/maximum_future_stamp", 0.06)),
        )

        try:
            self.detector_config = HorizontalBarrierConfig(
                min_forward_distance=get(
                    prefix + "detection/minimum_forward_distance", 0.10
                ),
                max_forward_distance=get(
                    prefix + "detection/maximum_forward_distance", 0.60
                ),
                half_width=get(
                    prefix + "detection/corridor_half_width", 0.24
                ),
                max_adjacent_beam_gap=get(
                    prefix + "detection/maximum_adjacent_beam_gap", 2
                ),
                max_point_gap=get(
                    prefix + "detection/maximum_point_gap", 0.08
                ),
                min_points=get(
                    prefix + "detection/minimum_cluster_points", 6
                ),
                min_lateral_span=get(
                    prefix + "detection/minimum_lateral_span", 0.18
                ),
                max_depth_spread=get(
                    prefix + "detection/maximum_depth_spread", 0.09
                ),
            )
        except ValueError as error:
            raise rospy.ROSInitException(
                "invalid level-crossing detection parameters: %s" % error
            )

        self.stop_forward_distance = float(
            get(prefix + "detection/stop_forward_distance", 0.45)
        )
        if (
            not math.isfinite(self.stop_forward_distance)
            or self.stop_forward_distance
            < self.detector_config.min_forward_distance
            or self.stop_forward_distance
            > self.detector_config.max_forward_distance
        ):
            raise rospy.ROSInitException(
                "level-crossing stop_forward_distance must lie inside the "
                "detection forward range"
            )

        self.closed_scans = max(
            1,
            int(get(prefix + "confirmation/closed_scans", 3)),
        )
        self.open_scans = max(
            1,
            int(get(prefix + "confirmation/open_scans", 5)),
        )
        self.arrival_open_scans = max(
            1,
            int(
                get(
                    prefix + "confirmation/open_at_arrival_scans",
                    self.open_scans,
                )
            ),
        )
        self.approach_lost_scans = max(
            1,
            int(get(prefix + "confirmation/approach_lost_scans", 3)),
        )
        self.post_stop_closed_scans = max(
            1,
            int(get(prefix + "confirmation/post_stop_closed_scans", 1)),
        )
        self.stopped_odom_samples = max(
            1,
            int(get(prefix + "confirmation/stopped_odom_samples", 2)),
        )
        self.stopped_linear_velocity = float(
            get(
                prefix + "confirmation/stopped_linear_velocity",
                0.03,
            )
        )
        self.stopped_angular_velocity = float(
            get(
                prefix + "confirmation/stopped_angular_velocity",
                0.10,
            )
        )
        if (
            not math.isfinite(self.stopped_linear_velocity)
            or self.stopped_linear_velocity < 0.0
            or not math.isfinite(self.stopped_angular_velocity)
            or self.stopped_angular_velocity < 0.0
        ):
            raise rospy.ROSInitException(
                "level-crossing stopped velocity thresholds must be finite "
                "and non-negative"
            )
        self.maximum_scan_gap = max(
            0.01,
            float(get(prefix + "confirmation/maximum_scan_gap", 0.25)),
        )
        self.control_period = max(
            0.02, float(get(prefix + "control/period", 0.05))
        )
        self.scan_timeout = max(
            0.05, float(get(prefix + "timeouts/scan", 0.40))
        )
        self.odom_timeout = float(get(prefix + "timeouts/odometry", 0.40))
        if not math.isfinite(self.odom_timeout) or self.odom_timeout < 0.05:
            raise rospy.ROSInitException(
                "level-crossing odometry timeout must be finite and at least "
                "0.05 seconds"
            )
        self.handoff_timeout = max(
            0.05, float(get(prefix + "timeouts/handoff", 0.30))
        )
        self.lane_stop_timeout = max(
            0.05, float(get(prefix + "timeouts/lane_stop", 0.30))
        )
        self.minimum_valid_scan_points = max(
            1,
            int(get(prefix + "detection/minimum_valid_scan_points", 10)),
        )

        self.footprint_front = max(
            0.01, float(get(prefix + "completion/footprint_front", 0.067645))
        )
        self.footprint_rear = max(
            0.01, float(get(prefix + "completion/footprint_rear", 0.118073))
        )
        self.footprint_half_width = max(
            0.01,
            float(get(prefix + "completion/footprint_half_width", 0.0903)),
        )
        self.footprint_padding = max(
            0.0, float(get(prefix + "completion/footprint_padding", 0.010))
        )
        self.completion_clearance_margin = max(
            0.0,
            float(get(prefix + "completion/clearance_margin", 0.15)),
        )

        self.lock = threading.RLock()
        self.state = self.WAIT_GATE
        self.zone_gate = False
        self.gate_requested = False
        self.manual_stop = False
        self.mission_has_control = False
        self.revoke_requested = False
        self.stop_requested = False
        self.resume_requested = False
        self.closed_confirmed = False
        self.stop_detection = None
        self.arrival_open_count = 0
        self.arrival_open_confirmed = False

        self.arm_seq = None
        self.arm_stamp = None
        self.crossing_frame = None
        self.registration_source_stamp = None
        self.registration_candidates = deque(
            maxlen=self.registration_samples
        )
        self.last_registration_stamp = None
        self.ready_published_seq = None

        self.scan_received = None
        self.scan_stamp = None
        self.scan_healthy = False
        self.valid_scan_points = 0
        self.last_confirmation_stamp = None
        self.closed_count = 0
        self.approach_lost_count = 0
        self.open_count = 0
        self.post_stop_closed_count = 0
        self.stationary_count = 0
        self.stationary_confirmed = False
        self.post_stop_closed_confirmed = False
        self.latest_detection = None
        self.gate_activated_at = None
        self.stop_command_stamp = None
        self.stationary_confirmed_stamp = None

        self.odom_received = None
        self.odom_stamp = None
        self.odom_healthy = False
        self.linear_speed = math.inf
        self.angular_speed = math.inf
        self.odom_x = self.odom_y = self.odom_yaw = 0.0
        self.odom_frame = "odom"
        self.odom_child_frame = self.base_frame
        self.odom_history = deque(maxlen=120)

        self.cmd_pub = rospy.Publisher(
            self.cmd_vel_topic, Twist, queue_size=1
        )
        self.state_pub = rospy.Publisher(
            "/level_crossing/state", String, queue_size=1, latch=True
        )
        self.barrier_pub = rospy.Publisher(
            "/level_crossing/barrier_down", Bool, queue_size=1, latch=True
        )
        self.ready_pub = rospy.Publisher(
            self.ready_topic, Header, queue_size=1, latch=True
        )
        self.diagnostics_pub = rospy.Publisher(
            "/level_crossing/diagnostics",
            Float64MultiArray,
            queue_size=1,
        )

        rospy.Subscriber(
            self.scan_topic, LaserScan, self.scan_callback, queue_size=1
        )
        rospy.Subscriber(
            self.odom_topic, Odometry, self.odom_callback, queue_size=1
        )
        rospy.Subscriber(
            self.gate_topic, Bool, self.gate_callback, queue_size=1
        )
        rospy.Subscriber(
            self.arm_topic, Header, self.arm_callback, queue_size=1
        )
        rospy.Subscriber(
            self.landmark_pose_topic,
            PoseStamped,
            self.landmark_pose_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.manual_stop_topic,
            Bool,
            self.manual_stop_callback,
            queue_size=1,
        )
        self.lane_service = rospy.ServiceProxy(
            self.lane_service_name, SetBool
        )
        self.lane_stop_service = rospy.ServiceProxy(
            self.lane_stop_service_name, SetBool
        )
        self.timer = rospy.Timer(
            rospy.Duration(self.control_period), self.control_callback
        )
        rospy.on_shutdown(self.shutdown)

        self._publish_state()
        self.barrier_pub.publish(Bool(data=False))
        rospy.loginfo(
            "Level-crossing LiDAR controller ready: scan=%s odom=%s "
            "arm=%s gate=%s landmark=%s fallback=%s",
            self.scan_topic,
            self.odom_topic,
            self.arm_topic,
            self.gate_topic,
            self.landmark_pose_topic,
            self.allow_barrier_anchor_fallback,
        )

    def _publish_state(self):
        self.state_pub.publish(String(data=self.state))

    def _set_state(self, state):
        if state == self.state:
            return
        self.state = state
        self._publish_state()
        rospy.loginfo("Level-crossing mission state: %s", state)

    def _reset_confirmation(self):
        self.last_confirmation_stamp = None
        self.closed_count = 0
        self.approach_lost_count = 0
        self.open_count = 0
        self.post_stop_closed_count = 0
        self.stop_requested = False
        self.resume_requested = False
        self.closed_confirmed = False
        self.arrival_open_count = 0
        self.arrival_open_confirmed = False
        self.stationary_count = 0
        self.stationary_confirmed = False
        self.post_stop_closed_confirmed = False
        self.latest_detection = None
        self.stop_detection = None
        self.stop_command_stamp = None
        self.stationary_confirmed_stamp = None

    def _reset_stopped_verification(self):
        self.last_confirmation_stamp = None
        self.open_count = 0
        self.post_stop_closed_count = 0
        self.resume_requested = False
        self.stationary_count = 0
        self.stationary_confirmed = False
        self.post_stop_closed_confirmed = False
        self.stationary_confirmed_stamp = None

    def _reset_registration(self):
        self.crossing_frame = None
        self.registration_source_stamp = None
        self.registration_candidates.clear()
        self.last_registration_stamp = None
        self.ready_published_seq = None

    def arm_callback(self, message):
        """Start one source-stamped local registration generation."""
        if message.frame_id != "level_crossing":
            return
        with self.lock:
            sequence = int(message.seq)
            if sequence <= 0:
                if self.state not in (self.WAIT_GATE, self.COMPLETE):
                    rospy.logwarn(
                        "Ignoring level-crossing disarm while %s", self.state
                    )
                    return
                self.arm_seq = None
                self.arm_stamp = None
                self.gate_requested = False
                self.zone_gate = False
                self._reset_registration()
                self._reset_confirmation()
                return
            if self.arm_seq == sequence:
                return
            if self.state not in (self.WAIT_GATE, self.COMPLETE):
                rospy.logwarn(
                    "Ignoring level-crossing arm generation %d while %s",
                    sequence,
                    self.state,
                )
                return
            stamp = (
                message.stamp
                if message.stamp != rospy.Time()
                else rospy.Time.now()
            )
            self.arm_seq = sequence
            self.arm_stamp = stamp
            self.gate_requested = False
            self.zone_gate = False
            self._reset_registration()
            self._reset_confirmation()
            if self.state == self.COMPLETE:
                self._set_state(self.WAIT_GATE)
            rospy.loginfo(
                "Level-crossing registration armed: generation=%d stamp=%.3f",
                sequence,
                stamp.to_sec(),
            )

    def _synchronized_odom_pose(self, stamp):
        if stamp is None or not self.odom_history:
            return None
        candidate = min(
            self.odom_history,
            key=lambda sample: abs((sample[0] - stamp).to_sec()),
        )
        if (
            abs((candidate[0] - stamp).to_sec())
            > self.registration_pose_stamp_skew
        ):
            return None
        return candidate[1:]

    def _pose_in_odom(self, message):
        stamp = message.header.stamp
        source_frame = (message.header.frame_id or "").lstrip("/")
        pose = message.pose
        values = (
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(pose.orientation),
        )
        if not source_frame or not all(math.isfinite(value) for value in values):
            return None
        if source_frame == self.odom_frame:
            return values

        synchronized = self._synchronized_odom_pose(stamp)
        if synchronized is None:
            return None
        base_pose = synchronized[:3]
        odom_frame = synchronized[3]
        child_frame = synchronized[4]
        if odom_frame != self.odom_frame:
            return None
        if source_frame in (self.base_frame, child_frame):
            return compose_pose_2d(base_pose, values)
        if source_frame == self.scan_frame:
            sensor_pose = compose_pose_2d(base_pose, self.scan_sensor_pose)
            return compose_pose_2d(sensor_pose, values)
        return None

    def landmark_pose_callback(self, message):
        """Consume a fixed-post/marker detector's source-stamped frame pose.

        The pose origin is the crossing plane centre and its +x axis points in
        the downstream travel direction.  The producer may publish in odom,
        base or configured scan coordinates.
        """
        stamp = message.header.stamp
        with self.lock:
            if not self._registration_stamp_is_eligible(stamp):
                return
            candidate = self._pose_in_odom(message)
            if candidate is None:
                rospy.logwarn_throttle(
                    1.0,
                    "Waiting for source-stamped level-crossing landmark pose",
                )
                return
            self._submit_registration_candidate(candidate, stamp, "landmark")

    def _registration_stamp_is_eligible(self, stamp):
        if self.arm_seq is None or self.arm_seq <= 0 or self.arm_stamp is None:
            return False
        if stamp == rospy.Time() or stamp <= self.arm_stamp:
            return False
        now = rospy.Time.now()
        if (stamp - now).to_sec() > self.maximum_future_stamp:
            return False
        if self.last_registration_stamp is not None and (
            stamp <= self.last_registration_stamp
        ):
            return False
        return True

    def _submit_registration_candidate(self, pose, stamp, source):
        if self.crossing_frame is not None:
            return False
        synchronized = self._synchronized_odom_pose(stamp)
        if synchronized is None:
            return False
        frame = CrossingFrame(*pose)
        robot_pose = synchronized[:3]
        forward_distance = -crossing_progress(frame, robot_pose)
        if not (
            self.registration_minimum_forward_distance
            <= forward_distance
            <= self.registration_maximum_forward_distance
        ):
            self.registration_candidates.clear()
            self.last_registration_stamp = stamp
            return False

        if self.registration_candidates:
            previous_stamp, previous = self.registration_candidates[-1]
            gap = (stamp - previous_stamp).to_sec()
            position_delta = math.hypot(
                frame.x - previous.x, frame.y - previous.y
            )
            heading_delta = abs(normalize_angle(frame.yaw - previous.yaw))
            if (
                gap <= 0.0
                or gap > self.registration_maximum_gap
                or position_delta > self.registration_maximum_position_delta
                or heading_delta > self.registration_maximum_heading_delta
            ):
                self.registration_candidates.clear()

        self.registration_candidates.append((stamp, frame))
        self.last_registration_stamp = stamp
        if len(self.registration_candidates) < self.registration_samples:
            return False

        count = float(len(self.registration_candidates))
        x = sum(item[1].x for item in self.registration_candidates) / count
        y = sum(item[1].y for item in self.registration_candidates) / count
        sine = sum(
            math.sin(item[1].yaw) for item in self.registration_candidates
        )
        cosine = sum(
            math.cos(item[1].yaw) for item in self.registration_candidates
        )
        self.crossing_frame = CrossingFrame(x, y, math.atan2(sine, cosine))
        self.registration_source_stamp = stamp
        ready = Header()
        ready.seq = int(self.arm_seq)
        ready.stamp = stamp
        # The sequence manager uses frame_id as mission identity.  The
        # registered target frame remains available in ``self.odom_frame``.
        ready.frame_id = "level_crossing"
        self.ready_pub.publish(ready)
        self.ready_published_seq = self.arm_seq
        rospy.loginfo(
            "Level crossing locally registered from %s: generation=%d "
            "frame=(%.3f, %.3f, %.1fdeg)",
            source,
            self.arm_seq,
            self.crossing_frame.x,
            self.crossing_frame.y,
            math.degrees(self.crossing_frame.yaw),
        )
        if self.gate_requested and self.state == self.WAIT_GATE:
            self._activate_gate()
        return True

    def _barrier_registration_candidate(self, message, detection):
        if (
            not self.allow_barrier_anchor_fallback
            or detection is None
            or self.crossing_frame is not None
            or not self._registration_stamp_is_eligible(message.header.stamp)
        ):
            return
        source_frame = (message.header.frame_id or "").lstrip("/")
        if source_frame and source_frame != self.scan_frame:
            return
        synchronized = self._synchronized_odom_pose(message.header.stamp)
        if synchronized is None:
            return
        base_pose = synchronized[:3]
        sensor_pose = compose_pose_2d(base_pose, self.scan_sensor_pose)
        candidate = compose_pose_2d(
            sensor_pose,
            (
                detection.forward_distance,
                detection.lateral_center,
                0.0,
            ),
        )
        self._submit_registration_candidate(
            candidate, message.header.stamp, "Gazebo barrier fallback"
        )

    def _activate_gate(self):
        self.revoke_requested = False
        self._reset_confirmation()
        self.gate_activated_at = rospy.Time.now()
        self.zone_gate = True
        self._set_state(self.APPROACH)

    def gate_callback(self, message):
        with self.lock:
            was_open = self.zone_gate
            self.gate_requested = bool(message.data)
            if self.gate_requested and not was_open:
                if (
                    self.crossing_frame is None
                    or self.ready_published_seq != self.arm_seq
                ):
                    rospy.logwarn_throttle(
                        1.0,
                        "Ignoring level-crossing enable before local "
                        "registration readiness",
                    )
                    return
                if self.mission_has_control:
                    # A gate publisher may restart or briefly flicker while the
                    # robot is stopped. Never abandon the zero-velocity owner or
                    # jump back to APPROACH in that situation.
                    self.zone_gate = True
                    self.revoke_requested = False
                    self.last_confirmation_stamp = None
                    self.open_count = 0
                    self.resume_requested = False
                    rospy.logwarn(
                        "Level-crossing gate returned while mission control is "
                        "active; remaining stopped"
                    )
                    return
                self._activate_gate()
            elif not self.gate_requested and was_open:
                self.zone_gate = False
                if self.state in (self.COMPLETE, self.FAILED):
                    return
                if self.mission_has_control:
                    self.revoke_requested = False
                    self.last_confirmation_stamp = None
                    self.open_count = 0
                    self.resume_requested = False
                    rospy.logerr(
                        "Level-crossing gate was revoked while stopped; "
                        "holding cmd_vel control fail-closed"
                    )
                else:
                    self.revoke_requested = True

    def manual_stop_callback(self, message):
        with self.lock:
            requested = bool(message.data)
            if requested == self.manual_stop:
                return
            self.manual_stop = requested
            # A manual pause invalidates partial consecutive sequences.  Once
            # the stopped pose and closed bar have both been verified, however,
            # retain those safety facts: clearing them could strand the robot if
            # the bar opens while the manual pause is active.
            if self.state == self.STOPPED:
                self.last_confirmation_stamp = None
                self.open_count = 0
                self.resume_requested = False
                if not self.post_stop_closed_confirmed:
                    self.post_stop_closed_count = 0
            else:
                self._reset_confirmation()

    @staticmethod
    def _message_stamp(message, received):
        return (
            message.header.stamp
            if message.header.stamp != rospy.Time()
            else received
        )

    def _plane_forward_distance_at(self, stamp):
        if self.crossing_frame is None:
            return None
        synchronized = self._synchronized_odom_pose(stamp)
        if synchronized is None:
            return None
        return -crossing_progress(self.crossing_frame, synchronized[:3])

    def _crossing_clearance(self):
        if self.crossing_frame is None or not self.odom_healthy:
            return -math.inf
        return crossing_rear_clearance(
            self.crossing_frame,
            (self.odom_x, self.odom_y, self.odom_yaw),
            self.footprint_front,
            self.footprint_rear,
            self.footprint_half_width,
            self.footprint_padding,
        )

    def _advance_confirmation(self, detection, stamp):
        detected = detection is not None
        gap = None
        if self.last_confirmation_stamp is not None:
            gap = (stamp - self.last_confirmation_stamp).to_sec()
        consecutive = (
            gap is not None and 0.0 < gap <= self.maximum_scan_gap
        )
        self.last_confirmation_stamp = stamp

        if self.state == self.APPROACH:
            self.open_count = 0
            if self.stop_requested:
                return

            if not self.closed_confirmed:
                if not detected:
                    self.closed_count = 0
                    self.approach_lost_count = 0
                    plane_distance = self._plane_forward_distance_at(stamp)
                    observable = bool(
                        plane_distance is not None
                        and self.detector_config.min_forward_distance
                        <= plane_distance
                        <= self.detector_config.max_forward_distance
                    )
                    if not observable:
                        self.arrival_open_count = 0
                        self.arrival_open_confirmed = False
                        return
                    self.arrival_open_count = (
                        self.arrival_open_count + 1 if consecutive else 1
                    )
                    if self.arrival_open_count >= self.arrival_open_scans:
                        self.arrival_open_confirmed = True
                    return
                self.arrival_open_count = 0
                self.arrival_open_confirmed = False
                self.closed_count = (
                    self.closed_count + 1 if consecutive else 1
                )
                if self.closed_count < self.closed_scans:
                    return
                self.closed_confirmed = True
                self.stop_detection = detection
                rospy.loginfo(
                    "Lowered barrier confirmed at %.3f m; approaching %.3f m "
                    "stop trigger",
                    detection.forward_distance,
                    self.stop_forward_distance,
                )

            # Confirmation is intentionally completed before the stop trigger.
            # Ignore one-off cluster dropouts, but stop conservatively after a
            # short consecutive loss instead of continuing indefinitely toward
            # an object whose range can no longer be measured.
            if not detected:
                self.approach_lost_count = (
                    self.approach_lost_count + 1 if consecutive else 1
                )
                if self.approach_lost_count >= self.approach_lost_scans:
                    self.stop_requested = True
                    rospy.logwarn(
                        "Confirmed barrier was lost for %d scans; stopping "
                        "conservatively",
                        self.approach_lost_count,
                    )
                return
            self.approach_lost_count = 0
            self.stop_detection = detection
            plane_distance = self._plane_forward_distance_at(stamp)
            if (
                plane_distance is not None
                and plane_distance <= self.stop_forward_distance
            ):
                self.stop_requested = True
            return

        if self.state == self.PASSING:
            self.resume_requested = False
            self.open_count = 0
            if self.stop_requested:
                return
            if not detected:
                self.closed_count = 0
                return
            self.closed_count = self.closed_count + 1 if consecutive else 1
            if self.closed_count >= self.closed_scans:
                self.stop_detection = detection
                self.stop_requested = True
            return

        if self.state != self.STOPPED:
            return

        self.resume_requested = False
        self.closed_count = 0

        # Clear scans observed while the base is still settling are not proof
        # that the bar opened.  Arm open confirmation only after distinct odom
        # samples report a stop and the closed bar is seen again at that pose.
        if not self.stationary_confirmed:
            self.open_count = 0
            self.post_stop_closed_count = 0
            return

        if not self.post_stop_closed_confirmed:
            self.open_count = 0
            if not detected:
                self.post_stop_closed_count = 0
                return
            self.post_stop_closed_count = (
                self.post_stop_closed_count + 1 if consecutive else 1
            )
            if self.post_stop_closed_count >= self.post_stop_closed_scans:
                self.post_stop_closed_confirmed = True
                self.last_confirmation_stamp = stamp
                rospy.loginfo(
                    "Stopped pose confirmed with lowered barrier at %.3f m",
                    detection.forward_distance,
                )
            return

        if detected:
            self.open_count = 0
            return
        self.open_count = self.open_count + 1 if consecutive else 1
        if self.open_count >= self.open_scans:
            self.resume_requested = True

    def odom_callback(self, message):
        received = rospy.Time.now()
        stamp = self._message_stamp(message, received)
        position_x = float(message.pose.pose.position.x)
        position_y = float(message.pose.pose.position.y)
        yaw = yaw_from_quaternion(message.pose.pose.orientation)
        linear_x = float(message.twist.twist.linear.x)
        linear_y = float(message.twist.twist.linear.y)
        angular_z = float(message.twist.twist.angular.z)
        odom_frame = (message.header.frame_id or "odom").lstrip("/")
        child_frame = (
            message.child_frame_id or self.base_frame
        ).lstrip("/")
        healthy = all(
            math.isfinite(value)
            for value in (
                position_x,
                position_y,
                yaw,
                linear_x,
                linear_y,
                angular_z,
            )
        )
        linear_speed = (
            math.hypot(linear_x, linear_y) if healthy else math.inf
        )
        angular_speed = abs(angular_z) if healthy else math.inf

        with self.lock:
            if self.odom_stamp is not None and stamp <= self.odom_stamp:
                rospy.logwarn_throttle(
                    2.0,
                    "Rejecting repeated or out-of-order level-crossing odometry",
                )
                return
            source_gap = (
                (stamp - self.odom_stamp).to_sec()
                if self.odom_stamp is not None
                else None
            )
            receipt_gap = (
                (received - self.odom_received).to_sec()
                if self.odom_received is not None
                else None
            )
            if self.state == self.STOPPED and (
                (source_gap is not None and source_gap > self.odom_timeout)
                or (
                    receipt_gap is not None
                    and receipt_gap > self.odom_timeout
                )
            ):
                # Motion during an odometry outage is unknowable. Require the
                # full stopped-and-closed sequence again after recovery.
                self._reset_stopped_verification()
            self.odom_received = received
            self.odom_stamp = stamp
            self.odom_healthy = healthy
            self.linear_speed = linear_speed
            self.angular_speed = angular_speed
            self.odom_x = position_x
            self.odom_y = position_y
            self.odom_yaw = yaw
            self.odom_frame = odom_frame
            self.odom_child_frame = child_frame
            if healthy:
                self.odom_history.append(
                    (
                        stamp,
                        position_x,
                        position_y,
                        yaw,
                        odom_frame,
                        child_frame,
                    )
                )

            if self.state != self.STOPPED:
                return
            # A stationary sample from before the zero command cannot prove
            # that braking has completed, even if callback delivery was late.
            if (
                self.stop_command_stamp is not None
                and stamp <= self.stop_command_stamp
            ):
                return
            stopped = (
                healthy
                and linear_speed <= self.stopped_linear_velocity
                and angular_speed <= self.stopped_angular_velocity
            )
            if not stopped:
                self._reset_stopped_verification()
                return
            if self.stationary_confirmed:
                return
            self.stationary_count += 1
            if (
                not self.stationary_confirmed
                and self.stationary_count >= self.stopped_odom_samples
            ):
                self.stationary_confirmed = True
                self.stationary_confirmed_stamp = stamp
                self.last_confirmation_stamp = None
                rospy.loginfo(
                    "Level-crossing base stopped: linear=%.3f m/s "
                    "angular=%.3f rad/s",
                    linear_speed,
                    angular_speed,
                )

    def _publish_diagnostics(self, detected):
        detection = self.latest_detection
        message = Float64MultiArray()
        message.data = [
            1.0 if detected else 0.0,
            (
                float(detection.forward_distance)
                if detection is not None
                else math.nan
            ),
            (
                float(detection.lateral_span)
                if detection is not None
                else 0.0
            ),
            (
                float(detection.depth_spread)
                if detection is not None
                else 0.0
            ),
            (
                float(detection.point_count)
                if detection is not None
                else 0.0
            ),
            float(self.closed_count),
            float(self.open_count),
            float(self.valid_scan_points),
        ]
        self.diagnostics_pub.publish(message)

    @staticmethod
    def _valid_range_count(message):
        count = 0
        for raw_range in message.ranges:
            try:
                distance = float(raw_range)
            except (TypeError, ValueError, OverflowError):
                continue
            if (
                math.isfinite(distance)
                and message.range_min <= distance <= message.range_max
            ):
                count += 1
        return count

    def _invalidate_confirmation(self):
        self.last_confirmation_stamp = None
        if self.state == self.APPROACH and not self.stop_requested:
            if not self.closed_confirmed:
                self.closed_count = 0
                self.arrival_open_count = 0
                self.arrival_open_confirmed = False
            self.approach_lost_count = 0
        elif self.state == self.STOPPED:
            self.open_count = 0
            self.resume_requested = False
            if not self.post_stop_closed_confirmed:
                self.post_stop_closed_count = 0
        elif self.state == self.PASSING and not self.stop_requested:
            self.closed_count = 0

    def scan_callback(self, message):
        received = rospy.Time.now()
        if not message.ranges:
            return
        stamp = self._message_stamp(message, received)
        try:
            detection = detect_horizontal_barrier(
                message.ranges,
                message.angle_min,
                message.angle_increment,
                message.range_min,
                message.range_max,
                self.detector_config,
            )
        except (TypeError, ValueError) as error:
            rospy.logwarn_throttle(
                2.0, "Rejecting invalid level-crossing scan: %s", error
            )
            return
        valid_scan_points = self._valid_range_count(message)
        scan_healthy = valid_scan_points >= self.minimum_valid_scan_points

        with self.lock:
            if self.scan_stamp is not None and stamp <= self.scan_stamp:
                rospy.logwarn_throttle(
                    2.0,
                    "Rejecting repeated or out-of-order level-crossing scan",
                )
                return
            self.scan_received = received
            self.scan_stamp = stamp
            self.scan_healthy = scan_healthy
            self.valid_scan_points = valid_scan_points
            self.latest_detection = detection
            if scan_healthy:
                self._barrier_registration_candidate(message, detection)
            if self.zone_gate and self.state in (
                self.APPROACH,
                self.STOPPED,
                self.PASSING,
            ):
                if self.manual_stop:
                    self._publish_diagnostics(detection is not None)
                    return
                # A queued scan captured before gate activation must not count
                # toward the initial closed-bar confirmation.
                if (
                    self.state == self.APPROACH
                    and self.gate_activated_at is not None
                    and stamp <= self.gate_activated_at
                ):
                    self._publish_diagnostics(detection is not None)
                    return
                # Likewise, only a scan captured after the odometry-confirmed
                # stop can satisfy the mandatory post-stop closed observation.
                if (
                    self.state == self.STOPPED
                    and self.stationary_confirmed_stamp is not None
                    and stamp <= self.stationary_confirmed_stamp
                ):
                    self._publish_diagnostics(detection is not None)
                    return
                if not scan_healthy:
                    self._invalidate_confirmation()
                    rospy.logwarn_throttle(
                        2.0,
                        "Level-crossing scan has only %d valid ranges; "
                        "confirmation is paused",
                        valid_scan_points,
                    )
                    self._publish_diagnostics(detection is not None)
                    return
                self._advance_confirmation(detection, stamp)
                self._publish_diagnostics(detection is not None)

    def _scan_is_fresh(self, now):
        if self.scan_received is None or self.scan_stamp is None:
            return False
        source_age = (now - self.scan_stamp).to_sec()
        receipt_age = (now - self.scan_received).to_sec()
        return (
            self.scan_healthy
            and 0.0 <= source_age <= self.scan_timeout
            and 0.0 <= receipt_age <= self.scan_timeout
        )

    def _odom_is_fresh(self, now):
        if self.odom_received is None or self.odom_stamp is None:
            return False
        source_age = (now - self.odom_stamp).to_sec()
        receipt_age = (now - self.odom_received).to_sec()
        return (
            self.odom_healthy
            and 0.0 <= source_age <= self.odom_timeout
            and 0.0 <= receipt_age <= self.odom_timeout
        )

    def _set_lane_controller(self, enabled):
        try:
            rospy.wait_for_service(
                self.lane_service_name, timeout=self.handoff_timeout
            )
            response = self.lane_service(enabled)
            if not response.success:
                raise rospy.ServiceException(response.message)
            self.mission_has_control = not enabled
            return True
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logerr("Level-crossing cmd_vel handoff failed: %s", error)
            return False

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())

    def _stop_lane_controller(self):
        try:
            rospy.wait_for_service(
                self.lane_stop_service_name, timeout=self.lane_stop_timeout
            )
            response = self.lane_stop_service(False)
            if not response.success:
                raise rospy.ServiceException(response.message)
            return True
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logerr(
                "Level-crossing lane emergency stop failed: %s", error
            )
            return False

    def _acquire_and_stop(self):
        detection = self.stop_detection or self.latest_detection
        if not self._set_lane_controller(False):
            self._fail_safe_stop("could not acquire cmd_vel control")
            return False
        # The lane handoff deliberately emits no command. Publish zero in the
        # same control cycle so there is no gap between the two owners.
        self._publish_stop()
        self.stop_command_stamp = rospy.Time.now()
        self.stop_requested = False
        self.resume_requested = False
        self.closed_confirmed = False
        self.closed_count = 0
        self.approach_lost_count = 0
        self.last_confirmation_stamp = None
        self._reset_stopped_verification()
        self.barrier_pub.publish(Bool(data=True))
        self._set_state(self.STOPPED)
        if detection is not None:
            rospy.loginfo(
                "Lowered barrier confirmed at %.3f m, span %.3f m; stopped",
                detection.forward_distance,
                detection.lateral_span,
            )
        return True

    def _release_after_open(self):
        self._publish_stop()
        if not self._set_lane_controller(True):
            self._fail_safe_stop(
                "could not return cmd_vel control after the barrier opened"
            )
            return
        self.resume_requested = False
        self._reset_confirmation()
        self.barrier_pub.publish(Bool(data=False))
        self._set_state(self.PASSING)
        rospy.loginfo(
            "Raised barrier confirmed; lane controller owns cmd_vel while "
            "clearing the crossing"
        )

    def _revoke(self):
        if self.mission_has_control:
            self._publish_stop()
            if not self._set_lane_controller(True):
                self.revoke_requested = False
                self._fail_safe_stop(
                    "could not return cmd_vel control after gate revocation"
                )
                return
        self.revoke_requested = False
        self._reset_confirmation()
        self.gate_activated_at = None
        self.barrier_pub.publish(Bool(data=False))
        self._set_state(self.WAIT_GATE)

    def _fail_safe_stop(self, reason):
        # A failed service response is ambiguous: the request may already have
        # changed the lane controller. Reassert mission ownership idempotently;
        # if that is unavailable, disable lane following through its independent
        # stop service before continuously publishing zero velocity.
        if not self._set_lane_controller(False):
            self._stop_lane_controller()
            self.mission_has_control = True
        self._publish_stop()
        self._set_state(self.FAILED)
        rospy.logerr("Level-crossing mission failed: %s", reason)

    def control_callback(self, _event):
        with self.lock:
            if self.manual_stop:
                if self.mission_has_control:
                    self._publish_stop()
                return
            if self.revoke_requested:
                self._revoke()
                return
            if self.state in (self.WAIT_GATE, self.COMPLETE):
                return
            if self.state == self.FAILED:
                if self.mission_has_control:
                    self._publish_stop()
                return

            now = rospy.Time.now()
            if self.state == self.PASSING:
                if not self._scan_is_fresh(now):
                    self._fail_safe_stop(
                        "LiDAR scan became unhealthy or stale while passing "
                        "the barrier"
                    )
                    return
                if not self._odom_is_fresh(now):
                    self._fail_safe_stop(
                        "odometry became unhealthy or stale while passing "
                        "the barrier"
                    )
                    return
                if self.stop_requested:
                    self._acquire_and_stop()
                    return
                clearance = self._crossing_clearance()
                if clearance >= self.completion_clearance_margin:
                    self._set_state(self.COMPLETE)
                    rospy.loginfo(
                        "Level crossing locally cleared by %.3f m",
                        clearance,
                    )
                return
            if self.state == self.APPROACH:
                if not self._scan_is_fresh(now):
                    if (
                        self.gate_activated_at is not None
                        and (now - self.gate_activated_at).to_sec()
                        <= self.scan_timeout
                    ):
                        return
                    self._fail_safe_stop(
                        "LiDAR scan became unhealthy or stale while "
                        "approaching the barrier"
                    )
                    return
                if self.stop_requested:
                    self._acquire_and_stop()
                    return
                if self.arrival_open_confirmed:
                    self._set_state(self.PASSING)
                    rospy.loginfo(
                        "Level crossing was already open; lane controller "
                        "continues through the registered plane"
                    )
                return

            # A stopped robot never treats missing/stale data as an open bar.
            self._publish_stop()
            odom_fresh = self._odom_is_fresh(now)
            if not odom_fresh:
                # Do this even when the scan is also stale. Otherwise a later
                # single odom sample could revive verification from before an
                # unobserved-motion interval.
                self._reset_stopped_verification()
            if not self._scan_is_fresh(now):
                rospy.logwarn_throttle(
                    2.0,
                    "Level-crossing scan is unhealthy or stale; keeping "
                    "cmd_vel at zero",
                )
                return
            if not odom_fresh or not self.stationary_confirmed:
                self.resume_requested = False
                self.open_count = 0
                rospy.logwarn_throttle(
                    2.0,
                    "Level-crossing odometry is moving, unhealthy or stale; "
                    "keeping cmd_vel at zero",
                )
                return
            if self.resume_requested:
                self._release_after_open()

    def shutdown(self):
        with self.lock:
            if self.mission_has_control:
                self._publish_stop()
        rospy.loginfo("Level-crossing LiDAR controller stopped")


if __name__ == "__main__":
    rospy.init_node("level_crossing_lidar_controller")
    LevelCrossingLidarController()
    rospy.spin()
