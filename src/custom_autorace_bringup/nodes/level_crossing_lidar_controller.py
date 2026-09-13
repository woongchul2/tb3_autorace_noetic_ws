#!/usr/bin/env python3
"""Stop for a lowered level-crossing bar detected in a raw LaserScan."""

import math
import threading

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64MultiArray, String
from std_srvs.srv import SetBool

from custom_autorace_bringup.level_crossing import (
    HorizontalBarrierConfig,
    detect_horizontal_barrier,
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
        self.gate_topic = str(
            get(
                prefix + "topics/zone_gate",
                "/mission/enable/level_crossing",
            )
        )
        self.clearance_topic = str(
            get(
                prefix + "topics/zone_clearance",
                "/mission/clear/level_crossing",
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

        self.closed_scans = max(
            1,
            int(get(prefix + "confirmation/closed_scans", 3)),
        )
        self.open_scans = max(
            1,
            int(get(prefix + "confirmation/open_scans", 5)),
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

        self.lock = threading.RLock()
        self.state = self.WAIT_GATE
        self.zone_gate = False
        self.zone_cleared = False
        self.manual_stop = False
        self.mission_has_control = False
        self.revoke_requested = False
        self.stop_requested = False
        self.resume_requested = False

        self.scan_received = None
        self.scan_stamp = None
        self.scan_healthy = False
        self.valid_scan_points = 0
        self.last_confirmation_stamp = None
        self.closed_count = 0
        self.open_count = 0
        self.latest_detection = None
        self.gate_activated_at = None

        self.cmd_pub = rospy.Publisher(
            self.cmd_vel_topic, Twist, queue_size=1
        )
        self.state_pub = rospy.Publisher(
            "/level_crossing/state", String, queue_size=1, latch=True
        )
        self.barrier_pub = rospy.Publisher(
            "/level_crossing/barrier_down", Bool, queue_size=1, latch=True
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
            self.gate_topic, Bool, self.gate_callback, queue_size=1
        )
        rospy.Subscriber(
            self.clearance_topic,
            Bool,
            self.clearance_callback,
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
            "Level-crossing LiDAR controller ready: scan=%s gate=%s",
            self.scan_topic,
            self.gate_topic,
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
        self.open_count = 0
        self.stop_requested = False
        self.resume_requested = False
        self.latest_detection = None

    def gate_callback(self, message):
        with self.lock:
            was_open = self.zone_gate
            self.zone_gate = bool(message.data)
            if self.zone_gate and not was_open:
                self.revoke_requested = False
                self.zone_cleared = False
                self._reset_confirmation()
                self.gate_activated_at = rospy.Time.now()
                self._set_state(self.APPROACH)
            elif (
                not self.zone_gate
                and was_open
                and self.state not in (self.COMPLETE, self.FAILED)
            ):
                self.revoke_requested = True

    def clearance_callback(self, message):
        with self.lock:
            self.zone_cleared = bool(message.data)

    def manual_stop_callback(self, message):
        with self.lock:
            requested = bool(message.data)
            if requested == self.manual_stop:
                return
            self.manual_stop = requested
            self._reset_confirmation()

    @staticmethod
    def _message_stamp(message, received):
        return (
            message.header.stamp
            if message.header.stamp != rospy.Time()
            else received
        )

    def _advance_confirmation(self, detected, stamp):
        gap = None
        if self.last_confirmation_stamp is not None:
            gap = (stamp - self.last_confirmation_stamp).to_sec()
        consecutive = (
            gap is not None and 0.0 < gap <= self.maximum_scan_gap
        )
        self.last_confirmation_stamp = stamp

        if self.state == self.APPROACH:
            self.open_count = 0
            # Once the required consecutive closed scans have been observed,
            # keep the stop request latched until the control timer completes
            # the handoff. A clear scan racing that timer must not cancel a
            # confirmed safety stop.
            if self.stop_requested:
                return
            if not detected:
                self.closed_count = 0
                return
            self.closed_count = self.closed_count + 1 if consecutive else 1
            if self.closed_count >= self.closed_scans:
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
                self.stop_requested = True
            return

        if self.state != self.STOPPED:
            return
        self.resume_requested = False
        self.closed_count = 0
        if detected:
            self.open_count = 0
            return
        self.open_count = self.open_count + 1 if consecutive else 1
        if self.open_count >= self.open_scans:
            self.resume_requested = True

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
            self.closed_count = 0
        elif self.state == self.STOPPED:
            self.open_count = 0
            self.resume_requested = False
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
            if self.zone_gate and self.state in (
                self.APPROACH,
                self.STOPPED,
                self.PASSING,
            ):
                if self.manual_stop:
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
                self._advance_confirmation(detection is not None, stamp)
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
        if not self._set_lane_controller(False):
            self._fail_safe_stop("could not acquire cmd_vel control")
            return False
        # The lane handoff deliberately emits no command. Publish zero in the
        # same control cycle so there is no gap between the two owners.
        self._publish_stop()
        self.stop_requested = False
        self.resume_requested = False
        self.last_confirmation_stamp = None
        self.open_count = 0
        self.barrier_pub.publish(Bool(data=True))
        self._set_state(self.STOPPED)
        detection = self.latest_detection
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
        self.last_confirmation_stamp = None
        self.closed_count = 0
        self.open_count = 0
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
                if self.zone_cleared:
                    self._set_state(self.COMPLETE)
                    rospy.loginfo("Level crossing cleared")
                    return
                if not self._scan_is_fresh(now):
                    self._fail_safe_stop(
                        "LiDAR scan became unhealthy or stale while passing "
                        "the barrier"
                    )
                    return
                if self.stop_requested:
                    self._acquire_and_stop()
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

            # A stopped robot never treats missing/stale data as an open bar.
            self._publish_stop()
            if not self._scan_is_fresh(now):
                rospy.logwarn_throttle(
                    2.0,
                    "Level-crossing scan is unhealthy or stale; keeping "
                    "cmd_vel at zero",
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
