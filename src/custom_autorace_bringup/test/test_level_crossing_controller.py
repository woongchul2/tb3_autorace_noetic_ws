#!/usr/bin/env python3

import math
import sys
import threading
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


NODE_DIR = Path(__file__).resolve().parents[1] / "nodes"
if str(NODE_DIR) not in sys.path:
    sys.path.insert(0, str(NODE_DIR))

import level_crossing_lidar_controller as controller_module
from custom_autorace_bringup.level_crossing import (
    CrossingFrame,
    HorizontalBarrierConfig,
)
from geometry_msgs.msg import PoseStamped
from level_crossing_lidar_controller import LevelCrossingLidarController
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Header


class RecordingPublisher:
    def __init__(self, events=None, name=None):
        self.messages = []
        self.events = events
        self.name = name

    def publish(self, message):
        self.messages.append(message)
        if self.events is not None:
            self.events.append(("publish", self.name, message))


class RecordingLaneService:
    def __init__(self, events, outcomes=None):
        self.calls = []
        self.events = events
        self.outcomes = list(outcomes or [])

    def __call__(self, enabled):
        enabled = bool(enabled)
        self.calls.append(enabled)
        self.events.append(("handoff", enabled))
        success = self.outcomes.pop(0) if self.outcomes else True
        return SimpleNamespace(success=success, message="ok" if success else "failed")


class RecordingLaneStopService:
    def __init__(self, events):
        self.calls = []
        self.events = events

    def __call__(self, enabled):
        enabled = bool(enabled)
        self.calls.append(enabled)
        self.events.append(("lane_stop", enabled))
        return SimpleNamespace(success=True, message="stopped")


class LevelCrossingLidarControllerTest(unittest.TestCase):
    def setUp(self):
        self.seconds = 10.0
        self.time_patch = mock.patch.object(
            controller_module.rospy.Time,
            "now",
            side_effect=lambda: controller_module.rospy.Time.from_sec(
                self.seconds
            ),
        )
        self.time_patch.start()
        self.addCleanup(self.time_patch.stop)
        self.wait_patch = mock.patch.object(
            controller_module.rospy, "wait_for_service", return_value=None
        )
        self.wait_patch.start()
        self.addCleanup(self.wait_patch.stop)

    def advance(self, seconds=0.10):
        self.seconds += seconds

    def make_controller(self, closed_scans=3, open_scans=5):
        controller = LevelCrossingLidarController.__new__(
            LevelCrossingLidarController
        )
        controller.lock = threading.RLock()
        controller.state = controller.WAIT_GATE
        controller.zone_gate = False
        controller.gate_requested = False
        controller.manual_stop = False
        controller.mission_has_control = False
        controller.revoke_requested = False
        controller.stop_requested = False
        controller.resume_requested = False
        controller.closed_confirmed = False
        controller.stop_detection = None
        controller.arrival_open_count = 0
        controller.arrival_open_confirmed = False

        controller.arm_seq = None
        controller.arm_stamp = None
        controller.crossing_frame = None
        controller.registration_source_stamp = None
        controller.registration_samples = 1
        controller.registration_candidates = deque(maxlen=1)
        controller.last_registration_stamp = None
        controller.ready_published_seq = None

        controller.scan_received = None
        controller.scan_stamp = None
        controller.scan_healthy = False
        controller.valid_scan_points = 0
        controller.last_confirmation_stamp = None
        controller.closed_count = 0
        controller.approach_lost_count = 0
        controller.open_count = 0
        controller.post_stop_closed_count = 0
        controller.stationary_count = 0
        controller.stationary_confirmed = False
        controller.post_stop_closed_confirmed = False
        controller.latest_detection = None
        controller.gate_activated_at = None
        controller.stop_command_stamp = None
        controller.stationary_confirmed_stamp = None

        controller.odom_received = None
        controller.odom_stamp = None
        controller.odom_healthy = False
        controller.linear_speed = math.inf
        controller.angular_speed = math.inf
        controller.odom_x = controller.odom_y = controller.odom_yaw = 0.0
        controller.odom_frame = "odom"
        controller.odom_child_frame = "base_footprint"
        controller.odom_history = []

        controller.closed_scans = closed_scans
        controller.open_scans = open_scans
        controller.arrival_open_scans = open_scans
        controller.approach_lost_scans = 3
        controller.post_stop_closed_scans = 1
        controller.stopped_odom_samples = 2
        controller.stopped_linear_velocity = 0.03
        controller.stopped_angular_velocity = 0.10
        controller.maximum_scan_gap = 0.25
        controller.scan_timeout = 0.40
        controller.odom_timeout = 0.40
        controller.handoff_timeout = 0.30
        controller.lane_stop_timeout = 0.30
        controller.minimum_valid_scan_points = 10
        controller.stop_forward_distance = 0.45
        controller.base_frame = "base_footprint"
        controller.scan_frame = "base_scan"
        controller.scan_sensor_pose = (0.0, 0.0, 0.0)
        controller.allow_barrier_anchor_fallback = False
        controller.registration_maximum_gap = 0.25
        controller.registration_maximum_position_delta = 0.05
        controller.registration_maximum_heading_delta = math.radians(6.0)
        controller.registration_minimum_forward_distance = 0.05
        controller.registration_maximum_forward_distance = 1.20
        controller.registration_pose_stamp_skew = 0.06
        controller.maximum_future_stamp = 0.06
        controller.footprint_front = 0.067645
        controller.footprint_rear = 0.118073
        controller.footprint_half_width = 0.0903
        controller.footprint_padding = 0.010
        controller.completion_clearance_margin = 0.15
        controller.detector_config = HorizontalBarrierConfig(
            min_forward_distance=0.10,
            max_forward_distance=0.60,
            half_width=0.24,
            max_adjacent_beam_gap=2,
            max_point_gap=0.08,
            min_points=6,
            min_lateral_span=0.18,
            max_depth_spread=0.09,
        )

        controller.lane_service_name = "/control/lane_mission_handoff"
        controller.lane_stop_service_name = "/control/lane_following"
        controller.events = []
        controller.lane_service = RecordingLaneService(controller.events)
        controller.lane_stop_service = RecordingLaneStopService(
            controller.events
        )
        controller.cmd_pub = RecordingPublisher(controller.events, "/cmd_vel")
        controller.state_pub = RecordingPublisher()
        controller.barrier_pub = RecordingPublisher()
        controller.ready_pub = RecordingPublisher()
        controller.diagnostics_pub = RecordingPublisher()
        return controller

    def scan(self, barrier_down, forward_distance=0.40):
        """Build a scan whose forward direction is not the array midpoint."""
        message = LaserScan()
        message.header.stamp = controller_module.rospy.Time.from_sec(
            self.seconds
        )
        message.header.frame_id = "base_scan"
        message.angle_min = -2.0
        message.angle_increment = 0.10
        message.angle_max = message.angle_min + 59 * message.angle_increment
        message.range_min = 0.05
        message.range_max = 10.0
        message.ranges = [math.inf] * 60
        # Ten finite background returns make this a healthy frame while staying
        # outside the forward detection corridor.
        for index in range(10):
            message.ranges[index] = 2.0
        if barrier_down:
            # With the metadata above, beams 17..23 cover -0.3..+0.3 rad.
            # The ranges describe a transverse bar at the requested x distance.
            for index in range(17, 24):
                angle = message.angle_min + index * message.angle_increment
                message.ranges[index] = forward_distance / math.cos(angle)
        return message

    def publish_scan(self, controller, barrier_down, forward_distance=0.40):
        self.advance()
        if (
            controller.state == controller.APPROACH
            and controller.crossing_frame is not None
        ):
            frame = controller.crossing_frame
            self.publish_odom_at_current_time(
                controller,
                x=frame.x - forward_distance * math.cos(frame.yaw),
                y=frame.y - forward_distance * math.sin(frame.yaw),
                yaw=frame.yaw,
            )
        controller.scan_callback(self.scan(barrier_down, forward_distance))

    def odom(
        self,
        linear_x=0.0,
        linear_y=0.0,
        angular_z=0.0,
        x=0.0,
        y=0.0,
        yaw=0.0,
    ):
        message = Odometry()
        message.header.stamp = controller_module.rospy.Time.from_sec(
            self.seconds
        )
        message.header.frame_id = "odom"
        message.child_frame_id = "base_footprint"
        message.pose.pose.position.x = x
        message.pose.pose.position.y = y
        message.pose.pose.orientation.z = math.sin(0.5 * yaw)
        message.pose.pose.orientation.w = math.cos(0.5 * yaw)
        message.twist.twist.linear.x = linear_x
        message.twist.twist.linear.y = linear_y
        message.twist.twist.angular.z = angular_z
        return message

    def publish_odom_at_current_time(
        self,
        controller,
        linear_x=0.0,
        linear_y=0.0,
        angular_z=0.0,
        x=None,
        y=None,
        yaw=None,
    ):
        x = controller.odom_x if x is None else x
        y = controller.odom_y if y is None else y
        yaw = controller.odom_yaw if yaw is None else yaw
        controller.odom_callback(
            self.odom(linear_x, linear_y, angular_z, x, y, yaw)
        )

    def publish_odom(
        self,
        controller,
        linear_x=0.0,
        linear_y=0.0,
        angular_z=0.0,
        x=None,
        y=None,
        yaw=None,
    ):
        self.advance(0.05)
        self.publish_odom_at_current_time(
            controller,
            linear_x,
            linear_y,
            angular_z,
            x,
            y,
            yaw,
        )

    def arm(self, controller, sequence=1):
        message = Header()
        message.seq = sequence
        message.stamp = controller_module.rospy.Time.from_sec(self.seconds)
        message.frame_id = "level_crossing"
        controller.arm_callback(message)

    def publish_landmark(self, controller, x=0.60, y=0.0, yaw=0.0):
        self.advance(0.05)
        self.publish_odom_at_current_time(
            controller,
            x=x - 0.60 * math.cos(yaw),
            y=y - 0.60 * math.sin(yaw),
            yaw=yaw,
        )
        message = PoseStamped()
        message.header.stamp = controller_module.rospy.Time.from_sec(
            self.seconds
        )
        message.header.frame_id = "odom"
        message.pose.position.x = x
        message.pose.position.y = y
        message.pose.orientation.z = math.sin(0.5 * yaw)
        message.pose.orientation.w = math.cos(0.5 * yaw)
        controller.landmark_pose_callback(message)

    def register_landmark(
        self, controller, x=0.60, y=0.0, yaw=0.0, sequence=1
    ):
        self.arm(controller, sequence)
        for _ in range(controller.registration_samples):
            self.publish_landmark(controller, x, y, yaw)
        self.assertIsNotNone(controller.crossing_frame)
        self.assertEqual(controller.ready_pub.messages[-1].seq, sequence)

    def open_gate(self, controller, crossing_x=0.60):
        if controller.crossing_frame is None:
            self.register_landmark(controller, x=crossing_x)
        controller.gate_callback(Bool(data=True))

    def request_stop_at_barrier(self, controller, confirm_distance=0.55):
        self.open_gate(controller)
        for _ in range(controller.closed_scans):
            self.publish_scan(controller, True, confirm_distance)
            controller.control_callback(None)
        if confirm_distance > controller.stop_forward_distance:
            self.publish_scan(controller, True, 0.44)
            controller.control_callback(None)
        self.assertEqual(controller.state, controller.STOPPED)

    def confirm_vehicle_stopped(self, controller):
        for _ in range(controller.stopped_odom_samples):
            self.publish_odom(controller)
        self.assertTrue(controller.stationary_confirmed)

    def reconfirm_closed_after_stop(self, controller):
        for _ in range(controller.post_stop_closed_scans):
            self.publish_scan(controller, True, 0.40)
        self.assertTrue(controller.post_stop_closed_confirmed)

    def stop_at_barrier(self, controller):
        self.request_stop_at_barrier(controller)
        self.confirm_vehicle_stopped(controller)
        self.reconfirm_closed_after_stop(controller)

    @staticmethod
    def assert_zero(command):
        assert command.linear.x == 0.0
        assert command.angular.z == 0.0

    def test_scan_outside_gate_has_no_control_effect(self):
        controller = self.make_controller()

        self.publish_scan(controller, True)

        self.assertEqual(controller.state, controller.WAIT_GATE)
        self.assertEqual(controller.closed_count, 0)
        self.assertEqual(controller.open_count, 0)
        self.assertFalse(controller.stop_requested)
        self.assertFalse(controller.resume_requested)
        self.assertEqual(controller.lane_service.calls, [])
        self.assertEqual(controller.cmd_pub.messages, [])
        self.assertEqual(controller.diagnostics_pub.messages, [])

    def test_fixed_landmark_ready_matches_fresh_arm_generation_and_stamp(self):
        controller = self.make_controller()
        self.arm(controller, sequence=17)
        self.publish_landmark(controller, x=4.20, y=-0.30, yaw=0.10)

        ready = controller.ready_pub.messages[-1]
        self.assertEqual(ready.seq, 17)
        self.assertEqual(ready.stamp, controller.registration_source_stamp)
        self.assertEqual(ready.frame_id, "level_crossing")
        self.assertAlmostEqual(controller.crossing_frame.x, 4.20)
        self.assertAlmostEqual(controller.crossing_frame.y, -0.30)
        self.assertAlmostEqual(controller.crossing_frame.yaw, 0.10)

    def test_arm_identity_and_zero_generation_disarm_registration(self):
        controller = self.make_controller()
        empty = Header()
        empty.seq = 3
        empty.stamp = controller_module.rospy.Time.from_sec(self.seconds)
        controller.arm_callback(empty)
        self.assertIsNone(controller.arm_seq)

        wrong = Header()
        wrong.seq = 4
        wrong.stamp = controller_module.rospy.Time.from_sec(self.seconds)
        wrong.frame_id = "tunnel"
        controller.arm_callback(wrong)
        self.assertIsNone(controller.arm_seq)

        valid = Header()
        valid.seq = 5
        valid.stamp = controller_module.rospy.Time.from_sec(self.seconds)
        valid.frame_id = "level_crossing"
        controller.arm_callback(valid)
        self.publish_landmark(controller, x=0.60)
        self.assertEqual(controller.ready_pub.messages[-1].seq, 5)
        ready_count = len(controller.ready_pub.messages)

        inactive = Header()
        inactive.seq = 0
        inactive.stamp = controller_module.rospy.Time.from_sec(self.seconds)
        inactive.frame_id = "level_crossing"
        controller.arm_callback(inactive)
        self.assertIsNone(controller.arm_seq)
        self.assertIsNone(controller.arm_stamp)
        self.assertIsNone(controller.crossing_frame)

        self.publish_landmark(controller, x=0.65)
        self.assertIsNone(controller.crossing_frame)
        self.assertEqual(len(controller.ready_pub.messages), ready_count)

    def test_enable_racing_readiness_activates_after_matching_registration(self):
        controller = self.make_controller()
        self.arm(controller, sequence=18)
        controller.gate_callback(Bool(data=True))
        self.assertEqual(controller.state, controller.WAIT_GATE)
        self.assertFalse(controller.zone_gate)

        self.publish_landmark(controller, x=0.70)

        self.assertEqual(controller.ready_pub.messages[-1].seq, 18)
        self.assertEqual(controller.state, controller.APPROACH)
        self.assertTrue(controller.zone_gate)

    def test_source_stamped_scan_frame_landmark_is_composed_with_odometry(self):
        controller = self.make_controller()
        controller.scan_sensor_pose = (0.10, 0.0, 0.0)
        self.arm(controller, sequence=19)
        self.advance(0.05)
        self.publish_odom_at_current_time(
            controller,
            x=2.0,
            y=1.0,
            yaw=math.pi / 2.0,
        )
        landmark = PoseStamped()
        landmark.header.stamp = controller_module.rospy.Time.from_sec(
            self.seconds
        )
        landmark.header.frame_id = "base_scan"
        landmark.pose.position.x = 0.50
        landmark.pose.orientation.w = 1.0

        controller.landmark_pose_callback(landmark)

        self.assertAlmostEqual(controller.crossing_frame.x, 2.0, places=6)
        self.assertAlmostEqual(controller.crossing_frame.y, 1.60, places=6)
        self.assertAlmostEqual(
            controller.crossing_frame.yaw, math.pi / 2.0, places=6
        )
        self.assertEqual(
            controller.ready_pub.messages[-1].stamp,
            landmark.header.stamp,
        )

    def test_landmark_before_arm_or_from_previous_generation_is_ignored(self):
        controller = self.make_controller()
        self.publish_odom(controller)
        stale = PoseStamped()
        stale.header.stamp = controller_module.rospy.Time.from_sec(self.seconds)
        stale.header.frame_id = "odom"
        stale.pose.position.x = 0.60
        stale.pose.orientation.w = 1.0
        controller.landmark_pose_callback(stale)
        self.assertIsNone(controller.crossing_frame)

        self.advance()
        self.arm(controller, sequence=2)
        controller.landmark_pose_callback(stale)
        self.assertIsNone(controller.crossing_frame)
        self.assertEqual(controller.ready_pub.messages, [])

    def test_bar_cluster_does_not_register_when_hardware_fallback_is_disabled(
        self,
    ):
        controller = self.make_controller()
        self.arm(controller, sequence=4)
        self.advance()
        self.publish_odom_at_current_time(controller)
        controller.scan_callback(self.scan(True, 0.50))

        self.assertIsNone(controller.crossing_frame)
        self.assertEqual(controller.ready_pub.messages, [])

    def test_gazebo_bar_fallback_registers_source_stamped_shifted_plane(self):
        controller = self.make_controller()
        controller.allow_barrier_anchor_fallback = True
        controller.registration_samples = 2
        controller.registration_candidates = deque(maxlen=2)
        self.arm(controller, sequence=8)

        self.advance()
        self.publish_odom_at_current_time(controller, x=5.00)
        controller.scan_callback(self.scan(True, 0.55))
        self.assertIsNone(controller.crossing_frame)

        self.advance()
        self.publish_odom_at_current_time(controller, x=5.05)
        controller.scan_callback(self.scan(True, 0.50))

        self.assertIsNotNone(controller.crossing_frame)
        self.assertAlmostEqual(controller.crossing_frame.x, 5.55, places=6)
        self.assertEqual(controller.ready_pub.messages[-1].seq, 8)
        self.assertEqual(
            controller.ready_pub.messages[-1].stamp,
            controller.scan_stamp,
        )

    def test_closed_confirmation_waits_for_distance_trigger_before_stop(self):
        controller = self.make_controller(closed_scans=3)
        self.open_gate(controller)

        for _ in range(controller.closed_scans):
            self.publish_scan(controller, True, 0.55)
            controller.control_callback(None)
            self.assertEqual(controller.state, controller.APPROACH)
            self.assertEqual(controller.lane_service.calls, [])
            self.assertEqual(controller.cmd_pub.messages, [])

        self.assertTrue(controller.closed_confirmed)
        self.assertFalse(controller.stop_requested)

        self.publish_scan(controller, True, 0.46)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.APPROACH)
        self.assertEqual(controller.lane_service.calls, [])

        self.publish_scan(controller, True, 0.45)
        controller.control_callback(None)

        self.assertEqual(controller.lane_service.calls, [False])
        self.assertTrue(controller.mission_has_control)
        self.assertEqual(controller.state, controller.STOPPED)
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        self.assert_zero(controller.cmd_pub.messages[-1])
        handoff_index = controller.events.index(("handoff", False))
        self.assertEqual(controller.events[handoff_index + 1][0:2], (
            "publish",
            "/cmd_vel",
        ))
        self.assertTrue(controller.barrier_pub.messages[-1].data)

    def test_near_barrier_still_requires_all_closed_confirmation_scans(self):
        controller = self.make_controller(closed_scans=3)
        self.open_gate(controller)

        for _ in range(controller.closed_scans - 1):
            self.publish_scan(controller, True, 0.40)
            controller.control_callback(None)
            self.assertEqual(controller.state, controller.APPROACH)
            self.assertEqual(controller.lane_service.calls, [])

        self.publish_scan(controller, True, 0.40)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.STOPPED)
        self.assertEqual(controller.lane_service.calls, [False])

    def test_one_cluster_dropout_after_early_confirmation_does_not_stop(self):
        controller = self.make_controller(closed_scans=3)
        self.open_gate(controller)
        for _ in range(controller.closed_scans):
            self.publish_scan(controller, True, 0.55)
        self.assertTrue(controller.closed_confirmed)

        self.publish_scan(controller, False)
        controller.control_callback(None)
        self.assertEqual(controller.approach_lost_count, 1)
        self.assertEqual(controller.state, controller.APPROACH)
        self.assertEqual(controller.lane_service.calls, [])

        self.publish_scan(controller, True, 0.50)
        controller.control_callback(None)
        self.assertEqual(controller.approach_lost_count, 0)
        self.assertEqual(controller.state, controller.APPROACH)

        self.publish_scan(controller, True, 0.44)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.STOPPED)
        self.assertEqual(controller.lane_service.calls, [False])

    def test_persistent_cluster_loss_after_confirmation_stops_fail_closed(self):
        controller = self.make_controller(closed_scans=3)
        self.open_gate(controller)
        for _ in range(controller.closed_scans):
            self.publish_scan(controller, True, 0.55)

        for _ in range(controller.approach_lost_scans - 1):
            self.publish_scan(controller, False)
            controller.control_callback(None)
            self.assertEqual(controller.state, controller.APPROACH)
        self.publish_scan(controller, False)
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.STOPPED)
        self.assertEqual(controller.lane_service.calls, [False])

    def test_open_confirmation_resumes_then_local_plane_clearance_completes(
        self,
    ):
        controller = self.make_controller(open_scans=3)
        self.stop_at_barrier(controller)

        for _ in range(controller.open_scans - 1):
            self.publish_odom(controller)
            self.publish_scan(controller, False)
            controller.control_callback(None)
            self.assertEqual(controller.state, controller.STOPPED)
            self.assertEqual(controller.lane_service.calls, [False])
            self.assert_zero(controller.cmd_pub.messages[-1])

        self.publish_odom(controller)
        self.publish_scan(controller, False)
        controller.control_callback(None)

        self.assertEqual(controller.lane_service.calls, [False, True])
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(controller.state, controller.PASSING)
        self.assertFalse(controller.barrier_pub.messages[-1].data)

        controller.control_callback(None)
        self.assertEqual(controller.state, controller.PASSING)
        required_center_progress = (
            controller.footprint_rear
            + controller.footprint_padding
            + controller.completion_clearance_margin
        )
        self.publish_odom(
            controller,
            x=controller.crossing_frame.x + required_center_progress + 0.01,
        )
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.COMPLETE)

    def test_braking_and_unverified_stop_cannot_be_mistaken_for_open_barrier(
        self,
    ):
        controller = self.make_controller(open_scans=5)
        self.request_stop_at_barrier(controller)

        # The bar can leave the ROI while the base is still braking. Even more
        # than the normal clear threshold must not release cmd_vel ownership.
        for _ in range(controller.open_scans + 1):
            self.publish_odom(controller, linear_x=0.08)
            self.publish_scan(controller, False)
            controller.control_callback(None)
        self.assertEqual(controller.state, controller.STOPPED)
        self.assertFalse(controller.stationary_confirmed)
        self.assertFalse(controller.post_stop_closed_confirmed)
        self.assertEqual(controller.open_count, 0)
        self.assertEqual(controller.lane_service.calls, [False])

        # Being stationary is still insufficient: the lowered bar must be seen
        # again at that exact stopped pose before clear scans can mean "open".
        self.confirm_vehicle_stopped(controller)
        for _ in range(controller.open_scans + 1):
            self.publish_odom(controller)
            self.publish_scan(controller, False)
            controller.control_callback(None)
        self.assertEqual(controller.state, controller.STOPPED)
        self.assertFalse(controller.post_stop_closed_confirmed)
        self.assertEqual(controller.open_count, 0)
        self.assertEqual(controller.lane_service.calls, [False])

        self.reconfirm_closed_after_stop(controller)
        for _ in range(controller.open_scans - 1):
            self.publish_odom(controller)
            self.publish_scan(controller, False)
            controller.control_callback(None)
            self.assertEqual(controller.state, controller.STOPPED)
        self.publish_odom(controller)
        self.publish_scan(controller, False)
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.PASSING)
        self.assertEqual(controller.lane_service.calls, [False, True])

    def test_pre_stop_and_repeated_odometry_cannot_confirm_stationary(self):
        controller = self.make_controller()
        self.open_gate(controller)
        self.publish_odom(controller)
        self.publish_odom(controller)
        self.request_stop_at_barrier(controller)

        self.assertFalse(controller.stationary_confirmed)
        self.assertEqual(controller.stationary_count, 0)

        self.publish_odom(controller)
        self.assertEqual(controller.stationary_count, 1)
        same_message = self.odom()
        controller.odom_callback(same_message)
        controller.control_callback(None)
        controller.control_callback(None)
        self.assertEqual(controller.stationary_count, 1)
        self.assertFalse(controller.stationary_confirmed)

        self.publish_odom(controller)
        self.assertTrue(controller.stationary_confirmed)

    def test_scan_captured_before_stationary_confirmation_is_not_reused(self):
        controller = self.make_controller()
        self.request_stop_at_barrier(controller)

        self.advance(0.05)
        queued_closed_scan = self.scan(True, 0.40)
        self.confirm_vehicle_stopped(controller)
        controller.scan_callback(queued_closed_scan)

        self.assertFalse(controller.post_stop_closed_confirmed)
        self.publish_scan(controller, True, 0.40)
        self.assertTrue(controller.post_stop_closed_confirmed)

    def test_scan_captured_before_gate_activation_does_not_confirm_barrier(self):
        controller = self.make_controller(closed_scans=1)
        queued_closed_scan = self.scan(True, 0.40)
        self.open_gate(controller)

        controller.scan_callback(queued_closed_scan)
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.APPROACH)
        self.assertFalse(controller.closed_confirmed)
        self.assertEqual(controller.lane_service.calls, [])

        self.publish_scan(controller, True, 0.40)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.STOPPED)

    def test_motion_after_post_stop_confirmation_disarms_release(self):
        controller = self.make_controller(open_scans=2)
        self.stop_at_barrier(controller)
        self.publish_odom(controller)
        self.publish_scan(controller, False)
        self.assertEqual(controller.open_count, 1)

        self.publish_odom(controller, linear_x=0.04)

        self.assertFalse(controller.stationary_confirmed)
        self.assertFalse(controller.post_stop_closed_confirmed)
        self.assertEqual(controller.open_count, 0)
        self.assertFalse(controller.resume_requested)

    def test_manual_pause_preserves_completed_post_stop_safety_checks(self):
        controller = self.make_controller(open_scans=2)
        self.stop_at_barrier(controller)
        self.assertTrue(controller.stationary_confirmed)
        self.assertTrue(controller.post_stop_closed_confirmed)

        controller.manual_stop_callback(Bool(data=True))
        self.publish_scan(controller, False)
        controller.manual_stop_callback(Bool(data=False))

        self.assertTrue(controller.stationary_confirmed)
        self.assertTrue(controller.post_stop_closed_confirmed)
        for _ in range(controller.open_scans):
            self.publish_odom(controller)
            self.publish_scan(controller, False)
            controller.control_callback(None)
        self.assertEqual(controller.state, controller.PASSING)

    def test_open_at_arrival_passes_without_cmd_vel_handoff_then_completes(self):
        controller = self.make_controller(open_scans=2)
        self.open_gate(controller)

        for _ in range(controller.arrival_open_scans):
            self.publish_scan(controller, False)
            controller.control_callback(None)

        self.assertEqual(controller.state, controller.PASSING)
        self.assertEqual(controller.open_count, 0)
        self.assertEqual(controller.lane_service.calls, [])
        self.assertEqual(controller.cmd_pub.messages, [])

        center_progress = (
            controller.footprint_rear
            + controller.footprint_padding
            + controller.completion_clearance_margin
            + 0.01
        )
        self.publish_odom(
            controller,
            x=controller.crossing_frame.x + center_progress,
        )
        self.publish_scan(controller, False)
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.COMPLETE)
        self.assertEqual(controller.lane_service.calls, [])
        self.assertEqual(controller.cmd_pub.messages, [])

    def test_local_completion_is_invariant_to_longitudinal_course_shift(self):
        for sequence, shift in enumerate((0.0, 3.75), start=20):
            controller = self.make_controller(open_scans=2)
            crossing_x = shift + 0.60
            self.register_landmark(
                controller,
                x=crossing_x,
                sequence=sequence,
            )
            controller.gate_callback(Bool(data=True))

            for _ in range(controller.arrival_open_scans):
                self.publish_scan(controller, False, 0.40)
                controller.control_callback(None)
            self.assertEqual(controller.state, controller.PASSING)

            center_progress = (
                controller.footprint_rear
                + controller.footprint_padding
                + controller.completion_clearance_margin
                + 0.01
            )
            self.publish_odom(
                controller,
                x=crossing_x + center_progress,
            )
            self.publish_scan(controller, False)
            controller.control_callback(None)

            self.assertEqual(controller.state, controller.COMPLETE)
            self.assertEqual(controller.lane_service.calls, [])

    def test_confirmed_closed_request_is_latched_until_timer_stops(self):
        controller = self.make_controller(closed_scans=2)
        self.open_gate(controller)

        self.publish_scan(controller, True)
        self.publish_scan(controller, True)
        self.assertTrue(controller.stop_requested)

        # A clear scan can race the timer after the required closed scans. The
        # already-confirmed stop remains conservative and cannot be cancelled.
        self.publish_scan(controller, False)
        self.assertTrue(controller.stop_requested)
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.STOPPED)
        self.assertEqual(controller.lane_service.calls, [False])
        self.assert_zero(controller.cmd_pub.messages[-1])

    def test_manual_stop_pauses_and_resets_scan_confirmation(self):
        controller = self.make_controller(closed_scans=2)
        self.open_gate(controller)
        self.publish_scan(controller, True)
        self.assertEqual(controller.closed_count, 1)

        controller.manual_stop_callback(Bool(data=True))
        self.publish_scan(controller, True)
        controller.control_callback(None)
        self.assertEqual(controller.closed_count, 0)
        self.assertEqual(controller.state, controller.APPROACH)
        self.assertEqual(controller.lane_service.calls, [])

        controller.manual_stop_callback(Bool(data=False))
        self.publish_scan(controller, True)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.APPROACH)
        self.publish_scan(controller, True)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.STOPPED)
        self.assertEqual(controller.lane_service.calls, [False])

    def test_stale_scan_while_stopped_keeps_zero_and_forbids_resume(self):
        controller = self.make_controller(open_scans=2)
        self.stop_at_barrier(controller)

        for _ in range(controller.open_scans):
            self.publish_scan(controller, False)
        self.assertTrue(controller.resume_requested)
        command_count = len(controller.cmd_pub.messages)

        self.advance(controller.scan_timeout + 0.01)
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.STOPPED)
        self.assertEqual(controller.lane_service.calls, [False])
        self.assertEqual(len(controller.cmd_pub.messages), command_count + 1)
        self.assert_zero(controller.cmd_pub.messages[-1])

        # The old, now-stale confirmation cannot be reused. A new clear scan is
        # also discarded until fresh stopped odometry is available.
        self.publish_scan(controller, False)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.STOPPED)
        self.assertEqual(controller.open_count, 0)
        self.assertFalse(controller.resume_requested)
        self.assertEqual(controller.lane_service.calls, [False])
        self.assert_zero(controller.cmd_pub.messages[-1])

    def test_stale_odometry_requires_full_stop_and_closed_reconfirmation(self):
        controller = self.make_controller(open_scans=2)
        self.stop_at_barrier(controller)

        self.advance(controller.odom_timeout + 0.01)
        controller.control_callback(None)
        self.assertFalse(controller.stationary_confirmed)
        self.assertFalse(controller.post_stop_closed_confirmed)

        self.publish_odom(controller)
        self.assertEqual(controller.stationary_count, 1)
        self.publish_odom(controller)
        self.assertTrue(controller.stationary_confirmed)
        self.assertFalse(controller.post_stop_closed_confirmed)

        for _ in range(controller.open_scans + 1):
            self.publish_odom(controller)
            self.publish_scan(controller, False)
            controller.control_callback(None)
        self.assertEqual(controller.state, controller.STOPPED)
        self.assertEqual(controller.open_count, 0)
        self.assertEqual(controller.lane_service.calls, [False])

    def test_all_invalid_scan_cannot_release_stopped_robot(self):
        controller = self.make_controller(open_scans=1)
        self.stop_at_barrier(controller)
        invalid = self.scan(False)
        invalid.ranges = [math.inf] * len(invalid.ranges)

        self.advance()
        invalid.header.stamp = controller_module.rospy.Time.from_sec(
            self.seconds
        )
        controller.scan_callback(invalid)
        controller.control_callback(None)

        self.assertFalse(controller.scan_healthy)
        self.assertEqual(controller.open_count, 0)
        self.assertFalse(controller.resume_requested)
        self.assertEqual(controller.state, controller.STOPPED)
        self.assertEqual(controller.lane_service.calls, [False])
        self.assert_zero(controller.cmd_pub.messages[-1])

    def test_barrier_reclosing_while_passing_stops_again(self):
        controller = self.make_controller(closed_scans=2, open_scans=1)
        self.stop_at_barrier(controller)
        self.publish_scan(controller, False)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.PASSING)

        self.publish_scan(controller, True)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.PASSING)
        self.publish_scan(controller, True)
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.STOPPED)
        self.assertEqual(controller.lane_service.calls, [False, True, False])
        self.assertTrue(controller.mission_has_control)
        self.assertFalse(controller.stationary_confirmed)
        self.assertFalse(controller.post_stop_closed_confirmed)
        self.assert_zero(controller.cmd_pub.messages[-1])

    def test_scan_callback_uses_angle_metadata_with_real_detector(self):
        controller = self.make_controller(closed_scans=1)
        self.open_gate(controller)

        self.publish_scan(controller, True)

        detection = controller.latest_detection
        self.assertIsNotNone(detection)
        self.assertAlmostEqual(detection.forward_distance, 0.40, places=6)
        self.assertEqual(detection.first_beam_index, 17)
        self.assertEqual(detection.last_beam_index, 23)
        self.assertGreaterEqual(detection.lateral_span, 0.18)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.STOPPED)
        self.assertEqual(controller.lane_service.calls, [False])

    def test_gate_revoke_before_handoff_returns_to_wait(self):
        controller = self.make_controller()
        self.open_gate(controller)

        controller.gate_callback(Bool(data=False))
        self.assertTrue(controller.revoke_requested)
        controller.control_callback(None)

        self.assertEqual(controller.lane_service.calls, [])
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(controller.state, controller.WAIT_GATE)
        self.assertFalse(controller.revoke_requested)
        self.assertFalse(controller.barrier_pub.messages[-1].data)

    def test_gate_revoke_while_stopped_holds_control_fail_closed(self):
        controller = self.make_controller()
        self.stop_at_barrier(controller)
        command_count = len(controller.cmd_pub.messages)

        controller.gate_callback(Bool(data=False))
        controller.control_callback(None)

        self.assertFalse(controller.revoke_requested)
        self.assertTrue(controller.mission_has_control)
        self.assertEqual(controller.state, controller.STOPPED)
        self.assertEqual(controller.lane_service.calls, [False])
        self.assertEqual(len(controller.cmd_pub.messages), command_count + 1)
        self.assert_zero(controller.cmd_pub.messages[-1])
        self.assertTrue(controller.barrier_pub.messages[-1].data)

    def test_gate_false_true_flicker_while_stopped_cannot_break_ownership(self):
        controller = self.make_controller()
        self.stop_at_barrier(controller)

        controller.gate_callback(Bool(data=False))
        controller.gate_callback(Bool(data=True))
        controller.control_callback(None)

        self.assertFalse(controller.revoke_requested)
        self.assertTrue(controller.zone_gate)
        self.assertTrue(controller.mission_has_control)
        self.assertEqual(controller.state, controller.STOPPED)
        self.assertEqual(controller.lane_service.calls, [False])
        self.assert_zero(controller.cmd_pub.messages[-1])

    def test_acquire_failure_disables_lane_and_holds_zero(self):
        controller = self.make_controller(closed_scans=1)
        controller.lane_service = RecordingLaneService(
            controller.events, outcomes=[False, False]
        )
        self.open_gate(controller)

        self.publish_scan(controller, True)
        controller.control_callback(None)

        self.assertEqual(controller.lane_service.calls, [False, False])
        self.assertEqual(controller.lane_stop_service.calls, [False])
        self.assertTrue(controller.mission_has_control)
        self.assertEqual(controller.state, controller.FAILED)
        self.assert_zero(controller.cmd_pub.messages[-1])

        command_count = len(controller.cmd_pub.messages)
        controller.control_callback(None)
        self.assertEqual(len(controller.cmd_pub.messages), command_count + 1)
        self.assert_zero(controller.cmd_pub.messages[-1])

    def test_release_failure_reacquires_control_and_holds_zero(self):
        controller = self.make_controller(closed_scans=1, open_scans=1)
        controller.lane_service = RecordingLaneService(
            controller.events, outcomes=[True, False, True]
        )
        self.stop_at_barrier(controller)

        self.publish_scan(controller, False)
        controller.control_callback(None)

        self.assertEqual(controller.lane_service.calls, [False, True, False])
        self.assertTrue(controller.mission_has_control)
        self.assertEqual(controller.state, controller.FAILED)
        self.assert_zero(controller.cmd_pub.messages[-1])


if __name__ == "__main__":
    unittest.main()
