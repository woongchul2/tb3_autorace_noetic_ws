#!/usr/bin/env python3

import math
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


NODE_DIR = Path(__file__).resolve().parents[1] / "nodes"
if str(NODE_DIR) not in sys.path:
    sys.path.insert(0, str(NODE_DIR))

import level_crossing_lidar_controller as controller_module
from custom_autorace_bringup.level_crossing import HorizontalBarrierConfig
from level_crossing_lidar_controller import LevelCrossingLidarController
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


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

    def make_controller(self, closed_scans=3, open_scans=3):
        controller = LevelCrossingLidarController.__new__(
            LevelCrossingLidarController
        )
        controller.lock = threading.RLock()
        controller.state = controller.WAIT_GATE
        controller.zone_gate = False
        controller.zone_cleared = False
        controller.manual_stop = False
        controller.mission_has_control = False
        controller.revoke_requested = False
        controller.stop_requested = False
        controller.resume_requested = False

        controller.scan_received = None
        controller.scan_stamp = None
        controller.scan_healthy = False
        controller.valid_scan_points = 0
        controller.last_confirmation_stamp = None
        controller.closed_count = 0
        controller.open_count = 0
        controller.latest_detection = None
        controller.gate_activated_at = None

        controller.closed_scans = closed_scans
        controller.open_scans = open_scans
        controller.maximum_scan_gap = 0.25
        controller.scan_timeout = 0.40
        controller.handoff_timeout = 0.30
        controller.lane_stop_timeout = 0.30
        controller.minimum_valid_scan_points = 1
        controller.detector_config = HorizontalBarrierConfig(
            min_forward_distance=0.10,
            max_forward_distance=0.70,
            half_width=0.25,
            max_adjacent_beam_gap=1,
            max_point_gap=0.08,
            min_points=5,
            min_lateral_span=0.14,
            max_depth_spread=0.02,
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
        controller.diagnostics_pub = RecordingPublisher()
        return controller

    def scan(self, barrier_down):
        """Build a scan whose forward direction is not the array midpoint."""
        message = LaserScan()
        message.header.stamp = controller_module.rospy.Time.from_sec(
            self.seconds
        )
        message.angle_min = -2.0
        message.angle_increment = 0.10
        message.angle_max = message.angle_min + 59 * message.angle_increment
        message.range_min = 0.05
        message.range_max = 10.0
        message.ranges = [math.inf] * 60
        # A finite return behind the robot marks the message as healthy but is
        # outside the forward detection corridor.
        message.ranges[0] = 2.0
        if barrier_down:
            # With the metadata above, beams 18..22 cover -0.2..+0.2 rad.
            # These ranges describe a transverse bar on x=0.40 m.
            for index in range(18, 23):
                angle = message.angle_min + index * message.angle_increment
                message.ranges[index] = 0.40 / math.cos(angle)
        return message

    def publish_scan(self, controller, barrier_down):
        self.advance()
        controller.scan_callback(self.scan(barrier_down))

    @staticmethod
    def open_gate(controller):
        controller.gate_callback(Bool(data=True))

    def stop_at_barrier(self, controller):
        self.open_gate(controller)
        for _ in range(controller.closed_scans):
            self.publish_scan(controller, True)
            controller.control_callback(None)
        self.assertEqual(controller.state, controller.STOPPED)

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

    def test_closed_confirmation_hands_off_and_stops_on_required_scan(self):
        controller = self.make_controller(closed_scans=3)
        self.open_gate(controller)

        for _ in range(controller.closed_scans - 1):
            self.publish_scan(controller, True)
            controller.control_callback(None)
            self.assertEqual(controller.state, controller.APPROACH)
            self.assertEqual(controller.lane_service.calls, [])
            self.assertEqual(controller.cmd_pub.messages, [])

        self.publish_scan(controller, True)
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

    def test_open_confirmation_resumes_then_clearance_completes(self):
        controller = self.make_controller(open_scans=3)
        self.stop_at_barrier(controller)

        for _ in range(controller.open_scans - 1):
            self.publish_scan(controller, False)
            controller.control_callback(None)
            self.assertEqual(controller.state, controller.STOPPED)
            self.assertEqual(controller.lane_service.calls, [False])
            self.assert_zero(controller.cmd_pub.messages[-1])

        self.publish_scan(controller, False)
        controller.control_callback(None)

        self.assertEqual(controller.lane_service.calls, [False, True])
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(controller.state, controller.PASSING)
        self.assertFalse(controller.barrier_pub.messages[-1].data)

        controller.control_callback(None)
        self.assertEqual(controller.state, controller.PASSING)
        controller.clearance_callback(Bool(data=True))
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.COMPLETE)

    def test_initial_clear_scans_cannot_complete_mission(self):
        controller = self.make_controller(open_scans=2)
        self.open_gate(controller)

        for _ in range(controller.open_scans + 2):
            self.publish_scan(controller, False)
            controller.control_callback(None)

        self.assertEqual(controller.state, controller.APPROACH)
        self.assertEqual(controller.open_count, 0)
        self.assertEqual(controller.lane_service.calls, [])
        self.assertEqual(controller.cmd_pub.messages, [])

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

        # The old, now-stale confirmation cannot be reused. A new clear scan
        # starts a fresh sequence at one rather than releasing immediately.
        self.publish_scan(controller, False)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.STOPPED)
        self.assertEqual(controller.open_count, 1)
        self.assertFalse(controller.resume_requested)
        self.assertEqual(controller.lane_service.calls, [False])
        self.assert_zero(controller.cmd_pub.messages[-1])

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
        self.assert_zero(controller.cmd_pub.messages[-1])

    def test_scan_callback_uses_angle_metadata_with_real_detector(self):
        controller = self.make_controller(closed_scans=1)
        self.open_gate(controller)

        self.publish_scan(controller, True)

        detection = controller.latest_detection
        self.assertIsNotNone(detection)
        self.assertAlmostEqual(detection.forward_distance, 0.40, places=6)
        self.assertEqual(detection.first_beam_index, 18)
        self.assertEqual(detection.last_beam_index, 22)
        self.assertGreaterEqual(detection.lateral_span, 0.14)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.STOPPED)
        self.assertEqual(controller.lane_service.calls, [False])

    def test_gate_revoke_returns_cmd_vel_ownership(self):
        controller = self.make_controller()
        self.stop_at_barrier(controller)
        command_count = len(controller.cmd_pub.messages)

        controller.gate_callback(Bool(data=False))
        self.assertTrue(controller.revoke_requested)
        controller.control_callback(None)

        self.assertEqual(controller.lane_service.calls, [False, True])
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(controller.state, controller.WAIT_GATE)
        self.assertFalse(controller.revoke_requested)
        self.assertEqual(len(controller.cmd_pub.messages), command_count + 1)
        self.assert_zero(controller.cmd_pub.messages[-1])
        self.assertFalse(controller.barrier_pub.messages[-1].data)

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
