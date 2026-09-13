#!/usr/bin/env python3

from collections import deque
import math
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


NODE_DIR = Path(__file__).resolve().parents[1] / "nodes"
if str(NODE_DIR) not in sys.path:
    sys.path.insert(0, str(NODE_DIR))

import obstacle_mission_controller as controller_module
from obstacle_mission_controller import ObstacleMissionController
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64MultiArray

from custom_autorace_bringup.obstacle_planner import Footprint, RectanglePathChecker
from custom_autorace_bringup.path_following import (
    CommonPath,
    GoalTolerance,
    PathFollower,
    Pose2D,
    RigidTransform2D,
    TrackingConfig,
    ValidationResult,
    normalize_angle,
)


class RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class ObstacleControllerTest(unittest.TestCase):
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

    def now(self):
        return controller_module.rospy.Time.from_sec(self.seconds)

    @staticmethod
    def tracking_config():
        return TrackingConfig(
            lookahead_distance=0.08,
            maximum_linear_velocity=0.20,
            maximum_angular_velocity=0.8,
            maximum_lateral_acceleration=0.08,
            linear_acceleration=0.5,
            linear_deceleration=0.8,
            angular_acceleration=1.0,
            heading_gain=0.35,
        )

    @staticmethod
    def straight_path():
        return CommonPath(
            x=np.asarray([0.0, 0.5, 1.0]),
            y=np.zeros(3),
            heading=np.zeros(3),
            curvature=np.zeros(3),
            speed=np.full(3, 0.10),
            frame_id="odom",
            goal_tolerance=GoalTolerance(0.03, math.radians(8.0), 0.10),
        )

    def make_path_harness(self):
        controller = ObstacleMissionController.__new__(
            ObstacleMissionController
        )
        controller.lock = threading.RLock()
        controller.shutting_down = False
        controller.committed_path = self.straight_path()
        controller.path_follower = PathFollower(self.tracking_config())
        controller.odom_x = 0.0
        controller.odom_y = 0.0
        controller.odom_yaw = 0.0
        controller.odom_linear_velocity = 0.0
        controller.odom_angular_velocity = 0.0
        controller.path_follower.reset(
            controller.committed_path, controller._current_pose()
        )
        controller.live_speed_limit = math.inf
        controller.control_period = 0.05
        controller.last_command_time = None
        controller.remaining_distance = controller.committed_path.length
        controller.diagnostics_pub = RecordingPublisher()
        return controller

    def make_map_pose_harness(self):
        controller = ObstacleMissionController.__new__(
            ObstacleMissionController
        )
        controller.lock = threading.RLock()
        controller.map_ready = False
        controller.map_pose_header_stamp = None
        controller.map_pose_stamp = None
        controller.map_pose_received = None
        controller.map_pose_timeout = 0.35
        controller.map_pose_future_tolerance = 0.05
        controller.maximum_scan_odom_skew = 0.075
        controller.maximum_map_pose_odom_skew = 0.075
        controller.odom_history = deque()
        controller.map_from_odom = None
        controller.odom_ready = False
        controller.odom_frame = "odom"
        return controller

    def make_scan_queue_harness(self):
        controller = self.make_map_pose_harness()
        controller.scan_timeout = 0.35
        controller.pending_scan_limit = 4
        controller.pending_scans = deque()
        controller.scan_points = np.asarray([[9.0, 9.0]], dtype=np.float64)
        controller.scan_odom_pose = (0.0, 0.0, 0.0)
        controller.scan_stamp = controller_module.rospy.Time.from_sec(9.95)
        controller.scan_generation = 1
        controller.scan_median_window = 1
        controller.scan_max_range = 2.0
        controller.scan_rear_limit = 0.20
        controller.cluster_link_distance = 0.08
        controller.cluster_minimum_points = 1
        controller.obstacle_point_spacing = 0.012
        controller.lidar_x = 0.0
        controller.lidar_y = 0.0
        controller.odom_history_duration = 0.75
        controller.odom_frame = "odom"
        controller.odom_x = 0.0
        controller.odom_y = 0.0
        controller.odom_yaw = 0.0
        controller.odom_linear_velocity = 0.0
        controller.odom_angular_velocity = 0.0
        controller.odom_stamp = self.now()
        controller.odom_ready = True
        return controller

    def make_runtime_harness(self):
        """Return an active controller with deterministic swept-safety inputs."""
        controller = self.make_path_harness()
        controller.state = controller.AVOIDING
        controller.zone_gate = True
        controller.zone_inside = True
        controller.zone_cleared = False
        controller.start_requested = False
        controller.revoke_requested = False
        controller.manual_stop = False
        controller.mission_has_control = True
        controller.handoff_ambiguous = False
        controller.state_pub = RecordingPublisher()
        controller.cmd_pub = RecordingPublisher()
        controller.speed_limit_pub = RecordingPublisher()
        controller.planner_status_pub = RecordingPublisher()
        controller.lane_resume_max_velocity = 0.30
        controller.tracking_config = self.tracking_config()
        controller.odom_ready = True
        controller.odom_stamp = self.now()
        controller.scan_stamp = self.now()
        controller.odom_timeout = 0.35
        controller.scan_timeout = 0.35
        controller.odom_frame = "odom"
        controller.odom_history_duration = 0.75
        controller.odom_history = deque(
            ((self.now(), 0.0, 0.0, 0.0, controller.odom_frame),)
        )
        controller.pending_scans = deque()
        controller.scan_points = np.empty((0, 2), dtype=np.float64)
        controller.scan_odom_pose = None
        controller.scan_generation = 1
        controller.map_ready = False
        controller.safety_reaction_time = 0.10
        controller.linear_deceleration = 0.80
        controller.safety_stop_margin = 0.005
        controller.spline_planner = mock.Mock(live_validation_distance=0.45)
        controller.footprint = Footprint(
            0.067645,
            0.118073,
            0.0903,
            obstacle_padding=0.010,
        )
        controller.validation_footprint = Footprint(
            0.067645,
            0.118073,
            0.0903,
            line_margin=0.005,
            localization_margin=0.002,
            tracking_margin=0.002,
        )
        controller.path_checker = RectanglePathChecker(controller.footprint)
        controller.fixed_path_validation = ValidationResult(True)
        controller._course_heading_error = mock.Mock(return_value=0.0)
        controller._planning_corridor = mock.Mock(
            return_value=(-0.25, 0.25, "surveyed")
        )
        controller._camera_corridor = mock.Mock(return_value=None)
        controller._lane_points = mock.Mock(
            return_value=np.empty((0, 2), dtype=np.float64)
        )
        controller._lane_points_in_odom = mock.Mock(
            return_value=np.empty((0, 2), dtype=np.float64)
        )
        controller._publish_diagnostics = mock.Mock()

        def set_lane(enabled):
            controller.mission_has_control = not enabled
            controller.handoff_ambiguous = False
            return True

        controller._set_lane_controller = mock.Mock(side_effect=set_lane)
        return controller

    def run_control_while_sweep_is_blocked(self, controller, while_blocked):
        """Run one timer tick and invoke a competing callback during its sweep."""
        entered_sweep = threading.Event()
        release_sweep = threading.Event()
        worker_errors = []
        captured = {}
        real_motion_safety = controller.path_checker.validator.motion_safety

        def blocked_motion_safety(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            captured["tracking"] = kwargs["tracking"]
            captured["pose"] = args[1]
            if controller.lock._is_owned():
                raise AssertionError("mission lock is held during swept validation")
            entered_sweep.set()
            if not release_sweep.wait(1.0):
                raise AssertionError("test did not release swept validation")
            return real_motion_safety(*args, **kwargs)

        def run_control():
            try:
                controller.control_callback(None)
            except BaseException as error:  # surfaced in the test thread below
                worker_errors.append(error)

        with mock.patch.object(
            controller.path_checker.validator,
            "motion_safety",
            side_effect=blocked_motion_safety,
        ):
            control_thread = threading.Thread(target=run_control)
            control_thread.start()
            self.assertTrue(entered_sweep.wait(0.5))
            try:
                while_blocked(captured)
            finally:
                release_sweep.set()
            control_thread.join(1.0)

        self.assertFalse(control_thread.is_alive())
        self.assertEqual(worker_errors, [])
        return captured

    @staticmethod
    def scan_message(stamp):
        message = LaserScan()
        message.header.stamp = stamp
        message.header.frame_id = "base_scan"
        message.angle_min = -0.01
        message.angle_increment = 0.01
        message.range_min = 0.01
        message.range_max = 3.5
        message.ranges = [1.0, 1.0, 1.0]
        return message

    def test_map_pose_callback_keeps_source_and_receipt_stamps(self):
        controller = self.make_map_pose_harness()
        message = PoseStamped()
        message.header.stamp = controller_module.rospy.Time.from_sec(9.90)
        message.pose.position.x = 1.2
        message.pose.position.y = -0.3

        controller.map_pose_callback(message)

        self.assertEqual(controller.map_pose_header_stamp, message.header.stamp)
        self.assertEqual(controller.map_pose_stamp, message.header.stamp)
        self.assertEqual(controller.map_pose_received, self.now())
        self.assertTrue(controller.map_ready)
        self.assertIsNone(controller.map_from_odom)

        message.header.stamp = controller_module.rospy.Time()
        self.seconds = 10.1
        controller.map_pose_callback(message)
        self.assertEqual(
            controller.map_pose_header_stamp, controller_module.rospy.Time()
        )
        self.assertEqual(controller.map_pose_stamp, self.now())
        self.assertEqual(controller.map_pose_received, self.now())

    def test_rejected_boundary_view_does_not_refresh_old_lane_geometry(self):
        controller = ObstacleMissionController.__new__(
            ObstacleMissionController
        )
        controller.lock = threading.RLock()
        controller.minimum_boundary_gap = 300.0
        controller.maximum_boundary_gap = 900.0
        controller.map_ready = True
        controller.odom_ready = True
        controller.boundary_heading_limit = 0.1
        controller.state = controller.AVOIDING
        controller.lane_width = 0.495
        controller.pixels_per_meter = 1850.0
        controller.scale_filter_alpha = 0.2
        controller.image_center = 383.0
        controller.boundary_sample_forward = 0.14
        controller.course_odom_heading = 0.0
        controller.odom_x = 0.0
        controller.odom_y = 0.0
        controller.line_center_absolute = 0.0
        old_stamp = controller_module.rospy.Time.from_sec(9.5)
        controller.line_observation_stamp = old_stamp
        controller._course_heading_error = mock.Mock(return_value=0.2)
        message = Float64MultiArray(data=[200.0, 600.0, 1.0, 1.0])

        controller.boundary_callback(message)

        self.assertEqual(controller.line_observation_stamp, old_stamp)
        self.assertEqual(controller.line_center_absolute, 0.0)

    def test_map_pose_freshness_checks_source_receipt_and_future_time(self):
        controller = self.make_map_pose_harness()
        controller.map_ready = True
        controller.map_pose_stamp = self.now()
        controller.map_pose_received = self.now()
        self.assertIsNone(controller._map_pose_problem(self.now()))

        controller.map_pose_stamp = controller_module.rospy.Time.from_sec(9.60)
        self.assertIn("source is stale", controller._map_pose_problem(self.now()))

        controller.map_pose_stamp = self.now()
        controller.map_pose_received = controller_module.rospy.Time.from_sec(9.60)
        self.assertIn("receipt is stale", controller._map_pose_problem(self.now()))

        controller.map_pose_received = self.now()
        controller.map_pose_stamp = controller_module.rospy.Time.from_sec(10.06)
        self.assertIn("source stamp", controller._map_pose_problem(self.now()))

        controller.map_pose_stamp = controller_module.rospy.Time.from_sec(10.04)
        self.assertIsNone(controller._map_pose_problem(self.now()))

    def test_start_run_rejects_stale_map_pose(self):
        controller = self.make_map_pose_harness()
        controller.map_ready = True
        controller.map_pose_stamp = controller_module.rospy.Time.from_sec(9.60)
        controller.map_pose_received = self.now()
        controller.map_from_odom = RigidTransform2D(
            0.0, 0.0, 0.0, source_frame="odom", target_frame="map"
        )
        controller.odom_ready = True
        controller.odom_stamp = self.now()
        controller.odom_timeout = 0.35
        controller.start_requested = True
        controller.speed_limit_pub = RecordingPublisher()
        controller._set_state = mock.Mock()

        controller._start_run()

        self.assertTrue(controller.start_requested)
        self.assertEqual(controller.speed_limit_pub.messages, [])
        controller._set_state.assert_not_called()

    def test_alignment_commit_rechecks_map_pose_freshness(self):
        controller = self.make_map_pose_harness()
        controller.map_ready = True
        controller.map_pose_stamp = self.now()
        controller.map_pose_received = self.now()
        controller.map_from_odom = RigidTransform2D(
            0.0, 0.0, 0.0, source_frame="odom", target_frame="map"
        )
        controller.odom_from_course = None
        controller.scan_generation = 1
        controller.last_plan_scan_generation = -1
        controller.planner_status_pub = RecordingPublisher()
        controller._course_heading_error = mock.Mock(return_value=0.0)
        controller._planning_corridor = mock.Mock(
            return_value=(-0.25, 0.25, 0.0)
        )
        controller._course_template_coordinates = mock.Mock(
            return_value=(0.0, 0.0)
        )
        controller._lane_points = mock.Mock(
            return_value=np.empty((0, 2), dtype=np.float64)
        )
        alignment = mock.Mock(
            progress=0.0,
            lateral=0.0,
            inliers=20,
            rms=0.001,
        )

        def finish_after_timeout(*_args):
            self.seconds += 0.40
            return alignment

        controller.spline_planner = mock.Mock()
        controller.spline_planner.align_pose.side_effect = finish_after_timeout
        controller._freeze_course_alignment = mock.Mock()

        self.assertFalse(controller._attempt_path(self.now()))
        controller._freeze_course_alignment.assert_not_called()
        self.assertEqual(
            controller.planner_status_pub.messages[-1].data,
            "ALIGNING",
        )

    def test_common_map_transform_uses_odom_at_map_source_stamp(self):
        controller = self.make_map_pose_harness()
        source_stamp = controller_module.rospy.Time.from_sec(10.05)
        controller.map_pose_stamp = source_stamp
        controller.odom_history = deque(
            (
                (
                    controller_module.rospy.Time.from_sec(10.00),
                    0.0,
                    0.0,
                    0.1,
                    "odom",
                ),
                (
                    controller_module.rospy.Time.from_sec(10.10),
                    0.2,
                    0.1,
                    0.3,
                    "odom",
                ),
            )
        )
        expected_yaw = 0.4
        expected_x = 1.5
        expected_y = -0.7
        interpolated_x = 0.1
        interpolated_y = 0.05
        interpolated_yaw = 0.2
        controller.map_x = expected_x + (
            math.cos(expected_yaw) * interpolated_x
            - math.sin(expected_yaw) * interpolated_y
        )
        controller.map_y = expected_y + (
            math.sin(expected_yaw) * interpolated_x
            + math.cos(expected_yaw) * interpolated_y
        )
        controller.map_yaw = expected_yaw + interpolated_yaw
        # These deliberately unrelated latest values expose the old callback-
        # order-dependent implementation if it is ever restored.
        controller.odom_x = 8.0
        controller.odom_y = -3.0
        controller.odom_yaw = -1.0

        self.assertTrue(controller._refresh_map_transform())

        transform = controller.map_from_odom
        self.assertIsInstance(transform, RigidTransform2D)
        self.assertEqual(transform.source_frame, "odom")
        self.assertEqual(transform.target_frame, "map")
        self.assertAlmostEqual(transform.target_from_source_x, expected_x, places=8)
        self.assertAlmostEqual(transform.target_from_source_y, expected_y, places=8)
        self.assertAlmostEqual(
            transform.target_from_source_yaw, expected_yaw, places=8
        )
        mapped = transform.apply_pose(
            Pose2D(interpolated_x, interpolated_y, interpolated_yaw)
        )
        self.assertAlmostEqual(mapped.x, controller.map_x, places=8)
        self.assertAlmostEqual(mapped.y, controller.map_y, places=8)
        self.assertAlmostEqual(mapped.yaw, controller.map_yaw, places=8)

    def test_common_map_transform_rejects_excessive_timestamp_skew(self):
        controller = self.make_map_pose_harness()
        controller.map_pose_stamp = controller_module.rospy.Time.from_sec(10.20)
        controller.map_x = 1.0
        controller.map_y = 2.0
        controller.map_yaw = 0.3
        controller.odom_history.append((self.now(), 0.0, 0.0, 0.0, "odom"))
        controller.map_from_odom = RigidTransform2D(
            0.0, 0.0, 0.0, source_frame="odom", target_frame="map"
        )

        self.assertFalse(controller._refresh_map_transform())
        self.assertIsNone(controller.map_from_odom)

    def test_lidar_alignment_freezes_common_course_transform(self):
        controller = ObstacleMissionController.__new__(
            ObstacleMissionController
        )
        controller.odom_x = 2.4
        controller.odom_y = -0.8
        controller.odom_yaw = math.radians(38.0)
        controller.course_odom_heading = math.radians(31.0)
        controller.odom_frame = "odom"
        progress = 0.37
        lateral = -0.045
        expected_robot_heading = normalize_angle(
            controller.odom_yaw - controller.course_odom_heading
        )

        controller._freeze_course_alignment(progress, lateral)

        transform = controller.odom_from_course
        self.assertIsInstance(transform, RigidTransform2D)
        self.assertEqual(transform.source_frame, "obstacle_course")
        self.assertEqual(transform.target_frame, "odom")
        recovered = transform.inverse().apply_pose(controller._current_pose())
        self.assertAlmostEqual(recovered.x, progress, places=12)
        self.assertAlmostEqual(recovered.y, lateral, places=12)
        self.assertAlmostEqual(
            recovered.yaw,
            expected_robot_heading,
            places=12,
        )
        self.assertIsNone(controller.course_odom_heading)
        self.assertEqual(
            controller._course_template_coordinates(),
            (recovered.x, -recovered.y),
        )

    def test_committed_path_reports_frozen_map_boundary_clearance(self):
        controller = ObstacleMissionController.__new__(
            ObstacleMissionController
        )
        controller.map_safety_boundary = (
            controller_module.AxisAlignedBoundsBoundary(
                -2.0, 2.0, -2.0, 2.0
            )
        )
        controller.map_from_odom = RigidTransform2D(
            0.20,
            -0.10,
            math.radians(5.0),
            source_frame="odom",
            target_frame="map",
        )
        path = self.straight_path()

        controller._attach_map_safety_boundary(path)
        validation = RectanglePathChecker(
            Footprint(0.067645, 0.118073, 0.0903)
        ).validator.validate_path(path)

        self.assertEqual(len(path.safety.map_boundaries), 1)
        self.assertTrue(validation.safe)
        self.assertTrue(math.isfinite(validation.minimum_map_clearance))
        self.assertGreater(validation.minimum_map_clearance, 0.0)

    def test_scan_waits_for_right_odom_bracket_without_erasing_valid_scan(self):
        controller = self.make_scan_queue_harness()
        controller.odom_history.append(
            (
                controller_module.rospy.Time.from_sec(10.00),
                0.0,
                0.0,
                0.0,
                "odom",
            )
        )
        old_points = controller.scan_points.copy()
        old_stamp = controller.scan_stamp
        scan_stamp = controller_module.rospy.Time.from_sec(10.05)
        self.seconds = 10.06

        controller.scan_callback(self.scan_message(scan_stamp))

        self.assertEqual(controller.scan_stamp, old_stamp)
        np.testing.assert_array_equal(controller.scan_points, old_points)
        self.assertEqual(controller.scan_generation, 1)
        self.assertEqual(len(controller.pending_scans), 1)

        odometry = Odometry()
        odometry.header.stamp = controller_module.rospy.Time.from_sec(10.10)
        odometry.header.frame_id = "odom"
        odometry.pose.pose.position.x = 0.10
        self.seconds = 10.11
        controller.odom_callback(odometry)

        self.assertEqual(len(controller.pending_scans), 0)
        self.assertEqual(controller.scan_stamp, scan_stamp)
        self.assertEqual(controller.scan_generation, 2)
        self.assertAlmostEqual(controller.scan_odom_pose[0], 0.05, places=8)
        self.assertAlmostEqual(controller.scan_odom_pose[1], 0.0, places=8)
        self.assertFalse(np.array_equal(controller.scan_points, old_points))

    def test_pending_scan_queue_is_bounded_without_erasing_valid_scan(self):
        controller = self.make_scan_queue_harness()
        controller.pending_scan_limit = 2
        old_points = controller.scan_points.copy()
        old_stamp = controller.scan_stamp
        for seconds in (10.01, 10.02, 10.03):
            scan = {
                "source_stamp": controller_module.rospy.Time.from_sec(seconds),
                "received": self.now(),
                "points": np.asarray([[seconds, 0.0]], dtype=np.float64),
            }
            controller._enqueue_pending_scan(scan, self.now())

        self.assertEqual(len(controller.pending_scans), 2)
        queued_stamps = [
            scan["source_stamp"] for scan in controller.pending_scans
        ]
        self.assertEqual(
            queued_stamps,
            [
                controller_module.rospy.Time.from_sec(10.02),
                controller_module.rospy.Time.from_sec(10.03),
            ],
        )
        self.assertEqual(controller.scan_stamp, old_stamp)
        np.testing.assert_array_equal(controller.scan_points, old_points)

    def test_pending_scan_with_excessive_bracket_skew_is_discarded_only(self):
        controller = self.make_scan_queue_harness()
        old_points = controller.scan_points.copy()
        old_stamp = controller.scan_stamp
        controller.odom_history.extend(
            (
                (
                    controller_module.rospy.Time.from_sec(10.00),
                    0.0,
                    0.0,
                    0.0,
                    "odom",
                ),
                (
                    controller_module.rospy.Time.from_sec(10.20),
                    0.2,
                    0.0,
                    0.0,
                    "odom",
                ),
            )
        )
        controller.pending_scans.append(
            {
                "source_stamp": controller_module.rospy.Time.from_sec(10.10),
                "received": self.now(),
                "points": np.asarray([[1.0, 0.0]], dtype=np.float64),
            }
        )
        self.seconds = 10.20

        self.assertEqual(controller._take_ready_scans(self.now()), [])

        self.assertEqual(len(controller.pending_scans), 0)
        self.assertEqual(controller.scan_stamp, old_stamp)
        np.testing.assert_array_equal(controller.scan_points, old_points)

    def test_gate_acquisition_takes_ownership_before_publishing(self):
        controller = self.make_path_harness()
        controller.lock = threading.RLock()
        controller.state = controller.WAIT_GATE
        controller.zone_gate = False
        controller.zone_inside = True
        controller.zone_cleared = False
        controller.start_requested = False
        controller.revoke_requested = False
        controller.manual_stop = False
        controller.mission_has_control = False
        controller.handoff_ambiguous = False
        controller.state_pub = RecordingPublisher()
        controller.cmd_pub = RecordingPublisher()
        controller.acquisition_heading_tolerance = 1.0
        controller.maximum_angular_velocity = 0.8
        controller.entry_velocity_cap = 0.10
        controller.tracking_position_tolerance = 0.055
        controller.tracking_heading_tolerance = math.radians(28.0)
        controller.observed_lane_linear = 0.06
        controller.observed_lane_angular = 0.0
        controller._start_run = mock.Mock(
            side_effect=lambda: controller._set_state(controller.ACQUIRING)
        )
        controller._course_heading_error = mock.Mock(return_value=0.0)
        controller._acquisition_data_problem = mock.Mock(return_value=None)
        candidate_path = controller.committed_path
        candidate_follower = controller.path_follower
        candidate_validation = ValidationResult(True)
        controller.committed_path = None
        controller.path_follower = PathFollower(self.tracking_config())
        controller.fixed_path_validation = None

        def commit_candidate(_now):
            controller.committed_path = candidate_path
            controller.path_follower = candidate_follower
            controller.fixed_path_validation = candidate_validation
            return True

        controller._attempt_path = mock.Mock(side_effect=commit_candidate)
        tracking = candidate_follower.calculate_tracking(
            controller._current_pose()
        )
        controller._validate_committed_path = mock.Mock(
            return_value=(tracking, controller._current_pose())
        )
        controller._common_path_command = mock.Mock(return_value=Twist())
        handoffs = []

        def handoff(enabled):
            handoffs.append(enabled)
            controller.mission_has_control = not enabled
            controller.handoff_ambiguous = False
            return True

        controller._set_lane_controller = handoff
        controller.gate_callback(Bool(data=True))
        controller.control_callback(None)

        self.assertEqual(handoffs, [])
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(controller.state, controller.ACQUIRING)
        self.assertEqual(len(controller.cmd_pub.messages), 0)
        controller.control_callback(None)

        controller._attempt_path.assert_called_once()
        self.assertEqual(handoffs, [False])
        self.assertTrue(controller.mission_has_control)
        self.assertEqual(controller.state, controller.AVOIDING)
        self.assertEqual(len(controller.cmd_pub.messages), 1)

    def test_committed_common_path_command_reuses_tracking(self):
        controller = self.make_path_harness()
        tracking = controller.path_follower.calculate_tracking(
            controller._current_pose()
        )
        with mock.patch.object(
            controller.path_follower,
            "calculate_tracking",
            side_effect=AssertionError("tracking was recalculated"),
        ):
            command = controller._common_path_command(
                self.now(), tracking=tracking
            )

        self.assertIsInstance(command, Twist)
        self.assertGreater(command.linear.x, 0.0)
        self.assertEqual(
            controller.path_follower.diagnostics.target_index,
            tracking.target_index,
        )

    def test_live_obstacle_inside_stopping_sweep_sets_zero_speed_limit(self):
        controller = self.make_path_harness()
        controller.odom_stamp = self.now()
        controller.scan_stamp = self.now()
        controller.odom_timeout = 0.35
        controller.scan_timeout = 0.35
        controller.safety_reaction_time = 0.10
        controller.linear_deceleration = 0.80
        controller.safety_stop_margin = 0.005
        controller.spline_planner = mock.Mock(live_validation_distance=0.45)
        controller.footprint = Footprint(
            0.067645,
            0.118073,
            0.0903,
            obstacle_padding=0.010,
        )
        controller.validation_footprint = Footprint(
            0.067645,
            0.118073,
            0.0903,
            line_margin=0.005,
            localization_margin=0.002,
            tracking_margin=0.002,
        )
        controller.path_checker = RectanglePathChecker(controller.footprint)
        controller.fixed_path_validation = ValidationResult(True)
        controller._course_heading_error = mock.Mock(return_value=0.0)
        controller._planning_corridor = mock.Mock(
            return_value=(-0.25, 0.25, "surveyed")
        )
        controller._camera_corridor = mock.Mock(return_value=None)
        # This return lies inside the current expanded rectangle, so an active
        # controller must hold rather than treating it as route-selection input.
        controller._lane_points = mock.Mock(
            return_value=np.asarray([[0.02, 0.0]], dtype=np.float64)
        )
        controller._lane_points_in_odom = mock.Mock(
            return_value=np.asarray([[0.02, 0.0]], dtype=np.float64)
        )
        controller._publish_diagnostics = mock.Mock()

        tracking, tracking_pose = controller._validate_committed_path(self.now())

        self.assertIsNotNone(tracking)
        self.assertEqual(tracking_pose, controller._current_pose())
        self.assertEqual(controller.live_speed_limit, 0.0)
        self.assertLessEqual(
            controller.path_follower.diagnostics.minimum_obstacle_clearance,
            0.0,
        )

    def test_inconsistent_camera_corridor_is_not_added_to_runtime_safety(self):
        controller = self.make_runtime_harness()
        controller.map_corridor_max_residual = 0.004
        controller._planning_corridor = mock.Mock(
            return_value=(-0.25, 0.25, 0.0)
        )
        controller._camera_corridor = mock.Mock(
            return_value=(-0.05, 0.45, 0.20)
        )
        validator = controller.path_checker.validator

        with mock.patch.object(
            validator,
            "motion_safety",
            wraps=validator.motion_safety,
        ) as motion_safety:
            result = controller._validate_committed_path(self.now())

        self.assertTrue(result)
        safety = motion_safety.call_args.kwargs["safety"]
        self.assertEqual(safety.line_boundaries, ())

    def test_runtime_sweep_does_not_starve_zone_or_sensor_callbacks(self):
        controller = self.make_path_harness()
        controller.state = controller.AVOIDING
        controller.zone_gate = True
        controller.zone_inside = True
        controller.zone_cleared = False
        controller.start_requested = False
        controller.revoke_requested = False
        controller.manual_stop = False
        controller.mission_has_control = True
        controller.state_pub = RecordingPublisher()
        controller.cmd_pub = RecordingPublisher()
        controller.odom_stamp = self.now()
        controller.scan_stamp = self.now()
        controller.odom_timeout = 0.35
        controller.scan_timeout = 0.35
        controller.safety_reaction_time = 0.10
        controller.linear_deceleration = 0.80
        controller.safety_stop_margin = 0.005
        controller.spline_planner = mock.Mock(live_validation_distance=0.45)
        controller.footprint = Footprint(
            0.067645,
            0.118073,
            0.0903,
            obstacle_padding=0.010,
        )
        controller.validation_footprint = Footprint(
            0.067645,
            0.118073,
            0.0903,
            line_margin=0.005,
            localization_margin=0.002,
            tracking_margin=0.002,
        )
        controller.path_checker = RectanglePathChecker(controller.footprint)
        controller.fixed_path_validation = ValidationResult(True)
        controller._course_heading_error = mock.Mock(return_value=0.0)
        controller._planning_corridor = mock.Mock(
            return_value=(-0.25, 0.25, "surveyed")
        )
        controller._camera_corridor = mock.Mock(return_value=None)
        controller._lane_points = mock.Mock(
            return_value=np.empty((0, 2), dtype=np.float64)
        )
        controller._lane_points_in_odom = mock.Mock(
            return_value=np.empty((0, 2), dtype=np.float64)
        )
        controller._publish_diagnostics = mock.Mock()
        controller._common_path_command = mock.Mock(return_value=Twist())

        entered_sweep = threading.Event()
        release_sweep = threading.Event()
        callback_finished = threading.Event()
        worker_errors = []
        real_motion_safety = controller.path_checker.validator.motion_safety

        def blocked_motion_safety(*args, **kwargs):
            entered_sweep.set()
            if not release_sweep.wait(1.0):
                raise AssertionError("test did not release swept validation")
            return real_motion_safety(*args, **kwargs)

        def run_control():
            try:
                controller.control_callback(None)
            except BaseException as error:  # surfaced in the test thread below
                worker_errors.append(error)

        def update_zone():
            controller.inside_callback(Bool(data=False))
            callback_finished.set()

        with mock.patch.object(
            controller.path_checker.validator,
            "motion_safety",
            side_effect=blocked_motion_safety,
        ):
            control_thread = threading.Thread(target=run_control)
            control_thread.start()
            self.assertTrue(entered_sweep.wait(0.5))
            callback_thread = threading.Thread(target=update_zone)
            callback_thread.start()
            try:
                self.assertTrue(
                    callback_finished.wait(0.2),
                    "mission lock stayed held during swept validation",
                )
            finally:
                release_sweep.set()
            callback_thread.join(1.0)
            control_thread.join(1.0)

        self.assertFalse(control_thread.is_alive())
        self.assertFalse(callback_thread.is_alive())
        self.assertEqual(worker_errors, [])
        self.assertEqual(controller.state, controller.REJOINING)

    def test_manual_stop_or_revoke_during_sweep_cannot_publish_motion(self):
        for interruption in ("manual_stop", "revoke"):
            with self.subTest(interruption=interruption):
                controller = self.make_runtime_harness()
                moving = Twist()
                moving.linear.x = 0.10
                controller._common_path_command = mock.Mock(return_value=moving)

                def interrupt(_captured):
                    if interruption == "manual_stop":
                        controller.manual_stop_callback(Bool(data=True))
                    else:
                        controller.gate_callback(Bool(data=False))

                self.run_control_while_sweep_is_blocked(controller, interrupt)

                controller._common_path_command.assert_not_called()
                self.assertTrue(controller.cmd_pub.messages)
                self.assertTrue(
                    all(
                        abs(message.linear.x) <= 1e-12
                        and abs(message.angular.z) <= 1e-12
                        for message in controller.cmd_pub.messages
                    )
                )
                if interruption == "manual_stop":
                    self.assertTrue(controller.manual_stop)
                    self.assertEqual(controller.state, controller.AVOIDING)
                else:
                    self.assertFalse(controller.mission_has_control)
                    self.assertEqual(controller.state, controller.WAIT_GATE)

    def test_shutdown_during_sweep_cannot_publish_motion_after_stop(self):
        controller = self.make_runtime_harness()
        moving = Twist()
        moving.linear.x = 0.10
        controller._common_path_command = mock.Mock(return_value=moving)

        self.run_control_while_sweep_is_blocked(
            controller, lambda _captured: controller.shutdown()
        )

        self.assertTrue(controller.shutting_down)
        controller._common_path_command.assert_not_called()
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        self.assertAlmostEqual(controller.cmd_pub.messages[0].linear.x, 0.0)
        self.assertAlmostEqual(controller.cmd_pub.messages[0].angular.z, 0.0)

    def test_odom_change_during_sweep_keeps_snapshot_pose_and_tracking(self):
        controller = self.make_runtime_harness()
        follower = controller.path_follower
        original_goal_status = follower.goal_status
        original_common_path_command = controller._common_path_command
        follower.goal_status = mock.Mock(wraps=original_goal_status)
        controller._common_path_command = mock.Mock(
            wraps=original_common_path_command
        )

        def advance_odometry(_captured):
            self.seconds += 0.02
            message = Odometry()
            message.header.stamp = self.now()
            message.header.frame_id = "odom"
            message.pose.pose.position.x = 0.40
            controller.odom_callback(message)

        with mock.patch.object(
            follower,
            "calculate_tracking",
            wraps=follower.calculate_tracking,
        ) as calculate_tracking:
            captured = self.run_control_while_sweep_is_blocked(
                controller, advance_odometry
            )

        self.assertAlmostEqual(controller.odom_x, 0.40, places=12)
        self.assertAlmostEqual(captured["pose"].x, 0.0, places=12)
        self.assertEqual(calculate_tracking.call_count, 1)
        self.assertEqual(calculate_tracking.call_args.args[0], captured["pose"])
        follower.goal_status.assert_called_once()
        self.assertEqual(
            follower.goal_status.call_args.args[0], captured["pose"]
        )
        self.assertIs(
            follower.goal_status.call_args.kwargs["tracking"],
            captured["tracking"],
        )
        controller._common_path_command.assert_called_once()
        self.assertEqual(
            controller._common_path_command.call_args.kwargs["pose"],
            captured["pose"],
        )
        self.assertIs(
            controller._common_path_command.call_args.kwargs["tracking"],
            captured["tracking"],
        )
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        self.assertGreater(controller.cmd_pub.messages[0].linear.x, 0.0)

    def test_sweep_result_is_discarded_when_its_snapshot_ages_out(self):
        controller = self.make_runtime_harness()
        controller.live_speed_limit = 0.073
        controller._common_path_command = mock.Mock()

        def age_snapshot(_captured):
            self.seconds += max(controller.odom_timeout, controller.scan_timeout) + 0.01

        self.run_control_while_sweep_is_blocked(controller, age_snapshot)

        controller._common_path_command.assert_not_called()
        self.assertAlmostEqual(controller.live_speed_limit, 0.073, places=12)
        self.assertTrue(controller.cmd_pub.messages)
        self.assertTrue(
            all(
                abs(message.linear.x) <= 1e-12
                and abs(message.angular.z) <= 1e-12
                for message in controller.cmd_pub.messages
            )
        )

    def test_sweep_result_is_discarded_after_path_or_follower_replacement(self):
        for replacement in ("path", "follower"):
            with self.subTest(replacement=replacement):
                controller = self.make_runtime_harness()
                controller.live_speed_limit = 0.073
                controller._common_path_command = mock.Mock()

                def replace_active_object(_captured):
                    with controller.lock:
                        if replacement == "path":
                            controller.committed_path = self.straight_path()
                        else:
                            follower = PathFollower(self.tracking_config())
                            follower.reset(
                                controller.committed_path,
                                controller._current_pose(),
                            )
                            controller.path_follower = follower

                self.run_control_while_sweep_is_blocked(
                    controller, replace_active_object
                )

                controller._common_path_command.assert_not_called()
                self.assertAlmostEqual(
                    controller.live_speed_limit, 0.073, places=12
                )
                self.assertEqual(controller.cmd_pub.messages, [])

    def test_completion_returns_cmd_vel_to_lane_controller(self):
        controller = ObstacleMissionController.__new__(
            ObstacleMissionController
        )
        controller.planner_status_pub = RecordingPublisher()
        controller.speed_limit_pub = RecordingPublisher()
        controller.state_pub = RecordingPublisher()
        controller.lane_resume_max_velocity = 0.30
        controller.state = controller.REJOINING
        controller.mission_has_control = True
        handoffs = []

        def handoff(enabled):
            handoffs.append(enabled)
            controller.mission_has_control = not enabled
            return True

        controller._set_lane_controller = handoff
        controller._complete()

        self.assertEqual(handoffs, [True])
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(controller.state, controller.COMPLETE)

    def test_completed_terminal_holds_until_zone_clear_then_hands_off(self):
        controller = self.make_runtime_harness()
        controller.state = controller.REJOINING
        controller.zone_inside = False
        controller.zone_cleared = False
        controller.odom_x = float(controller.committed_path.x[-1])
        controller.odom_y = float(controller.committed_path.y[-1])
        controller.odom_yaw = float(controller.committed_path.heading[-1])
        pose = controller._current_pose()
        controller.path_follower.reset(
            controller.committed_path,
            pose,
            initial_linear=0.10,
        )
        tracking = controller.path_follower.calculate_tracking(pose)
        controller._validate_committed_path = mock.Mock(
            return_value=(tracking, pose)
        )

        with mock.patch.object(
            controller.path_follower,
            "terminal_hold",
            wraps=controller.path_follower.terminal_hold,
        ) as terminal_hold:
            controller.control_callback(None)

        terminal_hold.assert_called_once()
        self.assertEqual(controller.state, controller.REJOINING)
        self.assertTrue(controller.mission_has_control)
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        self.assertGreaterEqual(controller.cmd_pub.messages[-1].linear.x, 0.0)
        self.assertLess(controller.cmd_pub.messages[-1].linear.x, 0.10)

        controller.zone_cleared = True
        tracking = controller.path_follower.calculate_tracking(pose)
        controller._validate_committed_path.return_value = (tracking, pose)
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.COMPLETE)
        self.assertFalse(controller.mission_has_control)
        controller._set_lane_controller.assert_called_once_with(True)
        self.assertEqual(len(controller.cmd_pub.messages), 1)

    def test_handoff_failure_uses_shared_stop_without_cmd_vel_publish(self):
        controller = ObstacleMissionController.__new__(
            ObstacleMissionController
        )
        controller.state = controller.ACQUIRING
        controller.mission_has_control = False
        controller.handoff_ambiguous = True
        controller.cmd_pub = RecordingPublisher()
        controller.emergency_stop_pub = RecordingPublisher()
        controller.planner_status_pub = RecordingPublisher()
        controller.state_pub = RecordingPublisher()
        controller._set_lane_controller = mock.Mock(return_value=False)
        controller._stop_lane_controller = mock.Mock(return_value=True)

        controller._fail("handoff unavailable")

        self.assertEqual(controller.cmd_pub.messages, [])
        controller._stop_lane_controller.assert_called_once_with()
        self.assertEqual(len(controller.emergency_stop_pub.messages), 1)
        self.assertTrue(controller.emergency_stop_pub.messages[0].data)
        self.assertEqual(controller.state, controller.FAILED)

    def test_scan_pose_interpolates_and_rejects_excessive_skew(self):
        controller = ObstacleMissionController.__new__(
            ObstacleMissionController
        )
        controller.maximum_scan_odom_skew = 0.075
        controller.odom_history = deque(
            (
                (self.now(), 0.0, 0.0, math.radians(179.0), "odom"),
                (
                    controller_module.rospy.Time.from_sec(self.seconds + 0.05),
                    0.10,
                    0.02,
                    math.radians(-179.0),
                    "odom",
                ),
            )
        )

        middle = controller._odom_pose_at(
            controller_module.rospy.Time.from_sec(self.seconds + 0.025)
        )
        self.assertAlmostEqual(middle[0], 0.05, places=12)
        self.assertAlmostEqual(middle[1], 0.01, places=12)
        self.assertAlmostEqual(abs(middle[2]), math.pi, places=9)
        self.assertIsNone(
            controller._odom_pose_at(
                controller_module.rospy.Time.from_sec(self.seconds + 0.20)
            )
        )


if __name__ == "__main__":
    unittest.main()
