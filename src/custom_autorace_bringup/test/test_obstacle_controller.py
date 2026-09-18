#!/usr/bin/env python3

from collections import deque
import math
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml


NODE_DIR = Path(__file__).resolve().parents[1] / "nodes"
CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "obstacle_mission_gazebo.yaml"
)
if str(NODE_DIR) not in sys.path:
    sys.path.insert(0, str(NODE_DIR))

import obstacle_mission_controller as controller_module
from obstacle_mission_controller import (
    DetectedBarrierFace,
    ObstacleMissionController,
    obstacle_barrier_template,
    register_obstacle_outer_faces,
)
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64MultiArray, Header

from custom_autorace_bringup.obstacle_planner import Footprint, RectanglePathChecker
from custom_autorace_bringup.local_registration import (
    RegistrationConfig,
    TemporalRegistrationConfig,
    TemporalRegistrationFilter,
    entry_plane_progress,
)
from custom_autorace_bringup.path_following import (
    CommonPath,
    GoalTolerance,
    PathFollower,
    PathSafety,
    Pose2D,
    RigidTransform2D,
    SafetyMargins,
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
        controller.odom_generation = 0
        controller.odom_stamp = self.now()
        controller.odom_frame = "odom"
        controller.path_follower.reset(
            controller.committed_path, controller._current_pose()
        )
        controller.live_speed_limit = math.inf
        controller.live_requires_stop = False
        controller.control_period = 0.05
        controller.last_command_time = None
        controller.entry_handoff_velocity_tolerance = 0.005
        controller.entry_projection_position_tolerance = 0.015
        controller.entry_projection_heading_tolerance = math.radians(8.0)
        controller.observed_lane_linear = 0.0
        controller.observed_lane_angular = 0.0
        controller.observed_lane_command_received = None
        controller.lane_command_generation = 0
        controller.gate_lane_command_generation = 0
        controller.gate_odom_generation = 0
        controller.gate_lane_command_received = None
        controller.gate_odom_stamp = None
        controller.gate_odom_frame = None
        controller.handoff_refresh_scan_generation = -1
        controller.handoff_refresh_lane_command_generation = -1
        controller.handoff_refresh_odom_generation = -1
        controller.handoff_refresh_lane_command_received = None
        controller.handoff_refresh_odom_stamp = None
        controller.handoff_refresh_odom_frame = None
        controller.handoff_service_pending = False
        controller.handoff_takeover_odom_generation = -1
        controller.handoff_takeover_odom_stamp = None
        controller.handoff_takeover_odom_frame = None
        controller.scan_generation = 0
        controller.remaining_distance = controller.committed_path.length
        controller.diagnostics_pub = RecordingPublisher()
        controller.planner_status_pub = RecordingPublisher()
        return controller

    def make_map_pose_harness(self):
        controller = ObstacleMissionController.__new__(
            ObstacleMissionController
        )
        controller.lock = threading.RLock()
        controller.maximum_scan_odom_skew = 0.075
        controller.odom_history = deque()
        controller.odom_ready = False
        controller.odom_frame = "odom"
        controller.odom_generation = 0
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
        controller.start_requested = False
        controller.revoke_requested = False
        controller.arm_generation = 1
        controller.armed_at = self.now()
        controller.prepared_generation = 1
        controller.prepared_stamp = self.now()
        controller.ready_published_generation = 1
        controller.last_ready_stamp = self.now()
        controller.registration_filter = TemporalRegistrationFilter()
        controller.registration_covariance = tuple()
        controller.registration_diagnostics = None
        controller.registration_source_stamp = None
        controller.last_registration_scan_generation = -1
        controller.last_plan_scan_generation = -1
        controller.planning_seconds = 0.0
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
        controller.maximum_scan_odom_skew = 0.075
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
        """Run one timer tick and update state during its unlocked route sweep."""
        entered_sweep = threading.Event()
        release_sweep = threading.Event()
        worker_errors = []
        captured = {}
        real_validate_path = controller.path_checker.validator.validate_path

        def blocked_route_safety(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            captured["pose"] = controller._current_pose()
            if controller.lock._is_owned():
                raise AssertionError("mission lock is held during route validation")
            entered_sweep.set()
            if not release_sweep.wait(1.0):
                raise AssertionError("test did not release swept validation")
            return real_validate_path(*args, **kwargs)

        def run_control():
            try:
                controller.control_callback(None)
            except BaseException as error:  # surfaced in the test thread below
                worker_errors.append(error)

        with mock.patch.object(
            controller.path_checker.validator,
            "validate_path",
            side_effect=blocked_route_safety,
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

    def test_rejected_boundary_view_does_not_refresh_old_lane_geometry(self):
        controller = ObstacleMissionController.__new__(
            ObstacleMissionController
        )
        controller.lock = threading.RLock()
        controller.minimum_boundary_gap = 300.0
        controller.maximum_boundary_gap = 900.0
        controller.odom_ready = True
        controller.odom_from_course = None
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

    def test_camera_line_boundary_keeps_only_absolute_observed_band(self):
        controller = ObstacleMissionController.__new__(
            ObstacleMissionController
        )
        controller.lock = threading.RLock()
        controller.minimum_boundary_gap = 300.0
        controller.maximum_boundary_gap = 900.0
        controller.odom_ready = True
        controller.odom_from_course = RigidTransform2D(
            0.0,
            0.0,
            0.0,
            source_frame="obstacle_course",
            target_frame="odom",
        )
        controller.boundary_heading_limit = math.radians(12.0)
        controller.state = controller.AVOIDING
        controller.lane_width = 0.50
        controller.pixels_per_meter = 1000.0
        controller.scale_filter_alpha = 0.0
        controller.line_position_alpha = 0.0
        controller.image_center = 500.0
        controller.boundary_sample_forward = 0.14
        controller.boundary_support_half_length = 0.004
        controller.camera_timeout = 0.50
        controller.map_corridor_max_residual = 0.004
        controller.odom_x = 1.0
        controller.odom_y = 0.10
        controller.odom_yaw = 0.0
        controller.line_center_absolute = None
        controller.line_observation_progress_absolute = None
        controller.line_observation_stamp = None

        controller.boundary_callback(
            Float64MultiArray(data=[200.0, 800.0, 1.0, 1.0])
        )
        boundary = controller._camera_line_boundary(
            self.now(), (-0.25, 0.25, 0.0)
        )

        self.assertIsNotNone(boundary)
        self.assertAlmostEqual(boundary.right, -0.15, places=12)
        self.assertAlmostEqual(boundary.left, 0.35, places=12)
        self.assertAlmostEqual(boundary.minimum_progress, 1.136, places=12)
        self.assertAlmostEqual(boundary.maximum_progress, 1.144, places=12)

        controller.odom_x = 1.08
        moved = controller._camera_line_boundary(
            self.now(), (-0.25, 0.25, 0.0)
        )
        self.assertAlmostEqual(
            moved.minimum_progress, boundary.minimum_progress, places=12
        )
        self.assertAlmostEqual(
            moved.maximum_progress, boundary.maximum_progress, places=12
        )
        self.assertTrue(
            math.isinf(
                moved.clearance(
                    Pose2D(1.40, 0.10, 0.0),
                    Footprint(0.067645, 0.118073, 0.0903),
                )
            )
        )

    @staticmethod
    def barrier_geometry():
        return (
            (0.5200, 0.1475, 0.1000, 0.2500),
            (0.9800, -0.1025, 0.1000, 0.2500),
            (1.4440, 0.1475, 0.1000, 0.2500),
        )

    def test_gazebo_config_separates_acquisition_from_ready_limits(self):
        with CONFIG_PATH.open("r", encoding="utf-8") as stream:
            obstacle = yaml.safe_load(stream)["obstacle"]
        registration = obstacle["template"]["registration"]

        self.assertEqual(
            obstacle["safety"]["local_bounds"], [-2.0, 2.0, -2.0, 2.0]
        )
        self.assertNotIn("map_bounds", obstacle["safety"])
        self.assertEqual(registration["outer_face_minimum_length"], 0.18)
        self.assertEqual(registration["outer_baseline_tolerance"], 0.035)
        self.assertEqual(
            registration["outer_parallel_heading_tolerance_deg"], 6.0
        )
        self.assertEqual(registration["finite_endpoint_tolerance"], 0.006)
        self.assertEqual(registration["side_minimum_length"], 0.055)
        self.assertEqual(registration["side_heading_tolerance_deg"], 12.0)
        self.assertEqual(registration["side_association_tolerance"], 0.025)
        self.assertEqual(
            registration["outer_baseline_lateral_tolerance"], 0.015
        )
        self.assertEqual(registration["outer_baseline_lateral_maximum"], 0.040)
        self.assertEqual(
            registration["axis_consistency_tolerance_deg"], 9.0
        )
        self.assertEqual(registration["ambiguity_position"], 0.004)
        self.assertEqual(registration["ambiguity_heading_deg"], 0.25)
        self.assertEqual(registration["confirmation_maximum_gap"], 1.10)
        self.assertEqual(registration["acquisition_maximum_lateral"], 0.80)
        self.assertEqual(
            registration["acquisition_maximum_heading_deg"], 105.0
        )
        self.assertEqual(registration["entry_maximum_lateral"], 0.20)
        self.assertEqual(registration["entry_maximum_heading_deg"], 15.0)
        planner = obstacle["planner"]
        self.assertEqual(planner["entry_connector_minimum_join_distance"], 0.12)
        self.assertEqual(planner["entry_connector_maximum_join_distance"], 0.32)
        self.assertEqual(planner["entry_connector_join_step"], 0.02)
        self.assertIn(0.12, planner["entry_connector_tangent_ratios"])
        self.assertEqual(
            obstacle["control"]["entry_handoff_velocity_tolerance"],
            0.005,
        )
        self.assertEqual(obstacle["control"]["entry_velocity_cap"], 0.09)
        self.assertEqual(
            obstacle["control"]["entry_projection_position_tolerance"],
            0.015,
        )
        self.assertEqual(
            obstacle["control"]["entry_projection_heading_tolerance_deg"],
            8.0,
        )
        self.assertEqual(
            obstacle["course"]["boundary_support_half_length"], 0.004
        )

    def test_outer_face_baseline_registers_side_approach_independent_of_order(self):
        template = obstacle_barrier_template(self.barrier_geometry())
        front_segments = tuple(
            segment
            for segment in template.segments
            if segment.name.endswith("_front")
        )
        side_segments = tuple(
            segment
            for segment in template.segments
            if segment.name.endswith("_side")
        )
        self.assertEqual(
            tuple(segment.longitudinal_weight for segment in front_segments),
            (0.0, 1.0, 0.0),
        )
        self.assertEqual(len(side_segments), 6)
        self.assertTrue(
            all(segment.longitudinal_weight == 0.0 for segment in side_segments)
        )
        expected = RigidTransform2D(
            0.84,
            1.77,
            math.radians(89.5),
            source_frame="obstacle_course",
            target_frame="odom",
        )
        first = template.segments[0]
        last = template.segments[2]
        faces = (
            # Reverse both the collection order and one PCA endpoint order.
            DetectedBarrierFace(
                expected.apply_point(last.end),
                expected.apply_point(last.start),
                rms=0.001,
            ),
            DetectedBarrierFace(
                expected.apply_point(first.start),
                expected.apply_point(first.end),
                rms=0.001,
            ),
        )
        robot = expected.apply_pose(
            Pose2D(0.24, 0.60, math.radians(-90.0))
        )

        result = register_obstacle_outer_faces(
            template,
            faces,
            160.71,
            "odom",
            robot,
            RegistrationConfig(
                minimum_inliers=2,
                minimum_template_coverage=0.60,
                minimum_inlier_coverage=1.0,
            ),
            -1.10,
            0.20,
            0.80,
            math.radians(105.0),
            minimum_face_length=0.18,
            baseline_tolerance=0.035,
            parallel_heading_tolerance=math.radians(6.0),
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(
            result.transform.target_from_source_x,
            expected.target_from_source_x,
            places=8,
        )
        self.assertAlmostEqual(
            result.transform.target_from_source_y,
            expected.target_from_source_y,
            places=8,
        )
        self.assertAlmostEqual(
            normalize_angle(
                result.transform.target_from_source_yaw
                - expected.target_from_source_yaw
            ),
            0.0,
            places=8,
        )
        progress = entry_plane_progress(
            template.entry_plane, robot, result.transform
        )
        self.assertAlmostEqual(progress.longitudinal, -0.28, places=8)
        self.assertAlmostEqual(progress.lateral, 0.60, places=8)
        self.assertAlmostEqual(
            math.degrees(progress.heading_error), -90.0, places=8
        )

    def test_clipped_outer_midpoints_need_independent_lateral_face(self):
        template = obstacle_barrier_template(self.barrier_geometry())
        expected = RigidTransform2D(
            0.84,
            1.77,
            math.radians(89.5),
            source_frame="obstacle_course",
            target_frame="odom",
        )
        first = template.segments[0]
        last = template.segments[2]

        # Both outer fronts are clipped from the same end.  Their midpoint
        # baseline is still perfect, but each midpoint is 24 mm course-left of
        # the physical face centre.  The former point-pair registrar therefore
        # shifted the complete path by that same 24 mm.
        clipped_faces = tuple(
            DetectedBarrierFace(
                expected.apply_point((landmark.start[0], 0.0705)),
                expected.apply_point((landmark.end[0], 0.2725)),
                rms=0.001,
            )
            for landmark in (first, last)
        )
        robot = expected.apply_pose(
            Pose2D(0.24, 0.60, math.radians(-90.0))
        )

        result = register_obstacle_outer_faces(
            template,
            clipped_faces,
            160.71,
            "odom",
            robot,
            RegistrationConfig(minimum_inliers=2),
            -1.10,
            0.20,
            0.80,
            math.radians(105.0),
            baseline_lateral_tolerance=0.020,
            axis_consistency_tolerance=math.radians(6.0),
        )

        self.assertIsNone(result)

        upper_side = next(
            segment
            for segment in template.segments
            if segment.name == "barrier_0_upper_side"
        )
        visible_side = DetectedBarrierFace(
            expected.apply_point((upper_side.start[0] + 0.005, upper_side.start[1])),
            expected.apply_point((upper_side.end[0] - 0.004, upper_side.end[1])),
            rms=0.001,
        )
        result = register_obstacle_outer_faces(
            template,
            (clipped_faces[1], visible_side, clipped_faces[0]),
            160.72,
            "odom",
            robot,
            RegistrationConfig(minimum_inliers=2),
            -1.10,
            0.20,
            0.80,
            math.radians(105.0),
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(
            result.transform.target_from_source_x,
            expected.target_from_source_x,
            places=8,
        )
        self.assertAlmostEqual(
            result.transform.target_from_source_y,
            expected.target_from_source_y,
            places=8,
        )
        self.assertAlmostEqual(
            normalize_angle(
                result.transform.target_from_source_yaw
                - expected.target_from_source_yaw
            ),
            0.0,
            places=8,
        )

    def test_gazebo_clipped_front_probe_uses_stable_visible_side(self):
        template = obstacle_barrier_template(self.barrier_geometry())
        expected = RigidTransform2D(
            1.6375,
            0.0175,
            math.radians(90.0),
            source_frame="obstacle_course",
            target_frame="odom",
        )
        robot = Pose2D(0.95, 0.255, 0.0)

        # Representative official-approach probe at robot world x ~= 1.03:
        # clipped first/last front midpoints and the stable visible side face.
        first_front = DetectedBarrierFace(
            (1.5818, 0.48505),
            (1.3684, 0.48915),
            rms=0.002,
        )
        last_front = DetectedBarrierFace(
            (1.61075, 1.41455),
            (1.34885, 1.42185),
            rms=0.002,
        )
        visible_side = DetectedBarrierFace(
            (1.36226, 0.48825),
            (1.36194, 0.57935),
            rms=0.002,
        )

        without_side = register_obstacle_outer_faces(
            template,
            (first_front, last_front),
            161.0,
            "odom",
            robot,
            RegistrationConfig(minimum_inliers=2),
            -1.10,
            0.20,
            0.80,
            math.radians(105.0),
        )
        self.assertIsNone(without_side)

        result = register_obstacle_outer_faces(
            template,
            (last_front, visible_side, first_front),
            161.1,
            "odom",
            robot,
            RegistrationConfig(minimum_inliers=2),
            -1.10,
            0.20,
            0.80,
            math.radians(105.0),
        )

        self.assertIsNotNone(result)
        transform_error = math.hypot(
            result.transform.target_from_source_x
            - expected.target_from_source_x,
            result.transform.target_from_source_y
            - expected.target_from_source_y,
        )
        self.assertLess(transform_error, 0.008)
        self.assertLess(
            abs(
                normalize_angle(
                    result.transform.target_from_source_yaw
                    - expected.target_from_source_yaw
                )
            ),
            math.radians(0.5),
        )

    def test_sparse_official_lidar_faces_confirm_before_obstacle_entry(self):
        template = obstacle_barrier_template(self.barrier_geometry())

        def face(midpoint, length, heading_deg):
            heading = math.radians(heading_deg)
            offset = (
                0.5 * length * math.cos(heading),
                0.5 * length * math.sin(heading),
            )
            return DetectedBarrierFace(
                (midpoint[0] - offset[0], midpoint[1] - offset[1]),
                (midpoint[0] + offset[0], midpoint[1] + offset[1]),
                rms=0.005,
            )

        # Three clean signatures replayed from the source-stamped official
        # Gazebo approach. Occluded scans occur between these observations;
        # only these accepted hypotheses count toward confirmation.
        replays = (
            (
                168.205,
                Pose2D(0.100, 2.009, math.radians(-0.8)),
                (
                    face((0.6664, 2.2434), 0.2011, 178.57),
                    face((0.6651, 3.1665), 0.2418, 176.07),
                    face((0.5653, 2.2938), 0.0875, 89.21),
                ),
            ),
            (
                168.705,
                Pose2D(0.239, 2.008, math.radians(-0.8)),
                (
                    face((0.6818, 2.2422), 0.2095, 178.95),
                    face((0.6715, 3.1707), 0.2507, 177.31),
                    face((0.5634, 2.2884), 0.1005, 89.44),
                ),
            ),
            (
                169.705,
                Pose2D(0.447, 2.006, math.radians(3.1)),
                (
                    face((0.6794, 2.2408), 0.2220, -178.94),
                    face((0.6761, 3.1661), 0.2317, 175.63),
                    face((0.5641, 2.2805), 0.0979, 90.67),
                ),
            ),
        )
        temporal_filter = TemporalRegistrationFilter(
            TemporalRegistrationConfig(
                required_confirmations=3,
                maximum_gap=1.10,
                maximum_position_delta=0.025,
                maximum_heading_delta=math.radians(3.0),
            )
        )
        states = []
        for stamp, robot, faces in replays:
            result = register_obstacle_outer_faces(
                template,
                faces,
                stamp,
                "odom",
                robot,
                RegistrationConfig(minimum_inliers=2),
                -1.10,
                0.20,
                0.80,
                math.radians(105.0),
                baseline_lateral_tolerance=0.015,
                baseline_lateral_maximum=0.030,
                axis_consistency_tolerance=math.radians(8.0),
                ambiguity_position=0.004,
                ambiguity_heading=math.radians(0.25),
            )
            self.assertIsNotNone(result)
            states.append(temporal_filter.update(result))

        self.assertEqual(
            [state.confirmation_count for state in states], [1, 2, 3]
        )
        self.assertTrue(states[-1].confirmed)
        self.assertLess(
            math.hypot(
                states[-1].transform.target_from_source_x - 0.84,
                states[-1].transform.target_from_source_y - 1.77,
            ),
            0.008,
        )
        self.assertLess(
            abs(
                normalize_angle(
                    states[-1].transform.target_from_source_yaw
                    - math.radians(90.0)
                )
            ),
            math.radians(0.5),
        )

    def test_run8_lidar_replay_confirms_with_robust_axis_consensus(self):
        template = obstacle_barrier_template(self.barrier_geometry())

        def face(midpoint, length, heading_deg, rms):
            heading = math.radians(heading_deg)
            offset = (
                0.5 * length * math.cos(heading),
                0.5 * length * math.sin(heading),
            )
            return DetectedBarrierFace(
                (midpoint[0] - offset[0], midpoint[1] - offset[1]),
                (midpoint[0] + offset[0], midpoint[1] + offset[1]),
                rms=rms,
            )

        # First confirming source-stamped PCA results from official-start run
        # 8.  The robust physical-axis vote retains a tightly clustered set
        # despite a different noisy line in each scan.
        replays = (
            (
                169.113,
                Pose2D(
                    0.144555114798,
                    2.009765597277,
                    math.radians(-0.408494679529),
                ),
                (
                    face(
                        (0.664858225581, 2.243059963641),
                        0.187847016293,
                        178.847576145260,
                        0.003384352167,
                    ),
                    face(
                        (0.663292963154, 3.167771940714),
                        0.263226493901,
                        174.144964110138,
                        0.005200420068,
                    ),
                    face(
                        (0.559203959816, 2.292539067892),
                        0.094089633807,
                        91.700177539214,
                        0.004868712703,
                    ),
                ),
            ),
            (
                169.813,
                Pose2D(
                    0.331427373487,
                    2.007204555180,
                    math.radians(-1.798774465471),
                ),
                (
                    face(
                        (0.670169024900, 2.245595563854),
                        0.189091229006,
                        0.357329772509,
                        0.003969928526,
                    ),
                    face(
                        (0.669733837989, 3.169419388382),
                        0.241340912065,
                        177.683684007780,
                        0.004827288593,
                    ),
                    face(
                        (0.562526320262, 2.294965445571),
                        0.096582633232,
                        95.518261209942,
                        0.004193353763,
                    ),
                ),
            ),
            (
                170.013,
                Pose2D(
                    0.366806154990,
                    2.006154107780,
                    math.radians(-0.424194969862),
                ),
                (
                    face(
                        (0.671180164501, 2.243272803659),
                        0.217543779680,
                        0.152894272865,
                        0.002887459492,
                    ),
                    face(
                        (0.676399970576, 3.168390715586),
                        0.239229878662,
                        178.833225036332,
                        0.006914571529,
                    ),
                    face(
                        (0.561799219950, 2.290631714992),
                        0.091364901962,
                        87.924294576794,
                        0.002449317982,
                    ),
                ),
            ),
        )
        temporal_filter = TemporalRegistrationFilter(
            TemporalRegistrationConfig(
                required_confirmations=3,
                maximum_gap=1.10,
                maximum_position_delta=0.025,
                maximum_heading_delta=math.radians(3.0),
            )
        )
        states = []
        for stamp, robot, faces in replays:
            result = register_obstacle_outer_faces(
                template,
                faces,
                stamp,
                "odom",
                robot,
                RegistrationConfig(minimum_inliers=2),
                -1.10,
                0.20,
                0.80,
                math.radians(105.0),
                baseline_lateral_tolerance=0.015,
                baseline_lateral_maximum=0.040,
                axis_consistency_tolerance=math.radians(8.0),
                ambiguity_position=0.004,
                ambiguity_heading=math.radians(0.25),
            )
            self.assertIsNotNone(result)
            states.append(temporal_filter.update(result))

        self.assertEqual(
            [state.confirmation_count for state in states], [1, 2, 3]
        )
        self.assertTrue(states[-1].confirmed)
        self.assertLess(states[-1].maximum_position_spread, 0.0048)
        self.assertLess(
            states[-1].maximum_heading_spread, math.radians(0.43)
        )
        self.assertAlmostEqual(
            states[-1].transform.target_from_source_x, 0.833079, places=5
        )
        self.assertAlmostEqual(
            states[-1].transform.target_from_source_y, 1.774069, places=5
        )
        self.assertAlmostEqual(
            math.degrees(states[-1].transform.target_from_source_yaw),
            89.9336,
            places=3,
        )

    def test_run9_lidar_replay_confirms_with_robust_axis_consensus(self):
        template = obstacle_barrier_template(self.barrier_geometry())

        def face(midpoint, length, heading_deg, rms):
            heading = math.radians(heading_deg)
            offset = (
                0.5 * length * math.cos(heading),
                0.5 * length * math.sin(heading),
            )
            return DetectedBarrierFace(
                (midpoint[0] - offset[0], midpoint[1] - offset[1]),
                (midpoint[0] + offset[0], midpoint[1] + offset[1]),
                rms=rms,
            )

        # Source-stamped PCA faces from official-start run 9.  The long outer
        # baseline and three independent line axes have one noisy member in
        # different scans.  Their robust consensus keeps all three transforms,
        # which themselves agree within 5.7 mm and 0.25 degrees.
        replays = (
            (
                160.606,
                Pose2D(
                    0.188882973017,
                    2.007000674941,
                    math.radians(-0.482516190788),
                ),
                (
                    face(
                        (0.666454372707, 2.240824374995),
                        0.198476741132,
                        0.481251380592,
                        0.004609462838,
                    ),
                    face(
                        (0.669734821707, 3.169608674413),
                        0.236944876212,
                        178.706627924962,
                        0.004056199166,
                    ),
                    face(
                        (0.560069564297, 2.286507602431),
                        0.100844494500,
                        91.737489583314,
                        0.003042890716,
                    ),
                ),
            ),
            (
                160.906,
                Pose2D(
                    0.272840628062,
                    2.005915063506,
                    math.radians(-0.914847475350),
                ),
                (
                    face(
                        (0.671785595070, 2.240597411736),
                        0.194909418094,
                        -0.140282189170,
                        0.004000578849,
                    ),
                    face(
                        (0.676203782912, 3.175514400879),
                        0.229632839946,
                        -0.428945182539,
                        0.007427187903,
                    ),
                    face(
                        (0.560672834356, 2.284415780532),
                        0.094443710243,
                        93.602832065988,
                        0.005205250384,
                    ),
                ),
            ),
            (
                161.506,
                Pose2D(
                    0.402126819420,
                    2.006022304445,
                    math.radians(2.718542524307),
                ),
                (
                    face(
                        (0.685498301343, 2.244126306818),
                        0.229824771636,
                        178.563990899368,
                        0.002766695476,
                    ),
                    face(
                        (0.685900349697, 3.169661726998),
                        0.238919759123,
                        179.524238554886,
                        0.004222522914,
                    ),
                    face(
                        (0.563724262704, 2.290275499292),
                        0.086124367677,
                        95.060294670237,
                        0.002494032534,
                    ),
                ),
            ),
        )
        temporal_filter = TemporalRegistrationFilter(
            TemporalRegistrationConfig(
                required_confirmations=3,
                maximum_gap=1.10,
                maximum_position_delta=0.025,
                maximum_heading_delta=math.radians(3.0),
            )
        )
        states = []
        for stamp, robot, faces in replays:
            common = dict(
                template=template,
                observed_faces=faces,
                stamp=stamp,
                target_frame="odom",
                robot_pose=robot,
                config=RegistrationConfig(minimum_inliers=2),
                minimum_entry_progress=-1.10,
                maximum_entry_progress=0.20,
                maximum_entry_lateral=0.80,
                maximum_entry_heading=math.radians(105.0),
                baseline_lateral_tolerance=0.015,
                axis_consistency_tolerance=math.radians(8.0),
                ambiguity_position=0.004,
                ambiguity_heading=math.radians(0.25),
            )
            result = register_obstacle_outer_faces(
                baseline_lateral_maximum=0.040,
                **common
            )
            self.assertIsNotNone(result)
            states.append(temporal_filter.update(result))

        self.assertEqual(
            [state.confirmation_count for state in states], [1, 2, 3]
        )
        self.assertTrue(states[-1].confirmed)
        self.assertLess(states[-1].maximum_position_spread, 0.0057)
        self.assertLess(
            states[-1].maximum_heading_spread, math.radians(0.25)
        )
        self.assertAlmostEqual(
            states[-1].transform.target_from_source_x, 0.832509, places=5
        )
        self.assertAlmostEqual(
            states[-1].transform.target_from_source_y, 1.774244, places=5
        )
        self.assertAlmostEqual(
            math.degrees(states[-1].transform.target_from_source_yaw),
            89.8340,
            places=3,
        )

        # Preserve the absolute cap as well as the measured acceptance.
        # Moving the last front 50 mm course-left leaves all line axes
        # unchanged but pushes the consensus baseline residual above 40 mm.
        last_front = faces[1]
        shifted_last = DetectedBarrierFace(
            (last_front.start[0] + 0.050, last_front.start[1]),
            (last_front.end[0] + 0.050, last_front.end[1]),
            rms=last_front.rms,
        )
        over_cap = dict(common)
        over_cap["observed_faces"] = (
            faces[0],
            shifted_last,
            faces[2],
        )
        self.assertIsNone(
            register_obstacle_outer_faces(
                baseline_lateral_maximum=0.040,
                **over_cap
            )
        )

    def test_run10_lidar_replay_confirms_with_robust_axis_consensus(self):
        template = obstacle_barrier_template(self.barrier_geometry())

        def face(midpoint, length, heading_deg, rms):
            heading = math.radians(heading_deg)
            offset = (
                0.5 * length * math.cos(heading),
                0.5 * length * math.sin(heading),
            )
            return DetectedBarrierFace(
                (midpoint[0] - offset[0], midpoint[1] - offset[1]),
                (midpoint[0] + offset[0], midpoint[1] + offset[1]),
                rms=rms,
            )

        # First confirming source-stamped PCA results from official-start run
        # 10.  The side-only lateral test measured 63-78 mm on these views;
        # the median of all three physical axes gives 14-37 mm while the
        # resulting transforms agree within 4.8 mm and 0.14 degrees.
        replays = (
            (
                161.806,
                Pose2D(
                    0.264466131567,
                    1.997855378078,
                    math.radians(-0.570648282576),
                ),
                (
                    face(
                        (0.676081925074, 2.238045470618),
                        0.188354127546,
                        -0.249605136715,
                        0.003408760056,
                    ),
                    face(
                        (0.666405088316, 3.161480396964),
                        0.248815296988,
                        177.415259052479,
                        0.004483073549,
                    ),
                    face(
                        (0.564254468468, 2.283992441882),
                        0.100824542747,
                        94.545092466261,
                        0.006659348509,
                    ),
                ),
            ),
            (
                161.906,
                Pose2D(
                    0.283988779805,
                    1.997826866345,
                    math.radians(0.160825544656),
                ),
                (
                    face(
                        (0.676690187088, 2.238533067409),
                        0.215459154822,
                        178.204363569948,
                        0.004545653217,
                    ),
                    face(
                        (0.669095121451, 3.164671723206),
                        0.248996039886,
                        177.367950290110,
                        0.004484876732,
                    ),
                    face(
                        (0.562955558482, 2.278540893322),
                        0.087351434240,
                        95.291564869670,
                        0.003059492685,
                    ),
                ),
            ),
            (
                162.106,
                Pose2D(
                    0.321440955661,
                    1.998521156914,
                    math.radians(1.950265542517),
                ),
                (
                    face(
                        (0.678168389201, 2.239192995281),
                        0.224016034513,
                        178.333951523294,
                        0.003066811372,
                    ),
                    face(
                        (0.669614458033, 3.162465818469),
                        0.247023464430,
                        178.892692645778,
                        0.005698191634,
                    ),
                    face(
                        (0.566925448341, 2.294600966856),
                        0.101009691946,
                        82.456853351205,
                        0.002440206047,
                    ),
                ),
            ),
        )
        temporal_filter = TemporalRegistrationFilter(
            TemporalRegistrationConfig(
                required_confirmations=3,
                maximum_gap=1.10,
                maximum_position_delta=0.025,
                maximum_heading_delta=math.radians(3.0),
            )
        )
        states = []
        thirty_mm_acceptances = 0
        for stamp, robot, faces in replays:
            common = dict(
                template=template,
                observed_faces=faces,
                stamp=stamp,
                target_frame="odom",
                robot_pose=robot,
                config=RegistrationConfig(minimum_inliers=2),
                minimum_entry_progress=-1.10,
                maximum_entry_progress=0.20,
                maximum_entry_lateral=0.80,
                maximum_entry_heading=math.radians(105.0),
                baseline_lateral_tolerance=0.015,
                axis_consistency_tolerance=math.radians(8.0),
                ambiguity_position=0.004,
                ambiguity_heading=math.radians(0.25),
            )
            thirty_mm_acceptances += int(
                register_obstacle_outer_faces(
                    baseline_lateral_maximum=0.030,
                    **common
                )
                is not None
            )
            result = register_obstacle_outer_faces(
                baseline_lateral_maximum=0.040,
                **common
            )
            self.assertIsNotNone(result)
            states.append(temporal_filter.update(result))

        self.assertEqual(thirty_mm_acceptances, 1)
        self.assertEqual(
            [state.confirmation_count for state in states], [1, 2, 3]
        )
        self.assertTrue(states[-1].confirmed)
        self.assertLess(states[-1].maximum_position_spread, 0.0048)
        self.assertLess(
            states[-1].maximum_heading_spread, math.radians(0.14)
        )
        self.assertAlmostEqual(
            states[-1].transform.target_from_source_x, 0.842027, places=5
        )
        self.assertAlmostEqual(
            states[-1].transform.target_from_source_y, 1.770271, places=5
        )
        self.assertAlmostEqual(
            math.degrees(states[-1].transform.target_from_source_yaw),
            90.5337,
            places=3,
        )

    def test_run11_lidar_replay_confirms_with_bounded_axis_uncertainty(self):
        template = obstacle_barrier_template(self.barrier_geometry())

        def face(midpoint, length, heading_deg, rms):
            heading = math.radians(heading_deg)
            offset = (
                0.5 * length * math.cos(heading),
                0.5 * length * math.sin(heading),
            )
            return DetectedBarrierFace(
                (midpoint[0] - offset[0], midpoint[1] - offset[1]),
                (midpoint[0] + offset[0], midpoint[1] + offset[1]),
                rms=rms,
            )

        # First source-stamped run-11 sequence that can satisfy the unchanged
        # 1.10 s confirmation window.  The middle view has only 0.99 mm of
        # lateral baseline residual, but its independently fitted line axes
        # span 8.965 degrees.  The bounded 9 degree tolerance admits that one
        # view while all three transforms agree within 3.3 mm/0.35 degrees.
        replays = (
            (
                160.006,
                Pose2D(
                    0.157256758104,
                    2.003429938073,
                    0.004015284064,
                ),
                (
                    face(
                        (0.665219029579, 2.238878635631),
                        0.182812360307,
                        177.032047535,
                        0.002776533393,
                    ),
                    face(
                        (0.676441985584, 3.169537377167),
                        0.241806000753,
                        178.807118986,
                        0.005637025670,
                    ),
                    face(
                        (0.565939794732, 2.287850361601),
                        0.095091433779,
                        90.749409419,
                        0.003183678590,
                    ),
                ),
            ),
            (
                160.206,
                Pose2D(
                    0.212762920187,
                    2.003735531542,
                    0.005979850465,
                ),
                (
                    face(
                        (0.673338512027, 2.240657111669),
                        0.201010751106,
                        179.596597196,
                        0.004541819093,
                    ),
                    face(
                        (0.678874592110, 3.167310012206),
                        0.232293998656,
                        174.377827700,
                        0.005302592171,
                    ),
                    face(
                        (0.563265825125, 2.286740362345),
                        0.096277839019,
                        93.342815246,
                        0.003043220080,
                    ),
                ),
            ),
            (
                161.206,
                Pose2D(
                    0.434491122157,
                    2.006702078642,
                    0.070829839559,
                ),
                (
                    face(
                        (0.677969664776, 2.238758686562),
                        0.209237628444,
                        179.919496949,
                        0.003262799151,
                    ),
                    face(
                        (0.687014552615, 3.163973701667),
                        0.235352518477,
                        -179.088198466,
                        0.002954312735,
                    ),
                    face(
                        (0.565912678224, 2.281400430638),
                        0.087166949166,
                        94.565179298,
                        0.002520090047,
                    ),
                ),
            ),
        )
        temporal_filter = TemporalRegistrationFilter(
            TemporalRegistrationConfig(
                required_confirmations=3,
                maximum_gap=1.10,
                maximum_position_delta=0.025,
                maximum_heading_delta=math.radians(3.0),
            )
        )
        states = []
        eight_degree_acceptances = 0
        for stamp, robot, faces in replays:
            common = dict(
                template=template,
                observed_faces=faces,
                stamp=stamp,
                target_frame="odom",
                robot_pose=robot,
                config=RegistrationConfig(minimum_inliers=2),
                minimum_entry_progress=-1.10,
                maximum_entry_progress=0.20,
                maximum_entry_lateral=0.80,
                maximum_entry_heading=math.radians(105.0),
                baseline_lateral_tolerance=0.015,
                ambiguity_position=0.004,
                ambiguity_heading=math.radians(0.25),
            )
            eight_degree_acceptances += int(
                register_obstacle_outer_faces(
                    baseline_lateral_maximum=0.040,
                    axis_consistency_tolerance=math.radians(8.0),
                    **common
                )
                is not None
            )
            result = register_obstacle_outer_faces(
                baseline_lateral_maximum=0.040,
                axis_consistency_tolerance=math.radians(9.0),
                **common
            )
            self.assertIsNotNone(result)
            states.append(temporal_filter.update(result))

        self.assertEqual(eight_degree_acceptances, 2)
        self.assertEqual(
            [state.confirmation_count for state in states], [1, 2, 3]
        )
        self.assertTrue(states[-1].confirmed)
        self.assertLess(states[-1].maximum_position_spread, 0.0033)
        self.assertLess(
            states[-1].maximum_heading_spread, math.radians(0.35)
        )
        self.assertAlmostEqual(
            states[-1].transform.target_from_source_x, 0.832771, places=5
        )
        self.assertAlmostEqual(
            states[-1].transform.target_from_source_y, 1.769684, places=5
        )
        self.assertAlmostEqual(
            math.degrees(states[-1].transform.target_from_source_yaw),
            89.4689,
            places=3,
        )

    def test_asymmetric_front_clipping_is_not_allowed_to_rotate_side_alignment(self):
        template = obstacle_barrier_template(self.barrier_geometry())
        expected = RigidTransform2D(
            0.84,
            1.77,
            math.radians(90.0),
            source_frame="obstacle_course",
            target_frame="odom",
        )
        first = template.segments[0]
        last = template.segments[2]
        upper_side = next(
            segment
            for segment in template.segments
            if segment.name == "barrier_0_upper_side"
        )
        # Both 200 mm front fragments are individually plausible, but their
        # unequal clipping moves the midpoint baseline by 20 mm course-left.
        # The exact perpendicular side proves that this is clipping, not yaw.
        faces = (
            DetectedBarrierFace(
                expected.apply_point((first.start[0], 0.0375)),
                expected.apply_point((first.end[0], 0.2375)),
            ),
            DetectedBarrierFace(
                expected.apply_point((last.start[0], 0.0575)),
                expected.apply_point((last.end[0], 0.2575)),
            ),
            DetectedBarrierFace(
                expected.apply_point(upper_side.start),
                expected.apply_point(upper_side.end),
            ),
        )
        robot = expected.apply_pose(Pose2D(0.24, 0.60, math.radians(-90.0)))

        result = register_obstacle_outer_faces(
            template,
            faces,
            161.2,
            "odom",
            robot,
            RegistrationConfig(minimum_inliers=2),
            -1.10,
            0.20,
            0.80,
            math.radians(105.0),
            baseline_lateral_tolerance=0.015,
            baseline_lateral_maximum=0.040,
            axis_consistency_tolerance=math.radians(9.0),
        )

        self.assertIsNone(result)

    def test_outer_face_signature_rejects_short_wrong_and_nonparallel_pairs(self):
        template = obstacle_barrier_template(self.barrier_geometry())
        expected = RigidTransform2D(
            0.84,
            1.77,
            math.radians(90.0),
            source_frame="obstacle_course",
            target_frame="odom",
        )
        robot = expected.apply_pose(
            Pose2D(0.24, 0.60, math.radians(-90.0))
        )
        first = template.segments[0]
        last = template.segments[2]
        valid_first = DetectedBarrierFace(
            expected.apply_point(first.start),
            expected.apply_point(first.end),
        )

        def register(second):
            return register_obstacle_outer_faces(
                template,
                (valid_first, second),
                160.71,
                "odom",
                robot,
                RegistrationConfig(minimum_inliers=2),
                -1.10,
                0.20,
                0.80,
                math.radians(105.0),
                minimum_face_length=0.18,
                baseline_tolerance=0.035,
                parallel_heading_tolerance=math.radians(6.0),
            )

        short_last_midpoint = last.midpoint
        short_last = DetectedBarrierFace(
            expected.apply_point(
                (short_last_midpoint[0], short_last_midpoint[1] - 0.08)
            ),
            expected.apply_point(
                (short_last_midpoint[0], short_last_midpoint[1] + 0.08)
            ),
        )
        self.assertIsNone(register(short_last))

        wrong_baseline = DetectedBarrierFace(
            expected.apply_point((last.start[0] - 0.08, last.start[1])),
            expected.apply_point((last.end[0] - 0.08, last.end[1])),
        )
        self.assertIsNone(register(wrong_baseline))

        midpoint = last.midpoint
        angle = math.radians(15.0)
        tangent = (-math.sin(angle), math.cos(angle))
        nonparallel = DetectedBarrierFace(
            expected.apply_point(
                (
                    midpoint[0] - 0.125 * tangent[0],
                    midpoint[1] - 0.125 * tangent[1],
                )
            ),
            expected.apply_point(
                (
                    midpoint[0] + 0.125 * tangent[0],
                    midpoint[1] + 0.125 * tangent[1],
                )
            ),
        )
        self.assertIsNone(register(nonparallel))

    def test_outer_face_signature_rejects_two_distinct_valid_hypotheses(self):
        template = obstacle_barrier_template(self.barrier_geometry())
        expected = RigidTransform2D(
            0.84,
            1.77,
            math.radians(90.0),
            source_frame="obstacle_course",
            target_frame="odom",
        )
        alternative = RigidTransform2D(
            0.96,
            1.77,
            math.radians(90.0),
            source_frame="obstacle_course",
            target_frame="odom",
        )
        faces = tuple(
            DetectedBarrierFace(
                transform.apply_point(segment.start),
                transform.apply_point(segment.end),
            )
            for transform in (expected, alternative)
            for segment in (template.segments[0], template.segments[2])
        )
        robot = expected.apply_pose(Pose2D(0.24, 0.30, math.radians(-90.0)))

        result = register_obstacle_outer_faces(
            template,
            faces,
            160.71,
            "odom",
            robot,
            RegistrationConfig(minimum_inliers=2),
            -1.10,
            0.20,
            0.80,
            math.radians(105.0),
            minimum_face_length=0.18,
            baseline_tolerance=0.035,
            parallel_heading_tolerance=math.radians(6.0),
        )

        self.assertIsNone(result)

    def test_close_competing_hypotheses_are_rejected_independent_of_order(self):
        template = obstacle_barrier_template(self.barrier_geometry())
        expected = RigidTransform2D(
            0.84,
            1.77,
            math.radians(90.0),
            source_frame="obstacle_course",
            target_frame="odom",
        )
        alternative = RigidTransform2D(
            0.845,
            1.77,
            math.radians(90.0),
            source_frame="obstacle_course",
            target_frame="odom",
        )
        groups = tuple(
            tuple(
                DetectedBarrierFace(
                    transform.apply_point(segment.start),
                    transform.apply_point(segment.end),
                )
                for segment in (template.segments[0], template.segments[2])
            )
            for transform in (expected, alternative)
        )
        robot = expected.apply_pose(Pose2D(0.24, 0.30, math.radians(-90.0)))

        for faces in (groups[0] + groups[1], groups[1] + groups[0]):
            result = register_obstacle_outer_faces(
                template,
                faces,
                161.3,
                "odom",
                robot,
                RegistrationConfig(minimum_inliers=2),
                -1.10,
                0.20,
                0.80,
                math.radians(105.0),
            )
            self.assertIsNone(result)

    def test_side_approach_acquisition_does_not_relax_ready_gate(self):
        controller = ObstacleMissionController.__new__(
            ObstacleMissionController
        )
        controller.registration_template = obstacle_barrier_template(
            self.barrier_geometry()
        )
        controller.odom_from_course = RigidTransform2D(
            0.84,
            1.77,
            math.radians(90.0),
            source_frame="obstacle_course",
            target_frame="odom",
        )
        controller.odom_ready = True
        controller.ready_lead_minimum_progress = -0.22
        controller.ready_lead_maximum_progress = 0.12
        controller.registration_entry_maximum_lateral = 0.20
        controller.registration_entry_maximum_heading = math.radians(15.0)
        side = controller.odom_from_course.apply_pose(
            Pose2D(0.24, 0.60, math.radians(-90.0))
        )
        controller.odom_x = side.x
        controller.odom_y = side.y
        controller.odom_yaw = side.yaw

        self.assertFalse(controller._entry_is_ready())

    def test_ready_lead_prevents_early_speed_cap(self):
        controller = ObstacleMissionController.__new__(
            ObstacleMissionController
        )
        controller.registration_template = obstacle_barrier_template(
            self.barrier_geometry()
        )
        controller.odom_from_course = RigidTransform2D(
            2.0,
            -0.4,
            math.radians(18.0),
            source_frame="obstacle_course",
            target_frame="odom",
        )
        controller.odom_ready = True
        controller.ready_lead_minimum_progress = -0.22
        controller.ready_lead_maximum_progress = 0.05
        controller.registration_entry_maximum_lateral = 0.20
        controller.registration_entry_maximum_heading = math.radians(15.0)
        far = controller.odom_from_course.apply_pose(Pose2D(-0.60, 0.0, 0.0))
        controller.odom_x = far.x
        controller.odom_y = far.y
        controller.odom_yaw = far.yaw
        self.assertFalse(controller._entry_is_ready())
        near = controller.odom_from_course.apply_pose(Pose2D(0.40, 0.0, 0.0))
        controller.odom_x = near.x
        controller.odom_y = near.y
        controller.odom_yaw = near.yaw
        self.assertTrue(controller._entry_is_ready())

    def test_official_handoff_pose_commits_full_path_and_publishes_ready(self):
        with CONFIG_PATH.open("r", encoding="utf-8") as stream:
            obstacle = yaml.safe_load(stream)["obstacle"]

        missing = object()

        def get_parameter(name, default=missing):
            value = obstacle
            prefix = "~obstacle/"
            relative_name = name[len(prefix) :] if name.startswith(prefix) else name
            for key in relative_name.split("/"):
                if not isinstance(value, dict) or key not in value:
                    if default is missing:
                        raise KeyError(name)
                    return default
                value = value[key]
            return value

        def publisher(*_args, **_kwargs):
            return RecordingPublisher()

        with mock.patch.object(
            controller_module.rospy, "get_param", side_effect=get_parameter
        ), mock.patch.object(
            controller_module.rospy, "Publisher", side_effect=publisher
        ), mock.patch.object(
            controller_module.rospy, "Subscriber", return_value=mock.Mock()
        ), mock.patch.object(
            controller_module.rospy, "ServiceProxy", return_value=mock.Mock()
        ), mock.patch.object(
            controller_module.rospy, "Timer", return_value=mock.Mock()
        ), mock.patch.object(
            controller_module.rospy, "on_shutdown"
        ):
            controller = ObstacleMissionController()

        # Captured official-start trajectory at 165.42 s, expressed in the
        # registered obstacle frame.  It is inside the strict entry envelope
        # and within the direct suffix-join tolerances.
        controller.odom_from_course = RigidTransform2D(
            0.0,
            0.0,
            0.0,
            source_frame="obstacle_course",
            target_frame="odom",
        )
        controller.odom_ready = True
        controller.odom_x = 0.448836
        controller.odom_y = -0.091623
        controller.odom_yaw = math.radians(-12.692)
        controller.odom_frame = "odom"
        controller.arm_generation = 7
        controller.armed_at = controller_module.rospy.Time.from_sec(9.0)
        controller.odom_stamp = self.now()
        controller.scan_stamp = controller_module.rospy.Time.from_sec(9.9)
        controller.scan_generation = 1
        controller.last_plan_scan_generation = -1
        controller.registration_covariance = (
            (0.01, 0.0, 0.0),
            (0.0, 0.01, 0.0),
            (0.0, 0.0, math.radians(10.0) ** 2),
        )
        controller._update_local_registration = mock.Mock(return_value=True)
        real_plan = controller.spline_planner.plan

        def plan_after_entry_cap(*args, **kwargs):
            self.assertEqual(len(controller.speed_limit_pub.messages), 1)
            self.assertEqual(
                controller.speed_limit_pub.messages[0].data,
                controller.entry_velocity_cap,
            )
            return real_plan(*args, **kwargs)

        controller.spline_planner.plan = mock.Mock(
            side_effect=plan_after_entry_cap
        )

        self.assertTrue(controller._attempt_path(self.now()))
        self.assertIsNotNone(controller.committed_path)
        self.assertTrue(controller.fixed_path_validation.safe)
        self.assertAlmostEqual(
            float(controller.committed_path.x[0]), controller.odom_x, places=9
        )
        self.assertAlmostEqual(
            float(controller.committed_path.y[0]), controller.odom_y, places=9
        )
        self.assertAlmostEqual(
            normalize_angle(
                float(controller.committed_path.heading[0])
                - controller.odom_yaw
            ),
            0.0,
            places=9,
        )
        self.assertEqual(controller.prepared_generation, 7)
        self.assertEqual(
            controller.planner_status_pub.messages[-1].data,
            "PATH_COMMITTED",
        )
        self.assertEqual(len(controller.ready_pub.messages), 1)
        self.assertEqual(controller.ready_pub.messages[0].seq, 7)
        self.assertEqual(controller.ready_pub.messages[0].frame_id, "obstacle")

        # A capped lane command may advance the robot after pre-gate planning.
        # Rebuild once from the same latest scan but the new odom pose, and
        # make that exact live pose the first executable sample.
        prepared_path = controller.committed_path
        controller.odom_x += 0.004
        self.assertTrue(controller._attempt_path(self.now(), refresh=True))
        self.assertEqual(len(controller.speed_limit_pub.messages), 1)
        self.assertIsNot(controller.committed_path, prepared_path)
        self.assertAlmostEqual(
            float(controller.committed_path.x[0]), controller.odom_x, places=9
        )

        # A rejected later takeover refresh is atomic: it cannot erase or
        # partially replace the last fully validated path while lane owns cmd.
        refreshed_path = controller.committed_path
        refreshed_follower = controller.path_follower
        controller.scan_generation += 1
        with mock.patch.object(
            controller.spline_planner, "connect_entry", return_value=None
        ):
            self.assertFalse(controller._attempt_path(self.now(), refresh=True))
        self.assertIs(controller.committed_path, refreshed_path)
        self.assertIs(controller.path_follower, refreshed_follower)

    def test_start_requires_same_generation_prepared_path(self):
        controller = self.make_path_harness()
        controller.odom_ready = True
        controller.odom_stamp = self.now()
        controller.odom_timeout = 0.35
        controller.arm_generation = 7
        controller.prepared_generation = 0
        controller.ready_published_generation = 0
        controller.odom_from_course = None
        controller.start_requested = True
        controller.revoke_requested = False
        controller.entry_velocity_cap = 0.09
        controller.speed_limit_pub = RecordingPublisher()
        controller._set_state = mock.Mock()

        controller._start_run()

        self.assertTrue(controller.start_requested)
        self.assertEqual(controller.speed_limit_pub.messages, [])
        controller._set_state.assert_not_called()

        controller.prepared_generation = 7
        controller.ready_published_generation = 7
        controller.last_ready_stamp = self.now()
        controller.scan_timeout = 0.35
        controller.maximum_scan_odom_skew = 0.075
        controller.odom_from_course = RigidTransform2D(
            0.0,
            0.0,
            0.0,
            source_frame="obstacle_course",
            target_frame="odom",
        )
        controller._start_run()
        self.assertFalse(controller.start_requested)
        # The cap is published on the rising gate edge, before this state
        # transition. _start_run must not publish it and acquire in one tick.
        self.assertEqual(controller.speed_limit_pub.messages, [])
        controller._set_state.assert_called_once_with(controller.ACQUIRING)

    def test_ready_echoes_arm_only_after_path_is_prepared(self):
        controller = self.make_runtime_harness()
        controller.arm_generation = 12
        controller.armed_at = controller_module.rospy.Time.from_sec(9.0)
        controller.ready_published_generation = 0
        controller.last_ready_stamp = None
        controller.prepared_generation = 0
        controller.prepared_stamp = controller_module.rospy.Time.from_sec(9.7)
        controller.scan_stamp = controller_module.rospy.Time.from_sec(9.80)
        controller.odom_stamp = controller_module.rospy.Time.from_sec(9.85)
        controller.ready_pub = RecordingPublisher()
        controller._entry_is_ready = mock.Mock(return_value=True)

        self.assertFalse(controller._publish_ready(self.now()))
        self.assertEqual(controller.ready_pub.messages, [])

        controller.prepared_generation = 12
        self.assertTrue(controller._publish_ready(self.now()))
        self.assertEqual(len(controller.ready_pub.messages), 1)
        ready = controller.ready_pub.messages[0]
        self.assertIsInstance(ready, Header)
        self.assertEqual(ready.seq, 12)
        self.assertEqual(ready.stamp, controller.scan_stamp)
        self.assertEqual(ready.frame_id, "obstacle")

        controller.scan_stamp = controller_module.rospy.Time.from_sec(9.95)
        controller.odom_stamp = self.now()
        self.assertTrue(controller._publish_ready(self.now()))
        self.assertEqual(len(controller.ready_pub.messages), 2)
        self.assertEqual(
            controller.ready_pub.messages[-1].stamp,
            controller.scan_stamp,
        )

    def test_ready_waits_for_fresh_sources_after_long_path_validation(self):
        controller = self.make_runtime_harness()
        controller.state = controller.WAIT_GATE
        controller.zone_gate = False
        controller.arm_generation = 4
        controller.armed_at = controller_module.rospy.Time.from_sec(8.0)
        controller.prepared_generation = 4
        controller.prepared_stamp = controller_module.rospy.Time.from_sec(9.0)
        controller.ready_published_generation = 0
        controller.last_ready_stamp = None
        controller.scan_stamp = controller_module.rospy.Time.from_sec(9.0)
        controller.odom_stamp = controller_module.rospy.Time.from_sec(9.0)
        controller.ready_pub = RecordingPublisher()
        controller._entry_is_ready = mock.Mock(return_value=True)

        self.assertFalse(controller._publish_ready(self.now()))
        self.assertEqual(controller.ready_pub.messages, [])

        controller.scan_stamp = controller_module.rospy.Time.from_sec(9.90)
        controller.odom_stamp = controller_module.rospy.Time.from_sec(9.95)
        controller.control_callback(None)

        self.assertEqual(len(controller.ready_pub.messages), 1)
        self.assertEqual(
            controller.ready_pub.messages[0].stamp,
            controller.scan_stamp,
        )
        self.assertIs(controller.committed_path, controller.path_follower.path)

    def test_arm_requires_exact_obstacle_frame(self):
        controller = self.make_runtime_harness()
        controller.state = controller.WAIT_GATE
        controller.zone_gate = False
        original_generation = controller.arm_generation

        for frame in ("", "odom", "Obstacle"):
            message = Header()
            message.seq = original_generation + 1
            message.stamp = self.now()
            message.frame_id = frame
            controller.arm_callback(message)
            self.assertEqual(controller.arm_generation, original_generation)

        zero_stamp = Header()
        zero_stamp.seq = original_generation + 1
        zero_stamp.frame_id = "obstacle"
        controller.arm_callback(zero_stamp)
        self.assertEqual(controller.arm_generation, original_generation)

        message = Header()
        message.seq = original_generation + 1
        message.stamp = self.now()
        message.frame_id = "obstacle"
        controller.arm_callback(message)
        self.assertEqual(controller.arm_generation, original_generation + 1)
        self.assertEqual(controller.armed_at, self.now())
        self.assertEqual(controller.prepared_generation, 0)
        self.assertIsNone(controller.committed_path)

    def test_pre_gate_registration_keeps_lane_control_and_cruise_limit(self):
        controller = self.make_runtime_harness()
        controller.state = controller.WAIT_GATE
        controller.zone_gate = False
        controller.mission_has_control = False
        controller.arm_generation = 5
        controller.armed_at = controller_module.rospy.Time.from_sec(9.0)
        controller.prepared_generation = 0
        controller.ready_published_generation = 0
        controller.speed_limit_pub = RecordingPublisher()
        controller._acquisition_data_problem = mock.Mock(return_value=None)
        controller._attempt_path = mock.Mock(return_value=False)
        controller._set_lane_controller = mock.Mock()

        controller.control_callback(None)

        controller._attempt_path.assert_called_once_with(self.now())
        controller._set_lane_controller.assert_not_called()
        self.assertEqual(controller.speed_limit_pub.messages, [])
        self.assertFalse(controller.mission_has_control)

    def test_committed_path_keeps_co_registered_clearance_margin(self):
        controller = ObstacleMissionController.__new__(
            ObstacleMissionController
        )
        controller.local_safety_boundary = (
            controller_module.AxisAlignedBoundsBoundary(
                -2.0, 2.0, -2.0, 2.0
            )
        )
        controller.odom_from_course = RigidTransform2D(
            0.20,
            -0.10,
            math.radians(5.0),
            source_frame="obstacle_course",
            target_frame="odom",
        )
        controller.validation_footprint = Footprint(
            0.067645, 0.118073, 0.0903
        )
        controller.registration_covariance = (
            (0.0001, 0.0, 0.0),
            (0.0, 0.000025, 0.0),
            (0.0, 0.0, 0.0),
        )
        path = self.straight_path()

        controller._attach_local_safety_boundary(path)
        validation = RectanglePathChecker(
            Footprint(0.067645, 0.118073, 0.0903)
        ).validator.validate_path(path)

        self.assertEqual(len(path.safety.map_boundaries), 1)
        self.assertAlmostEqual(path.safety.margins.localization, 0.0)
        self.assertTrue(validation.safe)
        self.assertTrue(math.isfinite(validation.minimum_map_clearance))
        self.assertGreater(validation.minimum_map_clearance, 0.0)

    def test_registration_covariance_is_not_double_counted_on_local_geometry(self):
        controller = ObstacleMissionController.__new__(
            ObstacleMissionController
        )
        controller.local_safety_boundary = (
            controller_module.AxisAlignedBoundsBoundary(
                -2.0, 2.0, -2.0, 2.0
            )
        )
        controller.odom_from_course = RigidTransform2D(
            0.20,
            -0.10,
            math.radians(12.0),
            source_frame="obstacle_course",
            target_frame="odom",
        )
        controller.validation_footprint = Footprint(
            0.067645, 0.118073, 0.0903
        )
        controller.registration_covariance = (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, math.radians(1.0) ** 2),
        )
        path = self.straight_path()
        path.safety = PathSafety(
            margins=SafetyMargins(localization=0.002)
        )

        controller._attach_local_safety_boundary(path)

        self.assertAlmostEqual(path.safety.margins.localization, 0.002)

    def test_runtime_live_sweep_does_not_reapply_global_registration_error(self):
        controller = self.make_runtime_harness()
        controller.registration_covariance = (
            (0.01, 0.0, 0.0),
            (0.0, 0.01, 0.0),
            (0.0, 0.0, math.radians(10.0) ** 2),
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
        self.assertAlmostEqual(
            safety.margins.localization,
            controller.validation_footprint.localization_margin,
        )

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

    def test_gate_acquisition_waits_for_fresh_capped_command_and_odometry(self):
        controller = self.make_path_harness()
        controller.lock = threading.RLock()
        controller.shutting_down = False
        controller.state = controller.WAIT_GATE
        controller.zone_gate = False
        controller.start_requested = False
        controller.revoke_requested = False
        controller.manual_stop = False
        controller.mission_has_control = False
        controller.handoff_ambiguous = False
        controller.arm_generation = 3
        controller.prepared_generation = 3
        controller.ready_published_generation = 3
        controller.last_ready_stamp = self.now()
        controller.maximum_scan_odom_skew = 0.075
        controller.scan_timeout = 0.35
        controller.odom_timeout = 0.35
        controller.odom_ready = True
        controller.odom_stamp = self.now()
        controller.odom_from_course = RigidTransform2D(
            0.0,
            0.0,
            0.0,
            source_frame="obstacle_course",
            target_frame="odom",
        )
        controller.state_pub = RecordingPublisher()
        controller.cmd_pub = RecordingPublisher()
        controller.speed_limit_pub = RecordingPublisher()
        controller.acquisition_heading_tolerance = 1.0
        controller.maximum_angular_velocity = 0.8
        controller.entry_velocity_cap = 0.10
        controller.entry_handoff_velocity_tolerance = 0.005
        controller.entry_projection_position_tolerance = 0.015
        controller.entry_projection_heading_tolerance = math.radians(8.0)
        controller.observed_lane_linear = 0.22
        controller.observed_lane_angular = 0.149
        # Reproduce the run4 seam: the future CommonPath curvature is not the
        # still-active lane controller's feedback curvature.  Geometric join
        # gates, not equality between those unrelated commands, decide the
        # ownership transfer.
        controller.committed_path.curvature[:] = 14.216568
        controller._course_heading_error = mock.Mock(return_value=0.0)
        controller._acquisition_data_problem = mock.Mock(return_value=None)
        controller._attempt_path = mock.Mock(side_effect=[False, True, True])
        candidate_follower = controller.path_follower
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

        self.assertEqual(len(controller.speed_limit_pub.messages), 1)
        self.assertEqual(controller.speed_limit_pub.messages[0].data, 0.10)
        self.assertFalse(controller.mission_has_control)
        controller.control_callback(None)

        self.assertEqual(handoffs, [])
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(controller.state, controller.ACQUIRING)
        self.assertEqual(controller.cmd_pub.messages, [])

        # New samples alone are insufficient while both commanded and
        # measured forward speed still exceed the cap. A callback generation
        # carrying a queued pre-gate odom source stamp is not fresh either.
        self.seconds = 10.01
        controller.lane_command_generation += 1
        controller.observed_lane_command_received = self.now()
        controller.odom_generation += 1
        controller.odom_linear_velocity = 0.22
        controller.control_callback(None)
        self.assertEqual(handoffs, [])
        self.assertEqual(controller._attempt_path.call_count, 0)

        self.seconds = 10.02
        controller.odom_generation += 1
        controller.odom_stamp = self.now()
        controller.control_callback(None)
        self.assertEqual(handoffs, [])

        # A capped command still leaves lane control active until fresh odom
        # confirms the vehicle itself has decelerated too.
        self.seconds = 10.03
        controller.lane_command_generation += 1
        controller.observed_lane_linear = 0.10
        controller.observed_lane_command_received = self.now()
        controller.control_callback(None)
        self.assertEqual(handoffs, [])

        self.seconds = 10.04
        controller.odom_generation += 1
        controller.odom_stamp = self.now()
        controller.odom_linear_velocity = 0.104
        controller.control_callback(None)

        # A fresh-pose connector that fails its complete swept validation
        # cannot take ownership or publish Twist. The next scan may retry.
        self.assertEqual(handoffs, [])
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(controller.state, controller.ACQUIRING)
        self.assertEqual(controller.cmd_pub.messages, [])
        controller.scan_generation += 1
        controller.control_callback(None)

        # Even a successful refresh cannot hand off on its sweep tick. The
        # lane keeps ownership until callbacks deliver newer capped command
        # and odometry samples.
        self.assertEqual(handoffs, [])
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(controller.cmd_pub.messages, [])
        refresh_odom_generation = controller.odom_generation
        refresh_command_generation = controller.lane_command_generation
        self.assertEqual(
            controller.handoff_refresh_odom_generation,
            refresh_odom_generation,
        )
        self.assertEqual(
            controller.handoff_refresh_lane_command_generation,
            refresh_command_generation,
        )
        self.assertEqual(controller.handoff_refresh_odom_stamp, self.now())
        self.assertEqual(
            controller.handoff_refresh_lane_command_received,
            controller.observed_lane_command_received,
        )

        # If the new post-refresh pose has moved outside join tolerance, keep
        # the validated candidate for diagnostics but clear the activation
        # marker and wait for another scan/refresh under lane ownership.
        prepared_path = controller.committed_path
        self.seconds = 10.05
        controller.lane_command_generation += 1
        controller.observed_lane_command_received = self.now()
        controller.odom_generation += 1
        controller.odom_stamp = self.now()
        controller.odom_y = 0.20
        controller.control_callback(None)
        self.assertEqual(handoffs, [])
        self.assertFalse(controller.mission_has_control)
        self.assertIs(controller.committed_path, prepared_path)
        self.assertEqual(controller.handoff_refresh_odom_generation, -1)

        controller.odom_y = 0.0
        controller.scan_generation += 1
        controller.control_callback(None)
        self.assertEqual(handoffs, [])
        self.assertFalse(controller.mission_has_control)

        self.seconds = 10.06
        controller.lane_command_generation += 1
        controller.observed_lane_command_received = self.now()
        controller.odom_generation += 1
        controller.odom_stamp = self.now()
        controller.control_callback(None)

        self.assertEqual(handoffs, [False])
        self.assertTrue(controller.mission_has_control)
        self.assertEqual(controller.state, controller.ACQUIRING)
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        self.assertAlmostEqual(controller.cmd_pub.messages[0].linear.x, 0.0)
        self.assertAlmostEqual(controller.cmd_pub.messages[0].angular.z, 0.0)

        # Ownership alone is not activation. A source-stamped odom sample
        # newer than the service response must reproject successfully first.
        self.seconds = 10.07
        controller.odom_generation += 1
        controller.odom_stamp = self.now()
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.AVOIDING)
        self.assertEqual(len(controller.cmd_pub.messages), 3)
        self.assertEqual(controller._attempt_path.call_count, 3)
        self.assertTrue(
            all(
                call.kwargs == {"refresh": True}
                for call in controller._attempt_path.call_args_list
            )
        )

    def test_handoff_service_does_not_block_odom_and_zero_gates_first_motion(self):
        controller = self.make_runtime_harness()
        controller.state = controller.ACQUIRING
        controller.zone_gate = True
        controller.mission_has_control = False
        controller.maximum_angular_velocity = 0.8
        controller.acquisition_heading_tolerance = 1.0
        controller.entry_velocity_cap = 0.10
        controller.observed_lane_linear = 0.09
        controller.observed_lane_angular = 0.0
        controller.observed_lane_command_received = (
            controller_module.rospy.Time.from_sec(10.0)
        )
        controller.lane_command_generation = 2
        controller.odom_generation = 2
        controller.odom_stamp = controller_module.rospy.Time.from_sec(10.0)
        controller.odom_frame = "odom"
        controller.handoff_refresh_scan_generation = controller.scan_generation
        controller.handoff_refresh_lane_command_generation = 1
        controller.handoff_refresh_odom_generation = 1
        controller.handoff_refresh_lane_command_received = (
            controller_module.rospy.Time.from_sec(9.99)
        )
        controller.handoff_refresh_odom_stamp = (
            controller_module.rospy.Time.from_sec(9.99)
        )
        controller.handoff_refresh_odom_frame = "odom"
        controller._course_heading_error = mock.Mock(return_value=0.0)

        tracking = controller.path_follower.calculate_tracking(
            controller._current_pose()
        )
        controller._validate_committed_path = mock.Mock(
            return_value=(tracking, controller._current_pose())
        )
        moving_command = Twist()
        moving_command.linear.x = 0.04
        controller._common_path_command = mock.Mock(
            return_value=moving_command
        )

        service_entered = threading.Event()
        release_service = threading.Event()
        service_times = {}

        def blocking_handoff(enabled):
            self.assertFalse(enabled)
            service_times["entered"] = time.monotonic()
            service_entered.set()
            if not release_service.wait(1.0):
                return False
            service_times["returned"] = time.monotonic()
            controller.mission_has_control = True
            controller.handoff_ambiguous = False
            return True

        controller._set_lane_controller = blocking_handoff
        control_errors = []

        def run_control():
            try:
                controller.control_callback(None)
            except BaseException as error:
                control_errors.append(error)

        control_thread = threading.Thread(target=run_control)
        control_thread.start()
        if not service_entered.wait(0.5):
            release_service.set()
            control_thread.join(0.5)
            self.fail(
                "lane handoff service was not reached: %r" % control_errors
            )

        odom_finished = threading.Event()
        odom_errors = []

        def deliver_odom(message):
            try:
                controller.odom_callback(message)
            except BaseException as error:
                odom_errors.append(error)
            finally:
                odom_finished.set()

        during_service = Odometry()
        during_service.header.stamp = controller_module.rospy.Time.from_sec(
            10.01
        )
        during_service.header.frame_id = "odom"
        during_service.pose.pose.orientation.w = 1.0
        during_service.twist.twist.linear.x = 0.09
        self.seconds = 10.01
        odom_thread = threading.Thread(
            target=deliver_odom, args=(during_service,)
        )
        odom_thread.start()
        try:
            self.assertTrue(
                odom_finished.wait(0.10),
                "odom callback was blocked by the lane handoff service",
            )
            self.assertTrue(control_thread.is_alive())
            remaining_hold = 0.22 - (
                time.monotonic() - service_times["entered"]
            )
            if remaining_hold > 0.0:
                threading.Event().wait(remaining_hold)
        finally:
            release_service.set()
            odom_thread.join(0.5)
            control_thread.join(0.5)

        self.assertFalse(odom_thread.is_alive())
        self.assertFalse(control_thread.is_alive())
        self.assertEqual(odom_errors, [])
        self.assertEqual(control_errors, [])
        self.assertGreaterEqual(
            service_times["returned"] - service_times["entered"], 0.20
        )
        self.assertTrue(controller.mission_has_control)
        self.assertEqual(controller.state, controller.ACQUIRING)
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        self.assertAlmostEqual(controller.cmd_pub.messages[0].linear.x, 0.0)
        self.assertAlmostEqual(controller.cmd_pub.messages[0].angular.z, 0.0)

        # Neither a timer tick nor a newer callback generation carrying the
        # same source stamp may release a nonzero mission command.
        controller.control_callback(None)
        same_source = Odometry()
        same_source.header.stamp = controller_module.rospy.Time.from_sec(
            10.01
        )
        same_source.header.frame_id = "odom"
        same_source.pose.pose.orientation.w = 1.0
        same_source.twist.twist.linear.x = 0.09
        controller.odom_callback(same_source)
        controller.control_callback(None)
        self.assertTrue(
            all(
                abs(message.linear.x) <= 1e-12
                and abs(message.angular.z) <= 1e-12
                for message in controller.cmd_pub.messages
            )
        )

        # The first nonzero command is permitted only after odometry whose
        # source stamp is strictly newer than the completed service barrier.
        newer_source = Odometry()
        newer_source.header.stamp = controller_module.rospy.Time.from_sec(
            10.02
        )
        newer_source.header.frame_id = "odom"
        newer_source.pose.pose.orientation.w = 1.0
        newer_source.twist.twist.linear.x = 0.09
        self.seconds = 10.02
        controller.odom_callback(newer_source)
        zero_count_before_activation = len(controller.cmd_pub.messages)
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.AVOIDING)
        nonzero_indexes = [
            index
            for index, message in enumerate(controller.cmd_pub.messages)
            if abs(message.linear.x) > 1e-12
            or abs(message.angular.z) > 1e-12
        ]
        self.assertEqual(nonzero_indexes, [len(controller.cmd_pub.messages) - 1])
        self.assertGreaterEqual(nonzero_indexes[0], zero_count_before_activation)

    def test_post_service_zero_reset_owns_heading_and_future_curvature(self):
        controller = self.make_runtime_harness()
        controller.committed_path.curvature[:] = 14.216568
        controller.path_follower.reset(
            controller.committed_path, controller._current_pose()
        )
        controller.state = controller.ACQUIRING
        controller.mission_has_control = True
        controller.handoff_takeover_odom_generation = 4
        controller.handoff_takeover_odom_stamp = (
            controller_module.rospy.Time.from_sec(9.99)
        )
        controller.handoff_takeover_odom_frame = "odom"
        controller.odom_generation = 5
        controller.odom_stamp = self.now()
        # The service returned after the robot crossed the sharp connector
        # seam.  It is still only 0 mm from the route, but its body heading is
        # now beyond the 8 degree pre-service join limit seen in run4.
        controller.odom_yaw = math.radians(9.183)
        controller.observed_lane_linear = 0.09
        controller.observed_lane_angular = 0.148696
        controller._fail = mock.Mock()

        self.assertTrue(
            controller._activate_after_handoff_service(self.now())
        )
        controller._fail.assert_not_called()
        self.assertEqual(controller.state, controller.AVOIDING)
        self.assertAlmostEqual(controller.path_follower.last_linear, 0.0)
        self.assertAlmostEqual(controller.path_follower.last_angular, 0.0)

        validated = controller._validate_committed_path(self.now())
        self.assertTrue(validated)
        tracking, pose = validated
        self.seconds += controller.control_period
        command = controller._common_path_command(
            self.now(), tracking=tracking, pose=pose
        )
        self.assertLessEqual(
            abs(command.linear.x),
            controller.tracking_config.linear_acceleration
            * controller.control_period
            + 1e-12,
        )
        self.assertLessEqual(
            abs(command.angular.z),
            controller.tracking_config.angular_acceleration
            * controller.control_period
            + 1e-12,
        )
        self.assertLessEqual(
            abs(command.linear.x * command.angular.z),
            controller.tracking_config.maximum_lateral_acceleration + 1e-12,
        )

    def test_post_service_handoff_rejects_pose_beyond_join_tolerance(self):
        controller = self.make_path_harness()
        controller.state = controller.ACQUIRING
        controller.state_pub = RecordingPublisher()
        controller.cmd_pub = RecordingPublisher()
        controller.mission_has_control = True
        controller.odom_timeout = 0.35
        controller.handoff_takeover_odom_generation = 4
        controller.handoff_takeover_odom_stamp = (
            controller_module.rospy.Time.from_sec(9.99)
        )
        controller.handoff_takeover_odom_frame = "odom"
        controller.odom_generation = 5
        controller.odom_stamp = self.now()
        controller.odom_y = 0.020
        controller._fail = mock.Mock()

        self.assertFalse(
            controller._activate_after_handoff_service(self.now())
        )

        controller._fail.assert_called_once()
        self.assertIn("outside join tolerance", controller._fail.call_args.args[0])
        self.assertEqual(controller.state, controller.ACQUIRING)

    def test_premature_gate_does_not_stop_preparation(self):
        controller = self.make_runtime_harness()
        controller.state = controller.WAIT_GATE
        controller.zone_gate = False
        controller.start_requested = False
        controller.mission_has_control = False
        controller.prepared_generation = 0
        controller.ready_published_generation = 0
        controller.committed_path = None
        controller.odom_from_course = None

        with mock.patch.object(controller_module.rospy, "logwarn_throttle"):
            controller.gate_callback(Bool(data=True))

        self.assertFalse(controller.zone_gate)
        self.assertFalse(controller.start_requested)

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
        self.assertTrue(controller.live_requires_stop)
        self.assertLessEqual(
            controller.path_follower.diagnostics.minimum_obstacle_clearance,
            0.0,
        )

    def test_required_stop_publishes_zero_twist_without_follower_rotation(self):
        controller = self.make_runtime_harness()
        # A live return inside the current expanded rectangle makes the
        # complete-stop sweep unsafe.  Even if the follower would request a
        # rotation, the controller must publish one exact zero Twist instead.
        controller._lane_points = mock.Mock(
            return_value=np.asarray([[0.02, 0.0]], dtype=np.float64)
        )
        controller._lane_points_in_odom = mock.Mock(
            return_value=np.asarray([[0.02, 0.0]], dtype=np.float64)
        )
        moving = Twist()
        moving.linear.x = 0.04
        moving.angular.z = 0.30
        controller._common_path_command = mock.Mock(return_value=moving)
        controller._publish_diagnostics.reset_mock()

        controller.control_callback(None)

        controller._common_path_command.assert_not_called()
        self.assertEqual(controller.state, controller.AVOIDING)
        self.assertTrue(controller.mission_has_control)
        self.assertTrue(controller.live_requires_stop)
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        self.assertAlmostEqual(controller.cmd_pub.messages[0].linear.x, 0.0)
        self.assertAlmostEqual(controller.cmd_pub.messages[0].angular.z, 0.0)
        self.assertAlmostEqual(controller.path_follower.last_linear, 0.0)
        self.assertAlmostEqual(controller.path_follower.last_angular, 0.0)
        self.assertAlmostEqual(
            controller.path_follower.diagnostics.commanded_linear, 0.0
        )
        self.assertAlmostEqual(
            controller.path_follower.diagnostics.commanded_angular, 0.0
        )
        self.assertGreaterEqual(controller._publish_diagnostics.call_count, 2)

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

    def test_runtime_sweep_does_not_starve_source_stamped_odom_callback(self):
        controller = self.make_runtime_harness()
        controller._common_path_command = mock.Mock(return_value=Twist())
        entered_sweep = threading.Event()
        release_sweep = threading.Event()
        callback_finished = threading.Event()
        worker_errors = []
        real_validate_path = controller.path_checker.validator.validate_path

        def blocked_route_safety(*args, **kwargs):
            entered_sweep.set()
            if not release_sweep.wait(1.0):
                raise AssertionError("test did not release swept validation")
            return real_validate_path(*args, **kwargs)

        def run_control():
            try:
                controller.control_callback(None)
            except BaseException as error:  # surfaced in the test thread below
                worker_errors.append(error)

        odometry = Odometry()
        odometry.header.stamp = controller_module.rospy.Time.from_sec(10.02)
        odometry.header.frame_id = "odom"
        odometry.pose.pose.position.x = 0.03

        def update_odometry():
            controller.odom_callback(odometry)
            callback_finished.set()

        with mock.patch.object(
            controller.path_checker.validator,
            "validate_path",
            side_effect=blocked_route_safety,
        ):
            control_thread = threading.Thread(target=run_control)
            control_thread.start()
            self.assertTrue(entered_sweep.wait(0.5))
            callback_thread = threading.Thread(target=update_odometry)
            callback_thread.start()
            try:
                self.assertTrue(
                    callback_finished.wait(0.2),
                    "mission lock stayed held during route validation",
                )
            finally:
                release_sweep.set()
            callback_thread.join(1.0)
            control_thread.join(1.0)

        self.assertFalse(control_thread.is_alive())
        self.assertFalse(callback_thread.is_alive())
        self.assertEqual(worker_errors, [])
        self.assertEqual(controller.state, controller.AVOIDING)
        self.assertEqual(controller.odom_stamp, odometry.header.stamp)
        self.assertAlmostEqual(controller.odom_x, 0.03)

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

    def test_sensor_generations_during_route_sweep_use_latest_pose_for_command(self):
        controller = self.make_runtime_harness()
        follower = controller.path_follower
        original_goal_status = follower.goal_status
        original_common_path_command = controller._common_path_command
        follower.goal_status = mock.Mock(wraps=original_goal_status)
        controller._common_path_command = mock.Mock(
            wraps=original_common_path_command
        )

        def advance_odometry(_captured):
            for index in range(1, 6):
                self.seconds = 10.0 + 0.004 * index
                message = Odometry()
                message.header.stamp = self.now()
                message.header.frame_id = "odom"
                message.pose.pose.position.x = 0.006 * index
                controller.odom_callback(message)
            with controller.lock:
                controller.scan_generation += 2
                controller.scan_stamp = self.now()

        with mock.patch.object(
            follower,
            "calculate_tracking",
            wraps=follower.calculate_tracking,
        ) as calculate_tracking:
            captured = self.run_control_while_sweep_is_blocked(
                controller, advance_odometry
            )

        self.assertAlmostEqual(controller.odom_x, 0.03, places=12)
        self.assertEqual(controller.odom_generation, 5)
        self.assertEqual(controller.scan_generation, 3)
        self.assertAlmostEqual(captured["pose"].x, 0.0, places=12)
        self.assertEqual(calculate_tracking.call_count, 1)
        latest_pose = calculate_tracking.call_args.args[0]
        self.assertAlmostEqual(latest_pose.x, 0.03, places=12)
        follower.goal_status.assert_called_once()
        controller._common_path_command.assert_called_once()
        commanded_pose = controller._common_path_command.call_args.kwargs["pose"]
        self.assertAlmostEqual(commanded_pose.x, 0.03, places=12)
        self.assertEqual(len(controller.cmd_pub.messages), 1)

    def test_route_sweep_pose_jump_beyond_checked_reserve_holds_zero(self):
        controller = self.make_runtime_harness()
        controller._common_path_command = mock.Mock()

        def jump_odometry(_captured):
            self.seconds += 0.02
            message = Odometry()
            message.header.stamp = self.now()
            message.header.frame_id = "odom"
            message.pose.pose.position.x = 0.40
            controller.odom_callback(message)

        self.run_control_while_sweep_is_blocked(controller, jump_odometry)

        controller._common_path_command.assert_not_called()
        self.assertTrue(controller.cmd_pub.messages)
        self.assertTrue(
            all(
                abs(message.linear.x) <= 1e-12
                and abs(message.angular.z) <= 1e-12
                for message in controller.cmd_pub.messages
            )
        )

    def test_odom_epoch_change_during_route_sweep_holds_zero(self):
        controller = self.make_runtime_harness()
        controller.odom_epoch = 0
        controller._common_path_command = mock.Mock()

        def rewind_odometry(_captured):
            self.seconds += 0.02
            message = Odometry()
            message.header.stamp = controller_module.rospy.Time.from_sec(9.0)
            message.header.frame_id = "odom"
            controller.odom_callback(message)

        self.run_control_while_sweep_is_blocked(controller, rewind_odometry)

        self.assertEqual(controller.odom_epoch, 1)
        self.assertIsNone(controller.scan_stamp)
        controller._common_path_command.assert_not_called()
        self.assertTrue(controller.cmd_pub.messages)
        self.assertTrue(
            all(
                abs(message.linear.x) <= 1e-12
                and abs(message.angular.z) <= 1e-12
                for message in controller.cmd_pub.messages
            )
        )

    def test_new_scan_during_route_sweep_is_used_for_latest_stop(self):
        controller = self.make_runtime_harness()
        controller._lane_points = mock.Mock(
            side_effect=(
                np.empty((0, 2), dtype=np.float64),
                np.asarray([[0.02, 0.0]], dtype=np.float64),
            )
        )
        controller._lane_points_in_odom = mock.Mock(
            side_effect=lambda points, _heading: np.asarray(
                points, dtype=np.float64
            )
        )
        controller._common_path_command = mock.Mock()

        def publish_new_scan(_captured):
            self.seconds += 0.02
            with controller.lock:
                controller.scan_generation += 1
                controller.scan_stamp = self.now()

        self.run_control_while_sweep_is_blocked(controller, publish_new_scan)

        self.assertTrue(controller.live_requires_stop)
        self.assertEqual(controller.live_speed_limit, 0.0)
        controller._common_path_command.assert_not_called()
        self.assertTrue(controller.cmd_pub.messages)
        self.assertTrue(
            all(
                abs(message.linear.x) <= 1e-12
                and abs(message.angular.z) <= 1e-12
                for message in controller.cmd_pub.messages
            )
        )

    def test_repeated_sensor_churn_cannot_starve_velocity_publication(self):
        controller = self.make_runtime_harness()
        command_times = []

        def command(*_args, **_kwargs):
            command_times.append(self.seconds)
            message = Twist()
            message.linear.x = 0.04
            return message

        controller._common_path_command = mock.Mock(side_effect=command)

        for cycle in range(5):
            def advance_sensors(_captured, cycle=cycle):
                start = self.seconds
                for index in range(1, 4):
                    self.seconds = start + 0.02 * index
                    message = Odometry()
                    message.header.stamp = self.now()
                    message.header.frame_id = "odom"
                    message.pose.pose.position.x = 0.003 * (
                        3 * cycle + index
                    )
                    controller.odom_callback(message)
                self.seconds = start + 0.08
                with controller.lock:
                    controller.scan_generation += 1
                    controller.scan_stamp = self.now()

            self.run_control_while_sweep_is_blocked(
                controller, advance_sensors
            )

        self.assertEqual(len(command_times), 5)
        self.assertEqual(len(controller.cmd_pub.messages), 5)
        self.assertLess(max(np.diff(command_times)), 0.5)

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

    def test_local_terminal_completes_without_polygon_heartbeat(self):
        controller = self.make_runtime_harness()
        controller.state = controller.AVOIDING
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

        controller.control_callback(None)

        self.assertEqual(controller.state, controller.COMPLETE)
        self.assertFalse(controller.mission_has_control)
        controller._set_lane_controller.assert_called_once_with(True)
        self.assertEqual(controller.cmd_pub.messages, [])

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
