#!/usr/bin/env python3

from collections import deque
from dataclasses import replace
import math
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


NODE_DIR = Path(__file__).resolve().parents[1] / "nodes"
if str(NODE_DIR) not in sys.path:
    sys.path.insert(0, str(NODE_DIR))

import safe_lane_controller as controller_module
from safe_lane_controller import SafeLaneController
from custom_autorace_bringup.path_following import (
    AsymmetricFootprint,
    PathSafety,
    PointCloudBoundary,
    SweptFootprintValidator,
    path_from_xy,
    project_to_path,
)
from std_msgs.msg import Float64


class RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class SafeLaneControllerTest(unittest.TestCase):
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

    def advance(self, seconds):
        self.seconds += seconds

    def make_controller(self):
        controller = SafeLaneController.__new__(SafeLaneController)
        controller.lock = threading.Lock()
        controller.target_center = 500.0
        controller.maximum_velocity = 0.30
        controller.lane_timeout = 0.50
        controller.green_received = True
        controller.manual_stop_requested = False
        controller.mission_has_control = False
        controller.enabled = True
        controller.last_valid_lane_time = None
        controller.last_command_time = None
        controller.last_command_linear = 0.0
        controller.last_command_angular = 0.0
        controller.control_period = 1.0 / 30.0
        controller.odom_timeout = 0.35
        controller.maximum_pose_stamp_skew = 0.06
        controller.odom_history = deque(maxlen=120)
        controller.path_calibration = controller_module.LanePathCalibration(
            top_forward_distance=0.60,
            bottom_forward_distance=0.16,
            lane_width_m=0.25,
            lane_width_pixels=640.0,
            target_center=500.0,
            minimum_confidence=0.20,
            minimum_samples=3,
            minimum_forward_span=0.10,
            maximum_near_support_distance=0.37,
            sample_spacing=0.01,
            shape_control_spacing=0.01,
            endpoint_plateau_pixels=1.0,
            maximum_fit_residual_pixels=25.0,
            maximum_endpoint_displacement_pixels=20.0,
            fusion_minimum_samples=4,
            fusion_minimum_overlap=0.20,
            fusion_maximum_disagreement=0.03,
            minimum_executable_velocity=0.04,
            preferred_boundary="white",
        )
        controller.speed_profile = controller_module.SpeedProfile(
            # Match the repeat-safe Gazebo production rolling-path profile.
            cruise_velocity=0.28,
            minimum_velocity=0.06,
            entry_velocity=0.28,
            exit_velocity=0.10,
            maximum_angular_velocity=2.0,
            maximum_lateral_acceleration=0.08,
            linear_acceleration=0.20,
            linear_deceleration=0.35,
            angular_acceleration=2.50,
        )
        controller.tracking_config = controller_module.TrackingConfig(
            lookahead_distance=0.065,
            maximum_linear_velocity=0.30,
            maximum_angular_velocity=2.0,
            maximum_lateral_acceleration=0.08,
            linear_acceleration=0.20,
            linear_deceleration=0.35,
            angular_acceleration=2.50,
            heading_gain=0.30,
            curvature_feedforward_weight=0.15,
            lateral_feedback_gain=1.0,
            search_back=3,
            search_ahead_distance=0.50,
            lookahead_time=0.35,
            minimum_lookahead_distance=0.08,
            maximum_lookahead_distance=0.16,
            speed_preview_distance=0.0,
        )
        controller.path_follower = controller_module.PathFollower(
            controller.tracking_config
        )
        controller.path_safety_config = controller_module.LanePathSafetyConfig(
            line_half_width=0.005,
            boundary_sample_spacing=0.004,
            line_margin=0.0,
            obstacle_margin=0.0,
            localization_margin=0.002,
            tracking_margin=0.002,
            reaction_time=0.10,
            stopping_distance_margin=0.005,
            lookahead_distance=0.45,
        )
        # Rolling camera paths must use the same complete asymmetric-rectangle
        # sweep as the path-based mission controllers.  These are the measured
        # Burger dimensions already used by the Gazebo mission validation.
        controller.footprint = AsymmetricFootprint(
            front=0.067645,
            rear=0.118073,
            half_width=0.0903,
        )
        controller.path_validator = SweptFootprintValidator(
            controller.footprint,
            translation_step=0.004,
            heading_step=math.radians(1.0),
        )
        controller.rolling_path = None
        controller.initialize_active_path_from_odometry = True
        controller.last_lane_diagnostics = None
        controller.cmd_vel_pub = RecordingPublisher()
        controller.manual_stop_pub = RecordingPublisher()
        controller.diagnostics_pub = RecordingPublisher()
        return controller

    def make_centerline(self, center_values=None):
        rows = [599.0, 450.0, 300.0, 150.0, 0.0]
        centers = [500.0] * len(rows) if center_values is None else center_values
        return SimpleNamespace(
            header=SimpleNamespace(
                stamp=controller_module.rospy.Time.from_sec(self.seconds)
            ),
            image_width=1000,
            image_height=600,
            sample_rows=rows,
            center_x=centers,
            yellow_x=[value - 320.0 for value in centers],
            white_x=[value + 320.0 for value in centers],
            confidence=[1.0] * len(rows),
            yellow_valid=[True] * len(rows),
            white_valid=[True] * len(rows),
        )

    def make_boundary_centerline(
        self,
        yellow_lateral=None,
        white_lateral=None,
        yellow_valid=None,
        white_valid=None,
    ):
        """Build detector output from metric painted-boundary samples."""

        rows = np.asarray([599.0, 450.0, 300.0, 150.0, 0.0])
        count = int(rows.size)
        scale = 0.25 / 640.0

        def pixels(values):
            if values is None:
                return np.full(count, np.nan)
            values = np.asarray(values, dtype=np.float64)
            if values.size == 1:
                values = np.full(count, float(values.reshape(-1)[0]))
            return 500.0 - values.reshape(-1) / scale

        return SimpleNamespace(
            header=SimpleNamespace(
                stamp=controller_module.rospy.Time.from_sec(self.seconds)
            ),
            image_width=1000,
            image_height=600,
            sample_rows=rows.tolist(),
            center_x=[500.0] * count,
            yellow_x=pixels(yellow_lateral).tolist(),
            white_x=pixels(white_lateral).tolist(),
            confidence=[1.0] * count,
            yellow_valid=(
                [False] * count
                if yellow_valid is None
                else list(yellow_valid)
            ),
            white_valid=(
                [False] * count
                if white_valid is None
                else list(white_valid)
            ),
        )

    def append_odom(self, controller, pose=None, speed=0.0, angular=0.0):
        controller.odom_history.append(
            controller_module.OdomSample(
                stamp=self.seconds,
                pose=pose or controller_module.Pose2D(0.0, 0.0, 0.0),
                linear_velocity=speed,
                angular_velocity=angular,
                frame_id="odom",
            )
        )

    @staticmethod
    def handoff(controller, lane_enabled):
        return controller.mission_handoff_callback(
            SimpleNamespace(data=lane_enabled)
        )

    def test_disabled_centerline_callback_only_warms_path_cache(self):
        controller = self.make_controller()
        self.handoff(controller, False)
        self.append_odom(controller)

        controller.lane_centerline_callback(self.make_centerline())

        self.assertFalse(controller.enabled)
        self.assertTrue(controller.mission_has_control)
        self.assertEqual(
            controller.last_valid_lane_time,
            controller_module.rospy.Time.from_sec(self.seconds),
        )
        self.assertIsNotNone(controller.rolling_path)
        self.assertEqual(controller.cmd_vel_pub.messages, [])

    def test_runtime_speed_cap_limits_common_path_follower(self):
        controller = self.make_controller()
        controller.maximum_velocity_callback(Float64(data=0.12))
        self.append_odom(controller, speed=0.12)

        controller.lane_centerline_callback(self.make_centerline())

        self.assertAlmostEqual(
            controller.cmd_vel_pub.messages[-1].linear.x,
            0.12,
        )

    def test_handoff_keeps_fresh_cache_and_watchdog_stays_silent(self):
        controller = self.make_controller()
        self.handoff(controller, False)
        self.append_odom(controller)
        controller.lane_centerline_callback(self.make_centerline())
        cached_time = controller.last_valid_lane_time

        self.advance(0.10)
        response = self.handoff(controller, True)
        controller.watchdog_callback(None)

        self.assertTrue(response.success)
        self.assertTrue(controller.enabled)
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(controller.last_valid_lane_time, cached_time)
        self.assertEqual(controller.cmd_vel_pub.messages, [])

    def test_handoff_discards_stale_cache_and_watchdog_stops(self):
        controller = self.make_controller()
        self.handoff(controller, False)
        self.append_odom(controller)
        controller.lane_centerline_callback(self.make_centerline())

        self.advance(controller.lane_timeout + 0.01)
        response = self.handoff(controller, True)
        controller.watchdog_callback(None)

        self.assertTrue(response.success)
        self.assertIsNone(controller.last_valid_lane_time)
        self.assertEqual(len(controller.cmd_vel_pub.messages), 1)
        stop = controller.cmd_vel_pub.messages[0]
        self.assertAlmostEqual(stop.linear.x, 0.0)
        self.assertAlmostEqual(stop.angular.z, 0.0)

    def test_manual_stop_is_published_only_by_the_current_owner(self):
        controller = self.make_controller()

        response = controller.enable_callback(SimpleNamespace(data=False))

        self.assertTrue(response.success)
        self.assertEqual(len(controller.cmd_vel_pub.messages), 1)
        self.assertTrue(controller.manual_stop_pub.messages[-1].data)

        controller = self.make_controller()
        self.handoff(controller, False)

        response = controller.enable_callback(SimpleNamespace(data=False))

        self.assertTrue(response.success)
        self.assertEqual(controller.cmd_vel_pub.messages, [])
        self.assertTrue(controller.manual_stop_pub.messages[-1].data)

    def test_shutdown_publishes_only_when_lane_controller_owns_cmd_vel(self):
        controller = self.make_controller()
        controller.shutdown()
        self.assertEqual(len(controller.cmd_vel_pub.messages), 1)

        controller = self.make_controller()
        self.handoff(controller, False)
        controller.shutdown()
        self.assertEqual(controller.cmd_vel_pub.messages, [])

    def test_metric_rolling_path_is_frozen_at_the_capture_odom_pose(self):
        controller = self.make_controller()
        capture = controller_module.Pose2D(1.0, 2.0, 0.5 * math.pi)

        rolling = controller_module.build_rolling_lane_path(
            self.make_centerline(),
            capture,
            "odom",
            controller.path_calibration,
            controller.speed_profile,
        )

        self.assertEqual(rolling.valid_samples, 5)
        self.assertAlmostEqual(rolling.mean_confidence, 1.0)
        self.assertAlmostEqual(rolling.horizon, 0.60)
        self.assertEqual(rolling.path.frame_id, "odom")
        self.assertAlmostEqual(rolling.path.x[0], 1.0, places=6)
        self.assertAlmostEqual(rolling.path.y[0], 2.0, places=6)
        self.assertAlmostEqual(rolling.path.x[-1], 1.0, places=5)
        self.assertAlmostEqual(rolling.path.y[-1], 2.60, places=5)
        self.assertAlmostEqual(rolling.path.length, 0.60, places=6)
        self.assertLess(float(np.max(np.abs(rolling.path.curvature))), 1e-6)
        self.assertLessEqual(float(np.max(rolling.path.speed)), 0.28 + 1e-12)

    def test_straight_two_boundary_candidates_fuse_at_lane_center(self):
        controller = self.make_controller()
        message = self.make_boundary_centerline(
            yellow_lateral=0.125,
            white_lateral=-0.125,
            yellow_valid=[True] * 5,
            white_valid=[True] * 5,
        )

        rolling = controller_module.build_rolling_lane_path(
            message,
            controller_module.Pose2D(0.0, 0.0, 0.0),
            "odom",
            controller.path_calibration,
            controller.speed_profile,
        )

        self.assertEqual(rolling.valid_samples, 5)
        np.testing.assert_allclose(rolling.path.y, 0.0, atol=1e-10)
        self.assertLess(float(np.max(np.abs(rolling.path.curvature))), 1e-8)

    def test_position_close_opposite_tangents_are_not_fused(self):
        controller = self.make_controller()
        calibration = replace(
            controller.path_calibration,
            fusion_minimum_overlap=0.08,
        )
        yellow = controller_module.BoundaryCenterCandidate(
            side="yellow",
            coefficients=np.asarray([0.0, 0.10, -0.025]),
            boundary_coefficients=np.asarray([0.0, 0.10, -0.150]),
            start_parameter=0.20,
            end_parameter=0.30,
            boundary_start_parameter=0.20,
            boundary_end_parameter=0.30,
            support_count=5,
            mean_confidence=1.0,
        )
        white = controller_module.BoundaryCenterCandidate(
            side="white",
            coefficients=np.asarray([0.0, -0.10, 0.025]),
            boundary_coefficients=np.asarray([0.0, -0.10, 0.150]),
            start_parameter=0.20,
            end_parameter=0.30,
            boundary_start_parameter=0.20,
            boundary_end_parameter=0.30,
            support_count=5,
            mean_confidence=1.0,
        )
        parameter = np.linspace(0.20, 0.30, 5)
        valid = np.ones(5, dtype=np.bool_)
        confidence = np.ones(5)

        strict = controller_module.boundary_geometry_options(
            yellow,
            white,
            parameter,
            valid,
            valid,
            confidence,
            calibration,
        )
        relaxed = controller_module.boundary_geometry_options(
            yellow,
            white,
            parameter,
            valid,
            valid,
            confidence,
            replace(
                calibration,
                fusion_maximum_heading_disagreement=math.radians(20.0),
            ),
        )

        self.assertEqual(len(strict[0][5]), 1)
        self.assertEqual(len(relaxed[0][5]), 2)

    def test_two_straight_boundaries_create_safe_full_footprint_sweep(self):
        controller = self.make_controller()
        message = self.make_boundary_centerline(
            yellow_lateral=0.125,
            white_lateral=-0.125,
            yellow_valid=[True] * 5,
            white_valid=[True] * 5,
        )

        rolling = controller_module.build_rolling_lane_path(
            message,
            controller_module.Pose2D(0.0, 0.0, 0.0),
            "odom",
            controller.path_calibration,
            controller.speed_profile,
            controller.path_safety_config,
        )

        self.assertIsInstance(rolling.path.safety, PathSafety)
        self.assertEqual(len(rolling.path.safety.line_boundaries), 2)
        self.assertNotEqual(
            controller.path_validator.footprint.front,
            controller.path_validator.footprint.rear,
        )
        validation = controller.path_validator.validate_path(rolling.path)
        self.assertTrue(validation.safe, validation)
        self.assertTrue(math.isfinite(validation.minimum_line_clearance))
        self.assertGreater(validation.minimum_line_clearance, 0.0)

    def test_rolling_path_detects_between_sample_rectangle_line_contact(self):
        controller = self.make_controller()
        self.append_odom(controller)
        boundary = PointCloudBoundary([[0.50, 0.04]])
        path = path_from_xy(
            [[0.0, 0.0], [1.0, 0.0]],
            "odom",
            target_speed=0.10,
            safety=PathSafety(line_boundaries=(boundary,)),
            label="camera_lane",
        )

        centre_only = AsymmetricFootprint(1e-6, 1e-6, 1e-6)
        self.assertGreater(
            boundary.clearance(
                controller_module.Pose2D(0.50, 0.0, 0.0), centre_only
            ),
            0.0,
        )
        # Both stored path poses clear the line with the complete footprint.
        # Only the swept rectangle between them reaches the boundary point.
        for index in (0, 1):
            pose = controller_module.Pose2D(
                float(path.x[index]),
                float(path.y[index]),
                float(path.heading[index]),
            )
            self.assertGreater(
                boundary.clearance(pose, controller.path_validator.footprint),
                0.0,
            )
        validation = controller.path_validator.validate_path(path)
        self.assertGreater(validation.samples, path.size)
        self.assertFalse(validation.safe)
        self.assertLessEqual(validation.minimum_line_clearance, 0.0)

        rolling = controller_module.RollingLanePath(
            path=path,
            valid_samples=5,
            mean_confidence=1.0,
            horizon=1.0,
            observed_start_station=0.0,
        )
        with mock.patch.object(
            controller_module,
            "build_rolling_lane_path",
            return_value=rolling,
        ), mock.patch.object(
            controller.path_validator,
            "motion_safety",
            wraps=controller.path_validator.motion_safety,
        ) as motion_safety:
            controller.lane_centerline_callback(self.make_centerline())

        motion_safety.assert_called_once()
        self.assertIs(controller.rolling_path, path)
        self.assertEqual(len(controller.cmd_vel_pub.messages), 1)
        self.assertEqual(len(controller.diagnostics_pub.messages), 1)
        diagnostics = list(controller.diagnostics_pub.messages[0].data)
        self.assertLessEqual(diagnostics[8], 0.0)

    def test_curved_single_boundary_produces_curved_center_path(self):
        controller = self.make_controller()
        rows = np.asarray([599.0, 450.0, 300.0, 150.0, 0.0])
        forward = 0.60 + rows / 599.0 * (0.16 - 0.60)
        yellow_lateral = 0.125 + 0.30 * (forward - 0.16) ** 2
        message = self.make_boundary_centerline(
            yellow_lateral=yellow_lateral,
            yellow_valid=[True] * 5,
        )

        rolling = controller_module.build_rolling_lane_path(
            message,
            controller_module.Pose2D(0.0, 0.0, 0.0),
            "odom",
            controller.path_calibration,
            controller.speed_profile,
        )

        self.assertEqual(rolling.valid_samples, 5)
        self.assertTrue(np.all(np.diff(rolling.path.x) > 0.0))
        self.assertGreater(rolling.path.y[-1] - rolling.path.y[0], 0.05)
        self.assertGreater(float(np.max(rolling.path.curvature)), 0.25)
        first_observed = int(
            np.searchsorted(
                rolling.path.station,
                rolling.observed_start_station,
                side="left",
            )
        )
        self.assertGreater(first_observed, 0)
        self.assertAlmostEqual(rolling.path.y[0], 0.0, places=12)
        self.assertAlmostEqual(rolling.path.heading[0], 0.0, places=12)
        self.assertTrue(np.all(np.diff(rolling.path.station) > 0.0))
        self.assertTrue(np.all(rolling.path.direction == 1))
        self.assertTrue(np.all(np.isfinite(rolling.path.curvature)))
        unwrapped_heading = np.unwrap(rolling.path.heading)
        self.assertLess(float(np.max(np.abs(np.diff(unwrapped_heading)))), 0.20)

    def test_one_straight_boundary_is_offset_to_lane_center(self):
        controller = self.make_controller()
        message = self.make_boundary_centerline(
            yellow_lateral=0.125,
            yellow_valid=[True] * 5,
        )

        rolling = controller_module.build_rolling_lane_path(
            message,
            controller_module.Pose2D(0.0, 0.0, 0.0),
            "odom",
            controller.path_calibration,
            controller.speed_profile,
        )

        self.assertEqual(rolling.valid_samples, 5)
        np.testing.assert_allclose(rolling.path.y, 0.0, atol=1e-10)

    def test_one_straight_boundary_still_creates_path_safety(self):
        controller = self.make_controller()
        message = self.make_boundary_centerline(
            yellow_lateral=0.125,
            yellow_valid=[True] * 5,
        )

        rolling = controller_module.build_rolling_lane_path(
            message,
            controller_module.Pose2D(0.0, 0.0, 0.0),
            "odom",
            controller.path_calibration,
            controller.speed_profile,
            controller.path_safety_config,
        )

        self.assertIsInstance(rolling.path.safety, PathSafety)
        self.assertEqual(len(rolling.path.safety.line_boundaries), 1)
        self.assertGreaterEqual(
            float(
                np.min(rolling.path.safety.line_boundaries[0].points[:, 0])
            ),
            controller.path_calibration.bottom_forward_distance - 1e-9,
        )
        validation = controller.path_validator.validate_path(rolling.path)
        self.assertTrue(validation.safe, validation)
        self.assertTrue(math.isfinite(validation.minimum_line_clearance))
        self.assertGreater(validation.minimum_line_clearance, 0.0)

    def test_side_switch_is_local_and_one_minority_point_cannot_move_path(self):
        controller = self.make_controller()
        rows = np.asarray([599.0, 450.0, 300.0, 150.0, 0.0])
        forward = 0.60 + rows / 599.0 * (0.16 - 0.60)
        white_curve = -0.125 + 0.60 * (forward - 0.16) ** 2
        yellow_curve = 0.125 - 0.60 * (forward - 0.16) ** 2
        minority_yellow = np.full(5, 0.18)
        minority_white = np.full(5, -0.18)

        white_only = self.make_boundary_centerline(
            white_lateral=white_curve,
            white_valid=[True] * 5,
        )
        white_with_one_yellow = self.make_boundary_centerline(
            yellow_lateral=minority_yellow,
            white_lateral=white_curve,
            yellow_valid=[True, False, False, False, False],
            white_valid=[True] * 5,
        )
        yellow_only = self.make_boundary_centerline(
            yellow_lateral=yellow_curve,
            yellow_valid=[True] * 5,
        )
        yellow_with_one_white = self.make_boundary_centerline(
            yellow_lateral=yellow_curve,
            white_lateral=minority_white,
            yellow_valid=[True] * 5,
            white_valid=[False, False, False, False, True],
        )

        def local_path(message):
            return controller_module.build_rolling_lane_path(
                message,
                controller_module.Pose2D(0.0, 0.0, 0.0),
                "odom",
                controller.path_calibration,
                controller.speed_profile,
            ).path

        white_reference = local_path(white_only)
        white_mixed = local_path(white_with_one_yellow)
        yellow_reference = local_path(yellow_only)
        yellow_mixed = local_path(yellow_with_one_white)
        np.testing.assert_allclose(white_mixed.x, white_reference.x, atol=1e-12)
        np.testing.assert_allclose(white_mixed.y, white_reference.y, atol=1e-12)
        np.testing.assert_allclose(yellow_mixed.x, yellow_reference.x, atol=1e-12)
        np.testing.assert_allclose(yellow_mixed.y, yellow_reference.y, atol=1e-12)
        self.assertGreater(white_mixed.y[-1], 0.05)
        self.assertLess(yellow_mixed.y[-1], -0.05)

    def test_configured_boundary_preference_wins_when_both_are_supported(self):
        controller = self.make_controller()
        rows = np.asarray([599.0, 450.0, 300.0, 150.0, 0.0])
        forward = 0.60 + rows / 599.0 * (0.16 - 0.60)
        white_curve = -0.125 + 0.40 * (forward - 0.16) ** 2
        yellow_curve = 0.125 - 0.40 * (forward - 0.16) ** 2
        white_support = [True, True, True, False, False]
        mixed = self.make_boundary_centerline(
            yellow_lateral=yellow_curve,
            white_lateral=white_curve,
            yellow_valid=[True] * 5,
            white_valid=white_support,
        )
        white_only = self.make_boundary_centerline(
            white_lateral=white_curve,
            white_valid=white_support,
        )
        yellow_only = self.make_boundary_centerline(
            yellow_lateral=yellow_curve,
            yellow_valid=[True] * 5,
        )

        def local_path(message):
            return controller_module.build_rolling_lane_path(
                message,
                controller_module.Pose2D(0.0, 0.0, 0.0),
                "odom",
                controller.path_calibration,
                controller.speed_profile,
            ).path

        selected = local_path(mixed)
        reference = local_path(white_only)
        np.testing.assert_allclose(selected.x, reference.x, atol=1e-12)
        np.testing.assert_allclose(selected.y, reference.y, atol=1e-12)
        self.assertGreater(selected.y[-1], 0.0)
        self.assertLess(selected.length, local_path(yellow_only).length)

    def test_sparse_boundary_fit_is_refit_from_observed_offset_nodes(self):
        controller = self.make_controller()
        parameter = np.asarray([0.19452, 0.24741, 0.30618, 0.36494])
        # Captured white-boundary fit at bag t=128.002. Its 0.125 m
        # normal offset reverses differential direction inside this interval.
        boundary_lateral = np.polyval(
            [5.393122779480086, -1.6941794143243125, 0.05798844820137858],
            parameter,
        )

        candidate = controller_module.build_boundary_candidate(
            "white",
            parameter,
            boundary_lateral,
            np.ones(4, dtype=np.bool_),
            np.ones(4),
            controller.path_calibration,
        )

        self.assertIsNotNone(candidate)
        _, forward, lateral = controller_module.sample_candidate(
            candidate,
            candidate.start_parameter,
            candidate.end_parameter,
            controller.path_calibration.sample_spacing,
        )
        path = path_from_xy(
            np.column_stack((forward, lateral)),
            "base_footprint",
            controller.speed_profile,
        )
        self.assertTrue(np.all(np.diff(path.x) > 0.0))
        self.assertLess(float(np.max(np.abs(path.curvature))), 10.0)

    def test_constrained_quadratic_cannot_reverse_monotone_endpoint(self):
        parameter = np.asarray([0.1945, 0.2474, 0.3062, 0.3650])
        pixels = np.asarray([625.0, 507.0, 423.0, 417.0])
        lateral = (500.0 - pixels) * 0.25 / 640.0
        unconstrained = np.polyfit(parameter, lateral, 2)
        constrained = controller_module.fit_endpoint_constrained_quadratic(
            parameter,
            lateral,
            np.ones(parameter.size),
            0.25 / 640.0,
        )
        derivative = np.polyval(
            np.polyder(constrained), [parameter[0], parameter[-1]]
        )

        self.assertLess(
            float(np.polyval(np.polyder(unconstrained), parameter[-1])),
            0.0,
        )
        self.assertGreaterEqual(derivative[0], -1e-10)
        self.assertGreaterEqual(derivative[-1], -1e-10)

    def test_constrained_quadratic_keeps_observed_s_direction_change(self):
        parameter = np.asarray([0.16, 0.25, 0.35, 0.46])
        lateral = np.asarray([-0.04, 0.005, 0.03, -0.01])
        constrained = controller_module.fit_endpoint_constrained_quadratic(
            parameter,
            lateral,
            np.ones(parameter.size),
            1e-6,
        )
        endpoint_derivative = np.polyval(
            np.polyder(constrained), [parameter[0], parameter[-1]]
        )

        self.assertGreater(endpoint_derivative[0], 0.0)
        self.assertLess(endpoint_derivative[-1], 0.0)
        turning_parameter = -constrained[1] / (2.0 * constrained[0])
        self.assertGreater(turning_parameter, parameter[0])
        self.assertLess(turning_parameter, parameter[-1])

    def test_subpixel_endpoint_delta_is_constrained_as_plateau(self):
        parameter = np.asarray([0.19, 0.25, 0.31, 0.37])
        pixels = np.asarray([805.58, 805.76, 755.16, 672.06])
        lateral = (500.0 - pixels) * 0.25 / 640.0
        constrained = controller_module.fit_endpoint_constrained_quadratic(
            parameter,
            lateral,
            np.ones(parameter.size),
            0.25 / 640.0,
        )

        self.assertAlmostEqual(
            float(np.polyval(np.polyder(constrained), parameter[0])),
            0.0,
            places=9,
        )

    def test_monotone_recorded_knots_do_not_reverse_endpoint_heading(self):
        controller = self.make_controller()
        rows = [552.0, 480.0, 400.0, 320.0]
        white = [625.0, 507.0, 423.0, 417.0]
        message = SimpleNamespace(
            header=SimpleNamespace(
                stamp=controller_module.rospy.Time.from_sec(self.seconds)
            ),
            image_width=1000,
            image_height=600,
            sample_rows=rows,
            center_x=[math.nan] * len(rows),
            yellow_x=[math.nan] * len(rows),
            white_x=white,
            confidence=[0.5] * len(rows),
            yellow_valid=[False] * len(rows),
            white_valid=[True] * len(rows),
        )

        rolling = controller_module.build_rolling_lane_path(
            message,
            controller_module.Pose2D(0.0, 0.0, 0.0),
            "odom",
            controller.path_calibration,
            controller.speed_profile,
        )

        self.assertEqual(rolling.valid_samples, 4)
        self.assertAlmostEqual(rolling.path.heading[0], 0.0, places=12)
        self.assertGreaterEqual(rolling.path.heading[-1], -1e-12)
        self.assertLess(float(np.max(np.abs(rolling.path.curvature))), 10.0)
        self.assertTrue(np.all(np.diff(rolling.path.station) > 0.0))
        self.assertTrue(np.all(np.isfinite(rolling.path.curvature)))

    def test_curvature_profile_slows_before_camera_detected_bend(self):
        controller = self.make_controller()
        scale = (
            controller.path_calibration.lane_width_m
            / controller.path_calibration.lane_width_pixels
        )
        rows = np.asarray(self.make_centerline().sample_rows)
        forward = 0.60 + rows / 599.0 * (0.16 - 0.60)
        lateral = 0.75 * forward ** 2
        centers = list(500.0 - lateral / scale)
        message = self.make_centerline(centers)
        fast_profile = controller_module.SpeedProfile(
            cruise_velocity=0.26,
            minimum_velocity=0.06,
            entry_velocity=0.26,
            exit_velocity=0.10,
            maximum_angular_velocity=2.0,
            maximum_lateral_acceleration=0.03,
            linear_acceleration=0.20,
            linear_deceleration=0.35,
            angular_acceleration=2.50,
        )

        rolling = controller_module.build_rolling_lane_path(
            message,
            controller_module.Pose2D(0.0, 0.0, 0.0),
            "odom",
            controller.path_calibration,
            fast_profile,
        )

        self.assertGreater(float(np.max(rolling.path.curvature)), 0.25)
        self.assertLess(float(np.min(rolling.path.speed)), 0.26)

    def test_camera_path_marks_first_observation_after_projection_connector(self):
        controller = self.make_controller()
        rolling = controller_module.build_rolling_lane_path(
            self.make_centerline(),
            controller_module.Pose2D(0.0, 0.0, 0.0),
            "odom",
            controller.path_calibration,
            controller.speed_profile,
        )

        self.assertAlmostEqual(rolling.path.x[0], 0.0, places=12)
        self.assertAlmostEqual(rolling.path.y[0], 0.0, places=12)
        self.assertAlmostEqual(rolling.path.heading[0], 0.0, places=12)
        self.assertGreater(rolling.observed_start_station, 0.16)
        self.assertLess(rolling.observed_start_station, 0.22)
        self.assertAlmostEqual(rolling.path.x[-1], 0.60, places=6)

    def test_ego_connector_matches_first_candidate_position_and_tangent(self):
        forward = np.linspace(0.20, 0.60, 41)
        lateral = 0.04 + 0.30 * (forward - 0.20) + 0.20 * (
            forward - 0.20
        ) ** 2
        observed_slope = controller_module.shape_preserving_slopes(
            forward, lateral
        )

        connected_forward, connected_lateral = (
            controller_module.connect_ego_to_lane_candidate(
                forward, lateral, 0.005
            )
        )

        join = int(np.flatnonzero(np.isclose(connected_forward, 0.20))[0])
        self.assertAlmostEqual(connected_forward[0], 0.0)
        self.assertAlmostEqual(connected_lateral[0], 0.0)
        self.assertAlmostEqual(connected_lateral[join], lateral[0], places=12)
        left_slope = (
            connected_lateral[join] - connected_lateral[join - 1]
        ) / (
            connected_forward[join] - connected_forward[join - 1]
        )
        self.assertAlmostEqual(left_slope, observed_slope[0], delta=0.02)
        first_slope = (
            connected_lateral[1] - connected_lateral[0]
        ) / (
            connected_forward[1] - connected_forward[0]
        )
        self.assertAlmostEqual(first_slope, 0.0, delta=0.02)

        connected_poses = controller_module.ego_connected_lane_poses(
            forward, lateral, 0.005
        )
        pose_join = int(
            np.flatnonzero(np.isclose(connected_poses[:, 0], 0.20))[0]
        )
        expected_join_heading = math.atan(observed_slope[0])
        self.assertAlmostEqual(connected_poses[0, 2], 0.0, places=12)
        self.assertAlmostEqual(
            connected_poses[pose_join, 2], expected_join_heading, places=10
        )
        self.assertTrue(np.all(np.isfinite(connected_poses)))
        self.assertLess(
            float(np.max(np.abs(np.diff(np.unwrap(connected_poses[:, 2]))))),
            0.10,
        )

    def test_c1_connector_keeps_a_conflicting_observed_tangent(self):
        forward = np.linspace(0.16, 0.60, 45)
        lateral = -0.25 * (forward - forward[0]) + 0.20 * (
            forward - forward[0]
        ) ** 2

        poses = controller_module.ego_connected_lane_poses(
            forward, lateral, 0.005
        )
        join = int(np.flatnonzero(np.isclose(poses[:, 0], forward[0]))[0])

        observed_slope = controller_module.shape_preserving_slopes(
            forward, lateral
        )[0]
        self.assertAlmostEqual(poses[0, 2], 0.0, places=12)
        self.assertAlmostEqual(
            poses[join, 2], math.atan(observed_slope), places=12
        )
        # With y(join)=0 and a non-zero join tangent, a C1 cubic must make a
        # small opposite-side excursion. Its analytic maximum is 4/27*x*|m|.
        self.assertGreater(float(np.max(poses[:join, 1])), 0.0)
        self.assertLessEqual(
            float(np.max(poses[:join, 1])),
            4.0 / 27.0 * forward[0] * abs(observed_slope) + 1e-12,
        )
        self.assertTrue(np.all(np.isfinite(poses)))

    def test_c1_connector_is_finite_bounded_and_mirror_symmetric(self):
        mirrored = []
        for sign in (-1.0, 1.0):
            forward = np.asarray([0.16, 0.18, 0.24, 0.33, 0.48, 0.60])
            lateral = sign * (
                0.01 + 1.0 * (forward - forward[0])
            )

            poses = controller_module.ego_connected_lane_poses(
                forward, lateral, 0.002
            )
            join = int(
                np.flatnonzero(np.isclose(poses[:, 0], forward[0]))[0]
            )
            observed_slope = controller_module.shape_preserving_slopes(
                forward, lateral
            )[0]
            bound = abs(lateral[0]) + (
                4.0 / 27.0 * forward[0] * abs(observed_slope)
            )
            self.assertAlmostEqual(
                math.tan(poses[join, 2]), observed_slope, places=12
            )
            self.assertLessEqual(
                float(np.max(np.abs(poses[: join + 1, 1]))),
                bound + 1e-12,
            )
            self.assertTrue(np.all(np.isfinite(poses)))
            mirrored.append(poses)
        np.testing.assert_allclose(
            mirrored[0][:, :1], mirrored[1][:, :1], atol=1e-12
        )
        np.testing.assert_allclose(
            mirrored[0][:, 1:], -mirrored[1][:, 1:], atol=1e-12
        )

    def test_shape_preserving_slopes_handle_nonuniform_plateau_and_mirror(self):
        parameter = np.asarray([0.0, 0.16, 0.19, 0.28, 0.60])
        values = np.asarray([0.0, 0.01, 0.04, 0.04, 0.02])
        slope = controller_module.shape_preserving_slopes(
            parameter, values, first_slope=0.0
        )
        mirrored = controller_module.shape_preserving_slopes(
            parameter, -values, first_slope=0.0
        )

        self.assertTrue(np.all(np.isfinite(slope)))
        self.assertAlmostEqual(slope[0], 0.0, places=12)
        self.assertAlmostEqual(slope[2], 0.0, places=12)
        self.assertAlmostEqual(slope[3], 0.0, places=12)
        np.testing.assert_allclose(mirrored, -slope, atol=1e-12)

    def test_normal_offset_with_nonincreasing_forward_is_rejected(self):
        controller = self.make_controller()
        parameter = np.linspace(0.16, 0.28, 7)
        boundary_lateral = 50.0 * (parameter - 0.22) ** 2

        candidate = controller_module.build_boundary_candidate(
            "white",
            parameter,
            boundary_lateral,
            np.ones(parameter.size, dtype=np.bool_),
            np.ones(parameter.size),
            controller.path_calibration,
        )

        self.assertIsNone(candidate)

    def test_quadratic_fit_residual_guard_rejects_mirrored_raw_motion(self):
        controller = self.make_controller()
        parameter = np.asarray([0.16, 0.23, 0.30, 0.37])
        observed = np.asarray([0.0, 0.02, -0.02, 0.0])

        for side, sign in (("white", 1.0), ("yellow", -1.0)):
            candidate = controller_module.build_boundary_candidate(
                side,
                parameter,
                sign * observed,
                np.ones(parameter.size, dtype=np.bool_),
                np.ones(parameter.size),
                controller.path_calibration,
            )
            self.assertIsNone(candidate)

    def test_endpoint_displacement_has_its_own_stricter_limit(self):
        controller = self.make_controller()
        parameter = np.asarray([0.1945, 0.2474, 0.3062, 0.3650])
        pixels = np.asarray([625.0, 507.0, 423.0, 417.0])
        boundary_lateral = (500.0 - pixels) * 0.25 / 640.0
        endpoint_limited = replace(
            controller.path_calibration,
            maximum_fit_residual_pixels=25.0,
            maximum_endpoint_displacement_pixels=5.0,
        )

        candidate = controller_module.build_boundary_candidate(
            "white",
            parameter,
            boundary_lateral,
            np.ones(parameter.size, dtype=np.bool_),
            np.ones(parameter.size),
            endpoint_limited,
        )

        self.assertIsNone(candidate)

    def test_far_only_boundary_fragment_cannot_define_ego_route(self):
        controller = self.make_controller()
        parameter = np.asarray([0.482, 0.541, 0.600])
        boundary_lateral = np.asarray([-0.10, -0.08, -0.06])

        candidate = controller_module.build_boundary_candidate(
            "yellow",
            parameter,
            boundary_lateral,
            np.ones(parameter.size, dtype=np.bool_),
            np.ones(parameter.size),
            controller.path_calibration,
        )

        self.assertIsNone(candidate)

    def test_recorded_join_spike_is_smoothed_without_weakening_speed_limit(self):
        controller = self.make_controller()
        rows = [552.0, 480.0, 400.0, 320.0, 240.0]
        white = [687.3, 572.6, 399.6, 280.7, 241.6]
        message = SimpleNamespace(
            image_width=1000,
            image_height=600,
            sample_rows=rows,
            center_x=[math.nan] * len(rows),
            yellow_x=[math.nan] * len(rows),
            white_x=white,
            confidence=[0.5] * len(rows),
            yellow_valid=[False] * len(rows),
            white_valid=[True] * len(rows),
        )
        residual_relaxed = replace(
            controller.path_calibration,
            maximum_fit_residual_pixels=1000.0,
            maximum_endpoint_displacement_pixels=1000.0,
        )

        rolling = controller_module.build_rolling_lane_path(
            message,
            controller_module.Pose2D(0.0, 0.0, 0.0),
            "odom",
            residual_relaxed,
            controller.speed_profile,
        )

        self.assertGreaterEqual(
            float(np.min(rolling.path.speed)),
            controller.path_calibration.minimum_executable_velocity,
        )
        self.assertGreater(float(np.max(np.abs(rolling.path.curvature))), 20.0)

    def test_run5_curve_uses_valid_same_frame_alternate_boundary(self):
        controller = self.make_controller()
        # Exact run5 frame at bag t=235.064. The white fragment starts only at
        # the middle row and its back-projected centre is outside the lane that
        # contains the robot. The longer yellow support is measured nearer in
        # this same frame and retains its actual lateral/heading error. No
        # previous path is an input to this pure per-frame construction.
        message = SimpleNamespace(
            image_width=1000,
            image_height=600,
            sample_rows=[552.0, 480.0, 400.0, 320.0, 240.0, 160.0, 80.0],
            center_x=[
                622.5,
                728.8200073242188,
                892.8400268554688,
                math.nan,
                121.62000274658203,
                347.2799987792969,
                573.0599975585938,
            ],
            yellow_x=[
                302.5,
                408.82000732421875,
                572.8400268554688,
                792.5399780273438,
                math.nan,
                math.nan,
                math.nan,
            ],
            white_x=[
                math.nan,
                math.nan,
                math.nan,
                217.6199951171875,
                441.6199951171875,
                667.280029296875,
                893.0599975585938,
            ],
            confidence=[
                0.2958124876022339,
                0.28824999928474426,
                0.28891345858573914,
                0.5841586589813232,
                0.3046322166919708,
                0.2698173224925995,
                0.2862403988838196,
            ],
            yellow_valid=[True, True, True, True, False, False, False],
            white_valid=[False, False, False, True, True, True, True],
        )

        rolling = controller_module.build_rolling_lane_path(
            message,
            controller_module.Pose2D(0.0, 0.0, 0.0),
            "odom",
            controller.path_calibration,
            controller.speed_profile,
        )

        physical_limit = min(
            controller.speed_profile.maximum_angular_velocity
            / controller.speed_profile.minimum_velocity,
            controller.speed_profile.maximum_lateral_acceleration
            / controller.speed_profile.minimum_velocity ** 2,
        )
        self.assertEqual(rolling.valid_samples, 4)
        self.assertAlmostEqual(rolling.horizon, 0.2584512581, places=8)
        self.assertLess(
            float(np.max(np.abs(rolling.path.curvature))), physical_limit
        )
        self.assertGreater(
            float(np.max(np.abs(rolling.path.curvature))), 3.0
        )
        self.assertLess(rolling.path.x[-1], 0.27)
        self.assertLess(rolling.path.y[-1], -0.17)

        # Removing yellow leaves a different current-frame white route. It is
        # allowed only because the new C1 connector makes its complete speed
        # profile executable; no historical route is consulted.
        white_only = SimpleNamespace(
            **dict(vars(message), yellow_valid=[False] * 7)
        )
        white_rolling = controller_module.build_rolling_lane_path(
            white_only,
            controller_module.Pose2D(0.0, 0.0, 0.0),
            "odom",
            controller.path_calibration,
            controller.speed_profile,
        )
        self.assertGreaterEqual(
            float(np.min(white_rolling.path.speed)),
            controller.path_calibration.minimum_executable_velocity,
        )
        self.assertFalse(
            np.allclose(
                rolling.path.y,
                white_rolling.path.y[: rolling.path.size],
                atol=1e-6,
            )
        )

    def test_run8_obstacle_approach_keeps_executable_tight_bend(self):
        controller = self.make_controller()
        # Exact stationary run-8 frame at bag t=224.468. The long, near yellow
        # support describes the local reference direction while preserving the
        # robot's offset from it. The short white fragment back-projects outside
        # the current lane and must remain rejected.
        message = SimpleNamespace(
            image_width=1000,
            image_height=600,
            sample_rows=[552.0, 480.0, 400.0, 320.0, 240.0, 160.0, 80.0, 0.0],
            center_x=[
                math.nan,
                math.nan,
                math.nan,
                math.nan,
                1101.8800048828125,
                1168.02001953125,
                1234.6800537109375,
                1299.7857666015625,
            ],
            yellow_x=[
                523.97998046875,
                583.9000244140625,
                649.9199829101562,
                715.719970703125,
                781.8800048828125,
                848.02001953125,
                914.6799926757812,
                979.7857055664062,
            ],
            white_x=[
                154.8000030517578,
                240.33999633789062,
                294.3399963378906,
                352.4800109863281,
                math.nan,
                math.nan,
                math.nan,
                math.nan,
            ],
            confidence=[
                0.6734134554862976,
                0.6992596387863159,
                0.6270769238471985,
                0.540134608745575,
                0.3930865526199341,
                0.4068197011947632,
                0.4258798062801361,
                0.4401785731315613,
            ],
            yellow_valid=[True] * 8,
            white_valid=[True, True, True, True, False, False, False, False],
        )

        rolling = controller_module.build_rolling_lane_path(
            message,
            controller_module.Pose2D(0.0, 0.0, 0.0),
            "odom",
            controller.path_calibration,
            controller.speed_profile,
        )

        self.assertEqual(rolling.valid_samples, 8)
        self.assertAlmostEqual(rolling.horizon, 0.5498266707, places=8)
        peak_curvature = float(np.max(np.abs(rolling.path.curvature)))
        self.assertGreater(peak_curvature, 26.0)
        self.assertLess(peak_curvature, 28.0)
        self.assertGreaterEqual(
            float(np.min(rolling.path.speed)),
            controller.path_calibration.minimum_executable_velocity,
        )
        self.assertLess(rolling.path.y[-1], -0.30)

        yellow_removed = SimpleNamespace(
            **dict(vars(message), yellow_valid=[False] * 8)
        )
        with self.assertRaisesRegex(ValueError, "physical curvature"):
            controller_module.build_rolling_lane_path(
                yellow_removed,
                controller_module.Pose2D(0.0, 0.0, 0.0),
                "odom",
                controller.path_calibration,
                controller.speed_profile,
            )

    def test_camera_stamp_pose_interpolates_position_velocity_and_wrapped_yaw(self):
        controller = self.make_controller()
        before = controller_module.OdomSample(
            stamp=9.98,
            pose=controller_module.Pose2D(1.0, 2.0, math.radians(179.0)),
            linear_velocity=0.10,
            angular_velocity=-0.20,
            frame_id="odom",
        )
        after = controller_module.OdomSample(
            stamp=10.02,
            pose=controller_module.Pose2D(1.2, 2.4, math.radians(-179.0)),
            linear_velocity=0.30,
            angular_velocity=0.20,
            frame_id="odom",
        )
        controller.odom_history.extend((before, after))

        capture, latest, skew = controller._odom_for_stamp_locked(10.0, 10.02)

        self.assertIs(latest, after)
        self.assertAlmostEqual(skew, 0.02, places=8)
        self.assertAlmostEqual(capture.stamp, 10.0)
        self.assertAlmostEqual(capture.pose.x, 1.1)
        self.assertAlmostEqual(capture.pose.y, 2.2)
        self.assertAlmostEqual(abs(capture.pose.yaw), math.pi, places=6)
        self.assertAlmostEqual(capture.linear_velocity, 0.20)
        self.assertAlmostEqual(capture.angular_velocity, 0.0)

    def test_centerline_uses_common_follower_as_the_only_command_path(self):
        controller = self.make_controller()
        self.append_odom(controller)

        controller.lane_centerline_callback(self.make_centerline())

        self.assertEqual(len(controller.cmd_vel_pub.messages), 1)
        command = controller.cmd_vel_pub.messages[0]
        self.assertGreater(command.linear.x, 0.0)
        self.assertAlmostEqual(command.angular.z, 0.0, places=6)
        self.assertEqual(len(controller.diagnostics_pub.messages), 1)

    def test_camera_path_steering_target_moves_farther_with_speed(self):
        controller = self.make_controller()
        rolling = controller_module.build_rolling_lane_path(
            self.make_centerline(),
            controller_module.Pose2D(0.0, 0.0, 0.0),
            "odom",
            controller.path_calibration,
            controller.speed_profile,
        )
        pose = controller_module.Pose2D(0.0, 0.0, 0.0)
        controller.path_follower.reset(rolling.path, pose)

        slow = controller.path_follower.calculate_tracking(
            pose,
            linear_velocity=0.05,
        )
        fast = controller.path_follower.calculate_tracking(
            pose,
            linear_velocity=0.28,
        )

        self.assertAlmostEqual(slow.steering_lookahead_distance, 0.0825)
        self.assertAlmostEqual(fast.steering_lookahead_distance, 0.16)
        self.assertGreater(fast.target_index, slow.target_index)
        self.assertLess(fast.steering_lookahead_distance, rolling.path.length)

    def test_lane_path_diagnostics_use_common_thirteen_value_schema(self):
        controller = self.make_controller()
        self.append_odom(controller)

        controller.lane_centerline_callback(self.make_centerline())

        self.assertEqual(len(controller.diagnostics_pub.messages), 1)
        diagnostics = list(controller.diagnostics_pub.messages[0].data)
        expected = controller.path_follower.diagnostics.as_array()
        self.assertEqual(len(expected), 13)
        self.assertEqual(len(diagnostics), 13)
        np.testing.assert_allclose(diagnostics, expected, atol=1e-12)
        self.assertEqual(controller.last_lane_diagnostics, diagnostics)

    def test_first_path_after_mission_handoff_starts_from_current_odom(self):
        controller = self.make_controller()
        controller.enabled = False
        controller.mission_has_control = True
        controller.rolling_path = object()
        controller.path_follower.last_linear = 0.29
        controller.path_follower.last_angular = -1.0
        self.append_odom(controller, speed=0.10, angular=0.15)

        response = self.handoff(controller, True)
        controller.lane_centerline_callback(self.make_centerline())

        self.assertTrue(response.success)
        self.assertTrue(controller.enabled)
        self.assertFalse(controller.initialize_active_path_from_odometry)
        command = controller.cmd_vel_pub.messages[-1]
        expected_period = controller.control_period
        self.assertAlmostEqual(
            command.linear.x,
            0.10 + controller.tracking_config.linear_acceleration * expected_period,
            places=7,
        )
        self.assertAlmostEqual(
            command.angular.z,
            0.15 - controller.tracking_config.angular_acceleration * expected_period,
            places=7,
        )

    def test_controller_rejects_unsynchronized_camera_path(self):
        controller = self.make_controller()
        self.append_odom(controller)
        self.advance(controller.maximum_pose_stamp_skew + 0.01)

        controller.lane_centerline_callback(self.make_centerline())

        self.assertEqual(controller.cmd_vel_pub.messages, [])
        self.assertEqual(controller.diagnostics_pub.messages, [])


if __name__ == "__main__":
    unittest.main()
