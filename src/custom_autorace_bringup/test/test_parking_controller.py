#!/usr/bin/env python3

import copy
from collections import deque
import math
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml


NODE_DIR = Path(__file__).resolve().parents[1] / "nodes"
if str(NODE_DIR) not in sys.path:
    sys.path.insert(0, str(NODE_DIR))

import parking_mission_controller as controller_module
from parking_mission_controller import ParkingMissionController
from custom_autorace_bringup.parking_geometry import (
    LEFT,
    RIGHT,
    choose_clear_space,
    map_from_odom_transform,
    map_pose_to_odom,
    normalize_angle,
    odom_pose_to_map,
    path_curvature_limits,
    quintic_pose_path,
    quintic_turn_path,
    rectangle_corners,
    scan_points_in_map,
)
from custom_autorace_bringup.path_following import (
    Pose2D,
    RigidTransform2D,
    footprint_points,
)
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64MultiArray

class RecordingPublisher:
    def __init__(self, topic, events):
        self.topic = topic
        self.events = events
        self.messages = []

    def publish(self, message):
        snapshot = copy.deepcopy(message)
        self.messages.append(snapshot)
        self.events.append(("publish", self.topic, snapshot))


class RecordingLaneService:
    def __init__(self, events):
        self.events = events
        self.success = True
        self.calls = []

    def __call__(self, enabled):
        enabled = bool(enabled)
        self.calls.append(enabled)
        self.events.append(("service", enabled, None))
        return SimpleNamespace(
            success=self.success,
            message="ok" if self.success else "rejected",
        )


class ParkingGeometryTest(unittest.TestCase):
    def test_quintic_pose_path_has_exact_pose_and_zero_endpoint_curvature(self):
        start = (1.0082, 1.7617, math.pi + math.radians(1.6))
        end = (0.7077, 1.7415, math.pi)
        path = quintic_pose_path(start, end, 0.0601, 0.0700, 1001)

        np.testing.assert_allclose(path[0, :2], start[:2], atol=1e-12)
        np.testing.assert_allclose(path[-1, :2], end[:2], atol=1e-12)
        self.assertAlmostEqual(
            math.sin(path[0, 2]), math.sin(start[2]), places=12
        )
        self.assertAlmostEqual(
            math.cos(path[0, 2]), math.cos(start[2]), places=12
        )
        self.assertAlmostEqual(
            math.sin(path[-1, 2]), math.sin(end[2]), places=12
        )
        self.assertAlmostEqual(
            math.cos(path[-1, 2]), math.cos(end[2]), places=12
        )
        self.assertTrue(
            np.all(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1) > 0.0)
        )

        # The equally spaced collinear endpoint controls make curvature tend
        # to zero at both ends; use a fine finite-difference estimate here.
        segment_lengths = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
        curvatures = np.diff(np.unwrap(path[:, 2])) / segment_lengths
        self.assertLess(abs(curvatures[0]), 0.02)
        self.assertLess(abs(curvatures[-1]), 0.02)

    def test_quintic_pose_path_rejects_invalid_or_degenerate_inputs(self):
        cases = (
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.0, 0.1, 101),
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.1, -0.1, 101),
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.1, 0.1, 2),
            ((0.0, 0.0, 0.0), (math.nan, 0.0, 0.0), 0.1, 0.1, 101),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    quintic_pose_path(*arguments)

    def test_quintic_turn_has_exact_pose_and_bounded_c2_geometry(self):
        path = quintic_turn_path(
            (0.7077, 1.7430, math.pi),
            -0.5 * math.pi,
            0.2000,
            0.0600,
            1001,
        )
        np.testing.assert_allclose(path[0, :2], (0.7077, 1.7430), atol=1e-12)
        np.testing.assert_allclose(path[-1, :2], (0.5077, 1.5430), atol=1e-12)
        self.assertAlmostEqual(math.sin(path[0, 2]), 0.0, places=12)
        self.assertAlmostEqual(math.cos(path[0, 2]), -1.0, places=12)
        self.assertAlmostEqual(math.sin(path[-1, 2]), -1.0, places=12)
        self.assertAlmostEqual(math.cos(path[-1, 2]), 0.0, places=12)

        maximum_curvature, maximum_rate = path_curvature_limits(path)
        self.assertLess(maximum_curvature, 5.87)
        self.assertLess(maximum_rate, 177.0)

    def test_quintic_turn_rejects_non_quarter_and_degenerate_parameters(self):
        cases = (
            ((0.0, 0.0, 0.0), math.pi, 0.2, 0.06, 101),
            ((0.0, 0.0, 0.0), 0.5 * math.pi, 0.0, 0.06, 101),
            ((0.0, 0.0, 0.0), 0.5 * math.pi, 0.2, 0.10, 101),
            ((0.0, 0.0, 0.0), 0.5 * math.pi, 0.2, 0.06, 2),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    quintic_turn_path(*arguments)

    def test_map_from_odom_round_trip_handles_wrapped_yaw(self):
        map_pose = (1.23, -0.42, math.radians(174.0))
        odom_pose = (-0.31, 0.67, math.radians(-169.0))

        transform = map_from_odom_transform(map_pose, odom_pose)
        recovered = map_pose_to_odom(map_pose, transform)

        self.assertAlmostEqual(recovered[0], odom_pose[0], places=9)
        self.assertAlmostEqual(recovered[1], odom_pose[1], places=9)
        self.assertAlmostEqual(
            math.sin(recovered[2]), math.sin(odom_pose[2]), places=9
        )
        self.assertAlmostEqual(
            math.cos(recovered[2]), math.cos(odom_pose[2]), places=9
        )

        projected = odom_pose_to_map(recovered, transform)
        self.assertAlmostEqual(projected[0], map_pose[0], places=9)
        self.assertAlmostEqual(projected[1], map_pose[1], places=9)
        self.assertAlmostEqual(
            math.sin(projected[2]), math.sin(map_pose[2]), places=9
        )
        self.assertAlmostEqual(
            math.cos(projected[2]), math.cos(map_pose[2]), places=9
        )

    def test_scan_projection_uses_angle_metadata_and_lidar_offset(self):
        map_pose = (0.48, 0.72, 0.63)
        lidar_x, lidar_y = -0.033073, 0.012
        distance = 0.42
        sensor_angle = 0.50
        projected = []

        # The same physical ray is deliberately placed at different indices.
        # A fixed-index or assumed-zero-angle implementation cannot make these
        # two projections agree.
        for angle_min, angle_increment, count in (
            (-1.0, 0.10, 24),
            (-0.25, 0.05, 32),
        ):
            ranges = np.full(count, np.inf, dtype=np.float64)
            index = int(round((sensor_angle - angle_min) / angle_increment))
            ranges[index] = distance
            points = scan_points_in_map(
                ranges,
                angle_min,
                angle_increment,
                0.10,
                0.65,
                map_pose,
                lidar_x,
                lidar_y,
            )
            self.assertEqual(points.shape, (1, 2))
            projected.append(points[0])

        base_x = lidar_x + distance * math.cos(sensor_angle)
        base_y = lidar_y + distance * math.sin(sensor_angle)
        cosine = math.cos(map_pose[2])
        sine = math.sin(map_pose[2])
        expected = np.asarray(
            [
                map_pose[0] + cosine * base_x - sine * base_y,
                map_pose[1] + sine * base_x + cosine * base_y,
            ]
        )
        np.testing.assert_allclose(projected[0], expected, atol=1e-12)
        np.testing.assert_allclose(projected[1], expected, atol=1e-12)

    def test_choose_clear_space_accepts_only_one_clear_bay(self):
        self.assertEqual(choose_clear_space(0, 4, 4, 1), LEFT)
        self.assertEqual(choose_clear_space(4, 0, 4, 1), RIGHT)
        self.assertIsNone(choose_clear_space(0, 0, 4, 1))
        self.assertIsNone(choose_clear_space(4, 4, 4, 1))
        self.assertIsNone(choose_clear_space(2, 4, 4, 1))

    def test_rectangle_corners_preserve_asymmetric_front_and_rear(self):
        corners = rectangle_corners((1.0, 2.0, 0.0), 0.07, 0.12, 0.09)
        self.assertEqual(
            set(corners),
            {
                (0.88, 1.91),
                (0.88, 2.09),
                (1.07, 1.91),
                (1.07, 2.09),
            },
        )


class ParkingControllerTest(unittest.TestCase):
    def setUp(self):
        self.seconds = 10.0
        self.events = []
        self.publishers = {}
        self.lane_service = RecordingLaneService(self.events)
        self.param_overrides = {}

        patches = (
            mock.patch.object(
                controller_module.rospy.Time,
                "now",
                side_effect=lambda: controller_module.rospy.Time.from_sec(
                    self.seconds
                ),
            ),
            mock.patch.object(
                controller_module.rospy,
                "get_param",
                side_effect=lambda name, default=None: self.param_overrides.get(
                    name, default
                ),
            ),
            mock.patch.object(
                controller_module.rospy,
                "Publisher",
                side_effect=self._publisher_factory,
            ),
            mock.patch.object(
                controller_module.rospy,
                "Subscriber",
                return_value=SimpleNamespace(),
            ),
            mock.patch.object(
                controller_module.rospy,
                "ServiceProxy",
                return_value=self.lane_service,
            ),
            mock.patch.object(
                controller_module.rospy, "Timer", return_value=SimpleNamespace()
            ),
            mock.patch.object(controller_module.rospy, "on_shutdown"),
            mock.patch.object(controller_module.rospy, "wait_for_service"),
            mock.patch.object(controller_module.rospy, "loginfo"),
            mock.patch.object(controller_module.rospy, "loginfo_throttle"),
            mock.patch.object(controller_module.rospy, "logwarn_throttle"),
            mock.patch.object(controller_module.rospy, "logerr"),
            mock.patch.object(controller_module.rospy, "logfatal"),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _publisher_factory(self, topic, *_args, **_kwargs):
        publisher = RecordingPublisher(topic, self.events)
        self.publishers[topic] = publisher
        return publisher

    def now(self):
        return controller_module.rospy.Time.from_sec(self.seconds)

    def advance(self, seconds=0.10):
        self.seconds += seconds

    def make_controller(self):
        controller = ParkingMissionController()
        self.events.clear()
        self.lane_service.calls.clear()
        return controller

    @staticmethod
    def pose_message(pose):
        x, y, yaw = pose
        message = PoseStamped()
        message.pose.position.x = x
        message.pose.position.y = y
        message.pose.orientation.z = math.sin(0.5 * yaw)
        message.pose.orientation.w = math.cos(0.5 * yaw)
        return message

    @staticmethod
    def odom_message(pose, linear_x=0.0, linear_y=0.0, angular_z=0.0):
        x, y, yaw = pose
        message = Odometry()
        message.pose.pose.position.x = x
        message.pose.pose.position.y = y
        message.pose.pose.orientation.z = math.sin(0.5 * yaw)
        message.pose.pose.orientation.w = math.cos(0.5 * yaw)
        message.twist.twist.linear.x = linear_x
        message.twist.twist.linear.y = linear_y
        message.twist.twist.angular.z = angular_z
        return message

    def update_poses(self, controller, map_pose, odom_pose=None):
        if odom_pose is None:
            odom_pose = map_pose
        controller.map_pose_callback(self.pose_message(map_pose))
        controller.odom_callback(self.odom_message(odom_pose))

    def assert_pose_almost_equal(self, actual, expected):
        self.assertAlmostEqual(actual[0], expected[0], places=9)
        self.assertAlmostEqual(actual[1], expected[1], places=9)
        self.assertAlmostEqual(math.sin(actual[2]), math.sin(expected[2]), places=9)
        self.assertAlmostEqual(math.cos(actual[2]), math.cos(expected[2]), places=9)

    @staticmethod
    def rigid_route_transform(x=0.0, y=0.0, yaw=0.0):
        return RigidTransform2D(x, y, yaw, "odom", "map")

    @staticmethod
    def route_pose_to_odom(pose, transform):
        transformed = transform.inverse().apply_pose(Pose2D(*pose))
        return transformed.x, transformed.y, transformed.yaw

    @staticmethod
    def lane_path_message(valid=True, **overrides):
        diagnostics = {
            "progress": 0.25,
            "remaining_distance": 0.60,
            "position_error": 0.02,
            "cross_track_error": 0.01,
            "heading_error": 0.03,
            "target_speed": 0.10,
            "target_index": 7.0,
            "curvature": 0.20,
            "minimum_line_clearance": 0.04 if valid else 0.0,
            "minimum_obstacle_clearance": math.inf,
            "minimum_map_clearance": math.inf,
            "commanded_linear": 0.08,
            "commanded_angular": 0.01,
        }
        diagnostics.update(overrides)
        return Float64MultiArray(data=list(diagnostics.values()))

    @staticmethod
    def lane_command(controller, linear=0.06, angular=0.0):
        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
        controller.command_observer_callback(command)
        return command

    def start_controller(self, controller):
        controller.gate_callback(Bool(data=True))
        self.update_poses(controller, (0.90, 1.75, math.pi))
        self.lane_command(controller)
        self.events.clear()
        self.lane_service.calls.clear()
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.PREPARE_APPROACH)
        self.assertEqual(self.lane_service.calls, [])

        # Lane control keeps moving during PREPARE. The path is latched from a
        # post-gate moving pose before parking acquires cmd_vel.
        self.advance(controller.prepare_settle_time + 0.01)
        controller.odom_callback(self.odom_message((0.90, 1.75, math.pi)))
        self.lane_command(controller)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.APPROACH)
        self.assertEqual(self.lane_service.calls, [False])
        return controller

    def confirm_current_goal(self, controller):
        expected_state = controller.state
        if controller.state in (controller.APPROACH, controller.TURN_IN):
            self.advance()
            self.update_poses(controller, controller.goal_map, controller.goal_odom)
            controller.control_callback(None)
            self.assertNotEqual(controller.state, expected_state)
            return
        for _ in range(controller.pose_confirm_samples):
            self.advance()
            self.update_poses(controller, controller.goal_map, controller.goal_odom)
            controller.control_callback(None)
        for _ in range(100):
            if controller.state != expected_state:
                break
            self.advance(controller.control_period)
            self.update_poses(controller, controller.goal_map, controller.goal_odom)
            controller.control_callback(None)
        self.assertNotEqual(controller.state, expected_state)

    def confirm_rotation(self, controller):
        expected_state = controller.state
        self.assertIn(expected_state, controller.ROTATION_STATES)
        target_pose = (
            controller.rotation_center_odom.x,
            controller.rotation_center_odom.y,
            controller.rotation_target_yaw,
        )
        for _ in range(controller.pose_confirm_samples):
            self.advance(controller.control_period)
            controller.map_pose_callback(self.pose_message(controller.goal_map))
            controller.odom_callback(self.odom_message(target_pose))
            controller.control_callback(None)
        self.assertNotEqual(controller.state, expected_state)

    def confirm_space_selection(self, controller, selected_space):
        self.assertEqual(controller.state, controller.SELECT_SPACE)
        for _ in range(controller.selection_confirm_scans):
            self.advance(0.05)
            if selected_space == LEFT:
                controller.left_points = 0
                controller.right_points = controller.occupied_minimum_points
            else:
                controller.left_points = controller.occupied_minimum_points
                controller.right_points = 0
            controller.scan_generation += 1
            controller.scan_stamp = self.now()
            controller.scan_received = self.now()
            controller.control_callback(None)
        self.assertEqual(controller.selected_space, selected_space)

    def test_production_route_targets_fit_the_complete_footprint(self):
        controller = self.make_controller()

        # Constructor validation covers the production aisle and both parking
        # targets. Moving the left target onto its dotted portal must reject it.
        controller._validate_route_geometry()
        controller.left_park_x = controller.left_portal_edge + 0.01
        with self.assertRaises(controller_module.rospy.ROSInitException):
            controller._validate_route_geometry()

    def test_route_validation_rejects_invalid_rotation_stages(self):
        cases = (
            (
                "zigzag_turn_order",
                "zigzag_straight_y",
                0.84,
                "decision_y must precede the zigzag turn start",
            ),
            (
                "left_goal",
                "left_park_x",
                0.50,
                "left goal must be beyond its in-place turn centre",
            ),
            (
                "right_goal",
                "right_park_x",
                0.52,
                "right goal must be beyond its in-place turn centre",
            ),
            (
                "entry_anchor_lower",
                "entry_anchor_min_y",
                1.7430,
                "entry anchor y bounds must contain entry_y",
            ),
            (
                "entry_anchor_upper",
                "entry_anchor_max_y",
                1.7420,
                "entry anchor y bounds must contain entry_y",
            ),
        )
        for label, attribute, value, message in cases:
            with self.subTest(label=label):
                controller = self.make_controller()
                setattr(controller, attribute, value)
                with self.assertRaisesRegex(
                    controller_module.rospy.ROSInitException, message
                ):
                    controller._validate_route_geometry()

    def test_gazebo_config_uses_frozen_amcl_map_to_odom_alignment(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "parking_mission_gazebo.yaml"
        )
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)["parking"]
        self.assertFalse(config["route"]["odom_aligned"])
        self.assertEqual(
            config["topics"]["lane_path_diagnostics"],
            "/control/lane_path_diagnostics",
        )
        self.assertNotIn("lane_boundaries", config["topics"])
        self.assertAlmostEqual(
            float(config["route"]["entry_anchor_min_y"]), 1.7425
        )
        self.assertAlmostEqual(
            float(config["route"]["entry_anchor_max_y"]), 1.7580
        )
        self.assertAlmostEqual(
            float(config["control"]["entry_curve_end_heading_tolerance_deg"]),
            4.0,
        )
        self.assertAlmostEqual(
            float(config["paint"]["parking_opening_right_edge"]), 0.3831
        )
        self.assertAlmostEqual(
            float(config["paint"]["parking_opening_left_edge"]), 0.6129
        )
        self.assertAlmostEqual(
            float(config["paint"]["parking_opening_upper_edge"]), 1.0001
        )
        self.assertAlmostEqual(
            float(config["rejoin"]["complete_min_x"]), 0.08
        )
        self.assertAlmostEqual(
            float(config["rejoin"]["complete_heading_tolerance_deg"]),
            6.0,
        )

    def test_run10_entry_terminal_continues_into_the_g2_turn(self):
        controller = self.make_controller()
        controller.route_transform = self.rigid_route_transform()
        controller.odom_pose = (0.95, 1.75, math.pi)
        controller.observed_lane_linear = controller.approach_speed

        self.assertTrue(controller._build_adaptive_entry())
        controller.state = controller.APPROACH
        self.assertTrue(
            controller._activate_common_path(
                controller.entry_connector_path,
                controller.APPROACH,
                initial_linear=controller.approach_speed,
                initial_angular=0.0,
            )
        )

        path = controller.entry_connector_path
        terminal_tangent = np.asarray(
            [math.cos(path.heading[-1]), math.sin(path.heading[-1])]
        )
        # Reproduce run 10's first terminal sample: position was only 2.6 mm
        # beyond the join, while plant steering lag left 3.62 deg of error.
        pose = (
            float(path.x[-1] + 0.0026 * terminal_tangent[0]),
            float(path.y[-1] + 0.0026 * terminal_tangent[1]),
            float(path.heading[-1] + math.radians(3.62)),
        )
        controller.odom_pose = pose
        tracking = controller.path_follower.calculate_tracking(pose)

        self.assertEqual(
            controller._continuous_path_completion_status(tracking),
            "READY",
        )

    def test_run11_return_bias_clears_the_finite_parking_opening(self):
        self.param_overrides.update(
            {
                "~parking/safety/localization_margin": 0.001,
                "~parking/safety/tracking_margin": 0.001,
            }
        )
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.selected_space = RIGHT
        controller.route_transform = self.rigid_route_transform()

        # Frozen-route pose reconstructed from official-start run 11. The old
        # infinite-corner approximation reported -0.752 mm even though the
        # native solid paint remained 8.57 mm outside the fully expanded body.
        start = (
            0.49203417869850774,
            0.6938828136475323,
            math.radians(90.1776092917495),
        )
        self.update_poses(controller, start, start)

        self.assertTrue(controller._begin_leave_aisle())
        self.assertEqual(controller.state, controller.LEAVE_AISLE)
        self.assertGreater(controller.active_path.line_clearance, 0.008)
        self.assertAlmostEqual(controller.goal_map[0], 0.4960, places=12)
        self.assertAlmostEqual(
            controller.zigzag_exit_curve_map[-1, 0], 0.2260, places=12
        )

    def test_finite_parking_opening_still_rejects_a_real_solid_arm_crossing(self):
        self.param_overrides.update(
            {
                "~parking/safety/localization_margin": 0.001,
                "~parking/safety/tracking_margin": 0.001,
            }
        )
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.selected_space = RIGHT
        controller.route_transform = self.rigid_route_transform()
        unsafe_start = (
            0.4700,
            0.6938828136475323,
            math.radians(90.1776092917495),
        )
        self.update_poses(controller, unsafe_start, unsafe_start)

        self.assertIsNone(controller._begin_leave_aisle())
        self.assertEqual(controller.state, controller.FAILED)

    def test_lane_handoff_consumes_common_path_diagnostics(self):
        controller = self.make_controller()
        controller.route_transform = self.rigid_route_transform()
        handoff_pose = (
            0.5 * (controller.handoff_min_x + controller.handoff_max_x),
            0.5 * (controller.handoff_min_y + controller.handoff_max_y),
            controller.zigzag_heading,
        )
        self.update_poses(controller, handoff_pose)
        message = self.lane_path_message(
            progress=0.35,
            remaining_distance=0.55,
            position_error=0.015,
            cross_track_error=-0.012,
            heading_error=0.025,
            target_speed=0.11,
            target_index=8.0,
            curvature=-0.30,
            minimum_line_clearance=0.035,
            commanded_linear=0.09,
            commanded_angular=-0.02,
        )
        controller.lane_path_diagnostics_callback(message)

        self.assertTrue(controller.lane_path_valid)
        self.assertTrue(
            controller._lane_observation_valid(self.now(), joining=False)
        )
        expected = message.data[: controller.LANE_PATH_DIAGNOSTIC_FIELDS]
        actual = [
            controller.lane_path_progress,
            controller.lane_path_remaining_distance,
            controller.lane_path_position_error,
            controller.lane_path_cross_track_error,
            controller.lane_path_heading_error,
            controller.lane_path_target_speed,
            controller.lane_path_target_index,
            controller.lane_path_curvature,
            controller.lane_path_minimum_line_clearance,
            controller.lane_path_minimum_obstacle_clearance,
            controller.lane_path_minimum_map_clearance,
            controller.lane_path_commanded_linear,
            controller.lane_path_commanded_angular,
        ]
        np.testing.assert_equal(actual, expected)
        summary = controller._lane_observation_summary()
        self.assertIn("progress=0.350", summary)
        self.assertIn("remaining=0.550", summary)
        self.assertIn("position=0.0150", summary)
        self.assertIn("min_line=0.0350", summary)
        self.assertNotIn("samples=", summary)
        self.assertNotIn("confidence=", summary)

        controller.lane_path_minimum_line_clearance = 0.0
        self.assertFalse(
            controller._lane_observation_valid(self.now(), joining=False)
        )

    def test_lane_handoff_rejects_invalid_common_path_diagnostics(self):
        valid_values = list(self.lane_path_message().data)
        cases = [
            ("short", valid_values[:12]),
            ("extra", valid_values + [0.0]),
            ("progress below zero", {0: -0.01}),
            ("progress above one", {0: 1.01}),
            ("negative remaining", {1: -0.01}),
            ("negative position error", {2: -0.01}),
            ("nonfinite cross track", {3: math.nan}),
            ("nonfinite heading", {4: math.inf}),
            ("unnormalized heading", {4: math.pi + 0.01}),
            ("negative target speed", {5: -0.01}),
            ("negative target index", {6: -1.0}),
            ("fractional target index", {6: 1.5}),
            ("nonfinite curvature", {7: math.nan}),
            ("zero line clearance", {8: 0.0}),
            ("infinite line clearance", {8: math.inf}),
            ("nan obstacle clearance", {9: math.nan}),
            ("negative infinite obstacle clearance", {9: -math.inf}),
            ("nan map clearance", {10: math.nan}),
            ("negative infinite map clearance", {10: -math.inf}),
            ("nonfinite linear command", {11: math.inf}),
            ("nonfinite angular command", {12: math.nan}),
        ]
        for label, mutation in cases:
            with self.subTest(label=label):
                controller = self.make_controller()
                if isinstance(mutation, list):
                    values = mutation
                else:
                    values = list(valid_values)
                    for index, value in mutation.items():
                        values[index] = value
                controller.lane_path_diagnostics_callback(
                    Float64MultiArray(data=values)
                )
                self.assertFalse(controller.lane_path_valid)
                self.assertIsNone(controller.last_lane_path_time)

    def test_cached_dense_footprint_is_geometrically_identical(self):
        controller = self.make_controller()
        pose = Pose2D(0.43, 1.27, math.radians(37.0))
        footprint = controller.common_footprint.expanded(0.011)
        expected = footprint_points(
            pose, footprint, spacing=0.001, perimeter_only=True
        )
        actual = controller._cached_perimeter_points(pose, footprint)
        np.testing.assert_allclose(actual, expected, atol=1e-15)

    def test_redesigned_turn_in_completes_with_positive_stopping_clearance(self):
        """Regress the TURN_IN condition that stopped runs 7fd180/b47481."""
        self.param_overrides.update(
            {
                "~parking/safety/localization_margin": 0.001,
                "~parking/safety/tracking_margin": 0.001,
                "~parking/control/arc_angular_scale": 1.10,
                "~parking/control/linear_deceleration": 0.03,
            }
        )
        controller = self.make_controller()
        controller.route_transform = self.rigid_route_transform()
        curve_start = (
            controller.aisle_x + controller.entry_curve_offset,
            controller.entry_y,
            controller.approach_heading,
        )
        poses = quintic_turn_path(
            curve_start,
            controller.aisle_heading,
            controller.entry_curve_offset,
            controller.entry_curve_tangent,
            controller.entry_curve_samples,
        )
        path = controller._common_path_from_poses(
            poses,
            1,
            controller.entry_turn_speed,
            controller.TURN_IN,
            "entry_turn",
            controller.entry_curve_end_position_tolerance,
            controller.entry_curve_end_heading_tolerance,
            controller.entry_curve_end_crossing_max_distance,
            feedforward_scale=controller.arc_angular_scale,
        )
        self.assertIsNotNone(path)
        controller.state = controller.TURN_IN
        pose = Pose2D(path.x[0], path.y[0], path.heading[0])
        controller.odom_pose = (pose.x, pose.y, pose.yaw)
        self.assertTrue(
            controller._activate_common_path(
                path,
                controller.TURN_IN,
                initial_linear=controller.entry_turn_speed,
                initial_angular=0.0,
            )
        )
        minimum_clearance = math.inf
        completed = False
        elapsed = controller.control_period
        for _ in range(300):
            tracking = controller.path_follower.calculate_tracking(pose)
            decision = controller.path_validator.motion_safety(
                path,
                pose,
                tracking.path_index,
                tracking.target_speed,
                controller.path_follower.last_linear,
                (
                    controller.path_follower.last_angular,
                    tracking.angular_velocity,
                ),
                controller.safety_reaction_time,
                controller.linear_deceleration,
                distance_margin=controller.safety_distance_margin,
                live_obstacles=np.empty((0, 2), dtype=np.float64),
                tracking=tracking,
            )
            minimum_clearance = min(
                minimum_clearance,
                decision.validation.minimum_line_clearance,
            )
            self.assertGreater(decision.speed_limit, 0.0)
            command, _ = controller.path_follower.command(
                pose,
                elapsed,
                speed_limit=decision.speed_limit,
                tracking=tracking,
            )

            # The measured Gazebo ground-path radius is 1.10 times the
            # differential-drive command radius.  This is the same plant
            # correction represented by feedforward_scale on the path.
            linear = command.linear_velocity
            angular = command.angular_velocity / controller.arc_angular_scale
            next_yaw = normalize_angle(pose.yaw + angular * elapsed)
            if abs(angular) <= 1e-9:
                pose = Pose2D(
                    pose.x + linear * elapsed * math.cos(pose.yaw),
                    pose.y + linear * elapsed * math.sin(pose.yaw),
                    next_yaw,
                )
            else:
                radius = linear / angular
                pose = Pose2D(
                    pose.x
                    + radius * (math.sin(next_yaw) - math.sin(pose.yaw)),
                    pose.y
                    - radius * (math.cos(next_yaw) - math.cos(pose.yaw)),
                    next_yaw,
                )
            if controller.path_follower.goal_status(pose).complete:
                completed = True
                break

        self.assertTrue(completed)
        self.assertGreater(minimum_clearance, 0.004)

    def test_common_path_contains_frozen_fixed_sign_obstacles(self):
        controller = self.make_controller()
        controller.route_transform = self.rigid_route_transform(
            0.31, -0.27, math.radians(2.0)
        )
        curve_start = (
            controller.aisle_x + controller.entry_curve_offset,
            controller.entry_y,
            controller.approach_heading,
        )
        route_poses = quintic_turn_path(
            curve_start,
            controller.aisle_heading,
            controller.entry_curve_offset,
            controller.entry_curve_tangent,
            controller.entry_curve_samples,
        )
        odom_poses = np.asarray(
            [controller._route_pose_to_odom(pose) for pose in route_poses]
        )
        path = controller._common_path_from_poses(
            odom_poses,
            1,
            controller.entry_turn_speed,
            controller.TURN_IN,
            "entry_turn",
            controller.entry_curve_end_position_tolerance,
            controller.entry_curve_end_heading_tolerance,
            controller.entry_curve_end_crossing_max_distance,
            feedforward_scale=controller.arc_angular_scale,
        )

        self.assertIsNotNone(path)
        self.assertGreater(path.safety.fixed_obstacles.shape[0], 400)
        expected = controller.route_transform.inverse().apply_point(
            controller.fixed_obstacle_points_route[0]
        )
        np.testing.assert_allclose(
            path.safety.fixed_obstacles[0], expected, atol=1e-12
        )
        self.assertGreater(path.obstacle_clearance, 0.02)


    def test_common_validator_rejects_a_path_through_fixed_sign(self):
        controller = self.make_controller()
        controller.route_transform = self.rigid_route_transform()
        path = controller._common_path_from_poses(
            ((0.40, 1.90, 0.0), (0.60, 1.90, 0.0)),
            1,
            controller.approach_speed,
            "fixed_sign_collision",
            None,
            controller.position_tolerance,
            controller.heading_tolerance,
            controller.overshoot_tolerance,
        )

        self.assertIsNone(path)
        self.assertEqual(controller.state, controller.FAILED)


    def test_scan_callback_scores_both_bays_from_map_pose_and_metadata(self):
        controller = self.make_controller()
        controller.scan_median_window = 1
        controller.state = controller.SELECT_SPACE
        controller.state_started = self.now()
        decision_pose = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        controller.map_pose_callback(self.pose_message(decision_pose))

        left_points = [
            (0.66, 0.74),
            (0.69, 0.78),
            (0.73, 0.84),
            (0.79, 0.88),
        ]
        controller.scan_callback(
            self.scan_message_for_map_points(
                decision_pose, controller, left_points, sample_count=721
            )
        )
        self.assertEqual(controller.left_points, 4)
        self.assertEqual(controller.right_points, 0)

        right_points = [
            (0.17, 0.74),
            (0.20, 0.78),
            (0.25, 0.84),
            (0.30, 0.88),
        ]
        controller.scan_callback(
            self.scan_message_for_map_points(
                decision_pose, controller, right_points, sample_count=1081
            )
        )
        self.assertEqual(controller.left_points, 0)
        self.assertEqual(controller.right_points, 4)

        generation = controller.scan_generation
        self.advance(controller.pose_timeout + 0.01)
        controller.scan_callback(
            self.scan_message_for_map_points(
                decision_pose, controller, left_points, sample_count=541
            )
        )
        self.assertEqual(controller.scan_generation, generation)

    def test_select_space_retries_latest_scan_after_main_lock_collision(self):
        controller = self.make_controller()
        controller.scan_median_window = 1
        decision_pose = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.update_poses(controller, decision_pose, decision_pose)
        left_points = [
            (0.66, 0.74),
            (0.69, 0.78),
            (0.73, 0.84),
            (0.79, 0.88),
        ]

        # A scan whose callback began before the stopped SELECT_SPACE epoch
        # may be retained for safety, but must not count as bay evidence.
        with controller.lock:
            self.advance(0.05)
            old_scan = self.scan_message_for_map_points(
                decision_pose, controller, left_points
            )
            old_worker = threading.Thread(
                target=controller.scan_callback, args=(old_scan,)
            )
            old_worker.start()
            old_worker.join(1.0)
            self.assertFalse(old_worker.is_alive())
            self.assertIsNotNone(controller.pending_selection_scan)
            self.advance(0.001)
            controller._set_state(controller.SELECT_SPACE)
        controller.control_callback(None)
        self.assertEqual(controller.scan_generation, 0)

        # Reproduce a persistent phase collision while SELECT_SPACE is active.
        # The subscriber returns immediately and the next control tick consumes
        # the distinct latest scan rather than losing it.
        with controller.lock:
            self.advance(0.05)
            new_scan = self.scan_message_for_map_points(
                decision_pose, controller, left_points
            )
            worker = threading.Thread(
                target=controller.scan_callback, args=(new_scan,)
            )
            worker.start()
            worker.join(1.0)
            self.assertFalse(worker.is_alive())
            self.assertIsNotNone(controller.pending_selection_scan)
            self.assertEqual(controller.scan_generation, 0)
        controller.control_callback(None)

        self.assertEqual(controller.scan_generation, 1)
        self.assertEqual(controller.left_points, 4)
        self.assertEqual(controller.right_points, 0)

    def test_safety_scan_uses_bracketing_odom_at_acquisition_stamp(self):
        controller = self.make_controller()
        controller.scan_median_window = 1

        first = self.odom_message((0.0, 0.0, 0.0))
        first.header.stamp = controller_module.rospy.Time.from_sec(10.00)
        controller.odom_callback(first)
        self.seconds = 10.08
        second = self.odom_message((0.08, 0.0, 0.0))
        second.header.stamp = controller_module.rospy.Time.from_sec(10.08)
        controller.odom_callback(second)

        scan = LaserScan()
        scan.header.stamp = controller_module.rospy.Time.from_sec(10.04)
        scan.angle_min = 0.0
        scan.angle_increment = 0.01
        scan.range_min = 0.10
        scan.range_max = 10.0
        scan.ranges = [0.40]
        controller.scan_callback(scan)

        self.assertEqual(controller.live_obstacle_points_odom.shape, (1, 2))
        self.assertAlmostEqual(
            controller.live_obstacle_points_odom[0, 0],
            0.04 + controller.lidar_x + 0.40,
            places=8,
        )
        self.assertAlmostEqual(
            controller.live_obstacle_points_odom[0, 1], 0.0, places=9
        )
        self.assertAlmostEqual(controller.safety_scan_stamp.to_sec(), 10.04)

        # Reverse the callback order: a scan newer than the latest odometry is
        # held until the next odom sample supplies its right-hand bracket.
        unbracketed = copy.deepcopy(scan)
        unbracketed.header.stamp = controller_module.rospy.Time.from_sec(10.09)
        controller.scan_callback(unbracketed)
        self.assertAlmostEqual(controller.safety_scan_stamp.to_sec(), 10.04)
        self.assertEqual(len(controller.pending_safety_scans), 1)

        self.seconds = 10.12
        third = self.odom_message((0.12, 0.0, 0.0))
        third.header.stamp = controller_module.rospy.Time.from_sec(10.12)
        controller.odom_callback(third)
        self.assertEqual(len(controller.pending_safety_scans), 0)
        self.assertAlmostEqual(controller.safety_scan_stamp.to_sec(), 10.09)
        self.assertAlmostEqual(
            controller.live_obstacle_points_odom[0, 0],
            0.09 + controller.lidar_x + 0.40,
            places=8,
        )

    def test_pending_safety_scans_are_bounded_and_discard_invalid_timing(self):
        controller = self.make_controller()
        controller.scan_median_window = 1
        initial = self.odom_message((0.0, 0.0, 0.0))
        initial.header.stamp = controller_module.rospy.Time.from_sec(10.00)
        controller.odom_callback(initial)

        def scan_at(stamp):
            message = LaserScan()
            message.header.stamp = controller_module.rospy.Time.from_sec(stamp)
            message.angle_min = 0.0
            message.angle_increment = 0.01
            message.range_min = 0.10
            message.range_max = 10.0
            message.ranges = [0.40]
            return message

        for index in range(controller.pending_safety_scan_limit + 1):
            controller.scan_callback(scan_at(10.01 + 0.01 * index))
        self.assertEqual(
            len(controller.pending_safety_scans),
            controller.pending_safety_scan_limit,
        )
        self.assertAlmostEqual(
            controller.pending_safety_scans[0]["source_stamp"].to_sec(),
            10.02,
        )

        # Once odom has passed these scans without an admissibly close left
        # and right sample, none may be projected with the newer pose.
        self.seconds = 10.20
        distant = self.odom_message((0.20, 0.0, 0.0))
        distant.header.stamp = controller_module.rospy.Time.from_sec(10.20)
        controller.odom_callback(distant)
        self.assertEqual(len(controller.pending_safety_scans), 0)
        self.assertIsNone(controller.safety_scan_stamp)

        # A scan that waits beyond the configured freshness timeout is also
        # discarded even when a later callback would otherwise bracket it.
        self.seconds = 10.21
        controller.scan_callback(scan_at(10.21))
        self.assertEqual(len(controller.pending_safety_scans), 1)
        self.seconds += controller.scan_timeout + 0.01
        late = self.odom_message((0.24, 0.0, 0.0))
        late.header.stamp = controller_module.rospy.Time.from_sec(10.24)
        controller.odom_callback(late)
        self.assertEqual(len(controller.pending_safety_scans), 0)
        self.assertIsNone(controller.safety_scan_stamp)

    @staticmethod
    def scan_message_for_map_points(
        map_pose, controller, map_points, sample_count=721
    ):
        message = LaserScan()
        message.angle_min = -math.pi
        message.angle_increment = 2.0 * math.pi / (sample_count - 1)
        message.range_min = 0.10
        message.range_max = 10.0
        ranges = np.full(sample_count, np.inf, dtype=np.float64)
        robot_x, robot_y, robot_yaw = map_pose
        cosine = math.cos(robot_yaw)
        sine = math.sin(robot_yaw)
        for map_x, map_y in map_points:
            dx = map_x - robot_x
            dy = map_y - robot_y
            base_x = cosine * dx + sine * dy
            base_y = -sine * dx + cosine * dy
            sensor_x = base_x - controller.lidar_x
            sensor_y = base_y - controller.lidar_y
            angle = math.atan2(sensor_y, sensor_x)
            distance = math.hypot(sensor_x, sensor_y)
            index = int(
                round((angle - message.angle_min) / message.angle_increment)
            )
            ranges[index] = distance
        message.ranges = ranges.tolist()
        return message

    def test_start_requires_fresh_amcl_but_prepare_keeps_lane_control(self):
        controller = self.make_controller()
        controller.gate_callback(Bool(data=True))

        controller.control_callback(None)
        self.assertEqual(controller.state, controller.WAIT_GATE)
        self.assertEqual(self.lane_service.calls, [])

        delayed_map = self.pose_message((0.90, 1.75, math.pi))
        delayed_map.header.stamp = controller_module.rospy.Time.from_sec(
            self.seconds - controller.pose_timeout - 0.01
        )
        controller.map_pose_callback(delayed_map)
        controller.odom_callback(self.odom_message((0.90, 1.75, math.pi)))
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.WAIT_GATE)
        self.assertEqual(self.lane_service.calls, [])

        self.update_poses(controller, (0.90, 1.75, math.pi))
        self.events.clear()
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.PREPARE_APPROACH)
        self.assertEqual(self.lane_service.calls, [])
        self.assertIsNone(controller.route_transform)
        self.assertIsNone(controller.goal_odom)
        self.assertEqual(
            self.publishers[controller.cmd_vel_topic].messages, []
        )

        # Parking remains silent even after settling until a fresh, capped lane
        # command newer than the gate has been observed.
        self.advance(controller.prepare_settle_time + 0.01)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.PREPARE_APPROACH)
        self.assertIsNone(controller.route_transform)
        self.assertEqual(self.lane_service.calls, [])
        self.assertEqual(
            self.publishers[controller.cmd_vel_topic].messages, []
        )

        self.lane_command(controller, 0.076, 0.015)
        controller.odom_callback(self.odom_message((0.90, 1.75, math.pi)))
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.APPROACH)
        self.assertIsNotNone(controller.route_transform)
        self.assertIsNotNone(controller.goal_odom)
        self.assertEqual(self.lane_service.calls, [False])
        service_index = next(
            index
            for index, event in enumerate(self.events)
            if event[0] == "service" and event[1] is False
        )
        command_index = next(
            index
            for index, event in enumerate(self.events)
            if event[0] == "publish" and event[1] == "/cmd_vel"
        )
        self.assertLess(service_index, command_index)

    def test_amcl_snapshot_aligns_entry_with_observed_odom_offset(self):
        self.param_overrides.update(
            {
                "~parking/route/odom_aligned": False,
                "~parking/control/arc_angular_scale": 1.10,
                "~parking/control/linear_acceleration": 0.03,
                "~parking/control/angular_acceleration": 0.55,
            }
        )
        controller = self.make_controller()
        map_pose = (
            1.0082,
            1.7617,
            math.pi,
        )
        odom_pose = (1.0300, 1.7400, math.radians(179.5))

        controller.gate_callback(Bool(data=True))
        self.update_poses(controller, map_pose, odom_pose)
        self.lane_command(controller)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.PREPARE_APPROACH)
        self.advance(controller.prepare_settle_time + 0.01)
        controller.odom_callback(self.odom_message(odom_pose))
        controller.control_callback(None)

        expected_join = (
            controller.aisle_x + controller.entry_curve_offset,
            controller.entry_y,
            controller.approach_heading,
        )
        anchored_join = self.route_pose_to_odom(
            expected_join, controller.route_transform
        )
        self.assertEqual(controller.state, controller.APPROACH)
        self.assert_pose_almost_equal(controller.entry_connector_odom[0], odom_pose)
        self.assert_pose_almost_equal(
            controller.entry_connector_odom[-1], anchored_join
        )
        self.assert_pose_almost_equal(controller.goal_odom, anchored_join)
        self.assertGreater(
            math.hypot(
                anchored_join[0] - expected_join[0],
                anchored_join[1] - expected_join[1],
            ),
            0.01,
        )
        self.assertGreaterEqual(
            controller.entry_connector_path.line_clearance
            + controller.entry_handoff_minimum_clearance,
            controller.line_margin,
        )
        self.assertAlmostEqual(
            controller.entry_connector_speed, controller.approach_speed
        )

    def test_north_amcl_outlier_is_corrected_without_an_entry_stop(self):
        self.param_overrides.update(
            {
                "~parking/route/odom_aligned": False,
                "~parking/control/entry_handoff_minimum_clearance": 0.003,
            }
        )
        controller = self.make_controller()
        validations = []
        original_validate = controller.path_validator.validate_path

        def record_validation(*args, **kwargs):
            result = original_validate(*args, **kwargs)
            validations.append(result)
            return result

        controller.path_validator.validate_path = mock.Mock(
            side_effect=record_validation
        )
        # Reproduce the synchronized AMCL/odom pair and latest moving odom
        # pose from official-start run 4DEAyD.  The uncorrected AMCL y made
        # the connector fail at line=-8.2 mm before parking took control.
        map_pose = (
            0.9764983254,
            1.7769530084,
            math.radians(-177.1324063),
        )
        synchronized_odom = (
            1.0051117724,
            1.7613431530,
            math.radians(-177.5382103),
        )
        latest_odom = (0.9983, 1.7611, math.radians(-177.6))

        controller.gate_callback(Bool(data=True))
        self.update_poses(controller, map_pose, synchronized_odom)
        self.lane_command(controller)
        controller.control_callback(None)
        self.advance(controller.prepare_settle_time + 0.01)
        controller.odom_callback(self.odom_message(latest_odom))
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.APPROACH)
        self.assertAlmostEqual(
            controller.route_anchor_map_pose[1],
            controller.entry_anchor_max_y,
        )
        self.assertLess(controller.route_anchor_lateral_correction, -0.018)
        self.assertTrue(validations)
        self.assertTrue(all(result.safe for result in validations))
        self.assertGreater(controller.entry_connector_path.line_clearance, 0.0)
        commands = self.publishers[controller.cmd_vel_topic].messages
        self.assertTrue(commands)
        self.assertGreater(commands[-1].linear.x, 0.0)
        self.assertFalse(
            any(
                abs(command.linear.x) <= 1e-9
                and abs(command.angular.z) <= 1e-9
                for command in commands
            )
        )

    def test_common_validator_accepts_a_safe_low_margin_handoff(self):
        self.param_overrides.update(
            {
                "~parking/route/odom_aligned": False,
                "~parking/control/entry_handoff_minimum_clearance": 0.003,
            }
        )
        controller = self.make_controller()
        map_pose = (0.9650, 1.7428, math.radians(176.1))
        odom_pose = (0.9573, 1.7446, math.radians(175.9))

        controller.gate_callback(Bool(data=True))
        self.update_poses(controller, map_pose, odom_pose)
        self.lane_command(controller)
        controller.control_callback(None)
        self.advance(controller.prepare_settle_time + 0.01)
        controller.odom_callback(self.odom_message(odom_pose))
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.APPROACH)
        validated_clearance = (
            controller.entry_connector_path.line_clearance
            + controller.entry_handoff_minimum_clearance
        )
        self.assertLess(validated_clearance, controller.line_margin)
        self.assertGreaterEqual(
            validated_clearance,
            controller.entry_handoff_minimum_clearance,
        )
        self.assertFalse(hasattr(controller, "_entry_pose_clearance"))
        self.assertFalse(hasattr(controller, "_entry_corridor_clearance"))

    def test_common_validator_stops_an_unsafe_live_entry_pose(self):
        controller = self.make_controller()
        controller.route_transform = self.rigid_route_transform()
        controller.odom_pose = (0.95, 1.75, math.pi)
        controller.observed_lane_linear = 0.06

        self.assertTrue(controller._build_adaptive_entry())
        controller.state = controller.APPROACH
        self.assertTrue(
            controller._activate_common_path(
                controller.entry_connector_path,
                controller.APPROACH,
                initial_linear=0.0,
                initial_angular=0.0,
            )
        )

        path = controller.entry_connector_path
        unsafe_x = float(path.x[0])
        unsafe_y = (
            controller.entry_lower_intercept
            + controller.entry_boundary_slope * unsafe_x
            + controller.half_width
            + controller.entry_handoff_minimum_clearance
            - 0.001
        )
        controller.odom_pose = (unsafe_x, unsafe_y, float(path.heading[0]))
        controller.last_command_time = self.now()
        decisions = []
        original_motion_safety = controller.path_validator.motion_safety

        def record_decision(*args, **kwargs):
            result = original_motion_safety(*args, **kwargs)
            decisions.append(result)
            return result

        controller.path_validator.motion_safety = mock.Mock(
            side_effect=record_decision
        )
        self.advance(controller.control_period)

        command = controller._common_path_command(self.now())

        self.assertEqual(controller.path_validator.motion_safety.call_count, 1)
        self.assertTrue(decisions[-1].requires_stop)
        self.assertFalse(decisions[-1].stopping.safe)
        self.assertEqual(command.linear.x, 0.0)
        self.assertEqual(command.angular.z, 0.0)
        self.assertEqual(controller.state, controller.APPROACH)

    def test_adaptive_entry_scales_a_shorter_amcl_aligned_connector(self):
        self.param_overrides.update(
            {
                "~parking/route/odom_aligned": False,
                "~parking/control/entry_handoff_minimum_clearance": 0.003,
                "~parking/control/arc_angular_scale": 1.10,
                "~parking/control/linear_acceleration": 0.03,
                "~parking/control/angular_acceleration": 0.55,
            }
        )
        controller = self.make_controller()
        map_pose = (0.9631, 1.7500, math.pi)
        odom_pose = (0.981398, 1.761707, math.radians(179.2))

        controller.gate_callback(Bool(data=True))
        self.update_poses(controller, map_pose, odom_pose)
        self.lane_command(controller)
        controller.control_callback(None)
        self.advance(controller.prepare_settle_time + 0.01)
        controller.odom_callback(self.odom_message(odom_pose))
        controller.control_callback(None)

        connector_length = controller.entry_connector_path.length
        self.assertEqual(controller.state, controller.APPROACH)
        self.assertLess(connector_length, 0.30)
        self.assertGreaterEqual(
            controller.entry_connector_path.line_clearance
            + controller.entry_handoff_minimum_clearance,
            controller.line_margin,
        )
        self.assertAlmostEqual(
            controller.entry_connector_speed, controller.approach_speed
        )




    def test_zigzag_exit_profile_obeys_all_motion_limits(self):
        self.param_overrides.update(
            {
                "~parking/control/arc_angular_scale": 1.10,
                "~parking/control/linear_acceleration": 0.03,
                "~parking/control/linear_deceleration": 0.03,
                "~parking/control/angular_acceleration": 0.55,
            }
        )

        controller = self.make_controller()
        controller.route_transform = self.rigid_route_transform()
        controller.odom_pose = controller._zigzag_turn_start_pose()
        self.assertTrue(controller._build_zigzag_exit_curve())
        path = controller.zigzag_exit_curve_path
        effective_curvature = path.curvature * path.feedforward_scale
        effective_angular = path.speed * effective_curvature
        self.assertLessEqual(
            float(np.max(np.abs(effective_angular))),
            controller.maximum_angular_velocity + 1e-9,
        )
        self.assertLessEqual(
            float(np.max(path.speed ** 2 * np.abs(effective_curvature))),
            controller.maximum_lateral_acceleration + 1e-9,
        )
        distance = np.diff(path.station)
        duration = 2.0 * distance / (path.speed[:-1] + path.speed[1:])
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(effective_angular)) / duration)),
            controller.angular_acceleration + 2e-6,
        )
        self.assertAlmostEqual(
            path.speed[-1], controller.profile_exit_velocity, places=12
        )
        self.assertLess(path.speed[-1], path.speed[0])





    def test_odom_callback_preserves_actual_planar_speed_for_safety(self):
        controller = self.make_controller()
        controller.state = controller.TURN_TO_EXIT

        reverse = self.odom_message((0.0, 0.0, 0.0))
        reverse.header.stamp = controller_module.rospy.Time.from_sec(4.2)
        # A 3-4-5 vector also covers Gazebo world-frame odometry, where the
        # planar velocity is not necessarily stored only in linear.x.
        reverse.twist.twist.linear.x = -0.021
        reverse.twist.twist.linear.y = 0.028
        reverse.twist.twist.angular.z = -0.17
        controller.odom_callback(reverse)
        self.assertAlmostEqual(controller.odom_linear_speed, 0.035, places=9)
        self.assertAlmostEqual(controller.odom_angular_velocity, -0.17, places=9)

        controller.state = controller.PARK_IN
        straight = self.odom_message((0.0, 0.0, 0.0))
        straight.header.stamp = controller_module.rospy.Time.from_sec(4.3)
        straight.twist.twist.linear.x = 0.012
        straight.twist.twist.angular.z = 0.03
        controller.odom_callback(straight)
        self.assertAlmostEqual(controller.odom_linear_speed, 0.012, places=9)
        self.assertAlmostEqual(controller.odom_angular_velocity, 0.03, places=9)

    def test_live_obstacle_sweep_limits_speed_and_reports_diagnostics(self):
        self.param_overrides["~parking/control/linear_deceleration"] = 0.03
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.route_transform = self.rigid_route_transform()
        start = (controller.aisle_x, 1.50, controller.aisle_heading)
        pose = (controller.aisle_x, 1.25, controller.aisle_heading)
        goal = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.update_poses(controller, start, start)
        self.assertTrue(
            controller._begin_drive(
                controller.ENTER_AISLE, goal, 1, controller.aisle_speed
            )
        )
        self.update_poses(controller, pose, pose)
        controller.path_follower.last_linear = 0.02
        controller.last_linear = 0.02
        controller.odom_linear_speed = 0.02

        # A collision farther down the nominal route is outside the current
        # complete-stop sweep, so it supplies a justified nonzero speed cap.
        controller.live_obstacle_points_odom = np.asarray(
            [[pose[0], pose[1] - 0.13]], dtype=np.float64
        )
        controller.safety_scan_stamp = self.now()
        controller.safety_scan_received = self.now()
        speed_limit, emergency = controller._common_safety_speed_limit(
            self.now()
        )
        self.assertFalse(emergency)
        self.assertGreater(speed_limit, 0.0)
        self.assertLess(speed_limit, controller.aisle_speed)

        # Once the obstacle enters the reaction-plus-braking footprint, the
        # same validator asks for zero target speed. The follower must still
        # apply configured deceleration instead of publishing an abrupt zero.
        controller.live_obstacle_points_odom = np.asarray(
            [[pose[0], pose[1] - 0.09]], dtype=np.float64
        )
        controller.path_follower.last_linear = 0.04
        controller.path_follower.last_angular = 0.10
        controller.last_linear = 0.04
        controller.last_angular = 0.10
        controller.odom_linear_speed = 0.04
        controller.odom_angular_velocity = 0.10
        controller.last_command_time = self.now()
        self.advance(controller.control_period)
        controller.odom_received = self.now()
        controller.odom_stamp = self.now()
        controller.safety_scan_stamp = self.now()
        controller.safety_scan_received = self.now()
        command = controller._common_path_command(self.now())

        self.assertGreater(command.linear.x, 0.0)
        self.assertAlmostEqual(
            command.linear.x,
            0.04
            - controller.linear_deceleration * controller.control_period,
            places=9,
        )
        diagnostics = self.publishers[controller.diagnostics_topic].messages[-1]
        self.assertEqual(len(diagnostics.data), 13)
        self.assertLessEqual(diagnostics.data[9], 0.0)
        self.assertAlmostEqual(diagnostics.data[11], command.linear.x)
        self.assertAlmostEqual(diagnostics.data[12], command.angular.z)

    def test_stale_scan_hard_stops_after_fresh_progress_without_regression(self):
        self.param_overrides["~parking/safety/require_live_scan"] = True
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.route_transform = self.rigid_route_transform()
        start = (controller.aisle_x, 1.50, controller.aisle_heading)
        pose0 = (controller.aisle_x, 1.30, controller.aisle_heading)
        pose1 = (controller.aisle_x, 1.25, controller.aisle_heading)
        goal = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.update_poses(controller, start, start)
        self.assertTrue(
            controller._begin_drive(
                controller.ENTER_AISLE, goal, 1, controller.aisle_speed
            )
        )
        controller.path_follower.last_linear = 0.12
        controller.last_linear = 0.12
        controller.last_command_time = self.now()

        self.advance(controller.control_period)
        controller.odom_callback(self.odom_message(pose0, linear_x=0.12))
        controller.safety_scan_stamp = self.now()
        controller.safety_scan_received = self.now()
        controller._control_motion(self.now())
        normal = self.publishers[controller.cmd_vel_topic].messages[-1]
        recorded_station = controller.path_follower.path_station

        controller.safety_scan_stamp = controller_module.rospy.Time.from_sec(
            self.seconds - controller.scan_timeout - 0.01
        )
        controller.safety_scan_received = controller.safety_scan_stamp
        self.advance(controller.control_period)
        controller.odom_callback(
            self.odom_message(pose1, linear_x=normal.linear.x)
        )
        controller._control_motion(self.now())
        stopped = self.publishers[controller.cmd_vel_topic].messages[-1]

        self.assertEqual(stopped.linear.x, 0.0)
        self.assertEqual(stopped.angular.z, 0.0)
        self.assertEqual(controller.path_follower.last_linear, 0.0)
        self.assertGreaterEqual(
            controller.path_follower.path_station, recorded_station
        )

        # Recovery calculates tracking from the current monotonic station and
        # must not raise after odometry advanced during the stale scan tick.
        self.advance(controller.control_period)
        controller.odom_callback(self.odom_message(pose1, linear_x=0.0))
        controller.safety_scan_stamp = self.now()
        controller.safety_scan_received = self.now()
        controller._control_motion(self.now())
        resumed = self.publishers[controller.cmd_vel_topic].messages[-1]

        self.assertGreater(resumed.linear.x, 0.0)
        self.assertEqual(controller.state, controller.ENTER_AISLE)

    def test_stale_scan_without_prior_safe_sweep_holds_exact_zero(self):
        self.param_overrides["~parking/safety/require_live_scan"] = True
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.route_transform = self.rigid_route_transform()
        start = (controller.aisle_x, 1.50, controller.aisle_heading)
        goal = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.update_poses(controller, start, start)
        self.assertTrue(
            controller._begin_drive(
                controller.ENTER_AISLE, goal, 1, controller.aisle_speed
            )
        )
        controller.path_follower.last_linear = 0.12
        controller.last_linear = 0.12
        controller.last_command_time = self.now()

        self.advance(controller.control_period)
        controller.odom_callback(self.odom_message(start, linear_x=0.12))
        command = controller._common_path_command(self.now())

        self.assertEqual(command.linear.x, 0.0)
        self.assertEqual(command.angular.z, 0.0)

    def test_persistent_stale_scan_hard_stops_on_second_response(self):
        self.param_overrides["~parking/safety/require_live_scan"] = True
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.route_transform = self.rigid_route_transform()
        start = (controller.aisle_x, 1.50, controller.aisle_heading)
        pose = (controller.aisle_x, 1.25, controller.aisle_heading)
        goal = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.update_poses(controller, start, start)
        self.assertTrue(
            controller._begin_drive(
                controller.ENTER_AISLE, goal, 1, controller.aisle_speed
            )
        )
        controller.odom_callback(self.odom_message(pose, linear_x=0.12))
        controller.path_follower.last_linear = 0.12
        controller.last_linear = 0.12
        controller.last_command_time = self.now()
        controller.safety_scan_stamp = self.now()
        controller.safety_scan_received = self.now()

        self.advance(controller.control_period)
        controller.odom_callback(self.odom_message(pose, linear_x=0.12))
        normal = controller._common_path_command(self.now())
        controller.safety_scan_stamp = controller_module.rospy.Time.from_sec(
            self.seconds - controller.scan_timeout - 0.01
        )
        controller.safety_scan_received = controller.safety_scan_stamp

        self.advance(controller.control_period)
        controller.odom_callback(
            self.odom_message(pose, linear_x=normal.linear.x)
        )
        first = controller._common_path_command(self.now())
        self.assertEqual(first.linear.x, 0.0)
        self.assertEqual(first.angular.z, 0.0)

        self.advance(controller.control_period)
        controller.odom_callback(
            self.odom_message(pose, linear_x=first.linear.x)
        )
        second = controller._common_path_command(self.now())

        self.assertEqual(second.linear.x, 0.0)
        self.assertEqual(second.angular.z, 0.0)
        self.assertEqual(controller.path_follower.last_linear, 0.0)

    def test_stale_odom_response_holds_exact_zero(self):
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.route_transform = self.rigid_route_transform()
        start = (controller.aisle_x, 1.50, controller.aisle_heading)
        pose = (controller.aisle_x, 1.25, controller.aisle_heading)
        goal = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.update_poses(controller, start, start)
        self.assertTrue(
            controller._begin_drive(
                controller.ENTER_AISLE, goal, 1, controller.aisle_speed
            )
        )
        controller.odom_callback(self.odom_message(pose, linear_x=0.12))
        controller.path_follower.last_linear = 0.12
        controller.last_linear = 0.12
        controller.last_command_time = self.now()
        self.advance(controller.control_period)
        controller.odom_callback(self.odom_message(pose, linear_x=0.12))
        normal = controller._common_path_command(self.now())
        controller.odom_stamp = controller_module.rospy.Time.from_sec(
            self.seconds - controller.odom_timeout - 0.01
        )
        controller.odom_received = controller.odom_stamp

        self.advance(controller.control_period)
        controller._control_motion(self.now())

        command = self.publishers[controller.cmd_vel_topic].messages[-1]
        self.assertEqual(command.linear.x, 0.0)
        self.assertEqual(command.angular.z, 0.0)
        self.assertEqual(controller.state, controller.ENTER_AISLE)

    def test_persistent_stale_scan_still_enforces_motion_timeout(self):
        self.param_overrides["~parking/safety/require_live_scan"] = True
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.route_transform = self.rigid_route_transform()
        start = (controller.aisle_x, 1.50, controller.aisle_heading)
        goal = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.update_poses(controller, start, start)
        self.assertTrue(
            controller._begin_drive(
                controller.ENTER_AISLE, goal, 1, controller.aisle_speed
            )
        )

        self.advance(controller.motion_timeout + 0.001)
        controller.odom_callback(self.odom_message(start, linear_x=0.0))
        controller.safety_scan_stamp = controller_module.rospy.Time.from_sec(
            self.seconds - controller.scan_timeout - 0.01
        )
        controller.safety_scan_received = controller.safety_scan_stamp
        controller._control_motion(self.now())

        self.assertEqual(controller.state, controller.FAILED)
        command = self.publishers[controller.cmd_vel_topic].messages[-1]
        self.assertEqual(command.linear.x, 0.0)
        self.assertEqual(command.angular.z, 0.0)

    def test_hard_stop_cannot_advance_segment_while_odom_is_moving(self):
        self.param_overrides["~parking/safety/require_live_scan"] = True
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.route_transform = self.rigid_route_transform()
        start = (controller.aisle_x, 1.50, controller.aisle_heading)
        goal = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.update_poses(controller, start, start)
        self.assertTrue(
            controller._begin_drive(
                controller.ENTER_AISLE, goal, 1, controller.aisle_speed
            )
        )
        controller.motion_goal_confirmed = True
        controller.path_follower.last_linear = 0.12
        controller.last_linear = 0.12
        controller.last_command_time = self.now()

        # A stale scan makes the command state zero immediately, while wheel
        # dynamics can leave the measured base speed nonzero.
        self.advance(controller.control_period)
        controller.odom_callback(self.odom_message(goal, linear_x=0.10))
        controller.safety_scan_stamp = controller_module.rospy.Time.from_sec(
            self.seconds - controller.scan_timeout - 0.01
        )
        controller.safety_scan_received = controller.safety_scan_stamp
        controller._control_motion(self.now())
        self.assertEqual(controller.path_follower.last_linear, 0.0)

        # Fresh inputs resume the terminal logic, but it must keep this state
        # until odometry independently confirms that the base has stopped.
        self.advance(controller.control_period)
        controller.odom_callback(self.odom_message(goal, linear_x=0.08))
        controller.safety_scan_stamp = self.now()
        controller.safety_scan_received = self.now()
        controller._control_motion(self.now())
        self.assertEqual(controller.state, controller.ENTER_AISLE)

        self.advance(controller.control_period)
        controller.odom_callback(self.odom_message(goal, angular_z=0.08))
        controller.safety_scan_stamp = self.now()
        controller.safety_scan_received = self.now()
        controller._control_motion(self.now())
        self.assertEqual(controller.state, controller.ENTER_AISLE)

        self.advance(controller.control_period)
        controller.odom_callback(self.odom_message(goal, linear_x=0.0))
        controller.safety_scan_stamp = self.now()
        controller.safety_scan_received = self.now()
        controller._control_motion(self.now())
        self.assertEqual(controller.state, controller.SELECT_SPACE)

    def test_turn_in_safety_benchmark_stays_below_control_period(self):
        controller = self.make_controller()
        controller.route_transform = self.rigid_route_transform()
        curve_start = (
            controller.aisle_x + controller.entry_curve_offset,
            controller.entry_y,
            controller.approach_heading,
        )
        poses = quintic_turn_path(
            curve_start,
            controller.aisle_heading,
            controller.entry_curve_offset,
            controller.entry_curve_tangent,
            controller.entry_curve_samples,
        )
        path = controller._common_path_from_poses(
            poses,
            1,
            controller.entry_turn_speed,
            controller.TURN_IN,
            "entry_turn",
            controller.entry_curve_end_position_tolerance,
            controller.entry_curve_end_heading_tolerance,
            controller.entry_curve_end_crossing_max_distance,
            feedforward_scale=controller.arc_angular_scale,
        )
        self.assertIsNotNone(path)
        index = int(0.324 * (path.size - 1))
        pose = (path.x[index], path.y[index], path.heading[index])
        controller.state = controller.TURN_IN
        controller.odom_pose = pose
        self.assertTrue(
            controller._activate_common_path(
                path,
                controller.TURN_IN,
                initial_linear=0.04,
                initial_angular=0.10,
            )
        )
        controller.odom_linear_speed = 0.04
        controller.odom_angular_velocity = 0.10
        angles = np.linspace(-math.pi, math.pi, 4000, endpoint=False)
        controller.live_obstacle_points_odom = np.column_stack(
            (
                pose[0] + 0.50 * np.cos(angles),
                pose[1] + 0.50 * np.sin(angles),
            )
        )
        controller.safety_scan_stamp = self.now()
        controller.safety_scan_received = self.now()

        # Warm the immutable-route boundary cache, then model the repeated
        # 20 Hz control load with a dense Mid-360-sized obstacle cloud.
        controller._common_safety_speed_limit(self.now())
        durations = []
        for _ in range(5):
            started = time.perf_counter()
            speed_limit, emergency = controller._common_safety_speed_limit(
                self.now()
            )
            durations.append(time.perf_counter() - started)
            self.assertFalse(emergency)
            self.assertGreater(speed_limit, 0.0)

        self.assertLess(float(np.median(durations)), controller.control_period)
        self.assertLess(max(durations), controller.control_period)

    def test_safety_scan_projection_is_not_starved_by_swept_validation(self):
        controller = self.make_controller()
        controller.scan_median_window = 1
        controller.require_live_safety_scan = False
        controller.route_transform = self.rigid_route_transform()
        old_cloud = np.asarray([[-9.0, -9.0]], dtype=np.float64)
        controller.live_obstacle_points_odom = old_cloud
        controller.safety_scan_stamp = self.now()
        controller.safety_scan_received = self.now()

        self.seconds = 10.00
        start = (controller.aisle_x, 1.50, controller.aisle_heading)
        goal = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        first_odom = self.odom_message(start)
        first_odom.header.stamp = controller_module.rospy.Time.from_sec(10.00)
        controller.odom_callback(first_odom)
        self.assertTrue(
            controller._begin_drive(
                controller.ENTER_AISLE,
                goal,
                1,
                controller.aisle_speed,
            )
        )

        validation_entered = threading.Event()
        release_validation = threading.Event()
        original_motion_safety = controller.path_validator.motion_safety
        validation_clouds = []

        def delayed_motion_safety(*args, **kwargs):
            validation_clouds.append(kwargs["live_obstacles"])
            validation_entered.set()
            self.assertTrue(release_validation.wait(1.0))
            return original_motion_safety(*args, **kwargs)

        controller.path_validator.motion_safety = delayed_motion_safety
        self.seconds = 10.08

        def control_work():
            # Match control_callback's ownership of the controller state lock.
            with controller.lock:
                controller._common_safety_speed_limit(self.now())

        control_thread = threading.Thread(target=control_work)
        control_thread.start()
        self.assertTrue(validation_entered.wait(1.0))

        scan = LaserScan()
        scan.header.stamp = controller_module.rospy.Time.from_sec(10.04)
        scan.angle_min = 0.0
        scan.angle_increment = 0.01
        scan.range_min = 0.10
        scan.range_max = 10.0
        scan.ranges = [0.40]
        scan_thread = threading.Thread(target=controller.scan_callback, args=(scan,))
        scan_thread.start()

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with controller.safety_data_lock:
                if controller.pending_safety_scans:
                    break
            time.sleep(0.001)
        else:
            self.fail("scan did not reach the safety synchronization queue")

        second_odom = self.odom_message((start[0], start[1] - 0.008, start[2]))
        second_odom.header.stamp = controller_module.rospy.Time.from_sec(10.08)
        odom_thread = threading.Thread(
            target=controller.odom_callback, args=(second_odom,)
        )
        odom_thread.start()

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            with controller.safety_data_lock:
                stamp = controller.safety_scan_stamp
                if stamp is not None and abs(stamp.to_sec() - 10.04) <= 1e-9:
                    break
            time.sleep(0.001)
        else:
            self.fail("swept validation starved the synchronized safety scan")

        # Neither subscriber worker may queue behind the long main-state lock.
        # The scan's safety projection is already current and the odom's newest
        # mission-state snapshot waits in one overwrite-only deferred slot.
        for thread in (scan_thread, odom_thread):
            thread.join(1.0)
            self.assertFalse(thread.is_alive())
        self.assertTrue(control_thread.is_alive())
        self.assertEqual(len(validation_clouds), 1)
        self.assertIs(validation_clouds[0], old_cloud)
        np.testing.assert_array_equal(validation_clouds[0], [[-9.0, -9.0]])
        self.assertAlmostEqual(controller.odom_stamp.to_sec(), 10.00)
        with controller.safety_data_lock:
            self.assertIsNotNone(controller.pending_odom_state)
            self.assertAlmostEqual(
                controller.pending_odom_state[4].to_sec(), 10.08
            )

        release_validation.set()
        control_thread.join(1.0)
        self.assertFalse(control_thread.is_alive())
        with controller.lock:
            controller._apply_pending_odom_state()
        self.assertAlmostEqual(controller.odom_stamp.to_sec(), 10.08)
        self.assertAlmostEqual(controller.odom_pose[1], start[1] - 0.008)
        fresh, next_cloud = controller._live_safety_snapshot(self.now())
        self.assertTrue(fresh)
        self.assertIsNot(next_cloud, old_cloud)
        self.assertEqual(next_cloud.shape, (1, 2))

    def test_deferred_odom_keeps_only_latest_original_timestamp(self):
        controller = self.make_controller()

        with controller.lock:
            for stamp, x in ((10.10, 0.10), (10.20, 0.20)):
                self.seconds = stamp
                message = self.odom_message((x, 0.0, 0.0), linear_x=x)
                message.header.stamp = controller_module.rospy.Time.from_sec(
                    stamp
                )
                worker = threading.Thread(
                    target=controller.odom_callback, args=(message,)
                )
                worker.start()
                worker.join(1.0)
                self.assertFalse(worker.is_alive())

            self.assertIsNone(controller.odom_pose)
            with controller.safety_data_lock:
                self.assertIsNotNone(controller.pending_odom_state)
                self.assertEqual(controller.pending_odom_state[0], 2)
                self.assertAlmostEqual(
                    controller.pending_odom_state[4].to_sec(), 10.20
                )

        with controller.lock:
            controller._apply_pending_odom_state()
        self.assertEqual(controller.odom_generation, 1)
        self.assertEqual(controller.applied_odom_arrival_generation, 2)
        self.assertAlmostEqual(controller.odom_pose[0], 0.20)
        self.assertAlmostEqual(controller.odom_linear_speed, 0.20)

        # Deferral must not rewrite receipt/source time and thereby disguise a
        # real publisher outage as a fresh sample.
        self.seconds = 10.20 + controller.odom_timeout + 0.01
        self.assertFalse(controller._odom_is_fresh(self.now()))

    def test_gate_drains_preexisting_odom_before_recording_its_edge(self):
        controller = self.make_controller()
        with controller.lock:
            message = self.odom_message((0.90, 1.75, math.pi))
            worker = threading.Thread(
                target=controller.odom_callback, args=(message,)
            )
            worker.start()
            worker.join(1.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(controller.odom_generation, 0)

        controller.gate_callback(Bool(data=True))

        self.assertEqual(controller.odom_generation, 1)
        self.assertEqual(controller.gate_odom_generation, 1)
        self.assertEqual(
            controller.gate_odom_generation, controller.odom_generation
        )

    def test_gate_counts_odom_staged_before_inflight_scan_projection(self):
        controller = self.make_controller()
        controller.scan_median_window = 1

        first = self.odom_message((0.0, 0.0, 0.0))
        first.header.stamp = controller_module.rospy.Time.from_sec(10.00)
        controller.odom_callback(first)

        self.seconds = 10.04
        scan = LaserScan()
        scan.header.stamp = controller_module.rospy.Time.from_sec(10.04)
        scan.angle_min = 0.0
        scan.angle_increment = 0.01
        scan.range_min = 0.10
        scan.range_max = 10.0
        scan.ranges = [0.40]
        controller.scan_callback(scan)
        self.assertEqual(len(controller.pending_safety_scans), 1)

        projection_entered = threading.Event()
        release_projection = threading.Event()
        original_projection = controller._project_safety_scan

        def delayed_projection(*args, **kwargs):
            projection_entered.set()
            self.assertTrue(release_projection.wait(1.0))
            return original_projection(*args, **kwargs)

        controller._project_safety_scan = delayed_projection
        self.seconds = 10.08
        second = self.odom_message((0.08, 0.0, 0.0))
        second.header.stamp = controller_module.rospy.Time.from_sec(10.08)
        worker = threading.Thread(target=controller.odom_callback, args=(second,))
        worker.start()
        self.assertTrue(projection_entered.wait(1.0))

        # The odom snapshot was staged before projection began, so the gate
        # drains it as pre-edge data. It cannot later satisfy the requirement
        # for a distinct post-gate moving sample.
        controller.gate_callback(Bool(data=True))
        self.assertEqual(controller.odom_generation, 2)
        self.assertEqual(controller.gate_odom_generation, 2)
        self.assertEqual(controller.applied_odom_arrival_generation, 2)

        release_projection.set()
        worker.join(1.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(controller.odom_generation, 2)

        self.seconds = 10.12
        third = self.odom_message((0.12, 0.0, 0.0))
        third.header.stamp = controller_module.rospy.Time.from_sec(10.12)
        controller.odom_callback(third)
        self.assertGreater(controller.odom_generation, controller.gate_odom_generation)

    def test_zigzag_exit_safety_benchmark_stays_below_control_period(self):
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.selected_space = LEFT
        controller.route_transform = self.rigid_route_transform()
        turn = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        start = (turn[0], turn[1], controller.outgoing_heading)
        self.update_poses(controller, start, start)
        self.assertTrue(controller._begin_leave_aisle())
        curve_start = controller._zigzag_turn_start_pose()
        self.update_poses(controller, curve_start, curve_start)
        self.assertTrue(controller._begin_zigzag_exit_curve())

        index = int(np.argmax(np.abs(controller.active_path.curvature)))
        pose = (
            controller.active_path.x[index],
            controller.active_path.y[index],
            controller.active_path.heading[index],
        )
        controller.odom_pose = pose
        # Exercise the requested upper bound directly. The complete-stop sweep
        # must retain enough 20 Hz headroom at the configured ceiling even at
        # the smooth exit turn's maximum-curvature sample.
        controller.path_follower.last_linear = controller.rejoin_speed
        controller.odom_linear_speed = controller.rejoin_speed
        angles = np.linspace(-math.pi, math.pi, 4000, endpoint=False)
        controller.live_obstacle_points_odom = np.column_stack(
            (
                pose[0] + 0.50 * np.cos(angles),
                pose[1] + 0.50 * np.sin(angles),
            )
        )
        controller.safety_scan_stamp = self.now()
        controller.safety_scan_received = self.now()

        controller._common_safety_speed_limit(self.now())
        durations = []
        for _ in range(10):
            started = time.perf_counter()
            speed_limit, emergency = controller._common_safety_speed_limit(
                self.now()
            )
            durations.append(time.perf_counter() - started)
            self.assertFalse(emergency)
            self.assertTrue(math.isfinite(speed_limit))

        self.assertLess(float(np.median(durations)), controller.control_period)
        self.assertLess(max(durations), controller.control_period)

    def test_leave_to_zigzag_turn_switches_paths_without_zero_command(self):
        for selected_space in (LEFT, RIGHT):
            with self.subTest(selected_space=selected_space):
                controller = self.make_controller()
                controller.mission_has_control = True
                controller.selected_space = selected_space
                controller.route_transform = self.rigid_route_transform()
                turn = (
                    controller.aisle_x,
                    controller.decision_y,
                    controller.aisle_heading,
                )
                start = (
                    turn[0],
                    turn[1],
                    controller.outgoing_heading,
                )
                self.update_poses(controller, start, start)
                self.assertTrue(controller._begin_leave_aisle())
                leave_path = controller.active_path
                turn_path = controller.zigzag_exit_curve_path
                self.assertAlmostEqual(
                    leave_path.speed[-1], turn_path.speed[0], places=12
                )
                controller.path_follower.last_linear = 0.08
                controller.last_linear = 0.08
                controller.path_follower.last_angular = 0.0
                controller.last_angular = 0.0
                controller.last_command_time = self.now()
                commands = self.publishers[controller.cmd_vel_topic].messages
                commands.clear()

                self.advance(controller.control_period)
                self.update_poses(
                    controller, controller.goal_map, controller.goal_odom
                )
                controller.control_callback(None)

                self.assertEqual(controller.state, controller.TURN_TO_ZIGZAG)
                self.assertIs(controller.active_path, turn_path)
                self.assertIsNot(controller.active_path, leave_path)
                self.assertEqual(len(commands), 1)
                self.assertGreater(commands[-1].linear.x, 0.0)
                self.assertFalse(
                    any(
                        abs(command.linear.x) <= 1e-9
                        and abs(command.angular.z) <= 1e-9
                        for command in commands
                    )
                )


    def test_confirmed_motion_segment_slews_to_zero_before_transition(self):
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.route_transform = self.rigid_route_transform()
        start = (controller.aisle_x, controller.entry_y, controller.aisle_heading)
        goal = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.update_poses(controller, start, start)
        self.assertTrue(
            controller._begin_drive(
                controller.ENTER_AISLE, goal, 1, controller.aisle_speed
            )
        )
        controller.path_follower.last_linear = 0.06
        controller.path_follower.last_angular = 0.12
        controller.last_linear = 0.06
        controller.last_angular = 0.12
        controller.last_command_time = self.now()

        for _ in range(controller.pose_confirm_samples):
            self.advance(controller.control_period)
            self.update_poses(controller, goal, controller.goal_odom)
            controller.control_callback(None)

        commands = self.publishers[controller.cmd_vel_topic].messages
        self.assertEqual(controller.state, controller.ENTER_AISLE)
        self.assertTrue(controller.motion_goal_confirmed)
        self.assertGreater(commands[-1].linear.x, 0.0)
        self.assertLessEqual(commands[-1].linear.x, 0.06 + 1e-12)
        self.assertLess(abs(commands[-1].angular.z), 0.12)

        previous_linear = commands[-1].linear.x
        previous_angular = commands[-1].angular.z
        for _ in range(100):
            if controller.state != controller.ENTER_AISLE:
                break
            self.advance(controller.control_period)
            self.update_poses(controller, goal, controller.goal_odom)
            controller.control_callback(None)
            command = commands[-1]
            self.assertLessEqual(
                previous_linear - command.linear.x,
                controller.linear_deceleration * controller.control_period
                + 1e-9,
            )
            self.assertLessEqual(
                abs(previous_angular - command.angular.z),
                controller.angular_acceleration * controller.control_period
                + 1e-9,
            )
            if controller.state == controller.ENTER_AISLE:
                self.assertGreater(
                    abs(command.linear.x) + abs(command.angular.z), 0.0
                )
            previous_linear = command.linear.x
            previous_angular = command.angular.z

        self.assertEqual(controller.state, controller.SELECT_SPACE)
        self.assertAlmostEqual(commands[-1].linear.x, 0.0)
        self.assertAlmostEqual(commands[-1].angular.z, 0.0)

    def test_motion_timeout_at_goal_still_confirms_and_stops_before_transition(self):
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.route_transform = self.rigid_route_transform()
        start = (
            controller.aisle_x,
            controller.entry_y,
            controller.aisle_heading,
        )
        goal = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.update_poses(controller, start, start)
        self.assertTrue(
            controller._begin_drive(
                controller.ENTER_AISLE,
                goal,
                1,
                controller.aisle_speed,
            )
        )
        controller.path_follower.last_linear = 0.06
        controller.path_follower.last_angular = 0.12
        controller.last_linear = 0.06
        controller.last_angular = 0.12
        controller.last_command_time = self.now()

        # The first fresh goal sample arrives just after the nominal timeout.
        # It must begin the normal two-sample confirmation instead of turning
        # an already successful motion into a timeout failure.
        self.advance(controller.motion_timeout + 0.001)
        self.update_poses(controller, goal, controller.goal_odom)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.ENTER_AISLE)
        self.assertEqual(controller.odom_confirmation_count, 1)
        self.assertFalse(controller.motion_goal_confirmed)

        self.advance(controller.control_period)
        self.update_poses(controller, goal, controller.goal_odom)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.ENTER_AISLE)
        self.assertTrue(controller.motion_goal_confirmed)

        commands = self.publishers[controller.cmd_vel_topic].messages
        self.assertGreater(
            abs(commands[-1].linear.x) + abs(commands[-1].angular.z),
            0.0,
        )
        for _ in range(100):
            if controller.state != controller.ENTER_AISLE:
                break
            self.advance(controller.control_period)
            self.update_poses(controller, goal, controller.goal_odom)
            controller.control_callback(None)

        self.assertEqual(controller.state, controller.SELECT_SPACE)
        self.assertAlmostEqual(commands[-1].linear.x, 0.0)
        self.assertAlmostEqual(commands[-1].angular.z, 0.0)

    def test_motion_timeout_outside_goal_tolerance_still_fails(self):
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.route_transform = self.rigid_route_transform()
        start = (
            controller.aisle_x,
            controller.entry_y,
            controller.aisle_heading,
        )
        goal = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.update_poses(controller, start, start)
        self.assertTrue(
            controller._begin_drive(
                controller.ENTER_AISLE,
                goal,
                1,
                controller.aisle_speed,
            )
        )

        outside_goal = (
            controller.goal_odom[0],
            controller.goal_odom[1] + 2.0 * controller.arc_position_tolerance,
            controller.goal_odom[2]
            + 2.0 * controller.arc_heading_tolerance,
        )
        self.advance(controller.motion_timeout + 0.001)
        self.update_poses(controller, outside_goal, outside_goal)
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.FAILED)
        command = self.publishers[controller.cmd_vel_topic].messages[-1]
        self.assertAlmostEqual(command.linear.x, 0.0)
        self.assertAlmostEqual(command.angular.z, 0.0)

    def test_amcl_correction_does_not_reproject_an_active_odom_goal(self):
        controller = self.start_controller(self.make_controller())
        frozen_transform = controller.route_transform
        frozen_goal = controller.goal_odom

        self.advance(0.05)
        controller.map_pose_callback(
            self.pose_message((1.08, 1.82, math.radians(168.0)))
        )
        controller.odom_callback(self.odom_message((0.90, 1.75, math.pi)))
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.APPROACH)
        self.assertEqual(controller.route_transform, frozen_transform)
        self.assertEqual(controller.goal_odom, frozen_goal)

    def test_entry_curve_and_aisle_goal_use_route_anchor_after_amcl_jump(self):
        controller = self.start_controller(self.make_controller())
        frozen_route = controller.route_transform
        self.confirm_current_goal(controller)
        self.assertEqual(controller.state, controller.TURN_IN)
        expected_turn = (
            controller.aisle_x,
            controller.entry_y - controller.entry_curve_offset,
            controller.aisle_heading,
        )
        self.assert_pose_almost_equal(controller.goal_map, expected_turn)
        self.assert_pose_almost_equal(
            controller.goal_odom,
            self.route_pose_to_odom(expected_turn, frozen_route),
        )
        self.assertTrue(np.all(controller.active_path.direction == 1))
        self.assertLessEqual(
            float(np.max(controller.active_path.speed)),
            controller.entry_turn_speed,
        )
        self.assertEqual(len(controller.entry_curve_odom), controller.entry_curve_samples)
        self.assert_pose_almost_equal(
            controller.entry_curve_odom[-1], controller.goal_odom
        )

        jumped_map = (1.10, 1.35, math.radians(-89.0))
        self.advance(0.05)
        controller.map_pose_callback(self.pose_message(jumped_map))
        controller.odom_callback(self.odom_message(controller.goal_odom))
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.ENTER_AISLE)
        expected_aisle = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.assert_pose_almost_equal(controller.goal_map, expected_aisle)
        self.assert_pose_almost_equal(
            controller.goal_odom,
            self.route_pose_to_odom(expected_aisle, frozen_route),
        )
        self.assertEqual(controller.route_transform, frozen_route)

    def test_entry_curve_joins_both_straights_without_zero_command(self):
        controller = self.start_controller(self.make_controller())
        controller.mission_has_control = True
        commands = self.publishers[controller.cmd_vel_topic].messages
        commands.clear()
        connector_path = controller.entry_connector_path
        curve_path = controller.entry_curve_path

        controller.last_linear = 0.060
        controller.last_angular = 0.0
        controller.last_command_time = self.now()
        approach_goal_map = controller.goal_map
        approach_goal_odom = controller.goal_odom
        self.advance(0.05)
        self.update_poses(controller, approach_goal_map, approach_goal_odom)
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.TURN_IN)
        self.assertIs(controller.active_path, curve_path)
        self.assertIsNot(connector_path, curve_path)
        self.assertEqual(len(commands), 1)
        self.assertGreater(commands[-1].linear.x, 0.0)
        self.assertLessEqual(
            abs(commands[-1].linear.x - 0.060),
            controller.linear_acceleration * 0.05 + 1e-8,
        )
        self.assertLessEqual(
            abs(commands[-1].angular.z),
            controller.angular_acceleration * 0.05 + 1e-8,
        )

        commands.clear()
        curve_goal_map = controller.goal_map
        curve_goal_odom = controller.goal_odom
        previous_linear = controller.last_linear
        previous_angular = controller.last_angular
        self.advance(0.05)
        self.update_poses(controller, curve_goal_map, curve_goal_odom)
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.ENTER_AISLE)
        self.assertEqual(len(commands), 1)
        self.assertGreater(commands[-1].linear.x, 0.0)
        self.assertLessEqual(
            abs(commands[-1].linear.x - previous_linear),
            controller.linear_acceleration * 0.05 + 1e-8,
        )
        self.assertLessEqual(
            abs(commands[-1].angular.z - previous_angular),
            controller.angular_acceleration * 0.05 + 1e-8,
        )

    def test_lane_handoff_continues_into_adaptive_entry_without_stop(self):
        self.param_overrides.update(
            {
                "~parking/route/odom_aligned": False,
                "~parking/control/arc_angular_scale": 1.10,
                "~parking/control/linear_acceleration": 0.03,
                "~parking/control/angular_acceleration": 0.55,
            }
        )
        controller = self.make_controller()
        pre_gate_command = self.lane_command(controller, 0.076, 0.015)
        map_pose = (0.9817, 1.7557, math.radians(-179.7))
        odom_pose = (1.0033, 1.7608, math.radians(-179.5))

        controller.gate_callback(Bool(data=True))
        self.update_poses(controller, map_pose, odom_pose)
        self.events.clear()
        self.lane_service.calls.clear()
        controller.control_callback(None)

        commands = self.publishers[controller.cmd_vel_topic].messages
        self.assertEqual(controller.state, controller.PREPARE_APPROACH)
        self.assertEqual(self.lane_service.calls, [])
        self.assertEqual(commands, [])
        self.assertFalse(controller.mission_has_control)

        # A pre-gate command and a post-gate command above the requested cap
        # must not be used for handoff; lane control remains the sole owner.
        self.advance(controller.prepare_settle_time + 0.01)
        latest_odom = (0.9945, 1.7608, math.radians(-179.5))
        controller.odom_callback(self.odom_message(latest_odom))
        self.lane_command(controller, 0.081, 0.015)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.PREPARE_APPROACH)
        self.assertEqual(self.lane_service.calls, [])
        self.assertEqual(commands, [])

        lane_command = self.lane_command(controller, 0.076, 0.015)
        self.events.clear()
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.APPROACH)
        self.assertEqual(self.lane_service.calls, [False])

        self.assertTrue(controller.mission_has_control)
        self.assert_pose_almost_equal(
            controller.entry_connector_odom[0], latest_odom
        )
        self.assertEqual(len(commands), 1)
        self.assertAlmostEqual(commands[0].linear.x, lane_command.linear.x)
        self.assertAlmostEqual(commands[0].angular.z, lane_command.angular.z)
        service_index = next(
            index
            for index, event in enumerate(self.events)
            if event[0] == "service" and event[1] is False
        )
        command_index = next(
            index
            for index, event in enumerate(self.events)
            if event[0] == "publish" and event[1] == "/cmd_vel"
        )
        self.assertLess(service_index, command_index)
        self.assertFalse(
            any(
                abs(command.linear.x) <= 1e-9
                and abs(command.angular.z) <= 1e-9
                for command in commands
            )
        )

        generation = controller.lane_command_generation
        controller.command_observer_callback(pre_gate_command)
        self.assertEqual(controller.lane_command_generation, generation)

        previous_linear = controller.last_linear
        previous_angular = controller.last_angular
        self.advance(0.05)
        controller.control_callback(None)
        self.assertGreater(commands[-1].linear.x, 0.0)
        self.assertLessEqual(
            abs(commands[-1].linear.x - previous_linear),
            controller.linear_acceleration * 0.05 + 1e-8,
        )
        self.assertLessEqual(
            abs(commands[-1].angular.z - previous_angular),
            controller.angular_acceleration * 0.05 + 1e-8,
        )

    def test_route_transform_uses_odom_at_amcl_source_stamp(self):
        controller = self.make_controller()
        controller.map_pose = (1.0, 1.75, 0.10)
        controller.map_pose_stamp = controller_module.rospy.Time.from_sec(10.0)
        controller.odom_pose = (0.20, 0.0, 0.0)
        controller.odom_pose_history = deque(
            (
                (9.98, (0.10, 0.0, 0.0)),
                (10.02, (0.14, 0.0, 0.0)),
            )
        )

        self.assertTrue(controller._latch_route_transform())
        synchronized = controller.route_transform.apply_pose(
            Pose2D(0.12, 0.0, 0.0)
        )
        self.assertAlmostEqual(synchronized.x, controller.map_pose[0])
        self.assertAlmostEqual(synchronized.y, controller.map_pose[1])
        self.assertAlmostEqual(synchronized.yaw, controller.map_pose[2])

        # Using the newest odom pose here would shift a moving 3 mm-clearance
        # route by 8 cm in this exaggerated regression fixture.
        newest = controller.route_transform.apply_pose(
            Pose2D(*controller.odom_pose)
        )
        self.assertGreater(math.hypot(newest.x - 1.0, newest.y - 1.75), 0.07)

    def test_safe_entry_is_not_blocked_by_the_removed_fixed_amcl_start_box(self):
        self.param_overrides.update(
            {
                "~parking/route/odom_aligned": False,
                "~parking/control/arc_angular_scale": 1.10,
                "~parking/control/linear_acceleration": 0.03,
                "~parking/control/angular_acceleration": 0.55,
            }
        )
        controller = self.make_controller()
        # x=1.04 was outside the deleted fixed max_x=1.03 condition. The
        # frozen map pose is safe and the raw odom offset is handled only by
        # the latched rigid transform, so the footprint check decides.
        map_pose = (1.04, 1.755, math.pi)
        odom_pose = (1.00, 1.755, math.pi)

        controller.gate_callback(Bool(data=True))
        self.update_poses(controller, map_pose, odom_pose)
        self.lane_command(controller)
        controller.control_callback(None)
        self.advance(controller.prepare_settle_time + 0.01)
        self.update_poses(controller, map_pose, odom_pose)
        self.lane_command(controller)
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.APPROACH)
        self.assertEqual(self.lane_service.calls, [False])
        self.assertGreaterEqual(
            controller.entry_connector_path.line_clearance
            + controller.entry_handoff_minimum_clearance,
            controller.line_margin,
        )


    def test_selection_starts_rotation_at_the_stopped_decision_pose(self):
        for selected_space, target_yaw in ((LEFT, 0.0), (RIGHT, math.pi)):
            with self.subTest(selected_space=selected_space):
                controller = self.make_controller()
                controller.mission_has_control = True
                controller.route_transform = self.rigid_route_transform()
                controller.state = controller.SELECT_SPACE
                controller.state_started = self.now()
                decision = (
                    controller.aisle_x,
                    controller.decision_y,
                    controller.aisle_heading,
                )
                self.update_poses(controller, decision)

                self.confirm_space_selection(controller, selected_space)

                self.assertEqual(controller.state, controller.TURN_TO_SPACE)
                self.assert_pose_almost_equal(
                    controller.goal_map,
                    (decision[0], decision[1], target_yaw),
                )
                self.assert_pose_almost_equal(
                    controller.goal_odom,
                    (decision[0], decision[1], target_yaw),
                )
                self.assertAlmostEqual(
                    controller.rotation_center_odom.x, decision[0]
                )
                self.assertAlmostEqual(
                    controller.rotation_center_odom.y, decision[1]
                )
                self.assertIsNone(controller.active_path)
                self.assertTrue(controller.active_path_validation.safe)

    def test_leave_aisle_uses_one_path_from_the_central_return_pose(self):
        for selected_space in (LEFT, RIGHT):
            with self.subTest(selected_space=selected_space):
                controller = self.make_controller()
                controller.mission_has_control = True
                controller.selected_space = selected_space
                controller.route_transform = self.rigid_route_transform()
                start = (
                    controller.aisle_x,
                    controller.decision_y,
                    controller.outgoing_heading,
                )
                self.update_poses(controller, start)

                self.assertTrue(controller._begin_leave_aisle())

                goal = controller._zigzag_turn_start_pose()
                distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
                tangent = controller.zigzag_approach_tangent_ratio * max(
                    0.01, distance
                )
                sample_count = max(
                    11,
                    int(math.ceil(distance / controller.path_sample_spacing))
                    + 1,
                )
                expected = quintic_pose_path(
                    start,
                    goal,
                    tangent,
                    tangent,
                    sample_count,
                )
                np.testing.assert_allclose(
                    controller.active_path.x,
                    expected[:, 0],
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    controller.active_path.y,
                    expected[:, 1],
                    atol=1e-12,
                )

    def test_rotation_is_zero_linear_and_obeys_rate_and_slew_limits(self):
        cases = (
            (
                "left_into_space",
                LEFT,
                "TURN_TO_SPACE",
                -0.5 * math.pi,
                0.0,
                1.0,
            ),
            (
                "right_into_space",
                RIGHT,
                "TURN_TO_SPACE",
                -0.5 * math.pi,
                math.pi,
                -1.0,
            ),
            (
                "left_to_exit",
                LEFT,
                "TURN_TO_EXIT",
                0.0,
                0.5 * math.pi,
                1.0,
            ),
            (
                "right_to_exit",
                RIGHT,
                "TURN_TO_EXIT",
                math.pi,
                0.5 * math.pi,
                -1.0,
            ),
        )
        for (
            label,
            selected_space,
            state_name,
            start_yaw,
            target_yaw,
            expected_sign,
        ) in cases:
            with self.subTest(label=label):
                controller = self.make_controller()
                controller.mission_has_control = True
                controller.selected_space = selected_space
                controller.route_transform = self.rigid_route_transform()
                turn = (
                    controller.aisle_x,
                    controller.decision_y,
                    controller.aisle_heading,
                )
                centre = (turn[0], turn[1], start_yaw)
                self.update_poses(controller, centre)
                self.assertTrue(
                    controller._begin_rotation(
                        getattr(controller, state_name), target_yaw
                    )
                )
                initial_diagnostics = self.publishers[
                    controller.diagnostics_topic
                ].messages[-1].data
                self.assertEqual(len(initial_diagnostics), 13)
                self.assertTrue(
                    all(math.isfinite(value) for value in initial_diagnostics[:8])
                )
                self.assertEqual(initial_diagnostics[2], 0.0)
                self.assertEqual(initial_diagnostics[3], 0.0)

                previous_angular = 0.0
                for _ in range(2):
                    self.advance(0.10)
                    controller.odom_callback(self.odom_message(centre))
                    controller.control_callback(None)
                    command = self.publishers[controller.cmd_vel_topic].messages[-1]
                    self.assertEqual(command.linear.x, 0.0)
                    self.assertEqual(
                        math.copysign(1.0, command.angular.z), expected_sign
                    )
                    self.assertLessEqual(
                        abs(command.angular.z), controller.rotation_velocity + 1e-12
                    )
                    self.assertLessEqual(
                        abs(command.angular.z - previous_angular),
                        controller.angular_acceleration * 0.10 + 1e-12,
                    )
                    previous_angular = command.angular.z

                diagnostics = self.publishers[
                    controller.diagnostics_topic
                ].messages[-1].data
                self.assertEqual(len(diagnostics), 13)
                self.assertEqual(diagnostics[11], 0.0)
                self.assertAlmostEqual(diagnostics[12], previous_angular)

    def test_rotation_revalidates_from_measured_pose_after_physical_drift(self):
        """Regress Run 6's safe 8 mm Gazebo spin drift."""
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.selected_space = RIGHT
        controller.route_transform = self.rigid_route_transform()
        centre = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.update_poses(controller, centre)
        self.assertTrue(
            controller._begin_rotation(
                controller.TURN_TO_SPACE,
                math.pi,
            )
        )

        # A differential-drive spin can translate slightly in Gazebo and on
        # the real floor even though every command has exactly zero linear
        # velocity.  The live swept-footprint check starts at this measured
        # pose, and the subsequent parking path is anchored at the measured
        # rotation endpoint, so cumulative drift alone is not a failure.
        drifted_pose = (
            centre[0] - 0.0085,
            centre[1] - 0.0060,
            centre[2] - math.radians(35.0),
        )
        self.advance(controller.control_period)
        controller.odom_callback(self.odom_message(drifted_pose))
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.TURN_TO_SPACE)
        self.assertTrue(controller.active_path_validation.safe)
        command = self.publishers[controller.cmd_vel_topic].messages[-1]
        self.assertEqual(command.linear.x, 0.0)
        diagnostics = self.publishers[
            controller.diagnostics_topic
        ].messages[-1].data
        self.assertAlmostEqual(
            diagnostics[2],
            math.hypot(0.0085, 0.0060),
        )

    def test_rotation_obstacle_margin_does_not_weaken_line_uncertainty(self):
        self.param_overrides.update(
            {
                "~parking/safety/localization_margin": 0.001,
                "~parking/safety/tracking_margin": 0.001,
                "~parking/safety/obstacle_margin": 0.009,
                "~parking/safety/rotation_obstacle_margin": 0.0,
            }
        )
        controller = self.make_controller()

        ordinary = controller._path_safety("parking")
        rotation = controller._rotation_path_safety()

        self.assertAlmostEqual(
            ordinary.margins.obstacle + ordinary.margins.uncertainty,
            0.011,
        )
        self.assertEqual(rotation.margins.obstacle, 0.0)
        self.assertEqual(rotation.margins.uncertainty, 0.0)
        self.assertAlmostEqual(
            rotation.margins.line,
            ordinary.margins.line + ordinary.margins.uncertainty,
        )

    def test_rotation_wrap_uses_the_shortest_direction_both_ways(self):
        cases = (
            (math.radians(179.0), math.radians(-179.0), 1.0),
            (math.radians(-179.0), math.radians(179.0), -1.0),
        )
        for current_yaw, target_yaw, expected_sign in cases:
            with self.subTest(
                current_yaw=current_yaw, target_yaw=target_yaw
            ):
                controller = self.make_controller()
                controller.mission_has_control = True
                controller.selected_space = LEFT
                controller.route_transform = self.rigid_route_transform()
                centre = (
                    controller.aisle_x,
                    controller.decision_y,
                    controller.aisle_heading,
                )
                pose = (centre[0], centre[1], current_yaw)
                self.update_poses(controller, pose)
                self.assertTrue(
                    controller._begin_rotation(
                        controller.TURN_TO_SPACE, target_yaw
                    )
                )

                self.advance(0.10)
                controller.odom_callback(self.odom_message(pose))
                controller.control_callback(None)

                command = self.publishers[controller.cmd_vel_topic].messages[-1]
                self.assertEqual(command.linear.x, 0.0)
                self.assertEqual(
                    math.copysign(1.0, command.angular.z), expected_sign
                )

    def test_static_full_rotation_sweep_rejects_an_endpoint_obstacle(self):
        self.param_overrides[
            "~parking/safety/rotation_obstacle_margin"
        ] = 0.0
        for selected_space, target_yaw, obstacle_dx in (
            (LEFT, 0.0, -0.11),
            (RIGHT, math.pi, 0.11),
        ):
            with self.subTest(selected_space=selected_space):
                controller = self.make_controller()
                controller.mission_has_control = True
                controller.selected_space = selected_space
                controller.route_transform = self.rigid_route_transform()
                centre = (
                    controller.aisle_x,
                    controller.decision_y,
                    controller.aisle_heading,
                )
                self.update_poses(controller, centre)
                controller.fixed_obstacle_points_route = np.asarray(
                    [[centre[0] + obstacle_dx, centre[1]]], dtype=np.float64
                )
                safety = controller._path_safety("parking")
                start_only = controller.path_validator.validate_poses(
                    (Pose2D(*centre),), safety=safety
                )
                self.assertTrue(start_only.safe)

                self.assertIsNone(
                    controller._begin_rotation(
                        controller.TURN_TO_SPACE, target_yaw
                    )
                )
                self.assertEqual(controller.state, controller.FAILED)
                command = self.publishers[controller.cmd_vel_topic].messages[-1]
                self.assertEqual(command.linear.x, 0.0)
                self.assertEqual(command.angular.z, 0.0)

    def test_live_full_rotation_sweep_holds_zero_for_a_future_obstacle(self):
        self.param_overrides["~parking/safety/require_live_scan"] = True
        for selected_space, target_yaw, obstacle_dx in (
            (LEFT, 0.0, -0.11),
            (RIGHT, math.pi, 0.11),
        ):
            with self.subTest(selected_space=selected_space):
                controller = self.make_controller()
                controller.mission_has_control = True
                controller.selected_space = selected_space
                controller.route_transform = self.rigid_route_transform()
                centre = (
                    controller.aisle_x,
                    controller.decision_y,
                    controller.aisle_heading,
                )
                self.update_poses(controller, centre)
                self.assertTrue(
                    controller._begin_rotation(
                        controller.TURN_TO_SPACE, target_yaw
                    )
                )
                controller.live_obstacle_points_odom = np.asarray(
                    [[centre[0] + obstacle_dx, centre[1]]], dtype=np.float64
                )
                controller.safety_scan_stamp = self.now()
                controller.safety_scan_received = self.now()

                self.advance(0.05)
                controller.odom_callback(self.odom_message(centre))
                controller.control_callback(None)

                self.assertEqual(controller.state, controller.TURN_TO_SPACE)
                self.assertFalse(controller.active_path_validation.safe)
                command = self.publishers[controller.cmd_vel_topic].messages[-1]
                self.assertEqual(command.linear.x, 0.0)
                self.assertEqual(command.angular.z, 0.0)
                diagnostics = self.publishers[
                    controller.diagnostics_topic
                ].messages[-1].data
                self.assertLessEqual(diagnostics[9], 0.0)
                self.assertEqual(diagnostics[11], 0.0)
                self.assertEqual(diagnostics[12], 0.0)

    def test_rotation_holds_zero_for_stale_scan_or_stale_odometry(self):
        self.param_overrides["~parking/safety/require_live_scan"] = True
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.selected_space = LEFT
        controller.route_transform = self.rigid_route_transform()
        centre = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.update_poses(controller, centre)
        self.assertTrue(
            controller._begin_rotation(controller.TURN_TO_SPACE, 0.0)
        )
        self.advance(0.05)
        controller.odom_callback(self.odom_message(centre))
        controller.control_callback(None)
        command = self.publishers[controller.cmd_vel_topic].messages[-1]
        self.assertEqual(command.linear.x, 0.0)
        self.assertEqual(command.angular.z, 0.0)
        self.assertEqual(controller.state, controller.TURN_TO_SPACE)

        self.param_overrides.clear()
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.selected_space = RIGHT
        controller.route_transform = self.rigid_route_transform()
        centre = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.update_poses(controller, centre)
        self.assertTrue(
            controller._begin_rotation(controller.TURN_TO_SPACE, math.pi)
        )
        self.advance(controller.odom_timeout + 0.01)
        controller.control_callback(None)
        command = self.publishers[controller.cmd_vel_topic].messages[-1]
        self.assertEqual(command.linear.x, 0.0)
        self.assertEqual(command.angular.z, 0.0)
        self.assertEqual(controller.state, controller.TURN_TO_SPACE)

    def test_rotation_requires_actual_stop_and_two_distinct_odom_samples(self):
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.selected_space = LEFT
        controller.route_transform = self.rigid_route_transform()
        centre = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.update_poses(controller, centre)
        self.assertTrue(
            controller._begin_rotation(controller.TURN_TO_SPACE, 0.0)
        )
        target = (centre[0], centre[1], 0.0)

        self.advance(controller.control_period)
        controller.odom_callback(
            self.odom_message(
                target,
                angular_z=controller.rotation_stopped_velocity + 0.01,
            )
        )
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.TURN_TO_SPACE)
        self.assertEqual(controller.odom_confirmation_count, 0)

        self.advance(controller.control_period)
        controller.odom_callback(self.odom_message(target))
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.TURN_TO_SPACE)
        self.assertEqual(controller.odom_confirmation_count, 1)

        controller.control_callback(None)
        self.assertEqual(controller.state, controller.TURN_TO_SPACE)
        self.assertEqual(controller.odom_confirmation_count, 1)

        self.advance(controller.control_period)
        controller.odom_callback(self.odom_message(target))
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.PARK_IN)
        self.assertIsNotNone(controller.parking_return_odom)

    def test_park_and_backout_complete_without_display_or_prepare_holds(self):
        for selected_space, bay_yaw in ((LEFT, 0.0), (RIGHT, math.pi)):
            with self.subTest(selected_space=selected_space):
                controller = self.make_controller()
                controller.mission_has_control = True
                controller.selected_space = selected_space
                controller.route_transform = self.rigid_route_transform()
                turn = (
                    controller.aisle_x,
                    controller.decision_y,
                    controller.aisle_heading,
                )
                return_pose = (turn[0], turn[1], bay_yaw)
                park_x = (
                    controller.left_park_x
                    if selected_space == LEFT
                    else controller.right_park_x
                )
                park_pose = (park_x, turn[1], bay_yaw)
                controller.parking_return_map = return_pose
                controller.parking_return_odom = return_pose
                controller.parking_return_amcl = return_pose
                self.update_poses(controller, return_pose)
                self.assertTrue(
                    controller._begin_drive(
                        controller.PARK_IN,
                        park_pose,
                        1,
                        controller.parking_speed,
                    )
                )

                for index in range(controller.pose_confirm_samples):
                    self.advance(controller.control_period)
                    self.update_poses(
                        controller, controller.goal_map, controller.goal_odom
                    )
                    controller.control_callback(None)
                    if index + 1 < controller.pose_confirm_samples:
                        self.assertEqual(controller.state, controller.PARK_IN)
                self.assertEqual(controller.state, controller.BACK_OUT)
                self.assertTrue(np.all(controller.active_path.direction == -1))
                self.assert_pose_almost_equal(
                    controller.goal_odom, controller.parking_return_odom
                )

                for index in range(controller.pose_confirm_samples):
                    self.advance(controller.control_period)
                    self.update_poses(
                        controller, controller.goal_map, controller.goal_odom
                    )
                    controller.control_callback(None)
                    if index + 1 < controller.pose_confirm_samples:
                        self.assertEqual(controller.state, controller.BACK_OUT)
                self.assertEqual(controller.state, controller.TURN_TO_EXIT)
                state_messages = [
                    message.data
                    for message in self.publishers["/parking/state"].messages
                ]
                self.assertNotIn("PARKED", state_messages)
                self.assertNotIn("PREPARE_EXIT", state_messages)

    def test_both_branches_complete_the_controller_sequence(self):
        expected_states = (
            "PREPARE_APPROACH",
            "APPROACH",
            "TURN_IN",
            "ENTER_AISLE",
            "SELECT_SPACE",
            "TURN_TO_SPACE",
            "PARK_IN",
            "BACK_OUT",
            "TURN_TO_EXIT",
            "LEAVE_AISLE",
            "TURN_TO_ZIGZAG",
            "VERIFY_ZIGZAG_LANE",
            "JOIN_ZIGZAG",
            "COMPLETE",
        )
        for selected_space in (LEFT, RIGHT):
            with self.subTest(selected_space=selected_space):
                controller = self.start_controller(self.make_controller())

                self.confirm_current_goal(controller)
                self.assertEqual(controller.state, controller.TURN_IN)
                self.confirm_current_goal(controller)
                self.assertEqual(controller.state, controller.ENTER_AISLE)
                self.confirm_current_goal(controller)
                self.assertEqual(controller.state, controller.SELECT_SPACE)
                self.confirm_space_selection(controller, selected_space)
                self.assertEqual(controller.state, controller.TURN_TO_SPACE)
                self.confirm_rotation(controller)
                self.assertEqual(controller.state, controller.PARK_IN)
                saved_return = controller.parking_return_odom

                self.confirm_current_goal(controller)
                self.assertEqual(controller.state, controller.BACK_OUT)
                self.assertTrue(np.all(controller.active_path.direction == -1))
                self.assert_pose_almost_equal(controller.goal_odom, saved_return)
                self.confirm_current_goal(controller)
                self.assertEqual(controller.state, controller.TURN_TO_EXIT)
                self.confirm_rotation(controller)
                self.assertEqual(controller.state, controller.LEAVE_AISLE)
                self.confirm_current_goal(controller)
                self.assertEqual(controller.state, controller.TURN_TO_ZIGZAG)
                terminal_speed = float(
                    controller.zigzag_exit_curve_path.speed[-1]
                )
                self.assertAlmostEqual(
                    terminal_speed,
                    controller.profile_exit_velocity,
                    places=12,
                )
                maximum_terminal_overrun = (
                    terminal_speed * controller.control_period
                    + terminal_speed * terminal_speed
                    / (2.0 * controller.linear_deceleration)
                )
                self.assertGreater(
                    controller.goal_map[0] - maximum_terminal_overrun,
                    controller.handoff_min_x,
                )
                self.confirm_current_goal(controller)
                self.assertEqual(
                    controller.state, controller.VERIFY_ZIGZAG_LANE
                )

                handoff_map = controller.goal_map
                handoff_odom = controller.goal_odom
                for _ in range(controller.handoff_confirm_frames):
                    self.advance(0.05)
                    self.update_poses(controller, handoff_map, handoff_odom)
                    controller.lane_path_diagnostics_callback(
                        self.lane_path_message()
                    )
                    controller.control_callback(None)
                self.assertEqual(controller.state, controller.JOIN_ZIGZAG)
                self.assertFalse(controller.mission_has_control)

                completion_map = (
                    0.5 * (controller.rejoin_min_x + controller.rejoin_max_x),
                    0.5 * (controller.rejoin_min_y + controller.rejoin_max_y),
                    controller.zigzag_heading,
                )
                completion_odom = self.route_pose_to_odom(
                    completion_map, controller.route_transform
                )
                for _ in range(controller.rejoin_confirm_frames):
                    self.advance(0.05)
                    self.update_poses(
                        controller, completion_map, completion_odom
                    )
                    controller.lane_path_diagnostics_callback(
                        self.lane_path_message()
                    )
                    controller.control_callback(None)
                controller.control_callback(None)
                self.assertEqual(controller.state, controller.COMPLETE)
                self.assertEqual(self.lane_service.calls, [False, True])

                state_messages = [
                    message.data
                    for message in self.publishers["/parking/state"].messages
                    if message.data in expected_states
                ]
                cursor = 0
                for state in state_messages:
                    if cursor < len(expected_states) and state == expected_states[cursor]:
                        cursor += 1
                self.assertEqual(cursor, len(expected_states))
                self.assertNotIn("PARKED", state_messages)
                self.assertNotIn("PREPARE_EXIT", state_messages)

    def test_moving_zigzag_handoff_preserves_forward_motion_and_sole_owner(self):
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.selected_space = LEFT
        controller.route_transform = self.rigid_route_transform()
        curve_start = controller._zigzag_turn_start_pose()
        self.update_poses(controller, curve_start, curve_start)
        self.assertTrue(controller._build_zigzag_exit_curve())
        self.assertTrue(controller._begin_zigzag_exit_curve())
        controller.path_follower.last_linear = 0.07
        controller.last_linear = 0.07
        controller.path_follower.last_angular = 0.0
        controller.last_angular = 0.0
        controller.last_command_time = self.now()

        commands = self.publishers[controller.cmd_vel_topic].messages
        commands.clear()
        self.publishers[controller.speed_limit_topic].messages.clear()
        self.events.clear()
        self.lane_service.calls.clear()

        for index in range(controller.handoff_confirm_frames):
            self.advance(controller.control_period)
            pose = (
                controller.handoff_max_x - 0.005 * (index + 1),
                controller.zigzag_straight_y,
                controller.zigzag_heading,
            )
            self.update_poses(controller, pose, pose)
            controller.lane_path_diagnostics_callback(
                self.lane_path_message()
            )
            controller.control_callback(None)
            if index + 1 < controller.handoff_confirm_frames:
                self.assertEqual(controller.state, controller.TURN_TO_ZIGZAG)
                self.assertEqual(self.lane_service.calls, [])

        self.assertEqual(controller.state, controller.JOIN_ZIGZAG)
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(self.lane_service.calls, [True])
        self.assertTrue(commands)
        self.assertTrue(all(command.linear.x > 0.0 for command in commands))

        service_index = next(
            index
            for index, event in enumerate(self.events)
            if event[0] == "service" and event[1] is True
        )
        limit_index = next(
            index
            for index, event in enumerate(self.events)
            if event[0] == "publish"
            and event[1] == controller.speed_limit_topic
        )
        command_indices = [
            index
            for index, event in enumerate(self.events)
            if event[0] == "publish" and event[1] == controller.cmd_vel_topic
        ]
        self.assertTrue(command_indices)
        self.assertLess(max(command_indices), limit_index)
        self.assertLess(limit_index, service_index)

        command_count = len(commands)
        self.advance(controller.control_period)
        controller.control_callback(None)
        self.assertEqual(len(commands), command_count)

    def test_zigzag_handoff_and_completion_ignore_live_amcl_jump(self):
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.selected_space = LEFT
        controller.route_transform = self.rigid_route_transform(
            0.12, -0.08, math.radians(3.0)
        )
        controller.state = controller.VERIFY_ZIGZAG_LANE
        controller.state_started = self.now()

        handoff_map = (
            0.5 * (controller.handoff_min_x + controller.handoff_max_x),
            0.5 * (controller.handoff_min_y + controller.handoff_max_y),
            controller.zigzag_heading,
        )
        handoff_odom = self.route_pose_to_odom(
            handoff_map, controller.route_transform
        )
        jumped_map = (
            handoff_map[0] + 0.21,
            handoff_map[1] + 0.21,
            handoff_map[2],
        )
        controller.map_pose_callback(self.pose_message(jumped_map))
        self.advance(controller.pose_timeout + 0.01)

        # AMCL is stale and translated outside the handoff box. Fresh frozen-
        # route odom and three new camera frames still agree with the pose.
        for _ in range(controller.handoff_confirm_frames):
            self.advance(0.05)
            controller.odom_callback(self.odom_message(handoff_odom))
            controller.lane_path_diagnostics_callback(
                self.lane_path_message(
                    commanded_linear=0.0,
                    commanded_angular=0.0,
                )
            )
            controller.control_callback(None)
        self.assertEqual(controller.state, controller.JOIN_ZIGZAG)
        self.assertEqual(self.lane_service.calls, [True])

        completion_map = (
            0.5 * (controller.rejoin_min_x + controller.rejoin_max_x),
            0.5 * (controller.rejoin_min_y + controller.rejoin_max_y),
            controller.zigzag_heading,
        )
        completion_odom = self.route_pose_to_odom(
            completion_map, controller.route_transform
        )
        for _ in range(controller.rejoin_confirm_frames):
            self.advance(0.05)
            controller.odom_callback(self.odom_message(completion_odom))
            controller.lane_path_diagnostics_callback(
                self.lane_path_message()
            )
            controller.control_callback(None)
        self.assertEqual(controller.state, controller.COMPLETE)
        self.assertEqual(self.lane_service.calls, [True])

    def test_zigzag_completion_survives_one_observed_camera_fit_gap(self):
        controller = self.make_controller()
        controller.route_transform = self.rigid_route_transform()
        controller.state = controller.JOIN_ZIGZAG
        controller.state_started = self.now()

        # Official-start run 13 produced eight safe frames, then one rejected
        # fit gap.  The fresh-frame counter must restart, while the remaining
        # surveyed straight still provides room for all nine confirmations.
        for index in range(controller.rejoin_confirm_frames - 1):
            self.advance(0.04)
            pose = (
                controller.rejoin_max_x - 0.004 * index,
                controller.zigzag_straight_y,
                controller.zigzag_heading,
            )
            self.update_poses(controller, pose, pose)
            controller.lane_path_diagnostics_callback(
                self.lane_path_message()
            )
            controller.control_callback(None)
        self.assertEqual(controller.state, controller.JOIN_ZIGZAG)

        self.advance(controller.rejoin_confirmation_max_gap + 0.11)
        for index in range(controller.rejoin_confirm_frames):
            self.advance(0.04)
            pose = (
                0.152 - 0.008 * index,
                controller.zigzag_straight_y,
                controller.zigzag_heading,
            )
            self.update_poses(controller, pose, pose)
            controller.lane_path_diagnostics_callback(
                self.lane_path_message()
            )
            controller.control_callback(None)

        self.assertEqual(controller.state, controller.COMPLETE)

    def test_zigzag_completion_accepts_verified_preview_heading(self):
        controller = self.make_controller()
        controller.route_transform = self.rigid_route_transform()
        controller.state = controller.JOIN_ZIGZAG
        controller.state_started = self.now()
        controller.rejoin_heading_tolerance = math.radians(6.0)

        # Official LEFT run 17 reached this straight window with positive
        # swept-line clearance while the speed-adaptive camera lookahead had
        # already begun the next bend. Its 5.7-degree preview heading remains
        # strictly inside Zigzag's separate 7-degree acquisition envelope.
        pose = (
            0.17,
            controller.zigzag_straight_y,
            controller.zigzag_heading + math.radians(5.7),
        )
        for _ in range(controller.rejoin_confirm_frames):
            self.advance(0.04)
            self.update_poses(controller, pose, pose)
            controller.lane_path_diagnostics_callback(
                self.lane_path_message(minimum_line_clearance=0.004)
            )
            controller.control_callback(None)

        self.assertEqual(controller.state, controller.COMPLETE)

    def test_zigzag_completion_rejects_heading_outside_preview_envelope(self):
        controller = self.make_controller()
        controller.route_transform = self.rigid_route_transform()
        controller.state = controller.JOIN_ZIGZAG
        controller.state_started = self.now()
        controller.rejoin_heading_tolerance = math.radians(6.0)
        pose = (
            0.17,
            controller.zigzag_straight_y,
            controller.zigzag_heading + math.radians(6.1),
        )

        for _ in range(controller.rejoin_confirm_frames):
            self.advance(0.04)
            self.update_poses(controller, pose, pose)
            controller.lane_path_diagnostics_callback(
                self.lane_path_message(minimum_line_clearance=0.004)
            )
            controller.control_callback(None)

        self.assertEqual(controller.state, controller.JOIN_ZIGZAG)

    def test_confirmed_zigzag_handoff_wins_on_timeout_edge(self):
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.selected_space = LEFT
        controller.route_transform = self.rigid_route_transform()
        controller.state = controller.VERIFY_ZIGZAG_LANE
        controller.state_started = self.now()
        handoff_pose = (
            0.5 * (controller.handoff_min_x + controller.handoff_max_x),
            0.5 * (controller.handoff_min_y + controller.handoff_max_y),
            controller.zigzag_heading,
        )

        for _ in range(controller.handoff_confirm_frames):
            self.advance(0.01)
            controller.odom_callback(self.odom_message(handoff_pose))
            controller.lane_path_diagnostics_callback(
                self.lane_path_message(
                    commanded_linear=0.0,
                    commanded_angular=0.0,
                )
            )

        controller.state_started = self.now() - controller_module.rospy.Duration(
            controller.handoff_timeout + 0.001
        )
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.JOIN_ZIGZAG)
        self.assertEqual(self.lane_service.calls, [True])


    def test_handoff_failures_never_publish_complete(self):
        controller = self.make_controller()
        self.lane_service.success = False
        controller.gate_callback(Bool(data=True))
        self.update_poses(controller, (0.90, 1.75, math.pi))
        self.lane_command(controller)
        controller.control_callback(None)
        self.advance(controller.prepare_settle_time + 0.01)
        self.update_poses(controller, (0.90, 1.75, math.pi))
        self.lane_command(controller)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.FAILED)
        self.assertFalse(controller.mission_has_control)

        controller = self.make_controller()
        controller.state = controller.JOIN_ZIGZAG
        controller.mission_has_control = True
        controller.selected_space = LEFT
        self.lane_service.success = False
        controller._complete()
        self.assertEqual(controller.state, controller.FAILED)
        states = [message.data for message in self.publishers["/parking/state"].messages]
        self.assertNotIn(controller.COMPLETE, states)
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(self.publishers["/cmd_vel"].messages, [])
        emergency = self.publishers["/control/manual_stop"].messages
        self.assertEqual(len(emergency), 1)
        self.assertTrue(emergency[0].data)

    def test_manual_stop_and_stale_amcl_hold_zero_without_advancing(self):
        controller = self.start_controller(self.make_controller())
        state = controller.state

        controller.manual_stop_callback(Bool(data=True))
        self.advance(2.0)
        controller.control_callback(None)
        self.assertEqual(controller.state, state)
        self.assertEqual(self.publishers["/cmd_vel"].messages[-1].linear.x, 0.0)

        controller.scan_generation = 7
        controller.manual_stop_callback(Bool(data=False))
        self.assertEqual(controller.state_started, self.now())
        self.assertEqual(controller.last_selection_scan_generation, 7)

        # Resuming does not make the old AMCL/odom data fresh. The controller
        # must remain in the same phase and continue holding zero.
        self.advance(controller.pose_timeout + 0.01)
        controller.control_callback(None)
        self.assertEqual(controller.state, state)
        self.assertEqual(self.publishers["/cmd_vel"].messages[-1].linear.x, 0.0)

    def test_stale_scan_holds_zero_in_space_selection(self):
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.state = controller.SELECT_SPACE
        controller.state_started = self.now()
        aisle_pose = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        self.update_poses(controller, aisle_pose)
        controller.left_points = controller.occupied_minimum_points
        controller.right_points = 0
        controller.scan_generation = 1
        controller.scan_stamp = controller_module.rospy.Time.from_sec(
            self.seconds - controller.scan_timeout - 0.01
        )
        # Receipt is current, but the source stamp is stale: a delayed scan
        # must not be accepted as a new parking-space observation.
        controller.scan_received = self.now()

        controller.control_callback(None)

        self.assertEqual(controller.state, controller.SELECT_SPACE)
        self.assertIsNone(controller.selected_space)
        self.assertEqual(self.publishers["/cmd_vel"].messages[-1].linear.x, 0.0)

    def test_selection_uses_fresh_amcl_projected_scan_after_translation_update(self):
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.state = controller.SELECT_SPACE
        controller.state_started = self.now()

        # ENTER_AISLE has already reached its odometry endpoint. AMCL may make
        # a translational scan correction there, but its fresh pose and aisle
        # heading are still valid for projecting returns into the bay ROIs.
        corrected_pose = (
            controller.aisle_x + 0.01,
            controller.decision_y - 2.0 * controller.checkpoint_position_tolerance,
            controller.aisle_heading,
        )
        stopped_odom_pose = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading,
        )
        controller.route_transform = self.rigid_route_transform()
        self.update_poses(controller, corrected_pose, stopped_odom_pose)
        self.assertGreater(
            math.hypot(
                corrected_pose[0] - controller.aisle_x,
                corrected_pose[1] - controller.decision_y,
            ),
            controller.checkpoint_position_tolerance,
        )

        for _ in range(controller.selection_confirm_scans):
            self.advance(0.05)
            controller.left_points = 0
            controller.right_points = controller.occupied_minimum_points
            controller.scan_generation += 1
            controller.scan_stamp = self.now()
            controller.scan_received = self.now()
            controller.control_callback(None)

        self.assertEqual(controller.selected_space, LEFT)
        self.assertEqual(controller.state, controller.TURN_TO_SPACE)

    def test_selection_rejects_scan_when_amcl_heading_is_outside_aisle(self):
        controller = self.make_controller()
        controller.mission_has_control = True
        controller.state = controller.SELECT_SPACE
        controller.state_started = self.now()
        wrong_heading_pose = (
            controller.aisle_x,
            controller.decision_y,
            controller.aisle_heading
            + controller.checkpoint_heading_tolerance
            + math.radians(0.5),
        )
        self.update_poses(controller, wrong_heading_pose)
        controller.left_points = 0
        controller.right_points = controller.occupied_minimum_points
        controller.scan_generation = 1
        controller.scan_stamp = self.now()
        controller.scan_received = self.now()

        issue = controller._selection_input_issue(self.now())
        controller.control_callback(None)

        self.assertIn("AMCL aisle heading error", issue)
        self.assertEqual(controller.state, controller.SELECT_SPACE)
        self.assertIsNone(controller.selected_space)


if __name__ == "__main__":
    unittest.main()
