#!/usr/bin/env python3

"""Focused regressions for parking-local registration and ordered handoff."""

import copy
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

import parking_mission_controller as controller_module
from parking_mission_controller import ParkingMissionController
from custom_autorace_bringup.parking_geometry import LEFT, RIGHT
from custom_autorace_bringup.path_following import Pose2D, RigidTransform2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Header


class RecordingPublisher:
    def __init__(self, topic, events):
        self.topic = topic
        self.events = events
        self.messages = []

    def publish(self, message):
        snapshot = copy.deepcopy(message)
        self.messages.append(snapshot)
        self.events.append(("publish", self.topic, snapshot))


class RecordingService:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def __call__(self, enabled):
        value = bool(enabled)
        self.calls.append(value)
        self.events.append(("service", value, None))
        return SimpleNamespace(success=True, message="ok")


class ParkingLocalRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.seconds = 10.0
        self.events = []
        self.publishers = {}
        self.publisher_options = {}
        self.services = []
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
                side_effect=self._service_factory,
            ),
            mock.patch.object(
                controller_module.rospy, "Timer", return_value=SimpleNamespace()
            ),
            mock.patch.object(controller_module.rospy, "on_shutdown"),
            mock.patch.object(controller_module.rospy, "wait_for_service"),
            mock.patch.object(controller_module.rospy, "loginfo"),
            mock.patch.object(controller_module.rospy, "loginfo_throttle"),
            mock.patch.object(controller_module.rospy, "logwarn"),
            mock.patch.object(controller_module.rospy, "logwarn_throttle"),
            mock.patch.object(controller_module.rospy, "logerr"),
            mock.patch.object(controller_module.rospy, "logfatal"),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _publisher_factory(self, topic, *_args, **options):
        publisher = RecordingPublisher(topic, self.events)
        self.publishers[topic] = publisher
        self.publisher_options[topic] = options
        return publisher

    def _service_factory(self, *_args, **_kwargs):
        service = RecordingService(self.events)
        self.services.append(service)
        return service

    def now(self):
        return controller_module.rospy.Time.from_sec(self.seconds)

    def advance(self, seconds=0.05):
        self.seconds += seconds

    def make_controller(self):
        controller = ParkingMissionController()
        self.events.clear()
        for publisher in self.publishers.values():
            publisher.messages.clear()
        for service in self.services:
            service.calls.clear()
        return controller

    def arm(self, controller, sequence=17):
        message = Header()
        message.seq = sequence
        message.stamp = self.now()
        message.frame_id = "parking"
        controller.arm_callback(message)
        return message

    @staticmethod
    def odom_message(pose, stamp):
        x, y, yaw = pose
        message = Odometry()
        message.header.stamp = stamp
        message.pose.pose.position.x = x
        message.pose.pose.position.y = y
        message.pose.pose.orientation.z = math.sin(0.5 * yaw)
        message.pose.pose.orientation.w = math.cos(0.5 * yaw)
        return message

    @staticmethod
    def transform_pose(transform, pose):
        result = transform.apply_pose(Pose2D(*pose))
        return result.x, result.y, result.yaw

    @staticmethod
    def transformed_fixture_segments(controller, transform):
        values = []
        for landmark in controller.registration_landmarks[:2]:
            start = transform.apply_point(landmark.start)
            end = transform.apply_point(landmark.end)
            values.append(
                (
                    start,
                    end,
                    math.hypot(end[0] - start[0], end[1] - start[1]),
                )
            )
        return tuple(values)

    @staticmethod
    def assert_transform_close(test_case, actual, expected, places=7):
        test_case.assertIsNotNone(actual)
        test_case.assertAlmostEqual(
            actual.target_from_source_x,
            expected.target_from_source_x,
            places=places,
        )
        test_case.assertAlmostEqual(
            actual.target_from_source_y,
            expected.target_from_source_y,
            places=places,
        )
        error = controller_module.normalize_angle(
            actual.target_from_source_yaw
            - expected.target_from_source_yaw
        )
        test_case.assertAlmostEqual(error, 0.0, places=places)

    def freeze_registration(self, controller, transform):
        local_robot = (0.92, 1.75, math.pi)
        odom_robot = self.transform_pose(transform, local_robot)
        segments = self.transformed_fixture_segments(controller, transform)
        controller._extract_registration_segments = mock.Mock(
            return_value=segments
        )

        for confirmation in range(1, 4):
            self.advance(0.05)
            scan = {"source_stamp": self.now()}
            controller._process_registration_scan(scan, odom_robot)
            self.assertEqual(
                controller.registration_confirmation_count, confirmation
            )
            if confirmation < 3:
                self.assertIsNone(controller.local_to_odom)
        return odom_robot

    def prepare_ready(self, controller, transform, sequence=17):
        self.arm(controller, sequence=sequence)
        controller.local_to_odom = transform
        controller.registration_source_stamp = self.now()

        entry = controller.registration_template.entry_plane
        local_pose = (
            entry.point[0] - entry.normal[0] * 0.20,
            entry.point[1] - entry.normal[1] * 0.20,
            entry.heading,
        )
        odom_pose = self.transform_pose(transform, local_pose)
        self.advance(0.02)
        controller.odom_callback(self.odom_message(odom_pose, self.now()))

        def build_preflight(preflight=False):
            self.assertTrue(preflight)
            controller.entry_connector_path = SimpleNamespace(label="entry")
            controller.entry_curve_path = SimpleNamespace(label="entry_turn")
            return True

        controller._build_adaptive_entry = mock.Mock(
            side_effect=build_preflight
        )
        return odom_pose

    def test_two_nonparallel_faces_freeze_on_third_source_stamped_scan(self):
        controller = self.make_controller()
        self.arm(controller)
        expected = RigidTransform2D(
            2.35, -1.10, math.radians(21.0), "parking_local", "odom"
        )

        self.freeze_registration(controller, expected)

        self.assert_transform_close(self, controller.local_to_odom, expected)
        self.assertEqual(controller.registration_confirmation_count, 3)
        self.assertEqual(controller.registration_source_stamp, self.now())
        self.assertEqual(
            np.asarray(controller.registration_covariance).shape, (3, 3)
        )
        self.assertEqual(
            controller.local_to_odom.source_frame, controller.route_frame
        )
        self.assertEqual(
            controller.local_to_odom.target_frame, "odom"
        )

    def test_configured_two_scan_confirmation_freezes_on_second_scan(self):
        self.param_overrides[
            "~parking/registration/confirmation_scans"
        ] = 2
        controller = self.make_controller()
        self.arm(controller)
        identity = RigidTransform2D(
            0.0, 0.0, 0.0, "parking_local", "odom"
        )
        robot = (0.92, 1.75, math.pi)
        controller._extract_registration_segments = mock.Mock(
            return_value=self.transformed_fixture_segments(
                controller, identity
            )
        )

        for expected_count in (1, 2):
            self.advance(0.05)
            controller._process_registration_scan(
                {"source_stamp": self.now()}, robot
            )
            self.assertEqual(
                controller.registration_confirmation_count,
                expected_count,
            )

        self.assert_transform_close(self, controller.local_to_odom, identity)

    def test_sparse_face_miss_preserves_only_fresh_confirmation_streak(self):
        controller = self.make_controller()
        self.arm(controller)
        identity = RigidTransform2D(
            0.0, 0.0, 0.0, "parking_local", "odom"
        )
        robot = (0.92, 1.75, math.pi)
        segments = self.transformed_fixture_segments(controller, identity)
        controller._extract_registration_segments = mock.Mock(
            side_effect=(segments, tuple(), segments, tuple(), segments)
        )

        expected_counts = (1, 1, 2, 2, 3)
        for expected in expected_counts:
            self.advance(0.05)
            controller._process_registration_scan(
                {"source_stamp": self.now()}, robot
            )
            self.assertEqual(
                controller.registration_confirmation_count, expected
            )

        self.assert_transform_close(self, controller.local_to_odom, identity)

        controller._reset_local_registration("gap regression")
        self.arm(controller, sequence=18)
        controller._extract_registration_segments = mock.Mock(
            side_effect=(segments, tuple())
        )
        self.advance(0.05)
        controller._process_registration_scan(
            {"source_stamp": self.now()}, robot
        )
        self.assertEqual(controller.registration_confirmation_count, 1)
        self.advance(
            controller.registration_filter.config.maximum_gap + 0.01
        )
        controller._process_registration_scan(
            {"source_stamp": self.now()}, robot
        )
        self.assertEqual(controller.registration_confirmation_count, 0)

    def test_course_heading_is_fixed_before_vehicle_heading_oscillates(self):
        controller = self.make_controller()
        self.arm(controller)
        identity = RigidTransform2D(
            0.0, 0.0, 0.0, "parking_local", "odom"
        )
        segments = self.transformed_fixture_segments(controller, identity)
        controller._extract_registration_segments = mock.Mock(
            return_value=segments
        )

        for yaw in (
            math.pi,
            math.pi - math.radians(1.2),
            math.pi + math.radians(0.8),
        ):
            self.advance(0.05)
            controller._process_registration_scan(
                {"source_stamp": self.now()},
                (0.92, 1.75, yaw),
            )

        self.assert_transform_close(self, controller.local_to_odom, identity)

    def test_finite_fixture_face_rejects_overhanging_segment(self):
        controller = self.make_controller()
        robot = (0.92, 1.75, math.pi)
        horizontal_overhang = (
            (0.43, 1.8875),
            (0.61, 1.8875),
            0.18,
        )
        vertical = self.transformed_fixture_segments(
            controller,
            RigidTransform2D(
                0.0, 0.0, 0.0, "parking_local", "odom"
            ),
        )[1]

        self.assertIsNone(
            controller._estimate_fixture_registration(
                (horizontal_overhang, vertical), robot, self.now()
            )
        )

    def test_registration_scan_survives_main_lock_phase_collision(self):
        controller = self.make_controller()
        self.arm(controller)
        expected = RigidTransform2D(
            0.02, -0.01, math.radians(0.4), "parking_local", "odom"
        )
        local_robot = (0.92, 1.75, math.pi)
        odom_robot = self.transform_pose(expected, local_robot)
        segments = self.transformed_fixture_segments(controller, expected)
        controller._extract_registration_segments = mock.Mock(
            return_value=segments
        )
        controller._project_safety_scan = mock.Mock()

        for confirmation in range(1, 4):
            self.advance(0.05)
            stamp = self.now()
            scan = {"source_stamp": stamp}
            controller._take_ready_safety_scans = mock.Mock(
                return_value=[(scan, odom_robot)]
            )
            odometry = self.odom_message(odom_robot, stamp)

            # Reproduce a scan/odom callback that reaches registration while
            # the 20 Hz controller owns its comparatively expensive main lock.
            with controller.lock:
                worker = threading.Thread(
                    target=controller.odom_callback, args=(odometry,)
                )
                worker.start()
                worker.join(1.0)
                self.assertFalse(worker.is_alive())
                self.assertIsNotNone(controller.pending_registration_scan)
                self.assertEqual(
                    controller.registration_confirmation_count,
                    confirmation - 1,
                )

            controller.control_callback(None)
            self.assertEqual(
                controller.registration_confirmation_count, confirmation
            )
            self.assertIsNone(controller.pending_registration_scan)

        self.assert_transform_close(self, controller.local_to_odom, expected)

    def test_laserscan_line_extraction_recovers_both_fixed_fixture_faces(self):
        controller = self.make_controller()
        robot = (0.92, 1.75, math.pi)
        sample_count = 2881
        angle_min = -math.pi
        angle_increment = 2.0 * math.pi / (sample_count - 1)
        ranges = np.full(sample_count, np.inf, dtype=np.float64)
        origin = np.asarray(
            [
                robot[0]
                + math.cos(robot[2]) * controller.lidar_x
                - math.sin(robot[2]) * controller.lidar_y,
                robot[1]
                + math.sin(robot[2]) * controller.lidar_x
                + math.cos(robot[2]) * controller.lidar_y,
            ]
        )

        def cross(first, second):
            return first[0] * second[1] - first[1] * second[0]

        for index in range(sample_count):
            sensor_angle = angle_min + index * angle_increment
            direction = np.asarray(
                [
                    math.cos(robot[2] + sensor_angle),
                    math.sin(robot[2] + sensor_angle),
                ]
            )
            # Raycast the complete solid fixtures and retain only the nearest
            # intersection, just as a real LaserScan does.  This catches a
            # registration template that accidentally selects an occluded
            # back face instead of the approach-visible face.
            fixture_edges = []
            for minimum_x, maximum_x, minimum_y, maximum_y in (
                controller.fixed_obstacle_rectangles
            ):
                fixture_edges.extend(
                    (
                        ((minimum_x, minimum_y), (maximum_x, minimum_y)),
                        ((maximum_x, minimum_y), (maximum_x, maximum_y)),
                        ((maximum_x, maximum_y), (minimum_x, maximum_y)),
                        ((minimum_x, maximum_y), (minimum_x, minimum_y)),
                    )
                )
            for edge_start, edge_end in fixture_edges:
                start = np.asarray(edge_start, dtype=np.float64)
                segment = np.asarray(edge_end, dtype=np.float64) - start
                denominator = cross(direction, segment)
                if abs(denominator) <= 1e-12:
                    continue
                offset = start - origin
                distance = cross(offset, segment) / denominator
                fraction = cross(offset, direction) / denominator
                if 0.05 <= distance <= 2.0 and 0.0 <= fraction <= 1.0:
                    ranges[index] = min(ranges[index], distance)

        scan = {
            "source_stamp": self.now(),
            "received": self.now(),
            "ranges": ranges,
            "angle_min": angle_min,
            "angle_increment": angle_increment,
            "range_min": 0.05,
            "maximum_range": 2.0,
        }
        segments = controller._extract_registration_segments(scan, robot)
        self.assertGreaterEqual(len(segments), 2)
        result = controller._estimate_fixture_registration(
            segments, robot, self.now()
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.accepted)
        expected = RigidTransform2D(
            0.0, 0.0, 0.0, "parking_local", "odom"
        )
        self.assert_transform_close(self, result.transform, expected, places=2)

    def _quantized_full_fixture_scan(self, controller, robot):
        sample_count = 360
        angle_min = -math.pi
        angle_increment = 2.0 * math.pi / (sample_count - 1)
        ranges = np.full(sample_count, np.inf, dtype=np.float64)
        cosine = math.cos(robot[2])
        sine = math.sin(robot[2])
        origin = np.asarray(
            [
                robot[0] + cosine * controller.lidar_x
                - sine * controller.lidar_y,
                robot[1] + sine * controller.lidar_x
                + cosine * controller.lidar_y,
            ]
        )

        def cross(first, second):
            return first[0] * second[1] - first[1] * second[0]

        edges = []
        for minimum_x, maximum_x, minimum_y, maximum_y in (
            controller.fixed_obstacle_rectangles
        ):
            edges.extend(
                (
                    ((minimum_x, minimum_y), (maximum_x, minimum_y)),
                    ((maximum_x, minimum_y), (maximum_x, maximum_y)),
                    ((maximum_x, maximum_y), (minimum_x, maximum_y)),
                    ((minimum_x, maximum_y), (minimum_x, minimum_y)),
                )
            )
        for index in range(sample_count):
            sensor_angle = angle_min + index * angle_increment
            direction = np.asarray(
                [
                    math.cos(robot[2] + sensor_angle),
                    math.sin(robot[2] + sensor_angle),
                ]
            )
            for edge_start, edge_end in edges:
                start = np.asarray(edge_start, dtype=np.float64)
                segment = np.asarray(edge_end, dtype=np.float64) - start
                denominator = cross(direction, segment)
                if abs(denominator) <= 1e-12:
                    continue
                offset = start - origin
                distance = cross(offset, segment) / denominator
                fraction = cross(offset, direction) / denominator
                if 0.10 <= distance <= 40.0 and 0.0 <= fraction <= 1.0:
                    ranges[index] = min(ranges[index], distance)
        finite = np.isfinite(ranges)
        ranges[finite] = np.round(ranges[finite] / 0.02) * 0.02
        message = LaserScan()
        message.ranges = ranges
        message.range_min = 0.10
        message.range_max = 40.0
        ranges = controller._median_ranges(message, 3)
        return {
            "source_stamp": self.now(),
            "received": self.now(),
            "ranges": ranges,
            "angle_min": angle_min,
            "angle_increment": angle_increment,
            "range_min": 0.10,
            "maximum_range": controller.registration_scan_maximum_range,
        }

    def test_actual_gazebo_scan_confirms_before_ready_window_end(self):
        controller = self.make_controller()
        self.arm(controller, sequence=23)
        freeze_x = None
        for robot_x in np.arange(1.098, 0.754, -0.026):
            self.advance(0.10)
            robot = (float(robot_x), controller.entry_y, math.pi)
            scan = self._quantized_full_fixture_scan(controller, robot)
            controller._process_registration_scan(scan, robot)
            if controller.local_to_odom is not None:
                freeze_x = float(robot_x)
                break

        self.assertIsNotNone(controller.local_to_odom)
        self.assertIsNotNone(freeze_x)
        self.assertGreaterEqual(freeze_x, 0.755)
        self.assertAlmostEqual(
            controller.local_to_odom.target_from_source_x, 0.0, delta=0.025
        )
        self.assertAlmostEqual(
            controller.local_to_odom.target_from_source_y, 0.0, delta=0.015
        )
        self.assertAlmostEqual(
            controller_module.normalize_angle(
                controller.local_to_odom.target_from_source_yaw
            ),
            0.0,
            delta=math.radians(1.0),
        )

    def test_single_or_parallel_fixture_faces_do_not_register(self):
        controller = self.make_controller()
        transform = RigidTransform2D(
            -0.70, 2.20, math.radians(-14.0), "parking_local", "odom"
        )
        local_robot = (0.92, 1.75, math.pi)
        odom_robot = self.transform_pose(transform, local_robot)
        segments = self.transformed_fixture_segments(controller, transform)

        self.assertIsNone(
            controller._estimate_fixture_registration(
                segments[:1], odom_robot, self.now()
            )
        )

        first = segments[0]
        offset = (0.0, 0.08)
        parallel = (
            (first[0][0] + offset[0], first[0][1] + offset[1]),
            (first[1][0] + offset[0], first[1][1] + offset[1]),
            first[2],
        )
        self.assertIsNone(
            controller._estimate_fixture_registration(
                (first, parallel), odom_robot, self.now()
            )
        )

    def test_equal_quality_perpendicular_decoy_pair_is_rejected(self):
        controller = self.make_controller()
        robot = (0.92, 1.75, math.pi)
        identity = RigidTransform2D(
            0.0, 0.0, 0.0, "parking_local", "odom"
        )
        shifted = RigidTransform2D(
            0.08, 0.0, 0.0, "parking_local", "odom"
        )
        real = self.transformed_fixture_segments(controller, identity)
        decoy = self.transformed_fixture_segments(controller, shifted)

        self.assertIsNone(
            controller._estimate_fixture_registration(
                real + decoy, robot, self.now()
            )
        )

    def test_arm_requires_exact_frame_and_treats_generation_as_token(self):
        controller = self.make_controller()
        wrong = Header(seq=8, stamp=self.now(), frame_id="intersection")
        controller.arm_callback(wrong)
        self.assertIsNone(controller.arm_seq)

        empty = Header(seq=8, stamp=self.now(), frame_id="")
        controller.arm_callback(empty)
        self.assertIsNone(controller.arm_seq)

        self.arm(controller, sequence=8)
        different_token = Header(seq=7, stamp=self.now(), frame_id="parking")
        controller.arm_callback(different_token)
        self.assertEqual(controller.arm_seq, 7)
        self.assertEqual(controller.arm_stamp, different_token.stamp)

    def test_entry_ready_is_local_and_independent_of_connecting_straight(self):
        for index, transform in enumerate(
            (
                RigidTransform2D(
                    1.0, -0.4, 0.0, "parking_local", "odom"
                ),
                RigidTransform2D(
                    4.8,
                    2.1,
                    math.radians(32.0),
                    "parking_local",
                    "odom",
                ),
            )
        ):
            with self.subTest(transform=index):
                controller = self.make_controller()
                odom_pose = self.prepare_ready(
                    controller, transform, sequence=31 + index
                )

                self.assertTrue(controller._try_publish_ready(self.now()))

                messages = self.publishers[controller.ready_topic].messages
                self.assertEqual(len(messages), 1)
                ready = messages[0]
                self.assertEqual(ready.seq, 31 + index)
                self.assertEqual(ready.stamp, controller.odom_stamp)
                self.assertEqual(ready.frame_id, "parking")
                self.assertEqual(controller.ready_published_seq, 31 + index)
                controller._build_adaptive_entry.assert_called_once_with(
                    preflight=True
                )

                # Until the Bool gate is received, refresh the ready stamp only
                # while the moving pose still projects onto the already swept
                # immutable path. Do not synthesize and validate a new suffix.
                prepared_connector = controller.entry_connector_path
                prepared_turn = controller.entry_curve_path
                controller._prepared_entry_projection = mock.Mock(
                    return_value=(SimpleNamespace(station=0.01), "")
                )
                self.advance(controller.ready_republish_period + 0.01)
                controller.odom_callback(
                    self.odom_message(odom_pose, self.now())
                )
                self.assertTrue(controller._try_publish_ready(self.now()))
                self.assertEqual(len(messages), 2)
                self.assertEqual(messages[-1].seq, 31 + index)
                self.assertEqual(messages[-1].stamp, controller.odom_stamp)
                self.assertEqual(
                    controller._build_adaptive_entry.call_count, 1
                )
                controller._prepared_entry_projection.assert_called_once_with()
                self.assertIs(
                    controller.entry_connector_path, prepared_connector
                )
                self.assertIs(
                    controller.entry_curve_path, prepared_turn
                )

    def test_neither_premature_gate_nor_ready_preflight_takes_control(self):
        controller = self.make_controller()
        arm = self.arm(controller, sequence=44)
        speed_limits = self.publishers[
            controller.speed_limit_topic
        ].messages
        self.assertEqual(len(speed_limits), 1)
        self.assertAlmostEqual(
            speed_limits[-1].data, controller.cruise_velocity, places=12
        )

        controller.gate_callback(Bool(data=True))
        controller.control_callback(None)
        self.assertFalse(controller.zone_gate)
        self.assertFalse(controller.start_requested)

        transform = RigidTransform2D(
            1.4, 0.6, math.radians(-8.0), "parking_local", "odom"
        )
        controller.arm_seq = arm.seq
        controller.arm_stamp = arm.stamp
        controller.local_to_odom = transform
        entry = controller.registration_template.entry_plane
        local_pose = (
            entry.point[0] - entry.normal[0] * 0.18,
            entry.point[1] - entry.normal[1] * 0.18,
            entry.heading,
        )
        self.advance(0.02)
        controller.odom_callback(
            self.odom_message(
                self.transform_pose(transform, local_pose), self.now()
            )
        )

        def build_preflight(preflight=False):
            self.assertTrue(preflight)
            controller.entry_connector_path = SimpleNamespace()
            controller.entry_curve_path = SimpleNamespace()
            return True

        controller._build_adaptive_entry = mock.Mock(
            side_effect=build_preflight
        )
        self.assertTrue(controller._try_publish_ready(self.now()))

        self.assertEqual(self.publishers[controller.cmd_vel_topic].messages, [])
        self.assertEqual(len(speed_limits), 1)
        self.assertTrue(all(not service.calls for service in self.services))
        self.assertFalse(controller.mission_has_control)

    def test_bay_rois_are_evaluated_in_frozen_parking_local_frame(self):
        controller = self.make_controller()
        transform = RigidTransform2D(
            3.2, -1.7, math.radians(41.0), "parking_local", "odom"
        )
        controller.local_to_odom = transform
        controller.state = controller.SELECT_SPACE
        controller.state_started = controller_module.rospy.Time.from_sec(9.0)
        local_robot = (controller.aisle_x, controller.decision_y, -0.5 * math.pi)
        odom_robot = self.transform_pose(transform, local_robot)

        cases = (
            (LEFT, controller.left_box, 5, 0),
            (RIGHT, controller.right_box, 0, 5),
        )
        for selected, box, expected_left, expected_right in cases:
            with self.subTest(selected=selected):
                self.advance(0.05)
                stamp = self.now()
                controller._record_odom_pose(stamp, odom_robot)
                xs = np.linspace(box[0] + 0.02, box[1] - 0.02, 5)
                local_points = np.column_stack(
                    (xs, np.full(xs.shape, 0.5 * (box[2] + box[3])))
                )
                odom_points = np.asarray(
                    [
                        tuple(transform.apply_point(point))
                        for point in local_points
                    ],
                    dtype=np.float64,
                )
                scan = {
                    "source_stamp": stamp,
                    "received": stamp,
                    "ranges": np.ones(5, dtype=np.float64),
                    "angle_min": 0.0,
                    "angle_increment": 0.1,
                    "range_min": 0.01,
                    "maximum_range": 2.0,
                }
                controller.pending_selection_scan = (scan, odom_robot)
                with mock.patch.object(
                    controller_module,
                    "scan_points_in_map",
                    return_value=odom_points,
                ):
                    message = controller._apply_pending_selection_scan()

                self.assertIsNotNone(message)
                self.assertEqual(
                    list(message.data), [expected_left, expected_right]
                )

    def test_forward_and_reverse_route_poses_share_the_same_frozen_transform(self):
        controller = self.make_controller()
        transform = RigidTransform2D(
            -2.0, 0.9, math.radians(-27.0), "parking_local", "odom"
        )
        controller.local_to_odom = transform
        route_poses = (
            (controller.aisle_x, controller.entry_y, controller.aisle_heading),
            (controller.left_park_x, controller.decision_y, 0.0),
            (controller.aisle_x, controller.decision_y, 0.0),
        )

        # PARK_IN uses +1 and BACK_OUT uses -1, but both geometry segments
        # must be projections through exactly this one frozen SE(2).
        for direction, route_pose in zip((1, 1, -1), route_poses):
            with self.subTest(direction=direction, route_pose=route_pose):
                odom_pose = controller._route_pose_to_odom(route_pose)
                recovered = controller._odom_pose_to_route(odom_pose)
                self.assertAlmostEqual(recovered[0], route_pose[0], places=9)
                self.assertAlmostEqual(recovered[1], route_pose[1], places=9)
                self.assertAlmostEqual(
                    controller_module.normalize_angle(
                        recovered[2] - route_pose[2]
                    ),
                    0.0,
                    places=9,
                )


if __name__ == "__main__":
    unittest.main()
