#!/usr/bin/env python3

from collections import deque
import math
from pathlib import Path
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
import yaml

from custom_autorace_bringup.path_following import (
    AsymmetricFootprint,
    CommonPath,
    GoalTolerance,
    PathFollower,
    PathSafety,
    Pose2D,
    RigidTransform2D,
    SafetyMargins,
    SpeedProfile,
    SweptFootprintValidator,
    TrackingConfig,
    ValidationResult,
    sample_path,
)
from custom_autorace_bringup.local_registration import (
    CurveRegistrationConfig,
    LocalCurveTemplate,
    TemporalRegistrationConfig,
    TemporalRegistrationFilter,
)
from custom_autorace_bringup.zigzag_path import (
    RasterPaintCorridorChecker,
    SurveyedCorridorChecker,
    build_zigzag_path,
    calculate_tracking,
    committed_route_path,
    limit_tracking_command,
    normalize_angle,
    transformed_path,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE_DIR / "config" / "zigzag_mission_gazebo.yaml"
PARKING_CONFIG = PACKAGE_DIR / "config" / "parking_mission_gazebo.yaml"
COURSE_IMAGE = (
    PACKAGE_DIR.parent
    / "turtlebot3_simulations"
    / "turtlebot3_gazebo"
    / "models"
    / "turtlebot3_autorace_2020"
    / "course"
    / "materials"
    / "textures"
    / "course.png"
)
NODE_DIR = PACKAGE_DIR / "nodes"
if str(NODE_DIR) not in sys.path:
    sys.path.insert(0, str(NODE_DIR))

import zigzag_mission_controller as controller_module
from zigzag_mission_controller import ZigzagMissionController


class RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class ZigzagPathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONFIG.open(encoding="utf-8") as stream:
            cls.config = yaml.safe_load(stream)["zigzag"]
        path_config = cls.config["path"]
        control = cls.config["control"]
        build_arguments = dict(
            knots=path_config["knots"],
            start_heading=math.radians(path_config["start_heading_deg"]),
            end_heading=math.radians(path_config["end_heading_deg"]),
            sample_spacing=path_config["sample_spacing"],
            cruise_velocity=control["cruise_velocity"],
            minimum_velocity=control["minimum_velocity"],
            entry_velocity=control["entry_velocity"],
            exit_velocity=control["exit_velocity"],
            maximum_angular_velocity=control["maximum_angular_velocity"],
            maximum_lateral_acceleration=control[
                "maximum_lateral_acceleration"
            ],
            linear_acceleration=control["linear_acceleration"],
            linear_deceleration=control["linear_deceleration"],
        )
        cls.speed_profile = SpeedProfile(
            cruise_velocity=control["cruise_velocity"],
            minimum_velocity=control["minimum_velocity"],
            entry_velocity=control["entry_velocity"],
            exit_velocity=control["exit_velocity"],
            maximum_angular_velocity=control["maximum_angular_velocity"],
            maximum_lateral_acceleration=control[
                "maximum_lateral_acceleration"
            ],
            linear_acceleration=control["linear_acceleration"],
            linear_deceleration=control["linear_deceleration"],
            angular_acceleration=control["angular_acceleration"],
        )
        exit_config = cls.config["exit"]
        common_arguments = dict(
            common_speed_profile=cls.speed_profile,
            frame_id=cls.config["route"]["frame_id"],
            goal_tolerance=GoalTolerance(
                position=exit_config["position_tolerance"],
                heading=math.radians(exit_config["heading_tolerance_deg"]),
                terminal_crossing=exit_config["position_tolerance"],
            ),
        )
        cls.full_geometry = build_zigzag_path(
            **build_arguments,
            maximum_angular_acceleration=control["angular_acceleration"],
            **common_arguments,
        )
        cls.path = build_zigzag_path(
            **build_arguments,
            guide_tail_length=path_config["guide_tail_length"],
            maximum_angular_acceleration=control["angular_acceleration"],
            **common_arguments,
        )
        footprint = cls.config["footprint"]
        corridor = cls.config["corridor"]
        cls.checker = SurveyedCorridorChecker(
            corridor["lower_boundary"],
            corridor["upper_boundary"],
        )
        texture = cls.config["texture"]
        course_rgb = np.asarray(Image.open(COURSE_IMAGE).convert("RGB"))
        cls.paint_checker = RasterPaintCorridorChecker(
            course_rgb,
            texture["course_size"],
            texture["course_yaw"],
            corridor["lower_boundary"],
            corridor["upper_boundary"],
            footprint["front"],
            footprint["rear"],
            footprint["half_width"],
            0.001,
            0.001,
            0.00025,
            texture["line_search_half_width"],
            texture["color_threshold"],
            texture["color_tolerance"],
            texture["yellow_blue_maximum"],
            texture["boundary_min_x"],
            texture["boundary_max_x"],
        )
        cls.runtime_paint_checker = RasterPaintCorridorChecker(
            course_rgb,
            texture["course_size"],
            texture["course_yaw"],
            corridor["lower_boundary"],
            corridor["upper_boundary"],
            footprint["front"],
            footprint["rear"],
            footprint["half_width"],
            footprint["validation_sample_spacing"],
            texture["boundary_x_step"],
            texture["boundary_y_step"],
            texture["line_search_half_width"],
            texture["color_threshold"],
            texture["color_tolerance"],
            texture["yellow_blue_maximum"],
            texture["boundary_min_x"],
            texture["boundary_max_x"],
        )

    def test_entry_cap_matches_the_parking_straight_handoff(self):
        with PARKING_CONFIG.open(encoding="utf-8") as stream:
            parking = yaml.safe_load(stream)["parking"]

        entry_cap = float(self.config["control"]["entry_velocity_cap"])
        parking_cap = float(
            parking["control"]["post_completion_velocity_cap"]
        )
        self.assertAlmostEqual(entry_cap, 0.10, places=12)
        self.assertAlmostEqual(entry_cap, parking_cap, places=12)

        # Parking completes around the westbound x=0.32 boundary.  At that
        # handoff pose the surveyed suffix already permits more than the cap,
        # so the higher handoff value cannot bypass the curvature profile.
        handoff_x = float(parking["rejoin"]["complete_max_x"])
        handoff_y = 0.5 * (
            float(parking["rejoin"]["complete_min_y"])
            + float(parking["rejoin"]["complete_max_y"])
        )
        index = int(
            np.argmin(
                np.hypot(
                    self.path.x - handoff_x,
                    self.path.y - handoff_y,
                )
            )
        )
        self.assertGreaterEqual(float(self.path.speed[index]), entry_cap)

    def test_constructor_latches_the_surveyed_path_before_the_gate(self):
        publishers = {}
        publisher_options = {}

        def get_param(name, default=None):
            prefix = "~zigzag/"
            key = name[len(prefix) :] if name.startswith(prefix) else name
            if key == "texture/enabled":
                return False
            value = self.config
            try:
                for part in key.split("/"):
                    value = value[part]
            except (KeyError, TypeError):
                return default
            return value

        def publisher(topic, _message_type, **options):
            recording = RecordingPublisher()
            publishers[topic] = recording
            publisher_options[topic] = options
            return recording

        validator = mock.Mock()
        validator.validate_path.return_value = ValidationResult(
            True,
            minimum_line_clearance=0.02,
            minimum_obstacle_clearance=math.inf,
            minimum_map_clearance=math.inf,
        )
        fixed_now = controller_module.rospy.Time.from_sec(42.0)
        with mock.patch.object(
            controller_module.rospy, "get_param", side_effect=get_param
        ), mock.patch.object(
            controller_module, "SweptFootprintValidator", return_value=validator
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
        ), mock.patch.object(
            controller_module.rospy, "loginfo"
        ), mock.patch.object(
            controller_module.rospy.Time, "now", return_value=fixed_now
        ):
            controller = ZigzagMissionController()

        self.assertIsNone(controller.committed_path)
        self.assertTrue(publisher_options["/zigzag/path"]["latch"])
        self.assertEqual(len(publishers["/zigzag/path"].messages), 1)
        message = publishers["/zigzag/path"].messages[0]
        self.assertEqual(message.header.stamp, controller_module.rospy.Time())
        self.assertEqual(controller.map_path.frame_id, "map")
        self.assertEqual(message.header.frame_id, "map")
        self.assertEqual(message.header.frame_id, controller.map_path.frame_id)
        self.assertEqual(len(message.poses), controller.map_path.size)
        for index in (0, controller.map_path.size // 2, -1):
            pose = message.poses[index].pose
            yaw = float(controller.map_path.heading[index])
            self.assertAlmostEqual(
                pose.position.x, float(controller.map_path.x[index]), places=12
            )
            self.assertAlmostEqual(
                pose.position.y, float(controller.map_path.y[index]), places=12
            )
            self.assertAlmostEqual(pose.position.z, 0.06, places=12)
            self.assertAlmostEqual(
                pose.orientation.z, math.sin(0.5 * yaw), places=12
            )
            self.assertAlmostEqual(
                pose.orientation.w, math.cos(0.5 * yaw), places=12
            )

    def _local_registration_controller(self):
        controller = ZigzagMissionController.__new__(ZigzagMissionController)
        registration = self.config["registration"]
        controller.lock = threading.RLock()
        controller.state = controller.WAIT_GATE
        controller.zone_gate = False
        controller.arm_generation = 0
        controller.armed_at = None
        controller.accepted_gate_generation = 0
        controller.prepared_path = None
        controller.prepared_route_from_odom = None
        controller.prepared_registration_transform = None
        controller.prepared_generation = 0
        controller.prepared_stamp = None
        controller.registration_covariance = tuple()
        controller.map_path = self.path
        controller.local_curve_template = LocalCurveTemplate(
            "zigzag", self.path.frame_id, tuple(zip(self.path.x, self.path.y))
        )
        controller.curve_registration_config = CurveRegistrationConfig(
            sample_count=registration["sample_count"],
            station_search_step=registration["station_search_step"],
            minimum_observed_length=registration["minimum_observed_length"],
            minimum_heading_variation=math.radians(
                registration["minimum_heading_variation_deg"]
            ),
            minimum_lateral_excitation=registration[
                "minimum_lateral_excitation"
            ],
            position_inlier_threshold=registration[
                "position_inlier_threshold"
            ],
            minimum_inlier_fraction=registration["minimum_inlier_fraction"],
            maximum_rms=registration["maximum_rms"],
            ambiguity_station_separation=registration[
                "ambiguity_station_separation"
            ],
            maximum_ambiguity_rms_difference=registration[
                "maximum_ambiguity_rms_difference"
            ],
            maximum_ambiguity_rms_ratio=registration[
                "maximum_ambiguity_rms_ratio"
            ],
        )
        controller.registration_filter = TemporalRegistrationFilter(
            TemporalRegistrationConfig(
                required_confirmations=registration["confirmation_frames"],
                maximum_gap=registration["confirmation_max_gap"],
                maximum_position_delta=registration[
                    "maximum_position_spread"
                ],
                maximum_heading_delta=math.radians(
                    registration["maximum_heading_spread_deg"]
                ),
            )
        )
        controller.registration_source_max_age = registration[
            "source_max_age"
        ]
        controller.registration_future_tolerance = registration[
            "future_tolerance"
        ]
        controller.ready_republish_period = registration[
            "ready_republish_period"
        ]
        controller.registration_maximum_start_station_error = registration[
            "maximum_start_station_error"
        ]
        controller.route_odom_aligned = False
        controller.odom_frame = "odom"
        controller.odom_ready = True
        controller.live_obstacle_points = np.empty((0, 2), dtype=np.float64)
        controller.footprint = AsymmetricFootprint(0.067645, 0.118073, 0.0903)
        controller.start_maximum_path_error = self.config["start"][
            "maximum_path_error"
        ]
        controller.start_maximum_join_heading_error = math.radians(
            self.config["start"]["maximum_join_heading_error_deg"]
        )
        controller.tracking_config = TrackingConfig(
            lookahead_distance=0.065,
            maximum_linear_velocity=0.14,
            maximum_angular_velocity=0.8,
            maximum_lateral_acceleration=0.035,
            linear_acceleration=0.04,
            linear_deceleration=0.12,
            angular_acceleration=0.8,
            heading_gain=0.35,
        )
        controller._path_safety = mock.Mock(return_value=PathSafety())
        controller.path_validator = mock.Mock()
        controller.path_validator.validate_path.return_value = ValidationResult(
            True,
            minimum_line_clearance=0.02,
            minimum_obstacle_clearance=0.10,
            minimum_map_clearance=0.20,
        )
        controller.ready_pub = RecordingPublisher()
        return controller

    @staticmethod
    def _rolling_path_message(points, stamp):
        message = controller_module.Path()
        message.header.frame_id = "odom"
        message.header.stamp = controller_module.rospy.Time.from_sec(stamp)
        for x, y in points:
            pose = controller_module.PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            message.poses.append(pose)
        return message

    def test_curved_lane_path_registers_shifted_local_route_before_gate(self):
        controller = self._local_registration_controller()
        empty_arm = controller_module.Header()
        empty_arm.seq = 16
        empty_arm.stamp = controller_module.rospy.Time.from_sec(19.9)
        controller.arm_callback(empty_arm)
        self.assertEqual(controller.arm_generation, 0)

        zero_stamp = controller_module.Header(
            seq=17,
            stamp=controller_module.rospy.Time(),
            frame_id="zigzag",
        )
        controller.arm_callback(zero_stamp)
        self.assertEqual(controller.arm_generation, 0)

        arm = controller_module.Header()
        arm.seq = 17
        arm.stamp = controller_module.rospy.Time.from_sec(20.0)
        arm.frame_id = "zigzag"
        controller.arm_callback(arm)

        start, end = 0.55, 1.05
        selected = (self.path.station >= start) & (self.path.station <= end)
        transform = RigidTransform2D(
            0.37, -0.22, math.radians(6.0), "map", "odom"
        )
        observed = [
            transform.apply_point(point)
            for point in zip(self.path.x[selected], self.path.y[selected])
        ]
        first_index = int(np.flatnonzero(selected)[0])
        robot_pose = transform.apply_pose(
            Pose2D(
                self.path.x[first_index],
                self.path.y[first_index],
                self.path.heading[first_index],
            )
        )
        controller.odom_x = robot_pose.x
        controller.odom_y = robot_pose.y
        controller.odom_yaw = robot_pose.yaw

        for stamp in (20.03, 20.06, 20.09):
            with mock.patch.object(
                controller_module.rospy.Time,
                "now",
                return_value=controller_module.rospy.Time.from_sec(
                    stamp + 0.01
                ),
            ):
                controller.lane_path_callback(
                    self._rolling_path_message(observed, stamp)
                )

        self.assertEqual(controller.prepared_generation, 17)
        self.assertEqual(len(controller.ready_pub.messages), 1)
        ready = controller.ready_pub.messages[0]
        self.assertEqual(ready.seq, 17)
        self.assertEqual(ready.frame_id, "zigzag")
        self.assertAlmostEqual(ready.stamp.to_sec(), 20.09, places=8)
        self.assertAlmostEqual(
            controller.prepared_path.x[first_index], robot_pose.x, delta=0.003
        )
        self.assertAlmostEqual(
            controller.prepared_path.y[first_index], robot_pose.y, delta=0.003
        )
        self.assertFalse(controller.zone_gate)

    def test_odom_aligned_run14_entry_uses_pose_bounded_station_search(self):
        controller = self._local_registration_controller()
        controller.route_odom_aligned = True
        arm = controller_module.Header(
            seq=18,
            stamp=controller_module.rospy.Time.from_sec(20.0),
            frame_id="zigzag",
        )
        controller.arm_callback(arm)

        # Run14's first rolling path matched the surveyed suffix near station
        # 0.23 while the raw-odom robot projection was near station 0.273.
        selected = (
            (self.path.station >= 0.23)
            & (self.path.station <= 0.915)
        )
        transform = RigidTransform2D(
            -0.821,
            1.732,
            math.radians(0.82),
            "map",
            "odom",
        )
        observed = [
            transform.apply_point(point)
            for point in zip(self.path.x[selected], self.path.y[selected])
        ]
        pose_index = int(np.argmin(np.abs(self.path.station - 0.273)))
        controller.odom_x = float(self.path.x[pose_index])
        controller.odom_y = float(self.path.y[pose_index])
        controller.odom_yaw = float(self.path.heading[pose_index])

        with mock.patch.object(
            controller_module,
            "register_curve_subset",
            wraps=controller_module.register_curve_subset,
        ) as matcher:
            for stamp in (20.03, 20.06, 20.09):
                with mock.patch.object(
                    controller_module.rospy.Time,
                    "now",
                    return_value=controller_module.rospy.Time.from_sec(
                        stamp + 0.01
                    ),
                ):
                    controller.lane_path_callback(
                        self._rolling_path_message(observed, stamp)
                    )

        self.assertEqual(controller.prepared_generation, 18)
        self.assertEqual(len(controller.ready_pub.messages), 1)
        self.assertEqual(matcher.call_count, 3)
        for call in matcher.call_args_list:
            lower, upper = call.kwargs["start_station_bounds"]
            self.assertLessEqual(lower, 0.23)
            self.assertGreaterEqual(upper, 0.23)
            self.assertAlmostEqual(
                upper - lower,
                2.0
                * self.config["registration"][
                    "maximum_start_station_error"
                ],
                delta=0.004,
            )

    def test_straight_lane_path_never_opens_zigzag_ready_gate(self):
        controller = self._local_registration_controller()
        arm = controller_module.Header()
        arm.seq = 9
        arm.stamp = controller_module.rospy.Time.from_sec(30.0)
        arm.frame_id = "zigzag"
        controller.arm_callback(arm)
        controller.odom_x = 0.0
        controller.odom_y = 0.0
        controller.odom_yaw = 0.0
        straight = [(value, 0.0) for value in np.linspace(0.0, 0.45, 40)]

        for stamp in (30.03, 30.06, 30.09, 30.12):
            with mock.patch.object(
                controller_module.rospy.Time,
                "now",
                return_value=controller_module.rospy.Time.from_sec(
                    stamp + 0.01
                ),
            ):
                controller.lane_path_callback(
                    self._rolling_path_message(straight, stamp)
                )

        self.assertEqual(controller.ready_pub.messages, [])
        self.assertIsNone(controller.prepared_path)

    def test_prepared_zigzag_refreshes_ready_without_replanning(self):
        controller = self._local_registration_controller()
        controller.route_odom_aligned = True
        arm = controller_module.Header(
            seq=17,
            stamp=controller_module.rospy.Time.from_sec(20.0),
            frame_id="zigzag",
        )
        controller.arm_callback(arm)
        selected = (self.path.station >= 0.55) & (self.path.station <= 1.05)
        transform = RigidTransform2D(
            0.37, -0.22, math.radians(6.0), "map", "odom"
        )
        observed = [
            transform.apply_point(point)
            for point in zip(self.path.x[selected], self.path.y[selected])
        ]
        first_index = int(np.flatnonzero(selected)[0])
        pose = Pose2D(
            self.path.x[first_index],
            self.path.y[first_index],
            self.path.heading[first_index],
        )
        controller.odom_x = pose.x
        controller.odom_y = pose.y
        controller.odom_yaw = pose.yaw

        for stamp in (20.03, 20.06, 20.09):
            with mock.patch.object(
                controller_module.rospy.Time,
                "now",
                return_value=controller_module.rospy.Time.from_sec(
                    stamp + 0.01
                ),
            ):
                controller.lane_path_callback(
                    self._rolling_path_message(observed, stamp)
                )
        frozen_path = controller.prepared_path
        frozen_transform = controller.prepared_route_from_odom
        self.assertAlmostEqual(
            controller.prepared_registration_transform.target_from_source_x,
            transform.target_from_source_x,
            delta=0.003,
        )
        self.assertAlmostEqual(
            frozen_transform.target_from_source_x, 0.0, places=12
        )
        self.assertEqual(controller.registration_covariance, tuple())
        self.assertEqual(
            controller._path_safety.call_args.kwargs["registration_margin"],
            0.0,
        )

        refresh_stamp = 20.25
        with mock.patch.object(
            controller_module.rospy.Time,
            "now",
            return_value=controller_module.rospy.Time.from_sec(
                refresh_stamp + 0.01
            ),
        ):
            controller.lane_path_callback(
                self._rolling_path_message(observed, refresh_stamp)
            )

        self.assertEqual(len(controller.ready_pub.messages), 2)
        self.assertAlmostEqual(
            controller.ready_pub.messages[-1].stamp.to_sec(),
            refresh_stamp,
            places=8,
        )
        self.assertIs(controller.prepared_path, frozen_path)
        self.assertEqual(controller.prepared_route_from_odom, frozen_transform)

    def test_premature_zigzag_gate_has_no_control_side_effect(self):
        controller = self._local_registration_controller()
        controller.speed_limit_pub = RecordingPublisher()
        controller.start_requested = False
        controller.revoke_requested = False
        controller.entry_velocity_cap = 0.10
        arm = controller_module.Header(
            seq=5,
            stamp=controller_module.rospy.Time.from_sec(10.0),
            frame_id="zigzag",
        )
        controller.arm_callback(arm)
        now = controller_module.rospy.Time.from_sec(10.1)
        with mock.patch.object(
            controller_module.rospy.Time, "now", return_value=now
        ), mock.patch.object(controller_module.rospy, "logwarn_throttle"):
            controller.gate_callback(controller_module.Bool(data=True))
            self.assertFalse(controller._start_run(now))

        self.assertFalse(controller.zone_gate)
        self.assertFalse(controller.start_requested)
        self.assertEqual(controller.speed_limit_pub.messages, [])

    def test_manager_accepted_gate_survives_local_age_and_timer_boundary(self):
        controller = self._local_registration_controller()
        controller.arm_generation = 5
        controller.armed_at = controller_module.rospy.Time.from_sec(9.9)
        controller.prepared_generation = 5
        controller.prepared_path = self.path
        controller.prepared_route_from_odom = RigidTransform2D(
            0.0, 0.0, 0.0, "odom", self.path.frame_id
        )
        controller.prepared_stamp = controller_module.rospy.Time.from_sec(10.0)
        controller.speed_limit_pub = RecordingPublisher()
        controller.state_pub = RecordingPublisher()
        controller.entry_velocity_cap = self.config["control"][
            "entry_velocity_cap"
        ]
        controller.start_requested = False
        controller.revoke_requested = False
        controller.mission_has_control = False

        gate_time = controller_module.rospy.Time.from_sec(10.379)
        self.assertFalse(controller._prepared_ready_for_gate(gate_time))
        with mock.patch.object(
            controller_module.rospy.Time, "now", return_value=gate_time
        ):
            controller.gate_callback(controller_module.Bool(data=True))

        self.assertTrue(controller.zone_gate)
        self.assertTrue(controller.start_requested)
        self.assertEqual(controller.accepted_gate_generation, 5)
        delayed_timer = controller_module.rospy.Time.from_sec(10.40)
        self.assertFalse(controller._prepared_ready_for_gate(delayed_timer))
        self.assertTrue(controller._start_run(delayed_timer))
        self.assertEqual(controller.state, controller.ACQUIRING)
        self.assertTrue(controller.zone_gate)
        self.assertFalse(controller.start_requested)
        self.assertAlmostEqual(
            controller.speed_limit_pub.messages[-1].data,
            controller.entry_velocity_cap,
            places=12,
        )

    def test_new_arm_invalidates_accepted_zigzag_gate_generation(self):
        controller = self._local_registration_controller()
        controller.arm_generation = 5
        controller.armed_at = controller_module.rospy.Time.from_sec(9.9)
        controller.prepared_generation = 5
        controller.prepared_path = self.path
        controller.prepared_route_from_odom = RigidTransform2D(
            0.0, 0.0, 0.0, "odom", self.path.frame_id
        )
        controller.prepared_stamp = controller_module.rospy.Time.from_sec(10.0)
        controller.start_requested = False
        controller.revoke_requested = False
        controller.gate_callback(controller_module.Bool(data=True))
        self.assertEqual(controller.accepted_gate_generation, 5)

        controller.arm_callback(
            controller_module.Header(
                seq=6,
                stamp=controller_module.rospy.Time.from_sec(10.1),
                frame_id="zigzag",
            )
        )

        self.assertEqual(controller.arm_generation, 6)
        self.assertEqual(controller.accepted_gate_generation, 0)
        self.assertFalse(controller.start_requested)
        self.assertIsNone(controller.prepared_path)

    def _configure_production_safety(self, controller):
        footprint_config = self.config["footprint"]
        control = self.config["control"]
        controller.route_from_odom = None
        controller.corridor_checker = self.checker
        controller.paint_checker = self.runtime_paint_checker
        controller.footprint_sample_spacing = footprint_config[
            "validation_sample_spacing"
        ]
        controller.minimum_outer_reserve = self.config["texture"][
            "minimum_outer_reserve"
        ]
        controller.inner_paint_blocking = self.config["validation"][
            "inner_paint_blocking"
        ]
        controller.safety_margins = SafetyMargins(
            line=footprint_config["line_margin"],
            obstacle=footprint_config["obstacle_margin"],
            localization=footprint_config["localization_error"],
            tracking=footprint_config["tracking_error"],
        )
        controller.live_route_safety = PathSafety(
            margins=controller.safety_margins
        )
        controller.path_validator = SweptFootprintValidator(
            AsymmetricFootprint(
                footprint_config["front"],
                footprint_config["rear"],
                footprint_config["half_width"],
            ),
            translation_step=footprint_config["sweep_translation_step"],
            heading_step=math.radians(
                footprint_config["sweep_heading_step_deg"]
            ),
        )
        controller.tracking_config = TrackingConfig(
            lookahead_distance=control["lookahead_distance"],
            maximum_linear_velocity=control["cruise_velocity"],
            maximum_angular_velocity=control["maximum_angular_velocity"],
            maximum_lateral_acceleration=control[
                "maximum_lateral_acceleration"
            ],
            linear_acceleration=control["linear_acceleration"],
            linear_deceleration=control["linear_deceleration"],
            angular_acceleration=control["angular_acceleration"],
            heading_gain=control["heading_gain"],
            curvature_feedforward_weight=control[
                "path_curvature_weight"
            ],
            lateral_feedback_gain=control["lateral_feedback_gain"],
            search_back=3,
            search_ahead_distance=control["nearest_search_ahead"],
        )
        controller.path_follower = PathFollower(controller.tracking_config)
        controller.committed_path = self.path
        controller.committed_path.safety = controller._path_safety()
        controller.last_linear = control["cruise_velocity"]
        controller.last_angular = 0.0
        controller.odom_speed = control["cruise_velocity"]
        controller.odom_angular_velocity = 0.0
        controller.prediction_reaction_time = self.config["texture"][
            "prediction_reaction_time"
        ]
        controller.prediction_distance_margin = self.config["texture"][
            "prediction_distance_margin"
        ]
        controller.linear_deceleration = control["linear_deceleration"]
        controller.nearest_search_ahead = control["nearest_search_ahead"]
        return controller

    def _production_safety_controller(self):
        return self._configure_production_safety(
            ZigzagMissionController.__new__(ZigzagMissionController)
        )

    def test_production_path_geometry_preserves_established_spline(self):
        path_config = self.config["path"]
        self.assertIs(type(self.path), CommonPath)
        self.assertFalse(
            self.config["validation"]["inner_paint_blocking"]
        )
        self.assertGreater(self.path.x.size, 900)
        self.assertAlmostEqual(self.path.x[0], 0.4327, places=6)
        self.assertAlmostEqual(self.path.y[0], 1.7500, places=6)
        self.assertAlmostEqual(self.path.x[-1], -1.39233, places=5)
        self.assertAlmostEqual(self.path.y[-1], 1.74893, places=5)
        self.assertLess(
            abs(normalize_angle(float(self.path.heading[0]) - math.pi)),
            1e-8,
        )
        self.assertLess(
            abs(normalize_angle(float(self.path.heading[-1]) - math.pi)),
            math.radians(3.0),
        )
        self.assertLess(abs(float(self.path.curvature[-1])), 0.30)
        self.assertTrue(np.all(np.diff(self.path.station) > 0.0))
        self.assertLessEqual(
            float(np.max(np.abs(self.path.curvature))),
            path_config["maximum_curvature"],
        )
        self.assertNotIn("maximum_nominal_line_overlap", path_config)

    def test_guide_tail_shapes_geometry_but_is_not_driven(self):
        guide_length = self.config["path"]["guide_tail_length"]
        active = self.path
        full = self.full_geometry
        self.assertAlmostEqual(full.length - active.length, guide_length, places=10)
        self.assertAlmostEqual(
            full.x[-1], self.config["path"]["knots"][-1][0], places=9
        )
        self.assertNotAlmostEqual(active.x[-1], full.x[-1], places=4)
        count = active.x.size
        np.testing.assert_allclose(active.x[:-1], full.x[: count - 1])
        np.testing.assert_allclose(active.y[:-1], full.y[: count - 1])
        self.assertLess(float(full.station[count - 2]), active.length)
        self.assertLess(active.length, float(full.station[count - 1]))
        self.assertAlmostEqual(
            math.hypot(
                float(active.x[-1] - full.x[count - 2]),
                float(active.y[-1] - full.y[count - 2]),
            ),
            active.length - float(full.station[count - 2]),
            places=9,
        )
        self.assertAlmostEqual(
            float(active.speed[-1]),
            self.config["control"]["exit_velocity"],
            places=12,
        )

    def test_speed_profile_respects_feedforward_angular_acceleration(self):
        segment = np.diff(self.path.station)
        duration = 2.0 * segment / (
            self.path.speed[:-1] + self.path.speed[1:]
        )
        omega = self.path.speed * self.path.curvature
        angular_acceleration = np.abs(np.diff(omega)) / duration
        self.assertLessEqual(
            float(np.max(angular_acceleration)),
            self.config["control"]["angular_acceleration"] + 2e-6,
        )
        acceleration = self.config["control"]["linear_acceleration"]
        self.assertTrue(
            np.all(
                self.path.speed[1:] ** 2
                <= self.path.speed[:-1] ** 2
                + 2.0 * acceleration * segment
                + 1e-9
            )
        )

    def test_actual_course_texture_extracts_ordered_paint_boundaries(self):
        self.assertFalse(
            self.config["validation"]["inner_paint_blocking"]
        )
        self.assertTrue(
            np.all(self.paint_checker.lower_outer < self.paint_checker.lower_inner)
        )
        self.assertTrue(
            np.all(self.paint_checker.lower_inner < self.paint_checker.upper_inner)
        )
        self.assertTrue(
            np.all(self.paint_checker.upper_inner < self.paint_checker.upper_outer)
        )

    def test_speed_profile_brakes_before_peak_curvature(self):
        peak = int(np.argmax(np.abs(self.path.curvature)))
        before = max(
            0,
            int(
                np.searchsorted(
                    self.path.station,
                    self.path.station[peak] - 0.04,
                )
            ),
        )
        cruise = self.config["control"]["cruise_velocity"]
        self.assertLess(self.path.speed[before], cruise)
        self.assertGreater(self.path.speed[before], self.path.speed[peak])
        segment = np.diff(self.path.station)
        deceleration = self.config["control"]["linear_deceleration"]
        self.assertTrue(
            np.all(
                self.path.speed[:-1] ** 2
                <= self.path.speed[1:] ** 2
                + 2.0 * deceleration * segment
                + 1e-9
            )
        )

    def test_runtime_full_stopping_sweep_fits_six_point_control_budget(self):
        controller = self._production_safety_controller()
        control = self.config["control"]
        empty_obstacles = np.empty((0, 2), dtype=np.float64)

        # Production performs a complete fixed-route sweep before control, which
        # also warms the exact local-footprint cache used by runtime checks.
        fixed = controller.path_validator.validate_path(
            controller.committed_path,
        )
        self.assertTrue(fixed.safe)

        elapsed = []
        for fraction in (0.0, 0.20, 0.40, 0.60, 0.80, 0.98):
            index = min(
                controller.committed_path.size - 1,
                int(fraction * (controller.committed_path.size - 1)),
            )
            pose = Pose2D(
                float(controller.committed_path.x[index]),
                float(controller.committed_path.y[index]),
                float(controller.committed_path.heading[index]),
            )
            controller.path_follower.reset(
                controller.committed_path,
                pose,
                initial_linear=control["cruise_velocity"],
            )
            tracking = controller.path_follower.calculate_tracking(pose)
            started = time.perf_counter()
            decision = controller._motion_safety(
                float(tracking.target_speed),
                empty_obstacles,
                pose=pose,
                tracking=tracking,
                path=controller.committed_path,
                validator=controller.path_validator,
            )
            elapsed.append(time.perf_counter() - started)
            self.assertTrue(decision.validation.safe)

        self.assertLess(
            max(elapsed),
            control["period"],
            "six-point runtime sweep seconds=%s" % elapsed,
        )

    def test_legacy_compatibility_lookahead_selects_future_target(self):
        index = int(np.searchsorted(self.path.station, 0.60))
        result = calculate_tracking(
            self.path,
            float(self.path.x[index]),
            float(self.path.y[index]),
            float(self.path.heading[index]),
            index,
            self.config["control"]["lookahead_distance"],
            self.config["control"]["maximum_angular_velocity"],
            self.config["control"]["heading_gain"],
            self.config["control"]["path_curvature_weight"],
            self.config["control"]["nearest_search_ahead"],
        )
        self.assertEqual(result.path_index, index)
        self.assertGreater(result.target_index, index)
        self.assertGreater(
            self.path.station[result.target_index] - self.path.station[index],
            0.95 * self.config["control"]["lookahead_distance"],
        )
        self.assertNotAlmostEqual(result.angular_velocity, 0.0, places=3)

    def test_legacy_compatibility_limiter_preserves_curvature(self):
        limited = limit_tracking_command(
            target_speed=0.05,
            reference_speed=0.10,
            target_angular_velocity=0.40,
            last_linear_velocity=0.05,
            last_angular_velocity=0.20,
            elapsed=0.05,
            linear_acceleration=0.04,
            linear_deceleration=0.12,
            angular_acceleration=0.80,
            maximum_angular_velocity=0.80,
            maximum_lateral_acceleration=0.035,
        )
        self.assertAlmostEqual(limited.linear_velocity, 0.05, places=12)
        self.assertAlmostEqual(
            limited.angular_velocity / limited.linear_velocity,
            4.0,
            places=12,
        )

    def test_legacy_compatibility_limiter_caps_steering_and_force(self):
        steering_limited = limit_tracking_command(
            target_speed=0.10,
            reference_speed=0.10,
            target_angular_velocity=0.40,
            last_linear_velocity=0.10,
            last_angular_velocity=0.0,
            elapsed=0.05,
            linear_acceleration=0.04,
            linear_deceleration=0.12,
            angular_acceleration=0.50,
            maximum_angular_velocity=0.80,
            maximum_lateral_acceleration=0.035,
        )
        self.assertAlmostEqual(steering_limited.angular_velocity, 0.025)
        self.assertLessEqual(steering_limited.linear_velocity, 0.00625 + 1e-12)

        force_limited = limit_tracking_command(
            target_speed=0.20,
            reference_speed=0.20,
            target_angular_velocity=0.80,
            last_linear_velocity=0.20,
            last_angular_velocity=0.80,
            elapsed=0.05,
            linear_acceleration=0.04,
            linear_deceleration=0.12,
            angular_acceleration=0.80,
            maximum_angular_velocity=0.80,
            maximum_lateral_acceleration=0.035,
        )
        self.assertLessEqual(
            force_limited.linear_velocity
            * abs(force_limited.angular_velocity),
            0.035 + 1e-12,
        )

    def test_segment_sweep_catches_rotation_between_safe_endpoints(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[128:133, 20:181] = (255, 255, 0)
        image[68:73, 20:181] = (255, 255, 255)
        checker = RasterPaintCorridorChecker(
            image,
            2.0,
            0.0,
            [(-0.7, -0.3), (0.7, -0.3)],
            [(-0.7, 0.3), (0.7, 0.3)],
            0.35,
            0.35,
            0.05,
            0.005,
            0.005,
            0.001,
            0.08,
            minimum_x=-0.7,
            maximum_x=0.7,
        )
        for yaw in (0.0, math.pi):
            clearance = checker.clearance(
                Pose2D(0.0, 0.0, yaw),
                AsymmetricFootprint(0.35, 0.35, 0.05),
            )
            self.assertGreater(clearance, 0.0)
        rotating_path = CommonPath(
            x=np.asarray([0.0, 0.000001]),
            y=np.asarray([0.0, 0.0]),
            heading=np.asarray([0.0, math.pi]),
            curvature=np.asarray([0.0, 0.0]),
            speed=np.asarray([0.05, 0.05]),
            safety=PathSafety(line_boundaries=(checker,)),
        )
        validation = SweptFootprintValidator(
            AsymmetricFootprint(0.35, 0.35, 0.05),
            translation_step=0.001,
            heading_step=math.radians(0.25),
        ).validate_path(rotating_path)
        self.assertFalse(validation.safe)
        self.assertLess(validation.minimum_line_clearance, 0.0)

    def test_surveyed_boundary_clearance_detects_shifted_footprint(self):
        index = int(np.argmax(np.abs(self.path.curvature)))
        footprint = self.config["footprint"]
        clearance = self.checker.clearance(
            Pose2D(
                float(self.path.x[index]),
                float(self.path.y[index]) + 0.04,
                float(self.path.heading[index]),
            ),
            AsymmetricFootprint(
                footprint["front"],
                footprint["rear"],
                footprint["half_width"],
            ),
            footprint_sample_spacing=footprint[
                "validation_sample_spacing"
            ],
        )
        self.assertLess(clearance, 0.0)
        self.assertNotIn("maximum_line_overlap", self.config["corridor"])

    def test_legacy_compatibility_rigid_transform_preserves_path_data(self):
        moved = transformed_path(self.path, 0.8, -0.4, 0.37)
        np.testing.assert_allclose(moved.station, self.path.station)
        np.testing.assert_allclose(moved.curvature, self.path.curvature)
        np.testing.assert_allclose(moved.speed, self.path.speed)
        self.assertAlmostEqual(moved.length, self.path.length, places=12)
        self.assertLess(
            abs(
                normalize_angle(
                    float(moved.heading[200])
                    - float(self.path.heading[200])
                    - 0.37
                )
            ),
            1e-12,
        )

    def test_legacy_compatibility_gazebo_route_odom_acquisition(self):
        self.assertTrue(self.config["route"]["odom_aligned"])
        for obsolete_key in (
            "min_x",
            "max_x",
            "min_y",
            "max_y",
            "heading_deg",
            "heading_tolerance_deg",
        ):
            self.assertNotIn(obsolete_key, self.config["start"])

        # Recorded in the official-start integrated run. AMCL had corrected
        # map about 18 cm away from world odom by parking completion, while the
        # actual zigzag handoff remained on the surveyed lead-in.
        map_pose = (0.4895, 1.9306, math.radians(177.0))
        odom_pose = (0.3060, 1.7474, math.radians(176.7))
        handoff = (0.178923, 1.752913, math.radians(178.280))
        aligned, route_from_odom = committed_route_path(
            self.path, map_pose, odom_pose, odom_aligned=True
        )
        shifted, _ = committed_route_path(
            self.path, map_pose, odom_pose, odom_aligned=False
        )

        aligned_index = int(
            np.argmin(np.hypot(aligned.x - handoff[0], aligned.y - handoff[1]))
        )
        shifted_error = float(
            np.min(np.hypot(shifted.x - handoff[0], shifted.y - handoff[1]))
        )
        self.assertEqual(route_from_odom, (0.0, 0.0, 0.0))
        self.assertGreater(aligned_index, 0)
        self.assertLess(
            math.hypot(
                float(aligned.x[aligned_index]) - handoff[0],
                float(aligned.y[aligned_index]) - handoff[1],
            ),
            0.006,
        )
        self.assertLess(
            abs(
                normalize_angle(
                    float(aligned.heading[aligned_index]) - handoff[2]
                )
            ),
            math.radians(4.0),
        )
        self.assertGreater(shifted_error, 0.15)

    def test_legacy_compatibility_hardware_route_transform_round_trip(self):
        route_pose = (0.25, 1.74, math.radians(179.0))
        odom_pose = (-0.31, 0.82, math.radians(176.5))
        moved, inverse = committed_route_path(
            self.path, route_pose, odom_pose, odom_aligned=False
        )
        nearest = int(
            np.argmin(
                np.hypot(
                    self.path.x - route_pose[0],
                    self.path.y - route_pose[1],
                )
            )
        )
        self.assertAlmostEqual(float(moved.x[nearest]), odom_pose[0], delta=0.02)
        self.assertAlmostEqual(float(moved.y[nearest]), odom_pose[1], delta=0.02)
        cosine = math.cos(inverse[2])
        sine = math.sin(inverse[2])
        recovered = (
            inverse[0] + cosine * odom_pose[0] - sine * odom_pose[1],
            inverse[1] + sine * odom_pose[0] + cosine * odom_pose[1],
            normalize_angle(inverse[2] + odom_pose[2]),
        )
        np.testing.assert_allclose(recovered[:2], route_pose[:2], atol=1e-12)
        self.assertLess(
            abs(normalize_angle(recovered[2] - route_pose[2])), 1e-12
        )

    def test_controller_exit_decelerates_before_verify_transition(self):
        control = self.config["control"]
        follower = PathFollower(
            TrackingConfig(
                lookahead_distance=control["lookahead_distance"],
                maximum_linear_velocity=control["cruise_velocity"],
                maximum_angular_velocity=control["maximum_angular_velocity"],
                maximum_lateral_acceleration=control[
                    "maximum_lateral_acceleration"
                ],
                linear_acceleration=control["linear_acceleration"],
                linear_deceleration=control["linear_deceleration"],
                angular_acceleration=control["angular_acceleration"],
                heading_gain=control["heading_gain"],
                curvature_feedforward_weight=control[
                    "path_curvature_weight"
                ],
                lateral_feedback_gain=control["lateral_feedback_gain"],
                search_ahead_distance=control["nearest_search_ahead"],
            )
        )
        follower.reset(
            self.path,
            Pose2D(
                float(self.path.x[-1]),
                float(self.path.y[-1]),
                float(self.path.heading[-1]),
            ),
            initial_linear=control["exit_velocity"],
            initial_angular=0.12,
        )
        controller = ZigzagMissionController.__new__(ZigzagMissionController)
        controller.path_follower = follower
        controller.control_period = control["period"]
        controller.last_command_time = controller_module.rospy.Time.from_sec(10.0)
        controller.last_linear = follower.last_linear
        controller.last_angular = follower.last_angular
        controller.mission_has_control = True
        controller.cmd_pub = RecordingPublisher()
        controller.state = controller.FOLLOWING
        controller._publish_diagnostics = lambda: None
        transitions = []

        def set_state(state, now=None):
            transitions.append(state)
            controller.state = state

        controller._set_state = set_state
        for step in range(1, 101):
            now = controller_module.rospy.Time.from_sec(
                10.0 + step * control["period"]
            )
            controller._decelerate_at_exit(now)
            if controller.state == controller.VERIFY_EXIT:
                break

        commands = controller.cmd_pub.messages
        self.assertGreater(len(commands), 1)
        self.assertGreater(commands[0].linear.x, 0.0)
        for previous, current in zip(commands, commands[1:]):
            self.assertLessEqual(
                abs(current.linear.x - previous.linear.x),
                control["linear_deceleration"] * control["period"] + 1e-9,
            )
            self.assertLessEqual(
                abs(current.angular.z - previous.angular.z),
                control["angular_acceleration"] * control["period"] + 1e-9,
            )
        self.assertEqual(controller.state, controller.VERIFY_EXIT)
        self.assertEqual(transitions, [controller.VERIFY_EXIT])
        self.assertAlmostEqual(commands[-1].linear.x, 0.0, places=12)
        self.assertAlmostEqual(commands[-1].angular.z, 0.0, places=12)

    def test_exit_confirmation_accepts_the_same_single_line_as_lane_control(self):
        controller = ZigzagMissionController.__new__(ZigzagMissionController)
        controller.lock = threading.RLock()
        controller.lane_width_min = self.config["exit"]["lane_width_min"]
        controller.lane_width_max = self.config["exit"]["lane_width_max"]
        controller.image_center = self.config["exit"]["image_center"]
        controller.single_line_center_offset = self.config["exit"][
            "single_line_center_offset"
        ]
        controller.lane_center_tolerance = self.config["exit"][
            "lane_center_tolerance"
        ]
        controller.exit_confirmation_max_gap = self.config["exit"][
            "confirmation_max_gap"
        ]
        controller.exit_confirmation_started = True
        controller.state = controller.VERIFY_EXIT
        controller.confirmation_started_at = controller_module.rospy.Time.from_sec(
            9.0
        )
        controller.boundary_confirmation_count = 0
        controller.last_boundary_confirmation_time = None
        message = SimpleNamespace(data=[math.nan, 772.0, 0.0, 1.0])

        with mock.patch.object(
            controller_module.rospy.Time,
            "now",
            return_value=controller_module.rospy.Time.from_sec(10.0),
        ):
            controller.boundary_callback(message)

        self.assertTrue(controller.boundary_valid)
        self.assertEqual(controller.boundary_confirmation_count, 1)


    def test_follow_checks_terminal_before_calculating_another_path_command(self):
        control = self.config["control"]
        follower = PathFollower(
            TrackingConfig(
                lookahead_distance=control["lookahead_distance"],
                maximum_linear_velocity=control["cruise_velocity"],
                maximum_angular_velocity=control["maximum_angular_velocity"],
                maximum_lateral_acceleration=control[
                    "maximum_lateral_acceleration"
                ],
                linear_acceleration=control["linear_acceleration"],
                linear_deceleration=control["linear_deceleration"],
                angular_acceleration=control["angular_acceleration"],
                heading_gain=control["heading_gain"],
                curvature_feedforward_weight=control[
                    "path_curvature_weight"
                ],
                lateral_feedback_gain=control["lateral_feedback_gain"],
                search_ahead_distance=control["nearest_search_ahead"],
            )
        )
        terminal_pose = Pose2D(
            float(self.path.x[-1]),
            float(self.path.y[-1]),
            float(self.path.heading[-1]),
        )
        follower.reset(
            self.path,
            terminal_pose,
            initial_linear=control["exit_velocity"],
        )
        controller = ZigzagMissionController.__new__(ZigzagMissionController)
        controller.lock = threading.RLock()
        controller.shutting_down = False
        controller.revoke_requested = False
        controller.manual_stop = False
        controller.mission_started = controller_module.rospy.Time.from_sec(9.0)
        controller.mission_timeout = 5.0
        controller.exit_stop_latched = False
        controller.odom_x = terminal_pose.x
        controller.odom_y = terminal_pose.y
        controller.odom_yaw = terminal_pose.yaw
        controller.odom_speed = control["exit_velocity"]
        controller.odom_angular_velocity = 0.0
        controller.path_follower = follower
        controller.path_validator = object()
        controller.committed_path = self.path
        controller.path_index = self.path.size - 1
        controller._tracking_input_problem = mock.Mock(return_value=None)
        controller._fresh_live_obstacles = mock.Mock(
            return_value=np.empty((0, 2), dtype=np.float64)
        )
        controller._motion_safety = mock.Mock(
            return_value=SimpleNamespace(
                speed_limit=math.inf,
                requires_stop=False,
                validation=ValidationResult(True),
            )
        )
        controller._common_command = mock.Mock(
            side_effect=AssertionError("terminal tick calculated a path command")
        )
        controller.exit_confirmation_started = True
        controller.handoff_remaining_distance = 0.02
        controller.exit_confirmation_lead_distance = 0.03
        controller.maximum_position_error_seen = 0.0
        controller.maximum_heading_error_seen = 0.0
        controller.control_period = control["period"]
        controller.last_command_time = controller_module.rospy.Time.from_sec(10.0)
        controller.last_linear = control["exit_velocity"]
        controller.last_angular = 0.0
        controller.linear_deceleration = control["linear_deceleration"]
        controller.mission_has_control = True
        controller.odom_stamp = controller_module.rospy.Time.from_sec(10.0)
        controller.scan_stamp = controller_module.rospy.Time.from_sec(10.0)
        controller.scan_received = controller_module.rospy.Time.from_sec(10.0)
        controller.scan_pose_stamp_delta = 0.0
        controller.maximum_pose_stamp_skew = self.config["start"][
            "maximum_pose_stamp_skew"
        ]
        controller.scan_pose_stamp_skew = self.config["scan"][
            "maximum_pose_stamp_skew"
        ]
        controller.odom_timeout = self.config["timeouts"]["odometry"]
        controller.scan_timeout = self.config["timeouts"]["scan"]
        controller.cmd_pub = RecordingPublisher()
        controller._publish_diagnostics = mock.Mock()
        controller.state = controller.FOLLOWING
        controller._set_state = mock.Mock()
        now = controller_module.rospy.Time.from_sec(10.0 + control["period"])

        with mock.patch.object(
            controller_module.rospy.Time, "now", return_value=now
        ):
            controller._follow(now)

        controller._common_command.assert_not_called()
        self.assertTrue(controller.exit_stop_latched)
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        self.assertLess(
            controller.cmd_pub.messages[0].linear.x,
            control["exit_velocity"],
        )

    def test_follow_brakes_with_common_slew_limit_for_stale_odometry(self):
        control = self.config["control"]
        follower = PathFollower(
            TrackingConfig(
                lookahead_distance=control["lookahead_distance"],
                maximum_linear_velocity=control["cruise_velocity"],
                maximum_angular_velocity=control["maximum_angular_velocity"],
                maximum_lateral_acceleration=control[
                    "maximum_lateral_acceleration"
                ],
                linear_acceleration=control["linear_acceleration"],
                linear_deceleration=control["linear_deceleration"],
                angular_acceleration=control["angular_acceleration"],
                heading_gain=control["heading_gain"],
                curvature_feedforward_weight=control[
                    "path_curvature_weight"
                ],
                lateral_feedback_gain=control["lateral_feedback_gain"],
                search_ahead_distance=control["nearest_search_ahead"],
            )
        )
        initial_linear = control["exit_velocity"]
        initial_angular = 0.12
        follower.reset(
            self.path,
            Pose2D(
                float(self.path.x[0]),
                float(self.path.y[0]),
                float(self.path.heading[0]),
            ),
            initial_linear=initial_linear,
            initial_angular=initial_angular,
        )
        controller = ZigzagMissionController.__new__(ZigzagMissionController)
        controller.lock = threading.RLock()
        controller.mission_started = controller_module.rospy.Time.from_sec(9.0)
        controller.mission_timeout = 5.0
        controller.path_follower = follower
        controller.control_period = control["period"]
        controller.last_command_time = controller_module.rospy.Time.from_sec(10.0)
        controller.last_linear = initial_linear
        controller.last_angular = initial_angular
        controller.mission_has_control = True
        controller.cmd_pub = RecordingPublisher()
        controller._publish_diagnostics = mock.Mock()
        controller._tracking_input_problem = mock.Mock(
            return_value="stale odometry"
        )
        controller._fail = mock.Mock()
        controller.state = controller.FOLLOWING
        now = controller_module.rospy.Time.from_sec(
            10.0 + control["period"]
        )

        with mock.patch.object(controller_module.rospy, "logwarn_throttle"):
            controller._follow(now)

        self.assertEqual(controller.state, controller.FOLLOWING)
        controller._fail.assert_not_called()
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        command = controller.cmd_pub.messages[0]
        self.assertGreater(command.linear.x, 0.0)
        self.assertLess(command.linear.x, initial_linear)
        self.assertAlmostEqual(
            initial_linear - command.linear.x,
            control["linear_deceleration"] * control["period"],
            places=12,
        )
        self.assertLess(abs(command.angular.z), abs(initial_angular))
        self.assertLessEqual(
            abs(command.angular.z - initial_angular),
            control["angular_acceleration"] * control["period"] + 1e-12,
        )
        self.assertAlmostEqual(follower.last_linear, command.linear.x, places=12)
        self.assertAlmostEqual(follower.last_angular, command.angular.z, places=12)
        controller._publish_diagnostics.assert_called_once_with()

    def test_failure_with_known_ownership_stops_without_rehandoff(self):
        controller = ZigzagMissionController.__new__(ZigzagMissionController)
        controller.state = controller.FOLLOWING
        controller.mission_has_control = True
        controller.handoff_ambiguous = False
        controller.speed_limit_pub = RecordingPublisher()
        controller.cmd_pub = RecordingPublisher()
        controller.emergency_stop_pub = RecordingPublisher()
        controller.state_pub = RecordingPublisher()
        controller.path_follower = SimpleNamespace(
            last_linear=0.07,
            last_angular=0.15,
        )
        controller.last_linear = 0.07
        controller.last_angular = 0.15
        handoff_calls = []
        controller._set_lane_controller = lambda enabled: handoff_calls.append(
            enabled
        )
        lane_stop_calls = []
        controller._stop_lane_controller = lambda: lane_stop_calls.append(False)
        fixed_now = controller_module.rospy.Time.from_sec(10.0)

        with mock.patch.object(
            controller_module.rospy.Time, "now", return_value=fixed_now
        ):
            controller._fail("test handoff failure")

        self.assertEqual(controller.state, controller.FAILED)
        self.assertTrue(controller.mission_has_control)
        self.assertEqual(handoff_calls, [])
        self.assertEqual(lane_stop_calls, [])
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        self.assertAlmostEqual(controller.cmd_pub.messages[0].linear.x, 0.0)
        self.assertAlmostEqual(controller.cmd_pub.messages[0].angular.z, 0.0)
        self.assertEqual(controller.speed_limit_pub.messages[-1].data, 0.0)
        self.assertEqual(controller.emergency_stop_pub.messages, [])

    def test_failed_ambiguous_ownership_reacquisition_never_publishes_cmd_vel(self):
        controller = ZigzagMissionController.__new__(ZigzagMissionController)
        controller.state = controller.FOLLOWING
        controller.mission_has_control = True
        controller.handoff_ambiguous = True
        controller.speed_limit_pub = RecordingPublisher()
        controller.cmd_pub = RecordingPublisher()
        controller.emergency_stop_pub = RecordingPublisher()
        controller.state_pub = RecordingPublisher()
        controller.path_follower = SimpleNamespace(
            last_linear=0.07,
            last_angular=0.15,
        )
        controller.last_linear = 0.07
        controller.last_angular = 0.15
        handoff_calls = []

        def fail_handoff(enabled):
            handoff_calls.append(enabled)
            return False

        controller._set_lane_controller = fail_handoff
        lane_stop_calls = []
        controller._stop_lane_controller = lambda: lane_stop_calls.append(False)
        fixed_now = controller_module.rospy.Time.from_sec(10.0)

        with mock.patch.object(
            controller_module.rospy.Time, "now", return_value=fixed_now
        ):
            controller._fail("test ambiguous handoff failure")

        self.assertEqual(controller.state, controller.FAILED)
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(handoff_calls, [False])
        self.assertEqual(lane_stop_calls, [False])
        self.assertEqual(controller.cmd_pub.messages, [])
        self.assertEqual(controller.speed_limit_pub.messages[-1].data, 0.0)
        self.assertTrue(controller.emergency_stop_pub.messages[-1].data)

    def test_shutdown_blocks_late_control_timer(self):
        controller = ZigzagMissionController.__new__(ZigzagMissionController)
        controller.lock = threading.RLock()
        controller.shutting_down = False
        controller.mission_has_control = True
        controller.cmd_pub = RecordingPublisher()
        controller.state = controller.FOLLOWING
        controller.revoke_requested = False
        controller.start_requested = False
        controller.zone_gate = True
        controller.manual_stop = False
        controller._follow = mock.Mock()

        fixed_now = controller_module.rospy.Time.from_sec(10.0)
        with mock.patch.object(
            controller_module.rospy.Time, "now", return_value=fixed_now
        ):
            controller.shutdown()
            controller.control_callback(None)

        self.assertTrue(controller.shutting_down)
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        controller._follow.assert_not_called()

    def test_odom_callback_is_not_starved_by_sweep_and_shutdown_wins(self):
        index = int(0.40 * (self.path.size - 1))
        pose = (
            float(self.path.x[index]),
            float(self.path.y[index]),
            float(self.path.heading[index]),
        )
        controller = self._configure_production_safety(
            self._scan_controller(robot_pose=pose, stamp=10.0)
        )
        controller.path_follower.reset(
            controller.committed_path,
            Pose2D(*pose),
            initial_linear=self.config["control"]["cruise_velocity"],
        )
        controller.state = controller.FOLLOWING
        controller.mission_has_control = True
        controller.shutting_down = False
        controller.revoke_requested = False
        controller.start_requested = False
        controller.zone_gate = True
        controller.manual_stop = False
        controller.mission_started = controller_module.rospy.Time.from_sec(9.0)
        controller.mission_timeout = self.config["timeouts"]["mission"]
        controller.odom_timeout = self.config["timeouts"]["odometry"]
        controller.scan_stamp = controller_module.rospy.Time.from_sec(10.0)
        controller.scan_received = controller_module.rospy.Time.from_sec(10.0)
        controller.scan_pose_stamp_delta = 0.0
        controller.exit_stop_latched = False
        controller.exit_confirmation_started = False
        controller.handoff_remaining_distance = self.config["exit"][
            "handoff_remaining_distance"
        ]
        controller.exit_confirmation_lead_distance = self.config["exit"][
            "confirmation_lead_distance"
        ]
        controller.maximum_position_error_seen = 0.0
        controller.maximum_heading_error_seen = 0.0
        controller.control_period = self.config["control"]["period"]
        controller.last_command_time = controller_module.rospy.Time.from_sec(10.0)
        controller.cmd_pub = RecordingPublisher()
        controller._publish_diagnostics = mock.Mock()

        sweep_entered = threading.Event()
        release_sweep = threading.Event()
        original_motion_safety = controller._motion_safety

        def blocked_motion_safety(*args, **kwargs):
            sweep_entered.set()
            self.assertTrue(release_sweep.wait(timeout=1.0))
            return original_motion_safety(*args, **kwargs)

        clock = [10.0]
        with mock.patch.object(
            controller_module.rospy.Time,
            "now",
            side_effect=lambda: controller_module.rospy.Time.from_sec(clock[0]),
        ), mock.patch.object(
            controller,
            "_motion_safety",
            side_effect=blocked_motion_safety,
        ):
            control_thread = threading.Thread(
                target=controller.control_callback,
                args=(None,),
            )
            control_thread.start()
            self.assertTrue(sweep_entered.wait(timeout=1.0))
            clock[0] = 10.02
            callback_thread = threading.Thread(
                target=controller.odom_callback,
                args=(self._odom_message(10.02, (pose[0] - 0.001, pose[1], pose[2])),),
            )
            callback_thread.start()
            callback_thread.join(timeout=0.20)
            callback_was_responsive = not callback_thread.is_alive()
            controller.shutdown()
            release_sweep.set()
            control_thread.join(timeout=2.0)
            callback_thread.join(timeout=2.0)

        self.assertTrue(callback_was_responsive)
        self.assertFalse(control_thread.is_alive())
        self.assertAlmostEqual(controller.odom_x, pose[0] - 0.001, places=12)
        self.assertTrue(controller.shutting_down)
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        self.assertAlmostEqual(controller.cmd_pub.messages[0].linear.x, 0.0)
        self.assertAlmostEqual(controller.cmd_pub.messages[0].angular.z, 0.0)

    def _acquiring_controller(self, pose, stamp=10.0):
        controller = self._configure_production_safety(
            self._scan_controller(robot_pose=pose, stamp=stamp)
        )
        fixed_stamp = controller_module.rospy.Time.from_sec(stamp)
        controller.map_path = self.path
        controller.arm_generation = 1
        controller.prepared_generation = 1
        controller.prepared_path = RigidTransform2D(
            0.0, 0.0, 0.0, self.path.frame_id, "odom"
        ).apply_path(self.path)
        controller.prepared_route_from_odom = RigidTransform2D(
            0.0, 0.0, 0.0, "odom", self.path.frame_id
        )
        controller.odom_timeout = self.config["timeouts"]["odometry"]
        controller.route_odom_aligned = self.config["route"]["odom_aligned"]
        controller.acquisition_timeout = self.config["timeouts"]["acquisition"]
        controller.start_maximum_path_error = self.config["start"][
            "maximum_path_error"
        ]
        controller.start_maximum_join_heading_error = math.radians(
            self.config["start"]["maximum_join_heading_error_deg"]
        )
        controller.entry_velocity_cap = self.config["control"][
            "entry_velocity_cap"
        ]
        controller.maximum_angular_velocity = self.config["control"][
            "maximum_angular_velocity"
        ]
        controller.observed_lane_linear = 0.098
        controller.observed_lane_angular = 0.0
        controller.scan_stamp = fixed_stamp
        controller.scan_received = fixed_stamp
        controller.scan_pose_stamp_delta = 0.0
        controller.live_obstacle_points = np.empty((0, 2), dtype=np.float64)
        controller.state = controller.ACQUIRING
        controller.state_started = controller_module.rospy.Time.from_sec(
            stamp - 0.05
        )
        controller.start_requested = False
        controller.revoke_requested = False
        controller.zone_gate = True
        controller.manual_stop = False
        controller.shutting_down = False
        controller.mission_has_control = False
        controller.handoff_ambiguous = False
        controller.mission_started = None
        controller.mission_timeout = self.config["timeouts"]["mission"]
        controller.committed_path = None
        controller.route_from_odom = None
        controller.exit_stop_latched = False
        controller.exit_confirmation_started = False
        controller.handoff_remaining_distance = self.config["exit"][
            "handoff_remaining_distance"
        ]
        controller.exit_confirmation_lead_distance = self.config["exit"][
            "confirmation_lead_distance"
        ]
        controller.maximum_position_error_seen = 0.0
        controller.maximum_heading_error_seen = 0.0
        controller.last_command_time = None
        controller.last_linear = 0.0
        controller.last_angular = 0.0
        controller.state_pub = RecordingPublisher()
        controller.path_pub = RecordingPublisher()
        controller.cmd_pub = RecordingPublisher()
        controller._publish_diagnostics = mock.Mock()
        return controller

    def test_acquire_sweep_does_not_starve_odom_and_uses_completion_time(self):
        entry_index = int(np.argmin(np.abs(self.path.x - 0.33)))
        latest_index = min(entry_index + 5, self.path.size - 1)
        entry_pose = (
            float(self.path.x[entry_index]),
            float(self.path.y[entry_index]),
            float(self.path.heading[entry_index]),
        )
        latest_pose = (
            float(self.path.x[latest_index]),
            float(self.path.y[latest_index]),
            float(self.path.heading[latest_index]),
        )
        controller = self._acquiring_controller(entry_pose)
        handoff_calls = []

        def handoff(enabled):
            handoff_calls.append(enabled)
            controller.mission_has_control = not enabled
            return True

        controller._set_lane_controller = handoff
        follow_calls = []
        original_follow = controller._follow

        def observed_follow(when):
            follow_calls.append(
                (when, controller._tracking_input_problem(when))
            )
            original_follow(when)

        controller._follow = observed_follow
        sweep_entered = threading.Event()
        release_sweep = threading.Event()
        original_validate = controller.path_validator.validate_path

        def blocked_validate(*args, **kwargs):
            sweep_entered.set()
            if not release_sweep.wait(timeout=1.0):
                raise RuntimeError("timed out waiting to release acquisition sweep")
            return original_validate(*args, **kwargs)

        clock = [10.0]
        control_errors = []

        def run_control():
            try:
                controller.control_callback(None)
            except BaseException as error:  # Thread failures must fail the test.
                control_errors.append(error)

        with mock.patch.object(
            controller_module.rospy.Time,
            "now",
            side_effect=lambda: controller_module.rospy.Time.from_sec(clock[0]),
        ), mock.patch.object(
            controller.path_validator,
            "validate_path",
            side_effect=blocked_validate,
        ):
            control_thread = threading.Thread(target=run_control)
            control_thread.start()
            self.assertTrue(sweep_entered.wait(timeout=1.0))
            clock[0] = 10.10
            callback_thread = threading.Thread(
                target=controller.odom_callback,
                args=(self._odom_message(10.10, latest_pose),),
            )
            callback_thread.start()
            callback_thread.join(timeout=0.20)
            callback_was_responsive = not callback_thread.is_alive()
            lane_command = controller_module.Twist()
            lane_command.linear.x = 0.099
            controller.command_observer_callback(lane_command)
            release_sweep.set()
            control_thread.join(timeout=2.0)
            callback_thread.join(timeout=2.0)

        self.assertTrue(callback_was_responsive)
        self.assertFalse(control_thread.is_alive())
        self.assertEqual(control_errors, [])
        self.assertEqual(handoff_calls, [False])
        self.assertEqual(controller.state, controller.FOLLOWING)
        self.assertEqual(len(controller.path_pub.messages), 1)
        self.assertEqual(controller.committed_path.frame_id, "odom")
        self.assertEqual(controller.path_pub.messages[0].header.frame_id, "map")
        self.assertEqual(len(follow_calls), 1)
        self.assertAlmostEqual(follow_calls[0][0].to_sec(), 10.10, places=8)
        self.assertIsNone(follow_calls[0][1])
        self.assertAlmostEqual(controller.last_command_time.to_sec(), 10.10, places=8)
        self.assertAlmostEqual(controller.mission_started.to_sec(), 10.10, places=8)
        self.assertAlmostEqual(controller.last_linear, 0.099, places=12)
        self.assertGreaterEqual(controller.path_index, latest_index - 1)
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        self.assertAlmostEqual(
            controller.cmd_pub.messages[0].linear.x,
            0.099,
            places=12,
        )

    def test_acquire_commits_prevalidated_rotated_local_route(self):
        entry_index = int(np.argmin(np.abs(self.path.x - 0.33)))
        map_pose = (
            float(self.path.x[entry_index]),
            float(self.path.y[entry_index]),
            float(self.path.heading[entry_index]),
        )
        odom_pose = (
            map_pose[0] + 0.31,
            map_pose[1] - 0.18,
            normalize_angle(map_pose[2] + math.radians(7.0)),
        )
        controller = self._acquiring_controller(map_pose)
        controller.route_odom_aligned = False
        controller.map_path = RigidTransform2D(
            0.0,
            0.0,
            0.0,
            self.path.frame_id,
            "map",
        ).apply_path(self.path)
        controller._publish_path(
            controller.map_path, stamp=controller_module.rospy.Time()
        )
        surveyed_message = controller.path_pub.messages[0]
        local_to_odom = RigidTransform2D.from_pose_pair(
            Pose2D(*map_pose),
            Pose2D(*odom_pose),
            source_frame="map",
            target_frame="odom",
        )
        controller.prepared_path = local_to_odom.apply_path(controller.map_path)
        controller.prepared_route_from_odom = local_to_odom.inverse()
        controller.odom_x, controller.odom_y, controller.odom_yaw = odom_pose
        fixed_stamp = controller_module.rospy.Time.from_sec(10.0)
        controller.odom_history = deque(
            [(fixed_stamp, *odom_pose, "odom")], maxlen=200
        )
        handoffs = []

        def handoff(enabled):
            handoffs.append(enabled)
            controller.mission_has_control = not enabled
            return True

        controller._set_lane_controller = handoff
        with mock.patch.object(
            controller_module.rospy.Time, "now", return_value=fixed_stamp
        ):
            self.assertTrue(controller._acquire(fixed_stamp))

        self.assertEqual(controller.state, controller.FOLLOWING)
        self.assertEqual(handoffs, [False])
        self.assertEqual(len(controller.path_pub.messages), 2)
        committed_message = controller.path_pub.messages[1]
        self.assertEqual(surveyed_message.header.frame_id, "map")
        self.assertEqual(committed_message.header.frame_id, "odom")
        self.assertEqual(controller.committed_path.frame_id, "odom")
        for index in (0, controller.committed_path.size // 2, -1):
            pose = committed_message.poses[index].pose
            self.assertAlmostEqual(
                pose.position.x,
                float(controller.committed_path.x[index]),
                places=12,
            )
            self.assertAlmostEqual(
                pose.position.y,
                float(controller.committed_path.y[index]),
                places=12,
            )
        self.assertNotAlmostEqual(
            surveyed_message.poses[0].pose.position.x,
            committed_message.poses[0].pose.position.x,
            places=6,
        )
        recovered_map_pose = controller.route_from_odom.apply_pose(
            Pose2D(*odom_pose)
        )
        self.assertAlmostEqual(recovered_map_pose.x, map_pose[0], places=10)
        self.assertAlmostEqual(recovered_map_pose.y, map_pose[1], places=10)
        self.assertAlmostEqual(
            normalize_angle(recovered_map_pose.yaw - map_pose[2]),
            0.0,
            places=10,
        )
        self.assertLess(
            controller.path_follower.diagnostics.position_error,
            1e-9,
        )

    def test_acquire_uses_prevalidated_local_route_without_map_pose(self):
        entry_index = int(np.argmin(np.abs(self.path.station - 0.55)))
        entry_pose = (
            float(self.path.x[entry_index]),
            float(self.path.y[entry_index]),
            float(self.path.heading[entry_index]),
        )
        controller = self._acquiring_controller(entry_pose)
        controller.arm_generation = 31
        controller.prepared_generation = 31
        controller.prepared_path = self.path
        controller.prepared_route_from_odom = RigidTransform2D(
            0.0, 0.0, 0.0, "odom", "map"
        )
        handoffs = []

        def handoff(enabled):
            handoffs.append(enabled)
            controller.mission_has_control = not enabled
            return True

        controller._set_lane_controller = handoff
        now = controller_module.rospy.Time.from_sec(10.0)
        with mock.patch.object(
            controller_module.rospy.Time, "now", return_value=now
        ):
            self.assertTrue(controller._acquire(now))

        self.assertEqual(handoffs, [False])
        self.assertEqual(controller.state, controller.FOLLOWING)
        self.assertIs(controller.committed_path, self.path)
        self.assertEqual(controller.route_from_odom.source_frame, "odom")
        self.assertEqual(controller.route_from_odom.target_frame, "map")

    def test_lane_join_success_returns_control_and_completes(self):
        controller = ZigzagMissionController.__new__(ZigzagMissionController)
        controller.state = controller.VERIFY_EXIT
        controller.state_started = controller_module.rospy.Time.from_sec(9.5)
        controller.mission_started = controller_module.rospy.Time.from_sec(8.0)
        controller.mission_has_control = True
        controller.handoff_ambiguous = False
        controller.join_velocity_cap = self.config["exit"]["join_velocity_cap"]
        controller.lane_resume_max_velocity = self.config["control"][
            "lane_resume_max_velocity"
        ]
        controller.join_timeout = self.config["exit"]["join_timeout"]
        controller.join_minimum_distance = self.config["exit"][
            "join_minimum_distance"
        ]
        controller.join_confirmation_frames = self.config["exit"][
            "join_confirmation_frames"
        ]
        controller.exit_confirmation_max_gap = self.config["exit"][
            "confirmation_max_gap"
        ]
        controller.boundary_timeout = self.config["exit"]["boundary_timeout"]
        controller.join_origin_ready = False
        controller.odom_x = -1.50
        controller.odom_y = 1.75
        controller.odom_yaw = math.pi
        controller.odom_stamp = controller_module.rospy.Time.from_sec(10.05)
        controller.maximum_position_error_seen = 0.004
        controller.maximum_heading_error_seen = math.radians(2.0)
        controller.path_follower = PathFollower(
            TrackingConfig(
                lookahead_distance=0.05,
                maximum_linear_velocity=0.10,
                maximum_angular_velocity=0.8,
                maximum_lateral_acceleration=0.04,
                linear_acceleration=0.1,
                linear_deceleration=0.1,
                angular_acceleration=0.8,
                heading_gain=0.35,
            )
        )
        controller.speed_limit_pub = RecordingPublisher()
        controller.state_pub = RecordingPublisher()
        controller._publish_diagnostics = mock.Mock()
        controller._tracking_input_problem = mock.Mock(return_value=None)
        handoffs = []

        def handoff(enabled):
            handoffs.append(enabled)
            controller.mission_has_control = not enabled
            return True

        controller._set_lane_controller = handoff
        start = controller_module.rospy.Time.from_sec(10.0)
        with mock.patch.object(
            controller_module.rospy.Time, "now", return_value=start
        ):
            controller._start_lane_join(start)

        self.assertEqual(controller.state, controller.JOINING_LANE)
        self.assertEqual(handoffs, [True])
        first = controller_module.rospy.Time.from_sec(10.05)
        controller._join_lane(first)
        self.assertTrue(controller.join_origin_ready)

        completed = controller_module.rospy.Time.from_sec(10.10)
        controller.odom_x -= controller.join_minimum_distance + 0.001
        controller.odom_stamp = completed
        controller.boundary_valid = True
        controller.boundary_stamp = completed
        controller.last_boundary_confirmation_time = completed
        controller.boundary_confirmation_count = (
            controller.join_confirmation_frames
        )
        controller._join_lane(completed)

        self.assertEqual(controller.state, controller.COMPLETE)
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(
            [message.data for message in controller.speed_limit_pub.messages],
            [controller.join_velocity_cap, controller.lane_resume_max_velocity],
        )

    def test_shutdown_during_acquire_sweep_prevents_late_handoff(self):
        entry_index = int(np.argmin(np.abs(self.path.x - 0.33)))
        pose = (
            float(self.path.x[entry_index]),
            float(self.path.y[entry_index]),
            float(self.path.heading[entry_index]),
        )
        controller = self._acquiring_controller(pose)
        controller._set_lane_controller = mock.Mock(return_value=True)
        controller._follow = mock.Mock()
        sweep_entered = threading.Event()
        release_sweep = threading.Event()
        original_validate = controller.path_validator.validate_path

        def blocked_validate(*args, **kwargs):
            sweep_entered.set()
            if not release_sweep.wait(timeout=1.0):
                raise RuntimeError("timed out waiting to release acquisition sweep")
            return original_validate(*args, **kwargs)

        fixed_now = controller_module.rospy.Time.from_sec(10.0)
        control_errors = []

        def run_control():
            try:
                controller.control_callback(None)
            except BaseException as error:  # Thread failures must fail the test.
                control_errors.append(error)

        with mock.patch.object(
            controller_module.rospy.Time, "now", return_value=fixed_now
        ), mock.patch.object(
            controller.path_validator,
            "validate_path",
            side_effect=blocked_validate,
        ):
            control_thread = threading.Thread(target=run_control)
            control_thread.start()
            self.assertTrue(sweep_entered.wait(timeout=1.0))
            shutdown_thread = threading.Thread(target=controller.shutdown)
            shutdown_thread.start()
            shutdown_thread.join(timeout=0.20)
            shutdown_was_responsive = not shutdown_thread.is_alive()
            release_sweep.set()
            control_thread.join(timeout=2.0)
            shutdown_thread.join(timeout=2.0)

        self.assertTrue(shutdown_was_responsive)
        self.assertFalse(control_thread.is_alive())
        self.assertEqual(control_errors, [])
        self.assertTrue(controller.shutting_down)
        controller._set_lane_controller.assert_not_called()
        controller._follow.assert_not_called()
        self.assertIsNone(controller.committed_path)
        self.assertEqual(controller.path_pub.messages, [])
        self.assertEqual(controller.cmd_pub.messages, [])

    def test_stale_acquire_snapshot_is_not_relabelled_with_fresh_inputs(self):
        entry_index = int(np.argmin(np.abs(self.path.x - 0.33)))
        pose = (
            float(self.path.x[entry_index]),
            float(self.path.y[entry_index]),
            float(self.path.heading[entry_index]),
        )
        controller = self._acquiring_controller(pose)
        controller._set_lane_controller = mock.Mock(return_value=True)
        controller._follow = mock.Mock()
        sweep_entered = threading.Event()
        release_sweep = threading.Event()
        original_validate = controller.path_validator.validate_path

        def blocked_validate(*args, **kwargs):
            sweep_entered.set()
            if not release_sweep.wait(timeout=1.0):
                raise RuntimeError("timed out waiting to release acquisition sweep")
            return original_validate(*args, **kwargs)

        clock = [10.0]
        control_errors = []

        def run_control():
            try:
                controller.control_callback(None)
            except BaseException as error:  # Thread failures must fail the test.
                control_errors.append(error)

        with mock.patch.object(
            controller_module.rospy.Time,
            "now",
            side_effect=lambda: controller_module.rospy.Time.from_sec(clock[0]),
        ), mock.patch.object(
            controller.path_validator,
            "validate_path",
            side_effect=blocked_validate,
        ), mock.patch.object(controller_module.rospy, "logwarn_throttle"):
            control_thread = threading.Thread(target=run_control)
            control_thread.start()
            self.assertTrue(sweep_entered.wait(timeout=1.0))
            clock[0] = 10.40
            fresh_stamp = controller_module.rospy.Time.from_sec(10.40)
            refresh_errors = []

            def refresh_inputs():
                try:
                    controller.odom_callback(self._odom_message(10.40, pose))
                    with controller.lock:
                        controller.prepared_generation += 1
                        controller.scan_stamp = fresh_stamp
                        controller.scan_received = fresh_stamp
                        controller.scan_pose_stamp_delta = 0.0
                except BaseException as error:
                    refresh_errors.append(error)

            refresh_thread = threading.Thread(target=refresh_inputs)
            refresh_thread.start()
            refresh_thread.join(timeout=0.20)
            refresh_was_responsive = not refresh_thread.is_alive()
            release_sweep.set()
            control_thread.join(timeout=2.0)
            refresh_thread.join(timeout=2.0)

        self.assertTrue(refresh_was_responsive)
        self.assertFalse(control_thread.is_alive())
        self.assertEqual(control_errors, [])
        self.assertEqual(refresh_errors, [])
        self.assertEqual(controller.state, controller.ACQUIRING)
        self.assertFalse(controller.mission_has_control)
        self.assertIsNone(controller.committed_path)
        controller._set_lane_controller.assert_not_called()
        controller._follow.assert_not_called()
        self.assertEqual(controller.path_pub.messages, [])
        self.assertEqual(controller.cmd_pub.messages, [])

    def test_acquire_drops_candidate_after_odometry_frame_change(self):
        entry_index = int(np.argmin(np.abs(self.path.x - 0.33)))
        pose = (
            float(self.path.x[entry_index]),
            float(self.path.y[entry_index]),
            float(self.path.heading[entry_index]),
        )
        controller = self._acquiring_controller(pose)
        controller._set_lane_controller = mock.Mock(return_value=True)
        controller._follow = mock.Mock()
        sweep_entered = threading.Event()
        release_sweep = threading.Event()
        original_validate = controller.path_validator.validate_path

        def blocked_validate(*args, **kwargs):
            sweep_entered.set()
            if not release_sweep.wait(timeout=1.0):
                raise RuntimeError("timed out waiting to release acquisition sweep")
            return original_validate(*args, **kwargs)

        clock = [10.0]
        control_errors = []

        def run_control():
            try:
                controller.control_callback(None)
            except BaseException as error:  # Thread failures must fail the test.
                control_errors.append(error)

        with mock.patch.object(
            controller_module.rospy.Time,
            "now",
            side_effect=lambda: controller_module.rospy.Time.from_sec(clock[0]),
        ), mock.patch.object(
            controller.path_validator,
            "validate_path",
            side_effect=blocked_validate,
        ):
            control_thread = threading.Thread(target=run_control)
            control_thread.start()
            self.assertTrue(sweep_entered.wait(timeout=1.0))
            clock[0] = 10.05
            changed_frame = self._odom_message(10.05, pose)
            changed_frame.header.frame_id = "replacement_odom"
            callback_thread = threading.Thread(
                target=controller.odom_callback,
                args=(changed_frame,),
            )
            callback_thread.start()
            callback_thread.join(timeout=0.20)
            callback_was_responsive = not callback_thread.is_alive()
            release_sweep.set()
            control_thread.join(timeout=2.0)
            callback_thread.join(timeout=2.0)

        self.assertTrue(callback_was_responsive)
        self.assertFalse(control_thread.is_alive())
        self.assertEqual(control_errors, [])
        self.assertEqual(controller.odom_frame, "replacement_odom")
        self.assertEqual(controller.state, controller.ACQUIRING)
        self.assertFalse(controller.mission_has_control)
        self.assertIsNone(controller.committed_path)
        controller._set_lane_controller.assert_not_called()
        controller._follow.assert_not_called()
        self.assertEqual(controller.path_pub.messages, [])
        self.assertEqual(controller.cmd_pub.messages, [])

    def _scan_controller(self, robot_pose=(0.0, 0.0, 0.0), stamp=10.0):
        controller = ZigzagMissionController.__new__(ZigzagMissionController)
        controller.lock = threading.RLock()
        controller.scan_frame = self.config["scan"]["frame_id"]
        controller.scan_maximum_range = self.config["scan"]["maximum_range"]
        controller.scan_pose_stamp_skew = self.config["scan"][
            "maximum_pose_stamp_skew"
        ]
        controller.odom_history_duration = self.config["scan"][
            "odom_history_duration"
        ]
        controller.pending_scan_limit = self.config["scan"][
            "pending_scan_limit"
        ]
        controller.scan_timeout = self.config["timeouts"]["scan"]
        controller.lidar_x = self.config["scan"]["lidar_x"]
        controller.lidar_y = self.config["scan"]["lidar_y"]
        controller.lidar_yaw = self.config["scan"]["lidar_yaw"]
        controller.maximum_pose_stamp_skew = self.config["start"][
            "maximum_pose_stamp_skew"
        ]
        controller.odom_frame = "odom"
        controller.odom_ready = True
        controller.odom_x = robot_pose[0]
        controller.odom_y = robot_pose[1]
        controller.odom_yaw = robot_pose[2]
        controller.odom_speed = 0.0
        controller.odom_angular_velocity = 0.0
        controller.odom_stamp = controller_module.rospy.Time.from_sec(stamp)
        controller.odom_history = deque(
            [
                (
                    controller_module.rospy.Time.from_sec(stamp),
                    robot_pose[0],
                    robot_pose[1],
                    robot_pose[2],
                    "odom",
                )
            ],
            maxlen=200,
        )
        controller.pending_scans = deque()
        controller.live_obstacle_points = np.empty((0, 2), dtype=np.float64)
        controller.scan_stamp = None
        controller.scan_received = None
        controller.scan_pose_stamp_delta = math.inf
        return controller

    def _scan_message(self, stamp, ranges):
        message = controller_module.LaserScan()
        message.header.stamp = controller_module.rospy.Time.from_sec(stamp)
        message.header.frame_id = self.config["scan"]["frame_id"]
        message.angle_min = 0.0
        message.angle_increment = 0.10
        message.range_min = 0.10
        message.range_max = 40.0
        message.ranges = list(ranges)
        return message

    @staticmethod
    def _odom_message(stamp, pose):
        message = controller_module.Odometry()
        message.header.stamp = controller_module.rospy.Time.from_sec(stamp)
        message.header.frame_id = "odom"
        message.pose.pose.position.x = pose[0]
        message.pose.pose.position.y = pose[1]
        message.pose.pose.orientation.z = math.sin(0.5 * pose[2])
        message.pose.pose.orientation.w = math.cos(0.5 * pose[2])
        return message

    def test_scan_uses_source_stamp_aligned_odom_pose(self):
        controller = self._scan_controller(robot_pose=(99.0, 99.0, 0.0))
        scan_stamp = 9.98
        controller.odom_history = deque(
            [
                (
                    controller_module.rospy.Time.from_sec(9.94),
                    -5.0,
                    -5.0,
                    0.0,
                    "odom",
                ),
                (
                    controller_module.rospy.Time.from_sec(scan_stamp),
                    1.0,
                    2.0,
                    0.5 * math.pi,
                    "odom",
                ),
                (
                    controller_module.rospy.Time.from_sec(10.00),
                    99.0,
                    99.0,
                    0.0,
                    "odom",
                ),
            ],
            maxlen=200,
        )
        now = controller_module.rospy.Time.from_sec(10.00)
        with mock.patch.object(
            controller_module.rospy.Time, "now", return_value=now
        ):
            controller.scan_callback(self._scan_message(scan_stamp, [1.0]))

        expected_sensor_forward = self.config["scan"]["lidar_x"] + 1.0
        np.testing.assert_allclose(
            controller.live_obstacle_points,
            [[1.0, 2.0 + expected_sensor_forward]],
            atol=1e-12,
        )
        self.assertEqual(
            controller.scan_stamp,
            controller_module.rospy.Time.from_sec(scan_stamp),
        )
        self.assertAlmostEqual(controller.scan_pose_stamp_delta, 0.0)
        self.assertIsNotNone(controller._fresh_live_obstacles(now))
        stale = controller_module.rospy.Time.from_sec(
            10.00 + self.config["timeouts"]["scan"] + 0.01
        )
        self.assertIsNone(controller._fresh_live_obstacles(stale))

    def test_scan_without_aligned_odom_has_no_current_pose_fallback(self):
        controller = self._scan_controller(
            robot_pose=(99.0, 99.0, 0.0),
            stamp=9.90,
        )
        now = controller_module.rospy.Time.from_sec(10.0)
        with mock.patch.object(
            controller_module.rospy.Time, "now", return_value=now
        ):
            controller.scan_callback(self._scan_message(10.0, [0.20]))

        self.assertIsNone(controller.scan_stamp)
        self.assertEqual(controller.live_obstacle_points.shape, (0, 2))
        self.assertIsNone(controller._fresh_live_obstacles(now))
        self.assertEqual(len(controller.pending_scans), 1)

    def test_scan_waits_for_right_odom_bracket_then_interpolates(self):
        controller = self._scan_controller(
            robot_pose=(0.0, 0.0, 0.0), stamp=10.00
        )
        scan_time = controller_module.rospy.Time.from_sec(10.02)
        with mock.patch.object(
            controller_module.rospy.Time, "now", return_value=scan_time
        ):
            controller.scan_callback(self._scan_message(10.02, [0.40]))

        self.assertIsNone(controller.scan_stamp)
        self.assertEqual(len(controller.pending_scans), 1)

        odom_time = controller_module.rospy.Time.from_sec(10.04)
        with mock.patch.object(
            controller_module.rospy.Time, "now", return_value=odom_time
        ):
            controller.odom_callback(
                self._odom_message(10.04, (0.04, 0.0, 0.0))
            )

        self.assertEqual(len(controller.pending_scans), 0)
        self.assertEqual(controller.scan_stamp, scan_time)
        self.assertAlmostEqual(controller.scan_pose_stamp_delta, 0.02)
        self.assertEqual(controller.live_obstacle_points.shape, (1, 2))
        self.assertAlmostEqual(
            controller.live_obstacle_points[0, 0],
            0.02 + self.config["scan"]["lidar_x"] + 0.40,
            delta=1e-8,
        )
        self.assertAlmostEqual(controller.live_obstacle_points[0, 1], 0.0)

    def test_pending_scan_queue_is_bounded_and_rejects_wide_brackets(self):
        controller = self._scan_controller(stamp=10.00)
        controller.pending_scan_limit = 2
        for stamp in (10.01, 10.02, 10.03):
            now = controller_module.rospy.Time.from_sec(stamp)
            with mock.patch.object(
                controller_module.rospy.Time, "now", return_value=now
            ):
                controller.scan_callback(self._scan_message(stamp, [0.40]))

        self.assertEqual(len(controller.pending_scans), 2)
        self.assertAlmostEqual(
            controller.pending_scans[0]["source_stamp"].to_sec(), 10.02
        )

        now = controller_module.rospy.Time.from_sec(10.20)
        with mock.patch.object(
            controller_module.rospy.Time, "now", return_value=now
        ):
            controller.odom_callback(
                self._odom_message(10.20, (0.20, 0.0, 0.0))
            )

        self.assertEqual(len(controller.pending_scans), 0)
        self.assertIsNone(controller.scan_stamp)
        self.assertEqual(controller.live_obstacle_points.shape, (0, 2))

    def test_live_lidar_point_drives_common_stopping_sweep(self):
        footprint_config = self.config["footprint"]
        control = self.config["control"]
        footprint = AsymmetricFootprint(
            footprint_config["front"],
            footprint_config["rear"],
            footprint_config["half_width"],
        )
        path = CommonPath(
            x=np.asarray([0.0, 0.50]),
            y=np.asarray([0.0, 0.0]),
            heading=np.asarray([0.0, 0.0]),
            curvature=np.asarray([0.0, 0.0]),
            speed=np.asarray([0.10, 0.10]),
            direction=1,
            frame_id="odom",
            safety=PathSafety(
                margins=SafetyMargins(
                    obstacle=footprint_config["obstacle_margin"],
                    localization=footprint_config["localization_error"],
                    tracking=footprint_config["tracking_error"],
                )
            ),
        )
        follower = PathFollower(
            TrackingConfig(
                lookahead_distance=control["lookahead_distance"],
                maximum_linear_velocity=control["cruise_velocity"],
                maximum_angular_velocity=control["maximum_angular_velocity"],
                maximum_lateral_acceleration=control[
                    "maximum_lateral_acceleration"
                ],
                linear_acceleration=control["linear_acceleration"],
                linear_deceleration=control["linear_deceleration"],
                angular_acceleration=control["angular_acceleration"],
                heading_gain=control["heading_gain"],
                curvature_feedforward_weight=control[
                    "path_curvature_weight"
                ],
                lateral_feedback_gain=control["lateral_feedback_gain"],
                search_ahead_distance=control["nearest_search_ahead"],
            )
        )
        follower.reset(path, Pose2D(0.0, 0.0, 0.0), initial_linear=0.10)
        controller = self._scan_controller()
        controller.path_validator = SweptFootprintValidator(
            footprint,
            translation_step=footprint_config["sweep_translation_step"],
            heading_step=math.radians(
                footprint_config["sweep_heading_step_deg"]
            ),
        )
        controller.path_follower = follower
        controller.committed_path = path
        controller.live_route_safety = PathSafety(
            margins=SafetyMargins(
                line=footprint_config["line_margin"],
                obstacle=footprint_config["obstacle_margin"],
                localization=footprint_config["localization_error"],
                tracking=footprint_config["tracking_error"],
            )
        )
        controller.path_index = 0
        controller.last_linear = 0.10
        controller.last_angular = 0.0
        controller.odom_speed = 0.10
        controller.odom_angular_velocity = 0.0
        controller.prediction_reaction_time = self.config["texture"][
            "prediction_reaction_time"
        ]
        controller.prediction_distance_margin = self.config["texture"][
            "prediction_distance_margin"
        ]
        controller.linear_deceleration = control["linear_deceleration"]
        controller.nearest_search_ahead = control["nearest_search_ahead"]

        now = controller_module.rospy.Time.from_sec(10.0)
        with mock.patch.object(
            controller_module.rospy.Time, "now", return_value=now
        ):
            controller.scan_callback(self._scan_message(10.0, [0.15]))
        live_obstacles = controller._fresh_live_obstacles(now)
        self.assertIsNotNone(live_obstacles)
        clear = controller._motion_safety(
            0.10, np.empty((0, 2), dtype=np.float64)
        )
        blocked = controller._motion_safety(0.10, live_obstacles)
        self.assertFalse(clear.requires_stop)
        self.assertTrue(blocked.requires_stop)
        self.assertLessEqual(
            blocked.validation.minimum_obstacle_clearance, 0.0
        )

        steering_tracking = follower.calculate_tracking(
            Pose2D(0.0, 0.0, math.radians(10.0))
        )
        self.assertNotAlmostEqual(steering_tracking.angular_velocity, 0.0)
        with mock.patch.object(
            controller.path_validator,
            "motion_safety",
            wraps=controller.path_validator.motion_safety,
        ) as motion_safety:
            controller._motion_safety(
                0.10,
                np.empty((0, 2), dtype=np.float64),
                pose=Pose2D(0.0, 0.0, math.radians(10.0)),
                tracking=steering_tracking,
            )
        expected_stopping_rates = follower.stopping_angular_velocities(
            steering_tracking,
            max(abs(controller.last_linear), controller.odom_speed),
            controller.odom_angular_velocity,
        )
        self.assertEqual(
            motion_safety.call_args.kwargs["angular_velocities"],
            expected_stopping_rates,
        )
        self.assertIs(
            motion_safety.call_args.kwargs["safety"],
            controller.committed_path.safety,
        )
        self.assertIs(
            motion_safety.call_args.kwargs["route_safety"],
            controller.live_route_safety,
        )

    def test_common_validator_diagnoses_inner_paint_without_blocking_route(self):
        footprint_config = self.config["footprint"]
        footprint = AsymmetricFootprint(
            footprint_config["front"],
            footprint_config["rear"],
            footprint_config["half_width"],
        )
        controller = ZigzagMissionController.__new__(ZigzagMissionController)
        controller.route_from_odom = None
        controller.corridor_checker = self.checker
        controller.paint_checker = self.runtime_paint_checker
        controller.footprint_sample_spacing = footprint_config[
            "validation_sample_spacing"
        ]
        controller.minimum_outer_reserve = self.config["texture"][
            "minimum_outer_reserve"
        ]
        controller.inner_paint_blocking = self.config["validation"][
            "inner_paint_blocking"
        ]
        controller.safety_margins = SafetyMargins(
            line=footprint_config["line_margin"],
            obstacle=footprint_config["obstacle_margin"],
            localization=footprint_config["localization_error"],
            tracking=footprint_config["tracking_error"],
        )
        validator = SweptFootprintValidator(
            footprint,
            translation_step=footprint_config["sweep_translation_step"],
            heading_step=math.radians(
                footprint_config["sweep_heading_step_deg"]
            ),
        )
        validation = validator.validate_path(
            self.path,
            safety=controller._path_safety(),
        )
        self.assertTrue(validation.safe)
        self.assertGreater(validation.minimum_line_clearance, 0.0)
        self.assertGreater(validation.minimum_map_clearance, 0.0)

        outside_outer_edge = transformed_path(self.path, 0.0, 0.20, 0.0)
        outside = validator.validate_path(
            outside_outer_edge,
            safety=controller._path_safety(),
        )
        self.assertFalse(outside.safe)
        self.assertLessEqual(outside.minimum_line_clearance, 0.0)

        controller.inner_paint_blocking = True
        blocking = validator.validate_path(
            self.path,
            safety=controller._path_safety(),
        )
        self.assertFalse(blocking.safe)
        self.assertLess(blocking.minimum_line_clearance, 0.0)

    def test_nonblocking_inner_policy_still_blocks_outer_and_map_range(self):
        controller = ZigzagMissionController.__new__(ZigzagMissionController)
        controller.inner_paint_blocking = False
        controller.minimum_outer_reserve = self.config["texture"][
            "minimum_outer_reserve"
        ]
        footprint_config = self.config["footprint"]
        controller.route_from_odom = None
        controller.corridor_checker = self.checker
        controller.paint_checker = self.runtime_paint_checker
        controller.footprint_sample_spacing = footprint_config[
            "validation_sample_spacing"
        ]
        controller.safety_margins = SafetyMargins(
            line=footprint_config["line_margin"],
            localization=footprint_config["localization_error"],
            tracking=footprint_config["tracking_error"],
        )
        validator = SweptFootprintValidator(
            AsymmetricFootprint(
                footprint_config["front"],
                footprint_config["rear"],
                footprint_config["half_width"],
            ),
            translation_step=footprint_config["sweep_translation_step"],
            heading_step=math.radians(
                footprint_config["sweep_heading_step_deg"]
            ),
        )
        outside_reference_range = CommonPath(
            x=np.asarray([0.80]),
            y=np.asarray([1.75]),
            heading=np.asarray([math.pi]),
            curvature=np.asarray([0.0]),
            speed=np.asarray([0.0]),
        )
        validation = validator.validate_path(
            outside_reference_range,
            safety=controller._path_safety(),
        )
        self.assertFalse(validation.safe)
        self.assertEqual(validation.minimum_line_clearance, math.inf)
        self.assertLess(validation.minimum_map_clearance, 0.0)

    def test_zigzag_publishes_only_common_diagnostics_schema(self):
        controller = ZigzagMissionController.__new__(ZigzagMissionController)
        controller.path_follower = PathFollower(
            TrackingConfig(
                lookahead_distance=0.05,
                maximum_linear_velocity=0.10,
                maximum_angular_velocity=0.80,
                maximum_lateral_acceleration=0.035,
                linear_acceleration=0.04,
                linear_deceleration=0.12,
                angular_acceleration=0.80,
                heading_gain=0.35,
            )
        )
        controller.diagnostics_pub = RecordingPublisher()

        controller._publish_diagnostics()

        self.assertEqual(len(controller.diagnostics_pub.messages), 1)
        self.assertEqual(
            len(controller.diagnostics_pub.messages[0].data),
            13,
        )



if __name__ == "__main__":
    unittest.main()
