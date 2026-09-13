#!/usr/bin/env python3

import math
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml


NODE_DIR = Path(__file__).resolve().parents[1] / "nodes"
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
if str(NODE_DIR) not in sys.path:
    sys.path.insert(0, str(NODE_DIR))

import intersection_mission_controller as controller_module
from intersection_mission_controller import IntersectionMissionController
from std_msgs.msg import Float64MultiArray
from custom_autorace_bringup.path_following import (
    AsymmetricFootprint,
    AxisAlignedBoundsBoundary,
    CommonPath,
    GoalTolerance,
    PathFollower,
    PathSafety,
    Pose2D,
    RasterCellBoundary,
    RigidTransform2D,
    SafetyMargins,
    SpeedProfile,
    SweptFootprintValidator,
    TrackingConfig,
)


class RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class IntersectionControllerTest(unittest.TestCase):
    # Three observations at the former 10 Hz rate and nine observations at the
    # current 30 Hz rate both represent the same confirmation evidence window.
    DIRECTION_EVIDENCE_WINDOW_SECONDS = 0.30

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

    def advance(self, seconds=0.10):
        self.seconds += seconds

    def make_course_map_harness(self):
        with (CONFIG_DIR / "intersection_mission.yaml").open(
            "r", encoding="utf-8"
        ) as stream:
            mission = yaml.safe_load(stream)["mission"]

        controller = IntersectionMissionController.__new__(
            IntersectionMissionController
        )
        controller.map_world_size = float(mission["map_world_size"])
        controller.map_resolution = float(mission["map_resolution"])
        controller.map_boundary_inflation = float(
            mission["map_boundary_inflation"]
        )
        controller.map_texture_package = str(mission["map_texture_package"])
        controller.map_texture_relative_path = str(
            mission["map_texture_relative_path"]
        )
        controller.map_frame = str(mission["map_frame"])
        controller.course_occupied = None
        controller.course_raw_occupied = None
        controller.course_map_pub = RecordingPublisher()

        gazebo_package = (
            CONFIG_DIR.parents[1]
            / "turtlebot3_simulations"
            / "turtlebot3_gazebo"
        )
        with mock.patch.object(
            controller_module.rospkg.RosPack,
            "get_path",
            return_value=str(gazebo_package),
        ):
            controller._publish_course_map()
            raw_occupied = controller.course_raw_occupied.copy()

        return controller, mission, raw_occupied

    @staticmethod
    def selected_entry_path(mission, direction):
        prefix = "map_left_entry" if direction == "LEFT" else "map_right_entry"
        goal_yaw = math.radians(
            float(mission["map_%s_arc_entry_yaw_deg" % direction.lower()])
        )
        start_point = mission["map_entry_start"]
        start_yaw = math.radians(float(mission["map_entry_start_yaw_deg"]))
        goal = mission[prefix + "_goal"]
        chord = math.hypot(
            float(goal[0]) - float(start_point[0]),
            float(goal[1]) - float(start_point[1]),
        )
        return IntersectionMissionController._cubic_path(
            start_point,
            start_yaw,
            goal,
            goal_yaw,
            float(mission["entry_start_tangent_ratio"]) * chord,
            float(mission["entry_end_tangent_ratio"]) * chord,
            int(mission["map_entry_samples"]),
        )

    @classmethod
    def aligned_entry_path(cls, mission, direction, start):
        """Build the production live-pose alignment plus selected branch."""
        entry = mission["map_entry_start"]
        entry_yaw = math.radians(float(mission["map_entry_start_yaw_deg"]))
        alignment_chord = math.hypot(
            float(entry[0]) - float(start[0]),
            float(entry[1]) - float(start[1]),
        )
        alignment_tangent = (
            float(mission["entry_alignment_tangent_ratio"])
            * alignment_chord
        )
        alignment = IntersectionMissionController._cubic_path(
            start[:2],
            float(start[2]),
            entry,
            entry_yaw,
            alignment_tangent,
            alignment_tangent,
            int(mission["entry_alignment_samples"]),
        )
        return alignment[:-1] + cls.selected_entry_path(mission, direction)

    @staticmethod
    def common_exit_path(mission):
        return IntersectionMissionController._bezier_path(
            mission["map_exit_control_points"],
            int(mission["map_exit_samples"]),
        )

    @classmethod
    def selected_exit_path(cls, mission, direction):
        branch = IntersectionMissionController._bezier_path(
            mission["map_%s_exit_control_points" % direction.lower()],
            int(mission["map_exit_branch_samples"]),
        )
        return branch[:-1] + cls.common_exit_path(mission)

    def configure_left_exit_generation(self, controller, mission, pose):
        controller.direction = controller.LEFT
        controller.map_exit_control_points = tuple(
            tuple(float(component) for component in point)
            for point in mission["map_exit_control_points"]
        )
        controller.map_left_exit_control_points = tuple(
            tuple(float(component) for component in point)
            for point in mission["map_left_exit_control_points"]
        )
        controller.map_right_exit_control_points = tuple(
            tuple(float(component) for component in point)
            for point in mission["map_right_exit_control_points"]
        )
        controller.map_exit_branch_samples = int(
            mission["map_exit_branch_samples"]
        )
        controller.exit_adaptive_join_ratio = float(
            mission["exit_adaptive_join_ratio"]
        )
        controller.exit_adaptive_tangent_ratio = float(
            mission["exit_adaptive_tangent_ratio"]
        )
        controller.exit_adaptive_connector_samples = int(
            mission["exit_adaptive_connector_samples"]
        )
        controller.map_exit_samples = int(mission["map_exit_samples"])
        controller.exit_map_snap_max_distance = float(
            mission["exit_map_snap_max_distance"]
        )
        controller.exit_goal_yaw = math.radians(
            float(mission["exit_goal_yaw_deg"])
        )
        controller.localized_map_x = pose[0]
        controller.localized_map_y = pose[1]
        controller.localized_map_yaw = math.radians(pose[2])
        controller.active_path_velocity = float(
            mission["exit_path_linear_velocity"]
        )
        controller._prepare_active_exit_parameters = mock.Mock()
        controller._tracking_pose = lambda: (
            pose[0],
            pose[1],
            math.radians(pose[2]),
        )
        controller._localized_map_pose_is_fresh = lambda: True
        controller._lookup_tracking_from_map = lambda: (0.0, 0.0, 0.0)
        controller._activate_path = mock.Mock()

    def make_transition_harness(self, direction):
        controller = IntersectionMissionController.__new__(
            IntersectionMissionController
        )
        controller.lock = threading.RLock()
        controller.pose_ready = True
        controller.last_odom_time = self.now()
        # This state-machine harness does not synthesize the normal 20 Hz odom
        # callback; freshness has its own focused ownership regression below.
        controller.odom_timeout = 100.0
        controller.manual_stop = False
        controller.shutting_down = False
        controller.state = controller.SEARCH_DIRECTION
        controller.state_started = self.now()
        controller.direction_candidate = controller.NONE
        controller.direction = controller.NONE
        controller.direction_count = 0
        controller.last_direction_confirmation_time = None
        controller.direction_search_timeout = 35.0
        controller.entry_handoff_timeout = 5.0
        controller.entry_takeover_pose_timeout = 0.75
        controller.entry_handoff_lead_distance = 0.200
        controller.yaw = 0.0
        controller.mission_has_control = False

        controller.total_distance = 0.0
        controller.odom_sequence = 10
        controller.path = [(0.0, 0.0), (0.1, 0.0)]
        controller.path_index = 1
        controller.path_started = self.now()
        controller.path_exit_yaw = 0.0
        controller.path_max_commanded_angular = 0.0
        controller.entry_elapsed = math.nan
        controller.entry_goal_error = math.nan
        controller.entry_yaw_error = math.nan
        controller.exit_path_elapsed = math.nan
        controller.exit_path_goal_error = math.nan
        controller.exit_path_yaw_error = math.nan
        controller.mission_started = self.now()

        controller.arc_lane_start_distance = None
        controller.entry_takeover_odom_sequence = -1
        controller.arc_lane_timeout = 10.0
        controller.arc_lane_max_distance = 2.0
        controller.exit_takeover_settle_time = 0.10
        controller.exit_takeover_pose_timeout = 0.75
        controller.exit_takeover_odom_sequence = -1
        controller.exit_takeover_max_distance = 0.06
        controller.map_exit_control_points = ((0.60, -0.75), (0.25, -0.30))
        controller.map_left_exit_control_points = (
            (0.7622, -1.0353),
            (0.6000, -0.7500),
        )
        controller.map_right_exit_control_points = (
            (0.7622, -0.4647),
            (0.6000, -0.7500),
        )
        controller.map_exit_branch_samples = 20
        controller.map_frame = "map"
        controller.exit_path_velocity = 0.10
        controller.map_entry_start = (1.395, -0.750)
        controller.map_entry_start_yaw = math.pi
        controller.localized_map_x = 0.80
        controller.localized_map_y = -0.75
        controller.arc_lane_max_velocity = 0.12
        controller.lane_resume_max_velocity = 0.30
        controller.final_lane_count = 0
        controller.final_lane_confirm_frames = 3
        controller.final_lane_confirmation_max_gap = 0.20
        controller.last_final_lane_confirmation_time = None
        controller.last_boundary_time = self.now()
        controller.boundary_timeout = 0.50
        controller.final_lane_join_velocity = 0.10
        controller.final_lane_verify_start_distance = None
        controller.final_lane_verify_timeout = 5.0
        controller.final_lane_verify_max_distance = 1.0

        controller.cmd_pub = RecordingPublisher()
        controller.lane_speed_limit_pub = RecordingPublisher()
        controller.state_history = []
        controller.handoff_history = []

        def set_state(state):
            controller.state = state
            controller.state_started = self.now()
            controller.state_history.append(state)

        def set_lane_controller(enabled):
            controller.handoff_history.append(enabled)
            controller.mission_has_control = not enabled
            return True

        controller._set_state = set_state
        controller._set_lane_controller = set_lane_controller
        controller._generate_map_entry_path = mock.Mock(return_value=True)
        controller._generate_exit_path = mock.Mock(return_value=True)
        controller._path_goal_reached = mock.Mock(return_value=True)
        controller._tracking_pose = lambda: (
            0.0,
            0.0,
            controller.path_exit_yaw,
        )
        controller._zone_signal_is_fresh = lambda *_args, **_kwargs: True
        controller._localized_map_pose_is_fresh = lambda: True
        controller._final_lane_observation_valid = lambda _now: True

        return controller

    def make_direction_harness(self, state):
        with (CONFIG_DIR / "intersection_mission.yaml").open(
            "r", encoding="utf-8"
        ) as stream:
            mission = yaml.safe_load(stream)["mission"]

        controller = IntersectionMissionController.__new__(
            IntersectionMissionController
        )
        controller.lock = threading.RLock()
        controller.state = state
        controller.state_started = self.now()
        controller.pose_ready = True
        controller.last_odom_time = self.now()
        controller.odom_timeout = 100.0
        controller.manual_stop = False
        controller.shutting_down = False
        controller.mission_has_control = False
        controller.zone_gate_open = False
        controller.direction = controller.direction_candidate = controller.NONE
        controller.direction_count = 0
        controller.last_direction_confirmation_time = None
        controller.direction_confirm_frames = int(
            mission["direction_confirm_frames"]
        )
        controller.direction_confirmation_max_gap = float(
            mission["direction_confirmation_max_gap"]
        )
        controller.direction_search_timeout = 35.0
        controller.direction_acquire_min_confidence = float(
            mission["direction_acquire_min_confidence"]
        )
        controller.direction_tracking_min_confidence = float(
            mission["direction_tracking_min_confidence"]
        )
        controller.direction_confirm_min_confidence = float(
            mission["direction_confirm_min_confidence"]
        )
        controller.direction_acquire_min_roi_area_ratio = float(
            mission["direction_acquire_min_roi_area_ratio"]
        )
        controller.direction_tracking_min_roi_area_ratio = float(
            mission["direction_tracking_min_roi_area_ratio"]
        )
        controller.direction_confirm_min_roi_area_ratio = float(
            mission["direction_confirm_min_roi_area_ratio"]
        )
        controller.forced_direction = controller.NONE
        controller.camera_width = 640
        controller.camera_height = 480
        controller.direction_pub = RecordingPublisher()
        controller._direction_observation_region_ready = mock.Mock(
            return_value=True
        )
        controller._set_lane_controller = mock.Mock()
        controller._fail = mock.Mock()

        def set_state(next_state):
            controller.state = next_state
            controller.state_started = self.now()

        controller._set_state = mock.Mock(side_effect=set_state)
        return controller

    @staticmethod
    def direction_message(direction):
        message = controller_module.TrafficSign()
        message.sign_type = message.DIRECTION
        message.direction = direction
        message.confidence = 0.90
        message.roi.x_offset = 45
        message.roi.y_offset = 20
        message.roi.width = 170
        message.roi.height = 100
        return message

    def direction_frame_period(self, controller):
        return (
            self.DIRECTION_EVIDENCE_WINDOW_SECONDS
            / float(controller.direction_confirm_frames)
        )

    def observe_direction(self, controller, message, count=None):
        observations = (
            controller.direction_confirm_frames if count is None else int(count)
        )
        for _ in range(observations):
            controller.sign_callback(message)
            self.advance(self.direction_frame_period(controller))

    def test_stale_odometry_stop_is_published_only_by_the_cmd_vel_owner(self):
        controller = self.make_transition_harness(
            IntersectionMissionController.LEFT
        )
        controller.odom_timeout = 0.05
        self.advance(0.10)

        controller.mission_has_control = False
        controller.control_callback(None)
        self.assertEqual(controller.cmd_pub.messages, [])

        controller.mission_has_control = True
        controller.control_callback(None)
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        self.assertEqual(controller.cmd_pub.messages[0].linear.x, 0.0)
        self.assertEqual(controller.cmd_pub.messages[0].angular.z, 0.0)

    def test_shutdown_latches_before_any_later_control_tick(self):
        controller = self.make_transition_harness(
            IntersectionMissionController.LEFT
        )
        controller.mission_has_control = True
        controller.cmd_pub.messages.clear()

        controller.shutdown()
        controller.control_callback(None)

        self.assertTrue(controller.shutting_down)
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        self.assertAlmostEqual(controller.cmd_pub.messages[0].linear.x, 0.0)
        self.assertAlmostEqual(controller.cmd_pub.messages[0].angular.z, 0.0)

    def test_handoff_failure_uses_shared_stop_without_cmd_vel_publish(self):
        controller = IntersectionMissionController.__new__(
            IntersectionMissionController
        )
        controller.state = controller.SEARCH_DIRECTION
        controller.mission_has_control = False
        controller.handoff_ambiguous = True
        controller.cmd_pub = RecordingPublisher()
        controller.lane_speed_limit_pub = RecordingPublisher()
        controller.emergency_stop_pub = RecordingPublisher()
        controller.state_pub = RecordingPublisher()
        controller._set_lane_controller = mock.Mock(return_value=False)
        controller._stop_lane_controller = mock.Mock(return_value=True)

        controller._fail("handoff unavailable")

        self.assertEqual(controller.cmd_pub.messages, [])
        controller._stop_lane_controller.assert_called_once_with()
        self.assertEqual(len(controller.emergency_stop_pub.messages), 1)
        self.assertTrue(controller.emergency_stop_pub.messages[0].data)
        self.assertEqual(controller.state, controller.FAILED)

    def test_path_tick_reuses_one_tracking_result_for_goal_safety_and_command(self):
        path = CommonPath(
            x=[0.0, 0.5, 1.0],
            y=[0.0, 0.0, 0.0],
            heading=[0.0, 0.0, 0.0],
            curvature=[0.0, 0.0, 0.0],
            speed=[0.10, 0.10, 0.10],
            frame_id="odom",
        )
        config = TrackingConfig(
            lookahead_distance=0.08,
            maximum_linear_velocity=0.20,
            maximum_angular_velocity=0.8,
            maximum_lateral_acceleration=0.08,
            linear_acceleration=0.5,
            linear_deceleration=0.8,
            angular_acceleration=1.0,
            heading_gain=0.35,
        )
        controller = IntersectionMissionController.__new__(
            IntersectionMissionController
        )
        controller.path = path
        controller.path_follower = PathFollower(config)
        controller.path_follower.reset(path, Pose2D(0.0, 0.0, 0.0))
        controller.path_validator = SweptFootprintValidator(
            AsymmetricFootprint(0.067645, 0.118073, 0.0903)
        )
        controller.path_index = 0
        controller.path_tick_pose = None
        controller.path_tick_tracking = None
        controller.odom_linear_velocity = 0.0
        controller.odom_angular_velocity = 0.0
        controller.safety_reaction_time = 0.10
        controller.path_linear_deceleration = 0.8
        controller.safety_stop_margin = 0.005
        controller.last_path_command_time = None
        controller.path_max_commanded_angular = 0.0
        controller.diagnostics_pub = RecordingPublisher()
        controller._tracking_pose = lambda: (0.0, 0.0, 0.0)

        with mock.patch.object(
            controller.path_follower,
            "calculate_tracking",
            wraps=controller.path_follower.calculate_tracking,
        ) as calculation:
            self.assertFalse(controller._path_goal_reached())
            command = controller._path_command()

        self.assertEqual(calculation.call_count, 1)
        self.assertGreater(command.linear.x, 0.0)

    def test_production_entry_envelope_and_exit_sweeps_are_safe(self):
        controller, mission, _raw_occupied = self.make_course_map_harness()
        footprint_config = mission["footprint"]
        footprint = AsymmetricFootprint(
            float(footprint_config["front"]),
            float(footprint_config["rear"]),
            float(footprint_config["half_width"]),
        )
        safety = PathSafety(
            line_boundaries=(
                RasterCellBoundary(
                    controller.course_boundary_points,
                    cell_size=controller.course_boundary_cell_size,
                ),
            ),
            margins=SafetyMargins(
                line=float(footprint_config["line_margin"]),
                localization=float(footprint_config["localization_error"]),
                tracking=float(footprint_config["tracking_error"]),
            ),
        )
        validator = SweptFootprintValidator(
            footprint,
            translation_step=float(mission["safety"]["sweep_step"]),
            heading_step=math.radians(
                float(mission["safety"]["sweep_angle_step_deg"])
            ),
        )
        production_routes = (
            (
                "ENTRY_LEFT",
                self.selected_entry_path(mission, "LEFT"),
                "map_left_arc_entry_yaw_deg",
            ),
            (
                "ENTRY_RIGHT",
                self.selected_entry_path(mission, "RIGHT"),
                "map_right_arc_entry_yaw_deg",
            ),
            (
                "EXIT_LEFT",
                self.selected_exit_path(mission, "LEFT"),
                "exit_goal_yaw_deg",
            ),
            (
                "EXIT_RIGHT",
                self.selected_exit_path(mission, "RIGHT"),
                "exit_goal_yaw_deg",
            ),
        )
        results = {}
        for label, path, yaw_parameter in production_routes:
            is_entry = label.startswith("ENTRY_")
            executable = controller_module.path_from_xy(
                path,
                "map",
                target_speed=0.10,
                initial_line_overlap_allowance=(
                    float(mission["path_initial_line_overlap_allowance"])
                    if is_entry
                    else 0.0
                ),
                line_egress_distance=(
                    min(
                        float(mission["path_line_egress_distance"]),
                        IntersectionMissionController._polyline_length(path),
                    )
                    if is_entry
                    else 0.0
                ),
                final_heading=math.radians(float(mission[yaw_parameter])),
                safety=safety,
            )
            results[label] = validator.validate_path(executable)

        self.assertTrue(
            all(result.safe for result in results.values()),
            "production sweep blocker: EL=%.4fm ER=%.4fm XL=%.4fm XR=%.4fm"
            % (
                results["ENTRY_LEFT"].minimum_line_clearance,
                results["ENTRY_RIGHT"].minimum_line_clearance,
                results["EXIT_LEFT"].minimum_line_clearance,
                results["EXIT_RIGHT"].minimum_line_clearance,
            ),
        )
        # The physical footprint itself must remain off the paint. The 3 mm
        # localization/tracking uncertainty envelope may consume only the
        # explicit station-dependent production allowance checked above.
        physical_safety = PathSafety(
            line_boundaries=safety.line_boundaries,
            margins=SafetyMargins(),
        )
        for side in ("LEFT", "RIGHT"):
            route = self.selected_entry_path(mission, side)
            physical = validator.validate_path(
                controller_module.path_from_xy(
                    route,
                    "map",
                    target_speed=0.10,
                    final_heading=math.radians(
                        float(mission["map_%s_arc_entry_yaw_deg" % side.lower()])
                    ),
                    safety=physical_safety,
                )
            )
            self.assertTrue(physical.safe)
            self.assertGreater(physical.minimum_line_clearance, 0.0)
        self.assertGreater(
            min(
                results["EXIT_LEFT"].minimum_line_clearance,
                results["EXIT_RIGHT"].minimum_line_clearance,
            ),
            0.003,
        )

    def test_map_plane_takeover_entries_complete_with_exact_safety(self):
        controller, mission, _ = self.make_course_map_harness()
        footprint_config = mission["footprint"]
        safety = PathSafety(
            line_boundaries=(
                RasterCellBoundary(
                    controller.course_boundary_points,
                    cell_size=controller.course_boundary_cell_size,
                ),
            ),
            margins=SafetyMargins(
                line=float(footprint_config["line_margin"]),
                localization=float(footprint_config["localization_error"]),
                tracking=float(footprint_config["tracking_error"]),
            ),
        )
        validator = SweptFootprintValidator(
            AsymmetricFootprint(
                float(footprint_config["front"]),
                float(footprint_config["rear"]),
                float(footprint_config["half_width"]),
            ),
            translation_step=float(mission["safety"]["sweep_step"]),
            heading_step=math.radians(
                float(mission["safety"]["sweep_angle_step_deg"])
            ),
        )
        # Every current handoff starts upstream of the measured entrance
        # centre.  Exercise ideal, run-2 and run-3 callback timings for both
        # selected branches through the same alignment-plus-branch builder.
        starts = {
            "left_camera_visible": (
                "left",
                Pose2D(1.595, -0.750, math.pi),
            ),
            "right_camera_visible": (
                "right",
                Pose2D(1.595, -0.750, math.pi),
            ),
            # Exact map-frame pose used to generate the run-2 RIGHT connector.
            # The former 0.25 end tangent failed its full rectangular sweep at
            # station 0.20116 m with -3.4886 mm line clearance.
            "right_official_run2": (
                "right",
                Pose2D(
                    1.576927153,
                    -0.762103593,
                    math.radians(164.122748),
                ),
            ),
            # Earliest map pose received after the same ownership handoff.
            # Keeping this end of the observed callback-timing envelope safe
            # prevents the generated path from depending on timer ordering.
            "right_official_run2_first_post_handoff": (
                "right",
                Pose2D(
                    1.587471286,
                    -0.765793892,
                    math.radians(161.435489),
                ),
            ),
            "left_official_run2": (
                "left",
                Pose2D(
                    1.576927153,
                    -0.762103593,
                    math.radians(164.122748),
                ),
            ),
            "left_official_run3": (
                "left",
                Pose2D(
                    1.571426921,
                    -0.749866698,
                    math.radians(161.102214),
                ),
            ),
            "left_official_run3_first_post_handoff": (
                "left",
                Pose2D(
                    1.584243874,
                    -0.753903779,
                    math.radians(162.530415),
                ),
            ),
            "right_official_run3": (
                "right",
                Pose2D(
                    1.571426921,
                    -0.749866698,
                    math.radians(161.102214),
                ),
            ),
            "right_official_run3_first_post_handoff": (
                "right",
                Pose2D(
                    1.584243874,
                    -0.753903779,
                    math.radians(162.530415),
                ),
            ),
        }
        step = 0.02
        for label, (side, start) in starts.items():
            with self.subTest(label=label):
                cruise = float(mission[side + "_path_linear_velocity"])
                maximum_angular = float(
                    mission[side + "_path_max_angular_velocity"]
                )
                goal_yaw = math.radians(
                    float(mission["map_%s_arc_entry_yaw_deg" % side])
                )
                aligned_route = self.aligned_entry_path(
                    mission,
                    side.upper(),
                    (start.x, start.y, start.yaw),
                )
                path = controller_module.path_from_xy(
                    aligned_route,
                    "map",
                    speed_profile=SpeedProfile(
                        cruise_velocity=cruise,
                        minimum_velocity=float(mission["path_min_velocity"]),
                        entry_velocity=float(mission["path_entry_velocity"]),
                        exit_velocity=float(mission["path_exit_velocity"]),
                        maximum_angular_velocity=maximum_angular,
                        maximum_lateral_acceleration=float(
                            mission["maximum_lateral_acceleration"]
                        ),
                        linear_acceleration=float(mission["linear_acceleration"]),
                        linear_deceleration=float(mission["linear_deceleration"]),
                        angular_acceleration=float(
                            mission["path_angular_acceleration"]
                        ),
                    ),
                    initial_line_overlap_allowance=float(
                        mission["path_initial_line_overlap_allowance"]
                    ),
                    line_egress_distance=min(
                        float(mission["path_line_egress_distance"]),
                        IntersectionMissionController._polyline_length(
                            aligned_route
                        ),
                    ),
                    final_heading=goal_yaw,
                    goal_tolerance=GoalTolerance(
                        float(mission["path_goal_tolerance"]),
                        math.radians(
                            float(mission["path_goal_heading_tolerance_deg"])
                        ),
                        float(mission["path_goal_crossing_max_distance"]),
                    ),
                    safety=safety,
                )
                full_validation = validator.validate_path(path)
                self.assertTrue(
                    full_validation.safe,
                    "%s entry sweep line clearance %.4fm"
                    % (side, full_validation.minimum_line_clearance),
                )
                self.assertAlmostEqual(
                    float(path.line_overlap_allowance[0]),
                    float(mission["path_initial_line_overlap_allowance"]),
                )
                self.assertTrue(
                    np.all(np.diff(path.line_overlap_allowance) <= 1.0e-12)
                )
                self.assertAlmostEqual(
                    float(path.line_overlap_allowance[-1]), 0.0, places=12
                )
                # The early camera handoff can occur while the approach lane
                # is still finishing a shallow bend opposite to the selected
                # branch.  Bound that measured connector bend by the same
                # topology limits as production instead of requiring every
                # numerical curvature sample to have one sign.
                heading = np.unwrap(np.asarray(path.heading))
                heading_delta = np.diff(heading)
                selected_sign = 1.0 if side == "left" else -1.0
                opposite_turn = float(
                    np.sum(np.maximum(-selected_sign * heading_delta, 0.0))
                )
                self.assertGreater(
                    selected_sign * float(heading[-1] - heading[0]), 0.0
                )
                self.assertLessEqual(
                    opposite_turn,
                    math.radians(float(mission["entry_max_opposite_turn_deg"])),
                )
                self.assertLessEqual(
                    float(np.sum(np.abs(heading_delta))),
                    math.radians(float(mission["entry_max_total_turn_deg"])),
                )
                follower = PathFollower(
                    TrackingConfig(
                        lookahead_distance=float(mission["path_lookahead"]),
                        maximum_linear_velocity=cruise,
                        maximum_angular_velocity=maximum_angular,
                        maximum_lateral_acceleration=float(
                            mission["maximum_lateral_acceleration"]
                        ),
                        linear_acceleration=float(mission["linear_acceleration"]),
                        linear_deceleration=float(mission["linear_deceleration"]),
                        angular_acceleration=float(
                            mission["path_angular_acceleration"]
                        ),
                        heading_gain=float(mission["path_heading_gain"]),
                        curvature_feedforward_weight=float(
                            mission["path_curvature_weight"]
                        ),
                        lateral_feedback_gain=float(
                            mission["path_lateral_feedback_gain"]
                        ),
                        search_ahead_distance=float(
                            mission["path_search_ahead_distance"]
                        ),
                    )
                )
                pose = start
                follower.reset(
                    path,
                    pose,
                    initial_linear=min(cruise, 0.20),
                    initial_angular=0.0,
                )
                tracking = follower.calculate_tracking(pose)
                initial = validator.validate_poses([pose], safety)
                self.assertLess(initial.minimum_line_clearance, 0.0)
                first_safety = validator.motion_safety(
                    path,
                    pose,
                    tracking.path_index,
                    tracking.target_speed,
                    # Cover the complete-stop region even if the incoming
                    # lane controller's 0.20 m/s sample has not decayed yet.
                    0.20,
                    (
                        -0.137039272,
                        follower.last_angular,
                        tracking.angular_velocity,
                    ),
                    float(mission["safety"]["reaction_time"]),
                    float(mission["linear_deceleration"]),
                    float(mission["safety"]["stop_distance_margin"]),
                    tracking=tracking,
                    route_safety=PathSafety(),
                )
                self.assertFalse(first_safety.requires_stop)

                complete = False
                became_strictly_clear = False
                first_command_angular = None
                maximum_error = 0.0
                minimum_net_line_clearance = math.inf
                for _ in range(int(math.ceil(float(mission["path_follow_timeout"]) / step))):
                    tracking = follower.calculate_tracking(pose)
                    maximum_error = max(
                        maximum_error, abs(tracking.cross_track_error)
                    )
                    allowance = float(
                        np.interp(
                            tracking.station,
                            path.station,
                            path.line_overlap_allowance,
                        )
                    )
                    pose_validation = validator.validate_poses(
                        [pose],
                        safety,
                        line_overlap_allowances=[allowance],
                    )
                    self.assertTrue(pose_validation.safe)
                    minimum_net_line_clearance = min(
                        minimum_net_line_clearance,
                        pose_validation.minimum_line_clearance + allowance,
                    )
                    became_strictly_clear |= (
                        pose_validation.minimum_line_clearance > 0.0
                    )
                    if follower.goal_status(pose, tracking=tracking).complete:
                        complete = True
                        break
                    decision = validator.motion_safety(
                        path,
                        pose,
                        tracking.path_index,
                        tracking.target_speed,
                        max(
                            0.20 if first_command_angular is None else 0.0,
                            abs(follower.last_linear),
                        ),
                        (
                            -0.137039272
                            if first_command_angular is None
                            else follower.last_angular,
                            tracking.angular_velocity,
                        ),
                        float(mission["safety"]["reaction_time"]),
                        float(mission["linear_deceleration"]),
                        float(mission["safety"]["stop_distance_margin"]),
                        tracking=tracking,
                        route_safety=PathSafety(),
                    )
                    self.assertFalse(decision.requires_stop)
                    command, _ = follower.command(
                        pose,
                        step,
                        speed_limit=decision.speed_limit,
                        tracking=tracking,
                    )
                    if first_command_angular is None:
                        first_command_angular = command.angular_velocity
                    pose = Pose2D(
                        pose.x
                        + command.linear_velocity * math.cos(pose.yaw) * step,
                        pose.y
                        + command.linear_velocity * math.sin(pose.yaw) * step,
                        controller_module.normalize_angle(
                            pose.yaw + command.angular_velocity * step
                        ),
                    )

                self.assertTrue(complete)
                self.assertTrue(became_strictly_clear)
                self.assertLess(maximum_error, 0.030)
                self.assertGreater(minimum_net_line_clearance, 0.0)

    def test_recorded_aligned_entries_keep_the_selected_turn_topology(self):
        with (CONFIG_DIR / "intersection_mission.yaml").open(
            "r", encoding="utf-8"
        ) as stream:
            mission = yaml.safe_load(stream)["mission"]

        recorded = (
            (
                "RIGHT_ROLLING_RUN2",
                "RIGHT",
                (
                    1.576927153,
                    -0.762103593,
                    math.radians(164.122748),
                ),
            ),
            (
                "RIGHT_ROLLING_RUN2_FIRST_POST_HANDOFF",
                "RIGHT",
                (
                    1.587471286,
                    -0.765793892,
                    math.radians(161.435489),
                ),
            ),
            (
                "LEFT_ROLLING_RUN3",
                "LEFT",
                (
                    1.584243874,
                    -0.753903779,
                    math.radians(162.530415),
                ),
            ),
            (
                "RIGHT_ROLLING_RUN3",
                "RIGHT",
                (
                    1.584243874,
                    -0.753903779,
                    math.radians(162.530415),
                ),
            ),
        )
        opposite_limit = math.radians(
            float(mission["entry_max_opposite_turn_deg"])
        )
        total_limit = math.radians(
            float(mission["entry_max_total_turn_deg"])
        )
        for label, side, start in recorded:
            with self.subTest(label=label):
                route = self.aligned_entry_path(mission, side, start)
                goal_yaw = math.radians(
                    float(mission["map_%s_arc_entry_yaw_deg" % side.lower()])
                )
                path = controller_module.path_from_xy(
                    route, "map", final_heading=goal_yaw
                )
                heading = np.unwrap(np.asarray(path.heading))
                delta = np.diff(heading)
                selected_sign = 1.0 if side == "LEFT" else -1.0
                net_turn = selected_sign * float(heading[-1] - heading[0])
                opposite_turn = float(
                    np.sum(np.maximum(-selected_sign * delta, 0.0))
                )

                self.assertGreater(net_turn, 0.0)
                self.assertLessEqual(opposite_turn, opposite_limit)
                self.assertLessEqual(float(np.sum(np.abs(delta))), total_limit)

    def test_path_follower_command_history_starts_at_handoff_zero(self):
        controller, mission, _ = self.make_course_map_harness()
        cruise = float(mission["right_path_linear_velocity"])
        maximum_angular = float(mission["right_path_max_angular_velocity"])
        angular_acceleration = float(mission["path_angular_acceleration"])
        pose = Pose2D(
            float(mission["map_entry_start"][0]),
            float(mission["map_entry_start"][1]),
            math.radians(float(mission["map_entry_start_yaw_deg"])),
        )
        goal_yaw = math.radians(mission["map_right_arc_entry_yaw_deg"])
        direct_points = self.selected_entry_path(mission, "RIGHT")
        path = controller_module.path_from_xy(
            direct_points,
            "map",
            speed_profile=SpeedProfile(
                cruise_velocity=cruise,
                minimum_velocity=float(mission["path_min_velocity"]),
                entry_velocity=float(mission["path_entry_velocity"]),
                exit_velocity=float(mission["path_exit_velocity"]),
                maximum_angular_velocity=maximum_angular,
                maximum_lateral_acceleration=float(
                    mission["maximum_lateral_acceleration"]
                ),
                linear_acceleration=float(mission["linear_acceleration"]),
                linear_deceleration=float(mission["linear_deceleration"]),
                angular_acceleration=angular_acceleration,
            ),
        )
        config = TrackingConfig(
            lookahead_distance=float(mission["path_lookahead"]),
            maximum_linear_velocity=cruise,
            maximum_angular_velocity=maximum_angular,
            maximum_lateral_acceleration=float(
                mission["maximum_lateral_acceleration"]
            ),
            linear_acceleration=float(mission["linear_acceleration"]),
            linear_deceleration=float(mission["linear_deceleration"]),
            angular_acceleration=angular_acceleration,
            heading_gain=float(mission["path_heading_gain"]),
            curvature_feedforward_weight=float(
                mission["path_curvature_weight"]
            ),
            lateral_feedback_gain=float(
                mission["path_lateral_feedback_gain"]
            ),
            search_ahead_distance=float(
                mission["path_search_ahead_distance"]
            ),
        )
        follower = PathFollower(config)
        follower.reset(path, pose, initial_linear=cruise, initial_angular=0.0)
        tracking = follower.calculate_tracking(pose)
        self.assertLess(float(np.max(path.curvature)), 0.0)
        self.assertLess(tracking.angular_velocity, 0.0)

        # The lane watchdog had already published zero before takeover, while
        # odometry still measured +0.6423 rad/s of physical rotation. Command
        # slew must start from the former; the latter remains for safety only.
        incompatible_lane_rate = 0.6423
        follower.reset(
            path,
            pose,
            initial_linear=0.0,
            initial_angular=0.0,
        )
        command, _ = follower.command(pose, 0.05)
        self.assertLess(command.angular_velocity, 0.0)
        self.assertLessEqual(
            abs(command.angular_velocity),
            angular_acceleration * 0.05 + 1e-12,
        )

        # Reproduce the removed initialization: treating measured rotation as
        # previous command bypasses the intended command-history slew bound.
        legacy = PathFollower(config)
        legacy.reset(
            path,
            pose,
            initial_linear=cruise,
            initial_angular=incompatible_lane_rate,
        )
        legacy_command, _ = legacy.command(pose, 0.05)
        self.assertGreater(
            legacy_command.angular_velocity, command.angular_velocity
        )

    def test_path_activation_starts_after_handoff_zero_barrier(self):
        controller = IntersectionMissionController.__new__(
            IntersectionMissionController
        )
        controller.course_boundary_points = np.asarray([[1.5, 1.5]])
        controller.course_boundary_cell_size = 0.01
        controller.map_world_size = 4.0
        controller.map_frame = controller.odom_frame = "map"
        controller.active_tracking_from_map = None
        controller.path_safety_margins = SafetyMargins()
        safe = SimpleNamespace(
            safe=True,
            minimum_line_clearance=1.0,
            minimum_obstacle_clearance=1.0,
            minimum_map_clearance=1.0,
        )
        controller.path_validator = SimpleNamespace(
            validate_path=mock.Mock(return_value=safe),
            validate_poses=mock.Mock(return_value=safe),
        )
        controller.active_path_min_velocity = 0.04
        controller.active_path_velocity = 0.12
        controller.active_path_entry_velocity = 0.04
        controller.active_path_exit_velocity = 0.04
        controller.active_path_max_angular = 0.90
        controller.maximum_lateral_acceleration = 0.08
        controller.path_linear_acceleration = 0.25
        controller.path_linear_deceleration = 0.50
        controller.path_angular_acceleration = 1.50
        controller.active_path_lookahead = 0.08
        controller.path_heading_gain = 0.35
        controller.path_curvature_weight = 0.25
        controller.path_lateral_feedback_gain = 1.0
        controller.path_search_ahead_distance = 0.30
        controller.active_path_goal_tolerance = 0.03
        controller.active_path_goal_heading_tolerance = math.radians(8.0)
        controller.active_path_goal_crossing_max_distance = 0.15
        controller.path_initial_line_overlap_allowance = 0.06
        controller.path_line_egress_distance = 0.20
        controller.exit_path_initial_line_overlap_allowance = 0.030
        controller.exit_path_line_egress_distance = 0.20
        controller.direction = controller.RIGHT
        controller.entry_max_opposite_turn = math.radians(2.0)
        controller.entry_max_total_turn = math.radians(120.0)
        controller.odom_linear_velocity = 0.12
        controller.odom_angular_velocity = 0.6423
        controller.odom_timeout = 0.35
        controller._tracking_pose = lambda: (0.0, 0.0, math.pi)
        controller._publish_path = mock.Mock()
        controller._publish_path_diagnostics = mock.Mock()

        self.assertFalse(
            controller._activate_path(
                [(0.0, 0.0), (-0.10, 0.0), (-0.10, -0.10)],
                -0.5 * math.pi,
                -0.5 * math.pi,
                "entry",
            )
        )
        clockwise_u_turn = [
            (math.cos(angle), math.sin(angle))
            for angle in np.linspace(0.0, -math.pi, 50)
        ]
        self.assertFalse(
            controller._activate_path(
                clockwise_u_turn,
                0.5 * math.pi,
                0.5 * math.pi,
                "entry",
            )
        )

        follower = mock.Mock()
        follower.path_index = 0
        follower.calculate_tracking.return_value = SimpleNamespace(
            angular_velocity=-0.30
        )
        with mock.patch.object(
            controller_module, "PathFollower", return_value=follower
        ):
            self.assertTrue(
                controller._activate_path(
                    [(0.0, 0.0), (-0.10, 0.0), (-0.10, 0.10)],
                    0.5 * math.pi,
                    0.5 * math.pi,
                    "entry",
                )
            )

        self.assertEqual(follower.reset.call_count, 1)
        final_reset = follower.reset.call_args
        self.assertEqual(final_reset.kwargs["initial_angular"], 0.0)
        self.assertEqual(
            final_reset.kwargs["initial_linear"],
            0.0,
        )

        # Exit takeover has already published mission-owned zero during its
        # settle interval and uses the same zero command history.
        follower.reset_mock()
        follower.path_index = 0
        with mock.patch.object(
            controller_module, "PathFollower", return_value=follower
        ):
            self.assertTrue(
                controller._activate_path(
                    [(0.0, 0.0), (-0.10, 0.0), (-0.10, 0.10)],
                    0.5 * math.pi,
                    0.5 * math.pi,
                    "exit",
                )
            )
        exit_reset = follower.reset.call_args
        self.assertEqual(exit_reset.kwargs["initial_linear"], 0.0)
        self.assertEqual(exit_reset.kwargs["initial_angular"], 0.0)

    def test_exit_handoff_projects_onto_branch_after_first_point(self):
        controller, mission, _ = self.make_course_map_harness()
        controller.direction = controller.RIGHT
        controller.map_right_exit_control_points = tuple(
            tuple(point) for point in mission["map_right_exit_control_points"]
        )
        controller.map_exit_branch_samples = int(
            mission["map_exit_branch_samples"]
        )
        controller.exit_path_velocity = float(
            mission["exit_path_linear_velocity"]
        )
        # Captured after the official-start RIGHT camera arc lost its last
        # lane frame. It is already beyond the branch's first stored point.
        controller.localized_map_x = 0.7071549453
        controller.localized_map_y = -0.5563684687

        first = controller.map_right_exit_control_points[0]
        self.assertGreater(
            math.hypot(
                controller.localized_map_x - first[0],
                controller.localized_map_y - first[1],
            ),
            0.10,
        )
        projection = controller._selected_exit_map_projection()
        self.assertLess(projection.distance, 0.04)
        self.assertGreater(projection.station, 0.09)

    def test_official_left_arc_exit_pose_uses_common_follower_and_safety(self):
        controller, mission, _ = self.make_course_map_harness()
        footprint_config = mission["footprint"]
        safety = PathSafety(
            line_boundaries=(
                RasterCellBoundary(
                    controller.course_boundary_points,
                    cell_size=controller.course_boundary_cell_size,
                ),
            ),
            margins=SafetyMargins(
                line=float(footprint_config["line_margin"]),
                localization=float(footprint_config["localization_error"]),
                tracking=float(footprint_config["tracking_error"]),
            ),
        )
        validator = SweptFootprintValidator(
            AsymmetricFootprint(
                float(footprint_config["front"]),
                float(footprint_config["rear"]),
                float(footprint_config["half_width"]),
            ),
            translation_step=float(mission["safety"]["sweep_step"]),
            heading_step=math.radians(
                float(mission["safety"]["sweep_angle_step_deg"])
            ),
        )
        cruise = float(mission["exit_path_linear_velocity"])
        maximum_angular = float(mission["exit_path_max_angular_velocity"])
        path = controller_module.path_from_xy(
            self.selected_exit_path(mission, "LEFT"),
            "map",
            speed_profile=SpeedProfile(
                cruise_velocity=cruise,
                minimum_velocity=float(mission["exit_path_min_velocity"]),
                entry_velocity=float(mission["exit_path_entry_velocity"]),
                exit_velocity=float(mission["exit_path_exit_velocity"]),
                maximum_angular_velocity=maximum_angular,
                maximum_lateral_acceleration=float(
                    mission["maximum_lateral_acceleration"]
                ),
                linear_acceleration=float(mission["linear_acceleration"]),
                linear_deceleration=float(mission["linear_deceleration"]),
                angular_acceleration=float(
                    mission["path_angular_acceleration"]
                ),
            ),
            initial_line_overlap_allowance=float(
                mission["exit_path_initial_line_overlap_allowance"]
            ),
            line_egress_distance=float(
                mission["exit_path_line_egress_distance"]
            ),
            final_heading=math.radians(float(mission["exit_goal_yaw_deg"])),
            goal_tolerance=GoalTolerance(
                float(mission["exit_path_goal_tolerance"]),
                math.radians(
                    float(mission["exit_path_goal_heading_tolerance_deg"])
                ),
                float(mission["exit_path_goal_crossing_max_distance"]),
            ),
            safety=safety,
        )
        follower = PathFollower(
            TrackingConfig(
                lookahead_distance=float(mission["exit_path_lookahead"]),
                maximum_linear_velocity=cruise,
                maximum_angular_velocity=maximum_angular,
                maximum_lateral_acceleration=float(
                    mission["maximum_lateral_acceleration"]
                ),
                linear_acceleration=float(mission["linear_acceleration"]),
                linear_deceleration=float(mission["linear_deceleration"]),
                angular_acceleration=float(
                    mission["path_angular_acceleration"]
                ),
                heading_gain=float(mission["path_heading_gain"]),
                curvature_feedforward_weight=float(
                    mission["path_curvature_weight"]
                ),
                lateral_feedback_gain=float(
                    mission["path_lateral_feedback_gain"]
                ),
                search_ahead_distance=float(
                    mission["path_search_ahead_distance"]
                ),
            )
        )
        pose = Pose2D(0.749, -1.037, math.radians(110.0))
        follower.reset(path, pose)
        became_strictly_clear = False
        complete = False
        step = 0.05
        for _ in range(300):
            tracking = follower.calculate_tracking(pose)
            allowance = float(
                np.interp(
                    tracking.station,
                    path.station,
                    path.line_overlap_allowance,
                )
            )
            pose_validation = validator.validate_poses(
                [pose], safety, line_overlap_allowances=[allowance]
            )
            self.assertTrue(pose_validation.safe)
            became_strictly_clear |= (
                pose_validation.minimum_line_clearance > 0.0
            )
            decision = validator.motion_safety(
                path,
                pose,
                tracking.path_index,
                tracking.target_speed,
                abs(follower.last_linear),
                (follower.last_angular, tracking.angular_velocity),
                float(mission["safety"]["reaction_time"]),
                float(mission["linear_deceleration"]),
                float(mission["safety"]["stop_distance_margin"]),
                tracking=tracking,
                route_safety=PathSafety(),
            )
            self.assertFalse(decision.requires_stop)
            if follower.goal_status(pose, tracking=tracking).complete:
                complete = True
                break
            command, _ = follower.command(
                pose,
                step,
                speed_limit=decision.speed_limit,
                tracking=tracking,
            )
            pose = Pose2D(
                pose.x + command.linear_velocity * math.cos(pose.yaw) * step,
                pose.y + command.linear_velocity * math.sin(pose.yaw) * step,
                controller_module.normalize_angle(
                    pose.yaw + command.angular_velocity * step
                ),
            )

        self.assertTrue(complete)
        self.assertTrue(became_strictly_clear)

    def test_recorded_right_adaptive_exit_completes_with_exact_swept_safety(self):
        controller, mission, _ = self.make_course_map_harness()
        footprint = mission["footprint"]
        pose = Pose2D(0.752, -0.422, math.radians(-109.7))
        branch = controller._bezier_path(
            mission["map_right_exit_control_points"],
            int(mission["map_exit_branch_samples"]),
        )
        fixed_branch = controller_module.path_from_xy(branch, "map")
        join_index = int(
            round(
                float(mission["exit_adaptive_join_ratio"])
                * (len(branch) - 1)
            )
        )
        join = branch[join_index]
        chord = math.hypot(join[0] - pose.x, join[1] - pose.y)
        tangent = float(mission["exit_adaptive_tangent_ratio"]) * chord
        connector = controller._cubic_path(
            (pose.x, pose.y),
            pose.yaw,
            join,
            float(fixed_branch.heading[join_index]),
            tangent,
            tangent,
            int(mission["exit_adaptive_connector_samples"]),
        )
        shared = controller._bezier_path(
            mission["map_exit_control_points"],
            int(mission["map_exit_samples"]),
        )
        route = connector[:-1] + branch[join_index:-1] + shared
        safety = PathSafety(
            line_boundaries=(
                RasterCellBoundary(
                    controller.course_boundary_points,
                    cell_size=controller.course_boundary_cell_size,
                ),
            ),
            map_boundaries=(
                AxisAlignedBoundsBoundary(-2.0, 2.0, -2.0, 2.0),
            ),
            margins=SafetyMargins(
                line=float(footprint["line_margin"]),
                localization=float(footprint["localization_error"]),
                tracking=float(footprint["tracking_error"]),
            ),
        )
        speed = SpeedProfile(
            cruise_velocity=float(mission["exit_path_linear_velocity"]),
            minimum_velocity=float(mission["exit_path_min_velocity"]),
            entry_velocity=float(mission["exit_path_entry_velocity"]),
            exit_velocity=float(mission["exit_path_exit_velocity"]),
            maximum_angular_velocity=float(
                mission["exit_path_max_angular_velocity"]
            ),
            maximum_lateral_acceleration=float(
                mission["maximum_lateral_acceleration"]
            ),
            linear_acceleration=float(mission["linear_acceleration"]),
            linear_deceleration=float(mission["linear_deceleration"]),
            angular_acceleration=float(
                mission["path_angular_acceleration"]
            ),
        )
        path = controller_module.path_from_xy(
            route,
            "map",
            speed_profile=speed,
            initial_line_overlap_allowance=float(
                mission["exit_path_initial_line_overlap_allowance"]
            ),
            line_egress_distance=float(
                mission["exit_path_line_egress_distance"]
            ),
            final_heading=math.radians(float(mission["exit_goal_yaw_deg"])),
            goal_tolerance=GoalTolerance(
                float(mission["exit_path_goal_tolerance"]),
                math.radians(
                    float(mission["exit_path_goal_heading_tolerance_deg"])
                ),
                float(mission["exit_path_goal_crossing_max_distance"]),
            ),
            safety=safety,
        )
        validator = SweptFootprintValidator(
            AsymmetricFootprint(
                float(footprint["front"]),
                float(footprint["rear"]),
                float(footprint["half_width"]),
            ),
            translation_step=float(mission["safety"]["sweep_step"]),
            heading_step=math.radians(
                float(mission["safety"]["sweep_angle_step_deg"])
            ),
        )
        self.assertTrue(validator.validate_path(path).safe)
        follower = PathFollower(
            TrackingConfig(
                lookahead_distance=float(mission["exit_path_lookahead"]),
                maximum_linear_velocity=float(
                    mission["exit_path_linear_velocity"]
                ),
                maximum_angular_velocity=float(
                    mission["exit_path_max_angular_velocity"]
                ),
                maximum_lateral_acceleration=float(
                    mission["maximum_lateral_acceleration"]
                ),
                linear_acceleration=float(mission["linear_acceleration"]),
                linear_deceleration=float(mission["linear_deceleration"]),
                angular_acceleration=float(
                    mission["path_angular_acceleration"]
                ),
                heading_gain=float(mission["path_heading_gain"]),
                curvature_feedforward_weight=float(
                    mission["path_curvature_weight"]
                ),
                lateral_feedback_gain=float(
                    mission["path_lateral_feedback_gain"]
                ),
                search_ahead_distance=float(
                    mission["path_search_ahead_distance"]
                ),
            )
        )
        follower.reset(path, pose, initial_linear=0.0, initial_angular=0.0)
        maximum_error = 0.0
        minimum_net_line_clearance = math.inf
        step = 0.02
        complete = False
        for index in range(
            int(math.ceil(float(mission["exit_path_follow_timeout"]) / step))
        ):
            tracking = follower.calculate_tracking(pose)
            allowance = float(
                np.interp(
                    tracking.station,
                    path.station,
                    path.line_overlap_allowance,
                )
            )
            pose_validation = validator.validate_poses(
                [pose], safety, line_overlap_allowances=[allowance]
            )
            self.assertTrue(pose_validation.safe)
            maximum_error = max(maximum_error, tracking.position_error)
            minimum_net_line_clearance = min(
                minimum_net_line_clearance,
                pose_validation.minimum_line_clearance + allowance,
            )
            if follower.goal_status(pose, tracking=tracking).complete:
                complete = True
                break
            decision = validator.motion_safety(
                path,
                pose,
                tracking.path_index,
                tracking.target_speed,
                max(0.10 if index == 0 else 0.0, abs(follower.last_linear)),
                (
                    -0.50 if index == 0 else follower.last_angular,
                    tracking.angular_velocity,
                ),
                float(mission["safety"]["reaction_time"]),
                float(mission["linear_deceleration"]),
                float(mission["safety"]["stop_distance_margin"]),
                tracking=tracking,
                route_safety=PathSafety(),
            )
            self.assertFalse(decision.requires_stop)
            command, _ = follower.command(
                pose,
                step,
                speed_limit=decision.speed_limit,
                tracking=tracking,
            )
            pose = Pose2D(
                pose.x + command.linear_velocity * math.cos(pose.yaw) * step,
                pose.y + command.linear_velocity * math.sin(pose.yaw) * step,
                controller_module.normalize_angle(
                    pose.yaw + command.angular_velocity * step
                ),
            )

        self.assertTrue(complete)
        self.assertLess(maximum_error, 0.01)
        self.assertGreater(minimum_net_line_clearance, 0.0)

    def test_latest_official_right_exit_pose_clears_initial_paint_overlap(self):
        """The live connector must egress the run-7 handoff, not fail planning."""
        controller, mission, _ = self.make_course_map_harness()
        footprint = mission["footprint"]
        pose = Pose2D(0.738, -0.433, math.radians(-103.1))
        branch = controller._bezier_path(
            mission["map_right_exit_control_points"],
            int(mission["map_exit_branch_samples"]),
        )
        fixed_branch = controller_module.path_from_xy(branch, "map")
        join_index = int(
            round(
                float(mission["exit_adaptive_join_ratio"])
                * (len(branch) - 1)
            )
        )
        join = branch[join_index]
        chord = math.hypot(join[0] - pose.x, join[1] - pose.y)
        tangent = float(mission["exit_adaptive_tangent_ratio"]) * chord
        connector = controller._cubic_path(
            (pose.x, pose.y),
            pose.yaw,
            join,
            float(fixed_branch.heading[join_index]),
            tangent,
            tangent,
            int(mission["exit_adaptive_connector_samples"]),
        )
        shared = controller._bezier_path(
            mission["map_exit_control_points"],
            int(mission["map_exit_samples"]),
        )
        safety = PathSafety(
            line_boundaries=(
                RasterCellBoundary(
                    controller.course_boundary_points,
                    cell_size=controller.course_boundary_cell_size,
                ),
            ),
            margins=SafetyMargins(
                line=float(footprint["line_margin"]),
                localization=float(footprint["localization_error"]),
                tracking=float(footprint["tracking_error"]),
            ),
        )
        path = controller_module.path_from_xy(
            connector[:-1] + branch[join_index:-1] + shared,
            "map",
            target_speed=float(mission["exit_path_linear_velocity"]),
            initial_line_overlap_allowance=float(
                mission["exit_path_initial_line_overlap_allowance"]
            ),
            line_egress_distance=float(
                mission["exit_path_line_egress_distance"]
            ),
            safety=safety,
        )
        validator = SweptFootprintValidator(
            AsymmetricFootprint(
                float(footprint["front"]),
                float(footprint["rear"]),
                float(footprint["half_width"]),
            ),
            translation_step=float(mission["safety"]["sweep_step"]),
            heading_step=math.radians(
                float(mission["safety"]["sweep_angle_step_deg"])
            ),
        )

        start = validator.validate_poses([pose], safety)
        result = validator.validate_path(path)
        downstream_index = int(np.searchsorted(path.station, 0.20))
        downstream_poses = [
            Pose2D(float(x), float(y), float(heading))
            for x, y, heading in zip(
                path.x[downstream_index:],
                path.y[downstream_index:],
                path.heading[downstream_index:],
            )
        ]
        downstream = validator.validate_poses(downstream_poses, safety)

        self.assertLess(start.minimum_line_clearance, 0.0)
        self.assertGreater(
            start.minimum_line_clearance
            + float(path.line_overlap_allowance[0]),
            0.0,
        )
        self.assertTrue(result.safe)
        self.assertGreaterEqual(path.station[downstream_index], 0.20)
        self.assertEqual(path.line_overlap_allowance[downstream_index], 0.0)
        self.assertTrue(downstream.safe)
        self.assertGreater(downstream.minimum_line_clearance, 0.0)

    def test_camera_choice_builds_aligned_entry_to_selected_fixed_goal(self):
        for direction, label in (
            (IntersectionMissionController.LEFT, "left"),
            (IntersectionMissionController.RIGHT, "right"),
        ):
            with self.subTest(label=label):
                controller, mission, _ = self.make_course_map_harness()
                controller.direction = direction
                controller.map_entry_start = tuple(mission["map_entry_start"])
                controller.map_entry_start_yaw = math.radians(
                    float(mission["map_entry_start_yaw_deg"])
                )
                for side in ("left", "right"):
                    setattr(
                        controller,
                        "map_%s_entry_goal" % side,
                        tuple(mission["map_%s_entry_goal" % side]),
                    )
                    setattr(
                        controller,
                        "map_%s_arc_entry_yaw" % side,
                        math.radians(
                            float(
                                mission[
                                    "map_%s_arc_entry_yaw_deg" % side
                                ]
                            )
                        ),
                    )
                controller.map_entry_samples = int(mission["map_entry_samples"])
                controller.entry_alignment_samples = int(
                    mission["entry_alignment_samples"]
                )
                controller.entry_alignment_tangent_ratio = float(
                    mission["entry_alignment_tangent_ratio"]
                )
                controller.entry_start_tangent_ratio = float(
                    mission["entry_start_tangent_ratio"]
                )
                controller.entry_end_tangent_ratio = float(
                    mission["entry_end_tangent_ratio"]
                )
                controller.map_snap_max_distance = float(
                    mission["map_snap_max_distance"]
                )
                # Current early-handoff envelope, captured in official run 3.
                handoff = (
                    1.584243874,
                    -0.753903779,
                    math.radians(162.530415),
                )
                controller.localized_map_x = handoff[0]
                controller.localized_map_y = handoff[1]
                controller._localized_map_pose_is_fresh = lambda: True
                controller._lookup_tracking_from_map = lambda: (0.0, 0.0, 0.0)
                controller._tracking_pose = lambda: handoff
                controller.active_path_velocity = 0.10
                controller._prepare_active_entry_parameters = mock.Mock()
                controller._activate_path = mock.Mock(return_value=True)

                self.assertTrue(controller._generate_map_entry_path())

                route, goal_yaw, exit_yaw, stage = (
                    controller._activate_path.call_args.args
                )
                self.assertEqual(stage, "entry")
                self.assertAlmostEqual(route[0][0], handoff[0])
                self.assertAlmostEqual(route[0][1], handoff[1])
                self.assertAlmostEqual(goal_yaw, exit_yaw)
                expected = self.aligned_entry_path(
                    mission, label.upper(), handoff
                )
                self.assertEqual(
                    len(route),
                    int(mission["entry_alignment_samples"])
                    + int(mission["map_entry_samples"])
                    - 1,
                )
                np.testing.assert_allclose(route, expected, atol=1.0e-12)
                expected_goal = tuple(mission["map_%s_entry_goal" % label])
                self.assertAlmostEqual(route[-1][0], expected_goal[0])
                self.assertAlmostEqual(route[-1][1], expected_goal[1])
                path = controller_module.path_from_xy(
                    route, "map", final_heading=goal_yaw
                )
                heading = np.unwrap(np.asarray(path.heading))
                heading_delta = np.diff(heading)
                selected_sign = (
                    1.0 if direction == controller.LEFT else -1.0
                )
                self.assertGreater(
                    selected_sign * float(heading[-1] - heading[0]), 0.0
                )
                self.assertLessEqual(
                    float(
                        np.sum(
                            np.maximum(-selected_sign * heading_delta, 0.0)
                        )
                    ),
                    math.radians(float(mission["entry_max_opposite_turn_deg"])),
                )

    def test_entry_generation_rejects_missing_direction(self):
        controller = IntersectionMissionController.__new__(
            IntersectionMissionController
        )
        controller.direction = controller.NONE
        controller._localized_map_pose_is_fresh = mock.Mock(return_value=True)
        controller._activate_path = mock.Mock(return_value=True)

        self.assertFalse(controller._generate_map_entry_path())
        controller._localized_map_pose_is_fresh.assert_not_called()
        controller._activate_path.assert_not_called()

    def test_camera_direction_is_confirmed_without_pre_path_cmd_vel(self):
        controller = self.make_direction_harness(
            IntersectionMissionController.SEARCH_DIRECTION
        )
        message = self.direction_message(
            controller_module.TrafficSign.DIRECTION_RIGHT
        )

        self.observe_direction(controller, message)

        self.assertEqual(controller.direction, controller.RIGHT)
        self.assertEqual(
            controller.direction_count, controller.direction_confirm_frames
        )
        self.assertEqual(controller.state, controller.WAIT_ENTRY_HANDOFF)
        self.assertEqual(len(controller.direction_pub.messages), 1)
        controller._set_state.assert_called_once_with(
            controller.WAIT_ENTRY_HANDOFF
        )
        controller._set_lane_controller.assert_not_called()

    def test_direction_is_preconfirmed_without_starting_gate_timeout_or_handoff(self):
        controller = self.make_direction_harness(
            IntersectionMissionController.WAIT_INTERSECTION
        )
        message = self.direction_message(
            controller_module.TrafficSign.DIRECTION_RIGHT
        )

        self.observe_direction(controller, message)

        self.assertEqual(controller.direction, controller.RIGHT)
        self.assertEqual(controller.state, controller.WAIT_INTERSECTION)
        self.assertEqual(len(controller.direction_pub.messages), 1)
        controller._set_state.assert_not_called()
        controller._set_lane_controller.assert_not_called()

        # Time spent observing before the ordered gate must not consume either
        # the search timeout or the later entry-handoff timeout.
        self.advance(controller.direction_search_timeout + 10.0)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.WAIT_INTERSECTION)
        controller._fail.assert_not_called()

        controller.zone_gate_open = True
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.WAIT_ENTRY_HANDOFF)
        controller._set_state.assert_called_once_with(
            controller.WAIT_ENTRY_HANDOFF
        )
        controller._set_lane_controller.assert_not_called()

    def test_direction_is_ignored_outside_the_upstream_observation_window(self):
        controller = self.make_direction_harness(
            IntersectionMissionController.WAIT_INTERSECTION
        )
        controller._direction_observation_region_ready.return_value = False
        message = self.direction_message(
            controller_module.TrafficSign.DIRECTION_RIGHT
        )

        self.observe_direction(controller, message)

        self.assertEqual(controller.direction, controller.NONE)
        self.assertEqual(controller.direction_count, 0)
        self.assertEqual(controller.state, controller.WAIT_INTERSECTION)
        self.assertEqual(controller.direction_pub.messages, [])

    def test_gate_preserves_a_fresh_partial_preconfirmation_streak(self):
        controller = self.make_direction_harness(
            IntersectionMissionController.WAIT_INTERSECTION
        )
        message = self.direction_message(
            controller_module.TrafficSign.DIRECTION_LEFT
        )

        partial_count = controller.direction_confirm_frames - 1
        self.observe_direction(controller, message, count=partial_count)
        self.assertEqual(controller.direction_count, partial_count)

        controller.zone_gate_open = True
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.SEARCH_DIRECTION)
        self.assertEqual(controller.direction_count, partial_count)

        controller.sign_callback(message)
        self.assertEqual(controller.direction, controller.LEFT)
        self.assertEqual(controller.state, controller.WAIT_ENTRY_HANDOFF)
        self.assertEqual(
            [call.args[0] for call in controller._set_state.call_args_list],
            [controller.SEARCH_DIRECTION, controller.WAIT_ENTRY_HANDOFF],
        )

    def test_failed_run_timing_preserves_spatially_gated_observations(self):
        controller = self.make_direction_harness(
            IntersectionMissionController.WAIT_INTERSECTION
        )

        # Reproduce the useful sequence from the GUI/RViz failure trace: a
        # small high-confidence sign before the ordered gate, followed by two
        # usable frames 1.01 s and 0.796 s later. The gate does not discard the
        # first three observations; the configured 10/30 Hz evidence count then
        # completes normally.
        first = self.direction_message(
            controller_module.TrafficSign.DIRECTION_RIGHT
        )
        first.confidence = 1.00
        first.roi.width = 112
        first.roi.height = 90
        first_area_ratio = float(first.roi.width * first.roi.height) / float(
            controller.camera_width * controller.camera_height
        )
        self.assertLess(first_area_ratio, 0.050)
        self.assertGreaterEqual(
            first_area_ratio, controller.direction_acquire_min_roi_area_ratio
        )
        controller.sign_callback(first)
        self.assertEqual(controller.direction_count, 1)
        self.assertEqual(controller.state, controller.WAIT_INTERSECTION)

        self.advance(0.42)
        controller.zone_gate_open = True
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.SEARCH_DIRECTION)
        self.assertEqual(controller.direction_count, 1)

        self.advance(0.59)
        second = self.direction_message(
            controller_module.TrafficSign.DIRECTION_RIGHT
        )
        second.confidence = 0.61
        second.roi.width = 110
        second.roi.height = 106
        controller.sign_callback(second)
        self.assertEqual(controller.direction_count, 2)

        self.advance(0.796)
        third = self.direction_message(
            controller_module.TrafficSign.DIRECTION_RIGHT
        )
        third.confidence = 0.69
        third.roi.width = 170
        third.roi.height = 154
        controller.sign_callback(third)

        for expected_count in range(4, controller.direction_confirm_frames + 1):
            self.advance(self.direction_frame_period(controller))
            controller.sign_callback(third)
            self.assertEqual(controller.direction_count, expected_count)

        self.assertEqual(controller.direction, controller.RIGHT)
        self.assertEqual(
            controller.direction_count, controller.direction_confirm_frames
        )
        self.assertEqual(controller.state, controller.WAIT_ENTRY_HANDOFF)
        self.assertEqual(len(controller.direction_pub.messages), 1)
        self.assertEqual(
            [call.args[0] for call in controller._set_state.call_args_list],
            [controller.SEARCH_DIRECTION, controller.WAIT_ENTRY_HANDOFF],
        )
        controller._set_lane_controller.assert_not_called()

    def test_direction_streak_resets_after_gap_or_opposite_observation(self):
        controller = self.make_direction_harness(
            IntersectionMissionController.WAIT_INTERSECTION
        )
        right = self.direction_message(
            controller_module.TrafficSign.DIRECTION_RIGHT
        )
        left = self.direction_message(
            controller_module.TrafficSign.DIRECTION_LEFT
        )

        controller.sign_callback(right)
        self.advance(controller.direction_confirmation_max_gap + 0.01)
        controller.sign_callback(right)
        self.assertEqual(controller.direction_candidate, controller.RIGHT)
        self.assertEqual(controller.direction_count, 1)

        self.advance(0.10)
        controller.sign_callback(left)
        self.assertEqual(controller.direction_candidate, controller.LEFT)
        self.assertEqual(controller.direction_count, 1)
        self.assertEqual(controller.direction, controller.NONE)

        for expected_count in range(2, controller.direction_confirm_frames + 1):
            self.advance(self.direction_frame_period(controller))
            controller.sign_callback(left)
            self.assertEqual(controller.direction_count, expected_count)

        self.assertEqual(controller.direction, controller.LEFT)
        self.assertEqual(controller.state, controller.WAIT_INTERSECTION)
        self.assertEqual(len(controller.direction_pub.messages), 1)

    def test_entry_handoff_uses_the_oriented_map_start_plane(self):
        controller = IntersectionMissionController.__new__(
            IntersectionMissionController
        )
        controller.map_entry_start = (1.395, -0.750)
        controller.map_entry_start_yaw = math.pi
        controller.entry_handoff_lead_distance = 0.020
        controller._localized_map_pose_is_fresh = lambda: True
        controller.localized_map_y = -0.690

        controller.localized_map_x = 1.416
        self.assertFalse(controller._entry_handoff_pose_ready())
        controller.localized_map_x = 1.414
        self.assertTrue(controller._entry_handoff_pose_ready())

        controller.map_entry_start = (2.0, 3.0)
        controller.map_entry_start_yaw = 0.5 * math.pi
        controller.localized_map_x = 2.2
        controller.localized_map_y = 2.979
        self.assertFalse(controller._entry_handoff_pose_ready())
        controller.localized_map_y = 2.981
        self.assertTrue(controller._entry_handoff_pose_ready())

        controller._localized_map_pose_is_fresh = lambda: False
        self.assertFalse(controller._entry_handoff_pose_ready())

    def test_production_handoff_precedes_recorded_camera_path_loss(self):
        with (CONFIG_DIR / "intersection_mission.yaml").open(
            "r", encoding="utf-8"
        ) as stream:
            mission = yaml.safe_load(stream)["mission"]

        controller = IntersectionMissionController.__new__(
            IntersectionMissionController
        )
        controller.map_entry_start = tuple(mission["map_entry_start"])
        controller.map_entry_start_yaw = math.radians(
            float(mission["map_entry_start_yaw_deg"])
        )
        controller.entry_handoff_lead_distance = float(
            mission["entry_handoff_lead_distance"]
        )
        controller._localized_map_pose_is_fresh = lambda: True
        # Fresh official-start run1 stopped here after the connected camera
        # path disappeared.  The mission must already own cmd_vel at this
        # position; waiting longer cannot make a stopped vehicle reach the old
        # 20 mm plane.
        controller.localized_map_x = 1.530
        controller.localized_map_y = -0.750

        self.assertAlmostEqual(
            controller._entry_handoff_progress(), -0.135, places=9
        )
        self.assertGreater(controller.entry_handoff_lead_distance, 0.135)
        self.assertTrue(controller._entry_handoff_pose_ready())

    def test_common_follower_completes_all_four_production_routes(self):
        with (CONFIG_DIR / "intersection_mission.yaml").open(
            "r", encoding="utf-8"
        ) as stream:
            mission = yaml.safe_load(stream)["mission"]

        routes = (
            (
                "LEFT",
                self.selected_entry_path(mission, "LEFT"),
                math.radians(float(mission["map_left_arc_entry_yaw_deg"])),
                float(mission["left_path_linear_velocity"]),
                float(mission["path_min_velocity"]),
                float(mission["path_entry_velocity"]),
                float(mission["path_exit_velocity"]),
                float(mission["path_lookahead"]),
                float(mission["path_goal_tolerance"]),
                float(mission["path_follow_timeout"]),
            ),
            (
                "RIGHT",
                self.selected_entry_path(mission, "RIGHT"),
                math.radians(float(mission["map_right_arc_entry_yaw_deg"])),
                float(mission["right_path_linear_velocity"]),
                float(mission["path_min_velocity"]),
                float(mission["path_entry_velocity"]),
                float(mission["path_exit_velocity"]),
                float(mission["path_lookahead"]),
                float(mission["path_goal_tolerance"]),
                float(mission["path_follow_timeout"]),
            ),
            (
                "EXIT_LEFT",
                self.selected_exit_path(mission, "LEFT"),
                math.radians(float(mission["exit_goal_yaw_deg"])),
                float(mission["exit_path_linear_velocity"]),
                float(mission["exit_path_min_velocity"]),
                float(mission["exit_path_entry_velocity"]),
                float(mission["exit_path_exit_velocity"]),
                float(mission["exit_path_lookahead"]),
                float(mission["exit_path_goal_tolerance"]),
                float(mission["exit_path_follow_timeout"]),
            ),
            (
                "EXIT_RIGHT",
                self.selected_exit_path(mission, "RIGHT"),
                math.radians(float(mission["exit_goal_yaw_deg"])),
                float(mission["exit_path_linear_velocity"]),
                float(mission["exit_path_min_velocity"]),
                float(mission["exit_path_entry_velocity"]),
                float(mission["exit_path_exit_velocity"]),
                float(mission["exit_path_lookahead"]),
                float(mission["exit_path_goal_tolerance"]),
                float(mission["exit_path_follow_timeout"]),
            ),
        )
        step = 0.02
        for (
            label,
            points,
            goal_yaw,
            cruise,
            minimum,
            entry,
            exit_velocity,
            lookahead,
            position_tolerance,
            timeout,
        ) in routes:
            with self.subTest(label=label):
                maximum_angular = float(
                    mission[
                        "exit_path_max_angular_velocity"
                        if label.startswith("EXIT")
                        else "path_max_angular_velocity"
                    ]
                )
                profile = SpeedProfile(
                    cruise_velocity=cruise,
                    minimum_velocity=minimum,
                    entry_velocity=entry,
                    exit_velocity=exit_velocity,
                    maximum_angular_velocity=maximum_angular,
                    maximum_lateral_acceleration=float(
                        mission["maximum_lateral_acceleration"]
                    ),
                    linear_acceleration=float(mission["linear_acceleration"]),
                    linear_deceleration=float(mission["linear_deceleration"]),
                    angular_acceleration=float(
                        mission["path_angular_acceleration"]
                    ),
                )
                path = controller_module.path_from_xy(
                    points,
                    "map",
                    speed_profile=profile,
                    final_heading=goal_yaw,
                    goal_tolerance=GoalTolerance(
                        position_tolerance,
                        math.radians(8.0),
                        float(
                            mission[
                                "exit_path_goal_crossing_max_distance"
                                if label.startswith("EXIT")
                                else "path_goal_crossing_max_distance"
                            ]
                        ),
                    ),
                )
                follower = PathFollower(
                    TrackingConfig(
                        lookahead_distance=lookahead,
                        maximum_linear_velocity=cruise,
                        maximum_angular_velocity=maximum_angular,
                        maximum_lateral_acceleration=float(
                            mission["maximum_lateral_acceleration"]
                        ),
                        linear_acceleration=float(
                            mission["linear_acceleration"]
                        ),
                        linear_deceleration=float(
                            mission["linear_deceleration"]
                        ),
                        angular_acceleration=float(
                            mission["path_angular_acceleration"]
                        ),
                        heading_gain=float(mission["path_heading_gain"]),
                        curvature_feedforward_weight=float(
                            mission["path_curvature_weight"]
                        ),
                        lateral_feedback_gain=float(
                            mission["path_lateral_feedback_gain"]
                        ),
                        search_ahead_distance=float(
                            mission["path_search_ahead_distance"]
                        ),
                    )
                )
                pose = Pose2D(
                    float(path.x[0]),
                    float(path.y[0]),
                    float(path.heading[0]),
                )
                follower.reset(path, pose)
                maximum_error = 0.0
                complete = False
                for _ in range(int(math.ceil(timeout / step))):
                    tracking = follower.calculate_tracking(pose)
                    maximum_error = max(
                        maximum_error, abs(tracking.cross_track_error)
                    )
                    if follower.goal_status(pose, tracking=tracking).complete:
                        complete = True
                        break
                    command, _ = follower.command(
                        pose, step, tracking=tracking
                    )
                    pose = Pose2D(
                        pose.x
                        + command.linear_velocity * math.cos(pose.yaw) * step,
                        pose.y
                        + command.linear_velocity * math.sin(pose.yaw) * step,
                        controller_module.normalize_angle(
                            pose.yaw + command.angular_velocity * step
                        ),
                    )

                self.assertTrue(complete, "%s did not complete" % label)
                self.assertLess(maximum_error, 0.02)

    def test_staged_state_and_handoff_order_for_both_directions(self):
        for direction in (
            IntersectionMissionController.LEFT,
            IntersectionMissionController.RIGHT,
        ):
            with self.subTest(direction=direction):
                controller = self.make_transition_harness(direction)

                # Camera confirmation latches the branch, but normal lane
                # following keeps cmd_vel until the AMCL entry plane.
                controller.direction = direction
                controller._set_state(controller.WAIT_ENTRY_HANDOFF)
                controller.localized_map_x = 1.60
                controller.control_callback(None)
                self.assertEqual(
                    controller.state, controller.WAIT_ENTRY_HANDOFF
                )
                self.assertEqual(controller.handoff_history, [])
                controller._generate_map_entry_path.assert_not_called()

                controller.localized_map_x = 1.53
                self.advance()
                controller.control_callback(None)
                self.assertEqual(controller.state, controller.PREPARE_ENTRY_PATH)
                controller._generate_map_entry_path.assert_not_called()
                entry_takeover_stop = controller.cmd_pub.messages[-1]
                self.assertEqual(entry_takeover_stop.linear.x, 0.0)
                self.assertEqual(entry_takeover_stop.angular.z, 0.0)

                # Path creation must use an EKF sample received after the
                # blocking lane-ownership service has returned.
                self.advance(0.05)
                controller.control_callback(None)
                self.assertEqual(controller.state, controller.PREPARE_ENTRY_PATH)
                controller._generate_map_entry_path.assert_not_called()

                controller.odom_sequence += 1
                self.advance(0.05)
                controller.control_callback(None)
                self.assertEqual(controller.state, controller.FOLLOW_ENTRY_PATH)
                controller._generate_map_entry_path.assert_called_once_with()

                publishes_before_arc_handoff = len(controller.cmd_pub.messages)
                self.advance()
                controller.control_callback(None)
                self.assertEqual(controller.state, controller.FOLLOW_ARC_LANE)
                self.assertEqual(
                    len(controller.cmd_pub.messages),
                    publishes_before_arc_handoff,
                )
                self.assertAlmostEqual(
                    controller.lane_speed_limit_pub.messages[-1].data,
                    0.12,
                )
                self.assertEqual(controller.handoff_history, [False, True])

                # The intersection controller remains silent while the lane
                # controller owns the semicircle.
                publishes_before_arc_end = len(controller.cmd_pub.messages)
                self.advance()
                controller.control_callback(None)
                self.assertEqual(controller.state, controller.FOLLOW_ARC_LANE)
                self.assertEqual(
                    len(controller.cmd_pub.messages),
                    publishes_before_arc_end,
                )
                controller.total_distance = (
                    controller.arc_lane_start_distance + 0.65
                )
                selected_start = controller._selected_exit_control_points()[0]
                controller.localized_map_x = selected_start[0]
                controller.localized_map_y = selected_start[1]
                self.advance()
                controller.control_callback(None)
                self.assertEqual(controller.state, controller.PREPARE_EXIT_PATH)
                self.assertEqual(
                    len(controller.cmd_pub.messages),
                    publishes_before_arc_end + 1,
                )
                takeover_stop = controller.cmd_pub.messages[-1]
                self.assertEqual(takeover_stop.linear.x, 0.0)
                self.assertEqual(takeover_stop.angular.z, 0.0)
                controller._generate_exit_path.assert_not_called()

                # Time alone is insufficient: path generation must wait for
                # an EKF odometry sample acquired after lane handoff.
                self.advance(0.05)
                controller.control_callback(None)
                self.assertEqual(controller.state, controller.PREPARE_EXIT_PATH)
                controller._generate_exit_path.assert_not_called()

                controller.odom_sequence += 1
                self.advance(0.06)
                controller.control_callback(None)
                self.assertEqual(controller.state, controller.FOLLOW_EXIT_PATH)
                controller._generate_exit_path.assert_called_once_with()

                publishes_before_final_handoff = len(controller.cmd_pub.messages)
                self.advance()
                controller.path_started = self.now()
                controller.control_callback(None)
                self.assertEqual(controller.state, controller.VERIFY_FINAL_LANE)
                self.assertEqual(
                    len(controller.cmd_pub.messages),
                    publishes_before_final_handoff,
                )
                self.assertAlmostEqual(
                    controller.lane_speed_limit_pub.messages[-1].data,
                    0.10,
                )

                # The rolling path already owns cmd_vel while the mission only
                # observes the final corridor.
                self.advance()
                controller.control_callback(None)
                self.assertEqual(controller.state, controller.VERIFY_FINAL_LANE)
                self.assertEqual(
                    len(controller.cmd_pub.messages),
                    publishes_before_final_handoff,
                )

                # Completion must not request a duplicate lane handoff or emit
                # a command from the observation-only mission state.
                controller.final_lane_count = controller.final_lane_confirm_frames
                controller.last_final_lane_confirmation_time = self.now()
                controller.last_boundary_time = self.now()
                self.advance()
                controller.control_callback(None)
                self.assertEqual(controller.state, controller.COMPLETE)
                self.assertEqual(
                    len(controller.cmd_pub.messages),
                    publishes_before_final_handoff,
                )

                self.assertEqual(
                    controller.handoff_history,
                    [False, True, False, True],
                )
                self.assertEqual(
                    controller.state_history,
                    [
                        controller.WAIT_ENTRY_HANDOFF,
                        controller.PREPARE_ENTRY_PATH,
                        controller.FOLLOW_ENTRY_PATH,
                        controller.FOLLOW_ARC_LANE,
                        controller.PREPARE_EXIT_PATH,
                        controller.FOLLOW_EXIT_PATH,
                        controller.VERIFY_FINAL_LANE,
                        controller.COMPLETE,
                    ],
                )

    def test_final_lane_verify_timeout_reacquires_control_and_publishes_one_stop(self):
        controller = self.make_transition_harness(
            IntersectionMissionController.RIGHT
        )
        controller.state = controller.VERIFY_FINAL_LANE
        controller.state_started = self.now()
        controller.mission_has_control = False
        controller.final_lane_verify_start_distance = controller.total_distance
        controller.final_lane_count = 0
        controller.last_final_lane_confirmation_time = None
        controller.handoff_history.clear()
        controller.cmd_pub.messages.clear()
        controller.lane_speed_limit_pub.messages.clear()

        self.advance(controller.final_lane_verify_timeout + 0.01)
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.FAILED)
        self.assertEqual(controller.handoff_history, [False])
        self.assertTrue(controller.mission_has_control)
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        stop = controller.cmd_pub.messages[0]
        self.assertEqual(stop.linear.x, 0.0)
        self.assertEqual(stop.angular.z, 0.0)
        self.assertEqual(controller.lane_speed_limit_pub.messages[-1].data, 0.0)

    def test_entry_handoff_timeout_fails_without_taking_lane_control(self):
        controller = self.make_transition_harness(
            IntersectionMissionController.LEFT
        )
        controller.direction = controller.LEFT
        controller.state = controller.WAIT_ENTRY_HANDOFF
        controller.state_started = self.now()
        controller.localized_map_x = 1.60
        controller.entry_handoff_timeout = 1.0
        controller._fail = mock.Mock()

        self.advance(1.01)
        controller.control_callback(None)

        controller._fail.assert_called_once()
        self.assertIn(
            "AMCL entry handoff plane timed out",
            controller._fail.call_args.args[0],
        )
        self.assertEqual(controller.handoff_history, [])
        controller._generate_map_entry_path.assert_not_called()

    def test_prepare_entry_times_out_without_post_handoff_odometry(self):
        controller = self.make_transition_harness(
            IntersectionMissionController.LEFT
        )
        controller.direction = controller.LEFT
        controller.state = controller.PREPARE_ENTRY_PATH
        controller.state_started = self.now()
        controller.entry_takeover_odom_sequence = controller.odom_sequence
        controller.entry_takeover_pose_timeout = 0.10
        controller.mission_has_control = True
        controller._fail = mock.Mock()

        self.advance(0.11)
        controller.control_callback(None)

        controller._fail.assert_called_once()
        self.assertIn(
            "no post-handoff EKF pose",
            controller._fail.call_args.args[0],
        )
        controller._generate_map_entry_path.assert_not_called()
        stop = controller.cmd_pub.messages[-1]
        self.assertEqual(stop.linear.x, 0.0)
        self.assertEqual(stop.angular.z, 0.0)

    def test_prepare_exit_retries_transient_tf_then_follows_generated_path(self):
        controller = self.make_transition_harness(
            IntersectionMissionController.LEFT
        )
        controller.state = controller.PREPARE_EXIT_PATH
        controller.state_started = self.now()
        controller.exit_takeover_odom_sequence = controller.odom_sequence
        controller.odom_sequence += 1
        controller._generate_exit_path = mock.Mock(
            side_effect=(None, True)
        )

        self.advance(controller.exit_takeover_settle_time + 0.01)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.PREPARE_EXIT_PATH)
        self.assertEqual(controller._generate_exit_path.call_count, 1)

        self.advance(0.05)
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.FOLLOW_EXIT_PATH)
        self.assertEqual(controller._generate_exit_path.call_count, 2)

    def test_both_directions_feed_one_fixed_common_exit(self):
        with (CONFIG_DIR / "intersection_mission.yaml").open(
            "r", encoding="utf-8"
        ) as stream:
            mission = yaml.safe_load(stream)["mission"]

        route = self.common_exit_path(mission)
        self.assertEqual(len(route), int(mission["map_exit_samples"]))
        for actual, expected in zip(
            route[0], mission["map_exit_control_points"][0]
        ):
            self.assertAlmostEqual(actual, float(expected), places=9)
        for actual, expected in zip(
            route[-1], mission["map_exit_control_points"][-1]
        ):
            self.assertAlmostEqual(actual, float(expected), places=9)
        self.assertAlmostEqual(route[0][1], route[1][1], places=4)
        self.assertAlmostEqual(route[-2][0], route[-1][0], places=4)

    def test_generate_exit_path_freezes_latest_alignment_then_uses_common_activation(self):
        controller, mission, _ = self.make_course_map_harness()
        pose = tuple(mission["map_left_exit_control_points"][0]) + (109.6,)
        self.configure_left_exit_generation(controller, mission, pose)
        frozen = (0.31, -0.22, math.radians(7.0))
        controller._lookup_tracking_from_map = mock.Mock(
            return_value=frozen
        )
        expected_start = controller._map_to_tracking_point(pose[:2], frozen)
        expected_start_yaw = controller._map_to_tracking_yaw(
            math.radians(pose[2]), frozen
        )
        controller._tracking_pose = lambda: (
            expected_start[0],
            expected_start[1],
            expected_start_yaw,
        )

        self.assertTrue(controller._generate_exit_path())
        controller._lookup_tracking_from_map.assert_called_once_with()
        controller._activate_path.assert_called_once()
        route, goal_yaw, exit_yaw, stage = (
            controller._activate_path.call_args.args
        )
        self.assertEqual(stage, "exit")
        expected_yaw = controller._map_to_tracking_yaw(
            math.pi / 2.0, frozen
        )
        self.assertAlmostEqual(goal_yaw, expected_yaw)
        self.assertAlmostEqual(exit_yaw, expected_yaw)
        self.assertEqual(
            len(route),
            int(mission["exit_adaptive_connector_samples"])
            + int(mission["map_exit_branch_samples"])
            + int(mission["map_exit_samples"])
            - int(
                round(
                    float(mission["exit_adaptive_join_ratio"])
                    * (int(mission["map_exit_branch_samples"]) - 1)
                )
            )
            - 2,
        )
        self.assertAlmostEqual(route[0][0], expected_start[0], places=9)
        self.assertAlmostEqual(route[0][1], expected_start[1], places=9)
        self.assertEqual(controller.active_tracking_from_map, frozen)

    def test_map_transform_round_trip_and_ekf_tracking_source(self):
        tracking_from_map = (1.2, -0.4, math.radians(37.0))
        map_point = (0.75, -0.9475)
        tracking_point = IntersectionMissionController._map_to_tracking_point(
            map_point, tracking_from_map
        )
        recovered = RigidTransform2D(
            *tracking_from_map, source_frame="map", target_frame="odom"
        ).inverse().apply_point(tracking_point)
        self.assertAlmostEqual(recovered[0], map_point[0], places=9)
        self.assertAlmostEqual(recovered[1], map_point[1], places=9)
        self.assertAlmostEqual(
            IntersectionMissionController._map_to_tracking_yaw(
                math.radians(90.0), tracking_from_map
            ),
            math.radians(127.0),
            places=9,
        )

        controller = IntersectionMissionController.__new__(
            IntersectionMissionController
        )
        controller.x, controller.y, controller.yaw = 0.3, -0.2, 0.7
        self.assertEqual(controller._tracking_pose(), (0.3, -0.2, 0.7))

    def make_final_lane_harness(self):
        controller = IntersectionMissionController.__new__(
            IntersectionMissionController
        )
        controller.lock = threading.RLock()
        controller.state = controller.VERIFY_FINAL_LANE
        controller.image_center = 500.0
        controller.final_lane_count = 0
        controller.last_final_lane_confirmation_time = None
        controller.final_lane_confirmation_max_gap = 0.20
        controller.boundary_timeout = 0.35
        controller.last_boundary_time = self.now()
        controller.yellow_x = 300.0
        controller.white_x = 700.0
        controller.yellow_valid = True
        controller.white_valid = True
        controller.final_lane_width_min = 120.0
        controller.final_lane_width_max = 900.0
        controller.final_lane_center_tolerance = 50.0
        controller.path_exit_yaw = 0.0
        controller.final_lane_heading_tolerance = math.radians(20.0)
        controller._tracking_pose = lambda: (0.0, 0.0, 0.0)
        return controller

    def test_final_lane_confirmation_requires_both_fresh_boundaries(self):
        controller = self.make_final_lane_harness()

        controller.boundary_callback(
            Float64MultiArray(data=[300.0, 700.0, 1.0, 0.0])
        )
        self.assertEqual(controller.final_lane_count, 0)

        controller.boundary_callback(
            Float64MultiArray(data=[300.0, 700.0, 1.0, 1.0])
        )
        self.assertEqual(controller.final_lane_count, 1)

        for expected in (2, 3):
            self.advance(0.10)
            controller.boundary_callback(
                Float64MultiArray(data=[300.0, 700.0, 1.0, 1.0])
            )
            self.assertEqual(controller.final_lane_count, expected)

        self.advance(controller.boundary_timeout + 0.01)
        self.assertFalse(controller._final_lane_observation_valid(self.now()))

        controller.boundary_callback(
            Float64MultiArray(data=[300.0, 700.0, 1.0, 1.0])
        )
        self.assertEqual(controller.final_lane_count, 1)
        controller.boundary_callback(
            Float64MultiArray(data=[300.0, 700.0, 1.0, 0.0])
        )
        self.assertEqual(controller.final_lane_count, 0)


if __name__ == "__main__":
    unittest.main()
