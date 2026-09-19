#!/usr/bin/env python3

import math
import os
import time
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import yaml

from custom_autorace_bringup.obstacle_planner import (
    CourseSplinePlanner,
    Footprint,
    RectanglePathChecker,
    rectangle_surface_points,
)
from custom_autorace_bringup.parking_geometry import curvature_matched_quintic
from custom_autorace_bringup.path_following import (
    CommonPath,
    PathFollower,
    Pose2D,
    TrackingConfig,
    normalize_angle,
)


class ObstaclePlannerTest(unittest.TestCase):
    def setUp(self):
        self.footprint = Footprint(
            front=0.067645,
            rear=0.118073,
            half_width=0.0903,
            obstacle_padding=0.014,
            line_margin=0.009,
        )
        self.checker = RectanglePathChecker(self.footprint)
        self.right_line = -0.2475
        self.left_line = 0.2475

    def production_spline(self):
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "config",
            "obstacle_mission_gazebo.yaml",
        )
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)["obstacle"]
        template = config["template"]
        validation = Footprint(
            front=self.footprint.front,
            rear=self.footprint.rear,
            half_width=self.footprint.half_width,
            obstacle_padding=config["footprint"][
                "validation_obstacle_padding"
            ],
            line_margin=config["footprint"]["validation_line_margin"],
        )
        lateral_scale = template.get("lateral_scale")
        if lateral_scale is None:
            lateral_scale = template.get("lateral_scales", [1.0])[0]
        spline = CourseSplinePlanner(
            collision_checker=self.checker,
            validation_footprint=validation,
            knots=template["knots"],
            exit_endpoint=template["exit_curve"]["endpoint"],
            exit_heading=math.radians(
                template["exit_curve"]["heading_deg"]
            ),
            exit_start_tangent_length=template["exit_curve"][
                "start_tangent_length"
            ],
            exit_end_tangent_length=template["exit_curve"][
                "end_tangent_length"
            ],
            start_slope=template["start_slope"],
            end_slope=template["end_slope"],
            lateral_scale=lateral_scale,
            sample_spacing=template["sample_spacing"],
            minimum_nominal_clearance=template[
                "minimum_nominal_clearance"
            ],
            live_validation_distance=template["live_validation_distance"],
            barrier_association_distance=template[
                "barrier_association_distance"
            ],
            entry_connector_minimum_join_distance=config["planner"][
                "entry_connector_minimum_join_distance"
            ],
            entry_connector_maximum_join_distance=config["planner"][
                "entry_connector_maximum_join_distance"
            ],
            entry_connector_join_step=config["planner"][
                "entry_connector_join_step"
            ],
            entry_connector_start_tangent_ratios=config["planner"][
                "entry_connector_start_tangent_ratios"
            ],
            entry_connector_end_tangent_ratios=config["planner"][
                "entry_connector_end_tangent_ratios"
            ],
            entry_connector_symmetric_tangent_ratios=config["planner"][
                "entry_connector_symmetric_tangent_ratios"
            ],
            cruise_velocity=template["cruise_velocity"],
            minimum_velocity=template["minimum_velocity"],
            entry_velocity=template["entry_velocity"],
            exit_velocity=template["exit_velocity"],
            maximum_angular_velocity=config["control"][
                "maximum_angular_velocity"
            ],
            maximum_lateral_acceleration=config["control"][
                "maximum_lateral_acceleration"
            ],
            linear_acceleration=config["control"]["linear_acceleration"],
            linear_deceleration=config["control"]["linear_deceleration"],
            angular_acceleration=config["control"]["angular_acceleration"],
        )
        obstacles = np.vstack(
            [
                rectangle_surface_points(
                    *barrier, template["barrier_surface_spacing"]
                )
                for barrier in template["barriers"]
            ]
        )
        return spline, obstacles

    def course_boundary_points(self):
        texture_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "turtlebot3_simulations",
                "turtlebot3_gazebo",
                "models",
                "turtlebot3_autorace_2020",
                "course",
                "materials",
                "textures",
                "course.png",
            )
        )
        with Image.open(texture_path) as texture:
            rgb = np.asarray(texture.convert("RGB"), dtype=np.int16)
        self.assertEqual(rgb.shape, (496, 520, 3))

        red, green, blue = np.moveaxis(rgb, 2, 0)
        minimum = np.minimum.reduce((red, green, blue))
        maximum = np.maximum.reduce((red, green, blue))
        white = (minimum >= 128) & ((maximum - minimum) <= 4)
        yellow = (
            (red >= 128)
            & (green >= 128)
            & (blue <= 16)
            & (np.abs(red - green) <= 4)
        )

        def row_centers(mask, first_row, last_row, first_column, last_column):
            centers = []
            for row in range(first_row, last_row):
                columns = np.flatnonzero(
                    mask[row, first_column:last_column]
                )
                self.assertGreater(columns.size, 0)
                centers.append(
                    (row, float(np.median(columns + first_column)))
                )
            return centers

        def column_centers(
            mask, first_column, last_column, first_row, last_row
        ):
            centers = []
            for column in range(first_column, last_column):
                rows = np.flatnonzero(mask[first_row:last_row, column])
                self.assertGreater(rows.size, 0)
                centers.append(
                    (float(np.median(rows + first_row)), column)
                )
            return centers

        white_pixels = row_centers(white, 350, 480, 0, 76)
        white_pixels += column_centers(white, 55, 171, 475, 491)
        yellow_pixels = row_centers(yellow, 350, 449, 75, 101)
        yellow_pixels += column_centers(yellow, 95, 171, 440, 456)
        self.assertEqual(len(white_pixels), 246)
        self.assertEqual(len(yellow_pixels), 175)

        def to_course_local(pixel_points):
            pixels = np.asarray(pixel_points, dtype=np.float64)
            return np.column_stack(
                (
                    pixels[:, 0] / 124.0 - 2.0200,
                    pixels[:, 1] / 130.0 - 0.3625,
                )
            )

        return {
            "white": to_course_local(white_pixels),
            "yellow": to_course_local(yellow_pixels),
        }

    def test_rectangle_checker_detects_collision_between_path_poses(self):
        path = CommonPath(
            x=np.asarray([0.0, 0.80]),
            y=np.asarray([0.0, 0.0]),
            heading=np.asarray([0.0, 0.0]),
            curvature=np.asarray([0.0, 0.0]),
        )
        obstacle = np.asarray([[0.40, 0.0]])
        self.assertFalse(
            self.checker.validator.validate_path(
                path,
                safety=self.checker.safety(
                    obstacle,
                    self.right_line,
                    self.left_line,
                    self.footprint,
                ),
            ).safe
        )
        self.assertTrue(
            self.checker.validator.validate_path(
                path,
                safety=self.checker.safety(
                    np.asarray([[0.40, 0.20]]),
                    self.right_line,
                    self.left_line,
                    self.footprint,
                ),
            ).safe
        )

    def test_course_spline_is_fast_smooth_and_rectangularly_safe(self):
        spline, obstacles = self.production_spline()
        self.assertTrue(
            spline.prepare(obstacles, self.right_line, self.left_line)
        )
        path = spline.plan(
            0.0,
            0.0,
            obstacles,
            self.right_line,
            self.left_line,
        )
        self.assertIsNotNone(path)
        self.assertTrue(
            self.checker.validator.validate_path(
                path,
                safety=self.checker.safety(
                    obstacles,
                    self.right_line,
                    self.left_line,
                    self.footprint,
                ),
            ).safe
        )
        self.assertLess(path.length, 2.30)
        self.assertLess(float(np.max(np.abs(path.curvature))), 6.30)
        self.assertLess(path.curvature_variation, 51.0)
        self.assertLess(path.expected_time, 38.0)
        self.assertGreaterEqual(
            min(path.obstacle_clearance, path.line_clearance),
            spline.minimum_nominal_clearance,
        )
        segment = np.hypot(np.diff(path.x), np.diff(path.y))
        interval_time = 2.0 * segment / (
            path.speed[:-1] + path.speed[1:]
        )
        yaw_acceleration = np.abs(
            np.diff(path.speed * path.curvature)
        ) / interval_time
        self.assertLessEqual(float(np.max(yaw_acceleration)), 0.551)

        template = spline._template
        self.assertAlmostEqual(float(template.x[-1]), 1.735, places=6)
        self.assertAlmostEqual(
            float(template.y[-1]), 0.150000000 * 0.995, places=6
        )
        self.assertAlmostEqual(float(template.heading[-1]), 0.5 * math.pi, places=6)
        map_x = 1.6375 - float(template.y[-1])
        map_y = 0.0200 + float(template.x[-1])
        self.assertAlmostEqual(map_x, 1.4883, places=3)
        self.assertAlmostEqual(map_y, 1.7550, places=3)

    def test_live_entry_connector_is_g2_and_keeps_the_early_sweep_safe(self):
        spline, obstacles = self.production_spline()
        self.assertTrue(
            spline.prepare(obstacles, self.right_line, self.left_line)
        )

        # Official-start replay just before the former late handoff.  Values
        # are in the registered course frame; the lane controller was still
        # following the approach at this sample.
        progress = 0.3782
        course_lateral = -0.1186
        lateral_offset = -course_lateral
        suffix = spline.plan(
            progress,
            lateral_offset,
            np.empty((0, 2), dtype=np.float64),
            self.right_line + lateral_offset,
            self.left_line + lateral_offset,
        )
        self.assertIsNotNone(suffix)
        start_heading = math.radians(-24.4)
        start_curvature = 3.17

        path = spline.connect_entry(
            suffix,
            start_heading,
            start_linear_velocity=0.07,
            start_angular_velocity=0.07 * start_curvature,
        )

        self.assertIsNotNone(path)
        self.assertAlmostEqual(float(path.x[0]), 0.0, places=12)
        self.assertAlmostEqual(float(path.y[0]), 0.0, places=12)
        self.assertAlmostEqual(float(path.heading[0]), start_heading, places=12)
        self.assertAlmostEqual(
            float(path.curvature[0]), start_curvature, places=12
        )
        self.assertTrue(self.checker.validator.validate_path(path).safe)
        self.assertGreater(path.line_clearance, 0.0)
        self.assertGreater(path.obstacle_clearance, 0.0)
        self.assertAlmostEqual(path.entry_join_distance, 0.200, places=2)
        self.assertEqual(path.entry_start_tangent_ratio, 0.12)
        self.assertEqual(path.entry_end_tangent_ratio, 0.36)
        self.assertGreaterEqual(float(np.min(path.speed)), 0.035 - 1e-12)
        self.assertLess(path.expected_time, 25.0)
        segment = np.diff(path.station)
        interval_time = 2.0 * segment / (
            path.speed[:-1] + path.speed[1:]
        )
        longitudinal_acceleration = (
            path.speed[1:] ** 2 - path.speed[:-1] ** 2
        ) / (2.0 * segment)
        yaw_acceleration = np.abs(
            np.diff(path.speed * path.curvature)
        ) / interval_time
        self.assertLessEqual(
            float(np.max(longitudinal_acceleration)),
            spline.linear_acceleration + 1e-9,
        )
        self.assertLessEqual(
            float(np.max(-longitudinal_acceleration)),
            spline.linear_deceleration + 1e-9,
        )
        self.assertLessEqual(
            float(np.max(yaw_acceleration)),
            spline.angular_acceleration + 1e-6,
        )

        spline.entry_connector_start_tangent_ratios = tuple(
            reversed(spline.entry_connector_start_tangent_ratios)
        )
        spline.entry_connector_end_tangent_ratios = tuple(
            reversed(spline.entry_connector_end_tangent_ratios)
        )
        spline.entry_connector_symmetric_tangent_ratios = tuple(
            reversed(spline.entry_connector_symmetric_tangent_ratios)
        )
        reordered = spline.connect_entry(
            suffix,
            start_heading,
            start_linear_velocity=0.07,
            start_angular_velocity=0.07 * start_curvature,
        )
        self.assertIsNotNone(reordered)
        self.assertAlmostEqual(
            reordered.expected_time, path.expected_time, places=9
        )
        self.assertEqual(
            reordered.entry_start_tangent_ratio,
            path.entry_start_tangent_ratio,
        )
        self.assertEqual(
            reordered.entry_end_tangent_ratio,
            path.entry_end_tangent_ratio,
        )

    def test_entry_search_keeps_faster_legacy_symmetric_candidate(self):
        spline, obstacles = self.production_spline()
        self.assertTrue(
            spline.prepare(obstacles, self.right_line, self.left_line)
        )
        lateral_offset = 0.12
        suffix = spline.plan(
            0.40,
            lateral_offset,
            np.empty((0, 2), dtype=np.float64),
            self.right_line + lateral_offset,
            self.left_line + lateral_offset,
        )

        path = spline.connect_entry(
            suffix,
            math.radians(-10.0),
            start_linear_velocity=0.07,
            start_angular_velocity=0.0,
        )

        self.assertIsNotNone(path)
        self.assertTrue(self.checker.validator.validate_path(path).safe)
        self.assertAlmostEqual(path.entry_join_distance, 0.180, places=2)
        self.assertEqual(path.entry_start_tangent_ratio, 0.22)
        self.assertEqual(path.entry_end_tangent_ratio, 0.22)
        self.assertLess(path.expected_time, 23.70)
        self.assertGreater(float(np.min(path.speed)), 0.064)

    def test_selected_entry_profile_is_continuous_across_suffix_seam(self):
        spline, obstacles = self.production_spline()
        self.assertTrue(
            spline.prepare(obstacles, self.right_line, self.left_line)
        )
        progress = 0.42018351253905656
        lateral_offset = 0.1219877281243856
        suffix = spline.plan(
            progress,
            lateral_offset,
            np.empty((0, 2), dtype=np.float64),
            self.right_line + lateral_offset,
            self.left_line + lateral_offset,
        )

        path = spline.connect_entry(
            suffix,
            math.radians(-27.044448760103883),
            start_linear_velocity=0.059948444130236575,
            start_angular_velocity=-0.04248550929273729,
        )

        self.assertIsNotNone(path)
        segment = np.diff(path.station)
        interval_time = 2.0 * segment / (
            path.speed[:-1] + path.speed[1:]
        )
        longitudinal_acceleration = (
            path.speed[1:] ** 2 - path.speed[:-1] ** 2
        ) / (2.0 * segment)
        yaw_acceleration = np.abs(
            np.diff(path.speed * path.curvature)
        ) / interval_time
        self.assertLessEqual(
            float(np.max(longitudinal_acceleration)),
            spline.linear_acceleration + 1e-9,
        )
        self.assertLessEqual(
            float(np.max(-longitudinal_acceleration)),
            spline.linear_deceleration + 1e-9,
        )
        self.assertLessEqual(
            float(np.max(yaw_acceleration)),
            spline.angular_acceleration + 1e-6,
        )
        self.assertAlmostEqual(
            path.expected_time,
            float(np.sum(interval_time)),
            places=9,
        )

    def test_infeasible_candidate_profile_does_not_abort_lattice(self):
        spline, obstacles = self.production_spline()
        self.assertTrue(
            spline.prepare(obstacles, self.right_line, self.left_line)
        )
        lateral_offset = 0.12
        suffix = spline.plan(
            0.35,
            lateral_offset,
            np.empty((0, 2), dtype=np.float64),
            self.right_line + lateral_offset,
            self.left_line + lateral_offset,
        )

        path = spline.connect_entry(
            suffix,
            math.radians(-10.0),
            start_linear_velocity=0.07,
            start_angular_velocity=0.105,
        )

        self.assertIsNotNone(path)
        self.assertTrue(self.checker.validator.validate_path(path).safe)

        # The first exact suffix sample is the quintic seam.  Both heading and
        # curvature must come from the same surveyed sample, not from a
        # two-point interpolation that is subsequently discarded.
        seam = None
        for connected_index in range(1, path.size):
            matches = np.flatnonzero(
                np.isclose(
                    suffix.x,
                    path.x[connected_index],
                    rtol=0.0,
                    atol=1e-12,
                )
                & np.isclose(
                    suffix.y,
                    path.y[connected_index],
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            if matches.size:
                seam = connected_index, int(matches[0])
                break
        self.assertIsNotNone(seam)
        connected_index, suffix_index = seam
        self.assertAlmostEqual(
            normalize_angle(
                float(path.heading[connected_index])
                - float(suffix.heading[suffix_index])
            ),
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(path.curvature[connected_index]),
            float(suffix.curvature[suffix_index]),
            places=12,
        )

    def test_late_intruding_replay_has_no_unsafe_connector_fallback(self):
        spline, obstacles = self.production_spline()
        self.assertTrue(
            spline.prepare(obstacles, self.right_line, self.left_line)
        )
        progress = 0.543867
        course_lateral = -0.145573
        lateral_offset = -course_lateral
        suffix = spline.plan(
            progress,
            lateral_offset,
            np.empty((0, 2), dtype=np.float64),
            self.right_line + lateral_offset,
            self.left_line + lateral_offset,
        )
        self.assertIsNotNone(suffix)

        self.assertIsNone(
            spline.connect_entry(
                suffix,
                math.radians(-1.678),
                start_linear_velocity=0.07,
                start_angular_velocity=0.0,
            )
        )

    def test_entry_search_bounds_exact_sweeps_before_final_route_validation(self):
        spline, obstacles = self.production_spline()
        self.assertTrue(
            spline.prepare(obstacles, self.right_line, self.left_line)
        )
        original_validate = self.checker.validator.validate_path
        self.checker.validator.validate_path = mock.Mock(
            wraps=original_validate
        )

        early_offset = 0.1186
        early = spline.plan(
            0.3782,
            early_offset,
            np.empty((0, 2), dtype=np.float64),
            self.right_line + early_offset,
            self.left_line + early_offset,
        )
        self.checker.validator.validate_path.reset_mock()
        self.assertIsNotNone(
            spline.connect_entry(
                early,
                math.radians(-24.4),
                start_linear_velocity=0.07,
                start_angular_velocity=0.07 * 3.17,
            )
        )
        # Candidates are time-ranked before exact sweeps; the fastest candidate
        # passes, so no slower candidate consumes another exact sweep.
        self.assertEqual(self.checker.validator.validate_path.call_count, 1)

        late_offset = 0.145573
        late = spline.plan(
            0.543867,
            late_offset,
            np.empty((0, 2), dtype=np.float64),
            self.right_line + late_offset,
            self.left_line + late_offset,
        )
        self.checker.validator.validate_path.reset_mock()
        self.assertIsNone(
            spline.connect_entry(
                late,
                math.radians(-1.678),
                start_linear_velocity=0.07,
                start_angular_velocity=0.0,
            )
        )
        # All late candidates already fail the cheap full-margin boundary
        # necessity check; none consume an exact obstacle sweep.
        self.assertEqual(self.checker.validator.validate_path.call_count, 0)

    def test_entry_lattice_runs_inside_sensor_period_budget(self):
        spline, obstacles = self.production_spline()
        self.assertTrue(
            spline.prepare(obstacles, self.right_line, self.left_line)
        )
        durations = []
        replays = (
            (0.3782, 0.1186, -24.4, 0.07, 0.07 * 3.17),
            (0.448836, 0.091623, -12.692, 0.09, 0.148696),
        )
        for progress, offset, heading, linear, angular in replays:
            suffix = spline.plan(
                progress,
                offset,
                np.empty((0, 2), dtype=np.float64),
                self.right_line + offset,
                self.left_line + offset,
            )
            for _ in range(8):
                started = time.monotonic()
                path = spline.connect_entry(
                    suffix,
                    math.radians(heading),
                    start_linear_velocity=linear,
                    start_angular_velocity=angular,
                )
                durations.append(time.monotonic() - started)
                self.assertIsNotNone(path)

        self.assertLess(float(np.percentile(durations, 95.0)), 0.20)

    def test_curvature_matched_quintic_matches_both_end_conditions(self):
        start = (0.0, 0.0, math.radians(-8.0))
        end = (0.24, 0.06, math.radians(31.0))
        points, heading, curvature = curvature_matched_quintic(
            start,
            end,
            -0.8,
            2.4,
            0.035,
            0.045,
            0.004,
        )

        np.testing.assert_allclose(points[0], start[:2], atol=1e-12)
        np.testing.assert_allclose(points[-1], end[:2], atol=1e-12)
        self.assertAlmostEqual(float(heading[0]), start[2], places=12)
        self.assertAlmostEqual(float(heading[-1]), end[2], places=12)
        self.assertAlmostEqual(float(curvature[0]), -0.8, places=12)
        self.assertAlmostEqual(float(curvature[-1]), 2.4, places=12)

    def test_template_and_live_path_each_run_one_swept_validation(self):
        spline, obstacles = self.production_spline()
        original_validate = self.checker.validator.validate_path
        self.checker.validator.validate_path = mock.Mock(
            wraps=original_validate
        )

        self.assertTrue(
            spline.prepare(obstacles, self.right_line, self.left_line)
        )
        self.assertEqual(self.checker.validator.validate_path.call_count, 1)

        path = spline.plan(
            0.0,
            0.0,
            obstacles,
            self.right_line,
            self.left_line,
        )
        self.assertIsNotNone(path)
        self.assertEqual(self.checker.validator.validate_path.call_count, 2)

    def test_production_spline_closes_the_loop_with_the_common_follower(self):
        spline, obstacles = self.production_spline()
        self.assertTrue(
            spline.prepare(obstacles, self.right_line, self.left_line)
        )
        path = spline.plan(
            0.0,
            0.0,
            np.empty((0, 2), dtype=np.float64),
            self.right_line,
            self.left_line,
        )
        self.assertIsInstance(path, CommonPath)

        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "config",
            "obstacle_mission_gazebo.yaml",
        )
        with open(config_path, "r") as config_file:
            control = yaml.safe_load(config_file)["obstacle"]["control"]
        tracking_config = TrackingConfig(
            lookahead_distance=control["lookahead_distance"],
            maximum_linear_velocity=spline.cruise_velocity,
            maximum_angular_velocity=control["maximum_angular_velocity"],
            maximum_lateral_acceleration=control[
                "maximum_lateral_acceleration"
            ],
            linear_acceleration=control["linear_acceleration"],
            linear_deceleration=control["linear_deceleration"],
            angular_acceleration=control["angular_acceleration"],
            heading_gain=control["heading_gain"],
            curvature_feedforward_weight=control[
                "curvature_feedforward_weight"
            ],
            lateral_feedback_gain=control["lateral_feedback_gain"],
            search_ahead_distance=control["search_ahead_distance"],
        )
        follower = PathFollower(tracking_config)
        pose = Pose2D(
            float(path.x[0]),
            float(path.y[0]),
            float(path.heading[0]),
        )
        follower.reset(path, pose)
        elapsed = 0.05
        previous_linear = 0.0
        previous_angular = 0.0
        maximum_position_error = 0.0
        maximum_heading_error = 0.0
        maximum_lateral_acceleration = 0.0
        completed = False
        for step in range(800):
            command, tracking = follower.command(pose, elapsed)
            maximum_position_error = max(
                maximum_position_error, tracking.position_error
            )
            maximum_heading_error = max(
                maximum_heading_error, abs(tracking.heading_error)
            )
            maximum_lateral_acceleration = max(
                maximum_lateral_acceleration,
                abs(command.linear_velocity * command.angular_velocity),
            )
            self.assertLessEqual(
                abs(command.linear_velocity - previous_linear),
                control["linear_deceleration"] * elapsed + 1e-12,
            )
            self.assertLessEqual(
                abs(command.angular_velocity - previous_angular),
                control["angular_acceleration"] * elapsed + 1e-12,
            )
            linear = command.linear_velocity
            angular = command.angular_velocity
            if abs(angular) <= 1e-12:
                pose = Pose2D(
                    pose.x + linear * elapsed * math.cos(pose.yaw),
                    pose.y + linear * elapsed * math.sin(pose.yaw),
                    pose.yaw,
                )
            else:
                next_yaw = pose.yaw + angular * elapsed
                radius = linear / angular
                pose = Pose2D(
                    pose.x
                    + radius * (math.sin(next_yaw) - math.sin(pose.yaw)),
                    pose.y
                    - radius * (math.cos(next_yaw) - math.cos(pose.yaw)),
                    math.atan2(math.sin(next_yaw), math.cos(next_yaw)),
                )
            previous_linear = linear
            previous_angular = angular
            if follower.goal_status(pose).complete:
                completed = True
                break

        self.assertTrue(completed)
        self.assertLess((step + 1) * elapsed, 32.0)
        self.assertLess(maximum_position_error, 0.010)
        self.assertLess(maximum_heading_error, math.radians(25.0))
        self.assertLessEqual(
            maximum_lateral_acceleration,
            control["maximum_lateral_acceleration"] + 1e-12,
        )

    def test_course_texture_lines_clear_rectangular_sweep(self):
        spline, obstacles = self.production_spline()
        self.assertTrue(
            spline.prepare(obstacles, self.right_line, self.left_line)
        )
        path = spline._template
        validation = spline.validation_footprint
        boundary_footprint = Footprint(
            front=validation.front,
            rear=validation.rear,
            half_width=validation.half_width,
            obstacle_padding=validation.line_margin,
            line_margin=0.0,
        )

        for color, boundary in self.course_boundary_points().items():
            result = self.checker.validator.validate_path(
                path,
                safety=self.checker.safety(
                    boundary,
                    -math.inf,
                    math.inf,
                    boundary_footprint,
                ),
            )
            self.assertTrue(
                result.safe,
                msg="rectangular sweep crosses the %s course line" % color,
            )
            self.assertGreaterEqual(
                result.minimum_obstacle_clearance,
                spline.minimum_nominal_clearance,
                msg="%s course-line clearance is %.6f m"
                % (color, result.minimum_obstacle_clearance),
            )

    def test_exit_bezier_is_heading_and_curvature_continuous_at_join(self):
        spline, obstacles = self.production_spline()
        self.assertTrue(
            spline.prepare(obstacles, self.right_line, self.left_line)
        )
        path = spline._template
        join_x = float(spline.knots[-1, 0])
        join_y = float(spline.knots[-1, 1] * spline.lateral_scale)
        matches = np.flatnonzero(
            np.isclose(path.x, join_x, rtol=0.0, atol=1e-12)
            & np.isclose(path.y, join_y, rtol=0.0, atol=1e-12)
        )
        self.assertEqual(matches.size, 1)
        join = int(matches[0])

        exit_points = np.column_stack((path.x[join:], path.y[join:]))
        self.assertGreaterEqual(exit_points.shape[0], 6)
        parameter = np.linspace(0.0, 1.0, exit_points.shape[0])
        opposite = 1.0 - parameter
        bernstein = np.column_stack(
            (
                opposite ** 5,
                5.0 * opposite ** 4 * parameter,
                10.0 * opposite ** 3 * parameter ** 2,
                10.0 * opposite ** 2 * parameter ** 3,
                5.0 * opposite * parameter ** 4,
                parameter ** 5,
            )
        )
        controls = np.linalg.lstsq(
            bernstein, exit_points, rcond=None
        )[0]
        self.assertLess(
            float(np.max(np.abs(bernstein.dot(controls) - exit_points))),
            1e-10,
        )

        first = 5.0 * (controls[1] - controls[0])
        second = 20.0 * (
            controls[2] - 2.0 * controls[1] + controls[0]
        )
        derivative_norm = float(np.hypot(first[0], first[1]))
        exit_heading = math.atan2(first[1], first[0])
        exit_curvature = (
            first[0] * second[1] - first[1] * second[0]
        ) / derivative_norm ** 3
        heading_error = math.atan2(
            math.sin(exit_heading - float(path.heading[join])),
            math.cos(exit_heading - float(path.heading[join])),
        )
        self.assertLess(abs(heading_error), 1e-9)
        self.assertLess(
            abs(exit_curvature - float(path.curvature[join])), 1e-8
        )

    def test_known_barrier_faces_are_robustly_stabilized(self):
        spline, obstacles = self.production_spline()
        self.assertAlmostEqual(spline.barrier_association_distance, 0.025)
        self.assertTrue(
            spline.prepare(obstacles, self.right_line, self.left_line)
        )
        progress = 0.40
        lateral = -0.11
        face_x = np.linspace(0.48, 0.56, 17)
        generator = np.random.RandomState(11)
        measured_y = 0.0225 + 0.006 + generator.normal(0.0, 0.004, face_x.size)
        measured_y[3] -= 0.018
        absolute = np.column_stack((face_x, measured_y))
        unrelated = np.asarray([[0.72, -0.01]], dtype=np.float64)
        relative = np.vstack((absolute, unrelated)) - np.asarray(
            [progress, lateral]
        )

        stabilized = spline.stabilize_known_barrier_returns(
            relative, progress, lateral
        )
        stabilized_absolute = stabilized + np.asarray([progress, lateral])

        self.assertLess(
            float(np.ptp(stabilized_absolute[:-1, 1])), 1e-12
        )
        self.assertAlmostEqual(
            float(stabilized_absolute[0, 1]),
            float(np.median(measured_y)),
            places=12,
        )
        np.testing.assert_allclose(stabilized_absolute[-1], unrelated[0])

    def test_stabilized_face_preserves_coherent_normal_displacement(self):
        spline, obstacles = self.production_spline()
        self.assertTrue(
            spline.prepare(obstacles, self.right_line, self.left_line)
        )
        progress = 0.40
        lateral = -0.11
        nominal_y = 0.0225
        measured_y = nominal_y + 0.018
        absolute = np.column_stack(
            (
                np.linspace(0.48, 0.56, 12),
                np.full(12, measured_y),
            )
        )
        relative = absolute - np.asarray([progress, lateral])

        stabilized = spline.stabilize_known_barrier_returns(
            relative, progress, lateral
        )
        stabilized_absolute = stabilized + np.asarray([progress, lateral])

        np.testing.assert_allclose(
            stabilized_absolute[:, 1],
            np.full(12, measured_y),
            atol=1e-12,
        )

    def test_stabilized_face_preserves_geometry_outside_association_gate(self):
        spline, obstacles = self.production_spline()
        self.assertTrue(
            spline.prepare(obstacles, self.right_line, self.left_line)
        )
        progress = 0.40
        lateral = -0.11
        shifted_y = 0.0225 - 0.035
        absolute = np.column_stack(
            (np.linspace(0.48, 0.56, 12), np.full(12, shifted_y))
        )
        relative = absolute - np.asarray([progress, lateral])

        stabilized = spline.stabilize_known_barrier_returns(
            relative, progress, lateral
        )
        stabilized_absolute = stabilized + np.asarray([progress, lateral])

        np.testing.assert_allclose(
            stabilized_absolute[:, 1], np.full(12, shifted_y), atol=1e-12
        )

    def test_live_collision_rejects_course_spline(self):
        spline, obstacles = self.production_spline()
        spline.prepare(obstacles, self.right_line, self.left_line)
        path = spline.plan(
            0.0,
            0.0,
            np.asarray([[0.04, 0.0]], dtype=np.float64),
            self.right_line,
            self.left_line,
        )
        self.assertIsNone(path)

    def test_circle_would_falsely_close_middle_barrier_passage(self):
        circumscribed_radius = math.hypot(
            self.footprint.rear, self.footprint.half_width
        )
        middle_barrier_left_edge = -0.1025 + 0.5 * 0.25
        free_gap = self.left_line - middle_barrier_left_edge
        rectangle_required = (
            2.0 * self.footprint.half_width
            + self.footprint.obstacle_padding
            + self.footprint.line_margin
        )
        circle_required = (
            2.0 * circumscribed_radius
            + self.footprint.obstacle_padding
            + self.footprint.line_margin
        )
        middle_barrier = rectangle_surface_points(
            0.98, -0.1025, 0.10, 0.25, spacing=0.002
        )
        result = self.checker.validator.validate_poses(
            (Pose2D(0.98, 0.1375, 0.0),),
            safety=self.checker.safety(
                middle_barrier,
                self.right_line,
                self.left_line,
                self.footprint,
            ),
        )
        self.assertTrue(result.safe)
        self.assertGreater(
            min(
                result.minimum_obstacle_clearance,
                result.minimum_line_clearance,
            ),
            0.010,
        )
        self.assertLess(rectangle_required, free_gap)
        self.assertGreater(circle_required, free_gap)


if __name__ == "__main__":
    unittest.main()
