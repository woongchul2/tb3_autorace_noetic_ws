#!/usr/bin/env python3

import math
import os
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
from custom_autorace_bringup.path_following import (
    CommonPath,
    PathFollower,
    Pose2D,
    TrackingConfig,
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
        registration = template.get("registration", {})
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
            cruise_velocity=template["cruise_velocity"],
            minimum_velocity=template["minimum_velocity"],
            maximum_angular_velocity=config["control"][
                "maximum_angular_velocity"
            ],
            linear_acceleration=config["control"]["linear_acceleration"],
            linear_deceleration=config["control"]["linear_deceleration"],
            angular_acceleration=config["control"]["angular_acceleration"],
            registration_longitudinal_search=registration.get(
                "longitudinal_search", 0.18
            ),
            registration_lateral_search=registration.get(
                "lateral_search", 0.14
            ),
            registration_coarse_step=registration.get("coarse_step", 0.02),
            registration_fine_window=registration.get("fine_window", 0.012),
            registration_fine_step=registration.get("fine_step", 0.002),
            registration_inlier_distance=registration.get(
                "inlier_distance", 0.025
            ),
            registration_minimum_inliers=registration.get(
                "minimum_inliers", 15
            ),
            registration_maximum_rms=registration.get("maximum_rms", 0.015),
            registration_line_exclusion=registration.get(
                "line_exclusion", 0.025
            ),
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

    def test_registration_fine_window_can_be_smaller_than_coarse_step(self):
        spline, _ = self.production_spline()
        self.assertAlmostEqual(spline.registration_coarse_step, 0.020)
        self.assertAlmostEqual(spline.registration_fine_step, 0.002)
        self.assertAlmostEqual(spline.registration_fine_window, 0.012)

    def test_lidar_registration_refines_amcl_seed_for_odom_latch(self):
        spline, obstacles = self.production_spline()
        spline.prepare(obstacles, self.right_line, self.left_line)
        true_progress = 0.405
        true_lateral = -0.122
        live = obstacles - np.asarray([true_progress, true_lateral])
        visible = (
            (live[:, 0] >= -0.18)
            & (live[:, 0] <= 1.35)
            & (np.abs(live[:, 1]) <= 0.43)
        )
        live = live[visible][::3]
        generator = np.random.RandomState(7)
        live += generator.normal(0.0, 0.0015, size=live.shape)
        outlier_x = np.linspace(-0.10, 1.20, 30)
        live = np.vstack(
            (
                live,
                np.column_stack(
                    (
                        outlier_x,
                        np.full_like(outlier_x, 0.2475 - true_lateral),
                    )
                ),
            )
        )
        alignment = spline.align_pose(
            true_progress + 0.090,
            true_lateral + 0.065,
            live,
        )
        self.assertIsNotNone(alignment)
        self.assertLess(abs(alignment.progress - true_progress), 0.006)
        self.assertLess(abs(alignment.lateral - true_lateral), 0.006)
        self.assertGreaterEqual(alignment.inliers, 15)
        self.assertLess(alignment.rms, 0.006)

    def test_lidar_registration_does_not_match_front_face_to_hidden_back(self):
        spline, obstacles = self.production_spline()
        spline.prepare(obstacles, self.right_line, self.left_line)
        minimum_x, maximum_x, minimum_y, maximum_y = (
            spline._registration_boxes[0]
        )
        self.assertAlmostEqual(maximum_x - minimum_x, 0.10, places=6)

        true_origin = np.asarray([0.30, -0.12])
        visible_front = np.column_stack(
            (
                np.full(25, minimum_x),
                np.linspace(minimum_y + 0.04, maximum_y - 0.04, 25),
            )
        )
        live_points = visible_front - true_origin
        correct_absolute = live_points + true_origin
        correct_distance, _ = spline._rectangle_surface_matches(
            correct_absolute, true_origin
        )
        self.assertLess(float(np.max(correct_distance)), 1e-12)

        wrong_origin = true_origin + np.asarray([0.10, 0.0])
        wrong_absolute = live_points + wrong_origin
        unfiltered_distance, _ = spline._rectangle_surface_matches(
            wrong_absolute
        )
        self.assertLess(float(np.max(unfiltered_distance)), 1e-12)
        visible_distance, _ = spline._rectangle_surface_matches(
            wrong_absolute, wrong_origin
        )
        self.assertGreater(
            float(np.min(visible_distance)),
            spline.registration_inlier_distance,
        )

    def test_lidar_registration_rejects_single_axis_match(self):
        spline, obstacles = self.production_spline()
        spline.prepare(obstacles, self.right_line, self.left_line)
        first_face = obstacles[
            (np.abs(obstacles[:, 0] - 0.47) < 1e-6)
            & (obstacles[:, 1] >= 0.03)
            & (obstacles[:, 1] <= 0.26)
        ]
        live = first_face - np.asarray([0.40, -0.12])
        self.assertIsNone(spline.align_pose(0.48, -0.06, live))

    def test_known_barrier_faces_are_robustly_stabilized(self):
        spline, obstacles = self.production_spline()
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
            0.0225,
            places=12,
        )
        np.testing.assert_allclose(stabilized_absolute[-1], unrelated[0])

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
