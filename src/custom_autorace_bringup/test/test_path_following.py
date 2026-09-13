#!/usr/bin/env python3

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import custom_autorace_bringup.path_following as path_following_module
from custom_autorace_bringup.path_following import (
    AsymmetricFootprint,
    AxisAlignedBoundsBoundary,
    CommonPath,
    GoalTolerance,
    PathFollower,
    PathSafety,
    PointCloudBoundary,
    Pose2D,
    RasterCellBoundary,
    RigidTransform2D,
    SafetyMargins,
    SpeedProfile,
    StraightCorridorBoundary,
    SweptFootprintValidator,
    TrackingConfig,
    build_speed_profile,
    freeze_path_in_odom,
    goal_status,
    in_place_rotation_command,
    limit_velocity_command,
    normalize_angle,
    path_from_poses,
    path_from_xy,
    project_to_path,
    sample_in_place_rotation,
    yaw_from_quaternion,
)


class InPlaceRotationTest(unittest.TestCase):
    @staticmethod
    def command(**overrides):
        arguments = dict(
            current_yaw=0.0,
            target_yaw=math.pi / 2.0,
            last_angular_velocity=0.0,
            elapsed=1.0,
            heading_gain=1.0,
            maximum_angular_velocity=0.55,
            minimum_angular_velocity=0.10,
            angular_acceleration=0.55,
            heading_tolerance=math.radians(1.0),
        )
        arguments.update(overrides)
        return in_place_rotation_command(**arguments)

    def test_rotation_sampler_keeps_position_and_includes_both_endpoints(self):
        start = Pose2D(1.25, -0.40, math.radians(10.0))
        target = math.radians(100.0)
        step = math.radians(30.0)

        poses = sample_in_place_rotation(start, target, step)

        self.assertEqual(len(poses), 4)
        self.assertEqual(poses[0], start)
        self.assertAlmostEqual(poses[-1].yaw, target, places=12)
        for pose in poses:
            self.assertAlmostEqual(pose.x, start.x, places=12)
            self.assertAlmostEqual(pose.y, start.y, places=12)
        changes = [
            normalize_angle(second.yaw - first.yaw)
            for first, second in zip(poses[:-1], poses[1:])
        ]
        self.assertTrue(
            all(0.0 < change <= step + 1e-12 for change in changes)
        )
        self.assertAlmostEqual(sum(changes), math.pi / 2.0, places=12)

    def test_rotation_sampler_uses_shortest_signed_sweep_across_wrap(self):
        counter_clockwise = sample_in_place_rotation(
            Pose2D(0.0, 0.0, math.radians(179.0)),
            math.radians(-179.0),
            math.radians(0.75),
        )
        clockwise = sample_in_place_rotation(
            Pose2D(0.0, 0.0, math.radians(-170.0)),
            math.radians(170.0),
            math.radians(7.0),
        )

        ccw_changes = [
            normalize_angle(second.yaw - first.yaw)
            for first, second in zip(
                counter_clockwise[:-1], counter_clockwise[1:]
            )
        ]
        cw_changes = [
            normalize_angle(second.yaw - first.yaw)
            for first, second in zip(clockwise[:-1], clockwise[1:])
        ]
        self.assertTrue(all(change > 0.0 for change in ccw_changes))
        self.assertAlmostEqual(sum(ccw_changes), math.radians(2.0), places=12)
        self.assertTrue(
            all(
                abs(change) <= math.radians(0.75) + 1e-12
                for change in ccw_changes
            )
        )
        self.assertTrue(all(change < 0.0 for change in cw_changes))
        self.assertAlmostEqual(sum(cw_changes), math.radians(-20.0), places=12)
        self.assertTrue(
            all(
                abs(change) <= math.radians(7.0) + 1e-12
                for change in cw_changes
            )
        )

    def test_rotation_sampler_returns_one_normalized_pose_when_aligned(self):
        poses = sample_in_place_rotation(
            Pose2D(0.4, 0.7, 2.0 * math.pi + 0.25),
            0.25,
            math.radians(1.0),
        )

        self.assertEqual(poses, (Pose2D(0.4, 0.7, 0.25),))

    def test_rotation_command_obeys_acceleration_and_maximum_rate(self):
        first = self.command(elapsed=0.10)
        saturated = self.command(elapsed=1.0)

        self.assertEqual(first.linear_velocity, 0.0)
        self.assertAlmostEqual(first.angular_velocity, 0.055, places=12)
        self.assertAlmostEqual(saturated.angular_velocity, 0.55, places=12)

    def test_rotation_command_applies_proportional_minimum_and_braking_caps(self):
        proportional = self.command(target_yaw=0.20)
        nominal_minimum = self.command(target_yaw=0.10, heading_gain=0.20)
        braking = self.command(
            target_yaw=0.002,
            last_angular_velocity=0.04,
            elapsed=0.05,
            heading_gain=100.0,
            heading_tolerance=0.001,
        )

        self.assertAlmostEqual(proportional.angular_velocity, 0.20, places=12)
        self.assertAlmostEqual(nominal_minimum.angular_velocity, 0.10, places=12)
        half_interval_rate = 0.5 * 0.55 * 0.05
        self.assertAlmostEqual(
            braking.angular_velocity,
            math.sqrt(
                half_interval_rate ** 2 + 2.0 * 0.55 * 0.002
            )
            - half_interval_rate,
            places=12,
        )
        self.assertLess(braking.angular_velocity, 0.10)

    def test_rotation_command_is_wrap_safe_and_slews_through_reversal(self):
        wrapped = self.command(
            current_yaw=math.radians(179.0),
            target_yaw=math.radians(-179.0),
        )
        reversing = self.command(
            target_yaw=-math.pi / 2.0,
            last_angular_velocity=0.20,
            elapsed=0.10,
        )

        self.assertGreater(wrapped.angular_velocity, 0.0)
        self.assertAlmostEqual(reversing.angular_velocity, 0.145, places=12)

    def test_rotation_command_slews_to_exact_stop_inside_tolerance(self):
        command = self.command(
            current_yaw=math.radians(89.5),
            last_angular_velocity=0.20,
            elapsed=0.01,
        )

        self.assertEqual(command.linear_velocity, 0.0)
        self.assertAlmostEqual(command.angular_velocity, 0.1945, places=12)

        stopped = self.command(
            current_yaw=math.radians(89.5),
            last_angular_velocity=0.004,
            elapsed=0.01,
        )
        self.assertEqual(stopped.linear_velocity, 0.0)
        self.assertEqual(stopped.angular_velocity, 0.0)

    def test_rotation_command_brakes_early_at_20hz_without_overshoot(self):
        period = 0.05
        acceleration = 0.55
        tolerance = math.radians(0.2)

        for start_yaw, target_yaw in (
            (-0.5 * math.pi, 0.0),
            (-0.5 * math.pi, math.pi),
        ):
            with self.subTest(start_yaw=start_yaw, target_yaw=target_yaw):
                yaw = start_yaw
                previous = 0.0
                errors = []
                for _ in range(120):
                    command = self.command(
                        current_yaw=yaw,
                        target_yaw=target_yaw,
                        last_angular_velocity=previous,
                        elapsed=period,
                        heading_gain=2.0,
                        minimum_angular_velocity=0.0,
                        angular_acceleration=acceleration,
                        heading_tolerance=tolerance,
                    )
                    self.assertLessEqual(
                        abs(command.angular_velocity - previous),
                        acceleration * period + 1e-12,
                    )
                    yaw = normalize_angle(
                        yaw + command.angular_velocity * period
                    )
                    error = normalize_angle(target_yaw - yaw)
                    errors.append(error)
                    previous = command.angular_velocity
                    if abs(previous) <= 1e-12:
                        break

                self.assertLess(len(errors), 120)
                self.assertLessEqual(abs(errors[-1]), tolerance + 1e-12)
                if normalize_angle(target_yaw - start_yaw) > 0.0:
                    self.assertGreaterEqual(min(errors), -tolerance)
                else:
                    self.assertLessEqual(max(errors), tolerance)

    def test_rotation_helpers_reject_invalid_limits(self):
        with self.assertRaisesRegex(ValueError, "heading step"):
            sample_in_place_rotation(Pose2D(0.0, 0.0, 0.0), 1.0, 0.0)
        with self.assertRaisesRegex(ValueError, "speed limits"):
            self.command(minimum_angular_velocity=0.56)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.command(target_yaw=float("nan"))


class CommonPathTest(unittest.TestCase):
    def tracking_config(self):
        return TrackingConfig(
            lookahead_distance=0.20,
            maximum_linear_velocity=0.50,
            maximum_angular_velocity=1.0,
            maximum_lateral_acceleration=0.5,
            linear_acceleration=1.0,
            linear_deceleration=2.0,
            angular_acceleration=3.0,
            heading_gain=0.5,
            curvature_feedforward_weight=1.0,
            lateral_feedback_gain=1.0,
        )

    def test_quaternion_conversion_is_shared_without_ros_dependency(self):
        yaw = math.radians(73.0)
        quaternion = SimpleNamespace(
            x=0.0,
            y=0.0,
            z=math.sin(0.5 * yaw),
            w=math.cos(0.5 * yaw),
        )
        self.assertAlmostEqual(yaw_from_quaternion(quaternion), yaw, places=12)

    @staticmethod
    def straight_path(direction=1, heading=0.0, safety=None):
        return CommonPath(
            x=np.asarray([0.0, 0.5, 1.0]),
            y=np.asarray([0.0, 0.0, 0.0]),
            heading=np.full(3, heading),
            curvature=np.zeros(3),
            station=np.asarray([0.0, 0.5, 1.0]),
            speed=np.asarray([0.2, 0.3, 0.1]),
            direction=direction,
            frame_id="map",
            goal_tolerance=GoalTolerance(0.02, math.radians(5.0), 0.08),
            safety=safety or PathSafety(),
        )

    def test_map_path_is_rigidly_transformed_and_frozen_in_odom(self):
        path = self.straight_path()
        aligned, odom_to_map = freeze_path_in_odom(
            path,
            source_pose=(2.0, 3.0, math.pi / 2.0),
            odom_pose=(10.0, -1.0, math.pi),
            odom_frame="odom",
        )
        transform = RigidTransform2D.from_pose_pair(
            (2.0, 3.0, math.pi / 2.0),
            (10.0, -1.0, math.pi),
        )
        expected = np.asarray([transform.apply_point(point) for point in zip(path.x, path.y)])
        np.testing.assert_allclose(aligned.x, expected[:, 0], atol=1e-12)
        np.testing.assert_allclose(aligned.y, expected[:, 1], atol=1e-12)
        np.testing.assert_allclose(aligned.station, path.station, atol=0.0)
        np.testing.assert_allclose(aligned.curvature, path.curvature, atol=0.0)
        self.assertEqual(aligned.frame_id, "odom")
        round_trip = odom_to_map.apply_pose(transform.apply_pose((0.4, 0.1, 0.2)))
        np.testing.assert_allclose(
            (round_trip.x, round_trip.y, round_trip.yaw),
            (0.4, 0.1, 0.2),
            atol=1e-12,
        )

    def test_line_egress_envelope_is_common_path_data_and_survives_freeze(self):
        path = path_from_xy(
            [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]],
            "map",
            target_speed=0.1,
            initial_line_overlap_allowance=0.03,
            line_egress_distance=0.15,
        )
        self.assertAlmostEqual(path.line_overlap_allowance[0], 0.03)
        self.assertAlmostEqual(path.line_overlap_allowance[1], 0.01)
        self.assertAlmostEqual(path.line_overlap_allowance[-1], 0.0)

        aligned, _ = freeze_path_in_odom(
            path,
            source_pose=(0.0, 0.0, 0.0),
            odom_pose=(1.0, -2.0, 0.3),
        )
        np.testing.assert_allclose(
            aligned.line_overlap_allowance,
            path.line_overlap_allowance,
            atol=0.0,
        )

    def test_line_egress_envelope_rejects_incomplete_configuration(self):
        with self.assertRaises(ValueError):
            path_from_xy(
                [[0.0, 0.0], [0.2, 0.0]],
                "map",
                target_speed=0.1,
                initial_line_overlap_allowance=0.01,
            )

    def test_projection_uses_segment_not_only_discrete_points(self):
        path = self.straight_path()
        projection = project_to_path(path, 0.25, 0.06)
        self.assertAlmostEqual(projection.x, 0.25, places=12)
        self.assertAlmostEqual(projection.y, 0.0, places=12)
        self.assertAlmostEqual(projection.station, 0.25, places=12)
        self.assertAlmostEqual(projection.distance, 0.06, places=12)
        self.assertAlmostEqual(projection.cross_track_error, 0.06, places=12)

    def test_projection_clamps_every_field_to_monotonic_execution_station(self):
        path = CommonPath(
            x=np.asarray([0.0, 1.0, 2.0, 3.0]),
            y=np.zeros(4),
            heading=np.asarray([0.0, 0.1, 0.2, 0.3]),
            curvature=np.asarray([0.0, 1.0, 2.0, 3.0]),
            speed=np.full(4, 0.2),
            direction=1,
        )
        projection = project_to_path(
            path,
            0.2,
            0.1,
            previous_index=2,
            minimum_station=1.8,
        )

        self.assertEqual(projection.segment_index, 1)
        self.assertEqual(projection.path_index, 2)
        self.assertAlmostEqual(projection.station, 1.8, places=12)
        self.assertAlmostEqual(projection.x, 1.8, places=12)
        self.assertAlmostEqual(projection.y, 0.0, places=12)
        self.assertAlmostEqual(projection.heading, 0.18, places=12)
        self.assertAlmostEqual(projection.curvature, 1.8, places=12)
        self.assertEqual(projection.direction, 1)
        self.assertAlmostEqual(
            projection.distance, math.hypot(1.6, 0.1), places=12
        )

        follower = PathFollower(self.tracking_config())
        initial = follower.reset(path, Pose2D(1.8, 0.0, 0.18))
        regressed_pose = Pose2D(1.2, 0.0, 0.12)
        tracking = follower.calculate_tracking(regressed_pose)
        self.assertGreaterEqual(tracking.station, initial.station)
        self.assertGreaterEqual(tracking.path_index, initial.path_index)
        follower.command(regressed_pose, 0.05, tracking=tracking)
        self.assertAlmostEqual(follower.path_station, 1.8, places=12)
        self.assertAlmostEqual(follower.diagnostics.progress, 0.6, places=12)

    def test_projection_supports_a_single_stationary_path_point(self):
        path = CommonPath(
            x=np.asarray([0.4]),
            y=np.asarray([-0.2]),
            heading=np.asarray([math.pi / 2.0]),
            curvature=np.asarray([0.0]),
            speed=np.asarray([0.0]),
            direction=1,
        )
        projection = project_to_path(path, 0.5, -0.2)
        self.assertEqual(projection.path_index, 0)
        self.assertAlmostEqual(projection.distance, 0.1, places=12)
        self.assertAlmostEqual(projection.cross_track_error, -0.1, places=12)

    def test_same_follower_supports_forward_and_reverse_paths(self):
        config = self.tracking_config()
        forward = PathFollower(config)
        forward.reset(self.straight_path(), Pose2D(0.0, 0.0, 0.0))
        forward_command, _ = forward.command(Pose2D(0.0, 0.0, 0.0), 0.10)
        self.assertGreater(forward_command.linear_velocity, 0.0)

        reverse = PathFollower(config)
        reverse.reset(
            self.straight_path(direction=-1, heading=math.pi),
            Pose2D(0.0, 0.0, math.pi),
        )
        reverse_command, tracking = reverse.command(
            Pose2D(0.0, 0.0, math.pi), 0.10
        )
        self.assertLess(reverse_command.linear_velocity, 0.0)
        self.assertEqual(tracking.direction, -1)
        self.assertAlmostEqual(tracking.heading_error, 0.0, places=12)

        generated_reverse = path_from_xy(
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            "odom",
            target_speed=0.2,
            direction=-1,
            final_heading=math.pi,
        )
        np.testing.assert_allclose(
            np.abs(generated_reverse.heading), np.full(3, math.pi), atol=1e-12
        )

    def test_curved_reverse_keeps_signed_linear_and_station_curvature_yaw(self):
        radius = 0.50
        angles = np.linspace(0.0, 0.40, 9)
        points = np.column_stack(
            (radius * np.sin(angles), radius * (1.0 - np.cos(angles)))
        )
        path = path_from_xy(
            points,
            "odom",
            target_speed=0.20,
            direction=-1,
        )
        follower = PathFollower(self.tracking_config())
        start = Pose2D(path.x[0], path.y[0], path.heading[0])
        follower.reset(path, start)

        command, tracking = follower.command(start, 0.10)

        self.assertEqual(tracking.direction, -1)
        self.assertLess(command.linear_velocity, 0.0)
        self.assertGreater(tracking.curvature_command, 0.0)
        self.assertGreater(command.angular_velocity, 0.0)

    def test_pose_path_constructor_preserves_reverse_body_heading(self):
        profile = SpeedProfile(
            cruise_velocity=0.20,
            minimum_velocity=0.05,
            entry_velocity=0.05,
            exit_velocity=0.05,
            maximum_angular_velocity=1.0,
            maximum_lateral_acceleration=0.5,
            linear_acceleration=1.0,
            linear_deceleration=1.0,
            angular_acceleration=3.0,
        )
        poses = np.asarray(
            [
                [0.0, 0.0, math.pi],
                [0.5, 0.0, math.pi],
                [1.0, 0.0, math.pi],
            ]
        )
        path = path_from_poses(
            poses,
            "odom",
            speed_profile=profile,
            direction=-1,
            initial_line_overlap_allowance=0.02,
            line_egress_distance=0.5,
            label="reverse_pose_path",
        )

        np.testing.assert_allclose(path.station, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(path.heading, [math.pi, math.pi, math.pi])
        np.testing.assert_allclose(path.curvature, 0.0, atol=1e-12)
        np.testing.assert_array_equal(path.direction, [-1, -1, -1])
        np.testing.assert_allclose(
            path.line_overlap_allowance, [0.02, 0.0, 0.0]
        )
        follower = PathFollower(self.tracking_config())
        follower.reset(path, Pose2D(0.0, 0.0, math.pi))
        command, tracking = follower.command(
            Pose2D(0.0, 0.0, math.pi), 0.10
        )
        self.assertLess(command.linear_velocity, 0.0)
        self.assertEqual(tracking.direction, -1)

    def test_reset_clears_monotonic_station_across_direction_change(self):
        follower = PathFollower(self.tracking_config())
        forward = self.straight_path()
        follower.reset(forward, Pose2D(0.9, 0.0, 0.0))
        follower.goal_status(Pose2D(1.0, 0.0, 0.0))
        self.assertAlmostEqual(follower.path_station, forward.length, places=12)

        reverse = self.straight_path(direction=-1, heading=math.pi)
        projection = follower.reset(reverse, Pose2D(0.0, 0.0, math.pi))
        tracking = follower.calculate_tracking(Pose2D(0.0, 0.0, math.pi))

        self.assertAlmostEqual(projection.station, 0.0, places=12)
        self.assertAlmostEqual(follower.path_station, 0.0, places=12)
        self.assertEqual(tracking.direction, -1)

    def test_speed_adaptive_lookahead_grows_and_clamps_in_both_directions(self):
        path = CommonPath(
            x=np.linspace(0.0, 1.0, 21),
            y=np.zeros(21),
            heading=np.zeros(21),
            curvature=np.zeros(21),
            speed=np.full(21, 0.30),
            direction=1,
        )
        arguments = dict(vars(self.tracking_config()))
        arguments.update(
            lookahead_distance=0.10,
            lookahead_time=0.50,
            minimum_lookahead_distance=0.14,
            maximum_lookahead_distance=0.24,
        )
        config = TrackingConfig(**arguments)

        stopped = path_following_module.calculate_tracking(
            path, Pose2D(0.0, 0.0, 0.0), 0, config, linear_velocity=0.0
        )
        moving = path_following_module.calculate_tracking(
            path, Pose2D(0.0, 0.0, 0.0), 0, config, linear_velocity=0.20
        )
        capped = path_following_module.calculate_tracking(
            path, Pose2D(0.0, 0.0, 0.0), 0, config, linear_velocity=1.0
        )
        reverse_path = CommonPath(
            x=path.x.copy(),
            y=path.y.copy(),
            heading=np.full(path.size, math.pi),
            curvature=path.curvature.copy(),
            speed=path.speed.copy(),
            direction=-1,
        )
        reverse_speed = path_following_module.calculate_tracking(
            reverse_path,
            Pose2D(0.0, 0.0, math.pi),
            0,
            config,
            linear_velocity=-0.20,
        )

        self.assertAlmostEqual(
            stopped.steering_lookahead_distance, 0.14, places=12
        )
        self.assertAlmostEqual(
            moving.steering_lookahead_distance, 0.20, places=12
        )
        self.assertAlmostEqual(
            capped.steering_lookahead_distance, 0.24, places=12
        )
        self.assertAlmostEqual(
            reverse_speed.steering_lookahead_distance,
            moving.steering_lookahead_distance,
            places=12,
        )
        self.assertLess(stopped.target_index, moving.target_index)
        self.assertGreater(
            capped.steering_lookahead_distance,
            moving.steering_lookahead_distance,
        )

    def test_default_tracking_config_preserves_fixed_lookahead_and_speed_window(self):
        path = CommonPath(
            x=np.asarray([0.0, 0.10, 0.20, 0.30]),
            y=np.zeros(4),
            heading=np.zeros(4),
            curvature=np.zeros(4),
            speed=np.asarray([0.26, 0.26, 0.06, 0.06]),
            direction=1,
        )
        arguments = dict(vars(self.tracking_config()))
        arguments["lookahead_distance"] = 0.30
        config = TrackingConfig(**arguments)

        tracking = path_following_module.calculate_tracking(
            path,
            Pose2D(0.0, 0.0, 0.0),
            0,
            config,
            linear_velocity=0.50,
        )

        self.assertEqual(config.lookahead_time, 0.0)
        self.assertIsNone(config.speed_preview_distance)
        self.assertAlmostEqual(
            tracking.steering_lookahead_distance, 0.30, places=12
        )
        self.assertAlmostEqual(
            tracking.speed_preview_distance, 0.30, places=12
        )
        self.assertAlmostEqual(tracking.target_speed, 0.06, places=12)

    def test_speed_preview_is_independent_of_long_steering_lookahead(self):
        path = CommonPath(
            x=np.asarray([0.0, 0.10, 0.20, 0.30]),
            y=np.zeros(4),
            heading=np.zeros(4),
            curvature=np.zeros(4),
            speed=np.asarray([0.26, 0.26, 0.06, 0.06]),
            direction=1,
        )
        arguments = dict(vars(self.tracking_config()))
        arguments.update(
            lookahead_distance=0.30,
            speed_preview_distance=0.0,
        )
        current_only = path_following_module.calculate_tracking(
            path,
            Pose2D(0.0, 0.0, 0.0),
            0,
            TrackingConfig(**arguments),
        )
        arguments["speed_preview_distance"] = 0.11
        future_window = path_following_module.calculate_tracking(
            path,
            Pose2D(0.0, 0.0, 0.0),
            0,
            TrackingConfig(**arguments),
        )

        self.assertAlmostEqual(
            current_only.steering_lookahead_distance, 0.30, places=12
        )
        self.assertAlmostEqual(
            current_only.speed_preview_distance, 0.0, places=12
        )
        self.assertAlmostEqual(current_only.target_speed, 0.26, places=12)
        self.assertAlmostEqual(
            future_window.speed_preview_distance, 0.11, places=12
        )
        self.assertAlmostEqual(future_window.target_speed, 0.06, places=12)

    def test_observed_target_floor_skips_blind_extension(self):
        path = CommonPath(
            x=np.linspace(0.0, 0.60, 7),
            y=np.zeros(7),
            heading=np.zeros(7),
            curvature=np.zeros(7),
            speed=np.full(7, 0.20),
            direction=1,
        )
        arguments = dict(vars(self.tracking_config()))
        arguments["lookahead_distance"] = 0.05
        config = TrackingConfig(**arguments)

        blind_target = path_following_module.calculate_tracking(
            path, Pose2D(0.0, 0.0, 0.0), 0, config
        )
        observed_target = path_following_module.calculate_tracking(
            path,
            Pose2D(0.0, 0.0, 0.0),
            0,
            config,
            minimum_target_station=0.20,
        )

        self.assertAlmostEqual(
            blind_target.steering_lookahead_distance, 0.05, places=12
        )
        self.assertAlmostEqual(
            observed_target.steering_lookahead_distance, 0.20, places=12
        )
        self.assertLess(blind_target.target_index, observed_target.target_index)

    def test_common_path_rejects_a_direction_change_inside_one_segment(self):
        with self.assertRaisesRegex(
            ValueError, "split forward/reverse motion into separate segments"
        ):
            CommonPath(
                x=np.asarray([0.0, 0.20, 0.40, 0.60]),
                y=np.zeros(4),
                heading=np.asarray([0.0, 0.0, math.pi, math.pi]),
                curvature=np.zeros(4),
                speed=np.full(4, 0.10),
                direction=np.asarray([1, 1, -1, -1]),
            )

    def test_follower_and_command_forward_adaptive_tracking_inputs(self):
        path = CommonPath(
            x=np.linspace(0.0, 1.0, 11),
            y=np.zeros(11),
            heading=np.zeros(11),
            curvature=np.zeros(11),
            speed=np.full(11, 0.30),
            direction=1,
        )
        arguments = dict(vars(self.tracking_config()))
        arguments.update(
            lookahead_distance=0.05,
            lookahead_time=0.50,
            maximum_lookahead_distance=0.30,
        )
        follower = PathFollower(TrackingConfig(**arguments))
        pose = Pose2D(0.0, 0.0, 0.0)
        follower.reset(path, pose, initial_linear=0.10)

        default = follower.calculate_tracking(pose)
        explicit = follower.calculate_tracking(
            pose,
            linear_velocity=0.20,
            minimum_target_station=0.25,
        )
        _, commanded = follower.command(
            pose,
            0.10,
            linear_velocity=0.20,
            minimum_target_station=0.25,
        )

        self.assertAlmostEqual(
            default.steering_lookahead_distance, 0.10, places=12
        )
        self.assertAlmostEqual(
            explicit.steering_lookahead_distance, 0.25, places=12
        )
        self.assertAlmostEqual(
            commanded.steering_lookahead_distance, 0.25, places=12
        )

    def test_adaptive_tracking_rejects_invalid_bounds_preview_and_target_floor(self):
        base = dict(vars(self.tracking_config()))
        invalid = (
            ("lookahead_time", -0.01),
            ("lookahead_time", math.nan),
            ("minimum_lookahead_distance", -0.01),
            ("maximum_lookahead_distance", 0.0),
            ("maximum_lookahead_distance", math.nan),
            ("speed_preview_distance", -0.01),
            ("speed_preview_distance", math.inf),
        )
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                arguments = dict(base)
                arguments[field] = value
                with self.assertRaises(ValueError):
                    TrackingConfig(**arguments)

        reversed_bounds = dict(base)
        reversed_bounds.update(
            minimum_lookahead_distance=0.30,
            maximum_lookahead_distance=0.20,
        )
        with self.assertRaisesRegex(ValueError, "bounds are reversed"):
            TrackingConfig(**reversed_bounds)

        path = self.straight_path()
        config = self.tracking_config()
        for floor in (-0.01, path.length + 0.01, math.nan, math.inf):
            with self.subTest(floor=floor):
                with self.assertRaisesRegex(ValueError, "active path"):
                    path_following_module.calculate_tracking(
                        path,
                        Pose2D(0.0, 0.0, 0.0),
                        0,
                        config,
                        minimum_target_station=floor,
                    )
        with self.assertRaisesRegex(ValueError, "linear velocity"):
            path_following_module.calculate_tracking(
                path,
                Pose2D(0.0, 0.0, 0.0),
                0,
                config,
                linear_velocity=math.nan,
            )

    def test_curvature_profile_obeys_angular_and_lateral_limits(self):
        station = np.asarray([0.0, 0.5, 1.0])
        curvature = np.asarray([0.0, 4.0, 0.0])
        profile = SpeedProfile(
            cruise_velocity=0.8,
            minimum_velocity=0.30,
            entry_velocity=0.8,
            exit_velocity=0.8,
            maximum_angular_velocity=1.0,
            maximum_lateral_acceleration=0.20,
            linear_acceleration=2.0,
            linear_deceleration=2.0,
            angular_acceleration=10.0,
        )
        speed = build_speed_profile(station, curvature, profile)
        self.assertLessEqual(speed[1] * abs(curvature[1]), 1.0 + 1e-9)
        self.assertLessEqual(speed[1] ** 2 * abs(curvature[1]), 0.20 + 1e-9)
        # A nominal minimum may never override a physical curvature limit.
        self.assertLess(speed[1], profile.minimum_velocity)

    def test_minimum_velocity_is_the_nominal_entry_and_exit_floor(self):
        arguments = dict(
            cruise_velocity=0.4,
            minimum_velocity=0.10,
            entry_velocity=0.10,
            exit_velocity=0.10,
            maximum_angular_velocity=1.0,
            maximum_lateral_acceleration=0.4,
            linear_acceleration=2.0,
            linear_deceleration=2.0,
        )
        profile = SpeedProfile(**arguments)
        speed = build_speed_profile(
            np.asarray([0.0, 0.5, 1.0]),
            np.zeros(3),
            profile,
        )
        self.assertGreaterEqual(float(np.min(speed)), profile.minimum_velocity)

        arguments["entry_velocity"] = 0.09
        with self.assertRaisesRegex(ValueError, "entry velocity"):
            SpeedProfile(**arguments)

    def test_one_point_profile_still_obeys_curvature_safety_cap(self):
        profile = SpeedProfile(
            cruise_velocity=0.4,
            minimum_velocity=0.10,
            entry_velocity=0.10,
            exit_velocity=0.10,
            maximum_angular_velocity=0.5,
            maximum_lateral_acceleration=0.04,
            linear_acceleration=1.0,
            linear_deceleration=1.0,
        )
        speed = build_speed_profile(
            np.asarray([0.0]),
            np.asarray([16.0]),
            profile,
        )
        self.assertLess(speed[0], profile.minimum_velocity)
        self.assertLessEqual(speed[0] ** 2 * 16.0, 0.04 + 1e-12)

    def test_one_tracking_result_is_reused_by_goal_safety_and_command(self):
        path = self.straight_path()
        pose = Pose2D(0.20, 0.01, 0.0)
        follower = PathFollower(self.tracking_config())
        follower.reset(path, pose)
        tracking = follower.calculate_tracking(pose)
        validator = SweptFootprintValidator(
            AsymmetricFootprint(front=0.10, rear=0.08, half_width=0.06)
        )
        safe = path_following_module.ValidationResult(True)

        # Downstream operations receive the immutable result and must not call
        # either projection or calculate_tracking for the same pose again.
        with mock.patch.object(
            path_following_module,
            "calculate_tracking",
            side_effect=AssertionError("tracking was recalculated"),
        ), mock.patch.object(
            validator,
            "validate_path",
            return_value=safe,
        ), mock.patch.object(
            validator,
            "stopping_sweep",
            return_value=safe,
        ):
            decision = validator.motion_safety(
                path,
                pose,
                tracking.path_index,
                tracking.target_speed,
                0.0,
                (tracking.angular_velocity,),
                0.1,
                1.0,
                tracking=tracking,
            )
            status = follower.goal_status(pose, tracking=tracking)
            command, returned = follower.command(
                pose,
                0.1,
                speed_limit=decision.speed_limit,
                tracking=tracking,
            )

        self.assertIs(returned, tracking)
        self.assertFalse(status.complete)
        self.assertAlmostEqual(
            follower.diagnostics.progress, tracking.progress, places=12
        )
        self.assertAlmostEqual(
            follower.diagnostics.remaining_distance,
            path.length - tracking.station,
            places=12,
        )
        self.assertAlmostEqual(
            follower.diagnostics.commanded_linear,
            command.linear_velocity,
            places=12,
        )

    def test_feedforward_calibration_does_not_change_path_curvature(self):
        path = CommonPath(
            x=np.asarray([0.0, 0.1, 0.2]),
            y=np.asarray([0.0, 0.0, 0.0]),
            heading=np.zeros(3),
            curvature=np.full(3, 2.0),
            speed=np.full(3, 0.1),
            feedforward_scale=1.10,
        )
        config = self.tracking_config()
        tracking = PathFollower(config)
        tracking.reset(path, Pose2D(0.0, 0.0, 0.0))
        _, result = tracking.command(Pose2D(0.0, 0.0, 0.0), 0.10)
        self.assertAlmostEqual(result.feedforward_curvature, 2.0)
        self.assertAlmostEqual(result.curvature_command, 2.2)
        np.testing.assert_allclose(path.curvature, np.full(3, 2.0))

    def test_safety_speed_limit_preserves_commanded_path_curvature(self):
        path = CommonPath(
            x=np.asarray([0.0, 0.1, 0.2]),
            y=np.asarray([0.0, 0.0, 0.0]),
            heading=np.zeros(3),
            curvature=np.full(3, 2.0),
            speed=np.full(3, 0.2),
        )
        follower = PathFollower(self.tracking_config())
        follower.reset(
            path,
            Pose2D(0.0, 0.0, 0.0),
            initial_linear=0.1,
            initial_angular=0.2,
        )
        command, _ = follower.command(
            Pose2D(0.0, 0.0, 0.0),
            elapsed=0.1,
            speed_limit=0.1,
        )
        self.assertAlmostEqual(command.linear_velocity, 0.1, places=12)
        self.assertAlmostEqual(command.angular_velocity, 0.2, places=12)
        self.assertAlmostEqual(
            command.angular_velocity / command.linear_velocity,
            2.0,
            places=12,
        )

    def test_goal_uses_pose_heading_and_terminal_crossing(self):
        path = self.straight_path()
        crossed = goal_status(path, Pose2D(1.03, 0.0, 0.0))
        self.assertTrue(crossed.crossed_terminal)
        self.assertTrue(crossed.complete)
        wrong_heading = goal_status(path, Pose2D(1.01, 0.0, math.pi / 2.0))
        self.assertFalse(wrong_heading.complete)

    def test_terminal_half_plane_needs_near_terminal_path_progress(self):
        # The start lies on the final westbound terminal plane, but it is one
        # metre and three path segments before the actual terminal event.
        path = CommonPath(
            x=np.asarray([0.0, 1.0, 1.0, 0.0]),
            y=np.asarray([0.0, 0.0, 1.0, 1.0]),
            heading=np.asarray([0.0, math.pi / 2.0, math.pi, math.pi]),
            curvature=np.zeros(4),
            speed=np.full(4, 0.2),
            goal_tolerance=GoalTolerance(
                position=0.02,
                heading=math.radians(5.0),
                terminal_crossing=0.08,
            ),
        )

        middle_half_plane = goal_status(path, Pose2D(0.0, 0.0, math.pi))
        self.assertFalse(middle_half_plane.crossed_terminal)
        self.assertFalse(middle_half_plane.complete)
        terminal = goal_status(path, Pose2D(-0.03, 1.0, math.pi))
        self.assertTrue(terminal.crossed_terminal)
        self.assertTrue(terminal.complete)

    def test_terminal_hold_slew_limits_to_zero_and_never_reapplies_exit_speed(self):
        path = self.straight_path()
        follower = PathFollower(self.tracking_config())
        pose = Pose2D(1.0, 0.0, 0.0)
        follower.reset(
            path,
            pose,
            initial_linear=0.2,
            initial_angular=0.1,
        )
        tracking = follower.calculate_tracking(pose)

        first, returned, status = follower.terminal_hold(
            pose,
            0.05,
            tracking=tracking,
        )
        self.assertIs(returned, tracking)
        self.assertTrue(status.complete)
        self.assertTrue(status.crossed_terminal)
        self.assertAlmostEqual(first.linear_velocity, 0.1, places=12)
        second, _, _ = follower.terminal_hold(pose, 0.05)
        self.assertAlmostEqual(second.linear_velocity, 0.0, places=12)
        self.assertAlmostEqual(second.angular_velocity, 0.0, places=12)
        third, _, _ = follower.terminal_hold(pose, 0.05)
        self.assertAlmostEqual(third.linear_velocity, 0.0, places=12)

        before_terminal = PathFollower(self.tracking_config())
        before_terminal.reset(path, Pose2D(0.5, 0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "near-path terminal crossing"):
            before_terminal.terminal_hold(Pose2D(0.5, 0.0, 0.0), 0.05)

    def test_regular_command_holds_crossed_terminal_while_correcting_heading(self):
        path = self.straight_path()
        follower = PathFollower(self.tracking_config())
        pose = Pose2D(1.01, 0.0, 0.10)
        follower.reset(
            path,
            pose,
            initial_linear=0.10,
            initial_angular=0.0,
        )
        status = follower.goal_status(pose)
        self.assertTrue(status.crossed_terminal)
        self.assertFalse(status.complete)

        command, _ = follower.command(pose, 0.05)

        self.assertAlmostEqual(command.linear_velocity, 0.0, places=12)
        self.assertLess(command.angular_velocity, 0.0)
        self.assertAlmostEqual(
            command.angular_velocity,
            follower.config.heading_gain * status.heading_error,
            places=12,
        )

    def test_linear_and_angular_slew_limits_apply_in_both_directions(self):
        config = self.tracking_config()
        accelerating = limit_velocity_command(
            0.5, 0.5, 0.0, 0.0, 0.0, 0.10, config
        )
        self.assertLessEqual(accelerating.linear_velocity, 0.10 + 1e-12)
        braking = limit_velocity_command(
            0.0, 0.2, 0.0, 0.2, 0.0, 0.05, config
        )
        self.assertGreaterEqual(braking.linear_velocity, 0.10 - 1e-12)
        reversing = limit_velocity_command(
            -0.5, 0.5, -1.0, 0.0, 0.0, 0.10, config
        )
        self.assertGreaterEqual(reversing.linear_velocity, -0.10 - 1e-12)
        self.assertLessEqual(abs(reversing.angular_velocity), 0.30 + 1e-12)

        curved_braking = limit_velocity_command(
            0.02, 0.10, 0.8, 0.10, 0.0, 0.05, config
        )
        self.assertGreaterEqual(curved_braking.linear_velocity, 0.0)
        self.assertLessEqual(
            0.10 - curved_braking.linear_velocity,
            config.linear_deceleration * 0.05 + 1e-12,
        )
        self.assertLessEqual(
            abs(curved_braking.angular_velocity),
            config.angular_acceleration * 0.05 + 1e-12,
        )
        lateral_transition = limit_velocity_command(
            0.5,
            0.5,
            0.0,
            0.10,
            0.30,
            0.10,
            TrackingConfig(
                lookahead_distance=0.1,
                maximum_linear_velocity=0.5,
                maximum_angular_velocity=1.0,
                maximum_lateral_acceleration=0.035,
                linear_acceleration=1.0,
                linear_deceleration=1.0,
                angular_acceleration=0.1,
                heading_gain=0.5,
            ),
        )
        self.assertLessEqual(
            abs(
                lateral_transition.linear_velocity
                * lateral_transition.angular_velocity
            ),
            0.035 + 1e-12,
        )


class SweptFootprintTest(unittest.TestCase):
    def setUp(self):
        self.footprint = AsymmetricFootprint(0.20, 0.10, 0.10)
        self.validator = SweptFootprintValidator(
            self.footprint, translation_step=0.02, heading_step=math.radians(2.0)
        )

    def test_shared_segment_endpoints_are_evaluated_once_exactly(self):
        evaluated_x = []

        def clearance(pose, _footprint):
            evaluated_x.append(pose.x)
            return 0.50 - pose.x

        validator = SweptFootprintValidator(
            self.footprint,
            translation_step=1.0,
            heading_step=math.pi,
        )
        result = validator.validate_poses(
            (
                Pose2D(0.0, 0.0, 0.0),
                Pose2D(0.1, 0.0, 0.0),
                Pose2D(0.2, 0.0, 0.0),
            ),
            safety=PathSafety(
                line_boundaries=(SimpleNamespace(clearance=clearance),)
            ),
        )

        np.testing.assert_allclose(evaluated_x, (0.0, 0.1, 0.2), atol=0.0)
        self.assertEqual(result.samples, 3)
        self.assertAlmostEqual(result.minimum_line_clearance, 0.30, places=12)

    def test_single_subdivision_normalizes_unwrapped_endpoint_yaw(self):
        evaluated_yaw = []

        def clearance(pose, _footprint):
            evaluated_yaw.append(pose.yaw)
            return 1.0

        validator = SweptFootprintValidator(
            self.footprint,
            translation_step=1.0,
            heading_step=math.pi,
        )
        result = validator.validate_poses(
            (
                Pose2D(0.0, 0.0, 0.0),
                Pose2D(0.1, 0.0, 2.0 * math.pi + 0.25),
            ),
            safety=PathSafety(
                line_boundaries=(SimpleNamespace(clearance=clearance),)
            ),
        )

        self.assertTrue(result.safe)
        self.assertEqual(result.samples, 2)
        self.assertAlmostEqual(evaluated_yaw[-1], 0.25, places=12)

    def test_obstacle_between_sparse_path_points_is_detected(self):
        safety = PathSafety(fixed_obstacles=np.asarray([[0.50, 0.0]]))
        path = path_from_xy(
            [[0.0, 0.0], [1.0, 0.0]],
            "map",
            target_speed=0.2,
            safety=safety,
        )
        result = self.validator.validate_path(path)
        self.assertFalse(result.safe)
        self.assertLess(result.minimum_obstacle_clearance, 0.0)
        self.assertGreater(result.samples, 2)

    def test_reaction_sweep_covers_intermediate_yaw_rates(self):
        footprint = AsymmetricFootprint(0.002, 0.002, 0.002)
        validator = SweptFootprintValidator(
            footprint,
            translation_step=0.001,
            heading_step=math.radians(2.0),
        )
        safety = PathSafety(fixed_obstacles=np.asarray([[0.055, 0.0]]))
        path = path_from_xy(
            [[0.0, 0.0], [0.20, 0.0]],
            "odom",
            target_speed=0.20,
            safety=safety,
        )
        arguments = dict(
            path=path,
            pose=Pose2D(0.0, 0.0, 0.0),
            path_index=0,
            linear_velocity=0.20,
            reaction_time=0.30,
            linear_deceleration=100.0,
        )

        # The two observed endpoint-rate arcs bend around this point. A
        # slew-limited actuator can occupy every intermediate rate, including
        # the straight reaction arc that intersects it.
        self.assertTrue(
            validator.stopping_sweep(
                angular_velocities=(-2.0,), **arguments
            ).safe
        )
        self.assertTrue(
            validator.stopping_sweep(
                angular_velocities=(2.0,), **arguments
            ).safe
        )
        envelope = validator.stopping_sweep(
            angular_velocities=(-2.0, 2.0), **arguments
        )
        self.assertFalse(envelope.safe)
        self.assertLessEqual(envelope.minimum_obstacle_clearance, 0.0)

    def test_numerically_zero_yaw_rates_share_the_exact_straight_sweep(self):
        path = path_from_xy(
            [[0.0, 0.0], [1.0, 0.0]],
            "odom",
            target_speed=0.20,
            safety=PathSafety(
                fixed_obstacles=np.asarray([[0.95, 0.25]], dtype=np.float64)
            ),
        )
        arguments = dict(
            path=path,
            pose=Pose2D(0.0, 0.0, 0.0),
            path_index=0,
            linear_velocity=0.20,
            reaction_time=0.10,
            linear_deceleration=0.03,
            distance_margin=0.005,
        )

        exact = self.validator.stopping_sweep(
            angular_velocities=(0.0,), **arguments
        )
        residual = self.validator.stopping_sweep(
            angular_velocities=(-5e-10, 0.0, 8e-10), **arguments
        )

        self.assertEqual(
            self.validator._reaction_angular_samples(
                (-5e-10, 0.0, 8e-10), 0.10, 0.20
            ),
            (0.0,),
        )
        self.assertEqual(residual, exact)

    def test_fixed_and_live_obstacles_share_one_nearest_clearance(self):
        safety = PathSafety(
            fixed_obstacles=np.asarray([[0.80, 0.0]], dtype=np.float64)
        )
        poses = (Pose2D(0.0, 0.0, 0.0), Pose2D(0.20, 0.0, 0.0))
        live_obstacles = np.asarray([[0.31, 0.0]], dtype=np.float64)

        combined = self.validator.validate_poses(
            poses, safety=safety, live_obstacles=live_obstacles
        )
        live_only = self.validator.validate_poses(
            poses, safety=PathSafety(), live_obstacles=live_obstacles
        )

        self.assertEqual(combined.samples, live_only.samples)
        self.assertEqual(
            combined.minimum_obstacle_clearance,
            live_only.minimum_obstacle_clearance,
        )
        self.assertEqual(
            combined.first_unsafe_distance,
            live_only.first_unsafe_distance,
        )
        self.assertEqual(combined.safe, live_only.safe)

    def test_reaction_arc_is_not_replaced_by_its_endpoint_chord(self):
        footprint = AsymmetricFootprint(0.0001, 0.0001, 0.0001)
        validator = SweptFootprintValidator(
            footprint,
            translation_step=0.02,
            heading_step=math.radians(2.0),
        )
        linear_velocity = 0.02
        angular_velocity = 10.0
        reaction_time = 0.20
        arc_midpoint = validator._advance_unicycle(
            Pose2D(0.0, 0.0, 0.0),
            linear_velocity,
            angular_velocity,
            0.5 * reaction_time,
        )
        path = path_from_xy(
            [[0.0, 0.0], [1.0, 0.0]],
            "odom",
            target_speed=linear_velocity,
            safety=PathSafety(
                fixed_obstacles=np.asarray([[arc_midpoint.x, arc_midpoint.y]])
            ),
        )

        result = validator.stopping_sweep(
            path,
            Pose2D(0.0, 0.0, 0.0),
            path_index=0,
            linear_velocity=linear_velocity,
            angular_velocities=(angular_velocity,),
            reaction_time=reaction_time,
            linear_deceleration=1e6,
        )

        self.assertFalse(result.safe)
        self.assertLessEqual(result.minimum_obstacle_clearance, 0.0)

    def test_line_sweep_includes_localization_and_tracking_error(self):
        corridor = StraightCorridorBoundary(-0.15, 0.15)
        physical_only = PathSafety(line_boundaries=(corridor,))
        uncertain = PathSafety(
            line_boundaries=(corridor,),
            margins=SafetyMargins(line=0.01, localization=0.02, tracking=0.02),
        )
        path = path_from_xy(
            [[0.0, 0.0], [0.5, 0.0]],
            "map",
            target_speed=0.2,
            safety=physical_only,
        )
        self.assertTrue(self.validator.validate_path(path).safe)
        expanded_result = self.validator.validate_path(path, safety=uncertain)
        self.assertFalse(expanded_result.safe)
        self.assertLessEqual(expanded_result.minimum_line_clearance, 0.0)

    def test_initial_line_overlap_can_egress_through_common_path_envelope(self):
        start_cell = RasterCellBoundary(
            [[0.0, 0.102]], cell_size=(0.01, 0.01)
        )
        safety = PathSafety(line_boundaries=(start_cell,))
        path = path_from_xy(
            [[0.0, 0.0], [0.15, 0.0], [0.30, 0.0]],
            "map",
            target_speed=0.10,
            initial_line_overlap_allowance=0.01,
            line_egress_distance=0.20,
            safety=safety,
        )

        result = self.validator.validate_path(path)

        self.assertTrue(result.safe)
        self.assertLess(result.minimum_line_clearance, 0.0)

    def test_line_overlap_after_egress_envelope_is_still_rejected(self):
        late_cell = RasterCellBoundary(
            [[0.25, 0.102]], cell_size=(0.01, 0.01)
        )
        path = path_from_xy(
            [[0.0, 0.0], [0.15, 0.0], [0.30, 0.0]],
            "map",
            target_speed=0.10,
            initial_line_overlap_allowance=0.01,
            line_egress_distance=0.10,
            safety=PathSafety(line_boundaries=(late_cell,)),
        )

        result = self.validator.validate_path(path)

        self.assertFalse(result.safe)
        self.assertLessEqual(result.minimum_line_clearance, 0.0)

    def test_stopping_sweep_uses_line_envelope_but_not_for_obstacles(self):
        start_cell = RasterCellBoundary(
            [[0.0, 0.102]], cell_size=(0.01, 0.01)
        )
        path = path_from_xy(
            [[0.0, 0.0], [0.15, 0.0], [0.30, 0.0]],
            "odom",
            target_speed=0.10,
            initial_line_overlap_allowance=0.01,
            line_egress_distance=0.20,
            safety=PathSafety(line_boundaries=(start_cell,)),
        )
        allowed = self.validator.stopping_sweep(
            path,
            Pose2D(0.0, 0.0, 0.0),
            path_index=0,
            linear_velocity=0.05,
            angular_velocities=(0.0,),
            reaction_time=0.10,
            linear_deceleration=1.0,
        )
        self.assertTrue(allowed.safe)
        self.assertLess(allowed.minimum_line_clearance, 0.0)

        path.safety.fixed_obstacles = np.asarray([[0.05, 0.0]])
        blocked = self.validator.stopping_sweep(
            path,
            Pose2D(0.0, 0.0, 0.0),
            path_index=0,
            linear_velocity=0.05,
            angular_velocities=(0.0,),
            reaction_time=0.10,
            linear_deceleration=1.0,
        )
        self.assertFalse(blocked.safe)
        self.assertLessEqual(blocked.minimum_obstacle_clearance, 0.0)

    def test_raster_line_points_include_their_finite_cell_extent(self):
        pose = Pose2D(0.0, 0.0, 0.0)
        centre_only = PointCloudBoundary([[0.205, 0.0]])
        finite_cell = PointCloudBoundary([[0.205, 0.0]], point_radius=0.01)
        self.assertGreater(centre_only.clearance(pose, self.footprint), 0.0)
        self.assertLess(finite_cell.clearance(pose, self.footprint), 0.0)

    def test_raster_cells_keep_rectangular_edge_clearance(self):
        pose = Pose2D(0.0, 0.0, 0.0)
        boundary = RasterCellBoundary([[0.0, 0.113]], cell_size=(0.02, 0.02))
        circle_approximation = PointCloudBoundary(
            [[0.0, 0.113]], point_radius=math.hypot(0.01, 0.01)
        )

        self.assertAlmostEqual(
            boundary.clearance(pose, self.footprint), 0.003, places=12
        )
        self.assertLess(
            circle_approximation.clearance(pose, self.footprint), 0.0
        )

        overlapping = RasterCellBoundary(
            [[0.0, 0.109]], cell_size=(0.02, 0.02)
        )
        self.assertAlmostEqual(
            overlapping.clearance(pose, self.footprint), -0.001, places=12
        )

    def test_raster_cell_transform_preserves_exact_source_geometry(self):
        boundary = RasterCellBoundary(
            [[0.23, 0.11], [-0.17, -0.14]], cell_size=(0.018, 0.026)
        )
        source_pose = Pose2D(0.03, -0.02, math.radians(23.0))
        first = RigidTransform2D(2.1, -1.7, math.radians(71.0))
        second = RigidTransform2D(
            -0.4,
            0.8,
            math.radians(-19.0),
            source_frame="odom",
            target_frame="tracking",
        )

        source_clearance = boundary.clearance(source_pose, self.footprint)
        once = boundary.transformed(first)
        twice = once.transformed(second)
        self.assertAlmostEqual(
            once.clearance(first.apply_pose(source_pose), self.footprint),
            source_clearance,
            places=12,
        )
        self.assertAlmostEqual(
            twice.clearance(
                second.apply_pose(first.apply_pose(source_pose)), self.footprint
            ),
            source_clearance,
            places=12,
        )

    def test_swept_validator_detects_an_occupied_raster_cell_between_points(self):
        boundary = RasterCellBoundary([[0.50, 0.0]], cell_size=(0.02, 0.02))
        path = path_from_xy(
            [[0.0, 0.0], [1.0, 0.0]],
            "map",
            target_speed=0.2,
            safety=PathSafety(line_boundaries=(boundary,)),
        )

        result = self.validator.validate_path(path)

        self.assertFalse(result.safe)
        self.assertLessEqual(result.minimum_line_clearance, 0.0)
        self.assertGreater(result.samples, 2)

    def test_transformed_map_boundary_uses_the_path_frame(self):
        boundary = AxisAlignedBoundsBoundary(-1.0, 1.0, -1.0, 1.0)
        transformed = boundary.transformed(
            RigidTransform2D(3.0, -2.0, math.pi / 4.0)
        )
        inside = RigidTransform2D(3.0, -2.0, math.pi / 4.0).apply_pose(
            Pose2D(0.0, 0.0, 0.0)
        )
        outside = RigidTransform2D(3.0, -2.0, math.pi / 4.0).apply_pose(
            Pose2D(1.2, 0.0, 0.0)
        )
        self.assertGreater(transformed.clearance(inside, self.footprint), 0.0)
        self.assertLess(transformed.clearance(outside, self.footprint), 0.0)

    def test_reaction_and_complete_stop_region_uses_actual_rectangle(self):
        path = self.straight_path_with_obstacle(0.31)
        result = self.validator.stopping_sweep(
            path,
            Pose2D(0.0, 0.0, 0.0),
            path_index=0,
            linear_velocity=0.40,
            angular_velocities=(0.0,),
            reaction_time=0.10,
            linear_deceleration=0.50,
        )
        self.assertFalse(result.safe)
        self.assertLessEqual(result.minimum_obstacle_clearance, 0.0)

    def test_runtime_safety_slows_before_a_bounded_stop_is_required(self):
        path = self.straight_path_with_obstacle(0.55)
        decision = self.validator.motion_safety(
            path,
            Pose2D(0.0, 0.0, 0.0),
            path_index=0,
            desired_speed=0.60,
            linear_velocity=0.10,
            angular_velocities=(0.0,),
            reaction_time=0.10,
            linear_deceleration=0.50,
        )
        self.assertFalse(decision.route.safe)
        self.assertTrue(decision.stopping.safe)
        self.assertGreater(decision.speed_limit, 0.0)
        self.assertLess(decision.speed_limit, 0.60)

    def test_runtime_route_safety_can_omit_prevalidated_fixed_boundaries(self):
        path = self.straight_path_with_obstacle(0.55)
        decision = self.validator.motion_safety(
            path,
            Pose2D(0.0, 0.0, 0.0),
            path_index=0,
            desired_speed=0.60,
            linear_velocity=0.10,
            angular_velocities=(0.0,),
            reaction_time=0.10,
            linear_deceleration=0.50,
            route_safety=PathSafety(),
        )

        self.assertTrue(decision.route.safe)
        self.assertTrue(decision.stopping.safe)
        self.assertAlmostEqual(decision.speed_limit, 0.60, places=12)

    def test_runtime_route_safety_never_weakens_actual_stopping_sweep(self):
        path = self.straight_path_with_obstacle(0.21)
        decision = self.validator.motion_safety(
            path,
            Pose2D(0.0, 0.0, 0.0),
            path_index=0,
            desired_speed=0.20,
            linear_velocity=0.10,
            angular_velocities=(0.0,),
            reaction_time=0.10,
            linear_deceleration=0.50,
            route_safety=PathSafety(),
        )

        self.assertTrue(decision.route.safe)
        self.assertFalse(decision.stopping.safe)
        self.assertLessEqual(decision.stopping.minimum_obstacle_clearance, 0.0)
        self.assertEqual(decision.speed_limit, 0.0)

    def test_runtime_route_split_keeps_future_actual_line_sweep(self):
        def future_actual_line(pose, _footprint):
            if pose.x >= 0.005 and pose.y >= 0.03:
                return -0.01
            return 0.01

        path = path_from_xy(
            [[0.0, 0.0], [0.20, 0.0]],
            "odom",
            target_speed=0.10,
            safety=PathSafety(
                line_boundaries=(
                    SimpleNamespace(clearance=future_actual_line),
                )
            ),
        )
        current_pose = Pose2D(0.0, 0.04, 0.0)
        self.assertGreater(
            future_actual_line(current_pose, self.footprint),
            0.0,
        )

        decision = self.validator.motion_safety(
            path,
            current_pose,
            path_index=0,
            desired_speed=0.10,
            linear_velocity=0.10,
            angular_velocities=(0.0,),
            reaction_time=0.10,
            linear_deceleration=0.50,
            route_safety=PathSafety(),
        )

        self.assertTrue(decision.route.safe)
        self.assertFalse(decision.stopping.safe)
        self.assertLessEqual(decision.stopping.minimum_line_clearance, 0.0)
        self.assertEqual(decision.speed_limit, 0.0)

    def test_braking_sweep_continues_from_a_steered_reaction_pose(self):
        path = self.straight_path_with_obstacle_at((0.50, 0.17))
        result = self.validator.stopping_sweep(
            path,
            Pose2D(0.0, 0.0, 0.0),
            path_index=0,
            linear_velocity=0.40,
            angular_velocities=(1.0,),
            reaction_time=0.50,
            linear_deceleration=0.50,
        )
        self.assertFalse(result.safe)
        self.assertLessEqual(result.minimum_obstacle_clearance, 0.0)

    def straight_path_with_obstacle(self, obstacle_x):
        return self.straight_path_with_obstacle_at((obstacle_x, 0.0))

    def straight_path_with_obstacle_at(self, obstacle):
        return path_from_xy(
            [[0.0, 0.0], [1.0, 0.0]],
            "map",
            target_speed=0.4,
            safety=PathSafety(
                fixed_obstacles=np.asarray([obstacle], dtype=np.float64)
            ),
        )


if __name__ == "__main__":
    unittest.main()
