#!/usr/bin/env python3

from dataclasses import FrozenInstanceError
import math
import unittest

import numpy as np

from custom_autorace_bringup.local_registration import (
    CurveRegistrationConfig,
    EntryPlane,
    LocalCurveTemplate,
    MissionLocalTemplate,
    ObservedCurve,
    OrientedSegmentLandmark,
    OrientedSegmentObservation,
    PointLandmark,
    PointObservation,
    RegistrationConfig,
    RegistrationDiagnostics,
    RegistrationObservation,
    RegistrationResult,
    TemporalRegistrationConfig,
    TemporalRegistrationFilter,
    entry_plane_progress,
    estimate_local_registration,
    register_point_with_heading,
    registration_covariance_with_floor,
    registration_radial_uncertainties,
    registration_radial_uncertainty,
    register_curve_subset,
)
from custom_autorace_bringup.path_following import (
    AsymmetricFootprint,
    Pose2D,
    RigidTransform2D,
)


def assert_transform_close(test, actual, expected, position=1e-8, heading=1e-8):
    test.assertIsNotNone(actual)
    test.assertLessEqual(
        math.hypot(
            actual.target_from_source_x - expected.target_from_source_x,
            actual.target_from_source_y - expected.target_from_source_y,
        ),
        position,
    )
    test.assertLessEqual(
        abs(
            math.atan2(
                math.sin(
                    actual.target_from_source_yaw
                    - expected.target_from_source_yaw
                ),
                math.cos(
                    actual.target_from_source_yaw
                    - expected.target_from_source_yaw
                ),
            )
        ),
        heading,
    )
    test.assertEqual(actual.source_frame, expected.source_frame)
    test.assertEqual(actual.target_frame, expected.target_frame)


def transformed_point_observations(template, transform, overrides=None):
    overrides = {} if overrides is None else overrides
    return tuple(
        PointObservation(
            landmark.name,
            overrides.get(
                landmark.name, transform.apply_point(landmark.point)
            ),
        )
        for landmark in template.points
    )


def transformed_segment_observations(template, transform, overrides=None):
    overrides = {} if overrides is None else overrides
    values = []
    for landmark in template.segments:
        start, end = overrides.get(
            landmark.name,
            (
                transform.apply_point(landmark.start),
                transform.apply_point(landmark.end),
            ),
        )
        values.append(OrientedSegmentObservation(landmark.name, start, end))
    return tuple(values)


def result_for(transform, stamp, accepted=True, reason=""):
    diagnostics = RegistrationDiagnostics(
        reason=reason,
        degenerate=False,
        total_landmarks=3,
        matched_landmarks=3,
        inlier_landmarks=3 if accepted else 0,
        point_inliers=3 if accepted else 0,
        segment_inliers=0,
        template_coverage=1.0,
        inlier_coverage=1.0 if accepted else 0.0,
        spatial_baseline=1.0,
        position_rms=0.0 if accepted else math.inf,
        heading_rms=0.0 if accepted else math.inf,
        robust_rms=0.0 if accepted else math.inf,
        maximum_position_residual=0.0 if accepted else math.inf,
        maximum_heading_residual=0.0 if accepted else math.inf,
        information_condition=2.0,
        iterations=1,
    )
    return RegistrationResult(
        accepted=accepted,
        stamp=stamp,
        transform=transform,
        covariance=((1e-6, 0.0, 0.0), (0.0, 1e-6, 0.0), (0.0, 0.0, 1e-6))
        if accepted
        else tuple(),
        diagnostics=diagnostics,
        inlier_landmarks=("a", "b", "c") if accepted else tuple(),
    )


def interpolate_curve(template, queries):
    points = np.asarray(template.points, dtype=np.float64)
    station = np.asarray(template.station, dtype=np.float64)
    return np.column_stack(
        (
            np.interp(queries, station, points[:, 0]),
            np.interp(queries, station, points[:, 1]),
        )
    )


class LocalRegistrationTypeTest(unittest.TestCase):
    def test_template_is_immutable_and_rejects_duplicate_names(self):
        landmark = PointLandmark("corner", (0.0, 0.0))
        template = MissionLocalTemplate(
            "obstacle", "obstacle_local", points=(landmark,)
        )

        with self.assertRaises(FrozenInstanceError):
            template.frame_id = "map"
        with self.assertRaises(ValueError):
            MissionLocalTemplate(
                "bad",
                "bad_local",
                points=(landmark,),
                segments=(
                    OrientedSegmentLandmark(
                        "corner", (0.0, 0.0), (1.0, 0.0)
                    ),
                ),
            )

    def test_segment_and_entry_plane_validation(self):
        with self.assertRaises(ValueError):
            OrientedSegmentLandmark("wall", (0.0, 0.0), (0.0, 0.0))
        with self.assertRaises(ValueError):
            EntryPlane((0.0, 0.0), (0.0, 0.0))

        plane = EntryPlane((1.0, 2.0), (3.0, 4.0))
        self.assertAlmostEqual(math.hypot(*plane.normal), 1.0, places=12)

    def test_registration_covariance_expands_at_furthest_body_corner(self):
        footprint = AsymmetricFootprint(0.2, 0.1, 0.15)
        covariance = (
            (0.0004, 0.0, 0.0),
            (0.0, 0.0001, 0.0),
            (0.0, 0.0, math.radians(2.0) ** 2),
        )

        uncertainty = registration_radial_uncertainty(
            covariance, footprint
        )

        self.assertAlmostEqual(
            uncertainty,
            0.02 + 0.25 * math.radians(2.0),
            places=12,
        )

    def test_registration_covariance_rejects_malformed_or_non_psd_input(self):
        footprint = AsymmetricFootprint(0.2, 0.1, 0.15)
        self.assertEqual(registration_radial_uncertainty(tuple(), footprint), 0.0)
        with self.assertRaises(ValueError):
            registration_radial_uncertainty(((1.0, 0.0),), footprint)
        with self.assertRaises(ValueError):
            registration_radial_uncertainty(
                ((-0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)),
                footprint,
            )

    def test_path_point_lever_arm_uses_full_registration_cross_covariance(self):
        footprint = AsymmetricFootprint(0.2, 0.1, 0.15)
        position_sigma = 0.005
        heading_sigma = math.radians(1.0)
        anchor = (0.3, 0.0)
        registration = register_point_with_heading(
            stamp=1.0,
            source_frame="local",
            target_frame="odom",
            local_point=anchor,
            target_point=anchor,
            target_from_source_yaw=0.0,
            position_standard_deviation=position_sigma,
            heading_standard_deviation=heading_sigma,
        )
        radius = math.hypot(footprint.front, footprint.half_width)

        at_anchor = registration_radial_uncertainty(
            registration.covariance,
            footprint,
            local_points=(anchor,),
            target_from_source_yaw=0.0,
        )
        one_metre_past_anchor = registration_radial_uncertainty(
            registration.covariance,
            footprint,
            local_points=((1.3, 0.0),),
            target_from_source_yaw=0.0,
        )

        # Heading error rotates about the observed landmark. At the anchor the
        # translation/yaw cross term cancels the lever arm; one metre farther
        # along the route the full one-metre centre displacement remains.
        self.assertAlmostEqual(
            at_anchor,
            position_sigma + radius * heading_sigma,
            places=12,
        )
        self.assertAlmostEqual(
            one_metre_past_anchor,
            math.hypot(position_sigma, heading_sigma)
            + radius * heading_sigma,
            places=12,
        )
        self.assertGreater(one_metre_past_anchor, at_anchor)
        profile = registration_radial_uncertainties(
            registration.covariance,
            footprint,
            local_points=(anchor, (1.3, 0.0)),
            target_from_source_yaw=0.0,
        )
        np.testing.assert_allclose(
            profile,
            (at_anchor, one_metre_past_anchor),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            registration_radial_uncertainty(
                registration.covariance,
                footprint,
                local_points=(anchor, (1.3, 0.0)),
                target_from_source_yaw=0.0,
            ),
            float(np.max(profile)),
            places=12,
        )


class PointHeadingRegistrationTest(unittest.TestCase):
    def test_point_and_independent_heading_recover_transform(self):
        expected = RigidTransform2D(
            1.2,
            -0.4,
            math.radians(23.0),
            "intersection_local",
            "odom",
        )
        local_point = (0.315, -0.022)

        result = register_point_with_heading(
            stamp=12.5,
            source_frame="intersection_local",
            target_frame="odom",
            local_point=local_point,
            target_point=expected.apply_point(local_point),
            target_from_source_yaw=expected.target_from_source_yaw,
            position_standard_deviation=0.005,
            heading_standard_deviation=math.radians(0.5),
        )

        self.assertTrue(result.accepted, result.diagnostics.reason)
        assert_transform_close(self, result.transform, expected)
        self.assertEqual(result.stamp, 12.5)
        self.assertEqual(result.inlier_landmarks, ("point", "heading"))

    def test_point_heading_covariance_preserves_lever_arm_and_cross_terms(self):
        position_sigma = 0.1
        heading_sigma = 0.2
        local_point = (1.0, 2.0)
        result = register_point_with_heading(
            stamp=1.0,
            source_frame="local",
            target_frame="odom",
            local_point=local_point,
            target_point=local_point,
            target_from_source_yaw=0.0,
            position_standard_deviation=position_sigma,
            heading_standard_deviation=heading_sigma,
        )

        # For t = q - R(theta)p at theta=0, dt/dtheta=(p_y, -p_x).
        derivative = np.asarray((2.0, -1.0), dtype=np.float64)
        expected = np.zeros((3, 3), dtype=np.float64)
        expected[:2, :2] = (
            np.eye(2) * position_sigma ** 2
            + np.outer(derivative, derivative) * heading_sigma ** 2
        )
        expected[:2, 2] = derivative * heading_sigma ** 2
        expected[2, :2] = expected[:2, 2]
        expected[2, 2] = heading_sigma ** 2

        self.assertTrue(np.allclose(np.asarray(result.covariance), expected))
        self.assertLess(result.covariance[0][1], 0.0)
        self.assertGreater(result.covariance[0][2], 0.0)
        self.assertLess(result.covariance[1][2], 0.0)

    def test_systematic_floor_is_not_reduced_by_temporal_confirmation(self):
        temporal_filter = TemporalRegistrationFilter(
            TemporalRegistrationConfig(required_confirmations=3)
        )
        for index, delta in enumerate((-0.001, 0.0, 0.001)):
            transform = RigidTransform2D(
                1.0 + delta,
                -0.2 + 0.5 * delta,
                0.1 + 2.0 * delta,
                "local",
                "odom",
            )
            state = temporal_filter.update(
                result_for(transform, 1.0 + 0.1 * index)
            )
        self.assertTrue(state.confirmed)

        position_floor = 0.012
        heading_floor = math.radians(1.0)
        covariance = registration_covariance_with_floor(
            state.covariance,
            position_floor,
            heading_floor,
        )
        before = np.asarray(state.covariance)
        after = np.asarray(covariance)

        self.assertAlmostEqual(
            after[0, 0] - before[0, 0], position_floor ** 2, places=15
        )
        self.assertAlmostEqual(
            after[1, 1] - before[1, 1], position_floor ** 2, places=15
        )
        self.assertAlmostEqual(
            after[2, 2] - before[2, 2], heading_floor ** 2, places=15
        )
        before_cross_terms = before - np.diag(np.diag(before))
        after_cross_terms = after - np.diag(np.diag(after))
        self.assertGreater(np.max(np.abs(before_cross_terms)), 0.0)
        self.assertTrue(np.allclose(after_cross_terms, before_cross_terms))

        # The swept validator consumes one sigma directly. Applying the floor
        # after temporal averaging therefore retains the full calibration
        # error radius rather than incorrectly dividing it by sqrt(N).
        footprint = AsymmetricFootprint(0.067645, 0.118073, 0.0903)
        radius = math.hypot(footprint.rear, footprint.half_width)
        self.assertGreaterEqual(
            registration_radial_uncertainty(covariance, footprint),
            position_floor + radius * heading_floor,
        )

    def test_point_heading_registration_and_floor_reject_invalid_uncertainty(self):
        arguments = dict(
            stamp=1.0,
            source_frame="local",
            target_frame="odom",
            local_point=(0.0, 0.0),
            target_point=(1.0, 2.0),
            target_from_source_yaw=0.0,
            position_standard_deviation=0.01,
            heading_standard_deviation=0.01,
        )
        for name, value in (
            ("position_standard_deviation", 0.0),
            ("heading_standard_deviation", -0.1),
            ("heading_standard_deviation", math.nan),
        ):
            invalid = dict(arguments)
            invalid[name] = value
            with self.assertRaises(ValueError):
                register_point_with_heading(**invalid)

        with self.assertRaises(ValueError):
            registration_covariance_with_floor(
                ((1.0, 0.0),), 0.012, math.radians(1.0)
            )
        with self.assertRaises(ValueError):
            registration_covariance_with_floor(tuple(), -0.001, 0.0)


class PointRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.template = MissionLocalTemplate(
            "intersection",
            "intersection_local",
            points=(
                PointLandmark("a", (0.0, 0.0)),
                PointLandmark("b", (1.0, 0.0)),
                PointLandmark("c", (0.2, 0.8)),
                PointLandmark("d", (1.1, 0.9)),
            ),
        )
        self.transform = RigidTransform2D(
            1.2,
            -0.4,
            math.radians(23.0),
            "intersection_local",
            "odom",
        )

    def test_exact_point_correspondences_recover_transform_and_covariance(self):
        result = estimate_local_registration(
            self.template,
            RegistrationObservation(
                stamp=12.5,
                target_frame="odom",
                points=transformed_point_observations(
                    self.template, self.transform
                ),
            ),
        )

        self.assertTrue(result.accepted, result.diagnostics.reason)
        assert_transform_close(self, result.transform, self.transform)
        self.assertEqual(result.inlier_landmarks, ("a", "b", "c", "d"))
        covariance = np.asarray(result.covariance)
        self.assertEqual(covariance.shape, (3, 3))
        self.assertTrue(np.allclose(covariance, covariance.T))
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) >= -1e-15))
        self.assertAlmostEqual(result.diagnostics.template_coverage, 1.0)
        self.assertGreater(result.diagnostics.spatial_baseline, 1.0)

    def test_one_gross_outlier_is_rejected_without_biasing_transform(self):
        observations = transformed_point_observations(
            self.template,
            self.transform,
            overrides={"d": (8.0, -7.0)},
        )
        result = estimate_local_registration(
            self.template,
            RegistrationObservation(1.0, "odom", points=observations),
            RegistrationConfig(
                minimum_inliers=3,
                minimum_inlier_coverage=0.70,
            ),
        )

        self.assertTrue(result.accepted, result.diagnostics.reason)
        assert_transform_close(self, result.transform, self.transform)
        self.assertEqual(result.inlier_landmarks, ("a", "b", "c"))
        self.assertAlmostEqual(result.diagnostics.inlier_coverage, 0.75)

    def test_missing_landmarks_report_template_coverage(self):
        observation = RegistrationObservation(
            1.0,
            "odom",
            points=transformed_point_observations(
                self.template, self.transform
            )[:2],
        )
        result = estimate_local_registration(
            self.template,
            observation,
            RegistrationConfig(minimum_template_coverage=0.75),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(
            result.diagnostics.reason, "insufficient template coverage"
        )
        self.assertAlmostEqual(result.diagnostics.template_coverage, 0.5)

    def test_single_point_is_rejected_as_unobservable(self):
        template = MissionLocalTemplate(
            "point", "point_local", points=(PointLandmark("only", (0, 0)),)
        )
        result = estimate_local_registration(
            template,
            RegistrationObservation(
                1.0, "odom", points=(PointObservation("only", (2, 3)),)
            ),
            RegistrationConfig(minimum_inliers=1),
        )

        self.assertFalse(result.accepted)
        self.assertTrue(result.diagnostics.degenerate)
        self.assertEqual(result.diagnostics.reason, "unobservable rigid transform")


class SegmentRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.transform = RigidTransform2D(
            0.65,
            -0.22,
            math.radians(31.0),
            "mission_local",
            "odom",
        )

    def test_two_nonparallel_unbounded_lines_recover_full_transform(self):
        template = MissionLocalTemplate(
            "tunnel",
            "mission_local",
            segments=(
                OrientedSegmentLandmark(
                    "north_wall",
                    (0.0, 0.0),
                    (1.0, 0.0),
                    longitudinal_weight=0.0,
                ),
                OrientedSegmentLandmark(
                    "portal",
                    (0.0, 0.0),
                    (0.0, 0.7),
                    longitudinal_weight=0.0,
                ),
            ),
        )
        result = estimate_local_registration(
            template,
            RegistrationObservation(
                2.0,
                "odom",
                segments=transformed_segment_observations(
                    template, self.transform
                ),
            ),
        )

        self.assertTrue(result.accepted, result.diagnostics.reason)
        assert_transform_close(self, result.transform, self.transform)
        self.assertEqual(result.diagnostics.segment_inliers, 2)

    def test_parallel_unbounded_walls_reject_longitudinal_ambiguity(self):
        template = MissionLocalTemplate(
            "corridor",
            "mission_local",
            segments=(
                OrientedSegmentLandmark(
                    "left", (0, 0), (1, 0), longitudinal_weight=0.0
                ),
                OrientedSegmentLandmark(
                    "right", (0, 1), (1, 1), longitudinal_weight=0.0
                ),
            ),
        )
        result = estimate_local_registration(
            template,
            RegistrationObservation(
                1.0,
                "odom",
                segments=transformed_segment_observations(
                    template, self.transform
                ),
            ),
        )

        self.assertFalse(result.accepted)
        self.assertTrue(result.diagnostics.degenerate)
        self.assertIn("unobservable", result.diagnostics.reason)

    def test_one_bounded_directed_segment_is_observable_when_explicitly_allowed(self):
        template = MissionLocalTemplate(
            "barrier",
            "mission_local",
            segments=(
                OrientedSegmentLandmark(
                    "bar", (-0.3, 0.0), (0.4, 0.0), longitudinal_weight=1.0
                ),
            ),
        )
        result = estimate_local_registration(
            template,
            RegistrationObservation(
                1.0,
                "odom",
                segments=transformed_segment_observations(
                    template, self.transform
                ),
            ),
            RegistrationConfig(minimum_inliers=1),
        )

        self.assertTrue(result.accepted, result.diagnostics.reason)
        assert_transform_close(self, result.transform, self.transform)

    def test_point_plus_unbounded_line_constrains_pose(self):
        template = MissionLocalTemplate(
            "sign_and_lane",
            "mission_local",
            points=(PointLandmark("sign", (0.4, 0.8)),),
            segments=(
                OrientedSegmentLandmark(
                    "lane", (0, 0), (1, 0), longitudinal_weight=0.0
                ),
            ),
        )
        result = estimate_local_registration(
            template,
            RegistrationObservation(
                1.0,
                "odom",
                points=transformed_point_observations(template, self.transform),
                segments=transformed_segment_observations(
                    template, self.transform
                ),
            ),
        )

        self.assertTrue(result.accepted, result.diagnostics.reason)
        assert_transform_close(self, result.transform, self.transform)

    def test_mismatched_oriented_line_is_removed_as_an_outlier(self):
        template = MissionLocalTemplate(
            "walls",
            "mission_local",
            segments=(
                OrientedSegmentLandmark(
                    "horizontal", (0, 0), (1, 0), longitudinal_weight=0.0
                ),
                OrientedSegmentLandmark(
                    "vertical", (0, 0), (0, 1), longitudinal_weight=0.0
                ),
                OrientedSegmentLandmark(
                    "mismatch", (1, 0), (1, 1), longitudinal_weight=0.0
                ),
            ),
        )
        observations = list(
            transformed_segment_observations(template, self.transform)
        )
        observations[-1] = OrientedSegmentObservation(
            "mismatch", (5.0, 5.0), (6.0, 5.0)
        )
        result = estimate_local_registration(
            template,
            RegistrationObservation(1.0, "odom", segments=tuple(observations)),
            RegistrationConfig(
                minimum_inliers=2,
                minimum_inlier_coverage=0.60,
            ),
        )

        self.assertTrue(result.accepted, result.diagnostics.reason)
        assert_transform_close(self, result.transform, self.transform)
        self.assertEqual(
            result.inlier_landmarks, ("horizontal", "vertical")
        )


class EntryPlaneProgressTest(unittest.TestCase):
    def test_progress_is_evaluated_in_registered_local_frame(self):
        transform = RigidTransform2D(
            2.0,
            -1.0,
            math.radians(90.0),
            "parking_local",
            "odom",
        )
        plane = EntryPlane((1.0, 0.0), (1.0, 0.0))
        local_pose = Pose2D(1.25, -0.08, math.radians(5.0))
        target_pose = transform.apply_pose(local_pose)

        progress = entry_plane_progress(plane, target_pose, transform)

        self.assertTrue(progress.crossed)
        self.assertAlmostEqual(progress.longitudinal, 0.25, places=12)
        self.assertAlmostEqual(progress.lateral, -0.08, places=12)
        self.assertAlmostEqual(progress.heading_error, math.radians(5.0), places=12)
        self.assertAlmostEqual(progress.local_pose.x, local_pose.x, places=12)

    def test_negative_progress_is_before_entry(self):
        transform = RigidTransform2D(0, 0, 0, "local", "odom")
        progress = entry_plane_progress(
            EntryPlane((0.5, 0), (1, 0)), Pose2D(0.3, 0, 0), transform
        )
        self.assertFalse(progress.crossed)
        self.assertAlmostEqual(progress.longitudinal, -0.2)


class TemporalRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.filter = TemporalRegistrationFilter(
            TemporalRegistrationConfig(
                required_confirmations=3,
                maximum_gap=0.20,
                maximum_position_delta=0.02,
                maximum_heading_delta=math.radians(2.0),
            )
        )

    def test_three_stable_source_ordered_results_confirm(self):
        values = (
            RigidTransform2D(1.000, 0.500, math.radians(179.5), "local", "odom"),
            RigidTransform2D(1.006, 0.497, math.radians(-179.8), "local", "odom"),
            RigidTransform2D(0.997, 0.504, math.radians(179.9), "local", "odom"),
        )
        states = [
            self.filter.update(result_for(transform, 1.0 + 0.1 * index))
            for index, transform in enumerate(values)
        ]

        self.assertFalse(states[0].confirmed)
        self.assertFalse(states[1].confirmed)
        self.assertTrue(states[2].confirmed)
        self.assertEqual(states[2].confirmation_count, 3)
        self.assertLess(states[2].maximum_position_spread, 0.02)
        self.assertLess(states[2].maximum_heading_spread, math.radians(1.0))
        self.assertEqual(np.asarray(states[2].covariance).shape, (3, 3))

    def test_large_transform_jump_starts_a_new_streak(self):
        first = RigidTransform2D(0.0, 0.0, 0.0, "local", "odom")
        second = RigidTransform2D(0.10, 0.0, 0.0, "local", "odom")
        self.filter.update(result_for(first, 1.0))
        state = self.filter.update(result_for(second, 1.1))

        self.assertFalse(state.confirmed)
        self.assertEqual(state.confirmation_count, 1)
        self.assertEqual(state.reason, "inconsistent transform")
        assert_transform_close(self, state.transform, second)

    def test_temporal_covariance_describes_the_mean_transform(self):
        temporal_filter = TemporalRegistrationFilter(
            TemporalRegistrationConfig(
                required_confirmations=3,
                maximum_gap=0.20,
                maximum_position_delta=0.50,
                maximum_heading_delta=math.radians(5.0),
            )
        )
        x_values = (-0.10, 0.0, 0.10)
        state = None
        for index, x_value in enumerate(x_values):
            state = temporal_filter.update(
                result_for(
                    RigidTransform2D(
                        x_value, 0.0, 0.0, "local", "odom"
                    ),
                    2.0 + 0.1 * index,
                )
            )

        covariance = np.asarray(state.covariance)
        expected_scatter_of_mean = np.var(
            np.asarray(x_values), ddof=1
        ) / len(x_values)
        expected_reported_of_mean = 1e-6 / len(x_values)
        self.assertAlmostEqual(
            covariance[0, 0],
            expected_scatter_of_mean + expected_reported_of_mean,
            places=12,
        )
        self.assertAlmostEqual(
            covariance[1, 1], expected_reported_of_mean, places=12
        )
        self.assertAlmostEqual(
            covariance[2, 2], expected_reported_of_mean, places=12
        )

    def test_time_gap_rejected_result_and_duplicate_stamp_do_not_confirm(self):
        transform = RigidTransform2D(0.0, 0.0, 0.0, "local", "odom")
        self.filter.update(result_for(transform, 1.0))
        gap_state = self.filter.update(result_for(transform, 1.4))
        self.assertEqual(gap_state.confirmation_count, 1)
        duplicate_state = self.filter.update(result_for(transform, 1.4))
        self.assertEqual(duplicate_state.confirmation_count, 1)
        self.assertEqual(duplicate_state.reason, "non-increasing source stamp")

        rejected = result_for(None, 1.5, accepted=False, reason="bad residual")
        state = self.filter.update(rejected)
        self.assertEqual(state.confirmation_count, 0)
        self.assertEqual(state.reason, "bad residual")


class CurveSubsetRegistrationTest(unittest.TestCase):
    def setUp(self):
        x = np.linspace(0.0, 2.0, 401)
        y = 0.12 * np.sin(2.7 * x) + 0.025 * x ** 2
        self.template = LocalCurveTemplate(
            "zigzag", "zigzag_local", tuple(zip(x, y))
        )
        self.transform = RigidTransform2D(
            0.7, -0.4, 0.35, "zigzag_local", "odom"
        )
        self.config = CurveRegistrationConfig(
            sample_count=25,
            station_search_step=0.005,
            minimum_heading_variation=math.radians(2.0),
            minimum_lateral_excitation=0.002,
            maximum_rms=0.005,
            ambiguity_station_separation=0.15,
            maximum_ambiguity_rms_difference=0.0005,
        )

    def test_curved_uncorresponded_subset_finds_station_and_transform(self):
        start, end = 0.55, 1.15
        local = interpolate_curve(self.template, np.linspace(start, end, 81))
        observed = ObservedCurve(
            4.0,
            "odom",
            tuple(self.transform.apply_point(point) for point in local),
        )

        result = register_curve_subset(self.template, observed, self.config)

        self.assertTrue(result.accepted, result.diagnostics.reason)
        self.assertAlmostEqual(result.start_station, start, delta=0.006)
        self.assertAlmostEqual(result.end_station, end, delta=0.008)
        assert_transform_close(
            self, result.transform, self.transform, position=2e-4, heading=2e-4
        )
        self.assertGreater(result.diagnostics.observed_heading_variation, 0.03)
        self.assertGreater(result.diagnostics.candidate_count, 100)
        self.assertTrue(result.registration.accepted)

    def test_straight_subset_is_rejected_before_station_search(self):
        template = LocalCurveTemplate(
            "straight",
            "straight_local",
            ((0, 0), (0.5, 0), (1.0, 0), (1.5, 0)),
        )
        observed = ObservedCurve(
            1.0,
            "odom",
            ((0.2, 1.0), (0.4, 1.0), (0.6, 1.0), (0.8, 1.0)),
        )

        result = register_curve_subset(template, observed)

        self.assertFalse(result.accepted)
        self.assertTrue(result.diagnostics.degenerate)
        self.assertIn("straight", result.diagnostics.reason)
        self.assertEqual(result.diagnostics.candidate_count, 0)

    def test_repeated_curve_motif_is_rejected_as_station_ambiguous(self):
        motif = np.asarray(
            ((0, 0), (0.25, 0.15), (0.5, 0), (0.75, -0.10), (1.0, 0)),
            dtype=np.float64,
        )
        repeated = np.vstack((motif, motif[1:] + (1.0, 0.0)))
        template = LocalCurveTemplate(
            "repeated", "repeated_local", tuple(map(tuple, repeated))
        )
        motif_length = float(
            np.sum(np.hypot(np.diff(motif[:, 0]), np.diff(motif[:, 1])))
        )
        local = interpolate_curve(template, np.linspace(0.0, motif_length, 61))
        transform = RigidTransform2D(
            0.3, 0.4, -0.2, "repeated_local", "odom"
        )
        observed = ObservedCurve(
            1.0,
            "odom",
            tuple(transform.apply_point(point) for point in local),
        )
        config = CurveRegistrationConfig(
            station_search_step=motif_length / 30.0,
            minimum_heading_variation=math.radians(2.0),
            minimum_lateral_excitation=0.002,
            maximum_rms=0.01,
            ambiguity_station_separation=0.5 * motif_length,
            maximum_ambiguity_rms_difference=0.001,
        )

        result = register_curve_subset(template, observed, config)

        self.assertFalse(result.accepted)
        self.assertTrue(result.diagnostics.degenerate)
        self.assertEqual(result.diagnostics.reason, "ambiguous template station")
        self.assertLess(result.diagnostics.station_margin, 1e-3)

        bounded = register_curve_subset(
            template,
            observed,
            config,
            start_station_bounds=(0.0, 0.25 * motif_length),
        )

        self.assertTrue(bounded.accepted, bounded.diagnostics.reason)
        self.assertAlmostEqual(bounded.start_station, 0.0, delta=1e-8)
        assert_transform_close(
            self,
            bounded.transform,
            transform,
            position=0.003,
            heading=0.003,
        )
        self.assertLess(
            bounded.diagnostics.candidate_count,
            result.diagnostics.candidate_count,
        )

    def test_start_station_bounds_validate_and_reject_empty_intersection(self):
        local = interpolate_curve(self.template, np.linspace(0.55, 1.15, 81))
        observed = ObservedCurve(
            4.0,
            "odom",
            tuple(self.transform.apply_point(point) for point in local),
        )

        for bounds in ((0.2,), (math.nan, 0.8), (0.8, 0.2)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                register_curve_subset(
                    self.template,
                    observed,
                    self.config,
                    start_station_bounds=bounds,
                )

        empty = register_curve_subset(
            self.template,
            observed,
            self.config,
            start_station_bounds=(
                self.template.length + 0.1,
                self.template.length + 0.2,
            ),
        )
        self.assertFalse(empty.accepted)
        self.assertEqual(empty.diagnostics.candidate_count, 0)
        self.assertEqual(
            empty.diagnostics.reason,
            "start station bounds do not overlap the feasible template",
        )

    def test_none_start_station_bounds_preserve_global_search(self):
        local = interpolate_curve(self.template, np.linspace(0.55, 1.15, 81))
        observed = ObservedCurve(
            4.0,
            "odom",
            tuple(self.transform.apply_point(point) for point in local),
        )

        implicit = register_curve_subset(self.template, observed, self.config)
        explicit = register_curve_subset(
            self.template,
            observed,
            self.config,
            start_station_bounds=None,
        )

        self.assertEqual(implicit, explicit)

    def test_short_curved_observation_is_rejected(self):
        local = interpolate_curve(self.template, np.linspace(0.5, 0.56, 10))
        observed = ObservedCurve(
            1.0,
            "odom",
            tuple(self.transform.apply_point(point) for point in local),
        )
        result = register_curve_subset(self.template, observed, self.config)

        self.assertFalse(result.accepted)
        self.assertTrue(result.diagnostics.degenerate)
        self.assertEqual(result.diagnostics.reason, "observed curve is too short")


if __name__ == "__main__":
    unittest.main()
