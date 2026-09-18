#!/usr/bin/env python3

from collections import deque
import math
from pathlib import Path
import sys
import unittest

import yaml


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PYTHON_DIR = PACKAGE_DIR / "src"
if str(PACKAGE_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PYTHON_DIR))

from custom_autorace_bringup.tunnel_registration import (
    PortalRegistrationConfig,
    detect_portal_from_indexed_points,
)
from custom_autorace_bringup.parking_geometry import (
    map_from_odom_transform,
    map_pose_to_odom,
)


class TunnelRegistrationTest(unittest.TestCase):
    def setUp(self):
        self.config = PortalRegistrationConfig(
            minimum_forward_distance=0.10,
            maximum_forward_distance=2.0,
            minimum_absolute_lateral=0.04,
            maximum_absolute_lateral=0.50,
            maximum_adjacent_beam_gap=2,
            maximum_point_gap=0.20,
            minimum_wall_points=6,
            minimum_wall_length=0.25,
            maximum_wall_residual=0.01,
            maximum_heading_deviation=math.radians(35.0),
            maximum_orthogonal_angle=math.radians(5.0),
            expected_portal_width=0.286,
            portal_width_tolerance=0.03,
            maximum_corner_skew=0.04,
        )

    @staticmethod
    def portal_points(
        center_x=0.50,
        center_y=0.0,
        yaw=0.0,
        width=0.286,
        longitudinal_start_offset=0.0,
    ):
        forward = (math.cos(yaw), math.sin(yaw))
        left = (-forward[1], forward[0])
        points = []
        right_corner = (
            center_x
            - 0.5 * width * left[0]
            + longitudinal_start_offset * forward[0],
            center_y
            - 0.5 * width * left[1]
            + longitudinal_start_offset * forward[1],
        )
        for offset in range(9):
            station = 0.05 * offset
            points.append(
                (
                    110 + offset,
                    right_corner[0] + station * forward[0],
                    right_corner[1] + station * forward[1],
                )
            )
        left_corner = (
            center_x + 0.5 * width * left[0],
            center_y + 0.5 * width * left[1],
        )
        for offset in range(9):
            station = 0.05 * offset
            points.append(
                (
                    210 + offset,
                    left_corner[0] + station * left[0],
                    left_corner[1] + station * left[1],
                )
            )
        return points

    @staticmethod
    def independently_biased_portal_points(
        center_x=0.50,
        center_y=0.0,
        width=0.286,
        longitudinal_yaw=math.radians(4.0),
        transverse_longitudinal_yaw=math.radians(-4.0),
    ):
        """Return two exact walls whose independent axes straddle zero."""
        along = (
            math.cos(longitudinal_yaw),
            math.sin(longitudinal_yaw),
        )
        transverse_yaw = transverse_longitudinal_yaw + 0.5 * math.pi
        across = (
            math.cos(transverse_yaw),
            math.sin(transverse_yaw),
        )
        right_corner = (center_x, center_y - 0.5 * width)
        left_corner = (center_x, center_y + 0.5 * width)
        points = []
        for offset in range(9):
            station = 0.05 * offset
            points.append(
                (
                    110 + offset,
                    right_corner[0] + station * along[0],
                    right_corner[1] + station * along[1],
                )
            )
        for offset in range(9):
            station = 0.05 * offset
            points.append(
                (
                    210 + offset,
                    left_corner[0] + station * across[0],
                    left_corner[1] + station * across[1],
                )
            )
        return points

    @staticmethod
    def gazebo_registration_config():
        with (PACKAGE_DIR / "config" / "tunnel_mission_gazebo.yaml").open(
            encoding="utf-8"
        ) as stream:
            registration = yaml.safe_load(stream)["tunnel"]["registration"]
        return PortalRegistrationConfig(
            minimum_forward_distance=registration[
                "minimum_forward_distance"
            ],
            maximum_forward_distance=registration[
                "maximum_forward_distance"
            ],
            minimum_absolute_lateral=registration[
                "minimum_absolute_lateral"
            ],
            maximum_absolute_lateral=registration[
                "maximum_absolute_lateral"
            ],
            maximum_adjacent_beam_gap=registration[
                "maximum_adjacent_beam_gap"
            ],
            maximum_point_gap=registration["maximum_point_gap"],
            minimum_wall_points=registration["minimum_wall_points"],
            minimum_wall_length=registration["minimum_wall_length"],
            maximum_wall_residual=registration["maximum_wall_residual"],
            maximum_heading_deviation=math.radians(
                registration["maximum_heading_deviation_deg"]
            ),
            maximum_orthogonal_angle=math.radians(
                registration["maximum_orthogonal_angle_deg"]
            ),
            expected_portal_width=registration["expected_portal_width"],
            portal_width_tolerance=registration["portal_width_tolerance"],
            maximum_corner_skew=registration["maximum_corner_skew"],
        )

    def test_two_walls_recover_portal_frame_and_longitudinal_shift(self):
        for shift in (0.0, 0.73):
            detection = detect_portal_from_indexed_points(
                self.portal_points(center_x=0.50 + shift, yaw=0.08),
                self.config,
            )
            self.assertIsNotNone(detection)
            self.assertAlmostEqual(detection.center_x, 0.50 + shift, places=6)
            self.assertAlmostEqual(detection.center_y, 0.0, places=6)
            self.assertAlmostEqual(detection.yaw, 0.08, places=6)
            self.assertAlmostEqual(detection.width, 0.286, places=6)

    def test_oppositely_biased_wall_axes_fuse_to_midpoint_heading(self):
        detection = detect_portal_from_indexed_points(
            self.independently_biased_portal_points(),
            self.config,
        )

        self.assertIsNotNone(detection)
        self.assertAlmostEqual(
            detection.longitudinal_wall.yaw,
            math.radians(4.0),
            places=6,
        )
        self.assertAlmostEqual(
            detection.transverse_wall.yaw,
            math.radians(86.0),
            places=6,
        )
        self.assertAlmostEqual(detection.yaw, 0.0, places=6)

    def test_fused_yaw_does_not_change_endpoint_center_or_skew(self):
        detection = detect_portal_from_indexed_points(
            self.independently_biased_portal_points(),
            self.config,
        )

        self.assertIsNotNone(detection)
        along_yaw = detection.longitudinal_wall.yaw
        if math.cos(along_yaw) < 0.0:
            along_yaw = (along_yaw + math.pi) % (2.0 * math.pi) - math.pi
        forward = (math.cos(along_yaw), math.sin(along_yaw))
        left = (-forward[1], forward[0])
        along_endpoints = (
            (
                detection.longitudinal_wall.corner_x,
                detection.longitudinal_wall.corner_y,
            ),
            (
                detection.longitudinal_wall.far_x,
                detection.longitudinal_wall.far_y,
            ),
        )
        along_corner = min(
            along_endpoints,
            key=lambda point: (
                point[0] * forward[0] + point[1] * forward[1]
            ),
        )
        across_endpoints = (
            (
                detection.transverse_wall.corner_x,
                detection.transverse_wall.corner_y,
            ),
            (
                detection.transverse_wall.far_x,
                detection.transverse_wall.far_y,
            ),
        )
        across_corner = min(across_endpoints, key=lambda point: abs(point[1]))
        dx = across_corner[0] - along_corner[0]
        dy = across_corner[1] - along_corner[1]

        self.assertAlmostEqual(
            detection.center_x,
            0.5 * (along_corner[0] + across_corner[0]),
            places=9,
        )
        self.assertAlmostEqual(
            detection.center_y,
            0.5 * (along_corner[1] + across_corner[1]),
            places=9,
        )
        self.assertAlmostEqual(
            detection.width,
            abs(dx * left[0] + dy * left[1]),
            places=9,
        )
        self.assertAlmostEqual(
            detection.corner_skew,
            abs(dx * forward[0] + dy * forward[1]),
            places=9,
        )

    def test_single_parallel_wall_cannot_observe_longitudinal_origin(self):
        only_left = [
            point for point in self.portal_points() if point[0] < 200
        ]
        self.assertIsNone(
            detect_portal_from_indexed_points(only_left, self.config)
        )

    def test_misaligned_corners_are_rejected_even_with_two_wall_directions(self):
        points = self.portal_points()
        shifted_right = [
            (index, x + (0.08 if index >= 200 else 0.0), y)
            for index, x, y in points
        ]
        self.assertIsNone(
            detect_portal_from_indexed_points(shifted_right, self.config)
        )

    def test_run17_visibility_skew_detects_and_confirms_three_samples(self):
        config = self.gazebo_registration_config()
        self.assertEqual(config.maximum_corner_skew, 0.10)
        candidates = deque(maxlen=3)
        for sample, sensor_x in enumerate((0.00, 0.02, 0.04)):
            detection = detect_portal_from_indexed_points(
                self.portal_points(
                    center_x=0.50 - sensor_x,
                    width=0.316,
                    longitudinal_start_offset=0.095,
                ),
                config,
            )
            self.assertIsNotNone(detection)
            self.assertAlmostEqual(detection.corner_skew, 0.095, places=6)
            portal_in_odom = (
                sensor_x + detection.center_x,
                detection.center_y,
                detection.yaw,
            )
            candidate = map_from_odom_transform(
                (-1.7475895, -0.105857, -0.5 * math.pi),
                portal_in_odom,
            )
            if candidates:
                previous = candidates[-1]
                self.assertLessEqual(
                    math.hypot(
                        candidate[0] - previous[0],
                        candidate[1] - previous[1],
                    ),
                    0.04,
                )
                heading_delta = (candidate[2] - previous[2] + math.pi) % (
                    2.0 * math.pi
                ) - math.pi
                self.assertLessEqual(abs(heading_delta), math.radians(4.0))
            candidates.append(candidate)
            self.assertEqual(len(candidates), sample + 1)
        self.assertEqual(len(candidates), 3)

    def test_corner_skew_beyond_run17_bound_is_rejected(self):
        config = self.gazebo_registration_config()
        detection = detect_portal_from_indexed_points(
            self.portal_points(
                width=0.316,
                longitudinal_start_offset=0.12,
            ),
            config,
        )
        self.assertIsNone(detection)

    def test_longitudinal_connector_shift_moves_every_template_pose_once(self):
        template_portal = (-1.757968, -0.041389, math.radians(-89.4336))
        template_exit = (0.20, -1.748373, 0.0)
        actual_a = (3.0, 2.0, -0.5 * math.pi)
        shift = 0.73
        actual_b = (
            actual_a[0] + shift * math.cos(actual_a[2]),
            actual_a[1] + shift * math.sin(actual_a[2]),
            actual_a[2],
        )
        transform_a = map_from_odom_transform(template_portal, actual_a)
        transform_b = map_from_odom_transform(template_portal, actual_b)
        exit_a = map_pose_to_odom(template_exit, transform_a)
        exit_b = map_pose_to_odom(template_exit, transform_b)

        self.assertAlmostEqual(
            exit_b[0] - exit_a[0], actual_b[0] - actual_a[0], places=9
        )
        self.assertAlmostEqual(
            exit_b[1] - exit_a[1], actual_b[1] - actual_a[1], places=9
        )
        self.assertAlmostEqual(exit_b[2], exit_a[2], places=9)

    def test_registration_feature_is_not_the_entry_clearance_plane(self):
        with (PACKAGE_DIR / "config" / "tunnel_mission_gazebo.yaml").open(
            encoding="utf-8"
        ) as stream:
            tunnel = yaml.safe_load(stream)["tunnel"]

        reference = tunnel["registration"]["reference_pose"]
        portal_plane_y = tunnel["entry"]["portal_plane_y"]
        self.assertAlmostEqual(reference[0], -1.757968, places=6)
        self.assertAlmostEqual(reference[1], -0.041389, places=6)
        self.assertAlmostEqual(reference[2], -89.4336, places=4)
        self.assertGreater(abs(reference[1] - portal_plane_y), 0.06)


if __name__ == "__main__":
    unittest.main()
