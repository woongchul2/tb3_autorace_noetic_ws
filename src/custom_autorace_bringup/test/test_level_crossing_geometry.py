#!/usr/bin/env python3

from dataclasses import FrozenInstanceError
import math
from pathlib import Path
import sys
import unittest


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PYTHON_DIR = PACKAGE_DIR / "src"
if str(PACKAGE_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PYTHON_DIR))

from custom_autorace_bringup.level_crossing import (
    HorizontalBarrierConfig,
    detect_horizontal_barrier,
)


ANGLE_INCREMENT = math.radians(1.0)
ANGLE_MIN = -math.pi
BEAM_COUNT = 361


def scan_index(angle):
    return int(round((angle - ANGLE_MIN) / ANGLE_INCREMENT))


def empty_scan():
    return [float("inf")] * BEAM_COUNT


def add_transverse_segment(ranges, x, minimum_angle_deg, maximum_angle_deg):
    first = scan_index(math.radians(minimum_angle_deg))
    last = scan_index(math.radians(maximum_angle_deg))
    for index in range(first, last + 1):
        angle = ANGLE_MIN + index * ANGLE_INCREMENT
        ranges[index] = x / math.cos(angle)
    return first, last


def add_diagonal_segment(
    ranges, intercept, slope, minimum_angle_deg, maximum_angle_deg
):
    """Add the intersections of rays with x = intercept + slope * y."""
    first = scan_index(math.radians(minimum_angle_deg))
    last = scan_index(math.radians(maximum_angle_deg))
    for index in range(first, last + 1):
        angle = ANGLE_MIN + index * ANGLE_INCREMENT
        ranges[index] = intercept / (
            math.cos(angle) - slope * math.sin(angle)
        )
    return first, last


class LevelCrossingGeometryTest(unittest.TestCase):
    def setUp(self):
        self.config = HorizontalBarrierConfig(
            min_forward_distance=0.20,
            max_forward_distance=1.20,
            half_width=0.70,
            max_adjacent_beam_gap=2,
            max_point_gap=0.050,
            min_points=8,
            min_lateral_span=0.25,
            max_depth_spread=0.060,
        )

    def detect(self, ranges, config=None, range_min=0.10, range_max=3.50):
        return detect_horizontal_barrier(
            ranges,
            ANGLE_MIN,
            ANGLE_INCREMENT,
            range_min,
            range_max,
            self.config if config is None else config,
        )

    def test_scan_angles_come_from_metadata_not_index_zero(self):
        ranges = empty_scan()
        first, last = add_transverse_segment(ranges, 0.60, -18, 18)

        detection = self.detect(ranges)

        self.assertIsNotNone(detection)
        self.assertEqual(detection.first_beam_index, first)
        self.assertEqual(detection.last_beam_index, last)
        self.assertAlmostEqual(detection.forward_distance, 0.60, places=12)
        self.assertAlmostEqual(detection.lateral_center, 0.0, places=12)

    def test_nan_inf_and_sensor_range_violations_are_rejected(self):
        ranges = empty_scan()
        first, last = add_transverse_segment(ranges, 0.60, -18, 18)
        invalid_values = {
            first + 5: float("nan"),
            first + 11: float("inf"),
            # Both finite values are inside the Cartesian detection corridor,
            # so only the LaserScan range bounds can reject them.
            first + 17: 0.30,
            first + 23: 1.00,
        }
        for index, value in invalid_values.items():
            ranges[index] = value

        detection = self.detect(ranges, range_min=0.40, range_max=0.90)

        self.assertIsNotNone(detection)
        self.assertEqual(
            detection.point_count, last - first + 1 - len(invalid_values)
        )
        invalid_only = empty_scan()
        for offset, value in enumerate(invalid_values.values()):
            invalid_only[scan_index(-0.02) + offset] = value
        self.assertIsNone(
            self.detect(invalid_only, range_min=0.40, range_max=0.90)
        )

    def test_narrow_object_is_not_a_lowered_bar(self):
        ranges = empty_scan()
        add_transverse_segment(ranges, 0.60, -3, 3)

        self.assertIsNone(self.detect(ranges))

    def test_depth_long_diagonal_structure_is_not_a_lowered_bar(self):
        ranges = empty_scan()
        add_diagonal_segment(ranges, 0.60, 0.45, -18, 18)

        self.assertIsNone(self.detect(ranges))

    def test_horizontal_bar_is_detected_and_nearest_cluster_wins(self):
        ranges = empty_scan()
        near_first, near_last = add_transverse_segment(
            ranges, 0.45, -48, -18
        )
        add_transverse_segment(ranges, 0.82, 10, 34)

        detection = self.detect(ranges)

        self.assertIsNotNone(detection)
        self.assertAlmostEqual(detection.forward_distance, 0.45, places=12)
        self.assertEqual(detection.first_beam_index, near_first)
        self.assertEqual(detection.last_beam_index, near_last)
        self.assertGreaterEqual(detection.lateral_span, 0.25)
        self.assertLessEqual(detection.depth_spread, 1e-12)

    def test_beam_gap_at_limit_is_bridged(self):
        ranges = empty_scan()
        add_transverse_segment(ranges, 0.60, -15, 15)
        ranges[scan_index(0.0)] = float("inf")

        detection = self.detect(ranges)

        self.assertIsNotNone(detection)
        self.assertEqual(detection.point_count, 30)

    def test_beam_gap_above_limit_splits_and_rejects_both_halves(self):
        ranges = empty_scan()
        add_transverse_segment(ranges, 0.60, -15, 15)
        ranges[scan_index(math.radians(-1.0))] = float("inf")
        ranges[scan_index(0.0)] = float("inf")

        self.assertIsNone(self.detect(ranges))

    def test_config_and_result_are_immutable(self):
        ranges = empty_scan()
        add_transverse_segment(ranges, 0.60, -18, 18)
        detection = self.detect(ranges)

        with self.assertRaises(FrozenInstanceError):
            self.config.half_width = 0.8
        with self.assertRaises(FrozenInstanceError):
            detection.forward_distance = 0.1


if __name__ == "__main__":
    unittest.main()
