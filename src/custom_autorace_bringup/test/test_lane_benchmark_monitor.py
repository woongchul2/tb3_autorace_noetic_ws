#!/usr/bin/env python3

import math
import sys
import unittest
from pathlib import Path

import numpy as np


NODE_DIR = Path(__file__).resolve().parents[1] / "nodes"
if str(NODE_DIR) not in sys.path:
    sys.path.insert(0, str(NODE_DIR))

from lane_benchmark_monitor import (  # noqa: E402
    command_dynamics_metrics,
    lane_path_metrics,
    parse_lane_path_diagnostic,
    validate_finish_crossing,
)


class LaneBenchmarkMetricsTest(unittest.TestCase):
    @staticmethod
    def finish_path():
        return type(
            "FinishPath",
            (),
            {
                "x": np.asarray([1.70, 1.62, 1.55, 1.48]),
                "y": np.asarray([-0.76, -0.76, -0.76, -0.76]),
                "heading": np.asarray([math.pi] * 4),
                "curvature": np.zeros(4),
                "station": np.asarray([1.70, 1.80, 1.86, 1.94]),
            },
        )()

    def test_path_diagnostic_contract_rejects_malformed_values(self):
        valid = [
            0.4,
            0.55,
            0.02,
            -0.01,
            0.10,
            0.18,
            5.0,
            -0.20,
            0.03,
            math.inf,
            math.inf,
            0.20,
            -0.10,
        ]
        self.assertEqual(parse_lane_path_diagnostic(valid), tuple(valid))
        self.assertIsNone(parse_lane_path_diagnostic(valid[:-1]))
        self.assertIsNone(parse_lane_path_diagnostic(valid + [0.0]))

        signed_clearance = list(valid)
        signed_clearance[8:11] = [-0.001, -0.002, -0.003]
        self.assertEqual(
            parse_lane_path_diagnostic(signed_clearance),
            tuple(signed_clearance),
        )

        for index, replacement in (
            (0, -0.1),
            (0, 1.1),
            (1, -1.0),
            (2, -0.01),
            (4, math.pi + 0.01),
            (5, -0.1),
            (6, -1.0),
            (6, 1.5),
            (8, math.inf),
            (9, -math.inf),
            (10, math.nan),
            (12, math.inf),
        ):
            malformed = list(valid)
            malformed[index] = replacement
            with self.subTest(index=index, replacement=replacement):
                self.assertIsNone(parse_lane_path_diagnostic(malformed))

    def test_command_metrics_measure_rate_and_ignore_deadband_signs(self):
        samples = [
            (0.0, 0.2, 0.0),
            (0.1, 0.2, 0.1),
            (0.2, 0.2, -0.1),
            (0.3, 0.2, -0.1),
        ]
        metrics = command_dynamics_metrics(samples, 0.3, 0.02)

        self.assertAlmostEqual(
            metrics["command_angular_rms"], math.sqrt(0.03 / 4.0)
        )
        self.assertEqual(metrics["command_angular_sign_change_count"], 1)
        self.assertAlmostEqual(
            metrics["command_angular_sign_changes_per_second"], 1.0 / 0.3
        )
        self.assertAlmostEqual(
            metrics["command_angular_acceleration_abs_p95"], 1.9
        )
        self.assertAlmostEqual(
            metrics["command_angular_acceleration_abs_maximum"], 2.0
        )

    def test_path_metrics_report_common_progress_error_and_clearance(self):
        samples = [
            (
                0.2,
                0.8,
                0.02,
                -0.01,
                0.10,
                0.20,
                3.0,
                -0.20,
                0.03,
                math.inf,
                math.inf,
                0.18,
                0.10,
            ),
            (
                0.6,
                0.4,
                0.04,
                -0.03,
                -0.20,
                0.10,
                5.0,
                0.40,
                0.01,
                0.05,
                math.inf,
                -0.10,
                -0.10,
            ),
        ]
        metrics = lane_path_metrics(samples)

        self.assertEqual(metrics["lane_path_diagnostic_count"], 2)
        self.assertAlmostEqual(metrics["lane_path_progress_final"], 0.6)
        self.assertAlmostEqual(
            metrics["lane_path_remaining_distance_final"], 0.4
        )
        self.assertAlmostEqual(
            metrics["lane_path_position_error_abs_maximum"], 0.04
        )
        self.assertAlmostEqual(metrics["lane_path_curvature_abs_maximum"], 0.4)
        self.assertAlmostEqual(metrics["lane_path_target_speed_mean"], 0.15)
        self.assertAlmostEqual(metrics["lane_path_target_speed_minimum"], 0.10)
        self.assertAlmostEqual(
            metrics["lane_path_minimum_line_clearance"], 0.01
        )
        self.assertAlmostEqual(
            metrics["lane_path_minimum_obstacle_clearance"], 0.05
        )
        self.assertTrue(
            math.isinf(metrics["lane_path_minimum_map_clearance"])
        )
        self.assertAlmostEqual(
            metrics["lane_path_commanded_linear_minimum"], -0.10
        )
        self.assertAlmostEqual(
            metrics["lane_path_commanded_angular_abs_maximum"], 0.10
        )
        self.assertGreater(metrics["lane_path_heading_error_abs_p95_deg"], 0.0)
        for removed in (
            "lane_path_valid_sample_count_mean",
            "lane_path_horizon_mean",
            "lane_path_confidence_mean",
            "lane_path_odom_skew_abs_p95",
        ):
            self.assertNotIn(removed, metrics)

    def test_finish_crossing_requires_minimum_station_and_heading(self):
        path = self.finish_path()

        valid, index, station, heading_error = validate_finish_crossing(
            path,
            1.55,
            -0.76,
            math.pi,
            0,
            1.85,
            math.pi,
            math.radians(30.0),
        )

        self.assertTrue(valid)
        self.assertEqual(index, 2)
        self.assertAlmostEqual(station, 1.86)
        self.assertAlmostEqual(heading_error, 0.0)

        too_early, _, _, _ = validate_finish_crossing(
            path,
            1.62,
            -0.76,
            math.pi,
            0,
            1.85,
            math.pi,
            math.radians(30.0),
        )
        wrong_heading, _, _, heading_error = validate_finish_crossing(
            path,
            1.55,
            -0.76,
            math.radians(130.0),
            0,
            1.85,
            math.pi,
            math.radians(30.0),
        )

        self.assertFalse(too_early)
        self.assertFalse(wrong_heading)
        self.assertAlmostEqual(math.degrees(heading_error), 50.0)


if __name__ == "__main__":
    unittest.main()
