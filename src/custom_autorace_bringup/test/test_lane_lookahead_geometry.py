#!/usr/bin/env python3

from copy import deepcopy
import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from custom_autorace_bringup.lane_lookahead_geometry import (
    resolve_lane_lookahead_geometry,
    validate_texture_revision,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE_DIR / "config" / "lane_lookahead_gazebo.yaml"
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


class LaneLookaheadGeometryTest(unittest.TestCase):
    def setUp(self):
        with CONFIG.open(encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)["lane_lookahead"]

    def resolved(self, length):
        config = deepcopy(self.config)
        config["geometry"]["straight"]["length_override"] = length
        return resolve_lane_lookahead_geometry(config)

    def test_reference_length_is_an_exact_identity_and_does_not_mutate_input(self):
        original = deepcopy(self.config)
        geometry = resolve_lane_lookahead_geometry(self.config)
        self.assertEqual(self.config, original)
        self.assertAlmostEqual(geometry.straight_length, 0.600, places=12)
        self.assertAlmostEqual(geometry.length_delta, 0.0, places=12)
        self.assertEqual(
            geometry.config["path"]["knots"], original["path"]["knots"]
        )
        self.assertEqual(geometry.config["finish"], original["finish"])
        self.assertAlmostEqual(geometry.path.length, 1.960435, places=5)

    def test_shorter_length_moves_the_straight_end_path_and_finish_together(self):
        geometry = self.resolved(0.450)
        knots = geometry.config["path"]["knots"]
        finish = geometry.config["finish"]
        self.assertEqual(geometry.straight_start, (0.8, -1.747))
        self.assertAlmostEqual(geometry.straight_end[0], 1.250, places=12)
        self.assertAlmostEqual(geometry.straight_end[1], -1.736941, places=12)
        self.assertAlmostEqual(knots[3][0], 1.250, places=12)
        self.assertAlmostEqual(knots[3][1], -1.736941, places=12)
        self.assertAlmostEqual(knots[4][0], 1.435046, places=12)
        self.assertAlmostEqual(knots[4][1], -1.722411, places=12)
        self.assertAlmostEqual(float(geometry.path.x[-1]), 1.348947, places=5)
        self.assertAlmostEqual(geometry.path.length, 1.810436, places=5)
        self.assertAlmostEqual(finish["x"], 1.400, places=12)
        self.assertAlmostEqual(finish["minimum_station"], 1.700001, places=5)

    def test_longer_length_within_canvas_uses_the_same_fixed_start_rule(self):
        geometry = self.resolved(0.700)
        knots = geometry.config["path"]["knots"]
        finish = geometry.config["finish"]
        self.assertEqual(geometry.straight_start, (0.8, -1.747))
        self.assertAlmostEqual(geometry.straight_end[0], 1.500, places=12)
        self.assertAlmostEqual(knots[4][0], 1.685046, places=12)
        self.assertAlmostEqual(float(geometry.path.x[-1]), 1.598947, places=5)
        self.assertAlmostEqual(finish["x"], 1.650, places=12)
        self.assertAlmostEqual(
            geometry.path.length - geometry.reference_path_length,
            0.100,
            places=4,
        )

    def test_transform_follows_a_ninety_degree_straight_heading(self):
        config = deepcopy(self.config)
        anchor_x, anchor_y = config["geometry"]["straight"]["start"]
        rotated = []
        for x, y in config["path"]["knots"]:
            dx, dy = x - anchor_x, y - anchor_y
            rotated.append([anchor_x - dy, anchor_y + dx])
        config["path"]["knots"] = rotated
        config["path"]["start_heading_deg"] += 90.0
        config["path"]["end_heading_deg"] += 90.0
        config["geometry"]["straight"]["heading_deg"] = 90.0
        config["geometry"]["straight"]["length_override"] = 0.450
        geometry = resolve_lane_lookahead_geometry(config)
        knots = geometry.config["path"]["knots"]
        self.assertAlmostEqual(knots[3][0], rotated[3][0], places=12)
        self.assertAlmostEqual(knots[3][1], anchor_y + 0.450, places=12)
        self.assertAlmostEqual(knots[4][0], rotated[4][0], places=12)
        self.assertAlmostEqual(knots[4][1], rotated[4][1] - 0.150, places=12)
        self.assertAlmostEqual(geometry.downstream_translation[0], 0.0, places=12)
        self.assertAlmostEqual(geometry.downstream_translation[1], -0.150, places=12)

    def test_invalid_lengths_fail_before_a_path_can_be_used(self):
        for value in (0.0, -0.1, float("nan"), float("inf"), "not-a-number"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.resolved(value)

    def test_adjusted_speed_profile_keeps_all_dynamic_limits(self):
        geometry = self.resolved(0.450)
        control = self.config["control"]
        curvature = np.abs(geometry.path.curvature)
        self.assertLessEqual(
            float(np.max(geometry.path.speed)), control["cruise_velocity"] + 1e-12
        )
        self.assertLessEqual(
            float(np.max(geometry.path.speed * curvature)),
            control["maximum_angular_velocity"] + 1e-12,
        )
        self.assertLessEqual(
            float(np.max(geometry.path.speed ** 2 * curvature)),
            control["maximum_lateral_acceleration"] + 1e-12,
        )
        self.assertLessEqual(float(np.max(curvature)), 4.86)

    def test_stock_texture_is_rejected_for_a_non_reference_length(self):
        geometry = self.resolved(0.590)
        texture = geometry.config["texture"]
        with self.assertRaisesRegex(ValueError, "reference course image"):
            validate_texture_revision(str(COURSE_IMAGE), texture, geometry)

        reference = self.resolved(0.600)
        digest = validate_texture_revision(
            str(COURSE_IMAGE), reference.config["texture"], reference
        )
        self.assertEqual(digest, texture["reference_sha256"])

    def test_path_benchmark_forwards_the_numeric_oracle_override(self):
        root = ET.parse(
            PACKAGE_DIR / "launch" / "gazebo_lane_path_test.launch"
        ).getroot()
        args = {element.attrib.get("name") for element in root.findall("arg")}
        self.assertIn("straight_length", args)
        self.assertIn("course_texture", args)
        matching = []
        for node in root.findall("node"):
            for param in node.findall("param"):
                if param.attrib.get("name") == (
                    "lane_lookahead/geometry/straight/length_override"
                ):
                    matching.append(param.attrib.get("value"))
        self.assertEqual(matching, ["$(arg straight_length)"])


if __name__ == "__main__":
    unittest.main()
