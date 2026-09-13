#!/usr/bin/env python3

import math
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import yaml

from custom_autorace_bringup.lane_lookahead_geometry import (
    resolve_lane_lookahead_geometry,
)
from custom_autorace_bringup.raster_route import RasterRouteCorridorChecker
from custom_autorace_bringup.zigzag_path import normalize_angle


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


class LaneLookaheadPathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONFIG.open(encoding="utf-8") as stream:
            cls.config = yaml.safe_load(stream)["lane_lookahead"]
        geometry = resolve_lane_lookahead_geometry(cls.config)
        cls.config = geometry.config
        cls.path = geometry.path
        footprint = cls.config["footprint"]
        texture = cls.config["texture"]
        course_rgb = np.asarray(Image.open(COURSE_IMAGE).convert("RGB"))
        cls.checker = RasterRouteCorridorChecker(
            course_rgb,
            texture["course_size"],
            texture["course_yaw"],
            cls.path,
            footprint["front"],
            footprint["rear"],
            footprint["half_width"],
            texture["expected_boundary_offset"],
            texture["line_search_half_width"],
            texture["boundary_step"],
            footprint["validation_sample_spacing"],
            texture["color_threshold"],
            texture["color_tolerance"],
            texture["yellow_blue_maximum"],
            texture["projection_search_distance"],
            texture["endpoint_extension"],
        )

    def test_surveyed_path_reaches_the_eroded_intersection_gate(self):
        finish = self.config["finish"]
        self.assertAlmostEqual(self.path.length, 1.96043, places=4)
        self.assertLessEqual(float(np.max(np.abs(self.path.curvature))), 4.85)
        self.assertLess(float(self.path.x[-1]), finish["x"])
        self.assertGreater(self.path.length, finish["minimum_station"])
        self.assertLess(
            abs(normalize_angle(float(self.path.heading[-1]) - math.pi)),
            math.radians(3.0),
        )

    def test_nominal_rectangular_footprint_stays_before_outer_paint_edges(self):
        safety = self.config["safety"]
        inner, outer = self.checker.path_metrics()
        self.assertLessEqual(
            inner, safety["maximum_nominal_inner_intrusion"]
        )
        self.assertGreaterEqual(
            outer, safety["minimum_nominal_outer_reserve"]
        )

    def test_curvature_profile_brakes_before_bends(self):
        control = self.config["control"]
        curvature = np.abs(self.path.curvature)
        self.assertLessEqual(
            float(np.max(self.path.speed)), control["cruise_velocity"] + 1e-12
        )
        self.assertLessEqual(
            float(np.max(self.path.speed * curvature)),
            control["maximum_angular_velocity"] + 1e-12,
        )
        self.assertLessEqual(
            float(np.max(self.path.speed ** 2 * curvature)),
            control["maximum_lateral_acceleration"] + 1e-12,
        )
        peak = int(np.argmax(curvature))
        before = max(0, peak - 30)
        self.assertLess(self.path.speed[before], control["cruise_velocity"])
        planned_time = float(
            np.sum(
                2.0
                * np.diff(self.path.station)
                / (self.path.speed[:-1] + self.path.speed[1:])
            )
        )
        self.assertLess(planned_time, 10.0)

if __name__ == "__main__":
    unittest.main()
