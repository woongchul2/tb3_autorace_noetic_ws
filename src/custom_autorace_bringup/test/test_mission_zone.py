#!/usr/bin/env python3

from pathlib import Path
import unittest
from unittest import mock

import yaml

import custom_autorace_bringup.mission_zone as mission_zone
from custom_autorace_bringup.mission_zone import (
    MissionZoneSequence,
    is_inside_with_margin,
    point_in_polygon,
    signed_polygon_distance,
    signed_polygon_distances,
)


SQUARE = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]


class PolygonTest(unittest.TestCase):
    def test_inside_boundary_and_outside(self):
        self.assertTrue(point_in_polygon((1.0, 1.0), SQUARE))
        self.assertTrue(point_in_polygon((0.0, 1.0), SQUARE))
        self.assertFalse(point_in_polygon((2.1, 1.0), SQUARE))
        self.assertAlmostEqual(signed_polygon_distance((1.0, 1.0), SQUARE), 1.0)
        self.assertAlmostEqual(signed_polygon_distance((2.2, 1.0), SQUARE), -0.2)

    def test_named_polygons_are_each_evaluated_once(self):
        translated_square = [
            (3.0, 0.0),
            (5.0, 0.0),
            (5.0, 2.0),
            (3.0, 2.0),
        ]
        entries = [
            (("mission", "intersection"), SQUARE),
            (("region", "intersection"), translated_square),
        ]
        with mock.patch.object(
            mission_zone,
            "signed_polygon_distance",
            wraps=signed_polygon_distance,
        ) as evaluator:
            distances = signed_polygon_distances((1.0, 1.0), entries)
        self.assertEqual(evaluator.call_count, 2)
        self.assertAlmostEqual(distances[("mission", "intersection")], 1.0)
        self.assertAlmostEqual(distances[("region", "intersection")], -2.0)

    def test_inside_margin_requires_depth_inside_polygon(self):
        self.assertTrue(is_inside_with_margin(0.0))
        self.assertTrue(is_inside_with_margin(0.25, 0.20))
        self.assertFalse(is_inside_with_margin(0.25, 0.30))
        self.assertFalse(is_inside_with_margin(-0.01, 0.0))
        self.assertTrue(is_inside_with_margin(0.0, -1.0))

    def test_zigzag_gate_contains_both_recorded_parking_handoffs(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "mission_zones_gazebo.yaml"
        )
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        polygon = config["missions"]["zigzag"]["polygon"]
        margin = config["zone"]["enter_margin"]
        # Official-start runs selected opposite parking bays and produced
        # opposite AMCL x corrections. Both must activate zigzag immediately.
        for pose in ((0.4895, 1.9306), (0.0733, 1.9628)):
            distance = signed_polygon_distance(pose, polygon)
            self.assertTrue(is_inside_with_margin(distance, margin))

    def test_intersection_gate_and_upstream_direction_window_are_separate(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "mission_zones_gazebo.yaml"
        )
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        mission = config["missions"]["intersection"]
        polygon = mission["polygon"]
        margin = config["zone"]["enter_margin"]
        observation = config["regions"]["intersection_direction_observation"]
        observation_polygon = observation["polygon"]
        observation_margin = observation["inside_margin"]

        # While travelling west, the independent direction window opens first.
        # The mission gate and the controller's map-entry plane remain later.
        self.assertTrue(
            is_inside_with_margin(
                signed_polygon_distance((1.68, -0.72), observation_polygon),
                observation_margin,
            )
        )
        self.assertFalse(
            is_inside_with_margin(
                signed_polygon_distance((1.68, -0.72), polygon), margin
            )
        )
        self.assertFalse(
            is_inside_with_margin(
                signed_polygon_distance((1.70, -0.72), observation_polygon),
                observation_margin,
            )
        )
        self.assertTrue(
            is_inside_with_margin(
                signed_polygon_distance((1.60, -0.72), polygon), margin
            )
        )
        self.assertNotEqual(mission["inside_topic"], observation["inside_topic"])

    def test_obstacle_gate_delays_speed_cap_until_recorded_turn_approach(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "mission_zones_gazebo.yaml"
        )
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        polygon = config["missions"]["obstacle"]["polygon"]
        margin = config["zone"]["enter_margin"]

        # The previous west edge enabled the 0.09 m/s acquisition cap around
        # this straight pose.  Keep normal lane speed here.
        self.assertFalse(
            is_inside_with_margin(
                signed_polygon_distance((1.20, 0.246), polygon), margin
            )
        )
        # The official-start bag first turned near x=1.36.  Arm about 50 mm
        # beforehand so the existing lane controller decelerates for the bend.
        self.assertTrue(
            is_inside_with_margin(
                signed_polygon_distance((1.32, 0.247), polygon), margin
            )
        )

    def test_level_crossing_gate_covers_bar_trigger_not_static_stop_sign(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "mission_zones_gazebo.yaml"
        )
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(
            config["sequence"][-2:], ["level_crossing", "tunnel"]
        )
        polygon = config["missions"]["level_crossing"]["polygon"]
        margin = config["zone"]["enter_margin"]

        # The upstream simulator lowers the bar around this odometry pose.
        self.assertTrue(
            is_inside_with_margin(
                signed_polygon_distance((-1.40, 1.25), polygon), margin
            )
        )
        # Its permanent warning sign must not open the physical-zone gate.
        self.assertFalse(point_in_polygon((-1.35, 1.04), polygon))

    def test_tunnel_gate_is_only_at_the_surveyed_north_west_portal(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "mission_zones_gazebo.yaml"
        )
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        polygon = config["missions"]["tunnel"]["polygon"]
        margin = config["zone"]["enter_margin"]

        # The official-start integration previously opened here, above the
        # solid north wall, and forced Hybrid A* into a 4.2 m approach detour.
        self.assertFalse(point_in_polygon((-0.3061, 0.3690), polygon))
        # The tunnel-only start is centred on the measured north-west portal.
        self.assertTrue(
            is_inside_with_margin(
                signed_polygon_distance((-1.75, 0.18), polygon), margin
            )
        )

class SequenceTest(unittest.TestCase):
    def setUp(self):
        self.sequence = MissionZoneSequence(
            [
                {
                    "name": "intersection",
                },
                {
                    "name": "obstacle",
                },
            ],
            enter_margin=0.1,
        )

    def test_order_gate_and_external_completion(self):
        self.sequence.update_signed_distance(-1.0)
        self.assertEqual(self.sequence.state, MissionZoneSequence.SEEKING)
        self.sequence.update_signed_distance(0.5)
        self.assertEqual(self.sequence.state, MissionZoneSequence.ACTIVE)
        self.assertFalse(self.sequence.complete_current("obstacle"))
        self.assertTrue(self.sequence.complete_current("intersection"))
        self.assertEqual(self.sequence.current_name, "obstacle")

    def test_precomputed_signed_distance_uses_same_sequence_path(self):
        self.sequence.update_signed_distance(0.5)
        self.assertEqual(self.sequence.state, MissionZoneSequence.ACTIVE)
        self.assertEqual(self.sequence.current_name, "intersection")

    def test_every_mission_requires_its_controller_completion(self):
        self.sequence.update_signed_distance(0.5)
        self.sequence.complete_current("intersection")
        self.sequence.update_signed_distance(0.5)
        self.assertEqual(self.sequence.state, MissionZoneSequence.ACTIVE)
        self.assertEqual(self.sequence.current_name, "obstacle")
        self.sequence.update_signed_distance(-1.0)
        self.assertEqual(self.sequence.state, MissionZoneSequence.ACTIVE)
        self.sequence.complete_current("obstacle")
        self.assertEqual(self.sequence.state, MissionZoneSequence.COMPLETE)


if __name__ == "__main__":
    unittest.main()
