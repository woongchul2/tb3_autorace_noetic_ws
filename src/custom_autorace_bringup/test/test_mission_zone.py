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

    def test_zigzag_diagnostic_polygon_contains_recorded_handoffs(self):
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

    def test_intersection_diagnostic_polygon_keeps_measured_boundary(self):
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
        self.assertEqual(config.get("regions", {}), {})
        self.assertAlmostEqual(max(point[0] for point in polygon), 1.72)

        # Polygon depth remains available for map diagnostics, but it no
        # longer opens the mission enable gate.
        self.assertFalse(
            is_inside_with_margin(
                signed_polygon_distance((1.70, -0.72), polygon), margin
            )
        )
        self.assertTrue(
            is_inside_with_margin(
                signed_polygon_distance((1.68, -0.72), polygon), margin
            )
        )
        self.assertTrue(
            is_inside_with_margin(
                signed_polygon_distance((1.60, -0.72), polygon), margin
            )
        )

    def test_obstacle_diagnostic_polygon_covers_recorded_turn_approach(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "mission_zones_gazebo.yaml"
        )
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        polygon = config["missions"]["obstacle"]["polygon"]
        margin = config["zone"]["enter_margin"]

        # The diagnostic boundary excludes the upstream straight pose.
        self.assertFalse(
            is_inside_with_margin(
                signed_polygon_distance((1.20, 0.246), polygon), margin
            )
        )
        # The official-start bag first turned near x=1.36.
        self.assertTrue(
            is_inside_with_margin(
                signed_polygon_distance((1.32, 0.247), polygon), margin
            )
        )

    def test_level_crossing_diagnostic_polygon_covers_bar_not_stop_sign(self):
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
        # Its permanent warning sign lies outside this diagnostic polygon.
        self.assertFalse(point_in_polygon((-1.35, 1.04), polygon))

    def test_tunnel_diagnostic_polygon_is_at_north_west_portal(self):
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

    def test_polygon_activation_is_explicitly_diagnostic_only(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "mission_zones_gazebo.yaml"
        )
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(config["activation"]["mode"], "ready")
        self.assertTrue(config["diagnostics"]["polygons_enabled"])
        for name in config["sequence"]:
            mission = config["missions"][name]
            self.assertEqual(mission["arm_topic"], "/mission/arm/" + name)
            self.assertEqual(mission["ready_topic"], "/mission/ready/" + name)

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
            armed_at=10.0,
            initial_generation=41,
        )

    def _ready_and_activate(self, stamp=10.1, received=10.2):
        self.assertTrue(
            self.sequence.mark_ready(
                self.sequence.current_name,
                self.sequence.generation,
                stamp,
                received,
                maximum_age=0.5,
                future_tolerance=0.05,
            )
        )
        self.assertEqual(self.sequence.state, MissionZoneSequence.READY)
        self.assertTrue(self.sequence.activate_current(received))
        self.assertEqual(self.sequence.state, MissionZoneSequence.ACTIVE)

    def test_first_mission_is_armed_without_any_pose_update(self):
        self.assertEqual(self.sequence.state, MissionZoneSequence.ARMED)
        self.assertEqual(self.sequence.current_name, "intersection")
        self.assertEqual(self.sequence.generation, 41)
        self.assertAlmostEqual(self.sequence.armed_at, 10.0)

    def test_only_ready_can_open_active_state(self):
        self.assertFalse(self.sequence.activate_current(10.1))
        self.assertEqual(self.sequence.state, MissionZoneSequence.ARMED)
        self._ready_and_activate()

    def test_out_of_order_readiness_is_rejected(self):
        self.assertIsNotNone(
            self.sequence.readiness_problem(
                "obstacle", 41, 10.1, 10.2, maximum_age=0.5
            )
        )
        self.assertFalse(
            self.sequence.mark_ready(
                "obstacle", 41, 10.1, 10.2, maximum_age=0.5
            )
        )
        self.assertEqual(self.sequence.state, MissionZoneSequence.ARMED)

    def test_wrong_generation_and_pre_arm_readiness_are_rejected(self):
        self.assertFalse(
            self.sequence.mark_ready(
                "intersection", 40, 10.1, 10.2, maximum_age=0.5
            )
        )
        self.assertFalse(
            self.sequence.mark_ready(
                "intersection", 41, 9.99, 10.0, maximum_age=0.5
            )
        )
        self.assertEqual(self.sequence.state, MissionZoneSequence.ARMED)

    def test_stale_and_future_readiness_are_rejected(self):
        self.assertFalse(
            self.sequence.mark_ready(
                "intersection", 41, 10.1, 10.7, maximum_age=0.5
            )
        )
        self.assertFalse(
            self.sequence.mark_ready(
                "intersection",
                41,
                10.3,
                10.2,
                maximum_age=0.5,
                future_tolerance=0.05,
            )
        )
        self.assertEqual(self.sequence.state, MissionZoneSequence.ARMED)

    def test_external_completion_arms_next_immediately(self):
        self._ready_and_activate()
        self.assertFalse(self.sequence.complete_current("obstacle", 10.3))
        self.assertTrue(self.sequence.complete_current("intersection", 11.0))
        self.assertEqual(self.sequence.current_name, "obstacle")
        self.assertEqual(self.sequence.state, MissionZoneSequence.ARMED)
        self.assertAlmostEqual(self.sequence.armed_at, 11.0)
        self.assertEqual(self.sequence.generation, 42)
        # The previous generation cannot activate the newly armed mission.
        self.assertFalse(
            self.sequence.mark_ready(
                "obstacle", 41, 11.1, 11.2, maximum_age=0.5
            )
        )

    def test_every_mission_requires_its_controller_completion(self):
        self._ready_and_activate()
        self.assertTrue(self.sequence.complete_current("intersection", 11.0))
        self.assertFalse(self.sequence.complete_current("obstacle", 11.1))
        self._ready_and_activate(stamp=11.1, received=11.2)
        self.assertTrue(self.sequence.complete_current("obstacle", 12.0))
        self.assertEqual(self.sequence.state, MissionZoneSequence.COMPLETE)
        self.assertEqual(self.sequence.completed, ["intersection", "obstacle"])


if __name__ == "__main__":
    unittest.main()
