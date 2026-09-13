#!/usr/bin/env python3

import math
import unittest
from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def load_yaml(name):
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class IntersectionRouteSelectionTest(unittest.TestCase):
    def setUp(self):
        self.mission = load_yaml("intersection_mission.yaml")["mission"]
        self.zones = load_yaml("mission_zones_gazebo.yaml")

    def test_production_config_has_only_absolute_map_routes(self):
        self.assertEqual(self.mission["map_frame"], "map")
        for obsolete_key in (
            "use_zone_gate",
            "path_planner",
            "intersection_confirm_frames",
            "map_lane_nodes",
            "map_lane_edges",
            "map_entry_node",
            "map_left_goal_node",
            "map_right_goal_node",
            "map_join_candidate_count",
            "map_path_spacing",
            "exit_merge_x",
            "exit_merge_y",
            "left_exit_merge_yaw_deg",
            "right_exit_merge_yaw_deg",
            "exit_branch_start_tangent",
            "exit_branch_end_tangent",
            "exit_common_start_tangent",
            "exit_common_end_tangent",
            "exit_branch_boundary_inflation",
            "exit_relative_forward",
            "exit_relative_lateral",
            "exit_common_relative_forward",
            "exit_common_relative_lateral",
            "map_pose_topic",
            "map_odom_origin_world_x",
            "map_odom_origin_world_y",
            "map_odom_origin_world_yaw_deg",
            "map_left_entry_start_tangent",
            "map_left_entry_end_tangent",
            "map_right_entry_start_tangent",
            "map_right_entry_end_tangent",
            "entry_adaptive_join_ratio",
            "entry_adaptive_tangent_ratio",
            "entry_adaptive_connector_samples",
            "lane_topic",
            "lane_timeout",
            "arc_lane_confirm_frames",
            "arc_lane_confirmation_max_gap",
            "arc_lane_seek_linear_velocity",
            "arc_lane_seek_timeout",
            "arc_lane_seek_max_distance",
            "lane_priority_kp",
            "lane_priority_max_angular_velocity",
            "one_line_search_angular_velocity",
            "seek_one_line_search_angular_velocity",
            "angular_acceleration_limit",
            "lane_seek_heading_kp",
            "lane_seek_max_angular_velocity",
            "seek_lane_correction_max",
            "exit_heading_tolerance_deg",
            "lane_center_tolerance",
            "final_lane_seek_linear_velocity",
            "final_lane_seek_timeout",
            "final_lane_seek_max_distance",
        ):
            self.assertNotIn(obsolete_key, self.mission)

        self.assertGreater(float(self.mission["entry_start_tangent_ratio"]), 0.0)
        self.assertGreater(float(self.mission["entry_end_tangent_ratio"]), 0.0)
        self.assertGreaterEqual(
            float(self.mission["entry_max_opposite_turn_deg"]), 0.0
        )
        self.assertGreater(float(self.mission["entry_max_total_turn_deg"]), 0.0)
        self.assertLess(float(self.mission["entry_max_total_turn_deg"]), 180.0)

    def test_camera_selector_has_two_independently_measured_entry_goals(self):
        start = self.mission["map_entry_start"]
        left = self.mission["map_left_entry_goal"]
        right = self.mission["map_right_entry_goal"]

        self.assertEqual(len(start), 2)
        self.assertLess(left[1], start[1])
        self.assertGreater(right[1], start[1])
        self.assertAlmostEqual(
            float(self.mission["map_entry_start_yaw_deg"]), 180.0
        )
        self.assertAlmostEqual(
            float(self.mission["map_left_arc_entry_yaw_deg"]), -90.0
        )
        self.assertAlmostEqual(
            float(self.mission["map_right_arc_entry_yaw_deg"]), 90.0
        )
        self.assertGreaterEqual(int(self.mission["map_entry_samples"]), 20)
        self.assertNotEqual(left, right)

    def test_exit_takeover_uses_direct_map_pose_without_exit_regions(self):
        self.assertEqual(
            set(self.zones.get("regions", {})),
            {"intersection_direction_observation"},
        )
        self.assertAlmostEqual(self.zones["zone"]["signal_period"], 0.10)
        self.assertEqual(self.mission["mission_map_pose_topic"], "/mission/map_pose")
        self.assertEqual(
            self.mission["direction_observation_topic"],
            self.zones["regions"]["intersection_direction_observation"][
                "inside_topic"
            ],
        )
        self.assertGreater(float(self.mission["exit_takeover_max_distance"]), 0.0)
        self.assertNotIn("zone_left_arc_end_topic", self.mission)
        self.assertNotIn("zone_right_arc_end_topic", self.mission)
        self.assertNotIn("arc_end_confirmations", self.mission)
        self.assertNotIn("arc_end_confirmation_min_interval", self.mission)

    def test_exit_control_polygon_has_the_configured_end_tangents(self):
        points = self.mission["map_exit_control_points"]
        self.assertGreaterEqual(len(points), 4)
        start_yaw = math.atan2(
            float(points[1][1]) - float(points[0][1]),
            float(points[1][0]) - float(points[0][0]),
        )
        end_yaw = math.atan2(
            float(points[-1][1]) - float(points[-2][1]),
            float(points[-1][0]) - float(points[-2][0]),
        )
        self.assertAlmostEqual(abs(start_yaw), math.pi, places=6)
        self.assertAlmostEqual(
            end_yaw,
            math.radians(float(self.mission["exit_goal_yaw_deg"])),
            places=6,
        )

    def test_final_exit_keeps_speed_through_lane_handoff(self):
        exit_velocity = float(self.mission["exit_path_linear_velocity"])
        self.assertAlmostEqual(
            float(self.mission["exit_path_min_velocity"]), exit_velocity
        )
        self.assertAlmostEqual(
            float(self.mission["final_lane_join_velocity"]),
            exit_velocity,
        )

    def test_arc_lane_uses_its_dedicated_speed_cap(self):
        self.assertAlmostEqual(
            float(self.mission["arc_lane_max_velocity"]), 0.12
        )
        self.assertLess(
            float(self.mission["arc_lane_max_velocity"]),
            float(self.mission["lane_resume_max_velocity"]),
        )


if __name__ == "__main__":
    unittest.main()
