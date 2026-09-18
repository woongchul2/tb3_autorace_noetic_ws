#!/usr/bin/env python3

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
import yaml


LAUNCH_FILE = Path(__file__).resolve().parents[1] / "launch" / "gazebo.launch"
PACKAGE_DIR = Path(__file__).resolve().parents[1]
DESCRIPTION_GAZEBO_LAUNCH = (
    PACKAGE_DIR.parent
    / "custom_autorace_description"
    / "launch"
    / "gazebo_autorace.launch"
)
LANE_PATH_LAUNCH = PACKAGE_DIR / "launch" / "gazebo_lane_path_test.launch"
LANE_CONFIG = PACKAGE_DIR / "config" / "lane_controller.yaml"
MISSION_ZONES = PACKAGE_DIR / "config" / "mission_zones_gazebo.yaml"
MISSION_CONFIGS = {
    "intersection": PACKAGE_DIR / "config" / "intersection_mission.yaml",
    "obstacle": PACKAGE_DIR / "config" / "obstacle_mission_gazebo.yaml",
    "parking": PACKAGE_DIR / "config" / "parking_mission_gazebo.yaml",
    "zigzag": PACKAGE_DIR / "config" / "zigzag_mission_gazebo.yaml",
}
ORDERED_MISSION_CONFIGS = {
    **MISSION_CONFIGS,
    "level_crossing": (
        PACKAGE_DIR / "config" / "level_crossing_mission_gazebo.yaml"
    ),
    "tunnel": PACKAGE_DIR / "config" / "tunnel_mission_gazebo.yaml",
}
MISSION_NODE_NAMES = {
    "intersection": "intersection_mission_controller",
    "obstacle": "obstacle_mission_controller",
    "parking": "parking_mission_controller",
    "zigzag": "zigzag_mission_controller",
    "level_crossing": "level_crossing_lidar_controller",
    "tunnel": "tunnel_mission_controller",
}
AMCL_CONFIG = PACKAGE_DIR / "config" / "amcl_gazebo.yaml"
GAZEBO_MAP = PACKAGE_DIR / "maps" / "autorace_gazebo.pgm"
RVIZ_CONFIG = PACKAGE_DIR.parents[1] / "autorace_sim.rviz"
TUNNEL_LAYOUTS = (
    PACKAGE_DIR.parent
    / "custom_autorace_description"
    / "config"
    / "tunnel_obstacle_layouts.yaml"
)
CORE_MISSION_LAUNCH = (
    PACKAGE_DIR.parent
    / "turtlebot3_autorace_2020"
    / "turtlebot3_autorace_core"
    / "launch"
    / "turtlebot3_autorace_mission.launch"
)
MODEL_ROOT = (
    PACKAGE_DIR.parent
    / "turtlebot3_simulations"
    / "turtlebot3_gazebo"
    / "models"
    / "turtlebot3_autorace_2020"
)


class GazeboLaunchWiringTest(unittest.TestCase):
    def setUp(self):
        self.root = ET.parse(str(LAUNCH_FILE)).getroot()

    def arg(self, name):
        return self.root.find("./arg[@name='%s']" % name)

    def node_param(self, node_name, param_name):
        return self.root.find(
            ".//node[@name='%s']/param[@name='%s']" % (node_name, param_name)
        )

    def test_default_race_launch_provides_filtered_odometry(self):
        self.assertEqual(self.arg("odometry_source").get("default"), "world")
        self.assertEqual(self.arg("fuse_imu").get("default"), "true")

    def test_integrated_bringup_shows_live_trajectory_by_default(self):
        self.assertEqual(self.arg("rviz").get("default"), "true")
        self.assertEqual(
            self.arg("publish_trajectories").get("default"), "true"
        )
        forwarded = self.root.find(
            ".//include/arg[@name='publish_trajectories']"
        )
        self.assertIsNotNone(forwarded)
        self.assertEqual(forwarded.get("value"), "$(arg publish_trajectories)")

        description_root = ET.parse(str(DESCRIPTION_GAZEBO_LAUNCH)).getroot()
        description_arg = description_root.find(
            "./arg[@name='publish_trajectories']"
        )
        # The lower-level simulator has no RViz of its own; the integrated
        # bringup explicitly enables this diagnostic display.
        self.assertEqual(description_arg.get("default"), "false")

        rviz = RVIZ_CONFIG.read_text(encoding="utf-8")
        self.assertIn("Name: Actual and EKF Trajectory", rviz)
        self.assertIn("Marker Topic: /trajectory/comparison", rviz)

    def test_integrated_bringup_shows_both_parking_plans_by_default(self):
        with RVIZ_CONFIG.open(encoding="utf-8") as stream:
            rviz = yaml.safe_load(stream)
        displays = {
            display.get("Name"): display
            for display in rviz["Visualization Manager"]["Displays"]
        }
        expected = {
            "Parking Left Planned Path": "/parking/planned_path/left",
            "Parking Right Planned Path": "/parking/planned_path/right",
        }
        for name, topic in expected.items():
            with self.subTest(name=name):
                display = displays[name]
                self.assertEqual(display["Class"], "rviz/Path")
                self.assertTrue(display["Enabled"])
                self.assertTrue(display["Value"])
                self.assertEqual(display["Topic"], topic)
        self.assertNotEqual(
            displays["Parking Left Planned Path"]["Color"],
            displays["Parking Right Planned Path"]["Color"],
        )

    def test_gazebo_amcl_uses_signed_subcentimetre_diff_model(self):
        with AMCL_CONFIG.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(config["odom_model_type"], "diff-signed")
        # Keep the existing scan-update cadence; the model, rather than a
        # course-specific angular threshold, fixes short reverse travel.
        self.assertAlmostEqual(config["update_min_a"], 0.05)

    def test_missions_follow_the_selected_odometry_publisher(self):
        selector = self.arg("mission_odometry_topic").get("default")
        self.assertIn("/odometry/filtered", selector)
        self.assertIn("/odom", selector)
        self.assertIn("arg('fuse_imu')", selector)

        intersection = self.node_param(
            "intersection_mission_controller", "mission/odom_topic"
        )
        obstacle = self.node_param(
            "obstacle_mission_controller", "obstacle/topics/odometry"
        )
        self.assertEqual(intersection.get("value"), "$(arg mission_odometry_topic)")
        self.assertEqual(obstacle.get("value"), "$(arg mission_odometry_topic)")

    def test_zigzag_survey_path_keeps_its_map_display_frame(self):
        with MISSION_CONFIGS["zigzag"].open(encoding="utf-8") as stream:
            route = yaml.safe_load(stream)["zigzag"]["route"]
        self.assertTrue(route["odom_aligned"])
        self.assertEqual(route["frame_id"], "map")

    def test_zigzag_aligner_consumes_the_single_generated_camera_path(self):
        with LANE_CONFIG.open(encoding="utf-8") as stream:
            lane = yaml.safe_load(stream)
        with MISSION_CONFIGS["zigzag"].open(encoding="utf-8") as stream:
            zigzag = yaml.safe_load(stream)["zigzag"]

        self.assertEqual(lane["path_topic"], "/control/lane_path")
        self.assertEqual(zigzag["topics"]["lane_path"], lane["path_topic"])

    def test_common_path_missions_share_direct_cmd_vel_and_handoff_wiring(self):
        with MISSION_ZONES.open(encoding="utf-8") as stream:
            zones = yaml.safe_load(stream)

        for name, config_path in MISSION_CONFIGS.items():
            node = self.root.find(
                ".//node[@name='%s_mission_controller']" % name
            )
            self.assertIsNotNone(node, name)
            self.assertIn(config_path.name, node.find("./rosparam").get("file"))

            with config_path.open(encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
            root = config["mission"] if name == "intersection" else config[name]
            topics = root if name == "intersection" else root["topics"]
            cmd_vel = topics.get("cmd_vel_topic", topics.get("cmd_vel"))
            gate = topics.get("zone_gate_topic", topics.get("zone_gate"))
            self.assertEqual(cmd_vel, "/cmd_vel", name)
            self.assertEqual(
                topics["lane_control_service"],
                "/control/lane_mission_handoff",
                name,
            )
            self.assertEqual(
                topics["lane_stop_service"], "/control/lane_following", name
            )
            self.assertEqual(gate, zones["missions"][name]["gate_topic"], name)
            self.assertEqual(
                zones["missions"][name]["completion_topic"],
                "/%s/state" % name,
                name,
            )

        self.assertEqual(zones.get("regions", {}), {})
        with MISSION_CONFIGS["intersection"].open(encoding="utf-8") as stream:
            intersection = yaml.safe_load(stream)["mission"]
        self.assertEqual(
            intersection["zone_gate_topic"],
            zones["missions"]["intersection"]["gate_topic"],
        )
        self.assertNotIn("direction_observation_topic", intersection)

    def test_intersection_has_no_legacy_map_keys_or_lane_path_wiring(self):
        with MISSION_CONFIGS["intersection"].open(encoding="utf-8") as stream:
            mission = yaml.safe_load(stream)["mission"]
        legacy_keys = {
            "direction_observation_topic",
            "mission_map_pose_topic",
            "entry_handoff_timeout",
            "entry_handoff_lead_distance",
            "map_frame",
            "map_transform_lookup_timeout",
            "map_transform_max_age",
            "map_world_size",
            "map_resolution",
            "map_boundary_inflation",
            "map_snap_max_distance",
            "exit_map_snap_max_distance",
            "map_texture_package",
            "map_texture_relative_path",
            "map_entry_start",
            "map_entry_start_yaw_deg",
            "map_left_entry_goal",
            "map_right_entry_goal",
            "map_left_arc_entry_yaw_deg",
            "map_right_arc_entry_yaw_deg",
            "map_entry_samples",
            "map_left_exit_control_points",
            "map_right_exit_control_points",
            "map_exit_branch_samples",
            "map_exit_control_points",
            "map_exit_samples",
            "exit_goal_yaw_deg",
            "zone_signal_timeout",
        }
        self.assertEqual(legacy_keys.intersection(mission), set())

        intersection_node = self.root.find(
            ".//node[@name='intersection_mission_controller']"
        )
        self.assertIsNotNone(intersection_node)
        parameters = {
            parameter.get("name"): parameter.get("value")
            for parameter in intersection_node.findall("./param")
        }
        self.assertNotIn("mission/lane_path_topic", parameters)
        self.assertNotIn("/control/lane_path", parameters.values())
        for remap in intersection_node.findall("./remap"):
            self.assertNotEqual(remap.get("from"), "/control/lane_path")
            self.assertNotEqual(remap.get("to"), "/control/lane_path")

        production_wiring = [
            parameter.get("value") for parameter in self.root.findall(".//param")
        ]
        for remap in self.root.findall(".//remap"):
            production_wiring.extend((remap.get("from"), remap.get("to")))
        self.assertNotIn("/control/lane_path", production_wiring)

    def test_every_ordered_controller_matches_manager_handshake_topics(self):
        with MISSION_ZONES.open(encoding="utf-8") as stream:
            zones = yaml.safe_load(stream)

        self.assertEqual(
            zones["sequence"], list(ORDERED_MISSION_CONFIGS.keys())
        )
        for name, config_path in ORDERED_MISSION_CONFIGS.items():
            with self.subTest(mission=name):
                node = self.root.find(
                    ".//node[@name='%s']" % MISSION_NODE_NAMES[name]
                )
                self.assertIsNotNone(node)
                self.assertIn(
                    config_path.name, node.find("./rosparam").get("file")
                )

                with config_path.open(encoding="utf-8") as stream:
                    config = yaml.safe_load(stream)
                root = (
                    config["mission"]
                    if name == "intersection"
                    else config[name]
                )
                topics = root if name == "intersection" else root["topics"]
                if name == "intersection":
                    arm = topics["arm_topic"]
                    ready = topics["ready_topic"]
                    gate = topics["zone_gate_topic"]
                elif name == "zigzag":
                    arm = topics["mission_arm"]
                    ready = topics["mission_ready"]
                    gate = topics["zone_gate"]
                else:
                    arm = topics["arm"]
                    ready = topics["ready"]
                    gate = topics["zone_gate"]

                managed = zones["missions"][name]
                self.assertEqual(managed["arm_topic"], arm)
                self.assertEqual(managed["ready_topic"], ready)
                self.assertEqual(managed["gate_topic"], gate)
                self.assertEqual(
                    managed["completion_topic"], "/%s/state" % name
                )
                self.assertIn("COMPLETE", managed["completion_values"])

    def test_level_crossing_controller_is_enabled_and_ordered_last(self):
        self.assertEqual(
            self.arg("level_crossing_mission").get("default"), "true"
        )
        node = self.root.find(
            ".//node[@name='level_crossing_lidar_controller']"
        )
        self.assertIsNotNone(node)
        self.assertEqual(node.get("type"), "level_crossing_lidar_controller.py")
        self.assertEqual(node.get("required"), "true")
        rosparam = node.find("./rosparam")
        self.assertIn("level_crossing_mission_gazebo.yaml", rosparam.get("file"))
        odometry = node.find(
            "./param[@name='level_crossing/topics/odometry']"
        )
        self.assertIsNotNone(odometry)
        self.assertEqual(odometry.get("value"), "$(arg mission_odometry_topic)")

        with MISSION_ZONES.open(encoding="utf-8") as stream:
            zones = yaml.safe_load(stream)
        self.assertEqual(
            zones["sequence"][-3:],
            ["zigzag", "level_crossing", "tunnel"],
        )
        crossing = zones["missions"]["level_crossing"]
        self.assertEqual(
            crossing["completion_topic"], "/level_crossing/state"
        )
        self.assertIn("COMPLETE", crossing["completion_values"])

    def test_level_crossing_regression_phase_is_explicit_and_default_is_normal(self):
        self.assertEqual(
            self.arg("mission_models_initial_state").get("default"), "1"
        )
        include_arg = self.root.find(
            ".//include/arg[@name='initial_traffic_state']"
        )
        self.assertEqual(
            include_arg.get("value"), "$(arg mission_models_initial_state)"
        )

        core_root = ET.parse(str(CORE_MISSION_LAUNCH)).getroot()
        core_arg = core_root.find("./arg[@name='initial_traffic_state']")
        core_param = core_root.find(
            ".//node[@name='core_node_mission']/param"
            "[@name='initial_traffic_state']"
        )
        self.assertEqual(core_arg.get("default"), "1")
        self.assertEqual(core_param.get("value"), "$(arg initial_traffic_state)")

    def test_lane_path_benchmark_disables_later_mission_controllers(self):
        root = ET.parse(str(LANE_PATH_LAUNCH)).getroot()
        for name in (
            "level_crossing_mission",
            "tunnel_mission",
            "mission_zones",
            "localization",
            "publish_trajectories",
        ):
            argument = root.find(
                ".//include/arg[@name='%s']" % name
            )
            self.assertIsNotNone(argument, str(LANE_PATH_LAUNCH))
            self.assertEqual(argument.get("value"), "false")

    def test_camera_lane_benchmark_selects_common_path_speed_once(self):
        root = ET.parse(str(LANE_PATH_LAUNCH)).getroot()
        args = {
            argument.get("name"): argument.get("default")
            for argument in root.findall("./arg")
        }
        self.assertEqual(args["test_velocity"], "0.28")
        self.assertNotIn("speed_cap", args)
        self.assertNotIn("lane_path_mode", args)

        controller = root.find("./node[@name='safe_lane_controller']")
        parameters = {
            parameter.get("name"): parameter.get("value")
            for parameter in controller.findall("./param")
        }
        self.assertNotIn("lane_topic", parameters)
        self.assertNotIn("lane_path/mode", parameters)
        self.assertNotIn("speed_cap", parameters)
        self.assertEqual(parameters["centerline_topic"], "/detect/lane_centerline")
        self.assertEqual(
            parameters["lane_path/control/cruise_velocity"],
            "$(arg test_velocity)",
        )
        self.assertEqual(
            parameters["lane_path/control/entry_velocity"],
            "$(arg test_velocity)",
        )
        self.assertEqual(parameters["odometry_topic"], "/odom")
        self.assertEqual(
            parameters["diagnostics_topic"],
            "/control/lane_path_diagnostics",
        )

        monitor = root.find("./node[@name='lane_benchmark_monitor']")
        monitor_parameters = {
            parameter.get("name"): parameter.get("value")
            for parameter in monitor.findall("./param")
        }
        self.assertEqual(
            monitor_parameters["requested_cruise_velocity"],
            "$(arg test_velocity)",
        )
        self.assertNotIn("lane_path_mode", monitor_parameters)
        self.assertNotIn("require_path_diagnostics", monitor_parameters)

    def test_tunnel_controller_uses_wall_only_map_and_is_ordered_last(self):
        self.assertEqual(self.arg("tunnel_mission").get("default"), "true")
        node = self.root.find(".//node[@name='tunnel_mission_controller']")
        self.assertIsNotNone(node)
        self.assertEqual(node.get("type"), "tunnel_mission_controller.py")
        self.assertEqual(node.get("required"), "true")
        self.assertIn(
            "tunnel_mission_gazebo.yaml", node.find("./rosparam").get("file")
        )
        odometry = node.find("./param[@name='tunnel/topics/odometry']")
        self.assertEqual(odometry.get("value"), "$(arg mission_odometry_topic)")

        with MISSION_ZONES.open(encoding="utf-8") as stream:
            zones = yaml.safe_load(stream)
        self.assertEqual(zones["sequence"][-1], "tunnel")
        tunnel = zones["missions"]["tunnel"]
        self.assertEqual(tunnel["completion_topic"], "/tunnel/state")
        self.assertIn("COMPLETE", tunnel["completion_values"])

    def test_tunnel_obstacle_layout_auto_cycle_and_named_cases_are_wired(self):
        self.assertEqual(
            self.arg("tunnel_obstacle_layout").get("default"), "auto"
        )
        forwarded = self.root.find(
            "./include/arg[@name='tunnel_obstacle_layout']"
        )
        self.assertIsNotNone(forwarded)
        self.assertEqual(forwarded.get("value"), "$(arg tunnel_obstacle_layout)")

        description = ET.parse(str(DESCRIPTION_GAZEBO_LAUNCH)).getroot()
        description_arg = description.find(
            "./arg[@name='tunnel_obstacle_layout']"
        )
        self.assertEqual(description_arg.get("default"), "auto")

        with TUNNEL_LAYOUTS.open(encoding="utf-8") as stream:
            layouts = yaml.safe_load(stream)["layouts"]
        self.assertEqual(
            {name: values["rotation_deg"] for name, values in layouts.items()},
            {"layout_a": 0.0, "layout_b": 90.0, "layout_c": 180.0},
        )

    def test_rviz_shows_zigzag_and_tunnel_paths(self):
        text = RVIZ_CONFIG.read_text(encoding="utf-8")
        self.assertIn("Name: Zigzag Surveyed Path", text)
        self.assertIn("Topic: /zigzag/path", text)
        self.assertIn("Name: Tunnel LiDAR Costmap", text)
        self.assertIn("Topic: /tunnel/costmap", text)
        self.assertIn("Name: Tunnel Active Path", text)
        self.assertIn("Topic: /tunnel/path", text)

    def test_amcl_map_contains_tunnel_walls_but_not_unknown_cylinders(self):
        image = Image.open(str(GAZEBO_MAP))

        def pixel_at_world(x, y):
            column = int((x + 2.5) // 0.02)
            map_row = int((y + 2.5) // 0.02)
            return image.getpixel((column, 249 - map_row))

        for centre in (
            (-0.618230, -1.338740),
            (-1.365180, -1.414270),
            (-1.140940, -0.627116),
        ):
            self.assertEqual(pixel_at_world(*centre), 254)
        for wall_point in (
            (-1.915480, -1.005857),
            (-0.994699, -1.921180),
            (-0.021303, -0.832071),
            (-0.814699, -0.030740),
        ):
            self.assertEqual(pixel_at_world(*wall_point), 0)

    def test_gazebo_bar_assets_have_one_pose_and_closed_scan_proxy(self):
        opened = ET.parse(
            str(MODEL_ROOT / "traffic_bar_down" / "model.sdf")
        ).getroot()
        opened_collision = opened.find(".//collision[@name='collision']")
        self.assertIsNone(opened_collision.find("pose"))

        closed = ET.parse(
            str(MODEL_ROOT / "traffic_bar_up" / "model.sdf")
        ).getroot()
        closed_model = closed.find("./model")
        closed_link = closed.find("./model/link")
        self.assertIsNone(closed_model.find("pose"))
        self.assertEqual(
            closed_link.find("pose").text.split(),
            ["-1.0", "1.2", "0.15", "0", "0", "1.57"],
        )
        physical = closed.find(".//collision[@name='collision']")
        proxy = closed.find(".//collision[@name='lidar_proxy']")
        visual = closed.find(".//visual[@name='visual']")
        self.assertIsNone(physical.find("pose"))
        self.assertIsNone(visual.find("pose"))
        self.assertEqual(
            physical.find("./geometry/box/size").text.strip(),
            "0.3 0.02 0.05",
        )
        self.assertIsNotNone(proxy)
        self.assertEqual(
            proxy.find("./geometry/box/size").text.strip(),
            "0.3 0.02 0.02",
        )
        self.assertAlmostEqual(
            float(proxy.find("pose").text.split()[2]), 0.04165
        )


if __name__ == "__main__":
    unittest.main()
