#!/usr/bin/env python3

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


PACKAGE_DIR = Path(__file__).resolve().parents[1]
HARDWARE_LAUNCH = PACKAGE_DIR / "launch" / "hardware.launch"
REALSENSE_LAUNCH = PACKAGE_DIR / "launch" / "realsense_d405.launch"
PERCEPTION_LAUNCH = PACKAGE_DIR / "launch" / "hardware_perception.launch"
PROJECTION_CONFIG = (
    PACKAGE_DIR / "config" / "d405_projection_uncalibrated.yaml"
)
SIGN_CONFIG = PACKAGE_DIR / "config" / "sign_detector_d405.yaml"
LANE_CONFIG = (
    PACKAGE_DIR / "config" / "lane_detector_d405_uncalibrated.yaml"
)


def argument(root, name):
    return root.find("./arg[@name='%s']" % name)


def child_named(parent, tag, name):
    return parent.find("./%s[@name='%s']" % (tag, name))


class HardwarePerceptionLaunchTest(unittest.TestCase):
    def test_hardware_defaults_to_one_30hz_camera_perception_pipeline(self):
        root = ET.parse(str(HARDWARE_LAUNCH)).getroot()
        self.assertEqual(
            argument(root, "start_perception").get("default"),
            "$(arg start_camera)",
        )
        self.assertEqual(argument(root, "camera_color_fps").get("default"), "30")

        expected_rate_defaults = {
            "lane_process_every_n_frames": "1",
            "lane_temporal_state_every_n_frames": "3",
            "traffic_light_process_every_n_frames": "1",
            "traffic_light_processing_rate_hz": "30.0",
            "sign_process_every_n_frames": "1",
        }
        for name, expected in expected_rate_defaults.items():
            self.assertEqual(argument(root, name).get("default"), expected)

        perception_includes = [
            include
            for include in root.findall(".//include")
            if "hardware_perception.launch" in include.get("file", "")
        ]
        self.assertEqual(len(perception_includes), 1)
        include = perception_includes[0]
        for name in expected_rate_defaults:
            forwarded = child_named(include, "arg", name)
            self.assertIsNotNone(forwarded, name)
            self.assertEqual(forwarded.get("value"), "$(arg %s)" % name)

    def test_realsense_profile_is_explicit_and_forwarded(self):
        hardware = ET.parse(str(HARDWARE_LAUNCH)).getroot()
        include = next(
            item
            for item in hardware.findall(".//include")
            if "realsense_d405.launch" in item.get("file", "")
        )
        expected = {
            "color_width": "camera_color_width",
            "color_height": "camera_color_height",
            "color_fps": "camera_color_fps",
        }
        for child_name, parent_name in expected.items():
            self.assertEqual(
                child_named(include, "arg", child_name).get("value"),
                "$(arg %s)" % parent_name,
            )

        realsense = ET.parse(str(REALSENSE_LAUNCH)).getroot()
        self.assertEqual(argument(realsense, "color_width").get("default"), "1280")
        self.assertEqual(argument(realsense, "color_height").get("default"), "720")
        self.assertEqual(argument(realsense, "color_fps").get("default"), "30")
        adapters = [
            node
            for node in realsense.findall(".//node")
            if node.get("type") == "autorace_image_adapter.py"
        ]
        self.assertEqual(len(adapters), 1)
        self.assertEqual(
            child_named(adapters[0], "param", "raw_output_topic").get(
                "value"
            ),
            "/camera/image_rect_color",
        )
        self.assertEqual(
            child_named(adapters[0], "param", "output_topic").get("value"),
            "/camera/image_rect_color/compressed",
        )

    def test_perception_launch_has_one_owner_for_each_stage_and_no_adapter(self):
        root = ET.parse(str(PERCEPTION_LAUNCH)).getroot()
        node_types = [node.get("type") for node in root.findall(".//node")]
        self.assertEqual(node_types.count("image_projection"), 1)
        self.assertEqual(node_types.count("image_compensation"), 1)
        self.assertEqual(node_types.count("detect_lane"), 1)
        self.assertEqual(node_types.count("detect_traffic_light"), 1)
        self.assertEqual(node_types.count("sign_detector.py"), 1)
        self.assertNotIn("autorace_image_adapter.py", node_types)
        self.assertFalse(any("controller" in (item or "") for item in node_types))

        serialized = PERCEPTION_LAUNCH.read_text(encoding="utf-8")
        self.assertNotIn("/cmd_vel", serialized)
        self.assertNotIn("gazebo_projection.yaml", serialized)

    def test_perception_topics_form_one_unbranched_lane_pipeline(self):
        root = ET.parse(str(PERCEPTION_LAUNCH)).getroot()
        projection = root.find(".//node[@name='image_projection']")
        compensation = root.find(
            ".//node[@name='image_compensation_projection']"
        )
        lane = root.find(".//node[@name='detect_lane']")
        traffic = root.find(".//node[@name='detect_traffic_light']")

        self.assertEqual(
            child_named(projection, "param", "input_topic").get("value"),
            "/camera/image_rect_color",
        )
        self.assertEqual(
            child_named(projection, "param", "input_image_type").get(
                "value"
            ),
            "raw",
        )
        self.assertEqual(
            child_named(projection, "param", "output_topic").get("value"),
            "/camera/image_projected",
        )
        self.assertEqual(
            child_named(projection, "param", "output_image_type").get("value"),
            "raw",
        )
        self.assertEqual(
            child_named(compensation, "param", "input_topic").get("value"),
            "/camera/image_projected",
        )
        self.assertEqual(
            child_named(compensation, "param", "input_image_type").get("value"),
            "raw",
        )
        self.assertEqual(
            child_named(compensation, "param", "output_topic").get("value"),
            "/camera/image_projected_compensated",
        )
        self.assertIsNotNone(
            lane.find(
                "./remap[@from='/detect/image_input']"
                "[@to='/camera/image_projected_compensated']"
            )
        )
        self.assertEqual(
            child_named(
                lane, "param", "temporal_state_every_n_frames"
            ).get("value"),
            "$(arg lane_temporal_state_every_n_frames)",
        )
        self.assertIsNotNone(
            traffic.find(
                "./remap[@from='/detect/image_input/compressed']"
                "[@to='/camera/image_rect_color/compressed']"
            )
        )

    def test_physical_configs_are_separate_and_marked_unverified(self):
        with PROJECTION_CONFIG.open(encoding="utf-8") as stream:
            projection = yaml.safe_load(stream)
        self.assertEqual(projection["calibration_status"], "unverified_template")
        for name in (
            "center_x",
            "top_x",
            "top_y",
            "bottom_x",
            "bottom_y",
            "output_width",
            "output_height",
            "destination_left_x",
            "destination_right_x",
        ):
            self.assertIn(name, projection)

        gazebo_projection = PACKAGE_DIR / "config" / "gazebo_projection.yaml"
        self.assertNotEqual(PROJECTION_CONFIG, gazebo_projection)

        with LANE_CONFIG.open(encoding="utf-8") as stream:
            lane = yaml.safe_load(stream)
        self.assertEqual(lane["calibration_status"], "unverified_template")
        self.assertEqual(lane["opencv_threads"], 1)
        self.assertEqual(lane["boundary"]["sample_y"], 350)
        self.assertEqual(lane["temporal_state_every_n_frames"], 3)
        self.assertEqual(lane["debug_every_n_frames"], 0)

        with SIGN_CONFIG.open(encoding="utf-8") as stream:
            sign = yaml.safe_load(stream)["sign_detector"]
        self.assertEqual(sign["process_every_n_frames"], 1)
        self.assertEqual(sign["opencv_threads"], 1)
        self.assertEqual(sign["processing_width"], 448)
        self.assertTrue(sign["mission_gating"]["enabled"])
        self.assertEqual(
            sign["mission_gating"]["active_missions"],
            ["intersection", "tunnel"],
        )
        template_filter = sign["mission_gating"]["template_filter"]
        self.assertFalse(template_filter["fail_open_without_mission"])
        self.assertEqual(
            template_filter["standalone_templates"],
            ["direction_left", "direction_right", "tunnel_warning"],
        )
        self.assertEqual(
            template_filter["mission_templates"]["intersection"],
            ["direction_left", "direction_right"],
        )
        self.assertEqual(
            template_filter["mission_templates"]["tunnel"],
            ["tunnel_warning"],
        )
        parking_file = sign["classifiers"]["template"]["entries"]["parking"][
            "file"
        ]
        self.assertEqual(parking_file, "parking.png")
        self.assertNotIn("turtlebot3_gazebo", parking_file)


if __name__ == "__main__":
    unittest.main()
