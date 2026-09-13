#!/usr/bin/env python3

import importlib.machinery
import importlib.util
import threading
import unittest
from unittest import mock
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


BRINGUP_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BRINGUP_DIR.parent
DETECT_DIR = (
    SRC_DIR
    / "turtlebot3_autorace_2020"
    / "turtlebot3_autorace_detect"
)


def launch_arg(root, name):
    return root.find("./arg[@name='%s']" % name)


def node_param(root, name):
    return root.find(".//node/param[@name='%s']" % name)


class DetectorRateConfigurationTest(unittest.TestCase):
    @staticmethod
    def load_traffic_detector_module():
        path = DETECT_DIR / "nodes" / "detect_traffic_light"
        loader = importlib.machinery.SourceFileLoader(
            "detect_traffic_light_resource_test", str(path)
        )
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module

    def test_original_detectors_use_configurable_rate_controls(self):
        lane_source = (DETECT_DIR / "nodes" / "detect_lane").read_text(
            encoding="utf-8"
        )
        traffic_source = (
            DETECT_DIR / "nodes" / "detect_traffic_light"
        ).read_text(encoding="utf-8")

        self.assertIn('get_param("~process_every_n_frames", 1)', lane_source)
        self.assertIn(
            'get_param("~temporal_state_every_n_frames", 3)', lane_source
        )
        self.assertIn(
            'get_param("~debug_every_n_frames", 0)', lane_source
        )
        self.assertIn(
            "self.counter % self.process_every_n_frames", lane_source
        )
        self.assertIn("if self.update_temporal_state:", lane_source)
        self.assertIn("message.header = self.cv_image_header", lane_source)
        self.assertIn(
            "self.pub_image_lane.get_num_connections() > 0", lane_source
        )
        self.assertIn(
            "self.debug_frame_counter >= self.debug_every_n_frames",
            lane_source,
        )
        self.assertIn("self.debug_every_n_frames > 0", lane_source)
        self.assertIn('get_param("~opencv_threads", 1)', lane_source)
        self.assertIn("buff_size=self.RAW_IMAGE_BUFFER_SIZE", lane_source)
        self.assertIn("tcp_nodelay=True", lane_source)
        self.assertNotIn("self.counter % 3", lane_source)

        self.assertIn(
            'get_param("~process_every_n_frames", 1)', traffic_source
        )
        self.assertIn(
            'get_param("~processing_rate_hz", 30.0)', traffic_source
        )
        self.assertIn(
            "rospy.Rate(self.processing_rate_hz)", traffic_source
        )
        self.assertIn("self.accepted_image_generation += 1", traffic_source)
        self.assertIn(
            "== self.processed_image_generation", traffic_source
        )
        self.assertIn("message.header = self.cv_image_header", traffic_source)
        self.assertIn(
            "self.pub_image_traffic_light.get_num_connections() > 0",
            traffic_source,
        )
        self.assertIn('get_param("~opencv_threads", 1)', traffic_source)
        self.assertIn("buff_size=self.IMAGE_BUFFER_SIZE", traffic_source)
        self.assertIn("tcp_nodelay=True", traffic_source)
        self.assertNotIn("self.counter % 3", traffic_source)
        self.assertNotIn("rospy.Rate(10)", traffic_source)
        self.assertIn(
            '~detect/traffic_light/color_confirmation_frames", 9',
            traffic_source,
        )
        self.assertIn(
            '~detect/traffic_light/stop_confirmation_frames", 24',
            traffic_source,
        )
        self.assertNotIn("~detect/lane/red/", traffic_source)
        self.assertNotIn("self.green_count >= 3", traffic_source)
        self.assertNotIn("self.stop_count >= 8", traffic_source)

    def test_finished_traffic_detector_releases_image_subscription_once(self):
        module = self.load_traffic_detector_module()
        for image_type in ("raw", "compressed"):
            with self.subTest(image_type=image_type):
                detector = module.DetectTrafficLight.__new__(
                    module.DetectTrafficLight
                )
                detector.image_lock = threading.Lock()
                detector.is_traffic_light_finished = False
                detector.pending_image = object()
                detector.pending_image_header = object()
                detector.sub_image_type = image_type
                subscriber = mock.Mock()
                detector.sub_image_original = subscriber

                with mock.patch.object(module.rospy, "loginfo"):
                    detector.finish_detection()
                    detector.finish_detection()

                self.assertTrue(detector.is_traffic_light_finished)
                self.assertIsNone(detector.pending_image)
                self.assertIsNone(detector.pending_image_header)
                self.assertIsNone(detector.sub_image_original)
                subscriber.unregister.assert_called_once_with()

    def test_finished_traffic_detector_rejects_late_camera_callbacks(self):
        module = self.load_traffic_detector_module()
        detector = module.DetectTrafficLight.__new__(module.DetectTrafficLight)
        detector.is_traffic_light_finished = True
        detector.counter = 1
        detector.process_every_n_frames = 1
        detector.sub_image_type = "raw"
        detector.cvBridge = mock.Mock()

        detector.cbGetImage(mock.Mock())

        detector.cvBridge.imgmsg_to_cv2.assert_not_called()

    def test_30hz_confirmation_windows_preserve_10hz_evidence_time(self):
        traffic_config = yaml.safe_load(
            (
                DETECT_DIR
                / "param"
                / "traffic_light"
                / "traffic_light.yaml"
            ).read_text(encoding="utf-8")
        )["detect"]["traffic_light"]
        self.assertEqual(traffic_config["color_confirmation_frames"], 9)
        self.assertEqual(traffic_config["stop_confirmation_frames"], 24)

        config_dir = BRINGUP_DIR / "config"
        intersection = yaml.safe_load(
            (config_dir / "intersection_mission.yaml").read_text(
                encoding="utf-8"
            )
        )["mission"]
        self.assertEqual(intersection["direction_confirm_frames"], 9)
        self.assertNotIn("arc_lane_confirm_frames", intersection)
        self.assertEqual(intersection["final_lane_confirm_frames"], 9)

        parking = yaml.safe_load(
            (config_dir / "parking_mission_gazebo.yaml").read_text(
                encoding="utf-8"
            )
        )["parking"]["rejoin"]
        self.assertEqual(parking["handoff_confirmation_frames"], 9)
        self.assertEqual(parking["complete_confirmation_frames"], 9)

        for filename, root_key in (
            ("zigzag_mission_gazebo.yaml", "zigzag"),
            ("tunnel_mission_gazebo.yaml", "tunnel"),
        ):
            mission = yaml.safe_load(
                (config_dir / filename).read_text(encoding="utf-8")
            )[root_key]["exit"]
            self.assertEqual(mission["confirmation_frames"], 6)
            self.assertEqual(mission["join_confirmation_frames"], 6)

    def test_original_launches_default_to_full_camera_rate(self):
        lane_root = ET.parse(
            str(DETECT_DIR / "launch" / "detect_lane.launch")
        ).getroot()
        self.assertEqual(
            launch_arg(lane_root, "process_every_n_frames").get("default"),
            "1",
        )
        self.assertEqual(
            node_param(lane_root, "process_every_n_frames").get("value"),
            "$(arg process_every_n_frames)",
        )
        self.assertEqual(
            launch_arg(lane_root, "temporal_state_every_n_frames").get(
                "default"
            ),
            "3",
        )
        self.assertEqual(
            node_param(lane_root, "temporal_state_every_n_frames").get(
                "value"
            ),
            "$(arg temporal_state_every_n_frames)",
        )

        traffic_root = ET.parse(
            str(DETECT_DIR / "launch" / "detect_traffic_light.launch")
        ).getroot()
        self.assertEqual(
            launch_arg(
                traffic_root, "process_every_n_frames"
            ).get("default"),
            "1",
        )
        self.assertEqual(
            launch_arg(traffic_root, "processing_rate_hz").get("default"),
            "30.0",
        )
        for name in ("process_every_n_frames", "processing_rate_hz"):
            self.assertEqual(
                node_param(traffic_root, name).get("value"),
                "$(arg %s)" % name,
            )

    def test_integrated_lane_launch_processes_every_frame(self):
        root = ET.parse(
            str(BRINGUP_DIR / "launch" / "gazebo_lane_detection.launch")
        ).getroot()
        self.assertEqual(
            launch_arg(root, "process_every_n_frames").get("default"), "1"
        )
        self.assertEqual(
            launch_arg(root, "temporal_state_every_n_frames").get("default"),
            "3",
        )
        for name in (
            "process_every_n_frames",
            "temporal_state_every_n_frames",
        ):
            include_arg = root.find(".//include/arg[@name='%s']" % name)
            self.assertEqual(include_arg.get("value"), "$(arg %s)" % name)

        projection = root.find(".//node[@name='image_projection']")
        compensation = root.find(
            ".//node[@name='image_compensation_projection']"
        )
        self.assertEqual(
            projection.find("./param[@name='input_image_type']").get("value"),
            "raw",
        )
        self.assertEqual(
            projection.find("./param[@name='input_topic']").get("value"),
            "/camera/image_rect_color",
        )
        self.assertEqual(
            projection.find("./param[@name='output_image_type']").get("value"),
            "raw",
        )
        self.assertEqual(
            projection.find("./param[@name='output_topic']").get("value"),
            "/camera/image_projected",
        )
        self.assertEqual(
            compensation.find("./param[@name='input_image_type']").get("value"),
            "raw",
        )
        self.assertEqual(
            compensation.find("./param[@name='input_topic']").get("value"),
            "/camera/image_projected",
        )

    def test_integrated_traffic_light_launch_runs_at_30hz(self):
        root = ET.parse(
            str(
                BRINGUP_DIR
                / "launch"
                / "gazebo_traffic_light_detection.launch"
            )
        ).getroot()
        self.assertEqual(
            launch_arg(root, "process_every_n_frames").get("default"), "1"
        )
        self.assertEqual(
            launch_arg(root, "processing_rate_hz").get("default"), "30.0"
        )
        for name in ("process_every_n_frames", "processing_rate_hz"):
            self.assertEqual(
                node_param(root, name).get("value"), "$(arg %s)" % name
            )

    def test_top_level_launch_exposes_detector_rate_controls(self):
        root = ET.parse(
            str(BRINGUP_DIR / "launch" / "gazebo.launch")
        ).getroot()
        expected_defaults = {
            "lane_process_every_n_frames": "1",
            "lane_temporal_state_every_n_frames": "3",
            "traffic_light_process_every_n_frames": "1",
            "traffic_light_processing_rate_hz": "30.0",
        }
        for name, expected in expected_defaults.items():
            self.assertEqual(launch_arg(root, name).get("default"), expected)

        lane_include = root.find(
            ".//include/arg[@name='process_every_n_frames']"
            "[@value='$(arg lane_process_every_n_frames)']/.."
        )
        self.assertIsNotNone(lane_include)
        lane_state_arg = root.find(
            ".//include/arg[@name='temporal_state_every_n_frames']"
            "[@value='$(arg lane_temporal_state_every_n_frames)']"
        )
        self.assertIsNotNone(lane_state_arg)

        traffic_frame_arg = root.find(
            ".//include/arg[@name='process_every_n_frames']"
            "[@value='$(arg traffic_light_process_every_n_frames)']"
        )
        traffic_rate_arg = root.find(
            ".//include/arg[@name='processing_rate_hz']"
            "[@value='$(arg traffic_light_processing_rate_hz)']"
        )
        self.assertIsNotNone(traffic_frame_arg)
        self.assertIsNotNone(traffic_rate_arg)


if __name__ == "__main__":
    unittest.main()
