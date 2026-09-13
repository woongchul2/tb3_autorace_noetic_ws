#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import rospkg
import rospy
from sensor_msgs.msg import Image
import yaml


PACKAGE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_SRC = PACKAGE_DIR.parent
NODE_DIR = PACKAGE_DIR / "nodes"
if str(NODE_DIR) not in sys.path:
    sys.path.insert(0, str(NODE_DIR))

from sign_detector import SignDetector
from custom_autorace_bringup.msg import TrafficSign


CONFIG_PATH = PACKAGE_DIR / "config" / "sign_detector.yaml"
GAZEBO_MODEL_DIR = (
    WORKSPACE_SRC
    / "turtlebot3_simulations"
    / "turtlebot3_gazebo"
    / "models"
    / "turtlebot3_autorace_2020"
)


class FakeRosPack:
    def __init__(self, packages):
        self.packages = packages

    def get_path(self, package_name):
        if package_name not in self.packages:
            raise rospkg.ResourceNotFound(package_name)
        return self.packages[package_name]


class SignDetectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with CONFIG_PATH.open(encoding="utf-8") as stream:
            cls.config = yaml.safe_load(stream)["sign_detector"]

    def make_detector(self):
        config = self.config
        template = config["classifiers"]["template"]
        direction = config["classifiers"]["direction_geometry"]
        detector = SignDetector.__new__(SignDetector)
        detector.mission_gating_enabled = False
        detector.current_mission = None
        detector.active_missions = frozenset(("intersection", "tunnel"))
        template_filter = config["mission_gating"]["template_filter"]
        # Geometry/template quality tests intentionally exercise every
        # registered class. Production keeps this false and bounded.
        detector.template_filter_fail_open_without_mission = True
        detector.standalone_template_labels = frozenset(
            template_filter["standalone_templates"]
        )
        detector.mission_template_labels = {
            mission: frozenset(labels)
            for mission, labels in template_filter[
                "mission_templates"
            ].items()
        }
        detector.minimum_confidence = float(config["minimum_confidence"])
        detector.ratio_test = float(template["ratio_test"])
        detector.ransac_threshold = float(
            template["ransac_reprojection_threshold"]
        )
        detector.minimum_inlier_ratio = float(
            template["minimum_inlier_ratio"]
        )
        detector.minimum_projected_area = float(
            template["minimum_projected_area"]
        )
        detector.maximum_projected_area_ratio = float(
            template["maximum_projected_area_ratio"]
        )
        detector.projection_margin_ratio = float(
            template["projection_margin_ratio"]
        )
        detector.orb = cv2.ORB_create(
            nfeatures=int(template["orb_nfeatures"]),
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=5,
            patchSize=31,
            fastThreshold=int(template["orb_fast_threshold"]),
        )
        detector.matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING, crossCheck=False
        )

        detector.direction_geometry_enabled = bool(direction["enabled"])
        detector.direction_blue_lower = np.asarray(
            direction["blue_hsv_lower"], dtype=np.uint8
        )
        detector.direction_blue_upper = np.asarray(
            direction["blue_hsv_upper"], dtype=np.uint8
        )
        detector.direction_white_saturation_max = int(
            direction["white_saturation_max"]
        )
        detector.direction_white_value_min = int(
            direction["white_value_min"]
        )
        detector.direction_minimum_blue_area_ratio = float(
            direction["minimum_blue_area_ratio"]
        )
        detector.direction_maximum_blue_area_ratio = float(
            direction["maximum_blue_area_ratio"]
        )
        detector.direction_minimum_hull_circularity = float(
            direction["minimum_hull_circularity"]
        )
        detector.direction_polygon_epsilon_ratio = float(
            direction["polygon_epsilon_ratio"]
        )
        detector.direction_minimum_polygon_vertices = int(
            direction["minimum_polygon_vertices"]
        )
        detector.direction_minimum_aspect_ratio = float(
            direction["minimum_aspect_ratio"]
        )
        detector.direction_maximum_aspect_ratio = float(
            direction["maximum_aspect_ratio"]
        )
        detector.direction_interior_erode_ratio = float(
            direction["interior_erode_ratio"]
        )
        detector.direction_minimum_arrow_pixels = int(
            direction["minimum_arrow_pixels"]
        )
        detector.direction_asymmetry_deadband = float(
            direction["asymmetry_deadband"]
        )
        detector.templates = []
        return detector

    @staticmethod
    def render_texture(path, corners, frame_shape=(480, 640)):
        texture = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if texture is None:
            raise AssertionError("missing test texture: %s" % path)
        height, width = texture.shape[:2]
        source = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        )
        homography = cv2.getPerspectiveTransform(source, np.float32(corners))
        frame = np.full(frame_shape + (3,), 175, dtype=np.uint8)
        warped = cv2.warpPerspective(
            texture[:, :, :3], homography, (frame.shape[1], frame.shape[0])
        )
        plane = cv2.warpPerspective(
            np.full((height, width), 255, dtype=np.uint8),
            homography,
            (frame.shape[1], frame.shape[0]),
        )
        frame[plane > 0] = warped[plane > 0]
        return frame

    def add_template(self, detector, label, path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = detector.orb.detectAndCompute(image, None)
        config = self.config["classifiers"]["template"]["entries"][label]
        sign_type, direction = detector.TEMPLATE_TYPES[label]
        detector.templates.append(
            detector.Template(
                label,
                sign_type,
                direction,
                image,
                keypoints,
                descriptors,
                int(config["min_good_matches"]),
                int(config["min_inliers"]),
                int(config["priority"]),
            )
        )

    def configured_template_path(self, label):
        template = self.config["classifiers"]["template"]
        template_directory = (
            WORKSPACE_SRC
            / "turtlebot3_autorace_2020"
            / "turtlebot3_autorace_detect"
            / template["directory"]
        )
        rospack = FakeRosPack(
            {
                "turtlebot3_gazebo": str(
                    WORKSPACE_SRC
                    / "turtlebot3_simulations"
                    / "turtlebot3_gazebo"
                )
            }
        )
        return Path(
            SignDetector._resolve_template_path(
                str(template_directory),
                template["entries"][label]["file"],
                rospack,
            )
        )

    def test_template_path_supports_relative_and_package_uri(self):
        rospack = FakeRosPack({"example_pkg": "/opt/example_pkg"})
        self.assertEqual(
            SignDetector._resolve_template_path(
                "/templates", "parking.png", rospack
            ),
            "/templates/parking.png",
        )
        self.assertEqual(
            SignDetector._resolve_template_path(
                "/unused",
                "package://example_pkg/images/parking.png",
                rospack,
            ),
            "/opt/example_pkg/images/parking.png",
        )
        with self.assertRaises(rospkg.ResourceNotFound):
            SignDetector._resolve_template_path(
                "/unused", "package://missing_pkg/parking.png", rospack
            )

    def test_30hz_config_processes_every_frame_at_448_pixels(self):
        self.assertEqual(self.config["process_every_n_frames"], 1)
        self.assertEqual(self.config["opencv_threads"], 1)
        self.assertEqual(self.config["processing_width"], 448)
        self.assertEqual(self.config["debug_every_n_frames"], 3)
        self.assertTrue(self.config["mission_gating"]["enabled"])
        self.assertEqual(
            self.config["mission_gating"]["active_missions"],
            ["intersection", "tunnel"],
        )
        template_filter = self.config["mission_gating"]["template_filter"]
        self.assertFalse(template_filter["fail_open_without_mission"])
        self.assertEqual(
            template_filter["standalone_templates"],
            ["direction_left", "direction_right", "tunnel_warning"],
        )
        self.assertEqual(
            template_filter["mission_templates"],
            {
                "intersection": ["direction_left", "direction_right"],
                "tunnel": ["tunnel_warning"],
            },
        )

    def test_mission_filter_matches_only_required_templates(self):
        detector = self.make_detector()
        labels = (
            "intersection_warning",
            "direction_left",
            "direction_right",
            "construction_warning",
            "tunnel_warning",
            "parking",
            "level_crossing_warning",
        )
        detector.templates = [SimpleNamespace(label=label) for label in labels]
        detector.template_filter_fail_open_without_mission = False
        detector.orb = mock.Mock()
        detector.orb.detectAndCompute.return_value = (
            [object(), object(), object(), object()],
            np.ones((4, 32), dtype=np.uint8),
        )
        detector._detect_direction_geometry = mock.Mock(return_value=None)
        detector._match_template = mock.Mock(return_value=None)
        frame = np.zeros((24, 32, 3), dtype=np.uint8)

        detector.current_mission = "intersection"
        result = detector._detect(frame)
        self.assertEqual(result.sign_type, TrafficSign.NONE)
        self.assertEqual(result.direction, TrafficSign.DIRECTION_NONE)
        self.assertEqual(
            [call.args[0].label for call in detector._match_template.call_args_list],
            ["direction_left", "direction_right"],
        )
        detector._detect_direction_geometry.assert_called_once()

        detector._match_template.reset_mock()
        detector._detect_direction_geometry.reset_mock()
        detector.current_mission = "tunnel"
        result = detector._detect(frame)
        self.assertEqual(result.sign_type, TrafficSign.NONE)
        self.assertEqual(
            [call.args[0].label for call in detector._match_template.call_args_list],
            ["tunnel_warning"],
        )
        detector._detect_direction_geometry.assert_not_called()

        detector._match_template.reset_mock()
        detector.current_mission = None
        detector._detect(frame)
        self.assertEqual(
            [call.args[0].label for call in detector._match_template.call_args_list],
            ["direction_left", "direction_right", "tunnel_warning"],
        )

        detector._match_template.reset_mock()
        detector.template_filter_fail_open_without_mission = True
        detector._detect(frame)
        self.assertEqual(
            [call.args[0].label for call in detector._match_template.call_args_list],
            list(labels),
        )

    def test_template_filter_rejects_unknown_labels(self):
        self.assertEqual(
            SignDetector._parse_template_labels(
                ["direction_left", "tunnel_warning"], "test"
            ),
            frozenset(("direction_left", "tunnel_warning")),
        )
        with self.assertRaises(rospy.ROSInitException):
            SignDetector._parse_template_labels(["not_a_sign"], "test")

    def test_detection_resize_preserves_aspect_ratio_without_upscaling(self):
        d405_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        resized, scale_x, scale_y = SignDetector._resize_for_detection(
            d405_image, self.config["processing_width"]
        )
        self.assertEqual(resized.shape, (252, 448, 3))
        self.assertAlmostEqual(scale_x, 0.35)
        self.assertAlmostEqual(scale_y, 0.35)

        gazebo_image = np.zeros((480, 640, 3), dtype=np.uint8)
        unchanged, scale_x, scale_y = SignDetector._resize_for_detection(
            gazebo_image, self.config["processing_width"]
        )
        self.assertEqual(unchanged.shape, (336, 448, 3))
        self.assertAlmostEqual(scale_x, 0.7)
        self.assertAlmostEqual(scale_y, 0.7)

        already_small = np.zeros((240, 320, 3), dtype=np.uint8)
        unchanged, scale_x, scale_y = SignDetector._resize_for_detection(
            already_small, self.config["processing_width"]
        )
        self.assertIs(unchanged, already_small)
        self.assertEqual((scale_x, scale_y), (1.0, 1.0))

    def test_detection_box_maps_back_to_original_camera_roi(self):
        source_shape = (720, 1280, 3)
        box = SignDetector._box_to_source(
            (70, 35, 56, 42), 0.35, 0.35, source_shape
        )
        self.assertEqual(box, (200, 100, 160, 120))
        processing_area_ratio = (56 * 42) / float(448 * 252)
        source_area_ratio = (box[2] * box[3]) / float(1280 * 720)
        self.assertAlmostEqual(source_area_ratio, processing_area_ratio)

        clipped = SignDetector._box_to_source(
            (434, 238, 28, 28), 0.35, 0.35, source_shape
        )
        self.assertEqual(clipped, (1240, 680, 40, 40))

    def test_callback_publishes_each_detection_and_throttles_only_debug(self):
        detector = self.make_detector()
        detector.frame_count = 0
        detector.processed_frame_count = 0
        detector.process_every_n_frames = 1
        detector.processing_width = 448
        detector.debug_every_n_frames = 3
        detector.bridge = mock.Mock()
        detector.bridge.imgmsg_to_cv2.return_value = np.zeros(
            (720, 1280, 3), dtype=np.uint8
        )
        detector.bridge.cv2_to_imgmsg.side_effect = lambda image, encoding: Image()
        detector.sign_pub = mock.Mock()
        detector.debug_pub = mock.Mock()
        detector.debug_pub.get_num_connections.return_value = 1
        detector._detect = mock.Mock(
            return_value=detector.Detection(
                TrafficSign.NONE,
                TrafficSign.DIRECTION_NONE,
                (100, 50, 80, 60),
                0.0,
                -1,
                "none",
            )
        )
        message = Image()
        message.header.seq = 17

        for _ in range(3):
            detector.image_callback(message)

        self.assertEqual(detector.sign_pub.publish.call_count, 3)
        published = detector.sign_pub.publish.call_args.args[0]
        self.assertEqual(published.header.seq, 17)
        self.assertEqual(
            (
                published.roi.x_offset,
                published.roi.y_offset,
                published.roi.width,
                published.roi.height,
            ),
            (286, 143, 228, 171),
        )
        self.assertEqual(detector.debug_pub.publish.call_count, 1)
        debug = detector.debug_pub.publish.call_args.args[0]
        self.assertEqual(debug.header.seq, 17)

    def test_inactive_mission_skips_orb_without_decoding_camera_frame(self):
        detector = self.make_detector()
        detector.frame_count = 0
        detector.processed_frame_count = 0
        detector.process_every_n_frames = 1
        detector.processing_width = 448
        detector.debug_every_n_frames = 3
        detector.mission_gating_enabled = True
        detector.current_mission = "parking"
        detector.bridge = mock.Mock()
        detector.sign_pub = mock.Mock()
        detector.debug_pub = mock.Mock()
        detector._detect = mock.Mock()

        detector.image_callback(Image())

        detector.bridge.imgmsg_to_cv2.assert_not_called()
        detector._detect.assert_not_called()
        detector.sign_pub.publish.assert_not_called()

        detector.mission_callback(SimpleNamespace(data="tunnel"))
        detector.bridge.imgmsg_to_cv2.return_value = np.zeros(
            (240, 320, 3), dtype=np.uint8
        )
        detector._detect.return_value = detector.Detection(
            TrafficSign.NONE,
            TrafficSign.DIRECTION_NONE,
            None,
            0.0,
            -1,
            "none",
        )
        detector.debug_pub.get_num_connections.return_value = 0

        detector.image_callback(Image())

        detector.bridge.imgmsg_to_cv2.assert_called_once()
        detector._detect.assert_called_once()
        detector.sign_pub.publish.assert_called_once()

    def test_all_templates_survive_gazebo_and_d405_processing_resize(self):
        detector = self.make_detector()
        entries = self.config["classifiers"]["template"]["entries"]
        paths = {
            label: self.configured_template_path(label) for label in entries
        }
        for label, path in paths.items():
            self.add_template(detector, label, path)

        camera_cases = (
            ((480, 640), 260),
            ((720, 1280), 390),
        )
        for frame_shape, projected_long_side in camera_cases:
            for expected_label, path in paths.items():
                texture = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                texture_height, texture_width = texture.shape[:2]
                projection_scale = projected_long_side / float(
                    max(texture_height, texture_width)
                )
                projected_width = texture_width * projection_scale
                projected_height = texture_height * projection_scale
                center_x = frame_shape[1] * 0.5
                center_y = frame_shape[0] * 0.5
                corners = (
                    [center_x - projected_width / 2, center_y - projected_height / 2],
                    [center_x + projected_width / 2, center_y - projected_height / 2 + 3],
                    [center_x + projected_width / 2 - 5, center_y + projected_height / 2],
                    [center_x - projected_width / 2 + 4, center_y + projected_height / 2 - 2],
                )
                frame = self.render_texture(path, corners, frame_shape)
                processing_image, scale_x, scale_y = detector._resize_for_detection(
                    frame, self.config["processing_width"]
                )
                result = detector._detect(processing_image, scale_x, scale_y)
                with self.subTest(
                    source_shape=frame_shape,
                    expected_label=expected_label,
                ):
                    self.assertEqual(result.label, expected_label)
                    self.assertGreaterEqual(
                        result.confidence, detector.minimum_confidence
                    )
                    source_box = detector._box_to_source(
                        result.box, scale_x, scale_y, frame.shape
                    )
                    self.assertIsNotNone(source_box)
                    self.assertLessEqual(
                        source_box[0] + source_box[2], frame_shape[1]
                    )
                    self.assertLessEqual(
                        source_box[1] + source_box[3], frame_shape[0]
                    )

    def test_missing_package_uri_fails_template_initialization(self):
        detector = self.make_detector()
        prefix = "~sign_detector/classifiers/template/"
        entries = {
            "parking": {
                "file": "package://missing_pkg/parking.png",
                "min_good_matches": 6,
                "min_inliers": 4,
                "priority": 10,
            }
        }
        parameters = {
            prefix + "package": "default_pkg",
            prefix + "directory": "image",
            prefix + "entries": entries,
        }
        rospack = FakeRosPack({"default_pkg": "/opt/default_pkg"})
        with mock.patch.object(
            rospy,
            "get_param",
            side_effect=lambda name, default=None: parameters.get(name, default),
        ), mock.patch.object(
            rospkg, "RosPack", return_value=rospack
        ), mock.patch.object(rospy, "logfatal") as logfatal:
            with self.assertRaises(rospy.ROSInitException):
                detector._load_templates(prefix)
        logfatal.assert_called_once()

    def test_all_configured_templates_resolve_and_load(self):
        detector = self.make_detector()
        prefix = "~sign_detector/classifiers/template/"
        template = self.config["classifiers"]["template"]
        parameters = {
            prefix + "package": template["package"],
            prefix + "directory": template["directory"],
            prefix + "entries": template["entries"],
        }
        rospack = FakeRosPack(
            {
                "turtlebot3_autorace_detect": str(
                    WORKSPACE_SRC
                    / "turtlebot3_autorace_2020"
                    / "turtlebot3_autorace_detect"
                ),
                "turtlebot3_gazebo": str(
                    WORKSPACE_SRC
                    / "turtlebot3_simulations"
                    / "turtlebot3_gazebo"
                ),
            }
        )
        with mock.patch.object(
            rospy,
            "get_param",
            side_effect=lambda name, default=None: parameters.get(name, default),
        ), mock.patch.object(
            rospkg, "RosPack", return_value=rospack
        ), mock.patch.object(rospy, "loginfo"):
            detector._load_templates(prefix)

        self.assertEqual(len(detector.templates), len(template["entries"]))
        parking = next(
            item for item in detector.templates if item.label == "parking"
        )
        self.assertEqual(parking.image.shape, (532, 272))

    def test_gazebo_parking_texture_wins_at_measured_camera_projection(self):
        detector = self.make_detector()
        parking_texture = (
            GAZEBO_MODEL_DIR
            / "traffic_parking"
            / "materials"
            / "textures"
            / "traffic_parking.png"
        )
        self.add_template(detector, "parking", parking_texture)

        # Full-plane corners measured by homography in the failed live frame.
        frame = self.render_texture(
            parking_texture,
            [[497, -4], [629, -5], [571, 253], [465, 253]],
        )

        detector.direction_minimum_polygon_vertices = 3
        false_direction = detector._detect_direction_geometry(frame)
        self.assertIsNotNone(false_direction)
        self.assertEqual(false_direction.direction, TrafficSign.DIRECTION_RIGHT)
        detector.direction_minimum_polygon_vertices = 8
        self.assertIsNone(detector._detect_direction_geometry(frame))
        result = detector._detect(frame)
        self.assertEqual(result.sign_type, TrafficSign.PARKING)
        self.assertGreater(result.confidence, 0.85)

    def test_left_and_right_disks_survive_polygon_gate_when_top_clipped(self):
        detector = self.make_detector()
        cases = (
            ("traffic_left", TrafficSign.DIRECTION_LEFT),
            ("traffic_right", TrafficSign.DIRECTION_RIGHT),
        )
        projections = (
            [[220, 10], [420, 10], [400, 410], [240, 410]],
            [[190, -100], [450, -100], [410, 440], [230, 440]],
        )
        for model_name, expected_direction in cases:
            texture = (
                GAZEBO_MODEL_DIR
                / model_name
                / "materials"
                / "textures"
                / (model_name + ".png")
            )
            for projection in projections:
                with self.subTest(
                    model=model_name, clipped=projection[0][1] < 0
                ):
                    frame = self.render_texture(texture, projection)
                    result = detector._detect_direction_geometry(frame)
                    self.assertIsNotNone(result)
                    self.assertEqual(result.direction, expected_direction)


if __name__ == "__main__":
    unittest.main()
