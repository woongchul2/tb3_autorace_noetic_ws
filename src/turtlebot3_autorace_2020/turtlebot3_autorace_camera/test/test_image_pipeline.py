#!/usr/bin/env python3

import importlib.machinery
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np


NODE_DIR = Path(__file__).resolve().parents[1] / "nodes"


def load_node(module_name, filename):
    loader = importlib.machinery.SourceFileLoader(
        module_name, str(NODE_DIR / filename)
    )
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


projection_module = load_node("image_projection_node", "image_projection")
compensation_module = load_node(
    "image_compensation_node", "image_compensation"
)


class ImagePipelineTest(unittest.TestCase):
    @staticmethod
    def make_projection():
        projection = projection_module.ImageProjection.__new__(
            projection_module.ImageProjection
        )
        projection.center_x = 160.0
        projection.top_x = 72.0
        projection.top_y = 4.0
        projection.bottom_x = 115.0
        projection.bottom_y = 119.0
        projection.output_width = 1000
        projection.output_height = 600
        projection.destination_left_x = 265.14
        projection.destination_right_x = 734.86
        projection.destination_bottom_y = 599.0
        projection.output_shift_x = -24.07
        projection.input_image_type = "compressed"
        projection.output_image_type = "compressed"
        projection.homography = projection._make_homography()
        return projection

    @staticmethod
    def make_extended_projection():
        projection = ImagePipelineTest.make_projection()
        projection.top_x = 28.0
        projection.top_y = 70.0
        projection.homography = projection._make_homography()
        return projection

    def test_projection_maps_source_to_shifted_destination(self):
        projection = self.make_projection()
        mapped = cv2.perspectiveTransform(
            projection._source_points().reshape(1, -1, 2),
            projection.homography,
        )[0]
        expected = np.float32(
            [
                [241.07, 0.0],
                [710.79, 0.0],
                [710.79, 599.0],
                [241.07, 599.0],
            ]
        )
        np.testing.assert_allclose(mapped, expected, atol=1e-3)

    def test_extended_projection_uses_source_row_110(self):
        projection = self.make_extended_projection()

        source = projection._source_points()

        np.testing.assert_allclose(
            source,
            np.float32(
                [
                    [132.0, 110.0],
                    [188.0, 110.0],
                    [275.0, 239.0],
                    [45.0, 239.0],
                ]
            ),
            atol=1e-6,
        )
        self.assertLess(
            source[0, 1], self.make_projection()._source_points()[0, 1]
        )

    def test_extended_projection_preserves_legacy_sample_ground_point(self):
        legacy = self.make_projection()
        extended = self.make_extended_projection()

        # The production lane centre and boundary sample used output row 350
        # before the field of view was extended. Carry that exact source-image
        # point through the new homography instead of changing where the
        # existing controller observes the ground.
        legacy_output = np.float32([[[500.0, 350.0]]])
        source_point = cv2.perspectiveTransform(
            legacy_output, np.linalg.inv(legacy.homography)
        )
        extended_output = cv2.perspectiveTransform(
            source_point, extended.homography
        )[0, 0]

        np.testing.assert_allclose(
            source_point[0, 0], np.float32([169.4419, 205.4901]), atol=1e-3
        )
        self.assertEqual(int(round(float(extended_output[1]))), 552)
        self.assertAlmostEqual(float(extended_output[0]), 500.0, delta=0.1)

        legacy_band = np.float32([[[500.0, 325.0], [500.0, 375.0]]])
        source_band = cv2.perspectiveTransform(
            legacy_band, np.linalg.inv(legacy.homography)
        )
        extended_band = cv2.perspectiveTransform(
            source_band, extended.homography
        )[0]
        new_sample_y = int(round(float(extended_output[1])))
        new_band_height = 10
        self.assertEqual(
            new_sample_y - new_band_height // 2,
            int(np.floor(extended_band[0, 1])),
        )
        self.assertEqual(
            new_sample_y - new_band_height // 2 + new_band_height,
            int(np.ceil(extended_band[1, 1])),
        )

    def test_flat_projection_parameter_precedes_legacy_tree(self):
        values = {
            "~top_x": 80.0,
            "~camera/extrinsic_camera_calibration/top_x": 72.0,
        }
        with mock.patch.object(
            projection_module.rospy,
            "has_param",
            side_effect=lambda name: name == "~top_x",
        ), mock.patch.object(
            projection_module.rospy,
            "get_param",
            side_effect=lambda name, default=None: values.get(name, default),
        ):
            self.assertEqual(projection_module._projection_param("top_x", 0), 80.0)

        with mock.patch.object(
            projection_module.rospy, "has_param", return_value=False
        ), mock.patch.object(
            projection_module.rospy,
            "get_param",
            side_effect=lambda name, default=None: values.get(name, default),
        ):
            self.assertEqual(projection_module._projection_param("top_x", 0), 72.0)

    def test_projection_preserves_output_shape_and_corner_masks(self):
        projection = self.make_projection()
        image = np.full((240, 320, 3), 127, dtype=np.uint8)

        result = projection._project_image(image)

        self.assertEqual(result.shape, (600, 1000, 3))
        np.testing.assert_array_equal(result[-1, 0], np.zeros(3, np.uint8))
        np.testing.assert_array_equal(result[-1, -1], np.zeros(3, np.uint8))

    def test_uniform_image_is_not_divided_by_zero(self):
        compensation = compensation_module.ImageCompensation.__new__(
            compensation_module.ImageCompensation
        )
        compensation.clip_hist_percent = 1.0
        image = np.full((32, 32, 3), 83, dtype=np.uint8)

        result = compensation._compensate_image(image)

        np.testing.assert_array_equal(result, image)

    def test_flat_compensation_parameter_precedes_legacy_tree(self):
        values = {
            "~clip_hist_percent": 2.0,
            "~camera/extrinsic_camera_calibration/clip_hist_percent": 1.0,
        }
        with mock.patch.object(
            compensation_module.rospy,
            "has_param",
            side_effect=lambda name: name == "~clip_hist_percent",
        ), mock.patch.object(
            compensation_module.rospy,
            "get_param",
            side_effect=lambda name, default=None: values.get(name, default),
        ):
            self.assertEqual(
                compensation_module._compensation_param(
                    "clip_hist_percent", 0.0
                ),
                2.0,
            )

        with mock.patch.object(
            compensation_module.rospy, "has_param", return_value=False
        ), mock.patch.object(
            compensation_module.rospy,
            "get_param",
            side_effect=lambda name, default=None: values.get(name, default),
        ):
            self.assertEqual(
                compensation_module._compensation_param(
                    "clip_hist_percent", 0.0
                ),
                1.0,
            )

    def test_default_internal_types_remain_compressed_compatible(self):
        def default_parameter(_name, default=None):
            return default

        with mock.patch.object(
            projection_module.rospy, "has_param", return_value=False
        ), mock.patch.object(
            projection_module.rospy,
            "get_param",
            side_effect=default_parameter,
        ), mock.patch.object(
            projection_module.rospy, "Publisher"
        ) as projection_publisher, mock.patch.object(
            projection_module.rospy, "Subscriber"
        ) as projection_subscriber, mock.patch.object(
            projection_module.rospy, "loginfo"
        ):
            projection = projection_module.ImageProjection()

        self.assertEqual(projection.output_image_type, "compressed")
        self.assertEqual(projection.input_image_type, "compressed")
        projection_publisher.assert_any_call(
            "/camera/image_output/compressed",
            projection_module.CompressedImage,
            queue_size=1,
        )
        projection_subscriber.assert_called_once_with(
            "/camera/image_input/compressed",
            projection_module.CompressedImage,
            projection.cbImageProjection,
            queue_size=1,
            buff_size=projection.RAW_IMAGE_BUFFER_SIZE,
            tcp_nodelay=True,
        )

        with mock.patch.object(
            compensation_module.rospy, "has_param", return_value=False
        ), mock.patch.object(
            compensation_module.rospy,
            "get_param",
            side_effect=default_parameter,
        ), mock.patch.object(
            compensation_module.rospy, "Publisher"
        ) as compensation_publisher, mock.patch.object(
            compensation_module.rospy, "Subscriber"
        ) as compensation_subscriber:
            compensation = compensation_module.ImageCompensation()

        self.assertEqual(compensation.input_image_type, "compressed")
        compensation_publisher.assert_called_once_with(
            "/camera/image_output",
            compensation_module.Image,
            queue_size=1,
        )
        compensation_subscriber.assert_called_once_with(
            "/camera/image_input/compressed",
            compensation_module.CompressedImage,
            compensation.cbImageCompensation,
            queue_size=1,
            buff_size=compensation.RAW_IMAGE_BUFFER_SIZE,
            tcp_nodelay=True,
        )

    def test_callbacks_preserve_input_header(self):
        class RecordingPublisher:
            def __init__(self):
                self.messages = []

            def publish(self, message):
                self.messages.append(message)

        class RecordingBridge:
            @staticmethod
            def cv2_to_compressed_imgmsg(image, dst_format="jpg"):
                return SimpleNamespace(header=None, image=image)

            @staticmethod
            def cv2_to_imgmsg(image, encoding="bgr8"):
                return SimpleNamespace(header=None, image=image)

        image = np.full((240, 320, 3), 83, dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(success)
        header = SimpleNamespace(stamp=123.0, frame_id="camera")
        message = SimpleNamespace(data=encoded.tobytes(), header=header)

        projection = self.make_projection()
        projection.is_calibration_mode = False
        projection.cvBridge = RecordingBridge()
        projection.pub_image_projected = RecordingPublisher()
        projection.cbImageProjection(message)
        self.assertIs(projection.pub_image_projected.messages[0].header, header)

        compensation = compensation_module.ImageCompensation.__new__(
            compensation_module.ImageCompensation
        )
        compensation.clip_hist_percent = 1.0
        compensation.input_image_type = "compressed"
        compensation.cvBridge = RecordingBridge()
        compensation.pub_image_compensated = RecordingPublisher()
        compensation.cbImageCompensation(message)
        self.assertIs(compensation.pub_image_compensated.messages[0].header, header)

    def test_raw_camera_to_compensation_skips_both_lane_jpeg_round_trips(self):
        class RecordingPublisher:
            def __init__(self):
                self.messages = []

            def publish(self, message):
                self.messages.append(message)

        class RawRecordingBridge:
            def __init__(self):
                self.compressed_output_count = 0
                self.raw_output_count = 0
                self.raw_input_count = 0

            def cv2_to_compressed_imgmsg(self, image, dst_format="jpg"):
                self.compressed_output_count += 1
                return SimpleNamespace(header=None, image=image)

            def cv2_to_imgmsg(self, image, encoding="bgr8"):
                self.raw_output_count += 1
                return SimpleNamespace(header=None, image=np.copy(image))

            def imgmsg_to_cv2(self, message, desired_encoding="bgr8"):
                self.raw_input_count += 1
                return np.copy(message.image)

        image = np.full((240, 320, 3), 83, dtype=np.uint8)
        header = SimpleNamespace(stamp=123.0, frame_id="camera")
        raw_input = SimpleNamespace(image=image, header=header)
        bridge = RawRecordingBridge()

        projection = self.make_projection()
        projection.input_image_type = "raw"
        projection.output_image_type = "raw"
        projection.is_calibration_mode = False
        projection.cvBridge = bridge
        projection.pub_image_projected = RecordingPublisher()
        projection.cbImageProjection(raw_input)

        self.assertEqual(bridge.compressed_output_count, 0)
        self.assertEqual(bridge.raw_output_count, 1)
        self.assertEqual(bridge.raw_input_count, 1)
        projected = projection.pub_image_projected.messages[0]
        self.assertIs(projected.header, header)
        self.assertEqual(projected.image.shape, (600, 1000, 3))

        compensation = compensation_module.ImageCompensation.__new__(
            compensation_module.ImageCompensation
        )
        compensation.clip_hist_percent = 1.0
        compensation.input_image_type = "raw"
        compensation.cvBridge = bridge
        compensation.pub_image_compensated = RecordingPublisher()
        compensation.cbImageCompensation(projected)

        self.assertEqual(bridge.compressed_output_count, 0)
        self.assertEqual(bridge.raw_input_count, 2)
        self.assertEqual(bridge.raw_output_count, 2)
        compensated = compensation.pub_image_compensated.messages[0]
        self.assertIs(compensated.header, header)
        self.assertEqual(compensated.image.shape, (600, 1000, 3))

    def test_empty_compressed_frames_are_ignored(self):
        projection = projection_module.ImageProjection.__new__(
            projection_module.ImageProjection
        )
        projection.input_image_type = "compressed"
        compensation = compensation_module.ImageCompensation.__new__(
            compensation_module.ImageCompensation
        )
        compensation.input_image_type = "compressed"
        message = SimpleNamespace(data=b"", header=None)

        with mock.patch.object(projection_module.rospy, "logwarn_throttle"):
            projection.cbImageProjection(message)
        with mock.patch.object(compensation_module.rospy, "logwarn_throttle"):
            compensation.cbImageCompensation(message)

    def test_clipped_range_stretches_nonuniform_image(self):
        compensation = compensation_module.ImageCompensation.__new__(
            compensation_module.ImageCompensation
        )
        compensation.clip_hist_percent = 1.0
        gray = np.tile(np.arange(40, 220, dtype=np.uint8), (16, 1))
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        result = compensation._compensate_image(image)

        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, image.shape)
        self.assertLessEqual(int(result.min()), 1)
        self.assertGreaterEqual(int(result.max()), 254)


if __name__ == "__main__":
    unittest.main()
