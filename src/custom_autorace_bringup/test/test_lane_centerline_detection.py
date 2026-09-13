#!/usr/bin/env python3

import importlib.machinery
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np


WORKSPACE_SRC = Path(__file__).resolve().parents[2]
DETECT_NODE = (
    WORKSPACE_SRC
    / "turtlebot3_autorace_2020"
    / "turtlebot3_autorace_detect"
    / "nodes"
    / "detect_lane"
)
MESSAGE_FILE = (
    WORKSPACE_SRC
    / "turtlebot3_autorace_2020"
    / "turtlebot3_autorace_msgs"
    / "msg"
    / "LaneCenterline.msg"
)


def load_detect_lane_module():
    loader = importlib.machinery.SourceFileLoader(
        "lane_centerline_detect_node", str(DETECT_NODE)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


detect_lane_module = load_detect_lane_module()


class RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class LaneCenterlineDetectionTest(unittest.TestCase):
    @staticmethod
    def detector(rows=(55, 40, 20, 0)):
        detector = detect_lane_module.DetectLane.__new__(
            detect_lane_module.DetectLane
        )
        detector.centerline_sample_rows = list(rows)
        detector.centerline_support_half_height = 2
        detector.centerline_support_half_width = 10
        detector.centerline_minimum_support_pixels = 3
        detector.centerline_branch_maximum_gap_bands = 1
        detector.centerline_branch_maximum_innovation = 80.0
        detector.centerline_branch_reversal_maximum_innovation = 60.0
        detector.centerline_lane_half_width = 320.0
        detector.centerline_minimum_lane_width = 400.0
        detector.centerline_maximum_lane_width = 900.0
        detector.boundary_sample_y = int(rows[0])
        detector.reliability_yellow_line = 100
        detector.reliability_white_line = 100
        detector.left_fitx = np.full(60, 180.0)
        detector.right_fitx = np.full(60, 820.0)
        detector.cv_image_header = SimpleNamespace(
            stamp=123.0, frame_id="camera_projected"
        )
        detector.pub_lane_centerline = RecordingPublisher()
        detector.pub_lane = RecordingPublisher()
        return detector

    @staticmethod
    def add_support(mask, row, x):
        y0 = max(0, row - 2)
        y1 = min(mask.shape[0], row + 3)
        mask[y0:y1, x - 2:x + 3] = 255

    @staticmethod
    def fit_through_points(height, points):
        ordered = sorted(points, key=lambda point: point[1])
        rows = np.asarray([point[1] for point in ordered], dtype=np.float64)
        columns = np.asarray(
            [point[0] for point in ordered], dtype=np.float64
        )
        return np.interp(np.arange(height, dtype=np.float64), rows, columns)

    def test_message_definition_has_the_requested_typed_fields(self):
        fields = [
            line.strip()
            for line in MESSAGE_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(
            fields,
            [
                "std_msgs/Header header",
                "uint32 image_width",
                "uint32 image_height",
                "float32[] sample_rows",
                "float32[] center_x",
                "float32[] yellow_x",
                "float32[] white_x",
                "float32[] confidence",
                "bool[] yellow_valid",
                "bool[] white_valid",
            ],
        )

    def test_two_supported_boundaries_publish_midpoints_near_to_far(self):
        detector = self.detector()
        yellow = np.zeros((60, 1000), dtype=np.uint8)
        white = np.zeros_like(yellow)
        cv2.line(yellow, (180, 59), (180, 0), 255, 5, cv2.LINE_8)
        cv2.line(white, (820, 59), (820, 0), 255, 5, cv2.LINE_8)

        detector.publish_lane_centerline(white, yellow, True, True)

        message = detector.pub_lane_centerline.messages[-1]
        self.assertIs(message.header, detector.cv_image_header)
        self.assertEqual(message.image_width, 1000)
        self.assertEqual(message.image_height, 60)
        self.assertEqual(list(message.sample_rows), [55.0, 40.0, 20.0, 0.0])
        np.testing.assert_allclose(message.center_x, 500.0)
        np.testing.assert_allclose(message.yellow_x, 180.0)
        np.testing.assert_allclose(message.white_x, 820.0)
        self.assertEqual(list(message.yellow_valid), [True] * 4)
        self.assertEqual(list(message.white_valid), [True] * 4)
        self.assertTrue(all(0.0 < value <= 1.0 for value in message.confidence))
        self.assertEqual(len(detector.pub_lane.messages), 1)
        self.assertAlmostEqual(detector.pub_lane.messages[0].data, 500.0)

    def test_disconnected_far_branch_is_not_reacquired_after_one_gap(self):
        # This reproduces the observed failure shape.  The historical global
        # fit points at x=674 on the far row, but the current-frame near branch
        # has already settled at x=516.  The one missing row band must not let
        # that disconnected far component become the active boundary.
        detector = self.detector(rows=(400, 320, 240, 160, 80, 0))
        detector.centerline_support_half_height = 4
        detector.centerline_branch_maximum_innovation = 120.0
        white = np.zeros((420, 1000), dtype=np.uint8)
        yellow = np.zeros_like(white)
        near_points = [(740, 400), (582, 320), (516, 240), (516, 160)]
        cv2.polylines(
            white,
            [np.asarray(near_points, dtype=np.int32)],
            False,
            255,
            7,
            cv2.LINE_8,
        )
        cv2.line(white, (674, 0), (674, 12), 255, 7, cv2.LINE_8)
        self.assertEqual(cv2.connectedComponents(white, connectivity=8)[0], 3)
        detector.right_fitx = self.fit_through_points(
            white.shape[0], [(674, 0)] + near_points
        )
        detector.left_fitx = np.full(white.shape[0], 180.0)

        detector.publish_lane_centerline(white, yellow, True, False)

        message = detector.pub_lane_centerline.messages[-1]
        self.assertEqual(
            list(message.sample_rows), [400.0, 320.0, 240.0, 160.0]
        )
        np.testing.assert_allclose(
            message.white_x, [740.0, 582.0, 516.0, 516.0], atol=3.0
        )
        self.assertNotIn(674.0, message.white_x)

    def test_one_missing_row_band_reconnects_predicted_component(self):
        detector = self.detector(rows=(55, 45, 35, 25, 15, 5))
        white = np.zeros((60, 1000), dtype=np.uint8)
        yellow = np.zeros_like(white)
        # Components are separated across the complete row-25 support band.
        # Row 15 may reconnect because only one sampled band was missed and it
        # remains on the x prediction from the lower component.
        white[33:60, 818:823] = 255
        white[0:18, 818:823] = 255
        self.assertEqual(cv2.connectedComponents(white, connectivity=8)[0], 3)

        detector.publish_lane_centerline(white, yellow, True, False)

        message = detector.pub_lane_centerline.messages[-1]
        self.assertEqual(
            list(message.sample_rows), [55.0, 45.0, 35.0, 15.0, 5.0]
        )
        np.testing.assert_allclose(message.white_x, 820.0, atol=0.1)

    def test_same_row_competing_cluster_keeps_predicted_branch(self):
        detector = self.detector(rows=(55, 45, 35, 25, 15, 5))
        white = np.zeros((60, 1000), dtype=np.uint8)
        yellow = np.zeros_like(white)
        cv2.line(white, (820, 59), (820, 0), 255, 5, cv2.LINE_8)
        cv2.line(white, (842, 30), (842, 0), 255, 5, cv2.LINE_8)
        # Connect the two candidates outside every sampled local row band.  A
        # global 8-connected label alone therefore cannot distinguish them.
        cv2.line(white, (820, 10), (842, 10), 255, 1, cv2.LINE_8)
        self.assertEqual(cv2.connectedComponents(white, connectivity=8)[0], 2)

        detector.publish_lane_centerline(white, yellow, True, False)

        message = detector.pub_lane_centerline.messages[-1]
        self.assertEqual(
            list(message.sample_rows), [55.0, 45.0, 35.0, 25.0, 15.0, 5.0]
        )
        np.testing.assert_allclose(message.white_x, 820.0, atol=1.0)

    def test_offscreen_prediction_ends_same_component_before_reentry(self):
        detector = self.detector(rows=(55, 45, 35, 25, 15, 5))
        white = np.zeros((60, 1000), dtype=np.uint8)
        yellow = np.zeros_like(white)
        supported_near = [(650, 55), (450, 45), (230, 35), (20, 25)]
        # The paint is one 8-connected component, but after leaving through
        # the image edge it folds back through farther rows.  The returning
        # pixels are a new visible branch, not forward support to reacquire.
        folded_far = [(0, 20), (100, 15), (200, 5)]
        cv2.polylines(
            white,
            [np.asarray(supported_near + folded_far, dtype=np.int32)],
            False,
            255,
            5,
            cv2.LINE_8,
        )
        self.assertEqual(cv2.connectedComponents(white, connectivity=8)[0], 2)
        detector.right_fitx = self.fit_through_points(
            white.shape[0], supported_near
        )

        detector.publish_lane_centerline(white, yellow, True, False)

        message = detector.pub_lane_centerline.messages[-1]
        self.assertEqual(
            list(message.sample_rows), [55.0, 45.0, 35.0, 25.0]
        )
        self.assertTrue(np.all(np.diff(message.white_x) < 0.0))
        self.assertLess(message.white_x[-1], 60.0)
        self.assertNotIn(100.0, message.white_x)
        self.assertNotIn(200.0, message.white_x)

    def test_run5_ninety_degree_curve_keeps_each_visible_colour_branch(self):
        """Preserve the complementary colour support seen at the run5 stop.

        In the recorded projected frames, yellow supplies the near half of
        the bend and white supplies its middle/far half.  Yellow's row-320
        tangent predicts row 240 just outside the image (about x=1014), while
        the same connected paint remains visible at x=974.  That 40 px
        innovation is valid current-frame support, not a returning component.
        """
        rows = (552, 480, 400, 320, 240, 160, 80, 0)
        detector = self.detector(rows=rows)
        detector.centerline_support_half_height = 12
        detector.centerline_support_half_width = 40
        detector.centerline_branch_maximum_innovation = 80.0
        yellow = np.zeros((600, 1000), dtype=np.uint8)
        white = np.zeros_like(yellow)
        yellow_points = [
            (303, 552),
            (410, 480),
            (575, 400),
            (794, 320),
            (974, 240),
        ]
        white_points = [
            (220, 320),
            (443, 240),
            (667, 160),
            (892, 80),
        ]
        cv2.polylines(
            yellow,
            [np.asarray(yellow_points, dtype=np.int32)],
            False,
            255,
            9,
            cv2.LINE_8,
        )
        cv2.polylines(
            white,
            [np.asarray(white_points, dtype=np.int32)],
            False,
            255,
            9,
            cv2.LINE_8,
        )
        detector.left_fitx = self.fit_through_points(
            yellow.shape[0], yellow_points
        )
        detector.right_fitx = self.fit_through_points(
            white.shape[0], white_points
        )

        detector.publish_lane_centerline(white, yellow, True, True)

        message = detector.pub_lane_centerline.messages[-1]
        self.assertEqual(
            list(message.sample_rows), [float(row) for row in rows[:-1]]
        )
        self.assertEqual(
            list(message.yellow_valid),
            [True, True, True, True, True, False, False],
        )
        self.assertEqual(
            list(message.white_valid),
            [False, False, False, True, True, True, True],
        )
        np.testing.assert_allclose(
            np.asarray(message.yellow_x[:5]),
            [303.0, 410.0, 575.0, 794.0, 974.0],
            atol=7.0,
        )
        np.testing.assert_allclose(
            np.asarray(message.white_x[3:]),
            [220.0, 443.0, 667.0, 892.0],
            atol=7.0,
        )

    def test_large_innovation_cannot_create_an_unobserved_reversal(self):
        detector = self.detector(rows=(55, 45, 35, 25, 15, 5))
        white = np.zeros((60, 1000), dtype=np.uint8)
        yellow = np.zeros_like(white)
        near_curve = [(625, 55), (545, 45), (524, 35)]
        # Local band fitting pulls the emitted cluster centre inward by a few
        # pixels.  Keep the first returning observation inside the general
        # 80 px innovation gate but decisively outside the 60 px reversal
        # gate so this fixture exercises the stricter branch-direction rule.
        returning_branch = [(600, 25), (670, 15), (740, 5)]
        cv2.polylines(
            white,
            [np.asarray(near_curve + returning_branch, dtype=np.int32)],
            False,
            255,
            5,
            cv2.LINE_8,
        )
        detector.right_fitx = self.fit_through_points(
            white.shape[0], near_curve
        )

        detector.publish_lane_centerline(white, yellow, True, False)

        message = detector.pub_lane_centerline.messages[-1]
        self.assertEqual(list(message.sample_rows), [55.0, 45.0, 35.0])
        self.assertTrue(np.all(np.diff(message.white_x) < 0.0))
        self.assertNotIn(600.0, message.white_x)

    def test_connected_s_curve_preserves_real_slope_reversal(self):
        detector = self.detector(rows=(55, 45, 35, 25, 15, 5))
        white = np.zeros((60, 1000), dtype=np.uint8)
        yellow = np.zeros_like(white)
        s_points = [
            (820, 55),
            (800, 45),
            (790, 35),
            (800, 25),
            (820, 15),
            (830, 5),
        ]
        cv2.polylines(
            white,
            [np.asarray(s_points, dtype=np.int32)],
            False,
            255,
            5,
            cv2.LINE_8,
        )
        self.assertEqual(cv2.connectedComponents(white, connectivity=8)[0], 2)
        detector.right_fitx = self.fit_through_points(white.shape[0], s_points)

        detector.publish_lane_centerline(white, yellow, True, False)

        message = detector.pub_lane_centerline.messages[-1]
        self.assertEqual(
            list(message.sample_rows), [55.0, 45.0, 35.0, 25.0, 15.0, 5.0]
        )
        np.testing.assert_allclose(
            message.white_x, [820.0, 800.0, 790.0, 800.0, 820.0, 830.0],
            atol=3.0,
        )
        self.assertLess(message.white_x[2], message.white_x[1])
        self.assertGreater(message.white_x[3], message.white_x[2])

    def test_one_boundary_uses_explicit_half_width_only_at_supported_rows(self):
        detector = self.detector(rows=(40, 20, 0))
        yellow = np.zeros((60, 1000), dtype=np.uint8)
        white = np.zeros_like(yellow)
        self.add_support(yellow, 40, 180)
        self.add_support(white, 20, 820)

        detector.publish_lane_centerline(white, yellow, True, True)

        message = detector.pub_lane_centerline.messages[-1]
        self.assertEqual(list(message.sample_rows), [40.0, 20.0])
        np.testing.assert_allclose(message.center_x, [500.0, 500.0])
        self.assertEqual(list(message.yellow_valid), [True, False])
        self.assertEqual(list(message.white_valid), [False, True])
        self.assertTrue(math.isnan(message.white_x[0]))
        self.assertTrue(math.isnan(message.yellow_x[1]))
        # The unsupported far row must not be filled from the polynomial fit.
        self.assertNotIn(0.0, message.sample_rows)

    def test_local_pixels_far_from_fit_do_not_authorize_extrapolation(self):
        detector = self.detector(rows=(20,))
        yellow = np.zeros((60, 1000), dtype=np.uint8)
        white = np.zeros_like(yellow)
        self.add_support(yellow, 20, 40)
        self.add_support(white, 20, 960)

        detector.publish_lane_centerline(white, yellow, True, True)

        self.assertEqual(
            list(detector.pub_lane_centerline.messages[-1].sample_rows), []
        )

    def test_inconsistent_horizontal_width_preserves_both_raw_boundaries(self):
        detector = self.detector(rows=(40,))
        detector.left_fitx[:] = 300.0
        detector.right_fitx[:] = 650.0
        yellow = np.zeros((60, 1000), dtype=np.uint8)
        white = np.zeros_like(yellow)
        self.add_support(yellow, 40, 300)
        self.add_support(white, 40, 650)

        detector.publish_lane_centerline(white, yellow, True, True)

        message = detector.pub_lane_centerline.messages[-1]
        self.assertEqual(list(message.sample_rows), [40.0])
        self.assertTrue(math.isnan(message.center_x[0]))
        self.assertAlmostEqual(message.yellow_x[0], 300.0)
        self.assertAlmostEqual(message.white_x[0], 650.0)
        self.assertTrue(message.yellow_valid[0])
        self.assertTrue(message.white_valid[0])
        # A same-row horizontal-width mismatch is not allowed onto the scalar
        # compatibility topic, but neither locally measured boundary is lost.
        self.assertEqual(detector.pub_lane.messages, [])

    def test_full_height_reliability_does_not_hide_supported_curve_rows(self):
        detector = self.detector(rows=(50,))
        detector.reliability_yellow_line = 0
        detector.reliability_white_line = 0
        white = np.zeros((60, 1000), dtype=np.uint8)
        yellow = np.zeros_like(white)
        self.add_support(white, 50, 820)

        detector.publish_lane_centerline(white, yellow, True, False)

        message = detector.pub_lane_centerline.messages[-1]
        self.assertEqual(list(message.sample_rows), [50.0])
        self.assertTrue(message.white_valid[0])
        self.assertAlmostEqual(message.center_x[0], 500.0)
        self.assertEqual(len(detector.pub_lane.messages), 1)
        self.assertAlmostEqual(detector.pub_lane.messages[0].data, 500.0)

    def test_supported_single_boundary_is_kept_when_pixel_centre_is_outside(self):
        detector = self.detector(rows=(40,))
        detector.right_fitx[:] = 200.0
        white = np.zeros((60, 1000), dtype=np.uint8)
        yellow = np.zeros_like(white)
        self.add_support(white, 40, 200)

        detector.publish_lane_centerline(white, yellow, True, False)

        message = detector.pub_lane_centerline.messages[-1]
        self.assertEqual(list(message.sample_rows), [40.0])
        self.assertFalse(message.yellow_valid[0])
        self.assertTrue(message.white_valid[0])
        self.assertAlmostEqual(message.white_x[0], 200.0)
        self.assertAlmostEqual(message.center_x[0], -120.0)
        self.assertEqual(detector.pub_lane.messages, [])

    def test_callback_converts_bgr_to_hsv_once_and_drops_duplicate_frame(self):
        detector = detect_lane_module.DetectLane.__new__(
            detect_lane_module.DetectLane
        )
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        detector.counter = 1
        detector.process_every_n_frames = 1
        detector.temporal_state_counter = 0
        detector.temporal_state_every_n_frames = 3
        detector.update_temporal_state = False
        detector.debug_every_n_frames = 3
        detector.debug_frame_counter = 0
        detector.last_input_stamp_ns = None
        detector.sub_image_type = "raw"
        detector.is_calibration_mode = False
        detector.pub_image_lane = SimpleNamespace(
            get_num_connections=lambda: 0
        )
        detector.cvBridge = SimpleNamespace(
            imgmsg_to_cv2=lambda _message, _encoding: image
        )
        detector.minimum_lane_pixels = 3
        detector.left_fitx = detector.right_fitx = None
        detector.left_fit = detector.right_fit = None
        detector.left_fit_history = detector.right_fit_history = []
        detector.publish_lane_boundaries = mock.Mock()
        detector.publish_lane_centerline = mock.Mock()
        detector.make_lane = mock.Mock()
        detector.maskWhiteLane = mock.Mock(return_value=(0, mask))
        detector.maskYellowLane = mock.Mock(return_value=(0, mask))
        message = SimpleNamespace(header=SimpleNamespace(stamp=1.0))

        original_cvt_color = cv2.cvtColor
        with mock.patch.object(
            detect_lane_module.cv2,
            "cvtColor",
            wraps=original_cvt_color,
        ) as convert, mock.patch.object(
            detect_lane_module.rospy,
            "logwarn_throttle",
        ):
            detector.cbFindLane(message)
            detector.cbFindLane(message)
            message.header.stamp = 1.0 + 1.0 / 30.0
            detector.cbFindLane(message)

        self.assertEqual(convert.call_count, 2)
        white_hsv = detector.maskWhiteLane.call_args.args[1]
        yellow_hsv = detector.maskYellowLane.call_args.args[1]
        self.assertIs(white_hsv, yellow_hsv)
        self.assertEqual(detector.publish_lane_centerline.call_count, 2)
        detector.publish_lane_centerline.assert_called_with(
            mask, mask, False, False
        )

    def test_callback_throttles_only_debug_image_not_control_outputs(self):
        detector = detect_lane_module.DetectLane.__new__(
            detect_lane_module.DetectLane
        )
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        detector.counter = 1
        detector.process_every_n_frames = 1
        detector.temporal_state_counter = 0
        detector.temporal_state_every_n_frames = 3
        detector.update_temporal_state = False
        detector.debug_every_n_frames = 3
        detector.debug_frame_counter = 0
        detector.last_input_stamp_ns = None
        detector.sub_image_type = "raw"
        detector.is_calibration_mode = False
        detector.pub_image_lane = SimpleNamespace(
            get_num_connections=lambda: 1
        )
        detector.cvBridge = SimpleNamespace(
            imgmsg_to_cv2=lambda _message, _encoding: image
        )
        detector.minimum_lane_pixels = 3
        detector.left_fitx = detector.right_fitx = None
        detector.left_fit = detector.right_fit = None
        detector.left_fit_history = detector.right_fit_history = []
        detector.publish_lane_boundaries = mock.Mock()
        detector.publish_lane_centerline = mock.Mock()
        detector.make_lane = mock.Mock()
        detector.maskWhiteLane = mock.Mock(return_value=(0, mask))
        detector.maskYellowLane = mock.Mock(return_value=(0, mask))
        message = SimpleNamespace(header=SimpleNamespace(stamp=1.0))

        for index in range(3):
            message.header.stamp = 1.0 + index / 30.0
            detector.cbFindLane(message)

        self.assertEqual(detector.publish_lane_centerline.call_count, 3)
        self.assertEqual(detector.publish_lane_boundaries.call_count, 3)
        self.assertEqual(detector.make_lane.call_count, 1)

        detector.debug_every_n_frames = 0
        message.header.stamp = 1.0 + 3.0 / 30.0
        detector.cbFindLane(message)
        self.assertEqual(detector.publish_lane_centerline.call_count, 4)
        self.assertEqual(detector.make_lane.call_count, 1)

        detector.is_calibration_mode = True
        message.header.stamp = 1.0 + 4.0 / 30.0
        detector.cbFindLane(message)
        self.assertEqual(detector.publish_lane_centerline.call_count, 5)
        self.assertEqual(detector.make_lane.call_count, 2)


if __name__ == "__main__":
    unittest.main()
