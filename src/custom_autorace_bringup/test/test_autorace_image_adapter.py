#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


NODE = Path(__file__).resolve().parents[1] / "nodes" / "autorace_image_adapter.py"
SPEC = importlib.util.spec_from_file_location("autorace_image_adapter", NODE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RecordingPublisher:
    def __init__(self, connections=0):
        self.connections = connections
        self.messages = []

    def get_num_connections(self):
        return self.connections

    def publish(self, message):
        self.messages.append(message)


class RecordingBridge:
    def __init__(self, image):
        self.image = image

    def imgmsg_to_cv2(self, _message, desired_encoding="bgr8"):
        self.asserted_input_encoding = desired_encoding
        return np.copy(self.image)

    def cv2_to_imgmsg(self, image, encoding="bgr8"):
        return SimpleNamespace(
            header=None, image=np.copy(image), encoding=encoding
        )


class AutoraceImageAdapterTest(unittest.TestCase):
    @staticmethod
    def make_adapter(compressed_connections):
        source = np.full((720, 1280, 3), 81, dtype=np.uint8)
        adapter = MODULE.AutoraceImageAdapter.__new__(
            MODULE.AutoraceImageAdapter
        )
        adapter.bridge = RecordingBridge(source)
        adapter.output_width = 320
        adapter.output_height = 240
        adapter.jpeg_quality = 90
        adapter.output_frame = "camera_optical_frame"
        adapter._rectify_lock = threading.Lock()
        adapter._rectify_key = None
        adapter._pending_rectify_key = None
        adapter._rectify_maps = None
        adapter.raw_image_pub = RecordingPublisher()
        adapter.compressed_image_pub = RecordingPublisher(
            compressed_connections
        )
        return adapter

    @staticmethod
    def camera_info(width=1280, height=720, distortion=0.1):
        return SimpleNamespace(
            width=width,
            height=height,
            distortion_model="plumb_bob",
            K=[600.0, 0.0, width / 2.0, 0.0, 600.0, height / 2.0, 0.0, 0.0, 1.0],
            D=[distortion, -0.01, 0.0, 0.0, 0.0],
            R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            P=[600.0, 0.0, width / 2.0, 0.0, 0.0, 600.0, height / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            binning_x=0,
            binning_y=0,
            roi=SimpleNamespace(
                x_offset=0,
                y_offset=0,
                width=0,
                height=0,
                do_rectify=False,
            ),
        )

    def test_identical_camera_info_reuses_one_rectification_map_pair(self):
        adapter = self.make_adapter(compressed_connections=0)
        info = self.camera_info()
        map_x = np.full((720, 1280), 1.0, dtype=np.float32)
        map_y = np.full((720, 1280), 2.0, dtype=np.float32)

        with mock.patch.object(
            MODULE.cv2,
            "initUndistortRectifyMap",
            return_value=(map_x, map_y),
        ) as initialize_maps:
            adapter._info_callback(info)
            first_pair = adapter._rectify_maps
            adapter._info_callback(info)

        initialize_maps.assert_called_once()
        self.assertIs(adapter._rectify_maps, first_pair)
        self.assertIs(adapter._rectify_maps[0], map_x)
        self.assertIs(adapter._rectify_maps[1], map_y)

    def test_newer_camera_info_wins_if_an_old_map_build_finishes_late(self):
        adapter = self.make_adapter(compressed_connections=0)
        old_info = self.camera_info(distortion=0.10)
        new_info = self.camera_info(distortion=0.20)
        old_maps = (
            np.full((720, 1280), 1.0, dtype=np.float32),
            np.full((720, 1280), 2.0, dtype=np.float32),
        )
        new_maps = (
            np.full((720, 1280), 3.0, dtype=np.float32),
            np.full((720, 1280), 4.0, dtype=np.float32),
        )
        old_started = threading.Event()
        release_old = threading.Event()

        def build_maps(_camera, distortion, *_args):
            if float(distortion[0]) == 0.10:
                old_started.set()
                release_old.wait(2.0)
                return old_maps
            return new_maps

        with mock.patch.object(
            MODULE.cv2,
            "initUndistortRectifyMap",
            side_effect=build_maps,
        ):
            old_thread = threading.Thread(
                target=adapter._info_callback, args=(old_info,)
            )
            old_thread.start()
            self.assertTrue(old_started.wait(1.0))
            adapter._info_callback(new_info)
            release_old.set()
            old_thread.join(1.0)

        self.assertFalse(old_thread.is_alive())
        self.assertIs(adapter._rectify_maps, new_maps)
        self.assertEqual(
            adapter._rectify_key,
            adapter._rectification_key(new_info),
        )

    def test_raw_lane_frame_is_published_without_running_jpeg_encoder(self):
        adapter = self.make_adapter(compressed_connections=0)
        original_header = SimpleNamespace(stamp=12.3, frame_id="d405")

        with mock.patch.object(MODULE.cv2, "imencode") as encoder:
            adapter._image_callback(SimpleNamespace(header=original_header))

        encoder.assert_not_called()
        self.assertEqual(len(adapter.raw_image_pub.messages), 1)
        self.assertEqual(len(adapter.compressed_image_pub.messages), 0)
        raw = adapter.raw_image_pub.messages[0]
        self.assertEqual(raw.image.shape, (240, 320, 3))
        self.assertEqual(raw.encoding, "bgr8")
        self.assertEqual(raw.header.stamp, 12.3)
        self.assertEqual(raw.header.frame_id, "camera_optical_frame")
        self.assertEqual(original_header.frame_id, "d405")

    def test_connected_legacy_stream_uses_the_same_resized_frame_header(self):
        adapter = self.make_adapter(compressed_connections=1)
        original_header = SimpleNamespace(stamp=45.6, frame_id="d405")

        adapter._image_callback(SimpleNamespace(header=original_header))

        self.assertEqual(len(adapter.raw_image_pub.messages), 1)
        self.assertEqual(len(adapter.compressed_image_pub.messages), 1)
        raw = adapter.raw_image_pub.messages[0]
        compressed = adapter.compressed_image_pub.messages[0]
        self.assertEqual(raw.image.shape, (240, 320, 3))
        self.assertEqual(compressed.header.stamp, raw.header.stamp)
        self.assertEqual(compressed.header.frame_id, raw.header.frame_id)
        self.assertGreater(len(compressed.data), 0)


if __name__ == "__main__":
    unittest.main()
