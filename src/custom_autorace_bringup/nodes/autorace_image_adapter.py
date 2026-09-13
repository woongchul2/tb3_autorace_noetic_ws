#!/usr/bin/env python3
"""Rectify and resize D405 color images for the AutoRace vision pipeline."""

import copy
import threading

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


class AutoraceImageAdapter:
    RAW_IMAGE_BUFFER_SIZE = 1 << 24

    def __init__(self):
        self.bridge = CvBridge()
        self.opencv_threads = max(
            1, int(rospy.get_param("~opencv_threads", 1))
        )
        cv2.setNumThreads(self.opencv_threads)
        self.output_width = int(rospy.get_param("~output_width", 320))
        self.output_height = int(rospy.get_param("~output_height", 240))
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 90))
        self.output_frame = rospy.get_param("~output_frame", "camera_optical_frame")
        self._rectify_lock = threading.Lock()
        self._rectify_key = None
        self._pending_rectify_key = None
        self._rectify_maps = None

        input_image = rospy.get_param(
            "~input_image", "/camera/color/image_raw"
        )
        input_info = rospy.get_param(
            "~input_camera_info", "/camera/color/camera_info"
        )
        output_topic = rospy.get_param(
            "~output_topic", "/camera/image_rect_color/compressed"
        )
        raw_output_topic = rospy.get_param(
            "~raw_output_topic", "/camera/image_rect_color"
        )

        self.compressed_image_pub = rospy.Publisher(
            output_topic, CompressedImage, queue_size=1
        )
        self.raw_image_pub = rospy.Publisher(
            raw_output_topic, Image, queue_size=1
        )

        rospy.Subscriber(input_info, CameraInfo, self._info_callback, queue_size=1)
        rospy.Subscriber(
            input_image,
            Image,
            self._image_callback,
            queue_size=1,
            buff_size=self.RAW_IMAGE_BUFFER_SIZE,
            tcp_nodelay=True,
        )

    @staticmethod
    def _rectification_key(message):
        roi = getattr(message, "roi", None)
        roi_key = (
            int(getattr(roi, "x_offset", 0)),
            int(getattr(roi, "y_offset", 0)),
            int(getattr(roi, "width", 0)),
            int(getattr(roi, "height", 0)),
            bool(getattr(roi, "do_rectify", False)),
        )
        return (
            int(message.width),
            int(message.height),
            str(message.distortion_model),
            tuple(float(value) for value in message.K),
            tuple(float(value) for value in message.D),
            tuple(float(value) for value in getattr(message, "R", ())),
            tuple(float(value) for value in getattr(message, "P", ())),
            int(getattr(message, "binning_x", 0)),
            int(getattr(message, "binning_y", 0)),
            roi_key,
        )

    def _info_callback(self, message):
        key = self._rectification_key(message)
        with self._rectify_lock:
            if key in (self._rectify_key, self._pending_rectify_key):
                return
            self._pending_rectify_key = key

        maps = None
        if message.width != 0 and message.height != 0 and message.D:
            if message.distortion_model != "plumb_bob":
                rospy.logwarn_once(
                    "D405 distortion model '%s' is not handled; forwarding without software rectification",
                    message.distortion_model,
                )
            else:
                camera_matrix = np.asarray(
                    message.K, dtype=np.float64
                ).reshape(3, 3)
                distortion = np.asarray(message.D, dtype=np.float64)
                if not np.allclose(distortion, 0.0):
                    maps = cv2.initUndistortRectifyMap(
                        camera_matrix,
                        distortion,
                        np.eye(3),
                        camera_matrix,
                        (message.width, message.height),
                        cv2.CV_32FC1,
                    )

        # CameraInfo and image callbacks run independently. Publish the pair as
        # one immutable reference, and do not let a stale calculation replace a
        # newer calibration that arrived while maps were being built.
        with self._rectify_lock:
            if self._pending_rectify_key == key:
                self._rectify_key = key
                self._rectify_maps = maps
                self._pending_rectify_key = None

    def _center_crop_4_by_3(self, image):
        height, width = image.shape[:2]
        target_aspect = 4.0 / 3.0
        current_aspect = float(width) / float(height)
        if current_aspect > target_aspect:
            crop_width = int(round(height * target_aspect))
            x0 = (width - crop_width) // 2
            return image[:, x0:x0 + crop_width]

        crop_height = int(round(width / target_aspect))
        y0 = (height - crop_height) // 2
        return image[y0:y0 + crop_height, :]

    def _image_callback(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as error:
            rospy.logerr_throttle(5.0, "D405 image conversion failed: %s", error)
            return

        with self._rectify_lock:
            rectify_maps = self._rectify_maps
        if (
            rectify_maps is not None
            and rectify_maps[0].shape[:2] == image.shape[:2]
        ):
            image = cv2.remap(
                image,
                rectify_maps[0],
                rectify_maps[1],
                interpolation=cv2.INTER_LINEAR,
            )

        cropped = self._center_crop_4_by_3(image)
        resized = cv2.resize(
            cropped,
            (self.output_width, self.output_height),
            interpolation=cv2.INTER_AREA,
        )

        header = copy.deepcopy(message.header)
        header.frame_id = self.output_frame
        raw = self.bridge.cv2_to_imgmsg(resized, encoding="bgr8")
        raw.header = header
        self.raw_image_pub.publish(raw)

        # Traffic-light detection and the external calibration tools still use
        # the legacy compressed stream.  JPEG is no longer on the lane-path
        # critical path, and it is skipped when that stream has no consumer.
        if self.compressed_image_pub.get_num_connections() == 0:
            return
        success, encoded = cv2.imencode(
            ".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not success:
            rospy.logerr_throttle(5.0, "D405 JPEG encoding failed")
            return
        compressed = CompressedImage()
        compressed.header = header
        compressed.format = "jpeg"
        compressed.data = np.asarray(encoded).tobytes()

        self.compressed_image_pub.publish(compressed)


if __name__ == "__main__":
    rospy.init_node("autorace_image_adapter")
    AutoraceImageAdapter()
    rospy.spin()
