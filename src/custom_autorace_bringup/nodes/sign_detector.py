#!/usr/bin/env python3
"""Template-based detector shared by AutoRace sign-consuming missions.

One node subscribes to the camera, compares the frame with only the templates
needed by the current mission, and publishes one typed ``TrafficSign`` result.
New missions add a template entry and a mission filter in
``sign_detector.yaml`` instead of adding another image subscriber or a
mission-specific sign topic.
"""

from collections import namedtuple
import os

import cv2
import numpy as np
import rospkg
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import String

from custom_autorace_bringup.msg import TrafficSign


class SignDetector:
    """Classify signs with ORB feature matching and RANSAC homography."""

    RAW_IMAGE_BUFFER_SIZE = 1 << 24

    Detection = namedtuple(
        "Detection", "sign_type direction box confidence priority label"
    )
    Template = namedtuple(
        "Template",
        (
            "label sign_type direction image keypoints descriptors "
            "min_good_matches min_inliers priority"
        ),
    )

    TEMPLATE_TYPES = {
        "intersection_warning": (
            TrafficSign.INTERSECTION_WARNING,
            TrafficSign.DIRECTION_NONE,
        ),
        "construction_warning": (
            TrafficSign.CONSTRUCTION_WARNING,
            TrafficSign.DIRECTION_NONE,
        ),
        "tunnel_warning": (
            TrafficSign.TUNNEL_WARNING,
            TrafficSign.DIRECTION_NONE,
        ),
        "parking": (TrafficSign.PARKING, TrafficSign.DIRECTION_NONE),
        "level_crossing_warning": (
            TrafficSign.LEVEL_CROSSING_WARNING,
            TrafficSign.DIRECTION_NONE,
        ),
        "direction_left": (TrafficSign.DIRECTION, TrafficSign.DIRECTION_LEFT),
        "direction_right": (
            TrafficSign.DIRECTION,
            TrafficSign.DIRECTION_RIGHT,
        ),
    }

    def __init__(self):
        self.bridge = CvBridge()
        self.frame_count = 0
        self.processed_frame_count = 0
        prefix = "~sign_detector/"
        get = rospy.get_param

        self.opencv_threads = max(
            1, int(get(prefix + "opencv_threads", 1))
        )
        cv2.setNumThreads(self.opencv_threads)

        self.process_every_n_frames = max(
            1, int(get(prefix + "process_every_n_frames", 1))
        )
        # ORB cost grows with the input area. Keep the camera's aspect ratio and
        # only cap its width so a 1280x720 D405 frame is classified at 448x252.
        # A zero value disables resizing, and smaller Gazebo images are never
        # enlarged.
        self.processing_width = max(
            0, int(get(prefix + "processing_width", 448))
        )
        self.debug_every_n_frames = max(
            1, int(get(prefix + "debug_every_n_frames", 3))
        )
        self.minimum_confidence = float(get(prefix + "minimum_confidence", 0.45))
        self.input_topic = get(prefix + "input_topic", "/camera/color/image_raw")
        self.output_topic = get(prefix + "output_topic", "/detect/signs")
        self.debug_topic = get(prefix + "debug_image_topic", "/detect/image_signs")
        self.mission_gating_enabled = bool(
            get(prefix + "mission_gating/enabled", False)
        )
        self.mission_topic = str(
            get(prefix + "mission_gating/topic", "/mission/current")
        ).strip()
        active_missions = get(
            prefix + "mission_gating/active_missions",
            ["intersection", "tunnel"],
        )
        if isinstance(active_missions, str):
            active_missions = [active_missions]
        self.active_missions = frozenset(
            str(value).strip()
            for value in active_missions
            if str(value).strip()
        )
        if self.mission_gating_enabled and (
            not self.mission_topic or not self.active_missions
        ):
            raise rospy.ROSInitException(
                "sign mission gating needs a topic and active missions"
            )
        self.current_mission = None

        self.template_filter_fail_open_without_mission = bool(
            get(
                prefix
                + "mission_gating/template_filter/fail_open_without_mission",
                False,
            )
        )
        standalone_labels = get(
            prefix + "mission_gating/template_filter/standalone_templates",
            ["direction_left", "direction_right", "tunnel_warning"],
        )
        self.standalone_template_labels = self._parse_template_labels(
            standalone_labels,
            "standalone_templates",
        )
        mission_labels = get(
            prefix + "mission_gating/template_filter/mission_templates",
            {
                "intersection": ["direction_left", "direction_right"],
                "tunnel": ["tunnel_warning"],
            },
        )
        if not isinstance(mission_labels, dict):
            raise rospy.ROSInitException(
                "sign mission template filters must be a mapping"
            )
        self.mission_template_labels = {
            str(mission).strip(): self._parse_template_labels(
                labels,
                "mission_templates/%s" % mission,
            )
            for mission, labels in mission_labels.items()
            if str(mission).strip()
        }

        template_prefix = prefix + "classifiers/template/"
        self.template_enabled = bool(get(template_prefix + "enabled", True))
        self.ratio_test = float(get(template_prefix + "ratio_test", 0.75))
        self.ransac_threshold = float(
            get(template_prefix + "ransac_reprojection_threshold", 4.0)
        )
        self.minimum_inlier_ratio = float(
            get(template_prefix + "minimum_inlier_ratio", 0.45)
        )
        self.minimum_projected_area = float(
            get(template_prefix + "minimum_projected_area", 80.0)
        )
        self.maximum_projected_area_ratio = float(
            get(template_prefix + "maximum_projected_area_ratio", 0.50)
        )
        self.projection_margin_ratio = float(
            get(template_prefix + "projection_margin_ratio", 0.20)
        )
        nfeatures = int(get(template_prefix + "orb_nfeatures", 1200))
        fast_threshold = int(get(template_prefix + "orb_fast_threshold", 5))
        self.orb = cv2.ORB_create(
            nfeatures=max(100, nfeatures),
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=5,
            patchSize=31,
            fastThreshold=max(0, fast_threshold),
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        direction_prefix = prefix + "classifiers/direction_geometry/"
        self.direction_geometry_enabled = bool(
            get(direction_prefix + "enabled", True)
        )
        self.direction_blue_lower = np.asarray(
            get(direction_prefix + "blue_hsv_lower", [90, 70, 30]),
            dtype=np.uint8,
        )
        self.direction_blue_upper = np.asarray(
            get(direction_prefix + "blue_hsv_upper", [140, 255, 255]),
            dtype=np.uint8,
        )
        self.direction_white_saturation_max = int(
            get(direction_prefix + "white_saturation_max", 90)
        )
        self.direction_white_value_min = int(
            get(direction_prefix + "white_value_min", 120)
        )
        self.direction_minimum_blue_area_ratio = float(
            get(direction_prefix + "minimum_blue_area_ratio", 0.015)
        )
        self.direction_maximum_blue_area_ratio = float(
            get(direction_prefix + "maximum_blue_area_ratio", 0.50)
        )
        self.direction_minimum_hull_circularity = float(
            get(direction_prefix + "minimum_hull_circularity", 0.70)
        )
        self.direction_polygon_epsilon_ratio = max(
            0.0,
            float(get(direction_prefix + "polygon_epsilon_ratio", 0.01)),
        )
        self.direction_minimum_polygon_vertices = max(
            3,
            int(get(direction_prefix + "minimum_polygon_vertices", 8)),
        )
        self.direction_minimum_aspect_ratio = float(
            get(direction_prefix + "minimum_aspect_ratio", 0.75)
        )
        self.direction_maximum_aspect_ratio = float(
            get(direction_prefix + "maximum_aspect_ratio", 2.00)
        )
        self.direction_interior_erode_ratio = float(
            get(direction_prefix + "interior_erode_ratio", 0.08)
        )
        self.direction_minimum_arrow_pixels = int(
            get(direction_prefix + "minimum_arrow_pixels", 200)
        )
        self.direction_asymmetry_deadband = float(
            get(direction_prefix + "asymmetry_deadband", 0.03)
        )

        self.templates = []
        if self.template_enabled:
            self._load_templates(template_prefix)

        self.sign_pub = rospy.Publisher(
            self.output_topic, TrafficSign, queue_size=1
        )
        self.debug_pub = rospy.Publisher(self.debug_topic, Image, queue_size=1)
        if self.mission_gating_enabled:
            rospy.Subscriber(
                self.mission_topic,
                String,
                self.mission_callback,
                queue_size=1,
            )
        rospy.Subscriber(
            self.input_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=self.RAW_IMAGE_BUFFER_SIZE,
            tcp_nodelay=True,
        )
        rospy.loginfo(
            "Unified sign detector input=%s output=%s templates=%d processing_width=%d",
            self.input_topic,
            self.output_topic,
            len(self.templates),
            self.processing_width,
        )

    @staticmethod
    def _resolve_template_path(template_directory, configured_file, rospack):
        configured_file = str(configured_file)
        prefix = "package://"
        if not configured_file.startswith(prefix):
            return os.path.join(template_directory, configured_file)

        resource = configured_file[len(prefix) :]
        package_name, separator, relative_path = resource.partition("/")
        if not separator or not package_name or not relative_path:
            raise ValueError("invalid package URI: %s" % configured_file)
        return os.path.join(rospack.get_path(package_name), relative_path)

    @classmethod
    def _parse_template_labels(cls, configured, parameter_name):
        if isinstance(configured, str):
            configured = [configured]
        if not isinstance(configured, (list, tuple, set, frozenset)):
            raise rospy.ROSInitException(
                "sign template filter %s must be a list" % parameter_name
            )
        labels = frozenset(
            str(value).strip()
            for value in configured
            if str(value).strip()
        )
        unknown = sorted(labels.difference(cls.TEMPLATE_TYPES))
        if unknown:
            raise rospy.ROSInitException(
                "sign template filter %s has unknown labels: %s"
                % (parameter_name, ", ".join(unknown))
            )
        return labels

    def _load_templates(self, template_prefix):
        package_name = rospy.get_param(
            template_prefix + "package", "turtlebot3_autorace_detect"
        )
        relative_directory = rospy.get_param(
            template_prefix + "directory", "image"
        )
        configured = rospy.get_param(template_prefix + "entries", {})
        rospack = rospkg.RosPack()
        try:
            package_path = rospack.get_path(package_name)
        except rospkg.ResourceNotFound as error:
            rospy.logfatal("Sign template package is missing: %s", error)
            raise
        template_directory = os.path.join(package_path, relative_directory)

        for label, config in configured.items():
            if not bool(config.get("enabled", True)):
                continue
            if label not in self.TEMPLATE_TYPES:
                rospy.logwarn("Ignoring unknown sign template label: %s", label)
                continue
            configured_file = str(config.get("file", ""))
            try:
                path = self._resolve_template_path(
                    template_directory, configured_file, rospack
                )
            except (rospkg.ResourceNotFound, ValueError) as error:
                message = "Cannot resolve sign template %s: %s" % (
                    label,
                    error,
                )
                rospy.logfatal(message)
                raise rospy.ROSInitException(message)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                if configured_file.startswith("package://"):
                    message = "Cannot read package sign template %s: %s" % (
                        label,
                        path,
                    )
                    rospy.logfatal(message)
                    raise rospy.ROSInitException(message)
                rospy.logerr("Cannot read sign template: %s", path)
                continue
            keypoints, descriptors = self.orb.detectAndCompute(image, None)
            if descriptors is None or len(keypoints) < 4:
                rospy.logerr("Not enough ORB features in sign template: %s", path)
                continue
            sign_type, direction = self.TEMPLATE_TYPES[label]
            self.templates.append(
                self.Template(
                    label=label,
                    sign_type=sign_type,
                    direction=direction,
                    image=image,
                    keypoints=keypoints,
                    descriptors=descriptors,
                    min_good_matches=max(
                        4, int(config.get("min_good_matches", 8))
                    ),
                    min_inliers=max(4, int(config.get("min_inliers", 5))),
                    priority=int(config.get("priority", 10)),
                )
            )
            rospy.loginfo(
                "Loaded sign template %-24s features=%d",
                label,
                len(keypoints),
            )

        if not self.templates:
            rospy.logwarn("No sign templates are active; /detect/signs will be NONE")

    def _projected_box(
        self,
        template,
        homography,
        frame_shape,
        scale_x=1.0,
        scale_y=1.0,
    ):
        template_height, template_width = template.image.shape[:2]
        corners = np.float32(
            [
                [0, 0],
                [template_width - 1, 0],
                [template_width - 1, template_height - 1],
                [0, template_height - 1],
            ]
        ).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        if not np.all(np.isfinite(projected)):
            return None

        frame_height, frame_width = frame_shape[:2]
        margin_x = frame_width * self.projection_margin_ratio
        margin_y = frame_height * self.projection_margin_ratio
        if (
            projected[:, 0].min() < -margin_x
            or projected[:, 0].max() > frame_width + margin_x
            or projected[:, 1].min() < -margin_y
            or projected[:, 1].max() > frame_height + margin_y
        ):
            return None

        polygon = np.rint(projected).astype(np.int32)
        if not cv2.isContourConvex(polygon):
            return None
        area = abs(float(cv2.contourArea(projected.astype(np.float32))))
        frame_area = float(frame_width * frame_height)
        minimum_area = self.minimum_projected_area * scale_x * scale_y
        if (
            area < minimum_area
            or area > frame_area * self.maximum_projected_area_ratio
        ):
            return None

        x, y, width, height = cv2.boundingRect(polygon)
        left = max(0, x)
        top = max(0, y)
        right = min(frame_width, x + width)
        bottom = min(frame_height, y + height)
        if right <= left or bottom <= top:
            return None
        return left, top, right - left, bottom - top

    def _match_template(
        self,
        template,
        frame_keypoints,
        frame_descriptors,
        shape,
        scale_x=1.0,
        scale_y=1.0,
    ):
        try:
            pairs = self.matcher.knnMatch(
                template.descriptors, frame_descriptors, k=2
            )
        except cv2.error:
            return None
        good = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < self.ratio_test * second.distance
        ]
        if len(good) < template.min_good_matches:
            return None

        template_points = np.float32(
            [template.keypoints[match.queryIdx].pt for match in good]
        ).reshape(-1, 1, 2)
        frame_points = np.float32(
            [frame_keypoints[match.trainIdx].pt for match in good]
        ).reshape(-1, 1, 2)
        homography, mask = cv2.findHomography(
            template_points,
            frame_points,
            cv2.RANSAC,
            max(0.1, self.ransac_threshold * min(scale_x, scale_y)),
        )
        if homography is None or mask is None:
            return None
        inliers = int(mask.ravel().sum())
        inlier_ratio = float(inliers) / float(len(good))
        if (
            inliers < template.min_inliers
            or inlier_ratio < self.minimum_inlier_ratio
        ):
            return None
        box = self._projected_box(
            template,
            homography,
            shape,
            scale_x,
            scale_y,
        )
        if box is None:
            return None

        # Absolute inlier count has more weight than ratio: a coincidental set
        # of four perfect matches must not beat a well-supported detection.
        support = min(1.0, inliers / float(2 * template.min_inliers))
        confidence = min(1.0, 0.70 * support + 0.30 * inlier_ratio)
        return self.Detection(
            template.sign_type,
            template.direction,
            box,
            confidence,
            template.priority,
            template.label,
        )

    def _detect_direction_geometry(self, image, scale_x=1.0, scale_y=1.0):
        """Classify a close blue direction sign even when its rim is clipped.

        A left arrow has its vertical stem and most white pixels on the right
        half of the blue disk; a right arrow has the opposite distribution.
        The convex-hull checks reject small distant signs and non-circular blue
        objects before that asymmetry is evaluated.
        """
        if not self.direction_geometry_enabled:
            return None

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(
            hsv, self.direction_blue_lower, self.direction_blue_upper
        )
        contours, _ = cv2.findContours(
            blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        frame_area = float(image.shape[0] * image.shape[1])
        best = None
        for contour in contours:
            hull = cv2.convexHull(contour)
            area = float(cv2.contourArea(hull))
            area_ratio = area / frame_area
            if not (
                self.direction_minimum_blue_area_ratio
                <= area_ratio
                <= self.direction_maximum_blue_area_ratio
            ):
                continue

            x, y, width, height = cv2.boundingRect(hull)
            if height <= 0:
                continue
            aspect_ratio = float(width) / float(height)
            if not (
                self.direction_minimum_aspect_ratio
                <= aspect_ratio
                <= self.direction_maximum_aspect_ratio
            ):
                continue

            perimeter = float(cv2.arcLength(hull, True))
            if perimeter <= 0.0:
                continue
            polygon = cv2.approxPolyDP(
                hull,
                self.direction_polygon_epsilon_ratio * perimeter,
                True,
            )
            if len(polygon) < self.direction_minimum_polygon_vertices:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.direction_minimum_hull_circularity:
                continue

            interior = np.zeros(blue.shape, dtype=np.uint8)
            cv2.drawContours(interior, [hull], -1, 255, -1)
            erosion_size = max(
                3,
                int(round(min(width, height) * self.direction_interior_erode_ratio)),
            )
            if erosion_size % 2 == 0:
                erosion_size += 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (erosion_size, erosion_size)
            )
            interior = cv2.erode(interior, kernel)
            white = cv2.inRange(
                hsv,
                np.array([0, 0, self.direction_white_value_min], dtype=np.uint8),
                np.array(
                    [179, self.direction_white_saturation_max, 255],
                    dtype=np.uint8,
                ),
            )
            white = cv2.bitwise_and(white, interior)
            _, white_x = np.nonzero(white)
            minimum_arrow_pixels = max(
                1,
                int(
                    round(
                        self.direction_minimum_arrow_pixels
                        * scale_x
                        * scale_y
                    )
                ),
            )
            if white_x.size < minimum_arrow_pixels:
                continue

            normalized_median = (
                float(np.median(white_x)) - float(x)
            ) / float(width)
            asymmetry = normalized_median - 0.5
            if abs(asymmetry) < self.direction_asymmetry_deadband:
                continue

            direction = (
                TrafficSign.DIRECTION_LEFT
                if asymmetry > 0.0
                else TrafficSign.DIRECTION_RIGHT
            )
            label = "direction_left" if asymmetry > 0.0 else "direction_right"
            confidence = min(
                1.0,
                0.60
                + 3.0
                * (abs(asymmetry) - self.direction_asymmetry_deadband),
            )
            detection = self.Detection(
                TrafficSign.DIRECTION,
                direction,
                (x, y, width, height),
                confidence,
                30,
                label,
            )
            if best is None or detection.confidence > best.confidence:
                best = detection
        return best

    def _detect(self, image, scale_x=1.0, scale_y=1.0):
        allowed_labels = self._allowed_template_labels()
        if allowed_labels is not None and not allowed_labels:
            return self._none_detection()

        detections = []
        direction_labels = frozenset(("direction_left", "direction_right"))
        if allowed_labels is None or allowed_labels.intersection(
            direction_labels
        ):
            direction_geometry = self._detect_direction_geometry(
                image, scale_x, scale_y
            )
            if (
                direction_geometry is not None
                and (
                    allowed_labels is None
                    or direction_geometry.label in allowed_labels
                )
            ):
                detections.append(direction_geometry)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_keypoints, frame_descriptors = self.orb.detectAndCompute(gray, None)
        if frame_descriptors is not None and len(frame_keypoints) >= 4:
            for template in self.templates:
                if (
                    allowed_labels is not None
                    and template.label not in allowed_labels
                ):
                    continue
                result = self._match_template(
                    template,
                    frame_keypoints,
                    frame_descriptors,
                    image.shape,
                    scale_x,
                    scale_y,
                )
                if (
                    result is not None
                    and result.confidence >= self.minimum_confidence
                ):
                    detections.append(result)
        if not detections:
            return self._none_detection()
        return max(
            detections,
            key=lambda result: (result.confidence, result.priority),
        )

    def _none_detection(self):
        return self.Detection(
            TrafficSign.NONE,
            TrafficSign.DIRECTION_NONE,
            None,
            0.0,
            -1,
            "none",
        )

    @staticmethod
    def _roi(box):
        roi = RegionOfInterest()
        if box is not None:
            x, y, width, height = box
            roi.x_offset = x
            roi.y_offset = y
            roi.width = width
            roi.height = height
        return roi

    @staticmethod
    def _resize_for_detection(image, processing_width):
        """Return an aspect-preserving detection image and source scales."""
        source_height, source_width = image.shape[:2]
        if processing_width <= 0 or source_width <= processing_width:
            return image, 1.0, 1.0

        target_width = int(processing_width)
        target_height = max(
            1,
            int(round(source_height * target_width / float(source_width))),
        )
        resized = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        return (
            resized,
            target_width / float(source_width),
            target_height / float(source_height),
        )

    @staticmethod
    def _box_to_source(box, scale_x, scale_y, source_shape):
        """Map a detection-image box to the original camera image."""
        if box is None:
            return None
        if scale_x <= 0.0 or scale_y <= 0.0:
            raise ValueError("detection image scales must be positive")

        source_height, source_width = source_shape[:2]
        x, y, width, height = box
        left = max(0, min(source_width, int(round(x / scale_x))))
        top = max(0, min(source_height, int(round(y / scale_y))))
        right = max(
            0,
            min(source_width, int(round((x + width) / scale_x))),
        )
        bottom = max(
            0,
            min(source_height, int(round((y + height) / scale_y))),
        )
        if right <= left or bottom <= top:
            return None
        return left, top, right - left, bottom - top

    def mission_callback(self, message):
        self.current_mission = str(message.data).strip()

    def _allowed_template_labels(self):
        current = getattr(self, "current_mission", None)
        mission_labels = getattr(self, "mission_template_labels", {})
        if current:
            return mission_labels.get(current, frozenset())
        if getattr(
            self,
            "template_filter_fail_open_without_mission",
            False,
        ):
            return None
        return getattr(
            self,
            "standalone_template_labels",
            frozenset(("direction_left", "direction_right", "tunnel_warning")),
        )

    def _mission_is_active(self):
        if not getattr(self, "mission_gating_enabled", False):
            return True
        current = getattr(self, "current_mission", None)
        return current is None or current in self.active_missions

    def image_callback(self, message):
        if not self._mission_is_active():
            return
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(message, "bgr8")
        except CvBridgeError as error:
            rospy.logerr_throttle(5.0, "Sign image conversion failed: %s", error)
            return

        processing_image, scale_x, scale_y = self._resize_for_detection(
            image, self.processing_width
        )
        self.processed_frame_count += 1
        result = self._detect(processing_image, scale_x, scale_y)
        if result.box is not None:
            result = result._replace(
                box=self._box_to_source(
                    result.box,
                    scale_x,
                    scale_y,
                    image.shape,
                )
            )
        detection = TrafficSign()
        detection.header = message.header
        detection.sign_type = result.sign_type
        detection.direction = result.direction
        detection.confidence = result.confidence
        detection.roi = self._roi(result.box)
        self.sign_pub.publish(detection)

        if result.sign_type != TrafficSign.NONE:
            rospy.loginfo_throttle(
                1.0,
                "Sign detected: %s confidence=%.2f",
                result.label,
                result.confidence,
            )

        # Avoid copying and serializing a 1280x720 debug frame at 30 Hz when no
        # viewer or recorder is connected. Once subscribed, the topic still
        # carries the original image coordinates and camera header.
        if (
            self.processed_frame_count % self.debug_every_n_frames != 0
            or self.debug_pub.get_num_connections() == 0
        ):
            return

        debug = image.copy()
        if result.box is not None:
            x, y, width, height = result.box
            cv2.rectangle(debug, (x, y), (x + width, y + height), (0, 255, 0), 2)
        label = "%s %.2f" % (result.label.upper(), result.confidence)
        cv2.putText(
            debug,
            label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if result.sign_type != TrafficSign.NONE else (0, 0, 255),
            2,
        )
        output = self.bridge.cv2_to_imgmsg(debug, "bgr8")
        output.header = message.header
        self.debug_pub.publish(output)


if __name__ == "__main__":
    rospy.init_node("sign_detector")
    SignDetector()
    rospy.spin()
