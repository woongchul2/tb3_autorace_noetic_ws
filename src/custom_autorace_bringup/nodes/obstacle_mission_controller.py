#!/usr/bin/env python3
"""Follow one surveyed curvature-continuous construction-course path."""

from collections import deque
from dataclasses import dataclass
from itertools import combinations
import math
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64, Float64MultiArray, Header, String
from std_srvs.srv import SetBool

from custom_autorace_bringup.obstacle_planner import (
    CourseSplinePlanner,
    Footprint,
    RectanglePathChecker,
    rectangle_surface_points,
)
from custom_autorace_bringup.local_registration import (
    EntryPlane,
    MissionLocalTemplate,
    OrientedSegmentLandmark,
    OrientedSegmentObservation,
    PointLandmark,
    PointObservation,
    RegistrationConfig,
    RegistrationObservation,
    TemporalRegistrationConfig,
    TemporalRegistrationFilter,
    entry_plane_progress,
    estimate_local_registration,
)
from custom_autorace_bringup.path_following import (
    AxisAlignedBoundsBoundary,
    CommonPath,
    PathFollower,
    PathSafety,
    Pose2D,
    RigidTransform2D,
    SafetyMargins,
    StraightCorridorBoundary,
    TrackingConfig,
    clamp,
    combine_validation_results,
    freeze_path_in_odom,
    normalize_angle,
    yaw_from_quaternion,
)


@dataclass(frozen=True)
class DetectedBarrierFace:
    """One finite, directed barrier face expressed in odom."""

    start: tuple
    end: tuple
    rms: float = 0.0

    @property
    def midpoint(self):
        return (
            0.5 * (self.start[0] + self.end[0]),
            0.5 * (self.start[1] + self.end[1]),
        )

    @property
    def length(self):
        return math.hypot(
            self.end[0] - self.start[0], self.end[1] - self.start[1]
        )

    @property
    def heading(self):
        return math.atan2(
            self.end[1] - self.start[1], self.end[0] - self.start[0]
        )


def obstacle_barrier_template(barriers):
    """Build the immutable mission-local front/side-face signature."""

    barriers = tuple(barriers)
    front_segments = []
    side_segments = []
    first_barrier_x = math.inf
    longitudinal_anchor_index = len(barriers) // 2
    for index, barrier in enumerate(barriers):
        if len(barrier) != 4:
            raise ValueError("each obstacle barrier must be [x, y, length, width]")
        center_x, center_y, length, width = (float(value) for value in barrier)
        if not all(
            math.isfinite(value)
            for value in (center_x, center_y, length, width)
        ) or length <= 0.0 or width <= 0.0:
            raise ValueError("obstacle barrier geometry must be finite and positive")
        first_barrier_x = min(first_barrier_x, center_x)
        front_x = center_x - 0.5 * length
        front_segments.append(
            OrientedSegmentLandmark(
                "barrier_%d_front" % index,
                (front_x, center_y - 0.5 * width),
                (front_x, center_y + 0.5 * width),
                # A partial outer-face midpoint is not a physical landmark.
                # Keep the common template honest; the early registrar below
                # promotes the outer midpoints only after both nearly full
                # faces satisfy the unique first-to-last baseline signature.
                longitudinal_weight=(
                    1.0 if index == longitudinal_anchor_index else 0.0
                ),
            )
        )
        back_x = center_x + 0.5 * length
        for side_name, side_y in (
            ("lower", center_y - 0.5 * width),
            ("upper", center_y + 0.5 * width),
        ):
            side_segments.append(
                OrientedSegmentLandmark(
                    "barrier_%d_%s_side" % (index, side_name),
                    (front_x, side_y),
                    (back_x, side_y),
                    # A clipped side still supplies its line-normal (course-y)
                    # offset, never a correspondence along the short face.
                    longitudinal_weight=0.0,
                )
            )
    if not front_segments:
        raise ValueError("obstacle barrier template must not be empty")
    return MissionLocalTemplate(
        mission="obstacle",
        frame_id="obstacle_course",
        # Keep fronts first so existing surveyed-path diagnostics retain their
        # stable order; side landmarks are registration evidence only.
        segments=tuple(front_segments + side_segments),
        # Two distinct front faces become visible only while the lane route is
        # passing the first barrier.  Readiness is therefore measured from
        # that physical landmark, not from the earlier surveyed path origin.
        # The path coordinates themselves remain unchanged and the common
        # follower projects the live pose onto the executable suffix.
        entry_plane=EntryPlane((first_barrier_x, 0.0), (1.0, 0.0)),
    )


def _fit_ordered_face(points, forward_yaw, maximum_rms):
    points = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    if points.shape[0] < 3 or not np.all(np.isfinite(points)):
        return None
    center = np.mean(points, axis=0)
    centered = points - center
    covariance = centered.T @ centered / float(points.shape[0])
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    projections = centered @ direction
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    rms = math.sqrt(float(np.mean(np.square(centered @ normal))))
    if rms > maximum_rms:
        return None
    start = center + float(np.min(projections)) * direction
    end = center + float(np.max(projections)) * direction
    # All local front faces are directed toward course-left.  Give an observed
    # PCA axis the same direction using the robot's lane heading as a weak
    # orientation prior; translation and yaw still come entirely from LiDAR.
    left = np.asarray((-math.sin(forward_yaw), math.cos(forward_yaw)))
    if float(np.dot(end - start, left)) < 0.0:
        start, end = end, start
    return DetectedBarrierFace(tuple(start), tuple(end), rms)


def extract_barrier_faces(
    ordered_points,
    forward_yaw,
    cluster_gap,
    minimum_points,
    minimum_length,
    maximum_length,
    maximum_rms,
    heading_tolerance,
):
    """Extract finite transverse faces from source-ordered LiDAR returns."""

    points = np.asarray(ordered_points, dtype=np.float64).reshape((-1, 2))
    if points.shape[0] < minimum_points or not np.all(np.isfinite(points)):
        return tuple()
    breaks = np.flatnonzero(
        np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1])) > cluster_gap
    )
    groups = np.split(points, breaks + 1)
    faces = []

    def split_fit(group):
        if group.shape[0] < minimum_points:
            return
        chord = group[-1] - group[0]
        chord_length = float(np.linalg.norm(chord))
        if chord_length > 1e-9:
            normal = np.asarray((-chord[1], chord[0])) / chord_length
            residual = np.abs((group - group[0]) @ normal)
            split = int(np.argmax(residual))
            if (
                float(residual[split]) > maximum_rms * 2.0
                and split >= minimum_points - 1
                and group.shape[0] - split >= minimum_points
            ):
                split_fit(group[: split + 1])
                split_fit(group[split:])
                return
        face = _fit_ordered_face(group, forward_yaw, maximum_rms)
        if face is None or not minimum_length <= face.length <= maximum_length:
            return
        expected = normalize_angle(forward_yaw + 0.5 * math.pi)
        if abs(normalize_angle(face.heading - expected)) > heading_tolerance:
            return
        faces.append(face)

    for group in groups:
        split_fit(group)
    return tuple(faces)


def _undirected_heading_error(first, second):
    """Return the smallest angle between two unoriented line axes."""

    difference = abs(normalize_angle(float(first) - float(second)))
    return min(difference, math.pi - difference)


def _axis_heading_delta(angle, reference):
    """Return the signed pi-periodic axis error nearest ``reference``."""

    difference = normalize_angle(float(angle) - float(reference))
    if difference > 0.5 * math.pi:
        difference -= math.pi
    elif difference < -0.5 * math.pi:
        difference += math.pi
    return difference


def register_obstacle_outer_faces(
    template,
    observed_faces,
    stamp,
    target_frame,
    robot_pose,
    config,
    minimum_entry_progress,
    maximum_entry_progress,
    maximum_entry_lateral,
    maximum_entry_heading,
    minimum_face_length=0.18,
    baseline_tolerance=0.035,
    parallel_heading_tolerance=math.radians(6.0),
    finite_endpoint_tolerance=0.006,
    side_minimum_length=0.055,
    side_heading_tolerance=math.radians(12.0),
    side_association_tolerance=0.025,
    baseline_lateral_tolerance=0.010,
    baseline_lateral_maximum=0.040,
    axis_consistency_tolerance=math.radians(2.0),
    ambiguity_position=0.004,
    ambiguity_heading=math.radians(0.25),
):
    """Register the course from its unique first-to-last barrier signature.

    The official approach crosses the course sideways, so neither robot-forward
    sorting nor a robot-yaw face prior describes the physical barrier order.
    The first and last front faces instead share one surveyed lateral coordinate
    and have a unique long baseline.  That pair fixes course yaw and longitudinal
    translation; its two directions are tested explicitly and the acquisition
    envelope rejects the reversed course hypothesis.

    A clipped front-face midpoint is not a physical lateral landmark.  Lateral
    translation therefore comes only from either two verified finite front-face
    endpoint spans or a verified perpendicular barrier side.  The short side
    is treated as an unbounded line and is not allowed to alter the yaw or the
    course-x translation established by the long parallel front faces.
    """

    front_landmarks = tuple(
        segment
        for segment in template.segments
        if segment.name.endswith("_front")
    )
    side_landmarks = tuple(
        segment
        for segment in template.segments
        if segment.name.endswith("_side")
    )
    if len(front_landmarks) < 3 or len(observed_faces) < 2:
        return None
    pose = Pose2D.from_value(robot_pose)
    first_landmark = front_landmarks[0]
    last_landmark = front_landmarks[-1]
    local_delta = (
        last_landmark.midpoint[0] - first_landmark.midpoint[0],
        last_landmark.midpoint[1] - first_landmark.midpoint[1],
    )
    expected_baseline = math.hypot(*local_delta)
    if expected_baseline <= 1e-9:
        return None
    local_baseline_heading = math.atan2(local_delta[1], local_delta[0])
    baseline_template = MissionLocalTemplate(
        mission=template.mission,
        frame_id=template.frame_id,
        points=(
            PointLandmark(
                first_landmark.name,
                first_landmark.midpoint,
                weight=first_landmark.weight,
            ),
            PointLandmark(
                last_landmark.name,
                last_landmark.midpoint,
                weight=last_landmark.weight,
            ),
        ),
        entry_plane=template.entry_plane,
    )
    finite_endpoint_tolerance = max(0.0, float(finite_endpoint_tolerance))
    side_minimum_length = max(0.0, float(side_minimum_length))
    side_heading_tolerance = max(0.0, float(side_heading_tolerance))
    side_association_tolerance = max(
        0.0, float(side_association_tolerance)
    )
    baseline_lateral_tolerance = max(
        0.0, float(baseline_lateral_tolerance)
    )
    baseline_lateral_maximum = max(
        baseline_lateral_tolerance, float(baseline_lateral_maximum)
    )
    axis_consistency_tolerance = max(
        0.0, float(axis_consistency_tolerance)
    )

    def face_confidence(face):
        return max(
            0.90,
            min(
                1.0,
                1.0
                - face.rms
                / max(1e-6, 10.0 * config.position_inlier_threshold),
            ),
        )

    def projected_segment_observation(landmark, face, target_heading):
        """Use a front line's normal offset without its clipped midpoint axis."""

        half_length = 0.5 * face.length
        direction = (math.cos(target_heading), math.sin(target_heading))
        midpoint = face.midpoint
        return OrientedSegmentObservation(
            landmark.name,
            (
                midpoint[0] - half_length * direction[0],
                midpoint[1] - half_length * direction[1],
            ),
            (
                midpoint[0] + half_length * direction[0],
                midpoint[1] + half_length * direction[1],
            ),
            confidence=face_confidence(face),
        )

    candidates = []

    def append_if_valid(result, score):
        if not result.accepted:
            return
        entry = entry_plane_progress(template.entry_plane, pose, result.transform)
        if not (
            minimum_entry_progress <= entry.longitudinal <= maximum_entry_progress
            and abs(entry.lateral) <= maximum_entry_lateral
            and abs(entry.heading_error) <= maximum_entry_heading
        ):
            return
        candidates.append((score, result))

    for first_index, second_index in combinations(
        range(len(observed_faces)), 2
    ):
        pair = (
            observed_faces[first_index],
            observed_faces[second_index],
        )
        if min(face.length for face in pair) < minimum_face_length:
            continue
        observed_distance = math.hypot(
            pair[1].midpoint[0] - pair[0].midpoint[0],
            pair[1].midpoint[1] - pair[0].midpoint[1],
        )
        baseline_error = abs(observed_distance - expected_baseline)
        if baseline_error > baseline_tolerance:
            continue
        if (
            _undirected_heading_error(pair[0].heading, pair[1].heading)
            > parallel_heading_tolerance
        ):
            continue

        for observed_first, observed_last in (pair, tuple(reversed(pair))):
            observed_delta = (
                observed_last.midpoint[0] - observed_first.midpoint[0],
                observed_last.midpoint[1] - observed_first.midpoint[1],
            )
            yaw = normalize_angle(
                math.atan2(observed_delta[1], observed_delta[0])
                - local_baseline_heading
            )
            expected_face_heading = normalize_angle(
                yaw + first_landmark.heading
            )
            if any(
                _undirected_heading_error(face.heading, expected_face_heading)
                > parallel_heading_tolerance
                for face in (observed_first, observed_last)
            ):
                continue
            selected_faces = (observed_first, observed_last)
            cosine = math.cos(yaw)
            sine = math.sin(yaw)
            course_forward = (cosine, sine)
            course_left = (-sine, cosine)
            selected_landmarks = (first_landmark, last_landmark)

            # A front line constrains only the course-forward origin offset.
            # Its midpoint coordinate along course-left is deliberately absent.
            forward_offsets = tuple(
                face.midpoint[0] * course_forward[0]
                + face.midpoint[1] * course_forward[1]
                - landmark.midpoint[0]
                for face, landmark in zip(selected_faces, selected_landmarks)
            )
            forward_offset = 0.5 * sum(forward_offsets)

            def transform_from_lateral(lateral_offset):
                return RigidTransform2D(
                    forward_offset * course_forward[0]
                    + lateral_offset * course_left[0],
                    forward_offset * course_forward[1]
                    + lateral_offset * course_left[1],
                    yaw,
                    template.frame_id,
                    target_frame,
                )

            # Finite-front fallback: both outer faces must account for both
            # surveyed endpoints.  A 180-240 mm clipped face cannot pass this
            # check merely because its midpoint baseline is correct.
            finite_offsets = []
            for face, landmark in zip(selected_faces, selected_landmarks):
                observed_span = sorted(
                    point[0] * course_left[0] + point[1] * course_left[1]
                    for point in (face.start, face.end)
                )
                local_span = sorted(
                    point[1] for point in (landmark.start, landmark.end)
                )
                endpoint_offsets = (
                    observed_span[0] - local_span[0],
                    observed_span[1] - local_span[1],
                )
                if (
                    abs(endpoint_offsets[1] - endpoint_offsets[0])
                    > finite_endpoint_tolerance
                ):
                    finite_offsets = []
                    break
                finite_offsets.append(0.5 * sum(endpoint_offsets))
            if (
                len(finite_offsets) == 2
                and abs(finite_offsets[1] - finite_offsets[0])
                <= finite_endpoint_tolerance
            ):
                initial_transform = transform_from_lateral(
                    0.5 * sum(finite_offsets)
                )
                observation = RegistrationObservation(
                    stamp=float(stamp),
                    target_frame=target_frame,
                    points=tuple(
                        PointObservation(
                            landmark.name,
                            face.midpoint,
                            confidence=face_confidence(face),
                        )
                        for face, landmark in zip(
                            selected_faces, baseline_template.points
                        )
                    ),
                )
                result = estimate_local_registration(
                    baseline_template,
                    observation,
                    config,
                    initial_transform=initial_transform,
                )
                append_if_valid(
                    result,
                    baseline_error
                    + result.diagnostics.position_rms
                    + 0.10 * result.diagnostics.heading_rms,
                )

            # Preferred clipped-face case: the front pair fixes yaw/course-x;
            # a perpendicular barrier side supplies only the course-y offset.
            # A candidate must lie over that barrier's finite front/back span,
            # have the surveyed side length, and be the side facing the robot.
            selected_indices = frozenset((first_index, second_index))
            for side_index, side_face in enumerate(observed_faces):
                if (
                    side_index in selected_indices
                    or side_face.length < side_minimum_length
                    or _undirected_heading_error(side_face.heading, yaw)
                    > side_heading_tolerance
                ):
                    continue
                side_local_x = (
                    side_face.midpoint[0] * course_forward[0]
                    + side_face.midpoint[1] * course_forward[1]
                    - forward_offset
                )
                side_target_left = (
                    side_face.midpoint[0] * course_left[0]
                    + side_face.midpoint[1] * course_left[1]
                )
                for side_landmark in side_landmarks:
                    side_length = math.hypot(
                        side_landmark.end[0] - side_landmark.start[0],
                        side_landmark.end[1] - side_landmark.start[1],
                    )
                    if (
                        side_face.length
                        > side_length + 2.0 * side_association_tolerance
                    ):
                        continue
                    side_x_min, side_x_max = sorted(
                        (side_landmark.start[0], side_landmark.end[0])
                    )
                    if not (
                        side_x_min - side_association_tolerance
                        <= side_local_x
                        <= side_x_max + side_association_tolerance
                    ):
                        continue
                    barrier_prefix = side_landmark.name.rsplit("_", 2)[0]
                    matching_front = next(
                        (
                            landmark
                            for landmark in front_landmarks
                            if landmark.name == barrier_prefix + "_front"
                        ),
                        None,
                    )
                    if matching_front is None:
                        continue

                    # The long midpoint baseline is the most stable primary
                    # yaw on the real scan, but unlike the two line axes it is
                    # corrupted when the outer fronts are clipped by different
                    # amounts. A verified perpendicular side supplies an
                    # independent course-forward axis. Require all three
                    # measured axes to agree, then reject a baseline carrying
                    # an unexplained course-left component. Do not average a
                    # contradictory sample into a plausible-looking transform.
                    front_axis_deltas = tuple(
                        _axis_heading_delta(
                            face.heading - landmark.heading,
                            yaw,
                        )
                        for face, landmark in zip(
                            selected_faces, selected_landmarks
                        )
                    )
                    side_axis_delta = _axis_heading_delta(
                        side_face.heading - side_landmark.heading,
                        yaw,
                    )
                    axis_deltas = front_axis_deltas + (side_axis_delta,)
                    axis_spread = max(axis_deltas) - min(axis_deltas)
                    if axis_spread > axis_consistency_tolerance:
                        continue
                    # The short side fragment is the noisiest axis, while one
                    # clipped front fit can also rotate.  Use the median of
                    # the three independently fitted physical line axes for
                    # this consistency measurement, so either single outlier
                    # is rejected by the other two.  Do not include the
                    # midpoint baseline in that vote: three agreeing axes must
                    # still expose asymmetric front clipping.  The transform
                    # yaw itself remains the unique long baseline, preserving
                    # the surveyed path.
                    consensus_axis_delta = sorted(axis_deltas)[1]
                    consensus_axis_yaw = yaw + consensus_axis_delta
                    consensus_course_left = (
                        -math.sin(consensus_axis_yaw),
                        math.cos(consensus_axis_yaw),
                    )
                    observed_baseline_lateral = (
                        observed_delta[0] * consensus_course_left[0]
                        + observed_delta[1] * consensus_course_left[1]
                    )
                    # A short LiDAR side fragment has noticeably noisier line
                    # heading than the long front-to-front baseline. Convert
                    # only the independently observed axis spread into the
                    # lateral uncertainty it can geometrically explain. The
                    # hard maximum prevents a noisy line from legitimising an
                    # arbitrarily skewed midpoint baseline. Conversely, equal
                    # line axes add no allowance, so the asymmetric-clipping
                    # case remains subject to the strict base tolerance.
                    lateral_uncertainty = expected_baseline * abs(
                        math.sin(axis_spread)
                    )
                    effective_lateral_tolerance = min(
                        baseline_lateral_maximum,
                        baseline_lateral_tolerance + lateral_uncertainty,
                    )
                    if (
                        abs(observed_baseline_lateral)
                        >= effective_lateral_tolerance - 1e-9
                    ):
                        continue
                    lateral_offset = (
                        side_target_left - side_landmark.midpoint[1]
                    )
                    initial_transform = transform_from_lateral(lateral_offset)
                    local_robot = initial_transform.inverse().apply_pose(pose)
                    side_sign = (
                        1.0
                        if side_landmark.midpoint[1]
                        > matching_front.midpoint[1]
                        else -1.0
                    )
                    if (
                        side_sign
                        * (local_robot.y - side_landmark.midpoint[1])
                        < -side_association_tolerance
                    ):
                        continue
                    line_template = MissionLocalTemplate(
                        mission=template.mission,
                        frame_id=template.frame_id,
                        segments=tuple(
                            OrientedSegmentLandmark(
                                landmark.name,
                                landmark.start,
                                landmark.end,
                                weight=landmark.weight,
                                longitudinal_weight=0.0,
                            )
                            for landmark in (
                                selected_landmarks + (side_landmark,)
                            )
                        ),
                        entry_plane=template.entry_plane,
                    )
                    line_observation = RegistrationObservation(
                        stamp=float(stamp),
                        target_frame=target_frame,
                        segments=tuple(
                            projected_segment_observation(
                                landmark,
                                face,
                                normalize_angle(yaw + landmark.heading),
                            )
                            for face, landmark in zip(
                                selected_faces + (side_face,),
                                selected_landmarks + (side_landmark,),
                            )
                        ),
                    )
                    result = estimate_local_registration(
                        line_template,
                        line_observation,
                        config,
                        initial_transform=initial_transform,
                    )
                    append_if_valid(
                        result,
                        baseline_error
                        + 0.10
                        * _undirected_heading_error(side_face.heading, yaw)
                        + abs(side_face.length - side_length)
                        + result.diagnostics.position_rms
                        + 0.10 * result.diagnostics.heading_rms,
                    )

    if not candidates:
        return None
    candidates.sort(key=lambda value: value[0])
    best = candidates[0][1]
    for _score, alternative in candidates[1:]:
        if (
            math.hypot(
                best.transform.target_from_source_x
                - alternative.transform.target_from_source_x,
                best.transform.target_from_source_y
                - alternative.transform.target_from_source_y,
            )
            > ambiguity_position
            or abs(
                normalize_angle(
                    best.transform.target_from_source_yaw
                    - alternative.transform.target_from_source_yaw
                )
            )
            > ambiguity_heading
        ):
            return None
    return best


class ObstacleMissionController:
    """Own cmd_vel only while traversing the fixed obstacle-course path."""

    # Source-stamped scan/map samples need both neighbouring odometry samples.
    # The controller deliberately releases its mission lock around live swept
    # validation, but initial alignment/full-path validation can still occupy a
    # timer callback for several hundred milliseconds.  A one-message rospy
    # input queue then drops the intermediate 30 Hz odometry samples and leaves
    # holes wider than maximum_scan_odom_skew in an otherwise healthy topic.
    # 64 samples retain over two seconds at the Gazebo EKF's nominal rate and
    # are pruned to odom_history_duration as soon as callbacks catch up.
    ODOM_SUBSCRIBER_QUEUE_SIZE = 64

    WAIT_GATE = "WAIT_GATE"
    ACQUIRING = "ACQUIRING"
    AVOIDING = "AVOIDING"
    REJOINING = "REJOINING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

    def __init__(self):
        get = rospy.get_param
        p = "~obstacle/"

        self.scan_topic = str(get(p + "topics/scan", "/scan_mid360_raw"))
        self.boundary_topic = str(
            get(p + "topics/lane_boundaries", "/detect/lane_boundaries")
        )
        self.odom_topic = str(get(p + "topics/odometry", "/odometry/filtered"))
        self.gate_topic = str(
            get(p + "topics/zone_gate", "/mission/enable/obstacle")
        )
        self.arm_topic = str(
            get(p + "topics/arm", "/mission/arm/obstacle")
        )
        self.ready_topic = str(
            get(p + "topics/ready", "/mission/ready/obstacle")
        )
        self.manual_stop_topic = str(
            get(p + "topics/manual_stop", "/control/manual_stop")
        )
        self.cmd_vel_topic = str(get(p + "topics/cmd_vel", "/cmd_vel"))
        self.speed_limit_topic = str(
            get(p + "topics/lane_speed_limit", "/control/max_vel")
        )
        self.lane_service_name = str(
            get(
                p + "topics/lane_control_service",
                "/control/lane_mission_handoff",
            )
        )
        self.lane_stop_service_name = str(
            get(p + "topics/lane_stop_service", "/control/lane_following")
        )

        self.lane_width = max(0.20, float(get(p + "course/lane_width", 0.495)))
        self.image_center = float(
            get(p + "course/boundary_image_center", 500.0)
        )
        self.pixels_per_meter = max(
            100.0, float(get(p + "course/boundary_pixels_per_meter", 1280.0))
        )
        self.scale_filter_alpha = clamp(
            float(get(p + "course/scale_filter_alpha", 0.20)), 0.0, 1.0
        )
        self.line_position_alpha = clamp(
            float(get(p + "course/line_position_filter_alpha", 0.12)),
            0.0,
            1.0,
        )
        self.boundary_sample_forward = float(
            get(p + "course/boundary_sample_forward", 0.14)
        )
        self.boundary_support_half_length = max(
            0.001,
            float(get(p + "course/boundary_support_half_length", 0.004)),
        )
        self.minimum_boundary_gap = float(
            get(p + "course/minimum_boundary_gap_px", 380.0)
        )
        self.maximum_boundary_gap = float(
            get(p + "course/maximum_boundary_gap_px", 850.0)
        )
        self.boundary_heading_limit = math.radians(
            abs(float(get(p + "course/boundary_max_heading_deg", 12.0)))
        )
        self.acquisition_heading_tolerance = math.radians(
            abs(float(get(p + "course/acquisition_max_heading_deg", 12.0)))
        )
        self.camera_timeout = max(
            0.05, float(get(p + "course/camera_timeout", 0.50))
        )
        self.map_corridor_max_residual = max(
            0.001,
            float(get(p + "course/map_corridor_max_residual", 0.004)),
        )

        self.localization_margin = max(
            0.0, float(get(p + "footprint/localization_margin", 0.0))
        )
        self.tracking_margin = max(
            0.0, float(get(p + "footprint/tracking_margin", 0.0))
        )
        self.footprint = Footprint(
            front=max(0.01, float(get(p + "footprint/front", 0.067645))),
            rear=max(0.01, float(get(p + "footprint/rear", 0.118073))),
            half_width=max(
                0.01, float(get(p + "footprint/half_width", 0.0903))
            ),
            obstacle_padding=max(
                0.0, float(get(p + "footprint/obstacle_padding", 0.014))
            ),
            line_margin=max(
                0.0, float(get(p + "footprint/line_margin", 0.009))
            ),
            localization_margin=self.localization_margin,
            tracking_margin=self.tracking_margin,
        )
        self.validation_footprint = Footprint(
            front=self.footprint.front,
            rear=self.footprint.rear,
            half_width=self.footprint.half_width,
            obstacle_padding=max(
                0.0,
                float(get(p + "footprint/validation_obstacle_padding", 0.0)),
            ),
            line_margin=max(
                0.0,
                float(get(p + "footprint/validation_line_margin", 0.009)),
            ),
            localization_margin=self.localization_margin,
            tracking_margin=self.tracking_margin,
        )
        # The committed path keeps the full surveyed reserve above.  Runtime
        # contact enforcement may use the physical footprint where the legal
        # construction channel cannot also absorb a second tracking reserve;
        # fixed-path diagnostics continue to expose the full reserved margin.
        self.runtime_line_margin = max(
            0.0,
            float(
                get(
                    p + "footprint/runtime_line_margin",
                    self.validation_footprint.line_margin,
                )
            ),
        )
        self.runtime_obstacle_padding = max(
            0.0,
            float(
                get(
                    p + "footprint/runtime_obstacle_padding",
                    self.footprint.obstacle_padding,
                )
            ),
        )
        self.runtime_localization_margin = max(
            0.0,
            float(
                get(
                    p + "footprint/runtime_localization_margin",
                    self.localization_margin,
                )
            ),
        )
        self.runtime_tracking_margin = max(
            0.0,
            float(
                get(
                    p + "footprint/runtime_tracking_margin",
                    self.tracking_margin,
                )
            ),
        )
        self.path_checker = RectanglePathChecker(
            footprint=self.footprint,
            collision_check_step=float(
                get(p + "planner/collision_check_step", 0.008)
            ),
            collision_check_angle=math.radians(
                float(get(p + "planner/collision_check_angle_deg", 3.0))
            ),
        )
        self.tracking_position_tolerance = max(
            0.01,
            float(get(p + "planner/tracking_position_tolerance", 0.055)),
        )
        self.tracking_heading_tolerance = math.radians(
            abs(float(get(p + "planner/tracking_heading_tolerance_deg", 28.0)))
        )
        local_bounds = get(
            p + "safety/local_bounds",
            [-2.0, 2.0, -2.0, 2.0],
        )
        if not isinstance(local_bounds, (list, tuple)) or len(local_bounds) != 4:
            raise rospy.ROSInitException(
                "obstacle safety/local_bounds must be [xmin, xmax, ymin, ymax]"
            )
        try:
            self.local_safety_boundary = AxisAlignedBoundsBoundary(
                *(float(value) for value in local_bounds)
            )
        except (TypeError, ValueError) as error:
            raise rospy.ROSInitException(
                "invalid obstacle safety/local_bounds: %s" % error
            )

        self.scan_max_range = max(
            0.30, float(get(p + "scan/maximum_planning_range", 1.40))
        )
        self.scan_rear_limit = max(
            0.0, float(get(p + "scan/rear_limit", 0.20))
        )
        self.cluster_link_distance = max(
            0.01, float(get(p + "scan/cluster_link_distance", 0.08))
        )
        self.cluster_minimum_points = max(
            1, int(get(p + "scan/cluster_minimum_points", 3))
        )
        self.obstacle_point_spacing = max(
            0.005, float(get(p + "scan/obstacle_point_spacing", 0.012))
        )
        self.scan_median_window = max(
            1, int(get(p + "scan/median_window", 5))
        )
        if self.scan_median_window % 2 == 0:
            self.scan_median_window += 1
        self.lidar_x = float(get(p + "scan/lidar_x", -0.033073))
        self.lidar_y = float(get(p + "scan/lidar_y", 0.0))
        self.maximum_scan_odom_skew = max(
            0.0, float(get(p + "scan/maximum_odom_skew", 0.075))
        )
        self.pending_scan_limit = max(
            1, int(get(p + "scan/pending_scan_limit", 4))
        )
        self.odom_history_duration = max(
            self.maximum_scan_odom_skew,
            float(get(p + "scan/odom_history_duration", 0.75)),
        )

        self.control_period = max(
            0.02, float(get(p + "control/period", 0.05))
        )
        self.entry_velocity_cap = max(
            0.0, float(get(p + "control/entry_velocity_cap", 0.09))
        )
        self.entry_handoff_velocity_tolerance = max(
            0.0,
            float(
                get(
                    p + "control/entry_handoff_velocity_tolerance",
                    0.005,
                )
            ),
        )
        self.entry_projection_position_tolerance = max(
            0.001,
            float(
                get(
                    p + "control/entry_projection_position_tolerance",
                    0.015,
                )
            ),
        )
        self.entry_projection_heading_tolerance = math.radians(
            abs(
                float(
                    get(
                        p
                        + "control/entry_projection_heading_tolerance_deg",
                        8.0,
                    )
                )
            )
        )
        self.lane_resume_max_velocity = max(
            self.entry_velocity_cap,
            float(get(p + "control/lane_resume_max_velocity", 0.30)),
        )
        self.maximum_angular_velocity = max(
            0.05, float(get(p + "control/maximum_angular_velocity", 0.85))
        )
        self.maximum_lateral_acceleration = max(
            0.001,
            float(get(p + "control/maximum_lateral_acceleration", 0.035)),
        )
        self.linear_acceleration = max(
            0.01, float(get(p + "control/linear_acceleration", 0.25))
        )
        self.linear_deceleration = max(
            0.01, float(get(p + "control/linear_deceleration", 0.80))
        )
        self.angular_acceleration = max(
            0.05, float(get(p + "control/angular_acceleration", 0.55))
        )
        self.lookahead_distance = max(
            0.03, float(get(p + "control/lookahead_distance", 0.06))
        )
        self.search_ahead_distance = max(
            self.lookahead_distance,
            float(get(p + "control/search_ahead_distance", 0.30)),
        )
        self.heading_gain = float(get(p + "control/heading_gain", 0.35))
        self.curvature_feedforward_weight = clamp(
            float(get(p + "control/curvature_feedforward_weight", 0.25)),
            0.0,
            1.0,
        )
        self.lateral_feedback_gain = max(
            0.0, float(get(p + "control/lateral_feedback_gain", 1.0))
        )
        self.safety_reaction_time = max(
            0.0, float(get(p + "control/safety_reaction_time", 0.10))
        )
        self.safety_stop_margin = max(
            0.0, float(get(p + "control/safety_stop_margin", 0.005))
        )

        self.cruise_velocity = max(
            0.005, float(get(p + "template/cruise_velocity", 0.07))
        )
        self.minimum_velocity = min(
            self.cruise_velocity,
            max(0.005, float(get(p + "template/minimum_velocity", 0.035))),
        )
        self.entry_velocity = min(
            self.cruise_velocity,
            max(
                self.minimum_velocity,
                float(get(p + "template/entry_velocity", 0.07)),
            ),
        )
        self.exit_velocity = min(
            self.cruise_velocity,
            max(
                self.minimum_velocity,
                float(get(p + "template/exit_velocity", 0.07)),
            ),
        )
        self.tracking_config = TrackingConfig(
            lookahead_distance=self.lookahead_distance,
            maximum_linear_velocity=self.cruise_velocity,
            maximum_angular_velocity=self.maximum_angular_velocity,
            maximum_lateral_acceleration=self.maximum_lateral_acceleration,
            linear_acceleration=self.linear_acceleration,
            linear_deceleration=self.linear_deceleration,
            angular_acceleration=self.angular_acceleration,
            heading_gain=self.heading_gain,
            curvature_feedforward_weight=self.curvature_feedforward_weight,
            lateral_feedback_gain=self.lateral_feedback_gain,
            search_ahead_distance=self.search_ahead_distance,
        )

        barriers = get(p + "template/barriers", [])
        try:
            self.registration_template = obstacle_barrier_template(barriers)
        except (TypeError, ValueError) as error:
            raise rospy.ROSInitException(
                "invalid obstacle local-registration template: %s" % error
            )
        self.registration_config = RegistrationConfig(
            position_inlier_threshold=max(
                0.005,
                float(get(p + "template/registration/inlier_distance", 0.025)),
            ),
            heading_inlier_threshold=math.radians(
                abs(
                    float(
                        get(
                            p + "template/registration/heading_inlier_deg",
                            6.0,
                        )
                    )
                )
            ),
            minimum_inliers=max(
                2, int(get(p + "template/registration/minimum_faces", 2))
            ),
            minimum_template_coverage=max(
                0.0,
                min(
                    1.0,
                    float(
                        get(
                            p + "template/registration/minimum_template_coverage",
                            0.60,
                        )
                    ),
                ),
            ),
            minimum_inlier_coverage=1.0,
        )
        self.registration_filter = TemporalRegistrationFilter(
            TemporalRegistrationConfig(
                required_confirmations=max(
                    1,
                    int(
                        get(
                            p + "template/registration/confirmation_samples",
                            3,
                        )
                    ),
                ),
                maximum_gap=max(
                    0.01,
                    float(
                        get(
                            p + "template/registration/confirmation_maximum_gap",
                            0.30,
                        )
                    ),
                ),
                maximum_position_delta=max(
                    0.001,
                    float(
                        get(
                            p
                            + "template/registration/confirmation_position_delta",
                            0.025,
                        )
                    ),
                ),
                maximum_heading_delta=math.radians(
                    abs(
                        float(
                            get(
                                p
                                + "template/registration/confirmation_heading_delta_deg",
                                3.0,
                            )
                        )
                    )
                ),
            )
        )
        self.registration_face_minimum_length = max(
            0.01,
            float(
                get(
                    p + "template/registration/face_minimum_length", 0.055
                )
            ),
        )
        self.registration_face_maximum_length = max(
            self.registration_face_minimum_length,
            float(
                get(
                    p + "template/registration/face_maximum_length", 0.32
                )
            ),
        )
        self.registration_face_maximum_rms = max(
            0.001,
            float(get(p + "template/registration/face_maximum_rms", 0.012)),
        )
        self.registration_face_heading_tolerance = math.radians(
            abs(
                float(
                    get(
                        p
                        + "template/registration/acquisition_face_heading_tolerance_deg",
                        180.0,
                    )
                )
            )
        )
        self.registration_outer_face_minimum_length = max(
            self.registration_face_minimum_length,
            float(
                get(
                    p
                    + "template/registration/outer_face_minimum_length",
                    0.18,
                )
            ),
        )
        self.registration_outer_baseline_tolerance = max(
            0.001,
            float(
                get(
                    p + "template/registration/outer_baseline_tolerance",
                    0.035,
                )
            ),
        )
        self.registration_outer_parallel_heading_tolerance = math.radians(
            abs(
                float(
                    get(
                        p
                        + "template/registration/outer_parallel_heading_tolerance_deg",
                        6.0,
                    )
                )
            )
        )
        self.registration_finite_endpoint_tolerance = max(
            0.001,
            float(
                get(
                    p
                    + "template/registration/finite_endpoint_tolerance",
                    0.006,
                )
            ),
        )
        self.registration_side_minimum_length = max(
            self.registration_face_minimum_length,
            float(
                get(
                    p + "template/registration/side_minimum_length",
                    0.055,
                )
            ),
        )
        self.registration_side_heading_tolerance = math.radians(
            abs(
                float(
                    get(
                        p
                        + "template/registration/side_heading_tolerance_deg",
                        12.0,
                    )
                )
            )
        )
        self.registration_side_association_tolerance = max(
            0.001,
            float(
                get(
                    p
                    + "template/registration/side_association_tolerance",
                    0.025,
                )
            ),
        )
        self.registration_baseline_lateral_tolerance = max(
            0.001,
            float(
                get(
                    p
                    + "template/registration/outer_baseline_lateral_tolerance",
                    0.010,
                )
            ),
        )
        self.registration_baseline_lateral_maximum = max(
            self.registration_baseline_lateral_tolerance,
            float(
                get(
                    p
                    + "template/registration/outer_baseline_lateral_maximum",
                    0.040,
                )
            ),
        )
        self.registration_axis_consistency_tolerance = math.radians(
            abs(
                float(
                    get(
                        p
                        + "template/registration/axis_consistency_tolerance_deg",
                        9.0,
                    )
                )
            )
        )
        self.registration_ambiguity_position = max(
            0.001,
            float(
                get(
                    p + "template/registration/ambiguity_position",
                    0.004,
                )
            ),
        )
        self.registration_ambiguity_heading = math.radians(
            abs(
                float(
                    get(
                        p
                        + "template/registration/ambiguity_heading_deg",
                        0.25,
                    )
                )
            )
        )
        self.registration_acquisition_minimum_progress = float(
            get(
                p + "template/registration/acquisition_minimum_progress",
                -1.10,
            )
        )
        self.registration_acquisition_maximum_progress = float(
            get(
                p + "template/registration/acquisition_maximum_progress",
                0.20,
            )
        )
        self.registration_acquisition_maximum_lateral = max(
            0.01,
            float(
                get(
                    p + "template/registration/acquisition_maximum_lateral",
                    0.80,
                )
            ),
        )
        self.registration_acquisition_maximum_heading = math.radians(
            abs(
                float(
                    get(
                        p
                        + "template/registration/acquisition_maximum_heading_deg",
                        105.0,
                    )
                )
            )
        )
        self.registration_entry_maximum_lateral = max(
            0.01,
            float(
                get(
                    p + "template/registration/entry_maximum_lateral", 0.20
                )
            ),
        )
        self.registration_entry_maximum_heading = math.radians(
            abs(
                float(
                    get(
                        p + "template/registration/entry_maximum_heading_deg",
                        15.0,
                    )
                )
            )
        )
        self.ready_lead_minimum_progress = float(
            get(p + "template/registration/ready_lead_minimum_progress", -0.22)
        )
        self.ready_lead_maximum_progress = float(
            get(p + "template/registration/ready_lead_maximum_progress", 0.05)
        )
        if not (
            self.registration_acquisition_minimum_progress
            < self.registration_acquisition_maximum_progress
            and self.ready_lead_minimum_progress
            < self.ready_lead_maximum_progress
        ):
            raise rospy.ROSInitException(
                "obstacle registration progress windows are reversed"
            )
        self.spline_planner = CourseSplinePlanner(
            collision_checker=self.path_checker,
            validation_footprint=self.validation_footprint,
            knots=get(p + "template/knots", []),
            exit_endpoint=get(p + "template/exit_curve/endpoint", []),
            exit_heading=math.radians(
                float(get(p + "template/exit_curve/heading_deg", 90.0))
            ),
            exit_start_tangent_length=float(
                get(p + "template/exit_curve/start_tangent_length", 0.15)
            ),
            exit_end_tangent_length=float(
                get(p + "template/exit_curve/end_tangent_length", 0.55)
            ),
            start_slope=float(get(p + "template/start_slope", 0.0)),
            end_slope=float(get(p + "template/end_slope", 0.0)),
            lateral_scale=float(get(p + "template/lateral_scale", 0.995)),
            sample_spacing=float(get(p + "template/sample_spacing", 0.004)),
            minimum_nominal_clearance=float(
                get(p + "template/minimum_nominal_clearance", 0.0035)
            ),
            live_validation_distance=float(
                get(p + "template/live_validation_distance", 0.45)
            ),
            barrier_association_distance=float(
                get(p + "template/barrier_association_distance", 0.025)
            ),
            entry_connector_minimum_join_distance=float(
                get(p + "planner/entry_connector_minimum_join_distance", 0.12)
            ),
            entry_connector_maximum_join_distance=float(
                get(p + "planner/entry_connector_maximum_join_distance", 0.32)
            ),
            entry_connector_join_step=float(
                get(p + "planner/entry_connector_join_step", 0.02)
            ),
            entry_connector_tangent_ratios=get(
                p + "planner/entry_connector_tangent_ratios",
                [0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.10, 0.24, 0.28],
            ),
            cruise_velocity=self.cruise_velocity,
            minimum_velocity=self.minimum_velocity,
            entry_velocity=self.entry_velocity,
            exit_velocity=self.exit_velocity,
            maximum_angular_velocity=self.maximum_angular_velocity,
            maximum_lateral_acceleration=self.maximum_lateral_acceleration,
            linear_acceleration=self.linear_acceleration,
            linear_deceleration=self.linear_deceleration,
            angular_acceleration=self.angular_acceleration,
            goal_position_tolerance=float(
                get(p + "control/goal_position_tolerance", 0.04)
            ),
            goal_heading_tolerance=math.radians(
                abs(float(get(p + "control/goal_heading_tolerance_deg", 10.0)))
            ),
            goal_crossing_max_distance=float(
                get(p + "control/goal_crossing_max_distance", 0.10)
            ),
        )
        nominal_obstacles = self._nominal_obstacle_points(
            barriers,
            float(get(p + "template/barrier_surface_spacing", 0.004)),
        )
        if not self.spline_planner.prepare(
            nominal_obstacles,
            -0.5 * self.lane_width,
            0.5 * self.lane_width,
        ):
            raise rospy.ROSInitException(
                "the configured obstacle path violates the nominal rectangle sweep"
            )

        self.scan_timeout = max(0.05, float(get(p + "timeouts/scan", 0.35)))
        self.odom_timeout = max(
            0.05, float(get(p + "timeouts/odometry", 0.35))
        )
        self.lock = threading.RLock()
        self.state = self.WAIT_GATE
        self.zone_gate = False
        self.arm_generation = 0
        self.armed_at = None
        self.prepared_generation = 0
        self.prepared_stamp = None
        self.ready_published_generation = 0
        self.last_ready_stamp = None
        self.start_requested = False
        self.revoke_requested = False
        self.mission_has_control = False
        self.handoff_ambiguous = False
        self.manual_stop = False
        self.shutting_down = False

        self.scan_points = np.empty((0, 2), dtype=np.float64)
        self.scan_odom_pose = None
        self.scan_stamp = None
        self.pending_scans = deque()
        self.scan_generation = 0
        self.last_plan_scan_generation = -1
        self.last_refresh_scan_generation = -1

        self.odom_ready = False
        self.odom_x = self.odom_y = self.odom_yaw = 0.0
        self.odom_linear_velocity = 0.0
        self.odom_angular_velocity = 0.0
        self.odom_frame = "odom"
        self.odom_stamp = None
        self.odom_generation = 0
        self.odom_epoch = 0
        self.odom_history = deque()
        self.odom_from_course = None
        self.registration_covariance = tuple()
        self.registration_diagnostics = None
        self.registration_source_stamp = None
        self.last_registration_scan_generation = -1

        self.line_center_absolute = None
        self.line_observation_progress_absolute = None
        self.line_observation_stamp = None
        self.committed_path = None
        self.path_follower = PathFollower(self.tracking_config)
        self.fixed_path_validation = None
        self.remaining_distance = 0.0
        self.live_speed_limit = math.inf
        self.live_requires_stop = False
        self.planning_seconds = 0.0

        self.last_command_time = None
        self.observed_lane_linear = 0.0
        self.observed_lane_angular = 0.0
        self.observed_lane_command_received = None
        self.lane_command_generation = 0
        self.gate_lane_command_generation = 0
        self.gate_odom_generation = 0
        self.gate_lane_command_received = None
        self.gate_odom_stamp = None
        self.gate_odom_frame = None
        self.handoff_refresh_scan_generation = -1
        self.handoff_refresh_lane_command_generation = -1
        self.handoff_refresh_odom_generation = -1
        self.handoff_refresh_lane_command_received = None
        self.handoff_refresh_odom_stamp = None
        self.handoff_refresh_odom_frame = None
        self.handoff_service_pending = False
        self.handoff_takeover_odom_generation = -1
        self.handoff_takeover_odom_stamp = None
        self.handoff_takeover_odom_frame = None

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.emergency_stop_pub = rospy.Publisher(
            self.manual_stop_topic, Bool, queue_size=1, latch=True
        )
        self.speed_limit_pub = rospy.Publisher(
            self.speed_limit_topic, Float64, queue_size=1, latch=True
        )
        self.state_pub = rospy.Publisher(
            "/obstacle/state", String, queue_size=1, latch=True
        )
        self.path_pub = rospy.Publisher(
            "/obstacle/local_path", Path, queue_size=1, latch=True
        )
        self.diagnostics_pub = rospy.Publisher(
            "/obstacle/diagnostics", Float64MultiArray, queue_size=1, latch=True
        )
        self.planner_status_pub = rospy.Publisher(
            "/obstacle/planner_status", String, queue_size=1, latch=True
        )
        self.ready_pub = rospy.Publisher(
            self.ready_topic, Header, queue_size=1, latch=True
        )

        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(
            self.boundary_topic,
            Float64MultiArray,
            self.boundary_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.odom_topic,
            Odometry,
            self.odom_callback,
            queue_size=self.ODOM_SUBSCRIBER_QUEUE_SIZE,
        )
        rospy.Subscriber(self.gate_topic, Bool, self.gate_callback, queue_size=1)
        rospy.Subscriber(self.arm_topic, Header, self.arm_callback, queue_size=1)
        rospy.Subscriber(
            self.manual_stop_topic, Bool, self.manual_stop_callback, queue_size=1
        )
        rospy.Subscriber(
            self.cmd_vel_topic, Twist, self.command_observer_callback, queue_size=1
        )
        self.lane_service = rospy.ServiceProxy(self.lane_service_name, SetBool)
        self.lane_stop_service = rospy.ServiceProxy(
            self.lane_stop_service_name, SetBool
        )
        self.timer = rospy.Timer(
            rospy.Duration(self.control_period), self.control_callback
        )
        rospy.on_shutdown(self.shutdown)
        self._publish_state()
        self.planner_status_pub.publish(String(data="TEMPLATE_READY"))
        rospy.loginfo(
            "Obstacle controller ready: one surveyed curvature-continuous path, scan=%s",
            self.scan_topic,
        )

    @staticmethod
    def _nominal_obstacle_points(barriers, spacing):
        surfaces = []
        for barrier in barriers:
            if len(barrier) != 4:
                raise rospy.ROSInitException(
                    "each template barrier must be [x, y, length, width]"
                )
            surfaces.append(
                rectangle_surface_points(
                    float(barrier[0]),
                    float(barrier[1]),
                    float(barrier[2]),
                    float(barrier[3]),
                    spacing,
                )
            )
        if not surfaces:
            raise rospy.ROSInitException("template/barriers is empty")
        return np.vstack(surfaces)

    def _publish_state(self):
        self.state_pub.publish(String(data=self.state))

    def _set_state(self, state):
        if state == self.state:
            return
        self.state = state
        self._publish_state()
        rospy.loginfo("Obstacle mission state: %s", state)

    def _reset_handoff_refresh(self):
        self.handoff_refresh_scan_generation = -1
        self.handoff_refresh_lane_command_generation = -1
        self.handoff_refresh_odom_generation = -1
        self.handoff_refresh_lane_command_received = None
        self.handoff_refresh_odom_stamp = None
        self.handoff_refresh_odom_frame = None
        self.handoff_service_pending = False
        self.handoff_takeover_odom_generation = -1
        self.handoff_takeover_odom_stamp = None
        self.handoff_takeover_odom_frame = None

    def _reset_preparation(self, reason):
        self.prepared_generation = 0
        self.prepared_stamp = None
        self.ready_published_generation = 0
        self.last_ready_stamp = None
        self.odom_from_course = None
        self.registration_covariance = tuple()
        self.registration_diagnostics = None
        self.registration_source_stamp = None
        self.last_registration_scan_generation = -1
        self.last_plan_scan_generation = -1
        self.last_refresh_scan_generation = -1
        self.registration_filter.reset(reason)
        self.committed_path = None
        self.path_follower = PathFollower(self.tracking_config)
        self.fixed_path_validation = None
        self.remaining_distance = 0.0
        self.live_speed_limit = math.inf
        self.live_requires_stop = False
        self.planning_seconds = 0.0
        self.line_center_absolute = None
        self.line_observation_progress_absolute = None
        self.line_observation_stamp = None
        self.gate_lane_command_received = None
        self.gate_odom_stamp = None
        self.gate_odom_frame = None
        self._reset_handoff_refresh()

    def arm_callback(self, message):
        """Start one local-registration generation while lane keeps control."""

        if message.frame_id != "obstacle":
            return
        with self.lock:
            generation = int(message.seq)
            if (
                generation <= 0
                or generation == self.arm_generation
                or message.stamp == rospy.Time()
            ):
                return
            if self.state not in (self.WAIT_GATE, self.COMPLETE):
                rospy.logwarn(
                    "Ignoring obstacle arm generation %d while %s",
                    generation,
                    self.state,
                )
                return
            self.arm_generation = generation
            self.armed_at = message.stamp
            self.start_requested = False
            self.revoke_requested = False
            self.zone_gate = False
            self._reset_preparation("new mission arm")
            if self.state == self.COMPLETE and generation > 0:
                self._set_state(self.WAIT_GATE)
            if generation > 0:
                rospy.loginfo(
                    "Obstacle local registration armed: generation=%d stamp=%.3f",
                    generation,
                    self.armed_at.to_sec() if self.armed_at is not None else 0.0,
                )

    def gate_callback(self, message):
        with self.lock:
            requested = bool(message.data)
            if requested and not self._prepared_ready_for_gate(rospy.Time.now()):
                rospy.logwarn_throttle(
                    1.0,
                    "Ignoring obstacle gate without a fresh matching ready",
                )
                return
            was_open = self.zone_gate
            self.zone_gate = requested
            if self.zone_gate and not was_open:
                # Keep lane control through the speed transition.  Obstacle
                # may acquire /cmd_vel only after both a lane command and an
                # odometry sample newer than this edge show the cap has taken
                # effect.
                self.gate_lane_command_generation = (
                    self.lane_command_generation
                )
                self.gate_odom_generation = self.odom_generation
                self.gate_lane_command_received = (
                    self.observed_lane_command_received
                )
                self.gate_odom_stamp = self.odom_stamp
                self.gate_odom_frame = self.odom_frame
                self._reset_handoff_refresh()
                self.start_requested = True
                self.speed_limit_pub.publish(
                    Float64(data=self.entry_velocity_cap)
                )
            elif not self.zone_gate and was_open and self.state != self.COMPLETE:
                self.revoke_requested = True

    def _prepared_ready_for_gate(self, now):
        if (
            self.arm_generation <= 0
            or self.prepared_generation != self.arm_generation
            or self.ready_published_generation != self.arm_generation
            or self.committed_path is None
            or self.odom_from_course is None
            or self.last_ready_stamp is None
        ):
            return False
        age = (now - self.last_ready_stamp).to_sec()
        return (
            -self.maximum_scan_odom_skew
            <= age
            <= max(self.scan_timeout, self.odom_timeout)
        )

    def odom_callback(self, message):
        received = rospy.Time.now()
        source_stamp = (
            message.header.stamp
            if message.header.stamp != rospy.Time()
            else received
        )
        with self.lock:
            odom_x = float(message.pose.pose.position.x)
            odom_y = float(message.pose.pose.position.y)
            odom_yaw = yaw_from_quaternion(message.pose.pose.orientation)
            odom_frame = message.header.frame_id or "odom"
            odom_epoch_changed = bool(
                self.odom_history
                and (
                    source_stamp < self.odom_history[-1][0]
                    or odom_frame != self.odom_history[-1][4]
                )
            )
            if odom_epoch_changed:
                # Simulation clock resets and frame changes invalidate temporal
                # interpolation across the discontinuity.
                self.odom_epoch = getattr(self, "odom_epoch", 0) + 1
                self.odom_history.clear()
                self.pending_scans.clear()
                self.scan_points = np.empty((0, 2), dtype=np.float64)
                self.scan_odom_pose = None
                self.scan_stamp = None
            self.odom_x = odom_x
            self.odom_y = odom_y
            self.odom_yaw = odom_yaw
            self.odom_linear_velocity = float(message.twist.twist.linear.x)
            self.odom_angular_velocity = float(message.twist.twist.angular.z)
            self.odom_frame = odom_frame
            self.odom_stamp = source_stamp
            self.odom_ready = True
            self.odom_generation += 1
            self.odom_history.append(
                (source_stamp, odom_x, odom_y, odom_yaw, odom_frame)
            )
            while self.odom_history and (
                source_stamp - self.odom_history[0][0]
            ).to_sec() > self.odom_history_duration:
                self.odom_history.popleft()
            for scan, aligned_pose in self._take_ready_scans(received):
                self._accept_scan(scan, aligned_pose)

    def _odom_pose_at(self, stamp, maximum_skew=None):
        """Return an odom pose using bracketed translation/yaw interpolation."""
        if not self.odom_history:
            return None
        allowed_skew = (
            self.maximum_scan_odom_skew
            if maximum_skew is None
            else max(0.0, float(maximum_skew))
        )
        target = float(stamp.to_sec())
        samples = tuple(self.odom_history)
        times = [float(sample[0].to_sec()) for sample in samples]
        right = int(np.searchsorted(times, target, side="left"))

        if right < len(samples) and abs(times[right] - target) <= 1e-9:
            sample = samples[right]
            return float(sample[1]), float(sample[2]), float(sample[3])

        if 0 < right < len(samples):
            left_sample = samples[right - 1]
            right_sample = samples[right]
            before = target - times[right - 1]
            after = times[right] - target
            if (
                before <= allowed_skew + 1e-12
                and after <= allowed_skew + 1e-12
            ):
                span = max(1e-12, before + after)
                fraction = before / span
                yaw_delta = normalize_angle(right_sample[3] - left_sample[3])
                return (
                    (1.0 - fraction) * left_sample[1]
                    + fraction * right_sample[1],
                    (1.0 - fraction) * left_sample[2]
                    + fraction * right_sample[2],
                    normalize_angle(left_sample[3] + fraction * yaw_delta),
                )

        # Never extrapolate a source-stamped sensor sample with the newest
        # pose. A later odom callback can provide the missing right bracket.
        return None

    def command_observer_callback(self, message):
        with self.lock:
            if (
                not self.mission_has_control
                and math.isfinite(message.linear.x)
                and math.isfinite(message.angular.z)
            ):
                self.observed_lane_linear = float(message.linear.x)
                self.observed_lane_angular = float(message.angular.z)
                self.observed_lane_command_received = rospy.Time.now()
                self.lane_command_generation += 1

    def boundary_callback(self, message):
        if len(message.data) < 4:
            return
        yellow = float(message.data[0])
        white = float(message.data[1])
        yellow_valid = bool(message.data[2] > 0.5 and math.isfinite(yellow))
        white_valid = bool(message.data[3] > 0.5 and math.isfinite(white))
        if not (yellow_valid and white_valid):
            return
        gap = white - yellow
        if not self.minimum_boundary_gap <= gap <= self.maximum_boundary_gap:
            return

        received = rospy.Time.now()
        with self.lock:
            if not self.odom_ready or self.odom_from_course is None:
                return
            heading_error = self._course_heading_error()
            if abs(heading_error) > self.boundary_heading_limit:
                return
            if self.state not in (
                self.WAIT_GATE,
                self.ACQUIRING,
                self.AVOIDING,
                self.REJOINING,
            ):
                return

            measured_scale = gap / self.lane_width
            self.pixels_per_meter = (
                (1.0 - self.scale_filter_alpha) * self.pixels_per_meter
                + self.scale_filter_alpha * measured_scale
            )
            pixel_center = 0.5 * (yellow + white)
            robot_lateral = (
                self.image_center - pixel_center
            ) / self.pixels_per_meter
            relative_center = (
                -math.sin(heading_error) * self.boundary_sample_forward
                + math.cos(heading_error) * robot_lateral
            )
            course_heading = self._course_heading_in_odom()
            if course_heading is None:
                course_heading = self.odom_yaw + heading_error
            robot_absolute = (
                -math.sin(course_heading) * self.odom_x
                + math.cos(course_heading) * self.odom_y
            )
            robot_progress_absolute = (
                math.cos(course_heading) * self.odom_x
                + math.sin(course_heading) * self.odom_y
            )
            measured_absolute = robot_absolute + relative_center
            measured_progress_absolute = robot_progress_absolute + (
                math.cos(heading_error) * self.boundary_sample_forward
                + math.sin(heading_error) * robot_lateral
            )
            if self.line_center_absolute is None:
                self.line_center_absolute = measured_absolute
            else:
                alpha = self.line_position_alpha
                self.line_center_absolute = (
                    (1.0 - alpha) * self.line_center_absolute
                    + alpha * measured_absolute
                )
            # Longitudinal support belongs to this image observation, not to
            # the moving robot.  Keep it absolute so a fresh-but-old camera
            # band cannot be dragged through the slalom for camera_timeout.
            self.line_observation_progress_absolute = measured_progress_absolute
            # Freshness belongs to an accepted two-line observation.  Invalid
            # views beside a barrier must not keep an older centre alive.
            self.line_observation_stamp = received

    def _pending_scan_expired(self, scan, now):
        return bool(
            (now - scan["source_stamp"]).to_sec() > self.scan_timeout
            or (now - scan["received"]).to_sec() > self.scan_timeout
        )

    def _enqueue_pending_scan(self, scan, now):
        while self.pending_scans and self._pending_scan_expired(
            self.pending_scans[0], now
        ):
            self.pending_scans.popleft()
        if self._pending_scan_expired(scan, now):
            return
        if len(self.pending_scans) >= self.pending_scan_limit:
            self.pending_scans.popleft()
            rospy.logwarn_throttle(
                1.0,
                "Obstacle dropped the oldest pending LaserScan before odom "
                "synchronization",
            )
        self.pending_scans.append(scan)

    def _take_ready_scans(self, now):
        """Return bracketed scans and retain only bounded future scans."""
        history = tuple(self.odom_history)
        latest_stamp = (
            float(history[-1][0].to_sec()) if history else -math.inf
        )
        ready = []
        waiting = deque()
        for scan in self.pending_scans:
            if self._pending_scan_expired(scan, now):
                continue
            aligned_pose = self._odom_pose_at(
                scan["source_stamp"],
                maximum_skew=self.maximum_scan_odom_skew,
            )
            if aligned_pose is not None:
                ready.append((scan, aligned_pose))
                continue
            if float(scan["source_stamp"].to_sec()) > latest_stamp:
                waiting.append(scan)
                continue
            rospy.logwarn_throttle(
                1.0,
                "Obstacle discarded a LaserScan whose odom bracket exceeded "
                "the synchronization tolerance",
            )
        self.pending_scans = waiting
        ready.sort(key=lambda item: item[0]["source_stamp"].to_sec())
        return ready

    def _accept_scan(self, scan, aligned_pose):
        with self.lock:
            if (
                self.scan_stamp is not None
                and scan["source_stamp"] <= self.scan_stamp
            ):
                return
            self.scan_points = scan["points"]
            self.scan_odom_pose = aligned_pose
            self.scan_stamp = scan["source_stamp"]
            self.scan_generation += 1

    def scan_callback(self, message):
        received = rospy.Time.now()
        source_stamp = (
            message.header.stamp
            if message.header.stamp != rospy.Time()
            else received
        )
        ranges = np.asarray(message.ranges, dtype=np.float64)
        if ranges.size == 0 or not math.isfinite(message.angle_increment):
            return
        indices = np.arange(ranges.size, dtype=np.float64)
        angles = float(message.angle_min) + indices * float(message.angle_increment)
        minimum_range = max(0.0, float(message.range_min))
        maximum_range = min(float(message.range_max), self.scan_max_range)
        valid = np.isfinite(ranges)
        valid[valid] &= (
            ranges[valid] >= minimum_range
        ) & (ranges[valid] <= maximum_range)
        filtered = ranges.copy()
        if self.scan_median_window > 1:
            half_window = self.scan_median_window // 2
            neighbors = []
            for offset in range(-half_window, half_window + 1):
                shifted = np.roll(ranges, offset).copy()
                shifted_valid = np.roll(valid, offset)
                shifted[~shifted_valid] = np.nan
                neighbors.append(shifted)
            neighbor_stack = np.vstack(neighbors)
            valid_neighbors = np.count_nonzero(
                np.isfinite(neighbor_stack), axis=0
            )
            filtered = np.ma.median(
                np.ma.masked_invalid(neighbor_stack), axis=0
            ).filled(np.nan)
            filtered[
                valid_neighbors < (self.scan_median_window // 2 + 1)
            ] = np.nan

        valid = np.isfinite(filtered)
        valid[valid] &= (
            filtered[valid] >= minimum_range
        ) & (filtered[valid] <= maximum_range)
        scan_indices = np.flatnonzero(valid)
        raw_points = np.empty((0, 2), dtype=np.float64)
        if scan_indices.size:
            selected_ranges = filtered[scan_indices]
            selected_angles = angles[scan_indices]
            raw_points = np.column_stack(
                (
                    self.lidar_x + selected_ranges * np.cos(selected_angles),
                    self.lidar_y + selected_ranges * np.sin(selected_angles),
                )
            )

        groups = []
        current = []
        previous_index = None
        previous_point = None
        for scan_index, point in zip(scan_indices.tolist(), raw_points):
            connected = (
                previous_index is not None
                and scan_index == previous_index + 1
                and np.linalg.norm(point - previous_point)
                <= self.cluster_link_distance
            )
            if current and not connected:
                groups.append(current)
                current = []
            current.append(point)
            previous_index = scan_index
            previous_point = point
        if current:
            groups.append(current)

        dense = []
        for group in groups:
            if len(group) < self.cluster_minimum_points:
                continue
            dense.append(group[0])
            for first, second in zip(group[:-1], group[1:]):
                distance = float(np.linalg.norm(second - first))
                samples = max(
                    1, int(math.ceil(distance / self.obstacle_point_spacing))
                )
                for sample in range(1, samples + 1):
                    dense.append(
                        first + (float(sample) / samples) * (second - first)
                    )
        points = (
            np.asarray(dense, dtype=np.float64).reshape((-1, 2))
            if dense
            else np.empty((0, 2), dtype=np.float64)
        )
        scan = {
            "source_stamp": source_stamp,
            "received": received,
            "points": points,
        }
        with self.lock:
            if self.scan_stamp is not None and source_stamp <= self.scan_stamp:
                return
            # Enqueue before re-reading history under the same lock. If the
            # scan callback won the race, the next odom callback supplies the
            # right bracket; if odom won, this pass accepts it immediately.
            self._enqueue_pending_scan(scan, received)
            for ready_scan, aligned_pose in self._take_ready_scans(received):
                self._accept_scan(ready_scan, aligned_pose)

    def manual_stop_callback(self, message):
        with self.lock:
            self.manual_stop = bool(message.data)
            if self.manual_stop and self.mission_has_control:
                self._publish_stop()

    def _set_lane_controller(self, enabled):
        try:
            rospy.wait_for_service(self.lane_service_name, timeout=2.0)
            response = self.lane_service(enabled)
            if not response.success:
                raise rospy.ServiceException(response.message)
            self.mission_has_control = not enabled
            self.handoff_ambiguous = False
            return True
        except (rospy.ROSException, rospy.ServiceException) as error:
            self.handoff_ambiguous = True
            rospy.logerr("Obstacle cmd_vel handoff failed: %s", error)
            return False

    def _stop_lane_controller(self):
        try:
            rospy.wait_for_service(self.lane_stop_service_name, timeout=1.0)
            response = self.lane_stop_service(False)
            if not response.success:
                raise rospy.ServiceException(response.message)
            return True
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logerr("Obstacle lane emergency stop failed: %s", error)
            return False

    def _start_run(self):
        now = rospy.Time.now()
        if (
            not self.odom_ready
            or self.odom_stamp is None
            or (now - self.odom_stamp).to_sec() > self.odom_timeout
        ):
            input_problem = "odometry is unavailable or stale"
        elif not self._prepared_ready_for_gate(now):
            input_problem = "locally registered path is not ready"
        else:
            input_problem = None
        if input_problem is not None:
            rospy.logwarn_throttle(
                2.0,
                "Waiting for obstacle local path before command handoff: %s",
                input_problem,
            )
            return
        self.start_requested = False
        self.revoke_requested = False
        self.live_speed_limit = math.inf
        self.live_requires_stop = False
        self.last_command_time = None
        self._set_state(self.ACQUIRING)

    def _entry_handoff_input_problem(
        self,
        now,
        minimum_odom_generation=None,
        minimum_lane_command_generation=None,
        minimum_odom_stamp=None,
        minimum_lane_command_received=None,
        minimum_odom_frame=None,
    ):
        """Return why lane control must still retain the obstacle entry."""

        minimum_odom_generation = (
            self.gate_odom_generation
            if minimum_odom_generation is None
            else int(minimum_odom_generation)
        )
        minimum_lane_command_generation = (
            self.gate_lane_command_generation
            if minimum_lane_command_generation is None
            else int(minimum_lane_command_generation)
        )
        minimum_odom_stamp = (
            self.gate_odom_stamp
            if minimum_odom_stamp is None
            else minimum_odom_stamp
        )
        minimum_lane_command_received = (
            self.gate_lane_command_received
            if minimum_lane_command_received is None
            else minimum_lane_command_received
        )
        minimum_odom_frame = (
            self.gate_odom_frame
            if minimum_odom_frame is None
            else str(minimum_odom_frame)
        )
        if self.odom_generation <= minimum_odom_generation:
            return "waiting for newer odometry"
        if self.lane_command_generation <= minimum_lane_command_generation:
            return "waiting for a newer lane command"
        if (
            minimum_odom_stamp is not None
            and (
                self.odom_stamp is None
                or self.odom_stamp <= minimum_odom_stamp
            )
        ):
            return "waiting for newer source-stamped odometry"
        if (
            minimum_odom_frame is not None
            and self.odom_frame != minimum_odom_frame
        ):
            return "odometry frame changed during entry handoff"
        if (
            minimum_lane_command_received is not None
            and (
                self.observed_lane_command_received is None
                or self.observed_lane_command_received
                <= minimum_lane_command_received
            )
        ):
            return "waiting for a lane command received after the handoff edge"
        if (
            self.observed_lane_command_received is None
            or (now - self.observed_lane_command_received).to_sec()
            > self.odom_timeout
        ):
            return "lane command is unavailable or stale"

        maximum_handoff_velocity = (
            self.entry_velocity_cap
            + self.entry_handoff_velocity_tolerance
        )
        if not (
            -self.entry_handoff_velocity_tolerance
            <= self.observed_lane_linear
            <= maximum_handoff_velocity
        ):
            return "lane command has not reached the entry velocity cap"
        if not (
            -self.entry_handoff_velocity_tolerance
            <= self.odom_linear_velocity
            <= maximum_handoff_velocity
        ):
            return "measured velocity has not reached the entry velocity cap"
        if abs(self.observed_lane_angular) > self.maximum_angular_velocity:
            return "lane angular command exceeds the obstacle limit"
        return None

    def _activate_after_handoff_service(self, now):
        """Use only post-service odometry before the first mission command."""

        if self.odom_generation <= self.handoff_takeover_odom_generation:
            return False
        if (
            self.odom_stamp is None
            or self.handoff_takeover_odom_stamp is None
            or self.odom_stamp <= self.handoff_takeover_odom_stamp
        ):
            return False
        if self.odom_frame != self.handoff_takeover_odom_frame:
            self._fail("odometry frame changed after cmd_vel handoff")
            return False
        if (now - self.odom_stamp).to_sec() > self.odom_timeout:
            return False

        current_pose = self._current_pose()
        projection = self.path_follower.reset(
            self.committed_path,
            current_pose,
            initial_linear=0.0,
            initial_angular=0.0,
        )
        # Heading is gated immediately before the service call.  While that
        # call is in flight the lane owner can rotate across the very short,
        # high-curvature G2 seam even though it remains on the validated
        # route.  Once ownership is exclusive and velocity is reset to zero,
        # the common follower and the runtime swept-footprint check own that
        # heading transient; only lateral departure invalidates the join.
        if projection.distance > self.entry_projection_position_tolerance:
            self._fail(
                "post-service obstacle entry moved outside join tolerance: "
                "join=%.4fm"
                % projection.distance
            )
            return False
        self.remaining_distance = max(
            0.0, self.committed_path.length - projection.station
        )
        self.path_follower.diagnostics.commanded_linear = 0.0
        self.path_follower.diagnostics.commanded_angular = 0.0
        self.last_command_time = now
        self._set_state(self.AVOIDING)
        return True

    def _revoke(self):
        if self.mission_has_control:
            self._publish_stop()
            if not self._set_lane_controller(True):
                self._fail("lane-controller handoff failed after gate closed")
                return
        self.speed_limit_pub.publish(Float64(data=self.lane_resume_max_velocity))
        self.revoke_requested = False
        self._reset_preparation("gate revoked")
        self._set_state(self.WAIT_GATE)

    def _fail(self, reason):
        self.planner_status_pub.publish(String(data="FAILED"))
        owns_unambiguously = self.mission_has_control and not self.handoff_ambiguous
        if not owns_unambiguously:
            owns_unambiguously = self._set_lane_controller(False)
        if owns_unambiguously:
            self._publish_stop()
        else:
            # Service failure leaves ownership indeterminate. Never publish a
            # competing Twist. Relinquish direct output, stop the lane publisher
            # through its independent service, then pause every mission owner.
            self.mission_has_control = False
            if not self._stop_lane_controller():
                rospy.logfatal("Obstacle could not establish a sole cmd_vel stop owner")
            self.emergency_stop_pub.publish(Bool(data=True))
        self._set_state(self.FAILED)
        rospy.logerr("Obstacle mission failed: %s", reason)

    def _publish_stop(self):
        if self.mission_has_control:
            self.cmd_pub.publish(Twist())
        self.path_follower.last_linear = 0.0
        self.path_follower.last_angular = 0.0
        self.path_follower.diagnostics.commanded_linear = 0.0
        self.path_follower.diagnostics.commanded_angular = 0.0
        self.last_command_time = rospy.Time.now()

    def _course_heading_in_odom(self):
        odom_from_course = getattr(self, "odom_from_course", None)
        if odom_from_course is not None:
            return odom_from_course.target_from_source_yaw
        return None

    def _course_heading_error(self):
        course_heading = self._course_heading_in_odom()
        if course_heading is not None:
            return normalize_angle(course_heading - self.odom_yaw)
        return 0.0

    def _course_template_coordinates(self):
        """Return (progress, path lateral offset) through one frame transform."""
        odom_from_course = getattr(self, "odom_from_course", None)
        if odom_from_course is not None:
            course_pose = odom_from_course.inverse().apply_pose(
                self._current_pose()
            )
            return course_pose.x, -course_pose.y
        return None

    def _attach_local_safety_boundary(self, path):
        """Freeze the mission-local extent with the registered path frame."""
        if self.odom_from_course is None:
            raise ValueError("obstacle local-to-odom transform is unavailable")
        map_boundary_odom = self.local_safety_boundary.transformed(
            self.odom_from_course
        )
        safety = path.safety
        # The surveyed path, nominal barriers, straight line bounds and this
        # local extent all receive the same frozen rigid transform.  Global
        # registration covariance is therefore common-mode and cancels from
        # their relative clearance.  Adding it here would count the same
        # alignment error twice and can reject a physically unchanged course.
        # The covariance remains available as registration diagnostics; live
        # odom/LiDAR/camera geometry is checked independently while driving.
        path.safety = PathSafety(
            line_boundaries=safety.line_boundaries,
            map_boundaries=(*safety.map_boundaries, map_boundary_odom),
            fixed_obstacles=safety.fixed_obstacles,
            margins=safety.margins,
        )
        return path

    def _camera_corridor(self, now):
        if (
            self.line_center_absolute is None
            or self.line_observation_stamp is None
            or (now - self.line_observation_stamp).to_sec() > self.camera_timeout
        ):
            return None
        course_heading = self._course_heading_in_odom()
        if course_heading is None:
            return None
        robot_absolute = (
            -math.sin(course_heading) * self.odom_x
            + math.cos(course_heading) * self.odom_y
        )
        center = self.line_center_absolute - robot_absolute
        return (
            center - 0.5 * self.lane_width,
            center + 0.5 * self.lane_width,
            center,
        )

    def _surveyed_corridor(self, course_coordinates=None):
        if course_coordinates is None:
            course_coordinates = self._course_template_coordinates()
        if course_coordinates is None:
            return None
        center = course_coordinates[1]
        return (
            center - 0.5 * self.lane_width,
            center + 0.5 * self.lane_width,
            center,
        )

    def _camera_line_boundary(self, now, reference_corridor):
        camera = self._camera_corridor(now)
        progress = getattr(self, "line_observation_progress_absolute", None)
        course_heading = self._course_heading_in_odom()
        if (
            camera is None
            or reference_corridor is None
            or progress is None
            or course_heading is None
            or abs(camera[2] - reference_corridor[2])
            > self.map_corridor_max_residual
        ):
            return None
        half_support = self.boundary_support_half_length
        return StraightCorridorBoundary(
            self.line_center_absolute - 0.5 * self.lane_width,
            self.line_center_absolute + 0.5 * self.lane_width,
            origin=(0.0, 0.0),
            heading=course_heading,
            minimum_progress=progress - half_support,
            maximum_progress=progress + half_support,
        )

    def _planning_corridor(self, now, course_coordinates=None):
        surveyed = self._surveyed_corridor(course_coordinates)
        camera = self._camera_corridor(now)
        if surveyed is None:
            return camera
        if (
            camera is not None
            and abs(camera[2] - surveyed[2]) <= self.map_corridor_max_residual
        ):
            return camera
        return surveyed

    def _lane_points(self, heading_error, corridor):
        points = self.scan_points
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        if self.scan_odom_pose is not None:
            scan_x, scan_y, scan_yaw = self.scan_odom_pose
            scan_cosine = math.cos(scan_yaw)
            scan_sine = math.sin(scan_yaw)
            odom_x = scan_x + scan_cosine * points[:, 0] - scan_sine * points[:, 1]
            odom_y = scan_y + scan_sine * points[:, 0] + scan_cosine * points[:, 1]
            dx = odom_x - self.odom_x
            dy = odom_y - self.odom_y
            current_cosine = math.cos(self.odom_yaw)
            current_sine = math.sin(self.odom_yaw)
            points = np.column_stack(
                (
                    current_cosine * dx + current_sine * dy,
                    -current_sine * dx + current_cosine * dy,
                )
            )
        cosine = math.cos(heading_error)
        sine = math.sin(heading_error)
        lane_points = np.column_stack(
            (
                cosine * points[:, 0] + sine * points[:, 1],
                -sine * points[:, 0] + cosine * points[:, 1],
            )
        )
        right, left, _ = corridor
        usable = (
            (lane_points[:, 0] >= -self.scan_rear_limit)
            & (lane_points[:, 0] <= self.scan_max_range)
            & (lane_points[:, 1] >= right - 0.12)
            & (lane_points[:, 1] <= left + 0.12)
        )
        return lane_points[usable]

    def _current_pose(self):
        return Pose2D(self.odom_x, self.odom_y, self.odom_yaw)

    def _lane_points_in_odom(self, lane_points, heading_error):
        """Express the time-aligned LiDAR points in the frozen odom frame."""
        if lane_points.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        course_heading = self.odom_yaw + heading_error
        cosine = math.cos(course_heading)
        sine = math.sin(course_heading)
        return np.column_stack(
            (
                self.odom_x + cosine * lane_points[:, 0] - sine * lane_points[:, 1],
                self.odom_y + sine * lane_points[:, 0] + cosine * lane_points[:, 1],
            )
        )

    def _acquisition_data_problem(self, now):
        checks = (
            (self.scan_stamp, self.scan_timeout, "scan"),
            (self.odom_stamp, self.odom_timeout, "odometry"),
        )
        for stamp, timeout, name in checks:
            if stamp is None or (now - stamp).to_sec() > timeout:
                return name
        return None

    def _source_stamp_is_fresh(self, now, stamp, timeout):
        if stamp is None:
            return False
        age = (now - stamp).to_sec()
        future_tolerance = getattr(self, "maximum_scan_odom_skew", 0.0)
        return -future_tolerance <= age <= timeout

    def _scan_points_in_odom(self):
        """Return the ordered LiDAR samples in odom at their source stamp."""

        points = np.asarray(self.scan_points, dtype=np.float64).reshape((-1, 2))
        if points.size == 0 or self.scan_odom_pose is None:
            return np.empty((0, 2), dtype=np.float64)
        scan_x, scan_y, scan_yaw = self.scan_odom_pose
        cosine = math.cos(scan_yaw)
        sine = math.sin(scan_yaw)
        return np.column_stack(
            (
                scan_x + cosine * points[:, 0] - sine * points[:, 1],
                scan_y + sine * points[:, 0] + cosine * points[:, 1],
            )
        )

    def _update_local_registration(self):
        """Consume each source-stamped scan once and confirm one SE(2)."""

        if self.odom_from_course is not None:
            return True
        if (
            self.arm_generation <= 0
            or self.armed_at is None
            or self.scan_stamp is None
            or self.scan_stamp <= self.armed_at
            or self.scan_generation == self.last_registration_scan_generation
            or self.scan_odom_pose is None
        ):
            return False
        self.last_registration_scan_generation = self.scan_generation
        points = self._scan_points_in_odom()
        scan_pose = Pose2D.from_value(self.scan_odom_pose)
        faces = extract_barrier_faces(
            points,
            scan_pose.yaw,
            self.cluster_link_distance,
            self.cluster_minimum_points,
            self.registration_face_minimum_length,
            self.registration_face_maximum_length,
            self.registration_face_maximum_rms,
            self.registration_face_heading_tolerance,
        )
        result = register_obstacle_outer_faces(
            self.registration_template,
            faces,
            self.scan_stamp.to_sec(),
            self.odom_frame,
            scan_pose,
            self.registration_config,
            self.registration_acquisition_minimum_progress,
            self.registration_acquisition_maximum_progress,
            self.registration_acquisition_maximum_lateral,
            self.registration_acquisition_maximum_heading,
            minimum_face_length=self.registration_outer_face_minimum_length,
            baseline_tolerance=self.registration_outer_baseline_tolerance,
            parallel_heading_tolerance=(
                self.registration_outer_parallel_heading_tolerance
            ),
            finite_endpoint_tolerance=(
                self.registration_finite_endpoint_tolerance
            ),
            side_minimum_length=self.registration_side_minimum_length,
            side_heading_tolerance=(
                self.registration_side_heading_tolerance
            ),
            side_association_tolerance=(
                self.registration_side_association_tolerance
            ),
            baseline_lateral_tolerance=(
                self.registration_baseline_lateral_tolerance
            ),
            baseline_lateral_maximum=(
                self.registration_baseline_lateral_maximum
            ),
            axis_consistency_tolerance=(
                self.registration_axis_consistency_tolerance
            ),
            ambiguity_position=self.registration_ambiguity_position,
            ambiguity_heading=self.registration_ambiguity_heading,
        )
        if result is None:
            # A 10 Hz LiDAR view can alternate between a clean three-face
            # signature and an occluded scan while the lane controller keeps
            # moving. Keep only already accepted, source-stamped hypotheses;
            # TemporalRegistrationFilter clears them automatically when the
            # next accepted result exceeds maximum_gap. A rejected scan never
            # contributes evidence and therefore cannot confirm a transform.
            self.planner_status_pub.publish(String(data="REGISTERING"))
            rospy.logwarn_throttle(
                1.0,
                "Obstacle registration pending: extracted_faces=%d "
                "pose=(%.3f, %.3f, %.1fdeg)",
                len(faces),
                scan_pose.x,
                scan_pose.y,
                math.degrees(scan_pose.yaw),
            )
            return False
        temporal = self.registration_filter.update(result)
        self.registration_diagnostics = result.diagnostics
        if not temporal.confirmed:
            self.planner_status_pub.publish(String(data="REGISTERING"))
            rospy.loginfo_throttle(
                1.0,
                "Obstacle registration confirming: %d/%d faces=%d "
                "position_spread=%.4fm heading_spread=%.2fdeg",
                temporal.confirmation_count,
                self.registration_filter.config.required_confirmations,
                result.diagnostics.inlier_landmarks,
                temporal.maximum_position_spread,
                math.degrees(temporal.maximum_heading_spread),
            )
            return False
        self.odom_from_course = temporal.transform
        self.registration_covariance = temporal.covariance
        self.registration_source_stamp = self.scan_stamp
        entry = entry_plane_progress(
            self.registration_template.entry_plane,
            scan_pose,
            self.odom_from_course,
        )
        self.planner_status_pub.publish(String(data="REGISTERED"))
        rospy.loginfo(
            "LiDAR obstacle frame confirmed: progress=%.3fm lateral=%.3fm "
            "faces=%d rms=%.4fm yaw=%.2fdeg",
            entry.longitudinal,
            entry.lateral,
            result.diagnostics.inlier_landmarks,
            result.diagnostics.position_rms,
            math.degrees(self.odom_from_course.target_from_source_yaw),
        )
        return True

    def _entry_is_ready(self):
        if self.odom_from_course is None or not self.odom_ready:
            return False
        entry = entry_plane_progress(
            self.registration_template.entry_plane,
            self._current_pose(),
            self.odom_from_course,
        )
        return (
            self.ready_lead_minimum_progress
            <= entry.longitudinal
            <= self.ready_lead_maximum_progress
            and abs(entry.lateral) <= self.registration_entry_maximum_lateral
            and abs(entry.heading_error) <= self.registration_entry_maximum_heading
        )

    def _publish_ready(self, now=None):
        """Publish fresh readiness without rebuilding the frozen path.

        Full-path validation may take longer than the manager's readiness age
        limit.  Readiness therefore carries the newest source time for which
        both the already validated path and fresh LiDAR/odometry are available,
        and is refreshed on later sensor samples until the gate opens.
        """

        if (
            self.arm_generation <= 0
            or self.prepared_generation != self.arm_generation
            or self.prepared_stamp is None
        ):
            return False
        now = rospy.Time.now() if now is None else now
        if self._acquisition_data_problem(now) is not None:
            return False
        if not self._entry_is_ready():
            return False
        source_stamp = (
            self.scan_stamp
            if self.scan_stamp <= self.odom_stamp
            else self.odom_stamp
        )
        if (
            source_stamp is None
            or (self.armed_at is not None and source_stamp <= self.armed_at)
            or (
                self.last_ready_stamp is not None
                and source_stamp <= self.last_ready_stamp
            )
        ):
            return False
        ready = Header()
        ready.seq = self.arm_generation
        ready.stamp = source_stamp
        ready.frame_id = "obstacle"
        self.ready_pub.publish(ready)
        self.ready_published_generation = self.arm_generation
        self.last_ready_stamp = source_stamp
        return True

    def _attempt_path(self, now, refresh=False):
        if not self._update_local_registration():
            return False
        if not self._entry_is_ready():
            self.planner_status_pub.publish(String(data="APPROACHING_ENTRY"))
            return False
        if refresh:
            # The lane controller may move on new odometry while its entry
            # cap settles. Permit one takeover rebuild from the current scan,
            # then wait for the next scan if that fresh-pose sweep is rejected.
            if self.scan_generation == getattr(
                self, "last_refresh_scan_generation", -1
            ):
                return False
            self.last_refresh_scan_generation = self.scan_generation
        elif self.scan_generation == self.last_plan_scan_generation:
            return False
        self.last_plan_scan_generation = self.scan_generation
        if not refresh:
            # Full-path planning takes about one LiDAR period.  Begin the lane
            # controller's bounded deceleration before that sweep, instead of
            # spending the narrow swept-safe G2 join window at cruise speed.
            # The gate edge republishes the same cap and records fresh command
            # and odometry generations before ownership can change.
            self.speed_limit_pub.publish(
                Float64(data=self.entry_velocity_cap)
            )
        heading_error = self._course_heading_error()
        course_coordinates = self._course_template_coordinates()
        if course_coordinates is None:
            return False
        progress, lateral_offset = course_coordinates
        corridor = self._planning_corridor(now, course_coordinates)
        if corridor is None:
            return False
        started = time.monotonic()

        path = self.spline_planner.plan(
            progress,
            lateral_offset,
            # Route selection never detours around a transient return.  The
            # same scan is checked immediately after takeover by motion_safety.
            np.empty((0, 2), dtype=np.float64),
            corridor[0],
            corridor[1],
        )
        if path is None:
            self.planner_status_pub.publish(String(data="PATH_REJECTED"))
            return False

        current_heading = normalize_angle(-heading_error)
        start_curvature = 0.0
        if abs(self.observed_lane_linear) >= self.minimum_velocity:
            start_curvature = clamp(
                self.observed_lane_angular / self.observed_lane_linear,
                -self.maximum_angular_velocity / self.minimum_velocity,
                self.maximum_angular_velocity / self.minimum_velocity,
            )
        path = self.spline_planner.connect_entry(
            path,
            current_heading,
            start_curvature=start_curvature,
        )
        if path is None:
            rospy.logwarn_throttle(
                1.0,
                "Obstacle entry connector rejected: no curvature-continuous "
                "connector-plus-suffix passed the rectangle sweep",
            )
            self.planner_status_pub.publish(String(data="PATH_REJECTED"))
            return False

        candidate_path, _ = freeze_path_in_odom(
            path,
            source_pose=Pose2D(0.0, 0.0, current_heading),
            odom_pose=self._current_pose(),
            odom_frame=self.odom_frame,
        )
        self._attach_local_safety_boundary(candidate_path)
        fixed_validation = self.path_checker.validator.validate_path(
            candidate_path
        )
        if not fixed_validation.safe:
            rospy.logwarn_throttle(
                1.0,
                "Obstacle frozen path rejected by full rectangle sweep: "
                "line=%.4fm obstacle=%.4fm",
                fixed_validation.minimum_line_clearance,
                fixed_validation.minimum_obstacle_clearance,
            )
            self.planner_status_pub.publish(String(data="PATH_REJECTED"))
            return False
        candidate_follower = PathFollower(self.tracking_config)
        projection = candidate_follower.reset(
            candidate_path,
            self._current_pose(),
            initial_linear=min(
                max(0.0, self.observed_lane_linear), self.entry_velocity_cap
            ),
            initial_angular=clamp(
                self.observed_lane_angular,
                -self.maximum_angular_velocity,
                self.maximum_angular_velocity,
            ),
        )
        candidate_follower.update_clearance(fixed_validation)
        candidate_path.line_clearance = fixed_validation.minimum_line_clearance
        candidate_path.obstacle_clearance = (
            fixed_validation.minimum_obstacle_clearance
        )
        candidate_path.map_clearance = fixed_validation.minimum_map_clearance

        # Commit atomically.  If a takeover refresh is rejected, the lane
        # controller keeps ownership and the pre-gate path remains available
        # for diagnostics instead of being partially replaced.
        self.committed_path = candidate_path
        self.path_follower = candidate_follower
        self.fixed_path_validation = fixed_validation
        self.remaining_distance = max(
            0.0, self.committed_path.length - projection.station
        )
        self.live_speed_limit = math.inf
        self.live_requires_stop = False
        self.planning_seconds = time.monotonic() - started
        self.prepared_generation = self.arm_generation
        self.prepared_stamp = self.odom_stamp
        self._publish_path()
        self._publish_diagnostics()
        self.planner_status_pub.publish(String(data="PATH_COMMITTED"))
        self._publish_ready()
        rospy.loginfo(
            "Prepared obstacle path while lane retained control in %.3fs: "
            "generation=%d remaining=%.3fm clearance=%.4fm line=%.4fm",
            self.planning_seconds,
            self.prepared_generation,
            self.remaining_distance,
            self.committed_path.obstacle_clearance,
            self.committed_path.line_clearance,
        )
        return True

    def _validate_committed_path(self, now):
        # Freeze one LiDAR/path snapshot for the comparatively expensive route
        # sweep.  Odometry is intentionally allowed to advance during that
        # calculation; the actual reaction/complete-stop sweep is recomputed
        # below from the newest pose while holding the lock for only its short,
        # bounded calculation.  Requiring an unchanged 30 Hz odom generation
        # here can starve a 20 Hz command loop indefinitely.
        with self.lock:
            if self.committed_path is None or self.path_follower.path is None:
                self._fail("no frozen path is active")
                return False
            stale_inputs = []
            if not self._source_stamp_is_fresh(
                now, self.odom_stamp, self.odom_timeout
            ):
                stale_inputs.append("odometry")
            if not self._source_stamp_is_fresh(
                now, self.scan_stamp, self.scan_timeout
            ):
                stale_inputs.append("LiDAR")
            if stale_inputs:
                self._publish_stop()
                rospy.logwarn_throttle(
                    1.0,
                    "Obstacle holds zero for stale swept-safety input: %s",
                    ", ".join(stale_inputs),
                )
                return False
            path = self.committed_path
            follower = self.path_follower
            fixed_validation = self.fixed_path_validation
            route_scan_stamp = self.scan_stamp
            route_odom_frame = self.odom_frame
            route_odom_epoch = getattr(self, "odom_epoch", 0)
            pose = self._current_pose()
            heading_error = self._course_heading_error()
            course_coordinates = self._course_template_coordinates()
            corridor = self._planning_corridor(now, course_coordinates)
            if corridor is None:
                self._fail("lane corridor unavailable")
                return False
            lane_points = self._lane_points(heading_error, corridor)
            if course_coordinates is not None:
                progress, lateral_offset = course_coordinates
                lane_points = self.spline_planner.stabilize_known_barrier_returns(
                    lane_points,
                    progress,
                    -lateral_offset,
                )
            live_obstacles = self._lane_points_in_odom(
                lane_points, heading_error
            )
            line_boundaries = path.safety.line_boundaries
            live_line_boundaries = ()
            camera_boundary = self._camera_line_boundary(now, corridor)
            if camera_boundary is not None:
                live_line_boundaries = (camera_boundary,)
                line_boundaries += live_line_boundaries
            runtime_margins = SafetyMargins(
                line=getattr(
                    self,
                    "runtime_line_margin",
                    self.validation_footprint.line_margin,
                ),
                obstacle=getattr(
                    self,
                    "runtime_obstacle_padding",
                    self.footprint.obstacle_padding,
                ),
                localization=getattr(
                    self,
                    "runtime_localization_margin",
                    self.validation_footprint.localization_margin,
                ),
                tracking=getattr(
                    self,
                    "runtime_tracking_margin",
                    self.validation_footprint.tracking_margin,
                ),
            )
            runtime_safety = PathSafety(
                line_boundaries=line_boundaries,
                map_boundaries=path.safety.map_boundaries,
                fixed_obstacles=path.safety.fixed_obstacles,
                margins=runtime_margins,
            )
            # The immutable route, fixed barriers and surveyed boundaries were
            # already certified over the complete path.  The rolling route
            # sweep only needs new LiDAR points and a fresh camera boundary;
            # the latest-pose stop sweep below still checks every boundary and
            # fixed obstacle in runtime_safety.
            live_route_safety = PathSafety(
                line_boundaries=live_line_boundaries,
                margins=runtime_margins,
            )
            validator = self.path_checker.validator
            live_validation_distance = (
                self.spline_planner.live_validation_distance
            )
            maximum_snapshot_advance = (
                max(
                    getattr(
                        self,
                        "entry_velocity_cap",
                        follower.config.maximum_linear_velocity,
                    ),
                    follower.config.maximum_linear_velocity,
                    abs(self.odom_linear_velocity),
                    abs(follower.last_linear),
                )
                * max(self.odom_timeout, self.scan_timeout)
                + validator.translation_step
            )
            # Start at the last command-committed station, not a speculative
            # projection that is not recorded until a command is produced.
            # Every later tracking result is constrained not to regress behind
            # this value, so the prevalidated interval cannot start ahead of
            # the pose used for the stop sweep.
            route_start_station = follower.path_station

        route_validation = validator.validate_path(
            path,
            start_station=route_start_station,
            maximum_distance=(
                live_validation_distance + maximum_snapshot_advance
            ),
            safety=live_route_safety,
            live_obstacles=live_obstacles,
        )

        with self.lock:
            # A gate/restart can replace either object while the route geometry
            # runs. Never apply its result to a different route/follower.
            if self.committed_path is not path or self.path_follower is not follower:
                return False
            completed = rospy.Time.now()
            stale_inputs = []
            if not self._source_stamp_is_fresh(
                completed, self.odom_stamp, self.odom_timeout
            ):
                stale_inputs.append("odometry")
            if not self._source_stamp_is_fresh(
                completed, route_scan_stamp, self.scan_timeout
            ):
                stale_inputs.append("LiDAR")
            if getattr(self, "odom_epoch", 0) != route_odom_epoch:
                stale_inputs.append("odometry epoch")
            if self.odom_frame != route_odom_frame or self.odom_frame != path.frame_id:
                stale_inputs.append("odometry frame")
            if stale_inputs:
                self._publish_stop()
                rospy.logwarn_throttle(
                    1.0,
                    "Obstacle holds zero after swept-safety input aged out: %s",
                    ", ".join(stale_inputs),
                )
                return False

            # The long route result remains valid as odometry advances because
            # its checked interval included the maximum fresh-snapshot travel.
            # Calculate both tracking and the complete-stop envelope from the
            # latest measured pose.  This short sweep stays under the lock so a
            # newer pose cannot invalidate it before the safety decision is
            # committed.
            pose = self._current_pose()
            tracking = follower.calculate_tracking(pose)
            station_advance = tracking.station - route_start_station
            if (
                station_advance < -1e-9
                or station_advance > maximum_snapshot_advance + 1e-9
            ):
                self._publish_stop()
                rospy.logwarn_throttle(
                    1.0,
                    "Obstacle holds zero after route-sweep pose advance "
                    "%.4fm exceeded its %.4fm validated reserve",
                    station_advance,
                    maximum_snapshot_advance,
                )
                return False
            measured_speed = max(
                abs(self.odom_linear_velocity),
                abs(follower.last_linear),
            )
            angular_velocities = follower.stopping_angular_velocities(
                tracking,
                tracking.direction * measured_speed,
                self.odom_angular_velocity,
            )
            stopping_obstacles = live_obstacles
            if self.scan_stamp != route_scan_stamp:
                latest_heading_error = self._course_heading_error()
                latest_coordinates = self._course_template_coordinates()
                latest_corridor = self._planning_corridor(
                    completed, latest_coordinates
                )
                if latest_corridor is None:
                    self._publish_stop()
                    return False
                latest_lane_points = self._lane_points(
                    latest_heading_error, latest_corridor
                )
                if latest_coordinates is not None:
                    latest_progress, latest_lateral_offset = latest_coordinates
                    latest_lane_points = (
                        self.spline_planner.stabilize_known_barrier_returns(
                            latest_lane_points,
                            latest_progress,
                            -latest_lateral_offset,
                        )
                    )
                stopping_obstacles = self._lane_points_in_odom(
                    latest_lane_points, latest_heading_error
                )
            safety = validator.motion_safety(
                path,
                pose,
                tracking.path_index,
                tracking.target_speed,
                tracking.direction * measured_speed,
                angular_velocities,
                self.safety_reaction_time,
                self.linear_deceleration,
                self.safety_stop_margin,
                safety=runtime_safety,
                live_obstacles=stopping_obstacles,
                tracking=tracking,
                route_validation=route_validation,
                route_start_station=route_start_station,
            )

            validation = combine_validation_results(
                (fixed_validation, safety.validation)
            )
            self.live_speed_limit = safety.speed_limit
            self.live_requires_stop = safety.requires_stop
            follower.update_clearance(validation)
            path.obstacle_clearance = validation.minimum_obstacle_clearance
            path.line_clearance = validation.minimum_line_clearance
            path.map_clearance = validation.minimum_map_clearance
            self.remaining_distance = max(0.0, path.length - tracking.station)

            if safety.requires_stop:
                rospy.logwarn_throttle(
                    0.5,
                    "Obstacle path holds a bounded stop: line=%.4fm obstacle=%.4fm",
                    validation.minimum_line_clearance,
                    validation.minimum_obstacle_clearance,
                )
            elif safety.speed_limit + 1e-9 < tracking.target_speed:
                rospy.logwarn_throttle(
                    0.5,
                    "Obstacle path slows for predicted contact: limit=%.3fm/s "
                    "line=%.4fm obstacle=%.4fm",
                    safety.speed_limit,
                    validation.minimum_line_clearance,
                    validation.minimum_obstacle_clearance,
                )
            self._publish_diagnostics()
            return tracking, pose

    def _publish_path(self):
        message = Path()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.odom_frame
        for x, y, yaw in zip(
            self.committed_path.x,
            self.committed_path.y,
            self.committed_path.heading,
        ):
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.055
            pose.pose.orientation.z = math.sin(0.5 * float(yaw))
            pose.pose.orientation.w = math.cos(0.5 * float(yaw))
            message.poses.append(pose)
        self.path_pub.publish(message)

    def _publish_diagnostics(self):
        message = Float64MultiArray()
        message.data = self.path_follower.diagnostics.as_array()
        self.diagnostics_pub.publish(message)

    def _common_path_command(self, now, tracking=None, pose=None):
        if self.committed_path is None or self.path_follower.path is None:
            return None
        pose = self._current_pose() if pose is None else Pose2D.from_value(pose)
        if tracking is None:
            tracking = self.path_follower.calculate_tracking(pose)
        if self.last_command_time is None:
            elapsed = self.control_period
        else:
            elapsed = clamp((now - self.last_command_time).to_sec(), 0.0, 0.15)
        limited, tracking = self.path_follower.command(
            pose,
            elapsed,
            speed_limit=self.live_speed_limit,
            tracking=tracking,
        )
        self.remaining_distance = self.path_follower.diagnostics.remaining_distance
        self.last_command_time = now
        command = Twist()
        command.linear.x = limited.linear_velocity
        command.angular.z = limited.angular_velocity
        self._publish_diagnostics()
        return command

    def _terminal_hold_command(self, now, tracking, pose):
        """Bounded stop while the AMCL-owned handoff signal catches up."""
        if self.last_command_time is None:
            elapsed = self.control_period
        else:
            elapsed = clamp((now - self.last_command_time).to_sec(), 0.0, 0.15)
        limited, _tracking, _status = self.path_follower.terminal_hold(
            pose,
            elapsed,
            tracking=tracking,
        )
        self.remaining_distance = self.path_follower.diagnostics.remaining_distance
        self.last_command_time = now
        command = Twist()
        command.linear.x = limited.linear_velocity
        command.angular.z = limited.angular_velocity
        self._publish_diagnostics()
        return command

    def _complete(self):
        self.planner_status_pub.publish(String(data="COMPLETE"))
        self.speed_limit_pub.publish(Float64(data=self.lane_resume_max_velocity))
        if not self._set_lane_controller(True):
            self._fail("could not return cmd_vel to lane controller")
            return
        self._set_state(self.COMPLETE)
        rospy.loginfo("Obstacle course complete; lane controller owns cmd_vel")

    def control_callback(self, _event):
        handoff_service_requested = False
        with self.lock:
            now = rospy.Time.now()
            if self.shutting_down:
                return
            if self.revoke_requested:
                self._revoke()
                return
            if (
                self.state == self.WAIT_GATE
                and not self.zone_gate
                and self.arm_generation > 0
                and self.armed_at is not None
            ):
                if self.manual_stop:
                    return
                if self.prepared_generation == self.arm_generation:
                    self._publish_ready(now)
                else:
                    problem = self._acquisition_data_problem(now)
                    if problem is None:
                        self._attempt_path(now)
                    else:
                        rospy.logwarn_throttle(
                            2.0,
                            "Waiting for obstacle pre-registration input: %s",
                            problem,
                        )
                return
            if self.start_requested and self.zone_gate and self.state in (
                self.WAIT_GATE,
                self.COMPLETE,
            ):
                self._start_run()

            if self.state in (self.WAIT_GATE, self.COMPLETE):
                return
            if self.manual_stop:
                if self.mission_has_control:
                    self._publish_stop()
                return
            if self.state == self.FAILED:
                self._publish_stop()
                return

            if self.state == self.ACQUIRING:
                if self.committed_path is None:
                    rospy.logerr_throttle(
                        1.0,
                        "Obstacle active state lost its prepared CommonPath",
                    )
                    return
                if self.handoff_service_pending:
                    return

                now = rospy.Time.now()
                problem = self._acquisition_data_problem(now)
                if problem is not None:
                    rospy.logwarn_throttle(
                        1.0,
                        "Committed obstacle path waits for refreshed input: %s",
                        problem,
                    )
                    return
                if self.mission_has_control:
                    # The lane publisher is already disabled. Hold zero until
                    # an odom sample sourced after the service response proves
                    # that the frozen path is still joined to the real pose.
                    self._publish_stop()
                    if not self._activate_after_handoff_service(now):
                        rospy.logwarn_throttle(
                            1.0,
                            "Obstacle holds zero for post-service odometry "
                            "and entry reprojection",
                        )
                        return
                    now = rospy.Time.now()
                else:
                    if (
                        abs(self._course_heading_error())
                        > self.acquisition_heading_tolerance
                    ):
                        return
                    handoff_problem = self._entry_handoff_input_problem(now)
                    if handoff_problem is not None:
                        rospy.logwarn_throttle(
                            1.0,
                            "Obstacle waits for capped post-gate lane motion "
                            "while lane control remains active: %s",
                            handoff_problem,
                        )
                        return
                    if self.handoff_refresh_odom_generation < 0:
                        # The lane has moved while the cap settled. Rebuild the
                        # G2 connector from this exact post-gate pose and wait
                        # for newer pose/command callbacks before ownership.
                        if not self._attempt_path(now, refresh=True):
                            rospy.logwarn_throttle(
                                1.0,
                                "Obstacle waits for a fresh swept-valid entry "
                                "connector while lane control remains active",
                            )
                            return
                        self.handoff_refresh_scan_generation = self.scan_generation
                        self.handoff_refresh_lane_command_generation = (
                            self.lane_command_generation
                        )
                        self.handoff_refresh_odom_generation = self.odom_generation
                        self.handoff_refresh_lane_command_received = (
                            self.observed_lane_command_received
                        )
                        self.handoff_refresh_odom_stamp = self.odom_stamp
                        self.handoff_refresh_odom_frame = self.odom_frame
                        return

                    handoff_problem = self._entry_handoff_input_problem(
                        now,
                        minimum_odom_generation=(
                            self.handoff_refresh_odom_generation
                        ),
                        minimum_lane_command_generation=(
                            self.handoff_refresh_lane_command_generation
                        ),
                        minimum_odom_stamp=self.handoff_refresh_odom_stamp,
                        minimum_lane_command_received=(
                            self.handoff_refresh_lane_command_received
                        ),
                        minimum_odom_frame=self.handoff_refresh_odom_frame,
                    )
                    if handoff_problem is not None:
                        rospy.logwarn_throttle(
                            1.0,
                            "Obstacle waits for capped post-refresh lane motion "
                            "while lane control remains active: %s",
                            handoff_problem,
                        )
                        return
                    current_pose = self._current_pose()
                    projection = self.path_follower.reset(
                        self.committed_path,
                        current_pose,
                        initial_linear=min(
                            max(0.0, self.observed_lane_linear),
                            self.entry_velocity_cap,
                        ),
                        initial_angular=clamp(
                            self.observed_lane_angular,
                            -self.maximum_angular_velocity,
                            self.maximum_angular_velocity,
                        ),
                    )
                    join_heading_error = abs(
                        normalize_angle(projection.heading - current_pose.yaw)
                    )
                    if (
                        projection.distance
                        > self.entry_projection_position_tolerance
                        or join_heading_error
                        > self.entry_projection_heading_tolerance
                    ):
                        rospy.logwarn(
                            "Obstacle refreshed entry moved outside join tolerance: "
                            "join=%.4fm/%.2fdeg",
                            projection.distance,
                            math.degrees(join_heading_error),
                        )
                        self._reset_handoff_refresh()
                        self.planner_status_pub.publish(
                            String(data="PATH_REJECTED")
                        )
                        return
                    self.remaining_distance = max(
                        0.0,
                        self.committed_path.length - projection.station,
                    )
                    self.handoff_service_pending = True
                    handoff_service_requested = True

        if handoff_service_requested:
            handoff_succeeded = self._set_lane_controller(False)
            with self.lock:
                self.handoff_service_pending = False
                if not handoff_succeeded:
                    self._fail("could not acquire cmd_vel control")
                    return
                # The mission is now the sole publisher. Stop the old lane
                # command immediately, then require source-stamped odometry
                # strictly newer than this service response before moving.
                self._publish_stop()
                self.handoff_takeover_odom_generation = self.odom_generation
                self.handoff_takeover_odom_stamp = self.odom_stamp
                self.handoff_takeover_odom_frame = self.odom_frame
                self.last_command_time = None
                if self.revoke_requested:
                    self._revoke()
            return

        validated = self._validate_committed_path(now)
        if not validated:
            return
        tracking, tracking_pose = validated

        with self.lock:
            # Sensor and zone callbacks may run during the unlocked geometric
            # sweep. Honour any safety/ownership change before publishing.
            if self.shutting_down:
                return
            if self.revoke_requested:
                self._revoke()
                return
            if self.manual_stop:
                if self.mission_has_control:
                    self._publish_stop()
                return
            if self.state == self.FAILED or not self.mission_has_control:
                return
            if self.live_requires_stop:
                # A failed complete-stop sweep is already an immediate safety
                # decision.  Passing only a zero linear speed limit through the
                # follower can still produce angular motion and sweep the
                # asymmetric footprint farther into the predicted contact.
                self._publish_stop()
                self._publish_diagnostics()
                return
            goal = self.path_follower.goal_status(
                tracking_pose, tracking=tracking
            )
            self.remaining_distance = goal.remaining_distance
            if goal.complete:
                if self.state == self.AVOIDING:
                    self._set_state(self.REJOINING)
                self._complete()
                return
            if (
                goal.crossed_terminal
                and goal.position_error
                > self.committed_path.goal_tolerance.terminal_crossing
            ):
                self._fail("fixed path crossed its local terminal outside tolerance")
                return
            if goal.crossed_terminal:
                # Crossing the selected CommonPath ends translation even if a
                # heading correction still prevents handoff. The common hold
                # retains only the bounded endpoint-heading correction.
                self.cmd_pub.publish(
                    self._terminal_hold_command(
                        rospy.Time.now(), tracking, tracking_pose
                    )
                )
                return

            command = self._common_path_command(
                rospy.Time.now(), tracking=tracking, pose=tracking_pose
            )
            if command is None:
                self._fail("fixed path ended before local goal completion")
                return
            self.cmd_pub.publish(command)

    def shutdown(self):
        with self.lock:
            self.shutting_down = True
            if self.mission_has_control:
                self.cmd_pub.publish(Twist())
        rospy.loginfo("Obstacle mission controller stopped")


if __name__ == "__main__":
    rospy.init_node("obstacle_mission_controller")
    ObstacleMissionController()
    rospy.spin()
