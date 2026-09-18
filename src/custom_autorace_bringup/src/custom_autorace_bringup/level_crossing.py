"""ROS-independent LiDAR geometry for a lowered level-crossing bar."""

from dataclasses import dataclass
import math
import operator
from typing import Iterable, Optional


def _finite_float(name, value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("{} must be a finite number".format(name)) from error
    if not math.isfinite(result):
        raise ValueError("{} must be a finite number".format(name))
    return result


def _integer(name, value):
    if isinstance(value, bool):
        raise ValueError("{} must be an integer".format(name))
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError("{} must be an integer".format(name)) from error


@dataclass(frozen=True)
class HorizontalBarrierConfig:
    """Thresholds used to select one transverse cluster from a laser scan.

    ``max_adjacent_beam_gap`` is an index difference, so one means strictly
    consecutive scan beams and two permits one rejected or missing beam.
    """

    min_forward_distance: float
    max_forward_distance: float
    half_width: float
    max_adjacent_beam_gap: int
    max_point_gap: float
    min_points: int
    min_lateral_span: float
    max_depth_spread: float

    def __post_init__(self):
        float_fields = (
            "min_forward_distance",
            "max_forward_distance",
            "half_width",
            "max_point_gap",
            "min_lateral_span",
            "max_depth_spread",
        )
        for name in float_fields:
            object.__setattr__(self, name, _finite_float(name, getattr(self, name)))

        object.__setattr__(
            self,
            "max_adjacent_beam_gap",
            _integer("max_adjacent_beam_gap", self.max_adjacent_beam_gap),
        )
        object.__setattr__(
            self, "min_points", _integer("min_points", self.min_points)
        )

        if self.min_forward_distance < 0.0:
            raise ValueError("min_forward_distance must be non-negative")
        if self.max_forward_distance <= self.min_forward_distance:
            raise ValueError(
                "max_forward_distance must exceed min_forward_distance"
            )
        if self.half_width <= 0.0:
            raise ValueError("half_width must be positive")
        if self.max_adjacent_beam_gap < 1:
            raise ValueError("max_adjacent_beam_gap must be at least one")
        if self.max_point_gap <= 0.0:
            raise ValueError("max_point_gap must be positive")
        if self.min_points < 2:
            raise ValueError("min_points must be at least two")
        if self.min_lateral_span <= 0.0:
            raise ValueError("min_lateral_span must be positive")
        if self.max_depth_spread < 0.0:
            raise ValueError("max_depth_spread must be non-negative")


@dataclass(frozen=True)
class HorizontalBarrierDetection:
    """Geometry of one qualifying scan cluster.

    ``forward_distance`` is the nearest forward coordinate in the cluster,
    rather than radial range.  This makes comparisons valid across its width.
    """

    forward_distance: float
    lateral_center: float
    point_count: int
    lateral_span: float
    depth_spread: float
    first_beam_index: int
    last_beam_index: int


@dataclass(frozen=True)
class CrossingFrame:
    """One locally registered crossing plane in an odometry frame.

    The origin lies on the crossing plane and ``yaw`` points in the normal
    downstream travel direction.  Keeping this frame in odometry coordinates
    makes completion independent of any accumulated displacement in the
    course's global map.
    """

    x: float
    y: float
    yaw: float

    def __post_init__(self):
        object.__setattr__(self, "x", _finite_float("x", self.x))
        object.__setattr__(self, "y", _finite_float("y", self.y))
        object.__setattr__(self, "yaw", _finite_float("yaw", self.yaw))


def normalize_angle(angle):
    """Wrap one finite angle to ``[-pi, pi)``."""
    angle = _finite_float("angle", angle)
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compose_pose_2d(parent, child):
    """Compose ``parent <- child`` planar poses.

    Each input is ``(x, y, yaw)``.  This helper is used both for source-stamped
    landmark poses and for the Gazebo-only scan-cluster registration fallback.
    """
    if len(parent) != 3 or len(child) != 3:
        raise ValueError("planar poses must contain (x, y, yaw)")
    parent_x, parent_y, parent_yaw = (
        _finite_float("parent pose", value) for value in parent
    )
    child_x, child_y, child_yaw = (
        _finite_float("child pose", value) for value in child
    )
    cosine = math.cos(parent_yaw)
    sine = math.sin(parent_yaw)
    return (
        parent_x + cosine * child_x - sine * child_y,
        parent_y + sine * child_x + cosine * child_y,
        normalize_angle(parent_yaw + child_yaw),
    )


def crossing_progress(frame, pose):
    """Return base-centre progress along a registered crossing normal."""
    if not isinstance(frame, CrossingFrame):
        raise TypeError("frame must be a CrossingFrame")
    if len(pose) != 3:
        raise ValueError("pose must contain (x, y, yaw)")
    x, y, _yaw = (_finite_float("pose", value) for value in pose)
    return (x - frame.x) * math.cos(frame.yaw) + (
        y - frame.y
    ) * math.sin(frame.yaw)


def crossing_rear_clearance(
    frame,
    pose,
    front,
    rear,
    half_width,
    padding=0.0,
):
    """Return the rear-most rectangular-footprint progress past the plane."""
    if not isinstance(frame, CrossingFrame):
        raise TypeError("frame must be a CrossingFrame")
    if len(pose) != 3:
        raise ValueError("pose must contain (x, y, yaw)")
    pose = tuple(_finite_float("pose", value) for value in pose)
    front = _finite_float("front", front)
    rear = _finite_float("rear", rear)
    half_width = _finite_float("half_width", half_width)
    padding = _finite_float("padding", padding)
    if min(front, rear, half_width) <= 0.0 or padding < 0.0:
        raise ValueError(
            "footprint front/rear/half_width must be positive and padding "
            "must be non-negative"
        )

    heading_delta = normalize_angle(pose[2] - frame.yaw)
    longitudinal_projection = math.cos(heading_delta)
    minimum_longitudinal = min(
        -(rear + padding) * longitudinal_projection,
        (front + padding) * longitudinal_projection,
    )
    minimum_lateral = -(half_width + padding) * abs(
        math.sin(heading_delta)
    )
    return (
        crossing_progress(frame, pose)
        + minimum_longitudinal
        + minimum_lateral
    )


@dataclass(frozen=True)
class _ScanPoint:
    index: int
    x: float
    y: float


def _detection_from_cluster(points, config):
    if len(points) < config.min_points:
        return None

    minimum_x = min(point.x for point in points)
    maximum_x = max(point.x for point in points)
    minimum_y = min(point.y for point in points)
    maximum_y = max(point.y for point in points)
    lateral_span = maximum_y - minimum_y
    depth_spread = maximum_x - minimum_x
    if (
        lateral_span < config.min_lateral_span
        or depth_spread > config.max_depth_spread
    ):
        return None

    return HorizontalBarrierDetection(
        forward_distance=minimum_x,
        lateral_center=0.5 * (minimum_y + maximum_y),
        point_count=len(points),
        lateral_span=lateral_span,
        depth_spread=depth_spread,
        first_beam_index=points[0].index,
        last_beam_index=points[-1].index,
    )


def detect_horizontal_barrier(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    config: HorizontalBarrierConfig,
) -> Optional[HorizontalBarrierDetection]:
    """Return the nearest lowered-bar cluster in a LaserScan-like sequence.

    Each retained sample is projected with the scan metadata as
    ``x = range * cos(angle_min + index * angle_increment)`` and likewise for
    ``y``.  Samples must lie in the configured forward corridor.  A cluster
    only continues while both its beam-index gap and Cartesian point gap stay
    within their configured maxima.

    The function has no ROS dependency; callers pass the five corresponding
    ``sensor_msgs/LaserScan`` fields directly.
    """
    if not isinstance(config, HorizontalBarrierConfig):
        raise TypeError("config must be a HorizontalBarrierConfig")

    angle_min = _finite_float("angle_min", angle_min)
    angle_increment = _finite_float("angle_increment", angle_increment)
    range_min = _finite_float("range_min", range_min)
    range_max = _finite_float("range_max", range_max)
    if angle_increment == 0.0:
        raise ValueError("angle_increment must be non-zero")
    if range_min < 0.0 or range_max < range_min:
        raise ValueError("range bounds must satisfy 0 <= range_min <= range_max")

    detections = []
    cluster = []
    previous = None

    for index, raw_range in enumerate(ranges):
        try:
            distance = float(raw_range)
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            not math.isfinite(distance)
            or distance < range_min
            or distance > range_max
        ):
            continue

        angle = angle_min + index * angle_increment
        if not math.isfinite(angle):
            continue
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        if (
            x < config.min_forward_distance
            or x > config.max_forward_distance
            or abs(y) > config.half_width
        ):
            continue

        point = _ScanPoint(index=index, x=x, y=y)
        if previous is not None:
            beam_gap = point.index - previous.index
            point_gap = math.hypot(point.x - previous.x, point.y - previous.y)
            if (
                beam_gap > config.max_adjacent_beam_gap
                or point_gap > config.max_point_gap
            ):
                detection = _detection_from_cluster(cluster, config)
                if detection is not None:
                    detections.append(detection)
                cluster = []

        cluster.append(point)
        previous = point

    detection = _detection_from_cluster(cluster, config)
    if detection is not None:
        detections.append(detection)
    if not detections:
        return None
    return min(
        detections,
        key=lambda item: (
            item.forward_distance,
            abs(item.lateral_center),
            item.first_beam_index,
        ),
    )


__all__ = [
    "CrossingFrame",
    "HorizontalBarrierConfig",
    "HorizontalBarrierDetection",
    "compose_pose_2d",
    "crossing_progress",
    "crossing_rear_clearance",
    "detect_horizontal_barrier",
    "normalize_angle",
]
