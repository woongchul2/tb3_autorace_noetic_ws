#!/usr/bin/env python3
"""ROS-independent path representation, tracking, and swept safety checks.

Mission controllers select and align routes.  This module owns the common
coordinate freeze, projection, lookahead, feed-forward/feedback tracking,
speed profiling, command limiting, completion, and rectangular sweep rules.

``heading`` is the robot body heading while a sample is traversed. ``station``
always increases in execution order.  Consequently a reverse sample has a
negative ``direction`` while its motion tangent is ``heading + pi``.
"""

from dataclasses import dataclass, field
from functools import lru_cache
import math

import numpy as np


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(quaternion):
    """Return planar yaw from a quaternion-like object.

    Keeping this conversion beside the common planar pose/transform types
    prevents each mission controller from carrying an identical ROS-message
    conversion.  The object only needs ``x``, ``y``, ``z`` and ``w`` fields,
    so the path library remains ROS-independent and unit-testable.
    """
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2),
    )


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        return cls(*(float(component) for component in value))


def sample_in_place_rotation(start_pose, target_yaw, heading_step):
    """Sample the shortest signed rotation about one fixed planar position.

    The returned tuple includes both endpoints for a non-zero rotation and
    contains one endpoint for an already aligned pose.  Every adjacent heading
    change is no larger than ``heading_step``.  Keeping this representation
    separate from :class:`CommonPath` is intentional: a pure rotation has no
    strictly increasing XY station and therefore is not a translational path.
    """
    start = Pose2D.from_value(start_pose)
    target_yaw = float(target_yaw)
    heading_step = float(heading_step)
    if not all(
        math.isfinite(value)
        for value in (start.x, start.y, start.yaw, target_yaw, heading_step)
    ):
        raise ValueError("rotation pose and heading step must be finite")
    if heading_step <= 0.0:
        raise ValueError("rotation heading step must be positive")

    signed_sweep = normalize_angle(target_yaw - start.yaw)
    normalized_target = normalize_angle(target_yaw)
    if abs(signed_sweep) <= 1e-12:
        return (Pose2D(start.x, start.y, normalized_target),)

    segment_count = max(1, int(math.ceil(abs(signed_sweep) / heading_step)))
    poses = tuple(
        Pose2D(
            start.x,
            start.y,
            normalize_angle(
                start.yaw + signed_sweep * float(index) / segment_count
            ),
        )
        for index in range(segment_count + 1)
    )
    # Use the caller's normalized endpoint rather than its interpolated
    # equivalent so endpoint comparisons stay deterministic at the +/-pi wrap.
    return poses[:-1] + (Pose2D(start.x, start.y, normalized_target),)


@dataclass(frozen=True)
class AsymmetricFootprint:
    front: float
    rear: float
    half_width: float

    def __post_init__(self):
        if (
            not all(
                math.isfinite(value)
                for value in (self.front, self.rear, self.half_width)
            )
            or self.front <= 0.0
            or self.rear <= 0.0
            or self.half_width <= 0.0
        ):
            raise ValueError("footprint extents must be finite and positive")

    def expanded(self, margin):
        margin = max(0.0, float(margin))
        return AsymmetricFootprint(
            self.front + margin,
            self.rear + margin,
            self.half_width + margin,
        )


@dataclass(frozen=True)
class GoalTolerance:
    position: float = 0.03
    heading: float = math.radians(8.0)
    terminal_crossing: float = 0.10

    def __post_init__(self):
        if (
            not all(
                math.isfinite(value)
                for value in (
                    self.position,
                    self.heading,
                    self.terminal_crossing,
                )
            )
            or self.position < 0.0
            or self.heading < 0.0
            or self.terminal_crossing < self.position
        ):
            raise ValueError("path goal tolerances are inconsistent")


@dataclass(frozen=True)
class SafetyMargins:
    line: float = 0.0
    obstacle: float = 0.0
    localization: float = 0.0
    tracking: float = 0.0

    def __post_init__(self):
        values = (self.line, self.obstacle, self.localization, self.tracking)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("path safety margins must be finite and non-negative")

    @property
    def uncertainty(self):
        return self.localization + self.tracking


@dataclass
class PathSafety:
    """Safety evidence carried with an executable path.

    Boundary objects implement ``clearance(pose, footprint)`` and return a
    signed distance: positive is clear, zero is contact, negative is overlap.
    Point obstacles are expressed in the same frame as the path.
    """

    line_boundaries: tuple = field(default_factory=tuple)
    map_boundaries: tuple = field(default_factory=tuple)
    fixed_obstacles: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    margins: SafetyMargins = field(default_factory=SafetyMargins)

    def __post_init__(self):
        self.line_boundaries = tuple(self.line_boundaries)
        self.map_boundaries = tuple(self.map_boundaries)
        points = np.asarray(self.fixed_obstacles, dtype=np.float64)
        if points.size == 0:
            points = np.empty((0, 2), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("fixed obstacles must contain [x, y] points")
        if not np.all(np.isfinite(points)):
            raise ValueError("fixed obstacle points must be finite")
        self.fixed_obstacles = points.copy()


def _as_vector(values, count, name, default=None, dtype=np.float64):
    if values is None:
        if default is None:
            raise ValueError("%s is required" % name)
        values = default
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 0:
        array = np.full(count, array.item(), dtype=dtype)
    if array.shape != (count,):
        raise ValueError("%s must have one value per path point" % name)
    return array


@dataclass
class CommonPath:
    """One executable path segment shared by every path-based mission.

    One segment has one motion direction.  A forward/reverse transition is a
    controller state transition to another ``CommonPath`` followed by
    :meth:`PathFollower.reset`; this guarantees that the existing slew limiter
    brings the vehicle to rest before the new signed command is issued.
    """

    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    curvature: np.ndarray
    station: np.ndarray = None
    speed: np.ndarray = None
    direction: np.ndarray = 1
    feedforward_scale: np.ndarray = 1.0
    # A route may begin while the measured footprint is already over a line
    # (for example, after a camera-only approach through a very tight bend).
    # This per-station envelope is the only explicit exception: zero keeps the
    # normal no-contact rule, while a positive value permits that much signed
    # line overlap and can taper to zero as the common follower converges.
    line_overlap_allowance: np.ndarray = 0.0
    frame_id: str = "map"
    goal_tolerance: GoalTolerance = field(default_factory=GoalTolerance)
    safety: PathSafety = field(default_factory=PathSafety)
    label: str = ""
    obstacle_clearance: float = math.inf
    line_clearance: float = math.inf
    map_clearance: float = math.inf
    expected_time: float = math.inf
    curvature_variation: float = 0.0

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=np.float64).reshape(-1)
        count = int(self.x.size)
        if count == 0:
            raise ValueError("a path needs at least one point")
        self.y = _as_vector(self.y, count, "y")
        self.heading = _as_vector(self.heading, count, "heading")
        self.curvature = _as_vector(self.curvature, count, "curvature")
        if not all(
            np.all(np.isfinite(values))
            for values in (self.x, self.y, self.heading, self.curvature)
        ):
            raise ValueError("path geometry must be finite")

        measured_station = np.concatenate(
            ([0.0], np.cumsum(np.hypot(np.diff(self.x), np.diff(self.y))))
        )
        if self.station is None:
            self.station = measured_station
        else:
            self.station = _as_vector(self.station, count, "station")
            if not np.all(np.isfinite(self.station)):
                raise ValueError("path station must be finite")
            self.station = self.station - float(self.station[0])
        if count > 1 and np.any(np.diff(self.station) <= 1e-9):
            raise ValueError("path station must strictly increase")
        if count > 1 and np.any(np.hypot(np.diff(self.x), np.diff(self.y)) <= 1e-9):
            raise ValueError("path contains duplicate consecutive points")

        self.speed = _as_vector(self.speed, count, "speed", default=0.0)
        if not np.all(np.isfinite(self.speed)) or np.any(self.speed < 0.0):
            raise ValueError("path target speed must be finite and non-negative")
        self.direction = _as_vector(
            self.direction, count, "direction", default=1, dtype=np.int8
        )
        if np.any((self.direction != 1) & (self.direction != -1)):
            raise ValueError("path direction values must be +1 or -1")
        if np.any(self.direction != self.direction[0]):
            raise ValueError(
                "one CommonPath execution segment must have one direction; "
                "split forward/reverse motion into separate segments"
            )
        self.feedforward_scale = _as_vector(
            self.feedforward_scale,
            count,
            "feedforward_scale",
            default=1.0,
        )
        if (
            not np.all(np.isfinite(self.feedforward_scale))
            or np.any(self.feedforward_scale <= 0.0)
        ):
            raise ValueError("feedforward scale must be finite and positive")
        self.line_overlap_allowance = _as_vector(
            self.line_overlap_allowance,
            count,
            "line_overlap_allowance",
            default=0.0,
        )
        if (
            not np.all(np.isfinite(self.line_overlap_allowance))
            or np.any(self.line_overlap_allowance < 0.0)
        ):
            raise ValueError(
                "line overlap allowance must be finite and non-negative"
            )
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("path frame_id must be a non-empty string")
        if not isinstance(self.goal_tolerance, GoalTolerance):
            raise ValueError("goal_tolerance must be GoalTolerance")
        if not isinstance(self.safety, PathSafety):
            raise ValueError("safety must be PathSafety")

    @property
    def length(self):
        return float(self.station[-1]) if self.station.size else 0.0

    @property
    def size(self):
        return int(self.x.size)

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(zip(self.x, self.y))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(zip(self.x[index], self.y[index]))
        return float(self.x[index]), float(self.y[index])


@dataclass(frozen=True)
class SpeedProfile:
    cruise_velocity: float
    minimum_velocity: float
    entry_velocity: float
    exit_velocity: float
    maximum_angular_velocity: float
    maximum_lateral_acceleration: float
    linear_acceleration: float
    linear_deceleration: float
    angular_acceleration: float = math.inf

    def __post_init__(self):
        finite_positive = (
            self.cruise_velocity,
            self.minimum_velocity,
            self.entry_velocity,
            self.exit_velocity,
            self.maximum_angular_velocity,
            self.maximum_lateral_acceleration,
            self.linear_acceleration,
            self.linear_deceleration,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in finite_positive):
            raise ValueError("speed-profile limits must be finite and positive")
        if self.minimum_velocity > self.cruise_velocity:
            raise ValueError("minimum velocity exceeds cruise velocity")
        if self.entry_velocity < self.minimum_velocity:
            raise ValueError("entry velocity is below the nominal minimum")
        if self.exit_velocity < self.minimum_velocity:
            raise ValueError("exit velocity is below the nominal minimum")
        if self.entry_velocity > self.cruise_velocity:
            raise ValueError("entry velocity exceeds cruise velocity")
        if self.exit_velocity > self.cruise_velocity:
            raise ValueError("exit velocity exceeds cruise velocity")
        if not (
            math.isinf(self.angular_acceleration)
            or math.isfinite(self.angular_acceleration)
            and self.angular_acceleration > 0.0
        ):
            raise ValueError("angular acceleration must be positive")


def build_speed_profile(station, curvature, profile):
    """Build a nominal profile, then apply every physical upper bound.

    ``minimum_velocity`` is the lower bound of the *nominal* moving profile,
    including its explicit entry and exit targets.  It is deliberately not a
    hard actuator command: curvature, yaw-rate/yaw-acceleration, and the
    distance needed to brake for any of those limits may reduce the executable
    speed below that nominal floor.
    """
    station = np.asarray(station, dtype=np.float64).reshape(-1)
    curvature = np.asarray(curvature, dtype=np.float64).reshape(-1)
    if station.shape != curvature.shape:
        raise ValueError("station and curvature sizes differ")
    count = int(station.size)
    if count == 0:
        return np.empty(0, dtype=np.float64)
    absolute_curvature = np.maximum(np.abs(curvature), 1e-9)
    physical_limit = np.minimum(
        profile.maximum_angular_velocity / absolute_curvature,
        np.sqrt(profile.maximum_lateral_acceleration / absolute_curvature),
    )
    if count == 1:
        return np.asarray(
            [
                min(
                    profile.cruise_velocity,
                    profile.entry_velocity,
                    profile.exit_velocity,
                    float(physical_limit[0]),
                )
            ],
            dtype=np.float64,
        )
    segment = np.diff(station)
    if np.any(segment <= 0.0):
        raise ValueError("station must strictly increase")
    # ``SpeedProfile`` has already established that the cruise/entry/exit
    # nominal targets are at least the configured floor. ``physical_limit`` is
    # authoritative after that: a sharp bend is never driven faster merely to
    # maintain the nominal floor.
    nominal_speed = np.full(count, profile.cruise_velocity, dtype=np.float64)
    nominal_speed[0] = profile.entry_velocity
    nominal_speed[-1] = profile.exit_velocity
    speed = np.minimum(nominal_speed, physical_limit)

    def apply_longitudinal_limits():
        for index in range(count - 2, -1, -1):
            reachable = math.sqrt(
                max(
                    0.0,
                    speed[index + 1] ** 2
                    + 2.0 * profile.linear_deceleration * segment[index],
                )
            )
            speed[index] = min(speed[index], reachable)
        for index in range(count - 1):
            reachable = math.sqrt(
                max(
                    0.0,
                    speed[index] ** 2
                    + 2.0 * profile.linear_acceleration * segment[index],
                )
            )
            speed[index + 1] = min(speed[index + 1], reachable)

    apply_longitudinal_limits()
    if math.isfinite(profile.angular_acceleration):
        for _ in range(100):
            duration = 2.0 * segment / np.maximum(
                speed[:-1] + speed[1:], 1e-9
            )
            omega = speed * curvature
            measured = np.abs(np.diff(omega)) / np.maximum(duration, 1e-9)
            offending = np.flatnonzero(
                measured > profile.angular_acceleration * (1.0 + 1e-9)
            )
            if offending.size == 0:
                break
            for index in offending:
                scale = math.sqrt(
                    profile.angular_acceleration
                    / max(float(measured[index]), 1e-12)
                )
                scale *= 1.0 - 1e-6
                speed[index] *= scale
                speed[index + 1] *= scale
            apply_longitudinal_limits()
        else:
            raise ValueError("speed profile cannot satisfy angular acceleration")
    return speed


def geometry_from_xy(points, final_heading=None):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("path points must be an Nx2 array with N >= 2")
    segment = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    if np.any(segment <= 1e-9):
        raise ValueError("path contains duplicate consecutive points")
    station = np.concatenate(([0.0], np.cumsum(segment)))
    edge_heading = np.unwrap(np.arctan2(np.diff(points[:, 1]), np.diff(points[:, 0])))
    heading = np.empty(points.shape[0], dtype=np.float64)
    heading[0] = edge_heading[0]
    heading[-1] = edge_heading[-1]
    if heading.size > 2:
        heading[1:-1] = np.arctan2(
            np.sin(edge_heading[:-1]) + np.sin(edge_heading[1:]),
            np.cos(edge_heading[:-1]) + np.cos(edge_heading[1:]),
        )
    if final_heading is not None:
        heading[-1] = float(final_heading)
    unwrapped = np.unwrap(heading)
    curvature = np.gradient(unwrapped, station, edge_order=1)
    heading = np.asarray([normalize_angle(value) for value in heading])
    return heading, curvature, station


def geometry_from_poses(poses):
    """Return body heading, curvature, and station for explicit SE(2) poses.

    Unlike :func:`geometry_from_xy`, this keeps the supplied body heading.
    That distinction is required for reverse paths, whose body heading is
    opposite their direction of travel.
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or poses.shape[0] < 2:
        raise ValueError("path poses must be an Nx3 array with N >= 2")
    if not np.all(np.isfinite(poses)):
        raise ValueError("path poses must be finite")
    segment = np.hypot(np.diff(poses[:, 0]), np.diff(poses[:, 1]))
    if np.any(segment <= 1e-9):
        raise ValueError("path contains duplicate consecutive points")
    station = np.concatenate(([0.0], np.cumsum(segment)))
    unwrapped_heading = np.unwrap(poses[:, 2])
    curvature = np.gradient(unwrapped_heading, station, edge_order=1)
    heading = np.asarray(
        [normalize_angle(value) for value in poses[:, 2]], dtype=np.float64
    )
    return heading, curvature, station


def _line_egress_profile(
    station, initial_line_overlap_allowance, line_egress_distance
):
    initial_line_overlap_allowance = float(initial_line_overlap_allowance)
    line_egress_distance = float(line_egress_distance)
    if (
        not math.isfinite(initial_line_overlap_allowance)
        or initial_line_overlap_allowance < 0.0
        or not math.isfinite(line_egress_distance)
        or line_egress_distance < 0.0
    ):
        raise ValueError("line-egress limits must be finite and non-negative")
    if initial_line_overlap_allowance > 0.0 and line_egress_distance <= 0.0:
        raise ValueError(
            "positive initial line overlap allowance needs an egress distance"
        )
    if initial_line_overlap_allowance <= 0.0:
        return np.zeros(len(station), dtype=np.float64)
    return initial_line_overlap_allowance * np.maximum(
        0.0, 1.0 - np.asarray(station) / line_egress_distance
    )


def _finalize_path_metadata(path):
    moving_speed = np.maximum(path.speed, 1e-9)
    if path.size > 1:
        segment_time = 2.0 * np.diff(path.station) / np.maximum(
            moving_speed[:-1] + moving_speed[1:], 1e-9
        )
        path.expected_time = float(np.sum(segment_time))
        path.curvature_variation = float(
            np.sum(np.abs(np.diff(path.curvature)))
        )
    return path


def path_from_xy(
    points,
    frame_id,
    speed_profile=None,
    target_speed=None,
    direction=1,
    feedforward_scale=1.0,
    initial_line_overlap_allowance=0.0,
    line_egress_distance=0.0,
    final_heading=None,
    goal_tolerance=None,
    safety=None,
    label="",
):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("path points must be an Nx2 array with N >= 2")
    count = int(points.shape[0])
    directions = _as_vector(
        direction, count, "direction", default=1, dtype=np.int8
    )
    if np.any((directions != 1) & (directions != -1)):
        raise ValueError("path direction values must be +1 or -1")
    motion_final_heading = final_heading
    if final_heading is not None and directions[-1] < 0:
        motion_final_heading = normalize_angle(float(final_heading) + math.pi)
    heading, curvature, station = geometry_from_xy(points, motion_final_heading)
    heading = np.asarray(
        [
            normalize_angle(value + (math.pi if directions[index] < 0 else 0.0))
            for index, value in enumerate(heading)
        ],
        dtype=np.float64,
    )
    if final_heading is not None:
        heading[-1] = normalize_angle(float(final_heading))
    if speed_profile is not None:
        speed = build_speed_profile(station, curvature, speed_profile)
    else:
        speed = target_speed
    line_overlap_allowance = _line_egress_profile(
        station, initial_line_overlap_allowance, line_egress_distance
    )
    path = CommonPath(
        x=points[:, 0],
        y=points[:, 1],
        heading=heading,
        curvature=curvature,
        station=station,
        speed=speed,
        direction=directions,
        feedforward_scale=feedforward_scale,
        line_overlap_allowance=line_overlap_allowance,
        frame_id=frame_id,
        goal_tolerance=goal_tolerance or GoalTolerance(),
        safety=safety or PathSafety(),
        label=label,
    )
    return _finalize_path_metadata(path)


def path_from_poses(
    poses,
    frame_id,
    speed_profile=None,
    target_speed=None,
    direction=1,
    feedforward_scale=1.0,
    initial_line_overlap_allowance=0.0,
    line_egress_distance=0.0,
    goal_tolerance=None,
    safety=None,
    label="",
):
    """Create the common path format from explicit body poses.

    This is the shared constructor for missions whose surveyed route includes
    body orientation (notably forward/reverse parking segments). Geometry,
    station, curvature, speed limits, and safety metadata are produced once in
    this library instead of being reimplemented in a mission controller.
    """
    poses = np.asarray(poses, dtype=np.float64)
    heading, curvature, station = geometry_from_poses(poses)
    count = int(poses.shape[0])
    directions = _as_vector(
        direction, count, "direction", default=1, dtype=np.int8
    )
    if np.any((directions != 1) & (directions != -1)):
        raise ValueError("path direction values must be +1 or -1")
    feedforward = _as_vector(
        feedforward_scale,
        count,
        "feedforward_scale",
        default=1.0,
    )
    if speed_profile is not None:
        speed = build_speed_profile(
            station, curvature * feedforward, speed_profile
        )
    else:
        speed = target_speed
    path = CommonPath(
        x=poses[:, 0],
        y=poses[:, 1],
        heading=heading,
        curvature=curvature,
        station=station,
        speed=speed,
        direction=directions,
        feedforward_scale=feedforward,
        line_overlap_allowance=_line_egress_profile(
            station,
            initial_line_overlap_allowance,
            line_egress_distance,
        ),
        frame_id=frame_id,
        goal_tolerance=goal_tolerance or GoalTolerance(),
        safety=safety or PathSafety(),
        label=label,
    )
    return _finalize_path_metadata(path)


@dataclass(frozen=True)
class RigidTransform2D:
    target_from_source_x: float
    target_from_source_y: float
    target_from_source_yaw: float
    source_frame: str = "map"
    target_frame: str = "odom"

    @classmethod
    def from_pose_pair(
        cls, source_pose, target_pose, source_frame="map", target_frame="odom"
    ):
        source = Pose2D.from_value(source_pose)
        target = Pose2D.from_value(target_pose)
        yaw = normalize_angle(target.yaw - source.yaw)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return cls(
            target.x - (cosine * source.x - sine * source.y),
            target.y - (sine * source.x + cosine * source.y),
            yaw,
            source_frame,
            target_frame,
        )

    def apply_point(self, point):
        x, y = (float(value) for value in point)
        cosine = math.cos(self.target_from_source_yaw)
        sine = math.sin(self.target_from_source_yaw)
        return (
            self.target_from_source_x + cosine * x - sine * y,
            self.target_from_source_y + sine * x + cosine * y,
        )

    def apply_pose(self, pose):
        pose = Pose2D.from_value(pose)
        x, y = self.apply_point((pose.x, pose.y))
        return Pose2D(x, y, normalize_angle(self.target_from_source_yaw + pose.yaw))

    def inverse(self):
        yaw = -self.target_from_source_yaw
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        x = -(cosine * self.target_from_source_x - sine * self.target_from_source_y)
        y = -(sine * self.target_from_source_x + cosine * self.target_from_source_y)
        return RigidTransform2D(
            x,
            y,
            yaw,
            self.target_frame,
            self.source_frame,
        )

    def apply_path(self, path):
        cosine = math.cos(self.target_from_source_yaw)
        sine = math.sin(self.target_from_source_yaw)
        safety = PathSafety(
            line_boundaries=tuple(
                boundary.transformed(self)
                if hasattr(boundary, "transformed")
                else boundary
                for boundary in path.safety.line_boundaries
            ),
            map_boundaries=tuple(
                boundary.transformed(self)
                if hasattr(boundary, "transformed")
                else boundary
                for boundary in path.safety.map_boundaries
            ),
            fixed_obstacles=np.asarray(
                [self.apply_point(point) for point in path.safety.fixed_obstacles],
                dtype=np.float64,
            ).reshape((-1, 2)),
            margins=path.safety.margins,
        )
        return CommonPath(
            x=self.target_from_source_x + cosine * path.x - sine * path.y,
            y=self.target_from_source_y + sine * path.x + cosine * path.y,
            heading=np.asarray(
                [
                    normalize_angle(self.target_from_source_yaw + value)
                    for value in path.heading
                ],
                dtype=np.float64,
            ),
            curvature=path.curvature.copy(),
            station=path.station.copy(),
            speed=path.speed.copy(),
            direction=path.direction.copy(),
            feedforward_scale=path.feedforward_scale.copy(),
            line_overlap_allowance=path.line_overlap_allowance.copy(),
            frame_id=self.target_frame,
            goal_tolerance=path.goal_tolerance,
            safety=safety,
            label=path.label,
            obstacle_clearance=path.obstacle_clearance,
            line_clearance=path.line_clearance,
            map_clearance=path.map_clearance,
            expected_time=path.expected_time,
            curvature_variation=path.curvature_variation,
        )


def freeze_path_in_odom(
    path,
    source_pose,
    odom_pose,
    odom_aligned=False,
    odom_frame="odom",
):
    """Transform a selected route once and never consult localization again."""
    if odom_aligned:
        transform = RigidTransform2D(0.0, 0.0, 0.0, path.frame_id, odom_frame)
    else:
        transform = RigidTransform2D.from_pose_pair(
            source_pose, odom_pose, path.frame_id, odom_frame
        )
    return transform.apply_path(path), transform.inverse()


@dataclass(frozen=True)
class Projection:
    segment_index: int
    path_index: int
    station: float
    x: float
    y: float
    heading: float
    curvature: float
    direction: int
    distance: float
    cross_track_error: float


def _segment_window(path, previous_index, search_back, search_ahead_distance):
    if path.size <= 1:
        return 0, 0
    if previous_index is None:
        return 0, path.size - 1
    previous_index = max(0, min(int(previous_index), path.size - 1))
    first = max(0, previous_index - max(0, int(search_back)))
    maximum = float(path.station[previous_index]) + max(
        0.01, float(search_ahead_distance)
    )
    last_point = int(np.searchsorted(path.station, maximum, side="right"))
    last_point = min(path.size, max(first + 2, last_point))
    return first, last_point - 1


def _projection_from_segment(path, x, y, segment, fraction):
    """Build every projection field from one segment/station decision."""
    segment = max(0, min(int(segment), path.size - 2))
    fraction = clamp(float(fraction), 0.0, 1.0)
    # An exact interior vertex belongs to the following segment so subsequent
    # projection searches and the reported nearest-point index advance
    # together.
    if fraction >= 1.0 - 1e-12 and segment + 1 < path.size - 1:
        segment += 1
        fraction = 0.0
    start_x = float(path.x[segment])
    start_y = float(path.y[segment])
    end_x = float(path.x[segment + 1])
    end_y = float(path.y[segment + 1])
    projected_x = start_x + fraction * (end_x - start_x)
    projected_y = start_y + fraction * (end_y - start_y)
    station = float(
        path.station[segment]
        + fraction * (path.station[segment + 1] - path.station[segment])
    )
    heading_delta = normalize_angle(
        float(path.heading[segment + 1]) - float(path.heading[segment])
    )
    heading = normalize_angle(
        float(path.heading[segment]) + fraction * heading_delta
    )
    curvature = float(
        (1.0 - fraction) * path.curvature[segment]
        + fraction * path.curvature[segment + 1]
    )
    direction = int(path.direction[segment])
    motion_heading = heading + (math.pi if direction < 0 else 0.0)
    error_x = float(x) - projected_x
    error_y = float(y) - projected_y
    path_index = segment + (1 if fraction >= 0.5 else 0)
    return Projection(
        segment,
        path_index,
        station,
        projected_x,
        projected_y,
        heading,
        curvature,
        direction,
        math.hypot(error_x, error_y),
        -math.sin(motion_heading) * error_x
        + math.cos(motion_heading) * error_y,
    )


def _projection_at_station(path, x, y, station):
    station = clamp(float(station), 0.0, path.length)
    if station >= path.length - 1e-12:
        return _projection_from_segment(path, x, y, path.size - 2, 1.0)
    segment = max(
        0,
        min(
            path.size - 2,
            int(np.searchsorted(path.station, station, side="right")) - 1,
        ),
    )
    span = float(path.station[segment + 1] - path.station[segment])
    fraction = (station - float(path.station[segment])) / span
    return _projection_from_segment(path, x, y, segment, fraction)


def _minimum_station_for_index(path, path_index):
    """Lowest station whose nearest-point index is ``path_index``."""
    path_index = max(0, min(int(path_index), path.size - 1))
    if path_index == 0:
        return 0.0
    return 0.5 * float(
        path.station[path_index - 1] + path.station[path_index]
    )


def project_to_path(
    path,
    x,
    y,
    previous_index=None,
    search_back=3,
    search_ahead_distance=0.30,
    minimum_station=None,
):
    """Project a pose onto path segments, including between stored points."""
    x = float(x)
    y = float(y)
    if path.size == 1:
        heading = float(path.heading[0])
        motion_heading = heading + (math.pi if path.direction[0] < 0 else 0.0)
        dx = x - float(path.x[0])
        dy = y - float(path.y[0])
        return Projection(
            0,
            0,
            0.0,
            float(path.x[0]),
            float(path.y[0]),
            heading,
            float(path.curvature[0]),
            int(path.direction[0]),
            math.hypot(dx, dy),
            -math.sin(motion_heading) * dx + math.cos(motion_heading) * dy,
        )
    first, last = _segment_window(
        path, previous_index, search_back, search_ahead_distance
    )
    ax = path.x[first:last]
    ay = path.y[first:last]
    bx = path.x[first + 1 : last + 1]
    by = path.y[first + 1 : last + 1]
    vx = bx - ax
    vy = by - ay
    length_squared = vx * vx + vy * vy
    ratio = np.clip(((x - ax) * vx + (y - ay) * vy) / length_squared, 0.0, 1.0)
    px = ax + ratio * vx
    py = ay + ratio * vy
    distance_squared = (px - x) ** 2 + (py - y) ** 2
    local = int(np.argmin(distance_squared))
    segment = first + local
    fraction = float(ratio[local])
    projection = _projection_from_segment(path, x, y, segment, fraction)
    if previous_index is not None:
        previous_index = max(0, min(int(previous_index), path.size - 1))
        station_floor = _minimum_station_for_index(path, previous_index)
        if minimum_station is not None:
            minimum_station = float(minimum_station)
            if not math.isfinite(minimum_station):
                raise ValueError("minimum projection station must be finite")
            station_floor = max(
                station_floor,
                clamp(minimum_station, 0.0, path.length),
            )
        if (
            projection.station < station_floor - 1e-12
            or projection.path_index < previous_index
        ):
            projection = _projection_at_station(path, x, y, station_floor)
    elif minimum_station is not None:
        raise ValueError("minimum_station requires previous_index")
    return projection


@dataclass(frozen=True)
class PathSample:
    index: int
    station: float
    x: float
    y: float
    heading: float
    curvature: float
    speed: float
    direction: int
    feedforward_scale: float
    line_overlap_allowance: float


def sample_path(path, station):
    station = clamp(float(station), 0.0, path.length)
    right = min(
        path.size - 1,
        int(np.searchsorted(path.station, station, side="right")),
    )
    if right == 0:
        return PathSample(
            0,
            station,
            float(path.x[0]),
            float(path.y[0]),
            float(path.heading[0]),
            float(path.curvature[0]),
            float(path.speed[0]),
            int(path.direction[0]),
            float(path.feedforward_scale[0]),
            float(path.line_overlap_allowance[0]),
        )
    left = right - 1
    span = float(path.station[right] - path.station[left])
    fraction = clamp((station - float(path.station[left])) / span, 0.0, 1.0)
    heading_delta = normalize_angle(float(path.heading[right]) - float(path.heading[left]))
    return PathSample(
        right,
        station,
        float((1.0 - fraction) * path.x[left] + fraction * path.x[right]),
        float((1.0 - fraction) * path.y[left] + fraction * path.y[right]),
        normalize_angle(float(path.heading[left]) + fraction * heading_delta),
        float(
            (1.0 - fraction) * path.curvature[left]
            + fraction * path.curvature[right]
        ),
        float(min(path.speed[left], path.speed[right])),
        int(path.direction[left]),
        float(
            (1.0 - fraction) * path.feedforward_scale[left]
            + fraction * path.feedforward_scale[right]
        ),
        float(
            (1.0 - fraction) * path.line_overlap_allowance[left]
            + fraction * path.line_overlap_allowance[right]
        ),
    )


@dataclass(frozen=True)
class TrackingConfig:
    lookahead_distance: float
    maximum_linear_velocity: float
    maximum_angular_velocity: float
    maximum_lateral_acceleration: float
    linear_acceleration: float
    linear_deceleration: float
    angular_acceleration: float
    heading_gain: float
    curvature_feedforward_weight: float = 0.25
    lateral_feedback_gain: float = 1.0
    search_back: int = 3
    search_ahead_distance: float = 0.30
    # A zero time keeps the historical fixed-distance target exactly.  When
    # enabled, the steering target follows ``base + |velocity| * time`` within
    # the explicit distance bounds.  ``lookahead_time`` therefore has seconds
    # as its unit and behaves identically for forward and reverse motion.
    lookahead_time: float = 0.0
    minimum_lookahead_distance: float = 0.0
    maximum_lookahead_distance: float = math.inf
    # ``None`` preserves the historical coupling: target speed is the minimum
    # profile value between the projection and steering target.  A configured
    # non-negative distance gives speed selection its own independent horizon;
    # zero samples the already acceleration-shaped profile at current progress.
    speed_preview_distance: float = None

    def __post_init__(self):
        positive = (
            self.lookahead_distance,
            self.maximum_linear_velocity,
            self.maximum_angular_velocity,
            self.maximum_lateral_acceleration,
            self.linear_acceleration,
            self.linear_deceleration,
            self.angular_acceleration,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("tracking limits must be finite and positive")
        if not all(
            math.isfinite(value)
            for value in (
                self.heading_gain,
                self.curvature_feedforward_weight,
                self.lateral_feedback_gain,
                self.search_ahead_distance,
            )
        ):
            raise ValueError("tracking gains must be finite")
        if not (
            math.isfinite(self.lookahead_time)
            and self.lookahead_time >= 0.0
        ):
            raise ValueError("lookahead time must be finite and non-negative")
        if not (
            math.isfinite(self.minimum_lookahead_distance)
            and self.minimum_lookahead_distance >= 0.0
        ):
            raise ValueError(
                "minimum lookahead distance must be finite and non-negative"
            )
        if not (
            not math.isnan(self.maximum_lookahead_distance)
            and self.maximum_lookahead_distance > 0.0
        ):
            raise ValueError("maximum lookahead distance must be positive")
        if (
            self.minimum_lookahead_distance
            > self.maximum_lookahead_distance
        ):
            raise ValueError("lookahead distance bounds are reversed")
        if self.speed_preview_distance is not None and not (
            math.isfinite(self.speed_preview_distance)
            and self.speed_preview_distance >= 0.0
        ):
            raise ValueError(
                "speed preview distance must be finite and non-negative"
            )


@dataclass(frozen=True)
class TrackingResult:
    path_index: int
    target_index: int
    position_error: float
    heading_error: float
    cross_track_error: float
    target_speed: float
    direction: int
    feedforward_curvature: float
    feedback_curvature: float
    curvature_command: float
    feedforward_angular_velocity: float
    heading_feedback: float
    lateral_feedback: float
    angular_velocity: float
    station: float
    progress: float
    steering_lookahead_distance: float = 0.0
    speed_preview_distance: float = 0.0


def _minimum_speed_between(path, start_station, end_station):
    """Return the conservative profile value over one continuous interval."""

    start_station = clamp(float(start_station), 0.0, path.length)
    end_station = clamp(float(end_station), start_station, path.length)
    start = sample_path(path, start_station)
    end = sample_path(path, end_station)
    first_point = int(np.searchsorted(path.station, start_station, side="left"))
    past_last_point = int(
        np.searchsorted(path.station, end_station, side="right")
    )
    candidates = [start.speed, end.speed]
    if past_last_point > first_point:
        candidates.append(
            float(np.min(path.speed[first_point:past_last_point]))
        )
    return min(candidates)


def calculate_tracking(
    path,
    pose,
    previous_index,
    config,
    previous_station=None,
    linear_velocity=0.0,
    minimum_target_station=None,
):
    pose = Pose2D.from_value(pose)
    linear_velocity = float(linear_velocity)
    if not math.isfinite(linear_velocity):
        raise ValueError("tracking linear velocity must be finite")
    if minimum_target_station is not None:
        minimum_target_station = float(minimum_target_station)
        if not (
            math.isfinite(minimum_target_station)
            and 0.0 <= minimum_target_station <= path.length
        ):
            raise ValueError(
                "minimum target station must lie on the active path"
            )
    projection = project_to_path(
        path,
        pose.x,
        pose.y,
        previous_index,
        config.search_back,
        config.search_ahead_distance,
        minimum_station=previous_station,
    )
    direction = projection.direction
    steering_lookahead = clamp(
        config.lookahead_distance
        + config.lookahead_time * abs(linear_velocity),
        config.minimum_lookahead_distance,
        config.maximum_lookahead_distance,
    )
    requested_target_station = projection.station + steering_lookahead
    if minimum_target_station is not None:
        # A camera path can mark the first station supported by observed lane
        # geometry.  This floor keeps the steering target off a blind extension.
        requested_target_station = max(
            requested_target_station, minimum_target_station
        )
    target_station = min(
        path.length,
        requested_target_station,
    )
    target = sample_path(path, target_station)
    motion_yaw = normalize_angle(pose.yaw + (math.pi if direction < 0 else 0.0))
    desired_motion_heading = normalize_angle(
        target.heading + (math.pi if direction < 0 else 0.0)
    )
    dx = target.x - pose.x
    dy = target.y - pose.y
    local_x = math.cos(motion_yaw) * dx + math.sin(motion_yaw) * dy
    local_y = -math.sin(motion_yaw) * dx + math.cos(motion_yaw) * dy
    distance_squared = max(0.0025, local_x * local_x + local_y * local_y)
    pure_pursuit = 2.0 * local_y / distance_squared
    weight = clamp(config.curvature_feedforward_weight, 0.0, 1.0)
    lateral_gain = max(0.0, config.lateral_feedback_gain)
    feedback_curvature = lateral_gain * pure_pursuit
    scaled_feedforward_curvature = target.feedforward_scale * target.curvature
    curvature_command = (
        weight * scaled_feedforward_curvature
        + (1.0 - weight) * feedback_curvature
    )
    if config.speed_preview_distance is None:
        # Keep the exact historical index window when no independent preview
        # was requested, so every fixed mission retains its validated behavior.
        first = min(projection.path_index, target.index)
        last = max(projection.path_index, target.index)
        profile_speed = float(np.min(path.speed[first : last + 1]))
        speed_preview_station = target.station
    else:
        speed_preview_station = min(
            path.length,
            projection.station + config.speed_preview_distance,
        )
        profile_speed = _minimum_speed_between(
            path, projection.station, speed_preview_station
        )
    speed = min(config.maximum_linear_velocity, profile_speed)
    if abs(curvature_command) > 1e-9:
        speed = min(
            speed,
            config.maximum_angular_velocity / abs(curvature_command),
            math.sqrt(config.maximum_lateral_acceleration / abs(curvature_command)),
        )
    heading_error = normalize_angle(desired_motion_heading - motion_yaw)
    feedforward_angular = speed * weight * scaled_feedforward_curvature
    lateral_feedback = speed * (1.0 - weight) * feedback_curvature
    heading_feedback = config.heading_gain * heading_error
    angular = clamp(
        feedforward_angular + lateral_feedback + heading_feedback,
        -config.maximum_angular_velocity,
        config.maximum_angular_velocity,
    )
    return TrackingResult(
        projection.path_index,
        target.index,
        projection.distance,
        heading_error,
        projection.cross_track_error,
        speed,
        direction,
        target.curvature,
        feedback_curvature,
        curvature_command,
        feedforward_angular,
        heading_feedback,
        lateral_feedback,
        angular,
        projection.station,
        1.0 if path.length <= 0.0 else projection.station / path.length,
        max(0.0, target.station - projection.station),
        max(0.0, speed_preview_station - projection.station),
    )


@dataclass(frozen=True)
class VelocityCommand:
    linear_velocity: float
    angular_velocity: float


def in_place_rotation_command(
    current_yaw,
    target_yaw,
    last_angular_velocity,
    elapsed,
    heading_gain,
    maximum_angular_velocity,
    minimum_angular_velocity,
    angular_acceleration,
    heading_tolerance,
):
    """Return a zero-linear command for shortest-direction in-place rotation.

    Outside the heading tolerance, the proportional target is raised to the
    configured nominal minimum, then capped by both maximum yaw rate and the
    rate from which the robot can brake over the remaining heading error.  The
    braking cap includes half of the current control interval, matching the
    distance covered while a slew-limited rate ramps between two samples.  It
    therefore starts braking before a discrete control tick can step past the
    continuous-time stopping point.  The physical braking cap remains
    authoritative near the target and may reduce the command below the
    nominal minimum.  Every angular command change, including the final stop
    inside the heading tolerance, obeys the same acceleration bound.
    """
    values = (
        current_yaw,
        target_yaw,
        last_angular_velocity,
        elapsed,
        heading_gain,
        maximum_angular_velocity,
        minimum_angular_velocity,
        angular_acceleration,
        heading_tolerance,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("in-place rotation inputs must be finite")

    elapsed = float(elapsed)
    heading_gain = float(heading_gain)
    maximum_angular_velocity = float(maximum_angular_velocity)
    minimum_angular_velocity = float(minimum_angular_velocity)
    angular_acceleration = float(angular_acceleration)
    heading_tolerance = float(heading_tolerance)
    if elapsed < 0.0:
        raise ValueError("rotation elapsed time must be non-negative")
    if heading_gain < 0.0:
        raise ValueError("rotation heading gain must be non-negative")
    if maximum_angular_velocity <= 0.0:
        raise ValueError("maximum angular velocity must be positive")
    if (
        minimum_angular_velocity < 0.0
        or minimum_angular_velocity > maximum_angular_velocity
    ):
        raise ValueError("minimum angular velocity is outside the speed limits")
    if angular_acceleration <= 0.0:
        raise ValueError("angular acceleration must be positive")
    if heading_tolerance < 0.0:
        raise ValueError("heading tolerance must be non-negative")

    previous_angular = clamp(
        float(last_angular_velocity),
        -maximum_angular_velocity,
        maximum_angular_velocity,
    )
    maximum_change = angular_acceleration * elapsed
    heading_error = normalize_angle(float(target_yaw) - float(current_yaw))
    error_magnitude = abs(heading_error)
    if error_magnitude <= heading_tolerance:
        desired_angular = 0.0
    else:
        proportional_rate = heading_gain * error_magnitude
        half_interval_rate = 0.5 * maximum_change
        braking_rate = max(
            0.0,
            math.sqrt(
                half_interval_rate * half_interval_rate
                + 2.0 * angular_acceleration * error_magnitude
            )
            - half_interval_rate,
        )
        desired_magnitude = min(
            maximum_angular_velocity,
            braking_rate,
            max(minimum_angular_velocity, proportional_rate),
        )
        desired_angular = math.copysign(desired_magnitude, heading_error)
    angular = clamp(
        desired_angular,
        previous_angular - maximum_change,
        previous_angular + maximum_change,
    )
    return VelocityCommand(0.0, angular)


def limit_velocity_command(
    target_linear_velocity,
    reference_speed,
    target_angular_velocity,
    last_linear_velocity,
    last_angular_velocity,
    elapsed,
    config,
):
    """Limit signed motion without bypassing linear or angular slew bounds.

    When a new path asks for steering that is not yet reachable, angular rate
    ramps first and the vehicle temporarily understeers.  Dropping linear speed
    instantaneously to preserve target curvature would violate the configured
    deceleration limit at a controller handoff.
    """
    target_linear = clamp(
        float(target_linear_velocity),
        -config.maximum_linear_velocity,
        config.maximum_linear_velocity,
    )
    reference_speed = max(1e-9, abs(float(reference_speed)))
    curvature = float(target_angular_velocity) / reference_speed
    target_magnitude = abs(target_linear)
    if abs(curvature) > 1e-9:
        target_magnitude = min(
            target_magnitude,
            config.maximum_angular_velocity / abs(curvature),
            math.sqrt(config.maximum_lateral_acceleration / abs(curvature)),
        )
    target_linear = math.copysign(target_magnitude, target_linear) if target_magnitude else 0.0
    elapsed = max(0.0, float(elapsed))
    previous = clamp(
        float(last_linear_velocity),
        -config.maximum_linear_velocity,
        config.maximum_linear_velocity,
    )
    if previous * target_linear < 0.0:
        linear = math.copysign(
            max(0.0, abs(previous) - config.linear_deceleration * elapsed),
            previous,
        )
    else:
        acceleration = (
            config.linear_acceleration
            if abs(target_linear) >= abs(previous)
            else config.linear_deceleration
        )
        linear = clamp(
            target_linear,
            previous - acceleration * elapsed,
            previous + acceleration * elapsed,
        )
    # Curvature is parameterized by increasing station, so yaw rate depends on
    # speed magnitude for both forward and reverse sections.
    desired_angular = clamp(
        abs(linear) * curvature,
        -config.maximum_angular_velocity,
        config.maximum_angular_velocity,
    )
    if abs(linear) > 1e-9:
        desired_angular = clamp(
            desired_angular,
            -config.maximum_lateral_acceleration / abs(linear),
            config.maximum_lateral_acceleration / abs(linear),
        )
    previous_angular = clamp(
        float(last_angular_velocity),
        -config.maximum_angular_velocity,
        config.maximum_angular_velocity,
    )
    angular = clamp(
        desired_angular,
        previous_angular - config.angular_acceleration * elapsed,
        previous_angular + config.angular_acceleration * elapsed,
    )
    if abs(linear * angular) > config.maximum_lateral_acceleration + 1e-12:
        angular_cap = config.maximum_lateral_acceleration / max(abs(linear), 1e-9)
        feasible_angular = math.copysign(angular_cap, angular)
        if (
            abs(feasible_angular - previous_angular)
            <= config.angular_acceleration * elapsed + 1e-12
        ):
            angular = feasible_angular
        else:
            linear_cap = config.maximum_lateral_acceleration / max(
                abs(angular), 1e-9
            )
            feasible_linear = math.copysign(linear_cap, linear)
            change_limit = (
                config.linear_deceleration
                if previous * feasible_linear < 0.0
                or abs(feasible_linear) < abs(previous)
                else config.linear_acceleration
            ) * elapsed
            if abs(feasible_linear - previous) <= change_limit + 1e-12:
                linear = feasible_linear
            elif (
                abs(previous * previous_angular)
                <= config.maximum_lateral_acceleration + 1e-12
            ):
                # The previous bounded command is always a feasible point in
                # both slew intervals when neither one-axis projection is.
                linear = previous
                angular = previous_angular
            else:
                # A caller supplied an already-infeasible initial command.
                # Enforce the physical envelope immediately at that boundary.
                angular = math.copysign(
                    config.maximum_lateral_acceleration
                    / max(abs(previous), 1e-9),
                    previous_angular,
                )
                linear = previous
    return VelocityCommand(linear, angular)


@dataclass(frozen=True)
class GoalStatus:
    complete: bool
    position_error: float
    heading_error: float
    remaining_distance: float
    crossed_terminal: bool


def _goal_status_from_progress(path, pose, path_index, station):
    pose = Pose2D.from_value(pose)
    goal_x = float(path.x[-1])
    goal_y = float(path.y[-1])
    position_error = math.hypot(goal_x - pose.x, goal_y - pose.y)
    heading_error = normalize_angle(float(path.heading[-1]) - pose.yaw)
    if path.size > 1:
        tangent_x = float(path.x[-1] - path.x[-2])
        tangent_y = float(path.y[-1] - path.y[-2])
        crossed_half_plane = (
            (pose.x - goal_x) * tangent_x
            + (pose.y - goal_y) * tangent_y
            >= 0.0
        )
        # A terminal plane is infinite.  Membership in its forward half-plane
        # is only a real path-terminal event after projected execution progress
        # has entered the configured crossing-distance horizon.
        remaining_distance = max(0.0, path.length - float(station))
        near_terminal = (
            remaining_distance <= path.goal_tolerance.terminal_crossing + 1e-9
        )
        crossed = crossed_half_plane and near_terminal
    else:
        crossed = position_error <= path.goal_tolerance.position
        near_terminal = True
    tolerance = path.goal_tolerance
    terminal_event = position_error <= tolerance.position or (
        crossed and near_terminal and position_error <= tolerance.terminal_crossing
    )
    complete = bool(abs(heading_error) <= tolerance.heading and terminal_event)
    return GoalStatus(
        complete,
        position_error,
        heading_error,
        max(0.0, path.length - float(station)),
        bool(crossed),
    )


def goal_status(path, pose, projection=None):
    if projection is None:
        converted = Pose2D.from_value(pose)
        projection = project_to_path(path, converted.x, converted.y)
    return _goal_status_from_progress(
        path,
        pose,
        projection.path_index,
        projection.station,
    )


class StraightCorridorBoundary:
    """Two parallel boundaries in a rigid local corridor frame."""

    def __init__(
        self,
        right,
        left,
        origin=(0.0, 0.0),
        heading=0.0,
        minimum_progress=-math.inf,
        maximum_progress=math.inf,
    ):
        self.right = float(right)
        self.left = float(left)
        self.origin = (float(origin[0]), float(origin[1]))
        self.heading = float(heading)
        self.minimum_progress = float(minimum_progress)
        self.maximum_progress = float(maximum_progress)
        if not self.right < self.left:
            raise ValueError("corridor right boundary must be below left boundary")

    def clearance(self, pose, footprint):
        points = footprint_points(pose, footprint, perimeter_only=True)
        dx = points[:, 0] - self.origin[0]
        dy = points[:, 1] - self.origin[1]
        cosine = math.cos(self.heading)
        sine = math.sin(self.heading)
        progress = cosine * dx + sine * dy
        lateral = -sine * dx + cosine * dy
        if (
            float(np.max(progress)) < self.minimum_progress
            or float(np.min(progress)) > self.maximum_progress
        ):
            return math.inf
        return min(
            float(np.min(lateral)) - self.right,
            self.left - float(np.max(lateral)),
        )

    def transformed(self, transform):
        origin = transform.apply_point(self.origin)
        return StraightCorridorBoundary(
            self.right,
            self.left,
            origin,
            normalize_angle(self.heading + transform.target_from_source_yaw),
            self.minimum_progress,
            self.maximum_progress,
        )


class AxisAlignedBoundsBoundary:
    def __init__(self, minimum_x, maximum_x, minimum_y, maximum_y):
        self.minimum_x = float(minimum_x)
        self.maximum_x = float(maximum_x)
        self.minimum_y = float(minimum_y)
        self.maximum_y = float(maximum_y)
        if self.minimum_x >= self.maximum_x or self.minimum_y >= self.maximum_y:
            raise ValueError("axis-aligned safety bounds are reversed")

    def clearance(self, pose, footprint):
        points = footprint_points(pose, footprint, perimeter_only=True)
        return min(
            float(np.min(points[:, 0])) - self.minimum_x,
            self.maximum_x - float(np.max(points[:, 0])),
            float(np.min(points[:, 1])) - self.minimum_y,
            self.maximum_y - float(np.max(points[:, 1])),
        )

    def transformed(self, transform):
        return ConvexPolygonBoundary(
            [
                transform.apply_point((self.minimum_x, self.minimum_y)),
                transform.apply_point((self.maximum_x, self.minimum_y)),
                transform.apply_point((self.maximum_x, self.maximum_y)),
                transform.apply_point((self.minimum_x, self.maximum_y)),
            ]
        )


class ConvexPolygonBoundary:
    """Allowed interior of a measured convex map boundary."""

    def __init__(self, vertices):
        vertices = np.asarray(vertices, dtype=np.float64)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 2
            or vertices.shape[0] < 3
            or not np.all(np.isfinite(vertices))
        ):
            raise ValueError("convex boundary needs at least three finite vertices")
        edges = np.roll(vertices, -1, axis=0) - vertices
        lengths = np.hypot(edges[:, 0], edges[:, 1])
        if np.any(lengths <= 1e-9):
            raise ValueError("convex boundary contains duplicate vertices")
        twice_area = float(
            np.sum(
                vertices[:, 0] * np.roll(vertices[:, 1], -1)
                - vertices[:, 1] * np.roll(vertices[:, 0], -1)
            )
        )
        if abs(twice_area) <= 1e-9:
            raise ValueError("convex boundary has zero area")
        self.vertices = vertices.copy()
        self.orientation = 1.0 if twice_area > 0.0 else -1.0

    def clearance(self, pose, footprint):
        points = footprint_points(pose, footprint, perimeter_only=True)
        start = self.vertices
        edge = np.roll(start, -1, axis=0) - start
        offset = points[:, np.newaxis, :] - start[np.newaxis, :, :]
        cross = (
            edge[np.newaxis, :, 0] * offset[:, :, 1]
            - edge[np.newaxis, :, 1] * offset[:, :, 0]
        )
        signed = self.orientation * cross / np.hypot(edge[:, 0], edge[:, 1])
        return float(np.min(signed))

    def transformed(self, transform):
        return ConvexPolygonBoundary(
            [transform.apply_point(point) for point in self.vertices]
        )


class CallbackBoundary:
    """Adapter for a measured boundary field without duplicating sweep logic."""

    def __init__(self, callback):
        self.callback = callback

    def clearance(self, pose, footprint):
        return float(self.callback(pose, footprint))


class PointCloudBoundary:
    """A measured line/map boundary represented by finite surface points."""

    def __init__(self, points, point_radius=0.0):
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            points = np.empty((0, 2), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("boundary point cloud must contain [x, y] points")
        if not np.all(np.isfinite(points)):
            raise ValueError("boundary point cloud must be finite")
        self.point_radius = float(point_radius)
        if not math.isfinite(self.point_radius) or self.point_radius < 0.0:
            raise ValueError("boundary point radius must be finite and non-negative")
        self.points = points.copy()

    def clearance(self, pose, footprint):
        return (
            point_obstacle_clearance(pose, footprint, self.points)
            - self.point_radius
        )

    def transformed(self, transform):
        return PointCloudBoundary(
            np.asarray(
                [transform.apply_point(point) for point in self.points],
                dtype=np.float64,
            ).reshape((-1, 2)),
            point_radius=self.point_radius,
        )


class RasterCellBoundary:
    """Occupied axis-aligned raster cells kept exact in their source frame.

    A raster pixel is a rectangle, not a circular point obstacle.  Turning its
    centre into a point with a circumscribed radius overstates the cell near
    the middle of each edge.  This boundary instead measures the robot's
    oriented asymmetric rectangle against every relevant cell rectangle.

    ``transformed`` deliberately keeps the cells axis aligned in their
    original source frame.  At query time the robot pose is transformed back
    into that frame, so arbitrary map-to-odom rotations do not change the cell
    shape or introduce a geometric approximation.
    """

    def __init__(self, cell_centers, cell_size, _query_from_source=None):
        centers = np.asarray(cell_centers, dtype=np.float64)
        if centers.size == 0:
            centers = np.empty((0, 2), dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 2:
            raise ValueError("raster cells must contain [x, y] centres")
        if not np.all(np.isfinite(centers)):
            raise ValueError("raster cell centres must be finite")

        size = np.asarray(cell_size, dtype=np.float64)
        if size.ndim == 0:
            size = np.repeat(size.reshape(1), 2)
        if (
            size.shape != (2,)
            or not np.all(np.isfinite(size))
            or np.any(size <= 0.0)
        ):
            raise ValueError("raster cell size must be one or two positive values")
        if _query_from_source is not None and not isinstance(
            _query_from_source, RigidTransform2D
        ):
            raise ValueError("raster cell transform must be RigidTransform2D")

        self.cell_centers = centers.copy()
        self.cell_size = size.copy()
        self._half_size = 0.5 * size
        self._query_from_source = _query_from_source

    @staticmethod
    def _compose(outer, inner):
        """Return ``outer`` after ``inner`` as one rigid transform."""
        x, y = outer.apply_point(
            (inner.target_from_source_x, inner.target_from_source_y)
        )
        return RigidTransform2D(
            x,
            y,
            normalize_angle(
                outer.target_from_source_yaw + inner.target_from_source_yaw
            ),
            inner.source_frame,
            outer.target_frame,
        )

    def transformed(self, transform):
        if not isinstance(transform, RigidTransform2D):
            raise ValueError("raster cell transform must be RigidTransform2D")
        if self._query_from_source is None:
            query_from_source = transform
        else:
            query_from_source = self._compose(
                transform, self._query_from_source
            )
        return RasterCellBoundary(
            self.cell_centers,
            self.cell_size,
            _query_from_source=query_from_source,
        )

    @staticmethod
    def _point_to_centered_rectangle_distance(
        point_x, point_y, half_x, half_y
    ):
        outside_x = np.maximum(np.abs(point_x) - half_x, 0.0)
        outside_y = np.maximum(np.abs(point_y) - half_y, 0.0)
        return np.hypot(outside_x, outside_y)

    def clearance(self, pose, footprint):
        if self.cell_centers.size == 0:
            return math.inf
        pose = Pose2D.from_value(pose)
        if self._query_from_source is not None:
            pose = self._query_from_source.inverse().apply_pose(pose)

        cosine = math.cos(pose.yaw)
        sine = math.sin(pose.yaw)
        half_length = 0.5 * (footprint.front + footprint.rear)
        centre_offset = 0.5 * (footprint.front - footprint.rear)
        robot_centre_x = pose.x + cosine * centre_offset
        robot_centre_y = pose.y + sine * centre_offset
        robot_radius = math.hypot(half_length, footprint.half_width)
        cell_radius = math.hypot(self._half_size[0], self._half_size[1])

        # The nearest centre supplies a finite upper bound on the true set
        # distance.  Bounding-circle lower bounds then discard only cells that
        # provably cannot be closest; intersection candidates are never lost.
        centre_dx = self.cell_centers[:, 0] - robot_centre_x
        centre_dy = self.cell_centers[:, 1] - robot_centre_y
        centre_distance = np.hypot(centre_dx, centre_dy)
        nearest_centre_distance = float(np.min(centre_distance))
        combined_radius = robot_radius + cell_radius
        candidates = centre_distance <= (
            nearest_centre_distance + 2.0 * combined_radius + 1e-12
        )
        dx = centre_dx[candidates]
        dy = centre_dy[candidates]
        centres = self.cell_centers[candidates]

        # Rectangle-vs-rectangle SAT.  The four axes are exact because the two
        # shapes are rectangles.  A non-positive maximum means contact or
        # overlap; its magnitude is the minimum separating-axis translation.
        along = cosine * dx + sine * dy
        lateral = -sine * dx + cosine * dy
        half_cell_x = float(self._half_size[0])
        half_cell_y = float(self._half_size[1])
        absolute_cosine = abs(cosine)
        absolute_sine = abs(sine)
        separations = np.vstack(
            (
                np.abs(along)
                - (
                    half_length
                    + half_cell_x * absolute_cosine
                    + half_cell_y * absolute_sine
                ),
                np.abs(lateral)
                - (
                    footprint.half_width
                    + half_cell_x * absolute_sine
                    + half_cell_y * absolute_cosine
                ),
                np.abs(dx)
                - (
                    half_cell_x
                    + half_length * absolute_cosine
                    + footprint.half_width * absolute_sine
                ),
                np.abs(dy)
                - (
                    half_cell_y
                    + half_length * absolute_sine
                    + footprint.half_width * absolute_cosine
                ),
            )
        )
        separating_distance = np.max(separations, axis=0)
        signed = separating_distance.copy()
        separated = separating_distance > 0.0

        if np.any(separated):
            separated_centres = centres[separated]

            # Exact positive distance is the minimum of every vertex-to-edge
            # candidate in both directions.  This also covers parallel edges.
            robot_axis_x = np.asarray((cosine, sine), dtype=np.float64)
            robot_axis_y = np.asarray((-sine, cosine), dtype=np.float64)
            robot_corners = np.asarray(
                [
                    np.asarray((robot_centre_x, robot_centre_y))
                    + longitudinal_sign * half_length * robot_axis_x
                    + lateral_sign * footprint.half_width * robot_axis_y
                    for longitudinal_sign in (-1.0, 1.0)
                    for lateral_sign in (-1.0, 1.0)
                ],
                dtype=np.float64,
            )
            robot_to_cell = []
            for corner in robot_corners:
                robot_to_cell.append(
                    self._point_to_centered_rectangle_distance(
                        corner[0] - separated_centres[:, 0],
                        corner[1] - separated_centres[:, 1],
                        half_cell_x,
                        half_cell_y,
                    )
                )

            cell_to_robot = []
            for x_sign in (-1.0, 1.0):
                for y_sign in (-1.0, 1.0):
                    corner_x = (
                        separated_centres[:, 0]
                        + x_sign * half_cell_x
                        - robot_centre_x
                    )
                    corner_y = (
                        separated_centres[:, 1]
                        + y_sign * half_cell_y
                        - robot_centre_y
                    )
                    corner_along = cosine * corner_x + sine * corner_y
                    corner_lateral = -sine * corner_x + cosine * corner_y
                    cell_to_robot.append(
                        self._point_to_centered_rectangle_distance(
                            corner_along,
                            corner_lateral,
                            half_length,
                            footprint.half_width,
                        )
                    )
            signed[separated] = np.min(
                np.vstack(robot_to_cell + cell_to_robot), axis=0
            )

        return float(np.min(signed))


@lru_cache(maxsize=128)
def _local_footprint_points(
    front,
    rear,
    half_width,
    spacing,
    perimeter_only,
):
    longitudinal = np.linspace(
        -rear,
        front,
        max(2, int(math.ceil((front + rear) / spacing)) + 1),
    )
    lateral = np.linspace(
        -half_width,
        half_width,
        max(2, int(math.ceil(2.0 * half_width / spacing)) + 1),
    )
    if perimeter_only:
        local = np.vstack(
            (
                np.column_stack((longitudinal, np.full_like(longitudinal, lateral[0]))),
                np.column_stack((longitudinal, np.full_like(longitudinal, lateral[-1]))),
                np.column_stack((np.full_like(lateral, longitudinal[0]), lateral)),
                np.column_stack((np.full_like(lateral, longitudinal[-1]), lateral)),
            )
        )
    else:
        grid_x, grid_y = np.meshgrid(longitudinal, lateral)
        local = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    local.setflags(write=False)
    return local


def footprint_points(pose, footprint, spacing=0.008, perimeter_only=False):
    pose = Pose2D.from_value(pose)
    spacing = max(0.001, float(spacing))
    local = _local_footprint_points(
        float(footprint.front),
        float(footprint.rear),
        float(footprint.half_width),
        spacing,
        bool(perimeter_only),
    )
    cosine = math.cos(pose.yaw)
    sine = math.sin(pose.yaw)
    return np.column_stack(
        (
            pose.x + cosine * local[:, 0] - sine * local[:, 1],
            pose.y + sine * local[:, 0] + cosine * local[:, 1],
        )
    )


def point_obstacle_clearance(pose, footprint, obstacle_points):
    points = np.asarray(obstacle_points, dtype=np.float64)
    if points.size == 0:
        return math.inf
    points = points.reshape((-1, 2))
    pose = Pose2D.from_value(pose)
    cosine = math.cos(pose.yaw)
    sine = math.sin(pose.yaw)
    dx = points[:, 0] - pose.x
    dy = points[:, 1] - pose.y
    longitudinal = cosine * dx + sine * dy
    lateral = -sine * dx + cosine * dy
    outside_x = np.maximum.reduce(
        (
            -footprint.rear - longitudinal,
            longitudinal - footprint.front,
            np.zeros_like(longitudinal),
        )
    )
    outside_y = np.maximum(np.abs(lateral) - footprint.half_width, 0.0)
    signed = np.hypot(outside_x, outside_y)
    inside = (
        (longitudinal >= -footprint.rear)
        & (longitudinal <= footprint.front)
        & (np.abs(lateral) <= footprint.half_width)
    )
    signed[inside] = -np.minimum.reduce(
        (
            longitudinal[inside] + footprint.rear,
            footprint.front - longitudinal[inside],
            footprint.half_width - np.abs(lateral[inside]),
        )
    )
    return float(np.min(signed))


@dataclass(frozen=True)
class ValidationResult:
    safe: bool
    minimum_line_clearance: float = math.inf
    minimum_obstacle_clearance: float = math.inf
    minimum_map_clearance: float = math.inf
    first_unsafe_distance: float = math.inf
    samples: int = 0

    @property
    def minimum_clearance(self):
        return min(
            self.minimum_line_clearance,
            self.minimum_obstacle_clearance,
            self.minimum_map_clearance,
        )


@dataclass(frozen=True)
class SafetyDecision:
    """One runtime decision from nominal-route and complete-stop sweeps."""

    speed_limit: float
    route: ValidationResult
    stopping: ValidationResult

    @property
    def requires_stop(self):
        return self.speed_limit <= 1e-9

    @property
    def validation(self):
        return combine_validation_results((self.route, self.stopping))


class SweptFootprintValidator:
    """Validate the full asymmetric rectangle between every stored pose."""

    _STRAIGHT_ANGULAR_EPSILON = 1e-9

    def __init__(
        self,
        footprint,
        translation_step=0.008,
        heading_step=math.radians(3.0),
    ):
        self.footprint = footprint
        self.translation_step = max(0.001, float(translation_step))
        self.heading_step = max(math.radians(0.1), float(heading_step))

    def _pose_metrics(
        self,
        pose,
        safety,
        obstacle_points,
        line_footprint,
        obstacle_footprint,
    ):
        line = min(
            (
                boundary.clearance(pose, line_footprint)
                for boundary in safety.line_boundaries
            ),
            default=math.inf,
        )
        map_clearance = min(
            (
                boundary.clearance(pose, line_footprint)
                for boundary in safety.map_boundaries
            ),
            default=math.inf,
        )
        obstacle = point_obstacle_clearance(
            pose, obstacle_footprint, obstacle_points
        )
        return line, obstacle, map_clearance

    def _segment_subdivision_count(self, start, end):
        start = Pose2D.from_value(start)
        end = Pose2D.from_value(end)
        distance = math.hypot(end.x - start.x, end.y - start.y)
        yaw_delta = normalize_angle(end.yaw - start.yaw)

        return max(
            1,
            int(math.ceil(distance / self.translation_step)),
            int(math.ceil(abs(yaw_delta) / self.heading_step)),
        )

    def _segment_poses(self, start, end):
        start = Pose2D.from_value(start)
        end = Pose2D.from_value(end)
        yaw_delta = normalize_angle(end.yaw - start.yaw)
        count = self._segment_subdivision_count(start, end)
        for fraction in np.linspace(0.0, 1.0, count + 1):
            yield Pose2D(
                start.x + fraction * (end.x - start.x),
                start.y + fraction * (end.y - start.y),
                normalize_angle(start.yaw + fraction * yaw_delta),
            )

    def validate_poses(
        self,
        poses,
        safety=None,
        live_obstacles=None,
        line_overlap_allowances=None,
    ):
        safety = safety or PathSafety()
        if live_obstacles is None:
            live_obstacles = np.empty((0, 2), dtype=np.float64)
        live_obstacles = np.asarray(live_obstacles, dtype=np.float64)
        if live_obstacles.size == 0:
            live_obstacles = np.empty((0, 2), dtype=np.float64)
        else:
            live_obstacles = live_obstacles.reshape((-1, 2))
        if safety.fixed_obstacles.size == 0:
            obstacle_points = live_obstacles
        elif live_obstacles.size == 0:
            obstacle_points = safety.fixed_obstacles
        else:
            obstacle_points = np.vstack(
                (safety.fixed_obstacles, live_obstacles)
            )
        uncertainty = safety.margins.uncertainty
        line_footprint = self.footprint.expanded(
            safety.margins.line + uncertainty
        )
        obstacle_footprint = self.footprint.expanded(
            safety.margins.obstacle + uncertainty
        )
        poses = [Pose2D.from_value(pose) for pose in poses]
        if not poses:
            return ValidationResult(False, samples=0, first_unsafe_distance=0.0)
        allowances = _as_vector(
            line_overlap_allowances,
            len(poses),
            "line_overlap_allowances",
            default=0.0,
        )
        if (
            not np.all(np.isfinite(allowances))
            or np.any(allowances < 0.0)
        ):
            raise ValueError(
                "line overlap allowances must be finite and non-negative"
            )
        minimum_line = math.inf
        minimum_obstacle = math.inf
        minimum_map = math.inf
        travelled = 0.0
        first_unsafe = math.inf
        samples = 0
        previous = poses[0]
        line, obstacle, map_clearance = self._pose_metrics(
            previous,
            safety,
            obstacle_points,
            line_footprint,
            obstacle_footprint,
        )
        minimum_line = min(minimum_line, line)
        minimum_obstacle = min(minimum_obstacle, obstacle)
        minimum_map = min(minimum_map, map_clearance)
        samples = 1
        if (
            line + float(allowances[0]) <= 0.0
            or obstacle <= 0.0
            or map_clearance <= 0.0
        ):
            first_unsafe = 0.0
        for second_index in range(1, len(poses)):
            first_index = second_index - 1
            second_pose = poses[second_index]
            first_pose = poses[first_index]
            subdivision_count = self._segment_subdivision_count(
                first_pose, second_pose
            )
            if subdivision_count == 1:
                # Stopping sweeps already store poses at the validator spacing.
                # Their adjacent endpoints therefore need no temporary
                # linspace/tuple allocation.  Preserve the endpoint yaw
                # normalization performed by _segment_poses so this fast path
                # remains equivalent even for externally supplied unwrapped
                # headings.
                endpoint = Pose2D(
                    second_pose.x,
                    second_pose.y,
                    normalize_angle(
                        first_pose.yaw
                        + normalize_angle(second_pose.yaw - first_pose.yaw)
                    ),
                )
                segment_samples = ((1, endpoint),)
            else:
                segment_samples = enumerate(
                    tuple(self._segment_poses(first_pose, second_pose))[1:],
                    start=1,
                )
            # The first sample is the preceding segment's endpoint and has
            # already been evaluated with the same pose and allowance. Skipping
            # it preserves the exact continuous sweep while avoiding duplicate
            # boundary and obstacle calculations at every stored path pose.
            for sample_index, pose in segment_samples:
                fraction = float(sample_index) / float(
                    subdivision_count
                )
                line_allowance = float(
                    (1.0 - fraction) * allowances[first_index]
                    + fraction * allowances[second_index]
                )
                travelled += math.hypot(pose.x - previous.x, pose.y - previous.y)
                line, obstacle, map_clearance = self._pose_metrics(
                    pose,
                    safety,
                    obstacle_points,
                    line_footprint,
                    obstacle_footprint,
                )
                minimum_line = min(minimum_line, line)
                minimum_obstacle = min(minimum_obstacle, obstacle)
                minimum_map = min(minimum_map, map_clearance)
                samples += 1
                if (
                    line + line_allowance <= 0.0
                    or obstacle <= 0.0
                    or map_clearance <= 0.0
                ) and math.isinf(first_unsafe):
                    first_unsafe = travelled
                previous = pose
        return ValidationResult(
            math.isinf(first_unsafe),
            minimum_line,
            minimum_obstacle,
            minimum_map,
            first_unsafe,
            samples,
        )

    def validate_path(
        self,
        path,
        start_station=0.0,
        maximum_distance=None,
        safety=None,
        live_obstacles=None,
    ):
        start_station = clamp(float(start_station), 0.0, path.length)
        end_station = path.length
        if maximum_distance is not None:
            end_station = min(
                end_station, start_station + max(0.0, float(maximum_distance))
            )
        stations = [start_station]
        stations.extend(
            float(value)
            for value in path.station
            if start_station < float(value) < end_station
        )
        if end_station > start_station:
            stations.append(end_station)
        poses = []
        line_overlap_allowances = []
        for station in stations:
            sample = sample_path(path, station)
            poses.append(Pose2D(sample.x, sample.y, sample.heading))
            line_overlap_allowances.append(sample.line_overlap_allowance)
        return self.validate_poses(
            poses,
            safety or path.safety,
            live_obstacles=live_obstacles,
            line_overlap_allowances=line_overlap_allowances,
        )

    @staticmethod
    def _line_allowances_for_poses(path, poses, path_index):
        """Project a runtime sweep onto its non-regressing safety envelope."""
        if np.all(path.line_overlap_allowance == path.line_overlap_allowance[0]):
            # Most production paths use one immutable allowance (usually zero).
            # Projection cannot change that value, so avoid repeating a nearest-
            # path search for every reaction and braking sample.
            return [float(path.line_overlap_allowance[0])] * len(poses)
        allowances = []
        previous_index = max(0, min(int(path_index), path.size - 1))
        previous_station = _minimum_station_for_index(path, previous_index)
        for pose in poses:
            projection = project_to_path(
                path,
                pose.x,
                pose.y,
                previous_index=previous_index,
                search_ahead_distance=max(0.30, path.length),
                minimum_station=previous_station,
            )
            sample = sample_path(path, projection.station)
            allowances.append(sample.line_overlap_allowance)
            previous_index = projection.path_index
            previous_station = projection.station
        return allowances

    @staticmethod
    def _advance_unicycle(pose, linear, angular, elapsed):
        pose = Pose2D.from_value(pose)
        if abs(angular) <= SweptFootprintValidator._STRAIGHT_ANGULAR_EPSILON:
            return Pose2D(
                pose.x + linear * elapsed * math.cos(pose.yaw),
                pose.y + linear * elapsed * math.sin(pose.yaw),
                pose.yaw,
            )
        yaw = normalize_angle(pose.yaw + angular * elapsed)
        radius = linear / angular
        return Pose2D(
            pose.x + radius * (math.sin(yaw) - math.sin(pose.yaw)),
            pose.y - radius * (math.cos(yaw) - math.cos(pose.yaw)),
            yaw,
        )

    def _reaction_angular_samples(
        self, angular_velocities, reaction_time, linear_velocity=0.0
    ):
        """Cover the continuous yaw-rate interval during reaction latency.

        The measured, previously commanded, and newly requested yaw rates are
        discrete observations of one slew-limited actuator.  Checking only
        those three arcs can miss an obstacle swept by an intermediate rate.
        Sample the full interval densely enough that adjacent reaction-end
        headings and bounded centre-position separation respect the validator
        heading and translation steps.
        """
        values = tuple(
            0.0
            if abs(float(value))
            <= self._STRAIGHT_ANGULAR_EPSILON
            else float(value)
            for value in angular_velocities
            if math.isfinite(float(value))
        )
        if not values:
            return (0.0,)
        minimum = min(values)
        maximum = max(values)
        reaction_time = max(0.0, float(reaction_time))
        heading_span = abs(maximum - minimum) * reaction_time
        # Neighbouring constant-rate arcs must be close in both heading and
        # position.  The latter is bounded by integrating the difference in
        # their velocity headings over the reaction interval.
        spatial_span_bound = (
            0.5
            * abs(float(linear_velocity))
            * reaction_time
            * heading_span
        )
        intervals = max(
            1,
            int(math.ceil(heading_span / self.heading_step)),
            int(math.ceil(spatial_span_bound / self.translation_step)),
        )
        samples = set(values)
        samples.update(
            float(value)
            for value in np.linspace(minimum, maximum, intervals + 1)
        )
        if minimum <= 0.0 <= maximum:
            samples.add(0.0)
        return tuple(sorted(samples))

    def stopping_sweep(
        self,
        path,
        pose,
        path_index,
        linear_velocity,
        angular_velocities,
        reaction_time,
        linear_deceleration,
        distance_margin=0.0,
        safety=None,
        live_obstacles=None,
    ):
        """Sweep reaction arcs and an error-preserving complete stop region."""
        pose = Pose2D.from_value(pose)
        speed = abs(float(linear_velocity))
        reaction_time = max(0.0, float(reaction_time))
        deceleration = max(1e-6, float(linear_deceleration))
        braking_distance = speed * speed / (2.0 * deceleration) + max(
            0.0, float(distance_margin)
        )
        safety = safety or path.safety
        results = []
        angular_velocities = self._reaction_angular_samples(
            angular_velocities, reaction_time, linear_velocity
        )
        for angular in angular_velocities:
            steps = max(
                1,
                int(
                    math.ceil(
                        speed * reaction_time / self.translation_step
                    )
                ),
                int(
                    math.ceil(
                        abs(angular) * reaction_time / self.heading_step
                    )
                ),
            )
            reaction_sequence = [pose]
            for step in range(1, steps + 1):
                reaction_sequence.append(
                    self._advance_unicycle(
                        pose,
                        float(linear_velocity),
                        angular,
                        reaction_time * float(step) / steps,
                    )
                )
            reaction_end = reaction_sequence[-1]
            projection = project_to_path(
                path,
                reaction_end.x,
                reaction_end.y,
                previous_index=path_index,
                search_ahead_distance=max(0.30, braking_distance + 0.05),
            )
            reference = sample_path(path, projection.station)
            motion_heading = normalize_angle(
                reference.heading
                + (math.pi if reference.direction < 0 else 0.0)
            )
            dx = reaction_end.x - reference.x
            dy = reaction_end.y - reference.y
            along_error = (
                math.cos(motion_heading) * dx + math.sin(motion_heading) * dy
            )
            lateral_error = (
                -math.sin(motion_heading) * dx + math.cos(motion_heading) * dy
            )
            heading_error = normalize_angle(reaction_end.yaw - reference.heading)
            count = max(1, int(math.ceil(braking_distance / self.translation_step)))
            braking_sequence = []
            for travel in np.linspace(0.0, braking_distance, count + 1):
                requested_station = projection.station + float(travel)
                station = min(path.length, requested_station)
                sample = sample_path(path, station)
                sample_motion = normalize_angle(
                    sample.heading + (math.pi if sample.direction < 0 else 0.0)
                )
                overrun = max(0.0, requested_station - path.length)
                base_x = sample.x + overrun * math.cos(sample_motion)
                base_y = sample.y + overrun * math.sin(sample_motion)
                braking_sequence.append(
                    Pose2D(
                        base_x
                        + math.cos(sample_motion) * along_error
                        - math.sin(sample_motion) * lateral_error,
                        base_y
                        + math.sin(sample_motion) * along_error
                        + math.cos(sample_motion) * lateral_error,
                        normalize_angle(sample.heading + heading_error),
                    )
                )
            sequence = reaction_sequence + braking_sequence[1:]
            results.append(
                self.validate_poses(
                    sequence,
                    safety,
                    live_obstacles=live_obstacles,
                    line_overlap_allowances=self._line_allowances_for_poses(
                        path, sequence, path_index
                    ),
                )
            )
        return combine_validation_results(results)

    def motion_safety(
        self,
        path,
        pose,
        path_index,
        desired_speed,
        linear_velocity,
        angular_velocities,
        reaction_time,
        linear_deceleration,
        distance_margin=0.0,
        lookahead_distance=None,
        safety=None,
        live_obstacles=None,
        tracking=None,
        route_safety=None,
    ):
        """Return the only speed reduction justified by a predicted contact.

        The nominal route sweep provides gradual braking distance.  The sweep
        from the actual pose, retaining current lateral and heading error,
        decides whether a bounded stop is already required.  ``route_safety``
        may omit fixed boundaries that were already checked for the immutable
        path, but ``safety`` always remains authoritative for the actual
        reaction and complete-stop sweep.  Omitting ``route_safety`` preserves
        the original behavior and uses ``safety`` for both sweeps.
        """
        pose = Pose2D.from_value(pose)
        desired_speed = max(0.0, float(desired_speed))
        if tracking is None:
            projection = project_to_path(
                path,
                pose.x,
                pose.y,
                previous_index=path_index,
                search_ahead_distance=max(
                    0.30,
                    desired_speed * max(0.0, float(reaction_time))
                    + desired_speed * desired_speed
                    / (2.0 * max(1e-9, float(linear_deceleration)))
                    + max(0.0, float(distance_margin)),
                ),
            )
            start_station = projection.station
            stopping_path_index = projection.path_index
        else:
            if not isinstance(tracking, TrackingResult):
                raise TypeError("tracking must be a TrackingResult")
            if not 0 <= tracking.path_index < path.size:
                raise ValueError("tracking path index is outside the path")
            if not 0.0 <= tracking.station <= path.length + 1e-9:
                raise ValueError("tracking station is outside the path")
            start_station = clamp(tracking.station, 0.0, path.length)
            stopping_path_index = tracking.path_index
        route = self.validate_path(
            path,
            start_station=start_station,
            maximum_distance=lookahead_distance,
            safety=safety if route_safety is None else route_safety,
            live_obstacles=live_obstacles,
        )
        speed_limit = desired_speed
        if not route.safe:
            usable_distance = max(
                0.0, route.first_unsafe_distance - max(0.0, float(distance_margin))
            )
            speed_limit = min(
                speed_limit,
                safe_speed_for_distance(
                    usable_distance, reaction_time, linear_deceleration
                ),
            )
        stopping = self.stopping_sweep(
            path,
            pose,
            stopping_path_index,
            linear_velocity,
            angular_velocities,
            reaction_time,
            linear_deceleration,
            distance_margin,
            safety=safety,
            live_obstacles=live_obstacles,
        )
        if not stopping.safe:
            speed_limit = 0.0
        return SafetyDecision(speed_limit, route, stopping)


def combine_validation_results(results):
    results = tuple(results)
    if not results:
        return ValidationResult(True)
    return ValidationResult(
        all(result.safe for result in results),
        min(result.minimum_line_clearance for result in results),
        min(result.minimum_obstacle_clearance for result in results),
        min(result.minimum_map_clearance for result in results),
        min(result.first_unsafe_distance for result in results),
        sum(result.samples for result in results),
    )


def safe_speed_for_distance(distance, reaction_time, deceleration):
    """Maximum speed whose reaction plus braking distance fits ``distance``."""
    distance = max(0.0, float(distance))
    reaction_time = max(0.0, float(reaction_time))
    deceleration = max(1e-9, float(deceleration))
    term = deceleration * reaction_time
    return max(0.0, math.sqrt(term * term + 2.0 * deceleration * distance) - term)


@dataclass
class PathDiagnostics:
    progress: float = 0.0
    remaining_distance: float = math.inf
    position_error: float = math.inf
    cross_track_error: float = math.inf
    heading_error: float = math.inf
    target_speed: float = 0.0
    target_index: int = 0
    curvature: float = 0.0
    minimum_line_clearance: float = math.inf
    minimum_obstacle_clearance: float = math.inf
    minimum_map_clearance: float = math.inf
    commanded_linear: float = 0.0
    commanded_angular: float = 0.0

    @classmethod
    def from_array(cls, values):
        """Parse the one public 13-field diagnostics representation."""

        if len(values) != 13:
            raise ValueError("path diagnostics must contain exactly 13 values")
        try:
            parsed = tuple(float(value) for value in values)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("path diagnostics values must be numeric") from error
        clearance_indices = (8, 9, 10)
        for index, value in enumerate(parsed):
            if math.isnan(value) or value == -math.inf:
                raise ValueError("path diagnostics contain an invalid value")
            if math.isinf(value) and index not in clearance_indices:
                raise ValueError("only absent-boundary clearance may be infinite")
        if not 0.0 <= parsed[0] <= 1.0:
            raise ValueError("path diagnostic progress is outside [0, 1]")
        if any(parsed[index] < 0.0 for index in (1, 2, 5, 6)):
            raise ValueError("path diagnostic magnitude cannot be negative")
        if abs(parsed[4]) > math.pi + 1e-6:
            raise ValueError("path diagnostic heading error is not normalized")
        target_index = int(round(parsed[6]))
        if abs(parsed[6] - target_index) > 1e-6:
            raise ValueError("path diagnostic target index must be integral")
        return cls(
            progress=parsed[0],
            remaining_distance=parsed[1],
            position_error=parsed[2],
            cross_track_error=parsed[3],
            heading_error=parsed[4],
            target_speed=parsed[5],
            target_index=target_index,
            curvature=parsed[7],
            minimum_line_clearance=parsed[8],
            minimum_obstacle_clearance=parsed[9],
            minimum_map_clearance=parsed[10],
            commanded_linear=parsed[11],
            commanded_angular=parsed[12],
        )

    def as_array(self):
        return [
            float(self.progress),
            float(self.remaining_distance),
            float(self.position_error),
            float(self.cross_track_error),
            float(self.heading_error),
            float(self.target_speed),
            float(self.target_index),
            float(self.curvature),
            float(self.minimum_line_clearance),
            float(self.minimum_obstacle_clearance),
            float(self.minimum_map_clearance),
            float(self.commanded_linear),
            float(self.commanded_angular),
        ]


class PathFollower:
    """Stateful common follower; controllers only publish the returned command."""

    def __init__(self, config):
        self.config = config
        self.path = None
        self.path_index = 0
        self.path_station = 0.0
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.diagnostics = PathDiagnostics()

    def reset(self, path, pose, initial_linear=0.0, initial_angular=0.0):
        pose = Pose2D.from_value(pose)
        projection = project_to_path(path, pose.x, pose.y)
        self.path = path
        self.path_index = projection.path_index
        self.path_station = projection.station
        self.last_linear = clamp(
            float(initial_linear),
            -self.config.maximum_linear_velocity,
            self.config.maximum_linear_velocity,
        )
        self.last_angular = clamp(
            float(initial_angular),
            -self.config.maximum_angular_velocity,
            self.config.maximum_angular_velocity,
        )
        self.diagnostics = PathDiagnostics(
            progress=0.0 if path.length <= 0.0 else projection.station / path.length,
            remaining_distance=max(0.0, path.length - projection.station),
            position_error=projection.distance,
            cross_track_error=projection.cross_track_error,
            heading_error=normalize_angle(projection.heading - pose.yaw),
            minimum_line_clearance=path.line_clearance,
            minimum_obstacle_clearance=path.obstacle_clearance,
            minimum_map_clearance=path.map_clearance,
        )
        return projection

    def stop(self, elapsed):
        if self.path is None:
            self.last_linear = 0.0
            self.last_angular = 0.0
            return VelocityCommand(0.0, 0.0)
        command = limit_velocity_command(
            0.0,
            max(abs(self.last_linear), 1e-9),
            0.0,
            self.last_linear,
            self.last_angular,
            elapsed,
            self.config,
        )
        self.last_linear = command.linear_velocity
        self.last_angular = command.angular_velocity
        self.diagnostics.commanded_linear = command.linear_velocity
        self.diagnostics.commanded_angular = command.angular_velocity
        return command

    def _terminal_hold_command(self, status, elapsed):
        """Brake translation and finish endpoint heading without path overrun."""
        elapsed = max(0.0, float(elapsed))
        previous_linear = clamp(
            float(self.last_linear),
            -self.config.maximum_linear_velocity,
            self.config.maximum_linear_velocity,
        )
        reduction = self.config.linear_deceleration * elapsed
        linear = math.copysign(
            max(0.0, abs(previous_linear) - reduction),
            previous_linear,
        )
        if abs(linear) <= 1e-12:
            linear = 0.0

        previous_angular = clamp(
            float(self.last_angular),
            -self.config.maximum_angular_velocity,
            self.config.maximum_angular_velocity,
        )
        target_angular = clamp(
            self.config.heading_gain * status.heading_error,
            -self.config.maximum_angular_velocity,
            self.config.maximum_angular_velocity,
        )
        angular_change = self.config.angular_acceleration * elapsed
        angular = clamp(
            target_angular,
            previous_angular - angular_change,
            previous_angular + angular_change,
        )
        if abs(linear) > 1e-9:
            lateral_cap = self.config.maximum_lateral_acceleration / abs(linear)
            angular = clamp(angular, -lateral_cap, lateral_cap)

        command = VelocityCommand(linear, angular)
        self.last_linear = command.linear_velocity
        self.last_angular = command.angular_velocity
        self.diagnostics.commanded_linear = command.linear_velocity
        self.diagnostics.commanded_angular = command.angular_velocity
        return command

    def calculate_tracking(
        self,
        pose,
        linear_velocity=None,
        minimum_target_station=None,
    ):
        """Calculate this tick's immutable tracking result exactly once.

        Callers that also run swept safety or completion checks should pass the
        returned object to those operations and to :meth:`command`.  It is only
        valid for the currently active path and the pose used here.  Supplying
        measured ``linear_velocity`` lets an adaptive configuration select its
        steering horizon; the last bounded command is the deterministic default.
        """
        if self.path is None:
            raise ValueError("no path is active")
        if linear_velocity is None:
            linear_velocity = self.last_linear
        return calculate_tracking(
            self.path,
            pose,
            self.path_index,
            self.config,
            previous_station=self.path_station,
            linear_velocity=linear_velocity,
            minimum_target_station=minimum_target_station,
        )

    def _record_tracking(self, tracking):
        if not isinstance(tracking, TrackingResult):
            raise TypeError("tracking must be a TrackingResult")
        if not 0 <= tracking.path_index < self.path.size:
            raise ValueError("tracking path index is outside the active path")
        if not 0 <= tracking.target_index < self.path.size:
            raise ValueError("tracking target index is outside the active path")
        if not 0.0 <= tracking.station <= self.path.length + 1e-9:
            raise ValueError("tracking station is outside the active path")
        if tracking.station < self.path_station - 1e-9:
            raise ValueError("tracking station regressed behind active progress")
        if tracking.path_index < self.path_index:
            raise ValueError("tracking path index regressed behind active progress")
        self.path_index = tracking.path_index
        self.path_station = max(self.path_station, tracking.station)
        self.diagnostics.progress = (
            1.0
            if self.path.length <= 0.0
            else self.path_station / self.path.length
        )
        self.diagnostics.remaining_distance = max(
            0.0, self.path.length - self.path_station
        )
        self.diagnostics.position_error = tracking.position_error
        self.diagnostics.cross_track_error = tracking.cross_track_error
        self.diagnostics.heading_error = tracking.heading_error
        self.diagnostics.target_speed = tracking.target_speed
        self.diagnostics.target_index = tracking.target_index
        self.diagnostics.curvature = tracking.curvature_command

    def update_tracking(self, tracking):
        """Publish-ready diagnostics for a calculated, non-commanded path.

        A controller may keep validating a rolling path while another mission
        owns velocity publication. This records that already calculated
        tracking result without producing or slew-limiting a command.
        """

        self._record_tracking(tracking)
        return tracking

    def command(
        self,
        pose,
        elapsed,
        speed_scale=1.0,
        speed_limit=math.inf,
        tracking=None,
        linear_velocity=None,
        minimum_target_station=None,
    ):
        """Limit a command, optionally reusing a result calculated this tick.

        ``linear_velocity`` and ``minimum_target_station`` are forwarded only
        when this call performs tracking itself.  A caller that pre-calculates
        one immutable result should supply those inputs to
        :meth:`calculate_tracking` and pass that result here.
        """
        if self.path is None:
            raise ValueError("no path is active")
        if tracking is None:
            tracking = self.calculate_tracking(
                pose,
                linear_velocity=linear_velocity,
                minimum_target_station=minimum_target_station,
            )
        self._record_tracking(tracking)
        status = _goal_status_from_progress(
            self.path,
            pose,
            tracking.path_index,
            tracking.station,
        )
        if status.crossed_terminal:
            return self._terminal_hold_command(status, elapsed), tracking
        magnitude = tracking.target_speed * clamp(float(speed_scale), 0.0, 1.0)
        magnitude = min(magnitude, max(0.0, float(speed_limit)))
        target_linear = tracking.direction * magnitude
        limited = limit_velocity_command(
            target_linear,
            tracking.target_speed,
            # The limiter derives curvature from angular/reference speed and
            # then applies that curvature to the slew-limited linear command.
            # Scaling angular here as well would square a safety speed scale
            # and make the robot understeer precisely while braking near a
            # boundary or obstacle.
            tracking.angular_velocity,
            self.last_linear,
            self.last_angular,
            elapsed,
            self.config,
        )
        self.last_linear = limited.linear_velocity
        self.last_angular = limited.angular_velocity
        self.diagnostics.commanded_linear = limited.linear_velocity
        self.diagnostics.commanded_angular = limited.angular_velocity
        return limited, tracking

    def terminal_hold(self, pose, elapsed, tracking=None):
        """Stop translation after crossing and converge endpoint heading.

        Mission controllers use this while an external state/zone handshake is
        still pending.  It prevents a non-zero ``exit_velocity`` from driving
        beyond the frozen path, while preserving the common deceleration and
        angular-acceleration limits.  Position and heading completion may still
        be pending, but projected progress must establish a real path-terminal
        crossing before a controller can request this hold.
        """
        if self.path is None:
            raise ValueError("no path is active")
        if tracking is None:
            tracking = self.calculate_tracking(pose)
        self._record_tracking(tracking)
        status = _goal_status_from_progress(
            self.path,
            pose,
            tracking.path_index,
            tracking.station,
        )
        if not status.crossed_terminal:
            raise ValueError(
                "terminal hold requires a near-path terminal crossing"
            )
        return self._terminal_hold_command(status, elapsed), tracking, status

    def update_clearance(self, validation):
        self.diagnostics.minimum_line_clearance = min(
            self.path.line_clearance, validation.minimum_line_clearance
        )
        self.diagnostics.minimum_obstacle_clearance = min(
            self.path.obstacle_clearance, validation.minimum_obstacle_clearance
        )
        self.diagnostics.minimum_map_clearance = min(
            self.path.map_clearance, validation.minimum_map_clearance
        )

    def goal_status(self, pose, tracking=None):
        if self.path is None:
            return GoalStatus(False, math.inf, math.inf, math.inf, False)
        if tracking is None:
            tracking = self.calculate_tracking(pose)
        self._record_tracking(tracking)
        return _goal_status_from_progress(
            self.path,
            pose,
            tracking.path_index,
            tracking.station,
        )
