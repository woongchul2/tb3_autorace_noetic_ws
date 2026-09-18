#!/usr/bin/env python3
"""ROS-independent, fully-observed tunnel entrance registration.

The entrance used by the AutoRace course is an offset opening: one wall runs
into the tunnel and the other wall terminates across the entrance.  A single
longitudinal wall supplies heading and lateral offset, but cannot supply the
longitudinal origin.  Registration is therefore returned only when a second,
roughly perpendicular wall and both visible entrance endpoints are present.
"""

from dataclasses import dataclass
import math
from typing import Iterable, Optional

import numpy as np


def _finite(name, value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("{} must be finite".format(name)) from error
    if not math.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def _positive_int(name, value):
    if isinstance(value, bool):
        raise ValueError("{} must be a positive integer".format(name))
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("{} must be a positive integer".format(name)) from error
    if result != value or result <= 0:
        raise ValueError("{} must be a positive integer".format(name))
    return result


def normalize_angle(angle):
    angle = _finite("angle", angle)
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class PortalRegistrationConfig:
    minimum_forward_distance: float = 0.10
    maximum_forward_distance: float = 1.50
    minimum_absolute_lateral: float = 0.06
    maximum_absolute_lateral: float = 0.45
    maximum_adjacent_beam_gap: int = 3
    maximum_point_gap: float = 0.18
    minimum_wall_points: int = 5
    minimum_wall_length: float = 0.20
    maximum_wall_residual: float = 0.025
    maximum_heading_deviation: float = math.radians(35.0)
    maximum_orthogonal_angle: float = math.radians(10.0)
    expected_portal_width: float = 0.286
    portal_width_tolerance: float = 0.08
    maximum_corner_skew: float = 0.10

    def __post_init__(self):
        float_fields = (
            "minimum_forward_distance",
            "maximum_forward_distance",
            "minimum_absolute_lateral",
            "maximum_absolute_lateral",
            "maximum_point_gap",
            "minimum_wall_length",
            "maximum_wall_residual",
            "maximum_heading_deviation",
            "maximum_orthogonal_angle",
            "expected_portal_width",
            "portal_width_tolerance",
            "maximum_corner_skew",
        )
        for name in float_fields:
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        object.__setattr__(
            self,
            "maximum_adjacent_beam_gap",
            _positive_int(
                "maximum_adjacent_beam_gap", self.maximum_adjacent_beam_gap
            ),
        )
        object.__setattr__(
            self,
            "minimum_wall_points",
            _positive_int("minimum_wall_points", self.minimum_wall_points),
        )
        if self.minimum_forward_distance < 0.0 or (
            self.maximum_forward_distance <= self.minimum_forward_distance
        ):
            raise ValueError("forward registration bounds are invalid")
        if self.minimum_absolute_lateral < 0.0 or (
            self.maximum_absolute_lateral <= self.minimum_absolute_lateral
        ):
            raise ValueError("lateral registration bounds are invalid")
        if min(
            self.maximum_point_gap,
            self.minimum_wall_length,
            self.maximum_wall_residual,
            self.expected_portal_width,
            self.portal_width_tolerance,
            self.maximum_corner_skew,
        ) <= 0.0:
            raise ValueError("portal geometry lengths must be positive")
        if self.minimum_wall_points < 2:
            raise ValueError("minimum_wall_points must be at least two")
        if not (0.0 < self.maximum_heading_deviation < 0.5 * math.pi):
            raise ValueError("maximum_heading_deviation must be in (0, pi/2)")
        if not (0.0 < self.maximum_orthogonal_angle < 0.5 * math.pi):
            raise ValueError("maximum_orthogonal_angle must be in (0, pi/2)")


@dataclass(frozen=True)
class PortalWall:
    corner_x: float
    corner_y: float
    far_x: float
    far_y: float
    yaw: float
    length: float
    residual: float
    point_count: int


@dataclass(frozen=True)
class PortalDetection:
    center_x: float
    center_y: float
    yaw: float
    width: float
    corner_skew: float
    longitudinal_wall: PortalWall
    transverse_wall: PortalWall


@dataclass(frozen=True)
class _Point:
    index: int
    x: float
    y: float


def _clusters(points, config):
    result = []
    current = []
    previous = None
    for point in points:
        if previous is not None and (
            point.index - previous.index > config.maximum_adjacent_beam_gap
            or math.hypot(point.x - previous.x, point.y - previous.y)
            > config.maximum_point_gap
        ):
            if current:
                result.append(current)
            current = []
        current.append(point)
        previous = point
    if current:
        result.append(current)
    return result


def _fit_wall(points, config):
    if len(points) < config.minimum_wall_points:
        return None
    coordinates = np.asarray([(point.x, point.y) for point in points])
    center = np.mean(coordinates, axis=0)
    centered = coordinates - center
    covariance = centered.T.dot(centered) / float(len(points))
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    if direction[0] < 0.0:
        direction = -direction
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    station = centered.dot(direction)
    length = float(np.max(station) - np.min(station))
    if length < config.minimum_wall_length:
        return None
    normal = np.asarray((-direction[1], direction[0]))
    residual = float(np.sqrt(np.mean(np.square(centered.dot(normal)))))
    if residual > config.maximum_wall_residual:
        return None
    corner = center + float(np.min(station)) * direction
    far = center + float(np.max(station)) * direction
    return PortalWall(
        corner_x=float(corner[0]),
        corner_y=float(corner[1]),
        far_x=float(far[0]),
        far_y=float(far[1]),
        yaw=yaw,
        length=length,
        residual=residual,
        point_count=len(points),
    )


def _wall_candidates(points, config):
    candidates = []
    for cluster in _clusters(points, config):
        wall = _fit_wall(cluster, config)
        if wall is not None:
            candidates.append(wall)
    return candidates


def _axis_error(yaw, target):
    """Smallest unoriented line-angle error from ``target``."""
    return min(
        abs(normalize_angle(yaw - target)),
        abs(normalize_angle(yaw - target + math.pi)),
        abs(normalize_angle(yaw - target - math.pi)),
    )


def _endpoint_nearest_centerline(wall):
    first = (wall.corner_x, wall.corner_y)
    second = (wall.far_x, wall.far_y)
    return first if abs(first[1]) <= abs(second[1]) else second


def _fused_longitudinal_yaw(longitudinal_yaw, transverse_yaw):
    """Fuse the two unoriented wall axes into one longitudinal heading.

    ``_fit_wall`` gives each line an arbitrary but deterministic direction.
    First rotate the transverse axis by either signed quarter turn, choosing
    the representation nearest the already forward-oriented longitudinal
    axis.  Averaging the resulting wrapped angular difference avoids the
    discontinuity at ``+/-pi`` while giving both independently fitted walls
    equal weight.
    """
    transverse_longitudinal = min(
        (
            normalize_angle(transverse_yaw - 0.5 * math.pi),
            normalize_angle(transverse_yaw + 0.5 * math.pi),
        ),
        key=lambda candidate: abs(
            normalize_angle(candidate - longitudinal_yaw)
        ),
    )
    difference = normalize_angle(
        transverse_longitudinal - longitudinal_yaw
    )
    return normalize_angle(longitudinal_yaw + 0.5 * difference)


def detect_portal_from_indexed_points(
    points: Iterable, config: PortalRegistrationConfig
) -> Optional[PortalDetection]:
    """Detect complementary portal walls and their entrance endpoints.

    ``points`` contains ``(beam_index, x, y)`` triples in the LiDAR frame.
    A longitudinal wall and an independently fitted transverse wall are both
    mandatory.  In particular, one long wall cannot observe translation along
    itself and therefore never yields a registration.
    """
    if not isinstance(config, PortalRegistrationConfig):
        raise TypeError("config must be PortalRegistrationConfig")
    filtered = []
    for raw_index, raw_x, raw_y in points:
        try:
            index = int(raw_index)
            x = float(raw_x)
            y = float(raw_y)
        except (TypeError, ValueError, OverflowError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        absolute_y = abs(y)
        if not (
            config.minimum_forward_distance <= x <= config.maximum_forward_distance
            and config.minimum_absolute_lateral
            <= absolute_y
            <= config.maximum_absolute_lateral
        ):
            continue
        filtered.append(_Point(index=index, x=x, y=y))

    filtered.sort(key=lambda point: point.index)
    walls = _wall_candidates(filtered, config)
    longitudinal = [
        wall
        for wall in walls
        if _axis_error(wall.yaw, 0.0) <= config.maximum_heading_deviation
    ]
    transverse = [
        wall
        for wall in walls
        if _axis_error(wall.yaw, 0.5 * math.pi)
        <= config.maximum_orthogonal_angle
    ]
    if not longitudinal or not transverse:
        return None

    candidates = []
    for along_wall in longitudinal:
        yaw = along_wall.yaw
        if math.cos(yaw) < 0.0:
            yaw = normalize_angle(yaw + math.pi)
        forward_x, forward_y = math.cos(yaw), math.sin(yaw)
        left_x, left_y = -forward_y, forward_x
        # The endpoint behind the other one along the detected travel axis is
        # the longitudinal wall's entrance corner.
        along_endpoints = (
            (along_wall.corner_x, along_wall.corner_y),
            (along_wall.far_x, along_wall.far_y),
        )
        along_corner = min(
            along_endpoints,
            key=lambda point: point[0] * forward_x + point[1] * forward_y,
        )
        for across_wall in transverse:
            across_corner = _endpoint_nearest_centerline(across_wall)
            dx = across_corner[0] - along_corner[0]
            dy = across_corner[1] - along_corner[1]
            signed_width = dx * left_x + dy * left_y
            width = abs(signed_width)
            corner_skew = abs(dx * forward_x + dy * forward_y)
            if signed_width == 0.0:
                continue
            # Both corners must bracket the sensor centreline; otherwise a
            # random perpendicular interior surface could masquerade as the
            # opposite portal jamb.
            along_lateral = (
                along_corner[0] * left_x + along_corner[1] * left_y
            )
            across_lateral = (
                across_corner[0] * left_x + across_corner[1] * left_y
            )
            if along_lateral * across_lateral >= 0.0:
                continue
            if abs(width - config.expected_portal_width) > (
                config.portal_width_tolerance
            ):
                continue
            if corner_skew > config.maximum_corner_skew:
                continue
            score = (
                along_wall.residual + across_wall.residual,
                corner_skew,
                abs(width - config.expected_portal_width),
                -(along_wall.length + across_wall.length),
            )
            fused_yaw = _fused_longitudinal_yaw(
                yaw, across_wall.yaw
            )
            candidates.append(
                (
                    score,
                    PortalDetection(
                        center_x=0.5 * (along_corner[0] + across_corner[0]),
                        center_y=0.5 * (along_corner[1] + across_corner[1]),
                        yaw=fused_yaw,
                        width=width,
                        corner_skew=corner_skew,
                        longitudinal_wall=along_wall,
                        transverse_wall=across_wall,
                    ),
                )
            )
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def detect_portal(
    ranges: Iterable[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    config: PortalRegistrationConfig,
) -> Optional[PortalDetection]:
    angle_min = _finite("angle_min", angle_min)
    angle_increment = _finite("angle_increment", angle_increment)
    range_min = _finite("range_min", range_min)
    range_max = _finite("range_max", range_max)
    if angle_increment == 0.0:
        raise ValueError("angle_increment must be non-zero")
    if range_min < 0.0 or range_max < range_min:
        raise ValueError("range bounds must satisfy 0 <= min <= max")
    points = []
    for index, raw_range in enumerate(ranges):
        try:
            distance = float(raw_range)
        except (TypeError, ValueError, OverflowError):
            continue
        if not (
            math.isfinite(distance)
            and range_min <= distance <= range_max
        ):
            continue
        angle = angle_min + index * angle_increment
        points.append(
            (index, distance * math.cos(angle), distance * math.sin(angle))
        )
    return detect_portal_from_indexed_points(points, config)


__all__ = [
    "PortalDetection",
    "PortalRegistrationConfig",
    "PortalWall",
    "detect_portal",
    "detect_portal_from_indexed_points",
]
