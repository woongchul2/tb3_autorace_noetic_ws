#!/usr/bin/env python3
"""ROS-independent registration of immutable mission-local geometry.

The course between missions may change length without changing the geometry
inside a mission.  This module therefore estimates one rigid transform from a
mission-local template into a live target frame (normally ``odom``).  It does
not perform mission selection, command ownership, or path following.

Point landmarks constrain planar position.  Oriented segment landmarks add a
heading constraint and a normal-distance constraint.  A segment whose
``longitudinal_weight`` is zero models an unbounded wall or painted line: it
deliberately does *not* constrain translation along that line.  This makes the
information-rank test reject the unsafe single-straight-wall case instead of
returning an arbitrary longitudinal alignment.
"""

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np

from custom_autorace_bringup.path_following import (
    Pose2D,
    RigidTransform2D,
    normalize_angle,
)


__all__ = (
    "CurveRegistrationConfig",
    "CurveRegistrationDiagnostics",
    "CurveRegistrationResult",
    "EntryPlane",
    "EntryPlaneProgress",
    "LocalCurveTemplate",
    "MissionLocalTemplate",
    "ObservedCurve",
    "OrientedSegmentLandmark",
    "OrientedSegmentObservation",
    "PointLandmark",
    "PointObservation",
    "RegistrationConfig",
    "RegistrationDiagnostics",
    "RegistrationObservation",
    "RegistrationResult",
    "TemporalRegistrationConfig",
    "TemporalRegistrationFilter",
    "TemporalRegistrationState",
    "entry_plane_progress",
    "estimate_local_registration",
    "register_point_with_heading",
    "registration_covariance_with_floor",
    "registration_radial_uncertainties",
    "registration_radial_uncertainty",
    "register_curve_subset",
)


_EPSILON = 1e-12


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)
    return value


def _positive(value, name, allow_zero=False):
    value = _finite(value, name)
    if value < 0.0 or (not allow_zero and value <= 0.0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError("%s must be %s" % (name, qualifier))
    return value


def _point(value, name):
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        raise ValueError("%s must contain two finite coordinates" % name)
    if len(result) != 2 or not all(math.isfinite(component) for component in result):
        raise ValueError("%s must contain two finite coordinates" % name)
    return result


def _covariance_tuple(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return tuple()
    return tuple(tuple(float(value) for value in row) for row in matrix)


def registration_radial_uncertainties(
    covariance,
    footprint,
    local_points=None,
    target_from_source_yaw=0.0,
):
    """Return radial registration clearance for every mission-local point.

    The translational term is the largest one-sigma axis in ``x/y``.  Heading
    uncertainty is propagated independently to each supplied path point and
    converted to a linear displacement at the furthest footprint corner.
    Passing no points returns the origin calculation.  An empty covariance
    returns zeros because no estimated-registration term is applicable (for
    example, an explicitly odom-aligned simulation route).
    """

    if local_points is None:
        points = np.zeros((1, 2), dtype=np.float64)
    else:
        points = np.asarray(local_points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape((1, -1))
        if (
            points.ndim != 2
            or points.shape[0] < 1
            or points.shape[1] != 2
            or not np.all(np.isfinite(points))
        ):
            raise ValueError("local registration points must be finite [x, y] rows")
    if covariance is None:
        return np.zeros(points.shape[0], dtype=np.float64)
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.size == 0:
        return np.zeros(points.shape[0], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("registration covariance must be a finite 3x3 matrix")
    matrix = 0.5 * (matrix + matrix.T)
    covariance_eigenvalues = np.linalg.eigvalsh(matrix)
    yaw_variance = float(matrix[2, 2])
    tolerance = 1e-12
    if float(np.min(covariance_eigenvalues)) < -tolerance:
        raise ValueError("registration covariance must be positive semidefinite")
    yaw = _finite(target_from_source_yaw, "registration transform heading")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    centre_uncertainties = []
    for point_x, point_y in points:
        jacobian = np.asarray(
            (
                (1.0, 0.0, -sine * point_x - cosine * point_y),
                (0.0, 1.0, cosine * point_x - sine * point_y),
            ),
            dtype=np.float64,
        )
        point_covariance = jacobian @ matrix @ jacobian.T
        point_eigenvalues = np.linalg.eigvalsh(
            0.5 * (point_covariance + point_covariance.T)
        )
        centre_uncertainties.append(
            math.sqrt(max(0.0, float(np.max(point_eigenvalues))))
        )
    radius = math.hypot(
        max(float(footprint.front), float(footprint.rear)),
        float(footprint.half_width),
    )
    return np.asarray(centre_uncertainties, dtype=np.float64) + (
        radius * math.sqrt(max(0.0, yaw_variance))
    )


def registration_radial_uncertainty(
    covariance,
    footprint,
    local_points=None,
    target_from_source_yaw=0.0,
):
    """Return the conservative maximum over radial point uncertainties."""

    return float(
        np.max(
            registration_radial_uncertainties(
                covariance,
                footprint,
                local_points=local_points,
                target_from_source_yaw=target_from_source_yaw,
            )
        )
    )


@dataclass(frozen=True)
class PointLandmark:
    """A named point expressed in the mission-local frame."""

    name: str
    point: tuple
    weight: float = 1.0

    def __post_init__(self):
        name = str(self.name).strip()
        if not name:
            raise ValueError("point landmark name must not be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "point", _point(self.point, "point landmark"))
        object.__setattr__(self, "weight", _positive(self.weight, "landmark weight"))


@dataclass(frozen=True)
class OrientedSegmentLandmark:
    """A directed local segment or line feature.

    ``longitudinal_weight=1`` treats the observed midpoint as corresponding to
    the template midpoint.  ``0`` treats the feature as an unbounded line and
    uses only its direction and normal offset.  Intermediate values are useful
    for a weakly localized, partially visible endpoint pair.
    """

    name: str
    start: tuple
    end: tuple
    weight: float = 1.0
    longitudinal_weight: float = 1.0

    def __post_init__(self):
        name = str(self.name).strip()
        if not name:
            raise ValueError("segment landmark name must not be empty")
        start = _point(self.start, "segment start")
        end = _point(self.end, "segment end")
        if math.hypot(end[0] - start[0], end[1] - start[1]) <= _EPSILON:
            raise ValueError("segment landmark must have positive length")
        longitudinal = _finite(
            self.longitudinal_weight, "segment longitudinal weight"
        )
        if not 0.0 <= longitudinal <= 1.0:
            raise ValueError("segment longitudinal weight must lie in [0, 1]")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "weight", _positive(self.weight, "landmark weight"))
        object.__setattr__(self, "longitudinal_weight", longitudinal)

    @property
    def midpoint(self):
        return (
            0.5 * (self.start[0] + self.end[0]),
            0.5 * (self.start[1] + self.end[1]),
        )

    @property
    def heading(self):
        return math.atan2(
            self.end[1] - self.start[1], self.end[0] - self.start[0]
        )


@dataclass(frozen=True)
class EntryPlane:
    """A directed mission-local entry plane.

    ``normal`` points in the mission travel direction.  Progress is negative
    before the plane, zero on it, and positive after it.
    """

    point: tuple
    normal: tuple

    def __post_init__(self):
        point = _point(self.point, "entry-plane point")
        normal = _point(self.normal, "entry-plane normal")
        length = math.hypot(*normal)
        if length <= _EPSILON:
            raise ValueError("entry-plane normal must have positive length")
        object.__setattr__(self, "point", point)
        object.__setattr__(
            self, "normal", (normal[0] / length, normal[1] / length)
        )

    @property
    def heading(self):
        return math.atan2(self.normal[1], self.normal[0])


@dataclass(frozen=True)
class MissionLocalTemplate:
    """Immutable landmarks and entry geometry for one mission."""

    mission: str
    frame_id: str
    points: tuple = field(default_factory=tuple)
    segments: tuple = field(default_factory=tuple)
    entry_plane: EntryPlane = None

    def __post_init__(self):
        mission = str(self.mission).strip()
        frame_id = str(self.frame_id).strip()
        if not mission or not frame_id:
            raise ValueError("mission and local frame id must not be empty")
        points = tuple(self.points)
        segments = tuple(self.segments)
        if not points and not segments:
            raise ValueError("a mission-local template needs at least one landmark")
        if not all(isinstance(item, PointLandmark) for item in points):
            raise TypeError("template points must be PointLandmark values")
        if not all(isinstance(item, OrientedSegmentLandmark) for item in segments):
            raise TypeError(
                "template segments must be OrientedSegmentLandmark values"
            )
        names = [item.name for item in points] + [item.name for item in segments]
        if len(names) != len(set(names)):
            raise ValueError("mission-local landmark names must be unique")
        if self.entry_plane is not None and not isinstance(
            self.entry_plane, EntryPlane
        ):
            raise TypeError("entry_plane must be an EntryPlane")
        object.__setattr__(self, "mission", mission)
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "segments", segments)

    @property
    def landmark_names(self):
        return tuple(item.name for item in self.points + self.segments)

    @property
    def total_weight(self):
        return sum(item.weight for item in self.points + self.segments)


@dataclass(frozen=True)
class PointObservation:
    """A target-frame observation corresponding to one named point."""

    landmark: str
    point: tuple
    confidence: float = 1.0

    def __post_init__(self):
        landmark = str(self.landmark).strip()
        if not landmark:
            raise ValueError("point observation landmark must not be empty")
        confidence = _finite(self.confidence, "point observation confidence")
        if not 0.0 < confidence <= 1.0:
            raise ValueError("point observation confidence must lie in (0, 1]")
        object.__setattr__(self, "landmark", landmark)
        object.__setattr__(self, "point", _point(self.point, "point observation"))
        object.__setattr__(self, "confidence", confidence)


@dataclass(frozen=True)
class OrientedSegmentObservation:
    """A target-frame directed segment corresponding to one local feature."""

    landmark: str
    start: tuple
    end: tuple
    confidence: float = 1.0

    def __post_init__(self):
        landmark = str(self.landmark).strip()
        if not landmark:
            raise ValueError("segment observation landmark must not be empty")
        start = _point(self.start, "observed segment start")
        end = _point(self.end, "observed segment end")
        if math.hypot(end[0] - start[0], end[1] - start[1]) <= _EPSILON:
            raise ValueError("observed segment must have positive length")
        confidence = _finite(self.confidence, "segment observation confidence")
        if not 0.0 < confidence <= 1.0:
            raise ValueError("segment observation confidence must lie in (0, 1]")
        object.__setattr__(self, "landmark", landmark)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "confidence", confidence)

    @property
    def midpoint(self):
        return (
            0.5 * (self.start[0] + self.end[0]),
            0.5 * (self.start[1] + self.end[1]),
        )

    @property
    def heading(self):
        return math.atan2(
            self.end[1] - self.start[1], self.end[0] - self.start[0]
        )


@dataclass(frozen=True)
class RegistrationObservation:
    """One source-timestamped set of target-frame landmark observations."""

    stamp: float
    target_frame: str = "odom"
    points: tuple = field(default_factory=tuple)
    segments: tuple = field(default_factory=tuple)

    def __post_init__(self):
        stamp = _finite(self.stamp, "registration observation stamp")
        target_frame = str(self.target_frame).strip()
        if not target_frame:
            raise ValueError("registration target frame must not be empty")
        points = tuple(self.points)
        segments = tuple(self.segments)
        if not all(isinstance(item, PointObservation) for item in points):
            raise TypeError("observed points must be PointObservation values")
        if not all(
            isinstance(item, OrientedSegmentObservation) for item in segments
        ):
            raise TypeError(
                "observed segments must be OrientedSegmentObservation values"
            )
        names = [item.landmark for item in points] + [
            item.landmark for item in segments
        ]
        if len(names) != len(set(names)):
            raise ValueError("an observation may contain each landmark only once")
        object.__setattr__(self, "stamp", stamp)
        object.__setattr__(self, "target_frame", target_frame)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "segments", segments)


@dataclass(frozen=True)
class RegistrationConfig:
    """Robust fit, acceptance, and observability thresholds."""

    position_inlier_threshold: float = 0.035
    heading_inlier_threshold: float = math.radians(6.0)
    orientation_scale: float = 0.20
    huber_delta: float = 1.0
    maximum_iterations: int = 25
    convergence_tolerance: float = 1e-9
    minimum_inliers: int = 2
    minimum_template_coverage: float = 0.0
    minimum_inlier_coverage: float = 0.60
    maximum_condition_number: float = 1e8
    covariance_floor: float = 1e-6

    def __post_init__(self):
        for name in (
            "position_inlier_threshold",
            "heading_inlier_threshold",
            "orientation_scale",
            "huber_delta",
            "convergence_tolerance",
            "maximum_condition_number",
            "covariance_floor",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        iterations = int(self.maximum_iterations)
        inliers = int(self.minimum_inliers)
        if iterations < 1 or inliers < 1:
            raise ValueError("registration iterations and inliers must be positive")
        object.__setattr__(self, "maximum_iterations", iterations)
        object.__setattr__(self, "minimum_inliers", inliers)
        for name in ("minimum_template_coverage", "minimum_inlier_coverage"):
            value = _finite(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError("%s must lie in [0, 1]" % name)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class RegistrationDiagnostics:
    """Evidence for accepting or rejecting one fitted transform."""

    reason: str
    degenerate: bool
    total_landmarks: int
    matched_landmarks: int
    inlier_landmarks: int
    point_inliers: int
    segment_inliers: int
    template_coverage: float
    inlier_coverage: float
    spatial_baseline: float
    position_rms: float
    heading_rms: float
    robust_rms: float
    maximum_position_residual: float
    maximum_heading_residual: float
    information_condition: float
    iterations: int


@dataclass(frozen=True)
class RegistrationResult:
    """An immutable local-to-target registration result."""

    accepted: bool
    stamp: float
    transform: RigidTransform2D
    covariance: tuple
    diagnostics: RegistrationDiagnostics
    inlier_landmarks: tuple = field(default_factory=tuple)

    def __post_init__(self):
        object.__setattr__(self, "stamp", _finite(self.stamp, "result stamp"))
        if self.accepted and self.transform is None:
            raise ValueError("an accepted registration needs a transform")
        if self.transform is not None and not isinstance(
            self.transform, RigidTransform2D
        ):
            raise TypeError("registration transform must be RigidTransform2D")
        covariance = tuple(tuple(float(value) for value in row) for row in self.covariance)
        if covariance and (
            len(covariance) != 3
            or any(len(row) != 3 for row in covariance)
            or not all(math.isfinite(value) for row in covariance for value in row)
        ):
            raise ValueError("registration covariance must be an empty or finite 3x3 matrix")
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "inlier_landmarks", tuple(self.inlier_landmarks))


@dataclass(frozen=True)
class EntryPlaneProgress:
    """Robot pose expressed relative to a registered entry plane."""

    local_pose: Pose2D
    longitudinal: float
    lateral: float
    heading_error: float
    crossed: bool


@dataclass(frozen=True)
class TemporalRegistrationConfig:
    required_confirmations: int = 3
    maximum_gap: float = 0.25
    maximum_position_delta: float = 0.025
    maximum_heading_delta: float = math.radians(3.0)
    history_size: int = 8

    def __post_init__(self):
        required = int(self.required_confirmations)
        history_size = int(self.history_size)
        if required < 1 or history_size < required:
            raise ValueError("temporal history must hold all required confirmations")
        object.__setattr__(self, "required_confirmations", required)
        object.__setattr__(self, "history_size", history_size)
        for name in (
            "maximum_gap",
            "maximum_position_delta",
            "maximum_heading_delta",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))


@dataclass(frozen=True)
class TemporalRegistrationState:
    confirmed: bool
    confirmation_count: int
    transform: RigidTransform2D
    covariance: tuple
    first_stamp: float
    last_stamp: float
    maximum_position_spread: float
    maximum_heading_spread: float
    reason: str = ""


@dataclass(frozen=True)
class _MatchedPoint:
    name: str
    local: tuple
    target: tuple
    weight: float


@dataclass(frozen=True)
class _MatchedSegment:
    name: str
    local_midpoint: tuple
    local_heading: float
    target_midpoint: tuple
    target_heading: float
    weight: float
    longitudinal_weight: float


def _match(template, observation):
    observed_points = {item.landmark: item for item in observation.points}
    observed_segments = {item.landmark: item for item in observation.segments}
    points = []
    segments = []
    for landmark in template.points:
        value = observed_points.get(landmark.name)
        if value is not None:
            points.append(
                _MatchedPoint(
                    landmark.name,
                    landmark.point,
                    value.point,
                    landmark.weight * value.confidence,
                )
            )
    for landmark in template.segments:
        value = observed_segments.get(landmark.name)
        if value is not None:
            segments.append(
                _MatchedSegment(
                    landmark.name,
                    landmark.midpoint,
                    landmark.heading,
                    value.midpoint,
                    value.heading,
                    landmark.weight * value.confidence,
                    landmark.longitudinal_weight,
                )
            )
    return tuple(points), tuple(segments)


def _rotate(point, yaw):
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        cosine * point[0] - sine * point[1],
        sine * point[0] + cosine * point[1],
    )


def _transform_point(parameters, point):
    rotated = _rotate(point, parameters[2])
    return parameters[0] + rotated[0], parameters[1] + rotated[1]


def _point_theta_derivative(point, yaw):
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        -sine * point[0] - cosine * point[1],
        cosine * point[0] - sine * point[1],
    )


def _feature_residuals(parameters, points, segments):
    result = {}
    for feature in points:
        predicted = _transform_point(parameters, feature.local)
        position = math.hypot(
            predicted[0] - feature.target[0], predicted[1] - feature.target[1]
        )
        result[feature.name] = (position, 0.0, feature.weight, "point")
    for feature in segments:
        predicted = _transform_point(parameters, feature.local_midpoint)
        target_tangent = (
            math.cos(feature.target_heading),
            math.sin(feature.target_heading),
        )
        target_normal = (-target_tangent[1], target_tangent[0])
        delta = (
            predicted[0] - feature.target_midpoint[0],
            predicted[1] - feature.target_midpoint[1],
        )
        normal = target_normal[0] * delta[0] + target_normal[1] * delta[1]
        tangent = target_tangent[0] * delta[0] + target_tangent[1] * delta[1]
        position = math.hypot(normal, feature.longitudinal_weight * tangent)
        heading = abs(
            normalize_angle(
                parameters[2] + feature.local_heading - feature.target_heading
            )
        )
        result[feature.name] = (position, heading, feature.weight, "segment")
    return result


def _inlier_names(parameters, points, segments, config):
    residuals = _feature_residuals(parameters, points, segments)
    names = []
    for name, (position, heading, _weight, kind) in residuals.items():
        if position > config.position_inlier_threshold:
            continue
        if kind == "segment" and heading > config.heading_inlier_threshold:
            continue
        names.append(name)
    return tuple(names), residuals


def _translation_for_yaw(yaw, points, segments, allowed_names=None):
    rows = []
    values = []
    weights = []
    for feature in points:
        if allowed_names is not None and feature.name not in allowed_names:
            continue
        rotated = _rotate(feature.local, yaw)
        rows.extend(((1.0, 0.0), (0.0, 1.0)))
        values.extend(
            (feature.target[0] - rotated[0], feature.target[1] - rotated[1])
        )
        weights.extend((feature.weight, feature.weight))
    for feature in segments:
        if allowed_names is not None and feature.name not in allowed_names:
            continue
        rotated = _rotate(feature.local_midpoint, yaw)
        tangent = (
            math.cos(feature.target_heading),
            math.sin(feature.target_heading),
        )
        normal = (-tangent[1], tangent[0])
        offset = (
            feature.target_midpoint[0] - rotated[0],
            feature.target_midpoint[1] - rotated[1],
        )
        rows.append(normal)
        values.append(normal[0] * offset[0] + normal[1] * offset[1])
        weights.append(feature.weight)
        if feature.longitudinal_weight > _EPSILON:
            rows.append(tangent)
            values.append(tangent[0] * offset[0] + tangent[1] * offset[1])
            weights.append(feature.weight * feature.longitudinal_weight)
    if not rows:
        return None
    matrix = np.asarray(rows, dtype=np.float64)
    value = np.asarray(values, dtype=np.float64)
    sqrt_weight = np.sqrt(np.asarray(weights, dtype=np.float64))
    weighted = matrix * sqrt_weight[:, None]
    if np.linalg.matrix_rank(weighted, tol=1e-10) < 2:
        return None
    translation, _residuals, _rank, _singular = np.linalg.lstsq(
        weighted, value * sqrt_weight, rcond=None
    )
    return np.asarray((translation[0], translation[1], yaw), dtype=np.float64)


def _procrustes_seed(points, segments):
    local = []
    target = []
    weights = []
    for feature in points:
        local.append(feature.local)
        target.append(feature.target)
        weights.append(feature.weight)
    for feature in segments:
        if feature.longitudinal_weight > _EPSILON:
            local.append(feature.local_midpoint)
            target.append(feature.target_midpoint)
            weights.append(feature.weight * feature.longitudinal_weight)
    if len(local) < 2:
        return None
    local = np.asarray(local, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    local_mean = np.sum(local * weights[:, None], axis=0) / total
    target_mean = np.sum(target * weights[:, None], axis=0) / total
    local_centered = local - local_mean
    target_centered = target - target_mean
    if (
        float(np.sum(weights * np.sum(local_centered ** 2, axis=1)))
        <= _EPSILON
        or float(np.sum(weights * np.sum(target_centered ** 2, axis=1)))
        <= _EPSILON
    ):
        return None
    cross = local_centered.T @ (weights[:, None] * target_centered)
    yaw = math.atan2(cross[0, 1] - cross[1, 0], cross[0, 0] + cross[1, 1])
    rotated_mean = _rotate(local_mean, yaw)
    return np.asarray(
        (
            target_mean[0] - rotated_mean[0],
            target_mean[1] - rotated_mean[1],
            normalize_angle(yaw),
        ),
        dtype=np.float64,
    )


def _candidate_seeds(points, segments):
    seeds = []
    procrustes = _procrustes_seed(points, segments)
    if procrustes is not None:
        seeds.append(procrustes)

    # Directed segments give a direct yaw observation.  Translation is solved
    # from every available point/line constraint so an unbounded single line
    # cannot invent motion along its tangent.
    yaw_candidates = [
        normalize_angle(feature.target_heading - feature.local_heading)
        for feature in segments
    ]
    if yaw_candidates:
        sine = sum(math.sin(value) for value in yaw_candidates)
        cosine = sum(math.cos(value) for value in yaw_candidates)
        if math.hypot(sine, cosine) > _EPSILON:
            yaw_candidates.append(math.atan2(sine, cosine))
    for yaw in yaw_candidates:
        seed = _translation_for_yaw(yaw, points, segments)
        if seed is not None:
            seeds.append(seed)
        # Deterministic minimal-set seeds make line registration robust to a
        # mismatched wall.  One point (or one bounded segment midpoint) fixes
        # translation for a directed-yaw hypothesis; two non-parallel
        # unbounded lines do so through their independent normals.
        translation_features = list(points) + list(segments)
        for first_index, first in enumerate(translation_features):
            subsets = ((first.name,),)
            for second in translation_features[first_index + 1 :]:
                subsets += ((first.name, second.name),)
            for names in subsets:
                subset_seed = _translation_for_yaw(
                    yaw, points, segments, allowed_names=frozenset(names)
                )
                if subset_seed is not None:
                    seeds.append(subset_seed)

    anchors = [
        (feature.local, feature.target, feature.weight) for feature in points
    ] + [
        (
            feature.local_midpoint,
            feature.target_midpoint,
            feature.weight * feature.longitudinal_weight,
        )
        for feature in segments
        if feature.longitudinal_weight > _EPSILON
    ]
    for first_index, first in enumerate(anchors):
        for second in anchors[first_index + 1 :]:
            local_delta = (
                second[0][0] - first[0][0], second[0][1] - first[0][1]
            )
            target_delta = (
                second[1][0] - first[1][0], second[1][1] - first[1][1]
            )
            if min(math.hypot(*local_delta), math.hypot(*target_delta)) <= 1e-6:
                continue
            yaw = normalize_angle(
                math.atan2(target_delta[1], target_delta[0])
                - math.atan2(local_delta[1], local_delta[0])
            )
            rotated = _rotate(first[0], yaw)
            seeds.append(
                np.asarray(
                    (
                        first[1][0] - rotated[0],
                        first[1][1] - rotated[1],
                        yaw,
                    ),
                    dtype=np.float64,
                )
            )
    unique = []
    for seed in seeds:
        if any(
            math.hypot(seed[0] - other[0], seed[1] - other[1]) <= 1e-9
            and abs(normalize_angle(seed[2] - other[2])) <= 1e-9
            for other in unique
        ):
            continue
        unique.append(seed)
    return tuple(unique)


def _normalized_feature_error(residual, config):
    position, heading, _weight, kind = residual
    squared = (position / config.position_inlier_threshold) ** 2
    if kind == "segment":
        squared += (heading / config.heading_inlier_threshold) ** 2
    return math.sqrt(squared)


def _select_seed(seeds, points, segments, config):
    best = None
    for seed in seeds:
        inliers, residuals = _inlier_names(seed, points, segments, config)
        inlier_set = set(inliers)
        inlier_weight = sum(
            residual[2] for name, residual in residuals.items() if name in inlier_set
        )
        robust_cost = 0.0
        for residual in residuals.values():
            normalized = _normalized_feature_error(residual, config)
            if normalized <= config.huber_delta:
                loss = normalized ** 2
            else:
                loss = (
                    2.0 * config.huber_delta * normalized
                    - config.huber_delta ** 2
                )
            robust_cost += residual[2] * loss
        score = (-inlier_weight, robust_cost)
        if best is None or score < best[0]:
            best = (score, seed.copy())
    return None if best is None else best[1]


def _linearized_system(parameters, points, segments, config, allowed_names=None):
    rows = []
    values = []
    weights = []
    feature_rows = []
    yaw = float(parameters[2])
    for feature in points:
        if allowed_names is not None and feature.name not in allowed_names:
            continue
        predicted = _transform_point(parameters, feature.local)
        derivative = _point_theta_derivative(feature.local, yaw)
        start = len(rows)
        rows.extend(((1.0, 0.0, derivative[0]), (0.0, 1.0, derivative[1])))
        values.extend(
            (predicted[0] - feature.target[0], predicted[1] - feature.target[1])
        )
        weights.extend((feature.weight, feature.weight))
        feature_rows.append((feature.name, start, len(rows)))
    for feature in segments:
        if allowed_names is not None and feature.name not in allowed_names:
            continue
        predicted = _transform_point(parameters, feature.local_midpoint)
        derivative = _point_theta_derivative(feature.local_midpoint, yaw)
        tangent = (
            math.cos(feature.target_heading),
            math.sin(feature.target_heading),
        )
        normal = (-tangent[1], tangent[0])
        delta = (
            predicted[0] - feature.target_midpoint[0],
            predicted[1] - feature.target_midpoint[1],
        )
        start = len(rows)
        rows.append(
            (
                normal[0],
                normal[1],
                normal[0] * derivative[0] + normal[1] * derivative[1],
            )
        )
        values.append(normal[0] * delta[0] + normal[1] * delta[1])
        weights.append(feature.weight)
        if feature.longitudinal_weight > _EPSILON:
            scale = math.sqrt(feature.longitudinal_weight)
            rows.append(
                (
                    scale * tangent[0],
                    scale * tangent[1],
                    scale
                    * (tangent[0] * derivative[0] + tangent[1] * derivative[1]),
                )
            )
            values.append(
                scale * (tangent[0] * delta[0] + tangent[1] * delta[1])
            )
            weights.append(feature.weight)
        rows.append((0.0, 0.0, config.orientation_scale))
        values.append(
            config.orientation_scale
            * normalize_angle(
                yaw + feature.local_heading - feature.target_heading
            )
        )
        weights.append(feature.weight)
        feature_rows.append((feature.name, start, len(rows)))
    return (
        np.asarray(rows, dtype=np.float64).reshape((-1, 3)),
        np.asarray(values, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        tuple(feature_rows),
    )


def _refine(seed, points, segments, config, allowed_names=None):
    parameters = seed.copy()
    iterations = 0
    information = np.zeros((3, 3), dtype=np.float64)
    for iterations in range(1, config.maximum_iterations + 1):
        rows, values, base_weights, feature_rows = _linearized_system(
            parameters, points, segments, config, allowed_names
        )
        if rows.shape[0] < 3:
            return None, iterations, information, math.inf, "insufficient constraints"
        robust_weights = np.ones(values.shape[0], dtype=np.float64)
        for _name, start, end in feature_rows:
            magnitude = float(np.linalg.norm(values[start:end]))
            normalized = magnitude / max(
                config.position_inlier_threshold, _EPSILON
            )
            if normalized > config.huber_delta:
                robust_weights[start:end] = config.huber_delta / normalized
        combined = base_weights * robust_weights
        sqrt_weight = np.sqrt(combined)
        weighted_rows = rows * sqrt_weight[:, None]
        weighted_values = values * sqrt_weight
        information = weighted_rows.T @ weighted_rows
        rank = int(np.linalg.matrix_rank(information, tol=1e-10))
        if rank < 3:
            return None, iterations, information, math.inf, "rank-deficient geometry"
        condition = float(np.linalg.cond(information))
        if not math.isfinite(condition) or condition > config.maximum_condition_number:
            return None, iterations, information, condition, "ill-conditioned geometry"
        try:
            delta = -np.linalg.solve(information, weighted_rows.T @ weighted_values)
        except np.linalg.LinAlgError:
            return None, iterations, information, condition, "singular information matrix"
        parameters += delta
        parameters[2] = normalize_angle(parameters[2])
        if float(np.linalg.norm(delta)) <= config.convergence_tolerance:
            return parameters, iterations, information, condition, ""
    condition = float(np.linalg.cond(information))
    return parameters, iterations, information, condition, ""


def _spatial_baseline(points, segments):
    anchors = [feature.local for feature in points]
    anchors.extend(feature.local_midpoint for feature in segments)
    if len(anchors) < 2:
        return 0.0
    points_array = np.asarray(anchors, dtype=np.float64)
    delta = points_array[:, None, :] - points_array[None, :, :]
    return float(np.max(np.hypot(delta[:, :, 0], delta[:, :, 1])))


def _rejection(
    template,
    observation,
    reason,
    matched,
    template_coverage,
    degenerate=False,
    transform=None,
    condition=math.inf,
    iterations=0,
):
    diagnostics = RegistrationDiagnostics(
        reason=reason,
        degenerate=bool(degenerate),
        total_landmarks=len(template.points) + len(template.segments),
        matched_landmarks=int(matched),
        inlier_landmarks=0,
        point_inliers=0,
        segment_inliers=0,
        template_coverage=float(template_coverage),
        inlier_coverage=0.0,
        spatial_baseline=0.0,
        position_rms=math.inf,
        heading_rms=math.inf,
        robust_rms=math.inf,
        maximum_position_residual=math.inf,
        maximum_heading_residual=math.inf,
        information_condition=float(condition),
        iterations=int(iterations),
    )
    return RegistrationResult(
        accepted=False,
        stamp=observation.stamp,
        transform=transform,
        covariance=tuple(),
        diagnostics=diagnostics,
    )


def register_point_with_heading(
    stamp,
    source_frame,
    target_frame,
    local_point,
    target_point,
    target_from_source_yaw,
    position_standard_deviation,
    heading_standard_deviation,
):
    """Register one local point under an independent heading observation.

    This is the honest full-rank primitive for a point landmark paired with a
    heading prior such as AMCL.  It avoids manufacturing a second positional
    landmark from the first point merely to pass a generic observability test.
    The returned covariance propagates both source measurements into the
    translation of the local-frame origin.
    """

    stamp = _finite(stamp, "registration stamp")
    source_frame = str(source_frame).strip()
    target_frame = str(target_frame).strip()
    if not source_frame or not target_frame:
        raise ValueError("registration frame ids must not be empty")
    local_point = _point(local_point, "local registration point")
    target_point = _point(target_point, "target registration point")
    yaw = normalize_angle(_finite(target_from_source_yaw, "registration heading"))
    position_sigma = _positive(
        position_standard_deviation,
        "position standard deviation",
    )
    heading_sigma = _positive(
        heading_standard_deviation,
        "heading standard deviation",
    )

    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotated_x = cosine * local_point[0] - sine * local_point[1]
    rotated_y = sine * local_point[0] + cosine * local_point[1]
    transform = RigidTransform2D(
        target_point[0] - rotated_x,
        target_point[1] - rotated_y,
        yaw,
        source_frame=source_frame,
        target_frame=target_frame,
    )

    # t = q - R(theta) p.  Propagate isotropic point error and heading error
    # through dt/dtheta, retaining the translation/yaw cross covariance.
    derivative_x = sine * local_point[0] + cosine * local_point[1]
    derivative_y = -cosine * local_point[0] + sine * local_point[1]
    jacobian = np.asarray(
        (
            (1.0, 0.0, derivative_x),
            (0.0, 1.0, derivative_y),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    measurement_covariance = np.diag(
        (position_sigma ** 2, position_sigma ** 2, heading_sigma ** 2)
    )
    covariance = jacobian @ measurement_covariance @ jacobian.T
    diagnostics = RegistrationDiagnostics(
        reason="",
        degenerate=False,
        total_landmarks=2,
        matched_landmarks=2,
        inlier_landmarks=2,
        point_inliers=1,
        segment_inliers=1,
        template_coverage=1.0,
        inlier_coverage=1.0,
        spatial_baseline=math.hypot(*local_point),
        position_rms=0.0,
        heading_rms=0.0,
        robust_rms=0.0,
        maximum_position_residual=0.0,
        maximum_heading_residual=0.0,
        information_condition=1.0,
        iterations=0,
    )
    return RegistrationResult(
        accepted=True,
        stamp=stamp,
        transform=transform,
        covariance=_covariance_tuple(covariance),
        diagnostics=diagnostics,
        inlier_landmarks=("point", "heading"),
    )


def registration_covariance_with_floor(
    covariance,
    position_standard_deviation,
    heading_standard_deviation,
):
    """Add non-averaging systematic registration uncertainty."""

    position_sigma = _positive(
        position_standard_deviation,
        "systematic position standard deviation",
        allow_zero=True,
    )
    heading_sigma = _positive(
        heading_standard_deviation,
        "systematic heading standard deviation",
        allow_zero=True,
    )
    if covariance is not None and np.asarray(covariance).size:
        matrix = np.asarray(covariance, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("registration covariance must be a finite 3x3 matrix")
        matrix = 0.5 * (matrix + matrix.T)
        if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-12:
            raise ValueError("registration covariance must be positive semidefinite")
    else:
        matrix = np.zeros((3, 3), dtype=np.float64)
    matrix += np.diag(
        (position_sigma ** 2, position_sigma ** 2, heading_sigma ** 2)
    )
    return _covariance_tuple(matrix)


def estimate_local_registration(
    template, observation, config=None, initial_transform=None
):
    """Estimate a robust mission-local to target-frame rigid transform.

    Correspondences are explicit through landmark names.  Unknown observations
    are ignored; a known landmark with the wrong observation type is not used.
    The returned result always explains rejection through ``diagnostics``.
    """

    if not isinstance(template, MissionLocalTemplate):
        raise TypeError("template must be MissionLocalTemplate")
    if not isinstance(observation, RegistrationObservation):
        raise TypeError("observation must be RegistrationObservation")
    config = RegistrationConfig() if config is None else config
    if not isinstance(config, RegistrationConfig):
        raise TypeError("config must be RegistrationConfig")
    if initial_transform is not None:
        if not isinstance(initial_transform, RigidTransform2D):
            raise TypeError("initial_transform must be RigidTransform2D")
        if (
            initial_transform.source_frame != template.frame_id
            or initial_transform.target_frame != observation.target_frame
        ):
            raise ValueError("initial registration transform frames do not match")

    points, segments = _match(template, observation)
    matched_weight = sum(item.weight for item in points + segments)
    template_coverage = matched_weight / max(template.total_weight, _EPSILON)
    matched_count = len(points) + len(segments)
    if matched_count == 0:
        return _rejection(
            template, observation, "no corresponding landmarks", 0, 0.0
        )
    if template_coverage + _EPSILON < config.minimum_template_coverage:
        return _rejection(
            template,
            observation,
            "insufficient template coverage",
            matched_count,
            template_coverage,
        )

    seeds = (
        (
            np.asarray(
                (
                    initial_transform.target_from_source_x,
                    initial_transform.target_from_source_y,
                    initial_transform.target_from_source_yaw,
                ),
                dtype=np.float64,
            ),
        )
        if initial_transform is not None
        else _candidate_seeds(points, segments)
    )
    if not seeds:
        return _rejection(
            template,
            observation,
            "unobservable rigid transform",
            matched_count,
            template_coverage,
            degenerate=True,
        )
    seed = _select_seed(seeds, points, segments, config)
    seed_inliers, _seed_residuals = _inlier_names(
        seed, points, segments, config
    )
    seed_allowed = (
        frozenset(seed_inliers)
        if len(seed_inliers) >= config.minimum_inliers
        else None
    )
    parameters, iterations, information, condition, problem = _refine(
        seed, points, segments, config, allowed_names=seed_allowed
    )
    if parameters is None:
        return _rejection(
            template,
            observation,
            problem,
            matched_count,
            template_coverage,
            degenerate=True,
            condition=condition,
            iterations=iterations,
        )

    inlier_names, residuals = _inlier_names(parameters, points, segments, config)
    inlier_set = set(inlier_names)
    if len(inlier_names) < config.minimum_inliers:
        transform = RigidTransform2D(
            float(parameters[0]),
            float(parameters[1]),
            float(parameters[2]),
            template.frame_id,
            observation.target_frame,
        )
        return _rejection(
            template,
            observation,
            "insufficient inliers",
            matched_count,
            template_coverage,
            transform=transform,
            condition=condition,
            iterations=iterations,
        )
    inlier_weight = sum(residuals[name][2] for name in inlier_names)
    inlier_coverage = inlier_weight / max(matched_weight, _EPSILON)
    if inlier_coverage + _EPSILON < config.minimum_inlier_coverage:
        transform = RigidTransform2D(
            float(parameters[0]),
            float(parameters[1]),
            float(parameters[2]),
            template.frame_id,
            observation.target_frame,
        )
        return _rejection(
            template,
            observation,
            "insufficient inlier coverage",
            matched_count,
            template_coverage,
            transform=transform,
            condition=condition,
            iterations=iterations,
        )

    refined, extra_iterations, information, condition, problem = _refine(
        parameters, points, segments, config, allowed_names=inlier_set
    )
    iterations += extra_iterations
    if refined is None:
        return _rejection(
            template,
            observation,
            problem,
            matched_count,
            template_coverage,
            degenerate=True,
            condition=condition,
            iterations=iterations,
        )
    parameters = refined
    inlier_names, residuals = _inlier_names(parameters, points, segments, config)
    inlier_set = set(inlier_names)
    if len(inlier_names) < config.minimum_inliers:
        return _rejection(
            template,
            observation,
            "refinement lost required inliers",
            matched_count,
            template_coverage,
            condition=condition,
            iterations=iterations,
        )
    refined_inlier_weight = sum(residuals[name][2] for name in inlier_names)
    if (
        refined_inlier_weight / max(matched_weight, _EPSILON) + _EPSILON
        < config.minimum_inlier_coverage
    ):
        return _rejection(
            template,
            observation,
            "refinement lost required inlier coverage",
            matched_count,
            template_coverage,
            transform=RigidTransform2D(
                float(parameters[0]),
                float(parameters[1]),
                float(parameters[2]),
                template.frame_id,
                observation.target_frame,
            ),
            condition=condition,
            iterations=iterations,
        )

    rows, values, base_weights, _feature_rows = _linearized_system(
        parameters, points, segments, config, allowed_names=inlier_set
    )
    weighted_rows = rows * np.sqrt(base_weights)[:, None]
    information = weighted_rows.T @ weighted_rows
    rank = int(np.linalg.matrix_rank(information, tol=1e-10))
    condition = float(np.linalg.cond(information)) if rank == 3 else math.inf
    if (
        rank < 3
        or not math.isfinite(condition)
        or condition > config.maximum_condition_number
    ):
        return _rejection(
            template,
            observation,
            "inlier geometry is degenerate",
            matched_count,
            template_coverage,
            degenerate=True,
            condition=condition,
            iterations=iterations,
        )

    weighted_squared_error = float(np.sum(base_weights * values ** 2))
    degrees_of_freedom = max(1, int(values.size) - 3)
    variance = max(
        config.covariance_floor ** 2,
        weighted_squared_error / degrees_of_freedom,
    )
    try:
        covariance = variance * np.linalg.inv(information)
    except np.linalg.LinAlgError:
        covariance = np.full((3, 3), math.nan, dtype=np.float64)

    positions = [residuals[name][0] for name in inlier_names]
    headings = [
        residuals[name][1]
        for name in inlier_names
        if residuals[name][3] == "segment"
    ]
    weights = [residuals[name][2] for name in inlier_names]
    position_rms = math.sqrt(
        sum(weight * value * value for weight, value in zip(weights, positions))
        / max(sum(weights), _EPSILON)
    )
    segment_weights = [
        residuals[name][2]
        for name in inlier_names
        if residuals[name][3] == "segment"
    ]
    heading_rms = (
        math.sqrt(
            sum(
                weight * value * value
                for weight, value in zip(segment_weights, headings)
            )
            / max(sum(segment_weights), _EPSILON)
        )
        if headings
        else 0.0
    )
    normalized = [
        _normalized_feature_error(residuals[name], config) for name in inlier_names
    ]
    robust_rms = math.sqrt(
        sum(weight * value * value for weight, value in zip(weights, normalized))
        / max(sum(weights), _EPSILON)
    )
    point_names = {feature.name for feature in points}
    segment_names = {feature.name for feature in segments}
    inlier_weight = sum(residuals[name][2] for name in inlier_names)
    diagnostics = RegistrationDiagnostics(
        reason="",
        degenerate=False,
        total_landmarks=len(template.points) + len(template.segments),
        matched_landmarks=matched_count,
        inlier_landmarks=len(inlier_names),
        point_inliers=sum(name in point_names for name in inlier_names),
        segment_inliers=sum(name in segment_names for name in inlier_names),
        template_coverage=template_coverage,
        inlier_coverage=inlier_weight / max(matched_weight, _EPSILON),
        spatial_baseline=_spatial_baseline(
            tuple(feature for feature in points if feature.name in inlier_set),
            tuple(feature for feature in segments if feature.name in inlier_set),
        ),
        position_rms=position_rms,
        heading_rms=heading_rms,
        robust_rms=robust_rms,
        maximum_position_residual=max(positions),
        maximum_heading_residual=max(headings) if headings else 0.0,
        information_condition=condition,
        iterations=iterations,
    )
    transform = RigidTransform2D(
        float(parameters[0]),
        float(parameters[1]),
        float(parameters[2]),
        template.frame_id,
        observation.target_frame,
    )
    return RegistrationResult(
        accepted=True,
        stamp=observation.stamp,
        transform=transform,
        covariance=_covariance_tuple(covariance),
        diagnostics=diagnostics,
        inlier_landmarks=tuple(sorted(inlier_names)),
    )


def entry_plane_progress(entry_plane, target_pose, local_to_target):
    """Express a target-frame robot pose relative to a local entry plane."""

    if not isinstance(entry_plane, EntryPlane):
        raise TypeError("entry_plane must be EntryPlane")
    if not isinstance(local_to_target, RigidTransform2D):
        raise TypeError("local_to_target must be RigidTransform2D")
    local_pose = local_to_target.inverse().apply_pose(target_pose)
    delta = (
        local_pose.x - entry_plane.point[0],
        local_pose.y - entry_plane.point[1],
    )
    normal = entry_plane.normal
    tangent = (-normal[1], normal[0])
    longitudinal = normal[0] * delta[0] + normal[1] * delta[1]
    lateral = tangent[0] * delta[0] + tangent[1] * delta[1]
    return EntryPlaneProgress(
        local_pose=local_pose,
        longitudinal=float(longitudinal),
        lateral=float(lateral),
        heading_error=normalize_angle(local_pose.yaw - entry_plane.heading),
        crossed=bool(longitudinal >= 0.0),
    )


class TemporalRegistrationFilter:
    """Confirm independent, time-ordered registration results.

    A rejected result or an excessive source-time gap clears the streak.  An
    inconsistent accepted transform starts a new streak with that transform,
    which avoids mixing two competing local-frame hypotheses.
    """

    def __init__(self, config=None):
        self.config = (
            TemporalRegistrationConfig() if config is None else config
        )
        if not isinstance(self.config, TemporalRegistrationConfig):
            raise TypeError("config must be TemporalRegistrationConfig")
        self._history = deque(maxlen=self.config.history_size)
        self._last_reason = ""

    def reset(self, reason="reset"):
        self._history.clear()
        self._last_reason = str(reason)
        return self.state()

    @staticmethod
    def _mean_transform(results):
        transforms = [result.transform for result in results]
        x = sum(value.target_from_source_x for value in transforms) / len(transforms)
        y = sum(value.target_from_source_y for value in transforms) / len(transforms)
        sine = sum(
            math.sin(value.target_from_source_yaw) for value in transforms
        )
        cosine = sum(
            math.cos(value.target_from_source_yaw) for value in transforms
        )
        yaw = math.atan2(sine, cosine)
        first = transforms[0]
        return RigidTransform2D(
            x,
            y,
            yaw,
            first.source_frame,
            first.target_frame,
        )

    def _spreads(self):
        if not self._history:
            return 0.0, 0.0
        transforms = [result.transform for result in self._history]
        position = 0.0
        heading = 0.0
        for index, first in enumerate(transforms):
            for second in transforms[index + 1 :]:
                position = max(
                    position,
                    math.hypot(
                        first.target_from_source_x
                        - second.target_from_source_x,
                        first.target_from_source_y
                        - second.target_from_source_y,
                    ),
                )
                heading = max(
                    heading,
                    abs(
                        normalize_angle(
                            first.target_from_source_yaw
                            - second.target_from_source_yaw
                        )
                    ),
                )
        return position, heading

    def update(self, result):
        if not isinstance(result, RegistrationResult):
            raise TypeError("result must be RegistrationResult")
        if not result.accepted or result.transform is None:
            return self.reset(result.diagnostics.reason or "rejected registration")
        if self._history:
            latest = self._history[-1]
            if result.stamp <= latest.stamp:
                self._last_reason = "non-increasing source stamp"
                return self.state()
            if result.stamp - latest.stamp > self.config.maximum_gap:
                self._history.clear()
                self._last_reason = "confirmation gap"
            elif (
                result.transform.source_frame
                != latest.transform.source_frame
                or result.transform.target_frame
                != latest.transform.target_frame
            ):
                self._history.clear()
                self._last_reason = "registration frame changed"
        self._history.append(result)
        position_spread, heading_spread = self._spreads()
        if (
            position_spread > self.config.maximum_position_delta
            or heading_spread > self.config.maximum_heading_delta
        ):
            self._history.clear()
            self._history.append(result)
            self._last_reason = "inconsistent transform"
        else:
            self._last_reason = ""
        return self.state()

    def state(self):
        if not self._history:
            return TemporalRegistrationState(
                confirmed=False,
                confirmation_count=0,
                transform=None,
                covariance=tuple(),
                first_stamp=math.nan,
                last_stamp=math.nan,
                maximum_position_spread=0.0,
                maximum_heading_spread=0.0,
                reason=self._last_reason,
            )
        transform = self._mean_transform(self._history)
        position_spread, heading_spread = self._spreads()
        vectors = np.asarray(
            [
                (
                    result.transform.target_from_source_x,
                    result.transform.target_from_source_y,
                    normalize_angle(
                        result.transform.target_from_source_yaw
                        - transform.target_from_source_yaw
                    ),
                )
                for result in self._history
            ],
            dtype=np.float64,
        )
        if len(vectors) > 1:
            # The published transform is the arithmetic/circular mean of the
            # accepted source observations.  Independent transform scatter is
            # therefore a covariance of that mean, not of one observation.
            temporal_covariance = np.cov(
                vectors, rowvar=False, ddof=1
            ) / float(len(vectors))
        else:
            temporal_covariance = np.zeros((3, 3), dtype=np.float64)
        reported_covariances = [
            np.asarray(result.covariance, dtype=np.float64)
            for result in self._history
            if result.covariance
        ]
        if reported_covariances:
            temporal_covariance += sum(reported_covariances) / (
                len(vectors) * len(vectors)
            )
        return TemporalRegistrationState(
            confirmed=len(self._history) >= self.config.required_confirmations,
            confirmation_count=len(self._history),
            transform=transform,
            covariance=_covariance_tuple(temporal_covariance),
            first_stamp=self._history[0].stamp,
            last_stamp=self._history[-1].stamp,
            maximum_position_spread=position_spread,
            maximum_heading_spread=heading_spread,
            reason=self._last_reason,
        )


@dataclass(frozen=True)
class LocalCurveTemplate:
    """An ordered mission-local centreline or boundary curve."""

    name: str
    frame_id: str
    points: tuple
    station: tuple = field(init=False)

    @classmethod
    def from_xy(cls, name, frame_id, x, y):
        x = tuple(x)
        y = tuple(y)
        if len(x) != len(y):
            raise ValueError("curve x and y arrays must have equal length")
        return cls(name, frame_id, tuple(zip(x, y)))

    def __post_init__(self):
        name = str(self.name).strip()
        frame_id = str(self.frame_id).strip()
        points = tuple(_point(value, "curve template point") for value in self.points)
        if not name or not frame_id:
            raise ValueError("curve template name and frame id must not be empty")
        if len(points) < 3:
            raise ValueError("a curve template needs at least three points")
        station = _polyline_station(points)
        if station[-1] <= _EPSILON:
            raise ValueError("curve template must have positive length")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "station", tuple(float(value) for value in station))

    @property
    def length(self):
        return self.station[-1]


@dataclass(frozen=True)
class ObservedCurve:
    """An ordered curve subset in a live target frame."""

    stamp: float
    target_frame: str
    points: tuple
    station: tuple = field(init=False)

    @classmethod
    def from_xy(cls, stamp, target_frame, x, y):
        x = tuple(x)
        y = tuple(y)
        if len(x) != len(y):
            raise ValueError("observed curve x and y arrays must have equal length")
        return cls(stamp, target_frame, tuple(zip(x, y)))

    def __post_init__(self):
        stamp = _finite(self.stamp, "observed curve stamp")
        target_frame = str(self.target_frame).strip()
        points = tuple(_point(value, "observed curve point") for value in self.points)
        if not target_frame:
            raise ValueError("observed curve target frame must not be empty")
        if len(points) < 3:
            raise ValueError("an observed curve needs at least three points")
        station = _polyline_station(points)
        if station[-1] <= _EPSILON:
            raise ValueError("observed curve must have positive length")
        object.__setattr__(self, "stamp", stamp)
        object.__setattr__(self, "target_frame", target_frame)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "station", tuple(float(value) for value in station))

    @property
    def length(self):
        return self.station[-1]


@dataclass(frozen=True)
class CurveRegistrationConfig:
    """Station search and shape-observability thresholds for a path subset."""

    sample_count: int = 21
    station_search_step: float = 0.01
    minimum_observed_length: float = 0.15
    minimum_heading_variation: float = math.radians(8.0)
    minimum_lateral_excitation: float = 0.008
    position_inlier_threshold: float = 0.025
    minimum_inlier_fraction: float = 0.75
    maximum_rms: float = 0.018
    ambiguity_station_separation: float = 0.08
    maximum_ambiguity_rms_difference: float = 0.002
    maximum_ambiguity_rms_ratio: float = 1.15
    maximum_condition_number: float = 1e8

    def __post_init__(self):
        sample_count = int(self.sample_count)
        if sample_count < 5:
            raise ValueError("curve registration needs at least five samples")
        object.__setattr__(self, "sample_count", sample_count)
        for name in (
            "station_search_step",
            "minimum_observed_length",
            "minimum_heading_variation",
            "minimum_lateral_excitation",
            "position_inlier_threshold",
            "maximum_rms",
            "ambiguity_station_separation",
            "maximum_ambiguity_rms_difference",
            "maximum_ambiguity_rms_ratio",
            "maximum_condition_number",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        fraction = _finite(self.minimum_inlier_fraction, "minimum_inlier_fraction")
        if not 0.0 < fraction <= 1.0:
            raise ValueError("minimum_inlier_fraction must lie in (0, 1]")
        object.__setattr__(self, "minimum_inlier_fraction", fraction)
        if self.maximum_ambiguity_rms_ratio < 1.0:
            raise ValueError("maximum_ambiguity_rms_ratio must be at least one")


@dataclass(frozen=True)
class CurveRegistrationDiagnostics:
    reason: str
    degenerate: bool
    template_length: float
    observed_length: float
    observed_heading_variation: float
    observed_lateral_excitation: float
    candidate_count: int
    inlier_fraction: float
    best_rms: float
    second_best_rms: float
    station_margin: float


@dataclass(frozen=True)
class CurveRegistrationResult:
    """Best continuous template-window match for one observed curve subset."""

    accepted: bool
    stamp: float
    transform: RigidTransform2D
    covariance: tuple
    start_station: float
    end_station: float
    registration: RegistrationResult
    diagnostics: CurveRegistrationDiagnostics


def _polyline_station(points):
    array = np.asarray(points, dtype=np.float64)
    distance = np.hypot(np.diff(array[:, 0]), np.diff(array[:, 1]))
    return np.concatenate(([0.0], np.cumsum(distance)))


def _resample_polyline(points, station, queries):
    points = np.asarray(points, dtype=np.float64)
    station = np.asarray(station, dtype=np.float64)
    keep = np.concatenate(([True], np.diff(station) > _EPSILON))
    points = points[keep]
    station = station[keep]
    if station.size < 2:
        raise ValueError("polyline has fewer than two distinct points")
    queries = np.asarray(queries, dtype=np.float64)
    return np.column_stack(
        (
            np.interp(queries, station, points[:, 0]),
            np.interp(queries, station, points[:, 1]),
        )
    )


def _curve_excitation(points):
    points = np.asarray(points, dtype=np.float64)
    delta = np.diff(points, axis=0)
    length = np.hypot(delta[:, 0], delta[:, 1])
    valid = length > _EPSILON
    delta = delta[valid]
    if delta.shape[0] < 2:
        return 0.0, 0.0
    heading = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    heading_variation = float(np.sum(np.abs(np.diff(heading))))
    chord = points[-1] - points[0]
    chord_length = float(np.hypot(chord[0], chord[1]))
    if chord_length <= _EPSILON:
        lateral = float(np.max(np.hypot(*(points - points[0]).T)))
    else:
        normal = np.asarray((-chord[1], chord[0]), dtype=np.float64) / chord_length
        lateral = float(np.max(np.abs((points - points[0]) @ normal)))
    return heading_variation, lateral


@dataclass(frozen=True)
class _CurveCandidate:
    start_station: float
    end_station: float
    parameters: tuple
    rms: float
    inlier_fraction: float
    condition: float


def _fit_point_arrays(local_points, target_points, selected):
    """Closed-form rigid fit for one candidate curve window."""
    local = np.asarray(local_points, dtype=np.float64)[selected]
    target = np.asarray(target_points, dtype=np.float64)[selected]
    if local.shape[0] < 2:
        return None
    local_mean = np.mean(local, axis=0)
    target_mean = np.mean(target, axis=0)
    local_centered = local - local_mean
    target_centered = target - target_mean
    if min(
        float(np.sum(local_centered ** 2)),
        float(np.sum(target_centered ** 2)),
    ) <= _EPSILON:
        return None
    cross = local_centered.T @ target_centered
    yaw = math.atan2(
        cross[0, 1] - cross[1, 0], cross[0, 0] + cross[1, 1]
    )
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    translation = target_mean - rotation @ local_mean
    parameters = np.asarray((translation[0], translation[1], yaw))

    derivative = np.column_stack(
        (
            -sine * local[:, 0] - cosine * local[:, 1],
            cosine * local[:, 0] - sine * local[:, 1],
        )
    )
    rows = np.zeros((2 * local.shape[0], 3), dtype=np.float64)
    rows[0::2, 0] = 1.0
    rows[1::2, 1] = 1.0
    rows[0::2, 2] = derivative[:, 0]
    rows[1::2, 2] = derivative[:, 1]
    information = rows.T @ rows
    if np.linalg.matrix_rank(information, tol=1e-10) < 3:
        return None
    condition = float(np.linalg.cond(information))
    return parameters, condition


def _curve_candidate(local_points, target_points, start_station, end_station, config):
    local_points = np.asarray(local_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    minimum_count = max(
        3,
        int(
            math.ceil(
                config.minimum_inlier_fraction * local_points.shape[0]
            )
        ),
    )
    selected = np.arange(local_points.shape[0])
    parameters = None
    condition = math.inf
    distances = np.full(local_points.shape[0], math.inf, dtype=np.float64)
    for _iteration in range(3):
        fitted = _fit_point_arrays(local_points, target_points, selected)
        if fitted is None:
            return None
        parameters, condition = fitted
        cosine = math.cos(parameters[2])
        sine = math.sin(parameters[2])
        predicted = np.column_stack(
            (
                parameters[0]
                + cosine * local_points[:, 0]
                - sine * local_points[:, 1],
                parameters[1]
                + sine * local_points[:, 0]
                + cosine * local_points[:, 1],
            )
        )
        distances = np.hypot(
            predicted[:, 0] - target_points[:, 0],
            predicted[:, 1] - target_points[:, 1],
        )
        next_selected = np.sort(np.argsort(distances)[:minimum_count])
        if np.array_equal(next_selected, selected):
            break
        selected = next_selected
    inlier_fraction = float(
        np.mean(distances <= config.position_inlier_threshold)
    )
    rms = float(math.sqrt(np.mean(distances[selected] ** 2)))
    return _CurveCandidate(
        start_station=float(start_station),
        end_station=float(end_station),
        parameters=tuple(float(value) for value in parameters),
        rms=rms,
        inlier_fraction=inlier_fraction,
        condition=condition,
    )


def _curve_rejection(template, observed, config, reason, degenerate, heading, lateral):
    return CurveRegistrationResult(
        accepted=False,
        stamp=observed.stamp,
        transform=None,
        covariance=tuple(),
        start_station=math.nan,
        end_station=math.nan,
        registration=None,
        diagnostics=CurveRegistrationDiagnostics(
            reason=reason,
            degenerate=bool(degenerate),
            template_length=template.length,
            observed_length=observed.length,
            observed_heading_variation=heading,
            observed_lateral_excitation=lateral,
            candidate_count=0,
            inlier_fraction=0.0,
            best_rms=math.inf,
            second_best_rms=math.inf,
            station_margin=math.inf,
        ),
    )


def register_curve_subset(
    template,
    observed,
    config=None,
    start_station_bounds=None,
):
    """Register an ordered, uncorresponded curve to a template station window.

    This is intended for the camera rolling ``CommonPath`` used to recognize a
    zigzag or another curved local route.  A straight observation is rejected
    even though a mathematical point fit exists, because its station along a
    longer straight is not observable.  ``start_station_bounds`` optionally
    limits the template-window start to a finite inclusive ``(minimum,
    maximum)`` interval.  Callers may use a pose projection on an already
    aligned route as this prior; omitting it preserves the global search.
    """

    if not isinstance(template, LocalCurveTemplate):
        raise TypeError("template must be LocalCurveTemplate")
    if not isinstance(observed, ObservedCurve):
        raise TypeError("observed must be ObservedCurve")
    config = CurveRegistrationConfig() if config is None else config
    if not isinstance(config, CurveRegistrationConfig):
        raise TypeError("config must be CurveRegistrationConfig")
    if start_station_bounds is not None:
        try:
            start_station_bounds = tuple(start_station_bounds)
        except TypeError:
            raise ValueError(
                "start station bounds must contain two finite values"
            )
        if len(start_station_bounds) != 2:
            raise ValueError(
                "start station bounds must contain two finite values"
            )
        minimum_start_station = _finite(
            start_station_bounds[0], "minimum start station"
        )
        maximum_start_station_bound = _finite(
            start_station_bounds[1], "maximum start station"
        )
        if minimum_start_station > maximum_start_station_bound:
            raise ValueError(
                "minimum start station must not exceed maximum start station"
            )
    else:
        minimum_start_station = None
        maximum_start_station_bound = None

    uniform_station = np.linspace(0.0, observed.length, config.sample_count)
    target_points = _resample_polyline(
        observed.points, observed.station, uniform_station
    )
    heading_variation, lateral_excitation = _curve_excitation(target_points)
    if observed.length + _EPSILON < config.minimum_observed_length:
        return _curve_rejection(
            template,
            observed,
            config,
            "observed curve is too short",
            True,
            heading_variation,
            lateral_excitation,
        )
    if (
        heading_variation < config.minimum_heading_variation
        and lateral_excitation < config.minimum_lateral_excitation
    ):
        return _curve_rejection(
            template,
            observed,
            config,
            "straight curve subset is station-degenerate",
            True,
            heading_variation,
            lateral_excitation,
        )
    if observed.length > template.length + 1e-6:
        return _curve_rejection(
            template,
            observed,
            config,
            "observed curve is longer than the local template",
            False,
            heading_variation,
            lateral_excitation,
        )

    maximum_start = max(0.0, template.length - observed.length)
    if start_station_bounds is None:
        starts = list(
            np.arange(
                0.0,
                maximum_start + 0.5 * config.station_search_step,
                config.station_search_step,
            )
        )
        if not starts or maximum_start - starts[-1] > 1e-9:
            starts.append(maximum_start)
    else:
        bounded_minimum = max(0.0, minimum_start_station)
        bounded_maximum = min(
            maximum_start, maximum_start_station_bound
        )
        if bounded_minimum > bounded_maximum + _EPSILON:
            return _curve_rejection(
                template,
                observed,
                config,
                "start station bounds do not overlap the feasible template",
                False,
                heading_variation,
                lateral_excitation,
            )

        # Retain the global station grid so a bound does not change the fit
        # merely by changing the grid phase.  Explicit interval endpoints make
        # a narrow but non-empty prior searchable as well.
        first_grid_index = int(
            math.ceil(
                (bounded_minimum - _EPSILON)
                / config.station_search_step
            )
        )
        first_grid = first_grid_index * config.station_search_step
        starts = list(
            np.arange(
                first_grid,
                bounded_maximum + 0.5 * config.station_search_step,
                config.station_search_step,
            )
        )
        starts = [
            float(value)
            for value in starts
            if bounded_minimum - _EPSILON
            <= value
            <= bounded_maximum + _EPSILON
        ]
        starts.extend((bounded_minimum, bounded_maximum))
        starts = sorted(
            set(
                round(float(value), 12)
                for value in starts
                if bounded_minimum - _EPSILON
                <= value
                <= bounded_maximum + _EPSILON
            )
        )
    candidates = []
    template_station = np.asarray(template.station, dtype=np.float64)
    for start in starts:
        start = min(float(start), maximum_start)
        queries = start + uniform_station
        local_points = _resample_polyline(
            template.points, template_station, queries
        )
        candidate = _curve_candidate(
            local_points,
            target_points,
            start,
            start + observed.length,
            config,
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return _curve_rejection(
            template,
            observed,
            config,
            "no observable curve candidate",
            True,
            heading_variation,
            lateral_excitation,
        )
    candidates.sort(key=lambda value: (value.rms, -value.inlier_fraction))
    best = candidates[0]
    separated = [
        candidate
        for candidate in candidates[1:]
        if abs(candidate.start_station - best.start_station)
        >= config.ambiguity_station_separation
    ]
    second = separated[0] if separated else None
    second_rms = math.inf if second is None else second.rms
    station_margin = second_rms - best.rms
    ratio = (
        math.inf
        if second is None or best.rms <= _EPSILON
        else second_rms / best.rms
    )

    sample_queries = best.start_station + uniform_station
    best_local = _resample_polyline(
        template.points, template_station, sample_queries
    )
    point_landmarks = tuple(
        PointLandmark("curve_%03d" % index, tuple(point))
        for index, point in enumerate(best_local)
    )
    point_observations = tuple(
        PointObservation("curve_%03d" % index, tuple(point))
        for index, point in enumerate(target_points)
    )
    registration_config = RegistrationConfig(
        position_inlier_threshold=config.position_inlier_threshold,
        minimum_inliers=max(
            3,
            int(math.ceil(config.minimum_inlier_fraction * config.sample_count)),
        ),
        minimum_inlier_coverage=config.minimum_inlier_fraction,
        maximum_condition_number=config.maximum_condition_number,
    )
    registration = estimate_local_registration(
        MissionLocalTemplate(
            mission=template.name,
            frame_id=template.frame_id,
            points=point_landmarks,
        ),
        RegistrationObservation(
            stamp=observed.stamp,
            target_frame=observed.target_frame,
            points=point_observations,
        ),
        registration_config,
        initial_transform=RigidTransform2D(
            best.parameters[0],
            best.parameters[1],
            best.parameters[2],
            template.frame_id,
            observed.target_frame,
        ),
    )

    reason = ""
    degenerate = False
    if best.inlier_fraction + _EPSILON < config.minimum_inlier_fraction:
        reason = "insufficient curve inliers"
    elif best.rms > config.maximum_rms:
        reason = "curve residual exceeds limit"
    elif second is not None and (
        station_margin <= config.maximum_ambiguity_rms_difference
        or ratio <= config.maximum_ambiguity_rms_ratio
    ):
        reason = "ambiguous template station"
        degenerate = True
    elif not registration.accepted:
        reason = registration.diagnostics.reason
        degenerate = registration.diagnostics.degenerate
    accepted = not reason
    return CurveRegistrationResult(
        accepted=accepted,
        stamp=observed.stamp,
        transform=registration.transform,
        covariance=registration.covariance,
        start_station=best.start_station,
        end_station=best.end_station,
        registration=registration,
        diagnostics=CurveRegistrationDiagnostics(
            reason=reason,
            degenerate=degenerate,
            template_length=template.length,
            observed_length=observed.length,
            observed_heading_variation=heading_variation,
            observed_lateral_excitation=lateral_excitation,
            candidate_count=len(candidates),
            inlier_fraction=best.inlier_fraction,
            best_rms=best.rms,
            second_best_rms=second_rms,
            station_margin=station_margin,
        ),
    )
