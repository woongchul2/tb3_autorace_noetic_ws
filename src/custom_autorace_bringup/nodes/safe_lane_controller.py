#!/usr/bin/env python3
"""Follow a camera-derived rolling lane path with explicit control handoff."""

from collections import deque
from dataclasses import dataclass
import math
import threading

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float64, Float64MultiArray, UInt8
from std_srvs.srv import SetBool, SetBoolResponse
from turtlebot3_autorace_msgs.msg import LaneCenterline

from custom_autorace_bringup.path_following import (
    AsymmetricFootprint,
    PathFollower,
    PathSafety,
    PointCloudBoundary,
    Pose2D,
    RigidTransform2D,
    SafetyMargins,
    SpeedProfile,
    SweptFootprintValidator,
    TrackingConfig,
    combine_validation_results,
    path_from_poses,
    yaw_from_quaternion,
)


@dataclass(frozen=True)
class LanePathCalibration:
    """Metric interpretation of rows in the projected lane image."""

    top_forward_distance: float
    bottom_forward_distance: float
    lane_width_m: float
    lane_width_pixels: float
    target_center: float
    minimum_confidence: float
    minimum_samples: int
    minimum_forward_span: float
    maximum_near_support_distance: float
    sample_spacing: float
    shape_control_spacing: float
    endpoint_plateau_pixels: float
    maximum_fit_residual_pixels: float
    maximum_endpoint_displacement_pixels: float
    fusion_minimum_samples: int
    fusion_minimum_overlap: float
    fusion_maximum_disagreement: float
    fusion_maximum_heading_disagreement: float = math.radians(5.0)
    minimum_executable_velocity: float = 0.04
    preferred_boundary: str = "strongest"

    def __post_init__(self):
        finite = (
            self.top_forward_distance,
            self.bottom_forward_distance,
            self.lane_width_m,
            self.lane_width_pixels,
            self.target_center,
            self.minimum_confidence,
            self.minimum_forward_span,
            self.maximum_near_support_distance,
            self.sample_spacing,
            self.shape_control_spacing,
            self.endpoint_plateau_pixels,
            self.maximum_fit_residual_pixels,
            self.maximum_endpoint_displacement_pixels,
            self.fusion_minimum_overlap,
            self.fusion_maximum_disagreement,
            self.fusion_maximum_heading_disagreement,
            self.minimum_executable_velocity,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("lane-path calibration values must be finite")
        if self.top_forward_distance <= self.bottom_forward_distance:
            raise ValueError("top row must map farther ahead than bottom row")
        if self.bottom_forward_distance < 0.0:
            raise ValueError("bottom forward distance cannot be negative")
        if self.lane_width_m <= 0.0 or self.lane_width_pixels <= 0.0:
            raise ValueError("lane width calibration must be positive")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum lane confidence must be within [0, 1]")
        if self.minimum_samples < 3:
            raise ValueError("rolling lane path needs at least three samples")
        if (
            self.minimum_forward_span <= 0.0
            or self.maximum_near_support_distance <= 0.0
            or self.sample_spacing <= 0.0
            or self.shape_control_spacing <= 0.0
        ):
            raise ValueError("lane path span and spacing must be positive")
        if self.maximum_near_support_distance > self.top_forward_distance:
            raise ValueError("near-support limit exceeds the camera horizon")
        if self.endpoint_plateau_pixels < 0.0:
            raise ValueError("endpoint plateau tolerance cannot be negative")
        if (
            self.maximum_fit_residual_pixels <= 0.0
            or self.maximum_endpoint_displacement_pixels <= 0.0
        ):
            raise ValueError("lane fit residual limits must be positive")
        if self.fusion_minimum_samples < 3:
            raise ValueError("lane fusion needs at least three shared samples")
        if (
            self.fusion_minimum_overlap <= 0.0
            or self.fusion_maximum_disagreement <= 0.0
            or self.fusion_maximum_heading_disagreement <= 0.0
            or self.fusion_maximum_heading_disagreement >= math.pi
        ):
            raise ValueError("lane fusion limits must be positive")
        if self.minimum_executable_velocity <= 0.0:
            raise ValueError(
                "minimum executable lane velocity must be positive"
            )
        if self.preferred_boundary not in ("yellow", "white", "strongest"):
            raise ValueError(
                "preferred boundary must be yellow, white, or strongest"
            )


@dataclass(frozen=True)
class LanePathSafetyConfig:
    """Physical interpretation and prediction horizon for detected paint."""

    line_half_width: float
    boundary_sample_spacing: float
    line_margin: float
    obstacle_margin: float
    localization_margin: float
    tracking_margin: float
    reaction_time: float
    stopping_distance_margin: float
    lookahead_distance: float

    def __post_init__(self):
        non_negative = (
            self.line_half_width,
            self.line_margin,
            self.obstacle_margin,
            self.localization_margin,
            self.tracking_margin,
            self.reaction_time,
            self.stopping_distance_margin,
        )
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in non_negative
        ):
            raise ValueError("lane safety margins must be finite and non-negative")
        if not (
            math.isfinite(self.boundary_sample_spacing)
            and self.boundary_sample_spacing > 0.0
            and math.isfinite(self.lookahead_distance)
            and self.lookahead_distance > 0.0
        ):
            raise ValueError("lane safety sampling and lookahead must be positive")

    @property
    def margins(self):
        return SafetyMargins(
            line=self.line_margin,
            obstacle=self.obstacle_margin,
            localization=self.localization_margin,
            tracking=self.tracking_margin,
        )


@dataclass(frozen=True)
class OdomSample:
    stamp: float
    pose: Pose2D
    linear_velocity: float
    angular_velocity: float
    frame_id: str


@dataclass(frozen=True)
class RollingLanePath:
    path: object
    valid_samples: int
    mean_confidence: float
    horizon: float
    observed_start_station: float


@dataclass(frozen=True)
class RollingLanePathSelection:
    """One camera calculation split into observation and drive decisions."""

    observation: object
    executable: object
    rejection_reason: str


def common_path_message(path, stamp):
    """Expose the already-built rolling path without recomputing geometry."""

    message = Path()
    message.header.stamp = stamp
    message.header.frame_id = path.frame_id
    for x, y, heading in zip(path.x, path.y, path.heading):
        pose = PoseStamped()
        pose.header = message.header
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.z = math.sin(0.5 * float(heading))
        pose.pose.orientation.w = math.cos(0.5 * float(heading))
        message.poses.append(pose)
    return message


@dataclass(frozen=True)
class BoundaryCenterCandidate:
    """One centre path derived from one physical painted boundary only."""

    side: str
    coefficients: np.ndarray
    boundary_coefficients: np.ndarray
    start_parameter: float
    end_parameter: float
    boundary_start_parameter: float
    boundary_end_parameter: float
    support_count: int
    mean_confidence: float

    @property
    def support_span(self):
        return self.boundary_end_parameter - self.boundary_start_parameter

    def sample(self, parameter):
        """Evaluate the constrained centre without extrapolating support."""

        parameter = np.asarray(parameter, dtype=np.float64).reshape(-1)
        if parameter.size == 0 or (
            np.min(parameter) < self.start_parameter - 1e-9
            or np.max(parameter) > self.end_parameter + 1e-9
        ):
            raise ValueError("lane candidate cannot extrapolate its support")
        parameter = np.clip(
            parameter, self.start_parameter, self.end_parameter
        )
        return parameter, np.polyval(self.coefficients, parameter)

def fit_endpoint_constrained_quadratic(
    parameter, values, weights, plateau_tolerance
):
    """Fit a weighted quadratic with measured endpoint-direction constraints.

    The unconstrained fit supplies the desired pixel-noise smoothing.  If its
    derivative would reverse relative to either observed endpoint secant, the
    convex least-squares optimum is recomputed with that endpoint derivative
    fixed at zero.  A sub-pixel plateau also fixes the derivative at zero.
    """

    parameter = np.asarray(parameter, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    plateau_tolerance = float(plateau_tolerance)
    if (
        parameter.shape != values.shape
        or parameter.shape != weights.shape
        or parameter.size < 3
    ):
        raise ValueError("quadratic fit needs at least three weighted samples")
    if not (
        np.all(np.isfinite(parameter))
        and np.all(np.isfinite(values))
        and np.all(np.isfinite(weights))
        and math.isfinite(plateau_tolerance)
        and plateau_tolerance >= 0.0
    ):
        raise ValueError("quadratic fit inputs must be finite")
    if np.any(np.diff(parameter) <= 1e-9) or np.any(weights < 0.0):
        raise ValueError("quadratic samples and weights are invalid")

    origin = float(parameter[0])
    scale = float(parameter[-1] - parameter[0])
    normalized = (parameter - origin) / scale
    design = np.column_stack(
        (normalized * normalized, normalized, np.ones(parameter.size))
    )
    weighted_design = design * np.maximum(weights, 1e-3)[:, None]
    weighted_values = values * np.maximum(weights, 1e-3)
    hessian = weighted_design.T @ weighted_design
    right_hand_side = weighted_design.T @ weighted_values
    derivative_rows = np.asarray(
        [[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float64
    )
    endpoint_delta = np.asarray(
        [values[1] - values[0], values[-1] - values[-2]],
        dtype=np.float64,
    )
    endpoint_sign = np.sign(endpoint_delta)
    endpoint_sign[np.abs(endpoint_delta) <= plateau_tolerance] = 0.0

    best = None
    for active_mask in range(4):
        active = {
            index
            for index in range(2)
            if active_mask & (1 << index)
        }
        active.update(
            index
            for index in range(2)
            if endpoint_sign[index] == 0.0
        )
        if active:
            constraint = derivative_rows[sorted(active)]
            count = int(constraint.shape[0])
            system = np.block(
                [
                    [hessian, constraint.T],
                    [constraint, np.zeros((count, count))],
                ]
            )
            target = np.concatenate(
                (right_hand_side, np.zeros(count, dtype=np.float64))
            )
            try:
                normalized_coefficients = np.linalg.solve(system, target)[:3]
            except np.linalg.LinAlgError:
                normalized_coefficients = np.linalg.lstsq(
                    system, target, rcond=None
                )[0][:3]
        else:
            normalized_coefficients = np.linalg.lstsq(
                weighted_design, weighted_values, rcond=None
            )[0]
        endpoint_derivative = derivative_rows @ normalized_coefficients
        feasible = all(
            abs(endpoint_derivative[index]) <= 1e-9
            if endpoint_sign[index] == 0.0
            else endpoint_sign[index] * endpoint_derivative[index] >= -1e-9
            for index in range(2)
        )
        if not feasible:
            continue
        residual = weighted_design @ normalized_coefficients - weighted_values
        cost = float(residual @ residual)
        if best is None or cost < best[0]:
            best = (cost, normalized_coefficients)
    if best is None:
        raise ValueError("quadratic endpoint constraints are infeasible")

    normalized_coefficients = best[1]
    quadratic = normalized_coefficients[0] / (scale * scale)
    linear = (
        normalized_coefficients[1] / scale - 2.0 * quadratic * origin
    )
    constant = (
        normalized_coefficients[2]
        - normalized_coefficients[1] * origin / scale
        + quadratic * origin * origin
    )
    coefficients = np.asarray([quadratic, linear, constant])
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("constrained quadratic is not finite")
    return coefficients


def interpolate_odom_sample(before, after, stamp):
    """Interpolate a camera-time pose between two odometry observations."""

    stamp = float(stamp)
    if before.frame_id != after.frame_id:
        raise ValueError("cannot interpolate odometry across different frames")
    interval = float(after.stamp - before.stamp)
    if interval <= 0.0:
        raise ValueError("odometry interpolation needs increasing timestamps")
    fraction = (stamp - before.stamp) / interval
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("camera timestamp is outside the odometry interval")
    yaw_delta = math.atan2(
        math.sin(after.pose.yaw - before.pose.yaw),
        math.cos(after.pose.yaw - before.pose.yaw),
    )
    yaw = before.pose.yaw + fraction * yaw_delta
    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
    return OdomSample(
        stamp=stamp,
        pose=Pose2D(
            before.pose.x + fraction * (after.pose.x - before.pose.x),
            before.pose.y + fraction * (after.pose.y - before.pose.y),
            yaw,
        ),
        linear_velocity=(
            before.linear_velocity
            + fraction * (after.linear_velocity - before.linear_velocity)
        ),
        angular_velocity=(
            before.angular_velocity
            + fraction * (after.angular_velocity - before.angular_velocity)
        ),
        frame_id=before.frame_id,
    )


def build_boundary_candidate(
    side,
    parameter,
    boundary_lateral,
    valid,
    confidence,
    calibration,
):
    """Fit and normal-offset one boundary over its supported interval only."""

    indices = np.flatnonzero(valid)
    if indices.size < calibration.minimum_samples:
        return None
    supported_parameter = np.asarray(parameter[indices], dtype=np.float64)
    supported_lateral = np.asarray(boundary_lateral[indices], dtype=np.float64)
    supported_confidence = np.asarray(confidence[indices], dtype=np.float64)
    order = np.argsort(supported_parameter)
    supported_parameter = supported_parameter[order]
    supported_lateral = supported_lateral[order]
    supported_confidence = supported_confidence[order]
    if np.any(np.diff(supported_parameter) <= 1e-6):
        return None
    support_span = float(supported_parameter[-1] - supported_parameter[0])
    if support_span < calibration.minimum_forward_span:
        return None
    if supported_parameter[0] > calibration.maximum_near_support_distance:
        # A far-only fragment does not constrain the route between the robot
        # and that paint. Reject this current-frame candidate instead of
        # connecting to an unrelated same-colour branch near the horizon.
        return None

    plateau_tolerance = (
        calibration.endpoint_plateau_pixels
        * calibration.lane_width_m
        / calibration.lane_width_pixels
    )
    try:
        boundary_coefficients = fit_endpoint_constrained_quadratic(
            supported_parameter,
            supported_lateral,
            supported_confidence,
            plateau_tolerance,
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    boundary_at_nodes = np.polyval(
        boundary_coefficients, supported_parameter
    )
    residual_pixels = (
        np.abs(boundary_at_nodes - supported_lateral)
        * calibration.lane_width_pixels
        / calibration.lane_width_m
    )
    if (
        float(np.max(residual_pixels))
        > calibration.maximum_fit_residual_pixels
        or max(float(residual_pixels[0]), float(residual_pixels[-1]))
        > calibration.maximum_endpoint_displacement_pixels
    ):
        # Quadratic smoothing may not invent a centreline by moving the
        # current-frame paint farther than the measured residual envelope.
        return None
    boundary_slope = np.polyval(
        np.polyder(boundary_coefficients), supported_parameter
    )
    if side == "yellow":
        offset_direction = 1
    elif side == "white":
        offset_direction = -1
    else:
        raise ValueError("lane boundary side must be yellow or white")

    # Offset smoothed observations only at their supported rows. The weighted
    # quadratic retains the detector's noise rejection, while its endpoint
    # constraints prevent an unobserved tangent reversal.
    normal_scale = (0.5 * calibration.lane_width_m) / np.hypot(
        1.0, boundary_slope
    )
    centre_forward = (
        supported_parameter
        + offset_direction * normal_scale * boundary_slope
    )
    centre_lateral = boundary_at_nodes - offset_direction * normal_scale
    if not (
        np.all(np.isfinite(centre_forward))
        and np.all(np.isfinite(centre_lateral))
        and np.all(np.diff(centre_forward) > 1e-6)
    ):
        return None
    try:
        centre_coefficients = fit_endpoint_constrained_quadratic(
            centre_forward,
            centre_lateral,
            supported_confidence,
            plateau_tolerance,
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None
    return BoundaryCenterCandidate(
        side=side,
        coefficients=np.asarray(centre_coefficients, dtype=np.float64),
        boundary_coefficients=np.asarray(
            boundary_coefficients, dtype=np.float64
        ),
        start_parameter=float(centre_forward[0]),
        end_parameter=float(centre_forward[-1]),
        boundary_start_parameter=float(supported_parameter[0]),
        boundary_end_parameter=float(supported_parameter[-1]),
        support_count=int(indices.size),
        mean_confidence=float(np.mean(supported_confidence)),
    )


def sample_candidate(candidate, start, end, maximum_spacing):
    """Sample one boundary-derived centre without extrapolating its support."""

    span = float(end - start)
    if span <= 0.0:
        raise ValueError("lane candidate interval must be positive")
    count = max(3, int(math.ceil(span / float(maximum_spacing))) + 1)
    parameter = np.linspace(float(start), float(end), count)
    forward, lateral = candidate.sample(parameter)
    if not (
        np.all(np.isfinite(forward))
        and np.all(np.isfinite(lateral))
    ):
        raise ValueError("metric boundary offset is not finite")
    return parameter, forward, lateral


def shape_preserving_slopes(parameter, values, first_slope=None):
    """Return non-overshooting PCHIP tangents for nonuniform knots.

    The weighted constrained quadratic above remains the sole noise-smoothing
    fit.  These Fritsch-Carlson tangents only turn its already-smooth dense
    samples into one C1 curve. Therefore a direction change can occur only
    where adjacent measured knot secants change sign, never because of
    polynomial endpoint leakage.
    """

    parameter = np.asarray(parameter, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if parameter.shape != values.shape or parameter.size < 2:
        raise ValueError("shape-preserving curve needs paired knots")
    if not (
        np.all(np.isfinite(parameter)) and np.all(np.isfinite(values))
    ):
        raise ValueError("shape-preserving curve knots must be finite")
    interval = np.diff(parameter)
    if np.any(interval <= 1e-9):
        raise ValueError("shape-preserving curve knots must increase")
    secant = np.diff(values) / interval
    count = int(parameter.size)
    slope = np.empty(count, dtype=np.float64)
    if count == 2:
        slope[:] = secant[0]
    else:
        for index in range(1, count - 1):
            before = secant[index - 1]
            after = secant[index]
            if before == 0.0 or after == 0.0 or before * after <= 0.0:
                slope[index] = 0.0
            else:
                before_interval = interval[index - 1]
                after_interval = interval[index]
                weight_before = 2.0 * after_interval + before_interval
                weight_after = after_interval + 2.0 * before_interval
                slope[index] = (weight_before + weight_after) / (
                    weight_before / before + weight_after / after
                )

        def endpoint_slope(first_interval, second_interval, first, second):
            candidate = (
                (2.0 * first_interval + second_interval) * first
                - first_interval * second
            ) / (first_interval + second_interval)
            if candidate * first <= 0.0:
                return 0.0
            if first * second < 0.0 and abs(candidate) > 3.0 * abs(first):
                return 3.0 * first
            return candidate

        slope[0] = endpoint_slope(
            interval[0], interval[1], secant[0], secant[1]
        )
        slope[-1] = endpoint_slope(
            interval[-1], interval[-2], secant[-1], secant[-2]
        )
    if first_slope is not None:
        first_slope = float(first_slope)
        if not math.isfinite(first_slope):
            raise ValueError("fixed first slope must be finite")
        first_secant = float(secant[0])
        if (
            first_slope * first_secant < -1e-12
            or abs(first_slope) > 3.0 * abs(first_secant) + 1e-12
        ):
            raise ValueError("fixed first slope would overshoot its interval")
        slope[0] = first_slope
    if not np.all(np.isfinite(slope)):
        raise ValueError("shape-preserving tangents are not finite")
    return slope


def sample_shape_preserving_curve(
    parameter, values, maximum_spacing, first_slope=None
):
    """Sample one nonuniform piecewise-Hermite curve and analytic tangent."""

    maximum_spacing = float(maximum_spacing)
    if not math.isfinite(maximum_spacing) or maximum_spacing <= 0.0:
        raise ValueError("shape-preserving sample spacing must be positive")
    parameter = np.asarray(parameter, dtype=np.float64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    slope = shape_preserving_slopes(
        parameter, values, first_slope=first_slope
    )
    sampled_parameter = []
    sampled_values = []
    sampled_slope = []
    for index in range(parameter.size - 1):
        interval = float(parameter[index + 1] - parameter[index])
        segment_count = max(1, int(math.ceil(interval / maximum_spacing)))
        unit = np.linspace(0.0, 1.0, segment_count + 1)
        if index:
            unit = unit[1:]
        unit_squared = unit * unit
        unit_cubed = unit_squared * unit
        start_value = float(values[index])
        end_value = float(values[index + 1])
        start_slope = float(slope[index])
        end_slope = float(slope[index + 1])
        sampled_parameter.append(parameter[index] + unit * interval)
        sampled_values.append(
            (2.0 * unit_cubed - 3.0 * unit_squared + 1.0) * start_value
            + (unit_cubed - 2.0 * unit_squared + unit)
            * interval
            * start_slope
            + (-2.0 * unit_cubed + 3.0 * unit_squared) * end_value
            + (unit_cubed - unit_squared) * interval * end_slope
        )
        sampled_slope.append(
            (
                (6.0 * unit_squared - 6.0 * unit) * start_value
                + (3.0 * unit_squared - 4.0 * unit + 1.0)
                * interval
                * start_slope
                + (-6.0 * unit_squared + 6.0 * unit) * end_value
                + (3.0 * unit_squared - 2.0 * unit)
                * interval
                * end_slope
            )
            / interval
        )
    sampled_parameter = np.concatenate(sampled_parameter)
    sampled_values = np.concatenate(sampled_values)
    sampled_slope = np.concatenate(sampled_slope)
    if not all(
        np.all(np.isfinite(array))
        for array in (sampled_parameter, sampled_values, sampled_slope)
    ):
        raise ValueError("sampled shape-preserving curve is not finite")
    return sampled_parameter, sampled_values, sampled_slope


def ego_connected_lane_poses(forward, lateral, maximum_spacing):
    """Join the capture-time pose to the first observed lane sample.

    The camera cannot observe the ground immediately below the robot. The
    connector gives the common projector a continuous path from the capture
    pose.  First build the observed curve without the ego point, then join its
    first position and tangent with one cubic Hermite segment.  Including the
    ego point in the same PCHIP fit suppresses the upcoming tangent until the
    join and creates a false curvature spike immediately after it.

    The controller keeps the first camera-supported station as a diagnostic
    boundary.  The connector is constrained by that sample's position and
    tangent, may be selected as a short-range steering target, and remains in
    the common path for projection and swept-footprint validation.
    """

    forward = np.asarray(forward, dtype=np.float64).reshape(-1)
    lateral = np.asarray(lateral, dtype=np.float64).reshape(-1)
    if forward.size < 3 or forward.shape != lateral.shape:
        raise ValueError("lane candidate needs at least three paired points")
    if not (
        np.all(np.isfinite(forward)) and np.all(np.isfinite(lateral))
    ):
        raise ValueError("lane candidate points must be finite")
    if forward[0] <= 1e-6 or np.any(np.diff(forward) <= 1e-6):
        raise ValueError("lane candidate must progress forward from the ego pose")

    (
        observed_forward,
        observed_lateral,
        observed_slope,
    ) = sample_shape_preserving_curve(
        forward,
        lateral,
        maximum_spacing,
    )

    join_forward = float(observed_forward[0])
    join_lateral = float(observed_lateral[0])
    join_slope = float(observed_slope[0])
    connector_count = max(
        1, int(math.ceil(join_forward / float(maximum_spacing)))
    )
    unit = np.linspace(0.0, 1.0, connector_count + 1)
    unit_squared = unit * unit
    unit_cubed = unit_squared * unit
    connector_forward = unit * join_forward
    # Cubic Hermite boundary conditions:
    # y(0)=0, y'(0)=0, y(join)=observed y, y'(join)=observed slope.
    connector_lateral = (
        (-2.0 * unit_cubed + 3.0 * unit_squared) * join_lateral
        + (unit_cubed - unit_squared) * join_forward * join_slope
    )
    connector_slope = (
        (-6.0 * unit_squared + 6.0 * unit) * join_lateral
        + (3.0 * unit_squared - 2.0 * unit)
        * join_forward
        * join_slope
    ) / join_forward

    # Keep the observation endpoint only once. Its analytic tangent is shared
    # by both pieces, making the splice C1 instead of producing a join impulse.
    reference_forward = np.concatenate(
        (connector_forward[:-1], observed_forward)
    )
    reference_lateral = np.concatenate(
        (connector_lateral[:-1], observed_lateral)
    )
    reference_slope = np.concatenate(
        (connector_slope[:-1], observed_slope)
    )
    reference_heading = np.arctan(reference_slope)
    if (
        np.any(np.diff(reference_forward) <= 1e-6)
        or not np.all(np.isfinite(reference_lateral))
        or not np.all(np.isfinite(reference_heading))
    ):
        raise ValueError("reference lane path is not forward-progressing")
    return np.column_stack(
        (reference_forward, reference_lateral, reference_heading)
    )


def boundary_geometry_options(
    yellow_candidate,
    white_candidate,
    parameter,
    yellow_valid,
    white_valid,
    confidence,
    calibration,
):
    """Return same-frame centre candidates in their deterministic preference.

    Broadly agreeing yellow/white observations still form the first choice.
    Otherwise the candidate whose measured support begins nearest the robot is
    tried first; support quality and the configured course side break ties.
    The caller converts each option to a ``CommonPath`` and accepts only one
    that belongs to the current lane and is physically executable. This is
    deliberately not a previous-path fallback.
    """

    candidates = [
        candidate
        for candidate in (yellow_candidate, white_candidate)
        if candidate is not None
    ]
    if not candidates:
        raise ValueError("no lane boundary has enough local support")

    options = []
    if yellow_candidate is not None and white_candidate is not None:
        boundary_overlap_start = max(
            yellow_candidate.boundary_start_parameter,
            white_candidate.boundary_start_parameter,
        )
        boundary_overlap_end = min(
            yellow_candidate.boundary_end_parameter,
            white_candidate.boundary_end_parameter,
        )
        boundary_overlap = boundary_overlap_end - boundary_overlap_start
        shared = (
            yellow_valid
            & white_valid
            & (parameter >= boundary_overlap_start - 1e-9)
            & (parameter <= boundary_overlap_end + 1e-9)
        )
        shared_count = int(np.count_nonzero(shared))
        if (
            boundary_overlap >= calibration.fusion_minimum_overlap
            and shared_count >= calibration.fusion_minimum_samples
        ):
            centre_overlap_start = max(
                yellow_candidate.start_parameter,
                white_candidate.start_parameter,
            )
            centre_overlap_end = min(
                yellow_candidate.end_parameter,
                white_candidate.end_parameter,
            )
            if centre_overlap_end > centre_overlap_start:
                (
                    sampled,
                    yellow_forward,
                    yellow_lateral,
                ) = sample_candidate(
                    yellow_candidate,
                    centre_overlap_start,
                    centre_overlap_end,
                    calibration.shape_control_spacing,
                )
                (
                    white_parameter,
                    white_forward,
                    white_lateral,
                ) = sample_candidate(
                    white_candidate,
                    centre_overlap_start,
                    centre_overlap_end,
                    calibration.shape_control_spacing,
                )
                if not np.allclose(sampled, white_parameter, atol=1e-12):
                    raise ValueError("lane candidate samples are inconsistent")
                disagreement = np.hypot(
                    yellow_forward - white_forward,
                    yellow_lateral - white_lateral,
                )
                yellow_heading = np.arctan(
                    np.polyval(
                        np.polyder(yellow_candidate.coefficients),
                        sampled,
                    )
                )
                white_heading = np.arctan(
                    np.polyval(
                        np.polyder(white_candidate.coefficients),
                        sampled,
                    )
                )
                heading_disagreement = np.abs(
                    np.arctan2(
                        np.sin(yellow_heading - white_heading),
                        np.cos(yellow_heading - white_heading),
                    )
                )
                if (
                    float(np.max(disagreement))
                    <= calibration.fusion_maximum_disagreement
                    and float(np.max(heading_disagreement))
                    <= calibration.fusion_maximum_heading_disagreement
                ):
                    options.append(
                        (
                            0.5 * (yellow_forward + white_forward),
                            0.5 * (yellow_lateral + white_lateral),
                            shared_count,
                            float(np.mean(confidence[shared])),
                            float(centre_overlap_end),
                            (
                                (
                                    yellow_candidate,
                                    boundary_overlap_start,
                                    boundary_overlap_end,
                                ),
                                (
                                    white_candidate,
                                    boundary_overlap_start,
                                    boundary_overlap_end,
                                ),
                            ),
                        )
                    )

    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.start_parameter,
            -candidate.support_span,
            -candidate.support_count,
            -candidate.mean_confidence,
            candidate.side != calibration.preferred_boundary
            if calibration.preferred_boundary != "strongest"
            else False,
            candidate.side != "white",
        ),
    )
    for selected in ordered_candidates:
        _, forward, lateral = sample_candidate(
            selected,
            selected.start_parameter,
            selected.end_parameter,
            calibration.shape_control_spacing,
        )
        options.append(
            (
                forward,
                lateral,
                selected.support_count,
                selected.mean_confidence,
                selected.end_parameter,
                (
                    (
                        selected,
                        selected.boundary_start_parameter,
                        selected.boundary_end_parameter,
                    ),
                ),
            )
        )
    return options


def path_safety_from_detected_boundaries(boundary_supports, safety_config):
    """Represent only the selected current-frame paint as dense boundaries.

    ``PointCloudBoundary`` measures the asymmetric robot rectangle against
    every point. Half of the largest chord is included in each point radius so
    the continuous paint between samples is covered as well as the samples.
    """

    if not isinstance(safety_config, LanePathSafetyConfig):
        raise TypeError("lane path safety config is required")
    boundaries = []
    for candidate, requested_start, requested_end in boundary_supports:
        start = max(
            float(requested_start), candidate.boundary_start_parameter
        )
        end = min(float(requested_end), candidate.boundary_end_parameter)
        if end <= start:
            raise ValueError("selected lane boundary support is empty")
        count = max(
            2,
            int(
                math.ceil(
                    (end - start) / safety_config.boundary_sample_spacing
                )
            )
            + 1,
        )
        parameter = np.linspace(start, end, count)
        lateral = np.polyval(candidate.boundary_coefficients, parameter)
        points = np.column_stack((parameter, lateral))
        if not np.all(np.isfinite(points)):
            raise ValueError("selected lane boundary is not finite")
        chord = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
        point_radius = safety_config.line_half_width
        if chord.size:
            point_radius += 0.5 * float(np.max(chord))
        boundaries.append(
            PointCloudBoundary(points, point_radius=point_radius)
        )
    if not boundaries:
        raise ValueError("lane path needs at least one detected boundary")
    return PathSafety(
        line_boundaries=tuple(boundaries),
        margins=safety_config.margins,
    )


def validate_local_lane_path(
    local_path,
    speed_profile,
    minimum_executable_velocity,
):
    """Reject geometry whose common speed profile is below control authority.

    ``SpeedProfile.minimum_velocity`` is the nominal moving target, not a
    hard lower bound: its curvature and yaw-rate constraints deliberately may
    reduce a point below that value.  Geometry is therefore checked against
    the fully generated per-point speed profile and the separately measured
    controllable crawl speed. This retains physically executable tight bends
    while retaining the actual in-lane lateral and heading error for the common
    follower.  Paint intrusion is decided separately by the swept-footprint
    validator from the measured boundary support, not by an assumed centreline
    offset limit in the camera blind zone.
    """

    validate_local_lane_path_geometry(local_path)
    minimum_executable_velocity = validated_minimum_executable_velocity(
        speed_profile, minimum_executable_velocity
    )
    validate_local_lane_path_execution(
        local_path, minimum_executable_velocity
    )


def validated_minimum_executable_velocity(
    speed_profile,
    minimum_executable_velocity,
):
    """Validate the measured crawl threshold once per camera frame."""

    minimum_executable_velocity = float(minimum_executable_velocity)
    if (
        not math.isfinite(minimum_executable_velocity)
        or minimum_executable_velocity <= 0.0
        or minimum_executable_velocity > speed_profile.minimum_velocity
    ):
        raise ValueError("minimum executable lane velocity is invalid")
    return minimum_executable_velocity


def validate_local_lane_path_execution(
    local_path,
    minimum_executable_velocity,
):
    """Apply normal-lane drive authority after geometry is accepted once."""

    if float(np.min(local_path.speed)) < (
        minimum_executable_velocity * (1.0 - 1e-9)
    ):
        raise ValueError(
            "lane path physical curvature requires sub-controllable speed"
        )


def validate_local_lane_path_geometry(local_path):
    """Validate geometry independently from normal-lane control authority."""

    if not (
        np.all(np.isfinite(local_path.curvature))
        and np.all(np.isfinite(local_path.speed))
        and np.all(np.diff(local_path.x) > 1e-6)
        and np.all(np.diff(local_path.station) > 1e-9)
    ):
        raise ValueError("reference lane path geometry is invalid")


def build_rolling_lane_path_selection(
    message,
    capture_pose,
    odom_frame,
    calibration,
    speed_profile,
    safety_config=None,
):
    """Convert one timestamped image observation into one odom ``CommonPath``.

    Detection owns the image fit. This function converts its sampled boundary
    candidates into metric ground coordinates and lets the common path
    implementation derive heading, curvature and speed. If both colours are
    independently supported, only same-frame candidates are considered.
    """

    height = int(message.image_height)
    width = int(message.image_width)
    if height < 2 or width < 2:
        raise ValueError("lane centreline image dimensions are invalid")

    rows = np.asarray(message.sample_rows, dtype=np.float64).reshape(-1)
    centers = np.asarray(message.center_x, dtype=np.float64).reshape(-1)
    yellow = np.asarray(message.yellow_x, dtype=np.float64).reshape(-1)
    white = np.asarray(message.white_x, dtype=np.float64).reshape(-1)
    confidence = np.asarray(message.confidence, dtype=np.float64).reshape(-1)
    yellow_valid = np.asarray(message.yellow_valid, dtype=np.bool_).reshape(-1)
    white_valid = np.asarray(message.white_valid, dtype=np.bool_).reshape(-1)
    count = int(rows.size)
    if count == 0 or any(
        values.size != count
        for values in (
            centers,
            yellow,
            white,
            confidence,
            yellow_valid,
            white_valid,
        )
    ):
        raise ValueError("lane centreline arrays must have the same non-zero size")

    row_valid = (
        np.isfinite(rows)
        & np.isfinite(confidence)
        & (rows >= 0.0)
        & (rows <= float(height - 1))
        & (confidence >= calibration.minimum_confidence)
        & (confidence <= 1.0)
    )
    yellow_in_range = np.zeros(count, dtype=np.bool_)
    yellow_finite = np.isfinite(yellow)
    yellow_in_range[yellow_finite] = (
        (yellow[yellow_finite] >= 0.0)
        & (yellow[yellow_finite] <= float(width - 1))
    )
    white_in_range = np.zeros(count, dtype=np.bool_)
    white_finite = np.isfinite(white)
    white_in_range[white_finite] = (
        (white[white_finite] >= 0.0)
        & (white[white_finite] <= float(width - 1))
    )
    yellow_valid = row_valid & yellow_valid & yellow_in_range
    white_valid = row_valid & white_valid & white_in_range
    evidence_valid = row_valid & (yellow_valid | white_valid)
    rows = rows[evidence_valid]
    yellow = yellow[evidence_valid]
    white = white[evidence_valid]
    confidence = confidence[evidence_valid]
    yellow_valid = yellow_valid[evidence_valid]
    white_valid = white_valid[evidence_valid]
    if rows.size == 0:
        raise ValueError("no valid near/mid/far lane samples")

    # A calibrated bird-eye image is metric-affine: the top row is farthest
    # ahead and the bottom row is nearest to the robot.
    row_fraction = rows / float(height - 1)
    forward = calibration.top_forward_distance + row_fraction * (
        calibration.bottom_forward_distance
        - calibration.top_forward_distance
    )
    observation_order = np.argsort(forward)
    forward = forward[observation_order]
    yellow = yellow[observation_order]
    white = white[observation_order]
    confidence = confidence[observation_order]
    yellow_valid = yellow_valid[observation_order]
    white_valid = white_valid[observation_order]
    if np.any(np.diff(forward) <= 1e-6):
        raise ValueError("lane boundary sample rows must be unique")
    lateral_scale = calibration.lane_width_m / calibration.lane_width_pixels
    yellow_lateral = (calibration.target_center - yellow) * lateral_scale
    white_lateral = (calibration.target_center - white) * lateral_scale
    yellow_candidate = build_boundary_candidate(
        "yellow",
        forward,
        yellow_lateral,
        yellow_valid,
        confidence,
        calibration,
    )
    white_candidate = build_boundary_candidate(
        "white",
        forward,
        white_lateral,
        white_valid,
        confidence,
        calibration,
    )
    geometry_options = boundary_geometry_options(
        yellow_candidate,
        white_candidate,
        forward,
        yellow_valid,
        white_valid,
        confidence,
        calibration,
    )
    minimum_executable_velocity = validated_minimum_executable_velocity(
        speed_profile, calibration.minimum_executable_velocity
    )
    first_observation = None
    executable = None
    last_geometry_error = None
    last_execution_error = None
    for (
        local_forward,
        local_lateral,
        selected_samples,
        selected_confidence,
        selected_horizon,
        selected_boundary_supports,
    ) in geometry_options:
        try:
            local_poses = ego_connected_lane_poses(
                local_forward,
                local_lateral,
                calibration.sample_spacing,
            )
            observed_indices = np.flatnonzero(
                np.isclose(
                    local_poses[:, 0],
                    float(local_forward[0]),
                    rtol=0.0,
                    atol=1e-10,
                )
            )
            if observed_indices.size != 1:
                raise ValueError("lane connector lost its first observation")
            observed_start_index = int(observed_indices[0])
            local_path = path_from_poses(
                local_poses,
                frame_id="base_footprint",
                speed_profile=speed_profile,
                safety=(
                    path_safety_from_detected_boundaries(
                        selected_boundary_supports, safety_config
                    )
                    if safety_config is not None
                    else None
                ),
                label="camera_lane",
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            last_geometry_error = error
            continue
        try:
            validate_local_lane_path_geometry(local_path)
        except ValueError as error:
            last_geometry_error = error
            continue
        candidate = (
            local_path,
            selected_samples,
            selected_confidence,
            selected_horizon,
            observed_start_index,
        )
        if first_observation is None:
            first_observation = candidate
        try:
            validate_local_lane_path_execution(
                local_path,
                minimum_executable_velocity,
            )
        except ValueError as error:
            last_execution_error = error
            continue
        executable = candidate
        break
    if first_observation is None:
        if last_geometry_error is not None:
            raise last_geometry_error
        raise ValueError("no lane boundary has valid reference geometry")

    capture_pose = Pose2D.from_value(capture_pose)
    transform = RigidTransform2D(
        capture_pose.x,
        capture_pose.y,
        capture_pose.yaw,
        source_frame="base_footprint",
        target_frame=str(odom_frame) or "odom",
    )

    def transformed(candidate):
        local_path, samples, confidence, horizon, start_index = candidate
        return RollingLanePath(
            path=transform.apply_path(local_path),
            valid_samples=samples,
            mean_confidence=confidence,
            horizon=horizon,
            observed_start_station=float(local_path.station[start_index]),
        )

    selected_observation = (
        executable if executable is not None else first_observation
    )
    observation = transformed(selected_observation)
    return RollingLanePathSelection(
        observation=observation,
        executable=observation if executable is not None else None,
        rejection_reason=(
            ""
            if executable is not None
            else str(last_execution_error or "lane path is not executable")
        ),
    )


def build_rolling_lane_path(
    message,
    capture_pose,
    odom_frame,
    calibration,
    speed_profile,
    safety_config=None,
):
    """Build the normal-lane path, retaining the historical strict API."""

    selection = build_rolling_lane_path_selection(
        message,
        capture_pose,
        odom_frame,
        calibration,
        speed_profile,
        safety_config,
    )
    if selection.executable is None:
        raise ValueError(selection.rejection_reason)
    return selection.executable


class SafeLaneController:
    """Sole normal-lane ``cmd_vel`` publisher using the common path follower."""

    def __init__(self):
        self.centerline_topic = rospy.get_param(
            "~centerline_topic", "/detect/lane_centerline"
        )
        self.odometry_topic = rospy.get_param("~odometry_topic", "/odom")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.diagnostics_topic = rospy.get_param(
            "~diagnostics_topic", "/control/lane_path_diagnostics"
        )
        self.path_topic = rospy.get_param(
            "~path_topic", "/control/lane_path"
        )
        self.target_center = float(rospy.get_param("~target_center", 500.0))
        self.maximum_velocity = float(rospy.get_param("~maximum_velocity", 0.1))
        self.lane_timeout = float(rospy.get_param("~lane_timeout", 0.5))
        self.odom_timeout = float(rospy.get_param("~lane_path/odom_timeout", 0.35))
        self.maximum_pose_stamp_skew = float(
            rospy.get_param("~lane_path/maximum_pose_stamp_skew", 0.06)
        )
        self.control_period = max(
            0.01,
            float(rospy.get_param("~lane_path/control_period", 1.0 / 30.0)),
        )
        self.path_calibration = LanePathCalibration(
            top_forward_distance=float(
                rospy.get_param("~lane_path/top_forward_distance", 0.60)
            ),
            bottom_forward_distance=float(
                rospy.get_param("~lane_path/bottom_forward_distance", 0.16)
            ),
            lane_width_m=float(rospy.get_param("~lane_path/lane_width_m", 0.25)),
            lane_width_pixels=float(
                rospy.get_param("~lane_path/lane_width_pixels", 640.0)
            ),
            target_center=self.target_center,
            minimum_confidence=float(
                rospy.get_param("~lane_path/minimum_confidence", 0.20)
            ),
            minimum_samples=max(
                3, int(rospy.get_param("~lane_path/minimum_samples", 3))
            ),
            minimum_forward_span=float(
                rospy.get_param("~lane_path/minimum_forward_span", 0.10)
            ),
            maximum_near_support_distance=float(
                rospy.get_param(
                    "~lane_path/maximum_near_support_distance", 0.37
                )
            ),
            sample_spacing=float(
                rospy.get_param("~lane_path/sample_spacing", 0.01)
            ),
            shape_control_spacing=float(
                rospy.get_param("~lane_path/shape_control_spacing", 0.01)
            ),
            endpoint_plateau_pixels=float(
                rospy.get_param(
                    "~lane_path/endpoint_plateau_pixels", 1.0
                )
            ),
            maximum_fit_residual_pixels=float(
                rospy.get_param(
                    "~lane_path/maximum_fit_residual_pixels", 25.0
                )
            ),
            maximum_endpoint_displacement_pixels=float(
                rospy.get_param(
                    "~lane_path/maximum_endpoint_displacement_pixels", 20.0
                )
            ),
            fusion_minimum_samples=max(
                3,
                int(
                    rospy.get_param(
                        "~lane_path/fusion_minimum_samples", 4
                    )
                ),
            ),
            fusion_minimum_overlap=float(
                rospy.get_param("~lane_path/fusion_minimum_overlap", 0.20)
            ),
            fusion_maximum_disagreement=float(
                rospy.get_param(
                    "~lane_path/fusion_maximum_disagreement", 0.03
                )
            ),
            fusion_maximum_heading_disagreement=math.radians(
                float(
                    rospy.get_param(
                        "~lane_path/fusion_maximum_heading_disagreement_deg",
                        5.0,
                    )
                )
            ),
            minimum_executable_velocity=float(
                rospy.get_param(
                    "~lane_path/minimum_executable_velocity", 0.04
                )
            ),
            preferred_boundary=str(
                rospy.get_param(
                    "~lane_path/preferred_boundary", "white"
                )
            ).strip().lower(),
        )
        self.path_safety_config = LanePathSafetyConfig(
            line_half_width=float(
                rospy.get_param("~lane_path/safety/line_half_width", 0.005)
            ),
            boundary_sample_spacing=float(
                rospy.get_param(
                    "~lane_path/safety/boundary_sample_spacing", 0.004
                )
            ),
            line_margin=float(
                rospy.get_param("~lane_path/safety/line_margin", 0.0)
            ),
            obstacle_margin=float(
                rospy.get_param("~lane_path/safety/obstacle_margin", 0.0)
            ),
            localization_margin=float(
                rospy.get_param(
                    "~lane_path/safety/localization_margin", 0.002
                )
            ),
            tracking_margin=float(
                rospy.get_param("~lane_path/safety/tracking_margin", 0.002)
            ),
            reaction_time=float(
                rospy.get_param("~lane_path/safety/reaction_time", 0.10)
            ),
            stopping_distance_margin=float(
                rospy.get_param(
                    "~lane_path/safety/stopping_distance_margin", 0.005
                )
            ),
            lookahead_distance=float(
                rospy.get_param("~lane_path/safety/lookahead_distance", 0.45)
            ),
        )
        self.footprint = AsymmetricFootprint(
            front=float(
                rospy.get_param(
                    "~lane_path/safety/footprint/front", 0.067645
                )
            ),
            rear=float(
                rospy.get_param(
                    "~lane_path/safety/footprint/rear", 0.118073
                )
            ),
            half_width=float(
                rospy.get_param(
                    "~lane_path/safety/footprint/half_width", 0.0903
                )
            ),
        )
        self.path_validator = SweptFootprintValidator(
            self.footprint,
            translation_step=float(
                rospy.get_param(
                    "~lane_path/safety/swept_translation_step", 0.004
                )
            ),
            heading_step=math.radians(
                float(
                    rospy.get_param(
                        "~lane_path/safety/swept_heading_step_deg", 1.0
                    )
                )
            ),
        )
        cruise_velocity = float(
            rospy.get_param("~lane_path/control/cruise_velocity", 0.20)
        )
        self.speed_profile = SpeedProfile(
            cruise_velocity=cruise_velocity,
            minimum_velocity=float(
                rospy.get_param("~lane_path/control/minimum_velocity", 0.06)
            ),
            # A rolling path starts at the vehicle every camera frame. A low
            # fixed-route entry speed would therefore cap the entire run.
            entry_velocity=float(
                rospy.get_param(
                    "~lane_path/control/entry_velocity", cruise_velocity
                )
            ),
            exit_velocity=float(
                rospy.get_param("~lane_path/control/exit_velocity", 0.10)
            ),
            maximum_angular_velocity=float(
                rospy.get_param(
                    "~lane_path/control/maximum_angular_velocity", 2.0
                )
            ),
            maximum_lateral_acceleration=float(
                rospy.get_param(
                    "~lane_path/control/maximum_lateral_acceleration", 0.16
                )
            ),
            linear_acceleration=float(
                rospy.get_param("~lane_path/control/linear_acceleration", 0.20)
            ),
            linear_deceleration=float(
                rospy.get_param("~lane_path/control/linear_deceleration", 0.35)
            ),
            angular_acceleration=float(
                rospy.get_param("~lane_path/control/angular_acceleration", 2.50)
            ),
        )
        speed_preview_distance = rospy.get_param(
            "~lane_path/control/speed_preview_distance", None
        )
        self.tracking_config = TrackingConfig(
            lookahead_distance=float(
                rospy.get_param("~lane_path/control/lookahead_distance", 0.065)
            ),
            maximum_linear_velocity=float(
                rospy.get_param("~lane_path/control/maximum_linear_velocity", 0.30)
            ),
            maximum_angular_velocity=self.speed_profile.maximum_angular_velocity,
            maximum_lateral_acceleration=(
                self.speed_profile.maximum_lateral_acceleration
            ),
            linear_acceleration=self.speed_profile.linear_acceleration,
            linear_deceleration=self.speed_profile.linear_deceleration,
            angular_acceleration=self.speed_profile.angular_acceleration,
            heading_gain=float(
                rospy.get_param("~lane_path/control/heading_gain", 0.30)
            ),
            curvature_feedforward_weight=float(
                rospy.get_param(
                    "~lane_path/control/curvature_feedforward_weight", 0.15
                )
            ),
            lateral_feedback_gain=float(
                rospy.get_param(
                    "~lane_path/control/lateral_feedback_gain", 1.0
                )
            ),
            search_back=int(rospy.get_param("~lane_path/control/search_back", 3)),
            search_ahead_distance=float(
                rospy.get_param(
                    "~lane_path/control/search_ahead_distance", 0.50
                )
            ),
            lookahead_time=float(
                rospy.get_param("~lane_path/control/lookahead_time", 0.0)
            ),
            minimum_lookahead_distance=float(
                rospy.get_param(
                    "~lane_path/control/minimum_lookahead_distance", 0.0
                )
            ),
            maximum_lookahead_distance=float(
                rospy.get_param(
                    "~lane_path/control/maximum_lookahead_distance", math.inf
                )
            ),
            speed_preview_distance=(
                None
                if speed_preview_distance is None
                else float(speed_preview_distance)
            ),
        )
        configured_enabled = bool(rospy.get_param("~enabled", True))
        self.wait_for_green = bool(rospy.get_param("~wait_for_green", True))
        self.traffic_light_topic = rospy.get_param(
            "~traffic_light_topic", "/detect/traffic_light"
        )
        self.green_signal_value = int(rospy.get_param("~green_signal_value", 2))
        self.green_received = not self.wait_for_green
        self.manual_stop_requested = not configured_enabled
        self.mission_has_control = False
        self.enabled = configured_enabled and self.green_received

        self.lock = threading.Lock()
        self.last_valid_lane_time = None
        self.last_command_time = None
        self.last_command_linear = 0.0
        self.last_command_angular = 0.0
        self.last_lane_speed_limit = 0.0
        self.odom_history = deque(
            maxlen=max(10, int(rospy.get_param("~lane_path/odom_buffer_size", 120)))
        )
        self.path_follower = PathFollower(self.tracking_config)
        self.rolling_path = None
        # A controller-state transition can leave the follower's last command
        # older than the robot velocity produced by the mission that owned
        # cmd_vel.  The first active path after every transition must therefore
        # start its slew limits from fresh odometry, not from that cached command.
        self.initialize_active_path_from_odometry = True
        self.last_lane_diagnostics = None

        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.manual_stop_pub = rospy.Publisher(
            "/control/manual_stop", Bool, queue_size=1, latch=True
        )
        self.diagnostics_pub = rospy.Publisher(
            self.diagnostics_topic, Float64MultiArray, queue_size=1
        )
        self.path_pub = rospy.Publisher(self.path_topic, Path, queue_size=1)
        rospy.Subscriber(
            self.centerline_topic,
            LaneCenterline,
            self.lane_centerline_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.odometry_topic, Odometry, self.odometry_callback, queue_size=1
        )
        rospy.Subscriber(
            "/control/max_vel",
            Float64,
            self.maximum_velocity_callback,
            queue_size=1,
        )
        if self.wait_for_green:
            rospy.Subscriber(
                self.traffic_light_topic,
                UInt8,
                self.traffic_light_callback,
                queue_size=1,
            )
        self.enable_service = rospy.Service(
            "/control/lane_following", SetBool, self.enable_callback
        )
        self.mission_handoff_service = rospy.Service(
            "/control/lane_mission_handoff", SetBool, self.mission_handoff_callback
        )
        self.watchdog = rospy.Timer(
            rospy.Duration(self.control_period), self.watchdog_callback
        )
        rospy.on_shutdown(self.shutdown)
        self.manual_stop_pub.publish(Bool(data=self.manual_stop_requested))

        if self.wait_for_green and not self.green_received:
            rospy.loginfo(
                "Lane following is stopped until green signal %d arrives on %s",
                self.green_signal_value,
                self.traffic_light_topic,
            )
        else:
            rospy.loginfo("Lane following starts enabled")
        rospy.loginfo(
            "Lane rolling path active, centreline=%s, odometry=%s",
            self.centerline_topic,
            self.odometry_topic,
        )
        rospy.loginfo("Stop/resume service: /control/lane_following")

    @staticmethod
    def _stamp_seconds(stamp, fallback):
        if stamp is None:
            return float(fallback)
        try:
            seconds = float(stamp.to_sec())
        except (AttributeError, TypeError, ValueError):
            return float(fallback)
        return seconds if math.isfinite(seconds) and seconds > 0.0 else float(fallback)

    def odometry_callback(self, message):
        pose = message.pose.pose
        twist = message.twist.twist
        values = (
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(pose.orientation),
            float(twist.linear.x),
            float(twist.angular.z),
        )
        if not all(math.isfinite(value) for value in values):
            return
        now = rospy.Time.now().to_sec()
        sample = OdomSample(
            stamp=self._stamp_seconds(message.header.stamp, now),
            pose=Pose2D(*values[:3]),
            linear_velocity=values[3],
            angular_velocity=values[4],
            frame_id=str(message.header.frame_id) or "odom",
        )
        with self.lock:
            self.odom_history.append(sample)

    def _odom_for_stamp_locked(self, stamp, now_seconds):
        if not self.odom_history:
            return None, None, None
        samples = sorted(self.odom_history, key=lambda sample: sample.stamp)
        latest = samples[-1]
        if abs(now_seconds - latest.stamp) > self.odom_timeout:
            return None, None, None

        before = None
        after = None
        for sample in samples:
            if sample.stamp <= stamp:
                before = sample
            if sample.stamp >= stamp:
                after = sample
                break

        if before is not None and abs(before.stamp - stamp) <= 1e-12:
            return before, latest, 0.0
        if after is not None and abs(after.stamp - stamp) <= 1e-12:
            return after, latest, 0.0

        if before is not None and after is not None:
            before_skew = stamp - before.stamp
            after_skew = after.stamp - stamp
            if (
                before.frame_id == after.frame_id
                and before_skew <= self.maximum_pose_stamp_skew
                and after_skew <= self.maximum_pose_stamp_skew
            ):
                capture = interpolate_odom_sample(before, after, stamp)
                # Diagnostics expose the nearest measured odometry support,
                # not the zero timestamp error of the interpolated sample.
                return capture, latest, min(before_skew, after_skew)

        capture = min(samples, key=lambda sample: abs(sample.stamp - stamp))
        skew = abs(capture.stamp - stamp)
        if skew > self.maximum_pose_stamp_skew:
            return None, None, skew
        return capture, latest, skew

    def maximum_velocity_callback(self, message):
        if math.isfinite(message.data) and message.data >= 0.0:
            with self.lock:
                self.maximum_velocity = message.data

    def traffic_light_callback(self, message):
        if message.data != self.green_signal_value:
            return

        with self.lock:
            if self.green_received:
                return
            self.green_received = True
            if not self.manual_stop_requested and not self.mission_has_control:
                self.enabled = True
                self.last_valid_lane_time = None
                self.last_command_time = None
                self.initialize_active_path_from_odometry = True

        if self.manual_stop_requested:
            rospy.loginfo(
                "Green signal received, but the manual stop remains active"
            )
        else:
            rospy.loginfo("Green signal received; lane following enabled")

    def lane_centerline_callback(self, message):
        now = rospy.Time.now()
        now_seconds = now.to_sec()
        stamp = self._stamp_seconds(message.header.stamp, now_seconds)
        if abs(now_seconds - stamp) > self.lane_timeout:
            rospy.logwarn_throttle(1.0, "Rejecting stale lane centreline")
            return

        with self.lock:
            capture, latest, skew = self._odom_for_stamp_locked(stamp, now_seconds)
            if capture is None or latest is None:
                rospy.logwarn_throttle(
                    1.0, "Waiting for synchronized lane-path odometry"
                )
                return
            try:
                selection = build_rolling_lane_path_selection(
                    message,
                    capture.pose,
                    capture.frame_id,
                    self.path_calibration,
                    self.speed_profile,
                    self.path_safety_config,
                )
            except (TypeError, ValueError, np.linalg.LinAlgError) as error:
                rospy.logwarn_throttle(
                    1.0, "Rejecting invalid lane centreline: %s", str(error)
                )
                return
            self.path_pub.publish(
                common_path_message(
                    selection.observation.path, message.header.stamp
                )
            )
            rolling = selection.executable
            if rolling is None:
                rospy.logwarn_throttle(
                    1.0,
                    "Rejecting invalid lane centreline: %s",
                    selection.rejection_reason,
                )
                return

            if (
                self.rolling_path is None
                or not self.enabled
                or self.initialize_active_path_from_odometry
            ):
                initial_linear = latest.linear_velocity
                initial_angular = latest.angular_velocity
            else:
                initial_linear = self.path_follower.last_linear
                initial_angular = self.path_follower.last_angular
            self.rolling_path = rolling.path
            self.path_follower.reset(
                rolling.path,
                latest.pose,
                initial_linear=initial_linear,
                initial_angular=initial_angular,
            )
            tracking = self.path_follower.calculate_tracking(
                latest.pose,
                linear_velocity=latest.linear_velocity,
            )
            desired_speed = min(
                tracking.target_speed, max(0.0, self.maximum_velocity)
            )
            decision = self.path_validator.motion_safety(
                rolling.path,
                latest.pose,
                tracking.path_index,
                desired_speed=desired_speed,
                linear_velocity=latest.linear_velocity,
                angular_velocities=self.path_follower.stopping_angular_velocities(
                    tracking,
                    latest.linear_velocity,
                    latest.angular_velocity,
                ),
                reaction_time=self.path_safety_config.reaction_time,
                linear_deceleration=self.speed_profile.linear_deceleration,
                distance_margin=(
                    self.path_safety_config.stopping_distance_margin
                ),
                lookahead_distance=(
                    self.path_safety_config.lookahead_distance
                ),
                tracking=tracking,
            )
            self.path_follower.update_clearance(
                combine_validation_results((decision.route, decision.stopping))
            )
            self.last_lane_speed_limit = min(
                max(0.0, self.maximum_velocity), decision.speed_limit
            )
            self.last_valid_lane_time = now
            if self.enabled:
                self.initialize_active_path_from_odometry = False
            else:
                self.path_follower.update_tracking(tracking)

            command = None
            if self.enabled:
                elapsed = self.control_period
                if self.last_command_time is not None:
                    elapsed = max(
                        0.0,
                        min(0.20, (now - self.last_command_time).to_sec()),
                    )
                limited, tracking = self.path_follower.command(
                    latest.pose,
                    elapsed,
                    speed_limit=self.last_lane_speed_limit,
                    tracking=tracking,
                )
                command = Twist()
                command.linear.x = limited.linear_velocity
                command.angular.z = limited.angular_velocity
                self.cmd_vel_pub.publish(command)
                self.last_command_time = now
                self.last_command_linear = command.linear.x
                self.last_command_angular = command.angular.z

            self._publish_path_diagnostics_locked()

    def _publish_path_diagnostics_locked(self):
        """Publish the shared 13-field ``PathDiagnostics`` contract."""

        message = Float64MultiArray()
        message.data = self.path_follower.diagnostics.as_array()
        self.last_lane_diagnostics = list(message.data)
        self.diagnostics_pub.publish(message)

    def enable_callback(self, request):
        with self.lock:
            self.manual_stop_requested = not request.data
            self.enabled = (
                request.data
                and self.green_received
                and not self.mission_has_control
            )
            self.last_valid_lane_time = None
            self.last_command_time = None
            self.initialize_active_path_from_odometry = True
            # Only the controller that owns cmd_vel publishes the stop.
            if not request.data and not self.mission_has_control:
                self.publish_stop_locked()
        self.manual_stop_pub.publish(Bool(data=self.manual_stop_requested))

        if request.data and self.mission_has_control:
            state = "resume requested; mission still has control"
        else:
            state = "enabled" if self.enabled else "stopped"
        rospy.loginfo("Lane following %s by service request", state)
        return SetBoolResponse(success=True, message=state)

    def mission_handoff_callback(self, request):
        """Transfer cmd_vel ownership without injecting a stop command."""
        now = rospy.Time.now()
        with self.lock:
            if request.data and self.mission_has_control:
                cache_age = (
                    math.inf
                    if self.last_valid_lane_time is None
                    else (now - self.last_valid_lane_time).to_sec()
                )
                maximum_cache_age = max(
                    0.0, self.lane_timeout - self.control_period
                )
                lane_cache_ready = bool(
                    self.enabled is False
                    and self.green_received
                    and not self.manual_stop_requested
                    and self.rolling_path is not None
                    and self.path_follower.path is self.rolling_path
                    and 0.0 <= cache_age <= maximum_cache_age
                    and self.last_lane_speed_limit > 1e-9
                )
                if not lane_cache_ready:
                    return SetBoolResponse(
                        success=False,
                        message=(
                            "retryable: fresh executable lane path is not ready"
                        ),
                    )
            self.mission_has_control = not request.data
            self.enabled = (
                request.data
                and self.green_received
                and not self.manual_stop_requested
            )
            self.last_command_time = None
            self.initialize_active_path_from_odometry = True
            if not self.enabled:
                # Camera callbacks may repopulate the rolling path cache while
                # a mission owns cmd_vel.
                self.last_valid_lane_time = None
            elif (
                self.last_valid_lane_time is None
                or (now - self.last_valid_lane_time).to_sec() > self.lane_timeout
            ):
                self.last_valid_lane_time = None

        owner = "lane controller" if request.data else "active mission controller"
        rospy.loginfo("cmd_vel control handed to %s", owner)
        return SetBoolResponse(success=True, message=owner)

    def watchdog_callback(self, _event):
        now = rospy.Time.now()
        with self.lock:
            if self.mission_has_control:
                return
            lane_is_stale = (
                self.last_valid_lane_time is None
                or (now - self.last_valid_lane_time).to_sec() > self.lane_timeout
            )
            if not self.enabled:
                self.publish_stop_locked()
            elif lane_is_stale:
                self.publish_stop_locked()
                rospy.logwarn_throttle(
                    2.0,
                    "Lane path is stale; holding zero velocity",
                )

    def publish_stop_locked(self):
        self.cmd_vel_pub.publish(Twist())
        self.last_command_linear = 0.0
        self.last_command_angular = 0.0
        if getattr(self, "path_follower", None) is not None:
            self.path_follower.last_linear = 0.0
            self.path_follower.last_angular = 0.0

    def shutdown(self):
        with self.lock:
            lane_owned_control = not self.mission_has_control
            if lane_owned_control:
                self.publish_stop_locked()
        if lane_owned_control:
            rospy.loginfo("Safe lane controller stopped; cmd_vel is zero")
        else:
            rospy.loginfo(
                "Safe lane controller stopped without publishing during "
                "mission cmd_vel ownership"
            )


if __name__ == "__main__":
    rospy.init_node("safe_lane_controller")
    SafeLaneController()
    rospy.spin()
