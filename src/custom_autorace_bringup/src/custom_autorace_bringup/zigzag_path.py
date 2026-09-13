#!/usr/bin/env python3
"""ROS-independent surveyed spline and lookahead helpers for zigzag driving."""

from dataclasses import dataclass
import math

import numpy as np

from custom_autorace_bringup.path_following import (
    CommonPath,
    GoalTolerance,
    PathSafety,
    Pose2D,
    SpeedProfile,
    build_speed_profile as build_common_speed_profile,
    footprint_points,
)


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _signed_curve_clearance(
    world_x,
    world_y,
    boundary_x,
    boundary_y,
    keep_above,
):
    """Signed perpendicular clearance to an x-monotone polyline."""
    indices = np.searchsorted(boundary_x, world_x, side="right") - 1
    indices = np.clip(indices, 0, boundary_x.size - 2)
    start_x = boundary_x[indices]
    end_x = boundary_x[indices + 1]
    start_y = boundary_y[indices]
    end_y = boundary_y[indices + 1]
    slope = (end_y - start_y) / (end_x - start_x)
    line_y = start_y + slope * (world_x - start_x)
    vertical = world_y - line_y if keep_above else line_y - world_y
    return vertical / np.sqrt(1.0 + slope * slope)


@dataclass(frozen=True)
class TrackingResult:
    path_index: int
    target_index: int
    position_error: float
    heading_error: float
    target_speed: float
    curvature_command: float
    angular_velocity: float


@dataclass(frozen=True)
class LimitedCommand:
    linear_velocity: float
    angular_velocity: float


def _clamped_second_derivatives(parameter, values, start_derivative, end_derivative):
    count = int(parameter.size)
    intervals = np.diff(parameter)
    matrix = np.zeros((count, count), dtype=np.float64)
    right_hand = np.zeros(count, dtype=np.float64)

    matrix[0, 0] = 2.0 * intervals[0]
    matrix[0, 1] = intervals[0]
    right_hand[0] = 6.0 * (
        (values[1] - values[0]) / intervals[0] - start_derivative
    )
    for index in range(1, count - 1):
        previous = intervals[index - 1]
        following = intervals[index]
        matrix[index, index - 1] = previous
        matrix[index, index] = 2.0 * (previous + following)
        matrix[index, index + 1] = following
        right_hand[index] = 6.0 * (
            (values[index + 1] - values[index]) / following
            - (values[index] - values[index - 1]) / previous
        )
    matrix[-1, -2] = intervals[-1]
    matrix[-1, -1] = 2.0 * intervals[-1]
    right_hand[-1] = 6.0 * (
        end_derivative
        - (values[-1] - values[-2]) / intervals[-1]
    )
    return np.linalg.solve(matrix, right_hand)


def _sample_clamped_spline(knots, start_heading, end_heading, sample_spacing):
    points = np.asarray(knots, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("zigzag path needs at least three [x, y] knots")
    if not np.all(np.isfinite(points)):
        raise ValueError("zigzag path knots must be finite")

    chord = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    if np.any(chord <= 1e-5):
        raise ValueError("zigzag path contains duplicate knots")
    parameter = np.concatenate(([0.0], np.cumsum(chord)))
    start_dx, start_dy = math.cos(start_heading), math.sin(start_heading)
    end_dx, end_dy = math.cos(end_heading), math.sin(end_heading)
    second_x = _clamped_second_derivatives(
        parameter, points[:, 0], start_dx, end_dx
    )
    second_y = _clamped_second_derivatives(
        parameter, points[:, 1], start_dy, end_dy
    )

    sampled_x = []
    sampled_y = []
    sampled_dx = []
    sampled_dy = []
    sampled_ddx = []
    sampled_ddy = []
    for index, interval in enumerate(chord):
        count = max(2, int(math.ceil(interval / sample_spacing)) + 1)
        local_parameter = np.linspace(
            parameter[index],
            parameter[index + 1],
            count,
            endpoint=index == chord.size - 1,
        )
        left = (parameter[index + 1] - local_parameter) / interval
        right = (local_parameter - parameter[index]) / interval
        sampled_x.append(
            left * points[index, 0]
            + right * points[index + 1, 0]
            + (
                (left ** 3 - left) * second_x[index]
                + (right ** 3 - right) * second_x[index + 1]
            )
            * interval ** 2
            / 6.0
        )
        sampled_y.append(
            left * points[index, 1]
            + right * points[index + 1, 1]
            + (
                (left ** 3 - left) * second_y[index]
                + (right ** 3 - right) * second_y[index + 1]
            )
            * interval ** 2
            / 6.0
        )
        sampled_dx.append(
            (points[index + 1, 0] - points[index, 0]) / interval
            + interval
            / 6.0
            * (
                -(3.0 * left ** 2 - 1.0) * second_x[index]
                + (3.0 * right ** 2 - 1.0) * second_x[index + 1]
            )
        )
        sampled_dy.append(
            (points[index + 1, 1] - points[index, 1]) / interval
            + interval
            / 6.0
            * (
                -(3.0 * left ** 2 - 1.0) * second_y[index]
                + (3.0 * right ** 2 - 1.0) * second_y[index + 1]
            )
        )
        sampled_ddx.append(
            left * second_x[index] + right * second_x[index + 1]
        )
        sampled_ddy.append(
            left * second_y[index] + right * second_y[index + 1]
        )

    x = np.concatenate(sampled_x)
    y = np.concatenate(sampled_y)
    dx = np.concatenate(sampled_dx)
    dy = np.concatenate(sampled_dy)
    ddx = np.concatenate(sampled_ddx)
    ddy = np.concatenate(sampled_ddy)
    derivative_norm = np.hypot(dx, dy)
    if float(np.min(derivative_norm)) <= 1e-6:
        raise ValueError("zigzag spline contains a zero-length tangent")
    heading = np.arctan2(dy, dx)
    curvature = (dx * ddy - dy * ddx) / derivative_norm ** 3
    station = np.concatenate(
        ([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y))))
    )
    return x, y, heading, curvature, station


def build_speed_profile(
    station,
    curvature,
    cruise_velocity,
    minimum_velocity,
    entry_velocity,
    exit_velocity,
    maximum_angular_velocity,
    maximum_lateral_acceleration,
    linear_acceleration,
    linear_deceleration,
    maximum_angular_acceleration=math.inf,
):
    count = int(station.size)
    if count < 2:
        return np.full(count, minimum_velocity, dtype=np.float64)
    segment = np.diff(station)
    absolute_curvature = np.maximum(np.abs(curvature), 1e-6)
    angular_limit = maximum_angular_velocity / absolute_curvature
    lateral_limit = np.sqrt(maximum_lateral_acceleration / absolute_curvature)
    speed = np.minimum.reduce(
        (
            np.full(count, cruise_velocity, dtype=np.float64),
            angular_limit,
            lateral_limit,
        )
    )
    speed = np.maximum(speed, minimum_velocity)
    speed[0] = min(speed[0], entry_velocity)
    speed[-1] = min(speed[-1], exit_velocity)

    def apply_linear_limits():
        for index in range(count - 2, -1, -1):
            reachable = math.sqrt(
                max(
                    0.0,
                    speed[index + 1] ** 2
                    + 2.0 * linear_deceleration * segment[index],
                )
            )
            speed[index] = min(speed[index], reachable)
        for index in range(count - 1):
            reachable = math.sqrt(
                max(
                    0.0,
                    speed[index] ** 2
                    + 2.0 * linear_acceleration * segment[index],
                )
            )
            speed[index + 1] = min(speed[index + 1], reachable)

    # The backward pass is the curvature look-ahead: a low corner speed is
    # propagated toward earlier samples by the configured deceleration bound.
    for _ in range(3):
        apply_linear_limits()

    maximum_angular_acceleration = float(maximum_angular_acceleration)
    if math.isfinite(maximum_angular_acceleration):
        maximum_angular_acceleration = max(0.05, maximum_angular_acceleration)
        # omega = v * curvature. Reducing both end speeds of an offending
        # segment by sqrt(limit / measured) reduces d(omega)/dt by the same
        # ratio squared. Longitudinal passes then propagate the braking ahead.
        for _ in range(100):
            duration = 2.0 * segment / np.maximum(
                speed[:-1] + speed[1:], 1e-6
            )
            omega = speed * curvature
            angular_acceleration = np.abs(np.diff(omega)) / np.maximum(
                duration, 1e-6
            )
            offending = np.flatnonzero(
                angular_acceleration
                > maximum_angular_acceleration * (1.0 + 1e-9)
            )
            if offending.size == 0:
                break
            for index in offending:
                scale = math.sqrt(
                    maximum_angular_acceleration
                    / max(float(angular_acceleration[index]), 1e-12)
                )
                scale *= 1.0 - 1e-6
                speed[index] *= scale
                speed[index + 1] *= scale
            apply_linear_limits()
        else:
            raise ValueError(
                "zigzag speed profile cannot satisfy angular acceleration"
            )
    return speed


def build_zigzag_path(
    knots,
    start_heading,
    end_heading,
    sample_spacing,
    cruise_velocity,
    minimum_velocity,
    entry_velocity,
    exit_velocity,
    maximum_angular_velocity,
    maximum_lateral_acceleration,
    linear_acceleration,
    linear_deceleration,
    guide_tail_length=0.0,
    maximum_angular_acceleration=math.inf,
    *,
    common_speed_profile=None,
    frame_id="map",
    goal_tolerance=None,
    safety=None,
    label="",
):
    sample_spacing = max(0.001, float(sample_spacing))
    cruise_velocity = max(0.005, float(cruise_velocity))
    minimum_velocity = clamp(float(minimum_velocity), 0.005, cruise_velocity)
    entry_velocity = clamp(float(entry_velocity), 0.005, cruise_velocity)
    exit_velocity = clamp(float(exit_velocity), 0.005, cruise_velocity)
    maximum_angular_velocity = max(0.05, float(maximum_angular_velocity))
    maximum_lateral_acceleration = max(
        0.005, float(maximum_lateral_acceleration)
    )
    linear_acceleration = max(0.005, float(linear_acceleration))
    linear_deceleration = max(0.005, float(linear_deceleration))

    x, y, heading, curvature, station = _sample_clamped_spline(
        knots,
        float(start_heading),
        float(end_heading),
        sample_spacing,
    )
    guide_tail_length = max(0.0, float(guide_tail_length))
    if guide_tail_length > 0.0:
        active_length = float(station[-1]) - guide_tail_length
        if active_length <= sample_spacing:
            raise ValueError("zigzag guide tail consumes the complete path")
        right = int(np.searchsorted(station, active_length, side="left"))
        if right >= station.size:
            raise ValueError("zigzag guide tail does not leave an active endpoint")
        left = max(0, right - 1)
        span = float(station[right] - station[left])
        fraction = clamp(
            (active_length - float(station[left])) / max(span, 1e-12),
            0.0,
            1.0,
        )

        def cropped(values, terminal):
            return np.concatenate((values[: left + 1], [float(terminal)]))

        terminal_x = (1.0 - fraction) * x[left] + fraction * x[right]
        terminal_y = (1.0 - fraction) * y[left] + fraction * y[right]
        terminal_heading = math.atan2(
            (1.0 - fraction) * math.sin(float(heading[left]))
            + fraction * math.sin(float(heading[right])),
            (1.0 - fraction) * math.cos(float(heading[left]))
            + fraction * math.cos(float(heading[right])),
        )
        terminal_curvature = (
            (1.0 - fraction) * curvature[left]
            + fraction * curvature[right]
        )
        x = cropped(x, terminal_x)
        y = cropped(y, terminal_y)
        heading = cropped(heading, terminal_heading)
        curvature = cropped(curvature, terminal_curvature)
        station = cropped(station, active_length)

    if common_speed_profile is None:
        # Tunnel and experimental lane code still use the historical helper.
        # Keep its exact numerical behaviour until those callers migrate.
        speed = build_speed_profile(
            station,
            curvature,
            cruise_velocity,
            minimum_velocity,
            entry_velocity,
            exit_velocity,
            maximum_angular_velocity,
            maximum_lateral_acceleration,
            linear_acceleration,
            linear_deceleration,
            maximum_angular_acceleration,
        )
    else:
        if not isinstance(common_speed_profile, SpeedProfile):
            raise ValueError("common_speed_profile must be SpeedProfile")
        speed = build_common_speed_profile(
            station,
            curvature,
            common_speed_profile,
        )
    return CommonPath(
        x=x,
        y=y,
        heading=heading,
        curvature=curvature,
        station=station,
        speed=speed,
        direction=1,
        frame_id=str(frame_id),
        goal_tolerance=goal_tolerance or GoalTolerance(),
        safety=safety or PathSafety(),
        label=str(label),
    )


def transformed_path(path, translation_x, translation_y, rotation):
    cosine = math.cos(rotation)
    sine = math.sin(rotation)
    return CommonPath(
        x=translation_x + cosine * path.x - sine * path.y,
        y=translation_y + sine * path.x + cosine * path.y,
        heading=np.asarray(
            [normalize_angle(rotation + value) for value in path.heading],
            dtype=np.float64,
        ),
        curvature=path.curvature.copy(),
        station=path.station.copy(),
        speed=path.speed.copy(),
    )


def committed_route_path(path, route_pose, odom_pose, odom_aligned=False):
    """Freeze a surveyed route in odom and return the inverse pose transform.

    Gazebo's world odometry already uses the coordinate system in which its
    course texture was surveyed.  In that case an AMCL correction must not
    move the millimetre-clearance driving path.  On hardware, the surveyed
    route is normally in ``map`` and the synchronized map/odom pose pair is
    still used to freeze it once at handoff.

    The second return value transforms a live odom pose back into the route
    frame for the footprint and paint guards.
    """
    if odom_aligned:
        return transformed_path(path, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    route_x, route_y, route_yaw = (float(value) for value in route_pose)
    odom_x, odom_y, odom_yaw = (float(value) for value in odom_pose)
    route_to_odom_yaw = normalize_angle(odom_yaw - route_yaw)
    cosine = math.cos(route_to_odom_yaw)
    sine = math.sin(route_to_odom_yaw)
    route_to_odom_x = odom_x - (cosine * route_x - sine * route_y)
    route_to_odom_y = odom_y - (sine * route_x + cosine * route_y)

    odom_to_route_yaw = normalize_angle(route_yaw - odom_yaw)
    inverse_cosine = math.cos(odom_to_route_yaw)
    inverse_sine = math.sin(odom_to_route_yaw)
    odom_to_route_x = route_x - (
        inverse_cosine * odom_x - inverse_sine * odom_y
    )
    odom_to_route_y = route_y - (
        inverse_sine * odom_x + inverse_cosine * odom_y
    )
    return (
        transformed_path(
            path,
            route_to_odom_x,
            route_to_odom_y,
            route_to_odom_yaw,
        ),
        (odom_to_route_x, odom_to_route_y, odom_to_route_yaw),
    )


def nearest_path_index(
    path,
    x,
    y,
    previous_index=0,
    search_back=3,
    search_ahead_distance=0.30,
):
    if path.x.size == 0:
        return 0
    first = max(0, min(int(previous_index) - int(search_back), path.x.size - 1))
    maximum_station = (
        float(path.station[max(0, min(int(previous_index), path.x.size - 1))])
        + max(0.05, float(search_ahead_distance))
    )
    last = int(np.searchsorted(path.station, maximum_station, side="right"))
    last = max(first + 1, min(last, path.x.size))
    distance = np.hypot(path.x[first:last] - x, path.y[first:last] - y)
    candidate = first + int(np.argmin(distance))
    return max(int(previous_index), candidate)


def calculate_tracking(
    path,
    x,
    y,
    yaw,
    previous_index,
    lookahead_distance,
    maximum_angular_velocity,
    heading_gain,
    path_curvature_weight,
    search_ahead_distance=0.30,
):
    index = nearest_path_index(
        path,
        x,
        y,
        previous_index,
        search_ahead_distance=search_ahead_distance,
    )
    position_error = math.hypot(path.x[index] - x, path.y[index] - y)
    heading_error = normalize_angle(float(path.heading[index]) - yaw)
    target_station = float(path.station[index]) + max(
        0.01, float(lookahead_distance)
    )
    target = min(
        int(np.searchsorted(path.station, target_station, side="left")),
        path.x.size - 1,
    )
    dx = float(path.x[target]) - x
    dy = float(path.y[target]) - y
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    target_x = cosine * dx + sine * dy
    target_y = -sine * dx + cosine * dy
    distance_squared = max(0.0025, target_x * target_x + target_y * target_y)
    pure_pursuit_curvature = 2.0 * target_y / distance_squared
    weight = clamp(float(path_curvature_weight), 0.0, 1.0)
    curvature_command = (
        (1.0 - weight) * pure_pursuit_curvature
        + weight * float(path.curvature[target])
    )
    target_speed = float(np.min(path.speed[index : target + 1]))
    target_heading_error = normalize_angle(float(path.heading[target]) - yaw)
    angular_velocity = clamp(
        target_speed * curvature_command + heading_gain * target_heading_error,
        -maximum_angular_velocity,
        maximum_angular_velocity,
    )
    return TrackingResult(
        path_index=index,
        target_index=target,
        position_error=position_error,
        heading_error=heading_error,
        target_speed=target_speed,
        curvature_command=curvature_command,
        angular_velocity=angular_velocity,
    )


def limit_tracking_command(
    target_speed,
    reference_speed,
    target_angular_velocity,
    last_linear_velocity,
    last_angular_velocity,
    elapsed,
    linear_acceleration,
    linear_deceleration,
    angular_acceleration,
    maximum_angular_velocity,
    maximum_lateral_acceleration,
):
    """Rate-limit a tracking command without changing its intended curve.

    ``target_angular_velocity`` was calculated at ``reference_speed``. If a
    line or tracking guard lowers the linear target, the same scale must be
    applied to angular velocity; otherwise slowing down makes the driven path
    turn more sharply. When steering cannot ramp up quickly enough, linear
    speed is reduced so the robot does not understeer into the paint.
    """
    target_speed = max(0.0, float(target_speed))
    reference_speed = max(1e-6, abs(float(reference_speed)))
    curvature = float(target_angular_velocity) / reference_speed
    elapsed = max(0.0, float(elapsed))
    linear_acceleration = max(0.0, float(linear_acceleration))
    linear_deceleration = max(0.0, float(linear_deceleration))
    angular_acceleration = max(0.0, float(angular_acceleration))
    maximum_angular_velocity = max(0.0, float(maximum_angular_velocity))
    maximum_lateral_acceleration = max(
        0.0, float(maximum_lateral_acceleration)
    )

    if abs(curvature) > 1e-9:
        target_speed = min(
            target_speed,
            math.sqrt(maximum_lateral_acceleration / abs(curvature)),
        )

    acceleration = (
        linear_acceleration
        if target_speed >= last_linear_velocity
        else linear_deceleration
    )
    linear = clamp(
        target_speed,
        float(last_linear_velocity) - acceleration * elapsed,
        float(last_linear_velocity) + acceleration * elapsed,
    )
    linear = max(0.0, linear)
    if abs(curvature) > 1e-9:
        linear = min(
            linear,
            math.sqrt(maximum_lateral_acceleration / abs(curvature)),
        )

    desired_angular = clamp(
        linear * curvature,
        -maximum_angular_velocity,
        maximum_angular_velocity,
    )
    angular = clamp(
        desired_angular,
        float(last_angular_velocity) - angular_acceleration * elapsed,
        float(last_angular_velocity) + angular_acceleration * elapsed,
    )

    # If the wheel-speed ramp cannot yet create the requested bend, wait with
    # linear motion rather than cutting the corner. Excess angular velocity
    # while unwinding cannot be fixed this way, so it is only bounded below by
    # the lateral-acceleration guard.
    if (
        abs(curvature) > 1e-9
        and abs(angular) + 1e-12 < abs(desired_angular)
    ):
        if angular * curvature > 0.0:
            linear = min(linear, abs(angular / curvature))
            desired_angular = linear * curvature
            angular = clamp(
                desired_angular,
                float(last_angular_velocity) - angular_acceleration * elapsed,
                float(last_angular_velocity) + angular_acceleration * elapsed,
            )
        else:
            linear = 0.0

    if abs(angular) > 1e-9:
        linear = min(linear, maximum_lateral_acceleration / abs(angular))
    return LimitedCommand(linear, angular)


class SurveyedCorridorChecker:
    """Provide the surveyed corridor as a signed-clearance boundary field.

    ``clearance`` deliberately evaluates one pose only. Continuous path and
    motion sweeps belong to ``SweptFootprintValidator``.
    """

    def __init__(
        self,
        lower_boundary,
        upper_boundary,
    ):
        lower = np.asarray(lower_boundary, dtype=np.float64)
        upper = np.asarray(upper_boundary, dtype=np.float64)
        if (
            lower.ndim != 2
            or upper.ndim != 2
            or lower.shape[1] != 2
            or upper.shape[1] != 2
            or lower.shape[0] < 2
            or upper.shape[0] < 2
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
        ):
            raise ValueError(
                "zigzag corridor boundaries need finite [x, y] points"
            )
        self.lower = lower[np.argsort(lower[:, 0])]
        self.upper = upper[np.argsort(upper[:, 0])]
        self.minimum_x = max(
            float(self.lower[0, 0]), float(self.upper[0, 0])
        )
        self.maximum_x = min(
            float(self.lower[-1, 0]), float(self.upper[-1, 0])
        )
        if self.minimum_x >= self.maximum_x:
            raise ValueError("zigzag corridor boundaries do not overlap")

    def map_clearance(self, pose, footprint):
        """Return longitudinal clearance inside the surveyed map extent."""
        pose = Pose2D.from_value(pose)
        cosine = math.cos(pose.yaw)
        sine = math.sin(pose.yaw)
        longitudinal = (-footprint.rear * cosine, footprint.front * cosine)
        lateral_extent = abs(sine) * footprint.half_width
        minimum_x = pose.x + min(longitudinal) - lateral_extent
        maximum_x = pose.x + max(longitudinal) + lateral_extent
        return min(
            minimum_x - self.minimum_x,
            self.maximum_x - maximum_x,
        )

    def line_clearance(
        self,
        pose,
        footprint,
        footprint_sample_spacing=0.008,
        inner_blocking=True,
    ):
        """Return clearance to the two surveyed painted references."""
        pose = Pose2D.from_value(pose)
        if not inner_blocking:
            return math.inf
        points = footprint_points(
            pose,
            footprint,
            spacing=footprint_sample_spacing,
            perimeter_only=True,
        )
        world_x = points[:, 0]
        world_y = points[:, 1]
        if (
            float(np.min(world_x)) < self.minimum_x - 1e-9
            or float(np.max(world_x)) > self.maximum_x + 1e-9
        ):
            # The separate map boundary owns the finite surveyed x range.
            return math.inf
        return float(
            min(
                np.min(
                    _signed_curve_clearance(
                        world_x,
                        world_y,
                        self.lower[:, 0],
                        self.lower[:, 1],
                        True,
                    )
                ),
                np.min(
                    _signed_curve_clearance(
                        world_x,
                        world_y,
                        self.upper[:, 0],
                        self.upper[:, 1],
                        False,
                    )
                ),
            )
        )

    def clearance(
        self,
        pose,
        footprint,
        footprint_sample_spacing=0.008,
        inner_blocking=True,
    ):
        """Compatibility aggregate of map-extent and painted-line clearance."""
        return min(
            self.map_clearance(pose, footprint),
            self.line_clearance(
                pose,
                footprint,
                footprint_sample_spacing=footprint_sample_spacing,
                inner_blocking=inner_blocking,
            ),
        )

class RasterPaintCorridorChecker:
    """Measure footprint intrusion and outer reserve from a course texture.

    The sparse surveyed curves select the intended yellow and white stripe.
    Pixel cells, including their finite area, then define the actual inner and
    outer paint edges. This is intended for deterministic Gazebo textures;
    hardware must supply separately measured physical boundaries.
    """

    def __init__(
        self,
        rgb_image,
        course_size,
        course_yaw,
        lower_reference,
        upper_reference,
        front,
        rear,
        half_width,
        footprint_sample_spacing=0.004,
        boundary_x_step=0.002,
        boundary_y_step=0.0005,
        line_search_half_width=0.080,
        color_threshold=128,
        color_tolerance=4,
        yellow_blue_maximum=16,
        minimum_x=None,
        maximum_x=None,
    ):
        image = np.asarray(rgb_image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("zigzag course texture must be an RGB image")
        image = image[:, :, :3].astype(np.int16, copy=False)
        self.image_height, self.image_width = image.shape[:2]
        self.course_size = max(0.1, float(course_size))
        self.course_yaw = float(course_yaw)
        self.course_cosine = math.cos(self.course_yaw)
        self.course_sine = math.sin(self.course_yaw)

        red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        minimum = np.minimum.reduce((red, green, blue))
        maximum = np.maximum.reduce((red, green, blue))
        threshold = int(color_threshold)
        tolerance = max(0, int(color_tolerance))
        self.white_mask = (minimum >= threshold) & (
            maximum - minimum <= tolerance
        )
        self.yellow_mask = (
            (red >= threshold)
            & (green >= threshold)
            & (blue <= int(yellow_blue_maximum))
            & (np.abs(red - green) <= tolerance)
        )

        lower = np.asarray(lower_reference, dtype=np.float64)
        upper = np.asarray(upper_reference, dtype=np.float64)
        if (
            lower.ndim != 2
            or upper.ndim != 2
            or lower.shape[1] != 2
            or upper.shape[1] != 2
            or lower.shape[0] < 2
            or upper.shape[0] < 2
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
        ):
            raise ValueError("zigzag raster checker needs surveyed references")
        lower = lower[np.argsort(lower[:, 0])]
        upper = upper[np.argsort(upper[:, 0])]
        common_minimum_x = max(float(lower[0, 0]), float(upper[0, 0]))
        common_maximum_x = min(float(lower[-1, 0]), float(upper[-1, 0]))
        if minimum_x is not None:
            common_minimum_x = max(common_minimum_x, float(minimum_x))
        if maximum_x is not None:
            common_maximum_x = min(common_maximum_x, float(maximum_x))
        if common_minimum_x >= common_maximum_x:
            raise ValueError("zigzag raster boundary range is empty")
        x_step = max(0.0005, float(boundary_x_step))
        count = max(
            2,
            int(
                math.ceil(
                    (common_maximum_x - common_minimum_x) / x_step
                )
            )
            + 1,
        )
        self.boundary_x = np.linspace(
            common_minimum_x, common_maximum_x, count
        )
        lower_reference_y = np.interp(
            self.boundary_x, lower[:, 0], lower[:, 1]
        )
        upper_reference_y = np.interp(
            self.boundary_x, upper[:, 0], upper[:, 1]
        )

        y_step = max(0.00025, float(boundary_y_step))
        search_half_width = max(0.02, float(line_search_half_width))
        pixel_diagonal = math.hypot(
            self.course_size / self.image_width,
            self.course_size / self.image_height,
        )
        maximum_gap_samples = int(math.floor(pixel_diagonal / y_step)) + 1

        lower_outer = []
        lower_inner = []
        upper_inner = []
        upper_outer = []
        for x, lower_y, upper_y in zip(
            self.boundary_x, lower_reference_y, upper_reference_y
        ):
            lower_band = self._stripe_band_at_x(
                self.yellow_mask,
                float(x),
                float(lower_y),
                search_half_width,
                y_step,
                maximum_gap_samples,
            )
            upper_band = self._stripe_band_at_x(
                self.white_mask,
                float(x),
                float(upper_y),
                search_half_width,
                y_step,
                maximum_gap_samples,
            )
            lower_outer.append(lower_band[0])
            lower_inner.append(lower_band[1])
            upper_inner.append(upper_band[0])
            upper_outer.append(upper_band[1])

        self.lower_outer = np.asarray(lower_outer, dtype=np.float64)
        self.lower_inner = np.asarray(lower_inner, dtype=np.float64)
        self.upper_inner = np.asarray(upper_inner, dtype=np.float64)
        self.upper_outer = np.asarray(upper_outer, dtype=np.float64)
        if np.any(self.lower_inner >= self.upper_inner):
            raise ValueError("zigzag raster paint edges cross")

        edge_step = max(0.001, float(footprint_sample_spacing))
        longitudinal_count = max(
            2, int(math.ceil((front + rear) / edge_step)) + 1
        )
        lateral_count = max(
            2, int(math.ceil(2.0 * half_width / edge_step)) + 1
        )
        local_edge = []
        for longitudinal in np.linspace(-rear, front, longitudinal_count):
            local_edge.extend(
                ((longitudinal, -half_width), (longitudinal, half_width))
            )
        for lateral in np.linspace(-half_width, half_width, lateral_count):
            local_edge.extend(((-rear, lateral), (front, lateral)))
        self.local_edge = np.asarray(local_edge, dtype=np.float64)

    def clearance(
        self,
        pose,
        footprint,
        footprint_sample_spacing=0.004,
        minimum_outer_reserve=0.0,
        inner_blocking=True,
    ):
        """Return one-pose paint clearance for ``CallbackBoundary``.

        The fixed route travels from larger to smaller x. Paint data becomes
        authoritative once the whole expanded footprint enters the extracted
        raster range. Leaving its lower-x end is unsafe. Inter-pose coverage is
        exclusively the common validator's responsibility.
        """
        pose = Pose2D.from_value(pose)
        points = footprint_points(
            pose,
            footprint,
            spacing=footprint_sample_spacing,
            perimeter_only=True,
        )
        world_x = points[:, 0]
        world_y = points[:, 1]
        minimum_x = float(self.boundary_x[0])
        maximum_x = float(self.boundary_x[-1])
        if float(np.min(world_x)) < minimum_x - 1e-9:
            return -math.inf
        if float(np.max(world_x)) > maximum_x + 1e-9:
            return math.inf

        lower_outer = _signed_curve_clearance(
            world_x,
            world_y,
            self.boundary_x,
            self.lower_outer,
            True,
        )
        upper_outer = _signed_curve_clearance(
            world_x,
            world_y,
            self.boundary_x,
            self.upper_outer,
            False,
        )
        clearances = [
            float(np.min(lower_outer)) - float(minimum_outer_reserve),
            float(np.min(upper_outer)) - float(minimum_outer_reserve),
        ]
        if inner_blocking:
            lower_inner = _signed_curve_clearance(
                world_x,
                world_y,
                self.boundary_x,
                self.lower_inner,
                True,
            )
            upper_inner = _signed_curve_clearance(
                world_x,
                world_y,
                self.boundary_x,
                self.upper_inner,
                False,
            )
            clearances.extend(
                (float(np.min(lower_inner)), float(np.min(upper_inner)))
            )
        return min(clearances)

    def _world_to_pixel(self, x, y):
        local_x = self.course_cosine * x + self.course_sine * y
        local_y = -self.course_sine * x + self.course_cosine * y
        u = np.floor(
            (local_x / self.course_size + 0.5) * self.image_width
        ).astype(np.int64)
        v = np.floor(
            (0.5 - local_y / self.course_size) * self.image_height
        ).astype(np.int64)
        return u, v

    def _stripe_band_at_x(
        self,
        mask,
        x,
        reference_y,
        search_half_width,
        y_step,
        maximum_gap_samples,
    ):
        count = int(math.ceil(2.0 * search_half_width / y_step)) + 1
        world_y = np.linspace(
            reference_y - search_half_width,
            reference_y + search_half_width,
            count,
        )
        world_x = np.full(world_y.size, x, dtype=np.float64)
        u, v = self._world_to_pixel(world_x, world_y)
        valid = (
            (u >= 0)
            & (u < self.image_width)
            & (v >= 0)
            & (v < self.image_height)
        )
        painted = np.zeros(world_y.size, dtype=bool)
        painted[valid] = mask[v[valid], u[valid]]
        indices = np.flatnonzero(painted)
        if indices.size == 0:
            raise ValueError("surveyed zigzag stripe is absent from texture")
        cuts = np.r_[
            0,
            np.flatnonzero(np.diff(indices) > maximum_gap_samples) + 1,
            indices.size,
        ]
        runs = [indices[a:b] for a, b in zip(cuts[:-1], cuts[1:])]
        selected = min(
            runs,
            key=lambda run: abs(
                0.5 * (world_y[run[0]] + world_y[run[-1]]) - reference_y
            ),
        )
        return (
            float(world_y[selected[0]] - 0.5 * y_step),
            float(world_y[selected[-1]] + 0.5 * y_step),
        )

    def _footprint_world(self, x, y, heading):
        cosine = math.cos(float(heading))
        sine = math.sin(float(heading))
        world_x = (
            float(x)
            + cosine * self.local_edge[:, 0]
            - sine * self.local_edge[:, 1]
        )
        world_y = (
            float(y)
            + sine * self.local_edge[:, 0]
            + cosine * self.local_edge[:, 1]
        )
        return world_x, world_y

    def pose_metrics(self, x, y, heading):
        """Return ``(inner_intrusion, outer_edge_reserve)`` in metres."""
        world_x, world_y = self._footprint_world(x, y, heading)
        if (
            float(np.min(world_x)) < float(self.boundary_x[0]) - 1e-9
            or float(np.max(world_x)) > float(self.boundary_x[-1]) + 1e-9
        ):
            return math.inf, -math.inf
        lower_outer = np.interp(world_x, self.boundary_x, self.lower_outer)
        lower_inner = np.interp(world_x, self.boundary_x, self.lower_inner)
        upper_inner = np.interp(world_x, self.boundary_x, self.upper_inner)
        upper_outer = np.interp(world_x, self.boundary_x, self.upper_outer)
        inner_intrusion = max(
            0.0,
            float(np.max(lower_inner - world_y)),
            float(np.max(world_y - upper_inner)),
        )
        outer_reserve = min(
            float(np.min(world_y - lower_outer)),
            float(np.min(upper_outer - world_y)),
        )
        return inner_intrusion, outer_reserve
