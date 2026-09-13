#!/usr/bin/env python3
"""Rectangular collision checks and one curvature-continuous course path.

Coordinates use the local course frame: +x follows the course and +y points
course-left. Collision checks use the robot's oriented, asymmetric rectangle;
a circumscribed circle would incorrectly close the narrow legal passages.
"""

from dataclasses import dataclass
import math

import numpy as np

from custom_autorace_bringup.path_following import (
    AsymmetricFootprint,
    CommonPath,
    GoalTolerance,
    PathSafety,
    SafetyMargins,
    SpeedProfile,
    StraightCorridorBoundary,
    SweptFootprintValidator,
    build_speed_profile,
)


@dataclass(frozen=True)
class Footprint(AsymmetricFootprint):
    """Obstacle-course margins wrapped around the common robot footprint."""

    obstacle_padding: float = 0.0
    line_margin: float = 0.0
    localization_margin: float = 0.0
    tracking_margin: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        margins = (
            self.obstacle_padding,
            self.line_margin,
            self.localization_margin,
            self.tracking_margin,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in margins):
            raise ValueError("footprint margins must be finite and non-negative")


@dataclass
class CourseAlignment:
    progress: float
    lateral: float
    inliers: int
    rms: float


class RectanglePathChecker:
    """Thin course-frame adapter around the common swept-footprint validator."""

    def __init__(
        self,
        footprint,
        collision_check_step=0.008,
        collision_check_angle=math.radians(3.0),
    ):
        self.footprint = footprint
        self.validator = SweptFootprintValidator(
            AsymmetricFootprint(
                footprint.front,
                footprint.rear,
                footprint.half_width,
            ),
            translation_step=collision_check_step,
            heading_step=collision_check_angle,
        )

    @staticmethod
    def safety(obstacle_points, right_line, left_line, footprint):
        return PathSafety(
            line_boundaries=(
                StraightCorridorBoundary(
                    float(right_line),
                    float(left_line),
                ),
            ),
            fixed_obstacles=np.asarray(obstacle_points, dtype=np.float64).reshape(
                (-1, 2)
            ),
            margins=SafetyMargins(
                line=footprint.line_margin,
                obstacle=footprint.obstacle_padding,
                localization=footprint.localization_margin,
                tracking=footprint.tracking_margin,
            ),
        )

def rectangle_surface_points(
    center_x,
    center_y,
    size_x,
    size_y,
    spacing=0.006,
):
    """Return dense points on a rectangular obstacle perimeter."""
    spacing = max(0.001, float(spacing))
    half_x = 0.5 * float(size_x)
    half_y = 0.5 * float(size_y)
    x_values = np.arange(
        center_x - half_x,
        center_x + half_x + 0.5 * spacing,
        spacing,
    )
    y_values = np.arange(
        center_y - half_y,
        center_y + half_y + 0.5 * spacing,
        spacing,
    )
    return np.unique(
        np.vstack(
            (
                np.column_stack(
                    (x_values, np.full_like(x_values, center_y - half_y))
                ),
                np.column_stack(
                    (x_values, np.full_like(x_values, center_y + half_y))
                ),
                np.column_stack(
                    (np.full_like(y_values, center_x - half_x), y_values)
                ),
                np.column_stack(
                    (np.full_like(y_values, center_x + half_x), y_values)
                ),
            )
        ),
        axis=0,
    )


class CourseSplinePlanner:
    """Build and validate the one surveyed obstacle-course path."""

    def __init__(
        self,
        collision_checker,
        validation_footprint,
        knots,
        exit_endpoint,
        exit_heading,
        exit_start_tangent_length,
        exit_end_tangent_length,
        start_slope=0.0,
        end_slope=0.0,
        lateral_scale=1.0,
        sample_spacing=0.004,
        minimum_nominal_clearance=0.0035,
        live_validation_distance=0.45,
        cruise_velocity=0.08,
        minimum_velocity=0.035,
        entry_velocity=None,
        exit_velocity=None,
        maximum_angular_velocity=0.85,
        maximum_lateral_acceleration=0.035,
        linear_acceleration=0.25,
        linear_deceleration=0.80,
        angular_acceleration=0.55,
        goal_position_tolerance=0.04,
        goal_heading_tolerance=math.radians(10.0),
        goal_crossing_max_distance=0.10,
        registration_longitudinal_search=0.18,
        registration_lateral_search=0.14,
        registration_coarse_step=0.02,
        registration_fine_window=0.012,
        registration_fine_step=0.002,
        registration_inlier_distance=0.025,
        registration_minimum_inliers=15,
        registration_maximum_rms=0.015,
        registration_line_exclusion=0.025,
    ):
        self.collision_checker = collision_checker
        self.validation_footprint = validation_footprint
        self.knots = np.asarray(knots, dtype=np.float64)
        if (
            self.knots.ndim != 2
            or self.knots.shape[0] < 3
            or self.knots.shape[1] != 2
            or np.any(np.diff(self.knots[:, 0]) <= 0.0)
        ):
            raise ValueError("course spline knots must be increasing [x, y] pairs")
        self.start_slope = float(start_slope)
        self.end_slope = float(end_slope)
        if not math.isfinite(self.start_slope) or not math.isfinite(
            self.end_slope
        ):
            raise ValueError("course spline endpoint slopes must be finite")
        self.lateral_scale = float(lateral_scale)
        if not math.isfinite(self.lateral_scale):
            raise ValueError("course spline lateral scale must be finite")
        self.exit_endpoint = np.asarray(exit_endpoint, dtype=np.float64)
        self.exit_heading = float(exit_heading)
        self.exit_start_tangent_length = float(exit_start_tangent_length)
        self.exit_end_tangent_length = float(exit_end_tangent_length)
        if (
            self.exit_endpoint.shape != (2,)
            or not np.all(np.isfinite(self.exit_endpoint))
            or not math.isfinite(self.exit_heading)
            or self.exit_start_tangent_length <= 0.0
            or self.exit_end_tangent_length <= 0.0
        ):
            raise ValueError("exit curve geometry must be finite and positive")
        self.sample_spacing = max(0.002, float(sample_spacing))
        self.minimum_nominal_clearance = max(
            0.0, float(minimum_nominal_clearance)
        )
        self.live_validation_distance = max(
            0.10, float(live_validation_distance)
        )
        self.cruise_velocity = max(0.01, float(cruise_velocity))
        self.minimum_velocity = min(
            self.cruise_velocity, max(0.005, float(minimum_velocity))
        )
        self.maximum_angular_velocity = max(
            0.05, float(maximum_angular_velocity)
        )
        self.maximum_lateral_acceleration = max(
            0.001, float(maximum_lateral_acceleration)
        )
        self.linear_acceleration = max(0.01, float(linear_acceleration))
        self.linear_deceleration = max(0.01, float(linear_deceleration))
        self.angular_acceleration = max(0.05, float(angular_acceleration))
        entry_velocity = (
            self.cruise_velocity
            if entry_velocity is None
            else float(entry_velocity)
        )
        exit_velocity = (
            self.cruise_velocity if exit_velocity is None else float(exit_velocity)
        )
        self.speed_profile = SpeedProfile(
            cruise_velocity=self.cruise_velocity,
            minimum_velocity=self.minimum_velocity,
            entry_velocity=entry_velocity,
            exit_velocity=exit_velocity,
            maximum_angular_velocity=self.maximum_angular_velocity,
            maximum_lateral_acceleration=self.maximum_lateral_acceleration,
            linear_acceleration=self.linear_acceleration,
            linear_deceleration=self.linear_deceleration,
            angular_acceleration=self.angular_acceleration,
        )
        self.goal_tolerance = GoalTolerance(
            position=float(goal_position_tolerance),
            heading=float(goal_heading_tolerance),
            terminal_crossing=float(goal_crossing_max_distance),
        )
        self.registration_longitudinal_search = max(
            0.02, float(registration_longitudinal_search)
        )
        self.registration_lateral_search = max(
            0.02, float(registration_lateral_search)
        )
        self.registration_coarse_step = max(
            0.005, float(registration_coarse_step)
        )
        self.registration_fine_step = max(
            0.001, float(registration_fine_step)
        )
        self.registration_fine_window = max(
            self.registration_fine_step,
            float(registration_fine_window),
        )
        self.registration_inlier_distance = max(
            0.005, float(registration_inlier_distance)
        )
        self.registration_minimum_inliers = max(
            5, int(registration_minimum_inliers)
        )
        self.registration_maximum_rms = max(
            0.002, float(registration_maximum_rms)
        )
        self.registration_line_exclusion = max(
            0.0, float(registration_line_exclusion)
        )
        self._template = None
        self._nominal_obstacles = np.empty((0, 2), dtype=np.float64)
        self._registration_obstacles = np.empty((0, 2), dtype=np.float64)
        self._registration_boxes = np.empty((0, 4), dtype=np.float64)
        self._nominal_right_line = -math.inf
        self._nominal_left_line = math.inf

    @staticmethod
    def _clamped_second_derivatives(x, y, start_slope=0.0, end_slope=0.0):
        count = int(x.size)
        intervals = np.diff(x)
        matrix = np.zeros((count, count), dtype=np.float64)
        right_hand = np.zeros(count, dtype=np.float64)
        matrix[0, 0] = 2.0 * intervals[0]
        matrix[0, 1] = intervals[0]
        right_hand[0] = 6.0 * (
            (y[1] - y[0]) / intervals[0] - start_slope
        )
        for index in range(1, count - 1):
            previous = intervals[index - 1]
            following = intervals[index]
            matrix[index, index - 1] = previous
            matrix[index, index] = 2.0 * (previous + following)
            matrix[index, index + 1] = following
            right_hand[index] = 6.0 * (
                (y[index + 1] - y[index]) / following
                - (y[index] - y[index - 1]) / previous
            )
        matrix[-1, -2] = intervals[-1]
        matrix[-1, -1] = 2.0 * intervals[-1]
        right_hand[-1] = 6.0 * (
            end_slope - (y[-1] - y[-2]) / intervals[-1]
        )
        return np.linalg.solve(matrix, right_hand)

    def _make_template(self):
        knot_x = self.knots[:, 0]
        knot_y = self.knots[:, 1] * self.lateral_scale
        second = self._clamped_second_derivatives(
            knot_x,
            knot_y,
            self.start_slope * self.lateral_scale,
            self.end_slope * self.lateral_scale,
        )
        x_values = []
        y_values = []
        slopes = []
        second_values = []
        for index, interval in enumerate(np.diff(knot_x)):
            samples = max(2, int(math.ceil(interval / self.sample_spacing)))
            local_x = np.linspace(
                knot_x[index],
                knot_x[index + 1],
                samples,
                endpoint=index == knot_x.size - 2,
            )
            ratio_right = (local_x - knot_x[index]) / interval
            ratio_left = 1.0 - ratio_right
            local_y = (
                ratio_left * knot_y[index]
                + ratio_right * knot_y[index + 1]
                + (
                    (ratio_left ** 3 - ratio_left) * second[index]
                    + (ratio_right ** 3 - ratio_right) * second[index + 1]
                )
                * interval ** 2
                / 6.0
            )
            local_slope = (
                (knot_y[index + 1] - knot_y[index]) / interval
                + interval
                / 6.0
                * (
                    -(3.0 * ratio_left ** 2 - 1.0) * second[index]
                    + (3.0 * ratio_right ** 2 - 1.0) * second[index + 1]
                )
            )
            local_second = (
                ratio_left * second[index]
                + ratio_right * second[index + 1]
            )
            x_values.append(local_x)
            y_values.append(local_y)
            slopes.append(local_slope)
            second_values.append(local_second)
        x = np.concatenate(x_values)
        y = np.concatenate(y_values)
        slope = np.concatenate(slopes)
        second_derivative = np.concatenate(second_values)
        heading = np.arctan(slope)
        curvature = second_derivative / np.power(1.0 + slope ** 2, 1.5)
        path = CommonPath(
            x=x,
            y=y,
            heading=heading,
            curvature=curvature,
            direction=1,
            frame_id="obstacle_course",
            goal_tolerance=self.goal_tolerance,
            label="obstacle_fixed_spline",
        )
        path = self._append_exit_curve(path)
        path.curvature_variation = float(
            np.sum(np.abs(np.diff(path.curvature)))
        )
        path.speed = build_speed_profile(
            path.station, path.curvature, self.speed_profile
        )
        segment = np.diff(path.station)
        path.expected_time = float(
            np.sum(
                2.0
                * segment
                / np.maximum(path.speed[:-1] + path.speed[1:], 1e-6)
            )
        )
        return path

    def _append_exit_curve(self, path):
        """Append the measured left turn while matching heading and curvature."""
        start = np.asarray([path.x[-1], path.y[-1]], dtype=np.float64)
        end = self.exit_endpoint.copy()
        end[1] *= self.lateral_scale

        start_heading = float(path.heading[-1])
        start_curvature = float(path.curvature[-1])
        start_tangent = self.exit_start_tangent_length * np.asarray(
            [math.cos(start_heading), math.sin(start_heading)], dtype=np.float64
        )
        start_normal = np.asarray(
            [-math.sin(start_heading), math.cos(start_heading)], dtype=np.float64
        )
        start_acceleration = (
            start_curvature
            * self.exit_start_tangent_length ** 2
            * start_normal
        )
        end_tangent = self.exit_end_tangent_length * np.asarray(
            [math.cos(self.exit_heading), math.sin(self.exit_heading)],
            dtype=np.float64,
        )

        controls = np.empty((6, 2), dtype=np.float64)
        controls[0] = start
        controls[1] = start + start_tangent / 5.0
        controls[2] = start_acceleration / 20.0 + 2.0 * controls[1] - start
        controls[5] = end
        controls[4] = end - end_tangent / 5.0
        controls[3] = -end + 2.0 * controls[4]

        control_length = float(
            np.sum(np.linalg.norm(np.diff(controls, axis=0), axis=1))
        )
        count = max(3, int(math.ceil(control_length / self.sample_spacing)) + 1)
        parameter = np.linspace(0.0, 1.0, count)
        opposite = 1.0 - parameter
        points = (
            opposite[:, None] ** 5 * controls[0]
            + 5.0 * (opposite ** 4 * parameter)[:, None] * controls[1]
            + 10.0 * (opposite ** 3 * parameter ** 2)[:, None] * controls[2]
            + 10.0 * (opposite ** 2 * parameter ** 3)[:, None] * controls[3]
            + 5.0 * (opposite * parameter ** 4)[:, None] * controls[4]
            + parameter[:, None] ** 5 * controls[5]
        )
        first = 5.0 * (
            opposite[:, None] ** 4 * (controls[1] - controls[0])
            + 4.0
            * (opposite ** 3 * parameter)[:, None]
            * (controls[2] - controls[1])
            + 6.0
            * (opposite ** 2 * parameter ** 2)[:, None]
            * (controls[3] - controls[2])
            + 4.0
            * (opposite * parameter ** 3)[:, None]
            * (controls[4] - controls[3])
            + parameter[:, None] ** 4 * (controls[5] - controls[4])
        )
        second = 20.0 * (
            opposite[:, None] ** 3
            * (controls[2] - 2.0 * controls[1] + controls[0])
            + 3.0
            * (opposite ** 2 * parameter)[:, None]
            * (controls[3] - 2.0 * controls[2] + controls[1])
            + 3.0
            * (opposite * parameter ** 2)[:, None]
            * (controls[4] - 2.0 * controls[3] + controls[2])
            + parameter[:, None] ** 3
            * (controls[5] - 2.0 * controls[4] + controls[3])
        )
        derivative_norm = np.hypot(first[:, 0], first[:, 1])
        if float(np.min(derivative_norm)) <= 1e-9:
            raise ValueError("exit curve contains a zero-length tangent")
        heading = np.arctan2(first[:, 1], first[:, 0])
        curvature = (
            first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        ) / derivative_norm ** 3

        return CommonPath(
            x=np.concatenate((path.x, points[1:, 0])),
            y=np.concatenate((path.y, points[1:, 1])),
            heading=np.concatenate((path.heading, heading[1:])),
            curvature=np.concatenate((path.curvature, curvature[1:])),
            direction=1,
            frame_id=path.frame_id,
            goal_tolerance=path.goal_tolerance,
            safety=path.safety,
            label=path.label,
        )

    def prepare(self, nominal_obstacle_points, right_line, left_line):
        """Build the surveyed template and verify its complete swept path."""
        nominal_obstacle_points = np.asarray(
            nominal_obstacle_points, dtype=np.float64
        ).reshape((-1, 2))
        self._nominal_obstacles = nominal_obstacle_points.copy()
        self._registration_obstacles = nominal_obstacle_points[::3].copy()
        sorted_x = np.sort(np.unique(nominal_obstacle_points[:, 0]))
        breaks = np.flatnonzero(np.diff(sorted_x) > 0.15) + 1
        x_groups = np.split(sorted_x, breaks)
        boxes = []
        for x_group in x_groups:
            selected = nominal_obstacle_points[
                (nominal_obstacle_points[:, 0] >= x_group[0] - 1e-9)
                & (nominal_obstacle_points[:, 0] <= x_group[-1] + 1e-9)
            ]
            if selected.shape[0] >= 4:
                boxes.append(
                    (
                        float(np.min(selected[:, 0])),
                        float(np.max(selected[:, 0])),
                        float(np.min(selected[:, 1])),
                        float(np.max(selected[:, 1])),
                    )
                )
        self._registration_boxes = np.asarray(
            boxes, dtype=np.float64
        ).reshape((-1, 4))
        self._nominal_right_line = float(right_line)
        self._nominal_left_line = float(left_line)

        path = self._make_template()
        path.safety = self.collision_checker.safety(
            nominal_obstacle_points,
            right_line,
            left_line,
            self.collision_checker.footprint,
        )
        validation = self.collision_checker.validator.validate_path(path)
        if not validation.safe:
            self._template = None
            return False
        obstacle = validation.minimum_obstacle_clearance
        line = validation.minimum_line_clearance
        path.obstacle_clearance = obstacle
        path.line_clearance = line
        if min(obstacle, line) + 1e-9 < self.minimum_nominal_clearance:
            self._template = None
            return False
        self._template = path
        return True

    def _rectangle_surface_matches(self, points, sensor_origin=None):
        """Return distance and face axis, excluding faces hidden from LiDAR."""
        count = points.shape[0]
        if count == 0 or self._registration_boxes.shape[0] == 0:
            return (
                np.full(count, math.inf),
                np.full(count, -1, dtype=np.int32),
            )
        all_distances = []
        all_axes = []
        for minimum_x, maximum_x, minimum_y, maximum_y in self._registration_boxes:
            clipped_y = np.clip(points[:, 1], minimum_y, maximum_y)
            clipped_x = np.clip(points[:, 0], minimum_x, maximum_x)
            faces = np.column_stack(
                (
                    np.hypot(points[:, 0] - minimum_x, points[:, 1] - clipped_y),
                    np.hypot(points[:, 0] - maximum_x, points[:, 1] - clipped_y),
                    np.hypot(points[:, 0] - clipped_x, points[:, 1] - minimum_y),
                    np.hypot(points[:, 0] - clipped_x, points[:, 1] - maximum_y),
                )
            )
            if sensor_origin is not None:
                sensor_x, sensor_y = sensor_origin
                visible = np.zeros(4, dtype=bool)
                if sensor_x <= minimum_x:
                    visible[0] = True
                elif sensor_x >= maximum_x:
                    visible[1] = True
                if sensor_y <= minimum_y:
                    visible[2] = True
                elif sensor_y >= maximum_y:
                    visible[3] = True
                faces[:, ~visible] = math.inf
            face = np.argmin(faces, axis=1)
            all_distances.append(faces[np.arange(count), face])
            all_axes.append((face >= 2).astype(np.int32))
        distance_matrix = np.column_stack(all_distances)
        rectangle = np.argmin(distance_matrix, axis=1)
        distance = distance_matrix[np.arange(count), rectangle]
        axes = np.column_stack(all_axes)[np.arange(count), rectangle]
        return distance, axes

    def align_pose(self, seed_progress, seed_lateral, live_obstacle_points):
        """Refine the AMCL translation with coarse/fine LiDAR matching."""
        points = np.asarray(live_obstacle_points, dtype=np.float64).reshape(
            (-1, 2)
        )
        points = points[np.all(np.isfinite(points), axis=1)]
        if (
            points.shape[0] < self.registration_minimum_inliers
            or self._registration_obstacles.shape[0] < 4
            or not math.isfinite(seed_progress)
            or not math.isfinite(seed_lateral)
        ):
            return None

        minimum_x = float(np.min(self._registration_obstacles[:, 0])) - 0.10
        maximum_x = float(np.max(self._registration_obstacles[:, 0])) + 0.10
        minimum_y = self._nominal_right_line - 0.10
        maximum_y = self._nominal_left_line + 0.10

        def evaluate(progress, lateral):
            absolute = points + np.asarray([progress, lateral])
            usable = (
                (absolute[:, 0] >= minimum_x)
                & (absolute[:, 0] <= maximum_x)
                & (absolute[:, 1] >= minimum_y)
                & (absolute[:, 1] <= maximum_y)
            )
            if self.registration_line_exclusion > 0.0:
                usable &= (
                    np.abs(absolute[:, 1] - self._nominal_right_line)
                    >= self.registration_line_exclusion
                ) & (
                    np.abs(absolute[:, 1] - self._nominal_left_line)
                    >= self.registration_line_exclusion
                )
            if np.count_nonzero(usable) < self.registration_minimum_inliers:
                return None
            distance, axes = self._rectangle_surface_matches(
                absolute[usable], (progress, lateral)
            )
            inlier_mask = distance < self.registration_inlier_distance
            inliers = int(np.count_nonzero(inlier_mask))
            inlier_distance = distance[inlier_mask]
            inlier_mse = (
                float(np.mean(inlier_distance ** 2))
                if inlier_distance.size
                else math.inf
            )
            robust_loss = float(
                np.mean(
                    np.minimum(
                        (distance / self.registration_inlier_distance) ** 2,
                        1.0,
                    )
                )
            )
            prior = 1e-7 * (
                ((progress - seed_progress) / 0.12) ** 2
                + ((lateral - seed_lateral) / 0.10) ** 2
            )
            return (
                robust_loss + prior,
                -inliers,
                inlier_mse,
                axes,
                distance,
            )

        def search(progress_values, lateral_values):
            best = None
            for progress in progress_values:
                for lateral in lateral_values:
                    score = evaluate(float(progress), float(lateral))
                    if score is None:
                        continue
                    candidate = score + (float(progress), float(lateral))
                    if best is None or candidate[:3] + candidate[-2:] < (
                        best[:3] + best[-2:]
                    ):
                        best = candidate
            return best

        coarse = search(
            np.arange(
                seed_progress - self.registration_longitudinal_search,
                seed_progress
                + self.registration_longitudinal_search
                + 0.5 * self.registration_coarse_step,
                self.registration_coarse_step,
            ),
            np.arange(
                seed_lateral - self.registration_lateral_search,
                seed_lateral
                + self.registration_lateral_search
                + 0.5 * self.registration_coarse_step,
                self.registration_coarse_step,
            ),
        )
        if coarse is None:
            return None
        coarse_progress, coarse_lateral = coarse[-2:]
        fine = search(
            np.arange(
                coarse_progress - self.registration_fine_window,
                coarse_progress
                + self.registration_fine_window
                + 0.5 * self.registration_fine_step,
                self.registration_fine_step,
            ),
            np.arange(
                coarse_lateral - self.registration_fine_window,
                coarse_lateral
                + self.registration_fine_window
                + 0.5 * self.registration_fine_step,
                self.registration_fine_step,
            ),
        )
        if fine is None:
            return None
        inliers = int(-fine[1])
        rms = math.sqrt(max(0.0, float(fine[2])))
        axes = fine[3]
        distances = fine[4]
        inlier_mask = distances < self.registration_inlier_distance
        x_normal_count = int(np.count_nonzero(inlier_mask & (axes == 0)))
        y_normal_count = int(np.count_nonzero(inlier_mask & (axes == 1)))
        if (
            inliers < self.registration_minimum_inliers
            or rms > self.registration_maximum_rms
            or x_normal_count < 4
            or y_normal_count < 4
        ):
            return None
        return CourseAlignment(
            progress=float(fine[-2]),
            lateral=float(fine[-1]),
            inliers=inliers,
            rms=rms,
        )

    def stabilize_known_barrier_returns(
        self,
        relative_points,
        sensor_progress,
        sensor_lateral,
        association_distance=None,
        minimum_face_points=4,
    ):
        """De-noise registered barrier faces while preserving other obstacles.

        The fixed path already contains the three registered barrier
        rectangles.  Feeding every noisy ray hit back as an additional point
        obstacle double-counts those rectangles and turns isolated range noise
        into a false collision.  The registration step has already frozen the
        measured course frame in odom, so a coherent return cluster belonging
        to a known face is represented by that face's surveyed plane.  Points
        that do not form a known face remain unchanged and continue to
        represent real, previously unknown obstacles.  A displacement larger
        than the association gate is likewise preserved as live geometry.
        """
        points = np.asarray(relative_points, dtype=np.float64).reshape((-1, 2))
        if points.size == 0 or self._registration_boxes.size == 0:
            return points.copy()
        if not all(
            math.isfinite(value) for value in (sensor_progress, sensor_lateral)
        ):
            return points.copy()

        threshold = (
            self.registration_inlier_distance
            if association_distance is None
            else max(0.0, float(association_distance))
        )
        minimum_face_points = max(2, int(minimum_face_points))
        absolute = points + np.asarray(
            [float(sensor_progress), float(sensor_lateral)], dtype=np.float64
        )
        count = absolute.shape[0]
        candidates = []
        sensor_x = float(sensor_progress)
        sensor_y = float(sensor_lateral)
        for rectangle, bounds in enumerate(self._registration_boxes):
            minimum_x, maximum_x, minimum_y, maximum_y = bounds
            clipped_y = np.clip(absolute[:, 1], minimum_y, maximum_y)
            clipped_x = np.clip(absolute[:, 0], minimum_x, maximum_x)
            distances = np.column_stack(
                (
                    np.hypot(absolute[:, 0] - minimum_x, absolute[:, 1] - clipped_y),
                    np.hypot(absolute[:, 0] - maximum_x, absolute[:, 1] - clipped_y),
                    np.hypot(absolute[:, 0] - clipped_x, absolute[:, 1] - minimum_y),
                    np.hypot(absolute[:, 0] - clipped_x, absolute[:, 1] - maximum_y),
                )
            )

            # A rectangle corner is compatible with both adjoining faces.
            # Select the face supported by the complete scan cluster instead
            # of assigning each noisy corner return to its nearest axis.
            visible = (
                sensor_x <= minimum_x,
                sensor_x >= maximum_x,
                sensor_y <= minimum_y,
                sensor_y >= maximum_y,
            )
            for face in range(4):
                if not visible[face]:
                    continue
                selected = distances[:, face] <= threshold
                support = int(np.count_nonzero(selected))
                if support >= minimum_face_points:
                    candidates.append(
                        (support, rectangle, face, selected)
                    )

        stabilized = absolute.copy()
        claimed = np.zeros(count, dtype=bool)
        for _, rectangle, face, compatible in sorted(
            candidates, key=lambda candidate: candidate[0], reverse=True
        ):
            selected = compatible & ~claimed
            if np.count_nonzero(selected) < minimum_face_points:
                continue
            bounds = self._registration_boxes[rectangle]
            minimum_x, maximum_x, minimum_y, maximum_y = bounds
            face_positions = (minimum_x, maximum_x, minimum_y, maximum_y)
            nominal_position = face_positions[face]
            axis = 0 if face < 2 else 1
            tangent_axis = 1 - axis
            stabilized[selected, axis] = nominal_position
            tangent_minimum, tangent_maximum = (
                (minimum_y, maximum_y)
                if axis == 0
                else (minimum_x, maximum_x)
            )
            stabilized[selected, tangent_axis] = np.clip(
                stabilized[selected, tangent_axis],
                tangent_minimum,
                tangent_maximum,
            )
            claimed[selected] = True

        return stabilized - np.asarray(
            [float(sensor_progress), float(sensor_lateral)], dtype=np.float64
        )

    def plan(
        self,
        progress,
        template_lateral_offset,
        live_obstacle_points,
        right_line,
        left_line,
    ):
        """Return the currently safe remainder of the one fixed template."""
        progress = float(progress)
        template_lateral_offset = float(template_lateral_offset)
        right_line = float(right_line)
        left_line = float(left_line)
        if (
            self._template is None
            or not all(
                math.isfinite(value)
                for value in (
                    progress,
                    template_lateral_offset,
                    right_line,
                    left_line,
                )
            )
            or right_line >= left_line
            or progress >= float(np.max(self._template.x)) - 1e-9
        ):
            return None

        template = self._template
        first = max(0, int(np.searchsorted(template.x, progress)) - 1)
        if first >= template.x.size - 1:
            return None
        nominal_obstacles = self._nominal_obstacles.copy()
        nominal_obstacles[:, 0] -= progress
        nominal_obstacles[:, 1] += template_lateral_offset
        path = CommonPath(
            x=template.x[first:] - progress,
            y=template.y[first:] + template_lateral_offset,
            heading=template.heading[first:].copy(),
            curvature=template.curvature[first:].copy(),
            speed=template.speed[first:].copy(),
            direction=template.direction[first:].copy(),
            frame_id="obstacle_course_local",
            goal_tolerance=template.goal_tolerance,
            safety=self.collision_checker.safety(
                nominal_obstacles,
                right_line,
                left_line,
                self.collision_checker.footprint,
            ),
            label=template.label,
            obstacle_clearance=template.obstacle_clearance,
            line_clearance=template.line_clearance,
            curvature_variation=float(
                np.sum(np.abs(np.diff(template.curvature[first:])))
            ),
        )
        live_safety = self.collision_checker.safety(
            live_obstacle_points,
            right_line,
            left_line,
            self.validation_footprint,
        )
        validation = self.collision_checker.validator.validate_path(
            path,
            maximum_distance=self.live_validation_distance,
            safety=live_safety,
        )
        if not validation.safe:
            return None
        live_obstacle = validation.minimum_obstacle_clearance
        live_line = validation.minimum_line_clearance
        path.obstacle_clearance = min(
            template.obstacle_clearance, live_obstacle
        )
        path.line_clearance = min(template.line_clearance, live_line)
        segment = np.hypot(np.diff(path.x), np.diff(path.y))
        path.expected_time = float(
            np.sum(
                2.0
                * segment
                / np.maximum(path.speed[:-1] + path.speed[1:], 1e-6)
            )
        )
        return path
