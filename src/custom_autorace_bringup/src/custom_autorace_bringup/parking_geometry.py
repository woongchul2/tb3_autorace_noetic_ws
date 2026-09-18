"""ROS-independent geometry used by the AMCL parking controller."""

import math
import operator

import numpy as np


LEFT = "LEFT"
RIGHT = "RIGHT"


def normalize_angle(angle):
    """Wrap an angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def quintic_pose_path(
    start_pose,
    end_pose,
    start_tangent_length,
    end_tangent_length,
    sample_count,
):
    """Sample a zero-end-curvature quintic between two planar poses.

    The first and last three control points are equally spaced and collinear
    with their respective pose headings.  Position, heading and curvature are
    therefore continuous when this path is joined to another path with the
    same endpoint pose and zero endpoint curvature.
    """
    try:
        start = np.asarray(start_pose, dtype=np.float64)
        end = np.asarray(end_pose, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "start_pose and end_pose must contain three finite values"
        ) from error
    if (
        start.shape != (3,)
        or end.shape != (3,)
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(end))
    ):
        raise ValueError(
            "start_pose and end_pose must contain three finite values"
        )

    try:
        start_tangent_length = float(start_tangent_length)
        end_tangent_length = float(end_tangent_length)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("tangent lengths must be finite and positive") from error
    if not (
        math.isfinite(start_tangent_length)
        and math.isfinite(end_tangent_length)
        and start_tangent_length > 0.0
        and end_tangent_length > 0.0
    ):
        raise ValueError("tangent lengths must be finite and positive")

    if isinstance(sample_count, (bool, np.bool_)):
        raise ValueError("sample_count must be an integer of at least three")
    try:
        sample_count = operator.index(sample_count)
    except TypeError as error:
        raise ValueError(
            "sample_count must be an integer of at least three"
        ) from error
    if sample_count < 3:
        raise ValueError("sample_count must be an integer of at least three")

    start_direction = np.asarray(
        [math.cos(start[2]), math.sin(start[2])], dtype=np.float64
    )
    end_direction = np.asarray(
        [math.cos(end[2]), math.sin(end[2])], dtype=np.float64
    )
    with np.errstate(over="ignore", invalid="ignore"):
        control_points = np.asarray(
            [
                start[:2],
                start[:2] + start_tangent_length * start_direction,
                start[:2] + 2.0 * start_tangent_length * start_direction,
                end[:2] - 2.0 * end_tangent_length * end_direction,
                end[:2] - end_tangent_length * end_direction,
                end[:2],
            ],
            dtype=np.float64,
        )
    if not np.all(np.isfinite(control_points)):
        raise ValueError("path dimensions produce non-finite control points")

    parameter = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    complement = 1.0 - parameter
    positions = np.zeros((sample_count, 2), dtype=np.float64)
    derivatives = np.zeros((sample_count, 2), dtype=np.float64)
    derivative_controls = 5.0 * np.diff(control_points, axis=0)
    for index in range(6):
        weight = (
            math.comb(5, index)
            * complement ** (5 - index)
            * parameter ** index
        )
        positions += weight[:, None] * control_points[index]
    for index in range(5):
        weight = (
            math.comb(4, index)
            * complement ** (4 - index)
            * parameter ** index
        )
        derivatives += weight[:, None] * derivative_controls[index]

    derivative_norm = np.linalg.norm(derivatives, axis=1)
    if not np.all(np.isfinite(derivative_norm)) or np.any(
        derivative_norm <= np.finfo(np.float64).tiny
    ):
        raise ValueError("path dimensions produce a degenerate path")
    headings = np.arctan2(derivatives[:, 1], derivatives[:, 0])
    headings[0] = normalize_angle(start[2])
    headings[-1] = normalize_angle(end[2])
    return np.column_stack((positions, headings))


def curvature_matched_quintic(
    start_pose,
    end_pose,
    start_curvature,
    end_curvature,
    start_tangent_length,
    end_tangent_length,
    sample_spacing,
):
    """Sample a quintic with matched endpoint pose and curvature.

    Tangent lengths are distances from each endpoint to its adjacent Bezier
    control point.  The next control at either end adds only the normal
    acceleration required by the requested curvature.  The returned points,
    headings and curvatures are therefore G2-continuous with matching route
    samples at both ends.
    """

    try:
        start = np.asarray(start_pose, dtype=np.float64)
        end = np.asarray(end_pose, dtype=np.float64)
        values = (
            float(start_curvature),
            float(end_curvature),
            float(start_tangent_length),
            float(end_tangent_length),
            float(sample_spacing),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "curvature-matched quintic inputs must be finite numbers"
        ) from error
    if (
        start.shape != (3,)
        or end.shape != (3,)
        or not np.all(np.isfinite(start))
        or not np.all(np.isfinite(end))
        or not all(math.isfinite(value) for value in values)
        or values[2] <= 0.0
        or values[3] <= 0.0
        or values[4] <= 0.0
    ):
        raise ValueError(
            "poses and curvatures must be finite; lengths and spacing must "
            "be positive"
        )

    start_direction = np.asarray(
        (math.cos(start[2]), math.sin(start[2])), dtype=np.float64
    )
    end_direction = np.asarray(
        (math.cos(end[2]), math.sin(end[2])), dtype=np.float64
    )
    start_normal = np.asarray(
        (-start_direction[1], start_direction[0]), dtype=np.float64
    )
    end_normal = np.asarray(
        (-end_direction[1], end_direction[0]), dtype=np.float64
    )
    start_tangent = values[2]
    end_tangent = values[3]
    controls = np.empty((6, 2), dtype=np.float64)
    controls[0] = start[:2]
    controls[1] = controls[0] + start_tangent * start_direction
    controls[2] = (
        2.0 * controls[1]
        - controls[0]
        + 1.25 * start_tangent ** 2 * values[0] * start_normal
    )
    controls[5] = end[:2]
    controls[4] = controls[5] - end_tangent * end_direction
    controls[3] = (
        2.0 * controls[4]
        - controls[5]
        + 1.25 * end_tangent ** 2 * values[1] * end_normal
    )
    if not np.all(np.isfinite(controls)):
        raise ValueError("curvature-matched quintic controls are non-finite")

    control_length = float(
        np.sum(np.linalg.norm(np.diff(controls, axis=0), axis=1))
    )
    sample_count = max(
        11, int(math.ceil(control_length / values[4])) + 1
    )
    parameter = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    complement = 1.0 - parameter
    points = np.zeros((sample_count, 2), dtype=np.float64)
    first = np.zeros((sample_count, 2), dtype=np.float64)
    second = np.zeros((sample_count, 2), dtype=np.float64)
    first_controls = 5.0 * np.diff(controls, axis=0)
    second_controls = 20.0 * (
        controls[2:] - 2.0 * controls[1:-1] + controls[:-2]
    )
    for index in range(6):
        weight = (
            math.comb(5, index)
            * complement ** (5 - index)
            * parameter ** index
        )
        points += weight[:, None] * controls[index]
    for index in range(5):
        weight = (
            math.comb(4, index)
            * complement ** (4 - index)
            * parameter ** index
        )
        first += weight[:, None] * first_controls[index]
    for index in range(4):
        weight = (
            math.comb(3, index)
            * complement ** (3 - index)
            * parameter ** index
        )
        second += weight[:, None] * second_controls[index]

    derivative_norm = np.linalg.norm(first, axis=1)
    if (
        not np.all(np.isfinite(derivative_norm))
        or np.any(derivative_norm <= 1e-9)
        or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1e-9)
    ):
        raise ValueError(
            "curvature-matched quintic contains a cusp or duplicate point"
        )
    headings = np.arctan2(first[:, 1], first[:, 0])
    curvatures = (
        first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    ) / derivative_norm ** 3
    headings[0] = normalize_angle(start[2])
    headings[-1] = normalize_angle(end[2])
    curvatures[0] = values[0]
    curvatures[-1] = values[1]
    return points, headings, curvatures


def quintic_turn_path(
    start_pose,
    target_yaw,
    offset,
    tangent_length,
    sample_count,
):
    """Sample a monotonic, zero-end-curvature 90-degree Bezier turn.

    ``offset`` is the displacement along both the start and target headings.
    The first and last three control points are equally spaced and collinear,
    which makes the path curvature zero at both endpoints.
    """
    try:
        start = np.asarray(start_pose, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("start_pose must contain three finite values") from error
    if start.shape != (3,) or not np.all(np.isfinite(start)):
        raise ValueError("start_pose must contain three finite values")

    try:
        target_yaw = float(target_yaw)
        offset = float(offset)
        tangent_length = float(tangent_length)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "target_yaw, offset and tangent_length must be finite numbers"
        ) from error
    if not all(
        math.isfinite(value)
        for value in (target_yaw, offset, tangent_length)
    ):
        raise ValueError(
            "target_yaw, offset and tangent_length must be finite numbers"
        )
    if offset <= 0.0:
        raise ValueError("offset must be positive")
    if tangent_length <= 0.0 or tangent_length >= 0.5 * offset:
        raise ValueError(
            "tangent_length must be positive and below half offset"
        )

    if isinstance(sample_count, (bool, np.bool_)):
        raise ValueError("sample_count must be an integer of at least three")
    try:
        sample_count = operator.index(sample_count)
    except TypeError as error:
        raise ValueError(
            "sample_count must be an integer of at least three"
        ) from error
    if sample_count < 3:
        raise ValueError("sample_count must be an integer of at least three")

    start_yaw = float(start[2])
    turn_angle = normalize_angle(target_yaw - start_yaw)
    if not math.isclose(
        abs(turn_angle), 0.5 * math.pi, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("target_yaw must define a 90-degree turn")

    start_direction = np.asarray(
        [math.cos(start_yaw), math.sin(start_yaw)], dtype=np.float64
    )
    target_direction = np.asarray(
        [math.cos(target_yaw), math.sin(target_yaw)], dtype=np.float64
    )
    with np.errstate(over="ignore", invalid="ignore"):
        end_point = start[:2] + offset * (
            start_direction + target_direction
        )
    end_pose = (end_point[0], end_point[1], target_yaw)
    try:
        return quintic_pose_path(
            start,
            end_pose,
            tangent_length,
            tangent_length,
            sample_count,
        )
    except ValueError as error:
        raise ValueError(
            str(error).replace("path dimensions", "turn dimensions")
        ) from error


def path_curvature_limits(poses):
    """Return maximum ``(|curvature|, |d curvature / ds|)`` for a pose path.

    Curvature is measured between adjacent pose headings. Its spatial rate is
    measured between those segment-centred curvature samples. This avoids the
    duplicated one-sided gradients that can dominate maxima at path endpoints.
    """
    try:
        values = np.asarray(poses, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("poses must be a finite Nx3 array") from error
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or values.shape[0] < 3
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "poses must be a finite Nx3 array with at least three rows"
        )

    segment_lengths = np.linalg.norm(np.diff(values[:, :2], axis=0), axis=1)
    coordinate_scale = max(1.0, float(np.max(np.abs(values[:, :2]))))
    minimum_length = 64.0 * np.finfo(np.float64).eps * coordinate_scale
    if np.any(segment_lengths <= minimum_length):
        raise ValueError("poses must not contain repeated or degenerate segments")

    headings = np.unwrap(values[:, 2])
    curvatures = np.diff(headings) / segment_lengths
    curvature_positions = np.cumsum(segment_lengths) - 0.5 * segment_lengths
    curvature_rates = np.diff(curvatures) / np.diff(curvature_positions)
    if not (
        np.all(np.isfinite(curvatures))
        and np.all(np.isfinite(curvature_rates))
    ):
        raise ValueError("poses produce non-finite curvature values")
    return (
        float(np.max(np.abs(curvatures))),
        float(np.max(np.abs(curvature_rates))),
    )


def map_from_odom_transform(map_pose, odom_pose):
    """Return the 2-D transform that maps odom coordinates into map."""
    map_x, map_y, map_yaw = map(float, map_pose)
    odom_x, odom_y, odom_yaw = map(float, odom_pose)
    yaw = normalize_angle(map_yaw - odom_yaw)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        map_x - (cosine * odom_x - sine * odom_y),
        map_y - (sine * odom_x + cosine * odom_y),
        yaw,
    )


def map_pose_to_odom(map_pose, map_from_odom):
    """Transform a map-frame pose into the odom frame."""
    map_x, map_y, map_yaw = map(float, map_pose)
    origin_x, origin_y, transform_yaw = map(float, map_from_odom)
    dx = map_x - origin_x
    dy = map_y - origin_y
    cosine = math.cos(transform_yaw)
    sine = math.sin(transform_yaw)
    return (
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        normalize_angle(map_yaw - transform_yaw),
    )


def odom_pose_to_map(odom_pose, map_from_odom):
    """Transform an odom-frame pose into the anchored map frame."""
    odom_x, odom_y, odom_yaw = map(float, odom_pose)
    origin_x, origin_y, transform_yaw = map(float, map_from_odom)
    cosine = math.cos(transform_yaw)
    sine = math.sin(transform_yaw)
    return (
        origin_x + cosine * odom_x - sine * odom_y,
        origin_y + sine * odom_x + cosine * odom_y,
        normalize_angle(odom_yaw + transform_yaw),
    )


def scan_points_in_map(
    ranges,
    angle_min,
    angle_increment,
    minimum_range,
    maximum_range,
    map_pose,
    lidar_x=0.0,
    lidar_y=0.0,
):
    """Project valid LaserScan samples into the AMCL map frame."""
    values = np.asarray(ranges, dtype=np.float64)
    if values.size == 0 or not math.isfinite(float(angle_increment)):
        return np.empty((0, 2), dtype=np.float64)

    angles = float(angle_min) + np.arange(values.size) * float(angle_increment)
    finite = np.isfinite(values)
    above_minimum = np.zeros(values.shape, dtype=bool)
    below_maximum = np.zeros(values.shape, dtype=bool)
    np.greater_equal(
        values,
        max(0.0, float(minimum_range)),
        out=above_minimum,
        where=finite,
    )
    np.less_equal(
        values,
        float(maximum_range),
        out=below_maximum,
        where=finite,
    )
    valid = finite & above_minimum & below_maximum
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)

    selected_ranges = values[valid]
    selected_angles = angles[valid]
    base_x = float(lidar_x) + selected_ranges * np.cos(selected_angles)
    base_y = float(lidar_y) + selected_ranges * np.sin(selected_angles)

    robot_x, robot_y, robot_yaw = map(float, map_pose)
    cosine = math.cos(robot_yaw)
    sine = math.sin(robot_yaw)
    return np.column_stack(
        (
            robot_x + cosine * base_x - sine * base_y,
            robot_y + sine * base_x + cosine * base_y,
        )
    )


def points_in_box(points, box):
    """Count points in an inclusive axis-aligned ``[xmin,xmax,ymin,ymax]`` box."""
    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return 0
    values = values.reshape((-1, 2))
    minimum_x, maximum_x, minimum_y, maximum_y = map(float, box)
    inside = (
        (values[:, 0] >= minimum_x)
        & (values[:, 0] <= maximum_x)
        & (values[:, 1] >= minimum_y)
        & (values[:, 1] <= maximum_y)
    )
    return int(np.count_nonzero(inside))


def choose_clear_space(
    left_points,
    right_points,
    occupied_minimum_points,
    clear_maximum_points,
):
    """Return the only clear bay, or ``None`` when the scan is ambiguous."""
    left_occupied = int(left_points) >= int(occupied_minimum_points)
    right_occupied = int(right_points) >= int(occupied_minimum_points)
    left_clear = int(left_points) <= int(clear_maximum_points)
    right_clear = int(right_points) <= int(clear_maximum_points)
    if left_clear and right_occupied:
        return LEFT
    if right_clear and left_occupied:
        return RIGHT
    return None


def rectangle_corners(pose, front, rear, half_width):
    """Return the four map-frame corners of an asymmetric robot footprint."""
    x, y, yaw = map(float, pose)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    corners = []
    for local_x in (-float(rear), float(front)):
        for local_y in (-float(half_width), float(half_width)):
            corners.append(
                (
                    x + cosine * local_x - sine * local_y,
                    y + sine * local_x + cosine * local_y,
                )
            )
    return corners
