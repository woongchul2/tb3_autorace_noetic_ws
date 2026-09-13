#!/usr/bin/env python3
"""ROS-independent forward Hybrid A* planner for the tunnel mission.

The planner deliberately keeps mapping and ROS message conversion outside this
module.  It consumes one occupancy-grid snapshot and plans in SE(2) with
forward, constant-curvature motion primitives.  Collision checks use the
robot's oriented, asymmetric rectangular footprint at every swept sample.

Occupancy-grid coordinates follow ``nav_msgs/OccupancyGrid`` conventions:
``data[grid_y, grid_x]`` and the origin is the lower-left corner of cell
``(0, 0)``.  Only axis-aligned grid origins are needed by the AutoRace maps.
"""

from dataclasses import dataclass
import heapq
import math

import numpy as np


_ANGLE_EPSILON = 1e-12


def normalize_angle(angle):
    """Wrap an angle to [-pi, pi)."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class RectangularFootprint:
    """Robot bounds relative to its base pose.

    ``front`` and ``rear`` are intentionally separate because the TurtleBot3
    base frame is not at the geometric centre of its footprint. ``padding`` is
    applied on all four sides before every collision check.
    """

    front: float
    rear: float
    half_width: float
    padding: float = 0.0

    def __post_init__(self):
        if self.front <= 0.0 or self.rear <= 0.0:
            raise ValueError("footprint front and rear must be positive")
        if self.half_width <= 0.0:
            raise ValueError("footprint half_width must be positive")
        if self.padding < 0.0:
            raise ValueError("footprint padding cannot be negative")


@dataclass
class PlannedPath:
    """A collision-checked path sampled densely enough for tracking."""

    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    curvature: np.ndarray
    cost: float
    expanded_nodes: int

    @property
    def length(self):
        if self.x.size < 2:
            return 0.0
        return float(np.sum(np.hypot(np.diff(self.x), np.diff(self.y))))

    @property
    def poses(self):
        return tuple(
            Pose2D(float(x), float(y), float(yaw))
            for x, y, yaw in zip(self.x, self.y, self.yaw)
        )


class OccupancyGrid:
    """Immutable-shape occupancy-grid snapshot with cached distance fields."""

    def __init__(
        self,
        data,
        resolution,
        origin_x=0.0,
        origin_y=0.0,
        occupied_threshold=50,
        unknown_is_occupied=True,
        soft_cost_data=None,
    ):
        values = np.asarray(data)
        if values.ndim != 2:
            raise ValueError("occupancy data must be a two-dimensional array")
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("occupancy data cannot be empty")
        if not math.isfinite(float(resolution)) or resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if occupied_threshold < 0:
            raise ValueError("occupied_threshold cannot be negative")

        self.data = values.copy()
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.occupied_threshold = int(occupied_threshold)
        self.unknown_is_occupied = bool(unknown_is_occupied)
        if self.data.dtype == np.bool_:
            occupied = self.data.copy()
        else:
            occupied = self.data >= self.occupied_threshold
        if self.unknown_is_occupied:
            occupied = occupied | (self.data < 0)
        self._occupied = np.asarray(occupied, dtype=bool)
        if soft_cost_data is None:
            soft_cost_values = np.zeros(self.data.shape, dtype=np.float64)
        else:
            soft_cost_values = np.asarray(soft_cost_data, dtype=np.float64)
            if soft_cost_values.shape != self.data.shape:
                raise ValueError(
                    "soft cost data must match occupancy data shape"
                )
            if (
                not np.all(np.isfinite(soft_cost_values))
                or np.any(soft_cost_values < 0.0)
                or np.any(soft_cost_values >= self.occupied_threshold)
            ):
                raise ValueError(
                    "soft cost values must be finite and in "
                    "[0, occupied_threshold)"
                )
        self._soft_cost_values = soft_cost_values.copy()
        self._soft_cost = self._soft_cost_values > 0.0
        self._has_soft_cost = bool(np.any(self._soft_cost))
        self._soft_cost_integral = None
        self._distance_field = None
        self._inflated_masks = {0.0: self._occupied}
        self._integral_masks = {}
        self._goal_distance_fields = {}

    @property
    def has_soft_cost(self):
        """Whether any positive, non-lethal traversal-cost cell exists."""
        return self._has_soft_cost

    def soft_cost_count(self, minimum_x, maximum_x, minimum_y, maximum_y):
        """Count positive sub-threshold cells in an inclusive rectangle."""
        if self._soft_cost_integral is None:
            integral = np.zeros(
                (self.height + 1, self.width + 1),
                dtype=np.int32,
            )
            integral[1:, 1:] = np.cumsum(
                np.cumsum(self._soft_cost, axis=0, dtype=np.int32),
                axis=1,
                dtype=np.int32,
            )
            self._soft_cost_integral = integral
        first_x = int(minimum_x)
        last_x = int(maximum_x) + 1
        first_y = int(minimum_y)
        last_y = int(maximum_y) + 1
        integral = self._soft_cost_integral
        return int(
            integral[last_y, last_x]
            - integral[first_y, last_x]
            - integral[last_y, first_x]
            + integral[first_y, first_x]
        )

    @classmethod
    def from_flat(
        cls,
        data,
        width,
        height,
        resolution,
        origin_x=0.0,
        origin_y=0.0,
        occupied_threshold=50,
        unknown_is_occupied=True,
        soft_cost_data=None,
    ):
        """Build a grid directly from flat ROS-style occupancy data."""
        width = int(width)
        height = int(height)
        values = np.asarray(data)
        if width <= 0 or height <= 0 or values.size != width * height:
            raise ValueError(
                "flat occupancy data does not match width * height"
            )
        soft_values = None
        if soft_cost_data is not None:
            soft_values = np.asarray(soft_cost_data)
            if soft_values.size != width * height:
                raise ValueError(
                    "flat soft cost data does not match width * height"
                )
            soft_values = soft_values.reshape((height, width))
        return cls(
            values.reshape((height, width)),
            resolution,
            origin_x,
            origin_y,
            occupied_threshold,
            unknown_is_occupied,
            soft_values,
        )

    @property
    def width(self):
        return int(self.data.shape[1])

    @property
    def height(self):
        return int(self.data.shape[0])

    @property
    def maximum_x(self):
        return self.origin_x + self.width * self.resolution

    @property
    def maximum_y(self):
        return self.origin_y + self.height * self.resolution

    def world_to_grid(self, x, y):
        """Return integer ``(grid_x, grid_y)`` for a world coordinate."""
        grid_x = int(math.floor((float(x) - self.origin_x) / self.resolution))
        grid_y = int(math.floor((float(y) - self.origin_y) / self.resolution))
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """Return the world coordinate at a cell centre."""
        return (
            self.origin_x + (float(grid_x) + 0.5) * self.resolution,
            self.origin_y + (float(grid_y) + 0.5) * self.resolution,
        )

    def contains_cell(self, grid_x, grid_y):
        return (
            0 <= int(grid_x) < self.width
            and 0 <= int(grid_y) < self.height
        )

    def is_occupied_cell(self, grid_x, grid_y, inflation_radius=0.0):
        """Treat out-of-map cells as occupied."""
        if not self.contains_cell(grid_x, grid_y):
            return True
        mask = self.inflated_mask(inflation_radius)
        return bool(mask[int(grid_y), int(grid_x)])

    def _compute_distance_field(self):
        """Build an 8-neighbour approximation of obstacle distance."""
        distances = np.full(self._occupied.shape, math.inf, dtype=np.float64)
        occupied_y, occupied_x = np.nonzero(self._occupied)
        if occupied_x.size == 0:
            self._distance_field = distances
            return

        queue = []
        for grid_x, grid_y in zip(occupied_x.tolist(), occupied_y.tolist()):
            distances[grid_y, grid_x] = 0.0
            heapq.heappush(queue, (0.0, grid_x, grid_y))

        diagonal = math.sqrt(2.0)
        neighbours = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, diagonal),
            (-1, 1, diagonal),
            (1, -1, diagonal),
            (1, 1, diagonal),
        )
        while queue:
            distance, grid_x, grid_y = heapq.heappop(queue)
            if distance > distances[grid_y, grid_x] + 1e-12:
                continue
            for offset_x, offset_y, edge_cost in neighbours:
                next_x = grid_x + offset_x
                next_y = grid_y + offset_y
                if not self.contains_cell(next_x, next_y):
                    continue
                candidate = distance + edge_cost
                if candidate + 1e-12 < distances[next_y, next_x]:
                    distances[next_y, next_x] = candidate
                    heapq.heappush(queue, (candidate, next_x, next_y))
        self._distance_field = distances * self.resolution

    @property
    def distance_field(self):
        """Cell-centre distance to the closest occupied cell centre."""
        if self._distance_field is None:
            self._compute_distance_field()
        return self._distance_field

    def distance_to_obstacle(self, x, y):
        grid_x, grid_y = self.world_to_grid(x, y)
        if not self.contains_cell(grid_x, grid_y):
            return 0.0
        return float(self.distance_field[grid_y, grid_x])

    def inflated_mask(self, inflation_radius=0.0):
        """Return a conservative raster inflation of occupied cells."""
        radius = max(0.0, float(inflation_radius))
        if radius <= 1e-12:
            return self._occupied
        key = round(radius, 12)
        if key not in self._inflated_masks:
            # A cell is marked if its square can intersect the requested
            # inflation disk.  The footprint SAT check then uses its full area.
            cell_half_diagonal = self.resolution / math.sqrt(2.0)
            self._inflated_masks[key] = (
                self.distance_field <= radius + cell_half_diagonal
            )
        return self._inflated_masks[key]

    def occupied_integral(self, inflation_radius=0.0):
        """Return a summed-area table for fast empty rectangle queries."""
        radius = max(0.0, float(inflation_radius))
        key = 0.0 if radius <= 1e-12 else round(radius, 12)
        if key not in self._integral_masks:
            mask = self.inflated_mask(radius)
            integral = np.zeros(
                (self.height + 1, self.width + 1),
                dtype=np.int32,
            )
            integral[1:, 1:] = np.cumsum(
                np.cumsum(mask, axis=0, dtype=np.int32),
                axis=1,
                dtype=np.int32,
            )
            self._integral_masks[key] = integral
        return self._integral_masks[key]

    def occupied_count(
        self,
        minimum_x,
        maximum_x,
        minimum_y,
        maximum_y,
        inflation_radius=0.0,
    ):
        """Count occupied cells in an inclusive in-bounds rectangle."""
        integral = self.occupied_integral(inflation_radius)
        first_x = int(minimum_x)
        last_x = int(maximum_x) + 1
        first_y = int(minimum_y)
        last_y = int(maximum_y) + 1
        return int(
            integral[last_y, last_x]
            - integral[first_y, last_x]
            - integral[last_y, first_x]
            + integral[first_y, first_x]
        )

    def goal_distance_field(self, goal_x, goal_y):
        """Raw-occupancy 2-D Dijkstra distances to one goal cell.

        This deliberately ignores the robot footprint.  Consequently it is a
        useful optimistic guide for SE(2) search while final feasibility still
        comes exclusively from oriented footprint sweep checks.
        """
        goal_grid_x, goal_grid_y = self.world_to_grid(goal_x, goal_y)
        key = goal_grid_x, goal_grid_y
        if key in self._goal_distance_fields:
            return self._goal_distance_fields[key]

        distances = np.full(self._occupied.shape, math.inf, dtype=np.float64)
        if (
            not self.contains_cell(goal_grid_x, goal_grid_y)
            or self._occupied[goal_grid_y, goal_grid_x]
        ):
            self._goal_distance_fields[key] = distances
            return distances

        distances[goal_grid_y, goal_grid_x] = 0.0
        queue = [(0.0, goal_grid_x, goal_grid_y)]
        diagonal = math.sqrt(2.0)
        neighbours = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, diagonal),
            (-1, 1, diagonal),
            (1, -1, diagonal),
            (1, 1, diagonal),
        )
        while queue:
            distance, grid_x, grid_y = heapq.heappop(queue)
            if distance > distances[grid_y, grid_x] + 1e-12:
                continue
            for offset_x, offset_y, edge_cost in neighbours:
                next_x = grid_x + offset_x
                next_y = grid_y + offset_y
                if (
                    not self.contains_cell(next_x, next_y)
                    or self._occupied[next_y, next_x]
                ):
                    continue
                candidate = distance + edge_cost
                if candidate + 1e-12 < distances[next_y, next_x]:
                    distances[next_y, next_x] = candidate
                    heapq.heappush(queue, (candidate, next_x, next_y))
        distances *= self.resolution
        self._goal_distance_fields[key] = distances
        return distances


@dataclass(frozen=True)
class _SearchNode:
    pose: Pose2D
    cost: float
    parent_id: int
    curvature_index: int
    travel_distance: float


class HybridAStarPlanner:
    """Forward-only SE(2) Hybrid A* with rectangular swept collision checks.

    All constructor arguments are scalar/list values suitable for loading from
    YAML.  Turning curvature is sampled uniformly from
    ``-1 / minimum_turning_radius`` through the corresponding left curvature;
    ``steering_samples`` must therefore be an odd number. Optional keep-in
    rectangles constrain the robot base centre to the union of surveyed route
    regions while physical geometry remains exclusively in the occupancy map.
    """

    def __init__(
        self,
        footprint,
        heading_bins=72,
        minimum_turning_radius=0.30,
        primitive_step=0.10,
        steering_samples=5,
        goal_position_tolerance=0.10,
        goal_heading_tolerance=math.radians(7.5),
        goal_curvature_tolerance=1e-6,
        non_straight_penalty=0.08,
        steering_change_penalty=0.04,
        inflation_radius=0.0,
        obstacle_cost_weight=0.10,
        obstacle_cost_distance=0.35,
        soft_obstacle_cost_weight=0.0,
        soft_cost_check_step=None,
        heading_heuristic_weight=0.15,
        use_goal_distance_heuristic=True,
        enable_analytic_goal_connection=True,
        analytic_lateral_tolerance=0.01,
        analytic_heading_tolerance=math.radians(3.0),
        collision_check_step=0.025,
        collision_check_angle=math.radians(3.0),
        path_sample_step=0.025,
        state_xy_resolution=None,
        maximum_iterations=100000,
        keep_in_rectangles=None,
    ):
        if not isinstance(footprint, RectangularFootprint):
            footprint = RectangularFootprint(**footprint)
        if int(heading_bins) < 8:
            raise ValueError("heading_bins must be at least 8")
        if minimum_turning_radius <= 0.0:
            raise ValueError("minimum_turning_radius must be positive")
        if primitive_step <= 0.0:
            raise ValueError("primitive_step must be positive")
        if int(steering_samples) < 3 or int(steering_samples) % 2 == 0:
            raise ValueError("steering_samples must be an odd number >= 3")
        if goal_position_tolerance <= 0.0:
            raise ValueError("goal_position_tolerance must be positive")
        if goal_heading_tolerance <= 0.0:
            raise ValueError("goal_heading_tolerance must be positive")
        if goal_curvature_tolerance < 0.0:
            raise ValueError("goal_curvature_tolerance cannot be negative")
        if non_straight_penalty < 0.0 or steering_change_penalty < 0.0:
            raise ValueError("motion penalties cannot be negative")
        if inflation_radius < 0.0:
            raise ValueError("inflation_radius cannot be negative")
        if obstacle_cost_weight < 0.0 or obstacle_cost_distance <= 0.0:
            raise ValueError("invalid obstacle-distance cost parameters")
        if soft_obstacle_cost_weight < 0.0:
            raise ValueError("soft_obstacle_cost_weight cannot be negative")
        if soft_cost_check_step is not None and soft_cost_check_step <= 0.0:
            raise ValueError("soft_cost_check_step must be positive")
        if heading_heuristic_weight < 0.0:
            raise ValueError("heading_heuristic_weight cannot be negative")
        if analytic_lateral_tolerance < 0.0:
            raise ValueError("analytic_lateral_tolerance cannot be negative")
        if analytic_heading_tolerance < 0.0:
            raise ValueError("analytic_heading_tolerance cannot be negative")
        if collision_check_step <= 0.0 or collision_check_angle <= 0.0:
            raise ValueError("collision sampling intervals must be positive")
        if path_sample_step <= 0.0:
            raise ValueError("path_sample_step must be positive")
        if state_xy_resolution is not None and state_xy_resolution <= 0.0:
            raise ValueError("state_xy_resolution must be positive")
        if int(maximum_iterations) <= 0:
            raise ValueError("maximum_iterations must be positive")
        keep_in_rectangles = self._validated_keep_in_rectangles(
            keep_in_rectangles
        )

        self.footprint = footprint
        self.heading_bins = int(heading_bins)
        self.minimum_turning_radius = float(minimum_turning_radius)
        self.primitive_step = float(primitive_step)
        self.steering_samples = int(steering_samples)
        self.goal_position_tolerance = float(goal_position_tolerance)
        self.goal_heading_tolerance = float(goal_heading_tolerance)
        self.goal_curvature_tolerance = float(goal_curvature_tolerance)
        self.non_straight_penalty = float(non_straight_penalty)
        self.steering_change_penalty = float(steering_change_penalty)
        self.inflation_radius = float(inflation_radius)
        self.obstacle_cost_weight = float(obstacle_cost_weight)
        self.obstacle_cost_distance = float(obstacle_cost_distance)
        self.soft_obstacle_cost_weight = float(soft_obstacle_cost_weight)
        self.soft_cost_check_step = (
            float(collision_check_step)
            if soft_cost_check_step is None
            else float(soft_cost_check_step)
        )
        self.heading_heuristic_weight = float(heading_heuristic_weight)
        self.use_goal_distance_heuristic = bool(
            use_goal_distance_heuristic
        )
        self.enable_analytic_goal_connection = bool(
            enable_analytic_goal_connection
        )
        self.analytic_lateral_tolerance = float(
            analytic_lateral_tolerance
        )
        self.analytic_heading_tolerance = float(
            analytic_heading_tolerance
        )
        self.collision_check_step = float(collision_check_step)
        self.collision_check_angle = float(collision_check_angle)
        self.path_sample_step = float(path_sample_step)
        self.state_xy_resolution = (
            None
            if state_xy_resolution is None
            else float(state_xy_resolution)
        )
        self.maximum_iterations = int(maximum_iterations)
        self.keep_in_rectangles = keep_in_rectangles

        maximum_curvature = 1.0 / self.minimum_turning_radius
        values = np.linspace(
            -maximum_curvature,
            maximum_curvature,
            self.steering_samples,
        )
        straight_index = self.steering_samples // 2
        values[straight_index] = 0.0
        # Trying straight first makes the deterministic tie-break favour the
        # shortest natural route in an empty tunnel.
        order = [straight_index]
        for offset in range(1, straight_index + 1):
            order.extend((straight_index + offset, straight_index - offset))
        self.curvatures = tuple(float(values[index]) for index in order)
        self._straight_index = 0

    @staticmethod
    def _coerce_pose(pose):
        if isinstance(pose, Pose2D):
            return Pose2D(
                float(pose.x),
                float(pose.y),
                normalize_angle(pose.yaw),
            )
        if len(pose) != 3:
            raise ValueError("a pose must contain x, y and yaw")
        return Pose2D(float(pose[0]), float(pose[1]), normalize_angle(pose[2]))

    @staticmethod
    def _validated_keep_in_rectangles(rectangles):
        if rectangles is None:
            return ()
        try:
            rectangles = list(rectangles)
        except TypeError as error:
            raise ValueError("keep_in_rectangles must be a sequence") from error

        validated = []
        for rectangle in rectangles:
            try:
                values = tuple(rectangle)
            except TypeError as error:
                raise ValueError(
                    "each keep-in rectangle must contain four values"
                ) from error
            if len(values) != 4:
                raise ValueError(
                    "each keep-in rectangle must contain four values"
                )
            try:
                values = tuple(float(value) for value in values)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(
                    "keep-in rectangle coordinates must be finite"
                ) from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError("keep-in rectangle coordinates must be finite")
            minimum_x, minimum_y, maximum_x, maximum_y = values
            if minimum_x >= maximum_x or minimum_y >= maximum_y:
                raise ValueError(
                    "keep-in rectangle minimums must be below maximums"
                )
            validated.append(values)
        return tuple(validated)

    @staticmethod
    def _propagate(start, curvature, distance):
        distance = float(distance)
        curvature = float(curvature)
        if abs(curvature) <= _ANGLE_EPSILON:
            return Pose2D(
                start.x + distance * math.cos(start.yaw),
                start.y + distance * math.sin(start.yaw),
                start.yaw,
            )
        end_yaw_unwrapped = start.yaw + distance * curvature
        return Pose2D(
            start.x
            + (math.sin(end_yaw_unwrapped) - math.sin(start.yaw))
            / curvature,
            start.y
            + (-math.cos(end_yaw_unwrapped) + math.cos(start.yaw))
            / curvature,
            normalize_angle(end_yaw_unwrapped),
        )

    def _motion_samples(
        self,
        start,
        curvature,
        distance,
        maximum_distance_step,
    ):
        count = self._motion_sample_count(
            curvature, distance, maximum_distance_step
        )
        for index in range(1, count + 1):
            yield self._propagate(
                start,
                curvature,
                distance * index / count,
            )

    def _motion_sample_count(
        self, curvature, distance, maximum_distance_step
    ):
        return max(
            1,
            int(math.ceil(abs(distance) / maximum_distance_step)),
            int(
                math.ceil(
                    abs(distance * curvature) / self.collision_check_angle
                )
            ),
        )

    def _footprint_geometry(self, pose):
        front = self.footprint.front + self.footprint.padding
        rear = self.footprint.rear + self.footprint.padding
        half_width = self.footprint.half_width + self.footprint.padding
        half_length = 0.5 * (front + rear)
        centre_offset = 0.5 * (front - rear)
        cosine = math.cos(pose.yaw)
        sine = math.sin(pose.yaw)
        centre_x = pose.x + centre_offset * cosine
        centre_y = pose.y + centre_offset * sine
        return (
            centre_x,
            centre_y,
            cosine,
            sine,
            half_length,
            half_width,
        )

    def _pose_is_collision_free(self, grid, pose):
        if not self._pose_is_within_keep_in(pose):
            return False

        epsilon = 1e-10
        (
            centre_x,
            centre_y,
            cosine,
            sine,
            half_length,
            half_width,
        ) = self._footprint_geometry(pose)

        absolute_cosine = abs(cosine)
        absolute_sine = abs(sine)
        extent_x = (
            absolute_cosine * half_length + absolute_sine * half_width
        )
        extent_y = (
            absolute_sine * half_length + absolute_cosine * half_width
        )
        footprint_minimum_x = centre_x - extent_x
        footprint_maximum_x = centre_x + extent_x
        footprint_minimum_y = centre_y - extent_y
        footprint_maximum_y = centre_y + extent_y
        if (
            footprint_minimum_x < grid.origin_x - epsilon
            or footprint_maximum_x > grid.maximum_x + epsilon
            or footprint_minimum_y < grid.origin_y - epsilon
            or footprint_maximum_y > grid.maximum_y + epsilon
        ):
            return False

        minimum_x = max(
            0,
            int(
                math.floor(
                    (footprint_minimum_x - grid.origin_x) / grid.resolution
                )
            ),
        )
        maximum_x = min(
            grid.width - 1,
            int(
                math.floor(
                    (footprint_maximum_x - grid.origin_x) / grid.resolution
                )
            ),
        )
        minimum_y = max(
            0,
            int(
                math.floor(
                    (footprint_minimum_y - grid.origin_y) / grid.resolution
                )
            ),
        )
        maximum_y = min(
            grid.height - 1,
            int(
                math.floor(
                    (footprint_maximum_y - grid.origin_y) / grid.resolution
                )
            ),
        )
        if (
            grid.occupied_count(
                minimum_x,
                maximum_x,
                minimum_y,
                maximum_y,
                self.inflation_radius,
            )
            == 0
        ):
            return True
        occupied = grid.inflated_mask(self.inflation_radius)[
            minimum_y : maximum_y + 1,
            minimum_x : maximum_x + 1,
        ]
        occupied_y, occupied_x = np.nonzero(occupied)
        if occupied_x.size == 0:
            return True

        occupied_x = occupied_x + minimum_x
        occupied_y = occupied_y + minimum_y
        cell_x = grid.origin_x + (occupied_x + 0.5) * grid.resolution
        cell_y = grid.origin_y + (occupied_y + 0.5) * grid.resolution
        delta_x = cell_x - centre_x
        delta_y = cell_y - centre_y
        cell_half = 0.5 * grid.resolution

        # Separating-axis test for the robot OBB and axis-aligned grid cells.
        along_robot_x = np.abs(delta_x * cosine + delta_y * sine)
        along_robot_y = np.abs(-delta_x * sine + delta_y * cosine)
        overlap = (
            (
                along_robot_x
                <= half_length
                + cell_half * (absolute_cosine + absolute_sine)
                + epsilon
            )
            & (
                along_robot_y
                <= half_width
                + cell_half * (absolute_cosine + absolute_sine)
                + epsilon
            )
            & (
                np.abs(delta_x)
                <= half_length * absolute_cosine
                + half_width * absolute_sine
                + cell_half
                + epsilon
            )
            & (
                np.abs(delta_y)
                <= half_length * absolute_sine
                + half_width * absolute_cosine
                + cell_half
                + epsilon
            )
        )
        return not bool(np.any(overlap))

    def _pose_soft_cost(self, grid, pose):
        """Maximum graded cost touched by the exact padded footprint."""
        if not grid.has_soft_cost:
            return 0.0

        (
            centre_x,
            centre_y,
            cosine,
            sine,
            half_length,
            half_width,
        ) = self._footprint_geometry(pose)
        absolute_cosine = abs(cosine)
        absolute_sine = abs(sine)
        extent_x = (
            absolute_cosine * half_length + absolute_sine * half_width
        )
        extent_y = (
            absolute_sine * half_length + absolute_cosine * half_width
        )
        minimum_x = max(
            0,
            int(
                math.floor(
                    (centre_x - extent_x - grid.origin_x)
                    / grid.resolution
                )
            ),
        )
        maximum_x = min(
            grid.width - 1,
            int(
                math.floor(
                    (centre_x + extent_x - grid.origin_x)
                    / grid.resolution
                )
            ),
        )
        minimum_y = max(
            0,
            int(
                math.floor(
                    (centre_y - extent_y - grid.origin_y)
                    / grid.resolution
                )
            ),
        )
        maximum_y = min(
            grid.height - 1,
            int(
                math.floor(
                    (centre_y + extent_y - grid.origin_y)
                    / grid.resolution
                )
            ),
        )
        if (
            minimum_x > maximum_x
            or minimum_y > maximum_y
            or grid.soft_cost_count(
                minimum_x, maximum_x, minimum_y, maximum_y
            )
            == 0
        ):
            return 0.0

        soft = grid._soft_cost[
            minimum_y : maximum_y + 1,
            minimum_x : maximum_x + 1,
        ]
        soft_y, soft_x = np.nonzero(soft)
        if soft_x.size == 0:
            return 0.0
        soft_x = soft_x + minimum_x
        soft_y = soft_y + minimum_y
        cell_x = grid.origin_x + (soft_x + 0.5) * grid.resolution
        cell_y = grid.origin_y + (soft_y + 0.5) * grid.resolution
        delta_x = cell_x - centre_x
        delta_y = cell_y - centre_y
        cell_half = 0.5 * grid.resolution
        epsilon = 1e-10
        along_robot_x = np.abs(delta_x * cosine + delta_y * sine)
        along_robot_y = np.abs(-delta_x * sine + delta_y * cosine)
        overlap = (
            (
                along_robot_x
                <= half_length
                + cell_half * (absolute_cosine + absolute_sine)
                + epsilon
            )
            & (
                along_robot_y
                <= half_width
                + cell_half * (absolute_cosine + absolute_sine)
                + epsilon
            )
            & (
                np.abs(delta_x)
                <= half_length * absolute_cosine
                + half_width * absolute_sine
                + cell_half
                + epsilon
            )
            & (
                np.abs(delta_y)
                <= half_length * absolute_sine
                + half_width * absolute_cosine
                + cell_half
                + epsilon
            )
        )
        if not np.any(overlap):
            return 0.0
        values = grid._soft_cost_values[
            soft_y[overlap], soft_x[overlap]
        ]
        denominator = max(1, grid.occupied_threshold - 1)
        return min(1.0, float(np.max(values)) / denominator)

    def pose_is_collision_free(self, grid, pose):
        """Check an oriented rectangle against bounds and occupied cells."""
        if not isinstance(grid, OccupancyGrid):
            raise TypeError("grid must be an OccupancyGrid")
        return self._pose_is_collision_free(grid, self._coerce_pose(pose))

    def _pose_is_within_keep_in(self, pose):
        if not self.keep_in_rectangles:
            return True
        epsilon = 1e-10
        return any(
            minimum_x - epsilon <= pose.x <= maximum_x + epsilon
            and minimum_y - epsilon <= pose.y <= maximum_y + epsilon
            for minimum_x, minimum_y, maximum_x, maximum_y
            in self.keep_in_rectangles
        )

    def pose_is_within_keep_in(self, pose):
        """Return whether the robot base centre is in a legal route region."""
        return self._pose_is_within_keep_in(self._coerce_pose(pose))

    def _primitive_is_collision_free(
        self,
        grid,
        start,
        curvature,
        distance,
        check_start,
    ):
        if check_start and not self._pose_is_collision_free(grid, start):
            return False
        sample_step = min(self.collision_check_step, 0.5 * grid.resolution)
        for pose in self._motion_samples(
            start,
            curvature,
            distance,
            sample_step,
        ):
            if not self._pose_is_collision_free(grid, pose):
                return False
        return True

    def _primitive_soft_cost_exposure(
        self, grid, start, curvature, distance
    ):
        """Integrate graded footprint/clearance-band overlap along a move."""
        if self.soft_obstacle_cost_weight <= 0.0 or not grid.has_soft_cost:
            return 0.0
        sample_step = min(self.soft_cost_check_step, grid.resolution)
        count = self._motion_sample_count(
            curvature, distance, sample_step
        )
        distance_per_sample = abs(float(distance)) / count
        exposure = 0.0
        for pose in self._motion_samples(
            start, curvature, distance, sample_step
        ):
            exposure += distance_per_sample * self._pose_soft_cost(grid, pose)
        return exposure

    def primitive_is_collision_free(
        self,
        grid,
        start,
        curvature,
        distance=None,
    ):
        """Check the complete swept primitive, including both endpoints."""
        if not isinstance(grid, OccupancyGrid):
            raise TypeError("grid must be an OccupancyGrid")
        start = self._coerce_pose(start)
        distance = self.primitive_step if distance is None else float(distance)
        if distance < 0.0:
            raise ValueError("forward primitive distance cannot be negative")
        return self._primitive_is_collision_free(
            grid,
            start,
            float(curvature),
            distance,
            check_start=True,
        )

    def _state_key(self, grid, pose, curvature_index):
        resolution = (
            grid.resolution
            if self.state_xy_resolution is None
            else self.state_xy_resolution
        )
        # Hybrid states are continuous poses quantised around lattice points,
        # rather than occupancy cells.  Nearest-bin rounding also prevents a
        # step exactly equal to the state resolution from being lost to binary
        # floating-point floor error (for example 0.6 / 0.1).
        grid_x = int(
            math.floor((pose.x - grid.origin_x) / resolution + 0.5)
        )
        grid_y = int(
            math.floor((pose.y - grid.origin_y) / resolution + 0.5)
        )
        yaw_unit = (normalize_angle(pose.yaw) + math.pi) / (2.0 * math.pi)
        heading_index = int(math.floor(yaw_unit * self.heading_bins + 0.5))
        heading_index %= self.heading_bins
        return grid_x, grid_y, heading_index, int(curvature_index)

    def _goal_reached(self, node, goal, goal_curvature):
        if math.hypot(node.pose.x - goal.x, node.pose.y - goal.y) > (
            self.goal_position_tolerance
        ):
            return False
        if abs(normalize_angle(node.pose.yaw - goal.yaw)) > (
            self.goal_heading_tolerance
        ):
            return False
        if goal_curvature is None:
            return True
        node_curvature = self.curvatures[node.curvature_index]
        return abs(node_curvature - float(goal_curvature)) <= (
            self.goal_curvature_tolerance
        )

    def _heuristic(self, grid, pose, goal, goal_distance_field):
        distance = math.hypot(pose.x - goal.x, pose.y - goal.y)
        heading_error = abs(normalize_angle(pose.yaw - goal.yaw))
        if goal_distance_field is not None:
            grid_x, grid_y = grid.world_to_grid(pose.x, pose.y)
            if not grid.contains_cell(grid_x, grid_y):
                return math.inf
            raw_distance = float(goal_distance_field[grid_y, grid_x])
            if not math.isfinite(raw_distance):
                return math.inf
            distance = max(distance, raw_distance)
        heading_distance = self.minimum_turning_radius * abs(
            heading_error
        )
        return distance + self.heading_heuristic_weight * heading_distance

    def _transition_cost(
        self,
        grid,
        end_pose,
        previous_curvature,
        curvature,
        travel_distance=None,
        soft_cost_exposure=0.0,
    ):
        distance = (
            self.primitive_step
            if travel_distance is None
            else float(travel_distance)
        )
        maximum_curvature = 1.0 / self.minimum_turning_radius
        curvature_ratio = abs(curvature) / maximum_curvature
        cost = distance * (
            1.0 + self.non_straight_penalty * curvature_ratio
        )
        if abs(curvature - previous_curvature) > 1e-12:
            change_ratio = abs(curvature - previous_curvature) / (
                2.0 * maximum_curvature
            )
            cost += self.steering_change_penalty * change_ratio
        clearance = grid.distance_to_obstacle(end_pose.x, end_pose.y)
        if clearance < self.obstacle_cost_distance:
            normalized = max(
                0.0,
                1.0 - clearance / self.obstacle_cost_distance,
            )
            cost += (
                self.obstacle_cost_weight
                * distance
                * normalized
                * normalized
            )
        cost += self.soft_obstacle_cost_weight * max(
            0.0, float(soft_cost_exposure)
        )
        return cost

    def _analytic_goal_node(
        self,
        grid,
        node,
        node_id,
        goal,
        goal_curvature,
    ):
        """Return a valid straight connection into the goal region."""
        if not self.enable_analytic_goal_connection:
            return None
        if (
            goal_curvature is not None
            and abs(float(goal_curvature)) > self.goal_curvature_tolerance
        ):
            return None
        if abs(normalize_angle(node.pose.yaw - goal.yaw)) > min(
            self.goal_heading_tolerance,
            self.analytic_heading_tolerance,
        ):
            return None

        delta_x = goal.x - node.pose.x
        delta_y = goal.y - node.pose.y
        cosine = math.cos(node.pose.yaw)
        sine = math.sin(node.pose.yaw)
        longitudinal = cosine * delta_x + sine * delta_y
        lateral = -sine * delta_x + cosine * delta_y
        if longitudinal <= 1e-9:
            return None
        if abs(lateral) > min(
            self.goal_position_tolerance,
            self.analytic_lateral_tolerance,
        ):
            return None

        end_pose = self._propagate(node.pose, 0.0, longitudinal)
        if not self._primitive_is_collision_free(
            grid,
            node.pose,
            0.0,
            longitudinal,
            check_start=False,
        ):
            return None
        soft_cost_exposure = self._primitive_soft_cost_exposure(
            grid, node.pose, 0.0, longitudinal
        )
        candidate = _SearchNode(
            pose=end_pose,
            cost=node.cost
            + self._transition_cost(
                grid,
                end_pose,
                self.curvatures[node.curvature_index],
                0.0,
                longitudinal,
                soft_cost_exposure,
            ),
            parent_id=node_id,
            curvature_index=self._straight_index,
            travel_distance=longitudinal,
        )
        if not self._goal_reached(candidate, goal, goal_curvature):
            return None
        return candidate

    def _reconstruct(self, nodes, goal_node_id, expanded_nodes):
        chain = []
        node_id = goal_node_id
        while node_id is not None:
            node = nodes[node_id]
            chain.append(node)
            node_id = node.parent_id
        chain.reverse()

        x_values = [chain[0].pose.x]
        y_values = [chain[0].pose.y]
        yaw_values = [chain[0].pose.yaw]
        curvature_values = [self.curvatures[chain[0].curvature_index]]
        for parent, child in zip(chain[:-1], chain[1:]):
            curvature = self.curvatures[child.curvature_index]
            curvature_values[-1] = curvature
            samples = self._motion_samples(
                parent.pose,
                curvature,
                child.travel_distance,
                self.path_sample_step,
            )
            for pose in samples:
                x_values.append(pose.x)
                y_values.append(pose.y)
                yaw_values.append(pose.yaw)
                curvature_values.append(curvature)
        return PlannedPath(
            x=np.asarray(x_values, dtype=np.float64),
            y=np.asarray(y_values, dtype=np.float64),
            yaw=np.asarray(yaw_values, dtype=np.float64),
            curvature=np.asarray(curvature_values, dtype=np.float64),
            cost=float(chain[-1].cost),
            expanded_nodes=int(expanded_nodes),
        )

    def plan(self, grid, start, goal, goal_curvature=0.0):
        """Plan from ``start`` to the goal region, or return ``None``.

        ``start`` and ``goal`` may be :class:`Pose2D` objects or ``(x, y,
        yaw)`` sequences.  Pass ``goal_curvature=None`` to omit the exit
        curvature constraint; the default requires a straight final primitive.
        """
        if not isinstance(grid, OccupancyGrid):
            raise TypeError("grid must be an OccupancyGrid")
        start = self._coerce_pose(start)
        goal = self._coerce_pose(goal)
        if not self._pose_is_collision_free(grid, start):
            return None
        if not self._pose_is_collision_free(grid, goal):
            return None

        start_node = _SearchNode(
            pose=start,
            cost=0.0,
            parent_id=None,
            curvature_index=self._straight_index,
            travel_distance=0.0,
        )
        nodes = {0: start_node}
        if self._goal_reached(start_node, goal, goal_curvature):
            return self._reconstruct(nodes, 0, 1)
        initial_analytic = self._analytic_goal_node(
            grid,
            start_node,
            0,
            goal,
            goal_curvature,
        )
        if initial_analytic is not None and (
            self.soft_obstacle_cost_weight <= 0.0
            or not grid.has_soft_cost
        ):
            nodes[1] = initial_analytic
            return self._reconstruct(nodes, 1, 1)

        goal_distance_field = None
        if self.use_goal_distance_heuristic:
            goal_distance_field = grid.goal_distance_field(goal.x, goal.y)
        start_heuristic = self._heuristic(
            grid,
            start,
            goal,
            goal_distance_field,
        )
        if not math.isfinite(start_heuristic):
            return None
        start_key = self._state_key(
            grid,
            start_node.pose,
            start_node.curvature_index,
        )
        best_cost = {start_key: 0.0}
        queue = []
        insertion_order = 0
        heapq.heappush(
            queue,
            (start_heuristic, insertion_order, 0, start_key),
        )
        expanded_nodes = 0
        next_node_id = 1
        if initial_analytic is not None:
            analytic_key = ("analytic_goal",)
            best_cost[analytic_key] = initial_analytic.cost
            nodes[next_node_id] = initial_analytic
            insertion_order += 1
            heapq.heappush(
                queue,
                (
                    initial_analytic.cost,
                    insertion_order,
                    next_node_id,
                    analytic_key,
                ),
            )
            next_node_id += 1

        while queue and expanded_nodes < self.maximum_iterations:
            _, _, node_id, state_key = heapq.heappop(queue)
            node = nodes[node_id]
            if node.cost > best_cost.get(state_key, math.inf) + 1e-12:
                continue
            expanded_nodes += 1
            if self._goal_reached(node, goal, goal_curvature):
                return self._reconstruct(nodes, node_id, expanded_nodes)
            if node_id != 0:
                analytic = self._analytic_goal_node(
                    grid,
                    node,
                    node_id,
                    goal,
                    goal_curvature,
                )
                if analytic is not None:
                    # Keep the straight connection as a goal candidate.  It is
                    # returned only when its complete cost reaches the front of
                    # the A* queue, so an early but needlessly long shot cannot
                    # replace a shorter lattice route.
                    analytic_key = ("analytic_goal",)
                    if analytic.cost + 1e-12 < best_cost.get(
                        analytic_key, math.inf
                    ):
                        best_cost[analytic_key] = analytic.cost
                        nodes[next_node_id] = analytic
                        insertion_order += 1
                        heapq.heappush(
                            queue,
                            (
                                analytic.cost,
                                insertion_order,
                                next_node_id,
                                analytic_key,
                            ),
                        )
                        next_node_id += 1

            previous_curvature = self.curvatures[node.curvature_index]
            for curvature_index, curvature in enumerate(self.curvatures):
                end_pose = self._propagate(
                    node.pose,
                    curvature,
                    self.primitive_step,
                )
                if not self._primitive_is_collision_free(
                    grid,
                    node.pose,
                    curvature,
                    self.primitive_step,
                    check_start=False,
                ):
                    continue
                soft_cost_exposure = self._primitive_soft_cost_exposure(
                    grid,
                    node.pose,
                    curvature,
                    self.primitive_step,
                )
                successor_key = self._state_key(
                    grid,
                    end_pose,
                    curvature_index,
                )
                successor_cost = node.cost + self._transition_cost(
                    grid,
                    end_pose,
                    previous_curvature,
                    curvature,
                    soft_cost_exposure=soft_cost_exposure,
                )
                if successor_cost + 1e-12 >= best_cost.get(
                    successor_key, math.inf
                ):
                    continue

                successor_heuristic = self._heuristic(
                    grid,
                    end_pose,
                    goal,
                    goal_distance_field,
                )
                if not math.isfinite(successor_heuristic):
                    continue

                best_cost[successor_key] = successor_cost
                successor = _SearchNode(
                    pose=end_pose,
                    cost=successor_cost,
                    parent_id=node_id,
                    curvature_index=curvature_index,
                    travel_distance=self.primitive_step,
                )
                successor_id = next_node_id
                next_node_id += 1
                nodes[successor_id] = successor
                insertion_order += 1
                priority = successor_cost + successor_heuristic
                heapq.heappush(
                    queue,
                    (
                        priority,
                        insertion_order,
                        successor_id,
                        successor_key,
                    ),
                )
        return None
