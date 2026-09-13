#!/usr/bin/env python3

import math
from pathlib import Path
import sys
import unittest

import numpy as np


PACKAGE_SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(PACKAGE_SOURCE) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SOURCE))

from custom_autorace_bringup.tunnel_planner import (  # noqa: E402
    HybridAStarPlanner,
    OccupancyGrid,
    Pose2D,
    RectangularFootprint,
    normalize_angle,
)


class TunnelPlannerTest(unittest.TestCase):
    def setUp(self):
        self.footprint = RectangularFootprint(
            front=0.16,
            rear=0.10,
            half_width=0.09,
            padding=0.01,
        )
        self.planner_arguments = dict(
            footprint=self.footprint,
            heading_bins=48,
            minimum_turning_radius=0.40,
            primitive_step=0.10,
            steering_samples=3,
            goal_position_tolerance=0.12,
            goal_heading_tolerance=math.radians(9.0),
            goal_curvature_tolerance=1e-9,
            non_straight_penalty=0.08,
            steering_change_penalty=0.04,
            inflation_radius=0.0,
            obstacle_cost_weight=0.10,
            obstacle_cost_distance=0.35,
            heading_heuristic_weight=0.30,
            collision_check_step=0.04,
            collision_check_angle=math.radians(3.0),
            path_sample_step=0.04,
            state_xy_resolution=0.10,
            maximum_iterations=50000,
        )
        self.planner = HybridAStarPlanner(**self.planner_arguments)

    @staticmethod
    def make_grid(
        width,
        height,
        resolution,
        origin_x,
        origin_y,
        occupied_rectangles=(),
    ):
        data = np.zeros((height, width), dtype=np.int8)
        grid = OccupancyGrid(data, resolution, origin_x, origin_y)
        for minimum_x, maximum_x, minimum_y, maximum_y in (
            occupied_rectangles
        ):
            first_x, first_y = grid.world_to_grid(minimum_x, minimum_y)
            last_x, last_y = grid.world_to_grid(maximum_x, maximum_y)
            first_x = max(0, first_x)
            first_y = max(0, first_y)
            last_x = min(width - 1, last_x)
            last_y = min(height - 1, last_y)
            data[first_y : last_y + 1, first_x : last_x + 1] = 100
        return OccupancyGrid(data, resolution, origin_x, origin_y)

    def assert_path_is_collision_free(self, grid, path):
        self.assertIsNotNone(path)
        for pose in path.poses:
            self.assertTrue(
                self.planner.pose_is_collision_free(grid, pose),
                msg="collision at ({:.3f}, {:.3f}, {:.1f} deg)".format(
                    pose.x,
                    pose.y,
                    math.degrees(pose.yaw),
                ),
            )

    def test_empty_grid_produces_straight_path(self):
        grid = self.make_grid(80, 50, 0.05, -0.5, -1.25)
        path = self.planner.plan(grid, (0.0, 0.0, 0.0), (2.6, 0.0, 0.0))

        self.assert_path_is_collision_free(grid, path)
        self.assertEqual(path.expanded_nodes, 1)
        self.assertAlmostEqual(path.length, 2.6, places=10)
        self.assertAlmostEqual(path.x[-1], 2.6, places=10)
        self.assertAlmostEqual(path.y[-1], 0.0, places=10)
        self.assertLess(float(np.max(np.abs(path.y))), 1e-10)
        self.assertLess(float(np.max(np.abs(path.yaw))), 1e-10)
        self.assertLess(float(np.max(np.abs(path.curvature))), 1e-10)

    def test_central_obstacle_is_avoided_and_exit_is_recovered(self):
        grid = self.make_grid(
            80,
            50,
            0.05,
            -0.5,
            -1.25,
            occupied_rectangles=((1.15, 1.45, -0.20, 0.20),),
        )
        goal = Pose2D(2.6, 0.0, 0.0)
        path = self.planner.plan(grid, Pose2D(0.0, 0.0, 0.0), goal)

        self.assert_path_is_collision_free(grid, path)
        self.assertGreater(path.expanded_nodes, 1)
        self.assertGreater(float(np.max(np.abs(path.y))), 0.30)
        self.assertGreater(float(np.max(np.abs(path.curvature))), 0.0)
        self.assertLess(
            math.hypot(path.x[-1] - goal.x, path.y[-1] - goal.y),
            self.planner.goal_position_tolerance + 1e-9,
        )
        self.assertLess(
            abs(normalize_angle(path.yaw[-1] - goal.yaw)),
            self.planner.goal_heading_tolerance + 1e-9,
        )

    def test_exit_heading_and_straight_curvature_are_constraints(self):
        grid = self.make_grid(70, 60, 0.05, -0.5, -0.5)
        goal = Pose2D(1.5, 0.6, math.pi / 2.0)
        path = self.planner.plan(grid, (0.0, 0.0, 0.0), goal)

        self.assert_path_is_collision_free(grid, path)
        self.assertLess(
            abs(normalize_angle(path.yaw[-1] - goal.yaw)),
            self.planner.goal_heading_tolerance + 1e-9,
        )
        self.assertAlmostEqual(path.curvature[-1], 0.0, places=12)
        self.assertGreater(float(np.max(np.abs(path.curvature[:-1]))), 0.0)
        self.assertLessEqual(
            float(np.max(np.abs(path.curvature))),
            1.0 / self.planner.minimum_turning_radius + 1e-12,
        )

    def test_production_steering_resolution_avoids_near_exit_loop(self):
        """An arbitrary near-goal yaw phase must not require a tunnel lap."""
        grid = self.make_grid(100, 100, 0.02, -1.0, -1.0)
        arguments = dict(self.planner_arguments)
        arguments.update(
            footprint=RectangularFootprint(
                front=0.067645,
                rear=0.118073,
                half_width=0.0903,
                padding=0.010,
            ),
            heading_bins=36,
            minimum_turning_radius=0.18,
            primitive_step=0.10,
            steering_samples=5,
            state_xy_resolution=0.050,
            goal_position_tolerance=0.035,
            goal_heading_tolerance=math.radians(8.0),
            collision_check_step=0.010,
            path_sample_step=0.020,
        )
        planner = HybridAStarPlanner(**arguments)
        path = planner.plan(
            grid,
            Pose2D(-0.306, -0.745, math.radians(8.6)),
            Pose2D(0.200, -0.745, 0.0),
            goal_curvature=0.0,
        )

        self.assertIsNotNone(path)
        self.assertLess(path.length, 1.0)
        self.assertLess(path.expanded_nodes, 100)

    def test_full_width_wall_returns_failure(self):
        grid = self.make_grid(
            42,
            18,
            0.05,
            -0.3,
            -0.45,
            occupied_rectangles=((0.65, 0.75, -0.45, 0.45),),
        )
        path = self.planner.plan(grid, (0.0, 0.0, 0.0), (1.5, 0.0, 0.0))
        self.assertIsNone(path)

    def test_keep_in_rectangles_constrain_base_to_their_union(self):
        arguments = dict(self.planner_arguments)
        arguments["keep_in_rectangles"] = (
            (-0.5, -0.2, 0.8, 0.2),
            (0.6, -0.2, 1.8, 0.8),
        )
        planner = HybridAStarPlanner(**arguments)
        grid = self.make_grid(50, 40, 0.05, -0.5, -0.5)

        self.assertTrue(
            planner.pose_is_collision_free(grid, (0.0, 0.0, 0.0))
        )
        self.assertTrue(
            planner.pose_is_collision_free(grid, (0.7, 0.1, 0.0))
        )
        self.assertTrue(
            planner.pose_is_collision_free(grid, (1.5, 0.6, 0.0))
        )
        self.assertTrue(planner.pose_is_within_keep_in((0.7, 0.1, 0.0)))
        self.assertFalse(
            planner.pose_is_collision_free(grid, (0.0, 0.3, 0.0))
        )
        self.assertFalse(
            planner.pose_is_collision_free(grid, (1.2, -0.3, 0.0))
        )
        self.assertFalse(planner.pose_is_within_keep_in((0.0, 0.3, 0.0)))

    def test_keep_in_rectangles_reject_malformed_geometry(self):
        arguments = dict(self.planner_arguments)
        arguments["keep_in_rectangles"] = ((0.0, 0.0, 1.0),)
        with self.assertRaisesRegex(ValueError, "four values"):
            HybridAStarPlanner(**arguments)
        arguments["keep_in_rectangles"] = ((1.0, 0.0, 0.0, 1.0),)
        with self.assertRaisesRegex(ValueError, "minimums"):
            HybridAStarPlanner(**arguments)

    def test_oriented_asymmetric_footprint_and_primitive_sweep(self):
        data = np.zeros((200, 200), dtype=np.int8)
        empty_grid = OccupancyGrid(data, 0.01, -1.0, -1.0)

        forward_cell = empty_grid.world_to_grid(0.13, 0.0)
        data[forward_cell[1], forward_cell[0]] = 100
        forward_grid = OccupancyGrid(data, 0.01, -1.0, -1.0)
        self.assertFalse(
            self.planner.pose_is_collision_free(
                forward_grid, Pose2D(0.0, 0.0, 0.0)
            )
        )
        self.assertTrue(
            self.planner.pose_is_collision_free(
                forward_grid, Pose2D(0.0, 0.0, math.pi)
            )
        )

        sweep_data = np.zeros((200, 200), dtype=np.int8)
        sweep_grid = OccupancyGrid(sweep_data, 0.01, -1.0, -1.0)
        obstacle_x, obstacle_y = sweep_grid.world_to_grid(0.25, 0.0)
        sweep_data[obstacle_y, obstacle_x] = 100
        sweep_grid = OccupancyGrid(sweep_data, 0.01, -1.0, -1.0)
        start = Pose2D(0.0, 0.0, 0.0)
        end = Pose2D(0.50, 0.0, 0.0)
        self.assertTrue(self.planner.pose_is_collision_free(sweep_grid, start))
        self.assertTrue(self.planner.pose_is_collision_free(sweep_grid, end))
        self.assertFalse(
            self.planner.primitive_is_collision_free(
                sweep_grid,
                start,
                curvature=0.0,
                distance=0.50,
            )
        )

    def test_soft_cost_is_explicit_and_primitive_sweep_integrated(self):
        data = np.zeros((80, 100), dtype=np.int8)
        # A sub-threshold value in occupancy data alone is not dynamic-halo
        # provenance and must not silently become a planner cost.
        data[40, 50] = 40
        ordinary = OccupancyGrid(data, 0.02, -0.5, -0.8)
        self.assertFalse(ordinary.has_soft_cost)

        soft = np.zeros_like(data)
        soft[37:44, 48:53] = 64
        grid = OccupancyGrid(
            data,
            0.02,
            -0.5,
            -0.8,
            occupied_threshold=65,
            soft_cost_data=soft,
        )
        arguments = dict(self.planner_arguments)
        arguments.update(
            soft_obstacle_cost_weight=1.0,
            collision_check_step=0.01,
        )
        planner = HybridAStarPlanner(**arguments)
        start = Pose2D(0.0, 0.0, 0.0)
        end = Pose2D(1.0, 0.0, 0.0)

        self.assertEqual(planner._pose_soft_cost(grid, start), 0.0)
        self.assertEqual(planner._pose_soft_cost(grid, end), 0.0)
        self.assertTrue(planner.pose_is_collision_free(grid, start))
        self.assertTrue(planner.pose_is_collision_free(grid, end))
        self.assertGreater(
            planner._primitive_soft_cost_exposure(
                grid, start, curvature=0.0, distance=1.0
            ),
            0.0,
        )

        with self.assertRaisesRegex(ValueError, "match occupancy"):
            OccupancyGrid(
                data,
                0.02,
                -0.5,
                -0.8,
                occupied_threshold=65,
                soft_cost_data=np.zeros((2, 2)),
            )

    def test_soft_cost_can_outweigh_a_straight_analytic_connection(self):
        data = np.zeros((100, 160), dtype=np.int8)
        soft = np.zeros_like(data)
        base = OccupancyGrid(data, 0.02, -0.5, -1.0)
        first_x, first_y = base.world_to_grid(0.85, -0.18)
        last_x, last_y = base.world_to_grid(1.15, 0.18)
        soft[first_y : last_y + 1, first_x : last_x + 1] = 64
        grid = OccupancyGrid(
            data,
            0.02,
            -0.5,
            -1.0,
            occupied_threshold=65,
            soft_cost_data=soft,
        )
        arguments = dict(self.planner_arguments)
        arguments.update(
            soft_obstacle_cost_weight=5.0,
            collision_check_step=0.01,
        )
        planner = HybridAStarPlanner(**arguments)

        path = planner.plan(grid, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))

        self.assertIsNotNone(path)
        self.assertGreater(float(np.max(np.abs(path.y))), 0.20)
        exposure = 0.0
        for index in range(path.x.size - 1):
            distance = math.hypot(
                path.x[index + 1] - path.x[index],
                path.y[index + 1] - path.y[index],
            )
            exposure += distance * planner._pose_soft_cost(
                grid, path.poses[index + 1]
            )
        self.assertLess(exposure, 0.03)

    def test_flat_ros_grid_api_preserves_origin_and_unknown_policy(self):
        values = [0, 0, 0, -1, 0, 100]
        grid = OccupancyGrid.from_flat(
            values,
            width=3,
            height=2,
            resolution=0.2,
            origin_x=-0.4,
            origin_y=1.0,
        )
        self.assertEqual(grid.world_to_grid(-0.1, 1.3), (1, 1))
        world_x, world_y = grid.grid_to_world(1, 1)
        self.assertAlmostEqual(world_x, -0.1)
        self.assertAlmostEqual(world_y, 1.3)
        self.assertTrue(grid.is_occupied_cell(0, 1))
        self.assertTrue(grid.is_occupied_cell(2, 1))
        self.assertTrue(grid.is_occupied_cell(-1, 0))

    def test_raw_goal_distance_field_routes_around_obstacle(self):
        grid = self.make_grid(
            80,
            50,
            0.05,
            -0.5,
            -1.25,
            occupied_rectangles=((1.15, 1.45, -0.20, 0.20),),
        )
        distances = grid.goal_distance_field(2.6, 0.0)
        start_x, start_y = grid.world_to_grid(0.0, 0.0)
        self.assertGreater(distances[start_y, start_x], 2.6)

    def test_goal_distance_heuristic_reduces_hybrid_expansions(self):
        grid = self.make_grid(
            80,
            50,
            0.05,
            -0.5,
            -1.25,
            occupied_rectangles=((1.15, 1.45, -0.20, 0.20),),
        )
        unguided_arguments = dict(self.planner_arguments)
        unguided_arguments.update(
            use_goal_distance_heuristic=False,
            enable_analytic_goal_connection=False,
        )
        guided_arguments = dict(unguided_arguments)
        guided_arguments["use_goal_distance_heuristic"] = True
        unguided = HybridAStarPlanner(**unguided_arguments).plan(
            grid,
            (0.0, 0.0, 0.0),
            (2.6, 0.0, 0.0),
        )
        guided = HybridAStarPlanner(**guided_arguments).plan(
            grid,
            (0.0, 0.0, 0.0),
            (2.6, 0.0, 0.0),
        )

        self.assertIsNotNone(unguided)
        self.assertIsNotNone(guided)
        self.assertLess(guided.expanded_nodes, unguided.expanded_nodes)
        self.assertLessEqual(guided.cost, unguided.cost + 1e-9)


if __name__ == "__main__":
    unittest.main()
