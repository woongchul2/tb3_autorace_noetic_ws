#!/usr/bin/env python3

import math
from pathlib import Path
import sys
import unittest


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PYTHON_DIR = PACKAGE_DIR / "src"
if str(PACKAGE_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PYTHON_DIR))

from custom_autorace_bringup.tunnel_costmap import TunnelCostmap


class TunnelCostmapTest(unittest.TestCase):
    def make_grid(self, width=7, height=3, data=None, **kwargs):
        if data is None:
            data = [0] * (width * height)
        return TunnelCostmap(
            static_data=data,
            width=width,
            height=height,
            resolution=1.0,
            origin_x=0.0,
            origin_y=0.0,
            **kwargs
        )

    @staticmethod
    def scan(grid, ranges, pose=(0.5, 1.5, 0.0), **kwargs):
        parameters = {
            "angle_min": 0.0,
            "angle_increment": 0.1,
            "range_min": 0.1,
            "range_max": 6.0,
            "sensor_pose": pose,
        }
        parameters.update(kwargs)
        return grid.update_scan(ranges=ranges, **parameters)

    def test_obstacle_free_infinite_scan_only_clears(self):
        grid = self.make_grid()

        update = self.scan(
            grid,
            [float("inf"), float("nan"), float("inf")],
            angle_min=-0.1,
        )

        self.assertEqual(grid.to_occupancy_data(), [0] * 21)
        self.assertEqual(update.beam_count, 3)
        self.assertEqual(update.hit_beams, 0)
        self.assertEqual(update.clearing_beams, 2)
        self.assertEqual(update.marked_cells, 0)

    def test_finite_hit_marks_endpoint_and_not_ray_cells(self):
        grid = self.make_grid()

        update = self.scan(grid, [3.0])

        self.assertEqual(grid.world_to_cell(3.5, 1.5), (3, 1))
        self.assertTrue(grid.is_dynamic_occupied(3, 1))
        self.assertFalse(grid.is_dynamic_occupied(2, 1))
        self.assertEqual(grid.to_occupancy_data()[1 * 7 + 3], 100)
        self.assertEqual(update.hit_beams, 1)
        self.assertEqual(update.marked_cells, 1)

    def test_infinite_ray_clears_a_previous_dynamic_hit(self):
        grid = self.make_grid(clear_observations=1)
        self.scan(grid, [3.0])
        self.assertTrue(grid.is_dynamic_occupied(3, 1))

        update = self.scan(grid, [float("inf")])

        self.assertFalse(grid.is_dynamic_occupied(3, 1))
        self.assertEqual(update.cleared_cells, 1)
        self.assertEqual(grid.to_occupancy_data(), [0] * 21)

    def test_clearing_never_removes_a_static_occupied_cell(self):
        static = [0] * 21
        static[1 * 7 + 3] = 100
        grid = self.make_grid(data=static)

        self.scan(grid, [float("inf")])
        self.scan(grid, [3.0])
        self.scan(grid, [float("inf")])

        self.assertTrue(grid.is_static_occupied(3, 1))
        self.assertFalse(grid.is_dynamic_occupied(3, 1))
        self.assertEqual(grid.to_occupancy_data()[1 * 7 + 3], 100)

    def test_static_hit_exclusion_suppresses_wall_offset_but_keeps_clearing(self):
        static = [0] * 21
        static[1 * 7 + 4] = 100
        grid = self.make_grid(
            data=static,
            static_hit_exclusion_radius=0.03,
        )
        self.scan(grid, [2.0])
        self.assertTrue(grid.is_dynamic_occupied(2, 1))

        # Endpoint x=3.98 lies in free cell 3, only 2 cm from the static wall
        # cell beginning at x=4.0.  It must not become a duplicate obstacle.
        update = self.scan(grid, [3.48])

        self.assertFalse(grid.is_dynamic_occupied(3, 1))
        self.assertFalse(grid.is_dynamic_occupied(2, 1))
        self.assertEqual(update.marked_cells, 0)
        self.assertEqual(update.cleared_cells, 1)
        self.assertEqual(grid.to_occupancy_data()[1 * 7 + 4], 100)

        # The YAML default of zero keeps the ordinary endpoint behaviour.
        without_exclusion = self.make_grid(data=static)
        self.scan(without_exclusion, [3.48])
        self.assertTrue(without_exclusion.is_dynamic_occupied(3, 1))

    def test_static_suppressed_endpoint_clears_old_dynamic_ghost(self):
        static = [0] * 21
        static[1 * 7 + 4] = 100
        grid = self.make_grid(
            data=static,
            static_hit_exclusion_radius=0.03,
            clear_observations=2,
        )

        # The first endpoint is safely away from the wall and creates a
        # dynamic observation in cell 3.
        self.scan(grid, [3.0])
        self.assertTrue(grid.is_dynamic_occupied(3, 1))

        # A later wall return lands in the same raster cell but is suppressed
        # because its endpoint is only 2 cm from static cell 4.  Since it is
        # not an accepted hit, it must count toward normal ray clearing.
        first_clear = self.scan(grid, [3.48])
        second_clear = self.scan(grid, [3.48])

        self.assertEqual(first_clear.marked_cells, 0)
        self.assertEqual(first_clear.cleared_cells, 0)
        self.assertEqual(second_clear.marked_cells, 0)
        self.assertEqual(second_clear.cleared_cells, 1)
        self.assertFalse(grid.is_dynamic_occupied(3, 1))

    def test_endpoint_clear_guard_preserves_nearby_hit_without_marking_neighbors(self):
        def small_grid(guard_radius):
            return TunnelCostmap(
                static_data=[0] * 30,
                width=10,
                height=3,
                resolution=0.02,
                origin_x=0.0,
                origin_y=0.0,
                endpoint_clear_guard_radius=guard_radius,
            )

        def small_scan(grid, distance):
            return grid.update_scan(
                ranges=[distance],
                angle_min=0.0,
                angle_increment=0.0,
                range_min=0.001,
                range_max=0.18,
                sensor_pose=(0.01, 0.03, 0.0),
            )

        guarded = small_grid(0.03)
        small_scan(guarded, 0.04)  # cell 2
        update = small_scan(guarded, 0.06)  # cell 3, traversing cell 2

        self.assertTrue(guarded.is_dynamic_occupied(2, 1))
        self.assertTrue(guarded.is_dynamic_occupied(3, 1))
        self.assertFalse(guarded.is_dynamic_occupied(4, 1))
        self.assertEqual(update.marked_cells, 1)
        self.assertEqual(update.cleared_cells, 0)

        # With the default radius, only the actual hit cell wins over clearing.
        unguarded = small_grid(0.0)
        small_scan(unguarded, 0.04)
        update = small_scan(unguarded, 0.06)
        self.assertFalse(unguarded.is_dynamic_occupied(2, 1))
        self.assertTrue(unguarded.is_dynamic_occupied(3, 1))
        self.assertEqual(update.cleared_cells, 1)

        # No-hit beams create no guard and retain their normal clearing role.
        update = small_scan(guarded, float("inf"))
        self.assertFalse(guarded.is_dynamic_occupied(2, 1))
        self.assertFalse(guarded.is_dynamic_occupied(3, 1))
        self.assertEqual(update.cleared_cells, 2)

    def test_unknown_cell_is_exported_as_dynamic_obstacle_then_restored(self):
        static = [-1] * 21
        grid = self.make_grid(data=static)

        self.scan(grid, [3.0])

        combined = grid.to_occupancy_data()
        self.assertEqual(combined[1 * 7 + 3], 100)
        self.assertEqual(combined[1 * 7 + 4], -1)

        self.scan(grid, [float("inf")])
        self.assertEqual(grid.to_occupancy_data()[1 * 7 + 3], -1)

    def test_dynamic_inflation_is_circular_and_export_only(self):
        static = [0] * 25
        static[2 * 5 + 3] = 80
        grid = TunnelCostmap(
            static_data=static,
            width=5,
            height=5,
            resolution=1.0,
            origin_x=0.0,
            origin_y=0.0,
            dynamic_occupied_value=100,
            dynamic_inflation_radius=1.0,
        )

        update = self.scan(
            grid,
            [2.0],
            pose=(0.5, 2.5, 0.0),
            range_max=4.0,
        )
        combined = grid.to_occupancy_data()

        self.assertEqual(update.marked_cells, 1)
        self.assertTrue(grid.is_dynamic_occupied(2, 2))
        self.assertFalse(grid.is_dynamic_occupied(1, 2))
        self.assertFalse(grid.is_dynamic_occupied(2, 1))
        self.assertEqual(combined[2 * 5 + 2], 100)
        self.assertEqual(combined[2 * 5 + 1], 100)
        self.assertEqual(combined[1 * 5 + 2], 100)
        self.assertEqual(combined[3 * 5 + 2], 100)
        self.assertEqual(combined[2 * 5 + 3], 80)
        self.assertEqual(combined[1 * 5 + 1], 0)
        self.assertEqual(combined[1 * 5 + 3], 0)
        self.assertEqual(grid.static_data, static)

    def test_dynamic_inflation_clips_at_grid_boundary(self):
        grid = TunnelCostmap(
            static_data=[0] * 16,
            width=4,
            height=4,
            resolution=1.0,
            origin_x=0.0,
            origin_y=0.0,
            dynamic_inflation_radius=1.5,
        )

        self.scan(
            grid,
            [0.1],
            pose=(0.5, 0.5, 0.0),
            range_min=0.0,
            range_max=3.0,
        )
        combined = grid.to_occupancy_data()

        self.assertEqual(len(combined), 16)
        self.assertEqual(
            [
                index
                for index, value in enumerate(combined)
                if value == 100
            ],
            [0, 1, 4, 5],
        )
        self.assertTrue(grid.is_dynamic_occupied(0, 0))
        self.assertFalse(grid.is_dynamic_occupied(1, 0))

    def test_soft_inflation_is_graded_while_raw_endpoint_stays_lethal(self):
        static = [0] * 49
        static[3 * 7 + 0] = -1
        grid = TunnelCostmap(
            static_data=static,
            width=7,
            height=7,
            resolution=1.0,
            origin_x=0.0,
            origin_y=0.0,
            dynamic_occupied_value=100,
            dynamic_inflation_value=64,
            dynamic_inflation_radius=3.0,
        )

        self.scan(
            grid,
            [3.0],
            pose=(0.5, 3.5, 0.0),
            range_max=6.0,
        )

        combined = grid.to_occupancy_data()
        soft = grid.to_soft_cost_data()
        raw = grid.to_raw_occupancy_data()
        row = 3 * 7
        self.assertEqual(combined[row + 3], 100)
        self.assertEqual(soft[row + 3], 0)
        self.assertEqual(raw[row + 3], 100)
        self.assertEqual(combined[row + 2], 64)
        self.assertEqual(combined[row + 1], 43)
        self.assertEqual(combined[row + 0], -1)
        self.assertEqual(soft[row + 2], 64)
        self.assertEqual(soft[row + 1], 43)
        self.assertEqual(soft[row + 0], 0)
        self.assertEqual(raw[row + 2], 0)
        self.assertEqual(raw[row + 1], 0)
        self.assertEqual(raw[row + 0], -1)

        # A diagonal offset in the outer radial band receives one-third risk.
        self.assertEqual(combined[1 * 7 + 1], 21)
        self.assertEqual(soft[1 * 7 + 1], 21)

    def test_soft_inflation_value_must_not_exceed_raw_value(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self.make_grid(
                dynamic_occupied_value=60,
                dynamic_inflation_value=61,
            )

    def test_dynamic_inflation_radius_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            self.make_grid(dynamic_inflation_radius=-0.01)

    def test_observation_counts_and_unobserved_decay_are_scan_based(self):
        grid = self.make_grid(mark_observations=2, decay_updates=2)

        # Repeated identical beams in one scan still count as one observation.
        self.scan(grid, [3.0, 3.0], angle_increment=0.0)
        self.assertFalse(grid.is_dynamic_occupied(3, 1))
        self.scan(grid, [3.0])
        self.assertTrue(grid.is_dynamic_occupied(3, 1))

        self.scan(grid, [])
        self.assertTrue(grid.is_dynamic_occupied(3, 1))
        update = self.scan(grid, [])
        self.assertFalse(grid.is_dynamic_occupied(3, 1))
        self.assertEqual(update.decayed_cells, 1)

    def test_unobserved_candidate_hit_does_not_satisfy_consecutive_count(self):
        grid = self.make_grid(mark_observations=2)

        self.scan(grid, [3.0])
        self.scan(grid, [])
        self.scan(grid, [3.0])

        self.assertFalse(grid.is_dynamic_occupied(3, 1))
        self.scan(grid, [3.0])
        self.assertTrue(grid.is_dynamic_occupied(3, 1))

    def test_crop_uses_grid_aligned_bounds_and_row_major_static_data(self):
        source = list(range(24))
        grid = TunnelCostmap(
            static_data=source,
            width=6,
            height=4,
            resolution=1.0,
            origin_x=-1.0,
            origin_y=-2.0,
            planning_bounds=(0.0, -1.0, 3.0, 1.0),
        )

        self.assertEqual(grid.shape, (2, 3))
        self.assertEqual((grid.origin_x, grid.origin_y), (0.0, -1.0))
        self.assertEqual(grid.bounds, (0.0, -1.0, 3.0, 1.0))
        self.assertEqual((grid.source_column, grid.source_row), (1, 1))
        self.assertEqual(grid.static_data, [7, 8, 9, 13, 14, 15])

    def test_ray_is_clipped_when_sensor_starts_outside_crop(self):
        grid = TunnelCostmap(
            static_data=[0] * 24,
            width=6,
            height=4,
            resolution=1.0,
            origin_x=-1.0,
            origin_y=-2.0,
            planning_bounds=(0.0, -1.0, 3.0, 1.0),
        )

        update = self.scan(
            grid,
            [3.5],
            pose=(-2.0, 0.5, 0.0),
            range_max=6.0,
        )

        # The ray enters through x=0 and marks its in-crop endpoint x=1.5.
        self.assertFalse(grid.is_dynamic_occupied(0, 1))
        self.assertTrue(grid.is_dynamic_occupied(1, 1))
        self.assertEqual(update.marked_cells, 1)

        # An infinite beam exits through the crop boundary and clears both
        # remaining in-grid cells without any out-of-range indexing.
        self.scan(
            grid,
            [float("inf")],
            pose=(1.5, 0.5, 0.0),
            range_max=6.0,
        )
        self.assertEqual(grid.to_occupancy_data(), [0] * 6)

    def test_rotated_sensor_pose_places_hit_in_map_frame(self):
        grid = self.make_grid(width=4, height=4)

        self.scan(
            grid,
            [2.0],
            pose=(1.5, 0.5, math.pi / 2.0),
            range_max=3.0,
        )

        self.assertTrue(grid.is_dynamic_occupied(1, 2))


if __name__ == "__main__":
    unittest.main()
