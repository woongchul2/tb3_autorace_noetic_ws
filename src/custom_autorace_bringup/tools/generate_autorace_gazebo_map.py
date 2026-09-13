#!/usr/bin/env python3
"""Generate the AMCL map from lidar-visible geometry in the Gazebo world."""

import math
from pathlib import Path


RESOLUTION = 0.02
ORIGIN_X = -2.5
ORIGIN_Y = -2.5
WIDTH = 250
HEIGHT = 250


def occupied_by_rectangle(x, y, center_x, center_y, size_x, size_y, yaw=0.0):
    dx, dy = x - center_x, y - center_y
    cosine, sine = math.cos(yaw), math.sin(yaw)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    return abs(local_x) <= 0.5 * size_x and abs(local_y) <= 0.5 * size_y


def make_map():
    # Only geometry whose pose is fixed by the course belongs in the AMCL map.
    # The three cylinders inside the tunnel are deliberately omitted: their
    # competition pose is not known before a run, so LiDAR adds them to the
    # tunnel controller's live costmap instead of leaking their Gazebo pose
    # into localization and global planning.
    rectangles = [
        # The world has a saved <state> whose absolute link poses override the
        # tunnel model's local link poses at startup.  AMCL must therefore use
        # these runtime collision centres, not model-pose-plus-link-pose sums.
        (-1.915480, -1.005857, 1.8, 0.05, math.pi / 2.0),
        (-0.994699, -1.921180, 1.8, 0.05, 0.0),
        (-0.021303, -0.832071, 1.53699, 0.05, math.pi / 2.0),
        (-0.814699, -0.030740, 1.58, 0.05, math.pi),
        # Fixed construction barriers in this Gazebo course revision.
        (1.49, 0.54, 0.25, 0.10, 0.0),
        (1.74, 1.00, 0.25, 0.10, 0.0),
        (1.49, 1.464, 0.25, 0.10, 0.0),
        # Fixed sign plates (x, y, plate x, plate y, world yaw).
        (0.74, 1.95, 0.12, 0.025, -math.pi / 2.0),
        (-1.544, 0.00, 0.12, 0.025, 0.0),
        (-1.35, 1.04, 0.12, 0.025, -math.pi / 2.0),
        (0.50, 1.90, 0.12, 0.025, 0.0),
        (1.00, 0.04, 0.12, 0.025, -math.pi / 2.0),
        (1.95, -1.00, 0.12, 0.025, 0.0),
        (0.95, -0.758, 0.12, 0.025, 0.0),
    ]
    pixels = bytearray([254] * (WIDTH * HEIGHT))
    for row in range(HEIGHT):
        y = ORIGIN_Y + (row + 0.5) * RESOLUTION
        for column in range(WIDTH):
            x = ORIGIN_X + (column + 0.5) * RESOLUTION
            occupied = any(
                occupied_by_rectangle(x, y, *rectangle)
                for rectangle in rectangles
            )
            if occupied:
                # PGM rows are top-to-bottom while map cells are bottom-to-top.
                pgm_row = HEIGHT - row - 1
                pixels[pgm_row * WIDTH + column] = 0
    return b"P5\n# AutoRace Gazebo lidar geometry\n250 250\n255\n" + bytes(pixels)


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "maps" / "autorace_gazebo.pgm"
    output.write_bytes(make_map())
    print(output)
