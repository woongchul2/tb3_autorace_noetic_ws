#!/usr/bin/env python3

import math
import sys
import unittest
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PACKAGE_PYTHON_DIR = PACKAGE_DIR / "src"
if str(PACKAGE_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PYTHON_DIR))

from custom_autorace_bringup.parking_geometry import (
    path_curvature_limits,
    quintic_pose_path,
    quintic_turn_path,
)

SOURCE_DIR = PACKAGE_DIR.parent
COURSE_IMAGE = (
    SOURCE_DIR
    / "turtlebot3_simulations"
    / "turtlebot3_gazebo"
    / "models"
    / "turtlebot3_autorace_2020"
    / "course"
    / "materials"
    / "textures"
    / "course.png"
)
BURGER_MODEL = (
    SOURCE_DIR
    / "turtlebot3_simulations"
    / "turtlebot3_gazebo"
    / "models"
    / "turtlebot3_burger"
    / "model.sdf"
)
CUSTOM_ROBOT_PARAMETERS = (
    SOURCE_DIR
    / "custom_autorace_description"
    / "urdf"
    / "robot_parameters.xacro"
)
PARKING_CONFIG = PACKAGE_DIR / "config" / "parking_mission_gazebo.yaml"

# turtlebot3_autorace_2020.world places the 4 m square course at yaw=-3.14.
COURSE_SIZE = 4.0
COURSE_YAW = -3.14
SAMPLE_COUNT = 1001
REGRESSION_CLEARANCE = 0.012


def normalize_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def pixel_corner_to_world(u, v, width, height):
    local_x = COURSE_SIZE * (float(u) / width - 0.5)
    local_y = COURSE_SIZE * (0.5 - float(v) / height)
    cosine = math.cos(COURSE_YAW)
    sine = math.sin(COURSE_YAW)
    return np.asarray(
        [
            cosine * local_x - sine * local_y,
            sine * local_x + cosine * local_y,
        ],
        dtype=np.float64,
    )


def mask_rectangles(mask):
    """Merge identical horizontal pixel runs on adjacent rows."""
    active = {}
    rectangles = []
    for v, row in enumerate(mask):
        columns = np.flatnonzero(row)
        runs = []
        if columns.size:
            starts = np.r_[0, np.flatnonzero(np.diff(columns) > 1) + 1]
            ends = np.r_[starts[1:] - 1, columns.size - 1]
            runs = [
                (int(columns[start]), int(columns[end]))
                for start, end in zip(starts, ends)
            ]

        current = set(runs)
        for run, (first_row, last_row) in list(active.items()):
            if run not in current:
                rectangles.append((run[0], run[1] + 1, first_row, last_row + 1))
                del active[run]
        for run in runs:
            if run in active:
                active[run] = (active[run][0], v)
            else:
                active[run] = (v, v)

    for run, (first_row, last_row) in active.items():
        rectangles.append((run[0], run[1] + 1, first_row, last_row + 1))
    return rectangles


def pixel_rectangles_to_world(rectangles, width, height):
    polygons = []
    for minimum_u, maximum_u, minimum_v, maximum_v in rectangles:
        polygons.append(
            np.asarray(
                [
                    pixel_corner_to_world(minimum_u, minimum_v, width, height),
                    pixel_corner_to_world(maximum_u, minimum_v, width, height),
                    pixel_corner_to_world(maximum_u, maximum_v, width, height),
                    pixel_corner_to_world(minimum_u, maximum_v, width, height),
                ]
            )
        )
    return polygons


def transformed_box(pose, minimum_x, maximum_x, minimum_y, maximum_y):
    x, y, yaw = pose
    local = np.asarray(
        [
            [minimum_x, minimum_y],
            [maximum_x, minimum_y],
            [maximum_x, maximum_y],
            [minimum_x, maximum_y],
        ],
        dtype=np.float64,
    )
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + np.asarray([x, y])


def point_segment_distance(point, start, end):
    direction = end - start
    fraction = np.dot(point - start, direction) / np.dot(direction, direction)
    fraction = min(1.0, max(0.0, float(fraction)))
    return float(np.linalg.norm(point - (start + fraction * direction)))


def convex_polygons_intersect(first, second):
    for polygon in (first, second):
        for index, start in enumerate(polygon):
            edge = polygon[(index + 1) % len(polygon)] - start
            axis = np.asarray([-edge[1], edge[0]])
            first_projection = first @ axis
            second_projection = second @ axis
            if (
                np.max(first_projection) < np.min(second_projection)
                or np.max(second_projection) < np.min(first_projection)
            ):
                return False
    return True


def convex_polygon_distance(first, second):
    if convex_polygons_intersect(first, second):
        return 0.0
    distances = []
    for point in first:
        for index, start in enumerate(second):
            distances.append(
                point_segment_distance(
                    point, start, second[(index + 1) % len(second)]
                )
            )
    for point in second:
        for index, start in enumerate(first):
            distances.append(
                point_segment_distance(
                    point, start, first[(index + 1) % len(first)]
                )
            )
    return min(distances)


def sample_in_place_rotation(start_pose, target_yaw):
    turn_angle = normalize_angle(float(target_yaw) - float(start_pose[2]))
    if not math.isclose(
        abs(turn_angle), 0.5 * math.pi, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("in-place parking rotation must be 90 degrees")
    return sample_rotation(start_pose, target_yaw)


def sample_rotation(start_pose, target_yaw, sample_count=SAMPLE_COUNT):
    turn_angle = normalize_angle(float(target_yaw) - float(start_pose[2]))
    yaw = float(start_pose[2]) + np.linspace(0.0, turn_angle, sample_count)
    return np.column_stack(
        (
            np.full(sample_count, float(start_pose[0])),
            np.full(sample_count, float(start_pose[1])),
            yaw,
        )
    )


def sample_straight(start_pose, goal_pose):
    fraction = np.linspace(0.0, 1.0, SAMPLE_COUNT)
    return np.column_stack(
        (
            start_pose[0] + fraction * (goal_pose[0] - start_pose[0]),
            start_pose[1] + fraction * (goal_pose[1] - start_pose[1]),
            np.full(SAMPLE_COUNT, start_pose[2]),
        )
    )


class ParkingRouteGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with PARKING_CONFIG.open(encoding="utf-8") as stream:
            cls.config = yaml.safe_load(stream)["parking"]
        cls.route = cls.config["route"]
        footprint = cls.config["footprint"]
        cls.front = float(footprint["front"])
        cls.rear = float(footprint["rear"])
        cls.half_width = float(footprint["half_width"])
        cls.line_margin = float(footprint["line_margin"])
        safety = cls.config["safety"]
        cls.expanded_footprint_margin = (
            cls.line_margin
            + float(safety["localization_margin"])
            + float(safety["tracking_margin"])
        )

        image = np.asarray(Image.open(COURSE_IMAGE).convert("RGB"))
        cls.image_height, cls.image_width = image.shape[:2]
        white = np.all(image >= 160, axis=2)
        yellow = (
            (image[:, :, 0] >= 160)
            & (image[:, :, 1] >= 160)
            & (image[:, :, 2] < 160)
        )

        # The two vertical three-pixel columns are the legal dotted entrances.
        # Their last dash ends at v=371; the horizontal arms beginning at v=372
        # are continuous solid paint and must remain forbidden.
        parking_solid = np.zeros_like(white)
        parking_solid[300:381, 135:251] = white[300:381, 135:251]
        parking_solid[320:372, 177:180] = False
        parking_solid[320:372, 209:212] = False
        cls.parking_solid = pixel_rectangles_to_world(
            mask_rectangles(parking_solid), cls.image_width, cls.image_height
        )

        # Only continuous yellow/white boundaries occur in the final-turn crop.
        final_solid = np.zeros_like(white)
        final_solid[410:496, 140:261] = (white | yellow)[410:496, 140:261]
        cls.final_solid = pixel_rectangles_to_world(
            mask_rectangles(final_solid), cls.image_width, cls.image_height
        )

        # Continuous approach/intersection paint around the north aisle entry.
        entry_solid = np.zeros_like(white)
        entry_solid[390:496, 90:270] = (white | yellow)[390:496, 90:270]
        cls.entry_solid = pixel_rectangles_to_world(
            mask_rectangles(entry_solid), cls.image_width, cls.image_height
        )
        cls.parking_route_solid = cls.parking_solid + cls.entry_solid

        # Runtime fixed-obstacle rectangles must describe these independently
        # measured Gazebo collision boxes exactly.
        cls.configured_entry_signs = [
            np.asarray(
                [
                    [minimum_x, minimum_y],
                    [maximum_x, minimum_y],
                    [maximum_x, maximum_y],
                    [minimum_x, maximum_y],
                ],
                dtype=np.float64,
            )
            for minimum_x, maximum_x, minimum_y, maximum_y in cls.config[
                "fixed_obstacles"
            ]["rectangles"]
        ]
        cls.entry_signs = [
            transformed_box((0.50, 1.90, 0.0), -0.06, 0.06, -0.0125, 0.0125),
            transformed_box(
                (0.74, 1.95, -0.5 * math.pi),
                -0.06,
                0.06,
                -0.0125,
                0.0125,
            ),
        ]

        parameters_root = ElementTree.parse(
            str(CUSTOM_ROBOT_PARAMETERS)
        ).getroot()
        parameter_values = {
            element.attrib["name"]: element.attrib["value"]
            for element in parameters_root.iter()
            if element.tag.endswith("}property")
            and "name" in element.attrib
            and "value" in element.attrib
        }

        def numeric_parameter(name):
            return float(parameter_values[name])

        wheel_radius = numeric_parameter("wheel_radius")
        wheel_width = numeric_parameter("wheel_width")
        wheel_separation = numeric_parameter("wheel_separation")
        wheel_x = numeric_parameter("wheel_x")
        base_length = numeric_parameter("base_length")
        base_width = numeric_parameter("base_width")
        base_height = numeric_parameter("base_height")
        base_x = numeric_parameter("base_x")
        base_y = numeric_parameter("base_y")
        base_z = numeric_parameter("base_z")
        base_bounds = (
            base_x - 0.5 * base_length,
            base_x + 0.5 * base_length,
            base_y - 0.5 * base_width,
            base_y + 0.5 * base_width,
        )
        base_z_bounds = (
            wheel_radius + base_z - 0.5 * base_height,
            wheel_radius + base_z + 0.5 * base_height,
        )
        wheel_half_width = 0.5 * wheel_width
        cls.robot_collision_shapes = [
            ("base", base_bounds, base_z_bounds),
            (
                "left_wheel",
                (
                    wheel_x - wheel_radius,
                    wheel_x + wheel_radius,
                    0.5 * wheel_separation - wheel_half_width,
                    0.5 * wheel_separation + wheel_half_width,
                ),
                (0.0, 2.0 * wheel_radius),
            ),
            (
                "right_wheel",
                (
                    wheel_x - wheel_radius,
                    wheel_x + wheel_radius,
                    -0.5 * wheel_separation - wheel_half_width,
                    -0.5 * wheel_separation + wheel_half_width,
                ),
                (0.0, 2.0 * wheel_radius),
            ),
        ]

        # The pitched D405 collision extends a fraction of a millimetre past
        # the base box, so include its exact horizontal projection as well.
        pitch_expression = parameter_values["camera_pitch"]
        expected_suffix = "*pi/180.0}"
        if not (
            pitch_expression.startswith("${")
            and pitch_expression.endswith(expected_suffix)
        ):
            raise AssertionError("camera pitch expression changed")
        pitch_degrees = float(
            pitch_expression[2 : -len(expected_suffix)]
        )
        camera_pitch = math.radians(pitch_degrees)
        camera_mount_x = numeric_parameter("camera_mount_x")
        camera_mount_y = numeric_parameter("camera_mount_y")
        camera_mount_z = numeric_parameter("camera_mount_z")
        camera_body_x = numeric_parameter("camera_body_x")
        camera_body_y = numeric_parameter("camera_body_y")
        camera_body_z = numeric_parameter("camera_body_z")
        camera_center_x = camera_mount_x + (
            0.5 * camera_body_x * math.cos(camera_pitch)
        )
        camera_center_y = camera_mount_y
        camera_center_z = wheel_radius + camera_mount_z - (
            0.5 * camera_body_x * math.sin(camera_pitch)
        )
        camera_half_x = (
            0.5 * camera_body_x * math.cos(camera_pitch)
            + 0.5 * camera_body_z * math.sin(camera_pitch)
        )
        camera_half_z = (
            0.5 * camera_body_x * math.sin(camera_pitch)
            + 0.5 * camera_body_z * math.cos(camera_pitch)
        )
        cls.robot_collision_shapes.append(
            (
                "camera",
                (
                    camera_center_x - camera_half_x,
                    camera_center_x + camera_half_x,
                    camera_center_y - 0.5 * camera_body_y,
                    camera_center_y + 0.5 * camera_body_y,
                ),
                (
                    camera_center_z - camera_half_z,
                    camera_center_z + camera_half_z,
                ),
            )
        )

        # These projected collisions are enclosed by the base box and cannot
        # reduce the union-to-obstacle distance.
        for center_x, center_y, radius in (
            (
                numeric_parameter("caster_x"),
                numeric_parameter("caster_y"),
                numeric_parameter("caster_radius"),
            ),
            (
                numeric_parameter("lidar_x"),
                numeric_parameter("lidar_y"),
                numeric_parameter("lidar_radius"),
            ),
        ):
            if not (
                base_bounds[0] <= center_x - radius
                and center_x + radius <= base_bounds[1]
                and base_bounds[2] <= center_y - radius
                and center_y + radius <= base_bounds[3]
            ):
                raise AssertionError(
                    "contained custom collision now extends past base"
                )

        model = ElementTree.parse(str(BURGER_MODEL)).getroot()
        base = model.find(".//link[@name='base']/collision[@name='base_collision']")
        base_pose = [float(value) for value in base.findtext("pose").split()]
        base_size = [
            float(value) for value in base.findtext("geometry/box/size").split()
        ]
        cls.burger_shapes = [
            (
                base_pose[0] - 0.5 * base_size[0],
                base_pose[0] + 0.5 * base_size[0],
                base_pose[1] - 0.5 * base_size[1],
                base_pose[1] + 0.5 * base_size[1],
            )
        ]
        cls.burger_collision_shapes = [
            (
                "base",
                cls.burger_shapes[0],
                (
                    base_pose[2] - 0.5 * base_size[2],
                    base_pose[2] + 0.5 * base_size[2],
                ),
            )
        ]
        for link_name in ("left_wheel", "right_wheel"):
            wheel = model.find(".//link[@name='%s']/collision" % link_name)
            wheel_pose = [float(value) for value in wheel.findtext("pose").split()]
            radius = float(wheel.findtext("geometry/cylinder/radius"))
            length = float(wheel.findtext("geometry/cylinder/length"))
            if not math.isclose(abs(wheel_pose[3]), 0.5 * math.pi, abs_tol=0.01):
                raise AssertionError("Burger wheel cylinder is no longer horizontal")
            cls.burger_shapes.append(
                (
                    wheel_pose[0] - radius,
                    wheel_pose[0] + radius,
                    wheel_pose[1] - 0.5 * length,
                    wheel_pose[1] + 0.5 * length,
                )
            )
            cls.burger_collision_shapes.append(
                (
                    link_name,
                    cls.burger_shapes[-1],
                    (
                        wheel_pose[2] - radius,
                        wheel_pose[2] + radius,
                    ),
                )
            )

        # The remaining projected collisions are contained by the base box.
        base_bounds = cls.burger_shapes[0]
        for link_name, collision_name, radius_path in (
            ("lidar", "lidar_sensor_collision", "geometry/cylinder/radius"),
            ("base", "caster_collision", "geometry/sphere/radius"),
        ):
            collision = model.find(
                ".//link[@name='%s']/collision[@name='%s']"
                % (link_name, collision_name)
            )
            pose = [float(value) for value in collision.findtext("pose").split()]
            radius = float(collision.findtext(radius_path))
            if not (
                base_bounds[0] <= pose[0] - radius
                and pose[0] + radius <= base_bounds[1]
                and base_bounds[2] <= pose[1] - radius
                and pose[1] + radius <= base_bounds[3]
            ):
                raise AssertionError(
                    "%s is no longer contained by the Burger base collision"
                    % collision_name
                )

    def robot_polygon(self, pose, expansion=0.0):
        expansion = float(expansion)
        return transformed_box(
            pose,
            -(self.rear + expansion),
            self.front + expansion,
            -(self.half_width + expansion),
            self.half_width + expansion,
        )

    @classmethod
    def parked_burger_polygons(cls, center_x):
        spawn = (float(center_x), 0.8, 0.5 * math.pi)
        return [transformed_box(spawn, *shape) for shape in cls.burger_shapes]

    @classmethod
    def parked_burger_collision_polygons(cls, center_x):
        spawn = (float(center_x), 0.8, 0.5 * math.pi)
        return [
            (name, transformed_box(spawn, *bounds), z_bounds)
            for name, bounds, z_bounds in cls.burger_collision_shapes
        ]

    def minimum_gazebo_collision_clearance(self, poses, parked_shapes):
        """Return raw XY clearance between vertically overlapping collisions."""
        minimum = math.inf
        for pose in poses:
            for _robot_name, bounds, robot_z in self.robot_collision_shapes:
                robot = transformed_box(pose, *bounds)
                for _parked_name, parked, parked_z in parked_shapes:
                    if min(robot_z[1], parked_z[1]) < max(
                        robot_z[0], parked_z[0]
                    ):
                        continue
                    minimum = min(
                        minimum, convex_polygon_distance(robot, parked)
                    )
        return minimum

    def minimum_clearance(self, poses, forbidden_polygons, expansion=0.0):
        minimum = math.inf
        forbidden_bounds = [
            (
                float(np.min(polygon[:, 0])),
                float(np.max(polygon[:, 0])),
                float(np.min(polygon[:, 1])),
                float(np.max(polygon[:, 1])),
            )
            for polygon in forbidden_polygons
        ]
        for pose in poses:
            robot = self.robot_polygon(pose, expansion=expansion)
            robot_minimum_x = float(np.min(robot[:, 0]))
            robot_maximum_x = float(np.max(robot[:, 0]))
            robot_minimum_y = float(np.min(robot[:, 1]))
            robot_maximum_y = float(np.max(robot[:, 1]))
            for polygon, bounds in zip(forbidden_polygons, forbidden_bounds):
                minimum_x, maximum_x, minimum_y, maximum_y = bounds
                separation_x = max(
                    0.0,
                    minimum_x - robot_maximum_x,
                    robot_minimum_x - maximum_x,
                )
                separation_y = max(
                    0.0,
                    minimum_y - robot_maximum_y,
                    robot_minimum_y - maximum_y,
                )
                if math.hypot(separation_x, separation_y) >= minimum:
                    continue
                minimum = min(minimum, convex_polygon_distance(robot, polygon))
        return minimum

    def parking_paths(self, label):
        aisle_x = float(self.route["aisle_x"])
        decision_y = float(self.route["decision_y"])
        zigzag_turn_start_x = float(self.route["zigzag_turn_start_x"])
        zigzag_straight_y = float(self.route["zigzag_straight_y"])
        turn_curve_offset = float(self.route["turn_curve_offset"])
        turn_curve_tangent = float(self.route["turn_curve_tangent"])
        zigzag_alignment_tail = float(
            self.route["zigzag_alignment_tail"]
        )
        zigzag_approach_tangent_ratio = float(
            self.route["zigzag_approach_tangent_ratio"]
        )
        aisle_yaw = math.radians(float(self.route["aisle_heading_deg"]))
        outgoing_yaw = math.radians(
            float(self.route["outgoing_heading_deg"])
        )
        zigzag_yaw = math.radians(
            float(self.route["zigzag_heading_deg"])
        )
        decision = (aisle_x, decision_y, aisle_yaw)
        if label == "LEFT":
            park_x = float(self.route["left_park_x"])
            park_yaw = 0.0
            parked_x = 0.23
        else:
            park_x = float(self.route["right_park_x"])
            park_yaw = math.pi
            parked_x = 0.73

        turn_space_pose = (aisle_x, decision_y, park_yaw)
        park_pose = (park_x, decision_y, park_yaw)
        turn_exit_pose = (aisle_x, decision_y, outgoing_yaw)
        exit_pose = (
            zigzag_turn_start_x,
            zigzag_straight_y - turn_curve_offset,
            outgoing_yaw,
        )

        turn_to_space = sample_in_place_rotation(decision, park_yaw)
        park_in = sample_straight(turn_space_pose, park_pose)
        back_out = sample_straight(park_pose, turn_space_pose)
        turn_to_exit = sample_in_place_rotation(
            turn_space_pose, outgoing_yaw
        )
        approach_distance = math.hypot(
            exit_pose[0] - turn_exit_pose[0],
            exit_pose[1] - turn_exit_pose[1],
        )
        approach_tangent = (
            zigzag_approach_tangent_ratio * approach_distance
        )
        leave_aisle = quintic_pose_path(
            turn_exit_pose,
            exit_pose,
            approach_tangent,
            approach_tangent,
            SAMPLE_COUNT,
        )
        turn_to_zigzag = quintic_turn_path(
            exit_pose,
            zigzag_yaw,
            turn_curve_offset,
            turn_curve_tangent,
            SAMPLE_COUNT,
        )
        curve_end = tuple(turn_to_zigzag[-1])
        tail_goal = (
            curve_end[0] + zigzag_alignment_tail * math.cos(zigzag_yaw),
            curve_end[1] + zigzag_alignment_tail * math.sin(zigzag_yaw),
            zigzag_yaw,
        )
        alignment_tail = sample_straight(curve_end, tail_goal)
        turn_to_zigzag = np.vstack(
            (turn_to_zigzag, alignment_tail[1:])
        )
        phases = (
            ("TURN_TO_SPACE", turn_to_space),
            ("PARK_IN", park_in),
            ("BACK_OUT", back_out),
            ("TURN_TO_EXIT", turn_to_exit),
            ("LEAVE_AISLE", leave_aisle),
            ("TURN_TO_ZIGZAG", turn_to_zigzag),
        )
        return phases, self.parked_burger_polygons(parked_x)

    def assert_clear(self, poses, forbidden, label):
        clearance = self.minimum_clearance(poses, forbidden)
        self.assertGreaterEqual(
            clearance,
            REGRESSION_CLEARANCE,
            "%s raw clearance %.3f mm is below %.1f mm"
            % (label, 1000.0 * clearance, 1000.0 * REGRESSION_CLEARANCE),
        )

    def assert_clearance_at_least(
        self, poses, forbidden, minimum, label, expansion=0.0
    ):
        clearance = self.minimum_clearance(
            poses, forbidden, expansion=expansion
        )
        self.assertGreaterEqual(
            clearance,
            minimum,
            "%s raw clearance %.3f mm is below %.1f mm"
            % (label, 1000.0 * clearance, 1000.0 * minimum),
        )

    def test_both_parking_branches_clear_solid_paint_and_parked_burger(self):
        for label in ("LEFT", "RIGHT"):
            phases, burger = self.parking_paths(label)
            parked_collision = self.parked_burger_collision_polygons(
                0.23 if label == "LEFT" else 0.73
            )
            for phase, poses in phases:
                with self.subTest(branch=label, phase=phase, forbidden="paint"):
                    self.assert_clear(
                        poses,
                        self.parking_route_solid,
                        "%s %s paint" % (label, phase),
                    )
                with self.subTest(branch=label, phase=phase, forbidden="burger"):
                    if phase in ("TURN_TO_SPACE", "TURN_TO_EXIT"):
                        clearance = self.minimum_gazebo_collision_clearance(
                            poses[::10], parked_collision
                        )
                        self.assertGreaterEqual(
                            clearance,
                            REGRESSION_CLEARANCE,
                            "%s %s actual Gazebo collision clearance %.3f mm"
                            % (label, phase, 1000.0 * clearance),
                        )
                    else:
                        self.assert_clear(
                            poses, burger, "%s %s Burger" % (label, phase)
                        )

    def test_both_parking_branches_clear_with_expanded_footprint(self):
        expansion = self.expanded_footprint_margin
        self.assertAlmostEqual(expansion, 0.011, places=12)
        for label in ("LEFT", "RIGHT"):
            phases, burger = self.parking_paths(label)
            for phase, poses in phases:
                for forbidden_name, forbidden in (
                    ("paint", self.parking_route_solid),
                    ("Burger", burger),
                ):
                    if (
                        forbidden_name == "Burger"
                        and phase in ("TURN_TO_SPACE", "TURN_TO_EXIT")
                    ):
                        # Rotation obstacle policy uses actual contact with no
                        # virtual margin; line/map margins remain expanded.
                        continue
                    with self.subTest(
                        branch=label,
                        phase=phase,
                        forbidden=forbidden_name,
                    ):
                        clearance = self.minimum_clearance(
                            poses,
                            forbidden,
                            expansion=expansion,
                        )
                        self.assertGreater(
                            clearance,
                            0.0,
                            "%s %s expanded footprint intersects %s"
                            % (label, phase, forbidden_name),
                        )

    def test_single_central_aisle_drive_replaces_position_connector(self):
        aisle_x = float(self.route["aisle_x"])
        decision_y = float(self.route["decision_y"])
        aisle_yaw = math.radians(float(self.route["aisle_heading_deg"]))
        expansion = self.expanded_footprint_margin

        self.assertAlmostEqual(aisle_x, 0.5000, places=12)
        self.assertAlmostEqual(decision_y, 0.6920, places=12)
        self.assertAlmostEqual(expansion, 0.011, places=12)
        self.assertAlmostEqual(
            float(self.config["safety"]["rotation_obstacle_margin"]),
            0.0,
            places=12,
        )
        for removed_name in (
            "parking_turn_y",
            "parking_turn_lateral_offset",
            "parking_connector_tangent",
            "parking_exit_connector_tangent",
            "parking_connector_samples",
        ):
            self.assertNotIn(removed_name, self.route)

        curve_end_y = float(self.route["entry_y"]) - float(
            self.route["turn_curve_offset"]
        )
        aisle_drive = sample_straight(
            (aisle_x, curve_end_y, aisle_yaw),
            (aisle_x, decision_y, aisle_yaw),
        )
        np.testing.assert_allclose(aisle_drive[:, 0], aisle_x, atol=1e-12)
        self.assertGreater(
            self.minimum_clearance(
                aisle_drive,
                self.parking_route_solid,
                expansion=expansion,
            ),
            0.0,
        )
        for parked_x in (0.23, 0.73):
            with self.subTest(parked_x=parked_x):
                self.assertGreater(
                    self.minimum_clearance(
                        aisle_drive,
                        self.parked_burger_polygons(parked_x),
                        expansion=expansion,
                    ),
                    0.0,
                )

        # The nominal direct spins retain at least 25 mm from the exact
        # Gazebo collision union in the tighter (RIGHT) obstacle layout.
        for label, target_yaw, parked_x, minimum_clearance in (
            ("LEFT", 0.0, 0.23, 0.060),
            ("RIGHT", math.pi, 0.73, 0.025),
        ):
            poses = sample_rotation(
                (aisle_x, decision_y, aisle_yaw), target_yaw, 101
            )
            clearance = self.minimum_gazebo_collision_clearance(
                poses, self.parked_burger_collision_polygons(parked_x)
            )
            with self.subTest(branch=label):
                self.assertGreaterEqual(clearance, minimum_clearance)

    def test_official_anchor_bias_clears_low_parked_wheel(self):
        """Regress the low-wheel contact observed only from the full start."""
        aisle_x = float(self.route["aisle_x"])
        decision_y = float(self.route["decision_y"])

        # In official RIGHT run Cf3tSn the frozen AMCL-to-odom route anchor and
        # stopped-pose error placed the physical pivot this far east/north of
        # the configured map target.  The parked Burger wheel is below the
        # Gazebo 2-D scan plane, so exact collision geometry owns this case.
        observed_physics_offset = (0.0142, 0.0104)
        observed_heading = math.radians(-90.5)
        poses = sample_rotation(
            (
                aisle_x + observed_physics_offset[0],
                decision_y + observed_physics_offset[1],
                observed_heading,
            ),
            math.pi,
            SAMPLE_COUNT,
        )
        clearance = self.minimum_gazebo_collision_clearance(
            poses, self.parked_burger_collision_polygons(0.73)
        )
        self.assertGreaterEqual(clearance, 0.008)

    def test_rotation_center_error_corners_clear_course_and_burger(self):
        aisle_x = float(self.route["aisle_x"])
        decision_y = float(self.route["decision_y"])
        aisle_yaw = math.radians(float(self.route["aisle_heading_deg"]))
        outgoing_yaw = math.radians(
            float(self.route["outgoing_heading_deg"])
        )
        expansion = self.expanded_footprint_margin

        position_errors = (-0.008, 0.008)
        heading_errors_deg = (-1.0, 1.0)
        for label, park_yaw, parked_x in (
            ("LEFT", 0.0, 0.23),
            ("RIGHT", math.pi, 0.73),
        ):
            parked_collision = self.parked_burger_collision_polygons(
                parked_x
            )
            for phase, start_yaw, target_yaw in (
                ("TURN_TO_SPACE", aisle_yaw, park_yaw),
                ("TURN_TO_EXIT", park_yaw, outgoing_yaw),
            ):
                for x_error in position_errors:
                    for y_error in position_errors:
                        for heading_error_deg in heading_errors_deg:
                            start = (
                                aisle_x + x_error,
                                decision_y + y_error,
                                start_yaw
                                + math.radians(heading_error_deg),
                            )
                            poses = sample_rotation(start, target_yaw, 101)
                            with self.subTest(
                                branch=label,
                                phase=phase,
                                x_error_mm=1000.0 * x_error,
                                y_error_mm=1000.0 * y_error,
                                heading_error_deg=heading_error_deg,
                            ):
                                paint_clearance = self.minimum_clearance(
                                    poses,
                                    self.parking_route_solid,
                                    expansion=expansion,
                                )
                                self.assertGreater(paint_clearance, 0.0)
                                collision_clearance = (
                                    self.minimum_gazebo_collision_clearance(
                                        poses, parked_collision
                                    )
                                )
                                self.assertGreater(collision_clearance, 0.0)

    def test_direct_exit_approach_error_envelope_clears_course_and_burger(self):
        aisle_x = float(self.route["aisle_x"])
        decision_y = float(self.route["decision_y"])
        outgoing_yaw = math.radians(
            float(self.route["outgoing_heading_deg"])
        )
        expansion = self.expanded_footprint_margin

        goal = (
            float(self.route["zigzag_turn_start_x"]),
            float(self.route["zigzag_straight_y"])
            - float(self.route["turn_curve_offset"]),
            outgoing_yaw,
        )
        tangent_ratio = float(
            self.route["zigzag_approach_tangent_ratio"]
        )
        error_positions = (-0.008, 0.0, 0.008)
        heading_errors_deg = (-1.0, 0.0, 1.0)
        for label, parked_x in (("LEFT", 0.23), ("RIGHT", 0.73)):
            parked_burger = self.parked_burger_polygons(parked_x)
            for x_error in error_positions:
                for y_error in error_positions:
                    for heading_error_deg in heading_errors_deg:
                        start = (
                            aisle_x + x_error,
                            decision_y + y_error,
                            outgoing_yaw
                            + math.radians(heading_error_deg),
                        )
                        distance = math.hypot(
                            goal[0] - start[0],
                            goal[1] - start[1],
                        )
                        tangent = tangent_ratio * max(0.01, distance)
                        approach = quintic_pose_path(
                            start,
                            goal,
                            tangent,
                            tangent,
                            SAMPLE_COUNT,
                        )
                        with self.subTest(
                            branch=label,
                            x_error_mm=1000.0 * x_error,
                            y_error_mm=1000.0 * y_error,
                            heading_error_deg=heading_error_deg,
                        ):
                            course_clearance = self.minimum_clearance(
                                approach,
                                self.parking_route_solid,
                                expansion=expansion,
                            )
                            self.assertGreater(course_clearance, 0.0)
                            burger_clearance = self.minimum_clearance(
                                approach,
                                parked_burger,
                                expansion=expansion,
                            )
                            self.assertGreater(burger_clearance, 0.0)

    def test_run11_exit_pose_clears_native_solid_paint_with_full_margin(self):
        start = (
            0.49203417869850774,
            0.6938828136475323,
            math.radians(90.1776092917495),
        )
        goal = (
            float(self.route["zigzag_turn_start_x"]),
            float(self.route["zigzag_straight_y"])
            - float(self.route["turn_curve_offset"]),
            math.radians(float(self.route["outgoing_heading_deg"])),
        )
        distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
        tangent = (
            float(self.route["zigzag_approach_tangent_ratio"])
            * max(0.01, distance)
        )
        approach = quintic_pose_path(
            start, goal, tangent, tangent, SAMPLE_COUNT
        )

        clearance = self.minimum_clearance(
            approach,
            self.parking_route_solid,
            expansion=self.expanded_footprint_margin,
        )
        # The west-shifted shared-turn placement keeps this measured biased
        # approach above 7 mm after the full 11 mm footprint expansion.
        self.assertGreater(clearance, 0.007)

        paint = self.config["paint"]
        # Rounded inward from the native solid pixels, never outward into a
        # painted cell. These three limits feed the runtime finite union.
        self.assertGreaterEqual(
            float(paint["parking_opening_right_edge"]), 0.3830222439
        )
        self.assertLessEqual(
            float(paint["parking_opening_left_edge"]), 0.6129434796
        )
        self.assertLessEqual(
            float(paint["parking_opening_upper_edge"]), 1.0001824994
        )

    def test_runtime_fixed_obstacles_match_gazebo_sign_collisions(self):
        self.assertEqual(
            len(self.configured_entry_signs), len(self.entry_signs)
        )
        for configured, measured in zip(
            self.configured_entry_signs, self.entry_signs
        ):
            configured = np.asarray(
                sorted(map(tuple, configured)), dtype=np.float64
            )
            measured = np.asarray(
                sorted(map(tuple, measured)), dtype=np.float64
            )
            np.testing.assert_allclose(configured, measured, atol=1e-12)

    def test_lane_handoff_speed_does_not_exceed_entry_connector_speed(self):
        control = self.config["control"]
        self.assertLessEqual(
            float(control["lane_entry_speed_limit"]),
            float(control["approach_velocity"]),
        )

    def test_motion_speed_targets_do_not_exceed_common_cruise_cap(self):
        control = self.config["control"]
        cruise = float(control["cruise_velocity"])
        self.assertAlmostEqual(cruise, 0.20, places=12)
        for name in (
            "aisle_velocity",
            "parking_velocity",
            "reverse_velocity",
        ):
            with self.subTest(straight_target=name):
                self.assertAlmostEqual(
                    float(control[name]), cruise, places=12
                )
        for name in (
            "approach_velocity",
            "aisle_velocity",
            "parking_velocity",
            "reverse_velocity",
            "entry_turn_velocity",
            "rejoin_velocity",
        ):
            with self.subTest(name=name):
                self.assertLessEqual(float(control[name]), cruise)

    def test_continuous_entry_left_curve_clears_paint_and_both_signs(self):
        aisle_x = float(self.route["aisle_x"])
        entry_y = float(self.route["entry_y"])
        offset = float(self.route["turn_curve_offset"])
        tangent = float(self.route["turn_curve_tangent"])
        start = (aisle_x + offset, entry_y, math.pi)
        approach = sample_straight((1.03, entry_y, math.pi), start)
        turn = quintic_turn_path(
            start,
            -0.5 * math.pi,
            offset,
            tangent,
            SAMPLE_COUNT,
        )

        self.assert_clear(approach, self.entry_solid, "entry straight paint")
        self.assert_clear(turn, self.entry_solid, "entry left curve paint")
        self.assert_clear(turn, self.entry_signs, "entry left curve signs")

    def test_observed_adaptive_connector_clears_entry_paint_and_signs(self):
        end = (
            float(self.route["aisle_x"])
            + float(self.route["turn_curve_offset"]),
            float(self.route["entry_y"]),
            math.pi,
        )
        approach_speed = float(self.config["control"]["approach_velocity"])
        angular_scale = float(self.config["control"]["arc_angular_scale"])
        maximum_angular = float(
            self.config["control"]["maximum_angular_velocity"]
        )
        maximum_angular_acceleration = float(
            self.config["control"]["angular_acceleration"]
        )
        tangent_ratio = float(
            self.config["control"]["entry_connector_tangent_ratio"]
        )

        # Both starts are expressed directly in parking_local. A translated or
        # rotated connecting straight changes local->odom, not this geometry;
        # no AMCL clamp, positioning state, or stopped correction is inserted.
        observed_starts = (
            ("previous", 1.0082, 1.7617, math.pi),
            (
                "north_anchor_corrected",
                0.969312,
                1.757642,
                math.radians(-177.2),
            ),
        )
        for start_label, start_x, start_y, start_yaw in observed_starts:
            for heading_offset in (-1.6, 0.0, 1.6):
                with self.subTest(
                    start=start_label, heading_offset=heading_offset
                ):
                    start = (
                        start_x,
                        start_y,
                        start_yaw + math.radians(heading_offset),
                    )
                    forward_distance = (
                        math.cos(end[2]) * (end[0] - start[0])
                        + math.sin(end[2]) * (end[1] - start[1])
                    )
                    connector = quintic_pose_path(
                        start,
                        end,
                        tangent_ratio * forward_distance,
                        tangent_ratio * forward_distance,
                        SAMPLE_COUNT,
                    )
                    maximum_curvature, maximum_rate = path_curvature_limits(
                        connector
                    )

                    self.assert_clearance_at_least(
                        connector,
                        self.entry_solid,
                        self.line_margin,
                        "adaptive entry connector paint",
                    )
                    self.assert_clear(
                        connector,
                        self.entry_signs,
                        "adaptive entry connector signs",
                    )
                    self.assertLessEqual(
                        angular_scale * approach_speed * maximum_curvature,
                        maximum_angular,
                    )
                    self.assertLessEqual(
                        angular_scale * approach_speed ** 2 * maximum_rate,
                        maximum_angular_acceleration,
                    )

    def test_smooth_zigzag_exit_has_exact_joins_clearance_and_alignment(self):
        offset = float(self.route["turn_curve_offset"])
        tangent = float(self.route["turn_curve_tangent"])
        tail_length = float(self.route["zigzag_alignment_tail"])
        zigzag_y = float(self.route["zigzag_straight_y"])
        start = (
            float(self.route["zigzag_turn_start_x"]),
            zigzag_y - offset,
            0.5 * math.pi,
        )
        curve = quintic_turn_path(
            start,
            math.pi,
            offset,
            tangent,
            SAMPLE_COUNT,
        )
        expected_curve_end = (
            start[0] - offset,
            zigzag_y,
            math.pi,
        )
        np.testing.assert_allclose(
            curve[-1, :2], expected_curve_end[:2], atol=1e-12
        )
        self.assertAlmostEqual(
            math.sin(curve[-1, 2]), math.sin(expected_curve_end[2]), places=12
        )
        self.assertAlmostEqual(
            math.cos(curve[-1, 2]), math.cos(expected_curve_end[2]), places=12
        )
        tail_goal = (
            expected_curve_end[0] - tail_length,
            zigzag_y,
            math.pi,
        )
        tail = sample_straight(expected_curve_end, tail_goal)
        poses = np.vstack((curve, tail[1:]))

        self.assertTrue(
            np.all(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1) > 0.0)
        )
        self.assertTrue(np.all(np.diff(np.unwrap(curve[:, 2])) >= 0.0))
        np.testing.assert_allclose(poses[-1], tail_goal, atol=1e-12)
        np.testing.assert_allclose(tail[:, 1], zigzag_y, atol=1e-12)
        np.testing.assert_allclose(tail[:, 2], math.pi, atol=1e-12)

        self.assert_clear(poses, self.final_solid, "smooth zigzag exit paint")
        self.assert_clear(poses, self.entry_signs, "smooth zigzag exit signs")
        expansion = self.expanded_footprint_margin
        self.assertGreaterEqual(
            self.minimum_clearance(
                poses, self.final_solid, expansion=expansion
            ),
            0.003,
        )
        self.assertGreater(
            self.minimum_clearance(
                poses, self.entry_signs, expansion=expansion
            ),
            0.0,
        )

        maximum_curvature, maximum_rate = path_curvature_limits(curve)
        self.assertLess(maximum_curvature, 11.4)
        self.assertLess(maximum_rate, 116.0)

        rejoin = self.config["rejoin"]
        self.assertTrue(
            float(rejoin["handoff_min_x"])
            <= tail_goal[0]
            <= float(rejoin["handoff_max_x"])
        )
        self.assertTrue(
            float(rejoin["handoff_min_y"])
            <= tail_goal[1]
            <= float(rejoin["handoff_max_y"])
        )


if __name__ == "__main__":
    unittest.main()
