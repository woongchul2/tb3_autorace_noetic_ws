"""Gazebo texture checks for a surveyed, arbitrarily curved lane route."""

import math

import numpy as np

from custom_autorace_bringup.zigzag_path import normalize_angle


class RasterRouteCorridorChecker:
    """Measure rectangular-footprint clearance from yellow/white paint.

    The configured path supplies the local tangent and selects the intended
    yellow stripe on its left and white stripe on its right.  The actual stripe
    edges still come from the finite pixels of the Gazebo course texture.
    """

    def __init__(
        self,
        rgb_image,
        course_size,
        course_yaw,
        path,
        front,
        rear,
        half_width,
        expected_boundary_offset=0.12,
        line_search_half_width=0.08,
        boundary_step=0.0005,
        footprint_sample_spacing=0.004,
        color_threshold=128,
        color_tolerance=4,
        yellow_blue_maximum=16,
        projection_search_distance=0.16,
        endpoint_extension=0.14,
    ):
        image = np.asarray(rgb_image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("course texture must be an RGB image")
        image = image[:, :, :3].astype(np.int16, copy=False)
        if path.x.size < 2 or not (
            path.x.size
            == path.y.size
            == path.heading.size
            == path.station.size
        ):
            raise ValueError("route corridor needs a non-empty sampled path")

        self.path = path
        self.image_height, self.image_width = image.shape[:2]
        self.course_size = max(0.1, float(course_size))
        self.course_yaw = float(course_yaw)
        self.course_cosine = math.cos(self.course_yaw)
        self.course_sine = math.sin(self.course_yaw)
        self.projection_search_distance = max(
            float(projection_search_distance), float(front), float(rear)
        )
        self.endpoint_extension = max(0.0, float(endpoint_extension))

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

        expected = max(0.02, float(expected_boundary_offset))
        search_half_width = max(0.02, float(line_search_half_width))
        boundary_step = max(0.00025, float(boundary_step))
        pixel_diagonal = math.hypot(
            self.course_size / self.image_width,
            self.course_size / self.image_height,
        )
        maximum_gap_samples = int(
            math.floor(pixel_diagonal / boundary_step)
        ) + 1

        left_inner = []
        left_outer = []
        right_inner = []
        right_outer = []
        for index, (x, y, heading) in enumerate(
            zip(path.x, path.y, path.heading)
        ):
            try:
                left_band = self._stripe_band_at_pose(
                    self.yellow_mask,
                    float(x),
                    float(y),
                    float(heading),
                    expected,
                    search_half_width,
                    boundary_step,
                    maximum_gap_samples,
                )
                right_band = self._stripe_band_at_pose(
                    self.white_mask,
                    float(x),
                    float(y),
                    float(heading),
                    -expected,
                    search_half_width,
                    boundary_step,
                    maximum_gap_samples,
                )
            except ValueError as error:
                raise ValueError(
                    "%s at route index %d pose (%.4f, %.4f)"
                    % (error, index, x, y)
                )
            left_inner.append(left_band[0])
            left_outer.append(left_band[1])
            right_outer.append(right_band[0])
            right_inner.append(right_band[1])

        self.left_inner = np.asarray(left_inner, dtype=np.float64)
        self.left_outer = np.asarray(left_outer, dtype=np.float64)
        self.right_inner = np.asarray(right_inner, dtype=np.float64)
        self.right_outer = np.asarray(right_outer, dtype=np.float64)
        if np.any(self.right_inner >= self.left_inner):
            raise ValueError("surveyed route paint edges cross")

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

    def _stripe_band_at_pose(
        self,
        mask,
        x,
        y,
        heading,
        expected_offset,
        search_half_width,
        boundary_step,
        maximum_gap_samples,
    ):
        count = int(math.ceil(2.0 * search_half_width / boundary_step)) + 1
        offsets = np.linspace(
            expected_offset - search_half_width,
            expected_offset + search_half_width,
            count,
        )
        normal_x = -math.sin(heading)
        normal_y = math.cos(heading)
        world_x = x + normal_x * offsets
        world_y = y + normal_y * offsets
        u, v = self._world_to_pixel(world_x, world_y)
        valid = (
            (u >= 0)
            & (u < self.image_width)
            & (v >= 0)
            & (v < self.image_height)
        )
        painted = np.zeros(offsets.size, dtype=bool)
        painted[valid] = mask[v[valid], u[valid]]
        indices = np.flatnonzero(painted)
        if indices.size == 0:
            color = "yellow" if expected_offset > 0.0 else "white"
            raise ValueError("surveyed route %s stripe is absent" % color)
        cuts = np.r_[
            0,
            np.flatnonzero(np.diff(indices) > maximum_gap_samples) + 1,
            indices.size,
        ]
        runs = [indices[a:b] for a, b in zip(cuts[:-1], cuts[1:])]
        selected = min(
            runs,
            key=lambda run: abs(
                0.5 * (offsets[run[0]] + offsets[run[-1]])
                - expected_offset
            ),
        )
        return (
            float(offsets[selected[0]] - 0.5 * boundary_step),
            float(offsets[selected[-1]] + 0.5 * boundary_step),
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

    def _nearest_indices(self, world_x, world_y, center_index):
        center_station = float(self.path.station[center_index])
        first = int(
            np.searchsorted(
                self.path.station,
                center_station - self.projection_search_distance,
                side="left",
            )
        )
        last = int(
            np.searchsorted(
                self.path.station,
                center_station + self.projection_search_distance,
                side="right",
            )
        )
        first = max(0, min(first, self.path.x.size - 1))
        last = max(first + 1, min(last, self.path.x.size))
        delta_x = world_x[:, None] - self.path.x[None, first:last]
        delta_y = world_y[:, None] - self.path.y[None, first:last]
        return first + np.argmin(delta_x * delta_x + delta_y * delta_y, axis=1)

    def covers_pose(self, x, y, heading):
        center_distance = np.hypot(self.path.x - x, self.path.y - y)
        center_index = int(np.argmin(center_distance))
        world_x, world_y = self._footprint_world(x, y, heading)
        indices = self._nearest_indices(world_x, world_y, center_index)
        return self._indices_within_endpoint_extension(
            world_x, world_y, indices
        )

    def _indices_within_endpoint_extension(self, world_x, world_y, indices):
        first = indices == 0
        if np.any(first):
            start_heading = float(self.path.heading[0])
            longitudinal = (
                math.cos(start_heading) * (world_x[first] - self.path.x[0])
                + math.sin(start_heading) * (world_y[first] - self.path.y[0])
            )
            if float(np.min(longitudinal)) < -self.endpoint_extension:
                return False
        last = indices == self.path.x.size - 1
        if np.any(last):
            end_heading = float(self.path.heading[-1])
            longitudinal = (
                math.cos(end_heading) * (world_x[last] - self.path.x[-1])
                + math.sin(end_heading) * (world_y[last] - self.path.y[-1])
            )
            if float(np.max(longitudinal)) > self.endpoint_extension:
                return False
        return True

    def pose_metrics(self, x, y, heading):
        """Return ``(inner_intrusion, outer_edge_reserve)`` in metres."""
        center_distance = np.hypot(self.path.x - x, self.path.y - y)
        center_index = int(np.argmin(center_distance))
        world_x, world_y = self._footprint_world(x, y, heading)
        indices = self._nearest_indices(world_x, world_y, center_index)
        if not self._indices_within_endpoint_extension(
            world_x, world_y, indices
        ):
            return math.inf, -math.inf

        path_x = self.path.x[indices]
        path_y = self.path.y[indices]
        heading_at_edge = self.path.heading[indices]
        lateral = (
            -np.sin(heading_at_edge) * (world_x - path_x)
            + np.cos(heading_at_edge) * (world_y - path_y)
        )
        inner_intrusion = max(
            0.0,
            float(np.max(lateral - self.left_inner[indices])),
            float(np.max(self.right_inner[indices] - lateral)),
        )
        outer_reserve = min(
            float(np.min(self.left_outer[indices] - lateral)),
            float(np.min(lateral - self.right_outer[indices])),
        )
        return inner_intrusion, outer_reserve

    def path_metrics(self, start_index=0, end_index=None):
        maximum_inner_intrusion = 0.0
        minimum_outer_reserve = math.inf
        start_index = max(0, int(start_index))
        if end_index is None:
            end_index = self.path.x.size
        end_index = min(
            self.path.x.size, max(start_index, int(end_index))
        )
        for x, y, heading in zip(
            self.path.x[start_index:end_index],
            self.path.y[start_index:end_index],
            self.path.heading[start_index:end_index],
        ):
            inner, outer = self.pose_metrics(x, y, heading)
            maximum_inner_intrusion = max(maximum_inner_intrusion, inner)
            minimum_outer_reserve = min(minimum_outer_reserve, outer)
        return maximum_inner_intrusion, minimum_outer_reserve

    def segment_metrics(
        self,
        start_pose,
        end_pose,
        translation_step=0.002,
        heading_step=math.radians(0.5),
    ):
        distance = math.hypot(
            float(end_pose[0]) - float(start_pose[0]),
            float(end_pose[1]) - float(start_pose[1]),
        )
        yaw_change = normalize_angle(float(end_pose[2]) - float(start_pose[2]))
        count = max(
            1,
            int(math.ceil(distance / max(0.001, float(translation_step)))),
            int(
                math.ceil(
                    abs(yaw_change)
                    / max(math.radians(0.1), float(heading_step))
                )
            ),
        )
        maximum_inner_intrusion = 0.0
        minimum_outer_reserve = math.inf
        for fraction in np.linspace(0.0, 1.0, count + 1):
            pose = (
                float(start_pose[0])
                + fraction * (float(end_pose[0]) - float(start_pose[0])),
                float(start_pose[1])
                + fraction * (float(end_pose[1]) - float(start_pose[1])),
                normalize_angle(float(start_pose[2]) + fraction * yaw_change),
            )
            inner, outer = self.pose_metrics(*pose)
            maximum_inner_intrusion = max(maximum_inner_intrusion, inner)
            minimum_outer_reserve = min(minimum_outer_reserve, outer)
        return maximum_inner_intrusion, minimum_outer_reserve
