"""ROS-independent polygon geometry and ordered mission-zone state machine."""

import math


def point_in_polygon(point, polygon):
    """Return True for points inside or on the boundary of a simple polygon."""
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        ax, ay = previous
        bx, by = current
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) <= 1e-9 and (
            min(ax, bx) - 1e-9 <= x <= max(ax, bx) + 1e-9
            and min(ay, by) - 1e-9 <= y <= max(ay, by) + 1e-9
        ):
            return True
        if (ay > y) != (by > y):
            intersection_x = ax + (y - ay) * (bx - ax) / (by - ay)
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside


def distance_to_polygon_boundary(point, polygon):
    """Return the shortest Euclidean distance from a point to polygon edges."""
    x, y = point
    shortest = float("inf")
    previous = polygon[-1]
    for current in polygon:
        ax, ay = previous
        bx, by = current
        dx, dy = bx - ax, by - ay
        length_squared = dx * dx + dy * dy
        if length_squared <= 1e-12:
            projected_x, projected_y = ax, ay
        else:
            ratio = max(
                0.0,
                min(1.0, ((x - ax) * dx + (y - ay) * dy) / length_squared),
            )
            projected_x = ax + ratio * dx
            projected_y = ay + ratio * dy
        shortest = min(shortest, math.hypot(x - projected_x, y - projected_y))
        previous = current
    return shortest


def signed_polygon_distance(point, polygon):
    """Return positive distance inside and negative distance outside polygon."""
    distance = distance_to_polygon_boundary(point, polygon)
    return distance if point_in_polygon(point, polygon) else -distance


def signed_polygon_distances(point, named_polygons):
    """Evaluate each ``(key, polygon)`` entry exactly once at ``point``."""
    return {
        key: signed_polygon_distance(point, polygon)
        for key, polygon in named_polygons
    }


def is_inside_with_margin(signed_distance, inside_margin=0.0):
    """Return whether a signed polygon distance clears an inside margin."""
    return signed_distance >= max(0.0, float(inside_margin))


class MissionZoneSequence:
    """Only activate the expected polygon in a configured mission sequence."""

    WAITING_FOR_POSE = "WAITING_FOR_POSE"
    SEEKING = "SEEKING"
    ACTIVE = "ACTIVE"
    COMPLETE = "COMPLETE"

    def __init__(
        self,
        missions,
        enter_margin=0.0,
    ):
        if not missions:
            raise ValueError("mission sequence must not be empty")
        self.missions = list(missions)
        self.enter_margin = max(0.0, float(enter_margin))
        self.reset()

    def reset(self):
        self.index = 0
        self.state = self.WAITING_FOR_POSE
        self.completed = []

    @property
    def current(self):
        if self.index >= len(self.missions):
            return None
        return self.missions[self.index]

    @property
    def current_name(self):
        return "" if self.current is None else self.current["name"]

    def update_signed_distance(self, signed_distance):
        """Consume a precomputed current-zone distance.

        The ROS owner computes every polygon distance once so it can publish
        physical-clearance topics without repeating the geometry here.
        """
        if self.state == self.WAITING_FOR_POSE:
            self.state = self.SEEKING
            changed = True
        else:
            changed = False
        mission = self.current
        if mission is None:
            return changed
        if signed_distance is None:
            raise ValueError("active mission needs a signed polygon distance")
        if self.state == self.SEEKING and signed_distance >= self.enter_margin:
            self.state = self.ACTIVE
            changed = True
        return changed

    def complete_current(self, mission_name):
        """Advance an ACTIVE mission after an external controller completes it."""
        mission = self.current
        if (
            mission is None
            or self.state != self.ACTIVE
            or mission["name"] != mission_name
        ):
            return False
        self._advance()
        return True

    def _advance(self):
        self.completed.append(self.current_name)
        self.index += 1
        self.state = self.COMPLETE if self.current is None else self.SEEKING
