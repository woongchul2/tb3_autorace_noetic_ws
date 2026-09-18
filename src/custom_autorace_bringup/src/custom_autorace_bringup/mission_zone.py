"""ROS-independent geometry and ordered mission-readiness state machine."""

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
    """Arm exactly one mission and activate it only from fresh readiness.

    A generation changes every time a mission is armed.  The controller must
    echo that generation with a source timestamp when it reports readiness.
    This prevents a latched message from a previous run or another mission
    from opening the command-ownership gate.
    """

    ARMED = "ARMED"
    READY = "READY"
    ACTIVE = "ACTIVE"
    COMPLETE = "COMPLETE"

    def __init__(
        self,
        missions,
        armed_at=0.0,
        initial_generation=1,
    ):
        if not missions:
            raise ValueError("mission sequence must not be empty")
        self.missions = list(missions)
        self.reset(armed_at, initial_generation)

    @staticmethod
    def _timestamp(value, name):
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("%s must be a finite non-negative timestamp" % name)
        return value

    @staticmethod
    def _generation(value):
        value = int(value)
        if value <= 0 or value > 0xFFFFFFFF:
            raise ValueError("generation must be within uint32 [1, 2^32-1]")
        return value

    @staticmethod
    def _next_generation(value):
        value = (int(value) + 1) & 0xFFFFFFFF
        return value if value != 0 else 1

    def reset(self, armed_at=0.0, initial_generation=1):
        self.index = 0
        self.state = self.ARMED
        self.completed = []
        self.generation = self._generation(initial_generation)
        self.armed_at = self._timestamp(armed_at, "armed_at")
        self.ready_at = None
        self.activated_at = None

    @property
    def current(self):
        if self.index >= len(self.missions):
            return None
        return self.missions[self.index]

    @property
    def current_name(self):
        return "" if self.current is None else self.current["name"]

    def readiness_problem(
        self,
        mission_name,
        generation,
        ready_at,
        received_at,
        maximum_age,
        future_tolerance=0.0,
        before_arm_tolerance=0.0,
    ):
        """Return ``None`` only for a current, fresh readiness sample."""
        if self.current is None:
            return "mission sequence is complete"
        if self.state != self.ARMED:
            return "mission is not armed"
        if str(mission_name) != self.current_name:
            return "readiness belongs to '%s', expected '%s'" % (
                mission_name,
                self.current_name,
            )
        try:
            generation = int(generation)
        except (TypeError, ValueError, OverflowError):
            return "readiness generation is invalid"
        if generation != self.generation:
            return "readiness generation %d does not match arm generation %d" % (
                generation,
                self.generation,
            )
        try:
            ready_at = self._timestamp(ready_at, "ready_at")
            received_at = self._timestamp(received_at, "received_at")
        except (TypeError, ValueError, OverflowError) as error:
            return str(error)
        try:
            maximum_age = float(maximum_age)
            future_tolerance = float(future_tolerance)
            before_arm_tolerance = float(before_arm_tolerance)
        except (TypeError, ValueError, OverflowError):
            return "readiness timing limits are invalid"
        if (
            not math.isfinite(maximum_age)
            or maximum_age < 0.0
            or not math.isfinite(future_tolerance)
            or future_tolerance < 0.0
            or not math.isfinite(before_arm_tolerance)
            or before_arm_tolerance < 0.0
        ):
            return "readiness timing limits are invalid"
        if ready_at + before_arm_tolerance < self.armed_at:
            return "readiness predates the current arm"
        if ready_at > received_at + future_tolerance:
            return "readiness timestamp is in the future"
        age = received_at - ready_at
        if age > maximum_age:
            return "readiness is stale by %.3fs" % age
        return None

    def mark_ready(
        self,
        mission_name,
        generation,
        ready_at,
        received_at,
        maximum_age,
        future_tolerance=0.0,
        before_arm_tolerance=0.0,
    ):
        """Move ARMED -> READY after validating identity, age and generation."""
        problem = self.readiness_problem(
            mission_name,
            generation,
            ready_at,
            received_at,
            maximum_age,
            future_tolerance,
            before_arm_tolerance,
        )
        if problem is not None:
            return False
        self.ready_at = float(ready_at)
        self.state = self.READY
        return True

    def activate_current(self, activated_at):
        """Move READY -> ACTIVE; no other state may open the enable gate."""
        if self.current is None or self.state != self.READY:
            return False
        activated_at = self._timestamp(activated_at, "activated_at")
        if activated_at < self.ready_at:
            return False
        self.activated_at = activated_at
        self.state = self.ACTIVE
        return True

    def complete_current(self, mission_name, completed_at):
        """Advance an ACTIVE mission after an external controller completes it."""
        mission = self.current
        if (
            mission is None
            or self.state != self.ACTIVE
            or mission["name"] != mission_name
        ):
            return False
        self._advance(completed_at)
        return True

    def _advance(self, completed_at):
        completed_at = self._timestamp(completed_at, "completed_at")
        if self.activated_at is not None and completed_at < self.activated_at:
            raise ValueError("completed_at cannot predate mission activation")
        self.completed.append(self.current_name)
        self.index += 1
        self.ready_at = None
        self.activated_at = None
        if self.current is None:
            self.state = self.COMPLETE
            self.armed_at = None
            return
        self.generation = self._next_generation(self.generation)
        self.armed_at = completed_at
        self.state = self.ARMED
