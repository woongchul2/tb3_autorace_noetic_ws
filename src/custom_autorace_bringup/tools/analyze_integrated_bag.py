#!/usr/bin/env python3
"""Summarize and validate one completed official-start AutoRace rosbag.

The analyzer is deliberately offline: it reads recorded messages and ROS
connection headers without starting roscore, Gazebo, or any controller.
"""

import argparse
import json
import math
import os
import statistics
import sys


MISSION_ORDER = (
    "intersection",
    "obstacle",
    "parking",
    "zigzag",
    "level_crossing",
    "tunnel",
)
MISSION_STATE_TOPICS = {
    name: "/%s/state" % name for name in MISSION_ORDER
}
EXPECTED_MISSION_CALLERIDS = {
    "intersection": "/intersection_mission_controller",
    "obstacle": "/obstacle_mission_controller",
    "parking": "/parking_mission_controller",
    "zigzag": "/zigzag_mission_controller",
    "level_crossing": "/level_crossing_lidar_controller",
    "tunnel": "/tunnel_mission_controller",
}
MISSION_EXECUTION_STATES = {
    "intersection": {
        "SEARCH_DIRECTION",
        "WAIT_ENTRY_HANDOFF",
        "PREPARE_ENTRY_PATH",
        "FOLLOW_ENTRY_PATH",
        "FOLLOW_ARC_LANE",
        "PREPARE_EXIT_PATH",
        "FOLLOW_EXIT_PATH",
        "VERIFY_FINAL_LANE",
    },
    "obstacle": {"ACQUIRING", "AVOIDING", "REJOINING"},
    "parking": {
        "PREPARE_APPROACH",
        "APPROACH",
        "TURN_IN",
        "ENTER_AISLE",
        "SELECT_SPACE",
        "TURN_TO_SPACE",
        "PARK_IN",
        "BACK_OUT",
        "TURN_TO_EXIT",
        "LEAVE_AISLE",
        "TURN_TO_ZIGZAG",
        "VERIFY_ZIGZAG_LANE",
        "JOIN_ZIGZAG",
    },
    "zigzag": {"ACQUIRING", "FOLLOWING", "VERIFY_EXIT", "JOINING_LANE"},
    "level_crossing": {"APPROACH", "STOPPED", "PASSING"},
    "tunnel": {
        "ACQUIRING",
        "ALIGNING_ENTRY",
        "ENTERING",
        "PLANNING",
        "FOLLOWING",
        "ALIGNING_EXIT",
        "EXITING",
        "VERIFY_EXIT",
        "JOINING_LANE",
    },
}
COMMON_DIAGNOSTIC_TOPICS = {
    name: "/%s/diagnostics" % name
    for name in ("intersection", "obstacle", "parking", "zigzag")
}
MANAGER_TOPICS = ("/mission/current", "/mission/state")
SEQUENCE_INDEX_TOPIC = "/mission/sequence_index"
MISSION_GATE_TOPICS = {
    name: "/mission/enable/%s" % name for name in MISSION_ORDER
}
EXPECTED_MANAGER_CALLERID = "/mission_zone_manager"
LANE_CENTERLINE_TOPIC = "/detect/lane_centerline"
EXPECTED_LANE_CALLERID = "/detect_lane"
CMD_VEL_TOPIC = "/cmd_vel"
MANUAL_STOP_TOPIC = "/control/manual_stop"
EXPECTED_MANUAL_STOP_CALLERID = "/safe_lane_controller"
COMMON_DIAGNOSTIC_FIELDS = 13
EXPECTED_RAW_CMD_VEL_CALLERID_ORDER = (
    "/safe_lane_controller",
    "/intersection_mission_controller",
    "/safe_lane_controller",
    "/intersection_mission_controller",
    "/safe_lane_controller",
    "/obstacle_mission_controller",
    "/safe_lane_controller",
    "/parking_mission_controller",
    "/safe_lane_controller",
    "/zigzag_mission_controller",
    "/safe_lane_controller",
    "/level_crossing_lidar_controller",
    "/safe_lane_controller",
    "/tunnel_mission_controller",
    "/safe_lane_controller",
)
EXPECTED_MOTION_CMD_VEL_CALLERID_ORDER = (
    "/safe_lane_controller",
    "/intersection_mission_controller",
    "/safe_lane_controller",
    "/intersection_mission_controller",
    "/safe_lane_controller",
    "/obstacle_mission_controller",
    "/safe_lane_controller",
    "/parking_mission_controller",
    "/safe_lane_controller",
    "/zigzag_mission_controller",
    "/safe_lane_controller",
    "/tunnel_mission_controller",
    "/safe_lane_controller",
)


def normalize_angle(value):
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _relative(stamp, bag_start):
    return None if stamp is None else float(stamp) - float(bag_start)


def _percentile(values, percent):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * float(percent) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def summarize_rate(
    stamps,
    minimum_rate_hz=29.0,
    maximum_steady_gap=0.20,
    window_start=None,
    window_end=None,
    minimum_coverage_fraction=0.98,
):
    """Return a sustained, gap-aware message-rate summary.

    A short 30 Hz burst is not evidence that perception stayed at 30 Hz for an
    official run.  When a window is supplied, samples must cover that whole
    interval, and the pass rate includes every interval rather than deleting
    stalls from the denominator.
    """

    ordered = [float(stamp) for stamp in stamps]
    if window_start is not None:
        ordered = [stamp for stamp in ordered if stamp >= float(window_start)]
    if window_end is not None:
        ordered = [stamp for stamp in ordered if stamp <= float(window_end)]
    intervals = [
        current - previous
        for previous, current in zip(ordered, ordered[1:])
        if current > previous
    ]
    nonpositive_intervals = max(0, len(ordered) - 1 - len(intervals))
    requested_duration = None
    if window_start is not None and window_end is not None:
        requested_duration = max(0.0, float(window_end) - float(window_start))
    result = {
        "message_count": len(ordered),
        "positive_interval_count": len(intervals),
        "nonpositive_interval_count": nonpositive_intervals,
        "steady_rate_hz": None,
        "overall_rate_hz": None,
        "median_interval_s": None,
        "p95_interval_s": None,
        "maximum_interval_s": None,
        "steady_interval_fraction": 0.0,
        "window_start": None if window_start is None else float(window_start),
        "window_end": None if window_end is None else float(window_end),
        "requested_duration_s": requested_duration,
        "covered_duration_s": 0.0,
        "coverage_fraction": 0.0,
        "minimum_coverage_fraction": float(minimum_coverage_fraction),
        "start_gap_s": None,
        "end_gap_s": None,
        "minimum_required_rate_hz": float(minimum_rate_hz),
        "maximum_allowed_gap_s": float(maximum_steady_gap),
        "pass": False,
    }
    if not intervals:
        return result

    median_interval = statistics.median(intervals)
    gap_limit = max(float(maximum_steady_gap), 3.0 * median_interval)
    steady = [interval for interval in intervals if interval <= gap_limit]
    steady_rate = (
        float(len(steady)) / sum(steady)
        if steady and sum(steady) > 0.0
        else None
    )
    covered_duration = max(0.0, ordered[-1] - ordered[0])
    overall_rate = (
        float(len(ordered) - 1) / covered_duration
        if covered_duration > 0.0
        else None
    )
    steady_fraction = float(len(steady)) / len(intervals)
    if requested_duration is None:
        coverage_fraction = 1.0
        start_gap = 0.0
        end_gap = 0.0
    elif requested_duration <= 0.0:
        coverage_fraction = 0.0
        start_gap = None
        end_gap = None
    else:
        coverage_fraction = min(1.0, covered_duration / requested_duration)
        start_gap = max(0.0, ordered[0] - float(window_start))
        end_gap = max(0.0, float(window_end) - ordered[-1])
    result.update(
        {
            "steady_rate_hz": steady_rate,
            "overall_rate_hz": overall_rate,
            "median_interval_s": median_interval,
            "p95_interval_s": _percentile(intervals, 95),
            "maximum_interval_s": max(intervals),
            "steady_gap_limit_s": gap_limit,
            "steady_interval_fraction": steady_fraction,
            "covered_duration_s": covered_duration,
            "coverage_fraction": coverage_fraction,
            "start_gap_s": start_gap,
            "end_gap_s": end_gap,
            "pass": bool(
                len(ordered) >= 30
                and nonpositive_intervals == 0
                and steady_rate is not None
                and steady_rate >= float(minimum_rate_hz)
                and overall_rate is not None
                and overall_rate >= float(minimum_rate_hz)
                and max(intervals) <= float(maximum_steady_gap)
                and coverage_fraction >= float(minimum_coverage_fraction)
                and start_gap is not None
                and start_gap <= float(maximum_steady_gap)
                and end_gap is not None
                and end_gap <= float(maximum_steady_gap)
            ),
        }
    )
    return result


def _decode_header_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value) if value is not None else ""


def callerid_from_header(header):
    if not header:
        return "<unknown>"
    callerid = _decode_header_value(header.get("callerid", "")).strip()
    return callerid or "<unknown>"


def quaternion_yaw(orientation):
    return math.atan2(
        2.0
        * (
            float(orientation.w) * float(orientation.z)
            + float(orientation.x) * float(orientation.y)
        ),
        1.0
        - 2.0
        * (
            float(orientation.y) * float(orientation.y)
            + float(orientation.z) * float(orientation.z)
        ),
    )


def extract_odom_pose(message):
    pose = message.pose.pose
    twist = message.twist.twist
    values = (
        float(pose.position.x),
        float(pose.position.y),
        quaternion_yaw(pose.orientation),
        math.hypot(float(twist.linear.x), float(twist.linear.y)),
        abs(float(twist.angular.z)),
    )
    return values if all(math.isfinite(value) for value in values) else None


def extract_model_pose(message, model_name, cached_index=None):
    index = cached_index
    if not (
        isinstance(index, int)
        and 0 <= index < len(message.name)
        and message.name[index] == model_name
    ):
        try:
            index = message.name.index(model_name)
        except ValueError:
            return None, None
    pose = message.pose[index]
    twist = message.twist[index]
    values = (
        float(pose.position.x),
        float(pose.position.y),
        quaternion_yaw(pose.orientation),
        math.hypot(float(twist.linear.x), float(twist.linear.y)),
        abs(float(twist.angular.z)),
    )
    if not all(math.isfinite(value) for value in values):
        return None, index
    return values, index


class DiagnosticSummary:
    """Streaming extrema for the public 13-value common path diagnostic."""

    METRICS = (
        ("maximum_position_error_m", 2, "max_abs"),
        ("maximum_absolute_cross_track_error_m", 3, "max_abs"),
        ("minimum_line_clearance_m", 8, "min"),
        ("minimum_obstacle_clearance_m", 9, "min"),
        ("minimum_map_clearance_m", 10, "min"),
    )
    CORE_FINITE_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 11, 12)

    def __init__(self):
        self.message_count = 0
        self.valid_schema_count = 0
        self.malformed_count = 0
        self.usable_core_count = 0
        self.invalid_core_count = 0
        self.minimum_progress = None
        self.maximum_progress = None
        self.extrema = {name: None for name, _, _ in self.METRICS}
        self.extrema_time = {name: None for name, _, _ in self.METRICS}
        self.finite_counts = {name: 0 for name, _, _ in self.METRICS}
        self.nonfinite_counts = {name: 0 for name, _, _ in self.METRICS}

    def observe(self, values, stamp):
        self.message_count += 1
        if len(values) != COMMON_DIAGNOSTIC_FIELDS:
            self.malformed_count += 1
            return
        self.valid_schema_count += 1
        core = [_finite(values[index]) for index in self.CORE_FINITE_INDICES]
        progress = core[0]
        remaining_distance = core[1]
        position_error = core[2]
        heading_error = core[4]
        target_speed = core[5]
        target_index = core[6]
        core_valid = bool(
            all(value is not None for value in core)
            and 0.0 <= progress <= 1.0
            and remaining_distance >= 0.0
            and position_error >= 0.0
            and abs(heading_error) <= math.pi + 1e-6
            and target_speed >= 0.0
            and target_index >= 0.0
            and abs(target_index - round(target_index)) <= 1e-6
        )
        if core_valid:
            self.usable_core_count += 1
            self.minimum_progress = (
                progress
                if self.minimum_progress is None
                else min(self.minimum_progress, progress)
            )
            self.maximum_progress = (
                progress
                if self.maximum_progress is None
                else max(self.maximum_progress, progress)
            )
        else:
            self.invalid_core_count += 1
        for name, index, operation in self.METRICS:
            value = _finite(values[index])
            if value is None:
                self.nonfinite_counts[name] += 1
                continue
            if operation == "max_abs":
                value = abs(value)
            self.finite_counts[name] += 1
            previous = self.extrema[name]
            replace = previous is None
            if previous is not None:
                replace = value > previous if operation == "max_abs" else value < previous
            if replace:
                self.extrema[name] = value
                self.extrema_time[name] = float(stamp)

    def result(self, bag_start):
        result = {
            "message_count": self.message_count,
            "valid_schema_count": self.valid_schema_count,
            "malformed_count": self.malformed_count,
            "usable_core_count": self.usable_core_count,
            "invalid_core_count": self.invalid_core_count,
            "minimum_progress": self.minimum_progress,
            "maximum_progress": self.maximum_progress,
        }
        for name, _, _ in self.METRICS:
            metric_stem = name[:-2] if name.endswith("_m") else name
            result[name] = self.extrema[name]
            result[metric_stem + "_time_s"] = _relative(
                self.extrema_time[name], bag_start
            )
            result[metric_stem + "_finite_sample_count"] = (
                self.finite_counts[name]
            )
            result[metric_stem + "_nonfinite_sample_count"] = (
                self.nonfinite_counts[name]
            )
        return result


class IntegratedRunAccumulator:
    """ROS-independent streaming state used by both rosbag I/O and tests."""

    def __init__(
        self,
        start_x=0.8,
        start_y=-1.747,
        start_yaw=0.0,
        start_position_tolerance=0.08,
        start_yaw_tolerance=math.radians(5.0),
        initial_speed_maximum=0.02,
        initial_angular_speed_maximum=0.05,
        maximum_first_pose_delay=0.50,
        departure_distance=0.03,
        departure_speed=0.03,
        departure_angular_speed=0.10,
        minimum_predeparture_duration=0.20,
        gazebo_pose_settling_duration=0.10,
        finish_x=1.033458,
        finish_min_y=-1.86,
        finish_max_y=-1.64,
        finish_heading=0.0,
        finish_heading_tolerance=math.radians(15.0),
        footprint_front=0.067645,
        footprint_rear=0.118073,
        footprint_half_width=0.0903,
        command_motion_threshold=0.005,
        parking_reverse_threshold=-0.001,
        parking_reverse_minimum_samples=3,
        parking_reverse_minimum_duration=0.08,
        parking_reverse_maximum_gap=0.20,
        minimum_lane_rate_hz=29.0,
        maximum_lane_steady_gap=0.20,
        minimum_lane_coverage_fraction=0.98,
        maximum_lane_source_age=0.25,
        minimum_diagnostic_progress=0.80,
        maximum_geometric_speed=0.75,
        maximum_geometric_angular_speed=4.0,
        maximum_pose_step=0.35,
        maximum_zero_time_pose_step=0.005,
        maximum_zero_time_yaw_step=math.radians(3.0),
    ):
        self.start_x = float(start_x)
        self.start_y = float(start_y)
        self.start_yaw = float(start_yaw)
        self.start_position_tolerance = float(start_position_tolerance)
        self.start_yaw_tolerance = float(start_yaw_tolerance)
        self.initial_speed_maximum = float(initial_speed_maximum)
        self.initial_angular_speed_maximum = float(initial_angular_speed_maximum)
        self.maximum_first_pose_delay = float(maximum_first_pose_delay)
        self.departure_distance = float(departure_distance)
        self.departure_speed = float(departure_speed)
        self.departure_angular_speed = float(departure_angular_speed)
        self.minimum_predeparture_duration = float(minimum_predeparture_duration)
        self.gazebo_pose_settling_duration = float(
            gazebo_pose_settling_duration
        )
        if (
            not math.isfinite(self.gazebo_pose_settling_duration)
            or self.gazebo_pose_settling_duration < 0.0
        ):
            raise ValueError(
                "gazebo_pose_settling_duration must be finite and non-negative"
            )
        self.finish_x = float(finish_x)
        self.finish_min_y = float(finish_min_y)
        self.finish_max_y = float(finish_max_y)
        self.finish_heading = float(finish_heading)
        self.finish_heading_tolerance = float(finish_heading_tolerance)
        self.footprint_front = float(footprint_front)
        self.footprint_rear = float(footprint_rear)
        self.footprint_half_width = float(footprint_half_width)
        expected_corner_x = [
            longitudinal * math.cos(self.finish_heading)
            - lateral * math.sin(self.finish_heading)
            for longitudinal in (self.footprint_front, -self.footprint_rear)
            for lateral in (self.footprint_half_width, -self.footprint_half_width)
        ]
        # ``finish_x`` remains the nominal base-footprint threshold used by
        # earlier reports.  Recover the physical east paint edge, then evaluate
        # every oriented corner against it at runtime.
        self.finish_boundary_x = self.finish_x + min(expected_corner_x)
        self.command_motion_threshold = float(command_motion_threshold)
        self.parking_reverse_threshold = float(parking_reverse_threshold)
        self.parking_reverse_minimum_samples = int(
            parking_reverse_minimum_samples
        )
        self.parking_reverse_minimum_duration = float(
            parking_reverse_minimum_duration
        )
        self.parking_reverse_maximum_gap = float(parking_reverse_maximum_gap)
        self.minimum_lane_rate_hz = float(minimum_lane_rate_hz)
        self.maximum_lane_steady_gap = float(maximum_lane_steady_gap)
        self.minimum_lane_coverage_fraction = float(
            minimum_lane_coverage_fraction
        )
        self.maximum_lane_source_age = float(maximum_lane_source_age)
        self.minimum_diagnostic_progress = float(minimum_diagnostic_progress)
        self.maximum_geometric_speed = float(maximum_geometric_speed)
        self.maximum_geometric_angular_speed = float(
            maximum_geometric_angular_speed
        )
        self.maximum_pose_step = float(maximum_pose_step)
        self.maximum_zero_time_pose_step = float(maximum_zero_time_pose_step)
        self.maximum_zero_time_yaw_step = float(maximum_zero_time_yaw_step)

        self.message_counts = {}
        self.state_transitions = {name: [] for name in MISSION_ORDER}
        self.last_states = {name: None for name in MISSION_ORDER}
        self.state_callerids = {name: set() for name in MISSION_ORDER}
        self.completion_events = []
        self.failure_events = []
        self.manager_transitions = {topic: [] for topic in MANAGER_TOPICS}
        self.manager_last = {topic: None for topic in MANAGER_TOPICS}
        self.manager_callerids = {topic: set() for topic in MANAGER_TOPICS}
        self.sequence_index_transitions = []
        self.last_sequence_index = None
        self.sequence_index_callerids = set()
        self.gate_events = {name: [] for name in MISSION_ORDER}
        self.gate_callerids = {name: set() for name in MISSION_ORDER}
        self.diagnostic_samples = {
            name: [] for name in COMMON_DIAGNOSTIC_TOPICS
        }
        self.lane_samples = []
        self.cmd_runs = []
        self.motion_cmd_runs = []
        self.first_motion_command_time = None
        self.parking_cmd_count = 0
        self.parking_state_epoch = 0
        self.parking_back_out_samples = []
        self.parking_negative = []
        self.parking_minimum_linear = None
        self.manual_stop_events = []
        self.manual_stop_callerids = set()
        self.first_pose = None
        self.previous_pose = None
        self.pose_count = 0
        self.predeparture_pose_samples = []
        self.first_pose_speed_departure = None
        self.first_pose_angular_departure = None
        self.first_pose_distance_departure = None
        self.east_crossings = []
        self.pose_discontinuities = []
        self.maximum_observed_geometric_speed = 0.0
        self.maximum_observed_geometric_angular_speed = 0.0
        self.maximum_observed_pose_step = 0.0

    def count_topic(self, topic):
        self.message_counts[topic] = self.message_counts.get(topic, 0) + 1

    def observe_state(self, mission, state, stamp, callerid="<unknown>"):
        state = str(state)
        stamp = float(stamp)
        callerid = str(callerid) if callerid else "<unknown>"
        self.state_callerids[mission].add(callerid)
        if self.last_states[mission] == state:
            return
        if mission == "parking":
            self.parking_state_epoch += 1
        self.last_states[mission] = state
        self.state_transitions[mission].append((stamp, state, callerid))
        if state == "COMPLETE":
            if not any(event[1] == mission for event in self.completion_events):
                self.completion_events.append((stamp, mission))
        if state.startswith("FAILED"):
            self.failure_events.append((stamp, mission, state))

    def observe_manager_state(
        self, topic, state, stamp, callerid="<unknown>"
    ):
        state = str(state)
        callerid = str(callerid) if callerid else "<unknown>"
        self.manager_callerids[topic].add(callerid)
        if self.manager_last[topic] == state:
            return
        self.manager_last[topic] = state
        self.manager_transitions[topic].append(
            (float(stamp), state, callerid)
        )

    def observe_sequence_index(self, value, stamp, callerid="<unknown>"):
        value = int(value)
        callerid = str(callerid) if callerid else "<unknown>"
        self.sequence_index_callerids.add(callerid)
        if self.last_sequence_index == value:
            return
        self.last_sequence_index = value
        self.sequence_index_transitions.append(
            (float(stamp), value, callerid)
        )

    def observe_gate(
        self, mission, enabled, stamp, callerid="<unknown>"
    ):
        callerid = str(callerid) if callerid else "<unknown>"
        self.gate_callerids[mission].add(callerid)
        self.gate_events[mission].append(
            (float(stamp), bool(enabled), callerid)
        )

    def observe_diagnostics(
        self, mission, values, stamp, callerid="<unknown>"
    ):
        callerid = str(callerid) if callerid else "<unknown>"
        self.diagnostic_samples[mission].append(
            (float(stamp), tuple(values), callerid)
        )

    def observe_lane_centerline(
        self, stamp, source_stamp=None, callerid="<unknown>"
    ):
        source_stamp = _finite(source_stamp)
        callerid = str(callerid) if callerid else "<unknown>"
        self.lane_samples.append(
            (float(stamp), source_stamp, callerid)
        )

    def observe_manual_stop(
        self, stamp, requested, callerid="<unknown>"
    ):
        callerid = str(callerid) if callerid else "<unknown>"
        self.manual_stop_callerids.add(callerid)
        self.manual_stop_events.append(
            (float(stamp), bool(requested), callerid)
        )

    def observe_cmd_vel(self, stamp, linear, angular, callerid):
        stamp = float(stamp)
        linear = float(linear)
        angular = float(angular)
        callerid = str(callerid) if callerid else "<unknown>"
        if not all(math.isfinite(value) for value in (stamp, linear, angular)):
            return
        if (
            self.first_motion_command_time is None
            and (
                abs(linear) > self.command_motion_threshold
                or abs(angular) > self.command_motion_threshold
            )
        ):
            self.first_motion_command_time = stamp

        if not self.cmd_runs or self.cmd_runs[-1]["callerid"] != callerid:
            self.cmd_runs.append(
                {
                    "callerid": callerid,
                    "start": stamp,
                    "end": stamp,
                    "message_count": 0,
                    "minimum_linear_velocity_mps": linear,
                    "maximum_linear_velocity_mps": linear,
                }
            )
        run = self.cmd_runs[-1]
        run["end"] = stamp
        run["message_count"] += 1
        run["minimum_linear_velocity_mps"] = min(
            run["minimum_linear_velocity_mps"], linear
        )
        run["maximum_linear_velocity_mps"] = max(
            run["maximum_linear_velocity_mps"], linear
        )

        moving = (
            abs(linear) > self.command_motion_threshold
            or abs(angular) > self.command_motion_threshold
        )
        if moving:
            if (
                not self.motion_cmd_runs
                or self.motion_cmd_runs[-1]["callerid"] != callerid
            ):
                self.motion_cmd_runs.append(
                    {
                        "callerid": callerid,
                        "start": stamp,
                        "end": stamp,
                        "message_count": 0,
                    }
                )
            motion_run = self.motion_cmd_runs[-1]
            motion_run["end"] = stamp
            motion_run["message_count"] += 1

        parking_state = self.last_states["parking"] or ""
        from_parking = (
            callerid == EXPECTED_MISSION_CALLERIDS["parking"]
        )
        if from_parking:
            self.parking_cmd_count += 1
            self.parking_minimum_linear = (
                linear
                if self.parking_minimum_linear is None
                else min(self.parking_minimum_linear, linear)
            )
            if parking_state == "BACK_OUT":
                reverse = linear < self.parking_reverse_threshold
                self.parking_back_out_samples.append(
                    (
                        stamp,
                        linear,
                        callerid,
                        parking_state,
                        reverse,
                        self.parking_state_epoch,
                    )
                )
            if parking_state == "BACK_OUT" and reverse:
                self.parking_negative.append(
                    (stamp, linear, callerid, parking_state)
                )

    def _footprint_corners(self, x, y, yaw):
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return [
            (
                x + cosine * longitudinal - sine * lateral,
                y + sine * longitudinal + cosine * lateral,
            )
            for longitudinal in (self.footprint_front, -self.footprint_rear)
            for lateral in (self.footprint_half_width, -self.footprint_half_width)
        ]

    def _finish_pose_metrics(self, x, y, yaw):
        corners = self._footprint_corners(x, y, yaw)
        return (
            min(point[0] for point in corners) - self.finish_boundary_x,
            min(point[1] for point in corners),
            max(point[1] for point in corners),
            abs(normalize_angle(yaw - self.finish_heading)),
        )

    def observe_pose(
        self, stamp, x, y, yaw, speed, angular_speed=0.0
    ):
        values = tuple(
            float(value)
            for value in (stamp, x, y, yaw, speed, angular_speed)
        )
        if not all(math.isfinite(value) for value in values):
            return
        stamp, x, y, yaw, speed, angular_speed = values
        yaw = normalize_angle(yaw)
        speed = abs(speed)
        angular_speed = abs(angular_speed)
        sample = (stamp, x, y, yaw, speed, angular_speed)
        self.pose_count += 1
        if self.first_pose is None:
            self.first_pose = sample
        else:
            displacement = math.hypot(x - self.first_pose[1], y - self.first_pose[2])
            if (
                self.first_pose_distance_departure is None
                and displacement >= self.departure_distance
            ):
                self.first_pose_distance_departure = stamp
        if (
            self.first_motion_command_time is None
            and self.first_pose_speed_departure is None
            and self.first_pose_angular_departure is None
            and self.first_pose_distance_departure is None
        ):
            self.predeparture_pose_samples.append(sample)
        if self.first_pose_speed_departure is None and speed > self.departure_speed:
            self.first_pose_speed_departure = stamp
        if (
            self.first_pose_angular_departure is None
            and angular_speed > self.departure_angular_speed
        ):
            self.first_pose_angular_departure = stamp

        previous = self.previous_pose
        if previous is not None:
            elapsed = stamp - previous[0]
            distance = math.hypot(x - previous[1], y - previous[2])
            yaw_delta = normalize_angle(yaw - previous[3])
            self.maximum_observed_pose_step = max(
                self.maximum_observed_pose_step, distance
            )
            if elapsed > 1e-9:
                geometric_speed = distance / elapsed
                geometric_angular_speed = abs(yaw_delta) / elapsed
                self.maximum_observed_geometric_speed = max(
                    self.maximum_observed_geometric_speed, geometric_speed
                )
                self.maximum_observed_geometric_angular_speed = max(
                    self.maximum_observed_geometric_angular_speed,
                    geometric_angular_speed,
                )
                if (
                    distance > self.maximum_pose_step
                    or geometric_speed > self.maximum_geometric_speed
                    or geometric_angular_speed
                    > self.maximum_geometric_angular_speed
                ):
                    if len(self.pose_discontinuities) < 50:
                        self.pose_discontinuities.append(
                            {
                                "time": stamp,
                                "step_m": distance,
                                "linear_rate_mps": geometric_speed,
                                "angular_rate_radps": geometric_angular_speed,
                            }
                        )
            elif (
                elapsed < 0.0
                or (
                    elapsed <= 1e-9
                    and (
                        distance > self.maximum_zero_time_pose_step
                        or abs(yaw_delta) > self.maximum_zero_time_yaw_step
                    )
                )
            ) and len(self.pose_discontinuities) < 50:
                self.pose_discontinuities.append(
                    {
                        "time": stamp,
                        "step_m": distance,
                        "linear_rate_mps": math.inf,
                        "angular_rate_radps": math.inf,
                    }
                )

            previous_clearance = self._finish_pose_metrics(
                previous[1], previous[2], previous[3]
            )[0]
            current_clearance = self._finish_pose_metrics(x, y, yaw)[0]
            if (
                previous_clearance <= 0.0 <= current_clearance
                and x > previous[1]
                and current_clearance > previous_clearance
            ):
                # A rotating asymmetric rectangle has a non-linear corner-x
                # clearance.  Solve the actual oriented footprint crossing
                # instead of linearly interpolating the endpoint clearances.
                if previous_clearance == 0.0:
                    fraction = 0.0
                else:
                    lower, upper = 0.0, 1.0
                    for _ in range(40):
                        fraction = 0.5 * (lower + upper)
                        trial_x = previous[1] + fraction * (x - previous[1])
                        trial_y = previous[2] + fraction * (y - previous[2])
                        trial_yaw = normalize_angle(
                            previous[3] + fraction * yaw_delta
                        )
                        trial_clearance = self._finish_pose_metrics(
                            trial_x, trial_y, trial_yaw
                        )[0]
                        if trial_clearance < 0.0:
                            lower = fraction
                        else:
                            upper = fraction
                    fraction = upper
                crossing_y = previous[2] + fraction * (y - previous[2])
                crossing_x = previous[1] + fraction * (x - previous[1])
                crossing_yaw = normalize_angle(
                    previous[3] + fraction * yaw_delta
                )
                crossing_time = previous[0] + fraction * elapsed
                (
                    _,
                    minimum_corner_y,
                    maximum_corner_y,
                    heading_error,
                ) = self._finish_pose_metrics(
                    crossing_x, crossing_y, crossing_yaw
                )
                within_y = bool(
                    minimum_corner_y >= self.finish_min_y
                    and maximum_corner_y <= self.finish_max_y
                )
                heading_valid = bool(
                    heading_error <= self.finish_heading_tolerance
                )
                if within_y and heading_valid:
                    self.east_crossings.append(
                        {
                            "time": crossing_time,
                            "x": crossing_x,
                            "y": crossing_y,
                            "yaw": crossing_yaw,
                            "minimum_corner_y": minimum_corner_y,
                            "maximum_corner_y": maximum_corner_y,
                            "heading_error": heading_error,
                        }
                    )
        self.previous_pose = sample

    @staticmethod
    def _value_at(transitions, stamp):
        value = None
        for event in transitions:
            if event[0] > stamp:
                break
            value = event[1]
        return value

    def _manager_result(self, bag_start):
        current_transitions = self.manager_transitions["/mission/current"]
        state_transitions = self.manager_transitions["/mission/state"]
        current_values = [value for _, value, _ in current_transitions]
        state_values = [value for _, value, _ in state_transitions]
        expected_current = list(MISSION_ORDER) + [""]
        expected_indices = list(range(len(MISSION_ORDER) + 1))
        observed_indices = [
            value for _, value, _ in self.sequence_index_transitions
        ]

        active_events = []
        for stamp, state, _ in state_transitions:
            if state != "ACTIVE":
                continue
            current = self._value_at(current_transitions, stamp)
            if current is not None:
                active_events.append((stamp, current))
        active_order = [mission for _, mission in active_events]
        active_times = {}
        for stamp, mission in active_events:
            if mission in MISSION_ORDER and mission not in active_times:
                active_times[mission] = stamp

        expected_manager_caller = {EXPECTED_MANAGER_CALLERID}
        manager_publishers_pass = bool(
            all(
                self.manager_callerids[topic] == expected_manager_caller
                for topic in MANAGER_TOPICS
            )
            and self.sequence_index_callerids == expected_manager_caller
            and all(
                self.gate_callerids[mission] == expected_manager_caller
                for mission in MISSION_ORDER
            )
        )
        gate_evidence = {}
        for mission in MISSION_ORDER:
            active_time = active_times.get(mission)
            complete_time = next(
                (
                    stamp
                    for stamp, completed_mission in self.completion_events
                    if completed_mission == mission
                ),
                None,
            )
            true_events = [
                stamp
                for stamp, enabled, callerid in self.gate_events[mission]
                if enabled and callerid == EXPECTED_MANAGER_CALLERID
            ]
            matching = [
                stamp
                for stamp in true_events
                if active_time is not None
                and stamp >= active_time - 0.10
                and (complete_time is None or stamp <= complete_time)
            ]
            gate_evidence[mission] = {
                "topic": MISSION_GATE_TOPICS[mission],
                "message_count": self.message_counts.get(
                    MISSION_GATE_TOPICS[mission], 0
                ),
                "callerids": sorted(self.gate_callerids[mission]),
                "true_event_times_s": [
                    _relative(stamp, bag_start) for stamp in true_events
                ],
                "matching_active_event": bool(matching),
            }

        return {
            "expected_current_order": expected_current,
            "observed_current_order": current_values,
            "current_order_pass": current_values == expected_current,
            "expected_active_order": list(MISSION_ORDER),
            "observed_active_order": active_order,
            "active_order_pass": active_order == list(MISSION_ORDER),
            "active_times": active_times,
            "expected_sequence_indices": expected_indices,
            "observed_sequence_indices": observed_indices,
            "sequence_index_pass": observed_indices == expected_indices,
            "final_complete_pass": bool(
                state_values and state_values[-1] == "COMPLETE"
            ),
            "publisher_pass": manager_publishers_pass,
            "gate_evidence": gate_evidence,
            "gate_evidence_pass": all(
                evidence["matching_active_event"]
                for evidence in gate_evidence.values()
            ),
            "transitions": {
                topic: [
                    {
                        "time_s": _relative(stamp, bag_start),
                        "state": state,
                        "callerid": callerid,
                    }
                    for stamp, state, callerid in self.manager_transitions[topic]
                ]
                for topic in MANAGER_TOPICS
            },
            "sequence_index_transitions": [
                {
                    "time_s": _relative(stamp, bag_start),
                    "index": value,
                    "callerid": callerid,
                }
                for stamp, value, callerid in self.sequence_index_transitions
            ],
        }

    def _mission_result(self, bag_start, manager):
        missions = {}
        active_times = manager["active_times"]
        for mission in MISSION_ORDER:
            transitions = self.state_transitions[mission]
            complete_times = [
                stamp
                for stamp, state, _ in transitions
                if state == "COMPLETE"
            ]
            failed = [
                (stamp, state)
                for stamp, state, _ in transitions
                if state.startswith("FAILED")
            ]
            active_time = active_times.get(mission)
            complete_time = complete_times[0] if complete_times else None
            execution_states = [
                state
                for stamp, state, _ in transitions
                if active_time is not None
                and complete_time is not None
                and active_time <= stamp <= complete_time
                and state in MISSION_EXECUTION_STATES[mission]
            ]
            idle_state = (
                "WAIT_INTERSECTION" if mission == "intersection" else "WAIT_GATE"
            )
            allowed_states = set(MISSION_EXECUTION_STATES[mission])
            allowed_states.update((idle_state, "COMPLETE", "FAILED"))
            unexpected_states = sorted(
                {
                    state
                    for _, state, _ in transitions
                    if state not in allowed_states
                    and not state.startswith("FAILED")
                }
            )
            linked = bool(
                active_time is not None
                and complete_time is not None
                and complete_time >= active_time
                and execution_states
                and manager["gate_evidence"][mission][
                    "matching_active_event"
                ]
            )
            expected_callerid = EXPECTED_MISSION_CALLERIDS[mission]
            callerids = sorted(self.state_callerids[mission])
            missions[mission] = {
                "topic": MISSION_STATE_TOPICS[mission],
                "message_count": self.message_counts.get(
                    MISSION_STATE_TOPICS[mission], 0
                ),
                "transitions": [
                    {
                        "time_s": _relative(stamp, bag_start),
                        "state": state,
                        "callerid": callerid,
                    }
                    for stamp, state, callerid in transitions
                ],
                "complete": bool(complete_times),
                "complete_time_s": _relative(
                    complete_time, bag_start
                ),
                "active_time_s": _relative(active_time, bag_start),
                "elapsed_s": (
                    complete_time - active_time
                    if active_time is not None and complete_time is not None
                    else None
                ),
                "active_complete_link_pass": linked,
                "execution_states": execution_states,
                "allowed_states": sorted(allowed_states),
                "unexpected_states": unexpected_states,
                "state_values_pass": not unexpected_states,
                "expected_callerid": expected_callerid,
                "callerids": callerids,
                "publisher_pass": callerids == [expected_callerid],
                "failed": bool(failed),
                "failed_events": [
                    {"time_s": _relative(stamp, bag_start), "state": state}
                    for stamp, state in failed
                ],
            }
        order = [mission for _, mission in sorted(self.completion_events)]
        all_complete = all(missions[name]["complete"] for name in MISSION_ORDER)
        links_pass = all(
            missions[name]["active_complete_link_pass"] for name in MISSION_ORDER
        )
        publishers_pass = all(
            missions[name]["publisher_pass"] for name in MISSION_ORDER
        )
        state_values_pass = all(
            missions[name]["state_values_pass"] for name in MISSION_ORDER
        )
        return {
            "expected_completion_order": list(MISSION_ORDER),
            "observed_completion_order": order,
            "completion_order_pass": order == list(MISSION_ORDER),
            "all_complete": all_complete,
            "any_failed": bool(self.failure_events),
            "active_complete_links_pass": links_pass,
            "state_publishers_pass": publishers_pass,
            "state_values_pass": state_values_pass,
            "missions": missions,
            "manager": {
                key: value
                for key, value in manager.items()
                if key != "active_times"
            },
        }

    def _predeparture_result(self, bag_start, pose_source):
        departure_candidates = [
            value
            for value in (
                self.first_motion_command_time,
                self.first_pose_speed_departure,
                self.first_pose_angular_departure,
                self.first_pose_distance_departure,
            )
            if value is not None
        ]
        departure = min(departure_candidates) if departure_candidates else None
        first_pose = self.first_pose
        initial_error = (
            math.hypot(first_pose[1] - self.start_x, first_pose[2] - self.start_y)
            if first_pose is not None
            else None
        )
        initial_yaw_error = (
            abs(normalize_angle(first_pose[3] - self.start_yaw))
            if first_pose is not None
            else None
        )
        bag_lead = departure - bag_start if departure is not None else None
        first_pose_delay = (
            first_pose[0] - bag_start if first_pose is not None else None
        )
        predeparture_samples = [
            sample
            for sample in self.predeparture_pose_samples
            if departure is not None and sample[0] < departure
        ]
        gazebo_pose_source = str(pose_source) in (
            "gazebo",
            "gazebo_model_states",
            "/gazebo/model_states",
        )
        pose_settling_applied = bool(
            gazebo_pose_source
            and first_pose is not None
            and self.gazebo_pose_settling_duration > 0.0
        )
        applied_settling_duration = (
            self.gazebo_pose_settling_duration
            if pose_settling_applied
            else 0.0
        )
        stationary_validation_start = (
            first_pose[0] + applied_settling_duration
            if first_pose is not None
            else None
        )
        settling_samples = [
            sample
            for sample in predeparture_samples
            if stationary_validation_start is not None
            and sample[0] < stationary_validation_start - 1e-9
        ]
        stationary_samples = [
            sample
            for sample in predeparture_samples
            if stationary_validation_start is not None
            and sample[0] >= stationary_validation_start - 1e-9
        ]
        position_errors = [
            math.hypot(sample[1] - self.start_x, sample[2] - self.start_y)
            for sample in predeparture_samples
        ]
        yaw_errors = [
            abs(normalize_angle(sample[3] - self.start_yaw))
            for sample in predeparture_samples
        ]
        linear_speeds = [sample[4] for sample in stationary_samples]
        angular_speeds = [sample[5] for sample in stationary_samples]
        settling_linear_speeds = [sample[4] for sample in settling_samples]
        settling_angular_speeds = [sample[5] for sample in settling_samples]
        stationary_observation = (
            departure - stationary_validation_start
            if departure is not None and stationary_validation_start is not None
            else None
        )
        raw_predeparture_observation = (
            departure - first_pose[0]
            if departure is not None and first_pose is not None
            else None
        )
        passed = bool(
            first_pose is not None
            and initial_error <= self.start_position_tolerance
            and initial_yaw_error <= self.start_yaw_tolerance
            and first_pose[4] <= self.initial_speed_maximum
            and first_pose[5] <= self.initial_angular_speed_maximum
            and departure is not None
            and first_pose_delay is not None
            and 0.0 <= first_pose_delay <= self.maximum_first_pose_delay
            and stationary_observation is not None
            and stationary_observation >= self.minimum_predeparture_duration
            and len(stationary_samples) >= 2
            and max(position_errors) <= self.start_position_tolerance
            and max(yaw_errors) <= self.start_yaw_tolerance
            and max(linear_speeds) <= self.initial_speed_maximum
            and max(angular_speeds) <= self.initial_angular_speed_maximum
        )
        return {
            "pass": passed,
            "configured_start": {
                "x": self.start_x,
                "y": self.start_y,
                "yaw_rad": self.start_yaw,
            },
            "start_position_tolerance_m": self.start_position_tolerance,
            "start_yaw_tolerance_rad": self.start_yaw_tolerance,
            "initial_speed_maximum_mps": self.initial_speed_maximum,
            "initial_angular_speed_maximum_radps": (
                self.initial_angular_speed_maximum
            ),
            "maximum_first_pose_delay_s": self.maximum_first_pose_delay,
            "minimum_predeparture_duration_s": self.minimum_predeparture_duration,
            "configured_gazebo_pose_settling_duration_s": (
                self.gazebo_pose_settling_duration
            ),
            "pose_settling_applied": pose_settling_applied,
            "applied_pose_settling_duration_s": applied_settling_duration,
            "stationary_validation_start_time_s": _relative(
                stationary_validation_start, bag_start
            ),
            "first_pose_time_s": _relative(
                first_pose[0] if first_pose is not None else None, bag_start
            ),
            "first_pose_x": first_pose[1] if first_pose is not None else None,
            "first_pose_y": first_pose[2] if first_pose is not None else None,
            "first_pose_yaw_rad": first_pose[3] if first_pose is not None else None,
            "first_pose_speed_mps": first_pose[4] if first_pose is not None else None,
            "first_pose_angular_speed_radps": (
                first_pose[5] if first_pose is not None else None
            ),
            "initial_position_error_m": initial_error,
            "initial_yaw_error_rad": initial_yaw_error,
            "first_pose_delay_s": first_pose_delay,
            "departure_time_s": _relative(departure, bag_start),
            "bag_predeparture_duration_s": bag_lead,
            "predeparture_duration_s": stationary_observation,
            "raw_predeparture_duration_s": raw_predeparture_observation,
            "pose_samples_before_departure": len(predeparture_samples),
            "pose_samples_during_settling": len(settling_samples),
            "pose_samples_after_settling": len(stationary_samples),
            "maximum_predeparture_position_error_m": (
                max(position_errors) if position_errors else None
            ),
            "maximum_predeparture_yaw_error_rad": (
                max(yaw_errors) if yaw_errors else None
            ),
            "maximum_predeparture_linear_speed_mps": (
                max(linear_speeds) if linear_speeds else None
            ),
            "maximum_predeparture_angular_speed_radps": (
                max(angular_speeds) if angular_speeds else None
            ),
            "maximum_settling_linear_speed_mps": (
                max(settling_linear_speeds)
                if settling_linear_speeds
                else None
            ),
            "maximum_settling_angular_speed_radps": (
                max(settling_angular_speeds)
                if settling_angular_speeds
                else None
            ),
            "departure_stamp": departure,
            "departure_evidence": {
                "first_nonzero_cmd_vel_time_s": _relative(
                    self.first_motion_command_time, bag_start
                ),
                "first_pose_speed_time_s": _relative(
                    self.first_pose_speed_departure, bag_start
                ),
                "first_pose_angular_speed_time_s": _relative(
                    self.first_pose_angular_departure, bag_start
                ),
                "first_pose_displacement_time_s": _relative(
                    self.first_pose_distance_departure, bag_start
                ),
            },
        }

    def _diagnostics_result(self, bag_start, manager):
        result = {}
        for mission, topic in COMMON_DIAGNOSTIC_TOPICS.items():
            active_time = manager["active_times"].get(mission)
            complete_time = next(
                (
                    stamp
                    for stamp, completed_mission in self.completion_events
                    if completed_mission == mission
                ),
                None,
            )
            samples = [
                sample
                for sample in self.diagnostic_samples[mission]
                if active_time is not None
                and complete_time is not None
                and active_time <= sample[0] <= complete_time
            ]
            summary = DiagnosticSummary()
            for stamp, values, _ in samples:
                summary.observe(values, stamp)
            values = summary.result(bag_start)
            window_callerids = sorted({sample[2] for sample in samples})
            callerids = sorted(
                {sample[2] for sample in self.diagnostic_samples[mission]}
            )
            expected_callerid = EXPECTED_MISSION_CALLERIDS[mission]
            progress_pass = bool(
                values["maximum_progress"] is not None
                and values["maximum_progress"]
                >= self.minimum_diagnostic_progress
            )
            line_finite_pass = bool(
                values["valid_schema_count"] > 0
                and values["minimum_line_clearance_finite_sample_count"]
                == values["valid_schema_count"]
            )
            diagnostic_pass = bool(
                samples
                and values["valid_schema_count"] > 0
                and values["malformed_count"] == 0
                and values["usable_core_count"] > 0
                and values["invalid_core_count"] == 0
                and progress_pass
                and line_finite_pass
                and callerids == [expected_callerid]
            )
            result[mission] = dict(
                {
                    "topic": topic,
                    "total_message_count": len(
                        self.diagnostic_samples[mission]
                    ),
                    "window_start_time_s": _relative(active_time, bag_start),
                    "window_end_time_s": _relative(complete_time, bag_start),
                    "expected_callerid": expected_callerid,
                    "callerids": callerids,
                    "window_callerids": window_callerids,
                    "minimum_required_progress": (
                        self.minimum_diagnostic_progress
                    ),
                    "progress_pass": progress_pass,
                    "line_clearance_finite_pass": line_finite_pass,
                    "pass": diagnostic_pass,
                },
                **values
            )
        return result

    def _parking_reverse_result(self, bag_start, manager):
        active_time = manager["active_times"].get("parking")
        complete_time = next(
            (
                stamp
                for stamp, completed_mission in self.completion_events
                if completed_mission == "parking"
                and active_time is not None
                and stamp >= active_time
            ),
            None,
        )
        scoped_samples = [
            sample
            for sample in self.parking_back_out_samples
            if active_time is not None
            and complete_time is not None
            and active_time <= sample[0] <= complete_time
        ]
        runs = []
        current_run = None
        for sample in scoped_samples:
            if not sample[4]:
                current_run = None
                continue
            if (
                current_run is None
                or sample[5] != current_run["state_epoch"]
                or sample[0] - current_run["end"]
                > self.parking_reverse_maximum_gap
            ):
                current_run = {
                    "start": sample[0],
                    "end": sample[0],
                    "sample_count": 0,
                    "minimum_linear_velocity_mps": sample[1],
                    "state_epoch": sample[5],
                }
                runs.append(current_run)
            current_run["end"] = sample[0]
            current_run["sample_count"] += 1
            current_run["minimum_linear_velocity_mps"] = min(
                current_run["minimum_linear_velocity_mps"], sample[1]
            )
        serialized_runs = []
        valid_runs = []
        for run in runs:
            duration = max(0.0, run["end"] - run["start"])
            valid = bool(
                run["sample_count"] >= self.parking_reverse_minimum_samples
                and duration >= self.parking_reverse_minimum_duration
            )
            serialized = {
                "start_time_s": _relative(run["start"], bag_start),
                "end_time_s": _relative(run["end"], bag_start),
                "duration_s": duration,
                "sample_count": run["sample_count"],
                "minimum_linear_velocity_mps": run[
                    "minimum_linear_velocity_mps"
                ],
                "valid": valid,
            }
            serialized_runs.append(serialized)
            if valid:
                valid_runs.append(serialized)
        return {
            "required_state": "BACK_OUT",
            "expected_callerid": EXPECTED_MISSION_CALLERIDS["parking"],
            "window_start_time_s": _relative(active_time, bag_start),
            "window_end_time_s": _relative(complete_time, bag_start),
            "parking_cmd_sample_count": self.parking_cmd_count,
            "back_out_cmd_sample_count": len(scoped_samples),
            "negative_linear_sample_count": sum(
                1 for sample in scoped_samples if sample[4]
            ),
            "reverse_threshold_mps": self.parking_reverse_threshold,
            "minimum_samples": self.parking_reverse_minimum_samples,
            "minimum_duration_s": self.parking_reverse_minimum_duration,
            "maximum_sample_gap_s": self.parking_reverse_maximum_gap,
            "minimum_linear_velocity_mps": self.parking_minimum_linear,
            "runs": serialized_runs,
            "valid_run_count": len(valid_runs),
            "pass": bool(valid_runs),
        }

    def _lane_result(self, bag_start, departure, finish_stamp):
        scoped = [
            sample
            for sample in self.lane_samples
            if departure is not None
            and finish_stamp is not None
            and departure <= sample[0] <= finish_stamp
        ]
        publish_stamps = [sample[0] for sample in scoped]
        lane = summarize_rate(
            publish_stamps,
            self.minimum_lane_rate_hz,
            self.maximum_lane_steady_gap,
            window_start=departure,
            window_end=finish_stamp,
            minimum_coverage_fraction=self.minimum_lane_coverage_fraction,
        )
        source_stamps = [sample[1] for sample in scoped if sample[1] is not None]
        source_complete = len(source_stamps) == len(scoped) and bool(scoped)
        source_rate = summarize_rate(
            source_stamps,
            self.minimum_lane_rate_hz,
            self.maximum_lane_steady_gap,
        )
        source_ages = [
            sample[0] - sample[1]
            for sample in scoped
            if sample[1] is not None
        ]
        freshness_pass = bool(
            source_complete
            and source_ages
            and min(source_ages) >= -0.05
            and max(source_ages) <= self.maximum_lane_source_age
        )
        callerids = sorted({sample[2] for sample in scoped})
        publisher_pass = callerids == [EXPECTED_LANE_CALLERID]
        lane.update(
            {
                "topic": LANE_CENTERLINE_TOPIC,
                "total_message_count": len(self.lane_samples),
                "first_time_s": _relative(
                    publish_stamps[0] if publish_stamps else None, bag_start
                ),
                "last_time_s": _relative(
                    publish_stamps[-1] if publish_stamps else None, bag_start
                ),
                "expected_callerid": EXPECTED_LANE_CALLERID,
                "callerids": callerids,
                "publisher_pass": publisher_pass,
                "source_stamp_complete": source_complete,
                "source_stamp_rate": source_rate,
                "minimum_source_age_s": min(source_ages) if source_ages else None,
                "maximum_source_age_s": max(source_ages) if source_ages else None,
                "maximum_allowed_source_age_s": self.maximum_lane_source_age,
                "freshness_pass": freshness_pass,
            }
        )
        lane["publish_rate_pass"] = lane["pass"]
        lane["pass"] = bool(
            lane["publish_rate_pass"]
            and source_rate["pass"]
            and freshness_pass
            and publisher_pass
        )
        return lane

    def result(self, bag_start, bag_end, bag_path, pose_source):
        bag_start = float(bag_start)
        manager = self._manager_result(bag_start)
        mission = self._mission_result(bag_start, manager)
        predeparture = self._predeparture_result(bag_start, pose_source)
        departure = predeparture.pop("departure_stamp")
        diagnostics = self._diagnostics_result(bag_start, manager)
        diagnostics_pass = all(
            summary["pass"] for summary in diagnostics.values()
        )

        tunnel_complete = next(
            (
                stamp
                for stamp, completed_mission in self.completion_events
                if completed_mission == "tunnel"
            ),
            None,
        )
        finish_after_tunnel = [
            crossing
            for crossing in self.east_crossings
            if tunnel_complete is not None
            and crossing["time"] >= tunnel_complete
        ]
        finish_stamp = (
            finish_after_tunnel[0]["time"] if finish_after_tunnel else None
        )
        finish = {
            "pose_source": pose_source,
            "pose_sample_count": self.pose_count,
            "nominal_base_x": self.finish_x,
            "physical_east_boundary_x": self.finish_boundary_x,
            "minimum_y": self.finish_min_y,
            "maximum_y": self.finish_max_y,
            "expected_heading_rad": self.finish_heading,
            "heading_tolerance_rad": self.finish_heading_tolerance,
            "footprint": {
                "front": self.footprint_front,
                "rear": self.footprint_rear,
                "half_width": self.footprint_half_width,
            },
            "oriented_footprint_crossing_count": len(self.east_crossings),
            "oriented_footprint_crossings": [
                {
                    "time_s": _relative(crossing["time"], bag_start),
                    "x": crossing["x"],
                    "y": crossing["y"],
                    "yaw_rad": crossing["yaw"],
                    "minimum_corner_y": crossing["minimum_corner_y"],
                    "maximum_corner_y": crossing["maximum_corner_y"],
                    "heading_error_rad": crossing["heading_error"],
                    "after_tunnel_complete": bool(
                        tunnel_complete is not None
                        and crossing["time"] >= tunnel_complete
                    ),
                }
                for crossing in self.east_crossings
            ],
            "after_tunnel_count": len(finish_after_tunnel),
            "first_after_tunnel_time_s": _relative(finish_stamp, bag_start),
            "pass": bool(finish_after_tunnel),
        }

        lane = self._lane_result(bag_start, departure, finish_stamp)

        publisher_runs = []
        for ordinal, run in enumerate(self.cmd_runs, 1):
            publisher_runs.append(
                {
                    "ordinal": ordinal,
                    "callerid": run["callerid"],
                    "start_time_s": _relative(run["start"], bag_start),
                    "end_time_s": _relative(run["end"], bag_start),
                    "duration_s": max(0.0, run["end"] - run["start"]),
                    "message_count": run["message_count"],
                    "minimum_linear_velocity_mps": run[
                        "minimum_linear_velocity_mps"
                    ],
                    "maximum_linear_velocity_mps": run[
                        "maximum_linear_velocity_mps"
                    ],
                }
            )
        unique_callerids = []
        for run in publisher_runs:
            if run["callerid"] not in unique_callerids:
                unique_callerids.append(run["callerid"])
        callerids_present = bool(publisher_runs) and all(
            run["callerid"] != "<unknown>" for run in publisher_runs
        )
        motion_publisher_runs = [
            {
                "ordinal": ordinal,
                "callerid": run["callerid"],
                "start_time_s": _relative(run["start"], bag_start),
                "end_time_s": _relative(run["end"], bag_start),
                "duration_s": max(0.0, run["end"] - run["start"]),
                "message_count": run["message_count"],
            }
            for ordinal, run in enumerate(self.motion_cmd_runs, 1)
        ]
        observed_raw_order = [run["callerid"] for run in publisher_runs]
        observed_motion_order = [
            run["callerid"] for run in motion_publisher_runs
        ]
        expected_raw_order = list(EXPECTED_RAW_CMD_VEL_CALLERID_ORDER)
        expected_motion_order = list(EXPECTED_MOTION_CMD_VEL_CALLERID_ORDER)
        raw_order_pass = observed_raw_order == expected_raw_order
        motion_order_pass = observed_motion_order == expected_motion_order

        parking_reverse = self._parking_reverse_result(bag_start, manager)
        manual_stop_true = [
            event for event in self.manual_stop_events if event[1]
        ]
        manual_stop = {
            "topic": MANUAL_STOP_TOPIC,
            "message_count": len(self.manual_stop_events),
            "expected_callerid": EXPECTED_MANUAL_STOP_CALLERID,
            "callerids": sorted(self.manual_stop_callerids),
            "true_event_times_s": [
                _relative(event[0], bag_start) for event in manual_stop_true
            ],
            "pass": bool(
                self.manual_stop_events
                and not manual_stop_true
                and self.manual_stop_callerids
                == {EXPECTED_MANUAL_STOP_CALLERID}
            ),
        }
        pose_integrity = {
            "maximum_allowed_geometric_speed_mps": self.maximum_geometric_speed,
            "maximum_allowed_geometric_angular_speed_radps": (
                self.maximum_geometric_angular_speed
            ),
            "maximum_allowed_pose_step_m": self.maximum_pose_step,
            "maximum_allowed_zero_time_pose_step_m": (
                self.maximum_zero_time_pose_step
            ),
            "maximum_allowed_zero_time_yaw_step_rad": (
                self.maximum_zero_time_yaw_step
            ),
            "maximum_observed_geometric_speed_mps": (
                self.maximum_observed_geometric_speed
            ),
            "maximum_observed_geometric_angular_speed_radps": (
                self.maximum_observed_geometric_angular_speed
            ),
            "maximum_observed_pose_step_m": self.maximum_observed_pose_step,
            "discontinuity_count": len(self.pose_discontinuities),
            "discontinuities": [
                dict(
                    event,
                    time_s=_relative(event["time"], bag_start),
                )
                for event in self.pose_discontinuities
            ],
            "pass": not self.pose_discontinuities,
        }

        for mission_name in MISSION_ORDER:
            values = mission["missions"][mission_name]
            if departure is not None:
                values["active_from_departure_s"] = (
                    None
                    if values["active_time_s"] is None
                    else values["active_time_s"]
                    - _relative(departure, bag_start)
                )
                values["complete_from_departure_s"] = (
                    None
                    if values["complete_time_s"] is None
                    else values["complete_time_s"]
                    - _relative(departure, bag_start)
                )
        total_elapsed = (
            finish_stamp - departure
            if finish_stamp is not None and departure is not None
            else None
        )
        performance = {
            "departure_time_s": _relative(departure, bag_start),
            "finish_time_s": _relative(finish_stamp, bag_start),
            "total_elapsed_s": total_elapsed,
            "departure_to_finish_s": total_elapsed,
            "recording_start_to_finish_s": _relative(finish_stamp, bag_start),
            "mission_elapsed_s": {
                name: mission["missions"][name]["elapsed_s"]
                for name in MISSION_ORDER
            },
        }

        checks = {
            "bag_started_before_departure": predeparture["pass"],
            "mission_manager_current_order": manager["current_order_pass"],
            "mission_manager_active_order": manager["active_order_pass"],
            "mission_manager_sequence_index": manager["sequence_index_pass"],
            "mission_manager_final_complete": manager["final_complete_pass"],
            "mission_manager_publishers": manager["publisher_pass"],
            "mission_gate_evidence": manager["gate_evidence_pass"],
            "all_missions_completed": mission["all_complete"],
            "mission_completion_order": mission["completion_order_pass"],
            "mission_active_complete_links": mission[
                "active_complete_links_pass"
            ],
            "mission_state_publishers": mission["state_publishers_pass"],
            "mission_state_values": mission["state_values_pass"],
            "no_mission_failed": not mission["any_failed"],
            "common_diagnostics_valid": diagnostics_pass,
            "lane_centerline_sustained_rate": lane["pass"],
            "parking_back_out_reverse": parking_reverse["pass"],
            "cmd_vel_callerids_present": callerids_present,
            "cmd_vel_raw_callerid_order": raw_order_pass,
            "cmd_vel_motion_callerid_order": motion_order_pass,
            "manual_stop_not_requested": manual_stop["pass"],
            "pose_continuity": pose_integrity["pass"],
            "finish_oriented_footprint_after_tunnel": finish["pass"],
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        return {
            "schema_version": 2,
            "bag": {
                "path": os.path.abspath(str(bag_path)),
                "start_time": bag_start,
                "end_time": float(bag_end),
                "duration_s": max(0.0, float(bag_end) - bag_start),
                "message_counts": dict(sorted(self.message_counts.items())),
            },
            "predeparture": predeparture,
            "mission_sequence": mission,
            "common_path_diagnostics": diagnostics,
            "lane_centerline_rate": lane,
            "parking_reverse": parking_reverse,
            "cmd_vel": {
                "topic": CMD_VEL_TOPIC,
                "message_count": self.message_counts.get(CMD_VEL_TOPIC, 0),
                "callerids_present": callerids_present,
                "unique_callerids": unique_callerids,
                "publisher_runs": publisher_runs,
                "motion_threshold": self.command_motion_threshold,
                "expected_raw_callerid_order": expected_raw_order,
                "observed_raw_callerid_order": observed_raw_order,
                "raw_callerid_order_pass": raw_order_pass,
                "expected_motion_callerid_order": expected_motion_order,
                "observed_motion_callerid_order": observed_motion_order,
                "motion_callerid_order_pass": motion_order_pass,
                "motion_publisher_runs": motion_publisher_runs,
            },
            "manual_stop": manual_stop,
            "pose_integrity": pose_integrity,
            "finish": finish,
            "performance": performance,
            "validation": {
                "pass": not failed_checks,
                "checks": checks,
                "failed_checks": failed_checks,
            },
        }


def select_pose_topic(available_topics, requested, odom_topic):
    if requested == "gazebo":
        topic = "/gazebo/model_states"
    elif requested == "odom":
        topic = str(odom_topic)
    elif "/gazebo/model_states" in available_topics:
        topic = "/gazebo/model_states"
    else:
        topic = str(odom_topic)
    if topic not in available_topics:
        raise ValueError("pose topic is absent from bag: %s" % topic)
    return topic


def analyze_bag(path, options):
    try:
        import rosbag
    except ImportError as error:
        raise RuntimeError(
            "ROS Noetic rosbag is unavailable; run inside the Noetic environment"
        ) from error

    if not os.path.isfile(path):
        raise ValueError("bag file does not exist: %s" % path)
    accumulator = IntegratedRunAccumulator(
        start_x=options.start_x,
        start_y=options.start_y,
        start_yaw=math.radians(options.start_yaw_deg),
        start_position_tolerance=options.start_position_tolerance,
        start_yaw_tolerance=math.radians(options.start_yaw_tolerance_deg),
        initial_speed_maximum=options.initial_speed_maximum,
        initial_angular_speed_maximum=options.initial_angular_speed_maximum,
        maximum_first_pose_delay=options.maximum_first_pose_delay,
        departure_distance=options.departure_distance,
        departure_speed=options.departure_speed,
        departure_angular_speed=options.departure_angular_speed,
        minimum_predeparture_duration=options.minimum_predeparture_duration,
        gazebo_pose_settling_duration=(
            options.gazebo_pose_settling_duration
        ),
        finish_x=options.finish_x,
        finish_min_y=options.finish_min_y,
        finish_max_y=options.finish_max_y,
        finish_heading=math.radians(options.finish_heading_deg),
        finish_heading_tolerance=math.radians(
            options.finish_heading_tolerance_deg
        ),
        footprint_front=options.footprint_front,
        footprint_rear=options.footprint_rear,
        footprint_half_width=options.footprint_half_width,
        command_motion_threshold=options.command_motion_threshold,
        parking_reverse_threshold=options.parking_reverse_threshold,
        parking_reverse_minimum_samples=(
            options.parking_reverse_minimum_samples
        ),
        parking_reverse_minimum_duration=(
            options.parking_reverse_minimum_duration
        ),
        parking_reverse_maximum_gap=options.parking_reverse_maximum_gap,
        minimum_lane_rate_hz=options.minimum_lane_rate_hz,
        maximum_lane_steady_gap=options.maximum_lane_steady_gap,
        minimum_lane_coverage_fraction=(
            options.minimum_lane_coverage_fraction
        ),
        maximum_lane_source_age=options.maximum_lane_source_age,
        minimum_diagnostic_progress=options.minimum_diagnostic_progress,
        maximum_geometric_speed=options.maximum_geometric_speed,
        maximum_geometric_angular_speed=(
            options.maximum_geometric_angular_speed
        ),
        maximum_pose_step=options.maximum_pose_step,
        maximum_zero_time_pose_step=options.maximum_zero_time_pose_step,
        maximum_zero_time_yaw_step=math.radians(
            options.maximum_zero_time_yaw_step_deg
        ),
    )
    with rosbag.Bag(path, "r") as bag:
        if bag.get_message_count() <= 0:
            raise ValueError("bag has no messages")
        bag_start = float(bag.get_start_time())
        bag_end = float(bag.get_end_time())
        available = set(bag.get_type_and_topic_info().topics)
        pose_topic = select_pose_topic(
            available, options.pose_source, options.odom_topic
        )
        topics = set(MISSION_STATE_TOPICS.values())
        topics.update(COMMON_DIAGNOSTIC_TOPICS.values())
        topics.update(MANAGER_TOPICS)
        topics.update(MISSION_GATE_TOPICS.values())
        topics.update(
            (
                SEQUENCE_INDEX_TOPIC,
                LANE_CENTERLINE_TOPIC,
                CMD_VEL_TOPIC,
                MANUAL_STOP_TOPIC,
                pose_topic,
            )
        )
        model_index = None
        state_by_topic = {
            topic: mission for mission, topic in MISSION_STATE_TOPICS.items()
        }
        diagnostic_by_topic = {
            topic: mission
            for mission, topic in COMMON_DIAGNOSTIC_TOPICS.items()
        }
        gate_by_topic = {
            topic: mission for mission, topic in MISSION_GATE_TOPICS.items()
        }
        for record in bag.read_messages(
            topics=sorted(topics), return_connection_header=True
        ):
            topic = record.topic
            message = record.message
            stamp = float(record.timestamp.to_sec())
            callerid = callerid_from_header(record.connection_header)
            accumulator.count_topic(topic)
            if topic in state_by_topic:
                accumulator.observe_state(
                    state_by_topic[topic], message.data, stamp, callerid
                )
            elif topic in MANAGER_TOPICS:
                accumulator.observe_manager_state(
                    topic, message.data, stamp, callerid
                )
            elif topic == SEQUENCE_INDEX_TOPIC:
                accumulator.observe_sequence_index(
                    message.data, stamp, callerid
                )
            elif topic in gate_by_topic:
                accumulator.observe_gate(
                    gate_by_topic[topic], message.data, stamp, callerid
                )
            elif topic in diagnostic_by_topic:
                accumulator.observe_diagnostics(
                    diagnostic_by_topic[topic], message.data, stamp, callerid
                )
            elif topic == LANE_CENTERLINE_TOPIC:
                accumulator.observe_lane_centerline(
                    stamp, message.header.stamp.to_sec(), callerid
                )
            elif topic == CMD_VEL_TOPIC:
                accumulator.observe_cmd_vel(
                    stamp,
                    message.linear.x,
                    message.angular.z,
                    callerid,
                )
            elif topic == MANUAL_STOP_TOPIC:
                accumulator.observe_manual_stop(
                    stamp, message.data, callerid
                )
            elif topic == "/gazebo/model_states":
                pose, model_index = extract_model_pose(
                    message, options.model_name, model_index
                )
                if pose is not None:
                    accumulator.observe_pose(stamp, *pose)
            elif topic == pose_topic:
                pose = extract_odom_pose(message)
                if pose is not None:
                    accumulator.observe_pose(stamp, *pose)
    return accumulator.result(
        bag_start,
        bag_end,
        path,
        "gazebo_model_states" if pose_topic == "/gazebo/model_states" else pose_topic,
    )


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", help="completed ROS1 .bag file")
    parser.add_argument(
        "--output",
        default="-",
        help="JSON output path; '-' writes to stdout (default)",
    )
    parser.add_argument("--pretty", action="store_true", help="indent JSON")
    parser.add_argument(
        "--no-fail-on-validation",
        action="store_true",
        help="return zero even when the JSON validation result is false",
    )
    parser.add_argument(
        "--pose-source", choices=("auto", "gazebo", "odom"), default="auto"
    )
    parser.add_argument("--model-name", default="custom_autorace")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--start-x", type=float, default=0.8)
    parser.add_argument("--start-y", type=float, default=-1.747)
    parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    parser.add_argument("--start-position-tolerance", type=float, default=0.08)
    parser.add_argument("--start-yaw-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--initial-speed-maximum", type=float, default=0.02)
    parser.add_argument(
        "--initial-angular-speed-maximum", type=float, default=0.05
    )
    parser.add_argument("--maximum-first-pose-delay", type=float, default=0.50)
    parser.add_argument("--departure-distance", type=float, default=0.03)
    parser.add_argument("--departure-speed", type=float, default=0.03)
    parser.add_argument("--departure-angular-speed", type=float, default=0.10)
    parser.add_argument("--minimum-predeparture-duration", type=float, default=0.20)
    parser.add_argument(
        "--gazebo-pose-settling-duration", type=float, default=0.10
    )
    # Nominal base-footprint threshold at the expected heading.  The analyzer
    # derives the physical paint edge from this and then checks oriented
    # corners, so the historical 1.033458 m value remains comparable.
    parser.add_argument("--finish-x", type=float, default=1.033458)
    parser.add_argument("--finish-min-y", type=float, default=-1.86)
    parser.add_argument("--finish-max-y", type=float, default=-1.64)
    parser.add_argument("--finish-heading-deg", type=float, default=0.0)
    parser.add_argument(
        "--finish-heading-tolerance-deg", type=float, default=15.0
    )
    parser.add_argument("--footprint-front", type=float, default=0.067645)
    parser.add_argument("--footprint-rear", type=float, default=0.118073)
    parser.add_argument("--footprint-half-width", type=float, default=0.0903)
    parser.add_argument("--command-motion-threshold", type=float, default=0.005)
    parser.add_argument("--parking-reverse-threshold", type=float, default=-0.001)
    parser.add_argument(
        "--parking-reverse-minimum-samples", type=int, default=3
    )
    parser.add_argument(
        "--parking-reverse-minimum-duration", type=float, default=0.08
    )
    parser.add_argument(
        "--parking-reverse-maximum-gap", type=float, default=0.20
    )
    parser.add_argument("--minimum-lane-rate-hz", type=float, default=29.0)
    parser.add_argument("--maximum-lane-steady-gap", type=float, default=0.20)
    parser.add_argument(
        "--minimum-lane-coverage-fraction", type=float, default=0.98
    )
    parser.add_argument("--maximum-lane-source-age", type=float, default=0.25)
    parser.add_argument("--minimum-diagnostic-progress", type=float, default=0.80)
    parser.add_argument("--maximum-geometric-speed", type=float, default=0.75)
    parser.add_argument(
        "--maximum-geometric-angular-speed", type=float, default=4.0
    )
    parser.add_argument("--maximum-pose-step", type=float, default=0.35)
    parser.add_argument(
        "--maximum-zero-time-pose-step", type=float, default=0.005
    )
    parser.add_argument(
        "--maximum-zero-time-yaw-step-deg", type=float, default=3.0
    )
    return parser


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main(argv=None):
    parser = build_argument_parser()
    options = parser.parse_args(argv)
    try:
        result = analyze_bag(options.bag, options)
    except Exception as error:
        failure = {
            "schema_version": 2,
            "bag": {"path": os.path.abspath(options.bag)},
            "validation": {
                "pass": False,
                "checks": {},
                "failed_checks": ["analysis_error"],
            },
            "error": str(error),
        }
        encoded = json.dumps(
            _json_safe(failure),
            indent=2 if options.pretty else None,
            sort_keys=True,
            allow_nan=False,
        )
        stream = sys.stdout if options.output == "-" else open(
            options.output, "w", encoding="utf-8"
        )
        try:
            stream.write(encoded + "\n")
        finally:
            if stream is not sys.stdout:
                stream.close()
        return 1

    encoded = json.dumps(
        _json_safe(result),
        indent=2 if options.pretty else None,
        sort_keys=True,
        allow_nan=False,
    )
    stream = sys.stdout if options.output == "-" else open(
        options.output, "w", encoding="utf-8"
    )
    try:
        stream.write(encoded + "\n")
    finally:
        if stream is not sys.stdout:
            stream.close()
    if result["validation"]["pass"] or options.no_fail_on_validation:
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
