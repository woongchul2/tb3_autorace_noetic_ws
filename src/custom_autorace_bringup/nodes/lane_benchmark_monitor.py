#!/usr/bin/env python3
"""Run one repeatable Gazebo lane-controller trial and emit JSON metrics."""

from collections import deque
import json
import math
import os
import threading
import time

import cv2
import numpy as np
import rospkg
import rosgraph
import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import SetBool

from custom_autorace_bringup.lane_lookahead_geometry import (
    resolve_lane_lookahead_geometry,
    texture_path_from_config,
    validate_texture_revision,
)
from custom_autorace_bringup.raster_route import RasterRouteCorridorChecker
from custom_autorace_bringup.path_following import PathDiagnostics
from custom_autorace_bringup.zigzag_path import (
    nearest_path_index,
    normalize_angle,
)


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2),
    )


def percentile(values, fraction):
    if len(values) == 0:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction))


def validate_finish_crossing(
    path,
    x,
    y,
    yaw,
    previous_path_index,
    minimum_station,
    target_heading,
    heading_tolerance,
):
    """Check that a geometric finish-plane crossing is on the intended return.

    The same x/y plane can be reached by an unsafe chord through the two bends.
    Use the nearest point on the ordered oracle path to distinguish that early
    crossing, then require the configured return heading as a second, independent
    condition.
    """

    nearest_index = nearest_path_index(
        path,
        float(x),
        float(y),
        int(previous_path_index),
        search_back=5,
        search_ahead_distance=0.35,
    )
    station = float(path.station[nearest_index])
    heading_error = abs(normalize_angle(float(yaw) - float(target_heading)))
    valid = bool(
        station >= float(minimum_station)
        and heading_error <= float(heading_tolerance)
    )
    return valid, nearest_index, station, heading_error


def parse_lane_path_diagnostic(values):
    """Validate the public common ``PathDiagnostics`` vector.

    The fields are progress, remaining distance, position error, cross-track
    error, heading error, target speed, target index, curvature, minimum line,
    obstacle and map clearance, and commanded linear and angular velocity.
    Positive infinity is meaningful only when no obstacle or map boundary is
    present.  Every other non-finite value means the diagnostic is unusable.
    """

    try:
        diagnostics = PathDiagnostics.from_array(values)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(diagnostics.minimum_line_clearance):
        return None
    return tuple(diagnostics.as_array())


def command_dynamics_metrics(samples, duration, sign_deadband):
    """Return timing-aware steering-vibration metrics for Twist samples."""

    if not samples:
        return {}
    angular = np.asarray([sample[2] for sample in samples], dtype=np.float64)
    accelerations = []
    for previous, current in zip(samples, samples[1:]):
        elapsed = float(current[0] - previous[0])
        if elapsed > 1e-4:
            accelerations.append(abs(float(current[2] - previous[2])) / elapsed)

    signs = []
    for value in angular:
        if abs(value) < sign_deadband:
            continue
        signs.append(1 if value > 0.0 else -1)
    sign_changes = sum(
        previous != current for previous, current in zip(signs, signs[1:])
    )
    metrics = {
        "command_angular_rms": float(math.sqrt(np.mean(angular ** 2))),
        "command_angular_abs_p95": percentile(np.abs(angular), 95),
        "command_angular_abs_maximum": float(np.max(np.abs(angular))),
        "command_angular_sign_change_count": int(sign_changes),
        "command_angular_sign_changes_per_second": (
            float(sign_changes) / duration if duration > 0.0 else math.nan
        ),
        "command_angular_sample_count": int(angular.size),
        "command_angular_sign_deadband": float(sign_deadband),
    }
    if accelerations:
        metrics.update(
            {
                "command_angular_acceleration_abs_p95": percentile(
                    accelerations, 95
                ),
                "command_angular_acceleration_abs_maximum": max(accelerations),
                "command_angular_acceleration_sample_count": len(accelerations),
            }
        )
    return metrics


def lane_path_metrics(samples):
    """Summarize common path diagnostics gathered between benchmark gates."""

    if not samples:
        return {}
    progress = [sample[0] for sample in samples]
    remaining_distance = [sample[1] for sample in samples]
    position_error = [abs(sample[2]) for sample in samples]
    cross_track_error = [abs(sample[3]) for sample in samples]
    heading_error = [abs(sample[4]) for sample in samples]
    target_speed = [sample[5] for sample in samples]
    target_index = [int(round(sample[6])) for sample in samples]
    curvature = [abs(sample[7]) for sample in samples]
    line_clearance = [sample[8] for sample in samples]
    obstacle_clearance = [sample[9] for sample in samples]
    map_clearance = [sample[10] for sample in samples]
    commanded_linear = [sample[11] for sample in samples]
    commanded_angular = [abs(sample[12]) for sample in samples]
    return {
        "lane_path_diagnostic_count": len(samples),
        "lane_path_progress_mean": float(np.mean(progress)),
        "lane_path_progress_minimum": min(progress),
        "lane_path_progress_maximum": max(progress),
        "lane_path_progress_final": progress[-1],
        "lane_path_remaining_distance_mean": float(
            np.mean(remaining_distance)
        ),
        "lane_path_remaining_distance_minimum": min(remaining_distance),
        "lane_path_remaining_distance_maximum": max(remaining_distance),
        "lane_path_remaining_distance_final": remaining_distance[-1],
        "lane_path_position_error_abs_p95": percentile(position_error, 95),
        "lane_path_position_error_abs_maximum": max(position_error),
        "lane_path_cross_track_error_abs_p95": percentile(
            cross_track_error, 95
        ),
        "lane_path_cross_track_error_abs_maximum": max(cross_track_error),
        "lane_path_heading_error_abs_p95_deg": math.degrees(
            percentile(heading_error, 95)
        ),
        "lane_path_heading_error_abs_maximum_deg": math.degrees(
            max(heading_error)
        ),
        "lane_path_target_speed_mean": float(np.mean(target_speed)),
        "lane_path_target_speed_minimum": min(target_speed),
        "lane_path_target_speed_maximum": max(target_speed),
        "lane_path_target_index_minimum": min(target_index),
        "lane_path_target_index_maximum": max(target_index),
        "lane_path_target_index_final": target_index[-1],
        "lane_path_curvature_abs_p95": percentile(curvature, 95),
        "lane_path_curvature_abs_maximum": max(curvature),
        "lane_path_minimum_line_clearance": min(line_clearance),
        "lane_path_minimum_obstacle_clearance": min(obstacle_clearance),
        "lane_path_minimum_map_clearance": min(map_clearance),
        "lane_path_commanded_linear_mean": float(np.mean(commanded_linear)),
        "lane_path_commanded_linear_minimum": min(commanded_linear),
        "lane_path_commanded_linear_maximum": max(commanded_linear),
        "lane_path_commanded_angular_abs_p95": percentile(
            commanded_angular, 95
        ),
        "lane_path_commanded_angular_abs_maximum": max(commanded_angular),
    }


def write_result_file(result_file, encoded):
    """Atomically replace one benchmark result, including startup failures."""

    if not result_file:
        return
    temporary = result_file + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
    os.replace(temporary, result_file)


class LaneBenchmarkMonitor:
    def __init__(self):
        get = rospy.get_param
        self.controller = str(get("~controller", "unknown"))
        self.trial_id = str(get("~trial_id", "single"))
        self.result_file = str(get("~result_file", "")).strip()
        self._write_result_file(
            json.dumps(
                {
                    "controller": self.controller,
                    "trial_id": self.trial_id,
                    "success": False,
                    "reason": "benchmark did not complete",
                    "status": "RUNNING",
                },
                sort_keys=True,
            )
        )
        self.model_name = str(get("~model_name", "custom_autorace"))
        self.expected_x = float(get("~expected_start/x", 0.8))
        self.expected_y = float(get("~expected_start/y", -1.747))
        self.expected_yaw = float(get("~expected_start/yaw", 0.0))
        self.startup_timeout = max(5.0, float(get("~startup_timeout", 60.0)))
        self.run_timeout = max(5.0, float(get("~run_timeout", 30.0)))
        self.enable_service = str(
            get("~enable_service", "/control/lane_following")
        )
        self.path_diagnostics_topic = str(
            get(
                "~path_diagnostics_topic",
                "/control/lane_path_diagnostics",
            )
        )
        requested_cruise_velocity = float(
            get("~requested_cruise_velocity", math.nan)
        )
        self.requested_cruise_velocity = (
            requested_cruise_velocity
            if math.isfinite(requested_cruise_velocity)
            else None
        )
        self.angular_sign_deadband = max(
            0.0, float(get("~angular_sign_deadband", 0.02))
        )
        config = get("~lane_lookahead", {})
        try:
            self.geometry = resolve_lane_lookahead_geometry(config)
        except ValueError as error:
            raise rospy.ROSInitException(str(error))
        config = self.geometry.config
        self.path = self.geometry.path
        finish = config["finish"]
        self.start_gate_x = float(get("~gates/start_x", 0.9028))
        self.start_gate_min_y = float(get("~gates/start_min_y", -1.86))
        self.start_gate_max_y = float(get("~gates/start_max_y", -1.64))
        self.finish_gate_x = float(get("~gates/finish_x", finish["x"]))
        self.finish_gate_min_y = float(
            get("~gates/finish_min_y", finish["min_y"])
        )
        self.finish_gate_max_y = float(
            get("~gates/finish_max_y", finish["max_y"])
        )
        self.finish_minimum_station = float(finish["minimum_station"])
        self.finish_heading = math.radians(float(finish["heading_deg"]))
        self.finish_heading_tolerance = math.radians(
            float(finish["heading_tolerance_deg"])
        )
        footprint = config["footprint"]
        safety = config["safety"]
        self.maximum_inner_intrusion = float(
            safety["maximum_inner_intrusion"]
        )
        self.minimum_outer_reserve = float(
            safety["minimum_outer_reserve"]
        )
        texture = config["texture"]
        try:
            self.texture_path = texture_path_from_config(
                texture, rospkg.RosPack().get_path
            )
            self.texture_sha256 = validate_texture_revision(
                self.texture_path, texture, self.geometry
            )
        except ValueError as error:
            raise rospy.ROSInitException(str(error))
        image = cv2.imread(self.texture_path, cv2.IMREAD_COLOR)
        if image is None:
            raise rospy.ROSInitException(
                "could not read benchmark texture %s" % self.texture_path
            )
        self.checker = RasterRouteCorridorChecker(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            texture["course_size"],
            texture["course_yaw"],
            self.path,
            footprint["front"],
            footprint["rear"],
            footprint["half_width"],
            texture["expected_boundary_offset"],
            texture["line_search_half_width"],
            texture["boundary_step"],
            footprint["validation_sample_spacing"],
            texture["color_threshold"],
            texture["color_tolerance"],
            texture["yellow_blue_maximum"],
            texture["projection_search_distance"],
            texture["endpoint_extension"],
        )

        self.lock = threading.RLock()
        self.latest_model = None
        self.model_generation = 0
        self.odom_received = None
        self.command_history = deque(maxlen=2000)
        self.path_diagnostic_history = deque(maxlen=2000)

        rospy.Subscriber(
            "/gazebo/model_states", ModelStates, self.model_callback, queue_size=1
        )
        rospy.Subscriber("/odom", rospy.AnyMsg, self.odom_callback, queue_size=1)
        rospy.Subscriber("/cmd_vel", Twist, self.command_callback, queue_size=1)
        rospy.Subscriber(
            self.path_diagnostics_topic,
            Float64MultiArray,
            self.path_diagnostics_callback,
            queue_size=1,
        )
        self.result_pub = rospy.Publisher(
            "/lane_benchmark/result", String, queue_size=1, latch=True
        )
        rospy.loginfo(
            "Lane benchmark geometry: fixed_anchor=start, straight=%.3fm "
            "(delta=%+.3fm), path_length=%.3fm, finish=(x=%.3f, "
            "y=%.3f..%.3f)",
            self.geometry.straight_length,
            self.geometry.length_delta,
            self.path.length,
            self.finish_gate_x,
            self.finish_gate_min_y,
            self.finish_gate_max_y,
        )

    def model_callback(self, message):
        try:
            index = message.name.index(self.model_name)
        except ValueError:
            return
        pose = message.pose[index]
        twist = message.twist[index]
        sample = (
            rospy.Time.now().to_sec(),
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(pose.orientation),
            math.hypot(float(twist.linear.x), float(twist.linear.y)),
            float(twist.linear.x),
            float(twist.angular.z),
        )
        if not all(math.isfinite(value) for value in sample):
            return
        with self.lock:
            self.latest_model = sample
            self.model_generation += 1

    def odom_callback(self, _message):
        with self.lock:
            self.odom_received = rospy.Time.now().to_sec()

    def command_callback(self, message):
        sample = (
            rospy.Time.now().to_sec(),
            float(message.linear.x),
            float(message.angular.z),
        )
        if all(math.isfinite(value) for value in sample):
            with self.lock:
                self.command_history.append(sample)

    def path_diagnostics_callback(self, message):
        parsed = parse_lane_path_diagnostic(list(message.data))
        if parsed is None:
            return
        with self.lock:
            self.path_diagnostic_history.append(
                (rospy.Time.now().to_sec(), parsed)
            )

    def _publisher_count(self, topic):
        try:
            publishers, _, _ = rosgraph.Master(rospy.get_name()).getSystemState()
        except Exception:
            return -1
        for name, nodes in publishers:
            if name == topic:
                return len(nodes)
        return 0

    def _ready(self, now):
        with self.lock:
            model = self.latest_model
            odom_received = self.odom_received
            path_diagnostics = [
                sample
                for stamp, sample in self.path_diagnostic_history
                if now - stamp <= 1.0
            ]
        if model is None or now - model[0] > 0.20:
            return False, "ground truth"
        _, x, y, yaw, speed, _, angular = model
        if (
            math.hypot(x - self.expected_x, y - self.expected_y) > 0.005
            or abs(normalize_angle(yaw - self.expected_yaw)) > math.radians(1.0)
        ):
            return False, "spawn pose"
        if speed > 0.005 or abs(angular) > 0.05:
            return False, "stationary robot"
        if odom_received is None or now - odom_received > 0.20:
            return False, "odometry"
        if len(path_diagnostics) < 5:
            return False, "five common-path diagnostic frames"
        recent = path_diagnostics[-5:]
        if not all(
            0.0 <= sample[0] <= 1.0
            and sample[1] > 0.0
            and sample[6] >= 0.0
            for sample in recent
        ):
            return False, "valid common lane path"
        return True, "ready"

    @staticmethod
    def _interpolate_sample(previous, current, coordinate, value):
        before = previous[coordinate]
        after = current[coordinate]
        fraction = (value - before) / (after - before)
        sample = [
            previous[index] + fraction * (current[index] - previous[index])
            for index in range(len(previous))
        ]
        sample[3] = normalize_angle(
            previous[3]
            + fraction * normalize_angle(current[3] - previous[3])
        )
        sample[coordinate] = value
        return tuple(sample)

    def run(self):
        # A wall-clock deadline is intentional here: simulated time can remain
        # at zero when Gazebo itself failed to start.
        started_wall = time.monotonic()
        stable_since = None
        last_problem = "ROS startup"
        poll_period = 0.02
        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            ready, problem = self._ready(now)
            last_problem = problem
            if ready:
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= 0.50:
                    break
            else:
                stable_since = None
            if time.monotonic() - started_wall > self.startup_timeout:
                return self._finish(False, "startup timeout: %s" % last_problem, {})
            time.sleep(poll_period)

        if self._publisher_count("/cmd_vel") != 1:
            return self._finish(False, "cmd_vel does not have exactly one publisher", {})
        try:
            rospy.wait_for_service(self.enable_service, timeout=5.0)
            enable = rospy.ServiceProxy(self.enable_service, SetBool)
            response = enable(True)
        except Exception as error:
            return self._finish(False, "enable service failed: %s" % error, {})
        if not response.success:
            return self._finish(False, "controller rejected enable", {})

        release_time = rospy.Time.now().to_sec()
        previous = None
        start_sample = None
        processed_generation = -1
        start_time = None
        finish_time = None
        measurement_end_time = None
        start_pose = None
        final_pose = None
        rejected_finish = None
        distance = 0.0
        speeds = []
        lateral_accelerations = []
        path_errors = []
        heading_errors = []
        maximum_inner = 0.0
        minimum_outer = math.inf
        path_index = 0
        failure_reason = ""
        run_started_wall = time.monotonic()
        next_publisher_check = run_started_wall + 1.0

        while not rospy.is_shutdown():
            with self.lock:
                current = self.latest_model
                generation = self.model_generation
            wall_now = time.monotonic()
            if wall_now >= next_publisher_check:
                if self._publisher_count("/cmd_vel") != 1:
                    failure_reason = (
                        "cmd_vel did not retain exactly one publisher"
                    )
                    break
                next_publisher_check = wall_now + 1.0
            if wall_now - run_started_wall > max(
                60.0, 4.0 * self.run_timeout
            ):
                failure_reason = "run wall-clock timeout"
                break
            if current is None or generation == processed_generation:
                time.sleep(poll_period)
                continue
            processed_generation = generation

            if previous is not None and start_time is None:
                if (
                    previous[1] < self.start_gate_x <= current[1]
                ):
                    crossing = self._interpolate_sample(
                        previous, current, 1, self.start_gate_x
                    )
                    if self.start_gate_min_y <= crossing[2] <= self.start_gate_max_y:
                        start_sample = crossing
                        start_time = crossing[0]
                        start_pose = crossing

            if start_time is not None:
                finish_sample = None
                if (
                    previous is not None
                    and previous[1] > self.finish_gate_x >= current[1]
                ):
                    crossing = self._interpolate_sample(
                        previous, current, 1, self.finish_gate_x
                    )
                    if self.finish_gate_min_y <= crossing[2] <= self.finish_gate_max_y:
                        (
                            crossing_valid,
                            _,
                            crossing_station,
                            crossing_heading_error,
                        ) = validate_finish_crossing(
                            self.path,
                            crossing[1],
                            crossing[2],
                            crossing[3],
                            path_index,
                            self.finish_minimum_station,
                            self.finish_heading,
                            self.finish_heading_tolerance,
                        )
                        if crossing_valid:
                            finish_sample = crossing
                        else:
                            if rejected_finish is None:
                                rejected_finish = (
                                    crossing[0],
                                    crossing[1],
                                    crossing[2],
                                    crossing[3],
                                    crossing_station,
                                    crossing_heading_error,
                                )
                            failure_reason = (
                                "finish crossing rejected: station %.3f < %.3f "
                                "or heading error %.1f deg > %.1f deg"
                                % (
                                    crossing_station,
                                    self.finish_minimum_station,
                                    math.degrees(crossing_heading_error),
                                    math.degrees(self.finish_heading_tolerance),
                                )
                            )

                segment_start = (
                    start_sample
                    if previous is not None and previous[0] < start_time
                    else previous
                )
                segment_end = finish_sample if finish_sample is not None else current
                if segment_start is not None:
                    distance += math.hypot(
                        segment_end[1] - segment_start[1],
                        segment_end[2] - segment_start[2],
                    )
                    inner, outer = self.checker.segment_metrics(
                        (segment_start[1], segment_start[2], segment_start[3]),
                        (segment_end[1], segment_end[2], segment_end[3]),
                    )
                    maximum_inner = max(maximum_inner, inner)
                    minimum_outer = min(minimum_outer, outer)
                measurement = segment_end
                measurement_end_time = measurement[0]
                final_pose = measurement
                speeds.append(measurement[4])
                lateral_accelerations.append(
                    measurement[4] * abs(measurement[6])
                )
                path_index = nearest_path_index(
                    self.path,
                    measurement[1],
                    measurement[2],
                    path_index,
                    search_back=5,
                    search_ahead_distance=0.35,
                )
                path_errors.append(
                    math.hypot(
                        float(self.path.x[path_index]) - measurement[1],
                        float(self.path.y[path_index]) - measurement[2],
                    )
                )
                heading_errors.append(
                    abs(
                        normalize_angle(
                            float(self.path.heading[path_index]) - measurement[3]
                        )
                    )
                )
                if finish_sample is not None:
                    finish_time = finish_sample[0]
                    break

            if current[0] - release_time > self.run_timeout:
                if not failure_reason:
                    failure_reason = "run timeout"
                break
            previous = current
            time.sleep(poll_period)

        try:
            enable(False)
        except Exception:
            pass
        if start_time is None or measurement_end_time is None:
            return self._finish(False, failure_reason or "gate was not crossed", {})
        evaluation_end_time = (
            finish_time if finish_time is not None else measurement_end_time
        )

        command_values = []
        path_diagnostics = []
        with self.lock:
            for stamp, linear, angular in self.command_history:
                if start_time <= stamp <= evaluation_end_time:
                    command_values.append((stamp, linear, angular))
            for stamp, diagnostic in self.path_diagnostic_history:
                if start_time <= stamp <= evaluation_end_time:
                    path_diagnostics.append(diagnostic)
        command_linear = [value[1] for value in command_values]
        if not (
            speeds
            and path_errors
            and heading_errors
            and command_linear
            and math.isfinite(minimum_outer)
        ):
            return self._finish(False, "benchmark samples were incomplete", {})
        if not path_diagnostics:
            return self._finish(
                False, "lane-path diagnostic samples were incomplete", {}
            )
        metrics = {
            "controller": self.controller,
            "evaluation_duration": evaluation_end_time - start_time,
            "release_to_evaluation_end": evaluation_end_time - release_time,
            "finish_crossed": finish_time is not None,
            "distance": distance,
            "mean_ground_truth_speed": float(np.mean(speeds)),
            "ground_truth_speed_p50": percentile(speeds, 50),
            "ground_truth_speed_p95": percentile(speeds, 95),
            "maximum_ground_truth_speed": max(speeds),
            "fraction_speed_at_least_0_19": float(
                np.mean(np.asarray(speeds) >= 0.19)
            ),
            "fraction_speed_at_least_0_21": float(
                np.mean(np.asarray(speeds) >= 0.21)
            ),
            "command_linear_mean": float(np.mean(command_linear)),
            "command_linear_maximum": max(command_linear),
            "path_error_p95": percentile(path_errors, 95),
            "path_error_maximum": max(path_errors),
            "heading_error_p95_deg": math.degrees(
                percentile(heading_errors, 95)
            ),
            "heading_error_maximum_deg": math.degrees(max(heading_errors)),
            "lateral_acceleration_p95": percentile(lateral_accelerations, 95),
            "lateral_acceleration_maximum": max(lateral_accelerations),
            "maximum_inner_intrusion": maximum_inner,
            "minimum_outer_reserve": minimum_outer,
            "sample_count": len(speeds),
            "start_pose_x": start_pose[1],
            "start_pose_y": start_pose[2],
            "final_pose_x": final_pose[1],
            "final_pose_y": final_pose[2],
            "final_pose_yaw_deg": math.degrees(final_pose[3]),
        }
        if finish_time is not None:
            metrics["gate_time"] = finish_time - start_time
            metrics["release_to_finish"] = finish_time - release_time
        if rejected_finish is not None:
            metrics.update(
                {
                    "first_rejected_finish_time": (
                        rejected_finish[0] - start_time
                    ),
                    "first_rejected_finish_x": rejected_finish[1],
                    "first_rejected_finish_y": rejected_finish[2],
                    "first_rejected_finish_yaw_deg": math.degrees(
                        rejected_finish[3]
                    ),
                    "first_rejected_finish_station": rejected_finish[4],
                    "first_rejected_finish_heading_error_deg": math.degrees(
                        rejected_finish[5]
                    ),
                }
            )
        metrics.update(
            command_dynamics_metrics(
                command_values,
                evaluation_end_time - start_time,
                self.angular_sign_deadband,
            )
        )
        metrics.update(lane_path_metrics(path_diagnostics))
        safety_pass = bool(
            maximum_inner <= self.maximum_inner_intrusion
            and minimum_outer >= self.minimum_outer_reserve
        )
        metrics["safety_pass"] = safety_pass
        if finish_time is None:
            return self._finish(
                False,
                failure_reason or "finish gate was not crossed",
                metrics,
            )
        return self._finish(
            safety_pass,
            "complete" if safety_pass else "paint limit exceeded",
            metrics,
        )

    def _finish(self, success, reason, metrics):
        result = dict(metrics)
        result.setdefault("controller", self.controller)
        result.setdefault("fixed_anchor", "start")
        result.setdefault("reference_straight_length", self.geometry.reference_straight_length)
        result.setdefault("straight_length", self.geometry.straight_length)
        result.setdefault("length_delta", self.geometry.length_delta)
        result.setdefault("path_length", self.path.length)
        result.setdefault("finish_gate_x", self.finish_gate_x)
        result.setdefault("finish_gate_min_y", self.finish_gate_min_y)
        result.setdefault("finish_gate_max_y", self.finish_gate_max_y)
        result.setdefault(
            "requested_cruise_velocity", self.requested_cruise_velocity
        )
        result.setdefault("course_texture", self.texture_path)
        result.setdefault("course_texture_sha256", self.texture_sha256)
        result["trial_id"] = self.trial_id
        result["success"] = bool(success)
        result["reason"] = str(reason)
        result["status"] = "COMPLETE"
        encoded = json.dumps(result, sort_keys=True)
        self.result_pub.publish(String(data=encoded))
        rospy.loginfo("LANE_BENCHMARK_RESULT %s", encoded)
        self._write_result_file(encoded)
        time.sleep(0.25)
        rospy.signal_shutdown("lane benchmark finished")
        return success

    def _write_result_file(self, encoded):
        try:
            write_result_file(self.result_file, encoded)
        except OSError as error:
            rospy.logerr("Could not write benchmark result: %s", error)


def report_initialization_failure(error):
    """Turn constructor failures into a terminal machine-readable result."""

    controller = str(rospy.get_param("~controller", "unknown"))
    trial_id = str(rospy.get_param("~trial_id", "single"))
    result_file = str(rospy.get_param("~result_file", "")).strip()
    result = {
        "controller": controller,
        "trial_id": trial_id,
        "success": False,
        "reason": "initialization failed: %s" % error,
        "status": "COMPLETE",
    }
    encoded = json.dumps(result, sort_keys=True)
    rospy.logerr("LANE_BENCHMARK_RESULT %s", encoded)
    try:
        write_result_file(result_file, encoded)
    except OSError as write_error:
        rospy.logerr("Could not write benchmark result: %s", write_error)


if __name__ == "__main__":
    rospy.init_node("lane_benchmark_monitor")
    try:
        monitor = LaneBenchmarkMonitor()
    except Exception as error:
        report_initialization_failure(error)
        raise SystemExit(1)
    monitor.run()
