#!/usr/bin/env python3
"""Plan and track a LiDAR-updated Hybrid A* path through the tunnel."""

from collections import deque
import math
import threading
import time

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid as OccupancyGridMessage, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64, Float64MultiArray, Header, String
from std_srvs.srv import SetBool

from custom_autorace_bringup.msg import TrafficSign
from custom_autorace_bringup.parking_geometry import (
    map_from_odom_transform,
    odom_pose_to_map,
    quintic_pose_path,
)
from custom_autorace_bringup.tunnel_costmap import TunnelCostmap
from custom_autorace_bringup.tunnel_registration import (
    PortalRegistrationConfig,
    detect_portal,
)
from custom_autorace_bringup.tunnel_planner import (
    HybridAStarPlanner,
    OccupancyGrid,
    Pose2D,
    RectangularFootprint,
)
from custom_autorace_bringup.path_following import (
    CommonPath,
    PathDiagnostics,
    path_from_poses,
)
from custom_autorace_bringup.zigzag_path import (
    build_speed_profile,
    calculate_tracking,
    clamp,
    committed_route_path,
    limit_tracking_command,
    nearest_path_index,
    normalize_angle,
)


def yaw_from_quaternion(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2),
    )


def quaternion_from_yaw(yaw):
    half = 0.5 * float(yaw)
    return math.sin(half), math.cos(half)


def propagate_twist(pose, linear_velocity, angular_velocity, duration):
    """Integrate one constant differential-drive command exactly."""
    linear_velocity = float(linear_velocity)
    angular_velocity = float(angular_velocity)
    duration = max(0.0, float(duration))
    yaw_delta = angular_velocity * duration
    if abs(angular_velocity) <= 1e-9:
        distance = linear_velocity * duration
        return Pose2D(
            pose.x + distance * math.cos(pose.yaw),
            pose.y + distance * math.sin(pose.yaw),
            pose.yaw,
        )
    radius = linear_velocity / angular_velocity
    end_yaw = pose.yaw + yaw_delta
    return Pose2D(
        pose.x + radius * (math.sin(end_yaw) - math.sin(pose.yaw)),
        pose.y - radius * (math.cos(end_yaw) - math.cos(pose.yaw)),
        normalize_angle(end_yaw),
    )


def pose_from_degrees(values, name):
    """Read one finite ``[x, y, yaw_deg]`` surveyed pose."""
    try:
        values = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("%s must contain [x, y, yaw_deg]" % name) from error
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("%s must contain three finite values" % name)
    return Pose2D(values[0], values[1], math.radians(values[2]))


def portal_alignment_path(
    start,
    target,
    tangent_ratio,
    minimum_tangent_length,
    maximum_tangent_length,
    sample_step,
    target_speed,
    frame_id,
    label="tunnel_entry_alignment",
):
    """Build a smooth zero-end-curvature path to a portal staging pose."""
    distance = math.hypot(target.x - start.x, target.y - start.y)
    if distance <= 1e-6:
        raise ValueError("portal alignment path is degenerate")
    # Keep the first/last control triplets ordered even when a tunnel-only
    # regression begins just centimetres before the staging pose.
    geometric_maximum = 0.24 * distance
    tangent_length = clamp(
        tangent_ratio * distance,
        min(minimum_tangent_length, geometric_maximum),
        min(maximum_tangent_length, geometric_maximum),
    )
    sample_count = max(3, int(math.ceil(distance / sample_step)) + 1)
    poses = quintic_pose_path(
        (start.x, start.y, start.yaw),
        (target.x, target.y, target.yaw),
        tangent_length,
        tangent_length,
        sample_count,
    )
    return path_from_poses(
        poses,
        frame_id=frame_id,
        target_speed=target_speed,
        label=label,
    )


def portal_straight_path(
    start,
    travel_yaw,
    distance,
    sample_step,
    target_speed,
    frame_id,
    label,
):
    """Build a sampled straight path beginning at the current base position."""
    distance = float(distance)
    if not math.isfinite(distance) or distance <= 1e-6:
        raise ValueError("portal straight distance must be finite and positive")
    sample_count = max(2, int(math.ceil(distance / sample_step)) + 1)
    station = np.linspace(0.0, distance, sample_count, dtype=np.float64)
    poses = np.column_stack(
        (
            start.x + station * math.cos(travel_yaw),
            start.y + station * math.sin(travel_yaw),
            np.full(sample_count, travel_yaw, dtype=np.float64),
        )
    )
    return path_from_poses(
        poses,
        frame_id=frame_id,
        target_speed=target_speed,
        label=label,
    )


def portal_entry_path(
    start,
    staging,
    inside,
    tangent_ratio,
    minimum_tangent_length,
    maximum_tangent_length,
    sample_step,
    target_speed,
    frame_id,
):
    """Join entrance alignment and crossing without stopping at the seam."""
    alignment = portal_alignment_path(
        start,
        staging,
        tangent_ratio,
        minimum_tangent_length,
        maximum_tangent_length,
        sample_step,
        target_speed,
        frame_id,
    )
    travel_yaw = staging.yaw
    straight_distance = (
        (inside.x - staging.x) * math.cos(travel_yaw)
        + (inside.y - staging.y) * math.sin(travel_yaw)
    )
    straight = portal_straight_path(
        staging,
        travel_yaw,
        straight_distance,
        sample_step,
        target_speed,
        frame_id,
        "tunnel_straight_entry",
    )
    poses = np.column_stack(
        (
            np.concatenate((alignment.x, straight.x[1:])),
            np.concatenate((alignment.y, straight.y[1:])),
            np.concatenate((alignment.heading, straight.heading[1:])),
        )
    )
    return (
        path_from_poses(
            poses,
            frame_id=frame_id,
            target_speed=target_speed,
            label="tunnel_aligned_entry",
        ),
        alignment.length,
    )


def portal_rear_clearance(
    pose,
    plane_point,
    travel_yaw,
    front,
    rear,
    half_width,
    padding,
):
    """Return the minimum footprint progress beyond a portal plane."""
    travel_x = math.cos(travel_yaw)
    travel_y = math.sin(travel_yaw)
    centre_progress = (
        (pose.x - plane_point[0]) * travel_x
        + (pose.y - plane_point[1]) * travel_y
    )
    heading_delta = normalize_angle(pose.yaw - travel_yaw)
    longitudinal_projection = math.cos(heading_delta)
    minimum_longitudinal = min(
        -(rear + padding) * longitudinal_projection,
        (front + padding) * longitudinal_projection,
    )
    lateral_projection = -(half_width + padding) * abs(
        math.sin(heading_delta)
    )
    return centre_progress + minimum_longitudinal + lateral_projection


def tracking_path_from_plan(
    plan,
    cruise_velocity,
    minimum_velocity,
    entry_velocity,
    exit_velocity,
    maximum_angular_velocity,
    maximum_lateral_acceleration,
    linear_acceleration,
    linear_deceleration,
    maximum_angular_acceleration,
):
    """Attach station and a dynamically feasible speed profile to a plan."""
    if plan is None or plan.x.size < 1:
        raise ValueError("a tunnel tracking path needs at least one pose")
    station = np.concatenate(
        ([0.0], np.cumsum(np.hypot(np.diff(plan.x), np.diff(plan.y))))
    )
    speed = build_speed_profile(
        station=station,
        curvature=plan.curvature,
        cruise_velocity=cruise_velocity,
        minimum_velocity=minimum_velocity,
        entry_velocity=entry_velocity,
        exit_velocity=exit_velocity,
        maximum_angular_velocity=maximum_angular_velocity,
        maximum_lateral_acceleration=maximum_lateral_acceleration,
        linear_acceleration=linear_acceleration,
        linear_deceleration=linear_deceleration,
        maximum_angular_acceleration=maximum_angular_acceleration,
    )
    return CommonPath(
        x=plan.x.copy(),
        y=plan.y.copy(),
        heading=plan.yaw.copy(),
        curvature=plan.curvature.copy(),
        station=station,
        speed=speed,
    )


def tracking_path_with_exit_connector(
    plan,
    connector,
    cruise_velocity,
    minimum_velocity,
    entry_velocity,
    exit_velocity,
    maximum_angular_velocity,
    maximum_lateral_acceleration,
    linear_acceleration,
    linear_deceleration,
    maximum_angular_acceleration,
    frame_id,
):
    """Join a Hybrid A* prefix and zero-curvature exit without a stop."""
    prefix = tracking_path_from_plan(
        plan,
        cruise_velocity,
        minimum_velocity,
        entry_velocity,
        exit_velocity,
        maximum_angular_velocity,
        maximum_lateral_acceleration,
        linear_acceleration,
        linear_deceleration,
        maximum_angular_acceleration,
    )
    if connector is None or connector.x.size < 2:
        raise ValueError("an exit connector needs at least two poses")
    position_gap = math.hypot(
        float(connector.x[0]) - float(prefix.x[-1]),
        float(connector.y[0]) - float(prefix.y[-1]),
    )
    heading_gap = abs(
        normalize_angle(
            float(connector.heading[0]) - float(prefix.heading[-1])
        )
    )
    if position_gap > 1e-6 or heading_gap > 1e-6:
        raise ValueError("Hybrid A* and exit connector poses do not meet")

    x = np.concatenate((prefix.x, connector.x[1:]))
    y = np.concatenate((prefix.y, connector.y[1:]))
    heading = np.concatenate((prefix.heading, connector.heading[1:]))
    # Preserve the planner's exact constant-curvature primitives.  The plan
    # requests zero terminal curvature and the quintic connector begins with
    # zero curvature, so keeping the prefix endpoint is the conservative seam.
    curvature = np.concatenate((prefix.curvature, connector.curvature[1:]))
    station = np.concatenate(
        ([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y))))
    )
    speed = build_speed_profile(
        station=station,
        curvature=curvature,
        cruise_velocity=cruise_velocity,
        minimum_velocity=minimum_velocity,
        entry_velocity=entry_velocity,
        exit_velocity=exit_velocity,
        maximum_angular_velocity=maximum_angular_velocity,
        maximum_lateral_acceleration=maximum_lateral_acceleration,
        linear_acceleration=linear_acceleration,
        linear_deceleration=linear_deceleration,
        maximum_angular_acceleration=maximum_angular_acceleration,
    )
    path = CommonPath(
        x=x,
        y=y,
        heading=heading,
        curvature=curvature,
        station=station,
        speed=speed,
        frame_id=frame_id,
        label="tunnel_hybrid_moving_exit",
    )
    connector_station = float(path.station[prefix.x.size - 1])
    return path, connector_station


def tracking_path_with_entry(
    entry,
    tunnel_path,
    cruise_velocity,
    minimum_velocity,
    entry_velocity,
    exit_velocity,
    maximum_angular_velocity,
    maximum_lateral_acceleration,
    linear_acceleration,
    linear_deceleration,
    maximum_angular_acceleration,
    frame_id,
):
    """Prepend the live portal entry to one preplanned tunnel route."""
    if entry is None or entry.x.size < 2:
        raise ValueError("a tunnel entry path needs at least two poses")
    if tunnel_path is None or tunnel_path.x.size < 2:
        raise ValueError("a preplanned tunnel path needs at least two poses")
    position_gap = math.hypot(
        float(tunnel_path.x[0]) - float(entry.x[-1]),
        float(tunnel_path.y[0]) - float(entry.y[-1]),
    )
    heading_gap = abs(
        normalize_angle(
            float(tunnel_path.heading[0]) - float(entry.heading[-1])
        )
    )
    if position_gap > 1e-6 or heading_gap > 1e-6:
        raise ValueError("tunnel entry and Hybrid A* poses do not meet")

    x = np.concatenate((entry.x, tunnel_path.x[1:]))
    y = np.concatenate((entry.y, tunnel_path.y[1:]))
    heading = np.concatenate((entry.heading, tunnel_path.heading[1:]))
    # Curvature is attached to the outgoing segment at each sample.  The
    # shared entry/tunnel pose therefore belongs to the first Hybrid A*
    # primitive, not to the final straight entry sample.
    curvature = np.concatenate((entry.curvature[:-1], tunnel_path.curvature))
    station = np.concatenate(
        ([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y))))
    )
    speed = build_speed_profile(
        station=station,
        curvature=curvature,
        cruise_velocity=cruise_velocity,
        minimum_velocity=minimum_velocity,
        entry_velocity=entry_velocity,
        exit_velocity=exit_velocity,
        maximum_angular_velocity=maximum_angular_velocity,
        maximum_lateral_acceleration=maximum_lateral_acceleration,
        linear_acceleration=linear_acceleration,
        linear_deceleration=linear_deceleration,
        maximum_angular_acceleration=maximum_angular_acceleration,
    )
    return CommonPath(
        x=x,
        y=y,
        heading=heading,
        curvature=curvature,
        station=station,
        speed=speed,
        frame_id=frame_id,
        label="tunnel_entry_hybrid_exit",
    )


class TunnelMissionController:
    """Own ``cmd_vel`` from the ordered entrance gate through lane rejoin."""

    WAIT_GATE = "WAIT_GATE"
    ACQUIRING = "ACQUIRING"
    ALIGNING_ENTRY = "ALIGNING_ENTRY"
    ENTERING = "ENTERING"
    PLANNING = "PLANNING"
    FOLLOWING = "FOLLOWING"
    EXITING = "EXITING"
    JOINING_LANE = "JOINING_LANE"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

    def __init__(self):
        get = rospy.get_param
        prefix = "~tunnel/"

        self.scan_topic = str(get(prefix + "topics/scan", "/scan_mid360_raw"))
        self.static_map_topic = str(get(prefix + "topics/static_map", "/map"))
        self.odom_topic = str(
            get(prefix + "topics/odometry", "/odometry/filtered")
        )
        self.map_pose_topic = str(
            get(prefix + "topics/map_pose", "/mission/map_pose")
        )
        self.gate_topic = str(
            get(prefix + "topics/zone_gate", "/mission/enable/tunnel")
        )
        self.arm_topic = str(
            get(prefix + "topics/arm", "/mission/arm/tunnel")
        )
        self.ready_topic = str(
            get(prefix + "topics/ready", "/mission/ready/tunnel")
        )
        self.lane_path_topic = str(
            get(
                prefix + "topics/lane_path_diagnostics",
                "/control/lane_path_diagnostics",
            )
        )
        self.sign_topic = str(get(prefix + "topics/signs", "/detect/signs"))
        self.manual_stop_topic = str(
            get(prefix + "topics/manual_stop", "/control/manual_stop")
        )
        self.cmd_vel_topic = str(get(prefix + "topics/cmd_vel", "/cmd_vel"))
        self.speed_limit_topic = str(
            get(prefix + "topics/lane_speed_limit", "/control/max_vel")
        )
        self.lane_service_name = str(
            get(
                prefix + "topics/lane_control_service",
                "/control/lane_mission_handoff",
            )
        )
        self.lane_stop_service_name = str(
            get(
                prefix + "topics/lane_stop_service",
                "/control/lane_following",
            )
        )
        self.map_frame = str(get(prefix + "frames/map", "map")).lstrip("/")
        if not self.map_frame:
            raise rospy.ROSInitException("tunnel map frame cannot be empty")

        bounds = tuple(float(value) for value in get(
            prefix + "map/planning_bounds", [-2.02, -2.02, 0.30, 0.55]
        ))
        if len(bounds) != 4:
            raise rospy.ROSInitException("tunnel planning_bounds needs four values")
        self.planning_bounds = bounds
        self.static_occupied_threshold = int(
            get(prefix + "map/static_occupied_threshold", 65)
        )
        self.unknown_is_occupied = bool(
            get(prefix + "map/unknown_is_occupied", True)
        )
        self.static_hit_exclusion_radius = max(
            0.0,
            float(get(prefix + "map/static_hit_exclusion_radius", 0.040)),
        )
        self.endpoint_clear_guard_radius = max(
            0.0,
            float(get(prefix + "map/endpoint_clear_guard_radius", 0.030)),
        )
        self.dynamic_inflation_radius = max(
            0.0,
            float(get(prefix + "map/dynamic_inflation_radius", 0.020)),
        )
        self.dynamic_occupied_value = int(
            get(prefix + "map/dynamic_occupied_value", 100)
        )
        self.dynamic_inflation_value = int(
            get(
                prefix + "map/dynamic_inflation_value",
                self.static_occupied_threshold - 1,
            )
        )
        if not (
            self.static_occupied_threshold
            <= self.dynamic_occupied_value
            <= 100
            and 0
            < self.dynamic_inflation_value
            < self.static_occupied_threshold
        ):
            raise rospy.ROSInitException(
                "tunnel dynamic occupancy must satisfy "
                "0 < inflation value < occupied threshold <= raw value <= 100"
            )
        self.mark_observations = max(
            1, int(get(prefix + "map/mark_observations", 2))
        )
        self.clear_observations = max(
            1, int(get(prefix + "map/clear_observations", 2))
        )
        self.decay_updates = max(0, int(get(prefix + "map/decay_updates", 0)))
        self.minimum_initial_scans = max(
            self.mark_observations,
            int(get(prefix + "map/minimum_initial_scans", 3)),
        )
        self.maximum_planning_range = max(
            0.2, float(get(prefix + "scan/maximum_planning_range", 3.0))
        )
        self.scan_transform_timeout = max(
            0.0, float(get(prefix + "scan/transform_timeout", 0.05))
        )
        self.maximum_scan_age = max(
            0.05, float(get(prefix + "scan/maximum_age", 0.25))
        )
        self.maximum_future_stamp = max(
            0.0, float(get(prefix + "scan/maximum_future_stamp", 0.06))
        )
        self.registration_samples = max(
            1, int(get(prefix + "registration/confirmation_samples", 3))
        )
        self.registration_maximum_gap = max(
            0.01, float(get(prefix + "registration/maximum_sample_gap", 0.20))
        )
        self.registration_maximum_position_delta = max(
            0.001,
            float(get(prefix + "registration/maximum_position_delta", 0.04)),
        )
        self.registration_maximum_heading_delta = math.radians(
            max(
                0.1,
                abs(
                    float(
                        get(
                            prefix + "registration/maximum_heading_delta_deg",
                            4.0,
                        )
                    )
                ),
            )
        )
        try:
            self.portal_registration_config = PortalRegistrationConfig(
                minimum_forward_distance=float(
                    get(prefix + "registration/minimum_forward_distance", 0.10)
                ),
                maximum_forward_distance=float(
                    get(prefix + "registration/maximum_forward_distance", 1.20)
                ),
                minimum_absolute_lateral=float(
                    get(prefix + "registration/minimum_absolute_lateral", 0.04)
                ),
                maximum_absolute_lateral=float(
                    get(prefix + "registration/maximum_absolute_lateral", 0.65)
                ),
                maximum_adjacent_beam_gap=int(
                    get(prefix + "registration/maximum_adjacent_beam_gap", 3)
                ),
                maximum_point_gap=float(
                    get(prefix + "registration/maximum_point_gap", 0.12)
                ),
                minimum_wall_points=int(
                    get(prefix + "registration/minimum_wall_points", 6)
                ),
                minimum_wall_length=float(
                    get(prefix + "registration/minimum_wall_length", 0.18)
                ),
                maximum_wall_residual=float(
                    get(prefix + "registration/maximum_wall_residual", 0.018)
                ),
                maximum_heading_deviation=math.radians(
                    abs(
                        float(
                            get(
                                prefix
                                + "registration/maximum_heading_deviation_deg",
                                30.0,
                            )
                        )
                    )
                ),
                maximum_orthogonal_angle=math.radians(
                    abs(
                        float(
                            get(
                                prefix
                                + "registration/maximum_orthogonal_angle_deg",
                                10.0,
                            )
                        )
                    )
                ),
                expected_portal_width=float(
                    get(prefix + "registration/expected_portal_width", 0.316)
                ),
                portal_width_tolerance=float(
                    get(prefix + "registration/portal_width_tolerance", 0.06)
                ),
                maximum_corner_skew=float(
                    get(prefix + "registration/maximum_corner_skew", 0.06)
                ),
            )
        except ValueError as error:
            raise rospy.ROSInitException(
                "invalid tunnel portal registration: %s" % error
            )
        self.ready_minimum_entry_lead = max(
            0.0,
            float(get(prefix + "registration/ready_minimum_entry_lead", 0.20)),
        )
        self.ready_maximum_entry_lead = max(
            self.ready_minimum_entry_lead,
            float(get(prefix + "registration/ready_maximum_entry_lead", 0.50)),
        )
        self.ready_maximum_heading_error = math.radians(
            max(
                0.1,
                abs(
                    float(
                        get(
                            prefix
                            + "registration/ready_maximum_heading_error_deg",
                            30.0,
                        )
                    )
                ),
            )
        )

        self.front = max(0.01, float(get(prefix + "footprint/front", 0.067645)))
        self.rear = max(0.01, float(get(prefix + "footprint/rear", 0.118073)))
        self.half_width = max(
            0.01, float(get(prefix + "footprint/half_width", 0.0903))
        )
        self.footprint_padding = max(
            0.0, float(get(prefix + "footprint/padding", 0.010))
        )
        footprint = RectangularFootprint(
            front=self.front,
            rear=self.rear,
            half_width=self.half_width,
            padding=self.footprint_padding,
        )

        self.goal = Pose2D(
            float(get(prefix + "goal/x", -0.18)),
            float(get(prefix + "goal/y", -1.748373)),
            math.radians(float(get(prefix + "goal/yaw_deg", 0.0))),
        )
        try:
            self.entry_staging_pose = pose_from_degrees(
                get(
                    prefix + "entry/staging_pose",
                    [-1.7475895, 0.14, -90.0],
                ),
                "tunnel entry staging_pose",
            )
            self.entry_inside_pose = pose_from_degrees(
                get(
                    prefix + "entry/inside_pose",
                    [-1.7475895, -0.28, -90.0],
                ),
                "tunnel entry inside_pose",
            )
            self.exit_outside_pose = pose_from_degrees(
                get(
                    prefix + "exit/outside_pose",
                    [0.20, -1.748373, 0.0],
                ),
                "tunnel exit outside_pose",
            )
        except ValueError as error:
            raise rospy.ROSInitException(str(error))
        self.entry_portal_plane_y = float(
            get(prefix + "entry/portal_plane_y", -0.105857)
        )
        try:
            self.registration_reference_pose = pose_from_degrees(
                get(
                    prefix + "registration/reference_pose",
                    [
                        self.entry_staging_pose.x,
                        self.entry_portal_plane_y,
                        math.degrees(self.entry_staging_pose.yaw),
                    ],
                ),
                "tunnel registration reference_pose",
            )
        except ValueError as error:
            raise rospy.ROSInitException(str(error))
        self.entry_clearance_margin = max(
            0.0, float(get(prefix + "entry/clearance_margin", 0.010))
        )
        self.exit_portal_plane_x = float(
            get(prefix + "exit/portal_plane_x", 0.003697)
        )
        self.exit_clearance_margin = max(
            0.0, float(get(prefix + "exit/clearance_margin", 0.010))
        )
        if not all(
            math.isfinite(value)
            for value in (
                self.entry_portal_plane_y,
                self.entry_clearance_margin,
                self.exit_portal_plane_x,
                self.exit_clearance_margin,
            )
        ):
            raise rospy.ROSInitException(
                "tunnel portal planes and margins must be finite"
            )
        entry_dx = self.entry_inside_pose.x - self.entry_staging_pose.x
        entry_dy = self.entry_inside_pose.y - self.entry_staging_pose.y
        entry_longitudinal = (
            entry_dx * math.cos(self.entry_staging_pose.yaw)
            + entry_dy * math.sin(self.entry_staging_pose.yaw)
        )
        entry_lateral = (
            -entry_dx * math.sin(self.entry_staging_pose.yaw)
            + entry_dy * math.cos(self.entry_staging_pose.yaw)
        )
        exit_dx = self.exit_outside_pose.x - self.goal.x
        exit_dy = self.exit_outside_pose.y - self.goal.y
        exit_longitudinal = (
            exit_dx * math.cos(self.goal.yaw) + exit_dy * math.sin(self.goal.yaw)
        )
        exit_lateral = (
            -exit_dx * math.sin(self.goal.yaw) + exit_dy * math.cos(self.goal.yaw)
        )
        if (
            entry_longitudinal <= 0.0
            or abs(entry_lateral) > 1e-6
            or abs(
                normalize_angle(
                    self.entry_inside_pose.yaw - self.entry_staging_pose.yaw
                )
            )
            > 1e-6
            or exit_longitudinal <= 0.0
            or abs(exit_lateral) > 1e-6
            or abs(
                normalize_angle(self.exit_outside_pose.yaw - self.goal.yaw)
            )
            > 1e-6
        ):
            raise rospy.ROSInitException(
                "tunnel straight portal poses must share one forward axis"
            )
        try:
            self.planner = HybridAStarPlanner(
                footprint=footprint,
                heading_bins=int(get(prefix + "planner/heading_bins", 36)),
                minimum_turning_radius=float(
                    get(prefix + "planner/minimum_turning_radius", 0.18)
                ),
                primitive_step=float(
                    get(prefix + "planner/primitive_step", 0.10)
                ),
                steering_samples=int(
                    get(prefix + "planner/steering_samples", 5)
                ),
                goal_position_tolerance=float(
                    get(prefix + "planner/goal_position_tolerance", 0.035)
                ),
                goal_heading_tolerance=math.radians(
                    float(get(prefix + "planner/goal_heading_tolerance_deg", 8.0))
                ),
                goal_curvature_tolerance=float(
                    get(prefix + "planner/goal_curvature_tolerance", 1e-6)
                ),
                non_straight_penalty=float(
                    get(prefix + "planner/non_straight_penalty", 0.08)
                ),
                steering_change_penalty=float(
                    get(prefix + "planner/steering_change_penalty", 0.04)
                ),
                inflation_radius=float(
                    get(prefix + "planner/static_inflation_radius", 0.0)
                ),
                obstacle_cost_weight=float(
                    get(prefix + "planner/obstacle_cost_weight", 0.12)
                ),
                obstacle_cost_distance=float(
                    get(prefix + "planner/obstacle_cost_distance", 0.30)
                ),
                soft_obstacle_cost_weight=float(
                    get(prefix + "planner/soft_obstacle_cost_weight", 1.0)
                ),
                soft_cost_check_step=float(
                    get(prefix + "planner/soft_cost_check_step", 0.020)
                ),
                heading_heuristic_weight=float(
                    get(prefix + "planner/heading_heuristic_weight", 0.25)
                ),
                collision_check_step=float(
                    get(prefix + "planner/collision_check_step", 0.010)
                ),
                collision_check_angle=math.radians(
                    float(get(prefix + "planner/collision_check_angle_deg", 3.0))
                ),
                path_sample_step=float(
                    get(prefix + "planner/path_sample_step", 0.020)
                ),
                state_xy_resolution=float(
                    get(prefix + "planner/state_xy_resolution", 0.050)
                ),
                maximum_iterations=int(
                    get(prefix + "planner/maximum_iterations", 60000)
                ),
                keep_in_rectangles=get(
                    prefix + "planner/keep_in_rectangles", []
                ),
            )
        except (TypeError, ValueError) as error:
            raise rospy.ROSInitException(
                "invalid tunnel Hybrid A* parameters: %s" % error
            )

        self.control_period = max(
            0.02, float(get(prefix + "control/period", 0.05))
        )
        self.entry_velocity_cap = max(
            0.0, float(get(prefix + "control/entry_velocity_cap", 0.04))
        )
        self.entry_handoff_velocity_tolerance = max(
            0.0,
            float(
                get(
                    prefix + "control/entry_handoff_velocity_tolerance",
                    0.005,
                )
            ),
        )
        self.preplan_lane_velocity_cap = max(
            0.005,
            min(
                self.entry_velocity_cap,
                float(
                    get(
                        prefix + "control/preplan_lane_velocity_cap",
                        0.025,
                    )
                ),
            ),
        )
        self.lane_resume_max_velocity = max(
            self.entry_velocity_cap,
            float(get(prefix + "control/lane_resume_max_velocity", 0.30)),
        )
        self.cruise_velocity = max(
            0.01, float(get(prefix + "control/cruise_velocity", 0.075))
        )
        self.minimum_velocity = clamp(
            float(get(prefix + "control/minimum_velocity", 0.035)),
            0.005,
            self.cruise_velocity,
        )
        self.entry_velocity = clamp(
            float(get(prefix + "control/entry_velocity", 0.035)),
            0.005,
            self.cruise_velocity,
        )
        self.exit_velocity = clamp(
            float(get(prefix + "control/exit_velocity", 0.075)),
            0.005,
            self.cruise_velocity,
        )
        self.entry_portal_velocity = clamp(
            float(get(prefix + "entry/velocity", 0.035)),
            0.005,
            self.cruise_velocity,
        )
        self.maximum_angular_velocity = max(
            0.05, float(get(prefix + "control/maximum_angular_velocity", 0.55))
        )
        self.maximum_lateral_acceleration = max(
            0.005,
            float(get(prefix + "control/maximum_lateral_acceleration", 0.030)),
        )
        self.linear_acceleration = max(
            0.005, float(get(prefix + "control/linear_acceleration", 0.15))
        )
        self.linear_deceleration = max(
            0.005, float(get(prefix + "control/linear_deceleration", 0.40))
        )
        self.angular_acceleration = max(
            0.05, float(get(prefix + "control/angular_acceleration", 0.55))
        )
        self.safety_linear_deceleration = max(
            0.01,
            float(get(prefix + "control/safety_linear_deceleration", 0.15)),
        )
        self.safety_angular_deceleration = max(
            0.05,
            float(get(prefix + "control/safety_angular_deceleration", 0.55)),
        )
        self.planning_stopped_linear = max(
            0.0,
            float(get(prefix + "control/planning_stopped_linear", 0.008)),
        )
        self.planning_stopped_angular = max(
            0.0,
            float(get(prefix + "control/planning_stopped_angular", 0.03)),
        )
        self.lookahead_distance = max(
            0.02, float(get(prefix + "control/lookahead_distance", 0.065))
        )
        self.heading_gain = float(get(prefix + "control/heading_gain", 0.30))
        self.path_curvature_weight = clamp(
            float(get(prefix + "control/path_curvature_weight", 0.20)), 0.0, 1.0
        )
        self.nearest_search_ahead = max(
            0.10, float(get(prefix + "control/nearest_search_ahead", 0.40))
        )
        self.soft_replan_lookahead_distance = max(
            0.05,
            float(
                get(
                    prefix + "control/soft_replan_lookahead_distance",
                    0.50,
                )
            ),
        )
        self.soft_start_prefix_max_distance = max(
            0.0,
            float(
                get(
                    prefix + "control/soft_start_prefix_max_distance",
                    0.35,
                )
            ),
        )
        self.tracking_position_tolerance = max(
            0.01,
            float(get(prefix + "control/tracking_position_tolerance", 0.030)),
        )
        self.tracking_heading_tolerance = math.radians(
            abs(float(get(prefix + "control/tracking_heading_tolerance_deg", 20.0)))
        )
        self.safety_reaction_time = max(
            0.0, float(get(prefix + "control/safety_reaction_time", 0.10))
        )
        self.safety_distance_margin = max(
            0.0, float(get(prefix + "control/safety_distance_margin", 0.010))
        )

        self.portal_sample_step = max(
            0.002, float(get(prefix + "entry/sample_step", 0.010))
        )
        self.entry_tangent_ratio = max(
            0.01, float(get(prefix + "entry/connector_tangent_ratio", 0.20))
        )
        self.entry_minimum_tangent_length = max(
            0.001,
            float(get(prefix + "entry/minimum_tangent_length", 0.008)),
        )
        self.entry_maximum_tangent_length = max(
            self.entry_minimum_tangent_length,
            float(get(prefix + "entry/maximum_tangent_length", 0.055)),
        )
        self.entry_path_position_tolerance = max(
            0.002, float(get(prefix + "entry/path_position_tolerance", 0.015))
        )
        self.entry_path_heading_tolerance = math.radians(
            abs(float(get(prefix + "entry/path_heading_tolerance_deg", 2.0)))
        )
        self.entry_path_remaining_tolerance = max(
            0.002,
            float(get(prefix + "entry/path_remaining_tolerance", 0.015)),
        )

        self.handoff_remaining_distance = max(
            0.005, float(get(prefix + "exit/handoff_remaining_distance", 0.035))
        )
        self.exit_position_tolerance = max(
            0.02, float(get(prefix + "exit/position_tolerance", 0.060))
        )
        self.exit_connector_tangent_ratio = max(
            0.01,
            float(get(prefix + "exit/connector_tangent_ratio", 0.20)),
        )
        self.exit_connector_minimum_tangent_length = max(
            0.001,
            float(get(prefix + "exit/minimum_tangent_length", 0.008)),
        )
        self.exit_connector_maximum_tangent_length = max(
            self.exit_connector_minimum_tangent_length,
            float(get(prefix + "exit/maximum_tangent_length", 0.055)),
        )
        self.exit_connector_sample_step = max(
            0.002,
            float(get(prefix + "exit/connector_sample_step", 0.010)),
        )
        self.exit_heading_tolerance = math.radians(
            abs(float(get(prefix + "exit/heading_tolerance_deg", 10.0)))
        )
        self.exit_confirmation_frames = max(
            1, int(get(prefix + "exit/confirmation_frames", 6))
        )
        self.exit_confirmation_max_gap = max(
            0.05, float(get(prefix + "exit/confirmation_max_gap", 0.20))
        )
        self.lane_path_timeout = max(
            0.05, float(get(prefix + "exit/lane_path_timeout", 0.35))
        )
        self.join_velocity_cap = max(
            0.0, float(get(prefix + "exit/join_velocity_cap", 0.06))
        )
        self.join_minimum_distance = max(
            0.0, float(get(prefix + "exit/join_minimum_distance", 0.04))
        )
        self.join_confirmation_frames = max(
            1, int(get(prefix + "exit/join_confirmation_frames", 6))
        )

        self.map_pose_timeout = max(
            0.05, float(get(prefix + "timeouts/map_pose", 0.40))
        )
        self.maximum_pose_stamp_skew = max(
            0.0, float(get(prefix + "timeouts/pose_stamp_skew", 0.05))
        )
        self.odom_timeout = max(
            0.05, float(get(prefix + "timeouts/odometry", 0.35))
        )
        self.scan_timeout = max(
            0.05, float(get(prefix + "timeouts/scan", 0.40))
        )
        self.acquisition_timeout = max(
            0.5, float(get(prefix + "timeouts/acquisition", 2.0))
        )
        self.entry_alignment_timeout = max(
            0.5, float(get(prefix + "timeouts/entry_alignment", 12.0))
        )
        self.entry_straight_timeout = max(
            0.5, float(get(prefix + "timeouts/entry_straight", 15.0))
        )
        self.planning_timeout = max(
            1.0, float(get(prefix + "timeouts/planning", 15.0))
        )
        self.plan_retry_period = max(
            0.1, float(get(prefix + "timeouts/plan_retry_period", 0.50))
        )
        self.mission_timeout = max(
            5.0, float(get(prefix + "timeouts/mission", 120.0))
        )
        self.exit_straight_timeout = max(
            0.5, float(get(prefix + "timeouts/exit_straight", 12.0))
        )
        self.join_timeout = max(
            0.5, float(get(prefix + "timeouts/lane_join", 2.0))
        )
        self.handoff_timeout = max(
            0.05, float(get(prefix + "timeouts/handoff", 0.30))
        )

        self.lock = threading.RLock()
        self.state = self.WAIT_GATE
        self.state_started = rospy.Time.now()
        self.mission_started = None
        self.zone_gate = False
        self.gate_requested = False
        self.start_requested = False
        self.revoke_requested = False
        self.manual_stop = False
        self.pause_started = None
        self.mission_has_control = False
        self.run_generation = 0
        self.input_fault = ""

        self.map_ready = False
        self.map_x = self.map_y = self.map_yaw = 0.0
        self.map_stamp = None
        self.map_received = None
        self.odom_ready = False
        self.odom_x = self.odom_y = self.odom_yaw = 0.0
        self.odom_linear_velocity = 0.0
        self.odom_angular_velocity = 0.0
        self.odom_stamp = None
        self.odom_received = None
        self.odom_frame = "odom"
        self.odom_history = deque(maxlen=100)
        self.map_from_odom = None
        self.frozen_odom_frame = ""
        self.arm_generation = 0
        self.armed_at = None
        self.registered_map_from_odom = None
        self.registered_odom_frame = ""
        self.registration_source_stamp = None
        self.registration_candidates = deque(maxlen=self.registration_samples)
        self.last_registration_stamp = None
        self.ready_published_generation = 0

        self.costmap = None
        self.costmap_version = 0
        self.planned_costmap_version = -1
        self.planned_grid = None
        self.soft_replan_contact_station = None
        self.scan_updates = 0
        self.minimum_planning_scan_updates = self.minimum_initial_scans
        self.scan_received = None
        self.scan_stamp = None
        self.last_scan_summary = None
        self.last_processed_scan_stamp = None
        self.grid_cache = None
        self.grid_cache_version = -1

        self.map_path = None
        self.odom_path = None
        self.entry_staging_station = None
        self.entry_inside_station = None
        self.exit_connector_station = None
        self.path_index = 0
        self.map_path_index = 0
        self.remaining_distance = math.inf
        self.last_position_error = 0.0
        self.last_heading_error = 0.0
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.last_command_time = None
        self.last_plan_attempt = None
        self.planning_generation = 0
        self.planning_thread = None
        self.last_plan_seconds = 0.0
        self.last_expanded_nodes = 0
        self.plan_attempts = 0
        self.replan_count = 0
        self.prepared_arm_generation = 0
        self.prepared_tunnel_path = None
        self.prepared_exit_connector_station = None
        self.prepared_grid = None
        self.prepared_costmap_version = -1
        self.prepared_soft_contact_station = None
        self.prepared_plan_kind = ""
        self.dynamic_preplan_generation = 0
        self.preplan_cap_generation = 0
        self.maximum_position_error_seen = 0.0
        self.maximum_heading_error_seen = 0.0
        self.amcl_anchor_position_delta = 0.0
        self.amcl_anchor_heading_delta = 0.0
        self.last_motion_safety_failure = ""

        self.sign_seen = False
        self.sign_confidence = 0.0
        self.lane_path_stamp = None
        self.lane_path_valid = False
        self.lane_path_minimum_line_clearance = math.nan
        self.lane_path_confirmation_count = 0
        self.last_lane_path_confirmation_time = None
        self.lane_command_linear = math.nan
        self.lane_command_angular = math.nan
        self.lane_command_stamp = None
        self.exit_confirmation_started = False
        self.confirmation_started_at = None
        self.lane_handoff_retry_pending = False
        self.join_start_x = self.join_start_y = self.join_start_yaw = 0.0
        self.join_origin_ready = False

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.speed_limit_pub = rospy.Publisher(
            self.speed_limit_topic, Float64, queue_size=1, latch=True
        )
        self.state_pub = rospy.Publisher(
            "/tunnel/state", String, queue_size=1, latch=True
        )
        self.path_pub = rospy.Publisher(
            "/tunnel/path", Path, queue_size=1, latch=True
        )
        self.costmap_pub = rospy.Publisher(
            "/tunnel/costmap", OccupancyGridMessage, queue_size=1, latch=True
        )
        self.diagnostics_pub = rospy.Publisher(
            "/tunnel/diagnostics", Float64MultiArray, queue_size=1, latch=True
        )
        self.ready_pub = rospy.Publisher(
            self.ready_topic, Header, queue_size=1, latch=True
        )

        # Laser endpoints must be projected with the transform that was valid
        # when the sensor acquired them.  A latest-pose approximation creates
        # moving wall ghosts while the robot turns.
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.Subscriber(
            self.static_map_topic,
            OccupancyGridMessage,
            self.static_map_callback,
            queue_size=1,
        )
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(
            self.map_pose_topic, PoseStamped, self.map_pose_callback, queue_size=1
        )
        rospy.Subscriber(self.gate_topic, Bool, self.gate_callback, queue_size=1)
        rospy.Subscriber(self.arm_topic, Header, self.arm_callback, queue_size=1)
        rospy.Subscriber(
            self.lane_path_topic,
            Float64MultiArray,
            self.lane_path_diagnostics_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.sign_topic, TrafficSign, self.sign_callback, queue_size=1
        )
        rospy.Subscriber(
            self.manual_stop_topic, Bool, self.manual_stop_callback, queue_size=1
        )

        self.lane_service = rospy.ServiceProxy(self.lane_service_name, SetBool)
        self.lane_stop_service = rospy.ServiceProxy(
            self.lane_stop_service_name, SetBool
        )
        self.timer = rospy.Timer(
            rospy.Duration(self.control_period), self.control_callback
        )
        rospy.on_shutdown(self.shutdown)
        self._publish_state()
        self._publish_diagnostics()
        rospy.loginfo(
            "Tunnel controller ready: goal=(%.3f, %.3f, %.1fdeg), scan=%s",
            self.goal.x,
            self.goal.y,
            math.degrees(self.goal.yaw),
            self.scan_topic,
        )

    def _publish_state(self):
        self.state_pub.publish(String(data=self.state))

    def _set_state(self, state, now=None):
        if state == self.state:
            return
        self.state = state
        self.state_started = rospy.Time.now() if now is None else now
        self._publish_state()
        rospy.loginfo("Tunnel mission state: %s", state)

    def _reset_registration(self):
        self.registered_map_from_odom = None
        self.registered_odom_frame = ""
        self.registration_source_stamp = None
        self.registration_candidates.clear()
        self.last_registration_stamp = None
        self.ready_published_generation = 0

    def _reset_preplan(self):
        self.planning_generation += 1
        self.last_plan_attempt = None
        self.plan_attempts = 0
        self.last_plan_seconds = 0.0
        self.last_expanded_nodes = 0
        self.prepared_arm_generation = 0
        self.prepared_tunnel_path = None
        self.prepared_exit_connector_station = None
        self.prepared_grid = None
        self.prepared_costmap_version = -1
        self.prepared_soft_contact_station = None
        self.prepared_plan_kind = ""
        self.dynamic_preplan_generation = 0
        self.preplan_cap_generation = 0

    def arm_callback(self, message):
        """Start one source-stamped portal-registration generation."""
        if message.frame_id != "tunnel":
            return
        with self.lock:
            generation = int(message.seq)
            if generation == self.arm_generation:
                return
            if self.state not in (self.WAIT_GATE, self.COMPLETE):
                rospy.logwarn(
                    "Ignoring tunnel arm generation %d while %s",
                    generation,
                    self.state,
                )
                return
            self.arm_generation = generation
            self.armed_at = (
                message.stamp
                if generation > 0 and message.stamp != rospy.Time()
                else rospy.Time.now()
                if generation > 0
                else None
            )
            self.zone_gate = False
            self.gate_requested = False
            self.start_requested = False
            self.revoke_requested = False
            self._reset_registration()
            self._reset_preplan()
            if generation > 0 and self.costmap is not None:
                # Start collecting the live mission-local layer while the
                # lane controller still drives.  Carry this prepared layer
                # across the later gate instead of inserting a scan dwell.
                self.costmap.reset_dynamic()
                self.costmap_version += 1
                self.grid_cache = None
                self.grid_cache_version = -1
                self.scan_updates = 0
                self.last_processed_scan_stamp = None
                self._publish_costmap(self.armed_at)
                self._start_preplan(self.armed_at, plan_kind="static")
            if generation > 0 and self.state == self.COMPLETE:
                self._set_state(self.WAIT_GATE)
            if generation > 0:
                rospy.loginfo(
                    "Tunnel portal registration armed: generation=%d "
                    "stamp=%.3f",
                    generation,
                    self.armed_at.to_sec(),
                )

    def _registration_stamp_is_eligible(self, stamp):
        if (
            self.arm_generation <= 0
            or self.armed_at is None
            or self.registered_map_from_odom is not None
        ):
            return False
        if stamp == rospy.Time() or stamp <= self.armed_at:
            return False
        now = rospy.Time.now()
        if (stamp - now).to_sec() > self.maximum_future_stamp:
            return False
        if self.last_registration_stamp is not None and (
            stamp <= self.last_registration_stamp
        ):
            return False
        return True

    def _submit_registration_candidate(
        self, map_from_odom, odom_frame, stamp
    ):
        """Confirm and freeze a single mission-template transform."""
        if not self._registration_stamp_is_eligible(stamp):
            return False
        candidate = tuple(float(value) for value in map_from_odom)
        if not all(math.isfinite(value) for value in candidate):
            return False
        if self.registration_candidates:
            previous_stamp, previous, previous_frame = (
                self.registration_candidates[-1]
            )
            gap = (stamp - previous_stamp).to_sec()
            position_delta = math.hypot(
                candidate[0] - previous[0], candidate[1] - previous[1]
            )
            heading_delta = abs(normalize_angle(candidate[2] - previous[2]))
            if (
                gap <= 0.0
                or gap > self.registration_maximum_gap
                or odom_frame != previous_frame
                or position_delta > self.registration_maximum_position_delta
                or heading_delta > self.registration_maximum_heading_delta
            ):
                self.registration_candidates.clear()
        self.registration_candidates.append((stamp, candidate, odom_frame))
        self.last_registration_stamp = stamp
        if len(self.registration_candidates) < self.registration_samples:
            return False

        count = float(len(self.registration_candidates))
        x = sum(item[1][0] for item in self.registration_candidates) / count
        y = sum(item[1][1] for item in self.registration_candidates) / count
        sine = sum(
            math.sin(item[1][2]) for item in self.registration_candidates
        )
        cosine = sum(
            math.cos(item[1][2]) for item in self.registration_candidates
        )
        self.registered_map_from_odom = (x, y, math.atan2(sine, cosine))
        self.registered_odom_frame = odom_frame
        self.registration_source_stamp = stamp
        rospy.loginfo(
            "Tunnel portal transform frozen: generation=%d "
            "template_from_%s=(%.3f, %.3f, %.1fdeg)",
            self.arm_generation,
            odom_frame,
            self.registered_map_from_odom[0],
            self.registered_map_from_odom[1],
            math.degrees(self.registered_map_from_odom[2]),
        )
        return True

    def _start_preplan(self, now, plan_kind="dynamic"):
        if plan_kind not in ("static", "dynamic"):
            raise ValueError("unknown tunnel preplan kind")
        if self.costmap is None:
            return False
        if plan_kind == "dynamic" and (
            self.scan_updates < self.minimum_initial_scans
        ):
            return False
        if self.last_plan_attempt is not None and (
            now - self.last_plan_attempt
        ).to_sec() < self.plan_retry_period:
            return False
        if self.planning_thread is not None and self.planning_thread.is_alive():
            return False

        self.last_plan_attempt = now
        self.plan_attempts += 1
        grid = self._planner_grid()
        costmap_version = self.costmap_version
        self.planning_generation += 1
        generation = self.planning_generation
        arm_generation = self.arm_generation
        worker = threading.Thread(
            target=self._plan_worker,
            args=(
                generation,
                grid,
                self.entry_inside_pose,
                None,
                costmap_version,
                self.plan_attempts,
            ),
            kwargs={
                "pre_gate": True,
                "arm_generation": arm_generation,
                "plan_kind": plan_kind,
            },
            name="tunnel_hybrid_astar_%s_preplan" % plan_kind,
        )
        worker.daemon = True
        self.planning_thread = worker
        worker.start()
        return True

    def _start_dynamic_preplan(self, now):
        """Plan once from the first confirmed live layer while lane owns cmd_vel."""
        if self.dynamic_preplan_generation == self.arm_generation:
            return False
        if self.planning_thread is not None and self.planning_thread.is_alive():
            return False
        self.prepared_arm_generation = 0
        self.prepared_tunnel_path = None
        self.prepared_exit_connector_station = None
        self.prepared_grid = None
        self.prepared_costmap_version = -1
        self.prepared_soft_contact_station = None
        self.prepared_plan_kind = ""
        self.last_plan_attempt = None
        if self.preplan_cap_generation != self.arm_generation:
            self.speed_limit_pub.publish(
                Float64(data=self.preplan_lane_velocity_cap)
            )
            self.preplan_cap_generation = self.arm_generation
        if not self._start_preplan(now, plan_kind="dynamic"):
            return False
        self.dynamic_preplan_generation = self.arm_generation
        rospy.loginfo(
            "Tunnel dynamic preplan started while lane owns cmd_vel: cap=%.3fm/s",
            self.preplan_lane_velocity_cap,
        )
        return True

    def _prepared_path_needs_dynamic_preplan(self, grid):
        """Return whether live evidence invalidated the prepared tunnel tail."""
        if self.prepared_tunnel_path is None:
            return True
        if not self._path_is_safe(grid, self.prepared_tunnel_path, 0):
            return True
        return bool(
            self.prepared_plan_kind == "static"
            and self._path_has_new_soft_cost(
                grid,
                self.prepared_tunnel_path,
                self.prepared_grid,
            )
        )

    def _restart_dynamic_preplan(self, now):
        """Replace one invalid prepared tail using the newest live layer."""
        if self.prepared_plan_kind == "dynamic":
            self.dynamic_preplan_generation = 0
        return self._start_dynamic_preplan(now)

    def _route_with_live_entry(self, start, entry_velocity=None):
        if (
            self.prepared_arm_generation != self.arm_generation
            or self.prepared_tunnel_path is None
            or self.prepared_exit_connector_station is None
        ):
            raise ValueError("tunnel Hybrid A* preplan is unavailable")
        entry_path, staging_station = portal_entry_path(
            start,
            self.entry_staging_pose,
            self.entry_inside_pose,
            self.entry_tangent_ratio,
            self.entry_minimum_tangent_length,
            self.entry_maximum_tangent_length,
            self.portal_sample_step,
            self.entry_portal_velocity,
            self.map_frame,
        )
        initial_velocity = (
            self.entry_velocity
            if entry_velocity is None
            else clamp(entry_velocity, 0.005, self.cruise_velocity)
        )
        route = tracking_path_with_entry(
            entry_path,
            self.prepared_tunnel_path,
            self.cruise_velocity,
            self.minimum_velocity,
            initial_velocity,
            self.exit_velocity,
            self.maximum_angular_velocity,
            self.maximum_lateral_acceleration,
            self.linear_acceleration,
            self.linear_deceleration,
            self.angular_acceleration,
            self.map_frame,
        )
        return (
            route,
            staging_station,
            entry_path.length,
            entry_path.length + self.prepared_exit_connector_station,
        )

    def _try_publish_ready(self, stamp):
        """Publish readiness after the complete tunnel route is preplanned."""
        if (
            self.arm_generation <= 0
            or self.registered_map_from_odom is None
            or not self.registered_odom_frame
            or self.ready_published_generation == self.arm_generation
            or stamp == rospy.Time()
            or self.registration_source_stamp is None
            or stamp < self.registration_source_stamp
        ):
            return False
        now = rospy.Time.now()
        source_age = (now - stamp).to_sec()
        if (
            source_age > self.scan_timeout
            or source_age < -self.maximum_future_stamp
        ):
            return False
        if self.costmap is None or self.scan_updates < self.minimum_initial_scans:
            return False
        if (
            self.prepared_arm_generation != self.arm_generation
            or self.prepared_tunnel_path is None
        ):
            self._start_dynamic_preplan(now)
            return False
        grid = self._planner_grid()
        if self._prepared_path_needs_dynamic_preplan(grid):
            self._restart_dynamic_preplan(now)
            return False
        synchronized = self._synchronized_odom_pose(stamp)
        if synchronized is None or self.odom_frame != self.registered_odom_frame:
            return False
        pose = Pose2D(
            *odom_pose_to_map(synchronized, self.registered_map_from_odom)
        )
        travel_x = math.cos(self.entry_staging_pose.yaw)
        travel_y = math.sin(self.entry_staging_pose.yaw)
        progress = (
            (pose.x - self.entry_staging_pose.x) * travel_x
            + (pose.y - self.entry_portal_plane_y) * travel_y
        )
        lead = -progress
        if not (
            self.ready_minimum_entry_lead
            <= lead
            <= self.ready_maximum_entry_lead
        ):
            return False
        if abs(normalize_angle(pose.yaw - self.entry_staging_pose.yaw)) > (
            self.ready_maximum_heading_error
        ):
            return False
        try:
            route, _, _, _ = self._route_with_live_entry(pose)
        except (TypeError, ValueError):
            return False
        if (
            not self._path_is_safe(grid, route, 0)
            or not self.planner.pose_is_collision_free(grid, self.goal)
        ):
            return False

        self.ready_published_generation = self.arm_generation
        ready = Header()
        ready.seq = self.arm_generation
        # Readiness is tied to the source scan whose synchronized odometry
        # lies inside the configured handoff lead.  The SE(2) estimate itself
        # remains the earlier frozen registration.
        ready.stamp = stamp
        ready.frame_id = "tunnel"
        self.ready_pub.publish(ready)
        rospy.loginfo(
            "Tunnel locally ready with preplanned route: generation=%d "
            "entry_lead=%.3fm",
            self.arm_generation,
            lead,
        )
        if self.gate_requested and self.state in (self.WAIT_GATE, self.COMPLETE):
            self.zone_gate = True
            self.start_requested = True
        return True

    def gate_callback(self, message):
        with self.lock:
            was_open = self.zone_gate
            self.gate_requested = bool(message.data)
            prepared = bool(
                self.arm_generation > 0
                and self.ready_published_generation == self.arm_generation
                and self.registered_map_from_odom is not None
                and self.registered_odom_frame
                and self.prepared_arm_generation == self.arm_generation
                and self.prepared_tunnel_path is not None
            )
            self.zone_gate = self.gate_requested and prepared
            if self.gate_requested and not prepared:
                rospy.logwarn_throttle(
                    1.0,
                    "Ignoring tunnel enable before matching local portal "
                    "registration readiness",
                )
                return
            if self.zone_gate and not was_open:
                self.start_requested = True
            elif not self.gate_requested and was_open and self.state != self.COMPLETE:
                self.revoke_requested = True

    def sign_callback(self, message):
        if message.sign_type != TrafficSign.TUNNEL_WARNING:
            return
        with self.lock:
            self.sign_seen = True
            self.sign_confidence = max(
                self.sign_confidence, float(message.confidence)
            )

    def map_pose_callback(self, message):
        now = rospy.Time.now()
        source_frame = (message.header.frame_id or self.map_frame).lstrip("/")
        if source_frame != self.map_frame:
            rospy.logerr_throttle(
                1.0,
                "Ignoring tunnel map pose in %s; expected %s",
                source_frame,
                self.map_frame,
            )
            return
        yaw = yaw_from_quaternion(message.pose.orientation)
        values = (message.pose.position.x, message.pose.position.y, yaw)
        if not all(math.isfinite(value) for value in values):
            rospy.logerr_throttle(1.0, "Ignoring non-finite tunnel map pose")
            return
        stamp = message.header.stamp if message.header.stamp != rospy.Time() else now
        with self.lock:
            self.map_x = float(values[0])
            self.map_y = float(values[1])
            self.map_yaw = float(values[2])
            self.map_stamp = stamp
            self.map_received = now
            self.map_ready = True
            if self.map_from_odom is not None:
                synchronized = self._synchronized_odom_pose(stamp)
                if synchronized is not None:
                    anchored = odom_pose_to_map(
                        synchronized, self.map_from_odom
                    )
                    self.amcl_anchor_position_delta = math.hypot(
                        self.map_x - anchored[0], self.map_y - anchored[1]
                    )
                    self.amcl_anchor_heading_delta = abs(
                        normalize_angle(self.map_yaw - anchored[2])
                    )

    def odom_callback(self, message):
        now = rospy.Time.now()
        stamp = message.header.stamp if message.header.stamp != rospy.Time() else now
        x = float(message.pose.pose.position.x)
        y = float(message.pose.pose.position.y)
        yaw = yaw_from_quaternion(message.pose.pose.orientation)
        linear_velocity = float(message.twist.twist.linear.x)
        angular_velocity = float(message.twist.twist.angular.z)
        frame = (message.header.frame_id or "odom").lstrip("/")
        if not all(
            math.isfinite(value)
            for value in (x, y, yaw, linear_velocity, angular_velocity)
        ):
            rospy.logerr_throttle(1.0, "Ignoring non-finite tunnel odometry")
            return
        with self.lock:
            if self.odom_stamp is not None and stamp < self.odom_stamp:
                if self.map_from_odom is not None:
                    self.input_fault = "odometry timestamp moved backwards"
                rospy.logerr_throttle(
                    1.0, "Ignoring out-of-order tunnel odometry"
                )
                return
            self.odom_x, self.odom_y, self.odom_yaw = x, y, yaw
            self.odom_linear_velocity = linear_velocity
            self.odom_angular_velocity = angular_velocity
            self.odom_stamp = stamp
            self.odom_received = now
            self.odom_frame = frame
            self.odom_ready = True
            self.odom_history.append((stamp, x, y, yaw, frame))

    def static_map_callback(self, message):
        source_frame = (message.header.frame_id or self.map_frame).lstrip("/")
        if source_frame != self.map_frame:
            rospy.logerr_throttle(
                2.0,
                "Rejected tunnel static map in %s; expected %s",
                source_frame,
                self.map_frame,
            )
            return
        orientation_yaw = yaw_from_quaternion(message.info.origin.orientation)
        if abs(normalize_angle(orientation_yaw)) > 1e-6:
            rospy.logerr_throttle(
                2.0, "Tunnel costmap requires an axis-aligned static map"
            )
            return
        try:
            candidate = TunnelCostmap(
                static_data=message.data,
                width=message.info.width,
                height=message.info.height,
                resolution=message.info.resolution,
                origin_x=message.info.origin.position.x,
                origin_y=message.info.origin.position.y,
                planning_bounds=self.planning_bounds,
                static_occupied_threshold=self.static_occupied_threshold,
                mark_observations=self.mark_observations,
                clear_observations=self.clear_observations,
                decay_updates=self.decay_updates,
                dynamic_occupied_value=self.dynamic_occupied_value,
                dynamic_inflation_value=self.dynamic_inflation_value,
                static_hit_exclusion_radius=self.static_hit_exclusion_radius,
                endpoint_clear_guard_radius=self.endpoint_clear_guard_radius,
                dynamic_inflation_radius=self.dynamic_inflation_radius,
            )
        except ValueError as error:
            rospy.logerr("Rejected tunnel static map: %s", error)
            return
        with self.lock:
            self.costmap = candidate
            self.costmap_version += 1
            self.grid_cache = None
            self.grid_cache_version = -1
            self.scan_updates = 0
            self.minimum_planning_scan_updates = self.minimum_initial_scans
            self.last_processed_scan_stamp = None
            self._publish_costmap(message.header.stamp)
            if (
                self.arm_generation > 0
                and self.state == self.WAIT_GATE
                and self.prepared_arm_generation != self.arm_generation
            ):
                self._start_preplan(rospy.Time.now(), plan_kind="static")
        rospy.loginfo(
            "Tunnel costmap ready: %dx%d at %.3fm, bounds=%s",
            candidate.width,
            candidate.height,
            candidate.resolution,
            candidate.bounds,
        )

    def scan_callback(self, message):
        now = rospy.Time.now()
        if not message.ranges or not math.isfinite(
            message.angle_min
        ) or not math.isfinite(message.angle_increment):
            return
        source_frame = message.header.frame_id.lstrip("/")
        if not source_frame:
            rospy.logerr_throttle(1.0, "Rejected tunnel scan without frame_id")
            return
        source_stamp = (
            message.header.stamp
            if message.header.stamp != rospy.Time()
            else now
        )
        source_age = (now - source_stamp).to_sec()
        if (
            source_age > self.maximum_scan_age
            or source_age < -self.maximum_future_stamp
        ):
            rospy.logwarn_throttle(
                1.0,
                "Rejected tunnel scan with source age %.3fs",
                source_age,
            )
            return
        with self.lock:
            registration_needed = self._registration_stamp_is_eligible(
                source_stamp
            )
            pre_gate_scan = bool(
                self.arm_generation > 0
                and self.state == self.WAIT_GATE
                and self.registered_map_from_odom is not None
                and self.registered_odom_frame
                and self.costmap is not None
            )
            active_scan = bool(
                self.zone_gate
                and self.costmap is not None
                and self.map_from_odom is not None
                and self.frozen_odom_frame
            )
            if not registration_needed and not pre_gate_scan and not active_scan:
                return
            if (
                (pre_gate_scan or active_scan)
                and
                self.last_processed_scan_stamp is not None
                and source_stamp <= self.last_processed_scan_stamp
            ):
                return
            target_frame = (
                self.odom_frame
                if registration_needed
                else self.registered_odom_frame
                if pre_gate_scan
                else self.frozen_odom_frame
            )

        detection = None
        if registration_needed:
            try:
                detection = detect_portal(
                    message.ranges,
                    message.angle_min,
                    message.angle_increment,
                    message.range_min,
                    message.range_max,
                    self.portal_registration_config,
                )
            except ValueError as error:
                rospy.logwarn_throttle(
                    1.0, "Rejected tunnel portal scan: %s", error
                )

        # Do not hold the controller lock while waiting for TF; odometry and
        # the fail-closed control timer must remain responsive.
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                source_stamp,
                rospy.Duration(self.scan_transform_timeout),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as error:
            rospy.logwarn_throttle(
                1.0,
                "Waiting for tunnel scan transform %s <- %s at %.3f: %s",
                target_frame,
                source_frame,
                source_stamp.to_sec(),
                error,
            )
            return

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        sensor_odom_pose = (
            float(translation.x),
            float(translation.y),
            yaw_from_quaternion(rotation),
        )
        if not all(math.isfinite(value) for value in sensor_odom_pose):
            rospy.logerr_throttle(1.0, "Rejected non-finite tunnel scan TF")
            return
        processed = rospy.Time.now()
        if (processed - source_stamp).to_sec() > self.maximum_scan_age:
            rospy.logwarn_throttle(1.0, "Tunnel scan TF arrived too late")
            return

        with self.lock:
            if registration_needed and self._registration_stamp_is_eligible(
                source_stamp
            ):
                if detection is None:
                    # Advance the source-stamp watermark even when geometry is
                    # incomplete, so an older scan can never complete a newer
                    # arm generation out of order.
                    self.last_registration_stamp = source_stamp
                elif target_frame == self.odom_frame:
                    portal_in_odom = odom_pose_to_map(
                        (
                            detection.center_x,
                            detection.center_y,
                            detection.yaw,
                        ),
                        sensor_odom_pose,
                    )
                    template_feature = (
                        self.registration_reference_pose.x,
                        self.registration_reference_pose.y,
                        self.registration_reference_pose.yaw,
                    )
                    candidate = map_from_odom_transform(
                        template_feature, portal_in_odom
                    )
                    self._submit_registration_candidate(
                        candidate, target_frame, source_stamp
                    )

            active_scan = bool(
                self.zone_gate
                and self.costmap is not None
                and self.map_from_odom is not None
                and self.frozen_odom_frame == target_frame
            )
            pre_gate_scan = bool(
                not active_scan
                and self.arm_generation > 0
                and self.state == self.WAIT_GATE
                and self.registered_map_from_odom is not None
                and self.registered_odom_frame == target_frame
                and self.costmap is not None
            )
            if not active_scan and not pre_gate_scan:
                return
            if (
                self.last_processed_scan_stamp is not None
                and source_stamp <= self.last_processed_scan_stamp
            ):
                return
            map_from_odom = (
                self.map_from_odom
                if active_scan
                else self.registered_map_from_odom
            )
            scan_generation = (
                self.run_generation if active_scan else self.arm_generation
            )

        sensor_pose = odom_pose_to_map(sensor_odom_pose, map_from_odom)
        if not all(math.isfinite(value) for value in sensor_pose):
            rospy.logerr_throttle(1.0, "Rejected non-finite tunnel scan TF")
            return
        effective_maximum = min(
            float(message.range_max), self.maximum_planning_range
        )
        if not math.isfinite(effective_maximum) or effective_maximum <= float(
            message.range_min
        ):
            return

        with self.lock:
            if active_scan:
                context_valid = bool(
                    self.zone_gate
                    and self.costmap is not None
                    and self.map_from_odom == map_from_odom
                    and self.frozen_odom_frame == target_frame
                    and self.run_generation == scan_generation
                )
            else:
                context_valid = bool(
                    self.arm_generation == scan_generation
                    and self.costmap is not None
                    and self.registered_map_from_odom == map_from_odom
                    and self.registered_odom_frame == target_frame
                    and self.state == self.WAIT_GATE
                )
            if not context_valid:
                return
            if (
                self.last_processed_scan_stamp is not None
                and source_stamp <= self.last_processed_scan_stamp
            ):
                return
            try:
                update = self.costmap.update_scan(
                    ranges=message.ranges,
                    angle_min=message.angle_min,
                    angle_increment=message.angle_increment,
                    range_min=message.range_min,
                    range_max=effective_maximum,
                    sensor_pose=sensor_pose,
                )
            except ValueError as error:
                rospy.logerr_throttle(1.0, "Rejected tunnel scan: %s", error)
                return
            self.scan_updates += 1
            self.scan_received = processed
            self.scan_stamp = source_stamp
            self.last_processed_scan_stamp = source_stamp
            self.last_scan_summary = update
            if update.marked_cells or update.cleared_cells or update.decayed_cells:
                self.costmap_version += 1
                self.grid_cache = None
                self.grid_cache_version = -1
                self._publish_costmap(self.scan_stamp)
            if (
                active_scan
                and self.state == self.ACQUIRING
                and not self.mission_has_control
                and self.prepared_arm_generation != self.arm_generation
                and self.dynamic_preplan_generation != self.arm_generation
            ):
                # A rejected pre-handoff plan is retried only after this newer
                # scan has updated the layer, never in a same-grid busy loop.
                self._start_dynamic_preplan(processed)
            elif not active_scan:
                self._try_publish_ready(source_stamp)

    def lane_path_diagnostics_callback(self, message):
        now = rospy.Time.now()
        try:
            diagnostics = PathDiagnostics.from_array(message.data)
        except (TypeError, ValueError):
            diagnostics = None
        valid = bool(
            diagnostics is not None
            and math.isfinite(diagnostics.minimum_line_clearance)
            and diagnostics.minimum_line_clearance > 0.0
        )
        with self.lock:
            self.lane_path_stamp = now
            self.lane_path_valid = valid
            self.lane_path_minimum_line_clearance = (
                diagnostics.minimum_line_clearance if valid else math.nan
            )
            if valid:
                self.lane_command_linear = float(diagnostics.commanded_linear)
                self.lane_command_angular = float(diagnostics.commanded_angular)
                self.lane_command_stamp = now
            if not valid or not (
                self.exit_confirmation_started or self.state == self.JOINING_LANE
            ):
                self.lane_path_confirmation_count = 0
                self.last_lane_path_confirmation_time = None
                return
            if (
                self.confirmation_started_at is not None
                and now <= self.confirmation_started_at
            ):
                self.lane_path_confirmation_count = 0
                self.last_lane_path_confirmation_time = None
                return
            if (
                self.last_lane_path_confirmation_time is None
                or (now - self.last_lane_path_confirmation_time).to_sec()
                > self.exit_confirmation_max_gap
            ):
                self.lane_path_confirmation_count = 1
            else:
                self.lane_path_confirmation_count += 1
            self.last_lane_path_confirmation_time = now

    def manual_stop_callback(self, message):
        with self.lock:
            requested = bool(message.data)
            if requested == self.manual_stop:
                return
            now = rospy.Time.now()
            if requested:
                self.manual_stop = True
                self.pause_started = now
                if self.mission_has_control:
                    self._publish_stop()
                return
            if self.pause_started is not None:
                paused = now - self.pause_started
                self.state_started += paused
                if self.mission_started is not None:
                    self.mission_started += paused
                if self.last_plan_attempt is not None:
                    self.last_plan_attempt += paused
            self.manual_stop = False
            self.pause_started = None
            self.lane_path_confirmation_count = 0
            self.last_lane_path_confirmation_time = None

    def _set_lane_controller(self, enabled):
        try:
            rospy.wait_for_service(
                self.lane_service_name, timeout=self.handoff_timeout
            )
            response = self.lane_service(enabled)
            if not response.success:
                rospy.logwarn_throttle(
                    0.5,
                    "Tunnel cmd_vel handoff deferred: %s",
                    response.message,
                )
                return False
            self.mission_has_control = not enabled
            return True
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logerr("Tunnel cmd_vel handoff failed: %s", error)
            return None

    def _stop_lane_controller(self):
        try:
            rospy.wait_for_service(
                self.lane_stop_service_name, timeout=self.handoff_timeout
            )
            response = self.lane_stop_service(False)
            return bool(response.success)
        except (rospy.ROSException, rospy.ServiceException):
            return False

    def _start_run(self, now):
        self.run_generation += 1
        self.input_fault = ""
        self.start_requested = False
        self.revoke_requested = False
        self.speed_limit_pub.publish(Float64(data=self.entry_velocity_cap))
        # Registration already projected confirmed live scans through the
        # frozen portal transform. Preserve that prepared costmap and its scan
        # count so activation adds no registration/planning dwell.
        self.minimum_planning_scan_updates = self.minimum_initial_scans
        self.map_path = None
        self.odom_path = None
        self.entry_staging_station = None
        self.entry_inside_station = None
        self.exit_connector_station = None
        self.planned_grid = None
        self.soft_replan_contact_station = None
        self.path_index = 0
        self.map_path_index = 0
        self.remaining_distance = math.inf
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.last_command_time = None
        self.last_plan_attempt = None
        self.map_from_odom = None
        self.frozen_odom_frame = ""
        self.amcl_anchor_position_delta = 0.0
        self.amcl_anchor_heading_delta = 0.0
        self.planning_generation += 1
        self.planning_thread = None
        self.replan_count = 0
        self.maximum_position_error_seen = 0.0
        self.maximum_heading_error_seen = 0.0
        self.exit_confirmation_started = False
        self.confirmation_started_at = None
        self.lane_handoff_retry_pending = False
        self.lane_path_stamp = None
        self.lane_path_valid = False
        self.lane_path_minimum_line_clearance = math.nan
        self.lane_path_confirmation_count = 0
        self.last_lane_path_confirmation_time = None
        self.join_origin_ready = False
        self._set_state(self.ACQUIRING, now)

    def _handoff_motion(self, now, route):
        measured_linear = max(0.0, float(self.odom_linear_velocity))
        measured_angular = clamp(
            self.odom_angular_velocity,
            -self.maximum_angular_velocity,
            self.maximum_angular_velocity,
        )
        if (
            self.lane_command_stamp is not None
            and (now - self.lane_command_stamp).to_sec() <= self.lane_path_timeout
            and math.isfinite(self.lane_command_linear)
            and math.isfinite(self.lane_command_angular)
        ):
            linear = max(0.0, self.lane_command_linear)
            angular = clamp(
                self.lane_command_angular,
                -self.maximum_angular_velocity,
                self.maximum_angular_velocity,
            )
        else:
            linear = measured_linear
            angular = measured_angular
        if linear > self.entry_velocity_cap:
            scale = self.entry_velocity_cap / linear
            linear = self.entry_velocity_cap
            angular *= scale
        if linear <= 1e-4:
            linear = min(float(route.speed[0]), self.entry_velocity_cap)
            angular = clamp(
                linear * float(route.curvature[0]),
                -self.maximum_angular_velocity,
                self.maximum_angular_velocity,
            )
        return linear, angular

    def _acquire(self, now):
        problem = self._input_problem(now, require_scan=False)
        if problem is not None:
            rospy.logwarn_throttle(
                1.0, "Waiting to use locally registered tunnel frame: %s", problem
            )
            return False
        if self.prepared_arm_generation != self.arm_generation:
            rospy.logwarn_throttle(
                1.0, "Waiting for tunnel Hybrid A* preplan"
            )
            return False
        self.map_from_odom = self.registered_map_from_odom
        self.frozen_odom_frame = self.registered_odom_frame
        grid = self._planner_grid()
        if self._prepared_path_needs_dynamic_preplan(grid):
            if self._restart_dynamic_preplan(now):
                self.state_started = now
            return False
        if (now - self.state_started).to_sec() > self.acquisition_timeout:
            self._fail("timed out while acquiring cmd_vel")
            return False
        if self.odom_linear_velocity > (
            self.entry_velocity_cap + self.entry_handoff_velocity_tolerance
        ):
            rospy.loginfo_throttle(
                1.0,
                "Tunnel lane handoff remains moving while entry cap settles: "
                "v=%.3f cap=%.3f",
                self.odom_linear_velocity,
                self.entry_velocity_cap,
            )
            return False
        start = Pose2D(*self._actual_map_pose())
        try:
            route, staging_station, inside_station, exit_station = (
                self._route_with_live_entry(
                    start,
                    entry_velocity=max(
                        self.entry_velocity,
                        min(
                            max(0.0, self.odom_linear_velocity),
                            self.entry_velocity_cap,
                        ),
                    ),
                )
            )
        except ValueError as error:
            self._fail("could not refresh preplanned tunnel entry: %s" % error)
            return False
        if not self._path_is_safe(grid, route, 0):
            self._fail("preplanned tunnel route changed before handoff")
            return False
        initial_linear, initial_angular = self._handoff_motion(now, route)
        if not self._command_is_safe(grid, initial_linear, initial_angular):
            self._fail("moving tunnel handoff command is not collision-free")
            return False
        self._commit_tracking_path(
            route,
            grid,
            now,
            initial_linear=initial_linear,
            initial_angular=initial_angular,
        )
        self.entry_staging_station = staging_station
        self.entry_inside_station = inside_station
        self.exit_connector_station = exit_station
        self.soft_replan_contact_station = (
            None
            if self.prepared_soft_contact_station is None
            else inside_station + self.prepared_soft_contact_station
        )
        if self._set_lane_controller(False) is not True:
            self._fail("could not acquire cmd_vel")
            return False
        rospy.loginfo(
            "Tunnel mission-local frame frozen from LiDAR: generation=%d "
            "template_from_%s=(%.4f, %.4f, %.2fdeg)",
            self.arm_generation,
            self.frozen_odom_frame,
            self.map_from_odom[0],
            self.map_from_odom[1],
            math.degrees(self.map_from_odom[2]),
        )
        command = Twist()
        command.linear.x = initial_linear
        command.angular.z = initial_angular
        self.cmd_pub.publish(command)
        self.last_command_time = now
        self.mission_started = now
        self._set_state(self.ALIGNING_ENTRY, now)
        rospy.loginfo(
            "Tunnel moving handoff: v=%.3fm/s w=%.3frad/s route=%.3fm",
            initial_linear,
            initial_angular,
            route.length,
        )
        return False

    def _input_problem(self, now, require_scan=True):
        if self.costmap is None:
            return "static map"
        if self.map_from_odom is None:
            if (
                self.arm_generation <= 0
                or self.ready_published_generation != self.arm_generation
                or self.registered_map_from_odom is None
                or not self.registered_odom_frame
                or self.registration_source_stamp is None
            ):
                return "source-stamped local portal registration"
        if not self.odom_ready or self.odom_received is None:
            return "odometry"
        if (
            self.map_from_odom is None
            and self.odom_frame != self.registered_odom_frame
        ):
            return "changed odometry frame since portal registration"
        if (
            self.map_from_odom is not None
            and self.odom_frame != self.frozen_odom_frame
        ):
            return "changed odometry frame"
        odom_age = (now - self.odom_stamp).to_sec()
        if odom_age < -self.maximum_pose_stamp_skew:
            return "future odometry"
        if (
            odom_age > self.odom_timeout
            or (now - self.odom_received).to_sec() > self.odom_timeout
        ):
            return "stale odometry"
        if require_scan:
            if self.scan_received is None:
                return "LiDAR scan"
            scan_age = (now - self.scan_stamp).to_sec()
            if scan_age < -self.maximum_future_stamp:
                return "future LiDAR scan"
            if (
                scan_age > self.scan_timeout
                or (now - self.scan_received).to_sec() > self.scan_timeout
            ):
                return "stale LiDAR scan"
            if self.scan_updates < self.minimum_planning_scan_updates:
                return "confirmed LiDAR costmap"
        return None

    def _actual_map_pose(self):
        if self.map_from_odom is None:
            return self.map_x, self.map_y, self.map_yaw
        return odom_pose_to_map(
            (self.odom_x, self.odom_y, self.odom_yaw),
            self.map_from_odom,
        )

    def _synchronized_odom_pose(self, map_stamp):
        """Return the odometry pose nearest one localized-map pose sample."""
        if map_stamp is None or not self.odom_history:
            return None
        candidate = min(
            self.odom_history,
            key=lambda sample: abs((sample[0] - map_stamp).to_sec()),
        )
        skew = abs((candidate[0] - map_stamp).to_sec())
        if skew > self.maximum_pose_stamp_skew:
            return None
        if candidate[4] != self.odom_frame:
            return None
        return candidate[1], candidate[2], candidate[3]

    def _planner_grid(self):
        if (
            self.grid_cache is not None
            and self.grid_cache_version == self.costmap_version
        ):
            return self.grid_cache
        self.grid_cache = OccupancyGrid.from_flat(
            self.costmap.to_occupancy_data(),
            width=self.costmap.width,
            height=self.costmap.height,
            resolution=self.costmap.resolution,
            origin_x=self.costmap.origin_x,
            origin_y=self.costmap.origin_y,
            occupied_threshold=self.static_occupied_threshold,
            unknown_is_occupied=self.unknown_is_occupied,
            soft_cost_data=self.costmap.to_soft_cost_data(),
        )
        self.grid_cache_version = self.costmap_version
        return self.grid_cache

    def _clear_tracking_path(self):
        self.map_path = None
        self.odom_path = None
        self.entry_staging_station = None
        self.entry_inside_station = None
        self.exit_connector_station = None
        self.planned_grid = None
        self.soft_replan_contact_station = None
        self.path_index = 0
        self.map_path_index = 0
        self.remaining_distance = math.inf

    def _commit_tracking_path(
        self,
        map_path,
        grid,
        now,
        initial_linear=0.0,
        initial_angular=0.0,
    ):
        """Freeze one already-validated map path into the odometry frame."""
        actual_map_pose = self._actual_map_pose()
        odom_pose = (self.odom_x, self.odom_y, self.odom_yaw)
        odom_path, _ = committed_route_path(
            map_path,
            actual_map_pose,
            odom_pose,
            odom_aligned=False,
        )
        self.map_path = map_path
        self.odom_path = odom_path
        self.planned_grid = grid
        self.soft_replan_contact_station = None
        self.path_index = 0
        self.map_path_index = 0
        self.remaining_distance = map_path.length
        self.planned_costmap_version = self.costmap_version
        self.last_linear = max(0.0, float(initial_linear))
        self.last_angular = clamp(
            float(initial_angular),
            -self.maximum_angular_velocity,
            self.maximum_angular_velocity,
        )
        self.last_command_time = now
        self._publish_path()
        self._publish_diagnostics()

    def _path_endpoint_ready(
        self,
        remaining_tolerance,
        position_tolerance,
        heading_tolerance,
    ):
        if self.odom_path is None:
            return False
        distance = math.hypot(
            float(self.odom_path.x[-1]) - self.odom_x,
            float(self.odom_path.y[-1]) - self.odom_y,
        )
        heading = abs(
            normalize_angle(float(self.odom_path.heading[-1]) - self.odom_yaw)
        )
        return bool(
            self.remaining_distance <= remaining_tolerance
            and distance <= position_tolerance
            and heading <= heading_tolerance
        )

    def _portal_path_problem(self, now, timeout, action):
        if self.mission_started is None or (
            now - self.mission_started
        ).to_sec() > self.mission_timeout:
            self._fail("tunnel mission timed out during %s" % action)
            return True
        if (now - self.state_started).to_sec() > timeout:
            self._fail("tunnel %s timed out" % action)
            return True
        return False

    def _entry_alignment_tick(self, now):
        if self.map_path is not None:
            self._follow(now)
            return
        self._fail("continuous preplanned tunnel route is missing at handoff")

    def _entering_tick(self, now):
        if self.map_path is not None:
            self._follow(now)
            return
        self._fail("continuous preplanned tunnel route was lost at entrance")

    def _entry_clearance_ready(self):
        clearance = portal_rear_clearance(
            Pose2D(*self._actual_map_pose()),
            (self.entry_staging_pose.x, self.entry_portal_plane_y),
            self.entry_staging_pose.yaw,
            self.front,
            self.rear,
            self.half_width,
            self.footprint_padding,
        )
        return bool(clearance >= self.entry_clearance_margin)

    def _planning_tick(self, now):
        self._publish_stop()
        if self.mission_started is not None and (
            now - self.mission_started
        ).to_sec() > self.mission_timeout:
            self._fail("tunnel mission timed out while planning")
            return False
        if (now - self.state_started).to_sec() > self.planning_timeout:
            self._fail("Hybrid A* could not find a tunnel path")
            return False
        problem = self._input_problem(now, require_scan=True)
        if problem is not None:
            rospy.logwarn_throttle(1.0, "Waiting for tunnel %s", problem)
            return False
        if self.map_from_odom is None:
            rospy.logwarn_throttle(
                1.0,
                "Waiting for frozen tunnel map/odom anchor",
            )
            return False
        if (
            abs(self.odom_linear_velocity) > self.planning_stopped_linear
            or abs(self.odom_angular_velocity) > self.planning_stopped_angular
        ):
            rospy.logwarn_throttle(
                1.0,
                "Waiting for tunnel stop before planning: v=%.3f w=%.3f",
                self.odom_linear_velocity,
                self.odom_angular_velocity,
            )
            return False
        start_odom = (self.odom_x, self.odom_y, self.odom_yaw)
        if self.last_plan_attempt is not None and (
            now - self.last_plan_attempt
        ).to_sec() < self.plan_retry_period:
            return False

        if self.planning_thread is not None and self.planning_thread.is_alive():
            return False

        self.last_plan_attempt = now
        self.plan_attempts += 1
        grid = self._planner_grid()
        actual_map_pose = self._actual_map_pose()
        start = Pose2D(*actual_map_pose)
        costmap_version = self.costmap_version
        self.planning_generation += 1
        generation = self.planning_generation
        worker = threading.Thread(
            target=self._plan_worker,
            args=(
                generation,
                grid,
                start,
                start_odom,
                costmap_version,
                self.plan_attempts,
            ),
            name="tunnel_hybrid_astar",
        )
        worker.daemon = True
        self.planning_thread = worker
        worker.start()
        return False

    def _build_moving_exit_path(self, plan, grid):
        """Append the mandatory, fully swept forward exit connector."""
        terminal = Pose2D(
            float(plan.x[-1]),
            float(plan.y[-1]),
            float(plan.yaw[-1]),
        )
        forward_distance = (
            (self.exit_outside_pose.x - terminal.x)
            * math.cos(self.goal.yaw)
            + (self.exit_outside_pose.y - terminal.y)
            * math.sin(self.goal.yaw)
        )
        if forward_distance <= 1e-6:
            return None, None, "outside pose is not ahead of the terminal pose"
        try:
            connector = portal_alignment_path(
                terminal,
                self.exit_outside_pose,
                self.exit_connector_tangent_ratio,
                self.exit_connector_minimum_tangent_length,
                self.exit_connector_maximum_tangent_length,
                self.exit_connector_sample_step,
                self.exit_velocity,
                self.map_frame,
                label="tunnel_moving_exit_connector",
            )
            forward_step = (
                np.diff(connector.x) * math.cos(self.goal.yaw)
                + np.diff(connector.y) * math.sin(self.goal.yaw)
            )
            if np.any(forward_step <= 1e-9):
                return None, None, "exit connector is not strictly forward"
            maximum_curvature = float(np.max(np.abs(connector.curvature)))
            minimum_turning_radius = float(
                getattr(self.planner, "minimum_turning_radius", 0.0)
            )
            if (
                minimum_turning_radius > 0.0
                and maximum_curvature
                > 1.0 / minimum_turning_radius + 1e-6
            ):
                return (
                    None,
                    None,
                    "exit connector curvature %.3f exceeds %.3f"
                    % (
                        maximum_curvature,
                        1.0 / minimum_turning_radius,
                    ),
                )
            path, connector_station = tracking_path_with_exit_connector(
                plan,
                connector,
                self.cruise_velocity,
                self.minimum_velocity,
                self.entry_velocity,
                self.exit_velocity,
                self.maximum_angular_velocity,
                self.maximum_lateral_acceleration,
                self.linear_acceleration,
                self.linear_deceleration,
                self.angular_acceleration,
                self.map_frame,
            )
        except ValueError as error:
            return None, None, str(error)
        # Hybrid A* validates only its prefix.  The connector and its seam must
        # pass the same exact oriented-footprint sweep even when no newer scan
        # arrived while the planning thread was running.
        if not self._path_is_safe(grid, path, 0):
            return None, None, "exit connector is not collision-free"
        return path, connector_station, ""

    def _plan_worker(
        self,
        generation,
        grid,
        start,
        start_odom,
        costmap_version,
        attempt,
        pre_gate=False,
        arm_generation=None,
        plan_kind="dynamic",
    ):
        wall_start = time.perf_counter()
        try:
            plan = self.planner.plan(
                grid, start, self.goal, goal_curvature=0.0
            )
        except Exception as error:
            rospy.logerr("Tunnel Hybrid A* raised an exception: %s", error)
            plan = None
        plan_seconds = time.perf_counter() - wall_start
        map_path = None
        exit_connector_station = None
        exit_connector_problem = ""
        if plan is not None:
            (
                map_path,
                exit_connector_station,
                exit_connector_problem,
            ) = self._build_moving_exit_path(plan, grid)

        with self.lock:
            if pre_gate:
                waiting_before_gate = bool(
                    self.state in (self.WAIT_GATE, self.COMPLETE)
                    and not self.zone_gate
                )
                waiting_to_acquire = bool(
                    self.state == self.ACQUIRING
                    and self.zone_gate
                    and not self.mission_has_control
                )
                context_valid = bool(
                    generation == self.planning_generation
                    and arm_generation == self.arm_generation
                    and (waiting_before_gate or waiting_to_acquire)
                    and not rospy.is_shutdown()
                )
            else:
                context_valid = bool(
                    generation == self.planning_generation
                    and self.state == self.PLANNING
                    and self.zone_gate
                    and not self.revoke_requested
                    and not rospy.is_shutdown()
                )
            if not context_valid:
                return
            self.planning_thread = None
            self.last_plan_seconds = plan_seconds
            self.last_expanded_nodes = 0 if plan is None else plan.expanded_nodes
            if plan is None or map_path is None:
                if pre_gate and plan_kind == "dynamic":
                    self.dynamic_preplan_generation = 0
                rospy.logwarn_throttle(
                    1.0,
                    "Hybrid A* has no complete collision-free tunnel path "
                    "(attempt %d): %s",
                    attempt,
                    exit_connector_problem or "planner returned no path",
                )
                self._publish_diagnostics()
                return

            # If newer evidence arrived during planning, accept the result only
            # after checking every sampled footprint against the latest layer.
            soft_contact_station = None
            if self.costmap_version != costmap_version:
                latest_grid = self._planner_grid()
                current_map_pose = (
                    start
                    if pre_gate
                    else Pose2D(*self._actual_map_pose())
                )
                if not self._path_is_safe(latest_grid, map_path, 0):
                    self.last_plan_attempt = None
                    if pre_gate and plan_kind == "dynamic":
                        self.dynamic_preplan_generation = 0
                    rospy.loginfo(
                        "Discarding tunnel plan affected by newer LiDAR evidence"
                    )
                    return
                soft_contact_station = self._path_future_soft_contact_station(
                    latest_grid,
                    map_path,
                    0,
                    current_map_pose,
                    baseline_grid=grid,
                )
                if (
                    soft_contact_station is not None
                    and soft_contact_station
                    <= self.soft_replan_lookahead_distance + 1e-9
                ):
                    self.last_plan_attempt = None
                    if pre_gate and plan_kind == "dynamic":
                        self.dynamic_preplan_generation = 0
                    rospy.loginfo(
                        "Discarding tunnel plan affected by newer LiDAR evidence"
                    )
                    return
            if pre_gate:
                self.prepared_arm_generation = arm_generation
                self.prepared_tunnel_path = map_path
                self.prepared_exit_connector_station = exit_connector_station
                # Keep the exact layer used by Hybrid A*.  Static-plan soft
                # validation must compare live costs against this baseline,
                # rather than treating newly observed clearance bands as old.
                self.prepared_grid = grid
                self.prepared_costmap_version = costmap_version
                self.prepared_soft_contact_station = soft_contact_station
                self.prepared_plan_kind = plan_kind
                if self.state == self.ACQUIRING:
                    self.state_started = rospy.Time.now()
                self._publish_diagnostics()
                rospy.loginfo(
                    "Hybrid A* tunnel %s preplan: %.3fm, %d expanded, %.3fs, "
                    "costmap=%d",
                    plan_kind,
                    map_path.length,
                    plan.expanded_nodes,
                    self.last_plan_seconds,
                    self.costmap_version,
                )
                if self.scan_stamp is not None:
                    self._try_publish_ready(self.scan_stamp)
                return
            current_map_pose = self._actual_map_pose()
            start_shift = math.hypot(
                current_map_pose[0] - start.x,
                current_map_pose[1] - start.y,
            )
            start_yaw_shift = abs(
                normalize_angle(current_map_pose[2] - start.yaw)
            )
            odom_shift = math.hypot(
                self.odom_x - start_odom[0], self.odom_y - start_odom[1]
            )
            odom_yaw_shift = abs(
                normalize_angle(self.odom_yaw - start_odom[2])
            )
            if (
                start_shift > 0.015
                or start_yaw_shift > math.radians(3.0)
                or odom_shift > 0.015
                or odom_yaw_shift > math.radians(3.0)
            ):
                self.last_plan_attempt = None
                rospy.loginfo(
                    "Discarding tunnel plan after pose shifted "
                    "map=%.3fm/%.1fdeg odom=%.3fm/%.1fdeg",
                    start_shift,
                    math.degrees(start_yaw_shift),
                    odom_shift,
                    math.degrees(odom_yaw_shift),
                )
                return

            # Preserve the map/odom relation from one synchronized sample pair;
            # mixing a plan-start map pose with latest odometry warps the route.
            odom_path, _ = committed_route_path(
                map_path,
                (start.x, start.y, start.yaw),
                start_odom,
                odom_aligned=False,
            )
            self.map_path = map_path
            self.odom_path = odom_path
            self.exit_connector_station = exit_connector_station
            # Preserve the immutable layer used by Hybrid A*.  Later soft
            # evidence is compared cell-by-cell with this baseline, so an
            # already-priced clearance band cannot cause a scan-by-scan loop.
            self.planned_grid = grid
            self.soft_replan_contact_station = soft_contact_station
            self.path_index = 0
            self.map_path_index = 0
            self.remaining_distance = map_path.length
            self.planned_costmap_version = self.costmap_version
            self.last_linear = 0.0
            self.last_angular = 0.0
            self.last_command_time = rospy.Time.now()
            self._publish_path()
            self._publish_diagnostics()
            self._set_state(self.FOLLOWING, rospy.Time.now())
            rospy.loginfo(
                "Hybrid A* tunnel path: %.3fm, %d expanded, %.3fs, costmap=%d",
                map_path.length,
                plan.expanded_nodes,
                self.last_plan_seconds,
                self.costmap_version,
            )
            rospy.loginfo(
                "Tunnel continuous exit appended at %.3fm; terminal speed "
                "%.3fm/s",
                exit_connector_station,
                float(map_path.speed[-1]),
            )
            return

    def _path_is_safe(self, grid, path, start_index):
        if path is None or path.x.size == 0:
            return False
        start_index = max(0, min(int(start_index), path.x.size - 1))
        for index in range(start_index, path.x.size - 1):
            start = Pose2D(
                float(path.x[index]),
                float(path.y[index]),
                float(path.heading[index]),
            )
            distance = math.hypot(
                float(path.x[index + 1] - path.x[index]),
                float(path.y[index + 1] - path.y[index]),
            )
            if not self.planner.primitive_is_collision_free(
                grid,
                start,
                float(path.curvature[index]),
                distance,
            ):
                return False
        return self.planner.pose_is_collision_free(
            grid,
            Pose2D(
                float(path.x[-1]),
                float(path.y[-1]),
                float(path.heading[-1]),
            ),
        )

    def _path_is_safe_until_station(
        self, grid, path, start_index, end_station
    ):
        """Sweep the live entry prefix without judging the Hybrid A* tail."""
        if path is None or path.x.size == 0:
            return False
        station = self._path_station(path)
        start_index = max(0, min(int(start_index), path.x.size - 1))
        end_index = min(
            path.x.size - 1,
            int(np.searchsorted(station, float(end_station), side="left")),
        )
        if end_index < start_index:
            end_index = start_index
        for index in range(start_index, end_index):
            start = Pose2D(
                float(path.x[index]),
                float(path.y[index]),
                float(path.heading[index]),
            )
            distance = math.hypot(
                float(path.x[index + 1] - path.x[index]),
                float(path.y[index + 1] - path.y[index]),
            )
            if not self.planner.primitive_is_collision_free(
                grid,
                start,
                float(path.curvature[index]),
                distance,
            ):
                return False
        return self.planner.pose_is_collision_free(
            grid,
            Pose2D(
                float(path.x[end_index]),
                float(path.y[end_index]),
                float(path.heading[end_index]),
            ),
        )

    def _remaining_path_is_safe(self, grid):
        if self.map_path is None:
            return False
        map_x, map_y, _ = self._actual_map_pose()
        self.map_path_index = nearest_path_index(
            self.map_path,
            map_x,
            map_y,
            self.map_path_index,
            search_ahead_distance=self.nearest_search_ahead,
        )
        # A remote obstacle may be observed once and then remain stable.  The
        # whole remaining swept route must therefore be approved on every
        # layer version, including the space between sampled poses.
        return self._path_is_safe(grid, self.map_path, self.map_path_index)

    @staticmethod
    def _path_station(path):
        station = getattr(path, "station", None)
        if station is not None and len(station) == path.x.size:
            return np.asarray(station, dtype=np.float64)
        return np.concatenate(
            (
                [0.0],
                np.cumsum(np.hypot(np.diff(path.x), np.diff(path.y))),
            )
        )

    @staticmethod
    def _soft_cost_increase_grid(grid, baseline_grid):
        """Return only per-cell positive soft-cost changes since planning."""
        compatible = bool(
            baseline_grid is not None
            and baseline_grid.data.shape == grid.data.shape
            and abs(baseline_grid.resolution - grid.resolution) <= 1e-12
            and abs(baseline_grid.origin_x - grid.origin_x) <= 1e-12
            and abs(baseline_grid.origin_y - grid.origin_y) <= 1e-12
            and baseline_grid.occupied_threshold == grid.occupied_threshold
        )
        if compatible:
            soft_cost_data = np.maximum(
                grid._soft_cost_values - baseline_grid._soft_cost_values,
                0.0,
            )
        else:
            # A replaced map cannot safely inherit the old plan's baseline.
            soft_cost_data = grid._soft_cost_values.copy()
        if not np.any(soft_cost_data > 0.0):
            return None
        return OccupancyGrid(
            np.zeros(grid.data.shape, dtype=np.int8),
            resolution=grid.resolution,
            origin_x=grid.origin_x,
            origin_y=grid.origin_y,
            occupied_threshold=grid.occupied_threshold,
            unknown_is_occupied=False,
            soft_cost_data=soft_cost_data,
        )

    def _path_has_new_soft_cost(self, grid, path, baseline_grid):
        """Check exact swept exposure added after a path was planned."""
        if path is None or path.x.size == 0:
            return False
        increase_grid = self._soft_cost_increase_grid(grid, baseline_grid)
        if increase_grid is None:
            return False
        for index in range(path.x.size - 1):
            start = Pose2D(
                float(path.x[index]),
                float(path.y[index]),
                float(path.heading[index]),
            )
            distance = math.hypot(
                float(path.x[index + 1] - path.x[index]),
                float(path.y[index + 1] - path.y[index]),
            )
            if self.planner._primitive_soft_cost_exposure(
                increase_grid,
                start,
                float(path.curvature[index]),
                distance,
            ) > 1e-12:
                return True
        terminal = Pose2D(
            float(path.x[-1]),
            float(path.y[-1]),
            float(path.heading[-1]),
        )
        return bool(self.planner._pose_soft_cost(increase_grid, terminal) > 0.0)

    def _path_future_soft_contact_station(
        self,
        grid,
        path,
        start_index,
        current_pose,
        baseline_grid=None,
    ):
        """Find the first newly increased clearance band on a route.

        A freshly planned route may start inside a graded band that appeared
        while the robot was stopping.  That prefix is allowed only until the
        footprint first clears it.  Any later positive cell-wise change is
        advance warning that an unseen obstacle surface is growing toward the
        committed path.
        """
        if path is None or path.x.size == 0 or not grid.has_soft_cost:
            return None
        increase_grid = self._soft_cost_increase_grid(grid, baseline_grid)
        if increase_grid is None:
            return None
        start_index = max(0, min(int(start_index), path.x.size - 1))
        station = self._path_station(path)
        # Only a positive change already under the current footprint earns an
        # escape grace.  A baseline band under the robot must not hide a
        # separate newly enlarged band farther along the same soft corridor.
        inside_start_prefix = (
            self.planner._pose_soft_cost(increase_grid, current_pose) > 0.0
        )
        prefix_cleared = not inside_start_prefix
        for index in range(start_index, path.x.size - 1):
            start = Pose2D(
                float(path.x[index]),
                float(path.y[index]),
                float(path.heading[index]),
            )
            distance = math.hypot(
                float(path.x[index + 1] - path.x[index]),
                float(path.y[index + 1] - path.y[index]),
            )
            increased_exposure = self.planner._primitive_soft_cost_exposure(
                increase_grid,
                start,
                float(path.curvature[index]),
                distance,
            )
            if increased_exposure > 1e-12:
                if prefix_cleared:
                    return float(station[index])
                if (
                    float(station[index + 1] - station[start_index])
                    > self.soft_start_prefix_max_distance + 1e-9
                ):
                    return float(station[start_index])
            elif not prefix_cleared:
                prefix_cleared = True

        final_pose = Pose2D(
            float(path.x[-1]),
            float(path.y[-1]),
            float(path.heading[-1]),
        )
        final_increase = (
            self.planner._pose_soft_cost(increase_grid, final_pose) > 0.0
        )
        if prefix_cleared and final_increase:
            return float(station[-1])
        if not prefix_cleared and final_increase:
            # The grace is an escape, not permission to finish a route that
            # never once clears the newly enlarged safety band.
            return float(station[start_index])
        return None

    def _path_has_future_soft_cost(
        self,
        grid,
        path,
        start_index,
        current_pose,
        baseline_grid=None,
    ):
        contact_station = self._path_future_soft_contact_station(
            grid,
            path,
            start_index,
            current_pose,
            baseline_grid=baseline_grid,
        )
        if contact_station is None:
            return False
        station = self._path_station(path)
        return bool(
            contact_station - float(station[start_index])
            <= self.soft_replan_lookahead_distance + 1e-9
        )

    def _remaining_path_has_future_soft_cost(self, grid):
        if self.map_path is None:
            return False
        self.soft_replan_contact_station = (
            self._path_future_soft_contact_station(
                grid,
                self.map_path,
                self.map_path_index,
                Pose2D(*self._actual_map_pose()),
                baseline_grid=self.planned_grid,
            )
        )
        return self._soft_replan_contact_is_due()

    def _soft_replan_contact_is_due(self):
        if (
            self.map_path is None
            or self.soft_replan_contact_station is None
        ):
            return False
        map_x, map_y, _ = self._actual_map_pose()
        self.map_path_index = nearest_path_index(
            self.map_path,
            map_x,
            map_y,
            self.map_path_index,
            search_ahead_distance=self.nearest_search_ahead,
        )
        station = self._path_station(self.map_path)
        current_index = max(
            0,
            min(int(self.map_path_index), station.size - 1),
        )
        return bool(
            self.soft_replan_contact_station - float(station[current_index])
            <= self.soft_replan_lookahead_distance + 1e-9
        )

    def _motion_is_safe(
        self, grid, start_pose, linear_velocity, angular_velocity
    ):
        """Check one measured/requested twist through reaction and braking."""
        pose = start_pose
        if not self.planner.pose_is_collision_free(grid, pose):
            self.last_motion_safety_failure = "start footprint"
            return False

        linear_velocity = float(linear_velocity)
        angular_velocity = float(angular_velocity)
        # A spatial margin must not be converted to extra reaction time.  At
        # near-zero linear speed that division made a small measured yaw rate
        # persist for seconds and falsely rejected every freshly planned path.
        reaction_remaining = self.safety_reaction_time
        elapsed = 0.0
        iterations = 0
        while (
            reaction_remaining > 1e-9
            or abs(linear_velocity) > 1e-6
            or abs(angular_velocity) > 1e-6
        ):
            iterations += 1
            if iterations > 1000:
                self.last_motion_safety_failure = "integration limit"
                return False
            distance_step = (
                math.inf
                if abs(linear_velocity) <= 1e-9
                else self.planner.collision_check_step
                / abs(linear_velocity)
            )
            angle_step = (
                math.inf
                if abs(angular_velocity) <= 1e-9
                else self.planner.collision_check_angle
                / abs(angular_velocity)
            )
            duration = min(0.02, distance_step, angle_step)
            if reaction_remaining > 1e-9:
                duration = min(duration, reaction_remaining)
            if duration <= 1e-9:
                self.last_motion_safety_failure = "zero integration step"
                return False

            pose = propagate_twist(
                pose, linear_velocity, angular_velocity, duration
            )
            elapsed += duration
            if not self.planner.pose_is_collision_free(grid, pose):
                phase = "reaction" if reaction_remaining > 1e-9 else "braking"
                self.last_motion_safety_failure = (
                    "%s at %.3fs pose=(%.3f, %.3f, %.1fdeg)"
                    % (phase, elapsed, pose.x, pose.y, math.degrees(pose.yaw))
                )
                return False

            if reaction_remaining > 1e-9:
                reaction_remaining = max(0.0, reaction_remaining - duration)
                continue
            linear_delta = self.safety_linear_deceleration * duration
            if linear_velocity > 0.0:
                linear_velocity = max(0.0, linear_velocity - linear_delta)
            else:
                linear_velocity = min(0.0, linear_velocity + linear_delta)
            angular_delta = self.safety_angular_deceleration * duration
            if angular_velocity > 0.0:
                angular_velocity = max(0.0, angular_velocity - angular_delta)
            else:
                angular_velocity = min(0.0, angular_velocity + angular_delta)
        if self.safety_distance_margin > 0.0 and not (
            self.planner.primitive_is_collision_free(
                grid,
                pose,
                0.0,
                self.safety_distance_margin,
            )
        ):
            self.last_motion_safety_failure = (
                "spatial margin at pose=(%.3f, %.3f, %.1fdeg)"
                % (pose.x, pose.y, math.degrees(pose.yaw))
            )
            return False
        self.last_motion_safety_failure = ""
        return True

    def _command_is_safe(self, grid, linear_velocity, angular_velocity):
        """Check both measured motion and the next requested command."""
        pose = Pose2D(*self._actual_map_pose())
        if not self._motion_is_safe(
            grid,
            pose,
            self.odom_linear_velocity,
            self.odom_angular_velocity,
        ):
            rospy.logwarn_throttle(
                0.5,
                "Unsafe measured tunnel stop sweep v=%.3f w=%.3f: %s",
                self.odom_linear_velocity,
                self.odom_angular_velocity,
                self.last_motion_safety_failure,
            )
            return False
        if not self._motion_is_safe(
            grid, pose, linear_velocity, angular_velocity
        ):
            rospy.logwarn_throttle(
                0.5,
                "Unsafe requested tunnel stop sweep v=%.3f w=%.3f: %s",
                linear_velocity,
                angular_velocity,
                self.last_motion_safety_failure,
            )
            return False
        return True

    def _request_replan(self, now, reason):
        self._publish_stop()
        self.replan_count += 1
        self.planning_generation += 1
        self.planning_thread = None
        self.last_plan_attempt = None
        self.map_path = None
        self.odom_path = None
        self.entry_staging_station = None
        self.entry_inside_station = None
        self.exit_connector_station = None
        self.planned_grid = None
        self.soft_replan_contact_station = None
        self.path_index = 0
        self.map_path_index = 0
        self.remaining_distance = math.inf
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.last_command_time = now
        self._set_state(self.PLANNING, now)
        rospy.logwarn("Tunnel path stopped for Hybrid A* replan: %s", reason)

    def _start_exit_confirmation(self, now):
        if self.exit_confirmation_started:
            return
        self.exit_confirmation_started = True
        self.confirmation_started_at = now
        self.lane_handoff_retry_pending = False
        self.lane_path_confirmation_count = 0
        self.last_lane_path_confirmation_time = None
        # Give the lane controller time to rebuild its rolling command at the
        # join speed before the ownership service can accept the handoff.
        self.speed_limit_pub.publish(Float64(data=self.join_velocity_cap))

    def _lane_confirmed(self, now, required_frames):
        return bool(
            self.confirmation_started_at is not None
            and self.lane_path_valid
            and self.lane_path_stamp is not None
            and self.lane_path_stamp > self.confirmation_started_at
            and (now - self.lane_path_stamp).to_sec() <= self.lane_path_timeout
            and self.last_lane_path_confirmation_time is not None
            and (now - self.last_lane_path_confirmation_time).to_sec()
            <= self.exit_confirmation_max_gap
            and self.lane_path_confirmation_count >= required_frames
        )

    def _exit_pose_ready(self):
        return self._path_endpoint_ready(
            self.handoff_remaining_distance,
            self.exit_position_tolerance,
            self.exit_heading_tolerance,
        )

    def _exit_clearance_ready(self):
        clearance = portal_rear_clearance(
            Pose2D(*self._actual_map_pose()),
            (self.exit_portal_plane_x, self.exit_outside_pose.y),
            self.exit_outside_pose.yaw,
            self.front,
            self.rear,
            self.half_width,
            self.footprint_padding,
        )
        return bool(clearance >= self.exit_clearance_margin)

    def _tracking_endpoint_ready(self):
        if self.state in (self.ALIGNING_ENTRY, self.ENTERING):
            return self._path_endpoint_ready(
                self.entry_path_remaining_tolerance,
                self.entry_path_position_tolerance,
                self.entry_path_heading_tolerance,
            )
        if self.state == self.EXITING:
            return self._exit_pose_ready()
        return False

    def _finish_tracking_phase(self, now):
        phase = self.state
        if phase == self.ALIGNING_ENTRY:
            self._fail(
                "continuous tunnel route ended before the staging transition"
            )
            return
        if phase == self.ENTERING:
            self._fail(
                "continuous tunnel route ended before the Hybrid A* segment"
            )
            return
        if phase == self.FOLLOWING:
            self._fail(
                "continuous tunnel exit connector was not reached before "
                "the tracking endpoint"
            )
            return
        if phase == self.EXITING:
            if not self._exit_clearance_ready():
                self._fail("robot footprint did not clear the tunnel exit")
                return
            if not self._lane_confirmed(now, self.exit_confirmation_frames):
                self._fail(
                    "safe rolling lane path was not confirmed before the "
                    "continuous tunnel exit ended"
                )
                return
            self._start_lane_join(now)
            return
        self._fail("unknown tunnel tracking phase %s" % phase)

    def _follow(self, now):
        if self.state == self.ALIGNING_ENTRY and self._portal_path_problem(
            now, self.entry_alignment_timeout, "entry alignment"
        ):
            return
        if self.state == self.ENTERING and self._portal_path_problem(
            now, self.entry_straight_timeout, "straight entry"
        ):
            return
        if self.state == self.EXITING and self._portal_path_problem(
            now, self.exit_straight_timeout, "straight exit"
        ):
            return
        if self.mission_started is None or (
            now - self.mission_started
        ).to_sec() > self.mission_timeout:
            self._fail("tunnel mission timed out")
            return
        if self.map_path is None or self.odom_path is None:
            self._fail("tunnel tracking path is missing")
            return
        problem = self._input_problem(now, require_scan=True)
        if problem is not None:
            self._fail("tunnel tracking lost %s" % problem)
            return
        grid = self._planner_grid()
        current_map_pose = Pose2D(*self._actual_map_pose())
        if not self.planner.pose_is_collision_free(grid, current_map_pose):
            self._fail(
                "localized tunnel footprint overlaps an obstacle at "
                "map=(%.4f, %.4f, %.2fdeg), costmap=%d"
                % (
                    current_map_pose.x,
                    current_map_pose.y,
                    math.degrees(current_map_pose.yaw),
                    self.costmap_version,
                )
            )
            return
        if self.costmap_version != self.planned_costmap_version:
            if not self._remaining_path_is_safe(grid):
                if self.state == self.FOLLOWING:
                    self._request_replan(
                        now, "new LiDAR obstacle intersects the path"
                    )
                elif (
                    self.state in (self.ALIGNING_ENTRY, self.ENTERING)
                    and self.entry_inside_station is not None
                    and self._path_is_safe_until_station(
                        grid,
                        self.map_path,
                        self.map_path_index,
                        self.entry_inside_station,
                    )
                ):
                    self._request_replan(
                        now,
                        "new LiDAR obstacle intersects the interior tail",
                    )
                else:
                    self._fail(
                        "new LiDAR obstacle intersects the surveyed portal path"
                    )
                return
            if (
                self.state == self.FOLLOWING
                and self._remaining_path_has_future_soft_cost(grid)
            ):
                self._request_replan(
                    now,
                    "new LiDAR obstacle safety band intersects the path",
                )
                return
            self.planned_costmap_version = self.costmap_version
        elif self.state == self.FOLLOWING and self._soft_replan_contact_is_due():
            # The layer can stay unchanged while the robot approaches a
            # farther soft intersection cached at the last map update.
            self._request_replan(
                now,
                "approaching a new LiDAR obstacle safety band",
            )
            return

        tracking = calculate_tracking(
            self.odom_path,
            self.odom_x,
            self.odom_y,
            self.odom_yaw,
            self.path_index,
            self.lookahead_distance,
            self.maximum_angular_velocity,
            self.heading_gain,
            self.path_curvature_weight,
            self.nearest_search_ahead,
        )
        self.path_index = tracking.path_index
        self.remaining_distance = max(
            0.0,
            self.odom_path.length
            - float(self.odom_path.station[self.path_index]),
        )
        self.last_position_error = tracking.position_error
        self.last_heading_error = tracking.heading_error
        self.maximum_position_error_seen = max(
            self.maximum_position_error_seen, tracking.position_error
        )
        self.maximum_heading_error_seen = max(
            self.maximum_heading_error_seen, abs(tracking.heading_error)
        )
        if tracking.position_error > self.tracking_position_tolerance:
            self._fail(
                "tunnel path error %.3fm exceeds %.3fm"
                % (tracking.position_error, self.tracking_position_tolerance)
            )
            return
        if abs(tracking.heading_error) > self.tracking_heading_tolerance:
            self._fail(
                "tunnel heading error %.1fdeg exceeds %.1fdeg"
                % (
                    math.degrees(abs(tracking.heading_error)),
                    math.degrees(self.tracking_heading_tolerance),
                )
            )
            return
        if (
            self.state == self.ALIGNING_ENTRY
            and self.entry_staging_station is not None
            and float(self.odom_path.station[self.path_index])
            >= self.entry_staging_station
        ):
            # Keep following the same continuous route. Stopping and
            # recommitting at the seam lets a constant-speed tracker overshoot
            # the narrow staging pose before its final heading has converged.
            self._set_state(self.ENTERING, now)
        if (
            self.state == self.ENTERING
            and self.entry_inside_station is not None
            and float(self.odom_path.station[self.path_index])
            >= self.entry_inside_station
        ):
            if not self._entry_clearance_ready():
                self._fail("robot footprint did not clear the tunnel entrance")
                return
            # The entry, Hybrid A* route, and exit are one prevalidated path.
            # Change only the phase; the same command stream continues.
            self._set_state(self.FOLLOWING, now)
        if (
            self.state == self.FOLLOWING
            and self.exit_connector_station is not None
            and float(self.odom_path.station[self.path_index])
            >= self.exit_connector_station
        ):
            # The Hybrid A* endpoint and exit connector are one continuous,
            # prevalidated path.  Change only the mission phase so the speed
            # limiter and the commanded motion remain continuous at the seam.
            self._start_exit_confirmation(now)
            self._set_state(self.EXITING, now)
        handoff_deferred = False
        if (
            self.state == self.EXITING
            and self._exit_clearance_ready()
            and self._lane_confirmed(now, self.exit_confirmation_frames)
        ):
            # The camera path is generated while tunnel control still owns
            # cmd_vel.  Hand it over as soon as the rear footprint clears the
            # portal; neither controller inserts a zero command at the seam.
            handoff = self._start_lane_join(now)
            if handoff is not False:
                return
            handoff_deferred = True
        self._publish_diagnostics()
        if self._tracking_endpoint_ready() and not handoff_deferred:
            self._finish_tracking_phase(now)
            return

        elapsed = (
            self.control_period
            if self.last_command_time is None
            else max(0.0, (now - self.last_command_time).to_sec())
        )
        limited = limit_tracking_command(
            target_speed=tracking.target_speed,
            reference_speed=tracking.target_speed,
            target_angular_velocity=tracking.angular_velocity,
            last_linear_velocity=self.last_linear,
            last_angular_velocity=self.last_angular,
            elapsed=elapsed,
            linear_acceleration=self.linear_acceleration,
            linear_deceleration=self.linear_deceleration,
            angular_acceleration=self.angular_acceleration,
            maximum_angular_velocity=self.maximum_angular_velocity,
            maximum_lateral_acceleration=self.maximum_lateral_acceleration,
        )
        command = Twist()
        command.linear.x = limited.linear_velocity
        command.angular.z = limited.angular_velocity
        if not self._command_is_safe(
            grid, command.linear.x, command.angular.z
        ):
            if self.state == self.FOLLOWING:
                self._request_replan(
                    now, "command reaction-and-stop sweep is not collision-free"
                )
            else:
                self._fail(
                    "portal command reaction-and-stop sweep is not collision-free"
                )
            return
        self.cmd_pub.publish(command)
        self.last_linear = limited.linear_velocity
        self.last_angular = limited.angular_velocity
        self.last_command_time = now

    def _start_lane_join(self, now):
        handoff = self._set_lane_controller(True)
        if handoff is None:
            self._fail("lane-control return outcome is unknown")
            return None
        if not handoff:
            # Keep ownership and let _follow finish this tick's normal tracking
            # and safety calculation. The next tick retries the service.
            self.lane_handoff_retry_pending = True
            return False
        self.lane_handoff_retry_pending = False
        started = rospy.Time.now()
        self.confirmation_started_at = started
        self.lane_path_confirmation_count = 0
        self.last_lane_path_confirmation_time = None
        self.join_origin_ready = False
        self._set_state(self.JOINING_LANE, started)
        return True

    def _join_lane(self, now):
        if not self.odom_ready or self.odom_received is None or (
            now - self.odom_received
        ).to_sec() > self.odom_timeout:
            self._fail("tunnel lane join lost odometry")
            return
        if not self.join_origin_ready:
            if self.odom_stamp is None or self.odom_stamp <= self.state_started:
                if (now - self.state_started).to_sec() > self.join_timeout:
                    self._fail("tunnel lane join received no fresh odometry")
                return
            self.join_start_x = self.odom_x
            self.join_start_y = self.odom_y
            self.join_start_yaw = self.odom_yaw
            self.join_origin_ready = True
        progress = (
            (self.odom_x - self.join_start_x) * math.cos(self.join_start_yaw)
            + (self.odom_y - self.join_start_y) * math.sin(self.join_start_yaw)
        )
        if progress >= self.join_minimum_distance and self._lane_confirmed(
            now, self.join_confirmation_frames
        ):
            self._complete(now)
            return
        if (now - self.state_started).to_sec() > self.join_timeout:
            self._fail("lane control did not complete the tunnel exit join")

    def _complete(self, now):
        if self.mission_has_control:
            self._fail("tunnel still owned cmd_vel at completion")
            return
        elapsed = (
            0.0
            if self.mission_started is None
            else (now - self.mission_started).to_sec()
        )
        self.speed_limit_pub.publish(
            Float64(data=self.lane_resume_max_velocity)
        )
        self._set_state(self.COMPLETE, now)
        self._publish_diagnostics()
        rospy.loginfo(
            "Tunnel complete in %.3fs; plans=%d replans=%d max_error=%.3fm "
            "max_heading=%.1fdeg AMCL_anchor_delta=%.3fm/%.1fdeg "
            "sign_seen=%s; lane controller owns cmd_vel",
            elapsed,
            self.plan_attempts,
            self.replan_count,
            self.maximum_position_error_seen,
            math.degrees(self.maximum_heading_error_seen),
            self.amcl_anchor_position_delta,
            math.degrees(self.amcl_anchor_heading_delta),
            self.sign_seen,
        )

    def _revoke(self):
        self.revoke_requested = False
        if self.state == self.COMPLETE:
            return
        self._fail("ordered tunnel gate closed before lane rejoin")

    def _fail(self, reason):
        if self.state == self.FAILED:
            return
        self.planning_generation += 1
        self.planning_thread = None
        self.speed_limit_pub.publish(Float64(data=0.0))
        if self.mission_has_control:
            # Lane control already records the active mission as sole owner.
            self._publish_stop()
        elif self._set_lane_controller(False):
            self._publish_stop()
        elif self._stop_lane_controller():
            # The lane controller remains the declared owner, but its manual
            # stop service and watchdog are now the sole zero publishers.
            self.mission_has_control = False
        else:
            # Both ownership services failed. Publishing zero is the last
            # collision-avoidance action even though ownership is now unknown.
            self.mission_has_control = True
            self._publish_stop()
            rospy.logfatal("Tunnel could not establish a sole cmd_vel stop owner")
        self._set_state(self.FAILED)
        self._publish_diagnostics()
        rospy.logerr("Tunnel mission failed: %s", reason)

    def _publish_stop(self):
        self.cmd_pub.publish(Twist())
        self.last_linear = 0.0
        self.last_angular = 0.0

    def _publish_path(self):
        if self.map_path is None:
            return
        now = rospy.Time.now()
        message = Path()
        message.header.stamp = now
        message.header.frame_id = self.map_frame
        for x, y, yaw in zip(
            self.map_path.x, self.map_path.y, self.map_path.heading
        ):
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.z, pose.pose.orientation.w = quaternion_from_yaw(
                yaw
            )
            message.poses.append(pose)
        self.path_pub.publish(message)

    def _publish_costmap(self, stamp=None):
        if self.costmap is None:
            return
        message = OccupancyGridMessage()
        message.header.stamp = (
            rospy.Time.now() if stamp is None or stamp == rospy.Time() else stamp
        )
        message.header.frame_id = self.map_frame
        message.info.resolution = self.costmap.resolution
        message.info.width = self.costmap.width
        message.info.height = self.costmap.height
        message.info.origin.position.x = self.costmap.origin_x
        message.info.origin.position.y = self.costmap.origin_y
        message.info.origin.orientation.w = 1.0
        message.data = self.costmap.to_occupancy_data()
        self.costmap_pub.publish(message)

    def _publish_diagnostics(self):
        elapsed = 0.0
        if self.mission_started is not None:
            elapsed = max(0.0, (rospy.Time.now() - self.mission_started).to_sec())
        self.diagnostics_pub.publish(
            Float64MultiArray(
                data=[
                    elapsed,
                    self.last_plan_seconds,
                    float(self.last_expanded_nodes),
                    float(self.costmap_version),
                    float(self.scan_updates),
                    float(self.path_index),
                    0.0 if not math.isfinite(self.remaining_distance) else self.remaining_distance,
                    self.last_position_error,
                    math.degrees(self.last_heading_error),
                    float(self.plan_attempts),
                    float(self.replan_count),
                    1.0 if self.sign_seen else 0.0,
                    self.sign_confidence,
                    self.amcl_anchor_position_delta,
                    math.degrees(self.amcl_anchor_heading_delta),
                ]
            )
        )

    def control_callback(self, _event):
        with self.lock:
            now = rospy.Time.now()
            if self.input_fault and self.state not in (
                self.WAIT_GATE,
                self.COMPLETE,
                self.FAILED,
            ):
                self._fail(self.input_fault)
                return
            if self.revoke_requested:
                self._revoke()
                return
            if (
                self.start_requested
                and self.zone_gate
                and self.state in (self.WAIT_GATE, self.COMPLETE)
            ):
                self._start_run(now)
            if self.state in (self.WAIT_GATE, self.COMPLETE):
                return
            if self.manual_stop:
                if self.mission_has_control:
                    self._publish_stop()
                return
            if self.state == self.FAILED:
                if self.mission_has_control:
                    self._publish_stop()
                return
            if self.state == self.ACQUIRING:
                if not self._acquire(now):
                    return
            if self.state == self.ALIGNING_ENTRY:
                self._entry_alignment_tick(now)
            elif self.state == self.ENTERING:
                self._entering_tick(now)
            elif self.state == self.PLANNING:
                self._planning_tick(now)
            elif self.state == self.FOLLOWING:
                self._follow(now)
            elif self.state == self.EXITING:
                self._follow(now)
            elif self.state == self.JOINING_LANE:
                self._join_lane(now)

    def shutdown(self):
        with self.lock:
            self.planning_generation += 1
            if self.mission_has_control:
                self.cmd_pub.publish(Twist())
        rospy.loginfo("Tunnel mission controller stopped")


if __name__ == "__main__":
    rospy.init_node("tunnel_mission_controller")
    TunnelMissionController()
    rospy.spin()
