#!/usr/bin/env python3
"""Follow one surveyed curvature-continuous construction-course path."""

from collections import deque
import math
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from std_srvs.srv import SetBool

from custom_autorace_bringup.obstacle_planner import (
    CourseSplinePlanner,
    Footprint,
    RectanglePathChecker,
    rectangle_surface_points,
)
from custom_autorace_bringup.path_following import (
    AxisAlignedBoundsBoundary,
    CommonPath,
    PathFollower,
    PathSafety,
    Pose2D,
    RigidTransform2D,
    SafetyMargins,
    StraightCorridorBoundary,
    TrackingConfig,
    clamp,
    combine_validation_results,
    freeze_path_in_odom,
    normalize_angle,
    yaw_from_quaternion,
)


class ObstacleMissionController:
    """Own cmd_vel only while traversing the fixed obstacle-course path."""

    # Source-stamped scan/map samples need both neighbouring odometry samples.
    # The controller deliberately releases its mission lock around live swept
    # validation, but initial alignment/full-path validation can still occupy a
    # timer callback for several hundred milliseconds.  A one-message rospy
    # input queue then drops the intermediate 30 Hz odometry samples and leaves
    # holes wider than maximum_scan_odom_skew in an otherwise healthy topic.
    # 64 samples retain over two seconds at the Gazebo EKF's nominal rate and
    # are pruned to odom_history_duration as soon as callbacks catch up.
    ODOM_SUBSCRIBER_QUEUE_SIZE = 64

    WAIT_GATE = "WAIT_GATE"
    ACQUIRING = "ACQUIRING"
    AVOIDING = "AVOIDING"
    REJOINING = "REJOINING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

    def __init__(self):
        get = rospy.get_param
        p = "~obstacle/"

        self.scan_topic = str(get(p + "topics/scan", "/scan_mid360_raw"))
        self.boundary_topic = str(
            get(p + "topics/lane_boundaries", "/detect/lane_boundaries")
        )
        self.odom_topic = str(get(p + "topics/odometry", "/odometry/filtered"))
        self.map_pose_topic = str(get(p + "topics/map_pose", "/mission/map_pose"))
        self.gate_topic = str(
            get(p + "topics/zone_gate", "/mission/enable/obstacle")
        )
        self.inside_topic = str(
            get(p + "topics/zone_inside", "/mission/inside/obstacle")
        )
        self.clearance_topic = str(
            get(p + "topics/zone_clearance", "/mission/clear/obstacle")
        )
        self.manual_stop_topic = str(
            get(p + "topics/manual_stop", "/control/manual_stop")
        )
        self.cmd_vel_topic = str(get(p + "topics/cmd_vel", "/cmd_vel"))
        self.speed_limit_topic = str(
            get(p + "topics/lane_speed_limit", "/control/max_vel")
        )
        self.lane_service_name = str(
            get(
                p + "topics/lane_control_service",
                "/control/lane_mission_handoff",
            )
        )
        self.lane_stop_service_name = str(
            get(p + "topics/lane_stop_service", "/control/lane_following")
        )

        self.course_heading = math.radians(
            float(get(p + "course/heading_deg", 90.0))
        )
        self.lane_width = max(0.20, float(get(p + "course/lane_width", 0.495)))
        self.image_center = float(
            get(p + "course/boundary_image_center", 500.0)
        )
        self.pixels_per_meter = max(
            100.0, float(get(p + "course/boundary_pixels_per_meter", 1280.0))
        )
        self.scale_filter_alpha = clamp(
            float(get(p + "course/scale_filter_alpha", 0.20)), 0.0, 1.0
        )
        self.line_position_alpha = clamp(
            float(get(p + "course/line_position_filter_alpha", 0.12)),
            0.0,
            1.0,
        )
        self.boundary_sample_forward = float(
            get(p + "course/boundary_sample_forward", 0.14)
        )
        self.minimum_boundary_gap = float(
            get(p + "course/minimum_boundary_gap_px", 380.0)
        )
        self.maximum_boundary_gap = float(
            get(p + "course/maximum_boundary_gap_px", 850.0)
        )
        self.boundary_heading_limit = math.radians(
            abs(float(get(p + "course/boundary_max_heading_deg", 12.0)))
        )
        self.acquisition_heading_tolerance = math.radians(
            abs(float(get(p + "course/acquisition_max_heading_deg", 12.0)))
        )
        self.camera_timeout = max(
            0.05, float(get(p + "course/camera_timeout", 0.50))
        )
        self.map_corridor_max_residual = max(
            0.001,
            float(get(p + "course/map_corridor_max_residual", 0.004)),
        )

        self.localization_margin = max(
            0.0, float(get(p + "footprint/localization_margin", 0.0))
        )
        self.tracking_margin = max(
            0.0, float(get(p + "footprint/tracking_margin", 0.0))
        )
        self.footprint = Footprint(
            front=max(0.01, float(get(p + "footprint/front", 0.067645))),
            rear=max(0.01, float(get(p + "footprint/rear", 0.118073))),
            half_width=max(
                0.01, float(get(p + "footprint/half_width", 0.0903))
            ),
            obstacle_padding=max(
                0.0, float(get(p + "footprint/obstacle_padding", 0.014))
            ),
            line_margin=max(
                0.0, float(get(p + "footprint/line_margin", 0.009))
            ),
            localization_margin=self.localization_margin,
            tracking_margin=self.tracking_margin,
        )
        self.validation_footprint = Footprint(
            front=self.footprint.front,
            rear=self.footprint.rear,
            half_width=self.footprint.half_width,
            obstacle_padding=max(
                0.0,
                float(get(p + "footprint/validation_obstacle_padding", 0.0)),
            ),
            line_margin=max(
                0.0,
                float(get(p + "footprint/validation_line_margin", 0.009)),
            ),
            localization_margin=self.localization_margin,
            tracking_margin=self.tracking_margin,
        )
        # The committed path keeps the full surveyed reserve above.  Runtime
        # contact enforcement may use the physical footprint where the legal
        # construction channel cannot also absorb a second tracking reserve;
        # fixed-path diagnostics continue to expose the full reserved margin.
        self.runtime_line_margin = max(
            0.0,
            float(
                get(
                    p + "footprint/runtime_line_margin",
                    self.validation_footprint.line_margin,
                )
            ),
        )
        self.runtime_obstacle_padding = max(
            0.0,
            float(
                get(
                    p + "footprint/runtime_obstacle_padding",
                    self.footprint.obstacle_padding,
                )
            ),
        )
        self.runtime_localization_margin = max(
            0.0,
            float(
                get(
                    p + "footprint/runtime_localization_margin",
                    self.localization_margin,
                )
            ),
        )
        self.runtime_tracking_margin = max(
            0.0,
            float(
                get(
                    p + "footprint/runtime_tracking_margin",
                    self.tracking_margin,
                )
            ),
        )
        self.path_checker = RectanglePathChecker(
            footprint=self.footprint,
            collision_check_step=float(
                get(p + "planner/collision_check_step", 0.008)
            ),
            collision_check_angle=math.radians(
                float(get(p + "planner/collision_check_angle_deg", 3.0))
            ),
        )
        self.tracking_position_tolerance = max(
            0.01,
            float(get(p + "planner/tracking_position_tolerance", 0.055)),
        )
        self.tracking_heading_tolerance = math.radians(
            abs(float(get(p + "planner/tracking_heading_tolerance_deg", 28.0)))
        )
        map_bounds = get(p + "safety/map_bounds", [-2.0, 2.0, -2.0, 2.0])
        if not isinstance(map_bounds, (list, tuple)) or len(map_bounds) != 4:
            raise rospy.ROSInitException(
                "obstacle safety/map_bounds must be [xmin, xmax, ymin, ymax]"
            )
        try:
            self.map_safety_boundary = AxisAlignedBoundsBoundary(
                *(float(value) for value in map_bounds)
            )
        except (TypeError, ValueError) as error:
            raise rospy.ROSInitException(
                "invalid obstacle safety/map_bounds: %s" % error
            )

        self.scan_max_range = max(
            0.30, float(get(p + "scan/maximum_planning_range", 1.40))
        )
        self.scan_rear_limit = max(
            0.0, float(get(p + "scan/rear_limit", 0.20))
        )
        self.cluster_link_distance = max(
            0.01, float(get(p + "scan/cluster_link_distance", 0.08))
        )
        self.cluster_minimum_points = max(
            1, int(get(p + "scan/cluster_minimum_points", 3))
        )
        self.obstacle_point_spacing = max(
            0.005, float(get(p + "scan/obstacle_point_spacing", 0.012))
        )
        self.scan_median_window = max(
            1, int(get(p + "scan/median_window", 5))
        )
        if self.scan_median_window % 2 == 0:
            self.scan_median_window += 1
        self.lidar_x = float(get(p + "scan/lidar_x", -0.033073))
        self.lidar_y = float(get(p + "scan/lidar_y", 0.0))
        self.maximum_scan_odom_skew = max(
            0.0, float(get(p + "scan/maximum_odom_skew", 0.075))
        )
        self.pending_scan_limit = max(
            1, int(get(p + "scan/pending_scan_limit", 4))
        )
        self.maximum_map_pose_odom_skew = max(
            0.0,
            float(get(p + "timeouts/map_pose_odom_skew", 0.075)),
        )
        self.odom_history_duration = max(
            self.maximum_scan_odom_skew,
            self.maximum_map_pose_odom_skew,
            float(get(p + "scan/odom_history_duration", 0.75)),
        )

        self.control_period = max(
            0.02, float(get(p + "control/period", 0.05))
        )
        self.entry_velocity_cap = max(
            0.0, float(get(p + "control/entry_velocity_cap", 0.09))
        )
        self.lane_resume_max_velocity = max(
            self.entry_velocity_cap,
            float(get(p + "control/lane_resume_max_velocity", 0.30)),
        )
        self.maximum_angular_velocity = max(
            0.05, float(get(p + "control/maximum_angular_velocity", 0.85))
        )
        self.maximum_lateral_acceleration = max(
            0.001,
            float(get(p + "control/maximum_lateral_acceleration", 0.035)),
        )
        self.linear_acceleration = max(
            0.01, float(get(p + "control/linear_acceleration", 0.25))
        )
        self.linear_deceleration = max(
            0.01, float(get(p + "control/linear_deceleration", 0.80))
        )
        self.angular_acceleration = max(
            0.05, float(get(p + "control/angular_acceleration", 0.55))
        )
        self.lookahead_distance = max(
            0.03, float(get(p + "control/lookahead_distance", 0.06))
        )
        self.search_ahead_distance = max(
            self.lookahead_distance,
            float(get(p + "control/search_ahead_distance", 0.30)),
        )
        self.heading_gain = float(get(p + "control/heading_gain", 0.35))
        self.curvature_feedforward_weight = clamp(
            float(get(p + "control/curvature_feedforward_weight", 0.25)),
            0.0,
            1.0,
        )
        self.lateral_feedback_gain = max(
            0.0, float(get(p + "control/lateral_feedback_gain", 1.0))
        )
        self.safety_reaction_time = max(
            0.0, float(get(p + "control/safety_reaction_time", 0.10))
        )
        self.safety_stop_margin = max(
            0.0, float(get(p + "control/safety_stop_margin", 0.005))
        )

        self.cruise_velocity = max(
            0.005, float(get(p + "template/cruise_velocity", 0.07))
        )
        self.minimum_velocity = min(
            self.cruise_velocity,
            max(0.005, float(get(p + "template/minimum_velocity", 0.035))),
        )
        self.entry_velocity = min(
            self.cruise_velocity,
            max(
                self.minimum_velocity,
                float(get(p + "template/entry_velocity", 0.07)),
            ),
        )
        self.exit_velocity = min(
            self.cruise_velocity,
            max(
                self.minimum_velocity,
                float(get(p + "template/exit_velocity", 0.07)),
            ),
        )
        self.tracking_config = TrackingConfig(
            lookahead_distance=self.lookahead_distance,
            maximum_linear_velocity=self.cruise_velocity,
            maximum_angular_velocity=self.maximum_angular_velocity,
            maximum_lateral_acceleration=self.maximum_lateral_acceleration,
            linear_acceleration=self.linear_acceleration,
            linear_deceleration=self.linear_deceleration,
            angular_acceleration=self.angular_acceleration,
            heading_gain=self.heading_gain,
            curvature_feedforward_weight=self.curvature_feedforward_weight,
            lateral_feedback_gain=self.lateral_feedback_gain,
            search_ahead_distance=self.search_ahead_distance,
        )

        self.spline_anchor_x = float(get(p + "template/map_anchor_x", 1.6375))
        self.spline_anchor_y = float(get(p + "template/map_anchor_y", 0.0200))
        self.course_from_map = RigidTransform2D(
            self.spline_anchor_x,
            self.spline_anchor_y,
            self.course_heading,
            source_frame="obstacle_course",
            target_frame="map",
        ).inverse()
        self.spline_planner = CourseSplinePlanner(
            collision_checker=self.path_checker,
            validation_footprint=self.validation_footprint,
            knots=get(p + "template/knots", []),
            exit_endpoint=get(p + "template/exit_curve/endpoint", []),
            exit_heading=math.radians(
                float(get(p + "template/exit_curve/heading_deg", 90.0))
            ),
            exit_start_tangent_length=float(
                get(p + "template/exit_curve/start_tangent_length", 0.15)
            ),
            exit_end_tangent_length=float(
                get(p + "template/exit_curve/end_tangent_length", 0.55)
            ),
            start_slope=float(get(p + "template/start_slope", 0.0)),
            end_slope=float(get(p + "template/end_slope", 0.0)),
            lateral_scale=float(get(p + "template/lateral_scale", 0.995)),
            sample_spacing=float(get(p + "template/sample_spacing", 0.004)),
            minimum_nominal_clearance=float(
                get(p + "template/minimum_nominal_clearance", 0.0035)
            ),
            live_validation_distance=float(
                get(p + "template/live_validation_distance", 0.45)
            ),
            cruise_velocity=self.cruise_velocity,
            minimum_velocity=self.minimum_velocity,
            entry_velocity=self.entry_velocity,
            exit_velocity=self.exit_velocity,
            maximum_angular_velocity=self.maximum_angular_velocity,
            maximum_lateral_acceleration=self.maximum_lateral_acceleration,
            linear_acceleration=self.linear_acceleration,
            linear_deceleration=self.linear_deceleration,
            angular_acceleration=self.angular_acceleration,
            goal_position_tolerance=float(
                get(p + "control/goal_position_tolerance", 0.04)
            ),
            goal_heading_tolerance=math.radians(
                abs(float(get(p + "control/goal_heading_tolerance_deg", 10.0)))
            ),
            goal_crossing_max_distance=float(
                get(p + "control/goal_crossing_max_distance", 0.10)
            ),
            registration_longitudinal_search=float(
                get(p + "template/registration/longitudinal_search", 0.18)
            ),
            registration_lateral_search=float(
                get(p + "template/registration/lateral_search", 0.14)
            ),
            registration_coarse_step=float(
                get(p + "template/registration/coarse_step", 0.02)
            ),
            registration_fine_window=float(
                get(p + "template/registration/fine_window", 0.012)
            ),
            registration_fine_step=float(
                get(p + "template/registration/fine_step", 0.002)
            ),
            registration_inlier_distance=float(
                get(p + "template/registration/inlier_distance", 0.025)
            ),
            registration_minimum_inliers=int(
                get(p + "template/registration/minimum_inliers", 15)
            ),
            registration_maximum_rms=float(
                get(p + "template/registration/maximum_rms", 0.015)
            ),
            registration_line_exclusion=float(
                get(p + "template/registration/line_exclusion", 0.025)
            ),
        )
        nominal_obstacles = self._nominal_obstacle_points(
            get(p + "template/barriers", []),
            float(get(p + "template/barrier_surface_spacing", 0.004)),
        )
        if not self.spline_planner.prepare(
            nominal_obstacles,
            -0.5 * self.lane_width,
            0.5 * self.lane_width,
        ):
            raise rospy.ROSInitException(
                "the configured obstacle path violates the nominal rectangle sweep"
            )

        self.scan_timeout = max(0.05, float(get(p + "timeouts/scan", 0.35)))
        self.odom_timeout = max(
            0.05, float(get(p + "timeouts/odometry", 0.35))
        )
        self.map_pose_timeout = max(
            0.05, float(get(p + "timeouts/map_pose", 0.35))
        )
        self.map_pose_future_tolerance = max(
            0.0,
            float(get(p + "timeouts/map_pose_future_tolerance", 0.05)),
        )
        self.lock = threading.RLock()
        self.state = self.WAIT_GATE
        self.zone_gate = False
        self.zone_inside = False
        self.zone_cleared = False
        self.start_requested = False
        self.revoke_requested = False
        self.mission_has_control = False
        self.handoff_ambiguous = False
        self.manual_stop = False
        self.shutting_down = False

        self.scan_points = np.empty((0, 2), dtype=np.float64)
        self.scan_odom_pose = None
        self.scan_stamp = None
        self.pending_scans = deque()
        self.scan_generation = 0
        self.last_plan_scan_generation = -1

        self.odom_ready = False
        self.odom_x = self.odom_y = self.odom_yaw = 0.0
        self.odom_linear_velocity = 0.0
        self.odom_angular_velocity = 0.0
        self.odom_frame = "odom"
        self.odom_stamp = None
        self.odom_history = deque()
        self.map_ready = False
        self.map_x = self.map_y = self.map_yaw = 0.0
        self.map_pose_header_stamp = None
        self.map_pose_stamp = None
        self.map_pose_received = None
        self.map_from_odom = None
        self.course_odom_heading = None
        self.odom_from_course = None

        self.line_center_absolute = None
        self.line_observation_stamp = None
        self.committed_path = None
        self.path_follower = PathFollower(self.tracking_config)
        self.fixed_path_validation = None
        self.remaining_distance = 0.0
        self.live_speed_limit = math.inf
        self.planning_seconds = 0.0

        self.last_command_time = None
        self.observed_lane_linear = 0.0
        self.observed_lane_angular = 0.0

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.emergency_stop_pub = rospy.Publisher(
            self.manual_stop_topic, Bool, queue_size=1, latch=True
        )
        self.speed_limit_pub = rospy.Publisher(
            self.speed_limit_topic, Float64, queue_size=1, latch=True
        )
        self.state_pub = rospy.Publisher(
            "/obstacle/state", String, queue_size=1, latch=True
        )
        self.path_pub = rospy.Publisher(
            "/obstacle/local_path", Path, queue_size=1, latch=True
        )
        self.diagnostics_pub = rospy.Publisher(
            "/obstacle/diagnostics", Float64MultiArray, queue_size=1, latch=True
        )
        self.planner_status_pub = rospy.Publisher(
            "/obstacle/planner_status", String, queue_size=1, latch=True
        )

        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(
            self.boundary_topic,
            Float64MultiArray,
            self.boundary_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.odom_topic,
            Odometry,
            self.odom_callback,
            queue_size=self.ODOM_SUBSCRIBER_QUEUE_SIZE,
        )
        rospy.Subscriber(
            self.map_pose_topic, PoseStamped, self.map_pose_callback, queue_size=1
        )
        rospy.Subscriber(self.gate_topic, Bool, self.gate_callback, queue_size=1)
        rospy.Subscriber(self.inside_topic, Bool, self.inside_callback, queue_size=1)
        rospy.Subscriber(
            self.clearance_topic, Bool, self.clearance_callback, queue_size=1
        )
        rospy.Subscriber(
            self.manual_stop_topic, Bool, self.manual_stop_callback, queue_size=1
        )
        rospy.Subscriber(
            self.cmd_vel_topic, Twist, self.command_observer_callback, queue_size=1
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
        self.planner_status_pub.publish(String(data="TEMPLATE_READY"))
        rospy.loginfo(
            "Obstacle controller ready: one surveyed curvature-continuous path, scan=%s",
            self.scan_topic,
        )

    @staticmethod
    def _nominal_obstacle_points(barriers, spacing):
        surfaces = []
        for barrier in barriers:
            if len(barrier) != 4:
                raise rospy.ROSInitException(
                    "each template barrier must be [x, y, length, width]"
                )
            surfaces.append(
                rectangle_surface_points(
                    float(barrier[0]),
                    float(barrier[1]),
                    float(barrier[2]),
                    float(barrier[3]),
                    spacing,
                )
            )
        if not surfaces:
            raise rospy.ROSInitException("template/barriers is empty")
        return np.vstack(surfaces)

    def _publish_state(self):
        self.state_pub.publish(String(data=self.state))

    def _set_state(self, state):
        if state == self.state:
            return
        self.state = state
        self._publish_state()
        rospy.loginfo("Obstacle mission state: %s", state)

    def gate_callback(self, message):
        with self.lock:
            was_open = self.zone_gate
            self.zone_gate = bool(message.data)
            if self.zone_gate and not was_open:
                self.start_requested = True
            elif not self.zone_gate and was_open and self.state != self.COMPLETE:
                self.revoke_requested = True

    def inside_callback(self, message):
        with self.lock:
            self.zone_inside = bool(message.data)

    def clearance_callback(self, message):
        with self.lock:
            self.zone_cleared = bool(message.data)

    def map_pose_callback(self, message):
        received = rospy.Time.now()
        header_stamp = message.header.stamp
        source_stamp = header_stamp if header_stamp != rospy.Time() else received
        with self.lock:
            self.map_x = float(message.pose.position.x)
            self.map_y = float(message.pose.position.y)
            self.map_yaw = yaw_from_quaternion(message.pose.orientation)
            self.map_ready = True
            self.map_pose_header_stamp = header_stamp
            self.map_pose_stamp = source_stamp
            self.map_pose_received = received
            self.map_from_odom = None
            problem = self._map_pose_problem(received)
            if self.odom_ready and problem is None:
                self._refresh_map_transform()
            elif problem is not None:
                rospy.logwarn_throttle(1.0, "Ignoring %s", problem)

    def odom_callback(self, message):
        received = rospy.Time.now()
        source_stamp = (
            message.header.stamp
            if message.header.stamp != rospy.Time()
            else received
        )
        with self.lock:
            odom_x = float(message.pose.pose.position.x)
            odom_y = float(message.pose.pose.position.y)
            odom_yaw = yaw_from_quaternion(message.pose.pose.orientation)
            odom_frame = message.header.frame_id or "odom"
            odom_epoch_changed = bool(
                self.odom_history
                and (
                    source_stamp < self.odom_history[-1][0]
                    or odom_frame != self.odom_history[-1][4]
                )
            )
            if odom_epoch_changed:
                # Simulation clock resets and frame changes invalidate temporal
                # interpolation across the discontinuity.
                self.odom_history.clear()
                self.pending_scans.clear()
                self.scan_points = np.empty((0, 2), dtype=np.float64)
                self.scan_odom_pose = None
                self.scan_stamp = None
            self.odom_x = odom_x
            self.odom_y = odom_y
            self.odom_yaw = odom_yaw
            self.odom_linear_velocity = float(message.twist.twist.linear.x)
            self.odom_angular_velocity = float(message.twist.twist.angular.z)
            self.odom_frame = odom_frame
            self.odom_stamp = source_stamp
            self.odom_ready = True
            self.odom_history.append(
                (source_stamp, odom_x, odom_y, odom_yaw, odom_frame)
            )
            while self.odom_history and (
                source_stamp - self.odom_history[0][0]
            ).to_sec() > self.odom_history_duration:
                self.odom_history.popleft()
            if self.map_ready and self._map_pose_problem(received) is None:
                self._refresh_map_transform()
            elif self.map_ready:
                self.map_from_odom = None
            for scan, aligned_pose in self._take_ready_scans(received):
                self._accept_scan(scan, aligned_pose)

    def _odom_pose_at(self, stamp, maximum_skew=None):
        """Return an odom pose using bracketed translation/yaw interpolation."""
        if not self.odom_history:
            return None
        allowed_skew = (
            self.maximum_scan_odom_skew
            if maximum_skew is None
            else max(0.0, float(maximum_skew))
        )
        target = float(stamp.to_sec())
        samples = tuple(self.odom_history)
        times = [float(sample[0].to_sec()) for sample in samples]
        right = int(np.searchsorted(times, target, side="left"))

        if right < len(samples) and abs(times[right] - target) <= 1e-9:
            sample = samples[right]
            return float(sample[1]), float(sample[2]), float(sample[3])

        if 0 < right < len(samples):
            left_sample = samples[right - 1]
            right_sample = samples[right]
            before = target - times[right - 1]
            after = times[right] - target
            if (
                before <= allowed_skew + 1e-12
                and after <= allowed_skew + 1e-12
            ):
                span = max(1e-12, before + after)
                fraction = before / span
                yaw_delta = normalize_angle(right_sample[3] - left_sample[3])
                return (
                    (1.0 - fraction) * left_sample[1]
                    + fraction * right_sample[1],
                    (1.0 - fraction) * left_sample[2]
                    + fraction * right_sample[2],
                    normalize_angle(left_sample[3] + fraction * yaw_delta),
                )

        # Never extrapolate a source-stamped sensor sample with the newest
        # pose. A later odom callback can provide the missing right bracket.
        return None

    def _refresh_map_transform(self):
        """Build the common map<-odom transform at the AMCL source stamp."""
        self.map_from_odom = None
        if self.map_pose_stamp is None:
            return False
        odom_pose = self._odom_pose_at(
            self.map_pose_stamp,
            maximum_skew=self.maximum_map_pose_odom_skew,
        )
        if odom_pose is None:
            return False
        self.map_from_odom = RigidTransform2D.from_pose_pair(
            Pose2D.from_value(odom_pose),
            Pose2D(self.map_x, self.map_y, self.map_yaw),
            source_frame=self.odom_frame,
            target_frame="map",
        )
        return True

    def _map_pose_problem(self, now):
        if (
            not self.map_ready
            or self.map_pose_stamp is None
            or self.map_pose_received is None
        ):
            return "AMCL map pose is unavailable"
        source_age = (now - self.map_pose_stamp).to_sec()
        receipt_age = (now - self.map_pose_received).to_sec()
        if source_age < -self.map_pose_future_tolerance:
            return "AMCL map pose source stamp is %.3fs in the future" % (
                -source_age
            )
        if receipt_age < -self.map_pose_future_tolerance:
            return "AMCL map pose receipt stamp is %.3fs in the future" % (
                -receipt_age
            )
        if source_age > self.map_pose_timeout:
            return "AMCL map pose source is stale by %.3fs" % source_age
        if receipt_age > self.map_pose_timeout:
            return "AMCL map pose receipt is stale by %.3fs" % receipt_age
        return None

    def command_observer_callback(self, message):
        with self.lock:
            if not self.mission_has_control:
                self.observed_lane_linear = float(message.linear.x)
                self.observed_lane_angular = float(message.angular.z)

    def boundary_callback(self, message):
        if len(message.data) < 4:
            return
        yellow = float(message.data[0])
        white = float(message.data[1])
        yellow_valid = bool(message.data[2] > 0.5 and math.isfinite(yellow))
        white_valid = bool(message.data[3] > 0.5 and math.isfinite(white))
        if not (yellow_valid and white_valid):
            return
        gap = white - yellow
        if not self.minimum_boundary_gap <= gap <= self.maximum_boundary_gap:
            return

        received = rospy.Time.now()
        with self.lock:
            if not self.map_ready or not self.odom_ready:
                return
            heading_error = self._course_heading_error()
            if abs(heading_error) > self.boundary_heading_limit:
                return
            if self.state not in (
                self.WAIT_GATE,
                self.ACQUIRING,
                self.AVOIDING,
                self.REJOINING,
            ):
                return

            measured_scale = gap / self.lane_width
            self.pixels_per_meter = (
                (1.0 - self.scale_filter_alpha) * self.pixels_per_meter
                + self.scale_filter_alpha * measured_scale
            )
            pixel_center = 0.5 * (yellow + white)
            robot_lateral = (
                self.image_center - pixel_center
            ) / self.pixels_per_meter
            relative_center = (
                -math.sin(heading_error) * self.boundary_sample_forward
                + math.cos(heading_error) * robot_lateral
            )
            course_heading = self._course_heading_in_odom()
            if course_heading is None:
                course_heading = self.odom_yaw + heading_error
            robot_absolute = (
                -math.sin(course_heading) * self.odom_x
                + math.cos(course_heading) * self.odom_y
            )
            measured_absolute = robot_absolute + relative_center
            if self.line_center_absolute is None:
                self.line_center_absolute = measured_absolute
            else:
                alpha = self.line_position_alpha
                self.line_center_absolute = (
                    (1.0 - alpha) * self.line_center_absolute
                    + alpha * measured_absolute
                )
            # Freshness belongs to an accepted two-line observation.  Invalid
            # views beside a barrier must not keep an older centre alive.
            self.line_observation_stamp = received

    def _pending_scan_expired(self, scan, now):
        return bool(
            (now - scan["source_stamp"]).to_sec() > self.scan_timeout
            or (now - scan["received"]).to_sec() > self.scan_timeout
        )

    def _enqueue_pending_scan(self, scan, now):
        while self.pending_scans and self._pending_scan_expired(
            self.pending_scans[0], now
        ):
            self.pending_scans.popleft()
        if self._pending_scan_expired(scan, now):
            return
        if len(self.pending_scans) >= self.pending_scan_limit:
            self.pending_scans.popleft()
            rospy.logwarn_throttle(
                1.0,
                "Obstacle dropped the oldest pending LaserScan before odom "
                "synchronization",
            )
        self.pending_scans.append(scan)

    def _take_ready_scans(self, now):
        """Return bracketed scans and retain only bounded future scans."""
        history = tuple(self.odom_history)
        latest_stamp = (
            float(history[-1][0].to_sec()) if history else -math.inf
        )
        ready = []
        waiting = deque()
        for scan in self.pending_scans:
            if self._pending_scan_expired(scan, now):
                continue
            aligned_pose = self._odom_pose_at(
                scan["source_stamp"],
                maximum_skew=self.maximum_scan_odom_skew,
            )
            if aligned_pose is not None:
                ready.append((scan, aligned_pose))
                continue
            if float(scan["source_stamp"].to_sec()) > latest_stamp:
                waiting.append(scan)
                continue
            rospy.logwarn_throttle(
                1.0,
                "Obstacle discarded a LaserScan whose odom bracket exceeded "
                "the synchronization tolerance",
            )
        self.pending_scans = waiting
        ready.sort(key=lambda item: item[0]["source_stamp"].to_sec())
        return ready

    def _accept_scan(self, scan, aligned_pose):
        with self.lock:
            if (
                self.scan_stamp is not None
                and scan["source_stamp"] <= self.scan_stamp
            ):
                return
            self.scan_points = scan["points"]
            self.scan_odom_pose = aligned_pose
            self.scan_stamp = scan["source_stamp"]
            self.scan_generation += 1

    def scan_callback(self, message):
        received = rospy.Time.now()
        source_stamp = (
            message.header.stamp
            if message.header.stamp != rospy.Time()
            else received
        )
        ranges = np.asarray(message.ranges, dtype=np.float64)
        if ranges.size == 0 or not math.isfinite(message.angle_increment):
            return
        indices = np.arange(ranges.size, dtype=np.float64)
        angles = float(message.angle_min) + indices * float(message.angle_increment)
        minimum_range = max(0.0, float(message.range_min))
        maximum_range = min(float(message.range_max), self.scan_max_range)
        valid = np.isfinite(ranges)
        valid[valid] &= (
            ranges[valid] >= minimum_range
        ) & (ranges[valid] <= maximum_range)
        filtered = ranges.copy()
        if self.scan_median_window > 1:
            half_window = self.scan_median_window // 2
            neighbors = []
            for offset in range(-half_window, half_window + 1):
                shifted = np.roll(ranges, offset).copy()
                shifted_valid = np.roll(valid, offset)
                shifted[~shifted_valid] = np.nan
                neighbors.append(shifted)
            neighbor_stack = np.vstack(neighbors)
            valid_neighbors = np.count_nonzero(
                np.isfinite(neighbor_stack), axis=0
            )
            filtered = np.ma.median(
                np.ma.masked_invalid(neighbor_stack), axis=0
            ).filled(np.nan)
            filtered[
                valid_neighbors < (self.scan_median_window // 2 + 1)
            ] = np.nan

        valid = np.isfinite(filtered)
        valid[valid] &= (
            filtered[valid] >= minimum_range
        ) & (filtered[valid] <= maximum_range)
        scan_indices = np.flatnonzero(valid)
        raw_points = np.empty((0, 2), dtype=np.float64)
        if scan_indices.size:
            selected_ranges = filtered[scan_indices]
            selected_angles = angles[scan_indices]
            raw_points = np.column_stack(
                (
                    self.lidar_x + selected_ranges * np.cos(selected_angles),
                    self.lidar_y + selected_ranges * np.sin(selected_angles),
                )
            )

        groups = []
        current = []
        previous_index = None
        previous_point = None
        for scan_index, point in zip(scan_indices.tolist(), raw_points):
            connected = (
                previous_index is not None
                and scan_index == previous_index + 1
                and np.linalg.norm(point - previous_point)
                <= self.cluster_link_distance
            )
            if current and not connected:
                groups.append(current)
                current = []
            current.append(point)
            previous_index = scan_index
            previous_point = point
        if current:
            groups.append(current)

        dense = []
        for group in groups:
            if len(group) < self.cluster_minimum_points:
                continue
            dense.append(group[0])
            for first, second in zip(group[:-1], group[1:]):
                distance = float(np.linalg.norm(second - first))
                samples = max(
                    1, int(math.ceil(distance / self.obstacle_point_spacing))
                )
                for sample in range(1, samples + 1):
                    dense.append(
                        first + (float(sample) / samples) * (second - first)
                    )
        points = (
            np.asarray(dense, dtype=np.float64).reshape((-1, 2))
            if dense
            else np.empty((0, 2), dtype=np.float64)
        )
        scan = {
            "source_stamp": source_stamp,
            "received": received,
            "points": points,
        }
        with self.lock:
            if self.scan_stamp is not None and source_stamp <= self.scan_stamp:
                return
            # Enqueue before re-reading history under the same lock. If the
            # scan callback won the race, the next odom callback supplies the
            # right bracket; if odom won, this pass accepts it immediately.
            self._enqueue_pending_scan(scan, received)
            for ready_scan, aligned_pose in self._take_ready_scans(received):
                self._accept_scan(ready_scan, aligned_pose)

    def manual_stop_callback(self, message):
        with self.lock:
            self.manual_stop = bool(message.data)
            if self.manual_stop and self.mission_has_control:
                self._publish_stop()

    def _set_lane_controller(self, enabled):
        try:
            rospy.wait_for_service(self.lane_service_name, timeout=2.0)
            response = self.lane_service(enabled)
            if not response.success:
                raise rospy.ServiceException(response.message)
            self.mission_has_control = not enabled
            self.handoff_ambiguous = False
            return True
        except (rospy.ROSException, rospy.ServiceException) as error:
            self.handoff_ambiguous = True
            rospy.logerr("Obstacle cmd_vel handoff failed: %s", error)
            return False

    def _stop_lane_controller(self):
        try:
            rospy.wait_for_service(self.lane_stop_service_name, timeout=1.0)
            response = self.lane_stop_service(False)
            if not response.success:
                raise rospy.ServiceException(response.message)
            return True
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logerr("Obstacle lane emergency stop failed: %s", error)
            return False

    def _start_run(self):
        now = rospy.Time.now()
        map_pose_problem = self._map_pose_problem(now)
        if map_pose_problem is not None:
            input_problem = map_pose_problem
        elif (
            not self.odom_ready
            or self.odom_stamp is None
            or (now - self.odom_stamp).to_sec() > self.odom_timeout
        ):
            input_problem = "odometry is unavailable or stale"
        elif self.map_from_odom is None:
            input_problem = "AMCL pose has no source-stamp-matched odometry"
        else:
            input_problem = None
        if input_problem is not None:
            rospy.logwarn_throttle(
                2.0,
                "Waiting for fresh odometry and AMCL pose before obstacle "
                "acquisition: %s",
                input_problem,
            )
            return
        self.start_requested = False
        self.revoke_requested = False
        self.speed_limit_pub.publish(Float64(data=self.entry_velocity_cap))
        course_origin_in_odom = self.map_from_odom.inverse().apply_pose(
            Pose2D(self.spline_anchor_x, self.spline_anchor_y, self.course_heading)
        )
        self.course_odom_heading = course_origin_in_odom.yaw
        self.odom_from_course = None
        self.committed_path = None
        self.path_follower = PathFollower(self.tracking_config)
        self.fixed_path_validation = None
        self.remaining_distance = 0.0
        self.live_speed_limit = math.inf
        self.last_plan_scan_generation = -1
        self.planning_seconds = 0.0
        self.last_command_time = None
        self._set_state(self.ACQUIRING)

    def _revoke(self):
        if self.mission_has_control:
            self._publish_stop()
            if not self._set_lane_controller(True):
                self._fail("lane-controller handoff failed after gate closed")
                return
        self.speed_limit_pub.publish(Float64(data=self.lane_resume_max_velocity))
        self.revoke_requested = False
        self.committed_path = None
        self.path_follower = PathFollower(self.tracking_config)
        self.fixed_path_validation = None
        self._set_state(self.WAIT_GATE)

    def _fail(self, reason):
        self.planner_status_pub.publish(String(data="FAILED"))
        owns_unambiguously = self.mission_has_control and not self.handoff_ambiguous
        if not owns_unambiguously:
            owns_unambiguously = self._set_lane_controller(False)
        if owns_unambiguously:
            self._publish_stop()
        else:
            # Service failure leaves ownership indeterminate. Never publish a
            # competing Twist. Relinquish direct output, stop the lane publisher
            # through its independent service, then pause every mission owner.
            self.mission_has_control = False
            if not self._stop_lane_controller():
                rospy.logfatal("Obstacle could not establish a sole cmd_vel stop owner")
            self.emergency_stop_pub.publish(Bool(data=True))
        self._set_state(self.FAILED)
        rospy.logerr("Obstacle mission failed: %s", reason)

    def _publish_stop(self):
        if self.mission_has_control:
            self.cmd_pub.publish(Twist())
        self.path_follower.last_linear = 0.0
        self.path_follower.last_angular = 0.0
        self.path_follower.diagnostics.commanded_linear = 0.0
        self.path_follower.diagnostics.commanded_angular = 0.0
        self.last_command_time = rospy.Time.now()

    def _current_pose_in_map(self):
        if self.map_from_odom is None:
            return Pose2D(self.map_x, self.map_y, self.map_yaw)
        return self.map_from_odom.apply_pose(self._current_pose())

    def _course_heading_in_odom(self):
        odom_from_course = getattr(self, "odom_from_course", None)
        if odom_from_course is not None:
            return odom_from_course.target_from_source_yaw
        return self.course_odom_heading

    def _course_heading_error(self):
        course_heading = self._course_heading_in_odom()
        if course_heading is not None:
            return normalize_angle(course_heading - self.odom_yaw)
        return normalize_angle(
            self.course_heading - self._current_pose_in_map().yaw
        )

    def _course_template_coordinates(self):
        """Return (progress, path lateral offset) through one frame transform."""
        odom_from_course = getattr(self, "odom_from_course", None)
        if odom_from_course is not None:
            course_pose = odom_from_course.inverse().apply_pose(
                self._current_pose()
            )
            return course_pose.x, -course_pose.y
        if not getattr(self, "map_ready", False):
            return None
        course_pose = self.course_from_map.apply_pose(
            self._current_pose_in_map()
        )
        return course_pose.x, -course_pose.y

    def _freeze_course_alignment(self, progress, lateral):
        """Freeze the LiDAR-aligned course frame in odom coordinates."""
        course_heading = self._course_heading_in_odom()
        robot_in_course = Pose2D(
            progress,
            lateral,
            normalize_angle(self.odom_yaw - course_heading),
        )
        self.odom_from_course = RigidTransform2D.from_pose_pair(
            robot_in_course,
            self._current_pose(),
            source_frame="obstacle_course",
            target_frame=self.odom_frame,
        )
        self.course_odom_heading = None

    def _attach_map_safety_boundary(self, path):
        """Freeze the surveyed map extent into the executable odom path."""
        if self.map_from_odom is None:
            raise ValueError("map-to-odom transform is unavailable")
        map_boundary_odom = self.map_safety_boundary.transformed(
            self.map_from_odom.inverse()
        )
        safety = path.safety
        path.safety = PathSafety(
            line_boundaries=safety.line_boundaries,
            map_boundaries=(*safety.map_boundaries, map_boundary_odom),
            fixed_obstacles=safety.fixed_obstacles,
            margins=safety.margins,
        )
        return path

    def _camera_corridor(self, now):
        if (
            self.line_center_absolute is None
            or self.line_observation_stamp is None
            or (now - self.line_observation_stamp).to_sec() > self.camera_timeout
        ):
            return None
        course_heading = self._course_heading_in_odom()
        robot_absolute = (
            -math.sin(course_heading) * self.odom_x
            + math.cos(course_heading) * self.odom_y
        )
        center = self.line_center_absolute - robot_absolute
        return (
            center - 0.5 * self.lane_width,
            center + 0.5 * self.lane_width,
            center,
        )

    def _surveyed_corridor(self, course_coordinates=None):
        if course_coordinates is None:
            course_coordinates = self._course_template_coordinates()
        if course_coordinates is None:
            return None
        center = course_coordinates[1]
        return (
            center - 0.5 * self.lane_width,
            center + 0.5 * self.lane_width,
            center,
        )

    def _planning_corridor(self, now, course_coordinates=None):
        surveyed = self._surveyed_corridor(course_coordinates)
        camera = self._camera_corridor(now)
        if surveyed is None:
            return camera
        if (
            camera is not None
            and abs(camera[2] - surveyed[2]) <= self.map_corridor_max_residual
        ):
            return camera
        return surveyed

    def _lane_points(self, heading_error, corridor):
        points = self.scan_points
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        if self.scan_odom_pose is not None:
            scan_x, scan_y, scan_yaw = self.scan_odom_pose
            scan_cosine = math.cos(scan_yaw)
            scan_sine = math.sin(scan_yaw)
            odom_x = scan_x + scan_cosine * points[:, 0] - scan_sine * points[:, 1]
            odom_y = scan_y + scan_sine * points[:, 0] + scan_cosine * points[:, 1]
            dx = odom_x - self.odom_x
            dy = odom_y - self.odom_y
            current_cosine = math.cos(self.odom_yaw)
            current_sine = math.sin(self.odom_yaw)
            points = np.column_stack(
                (
                    current_cosine * dx + current_sine * dy,
                    -current_sine * dx + current_cosine * dy,
                )
            )
        cosine = math.cos(heading_error)
        sine = math.sin(heading_error)
        lane_points = np.column_stack(
            (
                cosine * points[:, 0] + sine * points[:, 1],
                -sine * points[:, 0] + cosine * points[:, 1],
            )
        )
        right, left, _ = corridor
        usable = (
            (lane_points[:, 0] >= -self.scan_rear_limit)
            & (lane_points[:, 0] <= self.scan_max_range)
            & (lane_points[:, 1] >= right - 0.12)
            & (lane_points[:, 1] <= left + 0.12)
        )
        return lane_points[usable]

    def _current_pose(self):
        return Pose2D(self.odom_x, self.odom_y, self.odom_yaw)

    def _lane_points_in_odom(self, lane_points, heading_error):
        """Express the time-aligned LiDAR points in the frozen odom frame."""
        if lane_points.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        course_heading = self.odom_yaw + heading_error
        cosine = math.cos(course_heading)
        sine = math.sin(course_heading)
        return np.column_stack(
            (
                self.odom_x + cosine * lane_points[:, 0] - sine * lane_points[:, 1],
                self.odom_y + sine * lane_points[:, 0] + cosine * lane_points[:, 1],
            )
        )

    def _acquisition_data_problem(self, now):
        checks = (
            (self.scan_stamp, self.scan_timeout, "scan"),
            (self.odom_stamp, self.odom_timeout, "odometry"),
        )
        map_pose_problem = self._map_pose_problem(now)
        if map_pose_problem is not None:
            return map_pose_problem
        if self.map_from_odom is None:
            return "AMCL map-pose/odometry timestamp alignment"
        for stamp, timeout, name in checks:
            if stamp is None or (now - stamp).to_sec() > timeout:
                return name
        return None

    def _attempt_path(self, now):
        if self.odom_from_course is None:
            map_pose_problem = self._map_pose_problem(now)
            if map_pose_problem is not None or self.map_from_odom is None:
                self.planner_status_pub.publish(String(data="ALIGNING"))
                rospy.logwarn_throttle(
                    1.0,
                    "Obstacle alignment waits for a fresh AMCL pose: %s",
                    map_pose_problem or "map-to-odom alignment is unavailable",
                )
                return False
        if self.scan_generation == self.last_plan_scan_generation:
            return False
        self.last_plan_scan_generation = self.scan_generation
        heading_error = self._course_heading_error()
        course_coordinates = self._course_template_coordinates()
        if course_coordinates is None:
            return False
        progress, lateral_offset = course_coordinates
        corridor = self._planning_corridor(now, course_coordinates)
        if corridor is None:
            return False
        lane_points = self._lane_points(heading_error, corridor)
        started = time.monotonic()

        if self.odom_from_course is None:
            alignment = self.spline_planner.align_pose(
                progress,
                -lateral_offset,
                lane_points,
            )
            if alignment is None:
                self.planner_status_pub.publish(String(data="ALIGNING"))
                return False
            map_pose_problem = self._map_pose_problem(rospy.Time.now())
            if map_pose_problem is not None or self.map_from_odom is None:
                # Alignment can be non-trivial; recheck at the irreversible
                # odom-frame latch instead of trusting the earlier gate check.
                self.planner_status_pub.publish(String(data="ALIGNING"))
                rospy.logwarn_throttle(
                    1.0,
                    "Discarding obstacle alignment without fresh AMCL pose: %s",
                    map_pose_problem or "map-to-odom alignment is unavailable",
                )
                return False
            self._freeze_course_alignment(alignment.progress, alignment.lateral)
            progress = alignment.progress
            lateral_offset = -alignment.lateral
            corridor = (
                lateral_offset - 0.5 * self.lane_width,
                lateral_offset + 0.5 * self.lane_width,
                lateral_offset,
            )
            rospy.loginfo(
                "LiDAR course frame latched: progress=%.3fm lateral=%.3fm "
                "inliers=%d rms=%.4fm",
                alignment.progress,
                alignment.lateral,
                alignment.inliers,
                alignment.rms,
            )

        path = self.spline_planner.plan(
            progress,
            lateral_offset,
            # Route selection never detours around a transient return.  The
            # same scan is checked immediately after takeover by motion_safety.
            np.empty((0, 2), dtype=np.float64),
            corridor[0],
            corridor[1],
        )
        if path is None:
            self.planner_status_pub.publish(String(data="PATH_REJECTED"))
            return False

        current_heading = normalize_angle(-heading_error)
        connector = CommonPath(
            x=np.asarray([0.0, float(path.x[0])], dtype=np.float64),
            y=np.asarray([0.0, float(path.y[0])], dtype=np.float64),
            heading=np.asarray(
                [current_heading, float(path.heading[0])], dtype=np.float64
            ),
            curvature=np.zeros(2, dtype=np.float64),
            frame_id=path.frame_id,
        )
        connection_position = math.hypot(float(path.x[0]), float(path.y[0]))
        connection_heading = abs(
            normalize_angle(float(path.heading[0]) - current_heading)
        )
        if (
            connection_position > self.tracking_position_tolerance
            or connection_heading > self.tracking_heading_tolerance
            or not self.path_checker.validator.validate_path(
                connector,
                safety=self.path_checker.safety(
                    np.empty((0, 2), dtype=np.float64),
                    corridor[0],
                    corridor[1],
                    self.validation_footprint,
                ),
            ).safe
        ):
            self.planner_status_pub.publish(String(data="PATH_REJECTED"))
            return False

        self.committed_path, _ = freeze_path_in_odom(
            path,
            source_pose=Pose2D(0.0, 0.0, current_heading),
            odom_pose=self._current_pose(),
            odom_frame=self.odom_frame,
        )
        self._attach_map_safety_boundary(self.committed_path)
        fixed_validation = self.path_checker.validator.validate_path(
            self.committed_path
        )
        if not fixed_validation.safe:
            self.committed_path = None
            self.path_follower = PathFollower(self.tracking_config)
            self.planner_status_pub.publish(String(data="PATH_REJECTED"))
            return False
        self.fixed_path_validation = fixed_validation
        projection = self.path_follower.reset(
            self.committed_path,
            self._current_pose(),
            initial_linear=min(
                max(0.0, self.observed_lane_linear), self.entry_velocity_cap
            ),
            initial_angular=clamp(
                self.observed_lane_angular,
                -self.maximum_angular_velocity,
                self.maximum_angular_velocity,
            ),
        )
        self.remaining_distance = max(
            0.0, self.committed_path.length - projection.station
        )
        self.path_follower.update_clearance(fixed_validation)
        self.committed_path.line_clearance = (
            fixed_validation.minimum_line_clearance
        )
        self.committed_path.obstacle_clearance = (
            fixed_validation.minimum_obstacle_clearance
        )
        self.committed_path.map_clearance = (
            fixed_validation.minimum_map_clearance
        )
        self.live_speed_limit = math.inf
        self.planning_seconds = time.monotonic() - started
        self._publish_path()
        self._publish_diagnostics()
        self.planner_status_pub.publish(String(data="PATH_COMMITTED"))
        rospy.loginfo(
            "Committed the single obstacle path in %.3fs: "
            "remaining=%.3fm clearance=%.4fm line=%.4fm",
            self.planning_seconds,
            self.remaining_distance,
            self.committed_path.obstacle_clearance,
            self.committed_path.line_clearance,
        )
        return True

    def _validate_committed_path(self, now):
        # Assemble one coherent sensor/path snapshot under the mission lock.
        # The geometric sweep is intentionally evaluated after releasing it:
        # otherwise its CPU time starves the 30 Hz odometry callback and makes
        # otherwise synchronized 10 Hz scans appear stale.
        with self.lock:
            if self.committed_path is None or self.path_follower.path is None:
                self._fail("no frozen path is active")
                return False
            stale_inputs = []
            if (
                self.odom_stamp is None
                or (now - self.odom_stamp).to_sec() > self.odom_timeout
            ):
                stale_inputs.append("odometry")
            if (
                self.scan_stamp is None
                or (now - self.scan_stamp).to_sec() > self.scan_timeout
            ):
                stale_inputs.append("LiDAR")
            if stale_inputs:
                self._publish_stop()
                rospy.logwarn_throttle(
                    1.0,
                    "Obstacle holds zero for stale swept-safety input: %s",
                    ", ".join(stale_inputs),
                )
                return False
            path = self.committed_path
            follower = self.path_follower
            fixed_validation = self.fixed_path_validation
            odom_stamp = self.odom_stamp
            scan_stamp = self.scan_stamp
            pose = self._current_pose()
            heading_error = self._course_heading_error()
            course_coordinates = self._course_template_coordinates()
            corridor = self._planning_corridor(now, course_coordinates)
            if corridor is None:
                self._fail("lane corridor unavailable")
                return False
            lane_points = self._lane_points(heading_error, corridor)
            if course_coordinates is not None:
                progress, lateral_offset = course_coordinates
                lane_points = self.spline_planner.stabilize_known_barrier_returns(
                    lane_points,
                    progress,
                    -lateral_offset,
                )
            live_obstacles = self._lane_points_in_odom(
                lane_points, heading_error
            )
            tracking = follower.calculate_tracking(pose)
            measured_speed = max(
                abs(self.odom_linear_velocity),
                abs(follower.last_linear),
            )
            angular_velocities = (
                self.odom_angular_velocity,
                follower.last_angular,
                tracking.angular_velocity,
            )
            line_boundaries = path.safety.line_boundaries
            camera_corridor = self._camera_corridor(now)
            if (
                camera_corridor is not None
                and abs(camera_corridor[2] - corridor[2])
                <= self.map_corridor_max_residual
            ):
                line_boundaries += (
                    StraightCorridorBoundary(
                        camera_corridor[0],
                        camera_corridor[1],
                        origin=(pose.x, pose.y),
                        heading=pose.yaw + heading_error,
                    ),
                )
            runtime_safety = PathSafety(
                line_boundaries=line_boundaries,
                map_boundaries=path.safety.map_boundaries,
                fixed_obstacles=path.safety.fixed_obstacles,
                margins=SafetyMargins(
                    line=getattr(
                        self,
                        "runtime_line_margin",
                        self.validation_footprint.line_margin,
                    ),
                    obstacle=getattr(
                        self,
                        "runtime_obstacle_padding",
                        self.footprint.obstacle_padding,
                    ),
                    localization=getattr(
                        self,
                        "runtime_localization_margin",
                        self.validation_footprint.localization_margin,
                    ),
                    tracking=getattr(
                        self,
                        "runtime_tracking_margin",
                        self.validation_footprint.tracking_margin,
                    ),
                ),
            )
            validator = self.path_checker.validator
            live_validation_distance = (
                self.spline_planner.live_validation_distance
            )

        safety = validator.motion_safety(
            path,
            pose,
            tracking.path_index,
            tracking.target_speed,
            tracking.direction * measured_speed,
            angular_velocities,
            self.safety_reaction_time,
            self.linear_deceleration,
            self.safety_stop_margin,
            lookahead_distance=live_validation_distance,
            safety=runtime_safety,
            live_obstacles=live_obstacles,
            tracking=tracking,
        )

        with self.lock:
            # A gate/restart can replace either object while the pure geometry
            # call runs. Never apply a result to a different route/follower.
            if self.committed_path is not path or self.path_follower is not follower:
                return False
            completed = rospy.Time.now()
            stale_inputs = []
            if (completed - odom_stamp).to_sec() > self.odom_timeout:
                stale_inputs.append("odometry")
            if (completed - scan_stamp).to_sec() > self.scan_timeout:
                stale_inputs.append("LiDAR")
            if stale_inputs:
                self._publish_stop()
                rospy.logwarn_throttle(
                    1.0,
                    "Obstacle holds zero after swept-safety input aged out: %s",
                    ", ".join(stale_inputs),
                )
                return False

            validation = combine_validation_results(
                (fixed_validation, safety.validation)
            )
            self.live_speed_limit = safety.speed_limit
            follower.update_clearance(validation)
            path.obstacle_clearance = validation.minimum_obstacle_clearance
            path.line_clearance = validation.minimum_line_clearance
            path.map_clearance = validation.minimum_map_clearance
            self.remaining_distance = max(0.0, path.length - tracking.station)

            if safety.requires_stop:
                rospy.logwarn_throttle(
                    0.5,
                    "Obstacle path holds a bounded stop: line=%.4fm obstacle=%.4fm",
                    validation.minimum_line_clearance,
                    validation.minimum_obstacle_clearance,
                )
            elif safety.speed_limit + 1e-9 < tracking.target_speed:
                rospy.logwarn_throttle(
                    0.5,
                    "Obstacle path slows for predicted contact: limit=%.3fm/s "
                    "line=%.4fm obstacle=%.4fm",
                    safety.speed_limit,
                    validation.minimum_line_clearance,
                    validation.minimum_obstacle_clearance,
                )
            self._publish_diagnostics()
            return tracking, pose

    def _publish_path(self):
        message = Path()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.odom_frame
        for x, y, yaw in zip(
            self.committed_path.x,
            self.committed_path.y,
            self.committed_path.heading,
        ):
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.055
            pose.pose.orientation.z = math.sin(0.5 * float(yaw))
            pose.pose.orientation.w = math.cos(0.5 * float(yaw))
            message.poses.append(pose)
        self.path_pub.publish(message)

    def _publish_diagnostics(self):
        message = Float64MultiArray()
        message.data = self.path_follower.diagnostics.as_array()
        self.diagnostics_pub.publish(message)

    def _common_path_command(self, now, tracking=None, pose=None):
        if self.committed_path is None or self.path_follower.path is None:
            return None
        pose = self._current_pose() if pose is None else Pose2D.from_value(pose)
        if tracking is None:
            tracking = self.path_follower.calculate_tracking(pose)
        if self.last_command_time is None:
            elapsed = self.control_period
        else:
            elapsed = clamp((now - self.last_command_time).to_sec(), 0.0, 0.15)
        limited, tracking = self.path_follower.command(
            pose,
            elapsed,
            speed_limit=self.live_speed_limit,
            tracking=tracking,
        )
        self.remaining_distance = self.path_follower.diagnostics.remaining_distance
        self.last_command_time = now
        command = Twist()
        command.linear.x = limited.linear_velocity
        command.angular.z = limited.angular_velocity
        self._publish_diagnostics()
        return command

    def _terminal_hold_command(self, now, tracking, pose):
        """Bounded stop while the AMCL-owned handoff signal catches up."""
        if self.last_command_time is None:
            elapsed = self.control_period
        else:
            elapsed = clamp((now - self.last_command_time).to_sec(), 0.0, 0.15)
        limited, _tracking, _status = self.path_follower.terminal_hold(
            pose,
            elapsed,
            tracking=tracking,
        )
        self.remaining_distance = self.path_follower.diagnostics.remaining_distance
        self.last_command_time = now
        command = Twist()
        command.linear.x = limited.linear_velocity
        command.angular.z = limited.angular_velocity
        self._publish_diagnostics()
        return command

    def _complete(self):
        self.planner_status_pub.publish(String(data="COMPLETE"))
        self.speed_limit_pub.publish(Float64(data=self.lane_resume_max_velocity))
        if not self._set_lane_controller(True):
            self._fail("could not return cmd_vel to lane controller")
            return
        self._set_state(self.COMPLETE)
        rospy.loginfo("Obstacle course complete; lane controller owns cmd_vel")

    def control_callback(self, _event):
        with self.lock:
            now = rospy.Time.now()
            if self.shutting_down:
                return
            if self.revoke_requested:
                self._revoke()
                return
            if self.start_requested and self.zone_gate and self.state in (
                self.WAIT_GATE,
                self.COMPLETE,
            ):
                self._start_run()

            if self.state in (self.WAIT_GATE, self.COMPLETE):
                return
            if self.manual_stop:
                if self.mission_has_control:
                    self._publish_stop()
                return
            if self.state == self.FAILED:
                self._publish_stop()
                return

            if self.state == self.ACQUIRING:
                if abs(self._course_heading_error()) > self.acquisition_heading_tolerance:
                    return
                if self.committed_path is None:
                    problem = self._acquisition_data_problem(now)
                    if problem is not None:
                        rospy.logwarn_throttle(
                            2.0,
                            "Waiting for obstacle acquisition input: %s",
                            problem,
                        )
                        return
                    if not self._attempt_path(now):
                        return
                    # Alignment, planning and full-route validation are
                    # intentionally completed before ownership transfer. They
                    # can exceed one sensor timeout, so let callbacks refresh
                    # once and validate the current join pose on the next tick.
                    return

                now = rospy.Time.now()
                problem = self._acquisition_data_problem(now)
                if problem is not None:
                    rospy.logwarn_throttle(
                        1.0,
                        "Committed obstacle path waits for refreshed input: %s",
                        problem,
                    )
                    return
                current_pose = self._current_pose()
                projection = self.path_follower.reset(
                    self.committed_path,
                    current_pose,
                    initial_linear=min(
                        max(0.0, self.observed_lane_linear),
                        self.entry_velocity_cap,
                    ),
                    initial_angular=clamp(
                        self.observed_lane_angular,
                        -self.maximum_angular_velocity,
                        self.maximum_angular_velocity,
                    ),
                )
                join_heading_error = abs(
                    normalize_angle(projection.heading - current_pose.yaw)
                )
                if (
                    projection.distance > self.tracking_position_tolerance
                    or join_heading_error > self.tracking_heading_tolerance
                ):
                    rospy.logwarn(
                        "Discarding obstacle path after acquisition motion: "
                        "join=%.4fm/%.2fdeg",
                        projection.distance,
                        math.degrees(join_heading_error),
                    )
                    self.committed_path = None
                    self.path_follower = PathFollower(self.tracking_config)
                    self.fixed_path_validation = None
                    self.planner_status_pub.publish(String(data="PATH_REJECTED"))
                    return
                self.remaining_distance = max(
                    0.0,
                    self.committed_path.length - projection.station,
                )
                if not self._set_lane_controller(False):
                    self._fail("could not acquire cmd_vel control")
                    return
                self.path_follower.diagnostics.commanded_linear = (
                    self.path_follower.last_linear
                )
                self.path_follower.diagnostics.commanded_angular = (
                    self.path_follower.last_angular
                )
                self.last_command_time = rospy.Time.now()
                self._set_state(self.AVOIDING)
                now = rospy.Time.now()

        validated = self._validate_committed_path(now)
        if not validated:
            return
        tracking, tracking_pose = validated

        with self.lock:
            # Sensor and zone callbacks may run during the unlocked geometric
            # sweep. Honour any safety/ownership change before publishing.
            if self.shutting_down:
                return
            if self.revoke_requested:
                self._revoke()
                return
            if self.manual_stop:
                if self.mission_has_control:
                    self._publish_stop()
                return
            if self.state == self.FAILED or not self.mission_has_control:
                return
            if (
                self.state == self.AVOIDING
                and not self.zone_inside
            ):
                self._set_state(self.REJOINING)

            goal = self.path_follower.goal_status(
                tracking_pose, tracking=tracking
            )
            self.remaining_distance = goal.remaining_distance
            if (
                self.state == self.REJOINING
                and self.zone_cleared
                and goal.complete
            ):
                self._complete()
                return
            if (
                goal.crossed_terminal
                and goal.position_error
                > self.committed_path.goal_tolerance.terminal_crossing
            ):
                self._fail("fixed path ended before AMCL polygon clearance")
                return
            if goal.crossed_terminal:
                # Crossing the selected CommonPath ends translation even if a
                # heading correction or delayed AMCL polygon/state heartbeat
                # still prevents handoff.  The common hold retains only the
                # bounded endpoint-heading correction.
                self.cmd_pub.publish(
                    self._terminal_hold_command(
                        rospy.Time.now(), tracking, tracking_pose
                    )
                )
                return

            command = self._common_path_command(
                rospy.Time.now(), tracking=tracking, pose=tracking_pose
            )
            if command is None:
                self._fail("fixed path ended before AMCL polygon clearance")
                return
            self.cmd_pub.publish(command)

    def shutdown(self):
        with self.lock:
            self.shutting_down = True
            if self.mission_has_control:
                self.cmd_pub.publish(Twist())
        rospy.loginfo("Obstacle mission controller stopped")


if __name__ == "__main__":
    rospy.init_node("obstacle_mission_controller")
    ObstacleMissionController()
    rospy.spin()
