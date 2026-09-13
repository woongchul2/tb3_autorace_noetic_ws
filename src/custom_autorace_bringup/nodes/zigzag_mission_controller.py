#!/usr/bin/env python3
"""Follow one surveyed zigzag path with lookahead and curvature pre-braking."""

from collections import deque
import math
import os
import threading

import cv2
import numpy as np
import rospkg
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from std_srvs.srv import SetBool

from custom_autorace_bringup.path_following import (
    AsymmetricFootprint,
    CallbackBoundary,
    CommonPath,
    GoalTolerance,
    PathFollower,
    PathSafety,
    Pose2D,
    SafetyMargins,
    SpeedProfile,
    SweptFootprintValidator,
    TrackingConfig,
    clamp,
    freeze_path_in_odom,
    normalize_angle,
    yaw_from_quaternion,
)
from custom_autorace_bringup.zigzag_path import (
    RasterPaintCorridorChecker,
    SurveyedCorridorChecker,
    build_zigzag_path,
)


class ZigzagMissionController:
    """Own ``cmd_vel`` only between the ordered zigzag gate and lane exit."""

    # Keep enough 30 Hz samples to bracket LaserScan stamps without letting a
    # costly swept-validation tick leave a long queue of obsolete odometry for
    # the next control tick.  The callback stores the longer time history.
    ODOM_SUBSCRIBER_QUEUE_SIZE = 8

    WAIT_GATE = "WAIT_GATE"
    ACQUIRING = "ACQUIRING"
    FOLLOWING = "FOLLOWING"
    VERIFY_EXIT = "VERIFY_EXIT"
    JOINING_LANE = "JOINING_LANE"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

    def __init__(self):
        get = rospy.get_param
        p = "~zigzag/"

        self.odom_topic = str(get(p + "topics/odometry", "/odom"))
        self.scan_topic = str(
            get(p + "topics/scan", "/scan_mid360_raw")
        )
        self.map_pose_topic = str(
            get(p + "topics/map_pose", "/mission/map_pose")
        )
        self.gate_topic = str(
            get(p + "topics/zone_gate", "/mission/enable/zigzag")
        )
        self.boundary_topic = str(
            get(p + "topics/lane_boundaries", "/detect/lane_boundaries")
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

        self.route_odom_aligned = bool(
            get(p + "route/odom_aligned", False)
        )
        self.route_frame = str(
            get(
                p + "route/frame_id",
                "odom" if self.route_odom_aligned else "map",
            )
        )
        self.start_maximum_path_error = max(
            0.005, float(get(p + "start/maximum_path_error", 0.018))
        )
        self.start_maximum_join_heading_error = math.radians(
            abs(
                float(
                    get(p + "start/maximum_join_heading_error_deg", 7.0)
                )
            )
        )
        self.maximum_pose_stamp_skew = max(
            0.005, float(get(p + "start/maximum_pose_stamp_skew", 0.04))
        )
        self.scan_frame = str(get(p + "scan/frame_id", "base_scan")).lstrip(
            "/"
        )
        self.scan_maximum_range = max(
            0.10, float(get(p + "scan/maximum_range", 1.0))
        )
        self.scan_pose_stamp_skew = max(
            0.005,
            float(
                get(
                    p + "scan/maximum_pose_stamp_skew",
                    self.maximum_pose_stamp_skew,
                )
            ),
        )
        self.odom_history_duration = max(
            2.0 * self.scan_pose_stamp_skew,
            float(get(p + "scan/odom_history_duration", 1.0)),
        )
        self.pending_scan_limit = max(
            1, int(get(p + "scan/pending_scan_limit", 4))
        )
        self.lidar_x = float(get(p + "scan/lidar_x", -0.033073))
        self.lidar_y = float(get(p + "scan/lidar_y", 0.0))
        self.lidar_yaw = float(get(p + "scan/lidar_yaw", 0.0))
        self.front = max(0.01, float(get(p + "footprint/front", 0.067645)))
        self.rear = max(0.01, float(get(p + "footprint/rear", 0.118073)))
        self.half_width = max(
            0.01, float(get(p + "footprint/half_width", 0.0903))
        )
        self.footprint_sample_spacing = max(
            0.002,
            float(get(p + "footprint/validation_sample_spacing", 0.004)),
        )
        self.footprint = AsymmetricFootprint(
            self.front,
            self.rear,
            self.half_width,
        )
        self.safety_margins = SafetyMargins(
            line=max(0.0, float(get(p + "footprint/line_margin", 0.0))),
            obstacle=max(
                0.0, float(get(p + "footprint/obstacle_margin", 0.0))
            ),
            localization=max(
                0.0, float(get(p + "footprint/localization_error", 0.0))
            ),
            tracking=max(
                0.0, float(get(p + "footprint/tracking_error", 0.0))
            ),
        )
        # The immutable spline and its fixed paint/map boundaries are swept
        # once when the executable path is committed.  At control rate, the
        # nominal lookahead only needs newly observed LiDAR points; the
        # actual reaction-and-full-stop sweep still uses candidate.safety and
        # therefore retains every fixed boundary and uncertainty margin.
        self.live_route_safety = PathSafety(margins=self.safety_margins)
        self.path_validator = SweptFootprintValidator(
            self.footprint,
            translation_step=max(
                0.001,
                float(
                    get(
                        p + "footprint/sweep_translation_step",
                        self.footprint_sample_spacing,
                    )
                ),
            ),
            heading_step=math.radians(
                max(
                    0.1,
                    float(get(p + "footprint/sweep_heading_step_deg", 1.0)),
                )
            ),
        )
        self.inner_paint_blocking = bool(
            get(p + "validation/inner_paint_blocking", True)
        )

        self.control_period = max(
            0.02, float(get(p + "control/period", 0.05))
        )
        self.entry_velocity_cap = max(
            0.0, float(get(p + "control/entry_velocity_cap", 0.06))
        )
        self.lane_resume_max_velocity = max(
            self.entry_velocity_cap,
            float(get(p + "control/lane_resume_max_velocity", 0.30)),
        )
        self.cruise_velocity = max(
            0.01, float(get(p + "control/cruise_velocity", 0.14))
        )
        self.minimum_velocity = clamp(
            float(get(p + "control/minimum_velocity", 0.05)),
            0.005,
            self.cruise_velocity,
        )
        self.entry_velocity = clamp(
            float(get(p + "control/entry_velocity", 0.06)),
            self.minimum_velocity,
            self.cruise_velocity,
        )
        self.exit_velocity = clamp(
            float(get(p + "control/exit_velocity", 0.07)),
            self.minimum_velocity,
            self.cruise_velocity,
        )
        self.maximum_angular_velocity = max(
            0.05, float(get(p + "control/maximum_angular_velocity", 0.80))
        )
        self.maximum_lateral_acceleration = max(
            0.005,
            float(get(p + "control/maximum_lateral_acceleration", 0.035)),
        )
        self.linear_acceleration = max(
            0.005, float(get(p + "control/linear_acceleration", 0.04))
        )
        self.linear_deceleration = max(
            0.005, float(get(p + "control/linear_deceleration", 0.12))
        )
        self.angular_acceleration = max(
            0.05, float(get(p + "control/angular_acceleration", 0.80))
        )
        self.lookahead_distance = max(
            0.02, float(get(p + "control/lookahead_distance", 0.065))
        )
        self.heading_gain = float(get(p + "control/heading_gain", 0.35))
        self.path_curvature_weight = clamp(
            float(get(p + "control/path_curvature_weight", 0.25)), 0.0, 1.0
        )
        self.lateral_feedback_gain = max(
            0.0, float(get(p + "control/lateral_feedback_gain", 1.0))
        )
        self.nearest_search_ahead = max(
            0.05, float(get(p + "control/nearest_search_ahead", 0.25))
        )

        self.speed_profile = SpeedProfile(
            cruise_velocity=self.cruise_velocity,
            minimum_velocity=self.minimum_velocity,
            entry_velocity=self.entry_velocity,
            exit_velocity=self.exit_velocity,
            maximum_angular_velocity=self.maximum_angular_velocity,
            maximum_lateral_acceleration=self.maximum_lateral_acceleration,
            linear_acceleration=self.linear_acceleration,
            linear_deceleration=self.linear_deceleration,
            angular_acceleration=self.angular_acceleration,
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
            curvature_feedforward_weight=self.path_curvature_weight,
            lateral_feedback_gain=self.lateral_feedback_gain,
            search_back=3,
            search_ahead_distance=self.nearest_search_ahead,
        )
        self.path_follower = PathFollower(self.tracking_config)

        start_heading = math.radians(
            float(get(p + "path/start_heading_deg", 180.0))
        )
        end_heading = math.radians(
            float(get(p + "path/end_heading_deg", 180.0))
        )
        self.map_path = build_zigzag_path(
            knots=get(p + "path/knots", []),
            start_heading=start_heading,
            end_heading=end_heading,
            sample_spacing=float(get(p + "path/sample_spacing", 0.003)),
            cruise_velocity=self.cruise_velocity,
            minimum_velocity=self.minimum_velocity,
            entry_velocity=self.entry_velocity,
            exit_velocity=self.exit_velocity,
            maximum_angular_velocity=self.maximum_angular_velocity,
            maximum_lateral_acceleration=self.maximum_lateral_acceleration,
            linear_acceleration=self.linear_acceleration,
            linear_deceleration=self.linear_deceleration,
            guide_tail_length=float(get(p + "path/guide_tail_length", 0.0)),
            maximum_angular_acceleration=self.angular_acceleration,
            common_speed_profile=self.speed_profile,
            frame_id=self.route_frame,
            label="zigzag",
        )
        if not isinstance(self.map_path, CommonPath):
            raise rospy.ROSInitException("zigzag path is not in the common format")
        maximum_curvature = max(
            0.1, float(get(p + "path/maximum_curvature", 5.10))
        )
        actual_maximum_curvature = float(np.max(np.abs(self.map_path.curvature)))
        if actual_maximum_curvature > maximum_curvature + 1e-9:
            raise rospy.ROSInitException(
                "zigzag path curvature %.4f exceeds %.4f 1/m"
                % (actual_maximum_curvature, maximum_curvature)
            )
        self.corridor_checker = SurveyedCorridorChecker(
            get(p + "corridor/lower_boundary", []),
            get(p + "corridor/upper_boundary", []),
        )
        self.texture_enabled = bool(
            get(p + "texture/enabled", self.route_odom_aligned)
        )
        self.paint_checker = None
        self.minimum_outer_reserve = float(
            get(p + "texture/minimum_outer_reserve", 0.005)
        )
        if self.minimum_outer_reserve < 0.0:
            raise rospy.ROSInitException(
                "zigzag outer paint reserve must be non-negative"
            )

        if self.texture_enabled:
            texture_package = str(
                get(p + "texture/package", "turtlebot3_gazebo")
            )
            texture_relative_path = str(
                get(
                    p + "texture/relative_path",
                    "models/turtlebot3_autorace_2020/course/materials/"
                    "textures/course.png",
                )
            )
            try:
                texture_path = os.path.join(
                    rospkg.RosPack().get_path(texture_package),
                    texture_relative_path,
                )
                texture_bgr = cv2.imread(texture_path, cv2.IMREAD_COLOR)
                if texture_bgr is None:
                    raise IOError("could not read " + texture_path)
                texture_rgb = cv2.cvtColor(texture_bgr, cv2.COLOR_BGR2RGB)
                self.paint_checker = RasterPaintCorridorChecker(
                    texture_rgb,
                    float(get(p + "texture/course_size", 4.0)),
                    float(get(p + "texture/course_yaw", -3.14)),
                    get(p + "corridor/lower_boundary", []),
                    get(p + "corridor/upper_boundary", []),
                    self.front,
                    self.rear,
                    self.half_width,
                    self.footprint_sample_spacing,
                    float(get(p + "texture/boundary_x_step", 0.002)),
                    float(get(p + "texture/boundary_y_step", 0.0005)),
                    float(get(p + "texture/line_search_half_width", 0.080)),
                    int(get(p + "texture/color_threshold", 128)),
                    int(get(p + "texture/color_tolerance", 4)),
                    int(get(p + "texture/yellow_blue_maximum", 16)),
                    float(get(p + "texture/boundary_min_x", -1.58)),
                    float(get(p + "texture/boundary_max_x", 0.38)),
                )
            except (
                cv2.error,
                IOError,
                rospkg.ResourceNotFound,
                ValueError,
            ) as error:
                raise rospy.ROSInitException(
                    "zigzag paint texture validation failed: %s" % error
                )

        # One common swept validator owns the immutable-route decision. The
        # checker objects only expose signed clearance for each pose sampled
        # by that validator; they do not perform their own path sweep.
        self.route_from_odom = None
        self.map_path.safety = self._path_safety()
        nominal_validation = self.path_validator.validate_path(
            self.map_path,
        )
        if not nominal_validation.safe:
            raise rospy.ROSInitException(
                "zigzag common nominal sweep is unsafe: "
                "line=%.4fm obstacle=%.4fm map=%.4fm"
                % (
                    nominal_validation.minimum_line_clearance,
                    nominal_validation.minimum_obstacle_clearance,
                    nominal_validation.minimum_map_clearance,
                )
            )
        self.map_path.line_clearance = nominal_validation.minimum_line_clearance
        self.map_path.obstacle_clearance = (
            nominal_validation.minimum_obstacle_clearance
        )
        self.map_path.map_clearance = nominal_validation.minimum_map_clearance

        self.prediction_reaction_time = max(
            0.0,
            float(get(p + "texture/prediction_reaction_time", 0.10)),
        )
        self.prediction_distance_margin = max(
            0.0,
            float(get(p + "texture/prediction_distance_margin", 0.005)),
        )

        self.handoff_remaining_distance = max(
            0.001,
            float(get(p + "exit/handoff_remaining_distance", 0.005)),
        )
        self.exit_confirmation_lead_distance = max(
            0.0, float(get(p + "exit/confirmation_lead_distance", 0.030))
        )
        self.exit_position_tolerance = max(
            0.01, float(get(p + "exit/position_tolerance", 0.050))
        )
        self.exit_heading_tolerance = math.radians(
            abs(float(get(p + "exit/heading_tolerance_deg", 10.0)))
        )
        self.boundary_timeout = max(
            0.05, float(get(p + "exit/boundary_timeout", 0.35))
        )
        self.lane_width_min = float(get(p + "exit/lane_width_min", 450.0))
        self.lane_width_max = float(get(p + "exit/lane_width_max", 820.0))
        self.image_center = float(get(p + "exit/image_center", 500.0))
        self.single_line_center_offset = max(
            0.0,
            float(get(p + "exit/single_line_center_offset", 320.0)),
        )
        self.lane_center_tolerance = max(
            0.0, float(get(p + "exit/lane_center_tolerance", 180.0))
        )
        self.exit_confirmation_frames = max(
            1, int(get(p + "exit/confirmation_frames", 6))
        )
        self.exit_confirmation_max_gap = max(
            0.05, float(get(p + "exit/confirmation_max_gap", 0.20))
        )
        self.join_velocity_cap = max(
            0.0, float(get(p + "exit/join_velocity_cap", 0.07))
        )
        self.join_minimum_distance = max(
            0.0, float(get(p + "exit/join_minimum_distance", 0.035))
        )
        self.join_confirmation_frames = max(
            1, int(get(p + "exit/join_confirmation_frames", 6))
        )
        self.join_timeout = max(
            0.5, float(get(p + "exit/join_timeout", 1.5))
        )
        if not 0.0 < self.lane_width_min < self.lane_width_max:
            raise rospy.ROSInitException("zigzag exit lane width bounds are invalid")
        self.map_path.goal_tolerance = GoalTolerance(
            position=self.exit_position_tolerance,
            heading=self.exit_heading_tolerance,
            terminal_crossing=self.exit_position_tolerance,
        )

        self.map_timeout = max(
            0.05, float(get(p + "timeouts/map_pose", 0.35))
        )
        self.odom_timeout = max(
            0.05, float(get(p + "timeouts/odometry", 0.35))
        )
        self.scan_timeout = max(
            0.05, float(get(p + "timeouts/scan", 0.35))
        )
        self.acquisition_timeout = max(
            0.5, float(get(p + "timeouts/acquisition", 2.0))
        )
        self.mission_timeout = max(
            2.0, float(get(p + "timeouts/mission", 30.0))
        )
        self.exit_confirmation_timeout = max(
            0.5, float(get(p + "timeouts/exit_confirmation", 2.0))
        )

        self.lock = threading.RLock()
        self.state = self.WAIT_GATE
        self.state_started = rospy.Time.now()
        self.mission_started = None
        self.zone_gate = False
        self.start_requested = False
        self.revoke_requested = False
        self.mission_has_control = False
        self.handoff_ambiguous = False
        self.shutting_down = False
        self.manual_stop = False
        self.pause_started = None

        self.map_ready = False
        self.map_x = self.map_y = self.map_yaw = 0.0
        self.map_stamp = None
        self.odom_ready = False
        self.odom_x = self.odom_y = self.odom_yaw = 0.0
        self.odom_speed = 0.0
        self.odom_angular_velocity = 0.0
        self.odom_frame = "odom"
        self.odom_stamp = None
        self.odom_history = deque(maxlen=200)
        self.pending_scans = deque()
        self.live_obstacle_points = np.empty((0, 2), dtype=np.float64)
        self.scan_stamp = None
        self.scan_received = None
        self.scan_pose_stamp_delta = math.inf

        self.committed_path = None
        self.route_from_odom = None
        self.path_index = 0
        self.maximum_position_error_seen = 0.0
        self.maximum_heading_error_seen = 0.0
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.last_command_time = None
        self.observed_lane_linear = 0.0
        self.observed_lane_angular = 0.0

        self.boundary_stamp = None
        self.boundary_valid = False
        self.boundary_confirmation_count = 0
        self.last_boundary_confirmation_time = None
        self.exit_confirmation_started = False
        self.exit_stop_latched = False
        self.confirmation_started_at = None
        self.join_start_x = self.join_start_y = self.join_start_yaw = 0.0
        self.join_origin_ready = False

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.speed_limit_pub = rospy.Publisher(
            self.speed_limit_topic, Float64, queue_size=1, latch=True
        )
        self.emergency_stop_pub = rospy.Publisher(
            self.manual_stop_topic, Bool, queue_size=1, latch=True
        )
        self.state_pub = rospy.Publisher(
            "/zigzag/state", String, queue_size=1, latch=True
        )
        self.path_pub = rospy.Publisher(
            "/zigzag/path", Path, queue_size=1, latch=True
        )
        self.diagnostics_pub = rospy.Publisher(
            "/zigzag/diagnostics", Float64MultiArray, queue_size=1, latch=True
        )
        rospy.Subscriber(
            self.odom_topic,
            Odometry,
            self.odom_callback,
            queue_size=self.ODOM_SUBSCRIBER_QUEUE_SIZE,
        )
        rospy.Subscriber(
            self.scan_topic, LaserScan, self.scan_callback, queue_size=1
        )
        rospy.Subscriber(
            self.map_pose_topic, PoseStamped, self.map_pose_callback, queue_size=1
        )
        rospy.Subscriber(self.gate_topic, Bool, self.gate_callback, queue_size=1)
        rospy.Subscriber(
            self.boundary_topic,
            Float64MultiArray,
            self.boundary_callback,
            queue_size=1,
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
        self._publish_diagnostics()
        rospy.loginfo(
            "Zigzag controller ready: path=%.3fm max_curvature=%.3f/m "
            "line_clearance=%.4fm inner_blocking=%s",
            self.map_path.length,
            actual_maximum_curvature,
            self.map_path.line_clearance,
            self.inner_paint_blocking,
        )

    def _publish_state(self):
        self.state_pub.publish(String(data=self.state))

    def _set_state(self, state, now=None):
        if state == self.state:
            return
        self.state = state
        self.state_started = rospy.Time.now() if now is None else now
        self._publish_state()
        rospy.loginfo("Zigzag mission state: %s", state)

    def gate_callback(self, message):
        with self.lock:
            was_open = self.zone_gate
            self.zone_gate = bool(message.data)
            if self.zone_gate and not was_open:
                self.start_requested = True
            elif not self.zone_gate and was_open and self.state != self.COMPLETE:
                self.revoke_requested = True

    def map_pose_callback(self, message):
        received = rospy.Time.now()
        source_stamp = (
            message.header.stamp
            if message.header.stamp != rospy.Time()
            else received
        )
        with self.lock:
            if not all(
                math.isfinite(value)
                for value in (
                    message.pose.position.x,
                    message.pose.position.y,
                    yaw_from_quaternion(message.pose.orientation),
                )
            ):
                rospy.logerr_throttle(1.0, "Ignoring non-finite zigzag map pose")
                return
            self.map_x = float(message.pose.position.x)
            self.map_y = float(message.pose.position.y)
            self.map_yaw = yaw_from_quaternion(message.pose.orientation)
            self.map_stamp = source_stamp
            self.map_ready = True

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
            odom_linear_x = float(message.twist.twist.linear.x)
            odom_linear_y = float(message.twist.twist.linear.y)
            odom_angular = float(message.twist.twist.angular.z)
            if not all(
                math.isfinite(value)
                for value in (
                    odom_x,
                    odom_y,
                    odom_yaw,
                    odom_linear_x,
                    odom_linear_y,
                    odom_angular,
                )
            ):
                rospy.logerr_throttle(1.0, "Ignoring non-finite zigzag odometry")
                return
            new_odom_frame = message.header.frame_id or "odom"
            clock_rewound = bool(
                self.odom_history
                and source_stamp < self.odom_history[-1][0]
            )
            frame_changed = bool(
                self.odom_ready and new_odom_frame != self.odom_frame
            )
            if clock_rewound or frame_changed:
                self.odom_history.clear()
                self.pending_scans.clear()
                self.live_obstacle_points = np.empty((0, 2), dtype=np.float64)
                self.scan_stamp = None
                self.scan_received = None
                self.scan_pose_stamp_delta = math.inf
            self.odom_x = odom_x
            self.odom_y = odom_y
            self.odom_yaw = odom_yaw
            self.odom_speed = math.hypot(odom_linear_x, odom_linear_y)
            self.odom_angular_velocity = odom_angular
            self.odom_frame = new_odom_frame
            self.odom_stamp = source_stamp
            self.odom_ready = True
            self.odom_history.append(
                (
                    source_stamp,
                    self.odom_x,
                    self.odom_y,
                    self.odom_yaw,
                    self.odom_frame,
                )
            )
            newest_stamp = self.odom_history[-1][0]
            oldest_allowed = (
                newest_stamp.to_sec() - self.odom_history_duration
            )
            while (
                len(self.odom_history) > 1
                and self.odom_history[0][0].to_sec() < oldest_allowed
            ):
                self.odom_history.popleft()
            ready_scans = self._take_ready_scans(received)

        for scan, synchronized in ready_scans:
            self._project_scan(scan, synchronized)

    def _odom_pose_at_scan_stamp(self, stamp, history=None):
        """Return an odom pose interpolated exactly at a scan source stamp."""
        samples = tuple(self.odom_history if history is None else history)
        if not samples:
            return None
        target = float(stamp.to_sec())
        for sample in samples:
            delta = abs(float(sample[0].to_sec()) - target)
            if delta <= 1e-9:
                if sample[4] != self.odom_frame:
                    return None
                return (
                    stamp,
                    float(sample[1]),
                    float(sample[2]),
                    float(sample[3]),
                    sample[4],
                    0.0,
                )

        first_time = float(samples[0][0].to_sec())
        last_time = float(samples[-1][0].to_sec())
        if target < first_time or target > last_time:
            return None
        for first, second in zip(samples[:-1], samples[1:]):
            first_time = float(first[0].to_sec())
            second_time = float(second[0].to_sec())
            if not (first_time < target < second_time):
                continue
            left_delta = target - first_time
            right_delta = second_time - target
            bracket_skew = max(left_delta, right_delta)
            if bracket_skew > self.scan_pose_stamp_skew:
                return None
            if first[4] != self.odom_frame or second[4] != self.odom_frame:
                return None
            fraction = left_delta / (second_time - first_time)
            yaw_delta = normalize_angle(float(second[3]) - float(first[3]))
            return (
                stamp,
                float(first[1])
                + fraction * (float(second[1]) - float(first[1])),
                float(first[2])
                + fraction * (float(second[2]) - float(first[2])),
                normalize_angle(float(first[3]) + fraction * yaw_delta),
                first[4],
                bracket_skew,
            )
        return None

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
        if len(self.pending_scans) >= self.pending_scan_limit:
            self.pending_scans.popleft()
            rospy.logwarn_throttle(
                1.0,
                "Zigzag dropped the oldest pending LaserScan before odom "
                "synchronization",
            )
        self.pending_scans.append(scan)

    def _take_ready_scans(self, now):
        """Remove bracketed scans and retain only bounded future scans."""
        history = tuple(self.odom_history)
        latest_stamp = (
            float(history[-1][0].to_sec()) if history else -math.inf
        )
        ready = []
        waiting = deque()
        for scan in self.pending_scans:
            if self._pending_scan_expired(scan, now):
                continue
            synchronized = self._odom_pose_at_scan_stamp(
                scan["source_stamp"], history
            )
            if synchronized is not None:
                ready.append((scan, synchronized))
                continue
            if float(scan["source_stamp"].to_sec()) > latest_stamp:
                waiting.append(scan)
                continue
            rospy.logwarn_throttle(
                1.0,
                "Zigzag discarded a LaserScan whose odom bracket exceeded "
                "the synchronization tolerance",
            )
        self.pending_scans = waiting
        ready.sort(key=lambda item: item[0]["source_stamp"].to_sec())
        return ready

    def _project_scan(self, scan, synchronized):
        (
            _,
            robot_x,
            robot_y,
            robot_yaw,
            odom_frame,
            bracket_skew,
        ) = synchronized
        ranges = scan["ranges"]
        valid = np.isfinite(ranges)
        valid &= ranges >= scan["minimum_range"]
        valid &= ranges <= scan["maximum_range"]
        selected_ranges = ranges[valid]
        selected_angles = (
            scan["angle_min"]
            + np.flatnonzero(valid) * scan["angle_increment"]
            + self.lidar_yaw
        )
        base_x = self.lidar_x + selected_ranges * np.cos(selected_angles)
        base_y = self.lidar_y + selected_ranges * np.sin(selected_angles)
        cosine = math.cos(robot_yaw)
        sine = math.sin(robot_yaw)
        points = np.column_stack(
            (
                robot_x + cosine * base_x - sine * base_y,
                robot_y + sine * base_x + cosine * base_y,
            )
        ).reshape((-1, 2))

        with self.lock:
            if (
                odom_frame != self.odom_frame
                or (
                    self.scan_stamp is not None
                    and scan["source_stamp"] <= self.scan_stamp
                )
            ):
                return
            self.live_obstacle_points = points
            self.scan_stamp = scan["source_stamp"]
            self.scan_received = scan["received"]
            self.scan_pose_stamp_delta = float(bracket_skew)

    def scan_callback(self, message):
        received = rospy.Time.now()
        source_stamp = message.header.stamp
        if source_stamp == rospy.Time():
            rospy.logerr_throttle(
                1.0, "Rejected zigzag LiDAR scan without a source stamp"
            )
            return
        source_frame = message.header.frame_id.lstrip("/")
        if source_frame != self.scan_frame:
            rospy.logerr_throttle(
                1.0,
                "Rejected zigzag LiDAR frame %s; expected %s",
                source_frame or "<empty>",
                self.scan_frame,
            )
            return
        source_age = (received - source_stamp).to_sec()
        if (
            source_age < -self.scan_pose_stamp_skew
            or source_age > self.scan_timeout
        ):
            rospy.logwarn_throttle(
                1.0,
                "Rejected zigzag LiDAR scan with source age %.3fs",
                source_age,
            )
            return

        ranges = np.asarray(message.ranges, dtype=np.float64)
        if (
            ranges.size == 0
            or not math.isfinite(float(message.angle_min))
            or not math.isfinite(float(message.angle_increment))
            or abs(float(message.angle_increment)) <= 1e-12
            or not math.isfinite(float(message.range_min))
            or not math.isfinite(float(message.range_max))
        ):
            rospy.logerr_throttle(1.0, "Rejected malformed zigzag LiDAR scan")
            return
        minimum_range = max(0.0, float(message.range_min))
        maximum_range = min(
            float(message.range_max), self.scan_maximum_range
        )
        if maximum_range <= minimum_range:
            rospy.logerr_throttle(
                1.0, "Rejected zigzag LiDAR scan with invalid range bounds"
            )
            return

        scan = {
            "source_stamp": source_stamp,
            "received": received,
            "ranges": ranges.copy(),
            "angle_min": float(message.angle_min),
            "angle_increment": float(message.angle_increment),
            "minimum_range": minimum_range,
            "maximum_range": maximum_range,
        }
        with self.lock:
            if self.scan_stamp is not None and source_stamp <= self.scan_stamp:
                return
            self._enqueue_pending_scan(scan, received)
            ready_scans = self._take_ready_scans(received)
        for ready_scan, synchronized in ready_scans:
            self._project_scan(ready_scan, synchronized)

    def command_observer_callback(self, message):
        with self.lock:
            if (
                not self.mission_has_control
                and math.isfinite(message.linear.x)
                and math.isfinite(message.angular.z)
            ):
                self.observed_lane_linear = float(message.linear.x)
                self.observed_lane_angular = float(message.angular.z)

    def boundary_callback(self, message):
        now = rospy.Time.now()
        valid = False
        if len(message.data) >= 4:
            yellow = float(message.data[0])
            white = float(message.data[1])
            yellow_valid = bool(
                message.data[2] > 0.5 and math.isfinite(yellow)
            )
            white_valid = bool(message.data[3] > 0.5 and math.isfinite(white))
            if yellow_valid and white_valid:
                width = white - yellow
                center = 0.5 * (yellow + white)
                valid = bool(
                    self.lane_width_min <= width <= self.lane_width_max
                    and abs(center - self.image_center)
                    <= self.lane_center_tolerance
                )
            elif white_valid:
                # Match detect_lane.make_lane(): the normal lane controller
                # deliberately follows one visible boundary at transitions.
                center = white - self.single_line_center_offset
                valid = bool(
                    abs(center - self.image_center)
                    <= self.lane_center_tolerance
                )
            elif yellow_valid:
                center = yellow + self.single_line_center_offset
                valid = bool(
                    abs(center - self.image_center)
                    <= self.lane_center_tolerance
                )
        with self.lock:
            self.boundary_stamp = now
            self.boundary_valid = valid
            confirmation_enabled = bool(
                self.exit_confirmation_started
                or self.state == self.JOINING_LANE
            )
            if not valid or not confirmation_enabled:
                self.boundary_confirmation_count = 0
                self.last_boundary_confirmation_time = None
                return
            if (
                self.confirmation_started_at is not None
                and now <= self.confirmation_started_at
            ):
                self.boundary_confirmation_count = 0
                self.last_boundary_confirmation_time = None
                return
            if (
                self.last_boundary_confirmation_time is None
                or (now - self.last_boundary_confirmation_time).to_sec()
                > self.exit_confirmation_max_gap
            ):
                self.boundary_confirmation_count = 1
            else:
                self.boundary_confirmation_count += 1
            self.last_boundary_confirmation_time = now

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
            else:
                if self.pause_started is not None:
                    paused = now - self.pause_started
                    self.state_started += paused
                    if self.mission_started is not None:
                        self.mission_started += paused
                self.manual_stop = False
                self.pause_started = None
                self.boundary_confirmation_count = 0
                self.last_boundary_confirmation_time = None

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
            rospy.logerr("Zigzag cmd_vel handoff failed: %s", error)
            return False

    def _stop_lane_controller(self):
        try:
            rospy.wait_for_service(self.lane_stop_service_name, timeout=1.0)
            response = self.lane_stop_service(False)
            if not response.success:
                raise rospy.ServiceException(response.message)
            return True
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logerr("Zigzag lane emergency stop failed: %s", error)
            return False

    def _start_run(self, now):
        self.start_requested = False
        self.revoke_requested = False
        self.speed_limit_pub.publish(Float64(data=self.entry_velocity_cap))
        self.committed_path = None
        self.route_from_odom = None
        self.path_follower = PathFollower(self.tracking_config)
        self.path_index = 0
        self.last_command_time = None
        self.boundary_confirmation_count = 0
        self.last_boundary_confirmation_time = None
        self.exit_confirmation_started = False
        self.exit_stop_latched = False
        self.confirmation_started_at = None
        self.join_origin_ready = False
        self.maximum_position_error_seen = 0.0
        self.maximum_heading_error_seen = 0.0
        self._set_state(self.ACQUIRING, now)

    def _inputs_problem(self, now):
        if not self.map_ready or self.map_stamp is None:
            return "map pose"
        map_age = (now - self.map_stamp).to_sec()
        if map_age < -self.maximum_pose_stamp_skew:
            return "future map pose"
        if map_age > self.map_timeout:
            return "stale map pose"
        if not self.odom_ready or self.odom_stamp is None:
            return "odometry"
        odom_age = (now - self.odom_stamp).to_sec()
        if odom_age < -self.maximum_pose_stamp_skew:
            return "future odometry"
        if odom_age > self.odom_timeout:
            return "stale odometry"
        if self._synchronized_odom_pose() is None:
            return "map/odom timestamp synchronization"
        scan_problem = self._scan_input_problem(now)
        if scan_problem is not None:
            return scan_problem
        return None

    def _synchronized_odom_pose(self, stamp=None, maximum_skew=None):
        target_stamp = self.map_stamp if stamp is None else stamp
        if target_stamp is None or not self.odom_history:
            return None
        allowed_skew = (
            self.maximum_pose_stamp_skew
            if maximum_skew is None
            else max(0.0, float(maximum_skew))
        )
        synchronized = min(
            self.odom_history,
            key=lambda sample: abs((sample[0] - target_stamp).to_sec()),
        )
        if (
            abs((synchronized[0] - target_stamp).to_sec())
            > allowed_skew
        ):
            return None
        if synchronized[4] != self.odom_frame:
            return None
        return synchronized

    def _scan_input_problem(self, now):
        if self.scan_stamp is None or self.scan_received is None:
            return "LiDAR scan"
        source_age = (now - self.scan_stamp).to_sec()
        receipt_age = (now - self.scan_received).to_sec()
        if source_age < -self.scan_pose_stamp_skew or receipt_age < 0.0:
            return "future LiDAR scan"
        if source_age > self.scan_timeout or receipt_age > self.scan_timeout:
            return "stale LiDAR scan"
        if self.scan_pose_stamp_delta > self.scan_pose_stamp_skew:
            return "LiDAR/odometry timestamp synchronization"
        return None

    def _fresh_live_obstacles(self, now):
        if self._scan_input_problem(now) is not None:
            return None
        return self.live_obstacle_points

    def _tracking_input_problem(self, now):
        if not self.odom_ready or self.odom_stamp is None:
            return "odometry"
        odom_age = (now - self.odom_stamp).to_sec()
        if odom_age < -self.maximum_pose_stamp_skew:
            return "future odometry"
        if odom_age > self.odom_timeout:
            return "stale odometry"
        return None

    def _captured_acquisition_input_problem(self, snapshot, now):
        map_age = (now - snapshot["map_stamp"]).to_sec()
        if map_age < -self.maximum_pose_stamp_skew:
            return "future map-pose snapshot"
        if map_age > self.map_timeout:
            return "stale map-pose snapshot"
        if (
            abs(
                (
                    snapshot["synchronized_odom_stamp"]
                    - snapshot["map_stamp"]
                ).to_sec()
            )
            > self.maximum_pose_stamp_skew
        ):
            return "map/odometry snapshot synchronization"
        source_age = (now - snapshot["scan_stamp"]).to_sec()
        receipt_age = (now - snapshot["scan_received"]).to_sec()
        if source_age < -self.scan_pose_stamp_skew or receipt_age < 0.0:
            return "future LiDAR snapshot"
        if source_age > self.scan_timeout or receipt_age > self.scan_timeout:
            return "stale LiDAR snapshot"
        if snapshot["scan_pose_stamp_delta"] > self.scan_pose_stamp_skew:
            return "LiDAR/odometry snapshot synchronization"
        return None

    def _acquire(self, now):
        # Capture coherent localization and LiDAR input while lane following
        # still owns cmd_vel. The full dynamic-obstacle sweep is pure geometry
        # and must not starve odom, scan, gate, manual-stop or shutdown callbacks.
        with self.lock:
            if self.state != self.ACQUIRING:
                return False
            if (now - self.state_started).to_sec() > self.acquisition_timeout:
                self._fail(
                    "timed out before the surveyed zigzag path could be acquired"
                )
                return False
            problem = self._inputs_problem(now)
            if problem is not None:
                rospy.logwarn_throttle(1.0, "Waiting for zigzag %s", problem)
                return False
            synchronized = self._synchronized_odom_pose()
            live_obstacles = self._fresh_live_obstacles(now)
            if synchronized is None or live_obstacles is None:
                return False
            sync_stamp, sync_x, sync_y, sync_yaw, _ = synchronized
            snapshot = {
                "state_started": self.state_started,
                "map_path": self.map_path,
                "map_pose": Pose2D(self.map_x, self.map_y, self.map_yaw),
                "map_stamp": self.map_stamp,
                "synchronized_odom_pose": Pose2D(sync_x, sync_y, sync_yaw),
                "synchronized_odom_stamp": sync_stamp,
                "odom_frame": self.odom_frame,
                "scan_stamp": self.scan_stamp,
                "scan_received": self.scan_received,
                "scan_pose_stamp_delta": self.scan_pose_stamp_delta,
                "live_obstacles": np.array(
                    live_obstacles, dtype=np.float64, copy=True
                ),
                "validator": self.path_validator,
                "live_route_safety": self.live_route_safety,
                "tracking_config": self.tracking_config,
            }

        candidate, route_from_odom = freeze_path_in_odom(
            snapshot["map_path"],
            snapshot["map_pose"],
            snapshot["synchronized_odom_pose"],
            odom_aligned=self.route_odom_aligned,
            odom_frame=snapshot["odom_frame"],
        )
        # The surveyed line/map sweep was already accepted at initialization.
        # A rigid map-to-odom transform preserves those clearances, so only the
        # newly observed obstacle cloud needs another whole-route sweep here.
        candidate.safety = self._path_safety(route_from_odom)
        validation = snapshot["validator"].validate_path(
            candidate,
            safety=snapshot["live_route_safety"],
            live_obstacles=snapshot["live_obstacles"],
        )
        candidate.obstacle_clearance = min(
            candidate.obstacle_clearance,
            validation.minimum_obstacle_clearance,
        )
        follower = PathFollower(snapshot["tracking_config"])

        with self.lock:
            # Callback-side stops and ownership changes always win over a
            # geometry result calculated from an earlier snapshot.
            if self.shutting_down:
                return False
            if self.revoke_requested:
                self._revoke()
                return False
            if self.manual_stop:
                return False
            if (
                self.state != self.ACQUIRING
                or self.state_started != snapshot["state_started"]
                or self.map_path is not snapshot["map_path"]
                or self.odom_frame != snapshot["odom_frame"]
                or self.path_validator is not snapshot["validator"]
                or self.live_route_safety is not snapshot["live_route_safety"]
                or self.tracking_config is not snapshot["tracking_config"]
            ):
                return False

            completed = rospy.Time.now()
            problem = self._captured_acquisition_input_problem(
                snapshot, completed
            )
            if problem is None:
                problem = self._inputs_problem(completed)
            if problem is not None:
                rospy.logwarn_throttle(1.0, "Waiting for zigzag %s", problem)
                return False
            if not validation.safe:
                self._fail(
                    "zigzag live-obstacle path sweep is unsafe: "
                    "line=%.3fm obstacle=%.3fm map=%.3fm"
                    % (
                        candidate.line_clearance,
                        validation.minimum_obstacle_clearance,
                        candidate.map_clearance,
                    )
                )
                return False

            initial_linear = min(
                max(0.0, self.observed_lane_linear),
                self.entry_velocity_cap,
            )
            initial_angular = clamp(
                self.observed_lane_angular,
                -self.maximum_angular_velocity,
                self.maximum_angular_velocity,
            )
            pose = Pose2D(self.odom_x, self.odom_y, self.odom_yaw)
            projection = follower.reset(
                candidate,
                pose,
                initial_linear=initial_linear,
                initial_angular=initial_angular,
            )
            path_error = projection.distance
            heading_error = abs(normalize_angle(projection.heading - pose.yaw))
            if (
                path_error > self.start_maximum_path_error
                or heading_error > self.start_maximum_join_heading_error
            ):
                self._fail(
                    "zigzag entry does not join the path: "
                    "position=%.3fm heading=%.1fdeg"
                    % (path_error, math.degrees(heading_error))
                )
                return False
            follower.update_clearance(validation)
            if not self._set_lane_controller(False):
                self._fail("could not acquire cmd_vel control")
                return False

            # Publish no mission command before the lane-disable service has
            # confirmed the sole /cmd_vel owner. All shared route state is
            # committed together after that ownership boundary.
            committed_at = rospy.Time.now()
            self.committed_path = candidate
            self.route_from_odom = route_from_odom
            self.path_follower = follower
            self.path_index = projection.path_index
            self.last_linear = initial_linear
            self.last_angular = initial_angular
            self.last_command_time = committed_at
            self.mission_started = committed_at
            self._publish_path()
            self._set_state(self.FOLLOWING, committed_at)
            rospy.loginfo(
                "Zigzag path fixed in odom: entry_index=%d remaining=%.3fm",
                self.path_index,
                self.path_follower.diagnostics.remaining_distance,
            )
            return True

    def _revoke(self):
        if self.state not in (self.WAIT_GATE, self.COMPLETE):
            self.revoke_requested = False
            self._fail("zigzag gate closed before the safe lane join completed")
            return
        if self.mission_has_control:
            self._publish_stop()
            if not self._set_lane_controller(True):
                self._fail("lane-controller handoff failed after zigzag gate closed")
                return
        self.speed_limit_pub.publish(Float64(data=self.lane_resume_max_velocity))
        self.revoke_requested = False
        self.start_requested = False
        self.committed_path = None
        self._set_state(self.WAIT_GATE)

    def _fail(self, reason):
        if self.state == self.FAILED:
            return
        self.speed_limit_pub.publish(Float64(data=0.0))
        owns_unambiguously = (
            self.mission_has_control and not self.handoff_ambiguous
        )
        if not owns_unambiguously:
            # A handoff request may have been applied even when its response is
            # lost, so retry only when the current owner is genuinely unknown.
            owns_unambiguously = self._set_lane_controller(False)
        if not owns_unambiguously:
            # The failed handoff response cannot prove which publisher owns
            # /cmd_vel.  Relinquish direct publishing and use only the
            # independent stop authorities until ownership is explicit again.
            self.mission_has_control = False
            self._stop_lane_controller()
            self.emergency_stop_pub.publish(Bool(data=True))
        else:
            self.cmd_pub.publish(Twist())
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.path_follower.last_linear = 0.0
        self.path_follower.last_angular = 0.0
        self.last_command_time = rospy.Time.now()
        self._set_state(self.FAILED)
        rospy.logerr("Zigzag mission failed: %s", reason)

    def _publish_stop(self):
        if self.mission_has_control:
            self.cmd_pub.publish(Twist())
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.path_follower.last_linear = 0.0
        self.path_follower.last_angular = 0.0
        self.path_follower.diagnostics.commanded_linear = 0.0
        self.path_follower.diagnostics.commanded_angular = 0.0
        self.last_command_time = rospy.Time.now()

    def _publish_path(self):
        message = Path()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.committed_path.frame_id
        for x, y, yaw in zip(
            self.committed_path.x,
            self.committed_path.y,
            self.committed_path.heading,
        ):
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.06
            pose.pose.orientation.z = math.sin(0.5 * float(yaw))
            pose.pose.orientation.w = math.cos(0.5 * float(yaw))
            message.poses.append(pose)
        self.path_pub.publish(message)

    def _publish_diagnostics(self):
        message = Float64MultiArray()
        message.data = self.path_follower.diagnostics.as_array()
        self.diagnostics_pub.publish(message)

    def _command_elapsed(self, now):
        if self.last_command_time is None:
            return self.control_period
        return clamp((now - self.last_command_time).to_sec(), 0.0, 0.15)

    def _common_command(
        self,
        now,
        speed_limit=math.inf,
        pose=None,
        tracking=None,
    ):
        elapsed = self._command_elapsed(now)
        if pose is None:
            pose = Pose2D(self.odom_x, self.odom_y, self.odom_yaw)
        limited, tracking = self.path_follower.command(
            pose,
            elapsed,
            speed_limit=speed_limit,
            tracking=tracking,
        )
        command = Twist()
        command.linear.x = limited.linear_velocity
        command.angular.z = limited.angular_velocity
        self.last_linear = limited.linear_velocity
        self.last_angular = limited.angular_velocity
        self.last_command_time = now
        return command, tracking

    def _common_stop_command(self, now):
        """Build one slew-limited zero target with the shared follower."""
        limited = self.path_follower.stop(self._command_elapsed(now))
        command = Twist()
        command.linear.x = limited.linear_velocity
        command.angular.z = limited.angular_velocity
        self.last_linear = limited.linear_velocity
        self.last_angular = limited.angular_velocity
        self.last_command_time = now
        return command, limited

    def _decelerate_at_exit(self, now):
        command, limited = self._common_stop_command(now)
        if self.mission_has_control:
            self.cmd_pub.publish(command)
        self._publish_diagnostics()
        if (
            abs(limited.linear_velocity) <= 1e-9
            and abs(limited.angular_velocity) <= 1e-9
        ):
            self._set_state(self.VERIFY_EXIT, now)

    def _path_safety(self, route_from_odom=None):
        # Bind each candidate to its own immutable inverse transform. A sweep
        # running without the mission lock must never observe a later run's
        # mutable self.route_from_odom value.
        frozen_route_from_odom = (
            self.route_from_odom
            if route_from_odom is None
            else route_from_odom
        )

        def route_pose(pose):
            if frozen_route_from_odom is None:
                return Pose2D.from_value(pose)
            return frozen_route_from_odom.apply_pose(pose)

        def reference_line_clearance(pose, footprint):
            return self.corridor_checker.line_clearance(
                route_pose(pose),
                footprint,
                footprint_sample_spacing=self.footprint_sample_spacing,
                inner_blocking=self.inner_paint_blocking,
            )

        def reference_map_clearance(pose, footprint):
            return self.corridor_checker.map_clearance(
                route_pose(pose), footprint
            )

        line_boundaries = []
        if self.inner_paint_blocking:
            line_boundaries.append(CallbackBoundary(reference_line_clearance))
        if self.paint_checker is not None:

            def paint_clearance(pose, footprint):
                return self.paint_checker.clearance(
                    route_pose(pose),
                    footprint,
                    footprint_sample_spacing=self.footprint_sample_spacing,
                    minimum_outer_reserve=self.minimum_outer_reserve,
                    inner_blocking=self.inner_paint_blocking,
                )

            line_boundaries.append(CallbackBoundary(paint_clearance))
        return PathSafety(
            line_boundaries=tuple(line_boundaries),
            map_boundaries=(CallbackBoundary(reference_map_clearance),),
            margins=self.safety_margins,
        )

    def _motion_safety(
        self,
        desired_speed,
        live_obstacles,
        pose=None,
        tracking=None,
        path=None,
        validator=None,
        linear_speed=None,
        angular_velocities=None,
    ):
        path = self.committed_path if path is None else path
        validator = self.path_validator if validator is None else validator
        if pose is None:
            pose = Pose2D(self.odom_x, self.odom_y, self.odom_yaw)
        if tracking is None:
            tracking = self.path_follower.calculate_tracking(pose)
        speed = (
            max(abs(self.last_linear), self.odom_speed)
            if linear_speed is None
            else max(0.0, float(linear_speed))
        )
        if angular_velocities is None:
            angular_velocities = (
                self.last_angular,
                self.odom_angular_velocity,
                tracking.angular_velocity,
            )
        stopping_horizon = (
            desired_speed * self.prediction_reaction_time
            + desired_speed * desired_speed
            / (2.0 * self.linear_deceleration)
            + self.prediction_distance_margin
        )
        decision = validator.motion_safety(
            path=path,
            pose=pose,
            path_index=tracking.path_index,
            desired_speed=desired_speed,
            linear_velocity=speed,
            angular_velocities=angular_velocities,
            reaction_time=self.prediction_reaction_time,
            linear_deceleration=self.linear_deceleration,
            distance_margin=self.prediction_distance_margin,
            lookahead_distance=max(
                self.nearest_search_ahead,
                stopping_horizon,
            ),
            # Only the immutable nominal route omits boundaries already swept
            # at activation. The actual reaction-and-complete-stop sweep keeps
            # path.safety so cross-track and heading error cannot hide a future
            # paint/map contact between the current pose and the stopping pose.
            safety=path.safety,
            route_safety=self.live_route_safety,
            live_obstacles=live_obstacles,
            tracking=tracking,
        )
        return decision

    def _exit_lane_confirmed(self, now, required_frames):
        return bool(
            self.confirmation_started_at is not None
            and self.boundary_valid
            and self.boundary_stamp is not None
            and (now - self.boundary_stamp).to_sec() <= self.boundary_timeout
            and self.last_boundary_confirmation_time is not None
            and self.last_boundary_confirmation_time
            >= self.confirmation_started_at
            and (now - self.last_boundary_confirmation_time).to_sec()
            <= self.exit_confirmation_max_gap
            and self.boundary_confirmation_count >= required_frames
        )

    def _start_exit_confirmation(self, now):
        if self.exit_confirmation_started:
            return
        self.exit_confirmation_started = True
        self.confirmation_started_at = now
        self.boundary_confirmation_count = 0
        self.last_boundary_confirmation_time = None

    def _exit_pose_ready(self, pose=None, tracking=None):
        if self.committed_path is None:
            return False
        if pose is None:
            pose = Pose2D(self.odom_x, self.odom_y, self.odom_yaw)
        status = self.path_follower.goal_status(
            pose,
            tracking=tracking,
        )
        self.path_index = self.path_follower.path_index
        return bool(
            status.remaining_distance <= self.handoff_remaining_distance
            and status.complete
        )

    def _start_lane_join(self, now):
        if not self.mission_has_control:
            self._fail("cmd_vel ownership was lost before lane join")
            return
        self.speed_limit_pub.publish(Float64(data=self.join_velocity_cap))
        if not self._set_lane_controller(True):
            self._fail("could not start the low-speed lane join")
            return
        join_started = rospy.Time.now()
        self.boundary_confirmation_count = 0
        self.last_boundary_confirmation_time = None
        self.confirmation_started_at = join_started
        self.exit_confirmation_started = False
        self.join_origin_ready = False
        self._set_state(self.JOINING_LANE, join_started)

    def _complete(self, now):
        elapsed = (
            0.0
            if self.mission_started is None
            else (now - self.mission_started).to_sec()
        )
        if self.mission_has_control:
            self._fail("mission still owned cmd_vel at lane-join completion")
            return
        self.speed_limit_pub.publish(Float64(data=self.lane_resume_max_velocity))
        self._set_state(self.COMPLETE, now)
        self._publish_diagnostics()
        rospy.loginfo(
            "Zigzag complete in %.3fs; max path error=%.4fm, "
            "max heading error=%.2fdeg, min line=%.4fm, "
            "min obstacle=%.4fm; "
            "lane controller owns cmd_vel",
            elapsed,
            self.maximum_position_error_seen,
            math.degrees(self.maximum_heading_error_seen),
            self.path_follower.diagnostics.minimum_line_clearance,
            self.path_follower.diagnostics.minimum_obstacle_clearance,
        )

    def _join_lane(self, now):
        problem = self._tracking_input_problem(now)
        if problem is not None:
            self._fail("zigzag lane join lost %s" % problem)
            return
        if not self.join_origin_ready:
            if self.odom_stamp <= self.state_started:
                if (now - self.state_started).to_sec() > self.join_timeout:
                    self._fail("lane join did not receive fresh odometry")
                return
            self.join_start_x = self.odom_x
            self.join_start_y = self.odom_y
            self.join_start_yaw = self.odom_yaw
            self.join_origin_ready = True
        progress = (
            (self.odom_x - self.join_start_x) * math.cos(self.join_start_yaw)
            + (self.odom_y - self.join_start_y) * math.sin(self.join_start_yaw)
        )
        self._publish_diagnostics()
        if (
            progress >= self.join_minimum_distance
            and self._exit_lane_confirmed(now, self.join_confirmation_frames)
        ):
            self._complete(now)
            return
        if (now - self.state_started).to_sec() > self.join_timeout:
            self._fail("lane control did not complete the low-speed zigzag join")

    def _captured_follow_input_problem(self, snapshot, now):
        odom_age = (now - snapshot["odom_stamp"]).to_sec()
        if odom_age < -self.maximum_pose_stamp_skew:
            return "future odometry snapshot"
        if odom_age > self.odom_timeout:
            return "stale odometry snapshot"
        if snapshot["live_obstacles"] is None:
            return None
        source_age = (now - snapshot["scan_stamp"]).to_sec()
        receipt_age = (now - snapshot["scan_received"]).to_sec()
        if source_age < -self.scan_pose_stamp_skew or receipt_age < 0.0:
            return "future LiDAR snapshot"
        if source_age > self.scan_timeout or receipt_age > self.scan_timeout:
            return "stale LiDAR snapshot"
        if snapshot["scan_pose_stamp_delta"] > self.scan_pose_stamp_skew:
            return "LiDAR/odometry snapshot synchronization"
        return None

    def _follow(self, now):
        # Capture one coherent path/sensor state, then release the mission lock
        # for pure swept geometry. Odom, scan, manual-stop, gate and shutdown
        # callbacks must remain responsive during that calculation.
        with self.lock:
            if self.state != self.FOLLOWING or not self.mission_has_control:
                return
            if self.mission_started is None or (
                now - self.mission_started
            ).to_sec() > self.mission_timeout:
                self._fail("zigzag path timed out")
                return
            problem = self._tracking_input_problem(now)
            if problem is not None:
                command, _limited = self._common_stop_command(now)
                self.cmd_pub.publish(command)
                self._publish_diagnostics()
                rospy.logwarn_throttle(1.0, "Zigzag holds for %s", problem)
                return
            if self.exit_stop_latched:
                self._decelerate_at_exit(now)
                return
            path = self.committed_path
            follower = self.path_follower
            if path is None or follower.path is not path:
                self._fail("no frozen zigzag path is active")
                return
            pose = Pose2D(self.odom_x, self.odom_y, self.odom_yaw)
            tracking = follower.calculate_tracking(pose)
            live_obstacles = self._fresh_live_obstacles(now)
            snapshot = {
                "path": path,
                "follower": follower,
                "validator": self.path_validator,
                "pose": pose,
                "tracking": tracking,
                "linear_speed": max(abs(self.last_linear), self.odom_speed),
                "angular_velocities": (
                    self.last_angular,
                    self.odom_angular_velocity,
                    tracking.angular_velocity,
                ),
                "odom_stamp": self.odom_stamp,
                "scan_stamp": self.scan_stamp,
                "scan_received": self.scan_received,
                "scan_pose_stamp_delta": self.scan_pose_stamp_delta,
                "live_obstacles": (
                    None
                    if live_obstacles is None
                    else np.array(live_obstacles, dtype=np.float64, copy=True)
                ),
            }

        safety = None
        if snapshot["live_obstacles"] is not None:
            safety = self._motion_safety(
                float(tracking.target_speed),
                snapshot["live_obstacles"],
                pose=pose,
                tracking=tracking,
                path=path,
                validator=snapshot["validator"],
                linear_speed=snapshot["linear_speed"],
                angular_velocities=snapshot["angular_velocities"],
            )

        with self.lock:
            # Callbacks may have changed safety or ownership while the lock was
            # released. Apply the result only to the exact route/follower that
            # produced it, and never publish after an intervening stop/revoke.
            if self.shutting_down:
                return
            if self.revoke_requested:
                self._revoke()
                return
            if self.manual_stop:
                if self.mission_has_control:
                    self._publish_stop()
                return
            if (
                self.state != self.FOLLOWING
                or not self.mission_has_control
                or self.committed_path is not path
                or self.path_follower is not follower
                or self.path_validator is not snapshot["validator"]
            ):
                return
            completed = rospy.Time.now()
            problem = self._captured_follow_input_problem(snapshot, completed)
            if problem is not None:
                command, _limited = self._common_stop_command(completed)
                self.cmd_pub.publish(command)
                self._publish_diagnostics()
                rospy.logwarn_throttle(1.0, "Zigzag holds for %s", problem)
                return

            if safety is None:
                speed_limit = 0.0
                rospy.logwarn_throttle(
                    1.0,
                    "Zigzag common follower is braking for a stale LiDAR scan",
                )
            else:
                follower.update_clearance(safety.validation)
                speed_limit = safety.speed_limit
                if safety.requires_stop:
                    rospy.logwarn_throttle(
                        1.0,
                        "Zigzag common swept line/obstacle safety is braking "
                        "to a stop",
                    )

            remaining_distance = max(0.0, path.length - tracking.station)
            self.maximum_position_error_seen = max(
                self.maximum_position_error_seen, tracking.position_error
            )
            self.maximum_heading_error_seen = max(
                self.maximum_heading_error_seen, abs(tracking.heading_error)
            )
            if (
                not self.exit_confirmation_started
                and remaining_distance
                <= self.handoff_remaining_distance
                + self.exit_confirmation_lead_distance
            ):
                self._start_exit_confirmation(completed)
            if self._exit_pose_ready(pose=pose, tracking=tracking):
                self.exit_stop_latched = True
                # Check the common goal before calculating another positive path
                # command. Calculating it first advances last_command_time and
                # leaves the initial deceleration tick with dt=0.
                self._decelerate_at_exit(completed)
                return
            command, tracking = self._common_command(
                completed,
                speed_limit=speed_limit,
                pose=pose,
                tracking=tracking,
            )
            self.path_index = follower.path_index
            self._publish_diagnostics()
            self.cmd_pub.publish(command)

    def _verify_exit(self, now):
        self._publish_stop()
        problem = self._tracking_input_problem(now)
        if problem is not None:
            self._fail("zigzag exit confirmation lost %s" % problem)
            return
        self._publish_diagnostics()
        if self._exit_pose_ready() and self._exit_lane_confirmed(
            now, self.exit_confirmation_frames
        ):
            self._start_lane_join(now)
            return
        if (now - self.state_started).to_sec() > self.exit_confirmation_timeout:
            self._fail("zigzag exit lane was not confirmed")

    def control_callback(self, _event):
        acquire_now = None
        follow_now = None
        with self.lock:
            if self.shutting_down:
                return
            now = rospy.Time.now()
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
                acquire_now = now
            elif self.state == self.FOLLOWING:
                follow_now = now
            elif self.state == self.VERIFY_EXIT:
                self._verify_exit(now)
            elif self.state == self.JOINING_LANE:
                self._join_lane(now)
        if acquire_now is not None:
            if self._acquire(acquire_now):
                self._follow(rospy.Time.now())
            return
        if follow_now is not None:
            self._follow(follow_now)

    def shutdown(self):
        with self.lock:
            self.shutting_down = True
            if self.mission_has_control:
                self.cmd_pub.publish(Twist())
        rospy.loginfo("Zigzag mission controller stopped")


if __name__ == "__main__":
    rospy.init_node("zigzag_mission_controller")
    ZigzagMissionController()
    rospy.spin()
