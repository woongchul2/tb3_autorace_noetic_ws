#!/usr/bin/env python3
"""Select and align intersection paths around camera-path lane segments."""

import math
import os
import threading

import cv2
import numpy as np
import rospkg
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Bool, Float64, Float64MultiArray, String, UInt8
from std_srvs.srv import SetBool

from custom_autorace_bringup.msg import TrafficSign
from custom_autorace_bringup.path_following import (
    AsymmetricFootprint,
    AxisAlignedBoundsBoundary,
    GoalTolerance,
    PathFollower,
    PathSafety,
    Pose2D,
    RasterCellBoundary,
    RigidTransform2D,
    SafetyMargins,
    SpeedProfile,
    SweptFootprintValidator,
    TrackingConfig,
    clamp,
    normalize_angle,
    path_from_xy,
    project_to_path,
    yaw_from_quaternion,
)


class IntersectionMissionController:
    WAIT_INTERSECTION = "WAIT_INTERSECTION"
    SEARCH_DIRECTION = "SEARCH_DIRECTION"
    WAIT_ENTRY_HANDOFF = "WAIT_ENTRY_HANDOFF"
    PREPARE_ENTRY_PATH = "PREPARE_ENTRY_PATH"
    FOLLOW_ENTRY_PATH = "FOLLOW_ENTRY_PATH"
    FOLLOW_ARC_LANE = "FOLLOW_ARC_LANE"
    PREPARE_EXIT_PATH = "PREPARE_EXIT_PATH"
    FOLLOW_EXIT_PATH = "FOLLOW_EXIT_PATH"
    VERIFY_FINAL_LANE = "VERIFY_FINAL_LANE"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

    NONE, LEFT, RIGHT = 0, 2, 3

    def __init__(self):
        p = "~mission/"
        get = rospy.get_param
        self.odom_topic = get(p + "odom_topic", "/odometry/filtered")
        self.sign_topic = get(p + "sign_topic", "/detect/signs")
        self.camera_info_topic = get(
            p + "camera_info_topic", "/camera/color/camera_info"
        )
        self.boundary_topic = get(p + "boundary_topic", "/detect/lane_boundaries")
        self.cmd_vel_topic = get(p + "cmd_vel_topic", "/cmd_vel")
        self.manual_stop_topic = get(p + "manual_stop_topic", "/control/manual_stop")
        self.lane_speed_limit_topic = get(
            p + "lane_speed_limit_topic", "/control/max_vel"
        )
        self.lane_service_name = get(p + "lane_control_service", "/control/lane_mission_handoff")
        self.lane_stop_service_name = get(
            p + "lane_stop_service", "/control/lane_following"
        )
        self.zone_gate_topic = str(
            get(p + "zone_gate_topic", "/mission/enable/intersection")
        )
        self.direction_observation_topic = str(
            get(
                p + "direction_observation_topic",
                "/mission/inside/intersection_direction_observation",
            )
        )
        self.mission_map_pose_topic = str(
            get(p + "mission_map_pose_topic", "/mission/map_pose")
        )

        self.direction_confirm_frames = max(
            1, int(get(p + "direction_confirm_frames", 9))
        )
        # Non-zero only for repeatable simulation tests (2=left, 3=right).
        self.forced_direction = int(get(p + "forced_direction", self.NONE))
        self.direction_acquire_min_confidence = clamp(
            float(get(p + "direction_acquire_min_confidence", 0.62)), 0.0, 1.0
        )
        self.direction_tracking_min_confidence = clamp(
            float(get(p + "direction_tracking_min_confidence", 0.58)), 0.0, 1.0
        )
        self.direction_confirm_min_confidence = clamp(
            float(get(p + "direction_confirm_min_confidence", 0.60)), 0.0, 1.0
        )
        self.direction_acquire_min_roi_area_ratio = clamp(
            float(get(p + "direction_acquire_min_roi_area_ratio", 0.030)),
            0.0,
            1.0,
        )
        self.direction_tracking_min_roi_area_ratio = clamp(
            float(get(p + "direction_tracking_min_roi_area_ratio", 0.030)),
            0.0,
            1.0,
        )
        self.direction_confirm_min_roi_area_ratio = clamp(
            float(get(p + "direction_confirm_min_roi_area_ratio", 0.030)),
            0.0,
            1.0,
        )
        self.direction_confirmation_max_gap = max(
            0.05, float(get(p + "direction_confirmation_max_gap", 1.20))
        )
        self.direction_search_timeout = max(
            1.0, float(get(p + "direction_search_timeout", 35.0))
        )
        self.entry_handoff_timeout = max(
            1.0, float(get(p + "entry_handoff_timeout", 5.0))
        )
        self.entry_takeover_pose_timeout = max(
            0.10, float(get(p + "entry_takeover_pose_timeout", 0.75))
        )
        self.entry_handoff_lead_distance = max(
            0.0, float(get(p + "entry_handoff_lead_distance", 0.0))
        )
        self.path_follow_timeout = float(get(p + "path_follow_timeout", 10.0))
        self.path_lookahead = float(get(p + "path_lookahead", 0.08))
        self.path_goal_tolerance = float(get(p + "path_goal_tolerance", 0.03))
        self.path_goal_heading_tolerance = math.radians(
            abs(float(get(p + "path_goal_heading_tolerance_deg", 8.0)))
        )
        self.path_goal_crossing_max_distance = max(
            self.path_goal_tolerance,
            float(get(p + "path_goal_crossing_max_distance", 0.15)),
        )
        self.path_min_velocity = float(get(p + "path_min_velocity", 0.04))
        self.path_velocity = float(get(p + "path_linear_velocity", 0.05))
        self.path_entry_velocity = max(
            0.005, float(get(p + "path_entry_velocity", self.path_min_velocity))
        )
        self.path_exit_velocity = max(
            0.005, float(get(p + "path_exit_velocity", self.path_min_velocity))
        )
        self.left_path_velocity = float(
            get(p + "left_path_linear_velocity", self.path_velocity)
        )
        self.right_path_velocity = float(
            get(p + "right_path_linear_velocity", self.path_velocity)
        )
        self.active_path_velocity = self.path_velocity
        self.path_max_angular = abs(float(get(p + "path_max_angular_velocity", 0.55)))
        self.left_path_max_angular = abs(
            float(get(p + "left_path_max_angular_velocity", self.path_max_angular))
        )
        self.right_path_max_angular = abs(
            float(get(p + "right_path_max_angular_velocity", self.path_max_angular))
        )
        self.active_path_max_angular = self.path_max_angular
        self.maximum_lateral_acceleration = max(
            0.005,
            float(get(p + "maximum_lateral_acceleration", 0.08)),
        )
        self.path_linear_acceleration = max(
            0.005, float(get(p + "linear_acceleration", 0.25))
        )
        self.path_linear_deceleration = max(
            0.005, float(get(p + "linear_deceleration", 0.50))
        )
        self.path_angular_acceleration = max(
            0.05, float(get(p + "path_angular_acceleration", 1.50))
        )
        self.path_heading_gain = float(get(p + "path_heading_gain", 0.35))
        self.path_curvature_weight = clamp(
            float(get(p + "path_curvature_weight", 0.25)), 0.0, 1.0
        )
        self.path_lateral_feedback_gain = max(
            0.0, float(get(p + "path_lateral_feedback_gain", 1.0))
        )
        self.path_search_ahead_distance = max(
            0.05, float(get(p + "path_search_ahead_distance", 0.30))
        )
        # The official-start camera approach can hand over while the measured
        # rectangle is already touching the outer paint.  The common path
        # carries a station-dependent allowance for that initial convergence;
        # it reaches zero quickly.  The exit has its own separately measured
        # envelope below, so neither allowance leaks across path stages.
        self.path_initial_line_overlap_allowance = max(
            0.0,
            float(get(p + "path_initial_line_overlap_allowance", 0.060)),
        )
        self.path_line_egress_distance = max(
            0.0, float(get(p + "path_line_egress_distance", 0.100))
        )
        if (
            self.path_initial_line_overlap_allowance > 0.0
            and self.path_line_egress_distance <= 0.0
        ):
            raise rospy.ROSInitException(
                "path_line_egress_distance must be positive when an initial "
                "line overlap allowance is configured"
            )
        self.exit_path_initial_line_overlap_allowance = max(
            0.0,
            float(get(p + "exit_path_initial_line_overlap_allowance", 0.030)),
        )
        self.exit_path_line_egress_distance = max(
            0.0, float(get(p + "exit_path_line_egress_distance", 0.200))
        )
        if (
            self.exit_path_initial_line_overlap_allowance > 0.0
            and self.exit_path_line_egress_distance <= 0.0
        ):
            raise rospy.ROSInitException(
                "exit_path_line_egress_distance must be positive when an "
                "exit path line overlap allowance is configured"
            )

        self.footprint = AsymmetricFootprint(
            max(0.01, float(get(p + "footprint/front", 0.067645))),
            max(0.01, float(get(p + "footprint/rear", 0.118073))),
            max(0.01, float(get(p + "footprint/half_width", 0.0903))),
        )
        self.path_safety_margins = SafetyMargins(
            line=max(0.0, float(get(p + "footprint/line_margin", 0.009))),
            localization=max(
                0.0, float(get(p + "footprint/localization_error", 0.005))
            ),
            tracking=max(
                0.0, float(get(p + "footprint/tracking_error", 0.005))
            ),
        )
        self.safety_reaction_time = max(
            0.0, float(get(p + "safety/reaction_time", 0.10))
        )
        self.safety_stop_margin = max(
            0.0, float(get(p + "safety/stop_distance_margin", 0.005))
        )
        self.odom_timeout = max(
            0.05, float(get(p + "safety/odometry_timeout", 0.35))
        )
        self.path_validator = SweptFootprintValidator(
            self.footprint,
            max(0.002, float(get(p + "safety/sweep_step", 0.008))),
            math.radians(
                max(0.1, float(get(p + "safety/sweep_angle_step_deg", 3.0)))
            ),
        )

        # Routes are stored in the surveyed AMCL map. At path generation the
        # current map -> odom TF is snapshotted, and the controller then tracks
        # that frozen path only from /odometry/filtered. Ground Truth is never
        # part of the control loop.
        self.map_frame = str(get(p + "map_frame", "map"))
        self.map_transform_lookup_timeout = max(
            0.01, float(get(p + "map_transform_lookup_timeout", 0.20))
        )
        self.map_transform_max_age = max(
            0.05, float(get(p + "map_transform_max_age", 0.50))
        )
        self.map_world_size = float(get(p + "map_world_size", 4.0))
        self.map_resolution = float(get(p + "map_resolution", 0.01))
        self.map_boundary_inflation = max(
            0.0, float(get(p + "map_boundary_inflation", 0.03))
        )
        self.map_snap_max_distance = float(get(p + "map_snap_max_distance", 0.45))
        self.exit_map_snap_max_distance = max(
            0.01, float(get(p + "exit_map_snap_max_distance", 0.12))
        )
        self.map_texture_package = str(
            get(p + "map_texture_package", "turtlebot3_gazebo")
        )
        self.map_texture_relative_path = str(
            get(
                p + "map_texture_relative_path",
                "models/turtlebot3_autorace_2020/course/materials/textures/course.png",
            )
        )
        self.map_entry_start = self._map_point_parameter(
            get(p + "map_entry_start", [1.395, -0.750]),
            "map_entry_start",
        )
        self.map_entry_start_yaw = math.radians(
            float(get(p + "map_entry_start_yaw_deg", 180.0))
        )
        self.map_left_entry_goal = self._map_point_parameter(
            get(p + "map_left_entry_goal", [1.2522, -0.9257]),
            "map_left_entry_goal",
        )
        self.map_right_entry_goal = self._map_point_parameter(
            get(p + "map_right_entry_goal", [1.2513, -0.5886]),
            "map_right_entry_goal",
        )
        self.map_left_arc_entry_yaw = math.radians(
            float(get(p + "map_left_arc_entry_yaw_deg", -90.0))
        )
        self.map_right_arc_entry_yaw = math.radians(
            float(get(p + "map_right_arc_entry_yaw_deg", 90.0))
        )
        self.map_entry_samples = max(
            20, int(get(p + "map_entry_samples", 100))
        )
        self.entry_alignment_samples = max(
            20, int(get(p + "entry_alignment_samples", 61))
        )
        self.entry_alignment_tangent_ratio = clamp(
            float(get(p + "entry_alignment_tangent_ratio", 0.25)),
            0.05,
            1.50,
        )
        self.entry_start_tangent_ratio = clamp(
            float(get(p + "entry_start_tangent_ratio", 0.50)),
            0.05,
            1.50,
        )
        self.entry_end_tangent_ratio = clamp(
            float(get(p + "entry_end_tangent_ratio", 0.32)),
            0.05,
            1.50,
        )
        self.entry_max_opposite_turn = math.radians(
            max(0.0, float(get(p + "entry_max_opposite_turn_deg", 2.0)))
        )
        self.entry_max_total_turn = math.radians(
            max(1.0, float(get(p + "entry_max_total_turn_deg", 120.0)))
        )

        self.map_left_exit_control_points = tuple(
            self._map_point_parameter(value, "map_left_exit_control_points")
            for value in get(
                p + "map_left_exit_control_points",
                [
                    [0.762200, -1.035300],
                    [0.737117, -0.924915],
                    [0.742982, -0.941341],
                    [0.721843, -0.847479],
                    [0.757456, -0.750000],
                    [0.600000, -0.750000],
                ],
            )
        )
        self.map_right_exit_control_points = tuple(
            self._map_point_parameter(value, "map_right_exit_control_points")
            for value in get(
                p + "map_right_exit_control_points",
                [
                    [0.762200, -0.464700],
                    [0.737117, -0.575085],
                    [0.742982, -0.558659],
                    [0.721843, -0.652521],
                    [0.757456, -0.750000],
                    [0.600000, -0.750000],
                ],
            )
        )
        if (
            len(self.map_left_exit_control_points) < 2
            or len(self.map_right_exit_control_points) < 2
        ):
            raise rospy.ROSInitException(
                "each selected exit branch requires at least two control points"
            )
        self.map_exit_branch_samples = max(
            20, int(get(p + "map_exit_branch_samples", 100))
        )
        self.exit_adaptive_join_ratio = clamp(
            float(get(p + "exit_adaptive_join_ratio", 0.40)),
            0.05,
            0.95,
        )
        self.exit_adaptive_tangent_ratio = clamp(
            float(get(p + "exit_adaptive_tangent_ratio", 0.35)),
            0.05,
            0.80,
        )
        self.exit_adaptive_connector_samples = max(
            20, int(get(p + "exit_adaptive_connector_samples", 41))
        )

        # After the entry connector, normal lane following owns only the
        # painted semicircle. The intersection controller takes cmd_vel back
        # before visual convergence and follows the selected fixed branch.
        self.arc_lane_timeout = max(
            1.0, float(get(p + "arc_lane_timeout", 20.0))
        )
        self.arc_lane_max_distance = max(
            0.05,
            float(get(p + "arc_lane_max_distance", 1.30)),
        )
        self.arc_lane_max_velocity = max(
            0.01, float(get(p + "arc_lane_max_velocity", 0.12))
        )
        self.lane_resume_max_velocity = max(
            self.arc_lane_max_velocity,
            float(get(p + "lane_resume_max_velocity", 0.30)),
        )
        # After revoking the lane controller, hold zero until at least one new
        # EKF odometry sample arrives. This prevents the exit path from being
        # anchored to the pose cached before the cmd_vel ownership transfer.
        self.exit_takeover_settle_time = max(
            0.0, float(get(p + "exit_takeover_settle_time", 0.10))
        )
        self.exit_takeover_pose_timeout = max(
            self.exit_takeover_settle_time + 0.05,
            float(get(p + "exit_takeover_pose_timeout", 0.75)),
        )
        self.exit_takeover_max_distance = max(
            0.01, float(get(p + "exit_takeover_max_distance", 0.06))
        )

        self.map_exit_control_points = tuple(
            self._map_point_parameter(value, "map_exit_control_points")
            for value in get(
                p + "map_exit_control_points",
                [
                    [0.600000, -0.750000],
                    [0.341036, -0.750000],
                    [0.295513, -0.678066],
                    [0.256661, -0.509532],
                    [0.250000, -0.615049],
                    [0.250000, -0.300000],
                ],
            )
        )
        if len(self.map_exit_control_points) < 2:
            raise rospy.ROSInitException(
                "map_exit_control_points requires at least two points"
            )
        self.map_exit_samples = max(
            20, int(get(p + "map_exit_samples", 180))
        )
        self.exit_goal_yaw = math.radians(
            float(get(p + "exit_goal_yaw_deg", 90.0))
        )
        self.exit_path_lookahead = max(
            0.005, float(get(p + "exit_path_lookahead", 0.035))
        )
        self.exit_path_goal_tolerance = max(
            0.01, float(get(p + "exit_path_goal_tolerance", 0.025))
        )
        self.exit_path_goal_heading_tolerance = math.radians(
            abs(float(get(p + "exit_path_goal_heading_tolerance_deg", 8.0)))
        )
        self.exit_path_goal_crossing_max_distance = max(
            self.exit_path_goal_tolerance,
            float(get(p + "exit_path_goal_crossing_max_distance", 0.12)),
        )
        self.exit_path_min_velocity = max(
            0.0, float(get(p + "exit_path_min_velocity", 0.04))
        )
        self.exit_path_velocity = max(
            0.0, float(get(p + "exit_path_linear_velocity", 0.08))
        )
        self.exit_path_entry_velocity = max(
            0.005,
            float(get(p + "exit_path_entry_velocity", self.exit_path_velocity)),
        )
        self.exit_path_exit_velocity = max(
            0.005,
            float(get(p + "exit_path_exit_velocity", self.exit_path_velocity)),
        )
        self.exit_path_max_angular = abs(
            float(get(p + "exit_path_max_angular_velocity", 0.90))
        )
        self.exit_path_follow_timeout = max(
            1.0, float(get(p + "exit_path_follow_timeout", 8.0))
        )
        # The mission only observes the outgoing corridor here. Camera-path
        # construction and all lane steering remain in safe_lane_controller.
        self.image_center = float(get(p + "boundary_center_x", 500.0))
        self.boundary_timeout = float(get(p + "boundary_timeout", 0.35))
        self.zone_signal_timeout = max(
            0.05, float(get(p + "zone_signal_timeout", 0.50))
        )
        self.final_lane_confirm_frames = max(
            1, int(get(p + "final_lane_confirm_frames", 9))
        )
        self.final_lane_confirmation_max_gap = max(
            0.05, float(get(p + "final_lane_confirmation_max_gap", 0.20))
        )
        self.final_lane_width_min = max(
            0.0, float(get(p + "final_lane_width_min", 120.0))
        )
        self.final_lane_width_max = max(
            self.final_lane_width_min,
            float(get(p + "final_lane_width_max", 900.0)),
        )
        self.final_lane_center_tolerance = max(
            0.0, float(get(p + "final_lane_center_tolerance", 220.0))
        )
        self.final_lane_heading_tolerance = math.radians(
            abs(float(get(p + "final_lane_heading_tolerance_deg", 20.0)))
        )
        self.final_lane_join_velocity = max(
            0.0, float(get(p + "final_lane_join_velocity", 0.04))
        )
        self.final_lane_verify_timeout = max(
            0.5, float(get(p + "final_lane_verify_timeout", 6.0))
        )
        self.final_lane_verify_max_distance = max(
            0.05, float(get(p + "final_lane_verify_max_distance", 0.40))
        )

        self.lock = threading.RLock()
        self.state = self.WAIT_INTERSECTION
        self.state_started = rospy.Time.now()
        self.pose_ready = False
        self.last_odom_time = None
        self.odom_frame = "odom"
        self.x = self.y = self.yaw = 0.0
        self.odom_linear_velocity = 0.0
        self.odom_angular_velocity = 0.0
        self.last_x = self.last_y = self.last_wrapped_yaw = None
        self.odom_sequence = 0
        self.total_distance = 0.0
        self.direction_candidate = self.direction = self.NONE
        self.direction_count = 0
        self.last_direction_confirmation_time = None
        self.camera_width = max(1, int(get(p + "camera_width_fallback", 640)))
        self.camera_height = max(1, int(get(p + "camera_height_fallback", 480)))
        self.yellow_x = self.white_x = math.nan
        self.yellow_valid = self.white_valid = False
        self.last_boundary_time = None
        self.final_lane_count = 0
        self.last_final_lane_confirmation_time = None
        self.mission_has_control = False
        self.handoff_ambiguous = False
        self.shutting_down = False
        self.manual_stop = False
        self.pause_started = None
        self.path = None
        self.path_index = 0
        self.path_follower = None
        self.active_tracking_from_map = None
        self.active_path_stage = ""
        self.path_goal_yaw = 0.0
        self.path_exit_yaw = 0.0
        self.path_started = None
        self.mission_started = None
        self.entry_elapsed = math.nan
        self.entry_goal_error = math.nan
        self.entry_yaw_error = math.nan
        self.exit_path_elapsed = math.nan
        self.exit_path_goal_error = math.nan
        self.exit_path_yaw_error = math.nan
        self.active_path_goal_tolerance = self.path_goal_tolerance
        self.active_path_goal_heading_tolerance = (
            self.path_goal_heading_tolerance
        )
        self.active_path_goal_crossing_max_distance = (
            self.path_goal_crossing_max_distance
        )
        self.active_path_lookahead = self.path_lookahead
        self.active_path_min_velocity = self.path_min_velocity
        self.active_path_entry_velocity = self.path_entry_velocity
        self.active_path_exit_velocity = self.path_exit_velocity
        self.active_path_follow_timeout = self.path_follow_timeout
        self.path_max_commanded_angular = 0.0
        self.last_path_command_time = None
        self.path_tick_pose = None
        self.path_tick_tracking = None
        self.arc_lane_start_distance = None
        self.entry_takeover_odom_sequence = -1
        self.exit_takeover_odom_sequence = -1
        self.final_lane_verify_start_distance = None
        self.localized_map_pose_ready = False
        self.localized_map_x = self.localized_map_y = self.localized_map_yaw = 0.0
        self.last_localized_map_pose_time = None
        self.course_occupied = None
        self.course_raw_occupied = None
        self.course_boundary_points = np.empty((0, 2), dtype=np.float64)
        self.course_boundary_cell_size = None
        self.zone_gate_open = False
        self.direction_observation_inside = False
        self.last_direction_observation_time = None
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.lane_speed_limit_pub = rospy.Publisher(
            self.lane_speed_limit_topic, Float64, queue_size=1, latch=True
        )
        self.emergency_stop_pub = rospy.Publisher(
            self.manual_stop_topic, Bool, queue_size=1, latch=True
        )
        self.path_pub = rospy.Publisher(
            "/intersection/generated_path", Path, queue_size=1, latch=True
        )
        self.course_map_pub = rospy.Publisher(
            "/intersection/course_map", OccupancyGrid, queue_size=1, latch=True
        )
        self.diagnostics_pub = rospy.Publisher(
            "/intersection/diagnostics",
            Float64MultiArray,
            queue_size=1,
            latch=True,
        )
        self.state_pub = rospy.Publisher("/intersection/state", String, queue_size=1, latch=True)
        self.direction_pub = rospy.Publisher("/intersection/direction", UInt8, queue_size=1, latch=True)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(
            self.mission_map_pose_topic,
            PoseStamped,
            self.mission_map_pose_callback,
            queue_size=1,
        )
        rospy.Subscriber(self.sign_topic, TrafficSign, self.sign_callback, queue_size=1)
        rospy.Subscriber(
            self.camera_info_topic,
            CameraInfo,
            self.camera_info_callback,
            queue_size=1,
        )
        rospy.Subscriber(self.boundary_topic, Float64MultiArray, self.boundary_callback, queue_size=1)
        rospy.Subscriber(self.manual_stop_topic, Bool, self.manual_stop_callback, queue_size=1)
        rospy.Subscriber(
            self.zone_gate_topic, Bool, self.zone_gate_callback, queue_size=1
        )
        rospy.Subscriber(
            self.direction_observation_topic,
            Bool,
            self.direction_observation_callback,
            queue_size=1,
        )
        self.lane_service = rospy.ServiceProxy(self.lane_service_name, SetBool)
        self.lane_stop_service = rospy.ServiceProxy(
            self.lane_stop_service_name, SetBool
        )
        self.timer = rospy.Timer(rospy.Duration(0.05), self.control_callback)
        rospy.on_shutdown(self.shutdown)
        self._publish_state()
        self._publish_course_map()
        rospy.loginfo(
            "Intersection route waiting for the independent AMCL direction "
            "window and ordered mission gate"
        )

    def _publish_state(self):
        self.state_pub.publish(String(data=self.state))

    def _set_state(self, state):
        self.state = state
        self.state_started = rospy.Time.now()
        self._publish_state()
        rospy.loginfo("Intersection mission state: %s", state)

    def _state_age(self):
        return (rospy.Time.now() - self.state_started).to_sec()

    def odom_callback(self, message):
        with self.lock:
            x, y = message.pose.pose.position.x, message.pose.pose.position.y
            q = message.pose.pose.orientation
            wrapped = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                 1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
            if self.last_x is not None:
                self.total_distance += math.hypot(x - self.last_x, y - self.last_y)
            self.yaw = wrapped if self.last_wrapped_yaw is None else self.yaw + normalize_angle(wrapped - self.last_wrapped_yaw)
            self.x, self.y = x, y
            self.odom_linear_velocity = float(message.twist.twist.linear.x)
            self.odom_angular_velocity = float(message.twist.twist.angular.z)
            self.last_x, self.last_y, self.last_wrapped_yaw = x, y, wrapped
            self.odom_frame = message.header.frame_id or "odom"
            self.pose_ready = True
            self.last_odom_time = rospy.Time.now()
            self.odom_sequence += 1

    def mission_map_pose_callback(self, message):
        """Store the AMCL/odom-propagated pose used for physical region gates."""
        stamp = message.header.stamp
        if stamp == rospy.Time():
            stamp = rospy.Time.now()
        pose = message.pose
        with self.lock:
            self.localized_map_x = float(pose.position.x)
            self.localized_map_y = float(pose.position.y)
            self.localized_map_yaw = yaw_from_quaternion(pose.orientation)
            self.last_localized_map_pose_time = stamp
            self.localized_map_pose_ready = True

    def zone_gate_callback(self, message):
        with self.lock:
            was_open = self.zone_gate_open
            self.zone_gate_open = bool(message.data)
        if self.zone_gate_open and not was_open:
            rospy.loginfo(
                "Ordered intersection mission gate enabled"
            )

    def direction_observation_callback(self, message):
        with self.lock:
            self.direction_observation_inside = bool(message.data)
            self.last_direction_observation_time = rospy.Time.now()

    def _zone_signal_is_fresh(self, timestamp, require_after_state=False):
        if timestamp is None:
            return False
        if require_after_state and timestamp < self.state_started:
            return False
        return (
            rospy.Time.now() - timestamp
        ).to_sec() <= self.zone_signal_timeout

    def _direction_observation_region_ready(self):
        return (
            self.direction_observation_inside
            and self._zone_signal_is_fresh(self.last_direction_observation_time)
        )

    def _localized_map_pose_is_fresh(self):
        return (
            self.localized_map_pose_ready
            and self._zone_signal_is_fresh(self.last_localized_map_pose_time)
        )

    def _entry_handoff_progress(self):
        """Signed progress through the map-entry start normal plane."""
        delta_x = self.localized_map_x - self.map_entry_start[0]
        delta_y = self.localized_map_y - self.map_entry_start[1]
        return (
            delta_x * math.cos(self.map_entry_start_yaw)
            + delta_y * math.sin(self.map_entry_start_yaw)
        )

    def _entry_handoff_pose_ready(self):
        return (
            self._localized_map_pose_is_fresh()
            and self._entry_handoff_progress()
            >= -self.entry_handoff_lead_distance
        )

    def camera_info_callback(self, message):
        if message.width <= 0 or message.height <= 0:
            return
        with self.lock:
            self.camera_width = int(message.width)
            self.camera_height = int(message.height)

    def _begin_direction_search(self):
        # Preserve valid observations collected in the upstream direction
        # window.  A stale candidate is reset by the normal maximum-gap check
        # when the next sign message arrives.
        self._set_state(self.SEARCH_DIRECTION)
        rospy.loginfo(
            "Ordered intersection gate opened without a preconfirmed "
            "direction; timed search started while lane following continues"
        )

    def _direction_observation(self, message):
        if message.sign_type != TrafficSign.DIRECTION:
            return None
        if message.direction == TrafficSign.DIRECTION_LEFT:
            detected = self.LEFT
        elif message.direction == TrafficSign.DIRECTION_RIGHT:
            detected = self.RIGHT
        else:
            return None
        roi = message.roi
        if roi.width <= 0 or roi.height <= 0:
            return None
        frame_area = float(self.camera_width * self.camera_height)
        area_ratio = float(roi.width * roi.height) / frame_area
        center_ratio = (float(roi.x_offset) + 0.5 * float(roi.width)) / float(
            self.camera_width
        )
        acquiring = self.direction_candidate == self.NONE
        minimum_confidence = (
            self.direction_acquire_min_confidence
            if acquiring
            else self.direction_tracking_min_confidence
        )
        minimum_area_ratio = (
            self.direction_acquire_min_roi_area_ratio
            if acquiring
            else self.direction_tracking_min_roi_area_ratio
        )
        if message.confidence < minimum_confidence or area_ratio < minimum_area_ratio:
            return None
        return detected, center_ratio, area_ratio, message.confidence

    def sign_callback(self, message):
        with self.lock:
            if self.state not in (self.WAIT_INTERSECTION, self.SEARCH_DIRECTION):
                return
            # The independent AMCL direction window opens before the ordered
            # mission gate.  Only the direction is cached here; lane control
            # remains authoritative until the separately surveyed entry plane.
            if not self._direction_observation_region_ready():
                return
            observation = self._direction_observation(message)
            if observation is None:
                return
            detected, center_ratio, area_ratio, confidence = observation
            now = rospy.Time.now()
            if self.direction in (self.LEFT, self.RIGHT):
                return
            if (
                area_ratio < self.direction_confirm_min_roi_area_ratio
                or confidence < self.direction_confirm_min_confidence
            ):
                self.direction_candidate = self.NONE
                self.direction_count = 0
                self.last_direction_confirmation_time = None
                return

            selected = (
                self.forced_direction
                if self.forced_direction in (self.LEFT, self.RIGHT)
                else detected
            )
            separated = (
                self.last_direction_confirmation_time is None
                or (now - self.last_direction_confirmation_time).to_sec()
                > self.direction_confirmation_max_gap
            )
            if separated or selected != self.direction_candidate:
                self.direction_candidate, self.direction_count = selected, 1
                rospy.loginfo(
                    "Direction sign usable: confidence=%.2f area=%.1f%% "
                    "center=%.1f%%",
                    confidence,
                    100.0 * area_ratio,
                    100.0 * center_ratio,
                )
            else:
                self.direction_count += 1
            self.last_direction_confirmation_time = now
            if self.direction_count >= self.direction_confirm_frames:
                self.direction = self.direction_candidate
                self.direction_pub.publish(UInt8(data=self.direction))
                rospy.loginfo(
                    "Direction confirmed %d/%d without pre-path steering "
                    "(image center %.1f%%)",
                    self.direction_count,
                    self.direction_confirm_frames,
                    100.0 * center_ratio,
                )
                if self.state == self.SEARCH_DIRECTION:
                    self._set_state(self.WAIT_ENTRY_HANDOFF)
                    rospy.loginfo(
                        "Direction latched after the ordered gate; lane "
                        "following continues to the surveyed AMCL entry plane"
                    )
                else:
                    rospy.loginfo(
                        "Direction preconfirmed in the upstream AMCL window; "
                        "waiting for the ordered gate while lane following "
                        "continues"
                    )

    def boundary_callback(self, message):
        if len(message.data) < 4:
            return
        with self.lock:
            self.yellow_x, self.white_x = message.data[0], message.data[1]
            self.yellow_valid = message.data[2] > 0.5
            self.white_valid = message.data[3] > 0.5
            now = rospy.Time.now()
            self.last_boundary_time = now
            if self.state == self.VERIFY_FINAL_LANE:
                (
                    self.final_lane_count,
                    self.last_final_lane_confirmation_time,
                ) = self._update_lane_confirmation(
                    self.final_lane_count,
                    self.last_final_lane_confirmation_time,
                    self._final_lane_observation_valid(now),
                    now,
                    self.final_lane_confirmation_max_gap,
                )

    def _final_lane_observation_valid(self, now):
        boundary_fresh = (
            self.last_boundary_time is not None
            and (now - self.last_boundary_time).to_sec()
            <= self.boundary_timeout
        )
        width = (
            self.white_x - self.yellow_x
            if self.yellow_valid and self.white_valid
            else math.nan
        )
        boundary_center = (
            0.5 * (self.yellow_x + self.white_x)
            if self.yellow_valid and self.white_valid
            else math.nan
        )
        _, _, tracking_yaw = self._tracking_pose()
        return (
            boundary_fresh
            and self.yellow_valid
            and self.white_valid
            and math.isfinite(width)
            and self.final_lane_width_min <= width <= self.final_lane_width_max
            and math.isfinite(boundary_center)
            and abs(boundary_center - self.image_center)
            <= self.final_lane_center_tolerance
            and abs(normalize_angle(self.path_exit_yaw - tracking_yaw))
            <= self.final_lane_heading_tolerance
        )

    @staticmethod
    def _update_lane_confirmation(count, last_time, valid, now, maximum_gap):
        if not valid:
            return 0, None
        if last_time is None or (now - last_time).to_sec() > maximum_gap:
            return 1, now
        return count + 1, now

    def manual_stop_callback(self, message):
        with self.lock:
            requested = bool(message.data)
            if requested == self.manual_stop:
                return
            now = rospy.Time.now()
            if requested:
                self.manual_stop, self.pause_started = True, now
                if self.path_follower is not None:
                    self.path_follower.last_linear = 0.0
                    self.path_follower.last_angular = 0.0
                    self.last_path_command_time = None
                if self.state == self.VERIFY_FINAL_LANE:
                    self.final_lane_count = 0
                    self.last_final_lane_confirmation_time = None
                if self.mission_has_control:
                    self.cmd_pub.publish(Twist())
            else:
                if self.pause_started is not None:
                    paused_duration = now - self.pause_started
                    self.state_started += paused_duration
                    if self.path_started is not None:
                        self.path_started += paused_duration
                self.manual_stop, self.pause_started = False, None

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
            rospy.logerr("Lane controller service failed: %s", error)
            return False

    def _stop_lane_controller(self):
        try:
            rospy.wait_for_service(self.lane_stop_service_name, timeout=1.0)
            response = self.lane_stop_service(False)
            if not response.success:
                raise rospy.ServiceException(response.message)
            return True
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logerr("Lane controller emergency stop failed: %s", error)
            return False

    def _fail(self, reason):
        self.lane_speed_limit_pub.publish(Float64(data=0.0))
        owns_unambiguously = self.mission_has_control and not self.handoff_ambiguous
        if not owns_unambiguously:
            owns_unambiguously = self._set_lane_controller(False)
        if owns_unambiguously:
            self.cmd_pub.publish(Twist())
        else:
            # A timed-out service may have applied either ownership state. Do
            # not become a second /cmd_vel publisher. Relinquish direct output,
            # stop the lane publisher through its independent service, and
            # assert the shared pause observed by every mission controller.
            self.mission_has_control = False
            if not self._stop_lane_controller():
                rospy.logfatal(
                    "Intersection could not establish a sole cmd_vel stop owner"
                )
            self.emergency_stop_pub.publish(Bool(data=True))
        rospy.logerr("Intersection mission failed: %s", reason)
        self._set_state(self.FAILED)

    def _begin_arc_lane_follow(self):
        """Hand the selected semicircle directly to the rolling camera path."""
        self.lane_speed_limit_pub.publish(
            Float64(data=self.arc_lane_max_velocity)
        )
        self.arc_lane_start_distance = self.total_distance
        if not self._set_lane_controller(True):
            self._fail("could not hand off the selected arc lane")
            return False
        rospy.loginfo(
            "Selected %s arc handed directly to the rolling camera path "
            "follower",
            "LEFT" if self.direction == self.LEFT else "RIGHT",
        )
        self._set_state(self.FOLLOW_ARC_LANE)
        return True

    def _begin_final_lane_verification(self):
        """Restore rolling-path control, then observe the outgoing corridor."""
        self.lane_speed_limit_pub.publish(
            Float64(data=self.final_lane_join_velocity)
        )
        self.final_lane_verify_start_distance = self.total_distance
        self.final_lane_count = 0
        self.last_final_lane_confirmation_time = None
        if not self._set_lane_controller(True):
            self._fail("could not return the final lane to normal control")
            return False
        self._set_state(self.VERIFY_FINAL_LANE)
        return True

    def control_callback(self, _event):
        with self.lock:
            if self.shutting_down:
                return
            if not self.pose_ready:
                rospy.logwarn_throttle(
                    2.0,
                    "Waiting for intersection control odometry on %s",
                    self.odom_topic,
                )
                return
            if (
                self.last_odom_time is None
                or (rospy.Time.now() - self.last_odom_time).to_sec()
                > self.odom_timeout
            ):
                if self.mission_has_control:
                    self.cmd_pub.publish(Twist())
                rospy.logwarn_throttle(
                    1.0, "Intersection holds zero for stale control odometry"
                )
                return
            if self.manual_stop:
                if self.mission_has_control:
                    self.cmd_pub.publish(Twist())
                return

            if self.state == self.WAIT_INTERSECTION:
                if self.zone_gate_open:
                    if self.direction in (self.LEFT, self.RIGHT):
                        self._set_state(self.WAIT_ENTRY_HANDOFF)
                        rospy.loginfo(
                            "Ordered intersection gate accepted the "
                            "preconfirmed direction; lane following continues "
                            "to the surveyed AMCL entry plane"
                        )
                    elif self._direction_observation_region_ready():
                        self._begin_direction_search()

            elif self.state == self.SEARCH_DIRECTION:
                if self._state_age() > self.direction_search_timeout:
                    self._fail("direction sign search timed out inside polygon")

            elif self.state == self.WAIT_ENTRY_HANDOFF:
                if self._entry_handoff_pose_ready():
                    if not self._set_lane_controller(False):
                        self._fail("could not acquire cmd_vel control")
                        return
                    # The service call runs while this controller lock is
                    # held, so callbacks cannot refresh x/y/yaw during the
                    # ownership transfer. Flush the previous owner and defer
                    # path creation until at least one later EKF sample.
                    self.cmd_pub.publish(Twist())
                    self.entry_takeover_odom_sequence = self.odom_sequence
                    rospy.loginfo(
                        "Intersection acquired cmd_vel at the AMCL entry "
                        "plane; waiting for a post-handoff EKF pose"
                    )
                    self._set_state(self.PREPARE_ENTRY_PATH)
                    return
                elif self._state_age() > self.entry_handoff_timeout:
                    self._fail(
                        "AMCL entry handoff plane timed out "
                        "(progress=%.3fm map_pose_fresh=%s)"
                        % (
                            self._entry_handoff_progress(),
                            self._localized_map_pose_is_fresh(),
                        )
                    )

            elif self.state == self.PREPARE_ENTRY_PATH:
                # A short zero barrier establishes one command owner and lets
                # odometry expose all motion that occurred inside the service
                # call before the aligned entry path is frozen.
                self.cmd_pub.publish(Twist())
                if self.odom_sequence > self.entry_takeover_odom_sequence:
                    try:
                        generated = self._generate_map_entry_path()
                    except (ArithmeticError, TypeError, ValueError) as error:
                        rospy.logerr(
                            "Arc-entry path generation raised: %s", error
                        )
                        generated = False
                    if not generated:
                        self._fail("arc-entry path planning failed")
                        return
                    self.mission_started = rospy.Time.now()
                    self._set_state(self.FOLLOW_ENTRY_PATH)
                    return
                if self._state_age() > self.entry_takeover_pose_timeout:
                    self._fail(
                        "no post-handoff EKF pose for the arc-entry path: "
                        "odom_sequence=%d takeover_sequence=%d"
                        % (
                            self.odom_sequence,
                            self.entry_takeover_odom_sequence,
                        )
                    )
                    return

            elif self.state == self.FOLLOW_ENTRY_PATH:
                if self._path_goal_reached():
                    pose_x, pose_y, tracking_yaw = self._tracking_pose()
                    yaw_error = normalize_angle(self.path_exit_yaw - tracking_yaw)
                    self.entry_elapsed = (
                        rospy.Time.now() - self.path_started
                    ).to_sec()
                    self.entry_goal_error = math.hypot(
                        self.path[-1][0] - pose_x,
                        self.path[-1][1] - pose_y,
                    )
                    self.entry_yaw_error = math.degrees(yaw_error)
                    rospy.loginfo(
                        "Arc-entry endpoint reached: elapsed=%.3fs "
                        "goal_error=%.3fm yaw_error=%.1fdeg",
                        self.entry_elapsed,
                        self.entry_goal_error,
                        self.entry_yaw_error,
                    )
                    # Common goal_status already requires this segment's
                    # endpoint and heading tolerances. Do not add a second
                    # mission-specific rotate-in-place tracking formula.
                    self._begin_arc_lane_follow()
                    return
                if (
                    self.path_started is not None
                    and (rospy.Time.now() - self.path_started).to_sec()
                    > self.active_path_follow_timeout
                ):
                    self._fail("arc-entry path tracking timed out")
                    return
                self.cmd_pub.publish(self._path_command())

            elif self.state == self.FOLLOW_ARC_LANE:
                arc_distance = (
                    0.0
                    if self.arc_lane_start_distance is None
                    else self.total_distance - self.arc_lane_start_distance
                )
                map_pose_fresh = self._localized_map_pose_is_fresh()
                exit_projection = self._selected_exit_map_projection()
                exit_path_distance = exit_projection.distance
                # mission_map_pose is the same AMCL sample used by the map to
                # odom aligner. Projecting onto the whole selected branch is
                # important: a delayed camera frame can put the robot past the
                # first stored point while it is still correctly on the path.
                arc_end_ready = bool(
                    map_pose_fresh
                    and exit_path_distance <= self.exit_takeover_max_distance
                )
                if arc_end_ready:
                    if not self._set_lane_controller(False):
                        self._fail("could not reacquire control at the arc exit")
                        return
                    # Flush any command queued by the previous owner. Exit
                    # planning is deferred to PREPARE_EXIT_PATH so odometry can
                    # update after the ownership transfer.
                    self.cmd_pub.publish(Twist())
                    self.exit_takeover_odom_sequence = self.odom_sequence
                    rospy.loginfo(
                        "AMCL confirmed the %s semicircle end; intersection "
                        "controller reacquired cmd_vel after %.3fm; waiting "
                        "for a post-handoff EKF pose",
                        "LEFT" if self.direction == self.LEFT else "RIGHT",
                        arc_distance,
                    )
                    self._set_state(self.PREPARE_EXIT_PATH)
                    return
                if (
                    self._state_age() > self.arc_lane_timeout
                    or arc_distance > self.arc_lane_max_distance
                ):
                    self._fail(
                        "semicircle-end detection failed: age=%.2fs "
                        "distance=%.3fm map_pose_fresh=%s "
                        "exit_path_distance=%.3fm station=%.3fm (limit %.3fm)"
                        % (
                            self._state_age(),
                            arc_distance,
                            map_pose_fresh,
                            exit_path_distance,
                            exit_projection.station,
                            self.exit_takeover_max_distance,
                        )
                    )

            elif self.state == self.PREPARE_EXIT_PATH:
                # Repeated zero commands form a short barrier between the two
                # independent cmd_vel publishers. Planning starts only from a
                # pose sampled after that barrier, never from the lane owner's
                # cached pre-handoff pose.
                self.cmd_pub.publish(Twist())
                pose_updated = (
                    self.odom_sequence > self.exit_takeover_odom_sequence
                )
                settled = self._state_age() >= self.exit_takeover_settle_time
                if pose_updated and settled:
                    generation_result = self._generate_exit_path()
                    if generation_result is False:
                        self._fail("shared exit path planning failed")
                        return
                    if generation_result is True:
                        self._set_state(self.FOLLOW_EXIT_PATH)
                        return
                if self._state_age() > self.exit_takeover_pose_timeout:
                    self._fail(
                        "no usable post-handoff EKF/map transform for the "
                        "shared exit path: "
                        "odom_sequence=%d takeover_sequence=%d"
                        % (
                            self.odom_sequence,
                            self.exit_takeover_odom_sequence,
                        )
                    )
                    return

            elif self.state == self.FOLLOW_EXIT_PATH:
                if self._path_goal_reached():
                    pose_x, pose_y, tracking_yaw = self._tracking_pose()
                    yaw_error = normalize_angle(self.path_exit_yaw - tracking_yaw)
                    self.exit_path_elapsed = (
                        rospy.Time.now() - self.path_started
                    ).to_sec()
                    self.exit_path_goal_error = math.hypot(
                        self.path[-1][0] - pose_x,
                        self.path[-1][1] - pose_y,
                    )
                    self.exit_path_yaw_error = math.degrees(yaw_error)
                    rospy.loginfo(
                        "Shared exit-path endpoint reached: elapsed=%.3fs "
                        "goal_error=%.3fm yaw_error=%.1fdeg",
                        self.exit_path_elapsed,
                        self.exit_path_goal_error,
                        self.exit_path_yaw_error,
                    )
                    # Heading completion belongs to the common path goal.
                    self._begin_final_lane_verification()
                    return
                if (
                    self.path_started is not None
                    and (rospy.Time.now() - self.path_started).to_sec()
                    > self.active_path_follow_timeout
                ):
                    self._fail("shared exit path tracking timed out")
                    return
                self.cmd_pub.publish(self._path_command())

            elif self.state == self.VERIFY_FINAL_LANE:
                now = rospy.Time.now()
                confirmation_fresh = (
                    self.last_final_lane_confirmation_time is not None
                    and (
                        now - self.last_final_lane_confirmation_time
                    ).to_sec()
                    <= self.final_lane_confirmation_max_gap
                )
                final_lane_ready = (
                    self._final_lane_observation_valid(now)
                    and confirmation_fresh
                    and self.final_lane_count >= self.final_lane_confirm_frames
                )
                if final_lane_ready:
                    rospy.loginfo(
                        "Outgoing corridor confirmed between yellow and white "
                        "lines in %d frames while rolling-path control remained "
                        "active",
                        self.final_lane_count,
                    )
                    self._complete()
                    return
                verify_distance = (
                    0.0
                    if self.final_lane_verify_start_distance is None
                    else self.total_distance - self.final_lane_verify_start_distance
                )
                if (
                    self._state_age() > self.final_lane_verify_timeout
                    or verify_distance > self.final_lane_verify_max_distance
                ):
                    self._fail(
                        "final two-line corridor confirmation failed: "
                        "age=%.2fs distance=%.3fm valid_frames=%d/%d"
                        % (
                            self._state_age(),
                            verify_distance,
                            self.final_lane_count,
                            self.final_lane_confirm_frames,
                        )
                    )
                    return

            elif self.state == self.FAILED:
                if self.mission_has_control:
                    self.cmd_pub.publish(Twist())

    def _tracking_pose(self):
        """Return the sole local control pose: encoder/IMU EKF odometry."""
        return self.x, self.y, self.yaw

    def _lookup_tracking_from_map(self):
        """Snapshot T_odom_map for one generated path.

        AMCL remains the only map -> odom authority. Freezing this transform
        for each short path avoids steering jumps if AMCL corrects while the
        path is being tracked.
        """
        if self.odom_frame == self.map_frame:
            return 0.0, 0.0, 0.0
        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.map_frame,
                rospy.Time(0),
                rospy.Duration(self.map_transform_lookup_timeout),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as error:
            rospy.logerr(
                "No %s <- %s transform for map path: %s",
                self.odom_frame,
                self.map_frame,
                error,
            )
            return None

        stamp = transform.header.stamp
        if stamp != rospy.Time():
            age = (rospy.Time.now() - stamp).to_sec()
            if age > self.map_transform_max_age:
                rospy.logerr(
                    "%s <- %s transform is %.3fs old (limit %.3fs)",
                    self.odom_frame,
                    self.map_frame,
                    age,
                    self.map_transform_max_age,
                )
                return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            float(translation.x),
            float(translation.y),
            yaw_from_quaternion(rotation),
        )

    @staticmethod
    def _map_to_tracking_point(point, tracking_from_map):
        transform = RigidTransform2D(
            *tracking_from_map, source_frame="map", target_frame="odom"
        )
        return transform.apply_point(point)

    @staticmethod
    def _map_to_tracking_yaw(map_yaw, tracking_from_map):
        transform = RigidTransform2D(
            *tracking_from_map, source_frame="map", target_frame="odom"
        )
        return transform.apply_pose((0.0, 0.0, map_yaw)).yaw

    def _prepare_active_entry_parameters(self):
        self.active_path_velocity = (
            self.left_path_velocity
            if self.direction == self.LEFT
            else self.right_path_velocity
        )
        self.active_path_max_angular = (
            self.left_path_max_angular
            if self.direction == self.LEFT
            else self.right_path_max_angular
        )
        self.active_path_goal_tolerance = self.path_goal_tolerance
        self.active_path_goal_heading_tolerance = (
            self.path_goal_heading_tolerance
        )
        self.active_path_goal_crossing_max_distance = (
            self.path_goal_crossing_max_distance
        )
        self.active_path_lookahead = self.path_lookahead
        self.active_path_min_velocity = self.path_min_velocity
        self.active_path_entry_velocity = min(
            self.active_path_velocity, self.path_entry_velocity
        )
        self.active_path_exit_velocity = min(
            self.active_path_velocity, self.path_exit_velocity
        )
        self.active_path_follow_timeout = self.path_follow_timeout

    def _prepare_active_exit_parameters(self):
        self.active_path_velocity = self.exit_path_velocity
        self.active_path_max_angular = self.exit_path_max_angular
        self.active_path_goal_tolerance = self.exit_path_goal_tolerance
        self.active_path_goal_heading_tolerance = (
            self.exit_path_goal_heading_tolerance
        )
        self.active_path_goal_crossing_max_distance = (
            self.exit_path_goal_crossing_max_distance
        )
        self.active_path_lookahead = self.exit_path_lookahead
        self.active_path_min_velocity = self.exit_path_min_velocity
        self.active_path_entry_velocity = min(
            self.active_path_velocity, self.exit_path_entry_velocity
        )
        self.active_path_exit_velocity = min(
            self.active_path_velocity, self.exit_path_exit_velocity
        )
        self.active_path_follow_timeout = self.exit_path_follow_timeout

    @staticmethod
    def _map_point_parameter(value, name):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise rospy.ROSInitException("%s must contain exactly [x, y]" % name)
        point = (float(value[0]), float(value[1]))
        if not all(math.isfinite(component) for component in point):
            raise rospy.ROSInitException("%s must be finite" % name)
        return point

    @staticmethod
    def _cubic_path(p0, start_yaw, p3, goal_yaw, start_tangent, end_tangent, samples):
        p1 = (
            p0[0] + start_tangent * math.cos(start_yaw),
            p0[1] + start_tangent * math.sin(start_yaw),
        )
        p2 = (
            p3[0] - end_tangent * math.cos(goal_yaw),
            p3[1] - end_tangent * math.sin(goal_yaw),
        )
        path = []
        for index in range(samples):
            t = float(index) / float(samples - 1)
            u = 1.0 - t
            path.append(
                (
                    u ** 3 * p0[0]
                    + 3.0 * u ** 2 * t * p1[0]
                    + 3.0 * u * t ** 2 * p2[0]
                    + t ** 3 * p3[0],
                    u ** 3 * p0[1]
                    + 3.0 * u ** 2 * t * p1[1]
                    + 3.0 * u * t ** 2 * p2[1]
                    + t ** 3 * p3[1],
                )
            )
        return path

    @staticmethod
    def _bezier_path(control_points, samples):
        points = np.asarray(control_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
            raise ValueError("Bezier control points must contain finite [x, y] rows")
        if not np.all(np.isfinite(points)):
            raise ValueError("Bezier control points must be finite")
        degree = points.shape[0] - 1
        result = []
        for parameter in np.linspace(0.0, 1.0, max(2, int(samples))):
            inverse = 1.0 - parameter
            point = np.zeros(2, dtype=np.float64)
            for index, control in enumerate(points):
                weight = (
                    math.comb(degree, index)
                    * inverse ** (degree - index)
                    * parameter ** index
                )
                point += weight * control
            result.append((float(point[0]), float(point[1])))
        return result

    def _activate_path(self, path, goal_yaw, exit_yaw, stage):
        points = list(path)
        if len(points) < 2:
            rospy.logerr("Intersection %s path has fewer than two points", stage)
            return False
        if not self.course_boundary_points.size:
            rospy.logerr(
                "Intersection %s path cannot be validated without course paint",
                stage,
            )
            return False
        map_boundary = AxisAlignedBoundsBoundary(
            -0.5 * self.map_world_size,
            0.5 * self.map_world_size,
            -0.5 * self.map_world_size,
            0.5 * self.map_world_size,
        )
        if self.active_tracking_from_map is not None:
            map_boundary = map_boundary.transformed(
                RigidTransform2D(
                    *self.active_tracking_from_map,
                    source_frame=self.map_frame,
                    target_frame=self.odom_frame,
                )
            )
        if self.course_boundary_cell_size is None:
            rospy.logerr(
                "Intersection %s path has no course raster-cell geometry",
                stage,
            )
            return False
        # Preserve each native texture pixel as its exact rectangular cell.
        # A circumscribed point radius incorrectly grows the middle of every
        # cell edge and can close a physically usable narrow corridor.
        boundary = RasterCellBoundary(
            self.course_boundary_points,
            cell_size=self.course_boundary_cell_size,
        )
        if self.active_tracking_from_map is not None:
            boundary = boundary.transformed(
                RigidTransform2D(
                    *self.active_tracking_from_map,
                    source_frame=self.map_frame,
                    target_frame=self.odom_frame,
                )
            )
        safety = PathSafety(
            line_boundaries=(boundary,),
            map_boundaries=(map_boundary,),
            margins=self.path_safety_margins,
        )
        minimum_velocity = min(
            max(0.005, self.active_path_min_velocity),
            max(0.005, self.active_path_velocity),
        )
        profile = SpeedProfile(
            cruise_velocity=max(0.005, self.active_path_velocity),
            minimum_velocity=minimum_velocity,
            entry_velocity=max(
                minimum_velocity,
                min(
                    max(0.005, self.active_path_entry_velocity),
                    max(0.005, self.active_path_velocity),
                ),
            ),
            exit_velocity=max(
                minimum_velocity,
                min(
                    max(0.005, self.active_path_exit_velocity),
                    max(0.005, self.active_path_velocity),
                ),
            ),
            maximum_angular_velocity=self.active_path_max_angular,
            maximum_lateral_acceleration=self.maximum_lateral_acceleration,
            linear_acceleration=self.path_linear_acceleration,
            linear_deceleration=self.path_linear_deceleration,
            angular_acceleration=self.path_angular_acceleration,
        )
        initial_line_overlap_allowance = (
            self.path_initial_line_overlap_allowance
            if stage == "entry"
            else self.exit_path_initial_line_overlap_allowance
        )
        configured_line_egress_distance = (
            self.path_line_egress_distance
            if stage == "entry"
            else self.exit_path_line_egress_distance
        )
        line_egress_distance = configured_line_egress_distance
        if initial_line_overlap_allowance > 0.0:
            # A short live-pose connector must not carry a residual paint
            # allowance into the camera-followed arc at its endpoint.
            line_egress_distance = min(
                configured_line_egress_distance,
                self._polyline_length(points),
            )
        executable = path_from_xy(
            points,
            self.odom_frame,
            speed_profile=profile,
            direction=1,
            initial_line_overlap_allowance=initial_line_overlap_allowance,
            line_egress_distance=line_egress_distance,
            final_heading=goal_yaw,
            goal_tolerance=GoalTolerance(
                self.active_path_goal_tolerance,
                self.active_path_goal_heading_tolerance,
                self.active_path_goal_crossing_max_distance,
            ),
            safety=safety,
            label="intersection_%s" % stage,
        )
        direction = getattr(self, "direction", self.NONE)
        if stage == "entry" and direction in (self.LEFT, self.RIGHT):
            heading = np.unwrap(np.asarray(executable.heading, dtype=np.float64))
            heading_delta = np.diff(heading)
            selected_sign = 1.0 if direction == self.LEFT else -1.0
            net_turn = selected_sign * float(heading[-1] - heading[0])
            opposite_turn = float(
                np.sum(np.maximum(-selected_sign * heading_delta, 0.0))
            )
            total_turn = float(np.sum(np.abs(heading_delta)))
            if (
                net_turn <= 0.0
                or opposite_turn
                > getattr(
                    self,
                    "entry_max_opposite_turn",
                    math.radians(2.0),
                )
                or total_turn
                > getattr(
                    self,
                    "entry_max_total_turn",
                    math.radians(120.0),
                )
            ):
                rospy.logerr(
                    "Intersection %s entry has the wrong turn topology: "
                    "net=%.1fdeg opposite=%.1fdeg total=%.1fdeg",
                    "LEFT" if direction == self.LEFT else "RIGHT",
                    math.degrees(net_turn),
                    math.degrees(opposite_turn),
                    math.degrees(total_turn),
                )
                return False
        validation = self.path_validator.validate_path(executable)
        if not validation.safe:
            rospy.logerr(
                "Intersection %s rectangular sweep is unsafe: "
                "line=%.4fm map=%.4fm",
                stage,
                validation.minimum_line_clearance,
                validation.minimum_map_clearance,
            )
            return False
        start_validation = self.path_validator.validate_poses(
            [Pose2D(*self._tracking_pose())],
            safety,
            line_overlap_allowances=[
                float(executable.line_overlap_allowance[0])
            ],
        )
        if not start_validation.safe:
            rospy.logerr(
                "Intersection %s start footprint exceeds its common path "
                "envelope: line=%.4fm allowance=%.4fm obstacle=%.4fm map=%.4fm",
                stage,
                start_validation.minimum_line_clearance,
                float(executable.line_overlap_allowance[0]),
                start_validation.minimum_obstacle_clearance,
                start_validation.minimum_map_clearance,
            )
            return False
        if start_validation.minimum_line_clearance <= 0.0:
            rospy.logwarn(
                "Intersection %s starts in %.4fm measured line overlap; "
                "the common %.4fm/%.3fm taper will converge to zero",
                stage,
                -start_validation.minimum_line_clearance,
                float(executable.line_overlap_allowance[0]),
                line_egress_distance,
            )
        executable.line_clearance = min(
            validation.minimum_line_clearance,
            start_validation.minimum_line_clearance,
        )
        executable.obstacle_clearance = min(
            validation.minimum_obstacle_clearance,
            start_validation.minimum_obstacle_clearance,
        )
        executable.map_clearance = min(
            validation.minimum_map_clearance,
            start_validation.minimum_map_clearance,
        )
        follower_config = TrackingConfig(
            lookahead_distance=self.active_path_lookahead,
            maximum_linear_velocity=self.active_path_velocity,
            maximum_angular_velocity=self.active_path_max_angular,
            maximum_lateral_acceleration=self.maximum_lateral_acceleration,
            linear_acceleration=self.path_linear_acceleration,
            linear_deceleration=self.path_linear_deceleration,
            angular_acceleration=self.path_angular_acceleration,
            heading_gain=self.path_heading_gain,
            curvature_feedforward_weight=self.path_curvature_weight,
            lateral_feedback_gain=self.path_lateral_feedback_gain,
            search_ahead_distance=self.path_search_ahead_distance,
        )
        self.path = executable
        self.path_follower = PathFollower(follower_config)
        tracking_pose = Pose2D(*self._tracking_pose())
        now = rospy.Time.now()
        if stage == "entry":
            # PREPARE_ENTRY_PATH has already established a mission-owned zero
            # barrier while waiting for the fresh pose used by this path.
            observed_linear, observed_angular = 0.0, 0.0
        else:
            # PREPARE_EXIT_PATH has already held the mission-owned output at
            # zero through the settle interval. The earlier arc-lane command
            # observed before ownership transfer is no longer command history.
            observed_linear, observed_angular = 0.0, 0.0
        initial_linear = clamp(
            observed_linear,
            -self.active_path_velocity,
            self.active_path_velocity,
        )
        initial_angular = clamp(
            observed_angular,
            -self.active_path_max_angular,
            self.active_path_max_angular,
        )
        self.path_follower.reset(
            executable,
            tracking_pose,
            initial_linear=initial_linear,
            initial_angular=initial_angular,
        )
        self.path_follower.update_clearance(validation)
        self.path_index = self.path_follower.path_index
        self.path_goal_yaw = goal_yaw
        self.path_exit_yaw = exit_yaw
        self.active_path_stage = stage
        self.path_started = now
        self.path_max_commanded_angular = 0.0
        self.last_path_command_time = None
        self.path_tick_pose = None
        self.path_tick_tracking = None
        self._publish_path()
        self._publish_path_diagnostics()
        return True

    def _generate_map_entry_path(self):
        """Select one surveyed map entry and freeze it in the tracking frame."""
        if self.direction not in (self.LEFT, self.RIGHT):
            rospy.logerr("Intersection entry has no valid selected direction")
            return False
        if not self._localized_map_pose_is_fresh():
            rospy.logerr(
                "No fresh AMCL pose has arrived on %s",
                self.mission_map_pose_topic,
            )
            return False
        tracking_from_map = self._lookup_tracking_from_map()
        if tracking_from_map is None:
            return False
        snap_distance = math.hypot(
            self.map_entry_start[0] - self.localized_map_x,
            self.map_entry_start[1] - self.localized_map_y,
        )
        if snap_distance > self.map_snap_max_distance:
            rospy.logerr(
                "Localized pose is %.3f m from the map entry (limit %.3f m)",
                snap_distance,
                self.map_snap_max_distance,
            )
            return False

        if self.direction == self.LEFT:
            goal = self.map_left_entry_goal
            goal_map_yaw = self.map_left_arc_entry_yaw
        else:
            goal = self.map_right_entry_goal
            goal_map_yaw = self.map_right_arc_entry_yaw
        self._prepare_active_entry_parameters()
        tracking_start = Pose2D(*self._tracking_pose())
        tracking_entry = self._map_to_tracking_point(
            self.map_entry_start, tracking_from_map
        )
        entry_yaw = self._map_to_tracking_yaw(
            self.map_entry_start_yaw, tracking_from_map
        )
        tracking_goal = self._map_to_tracking_point(goal, tracking_from_map)
        goal_yaw = self._map_to_tracking_yaw(
            goal_map_yaw, tracking_from_map
        )
        alignment_chord = math.hypot(
            tracking_entry[0] - tracking_start.x,
            tracking_entry[1] - tracking_start.y,
        )
        alignment_tangent = max(
            0.005,
            self.entry_alignment_tangent_ratio * alignment_chord,
        )
        alignment = self._cubic_path(
            (tracking_start.x, tracking_start.y),
            tracking_start.yaw,
            tracking_entry,
            entry_yaw,
            alignment_tangent,
            alignment_tangent,
            self.entry_alignment_samples,
        )
        branch_chord = math.hypot(
            tracking_goal[0] - tracking_entry[0],
            tracking_goal[1] - tracking_entry[1],
        )
        start_tangent = max(
            0.005,
            self.entry_start_tangent_ratio * branch_chord,
        )
        end_tangent = max(
            0.005,
            self.entry_end_tangent_ratio * branch_chord,
        )
        branch = self._cubic_path(
            tracking_entry,
            entry_yaw,
            tracking_goal,
            goal_yaw,
            start_tangent,
            end_tangent,
            self.map_entry_samples,
        )
        route = alignment[:-1] + branch
        self.entry_elapsed = math.nan
        self.entry_goal_error = math.nan
        self.entry_yaw_error = math.nan
        self.active_tracking_from_map = tracking_from_map
        if not self._activate_path(route, goal_yaw, goal_yaw, "entry"):
            return False
        direction = "LEFT" if self.direction == self.LEFT else "RIGHT"
        rospy.loginfo(
            "Selected aligned %s arc-entry path; "
            "start=(%.3f, %.3f, %.1fdeg), "
            "alignment=(%.3f, %.3f, %.1fdeg), "
            "endpoint=(%.3f, %.3f, %.1fdeg), "
            "tangents=(%.3f, %.3f, %.3f)m, "
            "length=%.3fm speed=%.3fm/s",
            direction,
            route[0][0],
            route[0][1],
            math.degrees(tracking_start.yaw),
            tracking_entry[0],
            tracking_entry[1],
            math.degrees(entry_yaw),
            route[-1][0],
            route[-1][1],
            math.degrees(goal_yaw),
            alignment_tangent,
            start_tangent,
            end_tangent,
            self._polyline_length(route),
            self.active_path_velocity,
        )
        return True

    def _generate_exit_path(self):
        """Freeze the selected branch plus shared exit in the tracking frame.

        Return True on success, False for a deterministic unsafe path, and
        None when fresh localization/TF data may still arrive before the
        PREPARE_EXIT_PATH timeout.
        """
        self._prepare_active_exit_parameters()
        if not self._localized_map_pose_is_fresh():
            rospy.logwarn_throttle(
                0.5,
                "Waiting for a fresh AMCL map pose for the shared exit path",
            )
            return None
        # This is a new executable path, so align it once from the latest AMCL
        # map->odom relation after the camera semicircle, then freeze it for
        # the complete branch and shared exit. Reusing the entry snapshot here
        # carries dead-reckoning drift accumulated while vision owns cmd_vel
        # into the branch start.
        tracking_from_map = self._lookup_tracking_from_map()
        if tracking_from_map is None:
            return None
        branch_control_points = self._selected_exit_control_points()
        branch_route = self._bezier_path(
            branch_control_points, self.map_exit_branch_samples
        )
        shared_route = self._bezier_path(
            self.map_exit_control_points, self.map_exit_samples
        )
        map_route = branch_route[:-1] + shared_route
        map_common_path = path_from_xy(
            map_route,
            self.map_frame,
            target_speed=self.active_path_velocity,
        )
        map_projection = project_to_path(
            map_common_path,
            self.localized_map_x,
            self.localized_map_y,
        )
        snap_distance = map_projection.distance
        if snap_distance > self.exit_map_snap_max_distance:
            rospy.logerr(
                "Localized pose is %.3f m from the selected map exit path "
                "(limit %.3f m)",
                snap_distance,
                self.exit_map_snap_max_distance,
            )
            return False
        fixed_branch_path = path_from_xy(
            branch_route,
            self.map_frame,
            target_speed=self.active_path_velocity,
        )
        join_index = int(
            round(
                self.exit_adaptive_join_ratio
                * float(len(branch_route) - 1)
            )
        )
        join_index = max(1, min(len(branch_route) - 2, join_index))
        tracking_start = Pose2D(*self._tracking_pose())
        tracking_join = self._map_to_tracking_point(
            branch_route[join_index], tracking_from_map
        )
        tracking_join_yaw = self._map_to_tracking_yaw(
            float(fixed_branch_path.heading[join_index]),
            tracking_from_map,
        )
        connector_distance = math.hypot(
            tracking_join[0] - tracking_start.x,
            tracking_join[1] - tracking_start.y,
        )
        connector_tangent = max(
            0.005, self.exit_adaptive_tangent_ratio * connector_distance
        )
        connector = self._cubic_path(
            (tracking_start.x, tracking_start.y),
            tracking_start.yaw,
            tracking_join,
            tracking_join_yaw,
            connector_tangent,
            connector_tangent,
            self.exit_adaptive_connector_samples,
        )
        downstream_branch = [
            self._map_to_tracking_point(point, tracking_from_map)
            for point in branch_route[join_index:]
        ]
        downstream_shared = [
            self._map_to_tracking_point(point, tracking_from_map)
            for point in shared_route
        ]
        route = connector[:-1] + downstream_branch[:-1] + downstream_shared
        goal_yaw = self._map_to_tracking_yaw(
            self.exit_goal_yaw, tracking_from_map
        )
        rospy.loginfo(
            "Shared exit alignment: AMCL gate pose=(%.3f, %.3f, %.1fdeg), "
            "path_delta=%.3fm start_station=%.3fm",
            self.localized_map_x,
            self.localized_map_y,
            math.degrees(self.localized_map_yaw),
            snap_distance,
            map_projection.station,
        )

        self.exit_path_elapsed = math.nan
        self.exit_path_goal_error = math.nan
        self.exit_path_yaw_error = math.nan
        self.active_tracking_from_map = tracking_from_map
        if not self._activate_path(route, goal_yaw, goal_yaw, "exit"):
            return False
        rospy.loginfo(
            "Selected and aligned adaptive branch plus shared exit after the %s arc; "
            "start=(%.3f, %.3f, %.1fdeg), join_station=%.3fm, "
            "goal=(%.3f, %.3f, %.1fdeg), "
            "length=%.3fm speed=%.3fm/s",
            "LEFT" if self.direction == self.LEFT else "RIGHT",
            tracking_start.x,
            tracking_start.y,
            math.degrees(tracking_start.yaw),
            float(fixed_branch_path.station[join_index]),
            self.map_exit_control_points[-1][0],
            self.map_exit_control_points[-1][1],
            math.degrees(self.exit_goal_yaw),
            self._polyline_length(route),
            self.active_path_velocity,
        )
        return True

    def _selected_exit_control_points(self):
        if self.direction == self.LEFT:
            return self.map_left_exit_control_points
        if self.direction == self.RIGHT:
            return self.map_right_exit_control_points
        raise ValueError("intersection exit branch requested before selection")

    def _selected_exit_map_projection(self):
        """Project the live AMCL pose onto the selected fixed exit branch."""
        cache = getattr(self, "_exit_branch_path_cache", {})
        branch_path = cache.get(self.direction)
        if branch_path is None:
            branch_route = self._bezier_path(
                self._selected_exit_control_points(),
                self.map_exit_branch_samples,
            )
            branch_path = path_from_xy(
                branch_route,
                self.map_frame,
                target_speed=max(0.005, self.exit_path_velocity),
            )
            cache[self.direction] = branch_path
            self._exit_branch_path_cache = cache
        return project_to_path(
            branch_path,
            self.localized_map_x,
            self.localized_map_y,
        )

    @staticmethod
    def _polyline_length(points):
        return sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(points[:-1], points[1:])
        )

    def _inflate_course_boundaries(self, occupied, inflation):
        """Inflate paint by Euclidean distance instead of a square kernel."""
        radius = max(0.0, float(inflation)) / self.map_resolution
        cells = int(math.ceil(radius))
        if cells <= 0:
            return occupied.copy()
        offsets = np.arange(-cells, cells + 1, dtype=np.float32)
        dx, dy = np.meshgrid(offsets, offsets)
        kernel = ((dx * dx + dy * dy) <= radius * radius + 1e-6).astype(
            np.uint8
        )
        return cv2.dilate(occupied, kernel)

    def _publish_path(self):
        path = Path()
        path.header.stamp, path.header.frame_id = rospy.Time.now(), self.odom_frame
        headings = (
            self.path.heading
            if hasattr(self.path, "heading")
            else np.zeros(len(self.path), dtype=np.float64)
        )
        for (x, y), yaw in zip(self.path, headings):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y = x, y
            # Keep the route visibly above the Gazebo floor in RViz.
            pose.pose.position.z = 0.06
            pose.pose.orientation.z = math.sin(0.5 * float(yaw))
            pose.pose.orientation.w = math.cos(0.5 * float(yaw))
            path.poses.append(pose)
        self.path_pub.publish(path)

    def _publish_path_diagnostics(self):
        if self.path_follower is None:
            return
        message = Float64MultiArray()
        message.data = self.path_follower.diagnostics.as_array()
        self.diagnostics_pub.publish(message)

    def _publish_course_map(self):
        """Publish colour lane boundaries as an RViz OccupancyGrid."""
        try:
            package_path = rospkg.RosPack().get_path(self.map_texture_package)
            texture_path = os.path.join(
                package_path, self.map_texture_relative_path
            )
            texture = cv2.imread(texture_path, cv2.IMREAD_COLOR)
            if texture is None:
                raise IOError("could not read " + texture_path)
        except (rospkg.ResourceNotFound, IOError) as error:
            rospy.logwarn("Course map is unavailable: %s", error)
            return

        # Validator geometry comes directly from the native texture.  The
        # non-square source image is stretched across the square Gazebo model,
        # so x and y pixel sizes must remain distinct.  Gazebo rotates the
        # model by pi, yielding the coordinate conversion below.
        native_blue, native_green, native_red = cv2.split(texture)
        native_white = (
            (native_red > 150)
            & (native_green > 150)
            & (native_blue > 150)
        )
        native_yellow = (
            (native_red > 150)
            & (native_green > 150)
            & (native_blue < 120)
        )
        native_occupied = native_white | native_yellow
        native_y, native_x = np.nonzero(native_occupied)
        texture_height, texture_width = texture.shape[:2]
        native_cell_width = self.map_world_size / float(texture_width)
        native_cell_height = self.map_world_size / float(texture_height)
        self.course_boundary_points = np.column_stack(
            (
                0.5 * self.map_world_size
                - (native_x.astype(np.float64) + 0.5) * native_cell_width,
                -0.5 * self.map_world_size
                + (native_y.astype(np.float64) + 0.5) * native_cell_height,
            )
        )
        self.course_boundary_cell_size = (
            native_cell_width,
            native_cell_height,
        )

        width = max(1, int(round(self.map_world_size / self.map_resolution)))
        height = width
        # Gazebo rotates the square course model by pi. For each ROS grid cell
        # (world x/y increasing), sample the corresponding texture pixel.
        grid_x = -0.5 * self.map_world_size + (
            np.arange(width, dtype=np.float32) + 0.5
        ) * self.map_resolution
        grid_y = -0.5 * self.map_world_size + (
            np.arange(height, dtype=np.float32) + 0.5
        ) * self.map_resolution
        world_x, world_y = np.meshgrid(grid_x, grid_y)
        image_u = np.clip(
            ((0.5 * self.map_world_size - world_x) / self.map_world_size)
            * texture.shape[1],
            0,
            texture.shape[1] - 1,
        ).astype(np.int32)
        image_v = np.clip(
            ((0.5 * self.map_world_size + world_y) / self.map_world_size)
            * texture.shape[0],
            0,
            texture.shape[0] - 1,
        ).astype(np.int32)
        sampled = texture[image_v, image_u]
        blue, green, red = cv2.split(sampled)
        white = (red > 150) & (green > 150) & (blue > 150)
        yellow = (red > 150) & (green > 150) & (blue < 120)
        raw_occupied = (white | yellow).astype(np.uint8)
        self.course_raw_occupied = raw_occupied
        occupied = self._inflate_course_boundaries(
            raw_occupied, self.map_boundary_inflation
        )
        self.course_occupied = occupied

        message = OccupancyGrid()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.map_frame
        message.info.resolution = self.map_resolution
        message.info.width = width
        message.info.height = height
        message.info.origin.position.x = -0.5 * self.map_world_size
        message.info.origin.position.y = -0.5 * self.map_world_size
        message.info.origin.orientation.w = 1.0
        message.data = (occupied.reshape(-1) * 100).astype(np.int8).tolist()
        self.course_map_pub.publish(message)

    def _path_goal_reached(self):
        if self.path is None or self.path_follower is None:
            return False
        pose = Pose2D(*self._tracking_pose())
        tracking = self.path_follower.calculate_tracking(pose)
        status = self.path_follower.goal_status(pose, tracking=tracking)
        self.path_tick_pose = pose
        self.path_tick_tracking = tracking
        self.path_index = self.path_follower.path_index
        self._publish_path_diagnostics()
        return status.complete

    def _path_command(self):
        if self.path is None or self.path_follower is None:
            return Twist()
        now = rospy.Time.now()
        pose = self.path_tick_pose
        tracking = self.path_tick_tracking
        self.path_tick_pose = None
        self.path_tick_tracking = None
        if pose is None or tracking is None:
            pose = Pose2D(*self._tracking_pose())
            tracking = self.path_follower.calculate_tracking(pose)
        measured_speed = max(
            abs(self.odom_linear_velocity),
            abs(self.path_follower.last_linear),
        )
        safety = self.path_validator.motion_safety(
            self.path,
            pose,
            tracking.path_index,
            tracking.target_speed,
            tracking.direction * measured_speed,
            (
                self.odom_angular_velocity,
                self.path_follower.last_angular,
                tracking.angular_velocity,
            ),
            self.safety_reaction_time,
            self.path_linear_deceleration,
            self.safety_stop_margin,
            tracking=tracking,
            # The immutable course route was swept in full at activation.
            # Rechecking every remaining raster cell on every 20 Hz tick
            # starves the follower and itself creates centimetre-scale error.
            # The actual reaction and complete-stop sweep still uses the full
            # path safety below; only the duplicate nominal-route check is
            # omitted.
            route_safety=PathSafety(),
        )
        self.path_follower.update_clearance(safety.validation)
        elapsed = (
            0.05
            if self.last_path_command_time is None
            else clamp((now - self.last_path_command_time).to_sec(), 0.0, 0.15)
        )
        self.last_path_command_time = now
        limited, tracking = self.path_follower.command(
            pose,
            elapsed,
            speed_limit=safety.speed_limit,
            tracking=tracking,
        )
        self.path_index = tracking.path_index
        command = Twist()
        command.linear.x = limited.linear_velocity
        command.angular.z = limited.angular_velocity
        if safety.requires_stop:
            line_allowance = float(
                np.interp(
                    tracking.station,
                    self.path.station,
                    self.path.line_overlap_allowance,
                )
            )
            rospy.logwarn_throttle(
                0.5,
                "Intersection path holds a bounded stop: line=%.4fm "
                "obstacle=%.4fm map=%.4fm allowance=%.4fm "
                "station=%.4fm pose=(%.4f, %.4f, %.1fdeg) "
                "velocity=(%.4f, %.4f)",
                safety.stopping.minimum_line_clearance,
                safety.stopping.minimum_obstacle_clearance,
                safety.stopping.minimum_map_clearance,
                line_allowance,
                tracking.station,
                pose.x,
                pose.y,
                math.degrees(pose.yaw),
                self.odom_linear_velocity,
                self.odom_angular_velocity,
            )
        self.path_max_commanded_angular = max(
            self.path_max_commanded_angular, abs(command.angular.z)
        )
        self._publish_path_diagnostics()
        return command

    def _complete(self):
        progress = 0.0
        mission_elapsed = math.nan
        if self.path:
            progress = 100.0 * float(self.path_index) / float(max(1, len(self.path) - 1))
        if self.mission_started is not None:
            mission_elapsed = (rospy.Time.now() - self.mission_started).to_sec()
        if self.mission_has_control:
            if not self._set_lane_controller(True):
                self._fail("could not return control to lane controller")
                return
        self.lane_speed_limit_pub.publish(
            Float64(data=self.lane_resume_max_velocity)
        )
        self._set_state(self.COMPLETE)
        rospy.loginfo(
            "Intersection fully complete: elapsed=%.3fs entry_elapsed=%.3fs "
            "exit_path_elapsed=%.3fs exit_progress=%.1f%% "
            "entry_error=%.3fm/%.1fdeg exit_error=%.3fm/%.1fdeg "
            "max_cmd_angular=%.3frad/s; lane following owns cmd_vel",
            mission_elapsed,
            self.entry_elapsed,
            self.exit_path_elapsed,
            progress,
            self.entry_goal_error,
            self.entry_yaw_error,
            self.exit_path_goal_error,
            self.exit_path_yaw_error,
            self.path_max_commanded_angular,
        )

    def shutdown(self):
        # A completed intersection no longer owns cmd_vel. Publishing a late
        # zero during shutdown could otherwise interrupt the obstacle mission.
        with self.lock:
            self.shutting_down = True
            if self.mission_has_control:
                self.cmd_pub.publish(Twist())


if __name__ == "__main__":
    rospy.init_node("intersection_mission_controller")
    IntersectionMissionController()
    rospy.spin()
