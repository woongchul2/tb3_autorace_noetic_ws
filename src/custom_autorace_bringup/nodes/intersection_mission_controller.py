#!/usr/bin/env python3
"""Select a camera-sign route and align it to source-stamped local evidence."""

import math
import os
import threading
from collections import deque

import cv2
import numpy as np
import rospkg
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Bool, Float64, Float64MultiArray, Header, String, UInt8
from std_srvs.srv import SetBool

from custom_autorace_bringup.msg import TrafficSign
from custom_autorace_bringup.local_registration import (
    TemporalRegistrationConfig,
    TemporalRegistrationFilter,
    register_point_with_heading,
    registration_covariance_with_floor,
    registration_radial_uncertainties,
)
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
        self.arm_topic = str(
            get(p + "arm_topic", "/mission/arm/intersection")
        )
        self.ready_topic = str(
            get(p + "ready_topic", "/mission/ready/intersection")
        )
        self.mission_name = str(get(p + "mission_name", "intersection"))
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
        self.ready_gate_timeout = max(
            1.0, float(get(p + "ready_gate_timeout", 5.0))
        )
        self.entry_takeover_pose_timeout = max(
            0.10, float(get(p + "entry_takeover_pose_timeout", 0.75))
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

        # The surveyed mission geometry keeps its map heading, while the
        # direction sign fixes translation relative to the live vehicle.  The
        # map translation is deliberately ignored, so changing the length of
        # the straight before this mission cannot move the local route.
        self.diagnostic_map_frame = str(
            get(p + "diagnostic_map_frame", "map")
        )
        self.local_frame = str(
            get(p + "registration/local_frame", "intersection_local")
        )
        self.sign_landmarks_local = {
            self.LEFT: self._point_parameter(
                get(p + "registration/left_sign_landmark_in_local", [0.315, 0.088]),
                "registration/left_sign_landmark_in_local",
            ),
            self.RIGHT: self._point_parameter(
                get(p + "registration/right_sign_landmark_in_local", [0.315, -0.022]),
                "registration/right_sign_landmark_in_local",
            ),
        }
        self.sign_physical_height = max(
            0.001, float(get(p + "registration/sign_physical_height", 0.120))
        )
        self.sign_range_scale = {
            self.LEFT: max(
                0.01,
                float(get(p + "registration/left_sign_range_scale", 1.235)),
            ),
            self.RIGHT: max(
                0.01,
                float(get(p + "registration/right_sign_range_scale", 1.175)),
            ),
        }
        self.sign_center_bias_pixels = {
            self.LEFT: float(
                get(p + "registration/left_sign_center_bias_pixels", 24.0)
            ),
            self.RIGHT: float(
                get(p + "registration/right_sign_center_bias_pixels", 13.0)
            ),
        }
        camera_offset = self._point_parameter(
            get(p + "registration/camera_offset_in_base", [0.0580, 0.0090]),
            "registration/camera_offset_in_base",
        )
        self.camera_offset_in_base = camera_offset
        self.camera_fx = max(
            1.0, float(get(p + "registration/camera_fx_fallback", 337.0))
        )
        self.camera_fy = max(
            1.0, float(get(p + "registration/camera_fy_fallback", 337.0))
        )
        self.camera_cx = float(
            get(p + "registration/camera_cx_fallback", 320.0)
        )
        self.local_yaw_in_map = math.radians(
            float(get(p + "registration/local_yaw_in_map_deg", 180.52))
        )
        self.registration_tf_timeout = max(
            0.0, float(get(p + "registration/map_tf_timeout", 0.020))
        )
        self.registration_roi_edge_margin = max(
            0,
            int(get(p + "registration/roi_edge_margin_pixels", 2)),
        )
        self.registration_entry_lead_min = float(
            get(p + "registration/entry_lead_min", -0.15)
        )
        self.registration_entry_lead_max = float(
            get(p + "registration/entry_lead_max", 0.03)
        )
        self.registration_entry_lateral_max = max(
            0.01, float(get(p + "registration/entry_lateral_max", 0.03))
        )
        self.registration_entry_heading_max = math.radians(
            max(1.0, float(get(p + "registration/entry_heading_max_deg", 12.0)))
        )
        if self.registration_entry_lead_min >= self.registration_entry_lead_max:
            raise rospy.ROSInitException(
                "registration entry lead limits are inconsistent"
            )
        self.registration_observation_position_sigma = max(
            0.001,
            float(get(p + "registration/observation_position_stddev", 0.005)),
        )
        self.registration_observation_heading_sigma = math.radians(
            max(
                0.1,
                float(get(p + "registration/observation_heading_stddev_deg", 0.12)),
            )
        )
        self.registration_systematic_position_sigma = max(
            0.0,
            float(get(p + "registration/systematic_position_error", 0.0015)),
        )
        self.registration_systematic_heading_sigma = math.radians(
            max(
                0.0,
                float(get(p + "registration/systematic_heading_error_deg", 0.16)),
            )
        )
        registration_required_confirmations = max(
            1, int(get(p + "registration/required_confirmations", 3))
        )
        self.registration_temporal_config = TemporalRegistrationConfig(
            required_confirmations=registration_required_confirmations,
            maximum_gap=max(
                0.05, float(get(p + "registration/maximum_confirmation_gap", 0.25))
            ),
            maximum_position_delta=max(
                0.005, float(get(p + "registration/maximum_position_delta", 0.040))
            ),
            maximum_heading_delta=math.radians(
                max(
                    0.5,
                    float(get(p + "registration/maximum_heading_delta_deg", 4.0)),
                )
            ),
            history_size=max(
                registration_required_confirmations,
                int(get(p + "registration/history_size", 12)),
            ),
        )
        self.odom_history_duration = max(
            0.25, float(get(p + "registration/odom_history_duration", 2.0))
        )
        self.registration_pose_stamp_tolerance = max(
            0.0,
            float(get(p + "registration/pose_stamp_tolerance", 0.060)),
        )
        self.registration_source_max_age = max(
            0.05, float(get(p + "registration/source_max_age", 0.45))
        )
        self.registration_future_tolerance = max(
            0.0, float(get(p + "registration/future_tolerance", 0.05))
        )
        self.course_world_size = float(get(p + "course_world_size", 4.0))
        self.course_resolution = float(get(p + "course_resolution", 0.01))
        self.course_boundary_inflation = max(
            0.0, float(get(p + "course_boundary_inflation", 0.03))
        )
        self.exit_local_snap_max_distance = max(
            0.01, float(get(p + "exit_local_snap_max_distance", 0.12))
        )
        self.course_texture_package = str(
            get(p + "course_texture_package", "turtlebot3_gazebo")
        )
        self.course_texture_relative_path = str(
            get(
                p + "course_texture_relative_path",
                "models/turtlebot3_autorace_2020/course/materials/textures/course.png",
            )
        )
        texture_to_local = get(
            p + "course_texture_to_local", [1.395, -0.750, 180.0]
        )
        if not isinstance(texture_to_local, (list, tuple)) or len(texture_to_local) != 3:
            raise rospy.ROSInitException(
                "course_texture_to_local must be [x, y, yaw_deg]"
            )
        self.local_from_texture = RigidTransform2D(
            float(texture_to_local[0]),
            float(texture_to_local[1]),
            math.radians(float(texture_to_local[2])),
            source_frame="course_texture",
            target_frame=self.local_frame,
        )
        self.local_entry_start = self._point_parameter(
            get(p + "local_entry_start", [0.0, 0.0]),
            "local_entry_start",
        )
        self.local_entry_start_yaw = math.radians(
            float(get(p + "local_entry_start_yaw_deg", 0.0))
        )
        self.local_left_entry_goal = self._point_parameter(
            get(p + "local_left_entry_goal", [0.1428, 0.1757]),
            "local_left_entry_goal",
        )
        self.local_right_entry_goal = self._point_parameter(
            get(p + "local_right_entry_goal", [0.1437, -0.1614]),
            "local_right_entry_goal",
        )
        self.local_left_arc_entry_yaw = math.radians(
            float(get(p + "local_left_arc_entry_yaw_deg", 90.0))
        )
        self.local_right_arc_entry_yaw = math.radians(
            float(get(p + "local_right_arc_entry_yaw_deg", -90.0))
        )
        self.entry_samples = max(
            20, int(get(p + "entry_samples", 100))
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

        self.local_left_exit_control_points = tuple(
            self._point_parameter(value, "local_left_exit_control_points")
            for value in get(
                p + "local_left_exit_control_points",
                [
                    [0.632800, 0.285300],
                    [0.657883, 0.174915],
                    [0.652018, 0.191341],
                    [0.673157, 0.097479],
                    [0.637544, 0.000000],
                    [0.795000, 0.000000],
                ],
            )
        )
        self.local_right_exit_control_points = tuple(
            self._point_parameter(value, "local_right_exit_control_points")
            for value in get(
                p + "local_right_exit_control_points",
                [
                    [0.632800, -0.285300],
                    [0.657883, -0.174915],
                    [0.652018, -0.191341],
                    [0.673157, -0.097479],
                    [0.637544, 0.000000],
                    [0.795000, 0.000000],
                ],
            )
        )
        if (
            len(self.local_left_exit_control_points) < 2
            or len(self.local_right_exit_control_points) < 2
        ):
            raise rospy.ROSInitException(
                "each selected exit branch requires at least two control points"
            )
        self.exit_branch_samples = max(
            20, int(get(p + "exit_branch_samples", 100))
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

        self.local_exit_control_points = tuple(
            self._point_parameter(value, "local_exit_control_points")
            for value in get(
                p + "local_exit_control_points",
                [
                    [0.795000, 0.000000],
                    [1.053964, 0.000000],
                    [1.099487, -0.071934],
                    [1.138339, -0.240468],
                    [1.145000, -0.134951],
                    [1.145000, -0.450000],
                ],
            )
        )
        if len(self.local_exit_control_points) < 2:
            raise rospy.ROSInitException(
                "local_exit_control_points requires at least two points"
            )
        self.exit_samples = max(
            20, int(get(p + "exit_samples", 180))
        )
        self.local_exit_goal_yaw = math.radians(
            float(get(p + "local_exit_goal_yaw_deg", -90.0))
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
        self.odom_history = deque()
        self.total_distance = 0.0
        self.direction_candidate = self.direction = self.NONE
        self.direction_count = 0
        self.last_direction_confirmation_time = None
        self.pending_direction_confirmation = None
        self.camera_width = max(1, int(get(p + "camera_width_fallback", 640)))
        self.camera_height = max(1, int(get(p + "camera_height_fallback", 480)))
        self.registration_filter = TemporalRegistrationFilter(
            self.registration_temporal_config
        )
        self.registration_covariance = tuple()
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
        self.local_to_odom = None
        self.registration_source_stamp = None
        self.active_tracking_from_local = None
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
        self.course_occupied = None
        self.course_raw_occupied = None
        self.course_boundary_points = np.empty((0, 2), dtype=np.float64)
        self.course_boundary_cell_size = None
        self.course_boundary_local = None
        self.course_bounds_local = None
        self.zone_gate_open = False
        self.arm_seq = None
        self.arm_stamp = None
        self.ready_published_seq = None
        self.last_ready_stamp = None
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
        self.ready_pub = rospy.Publisher(
            self.ready_topic, Header, queue_size=1, latch=True
        )
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
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
            self.arm_topic, Header, self.arm_callback, queue_size=1
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
            "Intersection route waiting for arm=%s; gate=%s opens only after "
            "source-stamped local registration readiness",
            self.arm_topic,
            self.zone_gate_topic,
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
            stamp = message.header.stamp
            if stamp == rospy.Time():
                stamp = rospy.Time.now()
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
            frame = message.header.frame_id or "odom"
            if self.odom_history and frame != self.odom_history[-1][4]:
                self.odom_history.clear()
            if not self.odom_history or stamp > self.odom_history[-1][0]:
                self.odom_history.append((stamp, x, y, self.yaw, frame))
                while (
                    len(self.odom_history) > 1
                    and (stamp - self.odom_history[0][0]).to_sec()
                    > self.odom_history_duration
                ):
                    self.odom_history.popleft()

    def zone_gate_callback(self, message):
        with self.lock:
            was_open = self.zone_gate_open
            requested = bool(message.data)
            prepared = (
                self.arm_seq is not None
                and self.ready_published_seq == self.arm_seq
                and self.last_ready_stamp is not None
                and -self.registration_future_tolerance
                <= (rospy.Time.now() - self.last_ready_stamp).to_sec()
                <= self.registration_source_max_age
                and self.local_to_odom is not None
                and self.path is not None
                and self.active_path_stage == "entry"
            )
            self.zone_gate_open = requested and prepared
            if requested and not prepared:
                rospy.logwarn_throttle(
                    1.0,
                    "Ignoring intersection enable before matching local "
                    "registration readiness",
                )
        if self.zone_gate_open and not was_open:
            rospy.loginfo(
                "Ordered intersection mission gate enabled"
            )

    def _reset_registration(self):
        self.direction_candidate = self.NONE
        self.direction = self.NONE
        self.direction_count = 0
        self.last_direction_confirmation_time = None
        self.pending_direction_confirmation = None
        if hasattr(self, "registration_filter"):
            self.registration_filter.reset("new intersection arm")
        self.local_to_odom = None
        self.registration_source_stamp = None
        self.registration_covariance = tuple()
        self.active_tracking_from_local = None
        self.ready_published_seq = None
        self.last_ready_stamp = None
        self.path = None
        self.path_follower = None
        self.active_path_stage = ""
        self._exit_branch_path_cache = {}

    def arm_callback(self, message):
        """Arm exactly one ordered generation while lane control continues."""
        with self.lock:
            sequence = int(message.seq)
            if sequence == 0 or sequence == self.arm_seq:
                return
            if message.frame_id != self.mission_name:
                rospy.logwarn(
                    "Ignoring intersection arm frame '%s' (expected '%s')",
                    message.frame_id,
                    self.mission_name,
                )
                return
            if self.mission_has_control or self.state not in (
                self.WAIT_INTERSECTION,
                self.SEARCH_DIRECTION,
                self.WAIT_ENTRY_HANDOFF,
                self.COMPLETE,
                self.FAILED,
            ):
                rospy.logwarn(
                    "Ignoring intersection arm generation %d while %s",
                    sequence,
                    self.state,
                )
                return
            stamp = message.stamp
            if stamp == rospy.Time():
                rospy.logwarn(
                    "Ignoring intersection arm generation %d with zero stamp",
                    sequence,
                )
                return
            self.arm_seq = sequence
            self.arm_stamp = stamp
            self.zone_gate_open = False
            self._reset_registration()
            self._set_state(self.SEARCH_DIRECTION)
            rospy.loginfo(
                "Intersection registration armed: generation=%d stamp=%.3f; "
                "lane controller remains active",
                sequence,
                stamp.to_sec(),
            )

    def _synchronized_odom_pose(self, stamp):
        """Interpolate the encoder/IMU odometry pose at a camera source stamp."""
        if stamp is None or stamp == rospy.Time() or not self.odom_history:
            return None
        samples = tuple(self.odom_history)
        if stamp <= samples[0][0]:
            skew = abs((samples[0][0] - stamp).to_sec())
            if skew <= self.registration_pose_stamp_tolerance:
                return Pose2D(*samples[0][1:4])
            return None
        if stamp >= samples[-1][0]:
            skew = abs((stamp - samples[-1][0]).to_sec())
            if skew <= self.registration_pose_stamp_tolerance:
                return Pose2D(*samples[-1][1:4])
            return None
        for earlier, later in zip(samples[:-1], samples[1:]):
            if earlier[0] <= stamp <= later[0]:
                duration = (later[0] - earlier[0]).to_sec()
                if duration <= 0.0:
                    return None
                ratio = (stamp - earlier[0]).to_sec() / duration
                return Pose2D(
                    earlier[1] + ratio * (later[1] - earlier[1]),
                    earlier[2] + ratio * (later[2] - earlier[2]),
                    earlier[3]
                    + ratio * normalize_angle(later[3] - earlier[3]),
                )
        return None

    def _direction_source_stamp_eligible(self, stamp):
        if (
            self.arm_seq is None
            or self.arm_stamp is None
            or stamp is None
            or stamp == rospy.Time()
            or stamp < self.arm_stamp
        ):
            return False
        age = (rospy.Time.now() - stamp).to_sec()
        return (
            age <= self.registration_source_max_age
            and age >= -self.registration_future_tolerance
        )

    def camera_info_callback(self, message):
        if message.width <= 0 or message.height <= 0:
            return
        with self.lock:
            self.camera_width = int(message.width)
            self.camera_height = int(message.height)
            if len(message.K) >= 3:
                fx = float(message.K[0])
                cx = float(message.K[2])
                fy = float(message.K[4]) if len(message.K) >= 5 else math.nan
                if math.isfinite(fx) and fx > 1.0:
                    self.camera_fx = fx
                if math.isfinite(fy) and fy > 1.0:
                    self.camera_fy = fy
                if math.isfinite(cx):
                    self.camera_cx = cx

    def _map_aligned_local_yaw(self, source_stamp):
        """Return intersection-local heading in odom at the camera stamp."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.diagnostic_map_frame,
                source_stamp,
                rospy.Duration(self.registration_tf_timeout),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as error:
            rospy.logwarn_throttle(
                1.0,
                "Waiting for source-stamped intersection map heading: %s",
                error,
            )
            return None
        quaternion = transform.transform.rotation
        map_to_odom_yaw = math.atan2(
            2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2),
        )
        if not math.isfinite(map_to_odom_yaw):
            return None
        return normalize_angle(map_to_odom_yaw + self.local_yaw_in_map)

    def _registration_roi_unclipped(self, roi):
        margin = self.registration_roi_edge_margin
        return (
            int(roi.x_offset) >= margin
            and int(roi.y_offset) >= margin
            and int(roi.x_offset) + int(roi.width)
            <= self.camera_width - margin
            and int(roi.y_offset) + int(roi.height)
            <= self.camera_height - margin
        )

    def _direction_registration_result(
        self, selected, source_stamp, center_ratio, roi
    ):
        odom_pose = self._synchronized_odom_pose(source_stamp)
        if odom_pose is None or not self._registration_roi_unclipped(roi):
            return None, odom_pose
        local_yaw = self._map_aligned_local_yaw(source_stamp)
        if local_yaw is None:
            return None, odom_pose
        if (
            not math.isfinite(self.camera_fx)
            or self.camera_fx <= 1.0
            or not math.isfinite(self.camera_fy)
            or self.camera_fy <= 1.0
        ):
            return None, odom_pose
        apparent_height = float(roi.height)
        if apparent_height <= 0.0:
            return None, odom_pose

        # The upright sign's vertical diameter remains metric under an oblique
        # horizontal view.  Width does not, and can also be truncated at the
        # image edge.  Horizontal bearing still comes from the ROI centre.
        forward = (
            self.sign_range_scale[selected]
            * self.camera_fy
            * self.sign_physical_height
            / apparent_height
        )
        centre_pixels = (
            float(center_ratio) * float(self.camera_width)
            + self.sign_center_bias_pixels[selected]
        )
        lateral = -(centre_pixels - self.camera_cx) * forward / self.camera_fx
        base_forward = self.camera_offset_in_base[0] + forward
        base_lateral = self.camera_offset_in_base[1] + lateral
        cosine = math.cos(odom_pose.yaw)
        sine = math.sin(odom_pose.yaw)
        observed_sign = (
            odom_pose.x + cosine * base_forward - sine * base_lateral,
            odom_pose.y + sine * base_forward + cosine * base_lateral,
        )
        sign_local = self.sign_landmarks_local[selected]
        result = register_point_with_heading(
            source_stamp.to_sec(),
            self.local_frame,
            self.odom_frame,
            sign_local,
            observed_sign,
            local_yaw,
            self.registration_observation_position_sigma,
            self.registration_observation_heading_sigma,
        )
        return result, odom_pose

    def _entry_pose_eligible(self, transform, odom_pose, log_rejection=True):
        local_pose = transform.inverse().apply_pose(odom_pose)
        delta_x = local_pose.x - self.local_entry_start[0]
        delta_y = local_pose.y - self.local_entry_start[1]
        cosine = math.cos(self.local_entry_start_yaw)
        sine = math.sin(self.local_entry_start_yaw)
        entry_longitudinal = cosine * delta_x + sine * delta_y
        entry_lateral = -sine * delta_x + cosine * delta_y
        entry_heading = normalize_angle(
            local_pose.yaw - self.local_entry_start_yaw
        )
        eligible = (
            self.registration_entry_lead_min
            <= entry_longitudinal
            <= self.registration_entry_lead_max
            and abs(entry_lateral) <= self.registration_entry_lateral_max
            and abs(entry_heading)
            <= self.registration_entry_heading_max
        )
        if not eligible and log_rejection:
            rospy.logwarn_throttle(
                1.0,
                "Intersection aligned; waiting for local entry envelope: "
                "x=%.3f y=%.3f yaw=%.1fdeg",
                entry_longitudinal,
                entry_lateral,
                math.degrees(entry_heading),
            )
        return eligible

    def _registration_uncertainties(self, tracking_points=None):
        local_points = None
        transform = getattr(self, "active_tracking_from_local", None)
        if tracking_points is not None and transform is not None:
            inverse = transform.inverse()
            local_points = [inverse.apply_point(point) for point in tracking_points]
        return registration_radial_uncertainties(
            getattr(self, "registration_covariance", tuple()),
            getattr(self, "footprint", None),
            local_points=local_points,
            target_from_source_yaw=(
                0.0 if transform is None else transform.target_from_source_yaw
            ),
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
        return (
            detected,
            center_ratio,
            area_ratio,
            message.confidence,
            roi,
        )

    def sign_callback(self, message):
        with self.lock:
            if self.arm_seq is None:
                return
            if self.state == self.WAIT_ENTRY_HANDOFF:
                self._refresh_prepared_readiness(message)
                return
            if self.state != self.SEARCH_DIRECTION:
                return
            source_stamp = message.header.stamp
            if not self._direction_source_stamp_eligible(source_stamp):
                return
            observation = self._direction_observation(message)
            if observation is None:
                return
            detected, center_ratio, area_ratio, confidence, roi = observation
            if self.direction in (self.LEFT, self.RIGHT):
                return
            if (
                area_ratio < self.direction_confirm_min_roi_area_ratio
                or confidence < self.direction_confirm_min_confidence
            ):
                self.direction_candidate = self.NONE
                self.direction_count = 0
                self.last_direction_confirmation_time = None
                self.pending_direction_confirmation = None
                self.registration_filter.reset("unusable direction sign")
                return

            selected = (
                self.forced_direction
                if self.forced_direction in (self.LEFT, self.RIGHT)
                else detected
            )
            separated = (
                self.last_direction_confirmation_time is None
                or (source_stamp - self.last_direction_confirmation_time).to_sec()
                > self.direction_confirmation_max_gap
                or source_stamp <= self.last_direction_confirmation_time
            )
            if separated or selected != self.direction_candidate:
                self.direction_candidate, self.direction_count = selected, 1
                self.pending_direction_confirmation = None
                self.registration_filter.reset("new direction-sign streak")
                rospy.loginfo(
                    "Direction sign usable: confidence=%.2f area=%.1f%% "
                    "center=%.1f%%",
                    confidence,
                    100.0 * area_ratio,
                    100.0 * center_ratio,
                )
            else:
                self.direction_count += 1
            self.last_direction_confirmation_time = source_stamp
            registration, odom_pose = self._direction_registration_result(
                detected,
                source_stamp,
                center_ratio,
                roi,
            )
            if registration is None:
                # Selection and alignment are separate stages. A clipped ROI,
                # unavailable source-stamped TF, or late odometry may skip one
                # metric update without erasing a valid LEFT/RIGHT streak.
                return
            temporal = self.registration_filter.update(registration)
            if not registration.accepted:
                return
            if (
                self.direction_count >= self.direction_confirm_frames
                and temporal.confirmed
                and odom_pose is not None
                and self._entry_pose_eligible(temporal.transform, odom_pose)
            ):
                self.pending_direction_confirmation = (
                    selected,
                    source_stamp,
                    center_ratio,
                    temporal.transform,
                    temporal.covariance,
                )
                self._try_finalize_direction_registration()

    def _publish_ready(self, source_stamp):
        ready = Header()
        ready.seq = int(self.arm_seq)
        ready.stamp = source_stamp
        ready.frame_id = self.mission_name
        self.ready_pub.publish(ready)
        self.ready_published_seq = self.arm_seq
        self.last_ready_stamp = source_stamp

    def _refresh_prepared_readiness(self, message):
        """Refresh a validated route with new source evidence, without replanning."""
        if (
            self.zone_gate_open
            or self.local_to_odom is None
            or self.path is None
            or self.active_path_stage != "entry"
            or self.ready_published_seq != self.arm_seq
        ):
            return False
        source_stamp = message.header.stamp
        if (
            not self._direction_source_stamp_eligible(source_stamp)
            or (
                self.last_ready_stamp is not None
                and source_stamp <= self.last_ready_stamp
            )
        ):
            return False
        observation = self._direction_observation(message)
        if observation is None:
            return False
        detected = observation[0]
        selected = (
            self.forced_direction
            if self.forced_direction in (self.LEFT, self.RIGHT)
            else detected
        )
        if selected != self.direction:
            return False
        registration, odom_pose = self._direction_registration_result(
            detected,
            source_stamp,
            observation[1],
            observation[4],
        )
        if (
            registration is None
            or not registration.accepted
            or registration.transform is None
            or odom_pose is None
        ):
            return False
        position_delta = math.hypot(
            registration.transform.target_from_source_x
            - self.local_to_odom.target_from_source_x,
            registration.transform.target_from_source_y
            - self.local_to_odom.target_from_source_y,
        )
        heading_delta = abs(
            normalize_angle(
                registration.transform.target_from_source_yaw
                - self.local_to_odom.target_from_source_yaw
            )
        )
        temporal_config = self.registration_filter.config
        if (
            position_delta > temporal_config.maximum_position_delta
            or heading_delta > temporal_config.maximum_heading_delta
        ):
            return False
        self._publish_ready(source_stamp)
        return True

    def _try_finalize_direction_registration(self):
        if self.pending_direction_confirmation is None:
            return False
        (
            selected,
            source_stamp,
            center_ratio,
            local_to_odom,
            covariance,
        ) = self.pending_direction_confirmation
        if not self._direction_source_stamp_eligible(source_stamp):
            self.direction_candidate = self.NONE
            self.direction_count = 0
            self.last_direction_confirmation_time = None
            self.pending_direction_confirmation = None
            rospy.logwarn(
                "Discarded stale intersection direction confirmation before "
                "source-synchronized odometry became available"
            )
            return False
        odom_pose = self._synchronized_odom_pose(source_stamp)
        if odom_pose is None:
            return False

        self.direction = selected
        self.local_to_odom = local_to_odom
        self.registration_covariance = registration_covariance_with_floor(
            covariance,
            self.registration_systematic_position_sigma,
            self.registration_systematic_heading_sigma,
        )
        self.registration_source_stamp = source_stamp
        self.active_tracking_from_local = self.local_to_odom
        try:
            generated = self._generate_entry_path(odom_pose)
        except (ArithmeticError, TypeError, ValueError) as error:
            rospy.logerr("Local arc-entry path generation raised: %s", error)
            generated = False
        if not generated:
            rospy.logwarn(
                "Rejected source-stamped local arc-entry path before control "
                "handoff; keeping selector/alignment evidence while camera "
                "lane control advances into the entry envelope"
            )
            self.direction = self.NONE
            self.local_to_odom = None
            self.registration_covariance = tuple()
            self.registration_source_stamp = None
            self.active_tracking_from_local = None
            self.ready_published_seq = None
            self.last_ready_stamp = None
            self.path = None
            self.path_follower = None
            self.active_path_stage = ""
            self.pending_direction_confirmation = None
            self._exit_branch_path_cache = {}
            self._set_state(self.SEARCH_DIRECTION)
            return False

        self.direction_pub.publish(UInt8(data=self.direction))
        self._publish_ready(source_stamp)
        self.pending_direction_confirmation = None
        self._set_state(self.WAIT_ENTRY_HANDOFF)
        rospy.loginfo(
            "Direction confirmed %d/%d and intersection-local route validated: "
            "generation=%d source=%.3f direction=%s image_center=%.1f%%; "
            "lane control remains active until enable",
            self.direction_count,
            self.direction_confirm_frames,
            self.arm_seq,
            source_stamp.to_sec(),
            "LEFT" if self.direction == self.LEFT else "RIGHT",
            100.0 * center_ratio,
        )
        return True

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
                # Only a non-zero Header arm generation may start camera
                # evidence collection. A Bool enable never arms this state.
                pass

            elif self.state == self.SEARCH_DIRECTION:
                self._try_finalize_direction_registration()
                if self._state_age() > self.direction_search_timeout:
                    rospy.logwarn(
                        "Intersection direction/registration window expired; "
                        "restarting evidence while lane control continues"
                    )
                    self._reset_registration()
                    self._set_state(self.SEARCH_DIRECTION)

            elif self.state == self.WAIT_ENTRY_HANDOFF:
                if self.zone_gate_open:
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
                        "Intersection acquired cmd_vel after matching ready "
                        "generation; waiting for a post-handoff EKF pose"
                    )
                    self._set_state(self.PREPARE_ENTRY_PATH)
                    return
                elif self._state_age() > self.ready_gate_timeout:
                    rospy.logwarn(
                        "Intersection enable did not match fresh readiness; "
                        "reacquiring source-stamped geometry while lane control continues"
                    )
                    self._reset_registration()
                    self._set_state(self.SEARCH_DIRECTION)

            elif self.state == self.PREPARE_ENTRY_PATH:
                # A short zero barrier establishes one command owner and lets
                # odometry expose all motion that occurred inside the service
                # call before the aligned entry path is frozen.
                self.cmd_pub.publish(Twist())
                if self.odom_sequence > self.entry_takeover_odom_sequence:
                    if not self._start_prepared_entry_path():
                        self._fail("prepared arc-entry path could not start")
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
                exit_projection = self._selected_exit_local_projection()
                exit_path_distance = exit_projection.distance
                # Project odometry onto the whole branch transformed by the
                # same source-stamped local registration as the entry path.
                # Passing the first stored point therefore cannot miss the
                # camera-to-common-follower ownership handoff.
                arc_end_ready = bool(
                    exit_path_distance <= self.exit_takeover_max_distance
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
                        "Local route confirmed the %s semicircle end; intersection "
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
                        "distance=%.3fm exit_path_distance=%.3fm "
                        "station=%.3fm (limit %.3fm)"
                        % (
                            self._state_age(),
                            arc_distance,
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
                        "no usable post-handoff EKF pose for the shared exit path: "
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

    @staticmethod
    def _local_to_tracking_point(point, tracking_from_local):
        if not isinstance(tracking_from_local, RigidTransform2D):
            tracking_from_local = RigidTransform2D(
                *tracking_from_local,
                source_frame="intersection_local",
                target_frame="odom",
            )
        return tracking_from_local.apply_point(point)

    @staticmethod
    def _local_to_tracking_yaw(local_yaw, tracking_from_local):
        if not isinstance(tracking_from_local, RigidTransform2D):
            tracking_from_local = RigidTransform2D(
                *tracking_from_local,
                source_frame="intersection_local",
                target_frame="odom",
            )
        return tracking_from_local.apply_pose((0.0, 0.0, local_yaw)).yaw

    def _start_prepared_entry_path(self):
        """Reset the prevalidated entry follower at the post-handoff pose."""
        if (
            self.path is None
            or self.path_follower is None
            or self.active_path_stage != "entry"
            or self.ready_published_seq != self.arm_seq
        ):
            return False
        pose = Pose2D(*self._tracking_pose())
        self.path_follower.reset(
            self.path,
            pose,
            initial_linear=0.0,
            initial_angular=0.0,
        )
        self.path_index = self.path_follower.path_index
        self.path_started = rospy.Time.now()
        self.last_path_command_time = None
        self.path_tick_pose = None
        self.path_tick_tracking = None
        self._publish_path_diagnostics()
        return True

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
    def _point_parameter(value, name):
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

    def _activate_path(
        self,
        path,
        goal_yaw,
        exit_yaw,
        stage,
        validation_start_pose=None,
        begin_tracking=True,
    ):
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
        if self.active_tracking_from_local is None:
            rospy.logerr(
                "Intersection %s path has no frozen local->odom registration",
                stage,
            )
            return False
        if self.course_bounds_local is None:
            rospy.logerr(
                "Intersection %s path has no mission-local course bounds",
                stage,
            )
            return False
        map_boundary = self.course_bounds_local.transformed(
            self.active_tracking_from_local
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
        if self.course_boundary_local is None:
            rospy.logerr(
                "Intersection %s path has no mission-local paint boundary",
                stage,
            )
            return False
        boundary = self.course_boundary_local.transformed(
            self.active_tracking_from_local
        )
        registration_uncertainty = self._registration_uncertainties(points)
        safety = PathSafety(
            line_boundaries=(boundary,),
            map_boundaries=(map_boundary,),
            margins=SafetyMargins(
                line=self.path_safety_margins.line,
                obstacle=self.path_safety_margins.obstacle,
                localization=self.path_safety_margins.localization,
                tracking=self.path_safety_margins.tracking,
            ),
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
            localization_uncertainty=registration_uncertainty,
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
        validation_pose = (
            Pose2D(*self._tracking_pose())
            if validation_start_pose is None
            else Pose2D.from_value(validation_start_pose)
        )
        start_validation = self.path_validator.validate_poses(
            [validation_pose],
            safety,
            line_overlap_allowances=[
                float(executable.line_overlap_allowance[0])
            ],
            localization_uncertainties=[
                float(executable.localization_uncertainty[0])
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
        tracking_pose = validation_pose
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
        self.path_started = now if begin_tracking else None
        self.path_max_commanded_angular = 0.0
        self.last_path_command_time = None
        self.path_tick_pose = None
        self.path_tick_tracking = None
        self._publish_path()
        self._publish_path_diagnostics()
        return True

    def _generate_entry_path(self, synchronized_start_pose):
        """Build and fully validate the selected entrance from one local pose."""
        if self.direction not in (self.LEFT, self.RIGHT):
            rospy.logerr("Intersection entry has no valid selected direction")
            return False
        if self.local_to_odom is None:
            rospy.logerr("Intersection entry has no local->odom registration")
            return False

        if self.direction == self.LEFT:
            goal = self.local_left_entry_goal
            goal_local_yaw = self.local_left_arc_entry_yaw
        else:
            goal = self.local_right_entry_goal
            goal_local_yaw = self.local_right_arc_entry_yaw
        self._prepare_active_entry_parameters()
        tracking_start = Pose2D.from_value(synchronized_start_pose)
        tracking_goal = self._local_to_tracking_point(goal, self.local_to_odom)
        goal_yaw = self._local_to_tracking_yaw(
            goal_local_yaw, self.local_to_odom
        )
        route_chord = math.hypot(
            tracking_goal[0] - tracking_start.x,
            tracking_goal[1] - tracking_start.y,
        )
        start_tangent = max(
            0.005,
            self.entry_start_tangent_ratio * route_chord,
        )
        end_tangent = max(
            0.005,
            self.entry_end_tangent_ratio * route_chord,
        )
        route = self._cubic_path(
            (tracking_start.x, tracking_start.y),
            tracking_start.yaw,
            tracking_goal,
            goal_yaw,
            start_tangent,
            end_tangent,
            self.entry_samples,
        )
        self.entry_elapsed = math.nan
        self.entry_goal_error = math.nan
        self.entry_yaw_error = math.nan
        self.active_tracking_from_local = self.local_to_odom
        if not self._activate_path(
            route,
            goal_yaw,
            goal_yaw,
            "entry",
            validation_start_pose=tracking_start,
            begin_tracking=False,
        ):
            return False
        direction = "LEFT" if self.direction == self.LEFT else "RIGHT"
        rospy.loginfo(
            "Selected source-stamped local %s arc-entry path; "
            "start=(%.3f, %.3f, %.1fdeg), "
            "endpoint=(%.3f, %.3f, %.1fdeg), "
            "tangents=(%.3f, %.3f)m, "
            "length=%.3fm speed=%.3fm/s",
            direction,
            route[0][0],
            route[0][1],
            math.degrees(tracking_start.yaw),
            route[-1][0],
            route[-1][1],
            math.degrees(goal_yaw),
            start_tangent,
            end_tangent,
            self._polyline_length(route),
            self.active_path_velocity,
        )
        return True

    def _generate_exit_path(self):
        """Build the exit with the same frozen local transform as the entry."""
        self._prepare_active_exit_parameters()
        if self.local_to_odom is None:
            rospy.logerr("Shared exit has no frozen local->odom registration")
            return False
        branch_control_points = self._selected_exit_control_points()
        branch_route = self._bezier_path(
            branch_control_points, self.exit_branch_samples
        )
        shared_route = self._bezier_path(
            self.local_exit_control_points, self.exit_samples
        )
        local_route = branch_route[:-1] + shared_route
        local_common_path = path_from_xy(
            local_route,
            self.local_frame,
            target_speed=self.active_path_velocity,
        )
        tracking_pose = Pose2D(*self._tracking_pose())
        local_pose = self.local_to_odom.inverse().apply_pose(tracking_pose)
        local_projection = project_to_path(
            local_common_path,
            local_pose.x,
            local_pose.y,
        )
        snap_distance = local_projection.distance
        if snap_distance > self.exit_local_snap_max_distance:
            rospy.logerr(
                "Odometry is %.3f m from the registered local exit path "
                "(limit %.3f m)",
                snap_distance,
                self.exit_local_snap_max_distance,
            )
            return False
        fixed_branch_path = path_from_xy(
            branch_route,
            self.local_frame,
            target_speed=self.active_path_velocity,
        )
        join_index = int(
            round(
                self.exit_adaptive_join_ratio
                * float(len(branch_route) - 1)
            )
        )
        join_index = max(1, min(len(branch_route) - 2, join_index))
        tracking_start = tracking_pose
        tracking_join = self._local_to_tracking_point(
            branch_route[join_index], self.local_to_odom
        )
        tracking_join_yaw = self._local_to_tracking_yaw(
            float(fixed_branch_path.heading[join_index]), self.local_to_odom
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
            self._local_to_tracking_point(point, self.local_to_odom)
            for point in branch_route[join_index:]
        ]
        downstream_shared = [
            self._local_to_tracking_point(point, self.local_to_odom)
            for point in shared_route
        ]
        route = connector[:-1] + downstream_branch[:-1] + downstream_shared
        goal_yaw = self._local_to_tracking_yaw(
            self.local_exit_goal_yaw, self.local_to_odom
        )
        rospy.loginfo(
            "Shared exit uses frozen local registration: local_pose="
            "(%.3f, %.3f, %.1fdeg), "
            "path_delta=%.3fm start_station=%.3fm",
            local_pose.x,
            local_pose.y,
            math.degrees(local_pose.yaw),
            snap_distance,
            local_projection.station,
        )

        self.exit_path_elapsed = math.nan
        self.exit_path_goal_error = math.nan
        self.exit_path_yaw_error = math.nan
        self.active_tracking_from_local = self.local_to_odom
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
            self.local_exit_control_points[-1][0],
            self.local_exit_control_points[-1][1],
            math.degrees(self.local_exit_goal_yaw),
            self._polyline_length(route),
            self.active_path_velocity,
        )
        return True

    def _selected_exit_control_points(self):
        if self.direction == self.LEFT:
            return self.local_left_exit_control_points
        if self.direction == self.RIGHT:
            return self.local_right_exit_control_points
        raise ValueError("intersection exit branch requested before selection")

    def _selected_exit_local_projection(self):
        """Project odometry onto the frozen local branch in odom."""
        if self.local_to_odom is None:
            raise ValueError("intersection exit projection has no registration")
        cache = getattr(self, "_exit_branch_path_cache", {})
        cache_key = (self.direction, self.arm_seq)
        branch_path = cache.get(cache_key)
        if branch_path is None:
            local_branch_route = self._bezier_path(
                self._selected_exit_control_points(),
                self.exit_branch_samples,
            )
            branch_route = [
                self.local_to_odom.apply_point(point)
                for point in local_branch_route
            ]
            branch_path = path_from_xy(
                branch_route,
                self.odom_frame,
                target_speed=max(0.005, self.exit_path_velocity),
            )
            cache[cache_key] = branch_path
            self._exit_branch_path_cache = cache
        return project_to_path(
            branch_path,
            self.x,
            self.y,
        )

    @staticmethod
    def _polyline_length(points):
        return sum(
            math.hypot(b[0] - a[0], b[1] - a[1])
            for a, b in zip(points[:-1], points[1:])
        )

    def _inflate_course_boundaries(self, occupied, inflation):
        """Inflate paint by Euclidean distance instead of a square kernel."""
        radius = max(0.0, float(inflation)) / self.course_resolution
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
            package_path = rospkg.RosPack().get_path(self.course_texture_package)
            texture_path = os.path.join(
                package_path, self.course_texture_relative_path
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
        native_cell_width = self.course_world_size / float(texture_width)
        native_cell_height = self.course_world_size / float(texture_height)
        self.course_boundary_points = np.column_stack(
            (
                0.5 * self.course_world_size
                - (native_x.astype(np.float64) + 0.5) * native_cell_width,
                -0.5 * self.course_world_size
                + (native_y.astype(np.float64) + 0.5) * native_cell_height,
            )
        )
        self.course_boundary_cell_size = (
            native_cell_width,
            native_cell_height,
        )
        raw_boundary = RasterCellBoundary(
            self.course_boundary_points,
            cell_size=self.course_boundary_cell_size,
        )
        self.course_boundary_local = raw_boundary.transformed(
            self.local_from_texture
        )
        self.course_bounds_local = AxisAlignedBoundsBoundary(
            -0.5 * self.course_world_size,
            0.5 * self.course_world_size,
            -0.5 * self.course_world_size,
            0.5 * self.course_world_size,
        ).transformed(self.local_from_texture)

        width = max(1, int(round(self.course_world_size / self.course_resolution)))
        height = width
        # Gazebo rotates the square course model by pi. For each ROS grid cell
        # (world x/y increasing), sample the corresponding texture pixel.
        grid_x = -0.5 * self.course_world_size + (
            np.arange(width, dtype=np.float32) + 0.5
        ) * self.course_resolution
        grid_y = -0.5 * self.course_world_size + (
            np.arange(height, dtype=np.float32) + 0.5
        ) * self.course_resolution
        world_x, world_y = np.meshgrid(grid_x, grid_y)
        image_u = np.clip(
            ((0.5 * self.course_world_size - world_x) / self.course_world_size)
            * texture.shape[1],
            0,
            texture.shape[1] - 1,
        ).astype(np.int32)
        image_v = np.clip(
            ((0.5 * self.course_world_size + world_y) / self.course_world_size)
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
            raw_occupied, self.course_boundary_inflation
        )
        self.course_occupied = occupied

        message = OccupancyGrid()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.diagnostic_map_frame
        message.info.resolution = self.course_resolution
        message.info.width = width
        message.info.height = height
        message.info.origin.position.x = -0.5 * self.course_world_size
        message.info.origin.position.y = -0.5 * self.course_world_size
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
            self.path_follower.stopping_angular_velocities(
                tracking,
                tracking.direction * measured_speed,
                self.odom_angular_velocity,
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
