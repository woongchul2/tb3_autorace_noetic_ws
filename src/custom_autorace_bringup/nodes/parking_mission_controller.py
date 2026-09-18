#!/usr/bin/env python3
"""Register the surveyed parking fixture locally, then park in the empty bay."""

import math
import threading
from collections import OrderedDict, deque
from dataclasses import replace

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import (
    Bool,
    Float64,
    Float64MultiArray,
    Header,
    Int32MultiArray,
    String,
)
from std_srvs.srv import SetBool

from custom_autorace_bringup.parking_geometry import (
    LEFT,
    RIGHT,
    choose_clear_space,
    points_in_box,
    quintic_pose_path,
    quintic_turn_path,
    rectangle_corners,
    scan_points_in_map,
)
from custom_autorace_bringup.local_registration import (
    EntryPlane,
    MissionLocalTemplate,
    OrientedSegmentLandmark,
    OrientedSegmentObservation,
    RegistrationConfig,
    RegistrationObservation,
    TemporalRegistrationConfig,
    TemporalRegistrationFilter,
    entry_plane_progress,
    estimate_local_registration,
)
from custom_autorace_bringup.path_following import (
    AsymmetricFootprint,
    CallbackBoundary,
    GoalTolerance,
    PathDiagnostics,
    PathFollower,
    PathSafety,
    Pose2D,
    RigidTransform2D,
    SafetyDecision,
    SafetyMargins,
    SpeedProfile,
    SweptFootprintValidator,
    TrackingConfig,
    ValidationResult,
    calculate_tracking,
    clamp,
    combine_validation_results,
    normalize_angle,
    path_from_poses,
    project_to_path,
    footprint_points,
    in_place_rotation_command,
    sample_in_place_rotation,
    yaw_from_quaternion,
)


class ParkingMissionController:
    """Own ``cmd_vel`` only for the measured parking manoeuvre."""

    # Swept safety transforms each LaserScan with bracketed odometry.  Keep
    # enough 30 Hz samples to survive a short Python validation callback while
    # keeping the callback close to real time.  A depth of one loses the right
    # bracket; a deep backlog delays odometry behind the live safety scan.
    ODOM_SUBSCRIBER_QUEUE_SIZE = 8
    LANE_PATH_DIAGNOSTIC_FIELDS = 13

    WAIT_GATE = "WAIT_GATE"
    PREPARE_APPROACH = "PREPARE_APPROACH"
    APPROACH = "APPROACH"
    TURN_IN = "TURN_IN"
    ENTER_AISLE = "ENTER_AISLE"
    SELECT_SPACE = "SELECT_SPACE"
    TURN_TO_SPACE = "TURN_TO_SPACE"
    PARK_IN = "PARK_IN"
    BACK_OUT = "BACK_OUT"
    TURN_TO_EXIT = "TURN_TO_EXIT"
    LEAVE_AISLE = "LEAVE_AISLE"
    TURN_TO_ZIGZAG = "TURN_TO_ZIGZAG"
    VERIFY_ZIGZAG_LANE = "VERIFY_ZIGZAG_LANE"
    JOIN_ZIGZAG = "JOIN_ZIGZAG"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

    MOTION_STATES = {
        APPROACH,
        TURN_IN,
        ENTER_AISLE,
        PARK_IN,
        BACK_OUT,
        LEAVE_AISLE,
        TURN_TO_ZIGZAG,
    }
    ROTATION_STATES = {TURN_TO_SPACE, TURN_TO_EXIT}

    def __init__(self):
        get = rospy.get_param
        p = "~parking/"

        self.scan_topic = str(get(p + "topics/scan", "/scan_mid360_raw"))
        self.odom_topic = str(
            get(p + "topics/odometry", "/odometry/filtered")
        )
        self.map_pose_topic = str(
            get(p + "topics/map_pose", "/mission/map_pose")
        )
        self.gate_topic = str(
            get(p + "topics/zone_gate", "/mission/enable/parking")
        )
        self.arm_topic = str(
            get(p + "topics/arm", "/mission/arm/parking")
        )
        self.ready_topic = str(
            get(p + "topics/ready", "/mission/ready/parking")
        )
        self.mission_name = str(get(p + "mission_name", "parking")).strip()
        if not self.mission_name:
            raise rospy.ROSInitException("parking mission_name is empty")
        self.lane_path_topic = str(
            get(
                p + "topics/lane_path_diagnostics",
                "/control/lane_path_diagnostics",
            )
        )
        self.manual_stop_topic = str(
            get(p + "topics/manual_stop", "/control/manual_stop")
        )
        self.cmd_vel_topic = str(get(p + "topics/cmd_vel", "/cmd_vel"))
        self.diagnostics_topic = str(
            get(p + "topics/diagnostics", "/parking/diagnostics")
        )
        self.planned_left_path_topic = str(
            get(
                p + "topics/planned_path_left",
                "/parking/planned_path/left",
            )
        )
        self.planned_right_path_topic = str(
            get(
                p + "topics/planned_path_right",
                "/parking/planned_path/right",
            )
        )
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

        self.aisle_x = float(get(p + "route/aisle_x", 0.5000))
        self.entry_y = float(get(p + "route/entry_y", 1.7425))
        self.decision_y = float(get(p + "route/decision_y", 0.6920))
        self.zigzag_turn_start_x = float(
            get(p + "route/zigzag_turn_start_x", 0.4930)
        )
        self.zigzag_straight_y = float(
            get(p + "route/zigzag_straight_y", 1.7500)
        )
        self.turn_curve_offset = max(
            0.01, float(get(p + "route/turn_curve_offset", 0.1950))
        )
        self.turn_curve_tangent = max(
            0.001, float(get(p + "route/turn_curve_tangent", 0.0780))
        )
        self.zigzag_alignment_tail = max(
            0.01,
            float(get(p + "route/zigzag_alignment_tail", 0.0720)),
        )
        self.turn_curve_samples = max(
            11, int(get(p + "route/turn_curve_samples", 101))
        )
        self.zigzag_approach_tangent_ratio = clamp(
            float(
                get(p + "route/zigzag_approach_tangent_ratio", 0.10)
            ),
            0.01,
            0.24,
        )
        self.left_park_x = float(get(p + "route/left_park_x", 0.7741))
        self.right_park_x = float(get(p + "route/right_park_x", 0.2275))
        self.route_frame = str(
            get(p + "route/frame_id", "parking_local")
        ).strip().lstrip("/")
        if not self.route_frame:
            raise rospy.ROSInitException("parking route/frame_id is empty")
        if self.route_frame == "map":
            raise rospy.ROSInitException(
                "parking route/frame_id must be mission-local, not map"
            )
        self.approach_heading = math.radians(
            float(get(p + "route/approach_heading_deg", 180.0))
        )
        self.aisle_heading = math.radians(
            float(get(p + "route/aisle_heading_deg", -90.0))
        )
        self.outgoing_heading = math.radians(
            float(get(p + "route/outgoing_heading_deg", 90.0))
        )
        self.zigzag_heading = math.radians(
            float(get(p + "route/zigzag_heading_deg", 180.0))
        )

        self.fixed_obstacle_sample_spacing = max(
            0.001,
            float(get(p + "fixed_obstacles/sample_spacing", 0.004)),
        )
        self.fixed_obstacle_rectangles = self._rectangles_param(
            get(
                p + "fixed_obstacles/rectangles",
                [
                    [0.4400, 0.5600, 1.8875, 1.9125],
                    [0.7275, 0.7525, 1.8900, 2.0100],
                ],
            ),
            "fixed_obstacles/rectangles",
        )
        self.fixed_obstacle_points_route = self._sample_fixed_rectangles(
            self.fixed_obstacle_rectangles,
            self.fixed_obstacle_sample_spacing,
        )

        registration_local_frame = str(
            get(p + "registration/local_frame", self.route_frame)
        ).strip().lstrip("/")
        if registration_local_frame != self.route_frame:
            raise rospy.ROSInitException(
                "parking registration/local_frame must match route/frame_id"
            )
        if bool(
            get(p + "registration/use_fixed_obstacle_faces", True)
        ):
            visible_face_sides = get(
                p + "registration/fixed_obstacle_visible_faces",
                ["minimum_y", "maximum_x"],
            )
            self.registration_visible_faces = tuple(
                str(value).strip().lower() for value in visible_face_sides
            )
            self.registration_landmarks = (
                self._registration_landmarks_from_rectangles(
                    self.fixed_obstacle_rectangles,
                    self.registration_visible_faces,
                )
            )
        else:
            self.registration_visible_faces = tuple()
            self.registration_landmarks = self._registration_segments_param(
                get(p + "registration/segment_landmarks", [])
            )
        if len(self.registration_landmarks) != 2:
            raise rospy.ROSInitException(
                "parking registration requires exactly two nonparallel "
                "fixture faces"
            )
        self.registration_template = MissionLocalTemplate(
            mission=self.mission_name,
            frame_id=self.route_frame,
            segments=self.registration_landmarks,
            entry_plane=EntryPlane(
                point=tuple(
                    float(value)
                    for value in get(
                        p + "registration/entry_plane_point",
                        [self.aisle_x + self.turn_curve_offset, self.entry_y],
                    )
                ),
                normal=tuple(
                    float(value)
                    for value in get(
                        p + "registration/entry_plane_normal", [-1.0, 0.0]
                    )
                ),
            ),
        )
        self.registration_config = RegistrationConfig(
            position_inlier_threshold=max(
                0.001,
                float(
                    get(
                        p + "registration/position_inlier_threshold",
                        0.025,
                    )
                ),
            ),
            heading_inlier_threshold=math.radians(
                max(
                    0.1,
                    float(
                        get(
                            p + "registration/heading_inlier_threshold_deg",
                            8.0,
                        )
                    ),
                )
            ),
            minimum_inliers=2,
            minimum_template_coverage=1.0,
            minimum_inlier_coverage=1.0,
        )
        self.registration_filter = TemporalRegistrationFilter(
            TemporalRegistrationConfig(
                required_confirmations=max(
                    2,
                    int(get(p + "registration/confirmation_scans", 3)),
                ),
                maximum_gap=max(
                    0.05,
                    float(
                        get(p + "registration/confirmation_max_gap", 0.25)
                    ),
                ),
                maximum_position_delta=max(
                    0.001,
                    float(
                        get(
                            p + "registration/maximum_position_delta",
                            0.020,
                        )
                    ),
                ),
                maximum_heading_delta=math.radians(
                    max(
                        0.1,
                        float(
                            get(
                                p + "registration/maximum_heading_delta_deg",
                                2.0,
                            )
                        ),
                    )
                ),
            )
        )
        self.registration_source_max_age = max(
            0.05, float(get(p + "registration/source_max_age", 0.35))
        )
        self.ready_republish_period = max(
            0.05,
            float(get(p + "registration/ready_republish_period", 0.15)),
        )
        self.registration_future_tolerance = max(
            0.0, float(get(p + "registration/future_tolerance", 0.05))
        )
        self.registration_scan_maximum_range = max(
            0.1, float(get(p + "registration/maximum_range", 1.20))
        )
        self.registration_cluster_gap = max(
            0.005, float(get(p + "registration/cluster_gap", 0.035))
        )
        self.registration_line_residual = max(
            0.001, float(get(p + "registration/line_residual", 0.008))
        )
        self.registration_minimum_points = max(
            3, int(get(p + "registration/minimum_segment_points", 4))
        )
        self.registration_minimum_segment_length = max(
            0.01,
            float(get(p + "registration/minimum_segment_length", 0.045)),
        )
        self.registration_maximum_segment_length = max(
            self.registration_minimum_segment_length,
            float(get(p + "registration/maximum_segment_length", 0.18)),
        )
        self.registration_ambiguity_score = max(
            0.0,
            float(get(p + "registration/ambiguity_score", 0.002)),
        )
        self.registration_ambiguity_position = max(
            0.001,
            float(get(p + "registration/ambiguity_position", 0.04)),
        )
        self.registration_ambiguity_heading = math.radians(
            max(
                0.1,
                float(get(p + "registration/ambiguity_heading_deg", 2.0)),
            )
        )
        self.registration_face_heading_weight = clamp(
            float(get(p + "registration/face_heading_weight", 0.0)),
            0.0,
            1.0,
        )
        self.registration_heading_class_tolerance = math.radians(
            max(
                1.0,
                float(
                    get(
                        p
                        + "registration/face_heading_class_tolerance_deg",
                        40.0,
                    )
                ),
            )
        )
        self.registration_heading_class_margin = math.radians(
            max(
                0.0,
                float(
                    get(
                        p + "registration/face_heading_class_margin_deg",
                        10.0,
                    )
                ),
            )
        )
        self.registration_longitudinal_tolerance = max(
            0.0,
            float(
                get(
                    p + "registration/face_longitudinal_tolerance",
                    0.025,
                )
            ),
        )
        self.registration_robot_local_bounds = self._box_param(
            get(
                p + "registration/robot_local_bounds",
                [0.62, 1.60, 1.55, 1.90],
            ),
            "registration/robot_local_bounds",
        )
        self.registration_robot_heading_tolerance = math.radians(
            max(
                1.0,
                float(
                    get(
                        p + "registration/robot_heading_tolerance_deg",
                        25.0,
                    )
                ),
            )
        )
        self.entry_lead_minimum = float(
            get(p + "registration/entry_lead_minimum", -0.50)
        )
        self.entry_lead_maximum = float(
            get(p + "registration/entry_lead_maximum", -0.06)
        )
        self.entry_lead_lateral_tolerance = max(
            0.005,
            float(
                get(
                    p + "registration/entry_lead_lateral_tolerance",
                    0.06,
                )
            ),
        )
        self.entry_lead_heading_tolerance = math.radians(
            max(
                1.0,
                float(
                    get(
                        p + "registration/entry_lead_heading_tolerance_deg",
                        15.0,
                    )
                ),
            )
        )
        if self.entry_lead_minimum >= self.entry_lead_maximum:
            raise rospy.ROSInitException(
                "parking entry lead longitudinal bounds are reversed"
            )

        self.lane_path_timeout = max(
            0.1, float(get(p + "rejoin/lane_path_timeout", 0.35))
        )
        self.handoff_min_x = float(get(p + "rejoin/handoff_min_x", 0.215))
        self.handoff_max_x = float(get(p + "rejoin/handoff_max_x", 0.298))
        self.handoff_min_y = float(get(p + "rejoin/handoff_min_y", 1.738))
        self.handoff_max_y = float(get(p + "rejoin/handoff_max_y", 1.762))
        self.handoff_heading_tolerance = math.radians(
            abs(float(get(p + "rejoin/handoff_heading_tolerance_deg", 5.0)))
        )
        self.handoff_confirm_frames = max(
            1, int(get(p + "rejoin/handoff_confirmation_frames", 9))
        )
        self.handoff_confirmation_max_gap = max(
            0.05, float(get(p + "rejoin/handoff_confirmation_max_gap", 0.20))
        )
        self.handoff_timeout = max(
            0.5, float(get(p + "rejoin/handoff_timeout", 2.0))
        )
        self.rejoin_min_x = float(get(p + "rejoin/complete_min_x", 0.08))
        self.rejoin_max_x = float(get(p + "rejoin/complete_max_x", 0.19))
        self.rejoin_min_y = float(get(p + "rejoin/complete_min_y", 1.738))
        self.rejoin_max_y = float(get(p + "rejoin/complete_max_y", 1.762))
        self.rejoin_heading_tolerance = math.radians(
            abs(float(get(p + "rejoin/complete_heading_tolerance_deg", 7.0)))
        )
        self.rejoin_confirm_frames = max(
            1, int(get(p + "rejoin/complete_confirmation_frames", 9))
        )
        self.rejoin_confirmation_max_gap = max(
            0.05, float(get(p + "rejoin/complete_confirmation_max_gap", 0.20))
        )
        self.rejoin_timeout = max(
            1.0, float(get(p + "rejoin/complete_timeout", 8.0))
        )
        if not (
            self.handoff_min_x <= self.handoff_max_x
            and self.handoff_min_y <= self.handoff_max_y
            and self.rejoin_min_x <= self.rejoin_max_x
            and self.rejoin_min_y <= self.rejoin_max_y
        ):
            raise rospy.ROSInitException("parking rejoin bounds are invalid")

        self.left_box = self._box_param(
            get(p + "occupancy/left_roi", [0.60, 0.88, 0.65, 0.92]),
            "left_roi",
        )
        self.right_box = self._box_param(
            get(p + "occupancy/right_roi", [0.10, 0.38, 0.65, 0.92]),
            "right_roi",
        )
        self.occupied_minimum_points = max(
            1, int(get(p + "occupancy/occupied_minimum_points", 4))
        )
        self.clear_maximum_points = max(
            0, int(get(p + "occupancy/clear_maximum_points", 1))
        )
        if self.clear_maximum_points >= self.occupied_minimum_points:
            raise rospy.ROSInitException(
                "parking clear_maximum_points must be below occupied_minimum_points"
            )
        self.selection_confirm_scans = max(
            1, int(get(p + "occupancy/confirmation_scans", 3))
        )
        self.selection_confirmation_max_gap = max(
            0.05,
            float(get(p + "occupancy/confirmation_max_gap", 0.25)),
        )
        self.scan_maximum_range = max(
            0.1, float(get(p + "occupancy/maximum_range", 0.65))
        )
        self.scan_median_window = max(
            1, int(get(p + "occupancy/median_window", 3))
        )
        if self.scan_median_window % 2 == 0:
            raise rospy.ROSInitException("parking median_window must be odd")
        self.lidar_x = float(get(p + "scan/lidar_x", -0.033073))
        self.lidar_y = float(get(p + "scan/lidar_y", 0.0))

        self.front = max(0.01, float(get(p + "footprint/front", 0.067645)))
        self.rear = max(0.01, float(get(p + "footprint/rear", 0.118073)))
        self.half_width = max(
            0.01, float(get(p + "footprint/half_width", 0.0903))
        )
        self.line_margin = max(
            0.0, float(get(p + "footprint/line_margin", 0.009))
        )
        self.aisle_right_edge = float(
            get(p + "paint/aisle_right_edge", 0.3923)
        )
        self.aisle_left_edge = float(
            get(p + "paint/aisle_left_edge", 0.6231)
        )
        self.right_portal_edge = float(
            get(p + "paint/right_portal_edge", 0.3692)
        )
        self.left_portal_edge = float(
            get(p + "paint/left_portal_edge", 0.6385)
        )
        self.right_outer_edge = float(
            get(p + "paint/right_outer_edge", 0.1385)
        )
        self.left_outer_edge = float(
            get(p + "paint/left_outer_edge", 0.8615)
        )
        self.lower_outer_edge = float(
            get(p + "paint/lower_outer_edge", 0.5161)
        )
        self.upper_outer_edge = float(
            get(p + "paint/upper_outer_edge", 0.9919)
        )
        self.entry_boundary_slope = float(
            get(p + "paint/entry_boundary_slope", 0.001592655)
        )
        self.entry_lower_intercept = float(
            get(p + "paint/entry_lower_intercept", 1.63709885)
        )
        self.entry_upper_intercept = float(
            get(p + "paint/entry_upper_intercept", 1.87097011)
        )
        self.entry_turn_corner_relief = max(
            0.0, float(get(p + "paint/entry_turn_corner_relief", 0.010))
        )
        self.exit_turn_corner_relief = max(
            0.0, float(get(p + "paint/exit_turn_corner_relief", 0.026))
        )
        # The two dotted parking portals are legal openings. Their vertical
        # pixels end before the horizontal solid arms begin, so use the finite
        # arm endpoints instead of extending an aisle edge through a fictitious
        # closed corner.
        self.parking_opening_right_edge = float(
            get(p + "paint/parking_opening_right_edge", 0.3831)
        )
        self.parking_opening_left_edge = float(
            get(p + "paint/parking_opening_left_edge", 0.6129)
        )
        self.parking_opening_upper_edge = float(
            get(p + "paint/parking_opening_upper_edge", 1.0001)
        )
        if not (
            self.right_outer_edge
            < self.parking_opening_right_edge
            < self.parking_opening_left_edge
            < self.left_outer_edge
            and self.lower_outer_edge
            < self.parking_opening_upper_edge
        ):
            raise rospy.ROSInitException(
                "parking solid-arm opening edges are reversed"
            )
        if not (
            all(
                math.isfinite(value)
                for value in (
                    self.entry_boundary_slope,
                    self.entry_lower_intercept,
                    self.entry_upper_intercept,
                )
            )
            and self.entry_lower_intercept < self.entry_upper_intercept
        ):
            raise rospy.ROSInitException(
                "parking entry corridor paint edges are reversed"
            )
        self._validate_route_geometry()

        self.approach_speed = max(
            0.01, float(get(p + "control/approach_velocity", 0.09))
        )
        self.aisle_speed = max(
            0.01, float(get(p + "control/aisle_velocity", 0.20))
        )
        self.parking_speed = max(
            0.01, float(get(p + "control/parking_velocity", 0.20))
        )
        self.reverse_speed = max(
            0.01, float(get(p + "control/reverse_velocity", 0.20))
        )
        self.rejoin_speed = max(
            0.01, float(get(p + "control/rejoin_velocity", 0.10))
        )
        self.entry_turn_speed = max(
            0.005, float(get(p + "control/entry_turn_velocity", 0.10))
        )
        self.entry_connector_tangent_ratio = clamp(
            float(get(p + "control/entry_connector_tangent_ratio", 0.20)),
            0.01,
            0.24,
        )
        self.entry_connector_samples = max(
            11, int(get(p + "control/entry_connector_samples", 101))
        )
        self.entry_handoff_minimum_clearance = clamp(
            float(
                get(
                    p + "control/entry_handoff_minimum_clearance",
                    self.line_margin,
                )
            ),
            0.0,
            self.line_margin,
        )
        self.entry_handoff_maximum_path_error = max(
            0.0,
            float(
                get(
                    p + "control/entry_handoff_maximum_path_error",
                    0.015,
                )
            ),
        )
        self.entry_handoff_maximum_heading_error = math.radians(
            abs(
                float(
                    get(
                        p
                        + "control/entry_handoff_maximum_heading_error_deg",
                        8.0,
                    )
                )
            )
        )
        self.entry_curve_end_position_tolerance = max(
            0.003,
            float(
                get(
                    p + "control/entry_curve_end_position_tolerance",
                    0.010,
                )
            ),
        )
        self.entry_curve_end_crossing_max_distance = max(
            self.entry_curve_end_position_tolerance,
            float(
                get(
                    p + "control/entry_curve_end_crossing_max_distance",
                    0.025,
                )
            ),
        )
        self.entry_curve_end_heading_tolerance = math.radians(
            abs(
                float(
                    get(
                        p + "control/entry_curve_end_heading_tolerance_deg",
                        4.0,
                    )
                )
            )
        )
        self.arc_angular_scale = float(
            get(p + "control/arc_angular_scale", 1.0)
        )
        if not math.isfinite(self.arc_angular_scale) or self.arc_angular_scale <= 0.0:
            raise rospy.ROSInitException(
                "parking arc_angular_scale must be finite and positive"
            )
        self.lane_entry_speed_limit = max(
            0.0, float(get(p + "control/lane_entry_speed_limit", 0.08))
        )
        self.lane_resume_max_velocity = max(
            0.0, float(get(p + "control/lane_resume_max_velocity", 0.30))
        )
        self.post_completion_velocity_cap = max(
            0.0,
            float(
                get(
                    p + "control/post_completion_velocity_cap",
                    self.rejoin_speed,
                )
            ),
        )
        self.minimum_velocity = max(
            0.001, float(get(p + "control/minimum_velocity", 0.018))
        )
        self.cruise_velocity = max(
            self.minimum_velocity,
            float(
                get(
                    p + "control/cruise_velocity",
                    max(
                        self.approach_speed,
                        self.aisle_speed,
                        self.parking_speed,
                        self.reverse_speed,
                        self.rejoin_speed,
                        self.entry_turn_speed,
                    ),
                )
            ),
        )
        self.profile_entry_velocity = clamp(
            float(get(p + "control/entry_velocity", self.minimum_velocity)),
            self.minimum_velocity,
            self.cruise_velocity,
        )
        self.profile_exit_velocity = clamp(
            float(
                get(
                    p + "control/exit_velocity",
                    self.minimum_velocity,
                )
            ),
            self.minimum_velocity,
            self.cruise_velocity,
        )
        self.maximum_angular_velocity = max(
            0.05, float(get(p + "control/maximum_angular_velocity", 0.55))
        )
        self.rotation_velocity = clamp(
            float(
                get(
                    p + "control/in_place_rotation_velocity",
                    self.maximum_angular_velocity,
                )
            ),
            0.01,
            self.maximum_angular_velocity,
        )
        self.rotation_minimum_velocity = clamp(
            float(get(p + "control/in_place_rotation_minimum_velocity", 0.0)),
            0.0,
            self.rotation_velocity,
        )
        self.rotation_heading_gain = max(
            0.0, float(get(p + "control/in_place_rotation_heading_gain", 2.0))
        )
        self.rotation_command_tolerance = math.radians(
            abs(
                float(
                    get(
                        p + "control/in_place_rotation_command_tolerance_deg",
                        0.2,
                    )
                )
            )
        )
        self.rotation_heading_tolerance = math.radians(
            abs(
                float(
                    get(
                        p + "control/in_place_rotation_heading_tolerance_deg",
                        1.0,
                    )
                )
            )
        )
        self.rotation_stopped_velocity = max(
            0.0,
            float(
                get(
                    p + "control/in_place_rotation_stopped_velocity",
                    0.03,
                )
            ),
        )
        self.stopped_linear_velocity = max(
            0.0,
            float(get(p + "control/stopped_linear_velocity", 0.03)),
        )
        self.maximum_lateral_acceleration = max(
            0.001,
            float(get(p + "control/maximum_lateral_acceleration", 0.030)),
        )
        self.linear_acceleration = max(
            0.01, float(get(p + "control/linear_acceleration", 0.20))
        )
        self.linear_deceleration = max(
            0.01,
            float(
                get(
                    p + "control/linear_deceleration",
                    self.linear_acceleration,
                )
            ),
        )
        self.angular_acceleration = max(
            0.05, float(get(p + "control/angular_acceleration", 0.60))
        )
        self.common_lookahead_distance = max(
            0.005,
            float(get(p + "control/lookahead_distance", 0.050)),
        )
        self.curvature_feedforward_weight = clamp(
            float(get(p + "control/curvature_feedforward_weight", 0.25)),
            0.0,
            1.0,
        )
        self.lateral_feedback_gain = max(
            0.0, float(get(p + "control/lateral_feedback_gain", 1.0))
        )
        self.path_search_back = max(
            0, int(get(p + "control/path_search_back", 3))
        )
        self.path_search_ahead = max(
            self.common_lookahead_distance,
            float(get(p + "control/path_search_ahead", 0.30)),
        )
        self.path_heading_gain = float(
            get(p + "control/path_heading_gain", 0.35)
        )
        self.position_tolerance = max(
            0.005, float(get(p + "control/position_tolerance", 0.008))
        )
        self.arc_position_tolerance = max(
            0.003, float(get(p + "control/arc_position_tolerance", 0.005))
        )
        self.arc_distance_tolerance = max(
            self.arc_position_tolerance,
            float(get(p + "control/arc_distance_tolerance", 0.015)),
        )
        self.heading_tolerance = math.radians(
            abs(float(get(p + "control/heading_tolerance_deg", 4.0)))
        )
        self.arc_heading_tolerance = math.radians(
            abs(float(get(p + "control/arc_heading_tolerance_deg", 1.0)))
        )
        self.overshoot_tolerance = max(
            self.position_tolerance,
            float(get(p + "control/overshoot_tolerance", 0.05)),
        )
        self.pose_confirm_samples = max(
            1, int(get(p + "control/pose_confirmation_samples", 2))
        )
        self.control_period = max(
            0.02, float(get(p + "control/period", 0.05))
        )
        self.prepare_settle_time = max(
            0.0, float(get(p + "control/prepare_settle_time", 0.10))
        )
        self.checkpoint_position_tolerance = max(
            self.position_tolerance,
            float(get(p + "control/checkpoint_position_tolerance", 0.045)),
        )
        self.checkpoint_heading_tolerance = math.radians(
            abs(float(get(p + "control/checkpoint_heading_tolerance_deg", 10.0)))
        )

        self.pose_timeout = max(
            0.1, float(get(p + "timeouts/map_pose", 0.35))
        )
        self.odom_timeout = max(
            0.1, float(get(p + "timeouts/odometry", 0.35))
        )
        self.scan_timeout = max(
            0.1, float(get(p + "timeouts/scan", 0.35))
        )
        self.scan_odom_sync_tolerance = max(
            0.001,
            float(get(p + "safety/scan_odom_sync_tolerance", 0.05)),
        )
        self.odom_history_duration = max(
            2.0 * self.scan_odom_sync_tolerance,
            float(get(p + "safety/odom_history_duration", 1.0)),
        )
        self.pending_safety_scan_limit = max(
            1, int(get(p + "safety/pending_scan_limit", 4))
        )
        self.drive_timeout = max(
            1.0, float(get(p + "timeouts/drive", 35.0))
        )
        self.selection_timeout = max(
            1.0, float(get(p + "timeouts/selection", 4.0))
        )
        self.rotation_timeout = max(
            1.0, float(get(p + "timeouts/rotation", 8.0))
        )
        self.prepare_timeout = max(
            self.prepare_settle_time + 0.1,
            float(get(p + "timeouts/prepare", 0.75)),
        )

        self.localization_margin = max(
            0.0, float(get(p + "safety/localization_margin", 0.0))
        )
        self.tracking_margin = max(
            0.0, float(get(p + "safety/tracking_margin", 0.0))
        )
        self.obstacle_margin = max(
            0.0, float(get(p + "safety/obstacle_margin", self.line_margin))
        )
        self.rotation_obstacle_margin = max(
            0.0,
            float(
                get(
                    p + "safety/rotation_obstacle_margin",
                    self.obstacle_margin
                    + self.localization_margin
                    + self.tracking_margin,
                )
            ),
        )
        self.safety_reaction_time = max(
            0.0, float(get(p + "safety/reaction_time", 0.10))
        )
        self.safety_distance_margin = max(
            0.0, float(get(p + "safety/stopping_distance_margin", 0.005))
        )
        self.live_validation_distance = max(
            0.05, float(get(p + "safety/live_validation_distance", 0.15))
        )
        self.require_live_safety_scan = bool(
            get(p + "safety/require_live_scan", False)
        )
        self.swept_translation_step = max(
            0.001, float(get(p + "safety/swept_translation_step", 0.004))
        )
        self.swept_heading_step = math.radians(
            max(0.1, float(get(p + "safety/swept_heading_step_deg", 1.0)))
        )
        self.path_sample_spacing = max(
            0.002, float(get(p + "route/path_sample_spacing", 0.004))
        )
        self.local_safety_bounds = self._box_param(
            get(p + "safety/local_bounds", [-2.0, 2.0, -2.0, 2.0]),
            "safety/local_bounds",
        )

        self.common_tracking_config = TrackingConfig(
            lookahead_distance=self.common_lookahead_distance,
            maximum_linear_velocity=self.cruise_velocity,
            maximum_angular_velocity=self.maximum_angular_velocity,
            maximum_lateral_acceleration=self.maximum_lateral_acceleration,
            linear_acceleration=self.linear_acceleration,
            linear_deceleration=self.linear_deceleration,
            angular_acceleration=self.angular_acceleration,
            heading_gain=self.path_heading_gain,
            curvature_feedforward_weight=self.curvature_feedforward_weight,
            lateral_feedback_gain=self.lateral_feedback_gain,
            search_back=self.path_search_back,
            search_ahead_distance=self.path_search_ahead,
        )
        self.path_follower = PathFollower(self.common_tracking_config)
        self.common_footprint = AsymmetricFootprint(
            self.front, self.rear, self.half_width
        )
        self.path_validator = SweptFootprintValidator(
            self.common_footprint,
            translation_step=self.swept_translation_step,
            heading_step=self.swept_heading_step,
        )
        self._footprint_perimeter_cache = {}
        self._boundary_clearance_cache = OrderedDict()
        self._boundary_clearance_cache_limit = 8192

        self.lock = threading.RLock()
        # Swept-footprint validation can consume most of a control period.
        # Keep the timestamp-synchronized LiDAR pipeline independent from the
        # controller state lock so a validation cannot make its own input
        # stale by blocking both /scan and /odom callbacks.
        self.safety_data_lock = threading.RLock()
        # Sensor callbacks must never queue behind the comparatively expensive
        # swept-path control lock.  A blocked odom callback would also block
        # every newer message on that subscriber connection, making otherwise
        # healthy 30 Hz odometry appear stale.  Keep only its newest deferred
        # mission-state sample; the safety history is still updated immediately.
        self.pending_odom_state = None
        self.pending_registration_scan = None
        self.pending_selection_scan = None
        self.odom_arrival_generation = 0
        self.applied_odom_arrival_generation = 0
        self.state = self.WAIT_GATE
        self.zone_gate = False
        self.start_requested = False
        self.revoke_requested = False
        self.arm_seq = None
        self.arm_stamp = None
        self.ready_published_seq = None
        self.ready_source_stamp = None
        self.registration_source_stamp = None
        self.registration_confirmation_count = 0
        self.registration_rejection_reason = "not armed"
        self.registration_heading_prior = None
        self.local_to_odom = None
        self.registration_covariance = tuple()
        self.manual_stop = False
        self.mission_has_control = False
        self.handoff_ambiguous = False
        self.selected_space = None
        self.map_pose = None
        self.map_pose_stamp = None
        self.map_pose_received = None
        self.odom_pose = None
        self.odom_stamp = None
        self.odom_received = None
        self.odom_generation = 0
        self.odom_linear_speed = 0.0
        self.odom_angular_velocity = 0.0
        self.odom_pose_history = deque()
        self.pending_safety_scans = deque()
        self.scan_stamp = None
        self.scan_received = None
        self.scan_generation = 0
        self.safety_scan_stamp = None
        self.safety_scan_received = None
        self.live_obstacle_points_odom = np.empty((0, 2), dtype=np.float64)
        self.left_points = 0
        self.right_points = 0
        self.lane_path_valid = False
        self.lane_path_progress = math.nan
        self.lane_path_remaining_distance = math.nan
        self.lane_path_position_error = math.nan
        self.lane_path_cross_track_error = math.nan
        self.lane_path_heading_error = math.nan
        self.lane_path_target_speed = math.nan
        self.lane_path_target_index = math.nan
        self.lane_path_curvature = math.nan
        self.lane_path_minimum_line_clearance = math.nan
        self.lane_path_minimum_obstacle_clearance = math.nan
        self.lane_path_minimum_map_clearance = math.nan
        self.lane_path_commanded_linear = math.nan
        self.lane_path_commanded_angular = math.nan
        self.last_lane_path_time = None
        self.lane_confirmation_count = 0
        self.last_lane_confirmation_time = None

        # Sole frozen transform used by every parking route and safety check.
        self.goal_map = None
        self.goal_odom = None
        self.parking_goal_map = None
        self.parking_return_map = None
        self.parking_return_odom = None
        self.parking_goal_amcl = None
        self.parking_return_amcl = None
        self.rotation_center_odom = None
        self.rotation_target_yaw = None
        self.rotation_initial_error = 0.0
        self.rotation_static_validation = None
        self.motion_timeout = self.drive_timeout
        self.entry_curve_map = None
        self.entry_curve_odom = None
        self.entry_connector_odom = None
        self.entry_connector_speed = self.approach_speed
        self.entry_connector_path = None
        self.entry_curve_path = None
        self.entry_preparation_error = ""
        self.zigzag_exit_curve_map = None
        self.zigzag_exit_curve_odom = None
        self.zigzag_exit_curve_path = None
        self.active_path = None
        self.active_path_state = None
        self.active_path_validation = None
        self.state_started = rospy.Time.now()
        self.odom_confirmation_count = 0
        self.last_confirmed_odom_generation = -1
        self.motion_goal_confirmed = False
        self.prepare_odom_generation = -1
        self.selection_candidate = None
        self.selection_candidate_count = 0
        self.last_candidate_scan_stamp = None
        self.last_selection_scan_generation = -1
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.last_command_time = None
        self.observed_lane_linear = 0.0
        self.observed_lane_angular = 0.0
        self.observed_lane_command_received = None
        self.lane_command_generation = 0
        self.gate_lane_command_generation = 0
        self.gate_odom_generation = 0
        self.bounded_lane_command_generation = None
        self.bounded_lane_command_odom_generation = None
        self.shutting_down = False

        try:
            self.planned_left_route = self._planned_route_poses(LEFT)
            self.planned_right_route = self._planned_route_poses(RIGHT)
        except ValueError as error:
            raise rospy.ROSInitException(
                "cannot build parking preview paths: %s" % error
            )

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.state_pub = rospy.Publisher(
            "/parking/state", String, queue_size=1, latch=True
        )
        self.space_pub = rospy.Publisher(
            "/parking/selected_space", String, queue_size=1, latch=True
        )
        self.occupancy_pub = rospy.Publisher(
            "/parking/occupancy_points", Int32MultiArray, queue_size=1
        )
        self.diagnostics_pub = rospy.Publisher(
            self.diagnostics_topic,
            Float64MultiArray,
            queue_size=1,
            latch=True,
        )
        self.ready_pub = rospy.Publisher(
            self.ready_topic, Header, queue_size=1, latch=True
        )
        self.planned_left_path_pub = rospy.Publisher(
            self.planned_left_path_topic, Path, queue_size=1, latch=True
        )
        self.planned_right_path_pub = rospy.Publisher(
            self.planned_right_path_topic, Path, queue_size=1, latch=True
        )
        self.speed_limit_pub = rospy.Publisher(
            self.speed_limit_topic, Float64, queue_size=1
        )
        self.emergency_stop_pub = rospy.Publisher(
            self.manual_stop_topic, Bool, queue_size=1
        )
        self.lane_service = rospy.ServiceProxy(self.lane_service_name, SetBool)
        self.lane_stop_service = rospy.ServiceProxy(
            self.lane_stop_service_name, SetBool
        )

        rospy.Subscriber(
            self.gate_topic, Bool, self.gate_callback, queue_size=1
        )
        rospy.Subscriber(
            self.arm_topic, Header, self.arm_callback, queue_size=1
        )
        rospy.Subscriber(
            self.lane_path_topic,
            Float64MultiArray,
            self.lane_path_diagnostics_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.map_pose_topic,
            PoseStamped,
            self.map_pose_callback,
            queue_size=1,
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
            self.manual_stop_topic,
            Bool,
            self.manual_stop_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.cmd_vel_topic,
            Twist,
            self.command_observer_callback,
            queue_size=1,
        )
        self.control_timer = rospy.Timer(
            rospy.Duration(self.control_period), self.control_callback
        )
        rospy.on_shutdown(self.shutdown)

        self._publish_planned_route(
            self.planned_left_path_pub, self.planned_left_route
        )
        self._publish_planned_route(
            self.planned_right_path_pub, self.planned_right_route
        )
        self._publish_state()
        self.space_pub.publish(String(data="UNKNOWN"))
        rospy.loginfo(
            "Parking controller waiting for local fixture registration on %s: "
            "aisle x=%.4f, bay goals x=%.3f/%.3f",
            self.arm_topic,
            self.aisle_x,
            self.left_park_x,
            self.right_park_x,
        )

    @staticmethod
    def _box_param(raw, name):
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise rospy.ROSInitException(
                "parking %s must be [xmin, xmax, ymin, ymax]" % name
            )
        values = tuple(float(value) for value in raw)
        if values[0] >= values[1] or values[2] >= values[3]:
            raise rospy.ROSInitException("parking %s has reversed bounds" % name)
        return values

    @classmethod
    def _rectangles_param(cls, raw, name):
        if not isinstance(raw, (list, tuple)):
            raise rospy.ROSInitException(
                "parking %s must be a list of [xmin, xmax, ymin, ymax]"
                % name
            )
        return tuple(
            cls._box_param(rectangle, "%s[%d]" % (name, index))
            for index, rectangle in enumerate(raw)
        )

    @staticmethod
    def _registration_segments_param(raw):
        if not isinstance(raw, (list, tuple)):
            raise rospy.ROSInitException(
                "parking registration/segment_landmarks must be a list"
            )
        landmarks = []
        try:
            for index, item in enumerate(raw):
                if not isinstance(item, dict):
                    raise ValueError("landmark %d is not a dictionary" % index)
                landmarks.append(
                    OrientedSegmentLandmark(
                        name=item["name"],
                        start=item["start"],
                        end=item["end"],
                        longitudinal_weight=1.0,
                    )
                )
        except (KeyError, TypeError, ValueError) as error:
            raise rospy.ROSInitException(
                "invalid parking registration landmark: %s" % error
            )
        for index, first in enumerate(landmarks):
            for second in landmarks[index + 1 :]:
                angle = abs(normalize_angle(first.heading - second.heading))
                angle = min(angle, abs(math.pi - angle))
                if angle >= math.radians(20.0):
                    return tuple(landmarks)
        if len(landmarks) >= 2:
            raise rospy.ROSInitException(
                "parking registration landmarks are all parallel"
            )
        return tuple(landmarks)

    @staticmethod
    def _registration_landmarks_from_rectangles(rectangles, visible_faces):
        """Use one approach-visible long face from each fixed rectangle."""
        if not isinstance(visible_faces, (list, tuple)) or len(
            visible_faces
        ) != len(rectangles):
            raise rospy.ROSInitException(
                "parking registration/fixed_obstacle_visible_faces must "
                "select one face for every fixed rectangle"
            )
        landmarks = []
        for index, (rectangle, raw_face) in enumerate(
            zip(rectangles, visible_faces)
        ):
            minimum_x, maximum_x, minimum_y, maximum_y = rectangle
            width = maximum_x - minimum_x
            height = maximum_y - minimum_y
            face = str(raw_face).strip().lower()
            if width >= height:
                if face == "minimum_y":
                    start = (minimum_x, minimum_y)
                    end = (maximum_x, minimum_y)
                elif face == "maximum_y":
                    start = (minimum_x, maximum_y)
                    end = (maximum_x, maximum_y)
                else:
                    raise rospy.ROSInitException(
                        "parking rectangle %d long visible face must be "
                        "minimum_y or maximum_y" % index
                    )
            else:
                if face == "minimum_x":
                    start = (minimum_x, minimum_y)
                    end = (minimum_x, maximum_y)
                elif face == "maximum_x":
                    start = (maximum_x, minimum_y)
                    end = (maximum_x, maximum_y)
                else:
                    raise rospy.ROSInitException(
                        "parking rectangle %d long visible face must be "
                        "minimum_x or maximum_x" % index
                    )
            landmarks.append(
                OrientedSegmentLandmark(
                    name="fixed_rectangle_%d_visible_face" % index,
                    start=start,
                    end=end,
                    # LaserScan sees a range-quantized, partially occluded
                    # portion of each finite face.  The two nonparallel face
                    # lines still determine full SE(2); treating their cropped
                    # midpoints as the surveyed midpoints would bias it.
                    longitudinal_weight=0.0,
                )
            )
        # Reuse the same nonparallel validation as explicit landmarks.
        for index, first in enumerate(landmarks):
            for second in landmarks[index + 1 :]:
                angle = abs(normalize_angle(first.heading - second.heading))
                angle = min(angle, abs(math.pi - angle))
                if angle >= math.radians(20.0):
                    return tuple(landmarks)
        raise rospy.ROSInitException(
            "parking fixed obstacle faces do not observe full SE(2)"
        )

    @staticmethod
    def _sample_fixed_rectangles(rectangles, spacing):
        """Represent fixed collision boxes as a dense occupied point set."""
        clouds = []
        for minimum_x, maximum_x, minimum_y, maximum_y in rectangles:
            count_x = max(
                2, int(math.ceil((maximum_x - minimum_x) / spacing)) + 1
            )
            count_y = max(
                2, int(math.ceil((maximum_y - minimum_y) / spacing)) + 1
            )
            x = np.linspace(minimum_x, maximum_x, count_x)
            y = np.linspace(minimum_y, maximum_y, count_y)
            grid_x, grid_y = np.meshgrid(x, y)
            clouds.append(
                np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
            )
        if not clouds:
            return np.empty((0, 2), dtype=np.float64)
        return np.vstack(clouds)

    def _zigzag_turn_start_pose(self):
        return (
            self.zigzag_turn_start_x,
            self.zigzag_straight_y - self.turn_curve_offset,
            self.outgoing_heading,
        )

    def _zigzag_turn_end_pose(self):
        start_x, start_y, start_yaw = self._zigzag_turn_start_pose()
        offset = self.turn_curve_offset
        return (
            start_x
            + offset
            * (math.cos(start_yaw) + math.cos(self.zigzag_heading)),
            start_y
            + offset
            * (math.sin(start_yaw) + math.sin(self.zigzag_heading)),
            self.zigzag_heading,
        )

    def _zigzag_exit_goal_pose(self):
        end_x, end_y, end_yaw = self._zigzag_turn_end_pose()
        return (
            end_x + self.zigzag_alignment_tail * math.cos(end_yaw),
            end_y + self.zigzag_alignment_tail * math.sin(end_yaw),
            end_yaw,
        )

    def _quarter_turn_route_poses(self, start_pose, target_yaw):
        """Return the one surveyed quarter-turn shape used by parking."""
        return quintic_turn_path(
            start_pose,
            target_yaw,
            self.turn_curve_offset,
            self.turn_curve_tangent,
            self.turn_curve_samples,
        )

    def _entry_curve_route_poses(self):
        """Return the fixed surveyed entry turn in route coordinates."""
        return self._quarter_turn_route_poses(
            (
                self.aisle_x + self.turn_curve_offset,
                self.entry_y,
                self.approach_heading,
            ),
            self.aisle_heading,
        )

    def _zigzag_exit_route_poses(self):
        """Return the surveyed final left turn and alignment tail."""
        curve = self._quarter_turn_route_poses(
            self._zigzag_turn_start_pose(),
            self.zigzag_heading,
        )
        turn_end = curve[-1]
        tail_count = max(
            2,
            int(
                math.ceil(
                    self.zigzag_alignment_tail / self.path_sample_spacing
                )
            )
            + 1,
        )
        fraction = np.linspace(0.0, 1.0, tail_count)
        tail = np.column_stack(
            (
                turn_end[0]
                + fraction
                * self.zigzag_alignment_tail
                * math.cos(self.zigzag_heading),
                turn_end[1]
                + fraction
                * self.zigzag_alignment_tail
                * math.sin(self.zigzag_heading),
                np.full(tail_count, self.zigzag_heading),
            )
        )
        return np.vstack((curve, tail[1:]))

    def _straight_route_poses(self, start, goal):
        start = Pose2D.from_value(start)
        goal = Pose2D.from_value(goal)
        distance = math.hypot(goal.x - start.x, goal.y - start.y)
        count = max(
            2, int(math.ceil(distance / self.path_sample_spacing)) + 1
        )
        fraction = np.linspace(0.0, 1.0, count)
        yaw_delta = normalize_angle(goal.yaw - start.yaw)
        return np.column_stack(
            (
                start.x + fraction * (goal.x - start.x),
                start.y + fraction * (goal.y - start.y),
                start.yaw + fraction * yaw_delta,
            )
        )

    def _rotation_route_poses(self, start, target_yaw):
        poses = sample_in_place_rotation(
            start, target_yaw, self.swept_heading_step
        )
        return np.asarray(
            [(pose.x, pose.y, pose.yaw) for pose in poses],
            dtype=np.float64,
        )

    @staticmethod
    def _join_route_sections(*sections):
        joined = []
        for section in sections:
            values = np.asarray(section, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != 3 or not len(values):
                raise ValueError(
                    "parking preview section must be a non-empty Nx3 path"
                )
            joined.append(values if not joined else values[1:])
        return np.vstack(joined)

    def _nominal_exit_approach_route_poses(self):
        start = (self.aisle_x, self.decision_y, self.outgoing_heading)
        goal = self._zigzag_turn_start_pose()
        length = math.hypot(goal[0] - start[0], goal[1] - start[1])
        tangent = self.zigzag_approach_tangent_ratio * max(0.01, length)
        sample_count = max(
            11, int(math.ceil(length / self.path_sample_spacing)) + 1
        )
        return quintic_pose_path(
            start,
            goal,
            tangent,
            tangent,
            sample_count,
        )

    def _planned_route_poses(self, selected_space):
        """Build one complete nominal LEFT or RIGHT route for RViz."""
        if selected_space not in (LEFT, RIGHT):
            raise ValueError("parking preview requires LEFT or RIGHT")

        entry_curve = self._entry_curve_route_poses()
        decision = (self.aisle_x, self.decision_y, self.aisle_heading)
        aisle = self._straight_route_poses(entry_curve[-1], decision)
        park_yaw = 0.0 if selected_space == LEFT else math.pi
        park_x = (
            self.left_park_x
            if selected_space == LEFT
            else self.right_park_x
        )
        turn_to_space = self._rotation_route_poses(decision, park_yaw)
        return_pose = (self.aisle_x, self.decision_y, park_yaw)
        park_pose = (park_x, self.decision_y, park_yaw)
        park_in = self._straight_route_poses(return_pose, park_pose)
        back_out = self._straight_route_poses(park_pose, return_pose)
        turn_to_exit = self._rotation_route_poses(
            return_pose, self.outgoing_heading
        )
        exit_approach = self._nominal_exit_approach_route_poses()
        exit_curve = self._zigzag_exit_route_poses()
        route = self._join_route_sections(
            entry_curve,
            aisle,
            turn_to_space,
            park_in,
            back_out,
            turn_to_exit,
            exit_approach,
            exit_curve,
        )
        if not np.all(np.isfinite(route)):
            raise ValueError("parking preview route contains non-finite poses")
        return route

    def _publish_planned_route(self, publisher, route):
        """Latch one surveyed parking-local candidate with a zero stamp."""
        message = Path()
        message.header.stamp = rospy.Time()
        message.header.frame_id = self.route_frame
        for x, y, yaw in np.asarray(route, dtype=np.float64):
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.05
            pose.pose.orientation.z = math.sin(0.5 * float(yaw))
            pose.pose.orientation.w = math.cos(0.5 * float(yaw))
            message.poses.append(pose)
        publisher.publish(message)

    def _validate_route_geometry(self):
        if self.entry_y <= self.decision_y:
            raise rospy.ROSInitException(
                "parking entry_y must be north of decision_y"
            )
        if self.entry_y - self.turn_curve_offset <= self.decision_y:
            raise rospy.ROSInitException(
                "parking entry curve must end north of decision_y"
            )
        if self.turn_curve_tangent >= 0.5 * self.turn_curve_offset:
            raise rospy.ROSInitException(
                "parking turn curve tangent must be below half its offset"
            )
        entry_start = (
            self.aisle_x + self.turn_curve_offset,
            self.entry_y,
            self.approach_heading,
        )
        entry_corners = rectangle_corners(
            entry_start, self.front, self.rear, self.half_width
        )
        if any(
            not (
                self.entry_lower_intercept
                + self.entry_boundary_slope * x
                + self.line_margin
                <= y
                <= self.entry_upper_intercept
                + self.entry_boundary_slope * x
                - self.line_margin
            )
            for x, y in entry_corners
        ):
            raise rospy.ROSInitException(
                "parking entry curve start does not clear both solid lines"
            )
        turn_start = self._zigzag_turn_start_pose()
        if not self.decision_y < turn_start[1]:
            raise rospy.ROSInitException(
                "parking decision_y must precede the zigzag turn start"
            )
        turn_angle = abs(
            normalize_angle(self.zigzag_heading - self.outgoing_heading)
        )
        if not math.isclose(
            turn_angle, 0.5 * math.pi, rel_tol=0.0, abs_tol=1e-9
        ):
            raise rospy.ROSInitException(
                "parking zigzag exit must be a 90-degree turn"
            )
        aisle_pose = (self.aisle_x, self.decision_y, self.aisle_heading)
        aisle_corners = rectangle_corners(
            aisle_pose, self.front, self.rear, self.half_width
        )
        aisle_min_x = self.aisle_right_edge + self.line_margin
        aisle_max_x = self.aisle_left_edge - self.line_margin
        if any(not aisle_min_x <= x <= aisle_max_x for x, _ in aisle_corners):
            raise rospy.ROSInitException(
                "parking aisle target does not clear both dotted boundaries"
            )

        if self.left_park_x <= self.aisle_x:
            raise rospy.ROSInitException(
                "parking left goal must be beyond its in-place turn centre"
            )
        if self.right_park_x >= self.aisle_x:
            raise rospy.ROSInitException(
                "parking right goal must be beyond its in-place turn centre"
            )

        # A complete 90-degree spin must fit inside the horizontal parking
        # rectangle.  Checking the circumscribed radius of the margin-expanded
        # asymmetric footprint is conservative for every intermediate yaw.
        rotation_radius = max(
            math.hypot(
                self.front + self.line_margin,
                self.half_width + self.line_margin,
            ),
            math.hypot(
                self.rear + self.line_margin,
                self.half_width + self.line_margin,
            ),
        )
        if not (
            self.right_outer_edge + rotation_radius
            < self.aisle_x
            < self.left_outer_edge - rotation_radius
            and self.lower_outer_edge + rotation_radius
            < self.decision_y
            < self.upper_outer_edge - rotation_radius
        ):
            raise rospy.ROSInitException(
                "parking in-place turn does not clear the solid border"
            )

        left_pose = (self.left_park_x, self.decision_y, 0.0)
        right_pose = (self.right_park_x, self.decision_y, math.pi)
        left_min_x = self.left_portal_edge + self.line_margin
        left_max_x = self.left_outer_edge - self.line_margin
        right_min_x = self.right_outer_edge + self.line_margin
        right_max_x = self.right_portal_edge - self.line_margin
        lower_y = self.lower_outer_edge + self.line_margin
        upper_y = self.upper_outer_edge - self.line_margin
        for label, corners, minimum_x, maximum_x in (
            (
                LEFT,
                rectangle_corners(
                    left_pose, self.front, self.rear, self.half_width
                ),
                left_min_x,
                left_max_x,
            ),
            (
                RIGHT,
                rectangle_corners(
                    right_pose, self.front, self.rear, self.half_width
                ),
                right_min_x,
                right_max_x,
            ),
        ):
            if any(
                not (minimum_x <= x <= maximum_x and lower_y <= y <= upper_y)
                for x, y in corners
            ):
                raise rospy.ROSInitException(
                    "parking %s goal does not put the whole footprint beyond "
                    "the dotted line and inside the solid boundary" % label
                )

        turn_start_corners = rectangle_corners(
            turn_start, self.front, self.rear, self.half_width
        )
        if any(
            not aisle_min_x <= x <= aisle_max_x
            for x, _ in turn_start_corners
        ):
            raise rospy.ROSInitException(
                "parking zigzag turn start does not clear the aisle"
            )

        turn_end = self._zigzag_turn_end_pose()
        exit_goal = self._zigzag_exit_goal_pose()
        if not (
            math.isclose(
                turn_end[1],
                self.zigzag_straight_y,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                exit_goal[1],
                self.zigzag_straight_y,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise rospy.ROSInitException(
                "parking zigzag exit does not finish on its straight"
            )
        if not (
            self.handoff_min_x <= exit_goal[0] <= self.handoff_max_x
            and self.handoff_min_y <= exit_goal[1] <= self.handoff_max_y
        ):
            raise rospy.ROSInitException(
                "parking zigzag alignment tail does not end in the handoff bounds"
            )
        handoff_epsilon = 1e-9
        if not (
            min(turn_end[0], exit_goal[0]) - handoff_epsilon
            <= self.handoff_max_x
            <= max(turn_end[0], exit_goal[0]) + handoff_epsilon
        ):
            raise rospy.ROSInitException(
                "parking moving handoff does not begin on the alignment tail"
            )
        if self.rejoin_max_x >= self.handoff_min_x:
            raise rospy.ROSInitException(
                "parking completion bounds must follow the handoff bounds"
            )

    def _publish_state(self):
        self.state_pub.publish(String(data=self.state))

    def _set_state(self, state):
        if state == self.state:
            return
        self.state = state
        self.state_started = rospy.Time.now()
        self.odom_confirmation_count = 0
        self.last_confirmed_odom_generation = -1
        self.motion_goal_confirmed = False
        if state in (
            self.TURN_TO_ZIGZAG,
            self.VERIFY_ZIGZAG_LANE,
            self.JOIN_ZIGZAG,
        ):
            self.lane_confirmation_count = 0
            self.last_lane_confirmation_time = None
        self._publish_state()
        if self.map_pose is not None and self.odom_pose is not None:
            rospy.loginfo(
                "Parking mission state: %s diagnostic_map=(%.4f,%.4f,%.1fdeg) "
                "odom=(%.4f,%.4f,%.1fdeg)",
                state,
                self.map_pose[0],
                self.map_pose[1],
                math.degrees(self.map_pose[2]),
                self.odom_pose[0],
                self.odom_pose[1],
                math.degrees(self.odom_pose[2]),
            )
        else:
            rospy.loginfo("Parking mission state: %s", state)

    def _reset_local_registration(self, reason="new arm generation"):
        self.registration_filter.reset(reason)
        self.registration_confirmation_count = 0
        self.registration_rejection_reason = reason
        self.registration_source_stamp = None
        self.registration_heading_prior = None
        self.local_to_odom = None
        self.registration_covariance = tuple()
        self.ready_published_seq = None
        self.ready_source_stamp = None
        self.entry_curve_map = None
        self.entry_curve_odom = None
        self.entry_connector_odom = None
        self.entry_connector_path = None
        self.entry_curve_path = None
        self.active_path = None
        self.active_path_state = None
        self.active_path_validation = None
        self.bounded_lane_command_generation = None
        self.bounded_lane_command_odom_generation = None

    def arm_callback(self, message):
        """Arm one ordered generation without touching lane ownership."""
        with self.lock:
            sequence = int(message.seq)
            if sequence == 0 or sequence == self.arm_seq:
                return
            if message.frame_id != self.mission_name:
                rospy.logwarn(
                    "Ignoring parking arm frame '%s' (expected '%s')",
                    message.frame_id,
                    self.mission_name,
                )
                return
            if message.stamp == rospy.Time():
                rospy.logwarn(
                    "Ignoring parking arm generation %d with zero stamp",
                    sequence,
                )
                return
            if self.mission_has_control or self.state not in (
                self.WAIT_GATE,
                self.COMPLETE,
                self.FAILED,
            ):
                rospy.logwarn(
                    "Ignoring parking arm generation %d while %s",
                    sequence,
                    self.state,
                )
                return
            self.arm_seq = sequence
            self.arm_stamp = message.stamp
            self.zone_gate = False
            self.start_requested = False
            self.revoke_requested = False
            self._reset_local_registration()
            # Registration needs two independent LiDAR observations before
            # the fixed entry bundle can be advertised.  Keep lane ownership,
            # but apply the parking common cruise profile as soon as this
            # mission is armed so those observations cannot be crossed at the
            # normal-lane 0.28 m/s rate.  The stricter entry cap is still
            # applied only after the ordered gate opens.
            self.speed_limit_pub.publish(
                Float64(data=self.cruise_velocity)
            )
            if self.state != self.WAIT_GATE:
                self._set_state(self.WAIT_GATE)
            rospy.loginfo(
                "Parking registration armed: generation=%d stamp=%.3f; "
                "lane controller remains owner at %.3fm/s maximum",
                sequence,
                message.stamp.to_sec(),
                self.cruise_velocity,
            )

    def gate_callback(self, message):
        with self.lock:
            self._apply_pending_odom_state()
            was_open = self.zone_gate
            requested = bool(message.data)
            prepared = bool(
                self._ready_matches_current_arm(rospy.Time.now())
            )
            self.zone_gate = requested and prepared
            if requested and not prepared:
                rospy.logwarn_throttle(
                    1.0,
                    "Ignoring parking enable before matching local ready",
                )
                return
            if self.zone_gate and not was_open:
                # A command and odometry sample newer than this gate edge are
                # required before parking can take cmd_vel ownership. This
                # lets the lane controller apply the entry cap while it keeps
                # the robot moving.
                self.gate_lane_command_generation = (
                    self.lane_command_generation
                )
                self.gate_odom_generation = self.odom_generation
                self.bounded_lane_command_generation = None
                self.bounded_lane_command_odom_generation = None
                self.start_requested = True
                self.speed_limit_pub.publish(
                    Float64(data=self.lane_entry_speed_limit)
                )
            elif (
                not self.zone_gate
                and was_open
                and self.state not in (self.COMPLETE, self.WAIT_GATE)
            ):
                self.revoke_requested = True

    def map_pose_callback(self, message):
        received = rospy.Time.now()
        source_stamp = (
            message.header.stamp
            if message.header.stamp != rospy.Time()
            else received
        )
        with self.lock:
            self.map_pose = (
                float(message.pose.position.x),
                float(message.pose.position.y),
                yaw_from_quaternion(message.pose.orientation),
            )
            self.map_pose_stamp = source_stamp
            self.map_pose_received = received

    def _anchored_pose_in_bounds(
        self, minimum_x, maximum_x, minimum_y, maximum_y, heading_tolerance
    ):
        anchored_pose = self._anchored_local_pose()
        if anchored_pose is None:
            return False
        x, y, yaw = anchored_pose
        return (
            minimum_x <= x <= maximum_x
            and minimum_y <= y <= maximum_y
            and abs(normalize_angle(yaw - self.zigzag_heading))
            <= heading_tolerance
        )

    def _anchored_local_pose(self):
        if self.odom_pose is None or self._frozen_local_to_odom() is None:
            return None
        return self._odom_pose_to_route(self.odom_pose)

    def _lane_observation_summary(self):
        anchored_pose = self._anchored_local_pose()
        if anchored_pose is None:
            pose_text = "unavailable"
        else:
            pose_text = "(%.4f,%.4f,%.1fdeg)" % (
                anchored_pose[0],
                anchored_pose[1],
                math.degrees(anchored_pose[2]),
            )
        return (
            "anchored_pose=%s lane_path=(valid=%s progress=%.3f "
            "remaining=%.3f position=%.4f cross_track=%.4f heading=%.4f "
            "target_speed=%.3f target_index=%.1f curvature=%.4f "
            "min_line=%.4f min_obstacle=%.4f min_map=%.4f "
            "command=(%.3f,%.3f))"
            % (
                pose_text,
                self.lane_path_valid,
                self.lane_path_progress,
                self.lane_path_remaining_distance,
                self.lane_path_position_error,
                self.lane_path_cross_track_error,
                self.lane_path_heading_error,
                self.lane_path_target_speed,
                self.lane_path_target_index,
                self.lane_path_curvature,
                self.lane_path_minimum_line_clearance,
                self.lane_path_minimum_obstacle_clearance,
                self.lane_path_minimum_map_clearance,
                self.lane_path_commanded_linear,
                self.lane_path_commanded_angular,
            )
        )

    def _lane_observation_valid(self, now, joining):
        lane_path_fresh = (
            self.last_lane_path_time is not None
            and (now - self.last_lane_path_time).to_sec()
            <= self.lane_path_timeout
        )
        if joining:
            pose_valid = self._anchored_pose_in_bounds(
                self.rejoin_min_x,
                self.rejoin_max_x,
                self.rejoin_min_y,
                self.rejoin_max_y,
                self.rejoin_heading_tolerance,
            )
        else:
            pose_valid = self._anchored_pose_in_bounds(
                self.handoff_min_x,
                self.handoff_max_x,
                self.handoff_min_y,
                self.handoff_max_y,
                self.handoff_heading_tolerance,
            )
        return (
            lane_path_fresh
            and self._odom_is_fresh(now)
            and pose_valid
            and self.lane_path_valid
            and self.lane_path_minimum_line_clearance > 0.0
        )

    def lane_path_diagnostics_callback(self, message):
        with self.lock:
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
            self.lane_path_valid = bool(valid)
            if valid:
                self.lane_path_progress = diagnostics.progress
                self.lane_path_remaining_distance = diagnostics.remaining_distance
                self.lane_path_position_error = diagnostics.position_error
                self.lane_path_cross_track_error = diagnostics.cross_track_error
                self.lane_path_heading_error = diagnostics.heading_error
                self.lane_path_target_speed = diagnostics.target_speed
                self.lane_path_target_index = float(diagnostics.target_index)
                self.lane_path_curvature = diagnostics.curvature
                self.lane_path_minimum_line_clearance = (
                    diagnostics.minimum_line_clearance
                )
                self.lane_path_minimum_obstacle_clearance = (
                    diagnostics.minimum_obstacle_clearance
                )
                self.lane_path_minimum_map_clearance = (
                    diagnostics.minimum_map_clearance
                )
                self.lane_path_commanded_linear = diagnostics.commanded_linear
                self.lane_path_commanded_angular = diagnostics.commanded_angular
                self.last_lane_path_time = now
            if self.state not in (
                self.TURN_TO_ZIGZAG,
                self.VERIFY_ZIGZAG_LANE,
                self.JOIN_ZIGZAG,
            ):
                return
            joining = self.state == self.JOIN_ZIGZAG
            maximum_gap = (
                self.rejoin_confirmation_max_gap
                if joining
                else self.handoff_confirmation_max_gap
            )
            valid = self._lane_observation_valid(now, joining)
            if not valid:
                self.lane_confirmation_count = 0
                self.last_lane_confirmation_time = None
            elif (
                self.last_lane_confirmation_time is None
                or (now - self.last_lane_confirmation_time).to_sec() > maximum_gap
            ):
                self.lane_confirmation_count = 1
                self.last_lane_confirmation_time = now
            else:
                self.lane_confirmation_count += 1
                self.last_lane_confirmation_time = now

    def _record_odom_pose(self, stamp, pose):
        stamp_seconds = float(stamp.to_sec())
        sample = (stamp_seconds, tuple(float(value) for value in pose))
        if (
            self.odom_pose_history
            and stamp_seconds
            < self.odom_pose_history[-1][0] - self.odom_history_duration
        ):
            # Simulation time restarted; samples from the old epoch cannot
            # bracket a scan in the new one.
            self.odom_pose_history.clear()
            self.pending_safety_scans.clear()
        if self.odom_pose_history and abs(
            stamp_seconds - self.odom_pose_history[-1][0]
        ) <= 1e-9:
            self.odom_pose_history[-1] = sample
        else:
            self.odom_pose_history.append(sample)
            if (
                len(self.odom_pose_history) > 1
                and self.odom_pose_history[-2][0] > stamp_seconds
            ):
                self.odom_pose_history = deque(
                    sorted(self.odom_pose_history, key=lambda item: item[0])
                )
        newest_stamp = self.odom_pose_history[-1][0]
        oldest_allowed = newest_stamp - self.odom_history_duration
        while (
            len(self.odom_pose_history) > 1
            and self.odom_pose_history[0][0] < oldest_allowed
        ):
            self.odom_pose_history.popleft()

    def _odom_pose_at_scan_stamp(self, stamp, history):
        """Interpolate odometry at the LaserScan acquisition timestamp."""
        samples = tuple(history)
        if not samples:
            return None
        target = float(stamp.to_sec())
        for sample_stamp, pose in samples:
            if abs(sample_stamp - target) <= 1e-9:
                return pose
        if target < samples[0][0] or target > samples[-1][0]:
            return None
        for (first_stamp, first_pose), (second_stamp, second_pose) in zip(
            samples[:-1], samples[1:]
        ):
            if not (first_stamp < target < second_stamp):
                continue
            if max(
                target - first_stamp, second_stamp - target
            ) > self.scan_odom_sync_tolerance + 1e-9:
                return None
            fraction = (target - first_stamp) / (
                second_stamp - first_stamp
            )
            yaw_delta = normalize_angle(second_pose[2] - first_pose[2])
            return (
                first_pose[0] + fraction * (second_pose[0] - first_pose[0]),
                first_pose[1] + fraction * (second_pose[1] - first_pose[1]),
                normalize_angle(first_pose[2] + fraction * yaw_delta),
            )
        return None

    def _registration_scan_points(self, scan, odom_pose):
        """Return acquisition-ordered, indexed LiDAR points in odom."""
        ranges = np.asarray(scan["ranges"], dtype=np.float64)
        angles = scan["angle_min"] + np.arange(ranges.size) * scan[
            "angle_increment"
        ]
        valid = np.isfinite(ranges)
        finite_ranges = ranges[valid]
        valid[valid] = (
            (finite_ranges >= scan["range_min"])
            & (finite_ranges <= self.registration_scan_maximum_range)
        )
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            return indices, np.empty((0, 2), dtype=np.float64)
        ranges = ranges[indices]
        angles = angles[indices]
        base_x = self.lidar_x + ranges * np.cos(angles)
        base_y = self.lidar_y + ranges * np.sin(angles)
        cosine = math.cos(odom_pose[2])
        sine = math.sin(odom_pose[2])
        points = np.column_stack(
            (
                odom_pose[0] + cosine * base_x - sine * base_y,
                odom_pose[1] + sine * base_x + cosine * base_y,
            )
        )
        return indices, points

    @staticmethod
    def _fit_registration_line(points):
        points = np.asarray(points, dtype=np.float64)
        centre = np.mean(points, axis=0)
        covariance = (points - centre).T @ (points - centre)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        projection = (points - centre) @ direction
        start = centre + float(np.min(projection)) * direction
        end = centre + float(np.max(projection)) * direction
        normal = np.asarray((-direction[1], direction[0]))
        residuals = np.abs((points - centre) @ normal)
        return start, end, residuals

    def _split_registration_cluster(self, points):
        """Split a scan polyline at corners and retain straight fixture faces."""
        minimum = self.registration_minimum_points
        pending = [np.asarray(points, dtype=np.float64)]
        segments = []
        while pending:
            values = pending.pop()
            if len(values) < minimum:
                continue
            chord = values[-1] - values[0]
            chord_length = float(np.linalg.norm(chord))
            if chord_length > 1e-9:
                chord_normal = np.asarray((-chord[1], chord[0])) / chord_length
                chord_residuals = np.abs(
                    (values - values[0]) @ chord_normal
                )
                corner_index = int(np.argmax(chord_residuals))
                if (
                    float(chord_residuals[corner_index])
                    > self.registration_line_residual
                    and corner_index >= minimum - 1
                    and len(values) - corner_index >= minimum
                ):
                    pending.append(values[: corner_index + 1])
                    pending.append(values[corner_index:])
                    continue
            start, end, residuals = self._fit_registration_line(values)
            maximum_index = int(np.argmax(residuals))
            maximum_residual = float(residuals[maximum_index])
            if (
                maximum_residual > self.registration_line_residual
                and minimum <= maximum_index <= len(values) - minimum
            ):
                pending.append(values[: maximum_index + 1])
                pending.append(values[maximum_index:])
                continue
            length = float(np.linalg.norm(end - start))
            if (
                maximum_residual <= self.registration_line_residual
                and self.registration_minimum_segment_length
                <= length
                <= self.registration_maximum_segment_length
            ):
                segments.append((tuple(start), tuple(end), length))
        return segments

    def _extract_registration_segments(self, scan, odom_pose):
        indices, points = self._registration_scan_points(scan, odom_pose)
        if len(points) < self.registration_minimum_points:
            return tuple()
        breaks = np.flatnonzero(
            (np.diff(indices) != 1)
            | (
                np.linalg.norm(np.diff(points, axis=0), axis=1)
                > self.registration_cluster_gap
            )
        )
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks + 1, [len(points)]))
        segments = []
        for start, end in zip(starts, ends):
            if end - start >= self.registration_minimum_points:
                segments.extend(
                    self._split_registration_cluster(points[start:end])
                )
        return tuple(segments)

    @staticmethod
    def _undirected_angle(first, second):
        difference = abs(normalize_angle(first - second))
        return min(difference, abs(math.pi - difference))

    def _registration_pose_is_plausible(self, transform, odom_pose):
        local_pose = transform.inverse().apply_pose(odom_pose)
        minimum_x, maximum_x, minimum_y, maximum_y = (
            self.registration_robot_local_bounds
        )
        if not (
            minimum_x <= local_pose.x <= maximum_x
            and minimum_y <= local_pose.y <= maximum_y
            and abs(normalize_angle(local_pose.yaw - self.approach_heading))
            <= self.registration_robot_heading_tolerance
        ):
            return False
        for rectangle, face in zip(
            self.fixed_obstacle_rectangles, self.registration_visible_faces
        ):
            rect_min_x, rect_max_x, rect_min_y, rect_max_y = rectangle
            if face == "minimum_x" and local_pose.x > rect_min_x:
                return False
            if face == "maximum_x" and local_pose.x < rect_max_x:
                return False
            if face == "minimum_y" and local_pose.y > rect_min_y:
                return False
            if face == "maximum_y" and local_pose.y < rect_max_y:
                return False
        return True

    def _refine_fixture_registration(
        self, result, landmarks, assigned_segments, odom_pose
    ):
        """Stabilize sparse quantized faces with a frozen course heading.

        A 360-ray, 20 mm-resolution scan exposes only a cropped handful of
        returns on each 120 mm fixture face.  AMCL map-to-odom orientation, or
        the initial approach orientation when AMCL is unavailable, supplies an
        independent course-heading observation.  The measured face headings
        retain a configurable share, and the two nonparallel measured line
        offsets then solve translation without treating cropped midpoints as
        surveyed endpoints.
        """
        fitted = result.transform
        approach_yaw = (
            normalize_angle(odom_pose[2] - self.approach_heading)
            if self.registration_heading_prior is None
            else self.registration_heading_prior
        )
        yaw = normalize_angle(
            approach_yaw
            + self.registration_face_heading_weight
            * normalize_angle(
                fitted.target_from_source_yaw - approach_yaw
            )
        )
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        rotation = np.asarray(((cosine, -sine), (sine, cosine)))
        normals = []
        offsets = []
        for landmark, segment in zip(landmarks, assigned_segments):
            local_midpoint = np.asarray(landmark.midpoint, dtype=np.float64)
            observed_midpoint = 0.5 * (
                np.asarray(segment[0], dtype=np.float64)
                + np.asarray(segment[1], dtype=np.float64)
            )
            local_normal = np.asarray(
                (-math.sin(landmark.heading), math.cos(landmark.heading)),
                dtype=np.float64,
            )
            target_normal = rotation @ local_normal
            normals.append(target_normal)
            offsets.append(
                float(
                    target_normal
                    @ (observed_midpoint - rotation @ local_midpoint)
                )
            )
        try:
            translation = np.linalg.solve(
                np.asarray(normals, dtype=np.float64),
                np.asarray(offsets, dtype=np.float64),
            )
        except np.linalg.LinAlgError:
            return result
        refined = RigidTransform2D(
            float(translation[0]),
            float(translation[1]),
            yaw,
            fitted.source_frame,
            fitted.target_frame,
        )
        return replace(result, transform=refined)

    def _estimate_fixture_registration(self, segments, odom_pose, stamp):
        """Fit deterministically named approach-visible fixture faces.

        Both signs are 120 mm long, so trying every pair, name permutation and
        direction leaves symmetric hypotheses in a noisy 360-ray scan.  The
        lane-followed approach supplies only an axis prior: each extracted
        segment is assigned to the nearer surveyed horizontal/vertical face,
        directed consistently, and checked against that finite face's
        longitudinal extent.  The two measured line offsets still determine
        x/y; no global map translation is used.
        """
        if len(segments) < 2:
            return None
        landmarks = self.registration_landmarks
        if len(landmarks) != 2:
            return None
        approach_yaw = (
            normalize_angle(odom_pose[2] - self.approach_heading)
            if self.registration_heading_prior is None
            else self.registration_heading_prior
        )
        expected_headings = tuple(
            normalize_angle(approach_yaw + landmark.heading)
            for landmark in landmarks
        )
        classified = [[], []]
        for segment in segments:
            heading = math.atan2(
                segment[1][1] - segment[0][1],
                segment[1][0] - segment[0][0],
            )
            errors = tuple(
                self._undirected_angle(heading, expected)
                for expected in expected_headings
            )
            index = int(np.argmin(errors))
            other = 1 - index
            if (
                errors[index] > self.registration_heading_class_tolerance
                or errors[other] - errors[index]
                < self.registration_heading_class_margin
            ):
                continue
            start, end = segment[0], segment[1]
            observed_direction = (
                end[0] - start[0],
                end[1] - start[1],
            )
            expected_direction = (
                math.cos(expected_headings[index]),
                math.sin(expected_headings[index]),
            )
            if (
                observed_direction[0] * expected_direction[0]
                + observed_direction[1] * expected_direction[1]
                < 0.0
            ):
                start, end = end, start
                heading = normalize_angle(heading + math.pi)
            classified[index].append(
                (start, end, segment[2], errors[index], heading)
            )
        if not all(classified):
            return None

        candidates = []
        for first in classified[0]:
            for second in classified[1]:
                assigned = (first, second)
                observations = tuple(
                    OrientedSegmentObservation(
                        landmark=landmark.name,
                        start=candidate[0],
                        end=candidate[1],
                    )
                    for landmark, candidate in zip(landmarks, assigned)
                )
                observation = RegistrationObservation(
                    stamp=float(stamp.to_sec()),
                    target_frame="odom",
                    segments=observations,
                )
                result = estimate_local_registration(
                    self.registration_template,
                    observation,
                    self.registration_config,
                )
                if not result.accepted:
                    continue
                result = self._refine_fixture_registration(
                    result, landmarks, assigned, odom_pose
                )
                if not self._registration_pose_is_plausible(
                    result.transform, odom_pose
                ):
                    continue

                longitudinal_error = 0.0
                finite_faces_match = True
                for landmark, candidate in zip(landmarks, assigned):
                    expected_midpoint = result.transform.apply_point(
                        landmark.midpoint
                    )
                    heading = normalize_angle(
                        result.transform.target_from_source_yaw
                        + landmark.heading
                    )
                    tangent = (math.cos(heading), math.sin(heading))
                    endpoint_offsets = tuple(
                        tangent[0] * (endpoint[0] - expected_midpoint[0])
                        + tangent[1] * (endpoint[1] - expected_midpoint[1])
                        for endpoint in candidate[:2]
                    )
                    half_length = 0.5 * math.hypot(
                        landmark.end[0] - landmark.start[0],
                        landmark.end[1] - landmark.start[1],
                    )
                    excess = max(
                        0.0,
                        max(abs(value) for value in endpoint_offsets)
                        - half_length,
                    )
                    if excess > self.registration_longitudinal_tolerance:
                        finite_faces_match = False
                        break
                    longitudinal_error += excess
                if not finite_faces_match:
                    continue

                local_pose = result.transform.inverse().apply_pose(odom_pose)
                length_error = sum(
                    abs(
                        candidate[2]
                        - math.hypot(
                            landmark.end[0] - landmark.start[0],
                            landmark.end[1] - landmark.start[1],
                        )
                    )
                    for landmark, candidate in zip(landmarks, assigned)
                )
                heading_class_error = first[3] + second[3]
                score = (
                    result.diagnostics.robust_rms
                    + 2.0 * length_error
                    + longitudinal_error
                    + 0.02 * heading_class_error
                    + 0.05
                    * abs(
                        normalize_angle(
                            local_pose.yaw - self.approach_heading
                        )
                    )
                )
                candidates.append((score, result))
        if not candidates:
            return None
        candidates.sort(key=lambda value: value[0])
        best_score, best = candidates[0]
        for alternative_score, alternative in candidates[1:]:
            if alternative_score - best_score > self.registration_ambiguity_score:
                break
            first = best.transform
            second = alternative.transform
            if (
                math.hypot(
                    first.target_from_source_x
                    - second.target_from_source_x,
                    first.target_from_source_y
                    - second.target_from_source_y,
                )
                > self.registration_ambiguity_position
                or abs(
                    normalize_angle(
                        first.target_from_source_yaw
                        - second.target_from_source_yaw
                    )
                )
                > self.registration_ambiguity_heading
            ):
                return None
        return best

    def _process_registration_scan(self, scan, odom_pose):
        """Consume one synchronized scan while the caller owns ``self.lock``."""
        if (
            self.arm_seq is None
            or self.arm_stamp is None
            or self.local_to_odom is not None
            or self.state != self.WAIT_GATE
        ):
            return
        source_stamp = scan["source_stamp"]
        now = rospy.Time.now()
        if source_stamp < self.arm_stamp:
            self.registration_rejection_reason = "scan predates arm"
            return
        source_age = (now - source_stamp).to_sec()
        if not (
            -self.registration_future_tolerance
            <= source_age
            <= self.registration_source_max_age
        ):
            self.registration_rejection_reason = "stale registration scan"
            return
        if self.registration_heading_prior is None:
            map_orientation_is_fresh = bool(
                self.map_pose is not None
                and self.map_pose_stamp is not None
                and abs((source_stamp - self.map_pose_stamp).to_sec())
                <= self.pose_timeout
            )
            if map_orientation_is_fresh:
                self.registration_heading_prior = normalize_angle(
                    odom_pose[2] - self.map_pose[2]
                )
                heading_source = "AMCL map-to-odom"
            else:
                self.registration_heading_prior = normalize_angle(
                    odom_pose[2] - self.approach_heading
                )
                heading_source = "initial approach"
            rospy.loginfo(
                "Parking registration course heading fixed from %s: %.2fdeg",
                heading_source,
                math.degrees(self.registration_heading_prior),
            )
        segments = self._extract_registration_segments(scan, odom_pose)
        result = self._estimate_fixture_registration(
            segments, odom_pose, source_stamp
        )
        if result is None:
            self.registration_rejection_reason = (
                "two nonparallel fixture faces were not observable"
            )
            rospy.logwarn_throttle(
                1.0,
                "Parking registration pending: extracted_segments=%d "
                "pose=(%.3f, %.3f, %.1fdeg)",
                len(segments),
                odom_pose[0],
                odom_pose[1],
                math.degrees(odom_pose[2]),
            )
            # A sparse 360-ray scan can miss one of the two 120 mm faces for
            # a frame even though the accepted transforms immediately before
            # and after it are independent and consistent.  Preserve that
            # streak only inside the configured source-time gap; the temporal
            # filter still clears it when the next accepted result is late.
            temporal = self.registration_filter.state()
            if (
                temporal.confirmation_count > 0
                and float(source_stamp.to_sec()) - temporal.last_stamp
                > self.registration_filter.config.maximum_gap
            ):
                temporal = self.registration_filter.reset(
                    "confirmation gap"
                )
            self.registration_confirmation_count = (
                temporal.confirmation_count
            )
            return
        temporal = self.registration_filter.update(result)
        self.registration_confirmation_count = temporal.confirmation_count
        self.registration_rejection_reason = temporal.reason
        if not temporal.confirmed:
            rospy.loginfo(
                "Parking registration observation %d/%d: "
                "local->odom=(%.4f, %.4f, %.2fdeg)%s",
                temporal.confirmation_count,
                self.registration_filter.config.required_confirmations,
                result.transform.target_from_source_x,
                result.transform.target_from_source_y,
                math.degrees(result.transform.target_from_source_yaw),
                " reason=" + temporal.reason if temporal.reason else "",
            )
            return
        self.local_to_odom = temporal.transform
        self.registration_covariance = temporal.covariance
        self.registration_source_stamp = rospy.Time.from_sec(
            temporal.last_stamp
        )
        rospy.loginfo(
            "Parking local registration frozen after %d scans: "
            "local->odom=(%.4f, %.4f, %.2fdeg)",
            temporal.confirmation_count,
            self.local_to_odom.target_from_source_x,
            self.local_to_odom.target_from_source_y,
            math.degrees(self.local_to_odom.target_from_source_yaw),
        )

    def _stage_registration_scan(self, scan, odom_pose):
        """Keep the newest synchronized registration input without blocking."""
        with self.safety_data_lock:
            pending = self.pending_registration_scan
            if (
                pending is None
                or scan["source_stamp"] >= pending[0]["source_stamp"]
            ):
                self.pending_registration_scan = (scan, odom_pose)

    def _stage_selection_scan(self, scan, odom_pose):
        """Keep the newest source-synchronized bay-selection input."""
        with self.safety_data_lock:
            pending = self.pending_selection_scan
            if (
                pending is None
                or scan["source_stamp"] >= pending[0]["source_stamp"]
            ):
                self.pending_selection_scan = (scan, odom_pose)

    def _apply_pending_registration_scan(self):
        """Drain one newest-only registration slot under ``self.lock``."""
        with self.safety_data_lock:
            pending = self.pending_registration_scan
            self.pending_registration_scan = None
        if pending is not None:
            self._process_registration_scan(*pending)

    def _pending_scan_expired(self, scan, now):
        return (
            (now - scan["source_stamp"]).to_sec() > self.scan_timeout
            or (now - scan["received"]).to_sec() > self.scan_timeout
        )

    def _enqueue_pending_safety_scan(self, scan, now):
        while self.pending_safety_scans and self._pending_scan_expired(
            self.pending_safety_scans[0], now
        ):
            self.pending_safety_scans.popleft()
        if len(self.pending_safety_scans) >= self.pending_safety_scan_limit:
            self.pending_safety_scans.popleft()
            rospy.logwarn_throttle(
                1.0,
                "Parking dropped the oldest pending LaserScan before odom "
                "synchronization",
            )
        self.pending_safety_scans.append(scan)

    def _take_ready_safety_scans(self, now):
        """Remove ready scans while retaining only bounded future scans."""
        history = tuple(self.odom_pose_history)
        latest_stamp = history[-1][0] if history else -math.inf
        ready = []
        waiting = deque()
        for scan in self.pending_safety_scans:
            if self._pending_scan_expired(scan, now):
                continue
            pose = self._odom_pose_at_scan_stamp(
                scan["source_stamp"], history
            )
            if pose is not None:
                ready.append((scan, pose))
                continue
            scan_stamp = float(scan["source_stamp"].to_sec())
            if scan_stamp > latest_stamp:
                waiting.append(scan)
                continue
            rospy.logwarn_throttle(
                1.0,
                "Parking discarded a LaserScan whose odom bracket exceeded "
                "the synchronization tolerance",
            )
        self.pending_safety_scans = waiting
        return ready

    def _project_safety_scan(self, scan, odom_pose):
        safety_points = scan_points_in_map(
            scan["ranges"],
            scan["angle_min"],
            scan["angle_increment"],
            scan["range_min"],
            scan["maximum_range"],
            odom_pose,
            self.lidar_x,
            self.lidar_y,
        )
        with self.safety_data_lock:
            if (
                self.safety_scan_stamp is not None
                and scan["source_stamp"].to_sec()
                < self.safety_scan_stamp.to_sec() - 1e-9
            ):
                return
            self.live_obstacle_points_odom = safety_points
            self.safety_scan_stamp = scan["source_stamp"]
            self.safety_scan_received = scan["received"]

    def _apply_odom_state(self, state):
        """Apply one complete odom snapshot while holding ``self.lock``."""
        (
            arrival_generation,
            linear_speed,
            angular_velocity,
            odom_pose,
            source_stamp,
            received,
        ) = state
        self.odom_linear_speed = linear_speed
        self.odom_angular_velocity = angular_velocity
        self.odom_pose = odom_pose
        self.odom_stamp = source_stamp
        self.odom_received = received
        self.odom_generation += 1
        self.applied_odom_arrival_generation = arrival_generation

    def _apply_pending_odom_state(self):
        """Move the newest deferred odom snapshot into controller state."""
        with self.safety_data_lock:
            state = self.pending_odom_state
            self.pending_odom_state = None
        if (
            state is not None
            and state[0] > self.applied_odom_arrival_generation
        ):
            self._apply_odom_state(state)

    def odom_callback(self, message):
        received = rospy.Time.now()
        source_stamp = (
            message.header.stamp
            if message.header.stamp != rospy.Time()
            else received
        )
        linear_x = float(message.twist.twist.linear.x)
        linear_y = float(message.twist.twist.linear.y)
        angular_z = float(message.twist.twist.angular.z)
        odom_pose = (
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
            yaw_from_quaternion(message.pose.pose.orientation),
        )
        odom_state = (
            (
                math.hypot(linear_x, linear_y)
                if math.isfinite(linear_x) and math.isfinite(linear_y)
                else 0.0
            ),
            angular_z if math.isfinite(angular_z) else 0.0,
            odom_pose,
            source_stamp,
            received,
        )
        # Stage the controller snapshot and record the safety history in one
        # ordered critical section before doing scan projection.  The gate
        # callback uses this same lock ordering, so an odom callback is
        # unambiguously before or after the gate edge even if projection is
        # still in flight.
        with self.safety_data_lock:
            self.odom_arrival_generation += 1
            self.pending_odom_state = (
                self.odom_arrival_generation,
            ) + odom_state
            self._record_odom_pose(source_stamp, odom_pose)
            ready_safety_scans = self._take_ready_safety_scans(received)
        for scan, scan_pose in ready_safety_scans:
            self._project_safety_scan(scan, scan_pose)
            self._stage_registration_scan(scan, scan_pose)
            self._stage_selection_scan(scan, scan_pose)

        if not self.lock.acquire(False):
            return
        occupancy = None
        try:
            self._apply_pending_odom_state()
            self._apply_pending_registration_scan()
            occupancy = self._apply_pending_selection_scan()
        finally:
            self.lock.release()
        if occupancy is not None:
            self.occupancy_pub.publish(occupancy)

    @staticmethod
    def _median_ranges(message, window):
        ranges = np.asarray(message.ranges, dtype=np.float64)
        if window <= 1 or ranges.size == 0:
            return ranges
        minimum_range = max(0.0, float(message.range_min))
        maximum_range = float(message.range_max)
        valid = np.isfinite(ranges) & (ranges >= minimum_range)
        valid &= ranges <= maximum_range
        neighbors = []
        valid_neighbors = []
        half_window = window // 2
        for offset in range(-half_window, half_window + 1):
            shifted = np.roll(ranges, offset).copy()
            shifted_valid = np.roll(valid, offset)
            shifted[~shifted_valid] = np.nan
            neighbors.append(shifted)
            valid_neighbors.append(shifted_valid)
        stack = np.vstack(neighbors)
        filtered = np.ma.median(
            np.ma.masked_invalid(stack), axis=0
        ).filled(np.nan)
        enough = np.count_nonzero(np.vstack(valid_neighbors), axis=0) >= (
            half_window + 1
        )
        filtered[~enough] = np.nan
        return filtered

    def scan_callback(self, message):
        received = rospy.Time.now()
        source_stamp = (
            message.header.stamp
            if message.header.stamp != rospy.Time()
            else received
        )
        ranges = self._median_ranges(message, self.scan_median_window)
        maximum_range = min(float(message.range_max), self.scan_maximum_range)
        safety_scan = {
            "source_stamp": source_stamp,
            "received": received,
            "ranges": ranges.copy(),
            "angle_min": float(message.angle_min),
            "angle_increment": float(message.angle_increment),
            "range_min": float(message.range_min),
            "maximum_range": maximum_range,
        }
        with self.safety_data_lock:
            # Enqueue first, then re-check the latest history under the same
            # lock. This closes the scan/odom callback ordering race in both
            # directions without ever substituting the newest pose.
            self._enqueue_pending_safety_scan(safety_scan, received)
            ready_safety_scans = self._take_ready_safety_scans(received)
        for ready_scan, scan_pose in ready_safety_scans:
            self._project_safety_scan(ready_scan, scan_pose)
            self._stage_registration_scan(ready_scan, scan_pose)
            self._stage_selection_scan(ready_scan, scan_pose)

        # Bay occupancy is used only while SELECT_SPACE is stopped. Never let
        # this subscriber worker queue on the main lock; a control tick will
        # drain the same newest-only slot if this immediate attempt loses.
        if not self.lock.acquire(False):
            return
        try:
            self._apply_pending_registration_scan()
            occupancy = self._apply_pending_selection_scan()
        finally:
            self.lock.release()
        if occupancy is not None:
            self.occupancy_pub.publish(occupancy)

    def _apply_pending_selection_scan(self):
        """Consume the newest bay scan; caller must hold ``self.lock``."""
        with self.safety_data_lock:
            pending = self.pending_selection_scan
            self.pending_selection_scan = None
        if pending is None:
            return None
        scan, synchronized_odom_pose = pending
        if self.state != self.SELECT_SPACE:
            return None
        # A scan acquired or delivered before the stopped selection epoch must
        # never count as one of its independent confirmation samples.
        if (
            (scan["source_stamp"] - self.state_started).to_sec() < -1e-9
            or (scan["received"] - self.state_started).to_sec() < -1e-9
        ):
            return None
        received = scan["received"]
        local_to_odom = self._frozen_local_to_odom()
        if synchronized_odom_pose is None or local_to_odom is None:
            return None
        points_odom = scan_points_in_map(
            scan["ranges"],
            scan["angle_min"],
            scan["angle_increment"],
            scan["range_min"],
            scan["maximum_range"],
            synchronized_odom_pose,
            self.lidar_x,
            self.lidar_y,
        )
        odom_to_local = local_to_odom.inverse()
        points = np.asarray(
            [odom_to_local.apply_point(point) for point in points_odom],
            dtype=np.float64,
        ).reshape((-1, 2))
        left_points = points_in_box(points, self.left_box)
        right_points = points_in_box(points, self.right_box)
        self.left_points = left_points
        self.right_points = right_points
        self.scan_stamp = scan["source_stamp"]
        self.scan_received = received
        self.scan_generation += 1
        return Int32MultiArray(data=[left_points, right_points])

    def manual_stop_callback(self, message):
        with self.lock:
            previous = self.manual_stop
            self.manual_stop = bool(message.data)
            if self.manual_stop and self.mission_has_control:
                self._publish_stop()
            elif previous and not self.manual_stop:
                # A deliberate operator pause must not consume a phase timeout
                # or reuse scans gathered while motion was inhibited.
                self.state_started = rospy.Time.now()
                self.last_selection_scan_generation = self.scan_generation
                self.lane_confirmation_count = 0
                self.last_lane_confirmation_time = None

    def command_observer_callback(self, message):
        """Remember the lane command that must continue through handoff."""
        with self.lock:
            self._apply_pending_odom_state()
            if (
                not self.mission_has_control
                and math.isfinite(message.linear.x)
                and math.isfinite(message.angular.z)
            ):
                self.observed_lane_linear = float(message.linear.x)
                self.observed_lane_angular = float(message.angular.z)
                self.observed_lane_command_received = rospy.Time.now()
                self.lane_command_generation += 1
                handoff_speed_limit = min(
                    self.lane_entry_speed_limit, self.approach_speed
                )
                command_is_bounded = (
                    self.observed_lane_linear > 1e-4
                    and self.observed_lane_linear
                    <= handoff_speed_limit + 1e-9
                    and abs(self.observed_lane_angular)
                    <= self.maximum_angular_velocity + 1e-9
                )
                if (
                    self.zone_gate
                    and self.lane_command_generation
                    > self.gate_lane_command_generation
                    and command_is_bounded
                ):
                    if self.bounded_lane_command_generation is None:
                        # Require a subsequently applied odometry sample before
                        # projecting the moving base onto the prepared path.
                        self.bounded_lane_command_generation = (
                            self.lane_command_generation
                        )
                        self.bounded_lane_command_odom_generation = (
                            self.odom_generation
                        )
                elif not command_is_bounded:
                    self.bounded_lane_command_generation = None
                    self.bounded_lane_command_odom_generation = None

    def _map_pose_is_fresh(self, now):
        if (
            self.map_pose is None
            or self.map_pose_stamp is None
            or self.map_pose_received is None
        ):
            return False
        source_age = (now - self.map_pose_stamp).to_sec()
        receipt_age = (now - self.map_pose_received).to_sec()
        return (
            -self.scan_odom_sync_tolerance <= source_age <= self.pose_timeout
            and -self.scan_odom_sync_tolerance
            <= receipt_age
            <= self.pose_timeout
        )

    def _odom_is_fresh(self, now):
        if (
            self.odom_pose is None
            or self.odom_stamp is None
            or self.odom_received is None
        ):
            return False
        source_age = (now - self.odom_stamp).to_sec()
        receipt_age = (now - self.odom_received).to_sec()
        return (
            -self.scan_odom_sync_tolerance <= source_age <= self.odom_timeout
            and -self.scan_odom_sync_tolerance
            <= receipt_age
            <= self.odom_timeout
        )

    def _inputs_fresh(self, now):
        return self.local_to_odom is not None and self._odom_is_fresh(now)

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
            # A transport failure does not prove whether the service callback
            # applied the ownership change.  Keep that uncertainty explicit so
            # the failure path never becomes a second cmd_vel publisher.
            self.handoff_ambiguous = True
            rospy.logerr("Parking cmd_vel handoff failed: %s", error)
            return False

    def _stop_lane_controller(self):
        try:
            rospy.wait_for_service(self.lane_stop_service_name, timeout=1.0)
            response = self.lane_stop_service(False)
            if not response.success:
                raise rospy.ServiceException(response.message)
            return True
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logerr("Parking lane emergency stop failed: %s", error)
            return False

    def _frozen_local_to_odom(self):
        """Return the sole frozen parking-local registration."""
        return self.local_to_odom

    def _route_pose_to_odom(self, pose):
        """Project surveyed route coordinates into the local control frame."""
        local_to_odom = self._frozen_local_to_odom()
        if local_to_odom is None:
            raise ValueError("parking local->odom registration is unavailable")
        transformed = local_to_odom.apply_pose(pose)
        return transformed.x, transformed.y, transformed.yaw

    def _odom_pose_to_route(self, pose):
        """Project a local measured pose into surveyed route coordinates."""
        local_to_odom = self._frozen_local_to_odom()
        if local_to_odom is None:
            raise ValueError("parking local->odom registration is unavailable")
        transformed = local_to_odom.inverse().apply_pose(pose)
        return transformed.x, transformed.y, transformed.yaw

    def _latch_route_goal(self):
        if self.goal_map is None or self._frozen_local_to_odom() is None:
            return False
        self.goal_odom = self._route_pose_to_odom(self.goal_map)
        return True

    @staticmethod
    def _twist_from_velocity(command):
        message = Twist()
        message.linear.x = float(command.linear_velocity)
        message.angular.z = float(command.angular_velocity)
        return message

    def _route_pose_for_safety(self, pose):
        pose = Pose2D.from_value(pose)
        values = (pose.x, pose.y, pose.yaw)
        if self._frozen_local_to_odom() is None:
            return values
        return self._odom_pose_to_route(values)

    @staticmethod
    def _corners_for_footprint(pose, footprint):
        return rectangle_corners(
            pose,
            footprint.front,
            footprint.rear,
            footprint.half_width,
        )

    def _cached_perimeter_points(self, pose, footprint):
        """Transform one cached dense local perimeter without reallocating it."""
        spacing = min(0.001, self.swept_translation_step)
        footprint_key = (
            round(float(footprint.front), 12),
            round(float(footprint.rear), 12),
            round(float(footprint.half_width), 12),
            round(float(spacing), 12),
        )
        local = self._footprint_perimeter_cache.get(footprint_key)
        if local is None:
            local = footprint_points(
                Pose2D(0.0, 0.0, 0.0),
                footprint,
                spacing=spacing,
                perimeter_only=True,
            )
            self._footprint_perimeter_cache[footprint_key] = local
        pose = Pose2D.from_value(pose)
        cosine = math.cos(pose.yaw)
        sine = math.sin(pose.yaw)
        points = np.empty_like(local)
        points[:, 0] = pose.x + cosine * local[:, 0] - sine * local[:, 1]
        points[:, 1] = pose.y + sine * local[:, 0] + cosine * local[:, 1]
        return points

    @staticmethod
    def _boundary_cache_key(kind, pose, footprint, extra=0.0):
        pose = Pose2D.from_value(pose)
        return (
            str(kind),
            round(pose.x, 12),
            round(pose.y, 12),
            round(pose.yaw, 12),
            round(float(footprint.front), 12),
            round(float(footprint.rear), 12),
            round(float(footprint.half_width), 12),
            round(float(extra), 12),
        )

    def _cached_boundary_clearance(self, key):
        if key not in self._boundary_clearance_cache:
            return None
        value = self._boundary_clearance_cache.pop(key)
        self._boundary_clearance_cache[key] = value
        return value

    def _store_boundary_clearance(self, key, value):
        self._boundary_clearance_cache[key] = float(value)
        self._boundary_clearance_cache.move_to_end(key)
        while (
            len(self._boundary_clearance_cache)
            > self._boundary_clearance_cache_limit
        ):
            self._boundary_clearance_cache.popitem(last=False)
        return float(value)

    def _entry_line_clearance(self, pose, footprint):
        corners = self._corners_for_footprint(
            self._route_pose_for_safety(pose), footprint
        )
        return min(
            min(
                y
                - (
                    self.entry_lower_intercept
                    + self.entry_boundary_slope * x
                )
                for x, y in corners
            ),
            min(
                self.entry_upper_intercept
                + self.entry_boundary_slope * x
                - y
                for x, y in corners
            ),
        )

    def _aisle_line_clearance(self, pose, footprint):
        corners = self._corners_for_footprint(
            self._route_pose_for_safety(pose), footprint
        )
        return min(
            min(x for x, _ in corners) - self.aisle_right_edge,
            self.aisle_left_edge - max(x for x, _ in corners),
        )

    def _parking_union_clearance(self, pose, footprint):
        route_pose = Pose2D.from_value(self._route_pose_for_safety(pose))
        cache_key = self._boundary_cache_key(
            "parking", route_pose, footprint
        )
        cached = self._cached_boundary_clearance(cache_key)
        if cached is not None:
            return cached
        points = self._cached_perimeter_points(route_pose, footprint)
        aisle = np.minimum(
            points[:, 0] - self.aisle_right_edge,
            self.aisle_left_edge - points[:, 0],
        )
        parking = np.minimum.reduce(
            (
                points[:, 0] - self.right_outer_edge,
                self.left_outer_edge - points[:, 0],
                points[:, 1] - self.lower_outer_edge,
                self.upper_outer_edge - points[:, 1],
            )
        )
        return self._store_boundary_clearance(
            cache_key, np.min(np.maximum(aisle, parking))
        )

    def _parking_exit_union_clearance(self, pose, footprint):
        """Clear the measured finite solid arms at the dotted aisle opening."""
        route_pose = Pose2D.from_value(self._route_pose_for_safety(pose))
        cache_key = self._boundary_cache_key(
            "parking_exit", route_pose, footprint
        )
        cached = self._cached_boundary_clearance(cache_key)
        if cached is not None:
            return cached
        points = self._cached_perimeter_points(route_pose, footprint)
        opening = np.minimum(
            points[:, 0] - self.parking_opening_right_edge,
            self.parking_opening_left_edge - points[:, 0],
        )
        parking = np.minimum.reduce(
            (
                points[:, 0] - self.right_outer_edge,
                self.left_outer_edge - points[:, 0],
                points[:, 1] - self.lower_outer_edge,
                self.parking_opening_upper_edge - points[:, 1],
            )
        )
        return self._store_boundary_clearance(
            cache_key, np.min(np.maximum(opening, parking))
        )

    def _entry_aisle_union_clearance(self, pose, footprint, corner_relief):
        route_pose = Pose2D.from_value(self._route_pose_for_safety(pose))
        cache_key = self._boundary_cache_key(
            "entry_aisle", route_pose, footprint, extra=corner_relief
        )
        cached = self._cached_boundary_clearance(cache_key)
        if cached is not None:
            return cached
        points = self._cached_perimeter_points(route_pose, footprint)
        entry = np.minimum(
            points[:, 1]
            - (
                self.entry_lower_intercept
                + self.entry_boundary_slope * points[:, 0]
            ),
            self.entry_upper_intercept
            + self.entry_boundary_slope * points[:, 0]
            - points[:, 1],
        )
        aisle = np.minimum(
            points[:, 0] - self.aisle_right_edge,
            self.aisle_left_edge - points[:, 0],
        )
        # The surveyed horizontal solid line terminates before the aisle edge.
        # This measured relief represents that finite paint-free corner, rather
        # than treating both line equations as infinite through the junction.
        union = np.maximum(entry, aisle)
        near_finite_corner = (entry < corner_relief) & (
            aisle < corner_relief
        )
        union[near_finite_corner] += float(corner_relief)
        return self._store_boundary_clearance(cache_key, np.min(union))

    def _entry_turn_union_clearance(self, pose, footprint):
        return self._entry_aisle_union_clearance(
            pose, footprint, self.entry_turn_corner_relief
        )

    def _final_turn_union_clearance(self, pose, footprint):
        return self._entry_aisle_union_clearance(
            pose, footprint, self.exit_turn_corner_relief
        )

    def _local_boundary_clearance(self, pose, footprint):
        """Clearance to mission-local bounds through the frozen transform."""
        corners = self._corners_for_footprint(
            self._route_pose_for_safety(pose), footprint
        )
        minimum_x, maximum_x, minimum_y, maximum_y = self.local_safety_bounds
        return min(
            min(x for x, _ in corners) - minimum_x,
            maximum_x - max(x for x, _ in corners),
            min(y for _, y in corners) - minimum_y,
            maximum_y - max(y for _, y in corners),
        )

    def _fixed_obstacle_points_odom(self):
        """Freeze surveyed fixed collision points in the executable frame."""
        points = self.fixed_obstacle_points_route
        if points.size == 0:
            return points.copy()
        local_to_odom = self._frozen_local_to_odom()
        if local_to_odom is None:
            return points.copy()
        return np.asarray(
            [local_to_odom.apply_point(point) for point in points],
            dtype=np.float64,
        )

    def _path_safety(
        self,
        boundary_kind,
        total_line_margin=None,
    ):
        callbacks = {
            "entry": self._entry_line_clearance,
            "entry_turn": self._entry_turn_union_clearance,
            "aisle": self._aisle_line_clearance,
            "parking": self._parking_union_clearance,
            "parking_exit": self._parking_exit_union_clearance,
            "final_turn": self._final_turn_union_clearance,
        }
        callback = callbacks.get(boundary_kind)
        # The surveyed route, paint callbacks, fixed fixtures and local bounds
        # all receive the same frozen local->odom transform.  Registration
        # covariance is common-mode for that fixed sweep and must not be added
        # again as relative clearance.  Keep it as registration diagnostics;
        # configured odom/tracking margins and source-stamped live LiDAR own
        # the independent errors while the robot is moving.
        localization_margin = self.localization_margin
        uncertainty = localization_margin + self.tracking_margin
        if total_line_margin is None:
            total_line_margin = self.line_margin + uncertainty
        line_margin = max(0.0, float(total_line_margin) - uncertainty)
        return PathSafety(
            line_boundaries=(CallbackBoundary(callback),) if callback else (),
            map_boundaries=(CallbackBoundary(self._local_boundary_clearance),),
            fixed_obstacles=self._fixed_obstacle_points_odom(),
            margins=SafetyMargins(
                line=line_margin,
                obstacle=self.obstacle_margin,
                localization=localization_margin,
                tracking=self.tracking_margin,
            ),
        )

    def _rotation_path_safety(self):
        """Keep line uncertainty but check obstacles against the real body."""
        safety = self._path_safety("parking")
        uncertainty = safety.margins.uncertainty
        return PathSafety(
            line_boundaries=safety.line_boundaries,
            map_boundaries=safety.map_boundaries,
            fixed_obstacles=safety.fixed_obstacles,
            margins=SafetyMargins(
                # Moving uncertainty remains required at painted/map edges.
                # It is folded into the line margin before zeroing the common
                # uncertainty fields so it is not also forced onto obstacles.
                line=safety.margins.line + uncertainty,
                obstacle=self.rotation_obstacle_margin,
                localization=0.0,
                tracking=0.0,
            ),
        )

    def _speed_profile(self, cruise, entry=None, exit_velocity=None):
        cruise = min(self.cruise_velocity, max(0.001, float(cruise)))
        minimum = min(cruise, self.minimum_velocity)
        if entry is None:
            entry = min(cruise, self.profile_entry_velocity)
        if exit_velocity is None:
            exit_velocity = min(cruise, self.profile_exit_velocity)
        return SpeedProfile(
            cruise_velocity=cruise,
            minimum_velocity=minimum,
            entry_velocity=clamp(float(entry), minimum, cruise),
            exit_velocity=clamp(float(exit_velocity), minimum, cruise),
            maximum_angular_velocity=self.maximum_angular_velocity,
            maximum_lateral_acceleration=self.maximum_lateral_acceleration,
            linear_acceleration=self.linear_acceleration,
            linear_deceleration=self.linear_deceleration,
            angular_acceleration=self.angular_acceleration,
        )

    def _common_path_from_poses(
        self,
        poses,
        direction,
        speed,
        label,
        boundary_kind,
        position_tolerance,
        heading_tolerance,
        terminal_crossing,
        entry_velocity=None,
        exit_velocity=None,
        total_line_margin=None,
        feedforward_scale=1.0,
        fail_on_error=True,
    ):
        values = np.asarray(poses, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != 3
            or values.shape[0] < 2
            or not np.all(np.isfinite(values))
        ):
            if fail_on_error:
                self._fail("%s did not produce a finite pose path" % label)
            return None
        feedforward_scale = float(feedforward_scale)
        profile = self._speed_profile(
            speed, entry=entry_velocity, exit_velocity=exit_velocity
        )
        try:
            path = path_from_poses(
                values,
                "odom",
                speed_profile=profile,
                direction=int(1 if direction >= 0 else -1),
                feedforward_scale=feedforward_scale,
                goal_tolerance=GoalTolerance(
                    position=float(position_tolerance),
                    heading=float(heading_tolerance),
                    terminal_crossing=max(
                        float(position_tolerance), float(terminal_crossing)
                    ),
                ),
                safety=self._path_safety(
                    boundary_kind,
                    total_line_margin=total_line_margin,
                ),
                label=str(label),
            )
        except ValueError as error:
            if fail_on_error:
                self._fail("cannot build common %s path: %s" % (label, error))
            return None

        validation = self.path_validator.validate_path(path)
        path.line_clearance = validation.minimum_line_clearance
        path.obstacle_clearance = validation.minimum_obstacle_clearance
        path.map_clearance = validation.minimum_map_clearance
        if not validation.safe:
            if fail_on_error:
                self._fail(
                    "%s common swept path violates a fixed boundary "
                    "(line=%.4fm local=%.4fm)"
                    % (
                        label,
                        validation.minimum_line_clearance,
                        validation.minimum_map_clearance,
                    )
                )
            return None
        return path

    def _activate_common_path(
        self,
        path,
        state,
        initial_linear=None,
        initial_angular=None,
    ):
        if path is None or self.odom_pose is None:
            return False
        if initial_linear is None:
            initial_linear = self.last_linear
        if initial_angular is None:
            initial_angular = self.last_angular
        self.path_follower.reset(
            path,
            self.odom_pose,
            initial_linear=initial_linear,
            initial_angular=initial_angular,
        )
        self.active_path = path
        self.active_path_state = state
        # Every parking segment is validated once when it is converted to a
        # CommonPath.  Its geometry and embedded fixed safety evidence are
        # immutable for the segment lifetime, so activating the same object
        # must not repeat the full swept-footprint calculation.  Live LiDAR
        # and the reaction/braking sweep are still evaluated every control
        # tick by ``_common_safety_speed_limit`` below.
        self.active_path_validation = ValidationResult(
            True,
            minimum_line_clearance=path.line_clearance,
            minimum_obstacle_clearance=path.obstacle_clearance,
            minimum_map_clearance=path.map_clearance,
        )
        self.path_follower.update_clearance(self.active_path_validation)
        self.last_linear = float(initial_linear)
        self.last_angular = float(initial_angular)
        return True

    def _straight_path(self, goal_odom, direction, speed, state):
        current = Pose2D.from_value(self.odom_pose)
        goal = Pose2D.from_value(goal_odom)
        motion_heading = normalize_angle(
            goal.yaw + (math.pi if direction < 0 else 0.0)
        )
        tangent_x = math.cos(motion_heading)
        tangent_y = math.sin(motion_heading)
        along = tangent_x * (goal.x - current.x) + tangent_y * (
            goal.y - current.y
        )
        if along < -self.overshoot_tolerance:
            self._fail("passed the fixed anchored parking goal")
            return None
        length = max(along, self._state_position_tolerance(state), 1e-4)
        start_x = goal.x - length * tangent_x
        start_y = goal.y - length * tangent_y
        count = max(2, int(math.ceil(length / self.path_sample_spacing)) + 1)
        fraction = np.linspace(0.0, 1.0, count)
        poses = np.column_stack(
            (
                start_x + fraction * (goal.x - start_x),
                start_y + fraction * (goal.y - start_y),
                np.full(count, goal.yaw),
            )
        )
        boundary_kind = "parking" if state in (self.PARK_IN, self.BACK_OUT) else "aisle"
        return self._common_path_from_poses(
            poses,
            direction,
            speed,
            state,
            boundary_kind,
            self._state_position_tolerance(state),
            self._state_heading_tolerance(state),
            self.overshoot_tolerance,
        )

    def _live_safety_snapshot(self, now):
        """Return one coherent freshness/obstacle snapshot for a control tick."""
        with self.safety_data_lock:
            fresh = (
                self.safety_scan_stamp is not None
                and self.safety_scan_received is not None
                and (now - self.safety_scan_stamp).to_sec() <= self.scan_timeout
                and (now - self.safety_scan_received).to_sec() <= self.scan_timeout
            )
            obstacles = (
                self.live_obstacle_points_odom
                if fresh
                else np.empty((0, 2), dtype=np.float64)
            )
            return fresh, obstacles

    def _common_safety_speed_limit(self, now, tracking=None):
        if self.active_path is None or self.odom_pose is None:
            return math.inf, False
        scan_is_fresh, live_obstacles = self._live_safety_snapshot(now)
        if self.require_live_safety_scan and not scan_is_fresh:
            rospy.logwarn_throttle(
                1.0, "Parking holds zero for a stale swept-path safety scan"
            )
            return 0.0, True

        pose = Pose2D.from_value(self.odom_pose)
        if tracking is None:
            tracking = self.path_follower.calculate_tracking(pose)
        stopping_linear = tracking.direction * max(
            abs(self.path_follower.last_linear), self.odom_linear_speed
        )
        decision = self.path_validator.motion_safety(
            self.active_path,
            pose,
            tracking.path_index,
            tracking.target_speed,
            stopping_linear,
            self.path_follower.stopping_angular_velocities(
                tracking,
                stopping_linear,
                self.odom_angular_velocity,
            ),
            self.safety_reaction_time,
            self.linear_deceleration,
            distance_margin=self.safety_distance_margin,
            lookahead_distance=self.live_validation_distance,
            live_obstacles=live_obstacles,
            tracking=tracking,
        )
        self.active_path_validation = decision.validation
        self.path_follower.update_clearance(decision.validation)
        return decision.speed_limit, False

    def _publish_path_diagnostics(self):
        self.diagnostics_pub.publish(
            Float64MultiArray(data=self.path_follower.diagnostics.as_array())
        )

    def _hard_path_stop_command(self, now):
        """Reset commanded motion when a required live input is unavailable."""
        self.path_follower.last_linear = 0.0
        self.path_follower.last_angular = 0.0
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.last_command_time = now
        self.path_follower.diagnostics.commanded_linear = 0.0
        self.path_follower.diagnostics.commanded_angular = 0.0
        self._publish_path_diagnostics()
        return Twist()

    def _common_path_command(self, now, tracking=None, safety_decision=None):
        if (
            self.odom_pose is None
            or self.goal_odom is None
            or self.active_path is None
            or self.active_path_state != self.state
        ):
            return None
        if tracking is None:
            tracking = self.path_follower.calculate_tracking(self.odom_pose)
        if self.last_command_time is None:
            elapsed = self.control_period
        else:
            elapsed = clamp((now - self.last_command_time).to_sec(), 0.0, 0.15)
        if safety_decision is None:
            speed_limit, emergency_stop = self._common_safety_speed_limit(
                now, tracking
            )
        else:
            if not isinstance(safety_decision, SafetyDecision):
                raise TypeError("safety_decision must be a SafetyDecision")
            if safety_decision.requires_stop:
                return self._hard_path_stop_command(now)
            self.active_path_validation = safety_decision.validation
            self.path_follower.update_clearance(
                safety_decision.validation
            )
            speed_limit = safety_decision.speed_limit
            emergency_stop = False
        if emergency_stop:
            return self._hard_path_stop_command(now)
        command, _tracking = self.path_follower.command(
            self.odom_pose,
            elapsed,
            speed_limit=speed_limit,
            tracking=tracking,
        )
        self.last_linear = command.linear_velocity
        self.last_angular = command.angular_velocity
        self.last_command_time = now
        self._publish_path_diagnostics()
        return self._twist_from_velocity(command)

    def _common_terminal_hold(self, now, tracking):
        """Hold a completed path while its confirmation samples arrive."""
        if self.last_command_time is None:
            elapsed = self.control_period
        else:
            elapsed = clamp((now - self.last_command_time).to_sec(), 0.0, 0.15)
        command, _tracking, _status = self.path_follower.terminal_hold(
            self.odom_pose,
            elapsed,
            tracking=tracking,
        )
        self.last_linear = command.linear_velocity
        self.last_angular = command.angular_velocity
        self.last_command_time = now
        self._publish_path_diagnostics()
        return self._twist_from_velocity(command)

    def _ramped_path_stop(self, now):
        """Use the common slew limiter before a normal segment transition."""
        if self.last_command_time is None:
            elapsed = self.control_period
        else:
            elapsed = clamp((now - self.last_command_time).to_sec(), 0.0, 0.15)
        command = self.path_follower.stop(elapsed)
        self.last_linear = command.linear_velocity
        self.last_angular = command.angular_velocity
        self.last_command_time = now
        self._publish_path_diagnostics()
        return self._twist_from_velocity(command)

    def _begin_drive(
        self,
        state,
        goal,
        direction,
        speed,
        goal_odom_override=None,
        stop_before_start=True,
    ):
        self.goal_map = tuple(float(value) for value in goal)
        direction = 1 if direction >= 0 else -1
        speed = max(0.01, float(speed))
        self.motion_timeout = self.drive_timeout
        if goal_odom_override is not None:
            self.goal_odom = tuple(float(value) for value in goal_odom_override)
        elif not self._latch_route_goal():
            self._fail("cannot transform the fixed parking route goal into odom")
            return
        if stop_before_start:
            self._publish_stop()
        self._set_state(state)
        path = self._straight_path(
            self.goal_odom,
            direction,
            speed,
            state,
        )
        if path is None or self.state == self.FAILED:
            return
        if not self._activate_common_path(path, state):
            self._fail("cannot activate common %s path" % state)
            return
        return True

    def _begin_rotation(self, state, target_route_yaw):
        """Start a separately validated zero-linear in-place rotation."""
        if state not in self.ROTATION_STATES or self.odom_pose is None:
            self._fail("invalid in-place parking rotation request")
            return
        current_route = self._odom_pose_to_route(self.odom_pose)
        self.goal_map = (
            float(current_route[0]),
            float(current_route[1]),
            normalize_angle(float(target_route_yaw)),
        )
        if not self._latch_route_goal():
            self._fail("cannot transform the in-place rotation goal into odom")
            return

        centre = Pose2D.from_value(self.odom_pose)
        target_yaw = float(self.goal_odom[2])
        poses = sample_in_place_rotation(
            centre, target_yaw, self.swept_heading_step
        )
        validation = self.path_validator.validate_poses(
            poses,
            safety=self._rotation_path_safety(),
        )
        if not validation.safe:
            self._fail(
                "%s fixed in-place sweep is unsafe (line=%.4fm map=%.4fm)"
                % (
                    state,
                    validation.minimum_line_clearance,
                    validation.minimum_map_clearance,
                )
            )
            return

        self._publish_stop()
        self._set_state(state)
        self.motion_timeout = self.rotation_timeout
        self.rotation_center_odom = centre
        self.rotation_target_yaw = target_yaw
        self.rotation_initial_error = max(
            self.rotation_command_tolerance,
            abs(normalize_angle(target_yaw - centre.yaw)),
        )
        self.rotation_static_validation = validation
        self.active_path = None
        self.active_path_state = None
        self.active_path_validation = validation
        self.path_follower.path = None
        self.path_follower.diagnostics = PathDiagnostics(
            remaining_distance=self.rotation_initial_error,
            position_error=0.0,
            cross_track_error=0.0,
            heading_error=normalize_angle(target_yaw - centre.yaw),
            minimum_line_clearance=validation.minimum_line_clearance,
            minimum_obstacle_clearance=validation.minimum_obstacle_clearance,
            minimum_map_clearance=validation.minimum_map_clearance,
        )
        self._publish_path_diagnostics()
        return True

    def _rotation_safety(self, now, proposed_angular):
        """Validate the remaining spin and every observed complete-stop sweep."""
        scan_is_fresh, live_obstacles = self._live_safety_snapshot(now)
        if self.require_live_safety_scan and not scan_is_fresh:
            return None

        pose = Pose2D.from_value(self.odom_pose)
        sequences = [
            sample_in_place_rotation(
                pose, self.rotation_target_yaw, self.swept_heading_step
            )
        ]
        rates = {
            float(value)
            for value in (
                self.odom_angular_velocity,
                self.last_angular,
                proposed_angular,
            )
            if math.isfinite(float(value)) and abs(float(value)) > 1e-9
        }
        for rate in rates:
            # ``stopping_distance_margin`` is translational.  During a spin,
            # convert it to the equivalent conservative angular sweep at the
            # farthest footprint corner so the shared safety margin is not
            # silently lost merely because linear velocity is zero.
            footprint_radius = max(
                math.hypot(self.front, self.half_width),
                math.hypot(self.rear, self.half_width),
            )
            stopping_margin_angle = (
                self.safety_distance_margin / footprint_radius
            )
            stopping_angle = (
                abs(rate) * self.safety_reaction_time
                + rate * rate / (2.0 * self.angular_acceleration)
                + self.swept_heading_step
                + stopping_margin_angle
            )
            sequences.append(
                sample_in_place_rotation(
                    pose,
                    pose.yaw + math.copysign(stopping_angle, rate),
                    self.swept_heading_step,
                )
            )
        safety = self._rotation_path_safety()
        return combine_validation_results(
            self.path_validator.validate_poses(
                sequence,
                safety=safety,
                live_obstacles=live_obstacles,
            )
            for sequence in sequences
        )

    def _set_rotation_diagnostics(self, command, validation):
        pose = Pose2D.from_value(self.odom_pose)
        error = normalize_angle(self.rotation_target_yaw - pose.yaw)
        drift = math.hypot(
            pose.x - self.rotation_center_odom.x,
            pose.y - self.rotation_center_odom.y,
        )
        diagnostics = self.path_follower.diagnostics
        diagnostics.progress = clamp(
            1.0 - abs(error) / self.rotation_initial_error, 0.0, 1.0
        )
        diagnostics.remaining_distance = abs(error)
        diagnostics.position_error = drift
        diagnostics.cross_track_error = drift
        diagnostics.heading_error = error
        diagnostics.target_speed = 0.0
        diagnostics.target_index = 0
        diagnostics.curvature = 0.0
        diagnostics.minimum_line_clearance = validation.minimum_line_clearance
        diagnostics.minimum_obstacle_clearance = (
            validation.minimum_obstacle_clearance
        )
        diagnostics.minimum_map_clearance = validation.minimum_map_clearance
        diagnostics.commanded_linear = 0.0
        diagnostics.commanded_angular = command.angular_velocity

    def _control_rotation(self, now):
        if (now - self.state_started).to_sec() > self.motion_timeout:
            self._fail("%s exceeded its in-place rotation timeout" % self.state)
            return
        if not self._odom_is_fresh(now):
            self._publish_stop()
            rospy.logwarn_throttle(
                2.0, "Parking holds zero for stale rotation odometry"
            )
            return

        pose = Pose2D.from_value(self.odom_pose)
        elapsed = (
            self.control_period
            if self.last_command_time is None
            else clamp((now - self.last_command_time).to_sec(), 0.0, 0.15)
        )
        command = in_place_rotation_command(
            pose.yaw,
            self.rotation_target_yaw,
            self.last_angular,
            elapsed,
            self.rotation_heading_gain,
            self.rotation_velocity,
            self.rotation_minimum_velocity,
            self.angular_acceleration,
            self.rotation_command_tolerance,
        )
        validation = self._rotation_safety(now, command.angular_velocity)
        if validation is None:
            self._publish_stop()
            rospy.logwarn_throttle(
                1.0, "Parking holds zero for a stale rotation safety scan"
            )
            return
        if not validation.safe:
            self.active_path_validation = validation
            # Record the unsafe sweep before the stop helper publishes the
            # diagnostic vector. Otherwise the message would repeat the last
            # safe clearance even though this tick stopped for contact.
            self._set_rotation_diagnostics(command, validation)
            self._publish_stop()
            rospy.logwarn_throttle(
                1.0,
                "%s holds zero for an unsafe live rotation sweep",
                self.state,
            )
            return

        self.active_path_validation = validation
        self.last_linear = 0.0
        self.last_angular = command.angular_velocity
        self.last_command_time = now
        self._set_rotation_diagnostics(command, validation)
        self._publish_path_diagnostics()
        self.cmd_pub.publish(self._twist_from_velocity(command))

        error = abs(normalize_angle(self.rotation_target_yaw - pose.yaw))
        stopped = (
            error <= self.rotation_heading_tolerance
            and abs(command.angular_velocity) <= 1e-9
            and abs(self.odom_angular_velocity)
            <= self.rotation_stopped_velocity
        )
        if self.odom_generation != self.last_confirmed_odom_generation:
            self.last_confirmed_odom_generation = self.odom_generation
            self.odom_confirmation_count = (
                self.odom_confirmation_count + 1 if stopped else 0
            )
        if self.odom_confirmation_count >= self.pose_confirm_samples:
            self._advance_rotation()

    def _begin_leave_aisle(self):
        """Re-enter the aisle and approach the zigzag turn without stopping."""
        if not self._build_zigzag_exit_curve():
            return

        start = tuple(float(value) for value in self.odom_pose)
        goal_map = self._zigzag_turn_start_pose()
        goal_odom = self._route_pose_to_odom(goal_map)
        approach_length = math.hypot(
            goal_odom[0] - start[0],
            goal_odom[1] - start[1],
        )
        approach_tangent = self.zigzag_approach_tangent_ratio * max(
            0.01, approach_length
        )
        sample_count = max(
            11,
            int(math.ceil(approach_length / self.path_sample_spacing)) + 1,
        )
        try:
            approach = quintic_pose_path(
                start,
                goal_odom,
                approach_tangent,
                approach_tangent,
                sample_count,
            )
        except ValueError as error:
            self._fail("cannot form the continuous zigzag approach: %s" % error)
            return
        forward_direction = np.asarray(
            [math.cos(goal_odom[2]), math.sin(goal_odom[2])],
            dtype=np.float64,
        )
        if np.any(np.diff(approach[:, :2], axis=0) @ forward_direction <= 0.0):
            self._fail("continuous zigzag approach is not forward-monotonic")
            return

        self.goal_map = goal_map
        self.goal_odom = goal_odom
        self.motion_timeout = self.drive_timeout
        self._publish_stop()
        self._set_state(self.LEAVE_AISLE)
        path = self._common_path_from_poses(
            approach,
            1,
            self.aisle_speed,
            self.LEAVE_AISLE,
            "parking_exit",
            self.arc_position_tolerance,
            self.heading_tolerance,
            self.arc_distance_tolerance,
            # Match the already validated curve's executable first speed.
            # Its angular-acceleration profile may be below the nominal
            # rejoin cap even though both paths meet at zero curvature.
            exit_velocity=float(self.zigzag_exit_curve_path.speed[0]),
        )
        if path is None or self.state == self.FAILED:
            return
        if not self._activate_common_path(path, self.LEAVE_AISLE):
            self._fail("cannot activate parking aisle-exit path")
            return
        return True

    def _build_zigzag_exit_curve(self):
        """Build the smooth left turn and its westbound alignment tail."""
        if self.odom_pose is None or self._frozen_local_to_odom() is None:
            return False
        try:
            self.zigzag_exit_curve_map = self._zigzag_exit_route_poses()
        except ValueError as error:
            self._fail("cannot form the smooth zigzag exit turn: %s" % error)
            return False
        self.zigzag_exit_curve_odom = np.asarray(
            [
                self._route_pose_to_odom(tuple(pose))
                for pose in self.zigzag_exit_curve_map
            ],
            dtype=np.float64,
        )
        self.zigzag_exit_curve_path = self._common_path_from_poses(
            self.zigzag_exit_curve_odom,
            1,
            self.rejoin_speed,
            self.TURN_TO_ZIGZAG,
            "final_turn",
            self.arc_position_tolerance,
            self.handoff_heading_tolerance,
            self.arc_distance_tolerance,
            entry_velocity=self.rejoin_speed,
            # A normal run hands control to the rolling lane follower while
            # still moving through the straight tail.  If camera evidence is
            # temporarily unavailable, however, the immutable path must also
            # be able to stop inside that same handoff box.  Reaching the
            # terminal plane at rejoin speed would need about 0.17 m to brake
            # at the measured plant limit; the common profile exit velocity
            # keeps the complete stopping sweep within the surveyed box.
            exit_velocity=self.profile_exit_velocity,
            feedforward_scale=self.arc_angular_scale,
        )
        return (
            self.zigzag_exit_curve_path is not None
            and self.state != self.FAILED
        )

    def _begin_zigzag_exit_curve(self):
        if self.zigzag_exit_curve_path is None:
            self._fail("smooth zigzag exit curve was not prepared")
            return False
        self.goal_map = tuple(
            float(value) for value in self.zigzag_exit_curve_map[-1]
        )
        self.goal_odom = tuple(
            float(value) for value in self.zigzag_exit_curve_odom[-1]
        )
        self.motion_timeout = self.drive_timeout
        self._set_state(self.TURN_TO_ZIGZAG)
        if not self._activate_common_path(
            self.zigzag_exit_curve_path,
            self.TURN_TO_ZIGZAG,
            initial_linear=self.last_linear,
            initial_angular=self.last_angular,
        ):
            self._fail("cannot activate the smooth zigzag exit curve")
            return False
        return True

    def _build_adaptive_entry(self, preflight=False):
        """Atomically prepare the registered connector and fixed turn.

        Candidate geometry is kept local until both CommonPaths pass their
        complete swept-footprint checks.  A later rejected candidate therefore
        cannot destroy the immutable route advertised in an earlier ready.
        """
        self.entry_preparation_error = ""
        if self.odom_pose is None or self._frozen_local_to_odom() is None:
            self.entry_preparation_error = "local registration or odom unavailable"
            return False

        entry_curve_map = self._entry_curve_route_poses()
        entry_curve_odom = np.asarray(
            [
                self._route_pose_to_odom(tuple(pose))
                for pose in entry_curve_map
            ],
            dtype=np.float64,
        )

        connector_start = tuple(float(value) for value in self.odom_pose)
        connector_end = tuple(
            float(value) for value in entry_curve_odom[0]
        )
        delta_x = connector_end[0] - connector_start[0]
        delta_y = connector_end[1] - connector_start[1]
        forward_distance = (
            math.cos(connector_end[2]) * delta_x
            + math.sin(connector_end[2]) * delta_y
        )
        if forward_distance <= 1e-4:
            self.entry_preparation_error = (
                "adaptive parking entry has no forward distance to its turn"
            )
            if not preflight:
                self._fail(self.entry_preparation_error)
            return False

        start_tangent = self.entry_connector_tangent_ratio * forward_distance
        # Scale both controls with the available straight. This keeps short
        # handoffs forward-monotonic and avoids a fixed end tangent pulling the
        # connector toward the lower solid line. Both endpoint curvatures are
        # still zero, so the following fixed turn remains a continuous G2 join.
        end_tangent = start_tangent
        if end_tangent <= 1e-4:
            self.entry_preparation_error = (
                "adaptive parking entry is too short to form a safe path"
            )
            if not preflight:
                self._fail(self.entry_preparation_error)
            return False

        try:
            entry_connector_odom = quintic_pose_path(
                connector_start,
                connector_end,
                start_tangent,
                end_tangent,
                self.entry_connector_samples,
            )
        except ValueError as error:
            self.entry_preparation_error = (
                "cannot form adaptive parking entry: %s" % error
            )
            if not preflight:
                self._fail(self.entry_preparation_error)
            return False

        approach_direction = np.asarray(
            [math.cos(connector_end[2]), math.sin(connector_end[2])],
            dtype=np.float64,
        )
        forward_steps = np.diff(
            entry_connector_odom[:, :2], axis=0
        ) @ approach_direction
        if np.any(forward_steps <= 0.0):
            self.entry_preparation_error = (
                "adaptive parking entry path is not forward-monotonic"
            )
            if not preflight:
                self._fail(self.entry_preparation_error)
            return False

        entry_connector_speed = self.approach_speed
        connector_entry_velocity = min(
            entry_connector_speed,
            max(
                self.minimum_velocity,
                abs(self.observed_lane_linear),
            ),
        )
        entry_connector_path = self._common_path_from_poses(
            entry_connector_odom,
            1,
            entry_connector_speed,
            self.APPROACH,
            "entry",
            self.entry_curve_end_position_tolerance,
            self.entry_curve_end_heading_tolerance,
            self.entry_curve_end_crossing_max_distance,
            entry_velocity=connector_entry_velocity,
            exit_velocity=min(entry_connector_speed, self.entry_turn_speed),
            # The entry-line callback and common swept validation own both
            # handoff rejection and live stopping; no second corner-only
            # clearance gate is applied outside PathSafety.
            total_line_margin=self.entry_handoff_minimum_clearance,
            feedforward_scale=self.arc_angular_scale,
            fail_on_error=not preflight,
        )
        if entry_connector_path is None or self.state == self.FAILED:
            if not self.entry_preparation_error:
                self.entry_preparation_error = (
                    "adaptive parking connector failed full swept validation"
                )
            return False
        entry_curve_path = self._common_path_from_poses(
            entry_curve_odom,
            1,
            self.entry_turn_speed,
            self.TURN_IN,
            "entry_turn",
            self.entry_curve_end_position_tolerance,
            self.entry_curve_end_heading_tolerance,
            self.entry_curve_end_crossing_max_distance,
            entry_velocity=self.entry_turn_speed,
            exit_velocity=self.entry_turn_speed,
            feedforward_scale=self.arc_angular_scale,
            fail_on_error=not preflight,
        )
        if entry_curve_path is None or self.state == self.FAILED:
            if not self.entry_preparation_error:
                self.entry_preparation_error = (
                    "parking entry turn failed full swept validation"
                )
            return False

        # Commit the whole route as one prepared generation only after both
        # segments are valid.  Every later handoff starts from a projection on
        # these exact CommonPath objects; it never synthesizes a short suffix.
        self.entry_curve_map = entry_curve_map
        self.entry_curve_odom = entry_curve_odom
        self.entry_connector_odom = entry_connector_odom
        self.entry_connector_speed = entry_connector_speed
        self.entry_connector_path = entry_connector_path
        self.entry_curve_path = entry_curve_path
        self.goal_map = tuple(float(value) for value in entry_curve_map[0])
        self.goal_odom = tuple(float(value) for value in entry_curve_odom[0])
        self.motion_timeout = self.drive_timeout
        rospy.loginfo(
            "Adaptive parking entry: length=%.3fm lateral=%.3fm "
            "clearance=%.3fm speed=%.3fm/s",
            entry_connector_path.length,
            connector_end[1] - connector_start[1],
            entry_connector_path.line_clearance,
            entry_connector_speed,
        )
        return True

    def _prepared_entry_projection(self):
        """Project the latest moving pose onto the immutable ready path."""
        if self.entry_connector_path is None or self.odom_pose is None:
            return None, "prepared parking entry or odometry is unavailable"
        pose = Pose2D.from_value(self.odom_pose)
        projection = project_to_path(
            self.entry_connector_path,
            pose.x,
            pose.y,
            search_ahead_distance=max(
                self.path_search_ahead, self.entry_connector_path.length
            ),
        )
        heading_error = abs(normalize_angle(projection.heading - pose.yaw))
        if projection.distance > self.entry_handoff_maximum_path_error + 1e-9:
            return (
                None,
                "prepared entry projection error %.4fm exceeds %.4fm"
                % (
                    projection.distance,
                    self.entry_handoff_maximum_path_error,
                ),
            )
        if heading_error > self.entry_handoff_maximum_heading_error + 1e-9:
            return (
                None,
                "prepared entry heading error %.2fdeg exceeds %.2fdeg"
                % (
                    math.degrees(heading_error),
                    math.degrees(
                        self.entry_handoff_maximum_heading_error
                    ),
                ),
            )
        return projection, ""

    def _prepared_entry_handoff_safety(self, now):
        """Validate a takeover from the actual capped moving pose.

        Fixed geometry was already swept when the ready path was committed.
        The common validator is still run here with the newest LiDAR snapshot
        so both the remaining route and the reaction-plus-complete-stop sweep
        originate from the measured handoff pose.
        """
        if not self._odom_is_fresh(now):
            return None, None, "parking entry handoff odometry is stale"
        projection, issue = self._prepared_entry_projection()
        if projection is None:
            return None, None, issue
        scan_is_fresh, live_obstacles = self._live_safety_snapshot(now)
        if self.require_live_safety_scan and not scan_is_fresh:
            return None, None, "parking entry handoff safety scan is stale"

        pose = Pose2D.from_value(self.odom_pose)
        stopping_linear = projection.direction * max(
            abs(self.observed_lane_linear), self.odom_linear_speed
        )
        tracking = calculate_tracking(
            self.entry_connector_path,
            pose,
            projection.path_index,
            self.common_tracking_config,
            previous_station=projection.station,
            linear_velocity=stopping_linear,
        )
        angular_velocities = (
            self.observed_lane_angular,
        ) + self.path_follower.stopping_angular_velocities(
            tracking,
            stopping_linear,
            self.odom_angular_velocity,
        )
        decision = self.path_validator.motion_safety(
            self.entry_connector_path,
            pose,
            tracking.path_index,
            tracking.target_speed,
            stopping_linear,
            angular_velocities,
            self.safety_reaction_time,
            self.linear_deceleration,
            distance_margin=self.safety_distance_margin,
            lookahead_distance=self.live_validation_distance,
            live_obstacles=live_obstacles,
            tracking=tracking,
        )
        if decision.requires_stop:
            validation = decision.validation
            return (
                None,
                decision,
                "prepared entry requires a stop "
                "(line=%.4fm obstacle=%.4fm local=%.4fm)"
                % (
                    validation.minimum_line_clearance,
                    validation.minimum_obstacle_clearance,
                    validation.minimum_map_clearance,
                ),
            )
        return tracking, decision, ""

    def _begin_entry_curve(self):
        self.goal_map = tuple(float(value) for value in self.entry_curve_map[-1])
        self.goal_odom = tuple(float(value) for value in self.entry_curve_odom[-1])
        self.motion_timeout = self.drive_timeout
        self._set_state(self.TURN_IN)
        if not self._activate_common_path(
            self.entry_curve_path,
            self.TURN_IN,
            initial_linear=self.last_linear,
            initial_angular=self.last_angular,
        ):
            self._fail("cannot activate the common parking entry curve")

    def _continuous_path_completion_status(self, tracking=None):
        if self.active_path is None or self.odom_pose is None:
            return "TRACK"
        status = self.path_follower.goal_status(
            self.odom_pose, tracking=tracking
        )
        if status.complete:
            return "READY"
        if (
            status.crossed_terminal
            and status.position_error
            > self.active_path.goal_tolerance.terminal_crossing
        ):
            return "PASSED"
        return "TRACK"

    def _try_publish_ready(self, now):
        """Prepare and validate entry while the lane controller keeps driving."""
        if (
            self.arm_seq is None
            or self.arm_stamp is None
            or self.local_to_odom is None
            or self.state != self.WAIT_GATE
            or not self._odom_is_fresh(now)
            or self.odom_stamp is None
            or self.odom_stamp < self.arm_stamp
        ):
            return False
        already_prepared = self.ready_published_seq == self.arm_seq
        # READY is a physical handoff/braking window, so it must use the newest
        # odom pose rather than the older pose synchronized to the LiDAR scan.
        # The arm-time lane speed cap keeps the two-scan registration latency
        # inside this window without weakening its stopping-distance meaning.
        progress = entry_plane_progress(
            self.registration_template.entry_plane,
            self.odom_pose,
            self.local_to_odom,
        )
        if not (
            self.entry_lead_minimum
            <= progress.longitudinal
            <= self.entry_lead_maximum
            and abs(progress.lateral) <= self.entry_lead_lateral_tolerance
            and abs(progress.heading_error)
            <= self.entry_lead_heading_tolerance
        ):
            return False
        if already_prepared:
            if (
                self.ready_source_stamp is not None
                and (self.odom_stamp - self.ready_source_stamp).to_sec()
                < self.ready_republish_period
            ):
                return False
            projection, issue = self._prepared_entry_projection()
            if projection is None:
                rospy.logwarn_throttle(
                    1.0,
                    "Parking prepared entry is no longer reachable: %s",
                    issue,
                )
                return False
        else:
            if not self._build_adaptive_entry(preflight=True):
                rospy.logwarn_throttle(
                    1.0,
                    "Parking local entry is not ready: %s",
                    self.entry_preparation_error,
                )
                return False
        ready = Header()
        ready.seq = self.arm_seq
        ready.stamp = self.odom_stamp
        ready.frame_id = self.mission_name
        self.ready_pub.publish(ready)
        self.ready_published_seq = self.arm_seq
        self.ready_source_stamp = self.odom_stamp
        rospy.loginfo(
            "Parking ready generation=%d at local lead %.3fm: complete "
            "entry connector and turn swept validation passed",
            self.arm_seq,
            progress.longitudinal,
        )
        return True

    def _ready_matches_current_arm(self, now):
        if (
            self.arm_seq is None
            or self.ready_published_seq != self.arm_seq
            or self.ready_source_stamp is None
            or self.local_to_odom is None
            or self.entry_connector_path is None
            or self.entry_curve_path is None
        ):
            return False
        age = (now - self.ready_source_stamp).to_sec()
        return -self.registration_future_tolerance <= age <= self.odom_timeout

    def _start_run(self, now):
        if not (
            self._inputs_fresh(now)
            and self._ready_matches_current_arm(now)
        ):
            rospy.logwarn_throttle(
                2.0,
                "Waiting for matching parking ready and fresh local odometry",
            )
            return
        self.start_requested = False
        self.revoke_requested = False
        self.selected_space = None
        self.space_pub.publish(String(data="UNKNOWN"))
        self.parking_goal_map = None
        self.parking_return_map = None
        self.parking_return_odom = None
        self.parking_goal_amcl = None
        self.parking_return_amcl = None
        self.rotation_center_odom = None
        self.rotation_target_yaw = None
        self.rotation_initial_error = 0.0
        self.rotation_static_validation = None
        self.zigzag_exit_curve_map = None
        self.zigzag_exit_curve_odom = None
        self.zigzag_exit_curve_path = None
        self.active_path = None
        self.active_path_state = None
        self.active_path_validation = None
        self.path_follower.path = None
        self.path_follower.path_index = 0
        self.path_follower.path_station = 0.0
        self.path_follower.last_linear = 0.0
        self.path_follower.last_angular = 0.0
        self.lane_confirmation_count = 0
        self.last_lane_confirmation_time = None
        self.prepare_odom_generation = self.gate_odom_generation
        self._set_state(self.PREPARE_APPROACH)

    def _revoke(self):
        if self.mission_has_control:
            self._publish_stop()
            if not self._set_lane_controller(True):
                self._fail("lane-controller handoff failed after parking gate closed")
                return
        self.speed_limit_pub.publish(
            Float64(data=self.lane_resume_max_velocity)
        )
        self.revoke_requested = False
        self.start_requested = False
        self._set_state(self.WAIT_GATE)

    def _fail(self, reason):
        owns_unambiguously = (
            self.mission_has_control and not self.handoff_ambiguous
        )
        if not owns_unambiguously:
            owns_unambiguously = self._set_lane_controller(False)
        if owns_unambiguously:
            self._publish_stop()
        else:
            # A timed-out service may have applied either ownership state.
            # Relinquish direct output, stop the lane publisher through its
            # independent service, and pause every mission controller instead
            # of racing the lane controller on /cmd_vel.
            self.mission_has_control = False
            if not self._stop_lane_controller():
                rospy.logfatal("Parking could not establish a sole cmd_vel stop owner")
            self.emergency_stop_pub.publish(Bool(data=True))
        self._set_state(self.FAILED)
        rospy.logerr("Parking mission failed: %s", reason)

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
        self._publish_path_diagnostics()

    def _fresh_amcl_checkpoint(self):
        """Snapshot AMCL for diagnostics without making local motion depend on it."""
        if not self._map_pose_is_fresh(rospy.Time.now()):
            return None
        return tuple(float(value) for value in self.map_pose)

    def _report_amcl_checkpoint(self, phase, checkpoint, now):
        """Report localization correction while odom remains the local authority."""
        if checkpoint is None:
            rospy.logwarn(
                "%s AMCL checkpoint was unavailable; continuing from fresh "
                "settled odometry",
                phase,
            )
            return
        if not self._map_pose_is_fresh(now):
            rospy.logwarn(
                "%s current AMCL pose is stale; continuing from fresh settled "
                "odometry",
                phase,
            )
            return
        position_delta = math.hypot(
            self.map_pose[0] - checkpoint[0],
            self.map_pose[1] - checkpoint[1],
        )
        heading_delta = abs(normalize_angle(self.map_pose[2] - checkpoint[2]))
        message = (
            "%s observed AMCL checkpoint delta %.4fm/%.2fdeg; local parking "
            "motion remains anchored to settled odometry"
        )
        arguments = (
            phase,
            position_delta,
            math.degrees(heading_delta),
        )
        if (
            position_delta > self.checkpoint_position_tolerance
            or heading_delta > self.checkpoint_heading_tolerance
        ):
            rospy.logwarn(message, *arguments)
        else:
            rospy.loginfo(message, *arguments)

    def _odom_goal_reached(self, tracking=None):
        if self.odom_pose is None or self.goal_odom is None:
            return False
        if self.active_path is None or self.active_path_state != self.state:
            return False
        return self.path_follower.goal_status(
            self.odom_pose, tracking=tracking
        ).complete

    def _state_position_tolerance(self, state=None):
        state = self.state if state is None else state
        if state in (
            self.ENTER_AISLE,
            self.PARK_IN,
            self.BACK_OUT,
            self.LEAVE_AISLE,
            self.TURN_TO_ZIGZAG,
        ):
            return self.arc_position_tolerance
        return self.position_tolerance

    def _state_heading_tolerance(self, state=None):
        state = self.state if state is None else state
        if state in (
            self.TURN_IN,
            self.ENTER_AISLE,
            self.PARK_IN,
            self.BACK_OUT,
            self.LEAVE_AISLE,
            self.TURN_TO_ZIGZAG,
        ):
            return self.arc_heading_tolerance
        return self.heading_tolerance

    def _update_odom_confirmation(self, tracking=None, status=None):
        if self.odom_generation == self.last_confirmed_odom_generation:
            return False
        self.last_confirmed_odom_generation = self.odom_generation
        if status is None:
            reached = self._odom_goal_reached(tracking=tracking)
        else:
            reached = bool(status.complete)
        if reached:
            self.odom_confirmation_count += 1
        else:
            self.odom_confirmation_count = 0
        return self.odom_confirmation_count >= self.pose_confirm_samples

    def _fail_motion_timeout(self):
        """Report the latest anchored error before stopping a timed-out path."""
        if self.odom_pose is not None and self.goal_odom is not None:
            error_x = self.goal_odom[0] - self.odom_pose[0]
            error_y = self.goal_odom[1] - self.odom_pose[1]
            error_yaw = normalize_angle(
                self.goal_odom[2] - self.odom_pose[2]
            )
            rospy.logerr(
                "%s timeout error: position=%.4fm heading=%.2fdeg "
                "progress=%.1f%% remaining=%.4fm",
                self.state,
                math.hypot(error_x, error_y),
                math.degrees(error_yaw),
                100.0 * self.path_follower.diagnostics.progress,
                self.path_follower.diagnostics.remaining_distance,
            )
        self._fail("%s exceeded its motion timeout" % self.state)

    def _control_motion(self, now):
        timed_out = (now - self.state_started).to_sec() > self.motion_timeout
        if not self._odom_is_fresh(now):
            if timed_out:
                self._fail_motion_timeout()
                return
            if self.mission_has_control:
                self.cmd_pub.publish(self._hard_path_stop_command(now))
            rospy.logwarn_throttle(
                2.0, "Parking holds zero for stale local odometry"
            )
            return

        tracking = self.path_follower.calculate_tracking(self.odom_pose)

        # Completion checks also record path progress and can enter terminal
        # hold without calling _common_path_command().  Enforce the same live
        # scan contract before any of those branches so a stale input commands
        # an exact zero throughout the path lifecycle.
        if self.require_live_safety_scan:
            scan_is_fresh, _live_obstacles = self._live_safety_snapshot(now)
            if not scan_is_fresh:
                if timed_out:
                    self._fail_motion_timeout()
                    return
                if self.mission_has_control:
                    self.cmd_pub.publish(
                        self._hard_path_stop_command(now)
                    )
                rospy.logwarn_throttle(
                    1.0, "Parking holds zero for a stale swept-path safety scan"
                )
                return

        if self.state == self.APPROACH:
            if timed_out:
                self._fail_motion_timeout()
                return
            completion = self._continuous_path_completion_status(tracking)
            if completion == "PASSED":
                self._fail("passed the adaptive parking entry endpoint")
                return
            if completion == "READY":
                self._begin_entry_curve()
                tracking = self.path_follower.calculate_tracking(
                    self.odom_pose
                )
            command = self._common_path_command(now, tracking)
            if command is not None and self.state != self.FAILED:
                self.cmd_pub.publish(command)
            return

        if self.state == self.TURN_IN:
            if timed_out:
                self._fail_motion_timeout()
                return
            completion = self._continuous_path_completion_status(tracking)
            if completion == "PASSED":
                self._fail("passed the continuous parking entry-curve endpoint")
                return
            if completion == "READY":
                self._begin_drive(
                    self.ENTER_AISLE,
                    (self.aisle_x, self.decision_y, self.aisle_heading),
                    1,
                    self.aisle_speed,
                    stop_before_start=False,
                )
                command = self._common_path_command(now)
            else:
                command = self._common_path_command(now, tracking)
            if command is not None and self.state != self.FAILED:
                self.cmd_pub.publish(command)
            return

        if self.state == self.LEAVE_AISLE:
            if timed_out:
                self._fail_motion_timeout()
                return
            completion = self._continuous_path_completion_status(tracking)
            if completion == "PASSED":
                self._fail("passed the continuous zigzag-turn start")
                return
            if completion == "READY":
                if not self._begin_zigzag_exit_curve():
                    return
                tracking = self.path_follower.calculate_tracking(
                    self.odom_pose
                )
            command = self._common_path_command(now, tracking)
            if command is not None and self.state != self.FAILED:
                self.cmd_pub.publish(command)
            return

        if (
            self.state == self.TURN_TO_ZIGZAG
            and self._zigzag_lane_handoff_ready(now)
        ):
            command = self._common_path_command(now, tracking)
            if (
                command is not None
                and command.linear.x > 1e-9
                and self.active_path_validation is not None
                and self.active_path_validation.safe
            ):
                self.cmd_pub.publish(command)
                self._handoff_to_zigzag_lane(moving=True)
                return

        goal_status = None
        newly_confirmed = False
        if not self.motion_goal_confirmed:
            goal_status = self.path_follower.goal_status(
                self.odom_pose,
                tracking=tracking,
            )
            # A watchdog is not a reason to reject a pose that has physically
            # reached its frozen goal on this very tick.  Admit that sample so
            # the normal independent-odom confirmation and slew-limited stop
            # can finish; a pose still outside tolerance fails immediately.
            if timed_out and not goal_status.complete:
                self._fail_motion_timeout()
                return
            if self._update_odom_confirmation(
                tracking,
                status=goal_status,
            ):
                self.motion_goal_confirmed = True
                newly_confirmed = True

        if self.motion_goal_confirmed:
            stop_elapsed = (
                self.control_period
                if self.last_command_time is None
                else clamp(
                    (now - self.last_command_time).to_sec(), 0.0, 0.15
                )
            )
            command = self._ramped_path_stop(now)
            self.cmd_pub.publish(command)
            if (
                abs(command.linear.x) <= 1e-9
                and abs(command.angular.z) <= 1e-9
                and self.odom_linear_speed <= self.stopped_linear_velocity
                and abs(self.odom_angular_velocity)
                <= self.rotation_stopped_velocity
                and (not newly_confirmed or stop_elapsed > 1e-9)
            ):
                self._advance_motion()
            return

        if (
            goal_status is not None
            and goal_status.crossed_terminal
        ):
            # Confirmation deliberately spans distinct odometry samples.  Do
            # not keep applying the segment's exit speed between those samples
            # after the common terminal event has already occurred.
            self.cmd_pub.publish(self._common_terminal_hold(now, tracking))
            return

        command = self._common_path_command(now, tracking)
        if command is not None and self.state != self.FAILED:
            self.cmd_pub.publish(command)

    def _advance_motion(self):
        if self.state == self.ENTER_AISLE:
            self.selection_candidate = None
            self.selection_candidate_count = 0
            self.last_candidate_scan_stamp = None
            self.last_selection_scan_generation = self.scan_generation
            self._set_state(self.SELECT_SPACE)
        elif self.state == self.PARK_IN:
            # PARK_IN already stopped on fresh odometry and received the
            # configured confirmation samples.  Record the diagnostic AMCL
            # checkpoint and reverse immediately; there is no display hold.
            self.parking_goal_amcl = self._fresh_amcl_checkpoint()
            self._report_amcl_checkpoint(
                "PARKED", self.parking_goal_amcl, rospy.Time.now()
            )
            self._begin_drive(
                self.BACK_OUT,
                self.parking_return_map,
                -1,
                self.reverse_speed,
                goal_odom_override=self.parking_return_odom,
            )
        elif self.state == self.BACK_OUT:
            self._report_amcl_checkpoint(
                "BACK_OUT", self.parking_return_amcl, rospy.Time.now()
            )
            self._begin_rotation(self.TURN_TO_EXIT, self.outgoing_heading)
        elif self.state == self.TURN_TO_ZIGZAG:
            # Lane visibility is normally confirmed on the moving alignment
            # tail. Reaching its endpoint without enough fresh frames is a
            # safe exceptional fallback, so stop and retain the old verifier.
            self._set_state(self.VERIFY_ZIGZAG_LANE)
        else:
            self._fail("unexpected completed parking state %s" % self.state)

    def _advance_rotation(self):
        if self.state == self.TURN_TO_SPACE:
            goal_x = (
                self.left_park_x
                if self.selected_space == LEFT
                else self.right_park_x
            )
            goal_yaw = 0.0 if self.selected_space == LEFT else math.pi
            actual_return_map = self._odom_pose_to_route(self.odom_pose)
            # Preserve the measured spin centre as the endpoint shared by the
            # straight PARK_IN and BACK_OUT paths.  This adapts to normal
            # stopping error without an arbitrary map-pose portal gate.
            self.parking_return_odom = (
                float(self.odom_pose[0]),
                float(self.odom_pose[1]),
                float(self.goal_odom[2]),
            )
            self.parking_return_amcl = self._fresh_amcl_checkpoint()
            self.parking_return_map = (
                float(actual_return_map[0]),
                float(actual_return_map[1]),
                goal_yaw,
            )
            self.parking_goal_map = (
                goal_x,
                self.parking_return_map[1],
                goal_yaw,
            )
            lower_y = self.lower_outer_edge + self.line_margin
            upper_y = self.upper_outer_edge - self.line_margin
            if any(
                y < lower_y or y > upper_y
                for _, y in rectangle_corners(
                    self.parking_goal_map,
                    self.front,
                    self.rear,
                    self.half_width,
                )
            ):
                self._fail(
                    "actual parking entrance line y=%.4f would put the "
                    "footprint across the parking-area y paint [%.4f, %.4f]"
                    % (self.parking_goal_map[1], lower_y, upper_y)
                )
                return
            self._begin_drive(
                self.PARK_IN,
                self.parking_goal_map,
                1,
                self.parking_speed,
            )
        elif self.state == self.TURN_TO_EXIT:
            self._begin_leave_aisle()
        else:
            self._fail("unexpected completed rotation state %s" % self.state)

    def _selection_input_issue(self, now):
        """Explain why a new stopped parking-bay scan cannot be consumed."""
        if not self._odom_is_fresh(now):
            return "stale parking odometry"
        local_to_odom = self._frozen_local_to_odom()
        if local_to_odom is None:
            return "parking local registration unavailable"

        # ENTER_AISLE already confirmed the stopped endpoint from odometry.
        # Project that same odometry through the frozen fixture transform;
        # AMCL is diagnostic only and cannot suppress an unambiguous scan.
        local_pose = local_to_odom.inverse().apply_pose(self.odom_pose)
        heading_error = abs(
            normalize_angle(local_pose.yaw - self.aisle_heading)
        )
        if heading_error > self.checkpoint_heading_tolerance:
            return "local aisle heading error %.1fdeg exceeds %.1fdeg" % (
                math.degrees(heading_error),
                math.degrees(self.checkpoint_heading_tolerance),
            )
        if self.scan_received is None or self.scan_stamp is None:
            return "no LiDAR scan received"
        source_age = (now - self.scan_stamp).to_sec()
        receipt_age = (now - self.scan_received).to_sec()
        if source_age > self.scan_timeout or receipt_age > self.scan_timeout:
            return "stale LiDAR scan (source %.3fs, receipt %.3fs)" % (
                source_age,
                receipt_age,
            )
        if self.scan_generation == self.last_selection_scan_generation:
            return "waiting for a new LiDAR scan"
        return None

    def _select_space(self, now):
        self._publish_stop()
        input_issue = self._selection_input_issue(now)
        if (now - self.state_started).to_sec() > self.selection_timeout:
            candidate = choose_clear_space(
                self.left_points,
                self.right_points,
                self.occupied_minimum_points,
                self.clear_maximum_points,
            )
            if input_issue is not None:
                detail = input_issue
            elif candidate is None:
                detail = (
                    "ambiguous occupancy left=%d right=%d"
                    % (self.left_points, self.right_points)
                )
            else:
                detail = "confirming %s scan %d/%d" % (
                    candidate,
                    self.selection_candidate_count,
                    self.selection_confirm_scans,
                )
            self._fail("parking-bay selection timed out: %s" % detail)
            return
        if input_issue is not None:
            rospy.logwarn_throttle(
                1.0, "Parking-bay selection waiting: %s", input_issue
            )
            return
        self.last_selection_scan_generation = self.scan_generation
        candidate = choose_clear_space(
            self.left_points,
            self.right_points,
            self.occupied_minimum_points,
            self.clear_maximum_points,
        )
        if candidate is None:
            self.selection_candidate = None
            self.selection_candidate_count = 0
            self.last_candidate_scan_stamp = None
            return
        scan_gap = (
            None
            if self.last_candidate_scan_stamp is None
            else (self.scan_stamp - self.last_candidate_scan_stamp).to_sec()
        )
        if (
            candidate == self.selection_candidate
            and scan_gap is not None
            and 0.0 <= scan_gap <= self.selection_confirmation_max_gap
        ):
            self.selection_candidate_count += 1
        else:
            self.selection_candidate = candidate
            self.selection_candidate_count = 1
        self.last_candidate_scan_stamp = self.scan_stamp
        if self.selection_candidate_count < self.selection_confirm_scans:
            return

        self.selected_space = candidate
        self.space_pub.publish(String(data=candidate))
        rospy.loginfo(
            "Parking selected %s bay (LiDAR points left=%d right=%d)",
            candidate,
            self.left_points,
            self.right_points,
        )
        target_yaw = 0.0 if candidate == LEFT else math.pi
        self._begin_rotation(self.TURN_TO_SPACE, target_yaw)

    def _prepare_approach(self, now):
        elapsed = (now - self.state_started).to_sec()
        if elapsed > self.prepare_timeout:
            self._fail("fresh moving-pose parking handoff was not available")
            return
        if not self._inputs_fresh(now):
            rospy.logwarn_throttle(
                1.0, "Parking entry handoff lost fresh registered odometry"
            )
            return
        command_fresh = (
            self.observed_lane_command_received is not None
            and (now - self.observed_lane_command_received).to_sec()
            <= self.odom_timeout
        )
        command_after_gate = (
            self.lane_command_generation
            > self.gate_lane_command_generation
        )
        handoff_speed_limit = min(
            self.lane_entry_speed_limit, self.approach_speed
        )
        command_is_bounded = (
            self.observed_lane_linear > 1e-4
            and self.observed_lane_linear <= handoff_speed_limit + 1e-9
            and abs(self.observed_lane_angular)
            <= self.maximum_angular_velocity + 1e-9
        )
        if (
            elapsed < self.prepare_settle_time
            or self.odom_generation <= self.prepare_odom_generation
            or not command_after_gate
            or not command_fresh
            or not command_is_bounded
            or self.bounded_lane_command_generation is None
            or self.bounded_lane_command_odom_generation is None
            or self.odom_generation
            <= self.bounded_lane_command_odom_generation
        ):
            rospy.logwarn_throttle(
                1.0,
                "Parking waits for a fresh capped lane command followed by "
                "post-cap odometry while lane control remains active",
            )
            return
        handoff_tracking, _decision, handoff_issue = (
            self._prepared_entry_handoff_safety(now)
        )
        if handoff_tracking is None:
            rospy.logwarn_throttle(
                1.0,
                "Parking waits for a safe prepared-path handoff: %s",
                handoff_issue,
            )
            return

        # The entry was generated and completely swept before ready.  The
        # shared follower starts at the projected station of the latest
        # post-cap pose; no mission-specific suffix is regenerated here.
        handoff_linear = self.observed_lane_linear
        handoff_angular = self.observed_lane_angular
        if not self._set_lane_controller(False):
            self._fail("could not acquire cmd_vel control")
            return
        # Odom and the synchronized safety scan can arrive while the service
        # call owns the controller lock.  Apply that staged pose and run the
        # same common handoff check once more before the first mission command.
        self._apply_pending_odom_state()
        command_now = rospy.Time.now()
        handoff_tracking, handoff_decision, handoff_issue = (
            self._prepared_entry_handoff_safety(command_now)
        )
        if handoff_tracking is None:
            self._fail(
                "parking entry became unsafe during control handoff: %s"
                % handoff_issue
            )
            return
        self.last_linear = handoff_linear
        self.last_angular = handoff_angular
        self.last_command_time = command_now
        self._set_state(self.APPROACH)
        if not self._activate_common_path(
            self.entry_connector_path,
            self.APPROACH,
            initial_linear=handoff_linear,
            initial_angular=handoff_angular,
        ):
            self._fail("cannot activate the common adaptive parking entry")
            return
        self.cmd_pub.publish(
            self._common_path_command(
                command_now,
                tracking=handoff_tracking,
                safety_decision=handoff_decision,
            )
        )

    def _zigzag_lane_handoff_ready(self, now):
        confirmation_fresh = (
            self.last_lane_confirmation_time is not None
            and (now - self.last_lane_confirmation_time).to_sec()
            <= self.handoff_confirmation_max_gap
        )
        return bool(
            self._odom_is_fresh(now)
            and self._lane_observation_valid(now, joining=False)
            and confirmation_fresh
            and self.lane_confirmation_count >= self.handoff_confirm_frames
        )

    def _handoff_to_zigzag_lane(self, moving):
        self.speed_limit_pub.publish(Float64(data=self.rejoin_speed))
        if not self._set_lane_controller(True):
            self._fail("could not hand the visible zigzag lane to lane control")
            return False
        rospy.loginfo(
            "Zigzag lane visible in %d frames; %s lane rejoin started",
            self.lane_confirmation_count,
            "moving" if moving else "stopped fallback",
        )
        self._set_state(self.JOIN_ZIGZAG)
        return True

    def _verify_zigzag_lane(self, now):
        self._publish_stop()
        # A complete set of fresh observations acquired on the timeout edge is
        # still valid evidence.  Consume it before evaluating the watchdog so
        # timer scheduling jitter cannot discard an already-confirmed handoff.
        if self._zigzag_lane_handoff_ready(now):
            self._handoff_to_zigzag_lane(moving=False)
            return
        if (now - self.state_started).to_sec() > self.handoff_timeout:
            self._fail(
                "zigzag lane was not visible at the safe handoff pose: %s"
                % self._lane_observation_summary()
            )
            return

    def _join_zigzag(self, now):
        if (now - self.state_started).to_sec() > self.rejoin_timeout:
            self._fail(
                "lane control did not reach the confirmed zigzag entry: %s"
                % self._lane_observation_summary()
            )
            return
        confirmation_fresh = (
            self.last_lane_confirmation_time is not None
            and (now - self.last_lane_confirmation_time).to_sec()
            <= self.rejoin_confirmation_max_gap
        )
        if not (
            self._odom_is_fresh(now)
            and self._lane_observation_valid(now, joining=True)
            and confirmation_fresh
            and self.lane_confirmation_count >= self.rejoin_confirm_frames
        ):
            return
        self._complete()

    def _complete(self):
        if self.mission_has_control:
            if not self._set_lane_controller(True):
                self._fail("could not return cmd_vel to lane controller")
                return
        self._set_state(self.COMPLETE)
        self.speed_limit_pub.publish(
            Float64(data=self.post_completion_velocity_cap)
        )
        rospy.loginfo(
            "Parking complete in %s bay; zigzag lane entry is confirmed",
            self.selected_space,
        )

    def control_callback(self, _event):
        with self.lock:
            if self.shutting_down:
                return
            self._apply_pending_odom_state()
            self._apply_pending_registration_scan()
            occupancy = self._apply_pending_selection_scan()
            if occupancy is not None:
                self.occupancy_pub.publish(occupancy)
            now = rospy.Time.now()
            if self.revoke_requested:
                self._revoke()
                return
            if (
                self.start_requested
                and self.zone_gate
                and self.state == self.WAIT_GATE
            ):
                self._start_run(now)

            if self.state == self.WAIT_GATE:
                # Registration and full entry validation happen while the
                # lane controller retains cmd_vel ownership.  The arm callback
                # has already applied its bounded lane speed cap.
                self._try_publish_ready(now)
                return
            if self.state == self.COMPLETE:
                return
            if self.manual_stop:
                if self.mission_has_control:
                    self._publish_stop()
                return
            if self.state == self.FAILED:
                if self.mission_has_control:
                    self._publish_stop()
                return
            if self.state == self.PREPARE_APPROACH:
                self._prepare_approach(now)
                return
            if self.state == self.SELECT_SPACE:
                self._select_space(now)
                return
            if self.state in self.ROTATION_STATES:
                self._control_rotation(now)
                return
            if self.state == self.VERIFY_ZIGZAG_LANE:
                self._verify_zigzag_lane(now)
                return
            if self.state == self.JOIN_ZIGZAG:
                self._join_zigzag(now)
                return
            if self.state in self.MOTION_STATES:
                self._control_motion(now)
                return
            self._fail("unknown parking state %s" % self.state)

    def shutdown(self):
        with self.lock:
            self.shutting_down = True
            if self.mission_has_control:
                self.cmd_pub.publish(Twist())
        rospy.loginfo("Parking mission controller stopped")


if __name__ == "__main__":
    rospy.init_node("parking_mission_controller")
    ParkingMissionController()
    rospy.spin()
