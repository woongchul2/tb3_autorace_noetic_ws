#!/usr/bin/env python3

from collections import deque
import math
import numpy as np
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image
import yaml


NODE_DIR = Path(__file__).resolve().parents[1] / "nodes"
PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(NODE_DIR) not in sys.path:
    sys.path.insert(0, str(NODE_DIR))

import tunnel_mission_controller as controller_module
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64MultiArray, Header
from tunnel_mission_controller import TunnelMissionController


class RecordingPublisher:
    def __init__(self, events=None, name=None):
        self.messages = []
        self.events = events
        self.name = name

    def publish(self, message):
        self.messages.append(message)
        if self.events is not None:
            self.events.append(("publish", self.name, message))


class RecordingLaneService:
    def __init__(self, events):
        self.calls = []
        self.events = events

    def __call__(self, enabled):
        enabled = bool(enabled)
        self.calls.append(enabled)
        self.events.append(("handoff", enabled))
        return SimpleNamespace(success=True, message="ok")


class RecordingLaneStopService:
    def __init__(self, events):
        self.calls = []
        self.events = events

    def __call__(self, enabled):
        enabled = bool(enabled)
        self.calls.append(enabled)
        self.events.append(("lane_stop", enabled))
        return SimpleNamespace(success=True, message="ok")


class RecordingCostmap:
    def __init__(self):
        self.calls = []

    def update_scan(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            marked_cells=0,
            cleared_cells=0,
            decayed_cells=0,
        )


class TunnelMissionControllerTest(unittest.TestCase):
    def setUp(self):
        self.seconds = 10.0
        self.time_patch = mock.patch.object(
            controller_module.rospy.Time,
            "now",
            side_effect=lambda: controller_module.rospy.Time.from_sec(
                self.seconds
            ),
        )
        self.time_patch.start()
        self.addCleanup(self.time_patch.stop)
        self.wait_patch = mock.patch.object(
            controller_module.rospy, "wait_for_service", return_value=None
        )
        self.wait_patch.start()
        self.addCleanup(self.wait_patch.stop)

    @property
    def now(self):
        return controller_module.rospy.Time.from_sec(self.seconds)

    def advance(self, seconds=0.05):
        self.seconds += seconds

    def make_controller(self):
        controller = TunnelMissionController.__new__(TunnelMissionController)
        controller.lock = threading.RLock()
        controller.state = controller.WAIT_GATE
        controller.state_started = self.now
        controller.mission_started = None
        controller.zone_gate = False
        controller.gate_requested = False
        controller.start_requested = False
        controller.revoke_requested = False
        controller.manual_stop = False
        controller.pause_started = None
        controller.mission_has_control = False
        controller.run_generation = 0
        controller.input_fault = ""

        controller.map_ready = True
        controller.map_x = 0.0
        controller.map_y = 0.0
        controller.map_yaw = 0.0
        controller.map_stamp = self.now
        controller.map_received = self.now
        controller.odom_ready = True
        controller.odom_x = 0.0
        controller.odom_y = 0.0
        controller.odom_yaw = 0.0
        controller.odom_linear_velocity = 0.0
        controller.odom_angular_velocity = 0.0
        controller.odom_stamp = self.now
        controller.odom_received = self.now
        controller.odom_frame = "odom"
        controller.odom_history = deque(maxlen=100)
        controller.map_from_odom = None
        controller.frozen_odom_frame = ""
        controller.arm_generation = 1
        controller.armed_at = controller_module.rospy.Time.from_sec(9.0)
        controller.registered_map_from_odom = (0.0, 0.0, 0.0)
        controller.registered_odom_frame = "odom"
        controller.registration_source_stamp = self.now
        controller.registration_samples = 3
        controller.registration_maximum_gap = 0.20
        controller.registration_maximum_position_delta = 0.04
        controller.registration_maximum_heading_delta = math.radians(4.0)
        controller.registration_candidates = deque(maxlen=3)
        controller.last_registration_stamp = None
        controller.ready_published_generation = 1
        controller.ready_minimum_entry_lead = 0.20
        controller.ready_maximum_entry_lead = 0.50
        controller.ready_maximum_heading_error = math.radians(30.0)
        controller.portal_registration_config = (
            controller_module.PortalRegistrationConfig()
        )

        controller.costmap = None
        controller.costmap_version = 1
        controller.planned_costmap_version = 1
        controller.planned_grid = None
        controller.soft_replan_contact_station = None
        controller.scan_updates = 3
        controller.minimum_planning_scan_updates = 3
        controller.scan_received = self.now
        controller.scan_stamp = self.now
        controller.minimum_initial_scans = 3
        controller.maximum_planning_range = 3.0
        controller.scan_transform_timeout = 0.05
        controller.maximum_scan_age = 0.25
        controller.maximum_future_stamp = 0.06
        controller.last_processed_scan_stamp = None
        controller.last_scan_summary = None
        controller.grid_cache = None
        controller.grid_cache_version = -1

        controller.map_path = None
        controller.odom_path = None
        controller.entry_staging_station = None
        controller.entry_inside_station = None
        controller.exit_connector_station = None
        controller.path_index = 0
        controller.map_path_index = 0
        controller.remaining_distance = math.inf
        controller.last_position_error = 0.0
        controller.last_heading_error = 0.0
        controller.last_linear = 0.0
        controller.last_angular = 0.0
        controller.last_command_time = None
        controller.last_plan_attempt = None
        controller.planning_generation = 0
        controller.planning_thread = None
        controller.last_plan_seconds = 0.0
        controller.last_expanded_nodes = 0
        controller.plan_attempts = 0
        controller.replan_count = 0
        controller.prepared_arm_generation = 1
        controller.prepared_tunnel_path = None
        controller.prepared_exit_connector_station = None
        controller.prepared_grid = None
        controller.prepared_costmap_version = 1
        controller.prepared_soft_contact_station = None
        controller.prepared_plan_kind = "dynamic"
        controller.dynamic_preplan_generation = 0
        controller.preplan_cap_generation = 0
        controller.maximum_position_error_seen = 0.0
        controller.maximum_heading_error_seen = 0.0
        controller.amcl_anchor_position_delta = 0.0
        controller.amcl_anchor_heading_delta = 0.0
        controller.last_motion_safety_failure = ""
        controller.sign_seen = False
        controller.sign_confidence = 0.0

        controller.lane_path_stamp = None
        controller.lane_path_valid = False
        controller.lane_path_minimum_line_clearance = math.nan
        controller.lane_path_confirmation_count = 0
        controller.last_lane_path_confirmation_time = None
        controller.lane_command_linear = math.nan
        controller.lane_command_angular = math.nan
        controller.lane_command_stamp = None
        controller.exit_confirmation_started = False
        controller.confirmation_started_at = None
        controller.lane_handoff_retry_pending = False
        controller.join_start_x = 0.0
        controller.join_start_y = 0.0
        controller.join_start_yaw = 0.0
        controller.join_origin_ready = False

        controller.entry_velocity_cap = 0.04
        controller.preplan_lane_velocity_cap = 0.025
        controller.entry_handoff_velocity_tolerance = 0.005
        controller.lane_resume_max_velocity = 0.30
        controller.cruise_velocity = 0.075
        controller.minimum_velocity = 0.035
        controller.entry_velocity = 0.035
        controller.exit_velocity = 0.075
        controller.entry_portal_velocity = 0.035
        controller.join_velocity_cap = 0.06
        controller.acquisition_timeout = 2.0
        controller.entry_alignment_timeout = 12.0
        controller.entry_straight_timeout = 15.0
        controller.planning_timeout = 15.0
        controller.mission_timeout = 120.0
        controller.exit_straight_timeout = 12.0
        controller.join_timeout = 2.0
        controller.handoff_timeout = 0.30
        controller.map_pose_timeout = 0.40
        controller.maximum_pose_stamp_skew = 0.05
        controller.odom_timeout = 0.35
        controller.scan_timeout = 0.40
        controller.plan_retry_period = 0.50
        controller.safety_reaction_time = 0.10
        controller.safety_distance_margin = 0.01
        controller.linear_deceleration = 0.40
        controller.angular_acceleration = 0.55
        controller.safety_linear_deceleration = 0.15
        controller.safety_angular_deceleration = 0.55
        controller.planning_stopped_linear = 0.008
        controller.planning_stopped_angular = 0.03
        controller.control_period = 0.05
        controller.linear_acceleration = 0.15
        controller.maximum_angular_velocity = 0.55
        controller.maximum_lateral_acceleration = 0.03
        controller.lookahead_distance = 0.065
        controller.heading_gain = 0.30
        controller.path_curvature_weight = 0.20
        controller.nearest_search_ahead = 0.40
        controller.soft_replan_lookahead_distance = 0.50
        controller.soft_start_prefix_max_distance = 0.35
        controller.tracking_position_tolerance = 0.08
        controller.tracking_heading_tolerance = math.radians(35.0)
        controller.portal_sample_step = 0.01
        controller.entry_tangent_ratio = 0.20
        controller.entry_minimum_tangent_length = 0.008
        controller.entry_maximum_tangent_length = 0.055
        controller.entry_path_position_tolerance = 0.015
        controller.entry_path_heading_tolerance = math.radians(2.0)
        controller.entry_path_remaining_tolerance = 0.015
        controller.entry_staging_pose = controller_module.Pose2D(
            -1.7475895, 0.14, -0.5 * math.pi
        )
        controller.entry_inside_pose = controller_module.Pose2D(
            -1.7475895, -0.28, -0.5 * math.pi
        )
        controller.prepared_tunnel_path = controller_module.path_from_poses(
            np.asarray(
                (
                    (
                        controller.entry_inside_pose.x,
                        controller.entry_inside_pose.y,
                        controller.entry_inside_pose.yaw,
                    ),
                    (
                        controller.entry_inside_pose.x,
                        controller.entry_inside_pose.y - 0.10,
                        controller.entry_inside_pose.yaw,
                    ),
                )
            ),
            frame_id="map",
            target_speed=controller.entry_velocity,
            label="test_tunnel_preplan",
        )
        controller.prepared_exit_connector_station = 0.05
        controller.prepared_grid = object()
        controller.entry_portal_plane_y = -0.105857
        controller.registration_reference_pose = controller_module.Pose2D(
            -1.757968, -0.041389, math.radians(-89.4336)
        )
        controller.entry_clearance_margin = 0.010
        controller.front = 0.067645
        controller.rear = 0.118073
        controller.half_width = 0.0903
        controller.footprint_padding = 0.010
        controller.exit_portal_plane_x = 0.003697
        controller.exit_clearance_margin = 0.010
        controller.exit_outside_pose = controller_module.Pose2D(1.0, 0.0, 0.0)
        controller.exit_connector_tangent_ratio = 0.20
        controller.exit_connector_minimum_tangent_length = 0.008
        controller.exit_connector_maximum_tangent_length = 0.055
        controller.exit_connector_sample_step = 0.010
        controller.handoff_remaining_distance = 0.035
        controller.exit_position_tolerance = 0.060
        controller.exit_heading_tolerance = math.radians(10.0)
        controller.goal = controller_module.Pose2D(0.0, 0.0, 0.0)
        controller.map_frame = "map"
        controller.planner = SimpleNamespace(
            pose_is_collision_free=lambda _grid, _pose: True,
            primitive_is_collision_free=(
                lambda _grid, _pose, _curvature, _distance: True
            ),
            minimum_turning_radius=0.18,
            collision_check_step=0.01,
            collision_check_angle=math.radians(3.0),
        )

        controller.exit_confirmation_frames = 2
        controller.join_confirmation_frames = 2
        controller.exit_confirmation_max_gap = 0.20
        controller.lane_path_timeout = 0.35
        controller.join_minimum_distance = 0.04

        controller.lane_service_name = "/control/lane_mission_handoff"
        controller.lane_stop_service_name = "/control/lane_following"
        controller.events = []
        controller.lane_service = RecordingLaneService(controller.events)
        controller.lane_stop_service = RecordingLaneStopService(
            controller.events
        )
        controller.cmd_pub = RecordingPublisher(controller.events, "/cmd_vel")
        controller.speed_limit_pub = RecordingPublisher(
            controller.events, "/control/max_vel"
        )
        controller.state_pub = RecordingPublisher()
        controller.path_pub = RecordingPublisher()
        controller.costmap_pub = RecordingPublisher()
        controller.diagnostics_pub = RecordingPublisher()
        controller.ready_pub = RecordingPublisher()
        return controller

    @staticmethod
    def assert_zero(command):
        assert command.linear.x == 0.0
        assert command.angular.z == 0.0

    @staticmethod
    def valid_lane_path(line_clearance=0.010):
        diagnostics = controller_module.PathDiagnostics(
            progress=0.1,
            remaining_distance=0.5,
            position_error=0.001,
            cross_track_error=0.001,
            heading_error=0.01,
            target_speed=0.1,
            target_index=2,
            curvature=0.0,
            minimum_line_clearance=float(line_clearance),
            minimum_obstacle_clearance=math.inf,
            minimum_map_clearance=math.inf,
            commanded_linear=0.0,
            commanded_angular=0.0,
        )
        return Float64MultiArray(data=diagnostics.as_array())

    @staticmethod
    def invalid_lane_path():
        return TunnelMissionControllerTest.valid_lane_path(
            line_clearance=-0.001
        )

    def make_rolling_exit_controller(self):
        controller = self.make_controller()
        controller.state = controller.EXITING
        controller.state_started = self.now
        controller.mission_started = self.now
        controller.mission_has_control = True
        controller.map_from_odom = (0.0, 0.0, 0.0)
        controller.frozen_odom_frame = "odom"
        controller.odom_x = 0.5
        controller.odom_linear_velocity = 0.075
        path = controller_module.CommonPath(
            x=np.asarray([0.0, 0.5, 1.0]),
            y=np.zeros(3),
            heading=np.zeros(3),
            curvature=np.zeros(3),
            speed=np.full(3, 0.075),
            label="tunnel_hybrid_moving_exit",
        )
        controller.map_path = path
        controller.odom_path = path
        controller.exit_connector_station = 0.5
        controller.path_index = 1
        controller.remaining_distance = 0.5
        controller.last_linear = 0.075
        controller.last_command_time = controller_module.rospy.Time.from_sec(
            self.seconds - controller.control_period
        )
        controller.exit_confirmation_started = True
        controller.confirmation_started_at = self.now
        controller._input_problem = lambda now, require_scan=True: None
        controller._planner_grid = lambda: object()
        controller._command_is_safe = mock.Mock(return_value=True)
        return controller

    @staticmethod
    def rolling_exit_tracking():
        return SimpleNamespace(
            path_index=1,
            position_error=0.0,
            heading_error=0.0,
            target_speed=0.075,
            angular_velocity=0.0,
        )

    def odometry(self, x, y=0.0, yaw=0.0):
        message = Odometry()
        message.header.stamp = self.now
        message.header.frame_id = "odom"
        message.pose.pose.position.x = float(x)
        message.pose.pose.position.y = float(y)
        message.pose.pose.orientation.z = math.sin(0.5 * yaw)
        message.pose.pose.orientation.w = math.cos(0.5 * yaw)
        return message

    def scan(self, stamp=None):
        message = LaserScan()
        message.header.stamp = self.now if stamp is None else stamp
        message.header.frame_id = "/laser"
        message.angle_min = -0.10
        message.angle_increment = 0.10
        message.angle_max = 0.10
        message.range_min = 0.05
        message.range_max = 10.0
        message.ranges = [1.0, 1.1, 1.2]
        return message

    @staticmethod
    def production_grid_and_planner():
        with (PACKAGE_DIR / "config" / "tunnel_mission_gazebo.yaml").open(
            encoding="utf-8"
        ) as stream:
            config = yaml.safe_load(stream)["tunnel"]
        with (PACKAGE_DIR / "maps" / "autorace_gazebo.yaml").open(
            encoding="utf-8"
        ) as stream:
            map_config = yaml.safe_load(stream)

        image = np.asarray(
            Image.open(PACKAGE_DIR / "maps" / map_config["image"]),
            dtype=np.float64,
        )
        if map_config.get("negate", 0):
            occupancy_probability = image / 255.0
        else:
            occupancy_probability = (255.0 - image) / 255.0
        static_data = np.full(image.shape, -1, dtype=np.int8)
        static_data[
            occupancy_probability > float(map_config["occupied_thresh"])
        ] = 100
        static_data[
            occupancy_probability < float(map_config["free_thresh"])
        ] = 0
        static_data = np.flipud(static_data).reshape(-1).tolist()

        map_values = config["map"]
        origin = map_config["origin"]
        # nav_msgs/MapMetaData carries resolution as float32.  Reproduce that
        # wire quantization before cropping; using YAML's Python float selects
        # one fewer column at the production max-x boundary.
        map_resolution = float(np.float32(map_config["resolution"]))
        costmap = controller_module.TunnelCostmap(
            static_data=static_data,
            width=image.shape[1],
            height=image.shape[0],
            resolution=map_resolution,
            origin_x=origin[0],
            origin_y=origin[1],
            planning_bounds=map_values["planning_bounds"],
            static_occupied_threshold=map_values["static_occupied_threshold"],
            mark_observations=map_values["mark_observations"],
            clear_observations=map_values["clear_observations"],
            decay_updates=map_values["decay_updates"],
            dynamic_occupied_value=map_values["dynamic_occupied_value"],
            dynamic_inflation_value=map_values[
                "dynamic_inflation_value"
            ],
            static_hit_exclusion_radius=map_values[
                "static_hit_exclusion_radius"
            ],
            endpoint_clear_guard_radius=map_values[
                "endpoint_clear_guard_radius"
            ],
            dynamic_inflation_radius=map_values["dynamic_inflation_radius"],
        )
        grid = controller_module.OccupancyGrid.from_flat(
            costmap.to_occupancy_data(),
            width=costmap.width,
            height=costmap.height,
            resolution=costmap.resolution,
            origin_x=costmap.origin_x,
            origin_y=costmap.origin_y,
            occupied_threshold=map_values["static_occupied_threshold"],
            unknown_is_occupied=map_values["unknown_is_occupied"],
            soft_cost_data=costmap.to_soft_cost_data(),
        )

        footprint = config["footprint"]
        planner_values = config["planner"]
        planner = controller_module.HybridAStarPlanner(
            footprint=controller_module.RectangularFootprint(**footprint),
            heading_bins=planner_values["heading_bins"],
            minimum_turning_radius=planner_values["minimum_turning_radius"],
            primitive_step=planner_values["primitive_step"],
            steering_samples=planner_values["steering_samples"],
            goal_position_tolerance=planner_values["goal_position_tolerance"],
            goal_heading_tolerance=math.radians(
                planner_values["goal_heading_tolerance_deg"]
            ),
            goal_curvature_tolerance=planner_values[
                "goal_curvature_tolerance"
            ],
            non_straight_penalty=planner_values["non_straight_penalty"],
            steering_change_penalty=planner_values["steering_change_penalty"],
            inflation_radius=planner_values["static_inflation_radius"],
            obstacle_cost_weight=planner_values["obstacle_cost_weight"],
            obstacle_cost_distance=planner_values["obstacle_cost_distance"],
            soft_obstacle_cost_weight=planner_values[
                "soft_obstacle_cost_weight"
            ],
            soft_cost_check_step=planner_values["soft_cost_check_step"],
            heading_heuristic_weight=planner_values[
                "heading_heuristic_weight"
            ],
            collision_check_step=planner_values["collision_check_step"],
            collision_check_angle=math.radians(
                planner_values["collision_check_angle_deg"]
            ),
            path_sample_step=planner_values["path_sample_step"],
            state_xy_resolution=planner_values["state_xy_resolution"],
            maximum_iterations=planner_values["maximum_iterations"],
            keep_in_rectangles=planner_values["keep_in_rectangles"],
        )
        goal_values = config["goal"]
        goal = controller_module.Pose2D(
            goal_values["x"],
            goal_values["y"],
            math.radians(goal_values["yaw_deg"]),
        )
        return config, costmap, grid, planner, goal

    def replay_ideal_common_path(
        self,
        config,
        grid,
        planner,
        start,
        path,
        remaining_tolerance,
        position_tolerance,
        heading_tolerance,
        maximum_seconds,
    ):
        """Replay the production tracker and stop sweep on an ideal robot."""
        control = config["control"]
        controller = self.make_controller()
        controller.planner = planner
        controller.map_from_odom = (0.0, 0.0, 0.0)
        controller.odom_path = path
        controller.control_period = control["period"]
        controller.lookahead_distance = control["lookahead_distance"]
        controller.maximum_angular_velocity = control[
            "maximum_angular_velocity"
        ]
        controller.maximum_lateral_acceleration = control[
            "maximum_lateral_acceleration"
        ]
        controller.linear_acceleration = control["linear_acceleration"]
        controller.linear_deceleration = control["linear_deceleration"]
        controller.angular_acceleration = control["angular_acceleration"]
        controller.heading_gain = control["heading_gain"]
        controller.path_curvature_weight = control["path_curvature_weight"]
        controller.nearest_search_ahead = control["nearest_search_ahead"]
        controller.tracking_position_tolerance = control[
            "tracking_position_tolerance"
        ]
        controller.tracking_heading_tolerance = math.radians(
            control["tracking_heading_tolerance_deg"]
        )
        controller.safety_reaction_time = control["safety_reaction_time"]
        controller.safety_distance_margin = control["safety_distance_margin"]
        controller.safety_linear_deceleration = control[
            "safety_linear_deceleration"
        ]
        controller.safety_angular_deceleration = control[
            "safety_angular_deceleration"
        ]
        pose = controller_module.Pose2D(start.x, start.y, start.yaw)
        measured_linear = 0.0
        measured_angular = 0.0
        last_linear = 0.0
        last_angular = 0.0
        path_index = 0
        trace = []
        maximum_ticks = int(math.ceil(maximum_seconds / controller.control_period))
        for tick in range(maximum_ticks):
            controller.odom_x = pose.x
            controller.odom_y = pose.y
            controller.odom_yaw = pose.yaw
            if not planner.pose_is_collision_free(grid, pose):
                return (
                    False,
                    "current footprint",
                    tick * controller.control_period,
                    trace,
                )

            tracking = controller_module.calculate_tracking(
                path,
                pose.x,
                pose.y,
                pose.yaw,
                path_index,
                controller.lookahead_distance,
                controller.maximum_angular_velocity,
                controller.heading_gain,
                controller.path_curvature_weight,
                controller.nearest_search_ahead,
            )
            path_index = tracking.path_index
            trace.append((float(path.station[path_index]), pose))
            controller.remaining_distance = max(
                0.0,
                path.length - float(path.station[path_index]),
            )
            if (
                tracking.position_error
                > controller.tracking_position_tolerance
            ):
                return (
                    False,
                    "tracking position",
                    tick * controller.control_period,
                    trace,
                )
            if (
                abs(tracking.heading_error)
                > controller.tracking_heading_tolerance
            ):
                return (
                    False,
                    "tracking heading",
                    tick * controller.control_period,
                    trace,
                )
            if controller._path_endpoint_ready(
                remaining_tolerance,
                position_tolerance,
                heading_tolerance,
            ):
                return True, "endpoint", tick * controller.control_period, trace

            command = controller_module.limit_tracking_command(
                target_speed=tracking.target_speed,
                reference_speed=tracking.target_speed,
                target_angular_velocity=tracking.angular_velocity,
                last_linear_velocity=last_linear,
                last_angular_velocity=last_angular,
                elapsed=controller.control_period,
                linear_acceleration=controller.linear_acceleration,
                linear_deceleration=controller.linear_deceleration,
                angular_acceleration=controller.angular_acceleration,
                maximum_angular_velocity=controller.maximum_angular_velocity,
                maximum_lateral_acceleration=(
                    controller.maximum_lateral_acceleration
                ),
            )
            if not controller._motion_is_safe(
                grid, pose, measured_linear, measured_angular
            ):
                return (
                    False,
                    "measured stop sweep: %s"
                    % controller.last_motion_safety_failure,
                    tick * controller.control_period,
                    trace,
                )
            if not controller._motion_is_safe(
                grid,
                pose,
                command.linear_velocity,
                command.angular_velocity,
            ):
                return (
                    False,
                    "requested stop sweep: %s"
                    % controller.last_motion_safety_failure,
                    tick * controller.control_period,
                    trace,
                )

            measured_linear = command.linear_velocity
            measured_angular = command.angular_velocity
            last_linear = command.linear_velocity
            last_angular = command.angular_velocity
            pose = controller_module.propagate_twist(
                pose,
                command.linear_velocity,
                command.angular_velocity,
                controller.control_period,
            )
        return False, "replay timeout", maximum_seconds, trace

    def replay_ideal_tunnel_tracking(self, config, grid, planner, start, plan):
        """Replay the mandatory Hybrid A* plus rolling-exit path."""
        control = config["control"]
        exit_values = config["exit"]
        terminal = controller_module.Pose2D(
            float(plan.x[-1]),
            float(plan.y[-1]),
            float(plan.yaw[-1]),
        )
        outside = controller_module.pose_from_degrees(
            exit_values["outside_pose"], "exit/outside_pose"
        )
        connector = controller_module.portal_alignment_path(
            terminal,
            outside,
            exit_values["connector_tangent_ratio"],
            exit_values["minimum_tangent_length"],
            exit_values["maximum_tangent_length"],
            exit_values["connector_sample_step"],
            control["exit_velocity"],
            "map",
            label="tunnel_moving_exit_connector",
        )
        path, _ = controller_module.tracking_path_with_exit_connector(
            plan,
            connector,
            control["cruise_velocity"],
            control["minimum_velocity"],
            control["entry_velocity"],
            control["exit_velocity"],
            control["maximum_angular_velocity"],
            control["maximum_lateral_acceleration"],
            control["linear_acceleration"],
            control["linear_deceleration"],
            control["angular_acceleration"],
            "map",
        )
        checker = self.make_controller()
        checker.planner = planner
        self.assertTrue(checker._path_is_safe(grid, path, 0))
        replayed, reason, _, _ = self.replay_ideal_common_path(
            config,
            grid,
            planner,
            start,
            path,
            exit_values["handoff_remaining_distance"],
            exit_values["position_tolerance"],
            math.radians(exit_values["heading_tolerance_deg"]),
            config["timeouts"]["mission"],
        )
        return replayed, reason

    def test_production_helper_matches_ros_occupancy_grid_wire_geometry(self):
        _, costmap, grid, planner, goal = self.production_grid_and_planner()
        wire_resolution = float(np.float32(0.02))
        expected_bounds = (
            -2.5 + 24 * wire_resolution,
            -2.5 + 24 * wire_resolution,
            -2.5 + 141 * wire_resolution,
            -2.5 + 153 * wire_resolution,
        )

        self.assertEqual(costmap.shape, (129, 117))
        self.assertEqual((grid.height, grid.width), costmap.shape)
        self.assertEqual(costmap.resolution, wire_resolution)
        self.assertEqual(grid.resolution, wire_resolution)
        for actual, expected in zip(costmap.bounds, expected_bounds):
            self.assertAlmostEqual(actual, expected, places=12)
        self.assertEqual(
            (grid.origin_x, grid.origin_y, grid.maximum_x, grid.maximum_y),
            costmap.bounds,
        )
        self.assertTrue(
            planner.pose_is_collision_free(
                grid,
                controller_module.Pose2D(
                    -1.8434, 0.3652, math.radians(-88.28)
                ),
            )
        )
        self.assertTrue(planner.pose_is_collision_free(grid, goal))
        self.assertFalse(
            planner.pose_is_within_keep_in(
                controller_module.Pose2D(
                    0.0043, 0.1002, math.radians(-27.21)
                )
            )
        )

    def test_production_entry_centres_before_the_only_safe_straight_crossing(self):
        config, _, grid, planner, _ = self.production_grid_and_planner()
        entry = config["entry"]
        staging = controller_module.pose_from_degrees(
            entry["staging_pose"], "entry staging"
        )
        inside = controller_module.pose_from_degrees(
            entry["inside_pose"], "entry inside"
        )
        # Latest official-start failure anchored 64 mm left of the surveyed
        # aperture centre. Even after correcting yaw, going straight from
        # that x coordinate intersects the west-side portal wall.
        recorded_anchor = controller_module.Pose2D(
            -1.8115, 0.3648, math.radians(-88.64)
        )
        off_centre_start = controller_module.Pose2D(
            recorded_anchor.x, recorded_anchor.y, staging.yaw
        )

        def forward_distance(start, target, yaw):
            return (
                (target.x - start.x) * math.cos(yaw)
                + (target.y - start.y) * math.sin(yaw)
            )

        unsafe_straight = controller_module.portal_straight_path(
            off_centre_start,
            staging.yaw,
            forward_distance(off_centre_start, inside, staging.yaw),
            entry["sample_step"],
            entry["velocity"],
            "map",
            "unsafe_off_centre_entry",
        )
        aligned_entry, staging_station = controller_module.portal_entry_path(
            recorded_anchor,
            staging,
            inside,
            entry["connector_tangent_ratio"],
            entry["minimum_tangent_length"],
            entry["maximum_tangent_length"],
            entry["sample_step"],
            entry["velocity"],
            "map",
        )
        safe_straight = controller_module.portal_straight_path(
            staging,
            staging.yaw,
            forward_distance(staging, inside, staging.yaw),
            entry["sample_step"],
            entry["velocity"],
            "map",
            "surveyed_straight_entry",
        )
        controller = self.make_controller()
        controller.planner = planner

        self.assertFalse(controller._path_is_safe(grid, unsafe_straight, 0))
        self.assertTrue(controller._path_is_safe(grid, aligned_entry, 0))
        self.assertTrue(controller._path_is_safe(grid, safe_straight, 0))
        self.assertGreater(staging_station, 0.0)
        self.assertLess(staging_station, aligned_entry.length)
        self.assertAlmostEqual(aligned_entry.x[-1], inside.x)
        self.assertAlmostEqual(aligned_entry.y[-1], inside.y)
        suffix = aligned_entry.station >= staging_station - 1e-9
        self.assertTrue(np.allclose(aligned_entry.x[suffix], staging.x))
        self.assertTrue(np.allclose(aligned_entry.heading[suffix], staging.yaw))
        self.assertTrue(np.allclose(safe_straight.x, staging.x))
        self.assertTrue(np.allclose(safe_straight.heading, staging.yaw))

    def test_production_aligned_entry_replays_recorded_gate_poses(self):
        config, _, grid, planner, _ = self.production_grid_and_planner()
        entry = config["entry"]
        staging = controller_module.pose_from_degrees(
            entry["staging_pose"], "entry staging"
        )
        inside = controller_module.pose_from_degrees(
            entry["inside_pose"], "entry inside"
        )
        anchors = (
            (-1.8115, 0.3648, -88.64),
            (-1.8102, 0.3501, -87.57),
            (-1.7505, 0.1598, -90.11),
            (-1.8434, 0.3652, -88.28),
        )
        maximum_seconds = (
            config["timeouts"]["entry_alignment"]
            + config["timeouts"]["entry_straight"]
        )
        front_crossing_y = (
            entry["portal_plane_y"]
            + config["footprint"]["front"]
            + config["footprint"]["padding"]
        )

        for x, y, yaw_deg in anchors:
            with self.subTest(anchor=(x, y, yaw_deg)):
                start = controller_module.Pose2D(
                    x, y, math.radians(yaw_deg)
                )
                path, staging_station = controller_module.portal_entry_path(
                    start,
                    staging,
                    inside,
                    entry["connector_tangent_ratio"],
                    entry["minimum_tangent_length"],
                    entry["maximum_tangent_length"],
                    entry["sample_step"],
                    entry["velocity"],
                    "map",
                )
                replayed, reason, elapsed, trace = (
                    self.replay_ideal_common_path(
                        config,
                        grid,
                        planner,
                        start,
                        path,
                        entry["path_remaining_tolerance"],
                        entry["path_position_tolerance"],
                        math.radians(entry["path_heading_tolerance_deg"]),
                        maximum_seconds,
                    )
                )
                self.assertTrue(replayed, reason)
                seam_index = next(
                    index
                    for index, (station, _) in enumerate(trace)
                    if station >= staging_station
                )
                seam_seconds = seam_index * config["control"]["period"]
                self.assertLess(
                    seam_seconds, config["timeouts"]["entry_alignment"]
                )
                self.assertLess(
                    elapsed - seam_seconds,
                    config["timeouts"]["entry_straight"],
                )
                crossing_pose = next(
                    pose for _, pose in trace if pose.y <= front_crossing_y
                )
                self.assertLessEqual(abs(crossing_pose.x - staging.x), 0.010)
                self.assertLessEqual(
                    abs(
                        controller_module.normalize_angle(
                            crossing_pose.yaw - staging.yaw
                        )
                    ),
                    math.radians(2.0),
                )

    def test_production_keep_in_routes_through_unknown_obstacle_layouts(self):
        config, _, base_grid, planner, goal = self.production_grid_and_planner()
        obstacle_origin = np.asarray((-1.04145, -1.12671))
        obstacle_offsets = np.asarray(
            (
                (0.423220, -0.212034),
                (-0.323729, -0.287563),
                (-0.099492, 0.499594),
            )
        )
        rows, columns = np.indices(base_grid.data.shape)
        cell_x = base_grid.origin_x + (columns + 0.5) * base_grid.resolution
        cell_y = base_grid.origin_y + (rows + 0.5) * base_grid.resolution
        start = controller_module.pose_from_degrees(
            config["entry"]["inside_pose"], "entry inside"
        )

        def grid_for(rotation_deg, obstacle_radius):
            angle = math.radians(rotation_deg)
            rotation = np.asarray(
                (
                    (math.cos(angle), -math.sin(angle)),
                    (math.sin(angle), math.cos(angle)),
                )
            )
            centres = obstacle_origin + obstacle_offsets @ rotation.T
            data = base_grid.data.copy()
            for center_x, center_y in centres:
                occupied = (
                    (cell_x - center_x) ** 2
                    + (cell_y - center_y) ** 2
                    <= obstacle_radius ** 2 + 1e-15
                )
                data[occupied] = 100
            return controller_module.OccupancyGrid(
                data,
                base_grid.resolution,
                base_grid.origin_x,
                base_grid.origin_y,
                occupied_threshold=config["map"][
                    "static_occupied_threshold"
                ],
                unknown_is_occupied=config["map"]["unknown_is_occupied"],
            )

        # The runtime knows none of these centres. Full 170 mm disks are only
        # a conservative regression fixture for five distinct live layouts:
        # the saved Gazebo state and 45-degree compound rotations through 180.
        for rotation_deg in (0.0, 45.0, 90.0, 135.0, 180.0):
            with self.subTest(rotation_deg=rotation_deg):
                grid = grid_for(rotation_deg, 0.17)

                path = planner.plan(grid, start, goal, goal_curvature=0.0)

                self.assertIsNotNone(path)
                self.assertLess(path.length, 4.0)
                self.assertLess(path.expanded_nodes, 15000)
                self.assertTrue(
                    all(
                        planner.pose_is_within_keep_in(pose)
                        for pose in path.poses
                    )
                )
                replayed, reason = self.replay_ideal_tunnel_tracking(
                    config, grid, planner, start, path
                )
                self.assertTrue(replayed, reason)

        # The shorter, inside-to-inside Hybrid segment also remains executable
        # for the legacy 160 mm regression fixture.
        legacy_grid = grid_for(180.0, 0.16)
        legacy_path = planner.plan(
            legacy_grid, start, goal, goal_curvature=0.0
        )
        self.assertIsNotNone(legacy_path)
        replayed, reason = self.replay_ideal_tunnel_tracking(
            config, legacy_grid, planner, start, legacy_path
        )
        self.assertTrue(replayed, reason)

    def test_production_soft_halo_prefers_clearance_in_five_layouts(self):
        config, _, base_grid, planner, goal = self.production_grid_and_planner()
        obstacle_origin = np.asarray((-1.04145, -1.12671))
        obstacle_offsets = np.asarray(
            (
                (0.423220, -0.212034),
                (-0.323729, -0.287563),
                (-0.099492, 0.499594),
            )
        )
        rows, columns = np.indices(base_grid.data.shape)
        cell_x = base_grid.origin_x + (columns + 0.5) * base_grid.resolution
        cell_y = base_grid.origin_y + (rows + 0.5) * base_grid.resolution
        start = controller_module.pose_from_degrees(
            config["entry"]["inside_pose"], "entry inside"
        )

        def graded_grid(rotation_deg):
            angle = math.radians(rotation_deg)
            rotation = np.asarray(
                (
                    (math.cos(angle), -math.sin(angle)),
                    (math.sin(angle), math.cos(angle)),
                )
            )
            centres = obstacle_origin + obstacle_offsets @ rotation.T
            data = base_grid.data.copy()
            soft = np.zeros(data.shape, dtype=np.int8)
            for center_x, center_y in centres:
                squared_distance = (
                    (cell_x - center_x) ** 2
                    + (cell_y - center_y) ** 2
                )
                raw = squared_distance <= 0.11 ** 2 + 1e-15
                inner = (
                    (squared_distance > 0.11 ** 2 + 1e-15)
                    & (squared_distance <= 0.13 ** 2 + 1e-15)
                )
                middle = (
                    (squared_distance > 0.13 ** 2 + 1e-15)
                    & (squared_distance <= 0.15 ** 2 + 1e-15)
                )
                outer = (
                    (squared_distance > 0.15 ** 2 + 1e-15)
                    & (squared_distance <= 0.17 ** 2 + 1e-15)
                )
                data[raw] = 100
                soft[inner & (data < 65) & (data >= 0)] = np.maximum(
                    soft[inner & (data < 65) & (data >= 0)], 64
                )
                soft[middle & (data < 65) & (data >= 0)] = np.maximum(
                    soft[middle & (data < 65) & (data >= 0)], 43
                )
                soft[outer & (data < 65) & (data >= 0)] = np.maximum(
                    soft[outer & (data < 65) & (data >= 0)], 21
                )
            return controller_module.OccupancyGrid(
                data,
                base_grid.resolution,
                base_grid.origin_x,
                base_grid.origin_y,
                occupied_threshold=config["map"][
                    "static_occupied_threshold"
                ],
                unknown_is_occupied=config["map"]["unknown_is_occupied"],
                soft_cost_data=soft,
            )

        self.assertEqual(planner.soft_obstacle_cost_weight, 1.0)
        for rotation_deg in (0.0, 45.0, 90.0, 135.0, 180.0):
            with self.subTest(rotation_deg=rotation_deg):
                grid = graded_grid(rotation_deg)

                path = planner.plan(grid, start, goal, goal_curvature=0.0)

                self.assertIsNotNone(path)
                self.assertLess(path.length, 3.5)
                self.assertLess(path.expanded_nodes, 15000)
                self.assertTrue(
                    all(
                        planner.pose_is_within_keep_in(pose)
                        for pose in path.poses
                    )
                )
                exposure = 0.0
                for index in range(path.x.size - 1):
                    segment_start = path.poses[index]
                    distance = math.hypot(
                        float(path.x[index + 1] - path.x[index]),
                        float(path.y[index + 1] - path.y[index]),
                    )
                    curvature = float(path.curvature[index])
                    self.assertTrue(
                        planner.primitive_is_collision_free(
                            grid,
                            segment_start,
                            curvature,
                            distance,
                        )
                    )
                    exposure += planner._primitive_soft_cost_exposure(
                        grid,
                        segment_start,
                        curvature,
                        distance,
                    )
                self.assertLessEqual(exposure, 1e-12)

    def test_bag006_halo_only_start_remains_raw_safe_and_plannable(self):
        config, costmap, _, planner, goal = self.production_grid_and_planner()
        # Confirmed dynamic endpoints in the final v198 snapshot of the
        # official-start 006 bag, indexed in the cropped 117 x 129 grid.
        raw_indices = (
            2608, 2726, 2727, 2843, 2844, 2845, 2961, 2962, 2963,
            3078, 3079, 3080, 3111, 3196, 3197, 3225, 3226, 3227,
            3312, 3313, 3314, 3340, 3341, 3342, 3421, 3429, 3430,
            3431, 3457, 3458, 3459, 3539, 3545, 3546, 3547, 3574,
            3575, 3576, 3657, 3658, 3659, 3660, 3661, 3662, 3663,
            3664, 3690, 3691, 3692, 3693, 3778, 3779, 3780, 3808,
            3809, 3925, 3926, 3927, 4042, 4043, 4044, 4160, 4161,
            4162, 4163, 7178, 7179, 7180, 7181, 7182, 7183, 7293,
            7294, 7295, 7296, 7297, 7298, 7299, 7300, 7410, 7411,
            7412, 7413, 7414, 7415, 7416, 7417, 7418, 7527, 7528,
            7535, 7536, 7644, 7652, 7653, 7761, 7770, 7878, 7995,
            8112, 8348, 8349, 8350,
        )
        costmap._dynamic_occupied = bytearray(costmap.width * costmap.height)
        for index in raw_indices:
            costmap._dynamic_occupied[index] = 1
        grid = controller_module.OccupancyGrid.from_flat(
            costmap.to_occupancy_data(),
            width=costmap.width,
            height=costmap.height,
            resolution=costmap.resolution,
            origin_x=costmap.origin_x,
            origin_y=costmap.origin_y,
            occupied_threshold=config["map"]["static_occupied_threshold"],
            unknown_is_occupied=config["map"]["unknown_is_occupied"],
            soft_cost_data=costmap.to_soft_cost_data(),
        )
        start = controller_module.Pose2D(
            -0.8845451525965673,
            -1.5157538606558343,
            -0.9141153898684418,
        )

        self.assertTrue(planner.pose_is_collision_free(grid, start))
        self.assertFalse(grid.is_occupied_cell(62, 26))
        self.assertTrue(grid.is_occupied_cell(64, 28))
        self.assertAlmostEqual(
            planner._pose_soft_cost(grid, start), 21.0 / 64.0, places=12
        )

        controller = self.make_controller()
        controller.planner = planner
        control = config["control"]
        controller.safety_reaction_time = control["safety_reaction_time"]
        controller.safety_distance_margin = control["safety_distance_margin"]
        controller.safety_linear_deceleration = control[
            "safety_linear_deceleration"
        ]
        controller.safety_angular_deceleration = control[
            "safety_angular_deceleration"
        ]
        self.assertTrue(
            controller._motion_is_safe(
                grid,
                start,
                linear_velocity=0.06152,
                angular_velocity=-0.00817,
            ),
            controller.last_motion_safety_failure,
        )

        path = planner.plan(grid, start, goal, goal_curvature=0.0)

        self.assertIsNotNone(path)
        self.assertLess(path.length, 1.30)
        self.assertLess(path.expanded_nodes, 10000)
        stations = np.zeros(path.x.size, dtype=np.float64)
        if path.x.size > 1:
            stations[1:] = np.cumsum(
                np.hypot(np.diff(path.x), np.diff(path.y))
            )
        soft_indices = [
            index
            for index, pose in enumerate(path.poses)
            if planner._pose_soft_cost(grid, pose) > 0.0
        ]
        self.assertTrue(soft_indices)
        self.assertEqual(soft_indices[0], 0)
        self.assertEqual(
            soft_indices,
            list(range(soft_indices[-1] + 1)),
        )
        self.assertGreater(stations[soft_indices[-1]], 0.10)
        self.assertLess(stations[soft_indices[-1]], 0.20)
        for index in range(path.x.size - 1):
            distance = math.hypot(
                float(path.x[index + 1] - path.x[index]),
                float(path.y[index + 1] - path.y[index]),
            )
            self.assertTrue(
                planner.primitive_is_collision_free(
                    grid,
                    path.poses[index],
                    float(path.curvature[index]),
                    distance,
                )
            )

    def test_gate_handoff_preserves_motion_and_enters_entry_alignment(self):
        controller = self.make_controller()
        controller._input_problem = lambda now, require_scan=True: None
        controller._planner_grid = lambda: object()
        controller._path_is_safe = lambda _grid, _path, _index: True
        controller._command_is_safe = lambda _grid, _linear, _angular: True
        controller.odom_linear_velocity = 0.035
        controller.lane_command_linear = 0.035
        controller.lane_command_angular = 0.08
        controller.lane_command_stamp = self.now

        controller.gate_callback(Bool(data=True))
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.ALIGNING_ENTRY)
        self.assertEqual(controller.lane_service.calls, [False])
        self.assertTrue(controller.mission_has_control)
        self.assertIsNotNone(controller.map_path)
        self.assertIsNotNone(controller.odom_path)
        handoff_index = controller.events.index(("handoff", False))
        first_command_event = next(
            event
            for event in controller.events[handoff_index + 1 :]
            if event[0:2] == ("publish", "/cmd_vel")
        )
        self.assertAlmostEqual(first_command_event[2].linear.x, 0.035)
        self.assertAlmostEqual(first_command_event[2].angular.z, 0.08)

    def test_gate_handoff_clamps_a_stale_fast_lane_command_and_offsets_soft_contact(self):
        controller = self.make_controller()
        controller._input_problem = lambda now, require_scan=True: None
        controller._planner_grid = lambda: object()
        controller._path_is_safe = lambda _grid, _path, _index: True
        controller._command_is_safe = lambda _grid, _linear, _angular: True
        controller.odom_linear_velocity = 0.035
        controller.lane_command_linear = 0.30
        controller.lane_command_angular = 0.30
        controller.lane_command_stamp = self.now
        controller.prepared_soft_contact_station = 0.40

        controller.gate_callback(Bool(data=True))
        controller.control_callback(None)

        command = controller.cmd_pub.messages[-1]
        self.assertAlmostEqual(command.linear.x, controller.entry_velocity_cap)
        self.assertAlmostEqual(command.angular.z, 0.04)
        self.assertAlmostEqual(
            controller.soft_replan_contact_station,
            controller.entry_inside_station + 0.40,
        )

    def test_entry_hybrid_seam_uses_the_first_planner_curvature(self):
        entry = controller_module.path_from_poses(
            np.asarray(((0.0, 0.0, 0.0), (0.10, 0.0, 0.0))),
            frame_id="map",
            target_speed=0.035,
        )
        distance = 0.05
        curvature = 2.0
        tail = controller_module.CommonPath(
            x=np.asarray(
                [0.10, 0.10 + math.sin(curvature * distance) / curvature]
            ),
            y=np.asarray(
                [0.0, (1.0 - math.cos(curvature * distance)) / curvature]
            ),
            heading=np.asarray([0.0, curvature * distance]),
            curvature=np.asarray([curvature, curvature]),
            speed=np.asarray([0.035, 0.035]),
            frame_id="map",
        )

        combined = controller_module.tracking_path_with_entry(
            entry,
            tail,
            cruise_velocity=0.075,
            minimum_velocity=0.035,
            entry_velocity=0.035,
            exit_velocity=0.075,
            maximum_angular_velocity=0.55,
            maximum_lateral_acceleration=0.03,
            linear_acceleration=0.15,
            linear_deceleration=0.40,
            maximum_angular_acceleration=0.55,
            frame_id="map",
        )

        seam_index = entry.x.size - 1
        self.assertAlmostEqual(combined.curvature[seam_index], curvature)

    def test_registration_generation_source_stamp_and_ready_identity(self):
        controller = self.make_controller()
        prepared_path = controller.prepared_tunnel_path
        controller.arm_generation = 0
        controller.armed_at = None
        controller.ready_published_generation = 0
        controller.registered_map_from_odom = None
        controller.registered_odom_frame = ""
        controller.registration_source_stamp = None
        controller.registration_candidates.clear()

        arm = Header()
        arm.seq = 7
        arm.stamp = controller_module.rospy.Time.from_sec(9.50)
        arm.frame_id = "tunnel"
        controller.arm_callback(arm)

        stale = controller_module.rospy.Time.from_sec(9.49)
        self.assertFalse(
            controller._submit_registration_candidate(
                (0.0, 0.0, 0.0), "odom", stale
            )
        )
        for stamp_value in (9.70, 9.80, 9.90):
            controller._submit_registration_candidate(
                (0.0, 0.0, 0.0),
                "odom",
                controller_module.rospy.Time.from_sec(stamp_value),
            )
        self.assertEqual(controller.registered_map_from_odom, (0.0, 0.0, 0.0))
        self.assertEqual(controller.ready_pub.messages, [])

        ready_stamp = controller_module.rospy.Time.from_sec(10.0)
        lead = 0.30
        odom_y = controller.entry_portal_plane_y + lead
        controller.odom_history.append(
            (
                ready_stamp,
                controller.entry_staging_pose.x,
                odom_y,
                controller.entry_staging_pose.yaw,
                "odom",
            )
        )
        controller.costmap = object()
        controller._planner_grid = lambda: object()
        controller._path_is_safe = lambda _grid, _path, _index: True
        controller.prepared_arm_generation = 7
        controller.prepared_tunnel_path = prepared_path
        controller.prepared_exit_connector_station = 0.05

        self.assertTrue(controller._try_publish_ready(ready_stamp))
        ready = controller.ready_pub.messages[-1]
        self.assertEqual(ready.seq, 7)
        self.assertEqual(ready.stamp, ready_stamp)
        self.assertEqual(ready.frame_id, "tunnel")

        next_arm = Header()
        next_arm.seq = 8
        next_arm.stamp = ready_stamp
        next_arm.frame_id = "tunnel"
        controller.costmap = None
        controller.arm_callback(next_arm)
        self.assertIsNone(controller.registered_map_from_odom)
        self.assertFalse(
            controller._submit_registration_candidate(
                (0.0, 0.0, 0.0), "odom", ready_stamp
            )
        )
        wrong_identity = Header()
        wrong_identity.seq = 9
        wrong_identity.stamp = ready_stamp
        wrong_identity.frame_id = "level_crossing"
        controller.arm_callback(wrong_identity)
        self.assertEqual(controller.arm_generation, 8)
        empty_identity = Header()
        empty_identity.seq = 9
        empty_identity.stamp = ready_stamp
        controller.arm_callback(empty_identity)
        self.assertEqual(controller.arm_generation, 8)

        inactive = Header()
        inactive.seq = 0
        inactive.stamp = ready_stamp
        inactive.frame_id = "tunnel"
        controller.arm_callback(inactive)
        self.assertEqual(controller.arm_generation, 0)
        self.assertIsNone(controller.armed_at)

    def test_arm_starts_static_preplan_without_taking_lane_control(self):
        controller = self.make_controller()
        controller.arm_generation = 0
        controller.armed_at = None
        controller.costmap = SimpleNamespace(reset_dynamic=mock.Mock())
        controller._publish_costmap = mock.Mock()
        controller._start_preplan = mock.Mock(return_value=True)
        arm = Header()
        arm.seq = 7
        arm.stamp = controller_module.rospy.Time.from_sec(9.50)
        arm.frame_id = "tunnel"

        controller.arm_callback(arm)

        controller.costmap.reset_dynamic.assert_called_once_with()
        controller._start_preplan.assert_called_once_with(
            arm.stamp, plan_kind="static"
        )
        self.assertEqual(controller.lane_service.calls, [])
        self.assertEqual(controller.cmd_pub.messages, [])

    def test_static_collision_starts_one_capped_dynamic_preplan(self):
        controller = self.make_controller()
        controller.arm_generation = 3
        controller.ready_published_generation = 0
        controller.prepared_arm_generation = 3
        controller.prepared_plan_kind = "static"
        controller.costmap = object()
        controller.registration_source_stamp = (
            controller_module.rospy.Time.from_sec(9.8)
        )
        controller.odom_history.append(
            (
                self.now,
                controller.entry_staging_pose.x,
                controller.entry_portal_plane_y + 0.30,
                controller.entry_staging_pose.yaw,
                "odom",
            )
        )
        controller._planner_grid = lambda: object()
        controller._path_is_safe = mock.Mock(return_value=False)
        controller._start_preplan = mock.Mock(return_value=True)

        self.assertFalse(controller._try_publish_ready(self.now))
        self.assertEqual(
            [message.data for message in controller.speed_limit_pub.messages],
            [controller.preplan_lane_velocity_cap],
        )
        self.assertEqual(controller.dynamic_preplan_generation, 3)
        controller._start_preplan.assert_called_once_with(
            self.now, plan_kind="dynamic"
        )
        self.assertIsNone(controller.prepared_tunnel_path)
        self.assertFalse(controller._try_publish_ready(self.now))
        self.assertEqual(len(controller.speed_limit_pub.messages), 1)
        self.assertEqual(controller._start_preplan.call_count, 1)
        self.assertEqual(controller.lane_service.calls, [])
        self.assertEqual(controller.cmd_pub.messages, [])

    def test_static_soft_contact_starts_dynamic_preplan_but_baseline_cost_does_not(self):
        controller = self.make_controller()
        controller.arm_generation = 3
        controller.ready_published_generation = 0
        controller.prepared_arm_generation = 3
        controller.prepared_plan_kind = "static"
        controller.costmap = object()
        controller.registration_source_stamp = (
            controller_module.rospy.Time.from_sec(9.8)
        )
        controller.odom_history.append(
            (
                self.now,
                controller.entry_staging_pose.x,
                controller.entry_portal_plane_y + 0.30,
                controller.entry_staging_pose.yaw,
                "odom",
            )
        )
        latest_grid = object()
        controller._planner_grid = lambda: latest_grid
        controller._path_is_safe = mock.Mock(return_value=True)
        controller._path_has_new_soft_cost = mock.Mock(return_value=False)

        self.assertTrue(controller._try_publish_ready(self.now))
        controller._path_has_new_soft_cost.assert_called_once_with(
            latest_grid,
            controller.prepared_tunnel_path,
            controller.prepared_grid,
        )
        self.assertEqual(controller.speed_limit_pub.messages, [])

        controller.ready_published_generation = 0
        controller._path_has_new_soft_cost = mock.Mock(return_value=True)
        controller._start_preplan = mock.Mock(return_value=True)

        self.assertFalse(controller._try_publish_ready(self.now))
        self.assertEqual(
            [message.data for message in controller.speed_limit_pub.messages],
            [controller.preplan_lane_velocity_cap],
        )
        controller._start_preplan.assert_called_once_with(
            self.now, plan_kind="dynamic"
        )
        self.assertIsNone(controller.prepared_tunnel_path)

    def test_acquire_rechecks_static_soft_contact_before_cmd_vel_handoff(self):
        controller = self.make_controller()
        controller.state = controller.ACQUIRING
        controller.state_started = controller_module.rospy.Time.from_sec(9.0)
        controller.zone_gate = True
        controller._input_problem = lambda now, require_scan=True: None
        controller._planner_grid = lambda: object()
        controller._prepared_path_needs_dynamic_preplan = mock.Mock(
            return_value=True
        )
        controller._restart_dynamic_preplan = mock.Mock(return_value=True)

        self.assertFalse(controller._acquire(self.now))

        self.assertEqual(controller.map_from_odom, (0.0, 0.0, 0.0))
        self.assertEqual(controller.frozen_odom_frame, "odom")
        controller._restart_dynamic_preplan.assert_called_once_with(self.now)
        self.assertEqual(controller.state_started, self.now)
        self.assertEqual(controller.lane_service.calls, [])
        self.assertEqual(controller.cmd_pub.messages, [])

    def test_new_soft_cost_check_subtracts_the_planning_baseline(self):
        controller = self.make_controller()
        controller.planner = controller_module.HybridAStarPlanner(
            footprint=controller_module.RectangularFootprint(
                front=0.005,
                rear=0.005,
                half_width=0.005,
                padding=0.0,
            ),
            soft_obstacle_cost_weight=1.0,
            soft_cost_check_step=0.01,
            collision_check_step=0.01,
        )
        data = np.zeros((20, 30), dtype=np.int8)
        baseline_soft = np.zeros_like(data)
        baseline_soft[5, 7] = 43
        baseline = controller_module.OccupancyGrid(
            data,
            resolution=0.02,
            origin_x=0.0,
            origin_y=0.0,
            occupied_threshold=65,
            unknown_is_occupied=False,
            soft_cost_data=baseline_soft,
        )
        path = controller_module.path_from_poses(
            np.asarray(((0.10, 0.10, 0.0), (0.30, 0.10, 0.0))),
            frame_id="map",
            target_speed=0.035,
        )

        same = controller_module.OccupancyGrid(
            data,
            resolution=0.02,
            origin_x=0.0,
            origin_y=0.0,
            occupied_threshold=65,
            unknown_is_occupied=False,
            soft_cost_data=baseline_soft.copy(),
        )
        self.assertFalse(
            controller._path_has_new_soft_cost(same, path, baseline)
        )

        latest_soft = baseline_soft.copy()
        latest_soft[5, 12] = 64
        latest = controller_module.OccupancyGrid(
            data,
            resolution=0.02,
            origin_x=0.0,
            origin_y=0.0,
            occupied_threshold=65,
            unknown_is_occupied=False,
            soft_cost_data=latest_soft,
        )
        self.assertTrue(
            controller._path_has_new_soft_cost(latest, path, baseline)
        )

    def test_stale_scan_cannot_consume_the_ready_generation(self):
        controller = self.make_controller()
        controller.arm_generation = 3
        controller.ready_published_generation = 0
        controller.prepared_arm_generation = 3
        controller.costmap = object()
        controller.registration_source_stamp = (
            controller_module.rospy.Time.from_sec(9.0)
        )
        stale_stamp = controller_module.rospy.Time.from_sec(
            self.seconds - controller.scan_timeout - 0.01
        )

        self.assertFalse(controller._try_publish_ready(stale_stamp))
        self.assertEqual(controller.ready_published_generation, 0)
        self.assertEqual(controller.ready_pub.messages, [])

    def test_ready_before_gate_does_not_stop_or_take_lane_control(self):
        controller = self.make_controller()
        controller.arm_generation = 3
        controller.armed_at = controller_module.rospy.Time.from_sec(9.0)
        controller.ready_published_generation = 0
        controller.registered_map_from_odom = (0.0, 0.0, 0.0)
        controller.registered_odom_frame = "odom"
        controller.registration_source_stamp = (
            controller_module.rospy.Time.from_sec(9.8)
        )
        controller.prepared_arm_generation = 3
        ready_stamp = self.now
        lead = 0.30
        controller.odom_history.append(
            (
                ready_stamp,
                controller.entry_staging_pose.x,
                controller.entry_portal_plane_y + lead,
                controller.entry_staging_pose.yaw,
                "odom",
            )
        )
        controller.costmap = object()
        controller._planner_grid = lambda: object()
        controller._path_is_safe = lambda _grid, _path, _index: True

        controller.scan_updates = controller.minimum_initial_scans - 1
        self.assertFalse(controller._try_publish_ready(ready_stamp))
        self.assertEqual(controller.ready_pub.messages, [])
        controller.scan_updates = controller.minimum_initial_scans
        self.assertTrue(controller._try_publish_ready(ready_stamp))

        self.assertEqual(controller.state, controller.WAIT_GATE)
        self.assertFalse(controller.zone_gate)
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(controller.lane_service.calls, [])
        self.assertEqual(controller.cmd_pub.messages, [])
        self.assertEqual(controller.speed_limit_pub.messages, [])

    def test_hybrid_preplan_starts_before_the_ready_lead_without_lane_handoff(self):
        controller = self.make_controller()
        controller.prepared_arm_generation = 0
        controller.prepared_tunnel_path = None
        controller.prepared_exit_connector_station = None
        controller.ready_published_generation = 0
        controller.costmap = object()
        controller.scan_updates = controller.minimum_initial_scans
        controller.registration_source_stamp = (
            controller_module.rospy.Time.from_sec(9.8)
        )
        far_stamp = self.now
        controller.odom_history.append(
            (
                far_stamp,
                controller.entry_staging_pose.x,
                controller.entry_portal_plane_y + 0.80,
                controller.entry_staging_pose.yaw,
                "odom",
            )
        )
        controller._planner_grid = lambda: object()
        started = threading.Event()
        release = threading.Event()

        def blocking_worker(*_args, **_kwargs):
            started.set()
            release.wait(timeout=1.0)

        controller._plan_worker = blocking_worker

        self.assertFalse(controller._try_publish_ready(far_stamp))
        self.assertTrue(started.wait(timeout=0.25))
        self.assertEqual(controller.state, controller.WAIT_GATE)
        self.assertEqual(controller.lane_service.calls, [])
        self.assertEqual(controller.cmd_pub.messages, [])

        release.set()
        controller.planning_thread.join(timeout=0.50)
        self.assertFalse(controller.planning_thread.is_alive())

    def test_rearm_keeps_the_live_worker_reference_and_prevents_overlap(self):
        controller = self.make_controller()
        live_worker = SimpleNamespace(is_alive=lambda: True)
        controller.planning_thread = live_worker
        controller.costmap = object()
        controller._planner_grid = mock.Mock()

        controller._reset_preplan()

        self.assertIs(controller.planning_thread, live_worker)
        self.assertFalse(
            controller._start_preplan(self.now, plan_kind="static")
        )
        controller._planner_grid.assert_not_called()

    def test_completed_preplan_publishes_ready_without_taking_cmd_vel(self):
        controller = self.make_controller()
        tail = controller.prepared_tunnel_path
        controller.prepared_arm_generation = 0
        controller.prepared_tunnel_path = None
        controller.prepared_exit_connector_station = None
        controller.ready_published_generation = 0
        controller.costmap = object()
        controller.scan_updates = controller.minimum_initial_scans
        controller.scan_stamp = self.now
        controller.registration_source_stamp = (
            controller_module.rospy.Time.from_sec(9.8)
        )
        controller.odom_history.append(
            (
                self.now,
                controller.entry_staging_pose.x,
                controller.entry_portal_plane_y + 0.30,
                controller.entry_staging_pose.yaw,
                "odom",
            )
        )
        returned_plan = SimpleNamespace(expanded_nodes=12)
        controller.planner.plan = lambda *_args, **_kwargs: returned_plan
        controller._build_moving_exit_path = (
            lambda _plan, _grid: (tail, 0.05, "")
        )
        grid = object()
        controller._planner_grid = lambda: grid
        controller._path_is_safe = lambda _grid, _path, _index: True

        controller.planning_generation += 1
        generation = controller.planning_generation
        controller._plan_worker(
            generation,
            grid,
            controller.entry_inside_pose,
            None,
            controller.costmap_version,
            1,
            pre_gate=True,
            arm_generation=controller.arm_generation,
        )

        self.assertEqual(controller.state, controller.WAIT_GATE)
        self.assertEqual(
            controller.prepared_arm_generation, controller.arm_generation
        )
        self.assertIs(controller.prepared_tunnel_path, tail)
        self.assertEqual(controller.ready_pub.messages[-1].seq, 1)
        self.assertEqual(controller.lane_service.calls, [])
        self.assertEqual(controller.cmd_pub.messages, [])

    def test_rejected_dynamic_preplan_unlocks_one_newer_scan_retry(self):
        tail = self.make_controller().prepared_tunnel_path
        returned_plan = SimpleNamespace(expanded_nodes=12)

        for rejection in ("no_path", "new_hard", "near_soft"):
            with self.subTest(rejection=rejection):
                controller = self.make_controller()
                controller.state = controller.WAIT_GATE
                controller.zone_gate = False
                controller.prepared_arm_generation = 0
                controller.prepared_tunnel_path = None
                controller.prepared_exit_connector_station = None
                controller.prepared_plan_kind = ""
                controller.dynamic_preplan_generation = controller.arm_generation
                controller.costmap_version = 2
                controller.planner.plan = (
                    (lambda *_args, **_kwargs: None)
                    if rejection == "no_path"
                    else (lambda *_args, **_kwargs: returned_plan)
                )
                controller._build_moving_exit_path = (
                    lambda _plan, _grid: (tail, 0.05, "")
                )
                controller._planner_grid = lambda: object()
                controller._path_is_safe = mock.Mock(
                    return_value=rejection != "new_hard"
                )
                controller._path_future_soft_contact_station = mock.Mock(
                    return_value=(
                        0.10 if rejection == "near_soft" else None
                    )
                )

                controller._plan_worker(
                    controller.planning_generation,
                    object(),
                    controller.entry_inside_pose,
                    None,
                    1,
                    1,
                    pre_gate=True,
                    arm_generation=controller.arm_generation,
                    plan_kind="dynamic",
                )

                self.assertEqual(controller.dynamic_preplan_generation, 0)
                self.assertIsNone(controller.prepared_tunnel_path)
                controller._start_preplan = mock.Mock(return_value=True)
                self.assertTrue(controller._start_dynamic_preplan(self.now))
                self.assertEqual(
                    controller.dynamic_preplan_generation,
                    controller.arm_generation,
                )
                controller._start_preplan.assert_called_once_with(
                    self.now, plan_kind="dynamic"
                )

    def test_ready_lead_is_invariant_to_longitudinal_connector_shift(self):
        template_portal = (
            -1.7475895,
            -0.105857,
            -0.5 * math.pi,
        )
        local_robot = (
            template_portal[0],
            template_portal[1] + 0.30,
            template_portal[2],
        )
        ready_positions = []
        for shift in (0.0, -0.73):
            controller = self.make_controller()
            actual_portal = (4.0, 2.0 + shift, -0.5 * math.pi)
            transform = controller_module.map_from_odom_transform(
                template_portal, actual_portal
            )
            # The robot is the same 0.30 m upstream of the portal after the
            # preceding straight connector is shortened or lengthened.
            actual_robot = (
                actual_portal[0],
                actual_portal[1] + 0.30,
                actual_portal[2],
            )
            controller.arm_generation = 9
            controller.ready_published_generation = 0
            controller.registered_map_from_odom = transform
            controller.registered_odom_frame = "odom"
            controller.registration_source_stamp = (
                controller_module.rospy.Time.from_sec(9.8)
            )
            controller.prepared_arm_generation = 9
            controller.odom_history.append(
                (self.now,) + actual_robot + ("odom",)
            )
            controller.costmap = object()
            controller._planner_grid = lambda: object()
            controller._path_is_safe = lambda _grid, _path, _index: True

            self.assertTrue(controller._try_publish_ready(self.now))
            registered_robot = controller_module.odom_pose_to_map(
                actual_robot, transform
            )
            ready_positions.append(registered_robot)
            self.assertEqual(controller.ready_pub.messages[-1].seq, 9)

        for actual, expected in zip(ready_positions, (local_robot, local_robot)):
            for component, target in zip(actual, expected):
                self.assertAlmostEqual(component, target, places=9)

    def test_gate_preserves_prepared_scans_without_registration_dwell(self):
        controller = self.make_controller()
        controller.costmap = object()
        controller.scan_updates = 6
        prepared_stamp = controller_module.rospy.Time.from_sec(9.95)
        controller.scan_stamp = prepared_stamp
        controller.scan_received = self.now
        controller.last_processed_scan_stamp = prepared_stamp

        controller.gate_callback(Bool(data=True))
        controller._start_run(self.now)

        self.assertEqual(controller.state, controller.ACQUIRING)
        self.assertEqual(controller.scan_updates, 6)
        self.assertEqual(controller.scan_stamp, prepared_stamp)
        self.assertEqual(controller.last_processed_scan_stamp, prepared_stamp)
        self.assertEqual(
            controller.minimum_planning_scan_updates,
            controller.minimum_initial_scans,
        )
        self.assertEqual(controller.cmd_pub.messages, [])

    def test_entry_clear_continues_directly_into_preplanned_hybrid_path(self):
        controller = self.make_controller()
        controller.state = controller.ENTERING
        controller.state_started = self.now
        controller.mission_started = self.now
        controller.mission_has_control = True
        controller.map_from_odom = (0.0, 0.0, 0.0)
        controller.frozen_odom_frame = "odom"
        path = controller_module.path_from_poses(
            np.asarray(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0))),
            frame_id="map",
            target_speed=0.035,
        )
        controller.map_path = path
        controller.odom_path = path
        controller.entry_inside_station = 0.05
        controller.exit_connector_station = 1.0
        controller._input_problem = lambda now, require_scan=True: None
        controller._planner_grid = lambda: object()
        controller._command_is_safe = lambda _grid, _linear, _angular: True
        controller._entry_clearance_ready = lambda: True
        tracking = SimpleNamespace(
            path_index=1,
            position_error=0.0,
            heading_error=0.0,
            target_speed=0.035,
            angular_velocity=0.0,
        )

        with mock.patch.object(
            controller_module, "calculate_tracking", return_value=tracking
        ):
            controller.control_callback(None)

        self.assertEqual(controller.state, controller.FOLLOWING)
        self.assertIs(controller.map_path, path)
        self.assertIs(controller.odom_path, path)
        self.assertTrue(controller.cmd_pub.messages)
        self.assertGreater(controller.cmd_pub.messages[-1].linear.x, 0.0)

    def test_entry_staging_transition_keeps_the_continuous_path(self):
        controller = self.make_controller()
        controller.state = controller.ALIGNING_ENTRY
        controller.state_started = self.now
        controller.mission_started = self.now
        controller.mission_has_control = True
        controller.map_from_odom = (0.0, 0.0, 0.0)
        controller.frozen_odom_frame = "odom"
        controller.odom_x = 0.10
        path = controller_module.path_from_poses(
            np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (0.1, 0.0, 0.0),
                    (0.2, 0.0, 0.0),
                )
            ),
            frame_id="map",
            target_speed=0.035,
        )
        controller.map_path = path
        controller.odom_path = path
        controller.entry_staging_station = 0.10
        controller._input_problem = lambda now, require_scan=True: None
        controller._planner_grid = lambda: object()
        controller._command_is_safe = lambda _grid, _linear, _angular: True
        tracking = SimpleNamespace(
            path_index=1,
            position_error=0.0,
            heading_error=0.0,
            target_speed=0.035,
            angular_velocity=0.0,
        )

        with mock.patch.object(
            controller_module, "calculate_tracking", return_value=tracking
        ):
            controller.control_callback(None)

        self.assertEqual(controller.state, controller.ENTERING)
        self.assertIs(controller.map_path, path)
        self.assertIs(controller.odom_path, path)
        self.assertAlmostEqual(controller.entry_staging_station, 0.10)

    def test_scan_uses_frozen_map_transform_and_timestamped_sensor_tf(self):
        controller = self.make_controller()
        controller.zone_gate = True
        controller.map_from_odom = (10.0, -3.0, 0.5 * math.pi)
        controller.frozen_odom_frame = "odom"
        controller.costmap = RecordingCostmap()

        entered_lookup = threading.Event()
        release_lookup = threading.Event()
        transform = SimpleNamespace(
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=1.0, y=2.0, z=0.0),
                rotation=SimpleNamespace(
                    x=0.0,
                    y=0.0,
                    z=math.sin(math.radians(15.0)),
                    w=math.cos(math.radians(15.0)),
                ),
            )
        )

        class BlockingTransformBuffer:
            def __init__(self):
                self.calls = []
                self.lock_was_owned = None

            def lookup_transform(
                nested_self, target, source, stamp, timeout
            ):
                nested_self.calls.append((target, source, stamp, timeout))
                nested_self.lock_was_owned = controller.lock._is_owned()
                entered_lookup.set()
                release_lookup.wait(timeout=1.0)
                return transform

        controller.tf_buffer = BlockingTransformBuffer()
        scan_stamp = controller_module.rospy.Time.from_sec(9.90)
        worker = threading.Thread(
            target=controller.scan_callback,
            args=(self.scan(scan_stamp),),
        )
        worker.start()
        self.assertTrue(entered_lookup.wait(timeout=0.25))

        # The TF wait must not block odometry or the fail-closed timer from
        # acquiring the controller state lock.
        acquired = controller.lock.acquire(timeout=0.10)
        if acquired:
            controller.lock.release()
        self.assertTrue(acquired)
        self.assertFalse(controller.tf_buffer.lock_was_owned)
        release_lookup.set()
        worker.join(timeout=0.50)
        self.assertFalse(worker.is_alive())

        self.assertEqual(len(controller.tf_buffer.calls), 1)
        target, source, requested_stamp, _ = controller.tf_buffer.calls[0]
        self.assertEqual(target, "odom")
        self.assertEqual(source, "laser")
        self.assertEqual(requested_stamp, scan_stamp)
        self.assertEqual(len(controller.costmap.calls), 1)
        sensor_pose = controller.costmap.calls[0]["sensor_pose"]
        self.assertAlmostEqual(sensor_pose[0], 8.0, places=9)
        self.assertAlmostEqual(sensor_pose[1], -2.0, places=9)
        self.assertAlmostEqual(
            math.sin(sensor_pose[2]),
            math.sin(math.radians(120.0)),
            places=9,
        )
        self.assertAlmostEqual(
            math.cos(sensor_pose[2]),
            math.cos(math.radians(120.0)),
            places=9,
        )

        # Replayed and out-of-order scans are rejected before another TF
        # lookup or layer mutation.
        controller.scan_callback(self.scan(scan_stamp))
        controller.scan_callback(
            self.scan(controller_module.rospy.Time.from_sec(9.85))
        )
        self.assertEqual(len(controller.tf_buffer.calls), 1)
        self.assertEqual(len(controller.costmap.calls), 1)

    def test_wait_gate_keeps_updating_scans_after_ready_publication(self):
        controller = self.make_controller()
        controller.state = controller.WAIT_GATE
        controller.zone_gate = False
        controller.ready_published_generation = controller.arm_generation
        controller.map_from_odom = None
        controller.frozen_odom_frame = ""
        controller.registered_map_from_odom = (0.0, 0.0, 0.0)
        controller.registered_odom_frame = "odom"
        controller.costmap = RecordingCostmap()
        transform = SimpleNamespace(
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )
        controller.tf_buffer = SimpleNamespace(
            lookup_transform=lambda *_args, **_kwargs: transform
        )
        controller._try_publish_ready = mock.Mock(return_value=False)

        controller.scan_callback(
            self.scan(controller_module.rospy.Time.from_sec(9.90))
        )

        self.assertEqual(len(controller.costmap.calls), 1)
        controller._try_publish_ready.assert_called_once()

    def test_acquiring_new_scan_restarts_an_unlocked_dynamic_preplan(self):
        controller = self.make_controller()
        controller.state = controller.ACQUIRING
        controller.zone_gate = True
        controller.mission_has_control = False
        controller.map_from_odom = (0.0, 0.0, 0.0)
        controller.frozen_odom_frame = "odom"
        controller.prepared_arm_generation = 0
        controller.prepared_tunnel_path = None
        controller.dynamic_preplan_generation = 0
        controller.costmap = RecordingCostmap()
        transform = SimpleNamespace(
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            )
        )
        controller.tf_buffer = SimpleNamespace(
            lookup_transform=lambda *_args, **_kwargs: transform
        )
        controller._start_dynamic_preplan = mock.Mock(return_value=True)

        controller.scan_callback(
            self.scan(controller_module.rospy.Time.from_sec(9.90))
        )

        self.assertEqual(len(controller.costmap.calls), 1)
        controller._start_dynamic_preplan.assert_called_once_with(self.now)

    def test_synchronized_odom_selects_nearest_matching_frame(self):
        controller = self.make_controller()
        controller.odom_frame = "odom"
        controller.odom_history.extend(
            [
                (
                    controller_module.rospy.Time.from_sec(9.91),
                    -1.0,
                    0.0,
                    0.0,
                    "odom",
                ),
                (
                    controller_module.rospy.Time.from_sec(10.02),
                    2.0,
                    3.0,
                    0.4,
                    "odom",
                ),
                (
                    controller_module.rospy.Time.from_sec(10.04),
                    9.0,
                    9.0,
                    0.9,
                    "other_odom",
                ),
            ]
        )

        synchronized = controller._synchronized_odom_pose(
            controller_module.rospy.Time.from_sec(10.01)
        )
        self.assertEqual(synchronized, (2.0, 3.0, 0.4))

        controller.odom_frame = "renamed_odom"
        self.assertIsNone(
            controller._synchronized_odom_pose(
                controller_module.rospy.Time.from_sec(10.01)
            )
        )
        controller.costmap = object()
        controller.map_from_odom = (0.0, 0.0, 0.0)
        controller.frozen_odom_frame = "odom"
        self.assertEqual(
            controller._input_problem(self.now, require_scan=False),
            "changed odometry frame",
        )

    def test_changed_costmap_blocking_route_stops_and_replans_immediately(self):
        controller = self.make_controller()
        controller.state = controller.FOLLOWING
        controller.state_started = self.now
        controller.mission_started = self.now
        controller.mission_has_control = True
        controller.costmap_version = 2
        controller.planned_costmap_version = 1
        controller.map_path = object()
        controller.odom_path = object()
        controller._input_problem = lambda now, require_scan=True: None
        controller._planner_grid = lambda: object()
        controller._remaining_path_is_safe = lambda _grid: False
        soft_checks = []
        controller._remaining_path_has_future_soft_cost = (
            lambda _grid: soft_checks.append(True) or True
        )

        controller.control_callback(None)

        self.assertEqual(controller.state, controller.PLANNING)
        self.assertEqual(controller.replan_count, 1)
        self.assertIsNone(controller.map_path)
        self.assertIsNone(controller.odom_path)
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        self.assert_zero(controller.cmd_pub.messages[-1])
        self.assertEqual(soft_checks, [])

    def test_changed_costmap_soft_band_stops_before_it_becomes_lethal(self):
        controller = self.make_controller()
        controller.state = controller.FOLLOWING
        controller.state_started = self.now
        controller.mission_started = self.now
        controller.mission_has_control = True
        controller.costmap_version = 2
        controller.planned_costmap_version = 1
        controller.map_path = object()
        controller.odom_path = object()
        controller._input_problem = lambda now, require_scan=True: None
        controller._planner_grid = lambda: object()
        controller._remaining_path_is_safe = lambda _grid: True
        controller._remaining_path_has_future_soft_cost = lambda _grid: True

        controller.control_callback(None)

        self.assertEqual(controller.state, controller.PLANNING)
        self.assertEqual(controller.replan_count, 1)
        self.assertIsNone(controller.map_path)
        self.assertIsNone(controller.odom_path)
        self.assert_zero(controller.cmd_pub.messages[-1])

    def test_cached_soft_contact_replans_without_another_layer_change(self):
        controller = self.make_controller()
        controller.state = controller.FOLLOWING
        controller.state_started = self.now
        controller.mission_started = self.now
        controller.mission_has_control = True
        controller.costmap_version = 2
        controller.planned_costmap_version = 2
        controller.map_path = SimpleNamespace(
            x=np.asarray([0.0, 0.2, 0.4, 0.6, 0.8]),
            y=np.zeros(5),
            station=np.asarray([0.0, 0.2, 0.4, 0.6, 0.8]),
        )
        controller.odom_path = object()
        controller.path_index = 2
        controller.map_x = 0.4
        controller.soft_replan_contact_station = 0.8
        controller._input_problem = lambda now, require_scan=True: None
        controller._planner_grid = lambda: object()

        controller.control_callback(None)

        self.assertEqual(controller.state, controller.PLANNING)
        self.assertEqual(controller.replan_count, 1)
        self.assertIsNone(controller.map_path)
        self.assertIsNone(controller.odom_path)
        self.assert_zero(controller.cmd_pub.messages[-1])

    def test_soft_replan_ignores_only_one_contiguous_start_prefix(self):
        controller = self.make_controller()
        footprint = controller_module.RectangularFootprint(
            front=0.03,
            rear=0.03,
            half_width=0.03,
            padding=0.0,
        )
        controller.planner = controller_module.HybridAStarPlanner(
            footprint=footprint,
            minimum_turning_radius=0.18,
            primitive_step=0.10,
            steering_samples=5,
            soft_obstacle_cost_weight=1.0,
            soft_cost_check_step=0.01,
            collision_check_step=0.01,
        )
        data = np.zeros((30, 60), dtype=np.int8)
        path = SimpleNamespace(
            x=np.asarray([0.10, 0.20, 0.30, 0.40, 0.50]),
            y=np.zeros(5),
            heading=np.zeros(5),
            curvature=np.zeros(5),
        )

        def grid_with_soft_columns(columns):
            soft = np.zeros_like(data)
            for column in columns:
                soft[13:17, column] = 64
            return controller_module.OccupancyGrid(
                data,
                resolution=0.02,
                origin_x=0.0,
                origin_y=-0.30,
                occupied_threshold=65,
                unknown_is_occupied=True,
                soft_cost_data=soft,
            )

        start = controller_module.Pose2D(0.10, 0.0, 0.0)
        prefix_only = grid_with_soft_columns(range(3, 10))
        self.assertFalse(
            controller._path_has_future_soft_cost(
                prefix_only, path, 0, start
            )
        )

        prefix_and_future = grid_with_soft_columns(
            tuple(range(3, 10)) + tuple(range(26, 30))
        )
        self.assertTrue(
            controller._path_has_future_soft_cost(
                prefix_and_future, path, 0, start
            )
        )

        clear_start = controller_module.Pose2D(0.10, 0.15, 0.0)
        self.assertTrue(
            controller._path_has_future_soft_cost(
                prefix_only, path, 0, clear_start
            )
        )

        continuous_prefix = grid_with_soft_columns(range(3, 30))
        self.assertTrue(
            controller._path_has_future_soft_cost(
                continuous_prefix, path, 0, start
            )
        )

    def test_soft_replan_compares_cellwise_plan_baseline(self):
        controller = self.make_controller()
        footprint = controller_module.RectangularFootprint(
            front=0.03,
            rear=0.03,
            half_width=0.03,
            padding=0.0,
        )
        controller.planner = controller_module.HybridAStarPlanner(
            footprint=footprint,
            minimum_turning_radius=0.18,
            primitive_step=0.10,
            steering_samples=5,
            soft_obstacle_cost_weight=1.0,
            soft_cost_check_step=0.01,
            collision_check_step=0.01,
        )
        data = np.zeros((30, 80), dtype=np.int8)
        path = SimpleNamespace(
            x=np.arange(0.10, 0.71, 0.10),
            y=np.zeros(7),
            heading=np.zeros(7),
            curvature=np.zeros(7),
            station=np.arange(0.0, 0.61, 0.10),
        )

        def grid_with_soft_columns(columns):
            soft = np.zeros_like(data)
            for column in columns:
                soft[13:17, column] = 64
            return controller_module.OccupancyGrid(
                data,
                resolution=0.02,
                origin_x=0.0,
                origin_y=-0.30,
                occupied_threshold=65,
                unknown_is_occupied=True,
                soft_cost_data=soft,
            )

        baseline = grid_with_soft_columns(range(20, 24))
        unchanged = grid_with_soft_columns(range(20, 24))
        enlarged = grid_with_soft_columns(range(20, 27))
        moved_same_total = grid_with_soft_columns(range(24, 28))
        start = controller_module.Pose2D(0.10, 0.15, 0.0)

        self.assertFalse(
            controller._path_has_future_soft_cost(
                unchanged,
                path,
                0,
                start,
                baseline_grid=baseline,
            )
        )
        self.assertTrue(
            controller._path_has_future_soft_cost(
                enlarged,
                path,
                0,
                start,
                baseline_grid=baseline,
            )
        )
        self.assertTrue(
            controller._path_has_future_soft_cost(
                moved_same_total,
                path,
                0,
                start,
                baseline_grid=baseline,
            )
        )

    def test_baseline_soft_corridor_does_not_hide_new_delta(self):
        controller = self.make_controller()
        footprint = controller_module.RectangularFootprint(
            front=0.03,
            rear=0.03,
            half_width=0.03,
            padding=0.0,
        )
        controller.planner = controller_module.HybridAStarPlanner(
            footprint=footprint,
            minimum_turning_radius=0.18,
            primitive_step=0.10,
            steering_samples=5,
            soft_obstacle_cost_weight=1.0,
            soft_cost_check_step=0.01,
            collision_check_step=0.01,
        )
        data = np.zeros((30, 80), dtype=np.int8)
        baseline_soft = np.zeros_like(data)
        baseline_soft[13:17, 3:32] = 21
        latest_soft = baseline_soft.copy()
        latest_soft[13:17, 20:24] = 43

        def grid(soft):
            return controller_module.OccupancyGrid(
                data,
                resolution=0.02,
                origin_x=0.0,
                origin_y=-0.30,
                occupied_threshold=65,
                unknown_is_occupied=True,
                soft_cost_data=soft,
            )

        x = np.arange(0.10, 0.81, 0.10)
        path = SimpleNamespace(
            x=x,
            y=np.zeros(x.size),
            heading=np.zeros(x.size),
            curvature=np.zeros(x.size),
            station=x - x[0],
        )
        baseline = grid(baseline_soft)
        latest = grid(latest_soft)
        start = controller_module.Pose2D(0.10, 0.0, 0.0)

        contact = controller._path_future_soft_contact_station(
            latest,
            path,
            0,
            start,
            baseline_grid=baseline,
        )
        self.assertIsNotNone(contact)
        self.assertLessEqual(contact, 0.40)

    def test_new_soft_start_prefix_cannot_exceed_measured_escape_bound(self):
        controller = self.make_controller()
        footprint = controller_module.RectangularFootprint(
            front=0.03,
            rear=0.03,
            half_width=0.03,
            padding=0.0,
        )
        controller.planner = controller_module.HybridAStarPlanner(
            footprint=footprint,
            minimum_turning_radius=0.18,
            primitive_step=0.10,
            steering_samples=5,
            soft_obstacle_cost_weight=1.0,
            soft_cost_check_step=0.01,
            collision_check_step=0.01,
        )
        data = np.zeros((30, 80), dtype=np.int8)
        soft = np.zeros_like(data)
        soft[13:17, 3:32] = 64
        latest = controller_module.OccupancyGrid(
            data,
            resolution=0.02,
            origin_x=0.0,
            origin_y=-0.30,
            occupied_threshold=65,
            unknown_is_occupied=True,
            soft_cost_data=soft,
        )
        x = np.arange(0.10, 0.81, 0.10)
        path = SimpleNamespace(
            x=x,
            y=np.zeros(x.size),
            heading=np.zeros(x.size),
            curvature=np.zeros(x.size),
            station=x - x[0],
        )

        contact = controller._path_future_soft_contact_station(
            latest,
            path,
            0,
            controller_module.Pose2D(0.10, 0.0, 0.0),
        )
        self.assertEqual(contact, 0.0)

    def test_far_soft_contact_is_cached_until_lookahead_reaches_it(self):
        controller = self.make_controller()
        footprint = controller_module.RectangularFootprint(
            front=0.03,
            rear=0.03,
            half_width=0.03,
            padding=0.0,
        )
        controller.planner = controller_module.HybridAStarPlanner(
            footprint=footprint,
            minimum_turning_radius=0.18,
            primitive_step=0.10,
            steering_samples=5,
            soft_obstacle_cost_weight=1.0,
            soft_cost_check_step=0.01,
            collision_check_step=0.01,
        )
        data = np.zeros((30, 70), dtype=np.int8)
        soft = np.zeros_like(data)
        soft[13:17, 47:51] = 64
        grid = controller_module.OccupancyGrid(
            data,
            resolution=0.02,
            origin_x=0.0,
            origin_y=-0.30,
            occupied_threshold=65,
            unknown_is_occupied=True,
            soft_cost_data=soft,
        )
        x = np.arange(0.10, 1.11, 0.10)
        path = SimpleNamespace(
            x=x,
            y=np.zeros(x.size),
            heading=np.zeros(x.size),
            curvature=np.zeros(x.size),
            station=x - x[0],
        )
        start = controller_module.Pose2D(0.10, 0.15, 0.0)
        contact = controller._path_future_soft_contact_station(
            grid, path, 0, start
        )

        self.assertGreater(contact, controller.soft_replan_lookahead_distance)
        self.assertFalse(
            controller._path_has_future_soft_cost(grid, path, 0, start)
        )
        controller.map_path = path
        controller.soft_replan_contact_station = contact
        controller.path_index = 4
        controller.map_x = 0.50
        self.assertTrue(controller._soft_replan_contact_is_due())

    def test_layer_change_checks_obstacle_beyond_short_validation_horizon(self):
        controller = self.make_controller()
        controller.state = controller.FOLLOWING
        controller.state_started = self.now
        controller.mission_started = self.now
        controller.mission_has_control = True
        controller.costmap_version = 2
        controller.planned_costmap_version = 1
        controller.map_path = SimpleNamespace(
            x=np.asarray([0.0, 0.4, 0.8, 1.2]),
            y=np.zeros(4),
            heading=np.zeros(4),
            curvature=np.zeros(4),
            station=np.asarray([0.0, 0.4, 0.8, 1.2]),
        )
        controller.odom_path = object()
        controller._input_problem = lambda now, require_scan=True: None
        controller._planner_grid = lambda: object()
        checked_x = []

        def collision_free(_grid, pose):
            checked_x.append(pose.x)
            return pose.x < 1.1

        controller.planner.pose_is_collision_free = collision_free

        controller.control_callback(None)

        self.assertEqual(controller.state, controller.PLANNING)
        self.assertEqual(controller.replan_count, 1)
        self.assertTrue(any(x > 0.70 for x in checked_x))
        self.assertIn(1.2, checked_x)

    def test_changed_layer_checks_sweeps_between_sampled_poses(self):
        controller = self.make_controller()
        path = SimpleNamespace(
            x=np.asarray([0.0, 0.01]),
            y=np.asarray([0.0, 0.0]),
            heading=np.asarray([0.0, 0.05]),
            curvature=np.asarray([5.0, 5.0]),
        )
        controller.planner.pose_is_collision_free = lambda _grid, _pose: True
        swept = []

        def blocked_mid_sweep(_grid, pose, curvature, distance):
            swept.append((pose.x, curvature, distance))
            return False

        controller.planner.primitive_is_collision_free = blocked_mid_sweep

        self.assertFalse(controller._path_is_safe(object(), path, 0))
        self.assertEqual(len(swept), 1)
        self.assertAlmostEqual(swept[0][1], 5.0)
        self.assertAlmostEqual(swept[0][2], 0.01)

    def test_new_interior_collision_during_entry_requests_replan_not_failure(self):
        controller = self.make_controller()
        controller.state = controller.ALIGNING_ENTRY
        controller.state_started = self.now
        controller.mission_started = self.now
        controller.mission_has_control = True
        controller.zone_gate = True
        controller.map_from_odom = (0.0, 0.0, 0.0)
        controller.frozen_odom_frame = "odom"
        path = controller_module.path_from_poses(
            np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (0.10, 0.0, 0.0),
                    (0.20, 0.0, 0.0),
                    (0.30, 0.0, 0.0),
                )
            ),
            frame_id="map",
            target_speed=0.035,
        )
        controller.map_path = path
        controller.odom_path = path
        controller.entry_inside_station = 0.10
        controller.planned_costmap_version = 1
        controller.costmap_version = 2
        controller._input_problem = lambda now, require_scan=True: None
        controller._planner_grid = lambda: object()
        controller.planner.pose_is_collision_free = lambda _grid, _pose: True
        controller.planner.primitive_is_collision_free = (
            lambda _grid, pose, _curvature, _distance: pose.x < 0.20
        )

        controller.control_callback(None)

        self.assertEqual(controller.state, controller.PLANNING)
        self.assertEqual(controller.replan_count, 1)
        self.assertNotEqual(controller.state, controller.FAILED)
        self.assert_zero(controller.cmd_pub.messages[-1])

    def test_moving_exit_seam_preserves_cruise_command_without_zero(self):
        controller = self.make_controller()
        controller.state = controller.FOLLOWING
        controller.state_started = self.now
        controller.mission_started = self.now
        controller.mission_has_control = True
        controller.map_from_odom = (0.0, 0.0, 0.0)
        controller.frozen_odom_frame = "odom"
        controller.odom_x = 0.5
        controller.odom_linear_velocity = 0.075
        controller.goal = controller_module.Pose2D(0.5, 0.0, 0.0)
        path = controller_module.CommonPath(
            x=np.asarray([0.0, 0.5, 1.0]),
            y=np.zeros(3),
            heading=np.zeros(3),
            curvature=np.zeros(3),
            speed=np.full(3, 0.075),
            label="tunnel_hybrid_moving_exit",
        )
        controller.map_path = path
        controller.odom_path = path
        controller.exit_connector_station = 0.5
        controller.remaining_distance = 0.5
        controller.last_linear = 0.075
        controller.last_command_time = controller_module.rospy.Time.from_sec(
            self.seconds - controller.control_period
        )
        controller._input_problem = lambda now, require_scan=True: None
        controller._planner_grid = lambda: object()
        controller._command_is_safe = mock.Mock(return_value=True)
        tracking = SimpleNamespace(
            path_index=1,
            position_error=0.0,
            heading_error=0.0,
            target_speed=0.075,
            angular_velocity=0.0,
        )

        with mock.patch.object(
            controller_module, "calculate_tracking", return_value=tracking
        ):
            controller.control_callback(None)

            self.assertEqual(controller.state, controller.EXITING)
            self.assertIs(controller.map_path, path)
            self.assertIs(controller.odom_path, path)
            self.assertTrue(controller.exit_confirmation_started)
            confirmation_started_at = controller.confirmation_started_at
            self.assertEqual(confirmation_started_at, self.now)
            self.assertEqual(
                [message.data for message in controller.speed_limit_pub.messages],
                [controller.join_velocity_cap],
            )
            self.assertEqual(len(controller.cmd_pub.messages), 1)
            self.assertGreater(controller.cmd_pub.messages[-1].linear.x, 0.0)
            self.assertAlmostEqual(controller.last_linear, 0.075)

            controller._exit_pose_ready = lambda: True
            controller._exit_clearance_ready = lambda: True
            self.advance()
            controller.lane_path_diagnostics_callback(self.valid_lane_path())
            self.advance()
            controller.lane_path_diagnostics_callback(self.valid_lane_path())
            controller.control_callback(None)

        self.assertEqual(controller.state, controller.JOINING_LANE)
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(controller.lane_service.calls, [True])
        cap_event_index = next(
            index
            for index, event in enumerate(controller.events)
            if event[0:2] == ("publish", "/control/max_vel")
        )
        handoff_event_index = controller.events.index(("handoff", True))
        self.assertLess(cap_event_index, handoff_event_index)
        self.assertTrue(
            all(
                command.linear.x > 0.0
                for command in controller.cmd_pub.messages
            )
        )
        self.assertEqual(
            [message.data for message in controller.state_pub.messages],
            [controller.EXITING, controller.JOINING_LANE],
        )

    def test_exit_handoff_retries_while_tunnel_keeps_positive_command(self):
        controller = self.make_rolling_exit_controller()
        controller._exit_clearance_ready = lambda: True
        controller._lane_confirmed = lambda _now, _frames: True
        controller._tracking_endpoint_ready = lambda: False
        controller.last_linear = 0.075
        controller.last_angular = 0.02
        handoffs = []

        def deferred(_enabled):
            handoffs.append(True)
            controller.events.append(("handoff", True))
            return SimpleNamespace(success=False, message="lane path not ready")

        controller.lane_service = deferred
        tracking = SimpleNamespace(
            path_index=1,
            position_error=0.0,
            heading_error=0.0,
            target_speed=0.050,
            angular_velocity=-0.03,
        )
        with mock.patch.object(
            controller_module, "calculate_tracking", return_value=tracking
        ):
            for _ in range(20):
                controller.control_callback(None)
                self.assertEqual(controller.state, controller.EXITING)
                self.assertTrue(controller.mission_has_control)
                self.assertTrue(controller.lane_handoff_retry_pending)
                self.advance()

        self.assertEqual(len(handoffs), 20)
        self.assertEqual(len(controller.cmd_pub.messages), 20)
        self.assertTrue(
            all(command.linear.x > 0.0 for command in controller.cmd_pub.messages)
        )
        self.assertLess(controller.cmd_pub.messages[-1].linear.x, 0.075)

        controller.lane_service = RecordingLaneService(controller.events)
        with mock.patch.object(
            controller_module, "calculate_tracking", return_value=tracking
        ):
            controller.control_callback(None)
        self.assertEqual(controller.state, controller.JOINING_LANE)
        self.assertFalse(controller.mission_has_control)
        self.assertFalse(controller.lane_handoff_retry_pending)

    def test_moving_exit_builder_requires_safe_guarded_connector(self):
        controller = self.make_controller()
        controller.goal = controller_module.Pose2D(0.0, 0.0, 0.0)
        controller.exit_outside_pose = controller_module.Pose2D(
            1.0, 0.0, 0.0
        )

        def plan():
            return SimpleNamespace(
                x=np.asarray([-0.5, 0.0]),
                y=np.zeros(2),
                yaw=np.zeros(2),
                curvature=np.zeros(2),
            )

        path, station, reason = controller._build_moving_exit_path(
            plan(), object()
        )
        self.assertEqual(reason, "")
        self.assertEqual(path.label, "tunnel_hybrid_moving_exit")
        self.assertAlmostEqual(station, 0.5)
        self.assertAlmostEqual(path.speed[-1], 0.075)

        controller.planner.primitive_is_collision_free = (
            lambda _grid, _pose, _curvature, _distance: False
        )
        rejected, rejected_station, reason = (
            controller._build_moving_exit_path(plan(), object())
        )
        self.assertIsNone(rejected)
        self.assertIsNone(rejected_station)
        self.assertIn("not collision-free", reason)

    def test_rejected_moving_exit_rejects_the_whole_plan(self):
        controller = self.make_controller()
        controller.state = controller.PLANNING
        controller.state_started = self.now
        controller.mission_started = self.now
        controller.zone_gate = True
        controller.mission_has_control = True
        controller.map_from_odom = (0.0, 0.0, 0.0)
        controller.frozen_odom_frame = "odom"
        controller.goal = controller_module.Pose2D(1.0, 0.0, 0.0)
        controller.exit_outside_pose = controller_module.Pose2D(
            2.0, 0.0, 0.0
        )
        rejected_plan = SimpleNamespace(
            x=np.asarray([0.0, 1.0]),
            y=np.zeros(2),
            yaw=np.zeros(2),
            curvature=np.zeros(2),
            expanded_nodes=2,
        )
        controller.planner.plan = lambda *_args, **_kwargs: rejected_plan
        controller.planner.primitive_is_collision_free = (
            lambda _grid, pose, _curvature, _distance: pose.x < 1.0 - 1e-9
        )
        grid = object()
        controller._planner_grid = lambda: grid

        controller._plan_worker(
            controller.planning_generation,
            grid,
            controller_module.Pose2D(0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            controller.costmap_version,
            1,
        )

        self.assertEqual(controller.state, controller.PLANNING)
        self.assertIsNone(controller.map_path)
        self.assertIsNone(controller.odom_path)
        self.assertIsNone(controller.exit_connector_station)

    def test_planner_worker_does_not_block_control_callback(self):
        controller = self.make_controller()
        controller.state = controller.PLANNING
        controller.state_started = self.now
        controller.mission_started = self.now
        controller.zone_gate = True
        controller.mission_has_control = True
        controller.map_from_odom = (0.0, 0.0, 0.0)
        controller.frozen_odom_frame = "odom"
        controller.goal = controller_module.Pose2D(1.0, 0.0, 0.0)
        controller._input_problem = lambda now, require_scan=True: None
        controller._planner_grid = lambda: object()

        planner_started = threading.Event()
        release_planner = threading.Event()

        def blocking_plan(_grid, _start, _goal, goal_curvature=0.0):
            planner_started.set()
            release_planner.wait(timeout=1.0)
            return None

        controller.planner.plan = blocking_plan
        callback = threading.Thread(
            target=controller.control_callback,
            args=(None,),
        )
        callback.start()
        self.assertTrue(planner_started.wait(timeout=0.25))
        callback.join(timeout=0.10)

        self.assertFalse(callback.is_alive())
        worker = controller.planning_thread
        self.assertIsNotNone(worker)
        self.assertTrue(worker.is_alive())
        self.assert_zero(controller.cmd_pub.messages[-1])

        release_planner.set()
        worker.join(timeout=0.50)
        self.assertFalse(worker.is_alive())

    def test_plan_worker_revalidates_newer_hard_and_soft_evidence(self):
        data = np.zeros((100, 150), dtype=np.int8)
        baseline = controller_module.OccupancyGrid(
            data,
            resolution=0.02,
            origin_x=-1.0,
            origin_y=-1.0,
            occupied_threshold=65,
            unknown_is_occupied=True,
            soft_cost_data=np.zeros_like(data),
        )
        x = np.arange(0.0, 1.01, 0.10)
        returned_plan = SimpleNamespace(
            x=x,
            y=np.zeros(x.size),
            yaw=np.zeros(x.size),
            curvature=np.zeros(x.size),
            expanded_nodes=10,
        )

        def updated_grid(soft_x=None, hard_x=None):
            hard = data.copy()
            soft = np.zeros_like(data)
            row = int((0.0 + 1.0) / 0.02)
            if soft_x is not None:
                column = int((soft_x + 1.0) / 0.02)
                soft[row - 2 : row + 3, column] = 64
            if hard_x is not None:
                column = int((hard_x + 1.0) / 0.02)
                hard[row, column] = 100
            return controller_module.OccupancyGrid(
                hard,
                resolution=0.02,
                origin_x=-1.0,
                origin_y=-1.0,
                occupied_threshold=65,
                unknown_is_occupied=True,
                soft_cost_data=soft,
            )

        def run_worker(latest):
            controller = self.make_controller()
            controller.state = controller.PLANNING
            controller.zone_gate = True
            controller.mission_has_control = True
            controller.map_from_odom = (0.0, 0.0, 0.0)
            controller.frozen_odom_frame = "odom"
            controller.odom_x = 0.0
            controller.odom_y = 0.0
            controller.odom_yaw = 0.0
            controller.goal = controller_module.Pose2D(1.0, 0.0, 0.0)
            controller.exit_outside_pose = controller_module.Pose2D(
                1.5, 0.0, 0.0
            )
            controller.costmap_version = 2
            planner = controller_module.HybridAStarPlanner(
                footprint=controller_module.RectangularFootprint(
                    front=0.03,
                    rear=0.03,
                    half_width=0.03,
                    padding=0.0,
                ),
                minimum_turning_radius=0.18,
                primitive_step=0.10,
                steering_samples=5,
                soft_obstacle_cost_weight=1.0,
                soft_cost_check_step=0.01,
                collision_check_step=0.01,
            )
            planner.plan = lambda *_args, **_kwargs: returned_plan
            controller.planner = planner
            controller._planner_grid = lambda: latest
            controller._plan_worker(
                controller.planning_generation,
                baseline,
                controller_module.Pose2D(0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                1,
                1,
            )
            return controller

        near_soft = run_worker(updated_grid(soft_x=0.40))
        self.assertEqual(near_soft.state, near_soft.PLANNING)
        self.assertIsNone(near_soft.map_path)

        hard = run_worker(updated_grid(hard_x=0.50))
        self.assertEqual(hard.state, hard.PLANNING)
        self.assertIsNone(hard.map_path)

        far_soft = run_worker(updated_grid(soft_x=0.95))
        self.assertEqual(far_soft.state, far_soft.FOLLOWING)
        self.assertIs(far_soft.planned_grid, baseline)
        self.assertIsNotNone(far_soft.soft_replan_contact_station)
        self.assertGreater(
            far_soft.soft_replan_contact_station,
            far_soft.soft_replan_lookahead_distance,
        )

    def test_production_one_point_inner_goal_continues_into_moving_exit(self):
        config, _, grid, planner, goal = self.production_grid_and_planner()

        for start_x in (goal.x - 0.02, goal.x - 0.03):
            with self.subTest(start_x=start_x):
                controller = self.make_controller()
                controller.state = controller.PLANNING
                controller.state_started = self.now
                controller.mission_started = self.now
                controller.zone_gate = True
                controller.mission_has_control = True
                controller.map_from_odom = (0.0, 0.0, 0.0)
                controller.frozen_odom_frame = "odom"
                controller.odom_x = start_x
                controller.odom_y = goal.y
                controller.odom_yaw = goal.yaw
                controller.goal = goal
                controller.planner = planner
                control = config["control"]
                controller.cruise_velocity = control["cruise_velocity"]
                controller.minimum_velocity = control["minimum_velocity"]
                controller.entry_velocity = control["entry_velocity"]
                controller.exit_velocity = control["exit_velocity"]
                controller.maximum_angular_velocity = control[
                    "maximum_angular_velocity"
                ]
                controller.maximum_lateral_acceleration = control[
                    "maximum_lateral_acceleration"
                ]
                controller.linear_acceleration = control["linear_acceleration"]
                controller.linear_deceleration = control["linear_deceleration"]
                controller.angular_acceleration = control["angular_acceleration"]
                exit_values = config["exit"]
                controller.exit_connector_tangent_ratio = exit_values[
                    "connector_tangent_ratio"
                ]
                controller.exit_connector_minimum_tangent_length = exit_values[
                    "minimum_tangent_length"
                ]
                controller.exit_connector_maximum_tangent_length = exit_values[
                    "maximum_tangent_length"
                ]
                controller.exit_connector_sample_step = exit_values[
                    "connector_sample_step"
                ]
                controller.exit_outside_pose = controller_module.pose_from_degrees(
                    exit_values["outside_pose"], "exit/outside_pose"
                )
                controller._input_problem = lambda now, require_scan=True: None
                controller._planner_grid = lambda: grid

                plan = planner.plan(
                    grid,
                    controller_module.Pose2D(start_x, goal.y, goal.yaw),
                    goal,
                    goal_curvature=0.0,
                )
                self.assertIsNotNone(plan)
                self.assertEqual(plan.x.size, 1)
                self.assertAlmostEqual(plan.length, 0.0)

                controller._plan_worker(
                    controller.planning_generation,
                    grid,
                    controller_module.Pose2D(start_x, goal.y, goal.yaw),
                    (start_x, goal.y, goal.yaw),
                    controller.costmap_version,
                    1,
                )
                self.assertEqual(controller.state, controller.FOLLOWING)
                self.assertGreater(controller.map_path.x.size, 1)
                self.assertEqual(
                    controller.map_path.label,
                    "tunnel_hybrid_moving_exit",
                )
                self.assertAlmostEqual(controller.exit_connector_station, 0.0)

                self.advance()
                controller.control_callback(None)

                self.assertEqual(controller.state, controller.EXITING)
                self.assertTrue(controller.exit_confirmation_started)
                self.assertIsNotNone(controller.map_path)
                self.assertIsNotNone(controller.odom_path)
                self.assertTrue(
                    all(
                        command.linear.x > 0.0
                        for command in controller.cmd_pub.messages
                    )
                )

    def test_production_static_exclusion_rejects_recorded_wall_ghost_only(self):
        config, costmap, _, _, _ = self.production_grid_and_planner()
        self.assertAlmostEqual(
            config["map"]["static_hit_exclusion_radius"], 0.100
        )
        self.assertAlmostEqual(
            config["map"]["dynamic_inflation_radius"], 0.060
        )
        self.assertEqual(config["map"]["dynamic_occupied_value"], 100)
        self.assertEqual(config["map"]["dynamic_inflation_value"], 64)
        self.assertAlmostEqual(
            config["planner"]["soft_obstacle_cost_weight"], 1.0
        )
        self.assertAlmostEqual(
            config["control"]["soft_replan_lookahead_distance"], 0.50
        )
        self.assertAlmostEqual(
            config["control"]["soft_start_prefix_max_distance"], 0.35
        )

        # official_008 repeatedly observed this fixed north-west wall-end
        # return as far as 77.858 mm from the nominal static cell. It must not
        # flicker into a lethal dynamic obstacle at the entrance portal.
        for _ in range(config["map"]["mark_observations"]):
            update = costmap.update_scan(
                ranges=[0.082142],
                angle_min=0.0,
                angle_increment=0.0,
                range_min=0.05,
                range_max=3.0,
                sensor_pose=(-1.76, -0.026684, 0.0),
            )
        wall_cell = costmap.world_to_cell(-1.677858, -0.026684)
        self.assertIsNotNone(wall_cell)
        self.assertFalse(costmap.is_dynamic_occupied(*wall_cell))
        self.assertEqual(update.marked_cells, 0)

        # A surface point on the nearest observed cylinder was still 440 mm
        # from the static layer and must remain available to Hybrid A*.
        for _ in range(config["map"]["mark_observations"]):
            update = costmap.update_scan(
                ranges=[0.28],
                angle_min=0.0,
                angle_increment=0.0,
                range_min=0.05,
                range_max=3.0,
                sensor_pose=(-1.75, -1.39, 0.0),
            )
        cylinder_cell = costmap.world_to_cell(-1.47, -1.39)
        self.assertIsNotNone(cylinder_cell)
        self.assertTrue(costmap.is_dynamic_occupied(*cylinder_cell))
        self.assertEqual(update.marked_cells, 1)
        exported = costmap.to_occupancy_data()
        soft_cost = costmap.to_soft_cost_data()
        raw_index = cylinder_cell[1] * costmap.width + cylinder_cell[0]
        self.assertEqual(exported[raw_index], 100)
        self.assertEqual(soft_cost[raw_index], 0)
        three_cells_left = (
            cylinder_cell[1] * costmap.width + cylinder_cell[0] - 3
        )
        one_cell_left = raw_index - 1
        two_cells_left = raw_index - 2
        diagonal_two_cells = (
            (cylinder_cell[1] - 2) * costmap.width
            + cylinder_cell[0]
            - 2
        )
        self.assertEqual(exported[one_cell_left], 64)
        self.assertEqual(exported[two_cells_left], 43)
        self.assertEqual(exported[three_cells_left], 21)
        self.assertEqual(exported[diagonal_two_cells], 21)
        self.assertEqual(soft_cost[one_cell_left], 64)
        self.assertEqual(soft_cost[two_cells_left], 43)
        self.assertEqual(soft_cost[three_cells_left], 21)
        self.assertEqual(soft_cost[diagonal_two_cells], 21)

    def test_planning_waits_for_measured_robot_stop(self):
        controller = self.make_controller()
        controller.state = controller.PLANNING
        controller.state_started = self.now
        controller.mission_started = self.now
        controller.zone_gate = True
        controller.mission_has_control = True
        controller.map_from_odom = (0.0, 0.0, 0.0)
        controller.frozen_odom_frame = "odom"
        controller.odom_linear_velocity = 0.02
        controller._input_problem = lambda now, require_scan=True: None
        controller._planner_grid = mock.Mock(
            side_effect=AssertionError("planner must not start while moving")
        )

        controller.control_callback(None)

        self.assertEqual(controller.state, controller.PLANNING)
        self.assertIsNone(controller.planning_thread)
        controller._planner_grid.assert_not_called()
        self.assert_zero(controller.cmd_pub.messages[-1])

    def test_command_safety_checks_measured_and_requested_twists(self):
        controller = self.make_controller()
        controller.map_from_odom = (0.0, 0.0, 0.0)
        controller.odom_linear_velocity = 0.07
        controller.odom_angular_velocity = -0.20
        motions = []

        def record_motion(_grid, _pose, linear, angular):
            motions.append((linear, angular))
            return True

        controller._motion_is_safe = record_motion

        self.assertTrue(controller._command_is_safe(object(), 0.04, 0.30))
        self.assertEqual(motions, [(0.07, -0.20), (0.04, 0.30)])

    def test_near_zero_linear_speed_does_not_turn_margin_into_reaction_time(self):
        controller = self.make_controller()
        controller.last_motion_safety_failure = ""

        self.assertTrue(
            controller._motion_is_safe(
                object(),
                controller_module.Pose2D(0.0, 0.0, 0.0),
                1e-5,
                0.003,
            )
        )
        self.assertEqual(controller.last_motion_safety_failure, "")

    def test_exit_keeps_moving_until_clearance_and_lane_frames_are_ready(self):
        controller = self.make_rolling_exit_controller()
        exit_pose_ready = {"ready": False}
        exit_clearance_ready = {"ready": False}
        controller._exit_pose_ready = lambda: exit_pose_ready["ready"]
        controller._exit_clearance_ready = lambda: exit_clearance_ready["ready"]
        tracking = self.rolling_exit_tracking()

        with mock.patch.object(
            controller_module, "calculate_tracking", return_value=tracking
        ):
            self.advance()
            controller.lane_path_diagnostics_callback(self.valid_lane_path())
            self.advance()
            controller.lane_path_diagnostics_callback(self.valid_lane_path())
            controller.control_callback(None)

            self.assertEqual(controller.state, controller.EXITING)
            self.assertEqual(controller.lane_service.calls, [])
            self.assertGreater(controller.cmd_pub.messages[-1].linear.x, 0.0)

            exit_clearance_ready["ready"] = True
            self.advance()
            controller.lane_path_diagnostics_callback(self.invalid_lane_path())
            self.advance()
            controller.lane_path_diagnostics_callback(self.valid_lane_path())
            controller.control_callback(None)

            self.assertEqual(controller.state, controller.EXITING)
            self.assertEqual(controller.lane_service.calls, [])
            self.assertGreater(controller.cmd_pub.messages[-1].linear.x, 0.0)

            self.advance()
            controller.lane_path_diagnostics_callback(self.valid_lane_path())
            exit_pose_ready["ready"] = True
            controller.control_callback(None)

        self.assertEqual(controller.state, controller.JOINING_LANE)
        self.assertEqual(controller.lane_service.calls, [True])
        self.assertFalse(controller.mission_has_control)
        self.assertTrue(
            all(
                command.linear.x > 0.0
                for command in controller.cmd_pub.messages
            )
        )

    def test_exit_rejects_stale_lane_path_without_a_planned_stop(self):
        controller = self.make_rolling_exit_controller()
        controller._exit_pose_ready = lambda: False
        controller._exit_clearance_ready = lambda: True

        self.advance()
        controller.lane_path_diagnostics_callback(self.valid_lane_path())
        self.advance()
        controller.lane_path_diagnostics_callback(self.valid_lane_path())
        self.advance(controller.lane_path_timeout + 0.01)
        with mock.patch.object(
            controller_module,
            "calculate_tracking",
            return_value=self.rolling_exit_tracking(),
        ):
            controller.control_callback(None)

        self.assertEqual(controller.state, controller.EXITING)
        self.assertEqual(controller.lane_service.calls, [])
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        self.assertGreater(controller.cmd_pub.messages[-1].linear.x, 0.0)

    def test_exit_endpoint_without_safe_handoff_fails_closed(self):
        controller = self.make_rolling_exit_controller()
        controller._exit_pose_ready = lambda: True
        controller._exit_clearance_ready = lambda: True
        controller.lane_path_valid = False

        with mock.patch.object(
            controller_module,
            "calculate_tracking",
            return_value=self.rolling_exit_tracking(),
        ):
            controller.control_callback(None)

        self.assertEqual(controller.state, controller.FAILED)
        self.assertTrue(controller.mission_has_control)
        self.assertEqual(controller.lane_service.calls, [])
        self.assertEqual(len(controller.cmd_pub.messages), 1)
        self.assert_zero(controller.cmd_pub.messages[-1])

    def test_production_exit_uses_only_shared_lane_path_diagnostics(self):
        with (PACKAGE_DIR / "config" / "tunnel_mission_gazebo.yaml").open(
            encoding="utf-8"
        ) as stream:
            config = yaml.safe_load(stream)["tunnel"]

        self.assertEqual(
            config["topics"]["lane_path_diagnostics"],
            "/control/lane_path_diagnostics",
        )
        self.assertNotIn("lane_boundaries", config["topics"])
        self.assertNotIn("lane_center", config["topics"])
        self.assertAlmostEqual(config["exit"]["lane_path_timeout"], 0.35)
        self.assertNotIn("boundary_timeout", config["exit"])
        self.assertNotIn("lane_center_timeout", config["exit"])

    def test_production_moving_exit_retains_cruise_speed(self):
        with (PACKAGE_DIR / "config" / "tunnel_mission_gazebo.yaml").open(
            encoding="utf-8"
        ) as stream:
            config = yaml.safe_load(stream)["tunnel"]

        self.assertAlmostEqual(
            config["control"]["exit_velocity"],
            config["control"]["cruise_velocity"],
        )
        legacy_stop_keys = (
            "straight_velocity",
            "fallback_velocity",
            "moving_transition_enabled",
            "moving_heading_tolerance_deg",
            "moving_lateral_tolerance",
            "moving_longitudinal_tolerance",
            "staging_remaining_tolerance",
            "staging_position_tolerance",
            "staging_heading_tolerance_deg",
            "staging_lateral_tolerance",
            "alignment_tolerance_deg",
            "alignment_gain",
            "alignment_max_angular_velocity",
            "alignment_min_angular_velocity",
        )
        for key in legacy_stop_keys:
            self.assertNotIn(key, config["exit"])

    def test_release_completes_only_after_fresh_odom_progress_and_lane(self):
        controller = self.make_controller()
        controller.state = controller.JOINING_LANE
        controller.state_started = self.now
        controller.mission_started = self.now
        controller.mission_has_control = False
        controller.confirmation_started_at = self.now
        controller.odom_stamp = self.now
        controller.join_origin_ready = False

        controller.control_callback(None)
        self.assertFalse(controller.join_origin_ready)
        self.assertEqual(controller.state, controller.JOINING_LANE)

        self.advance()
        controller.odom_callback(self.odometry(1.0))
        controller.control_callback(None)
        self.assertTrue(controller.join_origin_ready)

        self.advance()
        controller.lane_path_diagnostics_callback(self.valid_lane_path())
        self.advance()
        controller.lane_path_diagnostics_callback(self.valid_lane_path())
        controller.control_callback(None)
        self.assertEqual(controller.state, controller.JOINING_LANE)

        self.advance()
        controller.odom_callback(self.odometry(1.05))
        controller.control_callback(None)

        self.assertEqual(controller.state, controller.COMPLETE)
        self.assertFalse(controller.mission_has_control)
        self.assertAlmostEqual(
            controller.speed_limit_pub.messages[-1].data,
            controller.lane_resume_max_velocity,
        )

    def test_stale_scan_while_following_fails_closed(self):
        controller = self.make_controller()
        controller.state = controller.FOLLOWING
        controller.state_started = self.now
        controller.mission_started = self.now
        controller.mission_has_control = True
        controller.costmap = object()
        controller.scan_received = controller_module.rospy.Time.from_sec(
            self.seconds - controller.scan_timeout - 0.01
        )

        controller.control_callback(None)

        self.assertEqual(controller.state, controller.FAILED)
        self.assertTrue(controller.mission_has_control)
        self.assertEqual(controller.lane_service.calls, [])
        self.assert_zero(controller.cmd_pub.messages[-1])
        self.assertEqual(controller.speed_limit_pub.messages[-1].data, 0.0)

    def test_lane_owned_failure_uses_lane_stop_as_sole_fallback_owner(self):
        controller = self.make_controller()
        controller.state = controller.JOINING_LANE
        controller.mission_has_control = False

        def failed_handoff(_enabled):
            raise controller_module.rospy.ServiceException("handoff failed")

        controller.lane_service = failed_handoff
        controller._fail("join failed")

        self.assertEqual(controller.state, controller.FAILED)
        self.assertFalse(controller.mission_has_control)
        self.assertEqual(controller.lane_stop_service.calls, [False])
        self.assertEqual(controller.cmd_pub.messages, [])


if __name__ == "__main__":
    unittest.main()
