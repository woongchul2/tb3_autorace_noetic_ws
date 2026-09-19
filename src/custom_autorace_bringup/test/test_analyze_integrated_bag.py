#!/usr/bin/env python3

import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from analyze_integrated_bag import (  # noqa: E402
    COMMON_DIAGNOSTIC_TOPICS,
    EXPECTED_MANAGER_CALLERID,
    EXPECTED_MISSION_CALLERIDS,
    EXPECTED_MOTION_CMD_VEL_CALLERID_ORDER,
    EXPECTED_RAW_CMD_VEL_CALLERID_ORDER,
    IntegratedRunAccumulator,
    MANAGER_TOPICS,
    MISSION_EXECUTION_STATES,
    MISSION_ORDER,
    callerid_from_header,
    extract_model_pose,
    summarize_rate,
)


class IntegratedBagAnalysisTest(unittest.TestCase):
    @staticmethod
    def diagnostic(progress, position, cross_track, line, obstacle, map_clearance):
        return [
            progress,
            max(0.0, 1.0 - progress),
            position,
            cross_track,
            0.0,
            0.2,
            5.0,
            0.1,
            line,
            obstacle,
            map_clearance,
            0.2,
            0.0,
        ]

    @staticmethod
    def execution_states(mission):
        return {
            "intersection": ("FOLLOW_ENTRY_PATH", "FOLLOW_ARC_LANE", "FOLLOW_EXIT_PATH"),
            "obstacle": ("ACQUIRING", "AVOIDING", "REJOINING"),
            "parking": ("APPROACH", "BACK_OUT", "LEAVE_AISLE"),
            "zigzag": ("ACQUIRING", "FOLLOWING", "JOINING_LANE"),
            "level_crossing": ("APPROACH", "STOPPED", "PASSING"),
            "tunnel": ("ACQUIRING", "PLANNING", "FOLLOWING"),
        }[mission]

    def complete_accumulator(self):
        accumulator = IntegratedRunAccumulator(
            minimum_predeparture_duration=0.2,
            minimum_lane_rate_hz=29.0,
        )

        # Recorder coverage begins while the freshly launched robot is still
        # stationary at the full official pose.
        for stamp in (0.05, 0.10, 0.15, 0.20):
            accumulator.observe_pose(stamp, 0.8, -1.747, 0.0, 0.0, 0.0)
        accumulator.observe_manual_stop(
            0.05, False, "/safe_lane_controller"
        )

        accumulator.observe_manager_state(
            "/mission/current", "intersection", 0.02, EXPECTED_MANAGER_CALLERID
        )
        accumulator.observe_manager_state(
            "/mission/state", "SEEKING", 0.03, EXPECTED_MANAGER_CALLERID
        )
        accumulator.observe_sequence_index(
            0, 0.04, EXPECTED_MANAGER_CALLERID
        )
        for mission in MISSION_ORDER:
            accumulator.observe_gate(
                mission, False, 0.04, EXPECTED_MANAGER_CALLERID
            )
            idle = "WAIT_INTERSECTION" if mission == "intersection" else "WAIT_GATE"
            accumulator.observe_state(
                mission,
                idle,
                0.04,
                EXPECTED_MISSION_CALLERIDS[mission],
            )

        # The first moving command defines departure.
        accumulator.observe_cmd_vel(
            0.50, 0.20, 0.0, "/safe_lane_controller"
        )

        active_times = {
            mission: 1.0 + 2.0 * index
            for index, mission in enumerate(MISSION_ORDER)
        }
        for index, mission in enumerate(MISSION_ORDER):
            active = active_times[mission]
            complete = active + 0.90
            accumulator.observe_manager_state(
                "/mission/state", "ACTIVE", active, EXPECTED_MANAGER_CALLERID
            )
            accumulator.observe_gate(
                mission, True, active + 0.001, EXPECTED_MANAGER_CALLERID
            )
            states = self.execution_states(mission)
            accumulator.observe_state(
                mission,
                states[0],
                active + 0.10,
                EXPECTED_MISSION_CALLERIDS[mission],
            )

            if mission in COMMON_DIAGNOSTIC_TOPICS:
                for offset, progress in ((0.20, 0.10), (0.70, 0.95)):
                    accumulator.observe_diagnostics(
                        mission,
                        self.diagnostic(
                            progress,
                            0.01 + index * 0.001,
                            -0.004 - index * 0.001,
                            0.02 - index * 0.001,
                            math.inf if mission == "intersection" else 0.03,
                            0.10,
                        ),
                        active + offset,
                        EXPECTED_MISSION_CALLERIDS[mission],
                    )

            if mission == "intersection":
                accumulator.observe_cmd_vel(
                    active + 0.20, 0.12, 0.10, EXPECTED_MISSION_CALLERIDS[mission]
                )
                accumulator.observe_state(
                    mission,
                    states[1],
                    active + 0.35,
                    EXPECTED_MISSION_CALLERIDS[mission],
                )
                accumulator.observe_cmd_vel(
                    active + 0.40, 0.10, 0.05, "/safe_lane_controller"
                )
                accumulator.observe_state(
                    mission,
                    states[2],
                    active + 0.55,
                    EXPECTED_MISSION_CALLERIDS[mission],
                )
                accumulator.observe_cmd_vel(
                    active + 0.60, 0.10, 0.10, EXPECTED_MISSION_CALLERIDS[mission]
                )
            elif mission == "level_crossing":
                accumulator.observe_state(
                    mission,
                    states[1],
                    active + 0.30,
                    EXPECTED_MISSION_CALLERIDS[mission],
                )
                # This controller is intentionally a zero-only owner.
                accumulator.observe_cmd_vel(
                    active + 0.35, 0.0, 0.0, EXPECTED_MISSION_CALLERIDS[mission]
                )
                accumulator.observe_state(
                    mission,
                    states[2],
                    active + 0.65,
                    EXPECTED_MISSION_CALLERIDS[mission],
                )
            else:
                accumulator.observe_cmd_vel(
                    active + 0.20,
                    0.12,
                    0.10,
                    EXPECTED_MISSION_CALLERIDS[mission],
                )

            if mission == "parking":
                accumulator.observe_state(
                    mission,
                    "BACK_OUT",
                    active + 0.30,
                    EXPECTED_MISSION_CALLERIDS[mission],
                )
                for offset in (0.40, 0.45, 0.50):
                    accumulator.observe_cmd_vel(
                        active + offset,
                        -0.08,
                        0.0,
                        EXPECTED_MISSION_CALLERIDS[mission],
                    )

            accumulator.observe_state(
                mission,
                "COMPLETE",
                complete,
                EXPECTED_MISSION_CALLERIDS[mission],
            )
            if mission != MISSION_ORDER[-1]:
                next_mission = MISSION_ORDER[index + 1]
                accumulator.observe_manager_state(
                    "/mission/state",
                    "SEEKING",
                    complete + 0.01,
                    EXPECTED_MANAGER_CALLERID,
                )
                accumulator.observe_manager_state(
                    "/mission/current",
                    next_mission,
                    complete + 0.02,
                    EXPECTED_MANAGER_CALLERID,
                )
                accumulator.observe_sequence_index(
                    index + 1,
                    complete + 0.03,
                    EXPECTED_MANAGER_CALLERID,
                )
            else:
                accumulator.observe_manager_state(
                    "/mission/state",
                    "COMPLETE",
                    complete + 0.01,
                    EXPECTED_MANAGER_CALLERID,
                )
                accumulator.observe_manager_state(
                    "/mission/current",
                    "",
                    complete + 0.02,
                    EXPECTED_MANAGER_CALLERID,
                )
                accumulator.observe_sequence_index(
                    len(MISSION_ORDER),
                    complete + 0.03,
                    EXPECTED_MANAGER_CALLERID,
                )
            for gate_mission in MISSION_ORDER:
                accumulator.observe_gate(
                    gate_mission,
                    False,
                    complete + 0.04,
                    EXPECTED_MANAGER_CALLERID,
                )
            accumulator.observe_cmd_vel(
                complete + 0.05, 0.20, 0.0, "/safe_lane_controller"
            )

        # Sustained detector output from departure through the full-footprint
        # finish crossing, with the real measured processing latency scale.
        for index in range(400):
            stamp = 0.50 + index / 30.0
            accumulator.observe_lane_centerline(
                stamp, stamp - 0.034, "/detect_lane"
            )

        # Sparse ground truth is enough for this pure accumulator fixture; all
        # increments remain below the configured physical speed guard.
        accumulator.observe_pose(12.80, 0.98, -1.75, 0.0, 0.20, 0.0)
        accumulator.observe_pose(13.20, 1.08, -1.75, 0.0, 0.20, 0.0)
        return accumulator

    def result(self, accumulator=None):
        return (accumulator or self.complete_accumulator()).result(
            0.0, 14.0, "/tmp/official.bag", "gazebo_model_states"
        )

    def test_complete_run_passes_and_reports_requested_metrics(self):
        result = self.result()

        self.assertTrue(result["validation"]["pass"], result["validation"])
        self.assertEqual(
            result["mission_sequence"]["observed_completion_order"],
            list(MISSION_ORDER),
        )
        parking = result["common_path_diagnostics"]["parking"]
        self.assertAlmostEqual(parking["maximum_position_error_m"], 0.012)
        self.assertAlmostEqual(
            parking["maximum_absolute_cross_track_error_m"], 0.006
        )
        self.assertAlmostEqual(parking["minimum_line_clearance_m"], 0.018)
        self.assertEqual(
            result["common_path_diagnostics"]["intersection"][
                "minimum_obstacle_clearance_m"
            ],
            None,
        )
        self.assertTrue(result["parking_reverse"]["pass"])
        self.assertTrue(result["finish"]["pass"])
        self.assertAlmostEqual(
            result["performance"]["departure_to_finish_s"], 12.5, places=1
        )
        self.assertEqual(
            result["performance"]["total_elapsed_s"],
            result["performance"]["departure_to_finish_s"],
        )
        self.assertAlmostEqual(
            result["performance"]["mission_elapsed_s"]["parking"], 0.9
        )
        self.assertGreaterEqual(
            result["lane_centerline_rate"]["overall_rate_hz"], 29.9
        )
        continuity = result["cmd_vel"]["nonparking_handoff_continuity"]
        self.assertTrue(continuity["pass"], continuity)
        self.assertEqual(len(continuity["excluded"]), 2)

    def test_level_crossing_physical_stop_zero_is_allowed(self):
        continuity = self.result()["cmd_vel"]["nonparking_handoff_continuity"]
        crossing_owner = EXPECTED_MISSION_CALLERIDS["level_crossing"]

        entry = next(
            event for event in continuity["checked"]
            if event["to"] == crossing_owner
        )
        exit_event = next(
            event for event in continuity["checked"]
            if event["from"] == crossing_owner
        )

        self.assertTrue(entry["next_command_is_zero"])
        self.assertTrue(entry["next_zero_allowed_for_physical_stop"])
        self.assertTrue(entry["pass"])
        self.assertTrue(exit_event["previous_command_is_zero"])
        self.assertTrue(
            exit_event["previous_zero_allowed_for_physical_stop"]
        )
        self.assertFalse(exit_event["next_command_is_zero"])
        self.assertTrue(exit_event["pass"])

    def test_level_crossing_lane_return_full_zero_is_rejected(self):
        accumulator = self.complete_accumulator()
        crossing_owner = EXPECTED_MISSION_CALLERIDS["level_crossing"]
        for index, run in enumerate(accumulator.cmd_runs[:-1]):
            if run["callerid"] == crossing_owner:
                lane_return = accumulator.cmd_runs[index + 1]
                lane_return["first_linear_velocity_mps"] = 0.0
                lane_return["first_angular_velocity_radps"] = 0.0
                break
        else:
            self.fail("level-crossing ownership run is missing")

        continuity = self.result(accumulator)["cmd_vel"][
            "nonparking_handoff_continuity"
        ]
        failed = [event for event in continuity["checked"] if not event["pass"]]

        self.assertFalse(continuity["pass"])
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["from"], crossing_owner)
        self.assertEqual(failed[0]["to"], "/safe_lane_controller")
        self.assertTrue(failed[0]["previous_command_is_zero"])
        self.assertTrue(failed[0]["next_command_is_zero"])

    def test_nonparking_new_owner_full_zero_is_rejected(self):
        accumulator = self.complete_accumulator()
        obstacle_owner = EXPECTED_MISSION_CALLERIDS["obstacle"]
        obstacle_run = next(
            run for run in accumulator.cmd_runs if run["callerid"] == obstacle_owner
        )
        obstacle_run["first_linear_velocity_mps"] = 0.0
        obstacle_run["first_angular_velocity_radps"] = 0.0

        result = self.result(accumulator)
        continuity = result["cmd_vel"]["nonparking_handoff_continuity"]

        self.assertFalse(continuity["pass"])
        self.assertIn(
            "nonparking_handoff_continuity",
            result["validation"]["failed_checks"],
        )
        failed = [event for event in continuity["checked"] if not event["pass"]]
        self.assertEqual(failed[0]["to"], obstacle_owner)

    def test_preexisting_lane_zero_does_not_become_a_handoff_failure(self):
        accumulator = self.complete_accumulator()
        lane_run = accumulator.cmd_runs[0]
        lane_run["last_linear_velocity_mps"] = 0.0
        lane_run["last_angular_velocity_radps"] = 0.0

        continuity = self.result(accumulator)["cmd_vel"][
            "nonparking_handoff_continuity"
        ]

        self.assertTrue(continuity["pass"], continuity)
        first = continuity["checked"][0]
        self.assertTrue(first["preexisting_lane_zero"])
        self.assertFalse(first["next_command_is_zero"])

    def test_mission_zero_before_lane_return_is_rejected(self):
        accumulator = self.complete_accumulator()
        intersection_owner = EXPECTED_MISSION_CALLERIDS["intersection"]
        mission_run = next(
            run
            for run in accumulator.cmd_runs
            if run["callerid"] == intersection_owner
        )
        mission_run["last_linear_velocity_mps"] = 0.0
        mission_run["last_angular_velocity_radps"] = 0.0

        continuity = self.result(accumulator)["cmd_vel"][
            "nonparking_handoff_continuity"
        ]

        self.assertFalse(continuity["pass"])
        failed = [event for event in continuity["checked"] if not event["pass"]]
        self.assertEqual(failed[0]["from"], intersection_owner)

    def test_pure_rotation_at_handoff_is_not_a_full_stop(self):
        accumulator = self.complete_accumulator()
        obstacle_owner = EXPECTED_MISSION_CALLERIDS["obstacle"]
        obstacle_run = next(
            run for run in accumulator.cmd_runs if run["callerid"] == obstacle_owner
        )
        obstacle_run["first_linear_velocity_mps"] = 0.0
        obstacle_run["first_angular_velocity_radps"] = 0.10

        continuity = self.result(accumulator)["cmd_vel"][
            "nonparking_handoff_continuity"
        ]

        self.assertTrue(continuity["pass"], continuity)

    def test_raw_and_motion_ownership_sequences_match_production(self):
        cmd = self.result()["cmd_vel"]

        self.assertEqual(
            cmd["observed_raw_callerid_order"],
            list(EXPECTED_RAW_CMD_VEL_CALLERID_ORDER),
        )
        self.assertEqual(
            cmd["observed_motion_callerid_order"],
            list(EXPECTED_MOTION_CMD_VEL_CALLERID_ORDER),
        )
        self.assertIn(
            "/level_crossing_lidar_controller",
            cmd["observed_raw_callerid_order"],
        )
        self.assertNotIn(
            "/level_crossing_lidar_controller",
            cmd["observed_motion_callerid_order"],
        )

    def test_removed_stop_preparation_states_are_not_accepted(self):
        self.assertNotIn(
            "PREPARE_ENTRY_PATH", MISSION_EXECUTION_STATES["intersection"]
        )
        self.assertNotIn(
            "PREPARE_EXIT_PATH", MISSION_EXECUTION_STATES["intersection"]
        )
        self.assertNotIn("VERIFY_EXIT", MISSION_EXECUTION_STATES["zigzag"])

    def test_zero_only_and_moving_extra_publishers_are_rejected(self):
        zero = self.complete_accumulator()
        zero.observe_cmd_vel(13.30, 0.0, 0.0, "/unexpected_zero_owner")
        zero_result = self.result(zero)
        self.assertFalse(zero_result["cmd_vel"]["raw_callerid_order_pass"])

        moving = self.complete_accumulator()
        moving.observe_cmd_vel(13.30, 0.1, 0.0, "/unexpected_motion_owner")
        moving_result = self.result(moving)
        self.assertFalse(moving_result["cmd_vel"]["raw_callerid_order_pass"])
        self.assertFalse(moving_result["cmd_vel"]["motion_callerid_order_pass"])

    def test_bag_that_starts_at_motion_fails_predeparture_validation(self):
        accumulator = IntegratedRunAccumulator(minimum_predeparture_duration=0.20)
        accumulator.observe_pose(10.0, 0.8, -1.747, 0.0, 0.0, 0.0)
        accumulator.observe_cmd_vel(10.05, 0.2, 0.0, "/safe_lane_controller")
        accumulator.observe_pose(10.06, 0.81, -1.747, 0.0, 0.2, 0.0)

        result = accumulator.result(10.0, 11.0, "late.bag", "/odom")

        self.assertFalse(result["predeparture"]["pass"])
        self.assertIn(
            "bag_started_before_departure",
            result["validation"]["failed_checks"],
        )

    def test_gazebo_spawn_impulse_is_scoped_to_settling_window(self):
        accumulator = self.complete_accumulator()
        spawn_impulse = 0.05228819636778588
        accumulator.predeparture_pose_samples.append(
            (0.09, 0.8, -1.747, 0.00007, 0.0072, spawn_impulse)
        )
        accumulator.predeparture_pose_samples.sort(key=lambda sample: sample[0])

        result = self.result(accumulator)
        predeparture = result["predeparture"]

        self.assertTrue(predeparture["pass"])
        self.assertTrue(predeparture["pose_settling_applied"])
        self.assertAlmostEqual(
            predeparture["configured_gazebo_pose_settling_duration_s"], 0.10
        )
        self.assertAlmostEqual(
            predeparture["applied_pose_settling_duration_s"], 0.10
        )
        self.assertAlmostEqual(
            predeparture["stationary_validation_start_time_s"], 0.15
        )
        self.assertAlmostEqual(predeparture["raw_predeparture_duration_s"], 0.45)
        self.assertAlmostEqual(predeparture["predeparture_duration_s"], 0.35)
        self.assertAlmostEqual(
            predeparture["maximum_settling_angular_speed_radps"],
            spawn_impulse,
        )
        self.assertEqual(
            predeparture["maximum_predeparture_angular_speed_radps"], 0.0
        )

    def test_gazebo_impulse_after_settling_is_not_ignored(self):
        accumulator = self.complete_accumulator()
        post_settling_angular_speed = 0.0523
        accumulator.predeparture_pose_samples.append(
            (0.16, 0.8, -1.747, 0.00007, 0.0, post_settling_angular_speed)
        )
        accumulator.predeparture_pose_samples.sort(key=lambda sample: sample[0])

        predeparture = self.result(accumulator)["predeparture"]

        self.assertFalse(predeparture["pass"])
        self.assertAlmostEqual(
            predeparture["maximum_predeparture_angular_speed_radps"],
            post_settling_angular_speed,
        )

    def test_gazebo_settling_does_not_hide_pose_command_or_departure(self):
        wrong_pose = self.complete_accumulator()
        wrong_pose.predeparture_pose_samples.append(
            (0.09, 0.90, -1.747, 0.0, 0.0, 0.0523)
        )
        wrong_pose.predeparture_pose_samples.sort(key=lambda sample: sample[0])
        wrong_pose_result = self.result(wrong_pose)["predeparture"]
        self.assertFalse(wrong_pose_result["pass"])
        self.assertAlmostEqual(
            wrong_pose_result["maximum_predeparture_position_error_m"], 0.10
        )

        early_command = self.complete_accumulator()
        early_command.first_motion_command_time = 0.10
        early_command_result = self.result(early_command)["predeparture"]
        self.assertFalse(early_command_result["pass"])
        self.assertAlmostEqual(early_command_result["departure_time_s"], 0.10)
        self.assertLess(early_command_result["predeparture_duration_s"], 0.0)

        early_pose_motion = self.complete_accumulator()
        early_pose_motion.first_pose_angular_departure = 0.09
        early_pose_result = self.result(early_pose_motion)["predeparture"]
        self.assertFalse(early_pose_result["pass"])
        self.assertAlmostEqual(early_pose_result["departure_time_s"], 0.09)

    def test_odom_predeparture_does_not_apply_gazebo_settling(self):
        accumulator = self.complete_accumulator()
        angular_speed = 0.0523
        accumulator.predeparture_pose_samples.append(
            (0.09, 0.8, -1.747, 0.0, 0.0, angular_speed)
        )
        accumulator.predeparture_pose_samples.sort(key=lambda sample: sample[0])

        predeparture = accumulator.result(
            0.0, 14.0, "/tmp/official.bag", "/odom"
        )["predeparture"]

        self.assertFalse(predeparture["pass"])
        self.assertFalse(predeparture["pose_settling_applied"])
        self.assertEqual(predeparture["applied_pose_settling_duration_s"], 0.0)
        self.assertAlmostEqual(
            predeparture["maximum_predeparture_angular_speed_radps"],
            angular_speed,
        )

    def test_wrong_start_yaw_and_late_first_pose_are_rejected(self):
        yaw = self.complete_accumulator()
        yaw.first_pose = (0.05, 0.8, -1.747, math.pi, 0.0, 0.0)
        yaw.predeparture_pose_samples = [
            (stamp, 0.8, -1.747, math.pi, 0.0, 0.0)
            for stamp in (0.05, 0.10, 0.15, 0.20)
        ]
        self.assertFalse(self.result(yaw)["predeparture"]["pass"])

        late = self.complete_accumulator()
        late.first_pose = (0.80, 0.8, -1.747, 0.0, 0.0, 0.0)
        late.predeparture_pose_samples = [
            (stamp, 0.8, -1.747, 0.0, 0.0, 0.0)
            for stamp in (0.80, 0.85)
        ]
        self.assertFalse(self.result(late)["predeparture"]["pass"])

    def test_manager_evidence_and_exact_complete_are_required(self):
        missing = self.complete_accumulator()
        missing.manager_transitions = {topic: [] for topic in MANAGER_TOPICS}
        missing.manager_callerids = {topic: set() for topic in MANAGER_TOPICS}
        self.assertFalse(self.result(missing)["validation"]["pass"])

        prefix = self.complete_accumulator()
        transitions = prefix.state_transitions["intersection"]
        prefix.state_transitions["intersection"] = [
            (stamp, "COMPLETE_PENDING" if state == "COMPLETE" else state, callerid)
            for stamp, state, callerid in transitions
        ]
        prefix.completion_events = [
            event for event in prefix.completion_events if event[1] != "intersection"
        ]
        self.assertFalse(
            self.result(prefix)["mission_sequence"]["missions"]["intersection"]["complete"]
        )

        lowercase = self.complete_accumulator()
        lowercase.state_transitions["intersection"] = [
            (stamp, "complete" if state == "COMPLETE" else state, callerid)
            for stamp, state, callerid in lowercase.state_transitions["intersection"]
        ]
        lowercase.completion_events = [
            event for event in lowercase.completion_events if event[1] != "intersection"
        ]
        self.assertFalse(
            self.result(lowercase)["mission_sequence"]["missions"]["intersection"]["complete"]
        )

        lowercase_active = self.complete_accumulator()
        lowercase_active.manager_transitions["/mission/state"] = [
            (stamp, "active" if state == "ACTIVE" else state, callerid)
            for stamp, state, callerid in lowercase_active.manager_transitions[
                "/mission/state"
            ]
        ]
        self.assertFalse(
            self.result(lowercase_active)["mission_sequence"]["manager"][
                "active_order_pass"
            ]
        )

    def test_manager_index_gate_and_publishers_are_required(self):
        bad_index = self.complete_accumulator()
        bad_index.sequence_index_transitions.pop()
        self.assertFalse(
            self.result(bad_index)["mission_sequence"]["manager"]["sequence_index_pass"]
        )

        bad_gate = self.complete_accumulator()
        bad_gate.gate_events["parking"] = [
            event for event in bad_gate.gate_events["parking"] if not event[1]
        ]
        self.assertFalse(
            self.result(bad_gate)["mission_sequence"]["manager"]["gate_evidence_pass"]
        )

        bad_publisher = self.complete_accumulator()
        bad_publisher.gate_callerids["zigzag"].add("/manual_gate")
        self.assertFalse(
            self.result(bad_publisher)["mission_sequence"]["manager"]["publisher_pass"]
        )

    def test_rate_summary_rejects_one_large_gap_and_partial_coverage(self):
        steady = summarize_rate([index / 30.0 for index in range(60)])
        one_gap = summarize_rate(
            [index / 30.0 for index in range(30)]
            + [300.0 + index / 30.0 for index in range(30)]
        )
        partial = summarize_rate(
            [index / 30.0 for index in range(90)],
            window_start=0.0,
            window_end=7.0,
        )

        self.assertTrue(steady["pass"])
        self.assertFalse(one_gap["pass"])
        self.assertFalse(partial["pass"])

    def test_lane_source_stamp_staleness_and_bursts_fail(self):
        stale = self.complete_accumulator()
        stale.lane_samples = [
            (stamp, source - 1.0, callerid)
            for stamp, source, callerid in stale.lane_samples
        ]
        self.assertFalse(self.result(stale)["lane_centerline_rate"]["pass"])

        burst = self.complete_accumulator()
        burst.lane_samples = [
            (stamp, stamp - 0.034, "/detect_lane")
            for stamp in (
                [0.50 + index / 30.0 for index in range(30)]
                + [12.00 + index / 30.0 for index in range(30)]
            )
        ]
        self.assertFalse(self.result(burst)["lane_centerline_rate"]["pass"])

    def test_diagnostics_need_exact_schema_finite_core_and_execution_window(self):
        malformed = self.complete_accumulator()
        stamp, values, callerid = malformed.diagnostic_samples["obstacle"][0]
        malformed.diagnostic_samples["obstacle"][0] = (
            stamp,
            tuple(values) + (99.0,),
            callerid,
        )
        self.assertFalse(
            self.result(malformed)["common_path_diagnostics"]["obstacle"]["pass"]
        )

        nonfinite = self.complete_accumulator()
        nonfinite.diagnostic_samples["zigzag"] = [
            (7.2, tuple([math.inf] * 13), EXPECTED_MISSION_CALLERIDS["zigzag"])
        ]
        self.assertFalse(
            self.result(nonfinite)["common_path_diagnostics"]["zigzag"]["pass"]
        )

        mixed = self.complete_accumulator()
        mixed.diagnostic_samples["zigzag"].append(
            (7.3, tuple([math.inf] * 13), EXPECTED_MISSION_CALLERIDS["zigzag"])
        )
        self.assertFalse(
            self.result(mixed)["common_path_diagnostics"]["zigzag"]["pass"]
        )

        nonfinite_line = self.complete_accumulator()
        values = self.diagnostic(0.90, 0.01, 0.0, math.inf, 0.03, 0.1)
        nonfinite_line.diagnostic_samples["obstacle"].append(
            (3.8, tuple(values), EXPECTED_MISSION_CALLERIDS["obstacle"])
        )
        self.assertFalse(
            self.result(nonfinite_line)["common_path_diagnostics"]["obstacle"][
                "pass"
            ]
        )

        outside = self.complete_accumulator()
        outside.diagnostic_samples["parking"] = [
            (
                0.2,
                self.diagnostic(0.95, 0.01, 0.0, 0.02, 0.03, 0.1),
                EXPECTED_MISSION_CALLERIDS["parking"],
            )
        ]
        self.assertFalse(
            self.result(outside)["common_path_diagnostics"]["parking"]["pass"]
        )

    def test_finish_requires_oriented_footprint_heading_and_y(self):
        correct = IntegratedRunAccumulator()
        correct.observe_state(
            "tunnel", "COMPLETE", 1.0, EXPECTED_MISSION_CALLERIDS["tunnel"]
        )
        correct.observe_pose(1.1, 0.98, -1.75, 0.0, 0.2, 0.0)
        correct.observe_pose(1.5, 1.08, -1.75, 0.0, 0.2, 0.0)
        self.assertTrue(correct.result(0.0, 2.0, "x", "gazebo")["finish"]["pass"])

        angled = IntegratedRunAccumulator()
        angled.observe_state(
            "tunnel", "COMPLETE", 1.0, EXPECTED_MISSION_CALLERIDS["tunnel"]
        )
        yaw = math.radians(10.0)
        angled.observe_pose(1.1, 0.98, -1.75, yaw, 0.2, 0.0)
        # The base has crossed nominal x, but its rear-left corner has not.
        angled.observe_pose(1.5, 1.04, -1.75, yaw, 0.2, 0.0)
        self.assertFalse(angled.result(0.0, 2.0, "x", "gazebo")["finish"]["pass"])

        outside_y = IntegratedRunAccumulator()
        outside_y.observe_state(
            "tunnel", "COMPLETE", 1.0, EXPECTED_MISSION_CALLERIDS["tunnel"]
        )
        outside_y.observe_pose(1.1, 0.98, -1.85, 0.0, 0.2, 0.0)
        outside_y.observe_pose(1.5, 1.08, -1.85, 0.0, 0.2, 0.0)
        self.assertFalse(outside_y.result(0.0, 2.0, "x", "gazebo")["finish"]["pass"])

    def test_finish_before_tunnel_complete_does_not_pass(self):
        accumulator = IntegratedRunAccumulator()
        accumulator.observe_pose(0.5, 0.98, -1.75, 0.0, 0.2, 0.0)
        accumulator.observe_pose(0.9, 1.08, -1.75, 0.0, 0.2, 0.0)
        accumulator.observe_state(
            "tunnel", "COMPLETE", 1.0, EXPECTED_MISSION_CALLERIDS["tunnel"]
        )
        self.assertFalse(
            accumulator.result(0.0, 2.0, "x", "gazebo")["finish"]["pass"]
        )

        tied = IntegratedRunAccumulator()
        tied.observe_pose(0.6, 0.98, -1.75, 0.0, 0.2, 0.0)
        tied.observe_pose(1.0, 1.08, -1.75, 0.0, 0.2, 0.0)
        # Rosbag may deliver the pose connection before the state connection
        # at the same recorded timestamp.
        crossing_time = tied.east_crossings[0]["time"]
        tied.observe_state(
            "tunnel",
            "COMPLETE",
            crossing_time,
            EXPECTED_MISSION_CALLERIDS["tunnel"],
        )
        self.assertTrue(
            tied.result(0.0, 2.0, "x", "gazebo")["finish"]["pass"]
        )

        exact_boundary = IntegratedRunAccumulator()
        exact_boundary.observe_state(
            "tunnel", "COMPLETE", 0.5, EXPECTED_MISSION_CALLERIDS["tunnel"]
        )
        exact_boundary.observe_pose(
            0.9, exact_boundary.finish_x, -1.75, 0.0, 0.2, 0.0
        )
        exact_boundary.observe_pose(1.0, 1.04, -1.75, 0.0, 0.2, 0.0)
        self.assertTrue(
            exact_boundary.result(0.0, 2.0, "x", "gazebo")["finish"]["pass"]
        )

    def test_parking_requires_sustained_back_out_reverse(self):
        accumulator = self.complete_accumulator()
        accumulator.parking_back_out_samples = [
            (
                5.4,
                -0.08,
                EXPECTED_MISSION_CALLERIDS["parking"],
                "BACK_OUT",
                True,
                1,
            )
        ]
        accumulator.parking_negative = [
            (5.4, -0.08, EXPECTED_MISSION_CALLERIDS["parking"], "BACK_OUT")
        ]
        self.assertFalse(self.result(accumulator)["parking_reverse"]["pass"])

        accumulator = self.complete_accumulator()
        accumulator.parking_back_out_samples = []
        accumulator.parking_negative = []
        self.assertFalse(self.result(accumulator)["parking_reverse"]["pass"])

        alternating = self.complete_accumulator()
        alternating.parking_back_out_samples = []
        alternating.parking_negative = []
        for index, linear in enumerate((-0.08, 0.01, -0.08, 0.01, -0.08)):
            stamp = 5.4 + 0.05 * index
            reverse = linear < -0.001
            alternating.parking_back_out_samples.append(
                (
                    stamp,
                    linear,
                    EXPECTED_MISSION_CALLERIDS["parking"],
                    "BACK_OUT",
                    reverse,
                    1,
                )
            )
            if reverse:
                alternating.parking_negative.append(
                    (
                        stamp,
                        linear,
                        EXPECTED_MISSION_CALLERIDS["parking"],
                        "BACK_OUT",
                    )
                )
        self.assertFalse(self.result(alternating)["parking_reverse"]["pass"])

        reentered = self.complete_accumulator()
        reentered.parking_back_out_samples = [
            (5.40, -0.08, EXPECTED_MISSION_CALLERIDS["parking"], "BACK_OUT", True, 2),
            (5.45, -0.08, EXPECTED_MISSION_CALLERIDS["parking"], "BACK_OUT", True, 2),
            (5.50, -0.08, EXPECTED_MISSION_CALLERIDS["parking"], "BACK_OUT", True, 4),
            (5.55, -0.08, EXPECTED_MISSION_CALLERIDS["parking"], "BACK_OUT", True, 4),
        ]
        self.assertFalse(self.result(reentered)["parking_reverse"]["pass"])

    def test_manual_stop_and_pose_jump_are_rejected(self):
        manual = self.complete_accumulator()
        manual.observe_manual_stop(6.0, True, "/safe_lane_controller")
        self.assertFalse(self.result(manual)["manual_stop"]["pass"])

        jump = self.complete_accumulator()
        jump.observe_pose(13.21, 2.0, -1.75, 0.0, 0.2, 0.0)
        self.assertFalse(self.result(jump)["pose_integrity"]["pass"])

        simultaneous_jump = self.complete_accumulator()
        simultaneous_jump.observe_pose(13.20, 2.0, -1.75, 0.0, 0.2, 0.0)
        self.assertFalse(
            self.result(simultaneous_jump)["pose_integrity"]["pass"]
        )

        simultaneous_yaw_jump = self.complete_accumulator()
        simultaneous_yaw_jump.observe_pose(
            13.20, 1.08, -1.75, math.pi / 2.0, 0.2, 0.0
        )
        self.assertFalse(
            self.result(simultaneous_yaw_jump)["pose_integrity"]["pass"]
        )

    def test_failed_and_out_of_order_mission_are_both_visible(self):
        accumulator = IntegratedRunAccumulator()
        accumulator.observe_state(
            "obstacle", "COMPLETE", 2.0, EXPECTED_MISSION_CALLERIDS["obstacle"]
        )
        accumulator.observe_state(
            "intersection", "FAILED", 3.0, EXPECTED_MISSION_CALLERIDS["intersection"]
        )
        accumulator.observe_state(
            "intersection", "COMPLETE", 4.0, EXPECTED_MISSION_CALLERIDS["intersection"]
        )

        mission = accumulator.result(0.0, 5.0, "bad.bag", "/odom")[
            "mission_sequence"
        ]

        self.assertTrue(mission["any_failed"])
        self.assertFalse(mission["completion_order_pass"])
        self.assertEqual(
            mission["observed_completion_order"], ["obstacle", "intersection"]
        )

    def test_connection_header_and_model_index_are_decoded_without_ros(self):
        self.assertEqual(
            callerid_from_header({"callerid": b"/parking_mission_controller"}),
            "/parking_mission_controller",
        )
        orientation = SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
        zero_twist = SimpleNamespace(
            linear=SimpleNamespace(x=0.0, y=0.0),
            angular=SimpleNamespace(z=0.0),
        )
        moving_twist = SimpleNamespace(
            linear=SimpleNamespace(x=0.1, y=0.0),
            angular=SimpleNamespace(z=-0.2),
        )
        model = SimpleNamespace(
            name=["ground", "custom_autorace"],
            pose=[
                SimpleNamespace(
                    position=SimpleNamespace(x=0.0, y=0.0), orientation=orientation
                ),
                SimpleNamespace(
                    position=SimpleNamespace(x=0.8, y=-1.747), orientation=orientation
                ),
            ],
            twist=[zero_twist, moving_twist],
        )

        pose, index = extract_model_pose(model, "custom_autorace")

        self.assertEqual(index, 1)
        self.assertEqual(pose, (0.8, -1.747, 0.0, 0.1, 0.2))


if __name__ == "__main__":
    unittest.main()
