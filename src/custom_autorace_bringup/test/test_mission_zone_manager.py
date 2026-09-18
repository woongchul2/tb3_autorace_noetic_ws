#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock

import rospy
from std_msgs.msg import Header

from custom_autorace_bringup.mission_zone import MissionZoneSequence


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "nodes"
    / "mission_zone_manager.py"
)
SPEC = importlib.util.spec_from_file_location(
    "mission_zone_manager_under_test", str(MODULE_PATH)
)
manager_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manager_module
SPEC.loader.exec_module(manager_module)


class MissionZoneManagerContractTest(unittest.TestCase):
    def make_manager(self):
        manager = manager_module.MissionZoneManager.__new__(
            manager_module.MissionZoneManager
        )
        manager.lock = threading.RLock()
        manager.ready_max_age = 0.50
        manager.ready_future_tolerance = 0.05
        manager.ready_before_arm_tolerance = 0.0
        manager.sequence = MissionZoneSequence(
            [{"name": "intersection"}],
            armed_at=10.0,
            initial_generation=17,
        )
        manager._log_transition = mock.Mock()
        manager._publish_status = mock.Mock()
        manager._publish_markers = mock.Mock()
        return manager

    @staticmethod
    def ready(frame_id):
        return Header(
            seq=17,
            stamp=rospy.Time.from_sec(10.10),
            frame_id=frame_id,
        )

    def test_empty_and_wrong_ready_frames_cannot_open_gate(self):
        mission = {
            "name": "intersection",
            "ready_topic": "/mission/ready/intersection",
        }
        for frame_id in ("", "obstacle"):
            with self.subTest(frame_id=frame_id or "empty"):
                manager = self.make_manager()
                with mock.patch.object(
                    manager_module.rospy.Time,
                    "now",
                    return_value=rospy.Time.from_sec(10.20),
                ), mock.patch.object(manager_module.rospy, "logwarn_throttle"):
                    manager.ready_callback(self.ready(frame_id), mission)
                self.assertEqual(
                    manager.sequence.state, MissionZoneSequence.ARMED
                )
                manager._publish_status.assert_not_called()

    def test_exact_ready_frame_activates_matching_generation(self):
        manager = self.make_manager()
        mission = {
            "name": "intersection",
            "ready_topic": "/mission/ready/intersection",
        }
        with mock.patch.object(
            manager_module.rospy.Time,
            "now",
            return_value=rospy.Time.from_sec(10.20),
        ):
            manager.ready_callback(self.ready("intersection"), mission)
        self.assertEqual(manager.sequence.state, MissionZoneSequence.ACTIVE)
        self.assertEqual(manager._publish_status.call_count, 2)

    def test_sim_time_waits_for_nonzero_clock_before_first_arm_stamp(self):
        zero = rospy.Time()
        valid = rospy.Time.from_sec(12.5)
        with mock.patch.object(
            manager_module.rospy.Time,
            "now",
            side_effect=(zero, zero, valid),
        ), mock.patch.object(
            manager_module.rospy,
            "get_param",
            return_value=True,
        ), mock.patch.object(
            manager_module.rospy,
            "is_shutdown",
            return_value=False,
        ), mock.patch.object(
            manager_module.rospy,
            "loginfo",
        ), mock.patch.object(
            manager_module.time,
            "sleep",
        ) as wall_sleep:
            result = manager_module.MissionZoneManager._wait_for_initial_ros_time()
        self.assertEqual(result, valid)
        self.assertEqual(wall_sleep.call_count, 2)

    def test_wall_time_mode_does_not_wait_even_if_mock_clock_is_zero(self):
        with mock.patch.object(
            manager_module.rospy.Time,
            "now",
            return_value=rospy.Time(),
        ), mock.patch.object(
            manager_module.rospy,
            "get_param",
            return_value=False,
        ), mock.patch.object(manager_module.time, "sleep") as wall_sleep:
            result = manager_module.MissionZoneManager._wait_for_initial_ros_time()
        self.assertEqual(result, rospy.Time())
        wall_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
