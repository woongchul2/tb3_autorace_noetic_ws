#!/usr/bin/env python3

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET


PACKAGE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_SRC = PACKAGE_DIR.parent
TURTLEBOT3 = WORKSPACE_SRC / "turtlebot3_simulations" / "turtlebot3_gazebo"
SERVER_PATH = PACKAGE_DIR / "nodes" / "gazebo_course_server.py"
SPEC = importlib.util.spec_from_file_location("gazebo_course_server", SERVER_PATH)
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GazeboCourseServerTest(unittest.TestCase):
    def setUp(self):
        self.world = (
            TURTLEBOT3 / "worlds" / "turtlebot3_autorace_2020.world"
        )
        self.course = (
            TURTLEBOT3 / "models" / "turtlebot3_autorace_2020" / "course"
        )
        self.tunnel_layouts = (
            PACKAGE_DIR / "config" / "tunnel_obstacle_layouts.yaml"
        )

    @staticmethod
    def tunnel_centres(world_path, prefix):
        world = ET.parse(str(world_path)).getroot().find("./world")
        model_path = (
            "./state/model[@name='tunnel_obstacle']"
            if prefix == "state"
            else "./model[@name='tunnel_obstacle']"
        )
        model = world.find(model_path)
        model_pose = [float(value) for value in model.find("pose").text.split()]
        centres = {}
        for link in model.findall("link"):
            pose = [float(value) for value in link.find("pose").text.split()]
            if prefix == "state":
                centres[link.get("name")] = tuple(pose[:2])
            else:
                centres[link.get("name")] = (
                    model_pose[0] + pose[0],
                    model_pose[1] + pose[1],
                )
        return centres

    def test_staging_changes_unique_uri_and_preserves_package_model(self):
        source_texture = self.course / "materials" / "textures" / "course.png"
        source_digest = digest(source_texture)
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            selected = temporary / "changed.png"
            selected.write_bytes(source_texture.read_bytes() + b"fixture revision")
            staged_world, model_root, metadata = SERVER.prepare_course_world(
                selected, self.world, self.course, temporary / "cache"
            )

            model_name = metadata["model_name"]
            staged_model = model_root / model_name
            staged_texture = (
                staged_model / "materials" / "textures" / "course.png"
            )
            self.assertEqual(staged_texture.read_bytes(), selected.read_bytes())
            self.assertEqual(digest(source_texture), source_digest)
            self.assertIn("model://%s" % model_name, staged_world.read_text())
            self.assertNotIn(SERVER.SOURCE_MODEL_URI, staged_world.read_text())
            self.assertIn(
                "model://%s/materials/textures" % model_name,
                (staged_model / "model.sdf").read_text(),
            )

            repeated = SERVER.prepare_course_world(
                selected, self.world, self.course, temporary / "cache"
            )
            self.assertEqual(repeated[0], staged_world)
            self.assertEqual(repeated[2]["cache_format"], 2)
            self.assertEqual(repeated[2]["texture_sha256"], digest(selected))
            self.assertEqual(repeated[2]["texture_width"], 520)
            self.assertEqual(repeated[2]["texture_height"], 496)

            staged_texture.write_bytes(b"damaged")
            with self.assertRaisesRegex(ValueError, "integrity"):
                SERVER.prepare_course_world(
                    selected, self.world, self.course, temporary / "cache"
                )

    def test_non_png_texture_is_rejected_before_gazebo_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            selected = temporary / "not-a-course.png"
            selected.write_bytes(b"not a png")
            with self.assertRaisesRegex(ValueError, "valid PNG"):
                SERVER.prepare_course_world(
                    selected, self.world, self.course, temporary / "cache"
                )

    def test_named_tunnel_layouts_are_distinct_reproducible_and_immutable(self):
        expected = {
            "layout_a": {
                "obstacle_1": (-0.618230, -1.338744),
                "obstacle_2": (-1.365179, -1.414273),
                "obstacle_3": (-1.140942, -0.627116),
            },
            "layout_b": {
                "obstacle_1": (-0.829416, -0.703490),
                "obstacle_2": (-0.753887, -1.450439),
                "obstacle_3": (-1.541044, -1.226202),
            },
            "layout_c": {
                "obstacle_1": (-1.464670, -0.914676),
                "obstacle_2": (-0.717721, -0.839147),
                "obstacle_3": (-0.941958, -1.626304),
            },
        }
        source_digest = digest(self.world)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            staged_paths = []
            for layout_name, expected_centres in expected.items():
                staged_world, metadata = SERVER.prepare_tunnel_world(
                    layout_name,
                    self.tunnel_layouts,
                    self.world,
                    cache,
                )
                staged_paths.append(staged_world)
                self.assertEqual(metadata["layout"], layout_name)
                actual = self.tunnel_centres(staged_world, "")
                saved = self.tunnel_centres(staged_world, "state")
                self.assertEqual(set(actual), set(expected_centres))
                for obstacle, centre in expected_centres.items():
                    self.assertAlmostEqual(actual[obstacle][0], centre[0], places=6)
                    self.assertAlmostEqual(actual[obstacle][1], centre[1], places=6)
                    self.assertAlmostEqual(saved[obstacle][0], centre[0], places=6)
                    self.assertAlmostEqual(saved[obstacle][1], centre[1], places=6)

                repeated_world, repeated_metadata = SERVER.prepare_tunnel_world(
                    layout_name,
                    self.tunnel_layouts,
                    self.world,
                    cache,
                )
                self.assertEqual(repeated_world, staged_world)
                self.assertEqual(repeated_metadata["centres"], metadata["centres"])

            self.assertEqual(len(set(staged_paths)), 3)
            self.assertEqual(digest(self.world), source_digest)

    def test_auto_tunnel_layout_advances_once_per_server_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            self.assertEqual(
                SERVER.select_tunnel_layout(
                    self.tunnel_layouts, "auto", cache
                ),
                "layout_b",
            )
            self.assertEqual(
                SERVER.select_tunnel_layout(
                    self.tunnel_layouts, "auto", cache
                ),
                "layout_c",
            )
            self.assertEqual(
                SERVER.select_tunnel_layout(
                    self.tunnel_layouts, "auto", cache
                ),
                "layout_a",
            )
            self.assertEqual(
                SERVER.select_tunnel_layout(
                    self.tunnel_layouts, "layout_c", cache
                ),
                "layout_c",
            )
            self.assertEqual(
                SERVER.select_tunnel_layout(
                    self.tunnel_layouts, "auto", cache
                ),
                "layout_b",
            )

    def test_unknown_tunnel_layout_is_rejected_before_gazebo_starts(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unknown tunnel obstacle layout"):
                SERVER.prepare_tunnel_world(
                    "layout_missing",
                    self.tunnel_layouts,
                    self.world,
                    Path(directory) / "cache",
                )

    def test_gazebo_client_uses_server_world_without_selecting_a_layout(self):
        environment = {
            SERVER.TEXTURE_ENVIRONMENT: SERVER.STOCK_TEXTURE,
            SERVER.TUNNEL_LAYOUT_ENVIRONMENT: SERVER.AUTO_TUNNEL_LAYOUT,
            SERVER.TUNNEL_LAYOUT_CONFIG_ENVIRONMENT: str(self.tunnel_layouts),
        }
        rospack = mock.Mock()
        rospack.get_path.return_value = str(TURTLEBOT3)
        with mock.patch.dict(SERVER.os.environ, environment, clear=True), mock.patch.object(
            SERVER.rospkg, "RosPack", return_value=rospack
        ), mock.patch.object(
            SERVER.sys, "argv", ["gazebo_course_server.py", "--client"]
        ), mock.patch.object(
            SERVER, "select_tunnel_layout"
        ) as select_layout, mock.patch.object(
            SERVER, "prepare_tunnel_world"
        ) as prepare_layout, mock.patch.object(
            SERVER, "_gazebo_ros_executable", return_value=Path("/bin/true")
        ), mock.patch.object(
            SERVER.os, "execve", side_effect=RuntimeError("exec intercepted")
        ):
            with self.assertRaisesRegex(RuntimeError, "exec intercepted"):
                SERVER.main()

        select_layout.assert_not_called()
        prepare_layout.assert_not_called()

    def test_launch_uses_the_same_argument_for_gazebo_and_benchmark_nodes(self):
        description = ET.parse(
            PACKAGE_DIR / "launch" / "gazebo_autorace.launch"
        ).getroot()
        argument_names = {
            argument.attrib.get("name") for argument in description.findall("arg")
        }
        self.assertNotIn("use_course_texture_override", argument_names)

        root = ET.parse(
            WORKSPACE_SRC
            / "custom_autorace_bringup"
            / "launch"
            / "gazebo_lane_path_test.launch"
        ).getroot()
        include = root.find("include")
        include_values = {
            argument.attrib.get("name"): argument.attrib.get("value")
            for argument in include.findall("arg")
        }
        self.assertEqual(include_values["course_texture"], "$(arg course_texture)")
        texture_parameters = [
            parameter.attrib.get("value")
            for node in root.findall("node")
            for parameter in node.findall("param")
            if parameter.attrib.get("name")
            == "lane_lookahead/texture/file_override"
        ]
        self.assertEqual(texture_parameters, ["$(arg course_texture)"])


if __name__ == "__main__":
    unittest.main()
