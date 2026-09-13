#!/usr/bin/env python3

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
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
