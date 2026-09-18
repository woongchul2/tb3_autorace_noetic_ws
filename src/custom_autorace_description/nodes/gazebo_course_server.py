#!/usr/bin/env python3
"""Start Gazebo with an isolated course texture and tunnel layout.

The upstream AutoRace world addresses its floor through a fixed ``model://``
URI.  This wrapper copies only that small model into an immutable cache,
changes the world to a digest-specific model URI, and then execs the normal
``gazebo_ros/gzserver`` wrapper.  It can also rotate the three stock tunnel
cylinders into a named, deterministic regression layout.  Package files are
never modified.
"""

import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET

import rospkg
import yaml


SOURCE_MODEL_URI = "model://turtlebot3_autorace_2020/course"
TEXTURE_ENVIRONMENT = "CUSTOM_AUTORACE_COURSE_TEXTURE"
CACHE_ENVIRONMENT = "CUSTOM_AUTORACE_COURSE_CACHE"
CACHE_FORMAT_VERSION = b"custom-autorace-course-v2\0"
STOCK_TEXTURE = "from_model"
TUNNEL_LAYOUT_ENVIRONMENT = "CUSTOM_AUTORACE_TUNNEL_LAYOUT"
TUNNEL_LAYOUT_CONFIG_ENVIRONMENT = "CUSTOM_AUTORACE_TUNNEL_LAYOUT_CONFIG"
TUNNEL_MODEL_NAME = "tunnel_obstacle"
TUNNEL_CACHE_FORMAT_VERSION = b"custom-autorace-tunnel-layout-v1\0"
AUTO_TUNNEL_LAYOUT = "auto"
TUNNEL_LAYOUT_SELECTION_VERSION = 1


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_dimensions(path):
    with path.open("rb") as stream:
        header = stream.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError("course texture must be a valid PNG file")
    width, height = struct.unpack(">II", header[16:24])
    if width < 2 or height < 2:
        raise ValueError("course texture PNG dimensions are invalid")
    return width, height


def _cache_key(texture_path, world_path, source_course):
    digest = hashlib.sha256(CACHE_FORMAT_VERSION)
    for path in (
        texture_path,
        world_path,
        source_course / "model.sdf",
        source_course / "model.config",
        source_course / "materials" / "scripts" / "course.material",
    ):
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:20]


def _validated_cache(
    target, target_world, target_model, target_texture, manifest, texture_digest
):
    """Load a complete immutable cache entry or fail closed if it is damaged."""

    if not target.exists():
        return None
    required = (target_world, target_model / "model.sdf", target_texture, manifest)
    if not all(path.is_file() for path in required):
        raise ValueError("staged Gazebo course cache is incomplete: %s" % target)
    try:
        with manifest.open(encoding="utf-8") as stream:
            cached = json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError("staged Gazebo course manifest is invalid: %s" % error)
    model_name = cached.get("model_name", "")
    model_uri = "model://%s" % model_name
    try:
        staged_width, staged_height = _png_dimensions(target_texture)
    except (OSError, ValueError) as error:
        raise ValueError(
            "staged Gazebo course cache failed its integrity check: %s" % error
        )
    if (
        cached.get("cache_format") != 2
        or cached.get("texture_sha256") != texture_digest
        or cached.get("texture_width") != staged_width
        or cached.get("texture_height") != staged_height
        or _file_digest(target_texture) != texture_digest
        or not model_name
        or model_name != target_model.name
        or target_world.read_text(encoding="utf-8").count(model_uri) != 1
        or model_uri not in (target_model / "model.sdf").read_text(encoding="utf-8")
    ):
        raise ValueError("staged Gazebo course cache failed its integrity check")
    return cached


def prepare_course_world(texture_path, world_path, source_course, cache_root):
    """Return an immutable staged world that renders ``texture_path``."""

    texture_path = Path(texture_path).expanduser().resolve(strict=True)
    world_path = Path(world_path).expanduser().resolve(strict=True)
    source_course = Path(source_course).expanduser().resolve(strict=True)
    cache_root = Path(cache_root).expanduser().resolve()
    if not texture_path.is_file():
        raise ValueError("course texture is not a regular file: %s" % texture_path)
    if not world_path.is_file():
        raise ValueError("Gazebo world is not a regular file: %s" % world_path)
    if not (source_course / "model.sdf").is_file():
        raise ValueError("source course model is incomplete: %s" % source_course)
    texture_width, texture_height = _png_dimensions(texture_path)

    key = _cache_key(texture_path, world_path, source_course)
    model_name = "custom_autorace_course_%s" % key
    target = cache_root / key
    target_world = target / "autorace.world"
    target_model = target / "models" / model_name
    target_texture = target_model / "materials" / "textures" / "course.png"
    manifest = target / "manifest.json"
    texture_digest = _file_digest(texture_path)

    cached = _validated_cache(
        target,
        target_world,
        target_model,
        target_texture,
        manifest,
        texture_digest,
    )
    if cached is not None:
        return target_world, target / "models", cached

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".%s." % key, dir=str(cache_root)))
    try:
        staged_model = temporary / "models" / model_name
        shutil.copytree(str(source_course), str(staged_model))
        shutil.copy2(
            str(texture_path),
            str(staged_model / "materials" / "textures" / "course.png"),
        )

        sdf_path = staged_model / "model.sdf"
        sdf = sdf_path.read_text(encoding="utf-8")
        occurrences = sdf.count(SOURCE_MODEL_URI)
        if occurrences < 1:
            raise ValueError("source course model has no expected material URI")
        sdf_path.write_text(
            sdf.replace(SOURCE_MODEL_URI, "model://%s" % model_name),
            encoding="utf-8",
        )

        world = world_path.read_text(encoding="utf-8")
        if world.count(SOURCE_MODEL_URI) != 1:
            raise ValueError("AutoRace world must contain exactly one course URI")
        (temporary / "autorace.world").write_text(
            world.replace(SOURCE_MODEL_URI, "model://%s" % model_name),
            encoding="utf-8",
        )

        metadata = {
            "cache_format": 2,
            "model_name": model_name,
            "source_texture": str(texture_path),
            "texture_sha256": texture_digest,
            "texture_width": texture_width,
            "texture_height": texture_height,
            "source_world": str(world_path),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            temporary.rename(target)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            shutil.rmtree(str(temporary))
            metadata = _validated_cache(
                target,
                target_world,
                target_model,
                target_texture,
                manifest,
                texture_digest,
            )
        return target_world, target / "models", metadata
    except Exception:
        if temporary.exists():
            shutil.rmtree(str(temporary))
        raise


def _pose_values(element, label):
    if element is None or not (element.text or "").strip():
        raise ValueError("%s has no pose" % label)
    try:
        values = [float(value) for value in element.text.split()]
    except ValueError as error:
        raise ValueError("%s pose is not numeric" % label) from error
    if len(values) != 6 or not all(math.isfinite(value) for value in values):
        raise ValueError("%s pose must contain six finite values" % label)
    return values


def _named_child(parent, tag, name, label):
    matches = [
        child
        for child in parent.findall(tag)
        if child.attrib.get("name") == name
    ]
    if len(matches) != 1:
        raise ValueError(
            "%s must contain exactly one %s named %s" % (label, tag, name)
        )
    return matches[0]


def load_tunnel_layout(layout_config, layout_name):
    """Return one finite rotation angle from the deterministic layout file."""

    layout_config = Path(layout_config).expanduser().resolve(strict=True)
    if not layout_config.is_file():
        raise ValueError(
            "tunnel layout config is not a regular file: %s" % layout_config
        )
    with layout_config.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    layouts = document.get("layouts") if isinstance(document, dict) else None
    if not isinstance(layouts, dict) or not layouts:
        raise ValueError("tunnel layout config needs a non-empty layouts map")
    layout = layouts.get(layout_name)
    if not isinstance(layout, dict):
        raise ValueError(
            "unknown tunnel obstacle layout %r; choose one of %s"
            % (layout_name, ", ".join(sorted(str(name) for name in layouts)))
        )
    try:
        rotation_deg = float(layout["rotation_deg"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "tunnel layout %s needs a numeric rotation_deg" % layout_name
        ) from error
    if not math.isfinite(rotation_deg):
        raise ValueError(
            "tunnel layout %s rotation_deg must be finite" % layout_name
        )
    return rotation_deg


def select_tunnel_layout(layout_config, requested_layout, cache_root):
    """Resolve ``auto`` for one gzserver and rotate layouts every launch."""

    if requested_layout != AUTO_TUNNEL_LAYOUT:
        load_tunnel_layout(layout_config, requested_layout)
        return requested_layout

    layout_config = Path(layout_config).expanduser().resolve(strict=True)
    with layout_config.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    layouts = document.get("layouts") if isinstance(document, dict) else None
    if not isinstance(layouts, dict) or not layouts:
        raise ValueError("tunnel layout config needs a non-empty layouts map")
    layout_names = sorted(str(name) for name in layouts)
    for name in layout_names:
        load_tunnel_layout(layout_config, name)

    selection_root = Path(cache_root).expanduser().resolve()
    selection_root.mkdir(parents=True, exist_ok=True)
    state_path = selection_root / "tunnel_layout_selection.json"
    lock_path = selection_root / "tunnel_layout_selection.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        state = {}
        if state_path.is_file():
            try:
                with state_path.open(encoding="utf-8") as state_stream:
                    loaded = json.load(state_stream)
                if isinstance(loaded, dict):
                    state = loaded
            except (OSError, ValueError):
                state = {}
        previous_selection = (
            state.get("selected_layout")
            if state.get("version") == TUNNEL_LAYOUT_SELECTION_VERSION
            else None
        )
        # Treat layout_a as the pre-auto default. A fresh cache therefore
        # selects layout_b first, then every new roslaunch advances B/C/A.
        if previous_selection not in layout_names:
            previous_selection = layout_names[0]
        selected_index = (
            layout_names.index(previous_selection) + 1
        ) % len(layout_names)
        selected_layout = layout_names[selected_index]
        selection = {
            "version": TUNNEL_LAYOUT_SELECTION_VERSION,
            "selected_layout": selected_layout,
        }
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".tunnel_layout_selection.",
            suffix=".json",
            dir=str(selection_root),
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(selection, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(str(temporary_path), str(state_path))
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return selected_layout


def _apply_tunnel_layout(tree, rotation_deg):
    """Rotate the three cylinder links and matching saved Gazebo state."""

    root = tree.getroot()
    world = root.find("./world")
    if root.tag != "sdf" or world is None:
        raise ValueError("AutoRace source must contain one SDF world")
    model = _named_child(world, "model", TUNNEL_MODEL_NAME, "AutoRace world")
    if (model.findtext("static") or "").strip().lower() not in ("1", "true"):
        raise ValueError("tunnel obstacle model must remain static")
    model_pose = _pose_values(
        model.find("pose"), "%s model" % TUNNEL_MODEL_NAME
    )
    if any(abs(value) > 1e-9 for value in model_pose[3:]):
        raise ValueError("tunnel obstacle model rotation must be zero")

    links = model.findall("link")
    link_names = sorted(link.attrib.get("name", "") for link in links)
    expected_names = ["obstacle_1", "obstacle_2", "obstacle_3"]
    if link_names != expected_names:
        raise ValueError(
            "tunnel obstacle model must contain obstacle_1, obstacle_2, obstacle_3"
        )

    angle = math.radians(rotation_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    centres = {}
    for name in expected_names:
        link = _named_child(model, "link", name, TUNNEL_MODEL_NAME)
        values = _pose_values(link.find("pose"), "%s/%s" % (TUNNEL_MODEL_NAME, name))
        local_x = cosine * values[0] - sine * values[1]
        local_y = sine * values[0] + cosine * values[1]
        values[0] = local_x
        values[1] = local_y
        link.find("pose").text = " ".join("%.9g" % value for value in values)
        centres[name] = [model_pose[0] + local_x, model_pose[1] + local_y]

    state = world.find("state")
    if state is None:
        raise ValueError("AutoRace world has no saved Gazebo state")
    state_model = _named_child(
        state, "model", TUNNEL_MODEL_NAME, "saved Gazebo state"
    )
    state_model_pose = _pose_values(
        state_model.find("pose"), "saved %s model" % TUNNEL_MODEL_NAME
    )
    if any(
        abs(actual - saved) > 1e-6
        for actual, saved in zip(model_pose, state_model_pose)
    ):
        raise ValueError("saved tunnel obstacle model pose disagrees with world")
    for name in expected_names:
        state_link = _named_child(
            state_model, "link", name, "saved %s" % TUNNEL_MODEL_NAME
        )
        values = _pose_values(
            state_link.find("pose"), "saved %s/%s" % (TUNNEL_MODEL_NAME, name)
        )
        values[0], values[1] = centres[name]
        state_link.find("pose").text = " ".join(
            "%.9g" % value for value in values
        )
    return centres


def _tunnel_cache_key(world_path, layout_config, layout_name):
    digest = hashlib.sha256(TUNNEL_CACHE_FORMAT_VERSION)
    digest.update(world_path.read_bytes())
    digest.update(b"\0")
    digest.update(layout_config.read_bytes())
    digest.update(b"\0")
    digest.update(layout_name.encode("utf-8"))
    return digest.hexdigest()[:20]


def _validated_tunnel_cache(
    target, target_world, manifest, source_world_digest, config_digest, layout_name
):
    if not target.exists():
        return None
    if not target_world.is_file() or not manifest.is_file():
        raise ValueError("staged Gazebo tunnel layout cache is incomplete: %s" % target)
    try:
        with manifest.open(encoding="utf-8") as stream:
            cached = json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError(
            "staged Gazebo tunnel layout manifest is invalid: %s" % error
        )
    if (
        cached.get("cache_format") != 1
        or cached.get("layout") != layout_name
        or cached.get("source_world_sha256") != source_world_digest
        or cached.get("layout_config_sha256") != config_digest
        or cached.get("staged_world_sha256") != _file_digest(target_world)
    ):
        raise ValueError("staged Gazebo tunnel layout cache failed its integrity check")
    return cached


def prepare_tunnel_world(layout_name, layout_config, world_path, cache_root):
    """Return an immutable world containing one named cylinder layout."""

    world_path = Path(world_path).expanduser().resolve(strict=True)
    layout_config = Path(layout_config).expanduser().resolve(strict=True)
    cache_root = Path(cache_root).expanduser().resolve()
    if not world_path.is_file():
        raise ValueError("Gazebo world is not a regular file: %s" % world_path)
    rotation_deg = load_tunnel_layout(layout_config, layout_name)
    key = _tunnel_cache_key(world_path, layout_config, layout_name)
    layout_root = cache_root / "tunnel_layouts"
    target = layout_root / key
    target_world = target / "autorace.world"
    manifest = target / "manifest.json"
    source_world_digest = _file_digest(world_path)
    config_digest = _file_digest(layout_config)
    cached = _validated_tunnel_cache(
        target,
        target_world,
        manifest,
        source_world_digest,
        config_digest,
        layout_name,
    )
    if cached is not None:
        return target_world, cached

    layout_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".%s." % key, dir=str(layout_root)))
    try:
        tree = ET.parse(str(world_path))
        centres = _apply_tunnel_layout(tree, rotation_deg)
        temporary_world = temporary / "autorace.world"
        tree.write(str(temporary_world), encoding="utf-8", xml_declaration=True)
        metadata = {
            "cache_format": 1,
            "layout": layout_name,
            "rotation_deg": rotation_deg,
            "centres": centres,
            "source_world": str(world_path),
            "source_world_sha256": source_world_digest,
            "layout_config": str(layout_config),
            "layout_config_sha256": config_digest,
            "staged_world_sha256": _file_digest(temporary_world),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            temporary.rename(target)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            shutil.rmtree(str(temporary))
            metadata = _validated_tunnel_cache(
                target,
                target_world,
                manifest,
                source_world_digest,
                config_digest,
                layout_name,
            )
        return target_world, metadata
    except Exception:
        if temporary.exists():
            shutil.rmtree(str(temporary))
        raise


def _replace_world_argument(arguments, source_world, staged_world):
    result = list(arguments)
    source = str(source_world)
    matches = [index for index, value in enumerate(result) if value == source]
    if len(matches) != 1:
        raise ValueError("gzserver arguments must contain the AutoRace world once")
    result[matches[0]] = str(staged_world)
    return result


def _gazebo_ros_executable(rospack, name):
    share = Path(rospack.get_path("gazebo_ros"))
    executable = share.parent.parent / "lib" / "gazebo_ros" / name
    if not executable.is_file() or not os.access(str(executable), os.X_OK):
        raise ValueError("gazebo_ros %s wrapper is not executable" % name)
    return executable


def main():
    rospack = rospkg.RosPack()
    turtlebot3 = Path(rospack.get_path("turtlebot3_gazebo"))
    source_world = (
        turtlebot3 / "worlds" / "turtlebot3_autorace_2020.world"
    ).resolve(strict=True)
    source_course = (
        turtlebot3 / "models" / "turtlebot3_autorace_2020" / "course"
    ).resolve(strict=True)
    texture_value = os.environ.get(TEXTURE_ENVIRONMENT, "").strip()
    if not texture_value:
        raise ValueError("%s is empty" % TEXTURE_ENVIRONMENT)
    layout_value = os.environ.get(TUNNEL_LAYOUT_ENVIRONMENT, "").strip()
    if not layout_value:
        raise ValueError("%s is empty" % TUNNEL_LAYOUT_ENVIRONMENT)
    layout_config_value = os.environ.get(
        TUNNEL_LAYOUT_CONFIG_ENVIRONMENT, ""
    ).strip()
    if not layout_config_value:
        raise ValueError("%s is empty" % TUNNEL_LAYOUT_CONFIG_ENVIRONMENT)
    cache_root = os.environ.get(
        CACHE_ENVIRONMENT, "/tmp/custom_autorace_gazebo_course"
    )
    arguments = list(sys.argv[1:])
    client = "--client" in arguments
    if client:
        arguments.remove("--client")
    if client and "--prepare-only" in arguments:
        raise ValueError("--client and --prepare-only cannot be combined")

    staged_world = source_world
    model_path = None
    texture_metadata = None
    if texture_value != STOCK_TEXTURE:
        staged_world, model_path, texture_metadata = prepare_course_world(
            texture_value, source_world, source_course, cache_root
        )
    selected_layout = None
    layout_metadata = None
    if not client:
        selected_layout = select_tunnel_layout(
            layout_config_value,
            layout_value,
            cache_root,
        )
        staged_world, layout_metadata = prepare_tunnel_world(
            selected_layout,
            layout_config_value,
            staged_world,
            cache_root,
        )

    if "--prepare-only" in arguments:
        prepared = {
            "world": str(staged_world),
            "model_path": "" if model_path is None else str(model_path),
            "requested_layout": layout_value,
            **layout_metadata,
        }
        if texture_metadata is not None:
            prepared["course_texture"] = texture_metadata
        print(json.dumps(prepared, sort_keys=True))
        return

    if not client:
        arguments = _replace_world_argument(arguments, source_world, staged_world)
    environment = os.environ.copy()
    if model_path is not None:
        previous_model_path = environment.get("GAZEBO_MODEL_PATH", "")
        environment["GAZEBO_MODEL_PATH"] = str(model_path)
        if previous_model_path:
            environment["GAZEBO_MODEL_PATH"] += os.pathsep + previous_model_path
    if texture_metadata is not None:
        print(
            "Gazebo course texture %s staged as %s (sha256=%s)"
            % (
                texture_value,
                texture_metadata["model_name"],
                texture_metadata["texture_sha256"],
            ),
            flush=True,
        )
    if not client:
        print(
            "Gazebo tunnel obstacle layout %s selected as %s at %.1f degrees"
            % (layout_value, selected_layout, layout_metadata["rotation_deg"]),
            flush=True,
        )
    executable = _gazebo_ros_executable(
        rospack, "gzclient" if client else "gzserver"
    )
    os.execve(
        str(executable),
        [str(executable)] + arguments,
        environment,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Could not prepare Gazebo world: %s" % error, file=sys.stderr)
        sys.exit(2)
