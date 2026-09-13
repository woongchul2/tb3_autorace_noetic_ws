#!/usr/bin/env python3
"""Start Gazebo with an isolated course model using one selected texture.

The upstream AutoRace world addresses its floor through a fixed ``model://``
URI.  This wrapper copies only that small model into an immutable cache,
changes the world to a digest-specific model URI, and then execs the normal
``gazebo_ros/gzserver`` wrapper.  The package's stock model is never modified.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

import rospkg


SOURCE_MODEL_URI = "model://turtlebot3_autorace_2020/course"
TEXTURE_ENVIRONMENT = "CUSTOM_AUTORACE_COURSE_TEXTURE"
CACHE_ENVIRONMENT = "CUSTOM_AUTORACE_COURSE_CACHE"
CACHE_FORMAT_VERSION = b"custom-autorace-course-v2\0"


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
        except FileExistsError:
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
    cache_root = os.environ.get(
        CACHE_ENVIRONMENT, "/tmp/custom_autorace_gazebo_course"
    )
    staged_world, model_path, metadata = prepare_course_world(
        texture_value, source_world, source_course, cache_root
    )

    arguments = list(sys.argv[1:])
    if "--prepare-only" in arguments:
        print(
            json.dumps(
                {
                    "world": str(staged_world),
                    "model_path": str(model_path),
                    **metadata,
                },
                sort_keys=True,
            )
        )
        return

    client = "--client" in arguments
    if client:
        arguments.remove("--client")
    else:
        arguments = _replace_world_argument(arguments, source_world, staged_world)
    environment = os.environ.copy()
    previous_model_path = environment.get("GAZEBO_MODEL_PATH", "")
    environment["GAZEBO_MODEL_PATH"] = str(model_path)
    if previous_model_path:
        environment["GAZEBO_MODEL_PATH"] += os.pathsep + previous_model_path
    print(
        "Gazebo course texture %s staged as %s (sha256=%s)"
        % (texture_value, metadata["model_name"], metadata["texture_sha256"]),
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
        print("Could not prepare Gazebo course texture: %s" % error, file=sys.stderr)
        sys.exit(2)
