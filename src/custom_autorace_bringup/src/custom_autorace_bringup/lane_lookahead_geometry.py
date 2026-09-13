"""Resolve the adjustable geometry for the isolated lane-lookahead trial.

The first measured straight has one fixed start anchor.  Changing its length
rescales only the knots that belong to that straight and rigidly translates all
later lookahead knots.  The production lane controller and mission geometry are
intentionally outside this module.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
import os

from custom_autorace_bringup.zigzag_path import build_zigzag_path


FROM_YAML = "from_yaml"


@dataclass(frozen=True)
class LaneLookaheadGeometry:
    """Fully resolved path and gates for one numeric straight length."""

    config: dict
    path: object
    reference_path_length: float
    reference_straight_length: float
    straight_length: float
    length_delta: float
    downstream_translation: tuple
    straight_start: tuple
    reference_straight_end: tuple
    straight_end: tuple


def _finite_float(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a finite number" % name)
    if not math.isfinite(result):
        raise ValueError("%s must be a finite number" % name)
    return result


def _point(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("%s must be [x, y]" % name)
    return (
        _finite_float(value[0], name + "[0]"),
        _finite_float(value[1], name + "[1]"),
    )


def _selected_length(straight):
    configured = _finite_float(straight.get("length"), "geometry/straight/length")
    override = straight.get("length_override", FROM_YAML)
    if isinstance(override, str) and override.strip().lower() == FROM_YAML:
        return configured
    return _finite_float(override, "geometry/straight/length_override")


def _build_path(config, knots):
    path = config.get("path", {})
    control = config.get("control", {})
    return build_zigzag_path(
        knots=knots,
        start_heading=math.radians(
            _finite_float(path.get("start_heading_deg"), "path/start_heading_deg")
        ),
        end_heading=math.radians(
            _finite_float(path.get("end_heading_deg"), "path/end_heading_deg")
        ),
        sample_spacing=_finite_float(path.get("sample_spacing"), "path/sample_spacing"),
        cruise_velocity=_finite_float(
            control.get("cruise_velocity"), "control/cruise_velocity"
        ),
        minimum_velocity=_finite_float(
            control.get("minimum_velocity"), "control/minimum_velocity"
        ),
        entry_velocity=_finite_float(
            control.get("entry_velocity"), "control/entry_velocity"
        ),
        exit_velocity=_finite_float(
            control.get("exit_velocity"), "control/exit_velocity"
        ),
        maximum_angular_velocity=_finite_float(
            control.get("maximum_angular_velocity"),
            "control/maximum_angular_velocity",
        ),
        maximum_lateral_acceleration=_finite_float(
            control.get("maximum_lateral_acceleration"),
            "control/maximum_lateral_acceleration",
        ),
        linear_acceleration=_finite_float(
            control.get("linear_acceleration"), "control/linear_acceleration"
        ),
        linear_deceleration=_finite_float(
            control.get("linear_deceleration"), "control/linear_deceleration"
        ),
        guide_tail_length=_finite_float(
            path.get("guide_tail_length", 0.0), "path/guide_tail_length"
        ),
        maximum_angular_acceleration=_finite_float(
            control.get("angular_acceleration"), "control/angular_acceleration"
        ),
    )


def resolve_lane_lookahead_geometry(config):
    """Return a copied config, rebuilt path and automatically shifted finish.

    The reference geometry remains untouched.  ``length_override`` wins over
    the editable YAML ``length`` only when it is not the literal ``from_yaml``.
    """

    if not isinstance(config, dict):
        raise ValueError("lane_lookahead config must be a YAML mapping")
    resolved = deepcopy(config)
    geometry = resolved.get("geometry", {})
    if not isinstance(geometry, dict):
        raise ValueError("geometry must be a YAML mapping")
    straight = geometry.get("straight", {})
    if not isinstance(straight, dict):
        raise ValueError("geometry/straight must be a YAML mapping")

    anchor = _point(straight.get("start"), "geometry/straight/start")
    heading = math.radians(
        _finite_float(straight.get("heading_deg"), "geometry/straight/heading_deg")
    )
    tangent = (math.cos(heading), math.sin(heading))
    normal = (-tangent[1], tangent[0])
    reference_length = _finite_float(
        straight.get("reference_length"), "geometry/straight/reference_length"
    )
    selected_length = _selected_length(straight)
    minimum_length = _finite_float(
        straight.get("minimum_length", 0.05), "geometry/straight/minimum_length"
    )
    maximum_length = _finite_float(
        straight.get("maximum_length", 5.0), "geometry/straight/maximum_length"
    )
    if reference_length <= 0.0:
        raise ValueError("geometry/straight/reference_length must be positive")
    if minimum_length <= 0.0 or maximum_length < minimum_length:
        raise ValueError("geometry/straight length limits are invalid")
    if not minimum_length <= selected_length <= maximum_length:
        raise ValueError(
            "geometry/straight/length %.6f is outside [%.6f, %.6f]"
            % (selected_length, minimum_length, maximum_length)
        )

    path_config = resolved.get("path", {})
    if not isinstance(path_config, dict):
        raise ValueError("path must be a YAML mapping")
    raw_knots = path_config.get("knots", [])
    if not isinstance(raw_knots, list):
        raise ValueError("path/knots must be a YAML list")
    knots = [_point(knot, "path/knots[%d]" % index) for index, knot in enumerate(raw_knots)]
    try:
        start_index = int(straight.get("path_start_knot", 0))
        end_index = int(straight.get("path_end_knot"))
    except (TypeError, ValueError):
        raise ValueError("straight knot indices must be integers")
    if not (0 <= start_index < end_index < len(knots)):
        raise ValueError("straight knot indices are outside path/knots")

    tolerance = _finite_float(
        straight.get("anchor_tolerance", 0.002),
        "geometry/straight/anchor_tolerance",
    )
    if tolerance < 0.0:
        raise ValueError("geometry/straight/anchor_tolerance cannot be negative")

    def local_coordinates(point):
        dx = point[0] - anchor[0]
        dy = point[1] - anchor[1]
        return (
            tangent[0] * dx + tangent[1] * dy,
            normal[0] * dx + normal[1] * dy,
        )

    start_longitudinal, start_lateral = local_coordinates(knots[start_index])
    end_longitudinal, end_lateral = local_coordinates(knots[end_index])
    if math.hypot(start_longitudinal, start_lateral) > tolerance:
        raise ValueError("straight start knot does not match its fixed start anchor")
    if abs(end_longitudinal - reference_length) > tolerance:
        raise ValueError(
            "straight end knot does not match reference_length along heading"
        )

    previous_longitudinal = -math.inf
    scale = selected_length / reference_length
    adjusted_knots = []
    for index, point in enumerate(knots):
        if start_index <= index <= end_index:
            longitudinal, lateral = local_coordinates(point)
            if longitudinal + tolerance < previous_longitudinal:
                raise ValueError("straight knots must progress monotonically")
            if longitudinal < -tolerance or longitudinal > reference_length + tolerance:
                raise ValueError("a straight knot lies outside the reference straight")
            previous_longitudinal = longitudinal
            adjusted_longitudinal = longitudinal * scale
            adjusted_knots.append(
                (
                    anchor[0]
                    + tangent[0] * adjusted_longitudinal
                    + normal[0] * lateral,
                    anchor[1]
                    + tangent[1] * adjusted_longitudinal
                    + normal[1] * lateral,
                )
            )
        elif index > end_index:
            delta = selected_length - reference_length
            adjusted_knots.append(
                (point[0] + tangent[0] * delta, point[1] + tangent[1] * delta)
            )
        else:
            adjusted_knots.append(point)

    reference_path = _build_path(resolved, knots)
    adjusted_path = _build_path(resolved, adjusted_knots)
    path_config["knots"] = [list(point) for point in adjusted_knots]
    straight["length"] = selected_length

    delta = selected_length - reference_length
    translation = (tangent[0] * delta, tangent[1] * delta)
    finish = resolved.get("finish", {})
    if not isinstance(finish, dict):
        raise ValueError("finish must be a YAML mapping")
    finish["x"] = _finite_float(finish.get("x"), "finish/x") + translation[0]
    finish["min_y"] = (
        _finite_float(finish.get("min_y"), "finish/min_y") + translation[1]
    )
    finish["max_y"] = (
        _finite_float(finish.get("max_y"), "finish/max_y") + translation[1]
    )
    finish["minimum_station"] = _finite_float(
        finish.get("minimum_station"), "finish/minimum_station"
    ) + (adjusted_path.length - reference_path.length)
    if finish["min_y"] > finish["max_y"]:
        raise ValueError("finish/min_y must not exceed finish/max_y")
    if adjusted_path.length <= finish["minimum_station"]:
        raise ValueError("adjusted lookahead path ends before the finish gate")

    reference_end = (
        anchor[0] + tangent[0] * reference_length + normal[0] * end_lateral,
        anchor[1] + tangent[1] * reference_length + normal[1] * end_lateral,
    )
    selected_end = (
        anchor[0] + tangent[0] * selected_length + normal[0] * end_lateral,
        anchor[1] + tangent[1] * selected_length + normal[1] * end_lateral,
    )
    return LaneLookaheadGeometry(
        config=resolved,
        path=adjusted_path,
        reference_path_length=float(reference_path.length),
        reference_straight_length=reference_length,
        straight_length=selected_length,
        length_delta=delta,
        downstream_translation=translation,
        straight_start=anchor,
        reference_straight_end=reference_end,
        straight_end=selected_end,
    )


def texture_path_from_config(texture, package_resolver):
    """Resolve an optional absolute override or the package-relative texture."""

    override = texture.get("file_override", FROM_YAML)
    if not (isinstance(override, str) and override.strip().lower() == FROM_YAML):
        path = str(override).strip()
        if not os.path.isabs(path):
            raise ValueError("texture/file_override must be an absolute path")
        return path
    direct = str(texture.get("file", "")).strip()
    if direct:
        if not os.path.isabs(direct):
            raise ValueError("texture/file must be an absolute path")
        return direct
    package = str(texture.get("package", "")).strip()
    relative = str(texture.get("relative_path", "")).strip()
    if not package or not relative:
        raise ValueError("texture needs file or package plus relative_path")
    return os.path.join(package_resolver(package), relative)


def validate_texture_revision(texture_path, texture, geometry):
    """Reject a resized route when it still points at the stock reference map."""

    reference_digest = str(texture.get("reference_sha256", "")).strip().lower()
    if not reference_digest:
        if abs(geometry.length_delta) > 1e-9:
            raise ValueError(
                "a non-reference straight length needs texture/reference_sha256"
            )
        return ""
    if len(reference_digest) != 64 or any(
        character not in "0123456789abcdef" for character in reference_digest
    ):
        raise ValueError("texture/reference_sha256 must contain 64 hex characters")
    try:
        digest = hashlib.sha256()
        with open(texture_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_digest = digest.hexdigest()
    except OSError as error:
        raise ValueError("could not read lookahead texture: %s" % error)
    if abs(geometry.length_delta) > 1e-9 and actual_digest == reference_digest:
        raise ValueError(
            "straight length changed but the configured texture is still the "
            "reference course image"
        )
    return actual_digest
