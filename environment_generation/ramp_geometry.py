"""Canonical geometry for walkable ramps.

Ramp authoring uses a low walkable endpoint, a horizontal run, and a vertical
rise.  The MuJoCo collider is a tilted box whose local x dimension is the
sloped length.  Keeping this calculation in one place prevents the authored
surface, physics collider, and deterministic checks from drifting apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


RAMP_GEOMETRY_VERSION = 2


@dataclass(frozen=True)
class RampGeometry:
    low_end: tuple[float, float, float]
    high_end: tuple[float, float, float]
    center: tuple[float, float, float]
    length: float
    width: float
    rise: float
    thickness: float
    yaw: float
    slope_length: float
    angle: float

    @property
    def x_axis(self) -> tuple[float, float, float]:
        """World-space uphill axis of the tilted collider."""

        return (
            math.cos(self.yaw) * math.cos(self.angle),
            math.sin(self.yaw) * math.cos(self.angle),
            math.sin(self.angle),
        )

    @property
    def y_axis(self) -> tuple[float, float, float]:
        """World-space cross-slope axis of the tilted collider."""

        return (-math.sin(self.yaw), math.cos(self.yaw), 0.0)

    def metadata(self, existing: Mapping[str, Any] | None = None) -> dict[str, Any]:
        value = dict(existing or {})
        value.pop("height", None)
        value.update(
            {
                "geometry_version": RAMP_GEOMETRY_VERSION,
                "rise": self.rise,
                "low_end": list(self.low_end),
            }
        )
        return value


def make_ramp_geometry(
    *,
    low_end: Sequence[float],
    length: float,
    width: float,
    rise: float,
    thickness: float,
    yaw: float,
) -> RampGeometry:
    low = _finite_vec3(low_end, "low_end")
    run = _positive_finite(length, "length")
    ramp_width = _positive_finite(width, "width")
    ramp_rise = _positive_finite(rise, "rise")
    ramp_thickness = _positive_finite(thickness, "thickness")
    ramp_yaw = _finite(yaw, "yaw")
    angle = math.atan2(ramp_rise, run)
    slope_length = math.hypot(run, ramp_rise)
    direction = (math.cos(ramp_yaw), math.sin(ramp_yaw))

    # low_end/high_end describe the edges of the walkable top face.  A tilted
    # box's thickness shifts that face slightly downhill from its centerline.
    center_along = run / 2.0 + ramp_thickness * math.sin(angle) / 2.0
    center = (
        low[0] + direction[0] * center_along,
        low[1] + direction[1] * center_along,
        low[2] + ramp_rise / 2.0 - ramp_thickness * math.cos(angle) / 2.0,
    )
    high = (
        low[0] + direction[0] * run,
        low[1] + direction[1] * run,
        low[2] + ramp_rise,
    )
    return RampGeometry(
        low_end=low,
        high_end=high,
        center=center,
        length=run,
        width=ramp_width,
        rise=ramp_rise,
        thickness=ramp_thickness,
        yaw=ramp_yaw,
        slope_length=slope_length,
        angle=angle,
    )


def ramp_geometry_from_object(obj: Mapping[str, Any]) -> RampGeometry:
    size = obj.get("size")
    position = obj.get("position")
    if not isinstance(size, Sequence) or len(size) != 3:
        raise ValueError("ramp size must contain three full dimensions")
    if not isinstance(position, Sequence) or len(position) != 3:
        raise ValueError("ramp position must contain three coordinates")
    metadata = obj.get("metadata") if isinstance(obj.get("metadata"), Mapping) else {}
    length = float(size[0])
    width = float(size[1])
    thickness = float(size[2])
    rise = float(metadata.get("rise", metadata.get("height", thickness)))
    yaw = float(obj.get("yaw") or 0.0)
    low_end = metadata.get("low_end")
    if not isinstance(low_end, Sequence) or isinstance(low_end, (str, bytes)) or len(low_end) != 3:
        angle = math.atan2(rise, length)
        center_along = length / 2.0 + thickness * math.sin(angle) / 2.0
        low_end = (
            float(position[0]) - math.cos(yaw) * center_along,
            float(position[1]) - math.sin(yaw) * center_along,
            float(position[2]) - rise / 2.0 + thickness * math.cos(angle) / 2.0,
        )
    return make_ramp_geometry(
        low_end=low_end,
        length=length,
        width=width,
        rise=rise,
        thickness=thickness,
        yaw=yaw,
    )


def ramp_object_fields(geometry: RampGeometry, *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "position": list(geometry.center),
        "size": [geometry.length, geometry.width, geometry.thickness],
        "yaw": geometry.yaw,
        "metadata": geometry.metadata(metadata),
    }


def ramp_bounds(geometry: RampGeometry) -> dict[str, float]:
    """Return the world AABB of the full tilted collider."""

    c_pitch = math.cos(geometry.angle)
    s_pitch = math.sin(geometry.angle)
    c_yaw = math.cos(geometry.yaw)
    s_yaw = math.sin(geometry.yaw)
    points: list[tuple[float, float, float]] = []
    for local_x in (-geometry.slope_length / 2.0, geometry.slope_length / 2.0):
        for local_y in (-geometry.width / 2.0, geometry.width / 2.0):
            for local_z in (-geometry.thickness / 2.0, geometry.thickness / 2.0):
                pitched_x = local_x * c_pitch - local_z * s_pitch
                world_z = geometry.center[2] + local_x * s_pitch + local_z * c_pitch
                world_x = geometry.center[0] + pitched_x * c_yaw - local_y * s_yaw
                world_y = geometry.center[1] + pitched_x * s_yaw + local_y * c_yaw
                points.append((world_x, world_y, world_z))
    return {
        "x1": min(point[0] for point in points),
        "x2": max(point[0] for point in points),
        "y1": min(point[1] for point in points),
        "y2": max(point[1] for point in points),
        "z1": min(point[2] for point in points),
        "z2": max(point[2] for point in points),
    }


def ramp_surface_height(geometry: RampGeometry, x: float, y: float) -> float | None:
    """Return top-surface z when an XY point lies on the authored ramp run."""

    dx = float(x) - geometry.low_end[0]
    dy = float(y) - geometry.low_end[1]
    c_yaw = math.cos(geometry.yaw)
    s_yaw = math.sin(geometry.yaw)
    along = dx * c_yaw + dy * s_yaw
    across = -dx * s_yaw + dy * c_yaw
    tolerance = 1e-8
    if along < -tolerance or along > geometry.length + tolerance:
        return None
    if abs(across) > geometry.width / 2.0 + tolerance:
        return None
    progress = min(1.0, max(0.0, along / geometry.length))
    return geometry.low_end[2] + geometry.rise * progress


def _finite_vec3(value: Sequence[float], name: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must contain three coordinates")
    return (
        _finite(value[0], f"{name}[0]"),
        _finite(value[1], f"{name}[1]"),
        _finite(value[2], f"{name}[2]"),
    )


def _positive_finite(value: float, name: str) -> float:
    number = _finite(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return number


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number
