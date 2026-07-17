"""Shape-aware geometric queries shared by verification and policy guidance.

The functions here describe geometry only. They do not know about semantic object
types, trial names, or task intent; callers choose a query based on the typed
predicate they are evaluating or guiding.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol, Sequence


_EPSILON = 1e-9
_ROUND_SHAPES = {"sphere", "cylinder", "capsule"}


class SceneShape(Protocol):
    shape: str
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float
    bounds: dict[str, float]


@dataclass(frozen=True)
class FootprintPoint:
    point: tuple[float, float]
    outward_normal: tuple[float, float]
    signed_distance: float
    inside: bool


@dataclass(frozen=True)
class Placement2D:
    center: tuple[float, float]
    target_point: tuple[float, float]
    outward_normal: tuple[float, float]
    distance: float


def closest_footprint_point(
    obj: SceneShape,
    point: tuple[float, float] | Sequence[float],
) -> FootprintPoint:
    """Return the closest boundary point and signed distance to a footprint."""

    query = (float(point[0]), float(point[1]))
    if _is_round(obj):
        center = _center_xy(obj)
        delta = (query[0] - center[0], query[1] - center[1])
        distance = math.hypot(*delta)
        normal = _normalized(delta, fallback=(1.0, 0.0))
        radius = _radius(obj)
        return FootprintPoint(
            point=(center[0] + normal[0] * radius, center[1] + normal[1] * radius),
            outward_normal=normal,
            signed_distance=distance - radius,
            inside=distance <= radius + _EPSILON,
        )

    local = _world_to_local(obj, query)
    half_x, half_y = _half_extents(obj)
    clamped = (
        min(half_x, max(-half_x, local[0])),
        min(half_y, max(-half_y, local[1])),
    )
    outside_delta = (local[0] - clamped[0], local[1] - clamped[1])
    outside_distance = math.hypot(*outside_delta)
    if outside_distance > _EPSILON:
        normal_local = _normalized(outside_delta)
        boundary_local = clamped
        signed_distance = outside_distance
        inside = False
    else:
        gap_x = half_x - abs(local[0])
        gap_y = half_y - abs(local[1])
        if gap_x <= gap_y:
            sign = 1.0 if local[0] >= 0.0 else -1.0
            boundary_local = (sign * half_x, local[1])
            normal_local = (sign, 0.0)
            signed_distance = -gap_x
        else:
            sign = 1.0 if local[1] >= 0.0 else -1.0
            boundary_local = (local[0], sign * half_y)
            normal_local = (0.0, sign)
            signed_distance = -gap_y
        inside = True
    return FootprintPoint(
        point=_local_to_world(obj, boundary_local),
        outward_normal=_rotate(normal_local, float(obj.yaw)),
        signed_distance=signed_distance,
        inside=inside,
    )


def footprint_signed_clearance(
    point: tuple[float, float] | Sequence[float],
    obj: SceneShape,
    *,
    radius: float = 0.0,
) -> float:
    """Signed XY clearance from a circular query to a shape footprint."""

    return closest_footprint_point(obj, point).signed_distance - float(radius)


def footprints_overlap(left: SceneShape, right: SceneShape, *, margin: float = 0.0) -> bool:
    """Return whether two exact XY footprints overlap or touch."""

    tolerance = float(margin)
    left_round = _is_round(left)
    right_round = _is_round(right)
    if left_round and right_round:
        return math.dist(_center_xy(left), _center_xy(right)) <= (
            _radius(left) + _radius(right) + tolerance
        )
    if left_round:
        return closest_footprint_point(right, _center_xy(left)).signed_distance <= (
            _radius(left) + tolerance
        )
    if right_round:
        return closest_footprint_point(left, _center_xy(right)).signed_distance <= (
            _radius(right) + tolerance
        )

    center_delta = (
        float(right.position[0]) - float(left.position[0]),
        float(right.position[1]) - float(left.position[1]),
    )
    axes = (*_axes(left), *_axes(right))
    return all(
        abs(_dot(center_delta, axis))
        <= support_radius_xy(left, axis) + support_radius_xy(right, axis) + tolerance
        for axis in axes
    )


def volumes_overlap(left: SceneShape, right: SceneShape, *, margin: float = 0.0) -> bool:
    """Return whether shape footprints and vertical intervals overlap."""

    tolerance = float(margin)
    left_z = _z_interval(left)
    right_z = _z_interval(right)
    vertical = (
        left_z[1] >= right_z[0] - tolerance
        and left_z[0] <= right_z[1] + tolerance
    )
    return vertical and footprints_overlap(left, right, margin=tolerance)


def footprint_contains(
    container: SceneShape,
    subject: SceneShape,
    *,
    margin: float = 0.0,
) -> bool:
    """Return whether the subject's full XY footprint is inside the container."""

    tolerance = float(margin)
    if _is_round(container):
        radius = _radius(container) + tolerance
        center = _center_xy(container)
        if _is_round(subject):
            return math.dist(center, _center_xy(subject)) + _radius(subject) <= radius + _EPSILON
        return all(math.dist(center, point) <= radius + _EPSILON for point in footprint_vertices(subject))

    half_x, half_y = _half_extents(container)
    if _is_round(subject):
        local = _world_to_local(container, _center_xy(subject))
        radius = _radius(subject)
        return (
            abs(local[0]) + radius <= half_x + tolerance + _EPSILON
            and abs(local[1]) + radius <= half_y + tolerance + _EPSILON
        )
    return all(
        abs(local[0]) <= half_x + tolerance + _EPSILON
        and abs(local[1]) <= half_y + tolerance + _EPSILON
        for local in (_world_to_local(container, point) for point in footprint_vertices(subject))
    )


def volume_contains(
    container: SceneShape,
    subject: SceneShape,
    *,
    margin: float = 0.0,
) -> bool:
    tolerance = float(margin)
    container_z = _z_interval(container)
    subject_z = _z_interval(subject)
    return (
        footprint_contains(container, subject, margin=tolerance)
        and subject_z[0] >= container_z[0] - tolerance
        and subject_z[1] <= container_z[1] + tolerance
    )


def support_radius_xy(obj: SceneShape, direction: tuple[float, float]) -> float:
    """Projection radius of a footprint along a world-space unit direction."""

    unit = _normalized(direction, fallback=(1.0, 0.0))
    if _is_round(obj):
        return _radius(obj)
    local_direction = _rotate(unit, -float(obj.yaw))
    half_x, half_y = _half_extents(obj)
    return abs(local_direction[0]) * half_x + abs(local_direction[1]) * half_y


def nearest_contact_center(subject: SceneShape, target: SceneShape) -> Placement2D:
    """Find the nearest subject-center pose whose footprint touches the target."""

    query = closest_footprint_point(target, _center_xy(subject))
    extent = support_radius_xy(subject, query.outward_normal)
    center = (
        query.point[0] + query.outward_normal[0] * extent,
        query.point[1] + query.outward_normal[1] * extent,
    )
    return Placement2D(
        center=center,
        target_point=query.point,
        outward_normal=query.outward_normal,
        distance=math.dist(_center_xy(subject), center),
    )


def contact_approach_centers(
    subject: SceneShape,
    target: SceneShape,
    *,
    clearance: float = 0.0,
) -> list[tuple[float, float]]:
    """Return reusable subject-center approach poses around a target footprint."""

    expansion = max(0.0, float(clearance))
    candidates = [nearest_contact_center(subject, target).center]
    if _is_round(target):
        normals = [
            (math.cos(index * math.pi / 4.0), math.sin(index * math.pi / 4.0))
            for index in range(8)
        ]
        target_center = _center_xy(target)
        for normal in normals:
            distance = _radius(target) + support_radius_xy(subject, normal) + expansion
            candidates.append(
                (
                    target_center[0] + normal[0] * distance,
                    target_center[1] + normal[1] * distance,
                )
            )
    else:
        axes = _axes(target)
        half_x, half_y = _half_extents(target)
        for axis, target_extent in (
            (axes[0], half_x),
            ((-axes[0][0], -axes[0][1]), half_x),
            (axes[1], half_y),
            ((-axes[1][0], -axes[1][1]), half_y),
        ):
            distance = target_extent + support_radius_xy(subject, axis) + expansion
            candidates.append(
                (
                    float(target.position[0]) + axis[0] * distance,
                    float(target.position[1]) + axis[1] * distance,
                )
            )
    unique: list[tuple[float, float]] = []
    for candidate in candidates:
        if not any(math.dist(candidate, existing) <= 1e-6 for existing in unique):
            unique.append(candidate)
    return unique


def nearest_interior_center(subject: SceneShape, target: SceneShape) -> Placement2D:
    """Find a nearby subject-center pose fully contained by the target footprint."""

    subject_center = _center_xy(subject)
    if _is_round(target):
        target_center = _center_xy(target)
        conservative_radius = (
            _radius(subject)
            if _is_round(subject)
            else math.hypot(float(subject.size[0]), float(subject.size[1])) * 0.5
        )
        available = max(0.0, _radius(target) - conservative_radius)
        delta = (subject_center[0] - target_center[0], subject_center[1] - target_center[1])
        distance = math.hypot(*delta)
        unit = _normalized(delta, fallback=(1.0, 0.0))
        radial = min(distance, available)
        center = (target_center[0] + unit[0] * radial, target_center[1] + unit[1] * radial)
    else:
        target_axes = _axes(target)
        half_x, half_y = _half_extents(target)
        inset_x = support_radius_xy(subject, target_axes[0])
        inset_y = support_radius_xy(subject, target_axes[1])
        available_x = max(0.0, half_x - inset_x)
        available_y = max(0.0, half_y - inset_y)
        local = _world_to_local(target, subject_center)
        center = _local_to_world(
            target,
            (
                min(available_x, max(-available_x, local[0])),
                min(available_y, max(-available_y, local[1])),
            ),
        )
    delta = (subject_center[0] - center[0], subject_center[1] - center[1])
    normal = _normalized(delta, fallback=(1.0, 0.0))
    return Placement2D(
        center=center,
        target_point=center,
        outward_normal=normal,
        distance=math.dist(subject_center, center),
    )


def footprint_vertices(obj: SceneShape) -> list[tuple[float, float]]:
    """Return exact rectangle corners; round shapes return cardinal boundary points."""

    if _is_round(obj):
        center = _center_xy(obj)
        radius = _radius(obj)
        return [
            (center[0] + radius, center[1]),
            (center[0], center[1] + radius),
            (center[0] - radius, center[1]),
            (center[0], center[1] - radius),
        ]
    half_x, half_y = _half_extents(obj)
    return [
        _local_to_world(obj, (x, y))
        for x, y in (
            (-half_x, -half_y),
            (-half_x, half_y),
            (half_x, half_y),
            (half_x, -half_y),
        )
    ]


def segment_intersects_footprint(
    start: tuple[float, float],
    end: tuple[float, float],
    obj: SceneShape,
    *,
    clearance: float = 0.0,
) -> bool:
    """Test a line segment against an optionally expanded footprint."""

    expansion = max(0.0, float(clearance))
    if _is_round(obj):
        return _point_segment_distance(_center_xy(obj), start, end) <= _radius(obj) + expansion
    local_start = _world_to_local(obj, start)
    local_end = _world_to_local(obj, end)
    half_x, half_y = _half_extents(obj)
    return _segment_intersects_aabb(
        local_start,
        local_end,
        (-half_x - expansion, half_x + expansion, -half_y - expansion, half_y + expansion),
    )


def yaw_from_rotation_matrix(matrix: Any, *, fallback: float = 0.0) -> float:
    try:
        if len(matrix) < 9:
            return float(fallback)
    except (TypeError, AttributeError):
        return float(fallback)
    return math.atan2(float(matrix[3]), float(matrix[0]))


def _is_round(obj: SceneShape) -> bool:
    return str(obj.shape).lower() in _ROUND_SHAPES


def _center_xy(obj: SceneShape) -> tuple[float, float]:
    return float(obj.position[0]), float(obj.position[1])


def _radius(obj: SceneShape) -> float:
    return float(obj.size[0]) * 0.5


def _half_extents(obj: SceneShape) -> tuple[float, float]:
    return float(obj.size[0]) * 0.5, float(obj.size[1]) * 0.5


def _axes(obj: SceneShape) -> tuple[tuple[float, float], tuple[float, float]]:
    yaw = float(obj.yaw)
    return (math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw))


def _world_to_local(obj: SceneShape, point: tuple[float, float]) -> tuple[float, float]:
    delta = (point[0] - float(obj.position[0]), point[1] - float(obj.position[1]))
    return _rotate(delta, -float(obj.yaw))


def _local_to_world(obj: SceneShape, point: tuple[float, float]) -> tuple[float, float]:
    rotated = _rotate(point, float(obj.yaw))
    return rotated[0] + float(obj.position[0]), rotated[1] + float(obj.position[1])


def _rotate(point: tuple[float, float], angle: float) -> tuple[float, float]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return point[0] * cosine - point[1] * sine, point[0] * sine + point[1] * cosine


def _normalized(
    vector: tuple[float, float],
    *,
    fallback: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    length = math.hypot(*vector)
    if length <= _EPSILON:
        return fallback
    return vector[0] / length, vector[1] / length


def _dot(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _z_interval(obj: SceneShape) -> tuple[float, float]:
    bounds = getattr(obj, "bounds", None)
    if isinstance(bounds, dict) and "z1" in bounds and "z2" in bounds:
        return float(bounds["z1"]), float(bounds["z2"])
    half = float(obj.size[2]) * 0.5
    return float(obj.position[2]) - half, float(obj.position[2]) + half


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    delta = (end[0] - start[0], end[1] - start[1])
    length_sq = _dot(delta, delta)
    if length_sq <= _EPSILON:
        return math.dist(point, start)
    parameter = max(0.0, min(1.0, _dot((point[0] - start[0], point[1] - start[1]), delta) / length_sq))
    nearest = (start[0] + delta[0] * parameter, start[1] + delta[1] * parameter)
    return math.dist(point, nearest)


def _segment_intersects_aabb(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> bool:
    delta = (end[0] - start[0], end[1] - start[1])
    minimum_t = 0.0
    maximum_t = 1.0
    for direction, offset in (
        (-delta[0], start[0] - bounds[0]),
        (delta[0], bounds[1] - start[0]),
        (-delta[1], start[1] - bounds[2]),
        (delta[1], bounds[3] - start[1]),
    ):
        if abs(direction) <= _EPSILON:
            if offset < 0.0:
                return False
            continue
        ratio = offset / direction
        if direction < 0.0:
            minimum_t = max(minimum_t, ratio)
        else:
            maximum_t = min(maximum_t, ratio)
        if minimum_t > maximum_t:
            return False
    return True
