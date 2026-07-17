from __future__ import annotations

import math

import pytest

from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.env_verification import scene_object_at, scene_objects
from environment_generation.scene_geometry import (
    closest_footprint_point,
    footprint_contains,
    footprint_signed_clearance,
    footprints_overlap,
    nearest_contact_center,
    nearest_interior_center,
    segment_intersects_footprint,
    volumes_overlap,
)


def _rotated_scene():
    builder = EnvSpec3DBuilder("shape_queries", description="generic shape queries")
    builder.add_ground_plane(12, 12)
    builder.add_static_box(
        0,
        0,
        width=8,
        depth=0.4,
        height=1,
        yaw=math.pi / 4,
        id="elongated_target",
    )
    builder.add_pushable_box(
        2.5,
        -2.5,
        width=0.6,
        depth=0.6,
        height=0.6,
        id="movable_subject",
    )
    objects = {obj.id: obj for obj in scene_objects(builder.finalize())}
    return objects["movable_subject"], objects["elongated_target"]


def test_oriented_footprints_reject_yaw_inflated_aabb_overlap() -> None:
    subject, target = _rotated_scene()

    aabbs_overlap = (
        subject.bounds["x2"] >= target.bounds["x1"]
        and subject.bounds["x1"] <= target.bounds["x2"]
        and subject.bounds["y2"] >= target.bounds["y1"]
        and subject.bounds["y1"] <= target.bounds["y2"]
    )

    assert aabbs_overlap is True
    assert footprints_overlap(subject, target) is False
    assert volumes_overlap(subject, target) is False
    assert footprint_signed_clearance(subject.position, target) > 2


def test_nearest_contact_pose_uses_target_surface_and_subject_extent() -> None:
    subject, target = _rotated_scene()

    placement = nearest_contact_center(subject, target)
    translated = scene_object_at(
        subject,
        (placement.center[0], placement.center[1], subject.position[2]),
    )

    assert placement.target_point != pytest.approx(target.position[:2])
    assert placement.distance < math.dist(subject.position[:2], target.position[:2])
    assert footprints_overlap(translated, target) is True


def test_round_and_oriented_shapes_share_the_same_contact_queries() -> None:
    builder = EnvSpec3DBuilder("round_queries", description="mixed shape queries")
    builder.add_ground_plane(12, 8)
    builder.add_ball(5, 0, radius=0.5, id="round_subject")
    builder.add_cylinder(0, 0, radius=2, height=1, id="round_target")
    objects = {obj.id: obj for obj in scene_objects(builder.finalize())}
    subject = objects["round_subject"]
    target = objects["round_target"]

    placement = nearest_contact_center(subject, target)
    translated = scene_object_at(
        subject,
        (placement.center[0], placement.center[1], subject.position[2]),
    )

    assert placement.center == pytest.approx((2.5, 0.0))
    assert closest_footprint_point(target, subject.position).signed_distance == pytest.approx(3)
    assert footprints_overlap(translated, target) is True


def test_nearest_interior_pose_respects_rotated_target_footprint() -> None:
    builder = EnvSpec3DBuilder("interior_queries", description="interior shape queries")
    builder.add_ground_plane(12, 8)
    builder.add_pushable_box(4, -3, width=0.8, depth=0.8, id="subject")
    builder.add_static_box(
        0,
        0,
        width=5,
        depth=3,
        height=0.2,
        yaw=0.6,
        id="target",
    )
    objects = {obj.id: obj for obj in scene_objects(builder.finalize())}
    subject = objects["subject"]
    target = objects["target"]

    placement = nearest_interior_center(subject, target)
    translated = scene_object_at(
        subject,
        (placement.center[0], placement.center[1], subject.position[2]),
    )

    assert footprint_contains(target, translated) is True
    assert placement.distance > 0


def test_segment_queries_use_oriented_footprints() -> None:
    _subject, target = _rotated_scene()

    assert segment_intersects_footprint((2.2, -2.2), (2.8, -2.8), target) is False
    assert segment_intersects_footprint((-2, -2), (2, 2), target) is True
