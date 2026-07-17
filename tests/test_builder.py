from __future__ import annotations

import pytest

from environment_generation.builder import BuilderOperationError, EnvSpec3DBuilder
from environment_generation.operations import execute_operation


def test_builder_recipe_finalizes_box_goal_scene() -> None:
    builder = EnvSpec3DBuilder("scene_1", description="box puzzle")

    builder.make_box_goal_scene()
    spec = builder.finalize()

    assert spec.id == "scene_1"
    assert {obj.semantic_type for obj in spec.objects} >= {"ground", "agent", "pushable_box", "wall", "goal"}
    box = next(obj for obj in spec.objects if obj.semantic_type == "pushable_box")
    assert box.position[2] == pytest.approx(0.55)


def test_finalize_requires_at_least_one_object() -> None:
    builder = EnvSpec3DBuilder("incomplete")

    report = builder.validate_draft()

    assert not report["valid"]
    assert report["issues"] == ["add at least one scene object"]


@pytest.mark.parametrize("kind", ["static", "dynamic", "agentless", "goalless", "groundless"])
def test_finalize_supports_optional_scene_roles(kind: str) -> None:
    builder = EnvSpec3DBuilder(f"optional_{kind}")
    if kind == "static":
        builder.add_wall(0, 0)
    elif kind == "dynamic":
        builder.add_ball(0, 0, z=2)
    elif kind == "agentless":
        builder.add_ground_plane()
    elif kind == "goalless":
        builder.add_ground_plane()
        builder.add_agent_spawn(0, 0)
    else:
        builder.add_agent_spawn(0, 0, z=2)

    spec = builder.finalize()

    assert spec.objects


def test_v11_scene_without_agent_and_goal_does_not_require_game_contract() -> None:
    builder = EnvSpec3DBuilder("v11_static")
    builder.schema_version = "1.1"
    builder.add_wall(0, 0)

    spec = builder.finalize()

    assert spec.schema_version == "1.1"
    assert spec.game is None


def test_requested_ids_are_unique() -> None:
    builder = EnvSpec3DBuilder("ids")
    builder.add_ground_plane(id="ground")

    with pytest.raises(BuilderOperationError):
        builder.add_wall(0, 0, id="ground")


def test_execute_operation_reports_validation_errors() -> None:
    builder = EnvSpec3DBuilder("ops")

    result = execute_operation(
        builder,
        {"op": "add_wall", "args": {"x": 0, "y": 0, "height": -1}},
    )

    assert result["success"] is False
    assert "height" in result["error"]
