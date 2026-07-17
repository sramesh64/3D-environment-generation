from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from environment_generation.artifacts import persist_artifacts
from environment_generation.builder import BuilderValidationError, EnvSpec3DBuilder
from environment_generation.courtyard import (
    DIFFICULTIES,
    LEVEL_FAMILIES,
    courtyard_shell_spec,
    is_courtyard_shell,
    validate_courtyard_layout,
)
from environment_generation.env_behavior_trials import fallback_locomotion_plan
from environment_generation.env_verification import spec_hash
from environment_generation.mujoco_compile import compile_spec_to_mjcf
from environment_generation.operations import execute_operation
from environment_generation.player import PlayableSimulation
from environment_generation.play_session import PlaySession
from environment_generation.schema import env_spec_to_dict
from environment_generation.session import SceneSession
from environment_generation.visual_scene import compile_visual_scene


def _level(*, family: str = "barrier_route", difficulty: str = "medium", seed: int = 7):
    builder = EnvSpec3DBuilder(f"courtyard_{family}_{difficulty}_{seed}", description="courtyard")
    builder.make_courtyard_level(family=family, difficulty=difficulty, seed=seed)
    return builder.finalize()


def test_v11_game_contract_and_v10_hash_compatibility() -> None:
    legacy = EnvSpec3DBuilder("legacy", description="legacy")
    legacy.add_wall(0, 0)
    legacy_spec = legacy.finalize()
    raw = legacy_spec.model_dump(mode="json")

    assert legacy_spec.schema_version == "1.0"
    assert "game" not in env_spec_to_dict(legacy_spec)
    assert spec_hash(raw) == spec_hash(env_spec_to_dict(legacy_spec))

    game = _level()
    assert game.schema_version == "1.1"
    assert game.theme == "robot_courtyard"
    assert game.game is not None
    assert game.game.mode == "reach_goal"


def test_generator_is_deterministic_and_seeded() -> None:
    first = env_spec_to_dict(_level(seed=123))
    repeated = env_spec_to_dict(_level(seed=123))
    changed = env_spec_to_dict(_level(seed=124))

    assert first == repeated
    assert first["generation"] == repeated["generation"]
    assert first["objects"] != changed["objects"]


def test_blank_shell_is_valid_and_can_become_a_game_level() -> None:
    shell = courtyard_shell_spec("blank_before", seed=41)

    assert shell.schema_version == "1.1"
    assert shell.game is None
    assert is_courtyard_shell(shell)
    assert {obj.semantic_type for obj in shell.objects} == {"ground", "wall"}
    assert {obj.id for obj in shell.objects} == {
        "ground",
        "north_boundary",
        "south_boundary",
        "west_boundary",
        "east_boundary",
    }
    assert shell.world_size == [30.0, 24.0, 10.0]
    ground = next(obj for obj in shell.objects if obj.semantic_type == "ground")
    boundaries = [obj for obj in shell.objects if obj.semantic_type == "wall"]
    assert ground.size[:2] == [24.0, 18.0]
    assert len(boundaries) == 4
    assert all(obj.appearance and obj.appearance.variant == "plain" for obj in boundaries)
    serialized_shell = env_spec_to_dict(shell)
    assert all(
        obj.get("appearance", {}).get("variant") == "plain"
        for obj in serialized_shell["objects"]
        if obj["semantic_type"] == "wall"
    )

    builder = EnvSpec3DBuilder.from_spec_dict(env_spec_to_dict(shell))
    builder.make_courtyard_level(family="slalom", difficulty="medium", seed=41)
    generated = builder.finalize()
    assert generated.schema_version == "1.1"
    assert generated.game is not None
    assert is_courtyard_shell(generated) is False


def test_new_courtyard_hazards_use_the_warning_design() -> None:
    builder = EnvSpec3DBuilder.from_spec_dict(env_spec_to_dict(courtyard_shell_spec("hazard_style")))
    hazard_id = builder.add_hazard_zone(0, 0)

    hazard = next(obj for obj in builder.finalize().objects if obj.id == hazard_id)

    assert hazard.appearance is not None
    assert hazard.appearance.asset_id == "courtyard_hazard"
    assert hazard.appearance.variant == "broken_paving"


def test_targeted_wall_segments_share_a_cohesive_default_appearance() -> None:
    builder = EnvSpec3DBuilder.from_spec_dict(env_spec_to_dict(courtyard_shell_spec("wall_style")))
    left_id = builder.add_wall(-3, 0, width=4, depth=0.35, height=1, id="divider_left")
    right_id = builder.add_wall(3, 0, width=4, depth=0.35, height=1, id="divider_right")

    objects = {obj.id: obj for obj in builder.finalize().objects}

    assert objects[left_id].appearance is not None
    assert objects[right_id].appearance is not None
    assert objects[left_id].appearance.model_dump() == objects[right_id].appearance.model_dump() == {
        "asset_id": "courtyard_boundary",
        "variant": "stone",
    }


def test_generator_validates_more_than_one_hundred_seeded_levels() -> None:
    generated = 0
    for family in sorted(LEVEL_FAMILIES):
        for difficulty in sorted(DIFFICULTIES):
            for seed in range(6):
                spec = _level(family=family, difficulty=difficulty, seed=seed)
                assert validate_courtyard_layout(spec) == []
                assert len([obj for obj in spec.objects if obj.semantic_type == "agent"]) == 1
                assert len([obj for obj in spec.objects if obj.semantic_type == "goal"]) == 1
                generated += 1
    assert generated == 108


def test_builder_operations_create_and_edit_courtyard() -> None:
    builder = EnvSpec3DBuilder("operation_level", description="operation")
    generated = execute_operation(
        builder,
        {"op": "make_courtyard_level", "args": {"family": "slalom", "difficulty": "hard", "seed": 8}},
    )
    rotated = execute_operation(builder, {"op": "rotate_object", "args": {"id": "slalom_prop_1", "yaw": 0.7}})
    appearance = execute_operation(
        builder,
        {
            "op": "set_object_appearance",
            "args": {"id": "slalom_prop_1", "asset_id": "courtyard_static_prop", "variant": "bench"},
        },
    )

    assert generated["success"] is True
    assert rotated["success"] is True
    assert appearance["success"] is True
    edited = next(obj for obj in builder.finalize().objects if obj.id == "slalom_prop_1")
    assert edited.yaw == pytest.approx(0.7)
    assert edited.appearance.variant == "bench"


def test_targeted_agent_and_goal_can_activate_game_without_recipe() -> None:
    shell = courtyard_shell_spec("targeted_game")
    before_visual = compile_visual_scene(shell)
    builder = EnvSpec3DBuilder.from_spec_dict(env_spec_to_dict(shell))
    agent_id = builder.add_agent_spawn(-7, -5, id="agent")
    goal_id = builder.add_goal_zone(7, 5, id="goal")

    result = builder.configure_reach_goal_game(agent_id, goal_id)
    spec = builder.finalize()

    assert result["game"]["mode"] == "reach_goal"
    assert spec.game is not None
    assert spec.game.agent_id == "agent"
    assert spec.game.goal_id == "goal"
    assert spec.game.play_bounds == [24.0, 18.0, 8.0]
    after_visual = compile_visual_scene(spec)
    before_boundaries = {
        obj["source_id"]: obj["visual_type"]
        for obj in before_visual["objects"]
        if obj.get("semantic_type") == "wall"
    }
    after_boundaries = {
        obj["source_id"]: obj["visual_type"]
        for obj in after_visual["objects"]
        if obj.get("semantic_type") == "wall"
    }
    assert before_boundaries == after_boundaries == {
        "north_boundary": "plain_wall",
        "south_boundary": "plain_wall",
        "west_boundary": "plain_wall",
        "east_boundary": "plain_wall",
    }


def test_boundary_walls_without_appearance_still_render_plain() -> None:
    raw = env_spec_to_dict(courtyard_shell_spec("missing_boundary_appearance"))
    for obj in raw["objects"]:
        if obj["semantic_type"] == "wall":
            obj.pop("appearance", None)

    visual = compile_visual_scene(raw)

    assert {
        obj["visual_type"]
        for obj in visual["objects"]
        if obj.get("semantic_type") == "wall"
    } == {"plain_wall"}


def test_session_rejects_unrequested_goal_and_full_level_recipe(tmp_path: Path) -> None:
    shell = courtyard_shell_spec("literal_edits")
    scene_dir = tmp_path / shell.id
    persist_artifacts(spec=shell, scene_dir=scene_dir, trace_records=[], render=False)
    session = SceneSession.resume(
        env_id=shell.id,
        output_root=tmp_path,
        prompt="put a box in the top left and an agent in the bottom right",
    )

    goal = session.apply_operation(
        {"op": "add_goal_zone", "args": {"x": 0, "y": 0, "id": "unrequested_goal"}}
    )
    recipe = session.apply_operation(
        {"op": "make_courtyard_level", "args": {"family": "push_lane", "difficulty": "medium", "seed": 4}}
    )

    assert goal["status"] == "error"
    assert "does not explicitly ask" in goal["operation_result"]["error"]
    assert recipe["status"] == "error"
    assert {obj["semantic_type"] for obj in session.builder.to_spec_dict()["objects"]} == {"ground", "wall"}

    negated = SceneSession.resume(
        env_id=shell.id,
        output_root=tmp_path,
        prompt="add an agent but do not add a goal",
    ).apply_operation({"op": "add_goal_zone", "args": {"x": 0, "y": 0}})
    explicit_session = SceneSession.resume(
        env_id=shell.id,
        output_root=tmp_path,
        prompt="add a charging-pad goal in the center",
    )
    explicit = explicit_session.apply_operation(
        {"op": "add_goal_zone", "args": {"x": 0, "y": 0, "id": "requested_goal"}}
    )

    assert negated["status"] == "error"
    assert explicit["status"] == "success"
    assert any(obj["id"] == "requested_goal" for obj in explicit_session.builder.to_spec_dict()["objects"])


def test_invalid_game_references_and_blocked_route_cannot_finalize() -> None:
    builder = EnvSpec3DBuilder("invalid_game", description="invalid")
    builder.make_courtyard_level(seed=3)
    builder.game["goal_id"] = "missing"
    with pytest.raises(BuilderValidationError, match="goal_id"):
        builder.finalize()


def test_switch_gate_compiles_real_joint_and_actuator() -> None:
    root = ET.fromstring(compile_spec_to_mjcf(_level(family="switch_gate")))

    assert root.find(".//body[@name='main_gate']/joint[@name='main_gate_slide']") is not None
    assert root.find("./actuator/position[@name='main_gate_actuator']") is not None
    assert root.find(".//geom[@name='gate_switch']").attrib["contype"] == "0"


def test_switch_latches_and_gate_opens_in_mujoco(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    spec = _level(family="switch_gate", seed=4)
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    simulation = PlayableSimulation.from_scene(scene_dir)
    switch = next(obj for obj in spec.objects if obj.semantic_type == "floor_switch")
    address = simulation.agent.qpos_address
    simulation.data.qpos[address : address + 3] = [switch.position[0], switch.position[1], 0.55]
    simulation.mujoco.mj_forward(simulation.model, simulation.data)

    for _ in range(120):
        simulation.step(right=0, forward=0, camera_azimuth=0)

    state = simulation.mechanism_states()[0]
    assert state["active"] is True
    assert state["progress"] > 0.9
    simulation.reset()
    assert simulation.mechanism_states()[0]["active"] is False


def test_goal_and_hazard_emit_events_without_ending_play(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    spec = _level(family="slalom", difficulty="medium", seed=9)
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    simulation = PlayableSimulation.from_scene(scene_dir)
    address = simulation.agent.qpos_address
    hazard = next(obj for obj in spec.objects if obj.semantic_type == "hazard")
    simulation.data.qpos[address : address + 3] = [hazard.position[0], hazard.position[1], 0.55]
    simulation.mujoco.mj_forward(simulation.model, simulation.data)
    simulation.step(right=0, forward=0, camera_azimuth=0)
    hazard_state = simulation.game_state()
    assert hazard_state["state"] == "playing"
    assert hazard_state["failure_reason"] == ""
    assert hazard.id in hazard_state["active_zones"]["hazard"]
    assert any(
        event["type"] == "zone_entered"
        and event["semantic_type"] == "hazard"
        and event["zone_id"] == hazard.id
        for event in simulation.zone_events_since()
    )
    agent_spawn = next(obj for obj in spec.objects if obj.semantic_type == "agent")
    simulation.data.qpos[address : address + 3] = agent_spawn.position
    simulation.mujoco.mj_forward(simulation.model, simulation.data)
    simulation.step(right=0, forward=0, camera_azimuth=0)
    assert any(
        event["type"] == "zone_exited" and event["zone_id"] == hazard.id
        for event in simulation.zone_events_since()
    )

    simulation.reset()
    goal = next(obj for obj in spec.objects if obj.semantic_type == "goal")
    simulation.data.qpos[address : address + 3] = [goal.position[0], goal.position[1], 0.55]
    simulation.mujoco.mj_forward(simulation.model, simulation.data)
    clock_values = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    session = PlaySession(id="play", env_id=spec.id, simulation=simulation, clock=lambda: next(clock_values))
    entered = session.advance(right=0, forward=0, camera_azimuth=0, jump=False)
    entered_steps = entered["game"]["metrics"]["steps"]
    continued = session.advance(right=1, forward=1, camera_azimuth=0, jump=True)

    assert entered["game"]["state"] == "playing"
    assert entered["game"]["metrics"]["goal_reached"] is True
    assert any(event["semantic_type"] == "goal" for event in entered["game"]["events"])
    assert continued["game"]["metrics"]["steps"] > entered_steps

    simulation.data.qpos[address] = float(spec.game.play_bounds[0])
    simulation.mujoco.mj_forward(simulation.model, simulation.data)
    failed = session.advance(right=0, forward=0, camera_azimuth=0, jump=False)
    failed_steps = failed["game"]["metrics"]["steps"]
    frozen = session.advance(right=1, forward=1, camera_azimuth=0, jump=True)
    assert failed["game"]["state"] == "failed"
    assert failed["game"]["failure_reason"] == "out_of_bounds"
    assert frozen["game"]["metrics"]["steps"] == failed_steps

    reset = session.reset()
    assert reset["game"]["state"] == "playing"
    assert reset["game"]["metrics"]["resets"] == 1


def test_game_fallback_trial_targets_goal_and_gate_in_order() -> None:
    spec = _level(family="switch_gate")
    plan = fallback_locomotion_plan(
        env_id=spec.id,
        prompt="reach the goal",
        operation_count=1,
        draft_spec=spec,
    )
    objective = plan["trials"][0]["objective"]

    assert [check["type"] for check in objective["checks"]] == ["mechanism_state", "zone_entry"]
    assert objective["ordered_check_ids"] == ["open_gate", "enter_goal"]


def test_courtyard_assets_are_local_and_licensed() -> None:
    root = Path(__file__).parents[1] / "environment_generation" / "studio_web"
    manifest = (root / "courtyard_assets.js").read_text(encoding="utf-8")
    assert "https://" not in manifest
    assert "kenney_nature/tree_default.glb" in manifest
    assert (root / "assets" / "courtyard" / "kenney_nature" / "tree_default.glb").is_file()
    assert (root / "assets" / "courtyard" / "kenney_platformer" / "crate.glb").is_file()
    assert (root / "assets" / "courtyard" / "kenney_platformer" / "Textures" / "colormap.png").is_file()
    assert "CC0" in (root / "assets" / "courtyard" / "SOURCES.md").read_text(encoding="utf-8")
