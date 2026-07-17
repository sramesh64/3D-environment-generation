from __future__ import annotations

import json
from pathlib import Path

import pytest

from environment_generation.artifacts import persist_artifacts
from environment_generation.behavior_guidance import (
    action_guidance,
    ground_route_guidance,
    interaction_guidance,
    objective_focus,
    semantic_affordances,
)
from environment_generation.behavior_trial import (
    BehaviorTrialSession,
    read_action_log,
    replay_behavior_actions,
)
from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.env_behavior_trials import normalize_behavior_trial_plan
from environment_generation.env_verification import scene_object_at, scene_objects


def _objective_state(*checks: tuple[str, bool]) -> dict:
    return {
        "satisfied": all(passed for _check_id, passed in checks),
        "checks": [
            {"id": check_id, "passed": passed, "metrics": {}}
            for check_id, passed in checks
        ],
    }


def _route_scene():
    builder = EnvSpec3DBuilder("guidance_route", description="route around a courtyard hazard")
    builder.add_ground_plane(14, 10)
    builder.add_agent_spawn(-5, 0, id="agent")
    builder.add_hazard_zone(0, 0, width=4, depth=4, id="hazard")
    builder.add_floor_switch(5, 0, id="switch")
    return builder.finalize()


def test_objective_focus_advances_ordered_push_then_zone_objectives() -> None:
    builder = EnvSpec3DBuilder("focus_scene", description="push then enter")
    builder.add_ground_plane(12, 8)
    builder.add_agent_spawn(-4, 0, id="agent")
    builder.add_pushable_box(0, 0, id="crate")
    builder.add_target_region(4, 0, id="drop_zone")
    spec = builder.finalize()
    objects = scene_objects(spec)
    trial = {
        "expected_outcome": "should_succeed",
        "objective": {
            "mode": "all",
            "ordered_check_ids": ["move_crate", "enter_zone"],
            "checks": [
                {
                    "id": "move_crate",
                    "type": "object_displacement",
                    "selector": {"id": "crate"},
                    "min_distance": 1.0,
                },
                {
                    "id": "enter_zone",
                    "type": "zone_entry",
                    "selector": {"id": "drop_zone"},
                    "min_count": 1,
                },
            ],
        },
    }

    push_focus = objective_focus(
        trial=trial,
        objective_state=_objective_state(("move_crate", False), ("enter_zone", False)),
        objects=objects,
    )
    zone_focus = objective_focus(
        trial=trial,
        objective_state=_objective_state(("move_crate", True), ("enter_zone", False)),
        objects=objects,
    )

    assert push_focus["type"] == "displacement"
    assert push_focus["subject_ids"] == ["crate"]
    assert push_focus["navigation_target_ids"] == ["crate"]
    assert "push" in push_focus["capabilities"]
    assert zone_focus["type"] == "overlap"
    assert zone_focus["target_ids"] == ["drop_zone"]


def test_objective_focus_revisits_a_condition_missing_an_ordered_witness() -> None:
    builder = EnvSpec3DBuilder("ordered_focus", description="ordered repeated contact")
    builder.add_ground_plane(12, 8)
    builder.add_agent_spawn(-4, 0, id="agent")
    builder.add_pushable_box(0, 0, id="crate")
    builder.add_goal_zone(4, 0, id="goal")
    objects = scene_objects(builder.finalize())
    trial = {
        "expected_outcome": "should_succeed",
        "objective": {
            "mode": "all",
            "ordered_check_ids": ["position_crate", "touch_crate", "enter_goal"],
            "checks": [
                {
                    "id": "position_crate",
                    "temporal": "eventually",
                    "predicate": {
                        "type": "displacement",
                        "subject": {"id": "crate"},
                        "min_value": 1.0,
                    },
                },
                {
                    "id": "touch_crate",
                    "temporal": "eventually",
                    "predicate": {
                        "type": "contact",
                        "subject": {"id": "agent"},
                        "target": {"id": "crate"},
                    },
                },
                {
                    "id": "enter_goal",
                    "temporal": "eventually",
                    "predicate": {
                        "type": "overlap",
                        "subject": {"id": "agent"},
                        "target": {"id": "goal"},
                    },
                },
            ],
        },
    }
    state = _objective_state(
        ("position_crate", True),
        ("touch_crate", True),
        ("enter_goal", False),
    )
    state["ordered_steps"] = [120, None, None]

    focus = objective_focus(trial=trial, objective_state=state, objects=objects)

    assert focus["check_id"] == "touch_crate"
    assert focus["type"] == "contact"
    assert focus["authoritative_check"]["passed"] is False


def test_generic_subject_target_guidance_stages_behind_movable_object() -> None:
    builder = EnvSpec3DBuilder("delivery_guidance", description="deliver an object")
    builder.add_ground_plane(12, 8)
    builder.add_agent_spawn(-4, 0, id="agent")
    builder.add_pushable_box(0, 0, id="delivery_object")
    builder.add_target_region(4, 0, id="destination")
    spec = builder.finalize()
    objects = scene_objects(spec)
    trial = {
        "expected_outcome": "should_succeed",
        "objective": {
            "ordered_check_ids": ["move_subject", "deliver"],
            "checks": [
                {
                    "id": "move_subject",
                    "description": "Move the selected object.",
                    "temporal": "eventually",
                    "predicate": {
                        "type": "displacement",
                        "subject": {"id": "delivery_object"},
                        "min_value": 0.5,
                    },
                },
                {
                    "id": "deliver",
                    "description": "Move the object into the destination.",
                    "temporal": "eventually",
                    "predicate": {
                        "type": "overlap",
                        "subject": {"id": "delivery_object"},
                        "target": {"id": "destination"},
                    },
                }
            ]
        },
    }
    focus = objective_focus(
        trial=trial,
        objective_state=_objective_state(("move_subject", False), ("deliver", False)),
        objects=objects,
    )
    guidance = interaction_guidance(
        objects=objects,
        focus=focus,
        agent_position=(-4.0, 0.0, 0.55),
        agent_radius=0.35,
        contact_ids=set(),
        camera_azimuth=-90.0,
    )

    assert focus["subject_ids"] == ["delivery_object"]
    assert focus["target_ids"] == ["destination"]
    assert focus["navigation_target_ids"] == ["delivery_object"]
    assert focus["destination_hint_check_id"] == "deliver"
    assert guidance["phase"] == "approach_staging_point"
    assert guidance["staging_position"][0] < 0
    assert guidance["push_direction_world"] == [1.0, 0.0]


def test_shape_guidance_never_substitutes_for_authoritative_objective_evidence() -> None:
    builder = EnvSpec3DBuilder("rotated_guidance", description="generic rotated geometry")
    builder.add_ground_plane(12, 12)
    builder.add_agent_spawn(3.5, -3.5, id="agent")
    builder.add_pushable_box(
        2.5,
        -2.5,
        width=0.6,
        depth=0.6,
        height=0.6,
        id="subject",
    )
    builder.add_static_box(
        0,
        0,
        width=8,
        depth=0.4,
        height=1,
        yaw=0.7853981633974483,
        id="target",
    )
    objects = scene_objects(builder.finalize())
    trial = {
        "expected_outcome": "should_succeed",
        "objective": {
            "checks": [
                {
                    "id": "make_contact",
                    "description": "Bring the selected subject into contact with the target.",
                    "temporal": "eventually",
                    "predicate": {
                        "type": "contact",
                        "subject": {"id": "subject"},
                        "target": {"id": "target"},
                    },
                }
            ]
        },
    }
    objective_state = _objective_state(("make_contact", False))
    focus = objective_focus(trial=trial, objective_state=objective_state, objects=objects)

    initial = interaction_guidance(
        objects=objects,
        focus=focus,
        agent_position=(3.5, -3.5, 0.55),
        agent_radius=0.35,
        contact_ids=set(),
        camera_azimuth=-90.0,
    )

    assert initial["geometric_destination_reached"] is False
    assert initial["authoritative_check_satisfied"] is False
    assert initial["phase"] != "objective_satisfied"
    assert "subject_in_target" not in initial
    assert initial["target_surface_point"] != pytest.approx([0.0, 0.0])

    desired = initial["desired_subject_center"]
    surface = initial["target_surface_point"]
    outward = (desired[0] - surface[0], desired[1] - surface[1])
    length = max(1e-9, (outward[0] ** 2 + outward[1] ** 2) ** 0.5)
    overlapping_center = (
        desired[0] - outward[0] / length * 0.01,
        desired[1] - outward[1] / length * 0.01,
    )
    moved_objects = [
        scene_object_at(obj, (*overlapping_center, obj.position[2]))
        if obj.id == "subject"
        else obj
        for obj in objects
    ]
    waiting = interaction_guidance(
        objects=moved_objects,
        focus=focus,
        agent_position=(3.5, -3.5, 0.55),
        agent_radius=0.35,
        contact_ids=set(),
        camera_azimuth=-90.0,
    )

    assert waiting["geometric_destination_reached"] is True
    assert waiting["authoritative_check_satisfied"] is False
    assert waiting["phase"] == "awaiting_objective_evidence"


def test_ground_route_goes_around_failure_zone_instead_of_using_direct_bearing() -> None:
    spec = _route_scene()
    objects = scene_objects(spec)
    focus = {
        "check_id": "activate_switch",
        "type": "zone_entry",
        "target_ids": ["switch"],
        "relation": None,
    }

    route = ground_route_guidance(
        spec=spec,
        objects=objects,
        focus=focus,
        agent_position=(-5.0, 0.0, 0.55),
        camera_azimuth=-90.0,
        mechanism_states=[],
    )

    assert route["available"] is True
    assert route["target_id"] == "switch"
    assert route["direct_path_blocked"] is True
    assert "hazard" in route["blocked_by"]
    assert len(route["waypoints"]) >= 2
    assert any(abs(point[1]) > 2.4 for point in route["waypoints"])


def test_route_preserves_clearance_recovery_before_turning_toward_target() -> None:
    spec = _route_scene()
    objects = scene_objects(spec)

    route = ground_route_guidance(
        spec=spec,
        objects=objects,
        focus={
            "check_id": "activate_switch",
            "type": "zone_entry",
            "target_ids": ["switch"],
            "relation": None,
        },
        # Outside the authored hazard, but inside its agent-radius safety margin.
        agent_position=(-2.38, 0.0, 0.55),
        camera_azimuth=-90.0,
        mechanism_states=[],
    )

    assert route["available"] is True
    assert route["status"] == "clearance_recovery"
    first = route["waypoints"][0]
    assert abs(first[1]) > 2.35 or first[0] < -2.45
    assert "recover safe clearance" in route["reason"]


def test_pushable_approach_guidance_does_not_require_a_goal() -> None:
    builder = EnvSpec3DBuilder("push_guidance", description="push a crate")
    builder.add_ground_plane(12, 8)
    builder.add_agent_spawn(-4, 0, id="agent")
    builder.add_pushable_box(3, 0, id="crate")
    spec = builder.finalize()
    objects = scene_objects(spec)
    crate = next(obj for obj in objects if obj.id == "crate")

    route = ground_route_guidance(
        spec=spec,
        objects=objects,
        focus={
            "check_id": "push",
            "type": "object_displacement",
            "target_ids": ["crate"],
            "relation": None,
        },
        agent_position=(-4.0, 0.0, 0.55),
        camera_azimuth=-90.0,
        mechanism_states=[],
    )

    assert route["available"] is True
    assert route["target_semantic_type"] == "pushable_box"
    assert "pushable" in semantic_affordances(crate)
    assert not any(obj.semantic_type == "goal" for obj in objects)


def test_sealed_wall_reports_no_ground_route_without_claiming_impossibility() -> None:
    builder = EnvSpec3DBuilder("sealed_wall", description="search for a wall bypass")
    builder.add_ground_plane(12, 8)
    builder.add_agent_spawn(-4, 0, id="agent")
    builder.add_wall(0, 0, width=0.5, depth=8, height=2, id="wall")
    spec = builder.finalize()

    route = ground_route_guidance(
        spec=spec,
        objects=scene_objects(spec),
        focus={
            "check_id": "past_wall",
            "type": "agent_relation",
            "relation": "right_of",
            "target_ids": ["wall"],
        },
        agent_position=(-4.0, 0.0, 0.55),
        camera_azimuth=-90.0,
        mechanism_states=[],
    )

    assert route["available"] is False
    assert route["status"] == "no_route"
    assert route["advisory_only"] is True
    assert "impossible" not in route["reason"].lower()


def test_no_target_objective_keeps_route_guidance_optional() -> None:
    spec = _route_scene()
    route = ground_route_guidance(
        spec=spec,
        objects=scene_objects(spec),
        focus={"check_id": "move", "type": "agent_displacement", "target_ids": []},
        agent_position=(-5.0, 0.0, 0.55),
        camera_azimuth=-90.0,
        mechanism_states=[],
    )

    assert route["available"] is False
    assert route["status"] == "not_applicable"


def test_action_guidance_uses_fine_batches_near_failure_zone() -> None:
    spec = _route_scene()
    objects = scene_objects(spec)
    focus = {"target_ids": ["switch"]}

    near = action_guidance(
        agent_position=(-2.5, 0.0, 0.55),
        agent_radius=0.35,
        objects=objects,
        focus=focus,
        route={"available": True, "next_waypoint_relative": {"distance": 2.0}},
    )
    far = action_guidance(
        agent_position=(-5.0, -4.0, 0.55),
        agent_radius=0.35,
        objects=objects,
        focus=focus,
        route={"available": True, "next_waypoint_relative": {"distance": 2.0}},
    )

    assert near["recommended_max_frames"] == 4
    assert far["recommended_max_frames"] == 18


def test_trial_observation_exposes_guidance_and_persists_audit_log(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    spec = _route_scene()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    trial = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="reach the switch by going around the hazard",
        operation_count=4,
        draft_spec=spec,
        trials=[
            {
                "id": "switch_route",
                "instruction": "Enter the switch without touching the hazard.",
                "objective": {
                    "checks": [
                        {"id": "switch", "type": "zone_entry", "selector": {"id": "switch"}}
                    ]
                },
            }
        ],
    )["trials"][0]
    observation_log = tmp_path / "observations.jsonl"
    session = BehaviorTrialSession(
        scene_dir=scene_dir,
        trial=trial,
        observation_log_path=observation_log,
        render_frames=False,
    )

    observation = session.start()
    session.close()

    assert observation["objective_focus"]["check_id"] == "switch"
    assert observation["route_guidance"]["direct_path_blocked"] is True
    assert observation["action_guidance"]["advisory_only"] is True
    hazard = next(item for item in observation["nearby_objects"] if item["id"] == "hazard")
    assert "failure_zone" in hazard["affordances"]
    records = [json.loads(line) for line in observation_log.read_text(encoding="utf-8").splitlines()]
    assert records[0]["sequence"] == 0
    assert records[0]["objective_focus"]["check_id"] == "switch"


def test_assisted_ground_route_logs_low_level_actions_for_exact_replay(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    builder = EnvSpec3DBuilder("assisted_route", description="reach a switch around a hazard")
    builder.add_ground_plane(10, 8)
    builder.add_agent_spawn(-3.5, 0, id="agent")
    builder.add_hazard_zone(0, 0, width=2, depth=2, id="hazard")
    builder.add_floor_switch(3.5, 0, id="switch")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    trial = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="reach the switch without entering the hazard",
        operation_count=4,
        draft_spec=spec,
        trials=[
            {
                "id": "reach_switch",
                "instruction": "Reach the switch around the hazard.",
                "max_steps": 900,
                "objective": {
                    "checks": [
                        {"id": "switch", "type": "zone_entry", "selector": {"id": "switch"}}
                    ]
                },
            }
        ],
    )["trials"][0]
    action_log = tmp_path / "assisted_actions.jsonl"
    observation_log = tmp_path / "assisted_observations.jsonl"
    session = BehaviorTrialSession(
        scene_dir=scene_dir,
        trial=trial,
        action_log_path=action_log,
        observation_log_path=observation_log,
        render_frames=False,
    )
    session.start()

    assist_calls = 1
    assisted = session.act(assist="ground_route", frames=300)
    if not session.terminal:
        assist_calls += 1
        assisted = session.act(assist="ground_route", frames=300)
    original = session.stop()
    session.close()
    actions = read_action_log(action_log)
    replay = replay_behavior_actions(scene_dir=scene_dir, trial=trial, actions=actions)

    assert assisted["assisted_action"]["controller_segments"] > 1
    assert assisted["assisted_action"]["authoritative_replay_uses_logged_controller_segments"] is True
    assert all(action.get("action") == "controller" for action in actions)
    assert len(observation_log.read_text(encoding="utf-8").splitlines()) == 1 + assist_calls
    assert original["status"] == "passed"
    assert replay["status"] == "passed"
    assert replay["steps_used"] == original["steps_used"]
    original_agent = next(item for item in original["final_state"]["objects"] if item["id"] == "agent")
    replay_agent = next(item for item in replay["final_state"]["objects"] if item["id"] == "agent")
    assert replay_agent["position"] == pytest.approx(original_agent["position"], abs=1e-9)


def test_assisted_ground_route_can_target_an_intermediate_scene_object(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    builder = EnvSpec3DBuilder("assisted_subgoal", description="route to an intermediate switch")
    builder.add_ground_plane(12, 8)
    builder.add_agent_spawn(-4, 0, id="agent")
    builder.add_pushable_box(-1, 0, id="crate")
    builder.add_floor_switch(4, 0, id="switch")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    trial = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="push the crate",
        operation_count=4,
        draft_spec=spec,
        trials=[
            {
                "id": "push_crate",
                "instruction": "Push the crate.",
                "max_steps": 900,
                "objective": {
                    "checks": [
                        {
                            "id": "move_crate",
                            "type": "object_displacement",
                            "selector": {"id": "crate"},
                            "min_distance": 1,
                        }
                    ]
                },
            }
        ],
    )["trials"][0]
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=trial, render_frames=False)
    initial = session.start()

    assisted = session.act(assist="ground_route", target_id="switch", frames=300)

    assert assisted["assisted_action"]["target_id"] == "switch"
    assert assisted["objective"]["satisfied"] is False
    assert assisted["agent"]["position"][0] > initial["agent"]["position"][0] + 5
    session.close()


def test_intermediate_switch_route_stops_when_linked_gate_is_ready(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    builder = EnvSpec3DBuilder("assisted_gate", description="wait only until the linked gate opens")
    builder.add_ground_plane(12, 8)
    builder.add_agent_spawn(-4, 0, id="agent")
    builder.add_pushable_box(4, 0, id="crate")
    switch_id = builder.add_floor_switch(0, 0, id="switch")
    gate_id = builder.add_sliding_gate(2, 0, id="gate")
    builder.link_switch_to_gate(switch_id, gate_id, id="gate_link")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    trial = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="push the crate",
        operation_count=6,
        draft_spec=spec,
        trials=[
            {
                "id": "push_crate",
                "instruction": "Push the crate.",
                "max_steps": 900,
                "objective": {
                    "checks": [
                        {
                            "id": "move_crate",
                            "type": "object_displacement",
                            "selector": {"id": "crate"},
                            "min_distance": 1,
                        }
                    ]
                },
            }
        ],
    )["trials"][0]
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=trial, render_frames=False)
    session.start()

    assisted = session.act(assist="ground_route", target_id="switch", frames=300)

    assert assisted["assisted_action"]["stop_reason"] == "mechanism_ready"
    assert assisted["assisted_action"]["frames_advanced"] < 300
    assert assisted["mechanisms"][0]["progress"] >= 0.9
    session.close()


def test_assisted_route_stops_before_solid_object_interaction(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    builder = EnvSpec3DBuilder("assisted_push", description="approach a crate without assuming a goal")
    builder.add_ground_plane(10, 8)
    builder.add_agent_spawn(-3.5, 0, id="agent")
    builder.add_pushable_box(2.5, 0, id="crate")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    trial = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="push the crate",
        operation_count=3,
        draft_spec=spec,
        trials=[
            {
                "id": "push",
                "instruction": "Push the crate.",
                "max_steps": 600,
                "objective": {
                    "checks": [
                        {
                            "id": "move_crate",
                            "type": "object_displacement",
                            "selector": {"id": "crate"},
                            "min_distance": 0.5,
                        }
                    ]
                },
            }
        ],
    )["trials"][0]
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=trial, render_frames=False)
    session.start()

    observation = session.act(assist="ground_route", frames=300)

    assert observation["assisted_action"]["stop_reason"] == "interaction_range"
    assert observation["objective"]["satisfied"] is False
    crate = next(item for item in observation["nearby_objects"] if item["id"] == "crate")
    assert crate["horizontal_clearance"] <= 0.45
    session.close()
