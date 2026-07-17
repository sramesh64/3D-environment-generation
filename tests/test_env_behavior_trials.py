from __future__ import annotations

import copy
import json
import os

import pytest

from environment_generation.artifacts import load_scene, persist_artifacts
from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.env_behavior_trials import (
    BEHAVIOR_CONTROLLER_VERSION,
    ENV_BEHAVIOR_TRIAL_SCHEMA_VERSION,
    EnvBehaviorTrialError,
    active_behavior_runs,
    behavior_plan_decision_issues,
    behavior_prompt_hash,
    behavior_trial_summary,
    behavior_trial_history,
    build_behavior_trial_report,
    fallback_locomotion_plan,
    load_behavior_trial_plan,
    load_behavior_trial_report,
    normalize_behavior_trial_result,
    normalize_behavior_trial_plan,
    upgrade_behavior_trial_plan,
    write_behavior_trial_plan,
)
from environment_generation.env_verification import spec_hash
from environment_generation.session import SceneSessionManager


def _agent_spec():
    builder = EnvSpec3DBuilder("behavior_scene", description="agent scene")
    builder.add_ground_plane()
    builder.add_agent_spawn(0, 0)
    builder.add_wall(2, 0, id="barrier")
    builder.add_pushable_box(0.8, 0, id="box")
    return builder.finalize()


def _hazard_spec():
    builder = EnvSpec3DBuilder("hazard_behavior_scene", description="agent facing a failure zone")
    builder.add_ground_plane(10, 6)
    builder.add_agent_spawn(-2, 0, id="agent")
    builder.add_hazard_zone(0, 0, width=1.2, depth=2, id="broken_paving")
    return builder.finalize()


def _negative_wall_plan(spec):
    return normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="the wall blocks the direct route",
        operation_count=3,
        draft_spec=spec,
        trials=[
            {
                "id": "wall_bypass",
                "instruction": "Try to get to the far side without jumping.",
                "expected_outcome": "should_not_succeed",
                "objective": {
                    "checks": [
                        {
                            "type": "agent_relation",
                            "target": {"id": "barrier"},
                            "relation": "right_of",
                        }
                    ]
                },
                "constraints": {"checks": [{"type": "jump_count", "max_count": 0}]},
            }
        ],
    )


def test_historical_results_expand_generic_check_labels_without_rewriting_the_plan() -> None:
    trial = {
        "expected_outcome": "should_succeed",
        "objective": {
            "checks": [
                {
                    "id": "clear_wall",
                    "description": "relation",
                    "predicate": {
                        "type": "relation",
                        "subject": {"id": "robot"},
                        "target": {"id": "barrier"},
                        "relation": "above",
                    },
                }
            ]
        },
    }
    result = normalize_behavior_trial_result(
        {
            "status": "inconclusive",
            "objective": {
                "satisfied": False,
                "checks": [
                    {
                        **trial["objective"]["checks"][0],
                        "passed": False,
                        "metrics": {"transition_count": 0},
                    }
                ],
            },
            "repair_hints": ["Review or repair the unmet objective checks: relation."],
        },
        trial=trial,
    )

    assert trial["objective"]["checks"][0]["description"] == "relation"
    assert result["objective"]["checks"][0]["description"] == "Robot is above barrier."
    assert result["repair_hints"] == [
        "Review or repair the unmet objective checks: Robot is above barrier."
    ]


def test_plan_normalizes_typed_affordance_objectives() -> None:
    plan = normalize_behavior_trial_plan(
        env_id="behavior_scene",
        prompt="climb the box tower",
        operation_count=4,
        draft_spec=_agent_spec(),
        trials=[
            {
                "id": "climb tower",
                "instruction": "Jump onto the box and become positioned above it.",
                "objective": {
                    "mode": "all",
                    "checks": [
                        {"id": "jumped", "type": "jump_count", "min_count": 1},
                        {"id": "touch_box", "type": "contact_count", "selector": {"id": "box"}},
                        {
                            "id": "above_box",
                            "type": "agent_relation",
                            "target": {"id": "box"},
                            "relation": "above",
                        },
                    ],
                    "ordered_check_ids": ["jumped", "touch_box", "above_box"],
                },
            }
        ],
    )

    trial = plan["trials"][0]
    assert trial["id"] == "climb_tower"
    assert trial["objective"]["ordered_check_ids"] == ["jumped", "touch_box", "above_box"]
    assert trial["objective"]["checks"][1]["selector"] == {"id": "box"}


def test_plan_accepts_generic_subject_target_assertions() -> None:
    builder = EnvSpec3DBuilder("generic_behavior", description="deliver any movable object")
    builder.add_ground_plane(12, 8)
    builder.add_agent_spawn(-4, 0, id="agent")
    builder.add_ball(0, 0, id="movable_subject")
    builder.add_target_region(4, 0, id="destination")
    builder.add_hazard_zone(0, 3, id="hazard")
    spec = builder.finalize()

    plan = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="move the ball into the region",
        operation_count=4,
        draft_spec=spec,
        trials=[
            {
                "instruction": "Push the selected object into the destination.",
                "objective": {
                    "checks": [
                        {
                            "id": "delivered",
                            "temporal": "eventually",
                            "predicate": {
                                "type": "overlap",
                                "subject": {"id": "movable_subject"},
                                "target": {"id": "destination"},
                            },
                        }
                    ]
                },
                "constraints": {
                    "checks": [
                        {
                            "id": "agent_avoids_hazards",
                            "temporal": "never",
                            "predicate": {
                                "type": "overlap",
                                "subject": {"semantic_type": "agent"},
                                "target": {"semantic_type": "hazard"},
                            },
                        }
                    ]
                },
            }
        ],
    )

    condition = plan["trials"][0]["objective"]["checks"][0]
    assert plan["schema_version"] == ENV_BEHAVIOR_TRIAL_SCHEMA_VERSION
    assert plan["controller_version"] == BEHAVIOR_CONTROLLER_VERSION
    assert condition["predicate"]["subject"] == {"id": "movable_subject"}
    assert condition["predicate"]["target"] == {"id": "destination"}


def test_climb_plan_rejects_aggregate_height_inside_ordered_phases() -> None:
    spec = _agent_spec()
    raw_trial = {
        "instruction": "Get onto the box and then finish above it.",
        "objective": {
            "checks": [
                {
                    "id": "reach_box",
                    "type": "agent_relation",
                    "target": {"id": "box"},
                    "relation": "on_surface",
                },
                {"id": "gain_height", "type": "agent_height_gain", "min_gain": 0.5},
                {
                    "id": "finish_above",
                    "type": "agent_relation",
                    "target": {"id": "box"},
                    "relation": "above",
                },
            ],
            "ordered_check_ids": ["reach_box", "gain_height", "finish_above"],
        },
    }

    with pytest.raises(EnvBehaviorTrialError, match="trial-global aggregate"):
        normalize_behavior_trial_plan(
            env_id=spec.id,
            prompt="climb the box",
            operation_count=3,
            draft_spec=spec,
            trials=[raw_trial],
        )

    corrected = copy.deepcopy(raw_trial)
    corrected["objective"]["ordered_check_ids"] = ["reach_box", "finish_above"]
    plan = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="climb the box",
        operation_count=3,
        draft_spec=spec,
        trials=[corrected],
    )

    assert plan["trials"][0]["objective"]["ordered_check_ids"] == [
        "reach_box",
        "finish_above",
    ]
    assert plan["trials"][0]["objective"]["checks"][1]["predicate"]["metric"] == "maximum"


def test_current_persisted_plan_with_ambiguous_order_is_marked_for_regeneration(
    tmp_path,
) -> None:
    spec = _agent_spec()
    good_plan = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="climb the box",
        operation_count=3,
        draft_spec=spec,
        trials=[
            {
                "instruction": "Get onto the box.",
                "objective": {
                    "checks": [
                        {"id": "gain_height", "type": "agent_height_gain", "min_gain": 0.5},
                        {
                            "id": "reach_box",
                            "type": "agent_relation",
                            "target": {"id": "box"},
                            "relation": "on_surface",
                        },
                    ],
                    "ordered_check_ids": ["reach_box"],
                },
            }
        ],
    )
    bad_plan = copy.deepcopy(good_plan)
    bad_plan["trials"][0]["objective"]["ordered_check_ids"] = [
        "gain_height",
        "reach_box",
    ]

    upgraded = upgrade_behavior_trial_plan(bad_plan, spec=spec)

    assert "trial-global aggregate" in upgraded["migration_issues"][0]

    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    write_behavior_trial_plan(scene_dir, bad_plan)
    loaded = load_behavior_trial_plan(scene_dir)
    summary = behavior_trial_summary(scene_dir, current_spec=spec)

    assert "trial-global aggregate" in loaded["migration_issues"][0]
    assert summary["status"] == "stale"


def test_legacy_zone_entry_cannot_silently_mean_object_delivery() -> None:
    with pytest.raises(EnvBehaviorTrialError, match="explicit subject and target"):
        normalize_behavior_trial_plan(
            env_id="behavior_scene",
            prompt="move the box",
            operation_count=2,
            draft_spec=_agent_spec(),
            trials=[
                {
                    "instruction": "Move the box into something.",
                    "objective": {
                        "checks": [
                            {
                                "id": "invalid_delivery",
                                "type": "zone_entry",
                                "selector": {"id": "box"},
                            }
                        ]
                    },
                }
            ],
        )


def test_loading_a_legacy_plan_upgrades_it_to_canonical_assertions(tmp_path) -> None:
    spec = _agent_spec()
    persist_artifacts(spec=spec, scene_dir=tmp_path, trace_records=[], render=False)
    legacy = {
        "schema_version": "1.0",
        "controller_version": "controller_v2",
        "env_id": spec.id,
        "prompt": "move the agent",
        "operation_count": 2,
        "draft_hash": spec_hash(spec),
        "trials": [
            {
                "id": "move",
                "instruction": "Move at least one meter.",
                "expected_outcome": "should_succeed",
                "severity": "advisory",
                "max_steps": 2400,
                "max_resets": 1,
                "objective": {
                    "checks": [
                        {
                            "id": "moved",
                            "type": "agent_displacement",
                            "min_distance": 1.0,
                        }
                    ]
                },
            }
        ],
    }
    write_behavior_trial_plan(tmp_path, legacy)

    upgraded = load_behavior_trial_plan(tmp_path)

    assert upgraded["schema_version"] == ENV_BEHAVIOR_TRIAL_SCHEMA_VERSION
    assert upgraded["controller_version"] == BEHAVIOR_CONTROLLER_VERSION
    assert upgraded["upgraded_from_schema_version"] == "1.0"
    assert upgraded["trials"][0]["objective"]["checks"][0]["predicate"] == {
        "type": "displacement",
        "subject": {"semantic_type": "agent"},
        "subject_quantifier": "any",
        "metric": "maximum",
        "space": "xy",
        "min_value": 1.0,
    }


def test_plan_rejects_unsafe_or_unsupported_objectives() -> None:
    with pytest.raises(EnvBehaviorTrialError, match="unsupported behavior check"):
        normalize_behavior_trial_plan(
            env_id="behavior_scene",
            prompt="bad",
            operation_count=1,
            draft_spec=_agent_spec(),
            trials=[
                {
                    "instruction": "Run arbitrary code.",
                    "verifier_code": "open('/tmp/x', 'w')",
                    "objective": {"checks": [{"type": "python_verifier"}]},
                }
            ],
        )


def test_plan_rejects_agentless_scene() -> None:
    builder = EnvSpec3DBuilder("static_scene")
    builder.add_wall(0, 0)

    with pytest.raises(EnvBehaviorTrialError, match="authored agent"):
        normalize_behavior_trial_plan(
            env_id="static_scene",
            prompt="wall",
            operation_count=1,
            draft_spec=builder.finalize(),
            trials=[
                {
                    "instruction": "Move.",
                    "objective": {"checks": [{"type": "agent_displacement"}]},
                }
            ],
        )


def test_agentless_summary_is_not_applicable_even_with_historical_plan(tmp_path) -> None:
    historical = fallback_locomotion_plan(
        env_id="behavior_scene",
        prompt="move",
        operation_count=1,
        draft_spec=_agent_spec(),
    )
    write_behavior_trial_plan(tmp_path, historical)
    builder = EnvSpec3DBuilder("behavior_scene")
    builder.add_wall(0, 0)

    summary = behavior_trial_summary(tmp_path, current_spec=builder.finalize())

    assert summary["status"] == "not_applicable"
    assert summary["trial_count"] == 0


def test_loaded_agentless_scene_exposes_non_playable_capabilities(tmp_path) -> None:
    builder = EnvSpec3DBuilder("static_scene")
    builder.add_wall(0, 0)
    scene_dir = tmp_path / "static_scene"
    persist_artifacts(spec=builder.finalize(), scene_dir=scene_dir, trace_records=[], render=False)

    scene = load_scene(scene_dir)

    assert scene["capabilities"] == {
        "has_agent": False,
        "has_goal": False,
        "has_ground": False,
        "playable": False,
        "behavior_testable": False,
        "task_testable": False,
        "gameplay_ready": False,
    }
    assert scene["env_behavior_trials"]["status"] == "not_applicable"


def test_fallback_is_basic_locomotion_only_for_agent_scene() -> None:
    plan = fallback_locomotion_plan(
        env_id="behavior_scene",
        prompt="an agent on a floor",
        operation_count=2,
        draft_spec=_agent_spec(),
    )

    assert plan["fallback"] is True
    assert plan["trials"][0]["id"] == "basic_locomotion"
    checks = plan["trials"][0]["objective"]["checks"]
    assert [check["type"] for check in checks] == ["agent_displacement", "agent_relation"]
    assert checks[1]["relation"] == "on_surface"
    assert checks[1]["when"] == "final"


def test_session_finalization_requires_explicit_fallback_decision_for_agent(tmp_path, monkeypatch) -> None:
    manager = SceneSessionManager(tmp_path)
    manager.create_scene(env_id="fallback_scene", prompt="an agent on the ground")
    manager.apply_operation(
        env_id="fallback_scene",
        operation={"op": "add_ground_plane", "args": {}},
    )
    manager.apply_operation(
        env_id="fallback_scene",
        operation={"op": "add_agent_spawn", "args": {"x": 0, "y": 0}},
    )
    monkeypatch.setattr("environment_generation.session.persist_artifacts", _fake_persist)

    blocked = manager.finalize_scene(env_id="fallback_scene")
    selected = manager.use_default_env_behavior_trial(
        env_id="fallback_scene",
        reason="The request only authored an agent and did not specify a concrete affordance.",
    )
    result = manager.finalize_scene(env_id="fallback_scene")

    assert blocked["status"] == "needs_changes"
    assert blocked["env_behavior_trials"]["status"] == "needs_decision"
    assert selected["plan"]["decision"] == "fallback"
    assert result["status"] == "success"
    assert result["env_behavior_trial_plan"]["fallback"] is True
    assert result["env_behavior_trial_plan"]["source_prompt_hash"] == behavior_prompt_hash(
        "an agent on the ground"
    )
    assert result["env_behavior_trial_plan"]["source_turn_id"].startswith("session-")
    assert (tmp_path / "fallback_scene" / "env_behavior_trials_plan.json").is_file()


def test_agentless_session_finalizes_without_behavior_decision(tmp_path, monkeypatch) -> None:
    manager = SceneSessionManager(tmp_path)
    manager.create_scene(env_id="static_scene", prompt="add one static courtyard wall")
    manager.apply_operation(
        env_id="static_scene",
        operation={"op": "add_wall", "args": {"x": 0, "y": 0, "id": "wall"}},
    )
    monkeypatch.setattr("environment_generation.session.persist_artifacts", _fake_persist)

    result = manager.finalize_scene(env_id="static_scene")

    assert result["status"] == "success"
    assert result["env_behavior_trial_plan"] is None
    assert not (tmp_path / "static_scene" / "env_behavior_trials_plan.json").exists()


def test_prompt_specific_plan_records_current_turn_provenance(tmp_path, monkeypatch) -> None:
    manager = SceneSessionManager(tmp_path)
    manager.create_scene(env_id="climb_scene", prompt="the robot should climb onto the box")
    manager.apply_operation(env_id="climb_scene", operation={"op": "add_ground_plane", "args": {}})
    manager.apply_operation(
        env_id="climb_scene",
        operation={"op": "add_agent_spawn", "args": {"x": 0, "y": 0}},
    )
    manager.apply_operation(
        env_id="climb_scene",
        operation={"op": "add_static_box", "args": {"x": 1, "y": 0, "id": "step"}},
    )
    monkeypatch.setattr("environment_generation.session.persist_artifacts", _fake_persist)

    defined = manager.define_env_behavior_trials(
        env_id="climb_scene",
        intent_summary="Demonstrate the latest request by climbing onto the authored step.",
        trials=[
            {
                "id": "climb_step",
                "instruction": "Climb onto the authored step.",
                "objective": {
                    "checks": [
                        {
                            "id": "on_step",
                            "type": "agent_relation",
                            "target": {"id": "step"},
                            "relation": "on_surface",
                        }
                    ]
                },
            }
        ],
    )
    finalized = manager.finalize_scene(env_id="climb_scene")

    assert defined["plan"]["decision"] == "prompt_specific"
    assert defined["plan"]["intent_summary"].startswith("Demonstrate the latest request")
    assert finalized["status"] == "success"


def test_preserve_behavior_trials_rebinds_plan_to_current_revision(tmp_path, monkeypatch) -> None:
    manager = SceneSessionManager(tmp_path)
    manager.create_scene(env_id="preserve_scene", prompt="add a controllable robot")
    manager.apply_operation(env_id="preserve_scene", operation={"op": "add_ground_plane", "args": {}})
    manager.apply_operation(
        env_id="preserve_scene",
        operation={"op": "add_agent_spawn", "args": {"x": 0, "y": 0}},
    )
    monkeypatch.setattr("environment_generation.session.persist_artifacts", _fake_persist)
    manager.use_default_env_behavior_trial(
        env_id="preserve_scene",
        reason="No concrete behavior was requested in the initial turn.",
    )
    assert manager.finalize_scene(env_id="preserve_scene")["status"] == "success"

    manager.resume_scene(env_id="preserve_scene", prompt="change only the visual presentation")
    preserved = manager.preserve_env_behavior_trials(
        env_id="preserve_scene",
        reason="The revision is presentation-only and cannot affect the existing locomotion test.",
    )
    finalized = manager.finalize_scene(env_id="preserve_scene")

    assert preserved["status"] == "success"
    assert preserved["plan"]["decision"] == "preserved"
    assert preserved["plan"]["fallback"] is True
    assert preserved["plan"]["source_prompt_hash"] == behavior_prompt_hash(
        "change only the visual presentation"
    )
    assert preserved["plan"]["preserved_from_turn_id"]
    assert finalized["status"] == "success"


def test_preserve_behavior_trials_rejects_removed_target(tmp_path, monkeypatch) -> None:
    manager = SceneSessionManager(tmp_path)
    manager.create_scene(env_id="changed_scene", prompt="the robot should climb onto the step")
    manager.apply_operation(env_id="changed_scene", operation={"op": "add_ground_plane", "args": {}})
    manager.apply_operation(
        env_id="changed_scene",
        operation={"op": "add_agent_spawn", "args": {"x": 0, "y": 0}},
    )
    manager.apply_operation(
        env_id="changed_scene",
        operation={"op": "add_static_box", "args": {"x": 1, "y": 0, "id": "step"}},
    )
    monkeypatch.setattr("environment_generation.session.persist_artifacts", _fake_persist)
    manager.define_env_behavior_trials(
        env_id="changed_scene",
        intent_summary="Demonstrate climbing onto the authored step.",
        trials=[
            {
                "id": "climb_step",
                "instruction": "Climb onto the authored step.",
                "objective": {
                    "checks": [
                        {
                            "id": "on_step",
                            "type": "agent_relation",
                            "target": {"id": "step"},
                            "relation": "on_surface",
                        }
                    ]
                },
            }
        ],
    )
    assert manager.finalize_scene(env_id="changed_scene")["status"] == "success"

    manager.resume_scene(env_id="changed_scene", prompt="remove the step")
    removed = manager.apply_operation(
        env_id="changed_scene",
        operation={"op": "remove_object", "args": {"id": "step"}},
    )
    preserved = manager.preserve_env_behavior_trials(
        env_id="changed_scene",
        reason="Keep the prior climbing test.",
    )

    assert removed["operation_result"]["success"] is True
    assert preserved["status"] == "needs_changes"
    assert "does not match any current scene object" in " ".join(preserved["issues"])


def test_behavior_plan_decision_rejects_plan_from_previous_turn() -> None:
    spec = _agent_spec()
    plan = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="old request",
        operation_count=1,
        draft_spec=spec,
        source_turn_id="turn-old",
        decision_reason="The old request introduced locomotion.",
        intent_summary="Move the agent.",
        trials=[
            {
                "id": "move",
                "instruction": "Move the agent.",
                "objective": {"checks": [{"type": "agent_displacement", "min_distance": 1}]},
            }
        ],
    )

    issues = behavior_plan_decision_issues(
        plan,
        current_spec=spec,
        prompt="new request",
        source_turn_id="turn-new",
    )

    assert "different user request" in " ".join(issues)
    assert "current Studio turn" in " ".join(issues)


def test_report_separates_inconclusive_and_forbidden_counterexample() -> None:
    spec = _agent_spec()
    plan = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="wall blocks direct route",
        operation_count=3,
        draft_spec=spec,
        trials=[
            {
                "id": "climb",
                "instruction": "Climb the box.",
                "expected_outcome": "should_succeed",
                "objective": {"checks": [{"type": "agent_height_gain", "min_gain": 1.0}]},
            },
            {
                "id": "walkaround",
                "instruction": "Try to walk around the wall without jumping.",
                "expected_outcome": "should_not_succeed",
                "objective": {
                    "checks": [
                        {"type": "agent_relation", "target": {"id": "barrier"}, "relation": "right_of"}
                    ]
                },
                "constraints": {"checks": [{"type": "jump_count", "max_count": 0}]},
            },
        ],
    )
    report = build_behavior_trial_report(
        env_id=spec.id,
        plan=plan,
        env_spec_hash=spec_hash(spec),
        run_id="run-0001-test",
        model="test-model",
        results=[
            {"trial_id": "climb", "status": "inconclusive"},
            {"trial_id": "walkaround", "status": "failed"},
        ],
    )

    assert report["status"] == "failed"
    assert report["blocking"] is False
    assert report["summary"]["inconclusive"] == 1
    assert report["summary"]["failed"] == 1


def test_negative_trial_rejects_limit_only_objective_and_accepts_attempt_constraints() -> None:
    spec = _agent_spec()
    with pytest.raises(EnvBehaviorTrialError, match="genuine attempt rules"):
        normalize_behavior_trial_plan(
            env_id=spec.id,
            prompt="the wall cannot be bypassed without jumping",
            operation_count=3,
            draft_spec=spec,
            trials=[
                {
                    "id": "bad_counterexample",
                    "instruction": "Try to get around the wall without jumping.",
                    "expected_outcome": "should_not_succeed",
                    "objective": {"checks": [{"type": "jump_count", "max_count": 0}]},
                }
            ],
        )

    plan = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="the wall cannot be bypassed without jumping",
        operation_count=3,
        draft_spec=spec,
        trials=[
            {
                "id": "wall_bypass",
                "instruction": "Try to get around the wall without jumping.",
                "expected_outcome": "should_not_succeed",
                "objective": {
                    "checks": [
                        {
                            "id": "past_wall",
                            "type": "agent_relation",
                            "target": {"id": "barrier"},
                            "relation": "right_of",
                        }
                    ]
                },
                "constraints": {
                    "checks": [{"id": "no_jump", "type": "jump_count", "max_count": 0}]
                },
            }
        ],
    )

    assert plan["trials"][0]["constraints"]["checks"][0]["id"] == "no_jump"


def test_plan_accepts_passive_hazard_entry_and_legacy_terminal_predicates() -> None:
    spec = _hazard_spec()
    plan = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="enter the broken paving",
        operation_count=3,
        draft_spec=spec,
        trials=[
            {
                "instruction": "Enter the broken paving.",
                "objective": {"checks": [{
                    "id": "entered_hazard",
                    "type": "zone_entry",
                    "selector": {"id": "broken_paving"},
                }]},
            },
            {
                "instruction": "Load a legacy hazard-event test.",
                "objective": {
                    "checks": [
                        {"id": "hazard_terminal", "type": "terminal_event", "event": "hazard"}
                    ]
                },
            },
        ],
    )

    assert plan["trials"][0]["objective"]["checks"][0]["predicate"]["type"] == "overlap"
    assert plan["trials"][1]["objective"]["checks"][0]["event"] == "hazard"


def test_negative_trial_passes_only_after_a_meaningful_bounded_search() -> None:
    spec = _agent_spec()
    plan = _negative_wall_plan(spec)
    report = build_behavior_trial_report(
        env_id=spec.id,
        plan=plan,
        env_spec_hash=spec_hash(spec),
        run_id="run-0002-test",
        model="test-model",
        results=[
            {
                "trial_id": "wall_bypass",
                "expected_outcome": "should_not_succeed",
                "status": "not_observed",
                "objective": {"satisfied": False},
                "constraints": {"satisfied": True},
                "termination_reason": "child_stopped",
                "actions": [
                    {
                        "action": "controller",
                        "forward": 1.0,
                        "frames_advanced": 240,
                    }
                ],
            }
        ],
    )

    assert report["status"] == "passed"
    assert report["passed"] is True
    assert report["summary"]["passed"] == 1
    assert report["results"][0]["search_evidence"] == {
        "valid": True,
        "reason": "active_search_completed",
        "active_steps": 240,
        "active_segments": 1,
        "required_active_steps": 240,
        "max_total_steps": 4800,
        "termination_reason": "child_stopped",
        "terminal_prevented_counterexample": False,
        "constraints_verified": True,
        "constraints_satisfied": True,
    }


def test_negative_trial_is_inconclusive_when_the_child_stops_without_searching() -> None:
    spec = _agent_spec()
    plan = _negative_wall_plan(spec)
    report = build_behavior_trial_report(
        env_id=spec.id,
        plan=plan,
        env_spec_hash=spec_hash(spec),
        run_id="run-0003-test",
        model="test-model",
        results=[
            {
                "trial_id": "wall_bypass",
                "expected_outcome": "should_not_succeed",
                "status": "not_observed",
                "objective": {"satisfied": False},
                "constraints": {"satisfied": True},
                "termination_reason": "child_stopped",
                "actions": [],
            }
        ],
    )

    assert report["status"] == "needs_attention"
    assert report["passed"] is False
    assert report["summary"]["inconclusive"] == 1
    assert report["results"][0]["search_evidence"]["reason"] == "insufficient_active_search"


def test_negative_trial_cannot_pass_after_violating_attempt_constraints() -> None:
    spec = _agent_spec()
    plan = _negative_wall_plan(spec)
    report = build_behavior_trial_report(
        env_id=spec.id,
        plan=plan,
        env_spec_hash=spec_hash(spec),
        run_id="run-0004-test",
        model="test-model",
        results=[
            {
                "trial_id": "wall_bypass",
                "expected_outcome": "should_not_succeed",
                "status": "not_observed",
                "objective": {"satisfied": False},
                "constraints": {"satisfied": False},
                "termination_reason": "child_stopped",
                "actions": [
                    {
                        "action": "controller",
                        "forward": 1.0,
                        "jump": True,
                        "frames_advanced": 240,
                    }
                ],
            }
        ],
    )

    assert report["status"] == "needs_attention"
    assert report["results"][0]["status"] == "inconclusive"
    assert report["results"][0]["search_evidence"]["reason"] == "attempt_constraints_violated"


def test_negative_trial_requires_evidence_for_defined_attempt_constraints() -> None:
    spec = _agent_spec()
    plan = _negative_wall_plan(spec)
    report = build_behavior_trial_report(
        env_id=spec.id,
        plan=plan,
        env_spec_hash=spec_hash(spec),
        run_id="run-0005-test",
        model="test-model",
        results=[
            {
                "trial_id": "wall_bypass",
                "expected_outcome": "should_not_succeed",
                "status": "not_observed",
                "objective": {"satisfied": False},
                "termination_reason": "child_stopped",
                "actions": [
                    {"action": "controller", "forward": 1.0, "frames_advanced": 240}
                ],
            }
        ],
    )

    assert report["status"] == "needs_attention"
    result = report["results"][0]
    assert result["status"] == "inconclusive"
    assert result["search_evidence"]["reason"] == "attempt_constraints_unverified"
    assert result["repair_hints"] == [
        "The result does not verify its attempt rules; rerun the test to collect complete evidence."
    ]


def test_summary_exposes_every_active_behavior_run(tmp_path) -> None:
    spec = _agent_spec()
    scene_dir = tmp_path / spec.id
    scene_dir.mkdir()
    (scene_dir / "env_spec_3d.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    write_behavior_trial_plan(scene_dir, _negative_wall_plan(spec))
    for index, trial_id in enumerate(("route", "counterexample"), start=1):
        run_dir = scene_dir / "behavior_trials" / f"run-{index}"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": f"run-{index}",
                    "status": "running",
                    "pid": os.getpid(),
                    "created_at": f"2026-01-01T00:00:0{index}+00:00",
                    "trial_ids": [trial_id],
                }
            ),
            encoding="utf-8",
        )

    active = active_behavior_runs(scene_dir)
    summary = behavior_trial_summary(scene_dir, current_spec=spec)

    assert [run["run_id"] for run in active] == ["run-1", "run-2"]
    assert [run["run_id"] for run in summary["active_runs"]] == ["run-1", "run-2"]
    assert summary["active_run"]["run_id"] == "run-2"
    assert summary["active_run_count"] == 2


def test_legacy_behavior_reports_normalize_without_rewriting_evidence(tmp_path) -> None:
    spec = _agent_spec()
    scene_dir = tmp_path / spec.id
    scene_dir.mkdir()
    (scene_dir / "env_spec_3d.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    plan = _negative_wall_plan(spec)
    write_behavior_trial_plan(scene_dir, plan)
    raw_report = {
        "status": "not_observed",
        "summary": {"total": 1, "not_observed": 1},
        "results": [
            {
                "trial_id": "wall_bypass",
                "expected_outcome": "should_not_succeed",
                "status": "not_observed",
                "objective": {"satisfied": False},
                "constraints": {"satisfied": True},
                "termination_reason": "child_stopped",
                "actions": [
                    {"action": "controller", "right": 1.0, "frames_advanced": 240}
                ],
            }
        ],
    }
    report_path = scene_dir / "env_behavior_trials_report.json"
    original = json.dumps(raw_report, indent=2)
    report_path.write_text(original, encoding="utf-8")
    history_dir = scene_dir / "behavior_trials" / "legacy-run"
    history_dir.mkdir(parents=True)
    (history_dir / "manifest.json").write_text(
        json.dumps({"run_id": "legacy-run", "status": "not_observed", "trial_ids": ["wall_bypass"]}),
        encoding="utf-8",
    )
    (history_dir / "report.json").write_text(original, encoding="utf-8")

    loaded = load_behavior_trial_report(scene_dir)
    history = behavior_trial_history(scene_dir)

    assert loaded["status"] == "passed"
    assert loaded["summary"]["passed"] == 1
    assert history[0]["status"] == "passed"
    assert history[0]["summary"]["passed"] == 1
    assert report_path.read_text(encoding="utf-8") == original


def _fake_persist(*, spec, scene_dir, trace_records, render=True):
    del trace_records, render
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "env_spec_3d.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return {"paths": {"env_spec": str(scene_dir / "env_spec_3d.json")}, "metadata": {"env_id": spec.id}}
