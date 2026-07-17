from __future__ import annotations

import pytest

from environment_generation.artifacts import persist_artifacts
from environment_generation.behavior_preflight import prepare_behavior_trial
from environment_generation.behavior_trial import BehaviorTrialSession
from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.env_behavior_trials import normalize_behavior_trial_plan


def _tower_scene(tmp_path, *, dynamic: bool):
    builder = EnvSpec3DBuilder("preflight_tower", description="a distant five-box tower")
    builder.add_ground_plane(16, 12)
    add_box = builder.add_pushable_box if dynamic else builder.add_static_box
    add_box(0, 0, z=0, id="stack_1")
    builder.add_agent_spawn(-7, -5, z=0.05, id="agent")
    for level in range(1, 5):
        add_box(0, 0, z=level, id=f"stack_{level + 1}")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    return scene_dir, spec


def _tower_trial(spec, *, allow_target_motion: bool = False, constraints=None):
    raw = {
        "id": "reach_top",
        "instruction": "Try to stand on the top box.",
        "expected_outcome": "should_not_succeed",
        "max_steps": 160,
        "max_resets": 1,
        "allow_target_motion": allow_target_motion,
        "objective": {
            "checks": [
                {
                    "id": "on_top",
                    "type": "agent_relation",
                    "target": {"id": "stack_5"},
                    "relation": "on_surface",
                }
            ]
        },
    }
    if constraints is not None:
        raw["constraints"] = constraints
    return normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="the central tower is too high to climb",
        operation_count=8,
        draft_spec=spec,
        trials=[raw],
    )["trials"][0]


def test_preflight_expands_each_attempt_and_aims_at_a_far_target(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, spec = _tower_scene(tmp_path, dynamic=False)

    runtime, preflight = prepare_behavior_trial(scene_dir=scene_dir, trial=_tower_trial(spec))

    navigation = preflight["navigation"]
    assert preflight["status"] == "ready"
    assert preflight["configured_steps_per_attempt"] == 160
    assert preflight["effective_steps_per_attempt"] > 160
    assert runtime["max_total_steps"] == preflight["effective_steps_per_attempt"] * 2
    assert navigation["primary_target_id"] == "stack_5"
    assert navigation["initial_distance_xy"] > 7
    assert navigation["initial_camera_azimuth"] == pytest.approx(-125.54, abs=0.1)
    assert navigation["initial_camera_elevation"] == 15

    session = BehaviorTrialSession(scene_dir=scene_dir, trial=runtime, render_frames=False)
    initial = session.start()
    moved = session.act(forward=1, frames=60)
    assert abs(initial["navigation"]["progress_xy"]) < 0.05
    assert abs(initial["navigation"]["heading_error_degrees"]) < 0.1
    assert initial["navigation"]["relative"]["forward"] > 0
    assert abs(initial["navigation"]["relative"]["right"]) < 0.05
    assert moved["navigation"]["progress_xy"] > 1
    session.close()


def test_preflight_rejects_collapsing_target_but_preserves_intentional_motion(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, spec = _tower_scene(tmp_path, dynamic=True)

    _runtime, preflight = prepare_behavior_trial(scene_dir=scene_dir, trial=_tower_trial(spec))
    _moving_runtime, moving_preflight = prepare_behavior_trial(
        scene_dir=scene_dir,
        trial=_tower_trial(spec, allow_target_motion=True),
    )

    assert preflight["status"] == "invalid_setup"
    assert preflight["stability"]["targets"][0]["max_vertical_drop"] > 1
    assert any("moves or collapses" in issue for issue in preflight["issues"])
    assert moving_preflight["status"] == "ready"
    assert any("intentional" in warning for warning in moving_preflight["warnings"])


def test_preflight_rejects_a_constraint_that_makes_counterexample_impossible(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, spec = _tower_scene(tmp_path, dynamic=False)
    trial = _tower_trial(
        spec,
        constraints={
            "checks": [
                {"id": "height_ceiling", "type": "agent_height_gain", "max_gain": 2.8}
            ]
        },
    )

    _runtime, preflight = prepare_behavior_trial(scene_dir=scene_dir, trial=trial)

    assert preflight["status"] == "invalid_setup"
    assert any("impossible by definition" in issue for issue in preflight["issues"])


def test_navigation_prefers_final_landing_surface_over_avoidance_zone(tmp_path) -> None:
    pytest.importorskip("mujoco")
    builder = EnvSpec3DBuilder("preflight_gap", description="jump to the far platform")
    builder.add_platform(-2.25, 0, 0, width=4, depth=4, id="left")
    builder.add_platform(2.25, 0, 0, width=4, depth=4, id="right")
    builder.add_agent_spawn(-2.5, 0, z=0.3, id="agent")
    builder.add_hazard_zone(0, 0, width=0.5, depth=3, id="gap")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    trial = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="jump across the gap without touching it",
        operation_count=4,
        draft_spec=spec,
        trials=[
            {
                "instruction": "Jump across and land on the right platform.",
                "objective": {
                    "checks": [
                        {
                            "type": "agent_relation",
                            "target": {"id": "gap"},
                            "relation": "right_of",
                        },
                        {"type": "zone_entry", "selector": {"id": "gap"}, "max_count": 0},
                        {
                            "type": "agent_relation",
                            "target": {"id": "right"},
                            "relation": "on_surface",
                            "when": "final",
                        },
                    ]
                },
            }
        ],
    )["trials"][0]

    _runtime, preflight = prepare_behavior_trial(scene_dir=scene_dir, trial=trial)

    assert preflight["navigation"]["primary_target_id"] == "right"


def test_preflight_accepts_hazard_entry_as_an_explicit_semantic_objective(tmp_path) -> None:
    pytest.importorskip("mujoco")
    builder = EnvSpec3DBuilder("legacy_hazard_trial", description="legacy ambiguous hazard trial")
    builder.add_ground_plane(10, 6)
    builder.add_agent_spawn(-2, 0, id="agent")
    builder.add_hazard_zone(0, 0, id="hazard")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    trial = {
        "id": "ambiguous_hazard",
        "instruction": "Enter the hazard without being reset.",
        "expected_outcome": "should_not_succeed",
        "max_steps": 300,
        "max_resets": 1,
        "objective": {
            "mode": "all",
            "checks": [
                {
                    "id": "entered_hazard",
                    "type": "zone_entry",
                    "selector": {"id": "hazard"},
                    "min_count": 1,
                }
            ],
            "ordered_check_ids": ["entered_hazard"],
        },
        "constraints": {"mode": "all", "checks": [], "ordered_check_ids": []},
    }

    _runtime, preflight = prepare_behavior_trial(scene_dir=scene_dir, trial=trial)

    assert preflight["status"] == "ready"
    assert preflight["issues"] == []


def test_preflight_budgets_actor_to_subject_and_subject_to_target_travel(tmp_path) -> None:
    pytest.importorskip("mujoco")
    builder = EnvSpec3DBuilder("delivery_preflight", description="a long delivery")
    builder.add_ground_plane(16, 8)
    builder.add_agent_spawn(-6, 0, id="agent")
    builder.add_pushable_box(-1, 0, id="subject")
    builder.add_target_region(6, 0, id="target")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    trial = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="deliver the object",
        operation_count=4,
        draft_spec=spec,
        trials=[
            {
                "instruction": "Push the subject into the target.",
                "max_steps": 60,
                "objective": {
                    "checks": [
                        {
                            "id": "delivered",
                            "predicate": {
                                "type": "overlap",
                                "subject": {"id": "subject"},
                                "target": {"id": "target"},
                            },
                        }
                    ]
                },
            }
        ],
    )["trials"][0]

    runtime, preflight = prepare_behavior_trial(scene_dir=scene_dir, trial=trial)

    navigation = preflight["navigation"]
    assert navigation["primary_target_id"] == "subject"
    assert navigation["estimated_trial_travel_distance_xy"] > navigation["initial_distance_xy"]
    assert runtime["max_steps"] > 60
