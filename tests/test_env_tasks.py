from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from environment_generation.artifacts import persist_artifacts
from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.env_tasks import (
    TaskDefinitionError,
    create_task_draft,
    delete_task,
    list_tasks,
    normalize_assertion_group,
    normalize_task_definition,
    read_task,
    task_catalog_summary,
    task_scene_hash,
)
from environment_generation.schema import env_spec_to_dict
from environment_generation.task_compiler import (
    TaskCompilerRepairContext,
    build_task_compiler_codex_args,
    build_task_compiler_prompt,
)


def _spec(*, hazard: bool = False, target: bool = True) -> dict:
    builder = EnvSpec3DBuilder("task_scene", description="task scene")
    builder.add_ground_plane(width=14, depth=10)
    builder.add_agent_spawn(-4, 0, id="robot")
    builder.add_pushable_box(0, 0, id="crate")
    builder.add_goal_zone(4, 0, id="charging_pad")
    if target:
        builder.add_target_region(2, 2, id="delivery_zone")
    if hazard:
        builder.add_hazard_zone(0, 3, id="broken_ground")
    return env_spec_to_dict(builder.finalize())


def _reach_goal_tests() -> list[dict]:
    return [
        {
            "id": "reach_goal",
            "description": "Reach the charging pad.",
            "mode": "all",
            "conditions": [
                {
                    "id": "enter_goal",
                    "description": "Robot enters the goal.",
                    "temporal": "eventually",
                    "predicate": {
                        "type": "overlap",
                        "subject": {"id": "robot"},
                        "target": {"id": "charging_pad"},
                    },
                }
            ],
            "ordered_condition_ids": [],
        }
    ]


def test_assertion_conditions_generate_specific_labels_for_generic_descriptions() -> None:
    group = normalize_assertion_group(
        raw={
            "checks": [
                {
                    "id": "touch_crate",
                    "description": "contact",
                    "predicate": {
                        "type": "contact",
                        "subject": {"id": "robot"},
                        "target": {"id": "crate"},
                    },
                },
                {
                    "id": "climb_crate",
                    "description": "relation",
                    "predicate": {
                        "type": "relation",
                        "subject": {"id": "robot"},
                        "target": {"id": "crate"},
                        "relation": "above",
                    },
                },
            ]
        },
        spec=_spec(),
    )

    assert [check["description"] for check in group["checks"]] == [
        "Robot makes contact with crate.",
        "Robot is above crate.",
    ]


def test_task_definition_supports_object_delivery_and_adds_visible_system_safety_tests() -> None:
    definition = normalize_task_definition(
        env_id="task_scene",
        instruction="Move the crate into the delivery zone without touching the hazard.",
        spec=_spec(hazard=True),
        tests=[
            {
                "id": "deliver",
                "description": "Deliver and settle the crate.",
                "mode": "all",
                "conditions": [
                    {
                        "id": "crate_inside",
                        "description": "Crate finishes in the target.",
                        "temporal": "at_end",
                        "predicate": {
                            "type": "overlap",
                            "subject": {"id": "crate"},
                            "target": {"semantic_type": "target_region"},
                        },
                    },
                    {
                        "id": "crate_settled",
                        "description": "Crate remains settled.",
                        "temporal": "sustained",
                        "frames": 20,
                        "predicate": {"type": "settled", "subject": {"id": "crate"}},
                    },
                    {
                        "id": "avoid_hazard",
                        "description": "Robot never enters broken ground.",
                        "temporal": "never",
                        "predicate": {
                            "type": "overlap",
                            "subject": {"id": "robot"},
                            "target": {"id": "broken_ground"},
                        },
                    },
                ],
                "ordered_condition_ids": [],
            }
        ],
    )

    assert definition["tests"][0]["conditions"][0]["predicate"]["target"] == {
        "semantic_type": "target_region"
    }
    assert {test["id"] for test in definition["tests"]} >= {
        "deliver",
        "system_stay_in_bounds",
    }
    assert "system_avoid_hazards" not in {test["id"] for test in definition["tests"]}
    assert definition["tests"][0]["conditions"][2]["id"] == "avoid_hazard"
    assert all(test["source"] == "system" for test in definition["tests"][1:])


@pytest.mark.parametrize(
    "tests,match",
    [
        ([], "non-empty"),
        (
            [
                {
                    "id": "wait",
                    "conditions": [
                        {
                            "id": "avoid",
                            "temporal": "never",
                            "predicate": {
                                "type": "overlap",
                                "subject": "agent",
                                "target": "goal",
                            },
                        }
                    ],
                }
            ],
            "doing nothing",
        ),
        (
            [
                {
                    "id": "missing",
                    "conditions": [
                        {
                            "id": "enter",
                            "temporal": "eventually",
                            "predicate": {
                                "type": "overlap",
                                "subject": "agent",
                                "target": {"id": "does_not_exist"},
                            },
                        }
                    ],
                }
            ],
            "does not match",
        ),
        (
            [
                {
                    "id": "bad_quantifier",
                    "conditions": [
                        {
                            "id": "move",
                            "temporal": "eventually",
                            "predicate": {
                                "type": "displacement",
                                "subject": "agent",
                                "subject_quantifier": "most",
                                "min_value": 1,
                            },
                        }
                    ],
                }
            ],
            "subject_quantifier",
        ),
    ],
)
def test_task_definition_rejects_malformed_or_vacuous_tests(tests: list[dict], match: str) -> None:
    with pytest.raises(TaskDefinitionError, match=match):
        normalize_task_definition(
            env_id="task_scene",
            instruction="A task",
            spec=_spec(),
            tests=tests,
        )


def test_task_definition_rejects_agentless_scene_selector() -> None:
    builder = EnvSpec3DBuilder("static", description="static")
    builder.add_ground_plane()
    spec = env_spec_to_dict(builder.finalize())
    with pytest.raises(TaskDefinitionError, match="does not match"):
        normalize_task_definition(
            env_id="static",
            instruction="Move",
            spec=spec,
            tests=_reach_goal_tests(),
        )


@pytest.mark.parametrize(
    "predicate",
    [
        {
            "type": "displacement",
            "subject": {"id": "robot"},
            "metric": "maximum",
            "min_value": 1,
        },
        {
            "type": "axis_delta",
            "subject": {"id": "robot"},
            "axis": "z",
            "metric": "maximum",
            "min_value": 0.5,
        },
        {
            "type": "axis_delta",
            "subject": {"id": "robot"},
            "axis": "z",
            "metric": "minimum",
            "max_value": -0.5,
        },
    ],
)
def test_task_definition_rejects_trial_global_aggregates_in_ordering(
    predicate: dict,
) -> None:
    with pytest.raises(TaskDefinitionError, match="trial-global aggregate"):
        normalize_task_definition(
            env_id="task_scene",
            instruction="Move, then touch the crate.",
            spec=_spec(),
            tests=[
                {
                    "id": "ordered_motion",
                    "conditions": [
                        {
                            "id": "aggregate_motion",
                            "temporal": "eventually",
                            "predicate": predicate,
                        },
                        {
                            "id": "touch_crate",
                            "temporal": "eventually",
                            "predicate": {
                                "type": "contact",
                                "subject": {"id": "robot"},
                                "target": {"id": "crate"},
                            },
                        },
                    ],
                    "ordered_condition_ids": ["aggregate_motion", "touch_crate"],
                }
            ],
        )


def test_task_definition_keeps_aggregate_metrics_as_unordered_evidence() -> None:
    definition = normalize_task_definition(
        env_id="task_scene",
        instruction="Gain height and touch the crate.",
        spec=_spec(),
        tests=[
            {
                "id": "height_and_contact",
                "conditions": [
                    {
                        "id": "height_gain",
                        "temporal": "eventually",
                        "predicate": {
                            "type": "axis_delta",
                            "subject": {"id": "robot"},
                            "axis": "z",
                            "metric": "maximum",
                            "min_value": 0.5,
                        },
                    },
                    {
                        "id": "touch_crate",
                        "temporal": "eventually",
                        "predicate": {
                            "type": "contact",
                            "subject": {"id": "robot"},
                            "target": {"id": "crate"},
                        },
                    },
                ],
                "ordered_condition_ids": ["touch_crate"],
            }
        ],
    )

    assert definition["tests"][0]["ordered_condition_ids"] == ["touch_crate"]
    assert definition["tests"][0]["conditions"][0]["predicate"]["metric"] == "maximum"


def test_task_scene_hash_ignores_visual_only_fields_but_tracks_physics() -> None:
    spec = _spec()
    visual_edit = copy.deepcopy(spec)
    visual_edit["theme"] = "another_theme"
    visual_edit["description"] = "new copy"
    visual_edit["objects"][0]["color"] = "#FFFFFF"
    visual_edit["objects"][0]["appearance"] = {
        "asset_id": "courtyard_ground",
        "variant": "pavers_grass",
    }
    physics_edit = copy.deepcopy(spec)
    physics_edit["objects"][1]["position"][0] += 1

    assert task_scene_hash(spec) == task_scene_hash(visual_edit)
    assert task_scene_hash(spec) != task_scene_hash(physics_edit)


def test_task_drafts_persist_list_stale_and_delete(tmp_path: Path) -> None:
    spec = _spec()
    scene_dir = tmp_path / "task_scene"
    persist_artifacts(
        spec=EnvSpec3DBuilder.from_spec_dict(spec).finalize(),
        scene_dir=scene_dir,
        trace_records=[],
        render=False,
    )
    task = create_task_draft(
        scene_dir=scene_dir,
        env_id="task_scene",
        instruction="Reach the charging pad.",
        compiler_output={"summary": "Reach the pad", "max_steps": 1000, "tests": _reach_goal_tests()},
    )

    assert task["status"] == "pending_oracle"
    assert read_task(scene_dir, task["task_id"])["effective_status"] == "pending_oracle"
    assert task_catalog_summary(scene_dir, current_spec=spec)["total"] == 1

    changed = copy.deepcopy(spec)
    changed["objects"][1]["position"][0] += 1
    assert list_tasks(scene_dir, current_spec=changed)[0]["effective_status"] == "stale"

    delete_task(scene_dir, task["task_id"])
    assert list_tasks(scene_dir, current_spec=spec) == []


def test_task_compiler_prompt_and_command_are_read_only_and_schema_bound(tmp_path: Path) -> None:
    prompt = build_task_compiler_prompt(
        instruction="Move the crate into the delivery zone.",
        spec=_spec(),
    )
    args = build_task_compiler_codex_args(
        prompt=prompt,
        output_path=tmp_path / "output.json",
        model="test-model",
        cwd=tmp_path,
    )

    assert '"id": "crate"' in prompt
    assert '"id": "delivery_zone"' in prompt
    assert "Never emit Python" in prompt
    assert "subject_quantifier" in prompt
    assert "passive semantic sensors" in prompt
    assert "Do not add hazard avoidance merely because a hazard exists" in prompt
    assert "read-only" in args
    assert "--ephemeral" in args
    assert "--output-schema" in args
    assert "test-model" in args


def test_task_compiler_repair_prompt_includes_rejected_output_and_validation_feedback() -> None:
    rejected = {
        "task_id": "stack_then_gate",
        "tests": [
            {
                "id": "sequence",
                "ordered_condition_ids": ["crate_stacked"],
                "conditions": [
                    {
                        "id": "crate_stacked",
                        "temporal": "at_end",
                        "predicate": {"type": "relation", "relation": "on_surface"},
                    }
                ],
            }
        ],
    }
    prompt = build_task_compiler_prompt(
        instruction="Stack the crate, then open the gate.",
        spec=_spec(),
        repair_context=TaskCompilerRepairContext(
            attempt=1,
            rejected_output=rejected,
            validation_errors=("ordered condition 'crate_stacked' cannot use at_end",),
        ),
    )

    assert "repair attempt 1 of 2" in prompt
    assert "semantic_validation_error" in prompt
    assert "ordered condition 'crate_stacked' cannot use at_end" in prompt
    assert '"task_id": "stack_then_gate"' in prompt
    assert "Preserve every requirement" in prompt
    assert "Do not remove an\nordering requirement" in prompt
    assert "complete corrected replacement" in prompt


def test_task_compiler_schema_is_valid_json() -> None:
    path = Path(__file__).parents[1] / "environment_generation" / "task_compiler_output_schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert schema["properties"]["tests"]["maxItems"] == 16
    assert schema["$defs"]["predicate"]["properties"]["subject_quantifier"]["enum"] == [
        "any",
        "all",
        None,
    ]
