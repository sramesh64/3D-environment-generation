from __future__ import annotations

from pathlib import Path

from environment_generation.behavior_evidence import (
    select_behavior_evidence,
    significant_frame_events,
)


def _path(index: int) -> Path:
    return Path(f"frame_{index:04d}.png")


def _check(check_id: str, passed: bool, description: str = "") -> dict:
    return {
        "id": check_id,
        "type": "zone_entry",
        "description": description,
        "passed": passed,
    }


def _record(
    step: int,
    *,
    attempt: int = 1,
    checks: list[dict] | None = None,
    constraints: list[dict] | None = None,
    events: list[dict] | None = None,
    target: str = "",
    distance: float | None = None,
    terminal: bool = False,
) -> dict:
    objective_checks = checks or []
    objective_satisfied = bool(objective_checks) and all(
        check.get("passed") for check in objective_checks
    )
    navigation = {"available": False, "primary_target_id": None}
    if target and distance is not None:
        navigation = {
            "available": True,
            "primary_target_id": target,
            "distance_xy": distance,
        }
    focus_check_id = next(
        (str(check.get("id")) for check in objective_checks if not check.get("passed")),
        "",
    )
    return {
        "total_step": step,
        "attempt": attempt,
        "events": events or [],
        "objective_focus": {
            "check_id": focus_check_id,
            "target_ids": [target] if target else [],
        },
        "navigation": navigation,
        "attempt_objective": {
            "satisfied": objective_satisfied,
            "checks": objective_checks,
        },
        "attempt_constraints": {
            "satisfied": all(check.get("passed") for check in constraints or []),
            "checks": constraints or [],
        },
        "terminal": terminal,
        "termination_reason": "objective_satisfied" if terminal and objective_satisfied else "",
    }


def test_ordered_trial_prefers_typed_milestones_over_support_contact_noise() -> None:
    steps = [0, 36, 54, 72, 432, 492, 621, 780, 1034, 1040]
    paths = [_path(index) for index in range(len(steps))]
    records = {}
    for index, (path, step) in enumerate(zip(paths, steps)):
        open_passed = step >= 492
        goal_passed = step >= 1040
        events = []
        if step in {36, 54, 72}:
            events = [
                {
                    "type": "contact_started",
                    "object_id": "ground",
                    "semantic_type": "ground",
                    "routine": True,
                }
            ]
        elif step == 432:
            events = [
                {
                    "type": "zone_entered",
                    "object_id": "switch",
                    "semantic_type": "floor_switch",
                    "objective_relevant": True,
                },
                {
                    "type": "mechanism_activated",
                    "object_id": "switch_opens_gate",
                },
            ]
        elif step == 1040:
            events = [
                {
                    "type": "zone_entered",
                    "object_id": "goal",
                    "semantic_type": "goal",
                    "objective_relevant": True,
                }
            ]
        target = "switch" if not open_passed else "goal"
        distance = max(0.0, (492 - step) / 40) if target == "switch" else max(0.0, (1040 - step) / 80)
        records[path.name] = _record(
            step,
            checks=[
                _check("open_gate", open_passed, "Open the gate"),
                _check("enter_goal", goal_passed, "Enter the goal"),
            ],
            events=events,
            target=target,
            distance=distance,
            terminal=step == 1040,
        )

    selected = select_behavior_evidence(paths, records, limit=6)
    selected_steps = [item.record["total_step"] for item in selected]

    assert selected_steps[0] == 0
    assert selected_steps[-1] == 1040
    assert 432 in selected_steps
    assert 492 in selected_steps
    assert 1034 in selected_steps
    assert not {36, 54, 72}.intersection(selected_steps)
    assert any(item.label == "Completed: Open the gate" for item in selected)
    assert selected[-1].label == "Completed: Enter the goal"
    completed_gate = next(item for item in selected if item.label == "Completed: Open the gate")
    assert completed_gate.focus_ids == ("switch",)
    assert selected[-1].focus_ids == ("goal",)


def test_objective_contact_is_signal_while_routine_ground_contact_is_not() -> None:
    routine = {
        "type": "contact_started",
        "object_id": "ground",
        "semantic_type": "ground",
        "routine": True,
    }
    target_contact = {
        "type": "contact_started",
        "object_id": "crate",
        "semantic_type": "pushable_box",
        "objective_relevant": True,
    }

    assert significant_frame_events({"events": [routine]}) == []
    assert significant_frame_events({"events": [routine, target_contact]}) == [target_contact]


def test_milestone_labels_expand_generic_typed_check_descriptions() -> None:
    paths = [_path(0), _path(1)]
    generic_check = {
        "id": "robot_above_wall",
        "description": "relation",
        "passed": False,
        "predicate": {
            "type": "relation",
            "subject": {"id": "blue_robot"},
            "target": {"id": "middle_wall"},
            "relation": "above",
        },
    }
    records = {
        paths[0].name: _record(0, checks=[generic_check]),
        paths[1].name: _record(10, checks=[{**generic_check, "passed": True}], terminal=True),
    }

    selected = select_behavior_evidence(paths, records, limit=2)

    assert selected[-1].label == "Completed: Blue robot is above middle wall"


def test_inconclusive_trial_preserves_attempts_and_closest_approach() -> None:
    paths = [_path(index) for index in range(8)]
    records = {
        path.name: _record(
            index * 20,
            attempt=1 if index < 4 else 2,
            checks=[_check("reach_platform", False, "Reach the platform")],
            target="reach_platform",
            distance=abs(5 - index) + 0.25,
        )
        for index, path in enumerate(paths)
    }

    selected = select_behavior_evidence(paths, records, limit=5)

    assert paths[0] in [item.path for item in selected]
    assert paths[4] in [item.path for item in selected]
    assert paths[5] in [item.path for item in selected]
    assert paths[-1] in [item.path for item in selected]
    assert any(item.kind == "attempt" and item.label == "Attempt 2 started" for item in selected)
    assert any(item.kind == "approach" for item in selected)


def test_constraint_regression_is_selected_as_a_milestone() -> None:
    paths = [_path(index) for index in range(5)]
    records = {}
    for index, path in enumerate(paths):
        no_jump = {
            "id": "no_jump",
            "type": "jump_count",
            "description": "Do not jump",
            "passed": index < 3,
        }
        records[path.name] = _record(
            index * 10,
            checks=[_check("far_side", False)],
            constraints=[no_jump],
        )

    selected = select_behavior_evidence(paths, records, limit=4)

    assert any(
        item.record["total_step"] == 30
        and item.label == "Constraint violated: Do not jump"
        for item in selected
    )
