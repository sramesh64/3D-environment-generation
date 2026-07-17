from __future__ import annotations

import math

from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.env_tasks import normalize_task_definition
from environment_generation.env_verification import scene_objects
from environment_generation.schema import env_spec_to_dict
from environment_generation import trajectory_assertions
from environment_generation.trajectory_assertions import (
    AssertionSatisfactionMonitor,
    evaluate_assertion_group,
    evaluate_assertion_tests as evaluate_task_tests,
)


def _spec() -> dict:
    builder = EnvSpec3DBuilder("trajectory", description="trajectory")
    builder.add_ground_plane(width=14, depth=10)
    builder.add_agent_spawn(-4, 0, id="robot")
    builder.add_pushable_box(-1, 0, id="crate")
    builder.add_target_region(3, 0, width=2, depth=2, id="drop_zone")
    builder.add_goal_zone(4, 3, id="goal")
    builder.add_hazard_zone(0, 3, id="hazard")
    return env_spec_to_dict(builder.finalize())


def _frame(step: int, *, robot=(-4, 0, 0.55), crate=(-1, 0, 0.5), contacts=(), jumps=0, events=()) -> dict:
    positions = {
        "ground": [0, 0, -0.1],
        "robot": list(robot),
        "crate": list(crate),
        "drop_zone": [3, 0, 0.1],
        "goal": [4, 3, 0.6],
        "hazard": [0, 3, 0.075],
    }
    rotations = {key: [1, 0, 0, 0, 1, 0, 0, 0, 1] for key in positions}
    velocities = {
        key: {"linear": [0, 0, 0], "angular": [0, 0, 0]}
        for key in positions
    }
    return {
        "step": step,
        "simulation_time": step * 0.01,
        "positions": positions,
        "rotations": rotations,
        "velocities": velocities,
        "contacts": [list(pair) for pair in contacts],
        "mechanisms": [],
        "jump_count": jumps,
        "reset_count": 0,
        "grounded": True,
        "terminal_events": list(events),
        "play_bounds": [14, 10, 8],
    }


def _task(tests: list[dict]) -> tuple[dict, list]:
    spec = _spec()
    task = normalize_task_definition(
        env_id="trajectory",
        instruction="Trajectory objective",
        tests=tests,
        spec=spec,
    )
    return task, scene_objects(spec)


def test_object_delivery_can_pass_without_agent_entering_goal() -> None:
    task, objects = _task(
        [
            {
                "id": "delivery",
                "description": "Deliver the crate but do not finish the level.",
                "mode": "all",
                "conditions": [
                    {
                        "id": "crate_in_zone",
                        "temporal": "at_end",
                        "predicate": {
                            "type": "overlap",
                            "subject": {"id": "crate"},
                            "target": {"id": "drop_zone"},
                        },
                    },
                    {
                        "id": "no_goal",
                        "temporal": "never",
                        "predicate": {
                            "type": "overlap",
                            "subject": {"id": "robot"},
                            "target": {"id": "goal"},
                        },
                    },
                ],
            }
        ]
    )
    frames = [_frame(0), _frame(1, crate=(3, 0, 0.5))]

    report = evaluate_task_tests(task=task, frames=frames, objects=objects)

    assert report["passed"] is True
    assert report["tests"][0]["conditions"][0]["witness"]["overlapping_pairs"] == [
        ["crate", "drop_zone"]
    ]


def test_overlap_assertion_uses_runtime_oriented_shapes_not_inflated_bounds() -> None:
    builder = EnvSpec3DBuilder("oriented_overlap", description="oriented overlap")
    builder.add_ground_plane(12, 12)
    builder.add_static_box(
        0,
        0,
        width=8,
        depth=0.4,
        height=1,
        yaw=math.pi / 4,
        id="target",
    )
    builder.add_static_box(
        2.5,
        -2.5,
        width=0.6,
        depth=0.6,
        height=0.6,
        id="subject",
    )
    objects = scene_objects(builder.finalize())
    by_id = {obj.id: obj for obj in objects}
    subject = by_id["subject"]
    target = by_id["target"]
    assert subject.bounds["x1"] <= target.bounds["x2"]
    assert subject.bounds["y2"] >= target.bounds["y1"]

    positions = {obj.id: list(obj.position) for obj in objects}
    rotations = {
        obj.id: [
            math.cos(obj.yaw),
            -math.sin(obj.yaw),
            0,
            math.sin(obj.yaw),
            math.cos(obj.yaw),
            0,
            0,
            0,
            1,
        ]
        for obj in objects
    }
    frame = {
        "step": 0,
        "simulation_time": 0.0,
        "positions": positions,
        "rotations": rotations,
        "velocities": {
            obj.id: {"linear": [0, 0, 0], "angular": [0, 0, 0]}
            for obj in objects
        },
        "contacts": [],
        "mechanisms": [],
        "jump_count": 0,
        "reset_count": 0,
        "grounded": True,
        "terminal_events": [],
        "play_bounds": [12, 12, 8],
    }
    result = evaluate_assertion_group(
        group={
            "mode": "all",
            "conditions": [
                {
                    "id": "overlap",
                    "description": "The selected shapes overlap.",
                    "temporal": "eventually",
                    "predicate": {
                        "type": "overlap",
                        "subject": {"id": "subject"},
                        "target": {"id": "target"},
                    },
                }
            ],
        },
        frames=[frame],
        objects=objects,
        final=True,
    )

    assert result["satisfied"] is False
    assert result["checks"][0]["witness"]["overlapping_pairs"] == []


def test_explicit_hazard_invariant_fails_even_when_positive_objective_passes() -> None:
    task, objects = _task(
        [
            {
                "id": "reach",
                "conditions": [
                    {
                        "id": "move",
                        "temporal": "eventually",
                        "predicate": {
                            "type": "displacement",
                            "subject": {"id": "robot"},
                            "min_value": 1,
                        },
                    }
                ],
            },
            {
                "id": "avoid_hazard",
                "conditions": [
                    {
                        "id": "never_enter_hazard",
                        "temporal": "never",
                        "predicate": {
                            "type": "overlap",
                            "subject": {"id": "robot"},
                            "target": {"id": "hazard"},
                        },
                    }
                ],
            },
        ]
    )
    frames = [_frame(0), _frame(1, robot=(0, 3, 0.55), events=("hazard",))]

    report = evaluate_task_tests(task=task, frames=frames, objects=objects)

    assert report["passed"] is False
    safety = next(test for test in report["tests"] if test["id"] == "avoid_hazard")
    assert safety["conditions"][0]["passed"] is False


def test_temporal_count_sustained_and_ordering_have_exact_witness_steps() -> None:
    task, objects = _task(
        [
            {
                "id": "ordered",
                "mode": "all",
                "conditions": [
                    {
                        "id": "jump_once",
                        "temporal": "count",
                        "min_count": 1,
                        "predicate": {"type": "jump_count", "min_value": 1},
                    },
                    {
                        "id": "touch_crate",
                        "temporal": "sustained",
                        "frames": 2,
                        "predicate": {
                            "type": "contact",
                            "subject": {"id": "robot"},
                            "target": {"id": "crate"},
                        },
                    },
                ],
                "ordered_condition_ids": ["jump_once", "touch_crate"],
            }
        ]
    )
    frames = [
        _frame(0),
        _frame(1, jumps=1),
        _frame(2, jumps=1, contacts=(("robot", "crate"),)),
        _frame(3, jumps=1, contacts=(("robot", "crate"),)),
    ]

    report = evaluate_task_tests(task=task, frames=frames, objects=objects)

    result = report["tests"][0]
    assert result["passed"] is True
    assert result["ordered_steps"] == [1, 3]
    assert result["conditions"][1]["metrics"]["longest_true_streak"] == 2


def test_ordering_selects_a_later_repeated_witness_instead_of_only_the_first() -> None:
    objects = scene_objects(_spec())
    group = {
        "mode": "all",
        "ordered_condition_ids": ["gate_open", "touch_crate", "goal_event"],
        "conditions": [
            {
                "id": "gate_open",
                "description": "The mechanism opens.",
                "temporal": "eventually",
                "predicate": {
                    "type": "mechanism_state",
                    "mechanism_id": "gate_link",
                    "state": "open",
                    "min_progress": 0.9,
                },
            },
            {
                "id": "touch_crate",
                "description": "The robot contacts the crate after the mechanism opens.",
                "temporal": "eventually",
                "predicate": {
                    "type": "contact",
                    "subject": {"id": "robot"},
                    "target": {"id": "crate"},
                },
            },
            {
                "id": "goal_event",
                "description": "The goal event occurs last.",
                "temporal": "eventually",
                "predicate": {"type": "terminal_event", "event": "goal"},
            },
        ],
    }

    def ordered_frame(step: int, *, gate_open: bool = False, contact: bool = False, goal: bool = False):
        frame = _frame(
            step,
            contacts=(("robot", "crate"),) if contact else (),
            events=("goal",) if goal else (),
        )
        frame["mechanisms"] = [
            {
                "id": "gate_link",
                "active": gate_open,
                "progress": 1.0 if gate_open else 0.0,
            }
        ]
        return frame

    no_later_contact = [
        ordered_frame(0),
        ordered_frame(1, contact=True),
        ordered_frame(2),
        ordered_frame(3, gate_open=True),
        ordered_frame(4, gate_open=True),
        ordered_frame(5, gate_open=True, goal=True),
    ]
    failed = evaluate_assertion_group(
        group=group,
        frames=no_later_contact,
        objects=objects,
    )

    assert failed["satisfied"] is False
    assert failed["ordered_steps"] == [3, None, None]

    valid_sequence = list(no_later_contact)
    valid_sequence[4] = ordered_frame(4, gate_open=True, contact=True)
    passed = evaluate_assertion_group(
        group=group,
        frames=valid_sequence,
        objects=objects,
    )
    monitor = AssertionSatisfactionMonitor(
        group=group,
        objects=objects,
        initial_frame=valid_sequence[0],
    )
    for frame in valid_sequence[1:]:
        monitor.update(frame)

    assert passed["satisfied"] is True
    assert passed["ordered_steps"] == [3, 4, 5]
    assert monitor.satisfied is True


def test_final_and_maximum_displacement_have_distinct_semantics() -> None:
    task, objects = _task(
        [
            {
                "id": "movement",
                "mode": "all",
                "conditions": [
                    {
                        "id": "moved_away",
                        "temporal": "eventually",
                        "predicate": {
                            "type": "displacement",
                            "subject": {"id": "crate"},
                            "metric": "maximum",
                            "min_value": 2,
                        },
                    },
                    {
                        "id": "returned",
                        "temporal": "at_end",
                        "predicate": {
                            "type": "displacement",
                            "subject": {"id": "crate"},
                            "metric": "final",
                            "max_value": 0.1,
                        },
                    },
                ],
            }
        ]
    )
    frames = [_frame(0), _frame(1, crate=(2, 0, 0.5)), _frame(2)]

    report = evaluate_task_tests(task=task, frames=frames, objects=objects)

    assert report["tests"][0]["passed"] is True


def test_maximum_displacement_scans_each_trajectory_frame_once(monkeypatch) -> None:
    task, objects = _task(
        [
            {
                "id": "movement",
                "conditions": [
                    {
                        "id": "moved_away",
                        "temporal": "eventually",
                        "predicate": {
                            "type": "displacement",
                            "subject": {"id": "crate"},
                            "metric": "maximum",
                            "min_value": 2,
                        },
                    }
                ],
            }
        ]
    )
    frames = [_frame(step, crate=(-1 + step * 0.1, 0, 0.5)) for step in range(40)]
    original_distance = trajectory_assertions._distance
    calls = 0

    def counted_distance(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_distance(*args, **kwargs)

    monkeypatch.setattr(trajectory_assertions, "_distance", counted_distance)

    report = evaluate_task_tests(task=task, frames=frames, objects=objects)

    assert report["tests"][0]["passed"] is True
    assert calls == len(frames)


def test_incremental_satisfaction_matches_authoritative_batch_evaluation() -> None:
    objects = scene_objects(_spec())
    group = {
        "mode": "all",
        "ordered_condition_ids": ["moved", "touched"],
        "conditions": [
            {
                "id": "moved",
                "description": "Move away from the spawn.",
                "temporal": "eventually",
                "predicate": {
                    "type": "displacement",
                    "subject": {"id": "robot"},
                    "subject_quantifier": "any",
                    "metric": "maximum",
                    "space": "xy",
                    "min_value": 1,
                },
            },
            {
                "id": "touched",
                "description": "Maintain contact with the crate.",
                "temporal": "sustained",
                "frames": 2,
                "predicate": {
                    "type": "contact",
                    "subject": {"id": "robot"},
                    "subject_quantifier": "any",
                    "target": {"id": "crate"},
                },
            },
            {
                "id": "safe",
                "description": "Never enter a terminal hazard state.",
                "temporal": "never",
                "predicate": {"type": "terminal_event", "event": "hazard"},
            },
        ],
    }
    frames = [
        _frame(0),
        _frame(1, robot=(-2.5, 0, 0.55)),
        _frame(2, robot=(-1.5, 0, 0.55), contacts=(("robot", "crate"),)),
        _frame(3, robot=(-1.2, 0, 0.55), contacts=(("robot", "crate"),)),
        _frame(4, robot=(-1.2, 0, 0.55), events=("hazard",)),
    ]
    monitor = AssertionSatisfactionMonitor(
        group=group,
        objects=objects,
        initial_frame=frames[0],
    )

    for index in range(len(frames)):
        if index:
            monitor.update(frames[index])
        authoritative = evaluate_assertion_group(
            group=group,
            frames=frames[: index + 1],
            objects=objects,
            final=False,
        )
        assert monitor.satisfied is authoritative["satisfied"]


def test_all_subject_quantifier_requires_every_matching_object() -> None:
    spec = _spec()
    duplicate = dict(spec["objects"][2])
    duplicate.update({"id": "crate_two", "position": [-1, 2, 0.5]})
    spec["objects"].append(duplicate)
    task = normalize_task_definition(
        env_id="trajectory",
        instruction="Move every crate.",
        spec=spec,
        tests=[
            {
                "id": "move_all",
                "conditions": [
                    {
                        "id": "all_crates_move",
                        "temporal": "eventually",
                        "predicate": {
                            "type": "displacement",
                            "subject": {"semantic_type": "pushable_box"},
                            "subject_quantifier": "all",
                            "min_value": 1,
                        },
                    }
                ],
            }
        ],
    )
    objects = scene_objects(spec)
    start = _frame(0)
    start["positions"]["crate_two"] = [-1, 2, 0.5]
    start["rotations"]["crate_two"] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    start["velocities"]["crate_two"] = {"linear": [0, 0, 0], "angular": [0, 0, 0]}
    partial = _frame(1, crate=(1, 0, 0.5))
    partial["positions"]["crate_two"] = [-1, 2, 0.5]
    partial["rotations"]["crate_two"] = start["rotations"]["crate_two"]
    partial["velocities"]["crate_two"] = start["velocities"]["crate_two"]
    complete = _frame(2, crate=(1, 0, 0.5))
    complete["positions"]["crate_two"] = [1, 2, 0.5]
    complete["rotations"]["crate_two"] = start["rotations"]["crate_two"]
    complete["velocities"]["crate_two"] = start["velocities"]["crate_two"]

    partial_report = evaluate_task_tests(task=task, frames=[start, partial], objects=objects)
    complete_report = evaluate_task_tests(task=task, frames=[start, partial, complete], objects=objects)

    assert partial_report["tests"][0]["passed"] is False
    assert complete_report["tests"][0]["passed"] is True
    assert complete_report["tests"][0]["conditions"][0]["witness"]["matched_subject_ids"] == [
        "crate",
        "crate_two",
    ]


def test_in_bounds_checks_complete_object_footprint() -> None:
    task, objects = _task(
        [
            {
                "id": "move",
                "conditions": [
                    {
                        "id": "move_far",
                        "temporal": "eventually",
                        "predicate": {
                            "type": "displacement",
                            "subject": {"id": "robot"},
                            "min_value": 1,
                        },
                    }
                ],
            }
        ]
    )
    report = evaluate_task_tests(
        task=task,
        frames=[_frame(0), _frame(1, robot=(6.8, 0, 0.55))],
        objects=objects,
    )

    safety = next(test for test in report["tests"] if test["id"] == "system_stay_in_bounds")
    assert safety["passed"] is False
    assert safety["conditions"][0]["witness"]["objects"] == {"robot": False}


def test_mechanism_state_and_terminal_event_can_be_composed() -> None:
    builder = EnvSpec3DBuilder.from_spec_dict(_spec())
    switch_id = builder.add_floor_switch(0, -2, id="switch")
    gate_id = builder.add_sliding_gate(2, -2, id="gate")
    builder.link_switch_to_gate(switch_id, gate_id, id="gate_link")
    builder.configure_reach_goal_game("robot", "goal")
    spec = builder.to_spec_dict()
    task = normalize_task_definition(
        env_id="trajectory",
        instruction="Open the gate, then finish.",
        spec=spec,
        tests=[
            {
                "id": "gate_then_goal",
                "mode": "all",
                "ordered_condition_ids": ["gate_open", "goal_event"],
                "conditions": [
                    {
                        "id": "gate_open",
                        "temporal": "eventually",
                        "predicate": {
                            "type": "mechanism_state",
                            "mechanism_id": "gate_link",
                            "state": "open",
                            "min_progress": 0.9,
                        },
                    },
                    {
                        "id": "goal_event",
                        "temporal": "eventually",
                        "predicate": {"type": "terminal_event", "event": "goal"},
                    },
                ],
            }
        ],
    )
    objects = scene_objects(spec)
    first = _frame(0)
    first["mechanisms"] = [{"id": "gate_link", "active": False, "progress": 0.0}]
    second = _frame(1)
    second["mechanisms"] = [{"id": "gate_link", "active": True, "progress": 1.0}]
    third = _frame(2, events=("goal",))
    third["mechanisms"] = second["mechanisms"]

    report = evaluate_task_tests(task=task, frames=[first, second, third], objects=objects)

    assert report["tests"][0]["passed"] is True
    assert report["tests"][0]["ordered_steps"] == [1, 2]


def test_ordered_atomic_sensor_and_mechanism_events_can_share_a_frame() -> None:
    builder = EnvSpec3DBuilder.from_spec_dict(_spec())
    switch_id = builder.add_floor_switch(0, -2, id="switch")
    gate_id = builder.add_sliding_gate(2, -2, id="gate")
    builder.link_switch_to_gate(switch_id, gate_id, id="gate_link")
    spec = builder.to_spec_dict()
    task = normalize_task_definition(
        env_id="trajectory",
        instruction="Touch the switch and open the gate.",
        spec=spec,
        tests=[
            {
                "id": "switch_then_gate",
                "mode": "all",
                "ordered_condition_ids": ["touch_switch", "open_gate"],
                "conditions": [
                    {
                        "id": "touch_switch",
                        "temporal": "eventually",
                        "predicate": {
                            "type": "overlap",
                            "subject": {"id": "robot"},
                            "target": {"id": "switch"},
                        },
                    },
                    {
                        "id": "open_gate",
                        "temporal": "eventually",
                        "predicate": {
                            "type": "mechanism_state",
                            "mechanism_id": "gate_link",
                            "state": "open",
                            "min_progress": 0.9,
                        },
                    },
                ],
            }
        ],
    )
    objects = scene_objects(spec)
    first = _frame(0)
    first["mechanisms"] = [{"id": "gate_link", "active": False, "progress": 0.0}]
    second = _frame(1, robot=(0, -2, 0.55))
    second["mechanisms"] = [{"id": "gate_link", "active": True, "progress": 1.0}]

    report = evaluate_task_tests(task=task, frames=[first, second], objects=objects)
    monitor = AssertionSatisfactionMonitor(
        group=next(test for test in task["tests"] if test["id"] == "switch_then_gate"),
        objects=objects,
        initial_frame=first,
    )
    monitor.update(second)

    assert report["tests"][0]["passed"] is True
    assert report["tests"][0]["ordered_steps"] == [1, 1]
    assert monitor.satisfied is True
