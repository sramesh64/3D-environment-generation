from __future__ import annotations

import json
import math

import numpy as np
import pytest
from PIL import Image

from environment_generation.artifacts import persist_artifacts
from environment_generation.behavior_trial import BehaviorTrialSession, replay_behavior_actions
from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.env_behavior_trials import fallback_locomotion_plan, normalize_behavior_trial_plan
from environment_generation.player import PlayableSimulation
from environment_generation.trajectory_assertions import capture_trajectory_frame


def _scene(tmp_path, *, goal_z: float | None = None):
    builder = EnvSpec3DBuilder("behavior_runtime", description="behavior runtime")
    builder.add_ground_plane(16, 10)
    builder.add_agent_spawn(-3, 0, id="agent")
    builder.add_pushable_box(-0.8, 0, id="box")
    builder.add_wall(2, 0, width=0.3, depth=5, height=2, id="wall")
    if goal_z is not None:
        builder.add_goal_zone(-3, 0, z=goal_z, id="goal")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    return scene_dir, spec


def _hazard_scene(tmp_path):
    builder = EnvSpec3DBuilder("behavior_hazard_runtime", description="hazard outcome runtime")
    builder.add_ground_plane(10, 6)
    builder.add_agent_spawn(-2, 0, id="agent")
    builder.add_hazard_zone(0, 0, width=1.2, depth=2, id="hazard")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    return scene_dir, spec


def _drive_until_terminal(session: BehaviorTrialSession) -> None:
    session.start()
    for _ in range(10):
        session.act(forward=1, frames=30)
        if session.terminal:
            return
    raise AssertionError("the agent did not reach the expected terminal state")


def test_fixed_step_locomotion_demonstrates_fallback(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, spec = _scene(tmp_path)
    plan = fallback_locomotion_plan(
        env_id=spec.id,
        prompt="move",
        operation_count=4,
        draft_spec=spec,
    )
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=plan["trials"][0], render_frames=False)
    observation = session.start()
    while not session.terminal:
        observation = session.act(forward=1, frames=30)
    result = session.stop()

    assert result["status"] == "passed"
    assert result["objective"]["checks"][0]["metrics"]["distance"] >= 1.0
    assert result["reward"] == 1.0
    assert result["steps_used"] <= plan["trials"][0]["max_steps"]
    assert observation["navigation"] == {
        "primary_target_id": None,
        "available": False,
        "status": "objective_satisfied",
    }


def test_raised_goal_requires_3d_overlap(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, _spec = _scene(tmp_path, goal_z=3.0)
    simulation = PlayableSimulation.from_scene(scene_dir)

    assert simulation.status() == "Exploring"
    qpos = simulation.agent.qpos_address
    simulation.data.qpos[qpos + 2] = 3.55
    simulation.mujoco.mj_forward(simulation.model, simulation.data)

    assert simulation.status() == "In goal"


def test_counterexample_success_is_hard_failure(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, spec = _scene(tmp_path)
    plan = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="do not simply pass the wall",
        operation_count=4,
        draft_spec=spec,
        trials=[
            {
                "id": "direct_route",
                "instruction": "Move to the right side of the wall without jumping.",
                "expected_outcome": "should_not_succeed",
                "objective": {
                    "mode": "all",
                    "checks": [
                        {
                            "id": "past_wall",
                            "type": "agent_relation",
                            "target": {"id": "wall"},
                            "relation": "right_of",
                        },
                    ],
                },
                "constraints": {
                    "checks": [{"id": "no_jump", "type": "jump_count", "max_count": 0}]
                },
            }
        ],
    )
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=plan["trials"][0], render_frames=False)
    session.start()
    qpos = session.simulation.agent.qpos_address
    session.simulation.data.qpos[qpos] = 3.0
    session.simulation.mujoco.mj_forward(session.simulation.model, session.simulation.data)
    session.attempts[-1].record_step(total_step=1, jump_started=False)
    session.total_steps = 1

    result = session.stop()

    assert result["objective"]["satisfied"] is True
    assert result["constraints"]["satisfied"] is True
    assert result["status"] == "failed"


def test_negative_trial_cannot_pass_when_the_child_stops_immediately(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, spec = _scene(tmp_path)
    trial = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="do not simply pass the wall",
        operation_count=4,
        draft_spec=spec,
        trials=[
            {
                "id": "direct_route",
                "instruction": "Try to move to the right side of the wall without jumping.",
                "expected_outcome": "should_not_succeed",
                "objective": {
                    "checks": [
                        {
                            "id": "past_wall",
                            "type": "agent_relation",
                            "target": {"id": "wall"},
                            "relation": "right_of",
                        }
                    ]
                },
                "constraints": {
                    "checks": [{"id": "no_jump", "type": "jump_count", "max_count": 0}]
                },
            }
        ],
    )["trials"][0]
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=trial, render_frames=False)

    session.start()
    result = session.stop()

    assert result["status"] == "inconclusive"
    assert result["passed"] is False
    assert result["search_evidence"]["active_steps"] == 0
    assert result["search_evidence"]["reason"] == "insufficient_active_search"


def test_hazard_entry_can_be_a_positive_objective(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, _spec = _hazard_scene(tmp_path)
    trial = {
        "id": "hazard_entry",
        "instruction": "Enter the hazard.",
        "expected_outcome": "should_succeed",
        "severity": "advisory",
        "max_steps": 500,
        "max_resets": 1,
        "objective": {
            "mode": "all",
            "checks": [
                {
                    "id": "entered_hazard",
                    "type": "zone_entry",
                    "description": "Enter the hazard.",
                    "selector": {"id": "hazard"},
                    "min_count": 1,
                }
            ],
            "ordered_check_ids": ["entered_hazard"],
        },
        "constraints": {"mode": "all", "checks": [], "ordered_check_ids": []},
    }
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=trial, render_frames=False)

    _drive_until_terminal(session)
    result = session.stop()

    assert result["termination_reason"] == "objective_satisfied"
    assert result["status"] == "passed"
    assert result["objective"]["satisfied"] is True
    assert result["objective"]["disqualified_by_terminal_event"] is None
    assert result["attempts"][0]["terminal_outcome"] is None


def test_explicit_hazard_avoidance_constraint_ends_the_attempt(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, spec = _hazard_scene(tmp_path)
    trial = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="move forward without entering the hazard",
        operation_count=3,
        draft_spec=spec,
        trials=[
            {
                "id": "avoid_hazard",
                "instruction": "Move forward without entering the hazard.",
                "objective": {
                    "checks": [
                        {
                            "id": "move_forward",
                            "type": "agent_displacement",
                            "min_distance": 4.0,
                        }
                    ],
                },
                "constraints": {
                    "checks": [
                        {
                            "id": "never_enter_hazard",
                            "description": "The agent never enters the hazard.",
                            "temporal": "never",
                            "predicate": {
                                "type": "overlap",
                                "subject": {"id": "agent"},
                                "target": {"id": "hazard"},
                            },
                        }
                    ]
                },
            }
        ],
    )["trials"][0]
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=trial, render_frames=False)

    _drive_until_terminal(session)
    before_reset = session.observe()
    result = session.stop()

    assert before_reset["termination_reason"] == "constraint_failed"
    assert before_reset["trial_satisfied"] is False
    assert result["status"] == "inconclusive"
    assert any(event["type"] == "zone_entered" for event in result["events"])
    assert result["attempts"][0]["terminal_outcome"] is None


def test_legacy_terminal_event_predicate_remains_compatible(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, spec = _hazard_scene(tmp_path)
    trial = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="load a legacy hazard event check",
        operation_count=3,
        draft_spec=spec,
        trials=[
            {
                "id": "legacy_hazard_event",
                "instruction": "Enter the hazard and record the legacy event predicate.",
                "objective": {
                    "checks": [
                        {"id": "hazard_terminal", "type": "terminal_event", "event": "hazard"}
                    ]
                },
            }
        ],
    )["trials"][0]
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=trial, render_frames=False)

    _drive_until_terminal(session)
    result = session.stop()

    assert result["termination_reason"] == "objective_satisfied"
    assert result["status"] == "passed"
    assert result["objective"]["satisfied"] is True
    assert result["attempts"][0]["terminal_outcome"] is None


def test_replay_matches_recorded_final_transform(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, spec = _scene(tmp_path)
    plan = fallback_locomotion_plan(
        env_id=spec.id,
        prompt="move",
        operation_count=4,
        draft_spec=spec,
    )
    trial = plan["trials"][0]
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=trial, render_frames=False)
    session.start()
    session.act(forward=0.8, right=0.25, look_x=0.2, frames=30)
    session.act(forward=1.0, frames=30)
    original = session.stop()

    replay = replay_behavior_actions(scene_dir=scene_dir, trial=trial, actions=original["actions"])
    original_agent = next(item for item in original["final_state"]["objects"] if item["id"] == "agent")
    replay_agent = next(item for item in replay["final_state"]["objects"] if item["id"] == "agent")

    assert replay_agent["position"] == pytest.approx(original_agent["position"], abs=1e-8)
    assert replay["objective"]["satisfied"] == original["objective"]["satisfied"]
    assert math.isfinite(replay_agent["position"][2])


def test_first_person_observation_is_nonblank_and_tracks_look_input(tmp_path, monkeypatch) -> None:
    pytest.importorskip("mujoco")
    scene_dir, spec = _scene(tmp_path)
    plan = fallback_locomotion_plan(
        env_id=spec.id,
        prompt="move",
        operation_count=4,
        draft_spec=spec,
    )
    session = BehaviorTrialSession(
        scene_dir=scene_dir,
        trial=plan["trials"][0],
        frame_dir=tmp_path / "frames",
        render_frames=True,
    )
    camera_azimuths = []

    class FakeRenderer:
        def __init__(self, _model, *, height, width):
            self.height = height
            self.width = width

        def update_scene(self, _data, *, camera, scene_option=None):
            camera_azimuths.append(float(camera.azimuth))
            assert scene_option.geomgroup[5] == 0

        def render(self):
            pixels = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            pixels[:, :, 1] = 120
            pixels[:, self.width // 2 :, 2] = 230
            return pixels

        def close(self):
            pass

    monkeypatch.setattr(session.simulation.mujoco, "Renderer", FakeRenderer)

    initial = session.start()
    turned = session.act(look_x=0.5, frames=1)

    assert Image.open(initial["path"]).size == (640, 360)
    assert Image.open(initial["path"]).getbbox() is not None
    assert turned["camera"]["azimuth"] == pytest.approx(-75.0)
    assert camera_azimuths[-1] != camera_azimuths[0]
    assert camera_azimuths[0] == pytest.approx(0.0)
    assert camera_azimuths[-1] == pytest.approx(-15.0)
    manifest = [
        json.loads(line)
        for line in (tmp_path / "frames" / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest[0]["camera"] == initial["camera"]
    assert manifest[-1]["camera"] == turned["camera"]
    assert manifest[-1]["navigation"] == turned["navigation"]
    assert manifest[-1]["agent"] == turned["agent"]
    assert manifest[-1]["attempt_objective"]["checks"][0]["id"] == "agent_moves_one_meter"
    assert manifest[-1]["events"]
    assert all(event.get("routine") for event in manifest[-1]["events"])
    session.close()


def test_box_tower_climb_records_ordered_jump_contact_height_and_relation(tmp_path) -> None:
    pytest.importorskip("mujoco")
    builder = EnvSpec3DBuilder("tower_climb", description="climb a short box tower")
    builder.add_ground_plane(12, 8)
    builder.add_agent_spawn(-2.4, 0, id="agent")
    builder.add_pushable_box(0, 0, width=2, depth=2, height=0.6, id="base_box")
    builder.add_pushable_box(0, 0, z=0.6, width=1.2, depth=1.2, height=0.6, id="top_box")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    plan = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="the agent can climb onto the tower of boxes",
        operation_count=4,
        draft_spec=spec,
        trials=[
            {
                "id": "climb_tower",
                "instruction": "Jump onto the box tower and finish above the top box.",
                "objective": {
                    "mode": "all",
                    "checks": [
                        {"id": "jumped", "type": "jump_count", "min_count": 1},
                        {"id": "touched_top", "type": "contact_count", "selector": {"id": "top_box"}},
                        {"id": "gained_height", "type": "agent_height_gain", "min_gain": 0.8},
                        {
                            "id": "above_top",
                            "type": "agent_relation",
                            "target": {"id": "top_box"},
                            "relation": "above",
                            "when": "final",
                        },
                    ],
                    "ordered_check_ids": ["jumped", "touched_top"],
                },
            }
        ],
    )
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=plan["trials"][0], render_frames=False)
    session.start()
    session.act(frames=20)
    for index in range(80):
        session.act(forward=1 if index < 30 else 0, jump=index in {0, 25, 50}, frames=5)
        if session.terminal:
            break

    result = session.stop()

    assert result["status"] == "passed"
    assert result["objective"]["order_satisfied"] is True
    ordered_steps = result["objective"]["ordered_steps"]
    assert ordered_steps == sorted(ordered_steps)
    assert ordered_steps[0] < ordered_steps[-1]
    by_id = {check["id"]: check for check in result["objective"]["checks"]}
    assert by_id["touched_top"]["metrics"]["count"] >= 1
    assert by_id["gained_height"]["metrics"]["height_gain"] >= 0.8


def test_final_relation_is_scored_independently_of_ordered_events(tmp_path) -> None:
    pytest.importorskip("mujoco")
    builder = EnvSpec3DBuilder("ordered_landing", description="ordered landing evidence")
    builder.add_ground_plane(8, 6)
    builder.add_agent_spawn(-2, 0, id="agent")
    builder.add_static_box(0, 0, width=2, depth=2, height=0.6, id="box")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    trial = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="jump and land on the box",
        operation_count=3,
        draft_spec=spec,
        trials=[
            {
                "instruction": "Jump, touch the box, and finish on it.",
                "objective": {
                    "checks": [
                        {"id": "jumped", "type": "jump_count", "min_count": 1},
                        {"id": "touched", "type": "contact_count", "selector": {"id": "box"}},
                        {
                            "id": "landed",
                            "type": "agent_relation",
                            "target": {"id": "box"},
                            "relation": "on_surface",
                            "when": "final",
                        },
                    ],
                    "ordered_check_ids": ["jumped", "touched"],
                },
            }
        ],
    )["trials"][0]
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=trial, render_frames=False)
    tracker = session.attempts[-1]
    jumped = capture_trajectory_frame(
        session.simulation,
        reset_count=0,
        total_step=36,
    )
    jumped["jump_count"] = 1
    touched = capture_trajectory_frame(
        session.simulation,
        reset_count=0,
        total_step=114,
    )
    touched["jump_count"] = 1
    touched["contacts"] = [["agent", "box"]]
    tracker.frames.extend([jumped, touched])
    qpos = session.simulation.agent.qpos_address
    session.simulation.data.qpos[qpos : qpos + 3] = [0.0, 0.0, 1.15]
    session.simulation.mujoco.mj_forward(session.simulation.model, session.simulation.data)
    landed = capture_trajectory_frame(
        session.simulation,
        reset_count=0,
        total_step=115,
    )
    landed["jump_count"] = 1
    tracker.frames.append(landed)

    objective = tracker.objective(final=True)

    assert objective["ordered_steps"] == [36, 114]
    assert objective["order_satisfied"] is True
    assert objective["satisfied"] is True
    session.close()


def test_generic_object_delivery_scores_the_subject_not_the_agent(tmp_path) -> None:
    pytest.importorskip("mujoco")
    builder = EnvSpec3DBuilder("generic_delivery", description="move an object into a region")
    builder.add_ground_plane(12, 8)
    builder.add_agent_spawn(-3, 0, id="agent")
    builder.add_pushable_box(0, 0, id="movable")
    builder.add_target_region(3, 0, width=1.5, depth=1.5, id="target_region")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    trial = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="move the object into the region",
        operation_count=4,
        draft_spec=spec,
        trials=[
            {
                "instruction": "Push the movable object into the target region.",
                "objective": {
                    "checks": [
                        {
                            "id": "object_in_region",
                            "temporal": "eventually",
                            "predicate": {
                                "type": "overlap",
                                "subject": {"id": "movable"},
                                "target": {"id": "target_region"},
                            },
                        }
                    ]
                },
            }
        ],
    )["trials"][0]
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=trial, render_frames=False)
    tracker = session.attempts[-1]

    agent_qpos = session.simulation.agent.qpos_address
    session.simulation.data.qpos[agent_qpos : agent_qpos + 3] = [3.0, 0.0, 0.55]
    session.simulation.mujoco.mj_forward(session.simulation.model, session.simulation.data)
    tracker.record_step(total_step=1, jump_started=False)
    assert tracker.objective(final=False)["satisfied"] is False

    body_id = session.simulation.dynamic_body_ids["movable"]
    joint_id = int(session.simulation.model.body_jntadr[body_id])
    object_qpos = int(session.simulation.model.jnt_qposadr[joint_id])
    session.simulation.data.qpos[object_qpos : object_qpos + 3] = [3.0, 0.0, 0.5]
    session.simulation.mujoco.mj_forward(session.simulation.model, session.simulation.data)
    tracker.record_step(total_step=2, jump_started=False)

    objective = tracker.objective(final=True)
    assert objective["satisfied"] is True
    assert objective["checks"][0]["metrics"]["counts_by_subject"] == {"movable": 1}
    session.close()


def test_push_trial_uses_contact_and_object_displacement_evidence(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, spec = _scene(tmp_path)
    plan = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="the agent can push the box",
        operation_count=4,
        draft_spec=spec,
        trials=[
            {
                "id": "push_box",
                "instruction": "Push the box at least 25 centimeters.",
                "objective": {
                    "checks": [
                        {
                            "id": "box_moves",
                            "type": "object_displacement",
                            "selector": {"id": "box"},
                            "min_distance": 0.25,
                        },
                        {"id": "touch_box", "type": "contact_count", "selector": {"id": "box"}},
                    ]
                },
            }
        ],
    )
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=plan["trials"][0], render_frames=False)
    session.start()
    for _ in range(30):
        session.act(forward=1, frames=10)
        if session.terminal:
            break

    result = session.stop()

    assert result["status"] == "passed"
    by_id = {check["id"]: check for check in result["objective"]["checks"]}
    assert by_id["box_moves"]["metrics"]["distance"] >= 0.25
    assert by_id["touch_box"]["metrics"]["count"] >= 1


def test_gap_clearance_demonstrates_jump_platform_access_and_hazard_avoidance(tmp_path) -> None:
    pytest.importorskip("mujoco")
    builder = EnvSpec3DBuilder("gap_clearance", description="two platforms separated by a marked gap")
    builder.add_platform(-2.25, 0, 0, width=4, depth=4, thickness=0.3, id="left_platform")
    builder.add_platform(2.25, 0, 0, width=4, depth=4, thickness=0.3, id="right_platform")
    builder.add_agent_spawn(-2.5, 0, z=0.3, id="agent")
    builder.add_hazard_zone(0, 0, width=0.5, depth=3, height=0.1, id="gap_hazard")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    plan = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="the agent can jump the gap without touching the hazard",
        operation_count=4,
        draft_spec=spec,
        trials=[
            {
                "id": "clear_gap",
                "instruction": "Jump across the gap and land on the right platform without entering the hazard.",
                "objective": {
                    "checks": [
                        {"id": "jumped", "type": "jump_count", "min_count": 1},
                        {
                            "id": "past_gap",
                            "type": "agent_relation",
                            "target": {"id": "gap_hazard"},
                            "relation": "right_of",
                        },
                        {
                            "id": "avoided_hazard",
                            "type": "zone_entry",
                            "selector": {"id": "gap_hazard"},
                            "max_count": 0,
                        },
                        {
                            "id": "landed",
                            "type": "agent_relation",
                            "target": {"id": "right_platform"},
                            "relation": "on_surface",
                            "when": "final",
                        },
                    ]
                },
            }
        ],
    )
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=plan["trials"][0], render_frames=False)
    session.start()
    session.act(frames=10)
    for index in range(20):
        session.act(forward=1, jump=index == 0, frames=10)
        if session.terminal:
            break

    result = session.stop()

    assert result["status"] == "passed"
    by_id = {check["id"]: check for check in result["objective"]["checks"]}
    assert by_id["jumped"]["metrics"]["count"] == 1
    assert by_id["avoided_hazard"]["metrics"]["count"] == 0
    assert by_id["landed"]["passed"] is True


def test_optional_goal_zone_entry_is_scored_without_baseline_goal_requirement(tmp_path) -> None:
    pytest.importorskip("mujoco")
    builder = EnvSpec3DBuilder("optional_goal", description="a small optional target")
    builder.add_ground_plane(10, 6)
    builder.add_agent_spawn(-2, 0, id="agent")
    builder.add_goal_zone(0, 0, id="optional_goal")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    plan = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="enter the target",
        operation_count=3,
        draft_spec=spec,
        trials=[
            {
                "id": "enter_goal",
                "instruction": "Enter the optional target zone.",
                "objective": {
                    "checks": [
                        {"id": "goal_entry", "type": "zone_entry", "selector": {"id": "optional_goal"}}
                    ]
                },
            }
        ],
    )
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=plan["trials"][0], render_frames=False)
    session.start()
    for _ in range(20):
        session.act(forward=1, frames=10)
        if session.terminal:
            break

    result = session.stop()

    assert result["status"] == "passed"
    assert result["objective"]["checks"][0]["metrics"]["counts"] == {"optional_goal": 1}


def test_step_and_reset_budgets_are_exact(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, spec = _scene(tmp_path)
    plan = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="bounded search",
        operation_count=4,
        draft_spec=spec,
        trials=[
            {
                "id": "bounded",
                "instruction": "Attempt an intentionally unreachable displacement.",
                "max_steps": 65,
                "max_resets": 1,
                "objective": {"checks": [{"type": "agent_displacement", "min_distance": 100}]},
            }
        ],
    )
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=plan["trials"][0], render_frames=False)
    session.start()

    first = session.act(forward=0.2, frames=17)
    assert first["steps_used"] == 17
    reset = session.reset()
    assert reset["attempt"] == 2
    reset_frames = [frame for frame in session.trajectory if frame["total_step"] == 17]
    assert [frame["attempt"] for frame in reset_frames] == [1, 2]
    final = session.act(forward=0.2, frames=60)

    assert final["last_action"]["frames_advanced"] == 60
    assert final["attempt_steps_used"] == 60
    assert final["steps_used"] == 77
    assert not final["termination_reason"]
    exhausted = session.act(forward=0.2, frames=10)
    assert exhausted["last_action"]["frames_advanced"] == 5
    assert exhausted["attempt_steps_used"] == 65
    assert exhausted["steps_used"] == 82
    assert exhausted["termination_reason"] == "step_budget"
    with pytest.raises(ValueError, match="reset budget"):
        session.reset()
    session.close()


def test_zero_reset_budget_is_preserved(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, spec = _scene(tmp_path)
    trial = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="one attempt only",
        operation_count=4,
        draft_spec=spec,
        trials=[
            {
                "instruction": "Move a long distance in one attempt.",
                "max_resets": 0,
                "objective": {"checks": [{"type": "agent_displacement", "min_distance": 100}]},
            }
        ],
    )["trials"][0]
    session = BehaviorTrialSession(scene_dir=scene_dir, trial=trial, render_frames=False)

    observation = session.start()

    assert observation["resets_remaining"] == 0
    with pytest.raises(ValueError, match="reset budget"):
        session.reset()
    session.close()


def test_real_first_person_frame_is_nonblank_when_graphics_context_is_available(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, spec = _scene(tmp_path)
    plan = fallback_locomotion_plan(
        env_id=spec.id,
        prompt="move",
        operation_count=4,
        draft_spec=spec,
    )
    session = BehaviorTrialSession(
        scene_dir=scene_dir,
        trial=plan["trials"][0],
        frame_dir=tmp_path / "actual_frames",
        render_frames=True,
    )
    observation = session.start()
    try:
        if observation.get("frame_error") == "invalid CoreGraphics connection":
            pytest.skip("sandbox does not expose a macOS graphics connection")
        assert "frame_error" not in observation
        pixels = np.asarray(Image.open(observation["path"]).convert("RGB"))
        assert pixels.shape == (360, 640, 3)
        assert float(pixels.std()) > 1.0
    finally:
        session.close()
