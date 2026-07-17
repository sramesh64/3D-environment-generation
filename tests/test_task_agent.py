from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from PIL import Image

import environment_generation.task_agent as task_agent_module
from environment_generation.artifacts import persist_artifacts
from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.env_tasks import normalize_task_definition
from environment_generation.schema import env_spec_to_dict
from environment_generation.task_agent import TASK_AGENT_OBSERVATION_MODE, TaskAgentSession
from environment_generation.styled_observation import (
    MUJOCO_OBSERVATION_RENDERER,
    STYLED_OBSERVATION_RENDERER,
    StyledObservationUnavailable,
)


PRIVILEGED_OBSERVATION_KEYS = {
    "agent",
    "available_actions",
    "camera",
    "mechanisms",
    "navigation",
    "nearby_objects",
    "termination_reason",
    "tests",
}

PRIVATE_CONTEXT_KEYS = {
    "azimuth",
    "camera",
    "camera_azimuth",
    "camera_elevation",
    "coordinates",
    "elevation",
    "fov_y_degrees",
    "heading_error_degrees",
    "position",
    "rotation_matrix",
    "semantic_type",
    "target",
    "target_bearing",
}


def _assert_observation_is_privacy_safe(observation: dict) -> None:
    assert PRIVILEGED_OBSERVATION_KEYS.isdisjoint(observation)

    def visit(value: object, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                assert key not in PRIVATE_CONTEXT_KEYS, (path, key)
                if not (not path and key == "task_id"):
                    assert key != "id" and not key.endswith("_id"), (path, key)
                visit(child, (*path, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))

    visit(observation)


def _runtime(
    tmp_path: Path,
    *,
    avoid_hazard: bool,
    live_state_log_path: Path | None = None,
) -> TaskAgentSession:
    builder = EnvSpec3DBuilder("task_agent_zones", description="task agent zones")
    builder.add_ground_plane(width=12, depth=6)
    builder.add_agent_spawn(-2, 0, id="robot")
    builder.add_hazard_zone(-0.25, 0, width=0.8, depth=1.6, id="hazard")
    builder.add_goal_zone(2, 0, width=1.2, depth=1.2, id="goal")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    tests = [{
        "id": "reach_goal",
        "conditions": [{
            "id": "enter_goal",
            "temporal": "eventually",
            "predicate": {
                "type": "overlap",
                "subject": {"id": "robot"},
                "target": {"id": "goal"},
            },
        }],
    }]
    if avoid_hazard:
        tests.append({
            "id": "avoid_hazard",
            "conditions": [{
                "id": "never_enter_hazard",
                "temporal": "never",
                "predicate": {
                    "type": "overlap",
                    "subject": {"id": "robot"},
                    "target": {"id": "hazard"},
                },
            }],
        })
    task = {
        "task_id": "cross_sensor",
        **normalize_task_definition(
            env_id=spec.id,
            instruction="Reach the goal." + (" Avoid the hazard." if avoid_hazard else ""),
            tests=tests,
            spec=env_spec_to_dict(spec),
            max_steps=600,
        ),
    }
    return TaskAgentSession(
        scene_dir=scene_dir,
        task=task,
        live_state_log_path=live_state_log_path,
        render_frames=False,
    )


def _drive_forward(session: TaskAgentSession) -> None:
    session.start()
    for _ in range(12):
        session.act(forward=1, frames=30)
        if session.terminal:
            return


def test_task_agent_observation_exposes_privacy_safe_context_and_session_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mujoco")
    session = _runtime(tmp_path, avoid_hazard=True)
    frame_path = tmp_path / "first-person.png"
    monkeypatch.setattr(session, "_render_first_person", lambda: frame_path)

    observation = session.start()

    assert observation["observation_mode"] == TASK_AGENT_OBSERVATION_MODE
    assert observation["instruction"] == "Reach the goal. Avoid the hazard."
    assert observation["outcome"] == "running"
    assert observation["terminal"] is False
    assert observation["steps_used"] == 0
    assert observation["steps_remaining"] == 600
    assert observation["resets_remaining"] == 2
    assert observation["grounded"] is session.simulation.has_ground_support()
    assert isinstance(observation["grounded"], bool)
    assert observation["collision"] is False
    assert observation["recent_events"] == []
    assert observation["recent_actions"] == []
    assert observation["frame"] == {
        "path": str(frame_path),
        "width": 640,
        "height": 360,
        "renderer": STYLED_OBSERVATION_RENDERER,
    }
    assert observation["recent_frames"] == [{
        **observation["frame"],
        "step": 0,
        "reset_count": 0,
    }]
    assert observation["recent_frames_order"] == "oldest_to_current"
    _assert_observation_is_privacy_safe(observation)


def test_task_agent_renders_live_state_through_shared_styled_scene(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mujoco")
    calls: list[dict] = []

    class FakeStyledRenderer:
        def __init__(self, **kwargs: object) -> None:
            calls.append({"init": kwargs})

        def render(self, **kwargs: object) -> dict:
            output_path = Path(str(kwargs["output_path"]))
            Image.new("RGB", (640, 360), (40, 150, 90)).save(output_path)
            calls.append({"render": kwargs})
            return {"colorRange": 110, "visiblePixels": 1000}

        def close(self) -> None:
            return

    monkeypatch.setattr(task_agent_module, "StyledObservationRenderer", FakeStyledRenderer)
    session = _runtime(tmp_path, avoid_hazard=False)
    session.render_frames = True
    session.frame_dir = tmp_path / "styled-frames"
    try:
        observation = session.start()
    finally:
        session.close()

    assert observation["frame"]["renderer"] == STYLED_OBSERVATION_RENDERER
    render_call = next(value["render"] for value in calls if "render" in value)
    assert render_call["hidden_source_ids"] == ["robot"]
    assert render_call["objects"]
    camera = render_call["camera"]
    agent_position = session.simulation.agent_position()
    assert camera["fov_y_degrees"] == 90.0
    assert camera["position"][:2] == pytest.approx(agent_position[:2])
    assert camera["position"][2] == pytest.approx(
        agent_position[2] + session.simulation.agent.height * 0.42
    )
    camera_delta = [
        camera["target"][index] - camera["position"][index]
        for index in range(3)
    ]
    assert math.hypot(*camera_delta[:2]) == pytest.approx(1.0)
    pitch = math.degrees(
        math.atan2(camera_delta[2], math.hypot(*camera_delta[:2]))
    )
    assert pitch == pytest.approx(-8.0)
    manifest = json.loads(
        (session.frame_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert manifest["renderer"] == STYLED_OBSERVATION_RENDERER
    assert manifest["camera"]["elevation"] == -8.0
    assert manifest["camera"]["fov_y_degrees"] == 90.0
    _assert_observation_is_privacy_safe(observation)


def test_task_agent_marks_explicit_mujoco_fallback_when_styled_capture_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mujoco")

    class BrokenStyledRenderer:
        def __init__(self, **_kwargs: object) -> None:
            raise StyledObservationUnavailable("browser unavailable")

    monkeypatch.setattr(task_agent_module, "StyledObservationRenderer", BrokenStyledRenderer)
    session = _runtime(tmp_path, avoid_hazard=False)
    session.render_frames = True
    session.frame_dir = tmp_path / "fallback-frames"
    fallback_path = session.frame_dir / "frame_0000.png"

    def fake_mujoco_frame() -> Path:
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 360), (20, 30, 40)).save(fallback_path)
        return fallback_path

    monkeypatch.setattr(session, "_render_mujoco_first_person", fake_mujoco_frame)
    observation = session.start()

    assert observation["frame"]["renderer"] == MUJOCO_OBSERVATION_RENDERER
    assert session._styled_renderer_unavailable is True
    assert session._styled_renderer_error == "browser unavailable"


def test_task_agent_exposes_only_anonymous_contact_ground_and_zone_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mujoco")
    session = _runtime(tmp_path, avoid_hazard=False)
    monkeypatch.setattr(session, "_render_first_person", lambda: tmp_path / "frame.png")
    contact = {
        "grounded": not session._last_grounded,
        "collision": False,
    }
    monkeypatch.setattr(
        session.simulation,
        "has_ground_support",
        lambda: contact["grounded"],
    )
    monkeypatch.setattr(
        session.simulation,
        "has_blocking_contact",
        lambda: contact["collision"],
    )

    def zone_events_since(sequence: int) -> list[dict]:
        if sequence >= 11:
            return []
        return [{
            "sequence": 11,
            "step": 7,
            "type": "zone_entered",
            "subject_id": "robot",
            "zone_id": "hazard",
            "semantic_type": "hazard",
        }]

    monkeypatch.setattr(session.simulation, "zone_events_since", zone_events_since)

    first = session.start()
    contact["grounded"] = not contact["grounded"]
    contact["collision"] = True
    second = session.observe()

    assert first["grounded"] is not second["grounded"]
    assert first["collision"] is False
    assert second["collision"] is True
    assert {event["type"] for event in second["recent_events"]} == {
        "airborne",
        "grounded",
        "zone_entered",
    }
    assert all(set(event) == {"type", "step"} for event in second["recent_events"])
    _assert_observation_is_privacy_safe(first)
    _assert_observation_is_privacy_safe(second)


def test_task_agent_retains_bounded_recent_frames_and_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mujoco")
    session = _runtime(tmp_path, avoid_hazard=False)
    monkeypatch.setattr(
        session,
        "_render_first_person",
        lambda: tmp_path / f"frame-{session.total_steps}.png",
    )

    session.start()
    observation = {}
    for _ in range(7):
        observation = session.act(look_x=0.1, frames=1)

    assert [frame["step"] for frame in observation["recent_frames"]] == [5, 6, 7]
    assert observation["recent_frames_order"] == "oldest_to_current"
    assert [action["index"] for action in observation["recent_actions"]] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert all(
        action["frames_requested"] == 1
        for action in observation["recent_actions"]
    )
    assert all(
        action["frames_advanced"] == 1
        for action in observation["recent_actions"]
    )
    assert observation["last_action"] == observation["recent_actions"][-1]
    _assert_observation_is_privacy_safe(observation)


def test_task_agent_stops_on_collision_and_logs_exact_advanced_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mujoco")
    session = _runtime(tmp_path, avoid_hazard=False)
    action_log = tmp_path / "actions.jsonl"
    session.action_log_path = action_log
    monkeypatch.setattr(session, "_render_first_person", lambda: tmp_path / "frame.png")
    monkeypatch.setattr(session.simulation, "has_blocking_contact", lambda: False)
    collision_checks = 0

    def movement_is_blocked(**_kwargs: float) -> bool:
        nonlocal collision_checks
        collision_checks += 1
        return collision_checks == 3

    monkeypatch.setattr(
        session.simulation,
        "movement_is_blocked",
        movement_is_blocked,
    )

    session.start()
    observation = session.act(forward=1, frames=30)

    logged_actions = [
        json.loads(line)
        for line in action_log.read_text(encoding="utf-8").splitlines()
    ]
    assert collision_checks == 3
    assert observation["steps_used"] == 3
    assert len(logged_actions) == 1
    assert logged_actions[0] == session.actions[0]
    assert logged_actions[0]["requested_frames"] == 30
    assert logged_actions[0]["frames"] == 3
    assert logged_actions[0]["frames_advanced"] == 3
    assert logged_actions[0]["total_step"] == 3
    assert logged_actions[0]["stopped_on_collision"] is True
    assert observation["last_action"]["frames_requested"] == 30
    assert observation["last_action"]["frames_advanced"] == 3
    assert observation["last_action"]["stopped_on_collision"] is True
    assert {"type": "collision", "step": 3} in observation["recent_events"]
    _assert_observation_is_privacy_safe(observation)


def test_unmentioned_hazard_does_not_stop_task_agent(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    session = _runtime(tmp_path, avoid_hazard=False)

    _drive_forward(session)
    observation = session.observe()
    result = session.stop()

    assert result["passed"] is True
    assert result["termination_reason"] == "task_satisfied"
    assert observation["outcome"] == "passed"
    assert any(
        event["type"] == "zone_entered"
        for event in observation["recent_events"]
    )
    assert all(
        set(event) == {"type", "step"}
        for event in observation["recent_events"]
    )
    _assert_observation_is_privacy_safe(observation)
    assert any(
        event["semantic_type"] == "hazard"
        for event in session.simulation.zone_events_since(0)
    )


def test_explicit_hazard_rule_stops_task_agent_on_violation(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    session = _runtime(tmp_path, avoid_hazard=True)

    _drive_forward(session)
    result = session.stop()

    assert result["passed"] is False
    assert result["termination_reason"] == "task_test_failed"
    assert session.observe()["outcome"] == "failed"


def test_task_agent_streams_sampled_styled_scene_states(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    live_log = tmp_path / "live" / "scene_frames.jsonl"
    session = _runtime(
        tmp_path,
        avoid_hazard=False,
        live_state_log_path=live_log,
    )

    session.start()
    session.act(forward=0.4, frames=18)

    frames = [json.loads(line) for line in live_log.read_text(encoding="utf-8").splitlines()]
    steps = [int(frame["total_step"]) for frame in frames]
    assert steps == sorted(steps)
    assert steps[0] == 0
    assert steps[-1] == 18
    assert len(frames) < 18
    robot = next(item for item in frames[-1]["objects"] if item["id"] == "robot")
    assert len(robot["position"]) == 3
    assert len(robot["rotation_matrix"]) == 9
    assert isinstance(frames[-1]["mechanisms"], list)
