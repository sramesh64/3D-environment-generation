from __future__ import annotations

import math

import pytest

from environment_generation.artifacts import persist_artifacts
from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.play_session import PlaySessionManager


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _scene_dir(tmp_path):
    builder = EnvSpec3DBuilder("embedded_play", description="embedded play scene")
    builder.make_box_goal_scene()
    scene_dir = tmp_path / "embedded_play"
    persist_artifacts(spec=builder.finalize(), scene_dir=scene_dir, trace_records=[], render=False)
    return scene_dir


def _transform(state, object_id):
    return next(item for item in state["objects"] if item["id"] == object_id)


def test_play_session_returns_dynamic_transforms_and_moves_agent(tmp_path) -> None:
    pytest.importorskip("mujoco")
    clock = FakeClock()
    manager = PlaySessionManager(clock=clock)
    started = manager.start(scene_dir=_scene_dir(tmp_path))
    session_id = started["session_id"]
    start_agent = _transform(started, "agent")

    assert len(session_id) == 32
    assert len(start_agent["rotation_matrix"]) == 9
    assert {item["id"] for item in started["objects"]} >= {"agent", "pushable_box"}

    clock.advance(0.08)
    advanced = manager.step(session_id, forward=1, right=0, camera_azimuth=-90)
    moved_agent = _transform(advanced, "agent")

    assert moved_agent["position"][0] > start_agent["position"][0]
    assert math.dist(moved_agent["position"][:2], start_agent["position"][:2]) > 0.05
    assert advanced["simulation_time"] > 0


def test_play_session_reset_and_stop(tmp_path) -> None:
    pytest.importorskip("mujoco")
    clock = FakeClock()
    manager = PlaySessionManager(clock=clock)
    started = manager.start(scene_dir=_scene_dir(tmp_path))
    session_id = started["session_id"]
    start_position = _transform(started, "agent")["position"]

    assert manager.has_env("embedded_play") is True

    clock.advance(0.1)
    manager.step(session_id, forward=1, camera_azimuth=-90)
    reset = manager.reset(session_id)

    assert _transform(reset, "agent")["position"] == pytest.approx(start_position)
    assert reset["simulation_time"] == 0
    assert manager.stop(session_id) is True
    assert manager.has_env("embedded_play") is False
    assert manager.stop(session_id) is False
    with pytest.raises(ValueError, match="not found"):
        manager.step(session_id)


def test_play_session_exposes_grounded_jump_state(tmp_path) -> None:
    pytest.importorskip("mujoco")
    clock = FakeClock()
    manager = PlaySessionManager(clock=clock)
    started = manager.start(scene_dir=_scene_dir(tmp_path))
    start_z = _transform(started, "agent")["position"][2]

    clock.advance(0.08)
    jumped = manager.step(started["session_id"], jump=True, camera_azimuth=-42)

    assert _transform(jumped, "agent")["position"][2] > start_z
    assert jumped["grounded"] is False


def test_play_session_validates_inputs_and_expires_idle_sessions(tmp_path) -> None:
    pytest.importorskip("mujoco")
    clock = FakeClock()
    manager = PlaySessionManager(clock=clock, ttl_seconds=1.0)
    session_id = manager.start(scene_dir=_scene_dir(tmp_path))["session_id"]

    with pytest.raises(ValueError, match="finite"):
        manager.step(session_id, right=float("nan"))
    with pytest.raises(ValueError, match="boolean"):
        manager.step(session_id, jump="yes")

    clock.advance(1.1)
    with pytest.raises(ValueError, match="expired"):
        manager.reset(session_id)
