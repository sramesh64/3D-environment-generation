from __future__ import annotations

import json
from pathlib import Path

import pytest

from environment_generation.artifacts import persist_artifacts
from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.env_tasks import create_task_draft, read_task
from environment_generation.task_oracle import TaskOracleSessionManager, replay_task_actions
from environment_generation.trajectory_assertions import TrajectoryRecorder


def _scene(tmp_path: Path) -> tuple[Path, dict]:
    builder = EnvSpec3DBuilder("oracle", description="oracle")
    builder.add_ground_plane(width=14, depth=10)
    builder.add_agent_spawn(-2, 0, id="robot")
    builder.add_hazard_zone(-0.25, 0, width=0.8, depth=1.6, id="hazard")
    builder.add_goal_zone(1.5, 0, width=1.4, depth=1.4, id="goal")
    scene_dir = tmp_path / "oracle"
    persist_artifacts(spec=builder.finalize(), scene_dir=scene_dir, trace_records=[], render=False)
    tests = [
        {
            "id": "reach",
            "description": "Reach the goal.",
            "conditions": [
                {
                    "id": "enter",
                    "temporal": "eventually",
                    "predicate": {
                        "type": "overlap",
                        "subject": {"id": "robot"},
                        "target": {"id": "goal"},
                    },
                }
            ],
        }
    ]
    task = create_task_draft(
        scene_dir=scene_dir,
        env_id="oracle",
        instruction="Reach the goal.",
        compiler_output={"summary": "Reach the goal", "max_steps": 500, "tests": tests},
    )
    return scene_dir, task


def test_authoritative_replay_passes_goal_task_deterministically(tmp_path: Path) -> None:
    scene_dir, task = _scene(tmp_path)
    actions = [
        {"right": 0, "forward": 1, "camera_azimuth": -90, "jump": False, "frames": 140}
    ]

    first = replay_task_actions(scene_dir=scene_dir, task=task, actions=actions)
    second = replay_task_actions(scene_dir=scene_dir, task=task, actions=actions)

    assert first["passed"] is True
    assert "hazard" in first["terminal_events"]
    assert first["step_count"] == second["step_count"] == 140
    assert first["tests"] == second["tests"]
    assert first["trajectory"][-1]["objects"] == second["trajectory"][-1]["objects"]


def test_oracle_manager_only_validates_after_server_replay_and_persists_provenance(tmp_path: Path) -> None:
    scene_dir, task = _scene(tmp_path)
    manager = TaskOracleSessionManager()
    state = manager.start(scene_dir=scene_dir, task_id=task["task_id"])
    assert state["timestep"] == pytest.approx(0.01)
    assert manager.has_scene(scene_dir) is True
    session_id = state["session_id"]
    for _ in range(12):
        state = manager.step(
            session_id,
            right=0,
            forward=1,
            camera_azimuth=-90,
            jump=False,
            frames=12,
        )
    result = manager.finish(session_id)
    assert manager.has_scene(scene_dir) is False

    saved = read_task(scene_dir, task["task_id"])
    assert result["status"] == "validated"
    assert saved["status"] == "validated"
    assert saved["oracle"]["provenance"] == "human_recording"
    assert saved["oracle"]["frames"] == 144
    actions_path = scene_dir / "tasks" / task["task_id"] / "attempts" / saved["oracle"]["attempt_id"] / "actions.json"
    assert json.loads(actions_path.read_text(encoding="utf-8"))["provenance"] == "human_recording"


def test_failing_oracle_is_retained_but_never_finalizes_task(tmp_path: Path) -> None:
    scene_dir, task = _scene(tmp_path)
    manager = TaskOracleSessionManager()
    state = manager.start(scene_dir=scene_dir, task_id=task["task_id"])
    manager.step(
        state["session_id"],
        right=0,
        forward=0,
        camera_azimuth=-90,
        jump=False,
        frames=12,
    )

    result = manager.finish(state["session_id"])
    saved = read_task(scene_dir, task["task_id"])

    assert result["status"] == "validation_failed"
    assert saved["oracle"] is None
    assert saved["oracle_attempts"][-1]["passed"] is False


def test_oracle_steps_defer_full_report_until_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_dir, task = _scene(tmp_path)
    report_calls = 0
    original_report = TrajectoryRecorder.report

    def tracked_report(self: TrajectoryRecorder, *, task: dict, final: bool) -> dict:
        nonlocal report_calls
        report_calls += 1
        return original_report(self, task=task, final=final)

    monkeypatch.setattr(TrajectoryRecorder, "report", tracked_report)
    manager = TaskOracleSessionManager()
    state = manager.start(scene_dir=scene_dir, task_id=task["task_id"])
    session_id = state["session_id"]

    assert report_calls == 1
    assert state["report_current"] is True
    assert state["report_step"] == 0

    state = manager.step(
        session_id,
        right=0,
        forward=1,
        camera_azimuth=-90,
        jump=False,
        frames=12,
    )
    assert report_calls == 1
    assert state["report_current"] is False
    assert state["report_step"] == 0
    assert state["ready_to_validate"] is False

    state = manager.step(
        session_id,
        right=0,
        forward=0,
        camera_azimuth=-90,
        jump=False,
        frames=1,
        evaluate_report=True,
    )
    assert report_calls == 2
    assert state["report_current"] is True
    assert state["report_step"] == state["steps"]

    manager.snapshot(session_id)
    assert report_calls == 3


def test_reset_discards_current_attempt_actions(tmp_path: Path) -> None:
    scene_dir, task = _scene(tmp_path)
    manager = TaskOracleSessionManager()
    state = manager.start(scene_dir=scene_dir, task_id=task["task_id"])
    manager.step(
        state["session_id"],
        right=0,
        forward=1,
        camera_azimuth=-90,
        jump=False,
        frames=12,
    )
    manager.reset(state["session_id"])

    with pytest.raises(ValueError, match="at least one action"):
        manager.finish(state["session_id"])


def test_interrupted_recording_recovers_to_pending_oracle(tmp_path: Path) -> None:
    scene_dir, task = _scene(tmp_path)
    saved = read_task(scene_dir, task["task_id"], include_staleness=False)
    saved["status"] = "recording"
    saved["active_oracle_session"] = {"session_id": "missing", "started_at": "now"}
    from environment_generation.env_tasks import write_task

    write_task(scene_dir, saved)
    manager = TaskOracleSessionManager()

    manager.recover_scene(scene_dir)

    recovered = read_task(scene_dir, task["task_id"], include_staleness=False)
    assert recovered["status"] == "pending_oracle"
    assert recovered["active_oracle_session"] is None
    assert "interrupted" in recovered["recording_error"]
