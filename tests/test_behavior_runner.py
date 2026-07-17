from __future__ import annotations

import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from environment_generation.artifacts import persist_artifacts
from environment_generation.behavior_runner import (
    _child_prompt,
    _child_timeout_seconds,
    _frame_capture_complete,
    _run_codex_child,
    dismiss_behavior_trials,
    recover_abandoned_behavior_runs,
    run_behavior_trials,
)
from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.env_behavior_trials import (
    fallback_locomotion_plan,
    load_behavior_trial_report,
    normalize_behavior_trial_plan,
    write_behavior_trial_plan,
)


def _prepared_scene(tmp_path: Path):
    builder = EnvSpec3DBuilder("runner_scene", description="runner scene")
    builder.add_ground_plane(12, 8)
    builder.add_agent_spawn(-2, 0)
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    plan = fallback_locomotion_plan(
        env_id=spec.id,
        prompt="move",
        operation_count=2,
        draft_spec=spec,
    )
    write_behavior_trial_plan(scene_dir, plan)
    return scene_dir, spec, plan


def _moving_child(_snapshot, _trial, _model, action_log, _frame_dir, _emit):
    action_log.write_text(
        "\n".join(
            json.dumps(
                {
                    "action": "controller",
                    "forward": 1.0,
                    "right": 0.0,
                    "look_x": 0.0,
                    "look_y": 0.0,
                    "jump": False,
                    "frames": 60,
                    "frames_advanced": 60,
                }
            )
            for _ in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    return {"exit_code": 0, "summary": "Moved forward.", "stderr": []}


def test_runner_replays_child_actions_and_persists_current_report(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, _spec, _plan = _prepared_scene(tmp_path)
    emitted = []

    result = run_behavior_trials(
        scene_dir=scene_dir,
        model="test-model",
        child_executor=_moving_child,
        emit=lambda event, data: emitted.append((event, data)),
    )

    assert result["report"]["status"] == "passed"
    assert result["report"]["results"][0]["status"] == "passed"
    assert result["report"]["results"][0]["trajectory_url"].endswith("trajectory.json")
    assert result["report"]["stale"] is False
    assert load_behavior_trial_report(scene_dir)["run_id"] == result["run_id"]
    assert (scene_dir / "behavior_trials" / result["run_id"] / "report.json").is_file()
    assert any(data.get("status") == "child_running" for event, data in emitted if event == "behavior_trials")


def test_disjoint_behavior_tests_run_concurrently_and_merge_results(tmp_path, monkeypatch) -> None:
    scene_dir, spec, _fallback_plan = _prepared_scene(tmp_path)
    plan = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="verify two independent movement behaviors",
        operation_count=3,
        draft_spec=spec,
        trials=[
            {
                "id": trial_id,
                "instruction": f"Run {trial_id}.",
                "objective": {
                    "checks": [{"type": "agent_displacement", "min_distance": 0.1}]
                },
            }
            for trial_id in ("move_a", "move_b")
        ],
    )
    write_behavior_trial_plan(scene_dir, plan)
    child_barrier = threading.Barrier(2)

    def fake_preflight(*, scene_dir, trial):
        del scene_dir
        return dict(trial), {"status": "ready", "repair_hints": []}

    def synchronized_child(_snapshot, _trial, _model, action_log, _frame_dir, _emit):
        child_barrier.wait(timeout=5)
        action_log.write_text(
            json.dumps({"action": "controller", "forward": 1.0, "frames_advanced": 1}) + "\n",
            encoding="utf-8",
        )
        return {"exit_code": 0, "summary": "Moved.", "stderr": []}

    def fake_replay(*, scene_dir, trial, actions, frame_dir):
        del scene_dir, actions, frame_dir
        return {
            "trial_id": trial["id"],
            "instruction": trial["instruction"],
            "expected_outcome": trial["expected_outcome"],
            "severity": trial["severity"],
            "status": "passed",
            "passed": True,
            "objective": {"satisfied": True, "checks": [], "passed_count": 1, "total_count": 1},
            "constraints": {"satisfied": True, "checks": [], "passed_count": 0, "total_count": 0},
            "reward": 1.0,
            "termination_reason": "objective_satisfied",
            "steps_used": 1,
            "reset_count": 0,
            "attempt_count": 1,
            "actions": [],
            "events": [],
            "attempts": [],
            "final_state": {},
            "trajectory": [],
            "repair_hints": [],
        }

    monkeypatch.setattr("environment_generation.behavior_runner.prepare_behavior_trial", fake_preflight)
    monkeypatch.setattr("environment_generation.behavior_runner.replay_behavior_actions", fake_replay)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                run_behavior_trials,
                scene_dir=scene_dir,
                trial_ids={trial_id},
                child_executor=synchronized_child,
            )
            for trial_id in ("move_a", "move_b")
        ]
        results = [future.result(timeout=10) for future in futures]

    current_report = load_behavior_trial_report(scene_dir)
    assert len({result["run_id"] for result in results}) == 2
    assert [result["trial_id"] for result in current_report["results"]] == ["move_a", "move_b"]
    assert current_report["summary"]["passed"] == 2


def test_same_behavior_test_cannot_start_twice(tmp_path) -> None:
    scene_dir, _spec, plan = _prepared_scene(tmp_path)
    trial_id = plan["trials"][0]["id"]
    active_dir = scene_dir / "behavior_trials" / "run-live"
    active_dir.mkdir(parents=True)
    (active_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-live",
                "status": "running",
                "pid": os.getpid(),
                "trial_ids": [trial_id],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"already running: {trial_id}"):
        run_behavior_trials(scene_dir=scene_dir, trial_ids={trial_id})


def test_late_run_is_historical_and_does_not_replace_current_report(tmp_path) -> None:
    pytest.importorskip("mujoco")
    scene_dir, spec, _plan = _prepared_scene(tmp_path)

    def edit_during_child(snapshot, trial, model, action_log, frame_dir, emit):
        result = _moving_child(snapshot, trial, model, action_log, frame_dir, emit)
        changed = spec.model_dump(mode="json")
        changed["description"] = "edited during behavior run"
        (scene_dir / "env_spec_3d.json").write_text(json.dumps(changed), encoding="utf-8")
        return result

    result = run_behavior_trials(scene_dir=scene_dir, child_executor=edit_during_child)

    assert result["report"]["stale"] is True
    assert load_behavior_trial_report(scene_dir) is None
    assert (scene_dir / "behavior_trials" / result["run_id"] / "report.json").is_file()


def test_dismiss_marks_current_plan_without_running(tmp_path) -> None:
    scene_dir, _spec, _plan = _prepared_scene(tmp_path)

    result = dismiss_behavior_trials(scene_dir=scene_dir)

    assert result["status"] == "dismissed"
    assert result["plan"]["dismissed"] is True
    assert result["dismissed_trial_ids"] == ["basic_locomotion"]


def test_child_command_is_ephemeral_read_only_and_bound(tmp_path, monkeypatch) -> None:
    scene_dir, _spec, plan = _prepared_scene(tmp_path)
    captured = {}
    monkeypatch.delenv("ENVIRONMENT_GENERATION_BEHAVIOR_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.setenv("XAUTHORITY", "/tmp/xvfb/Xauthority")

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr("environment_generation.behavior_runner.shutil.which", lambda _name: "/usr/bin/codex")
    monkeypatch.setattr("environment_generation.behavior_runner.subprocess.run", fake_run)

    _run_codex_child(
        scene_dir,
        plan["trials"][0],
        "test-model",
        tmp_path / "actions.jsonl",
        tmp_path / "frames",
        None,
    )

    args = captured["args"]
    assert "--ephemeral" in args
    assert args[args.index("-s") + 1] == "read-only"
    assert "workspace-write" not in args
    assert any("ENVIRONMENT_GENERATION_BEHAVIOR_SCENE_DIR" in item for item in args)
    assert any("ENVIRONMENT_GENERATION_BEHAVIOR_TRIAL_JSON" in item for item in args)
    assert any("mcp_servers.environment-generation.env.DISPLAY" in item for item in args)
    assert any("mcp_servers.environment-generation.env.XAUTHORITY" in item for item in args)
    assert captured["kwargs"]["timeout"] == 600


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, 600),
        ("", 600),
        ("900", 900),
        ("15", 60),
        ("99999", 1800),
        ("not-a-number", 600),
    ],
)
def test_child_timeout_configuration_is_bounded(monkeypatch, configured, expected) -> None:
    if configured is None:
        monkeypatch.delenv("ENVIRONMENT_GENERATION_BEHAVIOR_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("ENVIRONMENT_GENERATION_BEHAVIOR_TIMEOUT_SECONDS", configured)

    assert _child_timeout_seconds() == expected


def test_child_timeout_returns_retryable_execution_error(tmp_path, monkeypatch) -> None:
    scene_dir, _spec, plan = _prepared_scene(tmp_path)
    monkeypatch.setenv("ENVIRONMENT_GENERATION_BEHAVIOR_TIMEOUT_SECONDS", "900")

    def timed_out(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"], stderr="child exceeded its budget")

    monkeypatch.setattr("environment_generation.behavior_runner.shutil.which", lambda _name: "/usr/bin/codex")
    monkeypatch.setattr("environment_generation.behavior_runner.subprocess.run", timed_out)

    result = _run_codex_child(
        scene_dir,
        plan["trials"][0],
        "test-model",
        tmp_path / "actions.jsonl",
        tmp_path / "frames",
        None,
    )

    assert result["exit_code"] == 124
    assert any("exceeded 900 seconds" in line for line in result["stderr"])


def test_restart_recovery_marks_only_dead_behavior_runs_as_errors(tmp_path) -> None:
    scene_dir, _spec, _plan = _prepared_scene(tmp_path)
    root = scene_dir / "behavior_trials"
    dead_dir = root / "run-dead"
    live_dir = root / "run-live"
    dead_dir.mkdir(parents=True)
    live_dir.mkdir(parents=True)
    (dead_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run-dead", "status": "running", "pid": 999_999_999}),
        encoding="utf-8",
    )
    (live_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run-live", "status": "running", "pid": os.getpid()}),
        encoding="utf-8",
    )

    recover_abandoned_behavior_runs(scene_dir)

    dead = json.loads((dead_dir / "manifest.json").read_text(encoding="utf-8"))
    live = json.loads((live_dir / "manifest.json").read_text(encoding="utf-8"))
    assert dead["status"] == "error"
    assert "restarted" in dead["error"].lower()
    assert live["status"] == "running"


def test_replay_frame_capture_requires_a_frame_at_the_authoritative_final_step(tmp_path) -> None:
    frame_dir = tmp_path / "replay_frames"
    frame_dir.mkdir()
    (frame_dir / "frame_0000.png").write_bytes(b"png")
    (frame_dir / "frames.jsonl").write_text(
        json.dumps({"path": str(frame_dir / "frame_0000.png"), "total_step": 40}) + "\n",
        encoding="utf-8",
    )

    assert _frame_capture_complete(frame_dir, expected_step=40) is True
    assert _frame_capture_complete(frame_dir, expected_step=41) is False


def test_invalid_collapsing_structure_stops_before_child_execution(tmp_path) -> None:
    pytest.importorskip("mujoco")
    builder = EnvSpec3DBuilder("unstable_runner", description="a numerically unstable dynamic tower")
    builder.add_ground_plane(16, 12)
    builder.add_pushable_box(0, 0, z=0, id="stack_1")
    builder.add_agent_spawn(-7, -5, z=0.05, id="agent")
    for level in range(1, 5):
        builder.add_pushable_box(0, 0, z=level, id=f"stack_{level + 1}")
    spec = builder.finalize()
    scene_dir = tmp_path / spec.id
    persist_artifacts(spec=spec, scene_dir=scene_dir, trace_records=[], render=False)
    plan = normalize_behavior_trial_plan(
        env_id=spec.id,
        prompt="the tower is too high to climb",
        operation_count=8,
        draft_spec=spec,
        trials=[
            {
                "instruction": "Try to stand on the top box.",
                "expected_outcome": "should_not_succeed",
                "max_steps": 160,
                "objective": {
                    "checks": [
                        {
                            "type": "agent_relation",
                            "target": {"id": "stack_5"},
                            "relation": "on_surface",
                        }
                    ]
                },
            }
        ],
    )
    write_behavior_trial_plan(scene_dir, plan)

    def child_must_not_run(*_args, **_kwargs):
        raise AssertionError("child should not run for invalid preflight")

    result = run_behavior_trials(scene_dir=scene_dir, child_executor=child_must_not_run)

    trial_result = result["report"]["results"][0]
    assert result["report"]["status"] == "needs_attention"
    assert result["report"]["passed"] is False
    assert result["report"]["summary"]["invalid_setup"] == 1
    assert trial_result["status"] == "invalid_setup"
    assert any("moves or collapses" in hint for hint in trial_result["repair_hints"])


def test_child_prompt_contains_constraints_and_target_navigation_contract() -> None:
    prompt = _child_prompt(
        {
            "id": "wall_bypass",
            "instruction": "Try to reach the far side without jumping.",
            "expected_outcome": "should_not_succeed",
            "objective": {"checks": [{"id": "past_wall", "type": "agent_relation"}]},
            "constraints": {"checks": [{"id": "no_jump", "type": "jump_count", "max_count": 0}]},
            "navigation": {"primary_target_id": "wall", "initial_camera_azimuth": -90},
            "environment_request": "Build a wall that the robot cannot simply walk around.",
            "scene_description": "A courtyard with a dividing wall.",
            "max_steps": 500,
            "max_total_steps": 1000,
            "max_resets": 1,
        }
    )

    assert '"constraints"' in prompt
    assert '"no_jump"' in prompt
    assert '"primary_target_id": "wall"' in prompt
    assert "Build a wall" in prompt
    assert "heading_error_degrees" in prompt
    assert "fresh per-attempt step budget" in prompt
    assert "Do not assume the" in prompt
    assert "goal-reaching level" in prompt
    assert "direct_path_blocked" in prompt
    assert "objective_focus" in prompt
