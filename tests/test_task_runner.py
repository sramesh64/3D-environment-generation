from __future__ import annotations

import json
from pathlib import Path

import pytest

from environment_generation.artifacts import persist_artifacts
from environment_generation.builder import EnvSpec3DBuilder
from environment_generation.env_tasks import create_task_draft, read_task, write_task
from environment_generation.task_runner import (
    TaskCodexEventTranslator,
    TaskSceneFrameWatcher,
    _build_task_agent_codex_args,
    _child_prompt,
    _persist_evidence_frames,
    run_validated_task,
)
from environment_generation.task_agent import TASK_AGENT_OBSERVATION_MODE


def _validated_scene(tmp_path: Path) -> tuple[Path, dict]:
    builder = EnvSpec3DBuilder("runner", description="runner")
    builder.add_ground_plane(width=14, depth=10)
    builder.add_agent_spawn(-2, 0, id="robot")
    builder.add_goal_zone(1.5, 0, width=1.4, depth=1.4, id="goal")
    scene_dir = tmp_path / "runner"
    persist_artifacts(spec=builder.finalize(), scene_dir=scene_dir, trace_records=[], render=False)
    task = create_task_draft(
        scene_dir=scene_dir,
        env_id="runner",
        instruction="Reach the goal.",
        compiler_output={
            "summary": "Reach the goal",
            "max_steps": 500,
            "tests": [
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
            ],
        },
    )
    task["status"] = "validated"
    task["oracle"] = {
        "provenance": "human_recording",
        "env_spec_hash": task["env_spec_hash"],
        "task_definition_hash": task["task_definition_hash"],
    }
    write_task(scene_dir, task)
    return scene_dir, task


def test_task_run_requires_current_human_oracle(tmp_path: Path) -> None:
    scene_dir, task = _validated_scene(tmp_path)
    task["oracle"]["provenance"] = "generated"
    write_task(scene_dir, task)

    with pytest.raises(ValueError, match="human-recorded oracle"):
        run_validated_task(scene_dir=scene_dir, task_id=task["task_id"])


def test_task_run_parent_replay_is_authoritative(tmp_path: Path) -> None:
    scene_dir, task = _validated_scene(tmp_path)

    def child_executor(_scene, _task, _model, action_log, _frames, _emit):
        action_log.write_text(
            json.dumps(
                {
                    "action": "controller",
                    "right": 0,
                    "forward": 0,
                    "camera_azimuth": -90,
                    "jump": False,
                    "frames": 20,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"exit_code": 0, "summary": "I passed the task.", "stderr": []}

    result = run_validated_task(
        scene_dir=scene_dir,
        task_id=task["task_id"],
        child_executor=child_executor,
    )

    assert result["report"]["passed"] is False
    assert result["report"]["status"] == "failed"
    assert result["report"]["child_summary"] == "I passed the task."
    saved = read_task(scene_dir, task["task_id"])
    assert saved["latest_run"]["passed"] is False
    assert saved["latest_run"]["trajectory_url"].startswith("/generated/runner/tasks/")


def test_task_run_passing_actions_are_replayed_and_persisted(tmp_path: Path) -> None:
    scene_dir, task = _validated_scene(tmp_path)

    def child_executor(_scene, _task, _model, action_log, _frames, _emit):
        action_log.write_text(
            json.dumps(
                {
                    "action": "controller",
                    "right": 0,
                    "forward": 1,
                    "camera_azimuth": -90,
                    "jump": False,
                    "frames": 140,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"exit_code": 0, "summary": "Reached it.", "stderr": []}

    result = run_validated_task(
        scene_dir=scene_dir,
        task_id=task["task_id"],
        child_executor=child_executor,
    )

    assert result["report"]["passed"] is True
    assert result["report"]["status"] == "passed"
    run_dir = scene_dir / "tasks" / task["task_id"] / "runs" / result["run_id"]
    assert (run_dir / "snapshot" / "env_spec_3d.json").is_file()
    assert (run_dir / "actions.json").is_file()
    assert (run_dir / "trajectory.json").is_file()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pid"] is None
    assert manifest["observation_mode"] == TASK_AGENT_OBSERVATION_MODE
    assert result["report"]["observation_mode"] == TASK_AGENT_OBSERVATION_MODE
    saved = read_task(scene_dir, task["task_id"])
    assert saved["latest_run"]["observation_mode"] == TASK_AGENT_OBSERVATION_MODE


def test_scene_change_during_child_run_keeps_result_historical_and_stale(tmp_path: Path) -> None:
    scene_dir, task = _validated_scene(tmp_path)

    def child_executor(_scene, _task, _model, action_log, _frames, _emit):
        spec_path = scene_dir / "env_spec_3d.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        next(obj for obj in spec["objects"] if obj["id"] == "robot")["position"][1] += 0.25
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        action_log.write_text(
            json.dumps(
                {
                    "action": "controller",
                    "right": 0,
                    "forward": 1,
                    "camera_azimuth": -90,
                    "jump": False,
                    "frames": 140,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"exit_code": 0, "summary": "Reached it.", "stderr": []}

    result = run_validated_task(
        scene_dir=scene_dir,
        task_id=task["task_id"],
        child_executor=child_executor,
    )

    assert result["report"]["passed"] is True
    assert result["report"]["stale"] is True
    saved = read_task(scene_dir, task["task_id"], include_staleness=False)
    assert "latest_run" not in saved
    assert saved["run_summaries"][-1]["stale"] is True


def test_codex_events_stream_public_activity_without_reasoning() -> None:
    events: list[tuple[str, dict]] = []
    translator = TaskCodexEventTranslator(
        emit=lambda event, data: events.append((event, data)),
        task={"task_id": "task-1"},
        frame_url_prefix="/generated/runner/tasks/task-1/runs/run-1/child_frames",
    )

    translator.handle_line(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"id": "reason-1", "type": "reasoning", "text": "hidden reasoning"},
            }
        )
    )
    translator.handle_line(
        json.dumps(
            {
                "type": "item.completed",
                "item": {"id": "message-1", "type": "agent_message", "text": "I will inspect the route."},
            }
        )
    )
    translator.handle_line(
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "tool-1",
                    "type": "mcp_tool_call",
                    "tool": "observe_task_run",
                    "arguments": "{}",
                    "result": json.dumps(
                        {
                            "status": "success",
                            "observation_mode": TASK_AGENT_OBSERVATION_MODE,
                            "outcome": "running",
                            "steps_used": 24,
                            "steps_remaining": 476,
                            "grounded": True,
                            "collision": False,
                            "frame": {
                                "path": "/tmp/frames/frame_0003.png",
                                "renderer": "threejs_styled",
                            },
                        }
                    ),
                },
            }
        )
    )

    assert [event for event, _data in events] == ["text", "tool_start", "tool_result", "task_frame"]
    assert all("hidden reasoning" not in json.dumps(data) for _event, data in events)
    assert events[0][1]["delta"] == "I will inspect the route."
    assert events[0][1]["step"] == 0
    assert events[1][1]["step"] == 0
    assert events[2][1]["summary"] == {
        "status": "success",
        "observation_mode": TASK_AGENT_OBSERVATION_MODE,
        "outcome": "running",
        "step": 24,
        "steps_remaining": 476,
        "grounded": True,
        "collision": False,
        "terminal": False,
        "frame_url": "/generated/runner/tasks/task-1/runs/run-1/child_frames/frame_0003.png",
        "frame_renderer": "threejs_styled",
    }
    assert events[3][1]["url"].endswith("/child_frames/frame_0003.png")
    assert events[3][1]["renderer"] == "threejs_styled"
    assert events[2][1]["step"] == 24


def test_task_evidence_preserves_observation_renderer_and_camera(tmp_path: Path) -> None:
    source = tmp_path / "child_frames"
    source.mkdir()
    for index in range(2):
        (source / f"frame_{index:04d}.png").write_bytes(b"png")
    (source / "frames.jsonl").write_text(
        "\n".join(
            [
                json.dumps({
                    "path": str(source / "frame_0000.png"),
                    "step": 0,
                    "reset_count": 0,
                    "renderer": "threejs_styled",
                    "camera": {"azimuth": -90, "elevation": 0},
                }),
                json.dumps({
                    "path": str(source / "frame_0001.png"),
                    "step": 24,
                    "reset_count": 0,
                    "renderer": "mujoco_fallback",
                    "camera": {"azimuth": -60, "elevation": 10},
                }),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    evidence = _persist_evidence_frames(
        scene_dir=tmp_path / "scene",
        source_dir=source,
        target_dir=tmp_path / "evidence",
        task_id="task-1",
        run_id="run-1",
    )

    assert [item["renderer"] for item in evidence] == [
        "threejs_styled",
        "mujoco_fallback",
    ]
    assert [item["step"] for item in evidence] == [0, 24]
    assert evidence[1]["camera"]["azimuth"] == -60


def test_task_child_prompt_hides_verifier_and_structured_scene_state() -> None:
    prompt = _child_prompt(
        {
            "task_id": "private-verifier",
            "instruction": "Move the crate into the marked region.",
            "max_steps": 800,
            "tests": [
                {
                    "id": "secret_delivery_test",
                    "description": "Hidden exact overlap check.",
                    "conditions": [{"id": "secret_condition", "description": "Do not reveal this."}],
                }
            ],
        }
    )

    assert "Move the crate into the marked region." in prompt
    assert "first-person images" in prompt
    assert "earlier views remain in the conversation" in prompt
    assert "anonymous collision" in prompt
    assert "Visual vocabulary (appearance only, not extra task rules)" in prompt
    assert "Pushable boxes look like wooden supply crates." in prompt
    assert "Neutral target regions are low cyan square outlines" in prompt
    assert "Charging goals have a dark round plinth" in prompt
    assert "36-90 frame movement batches" in prompt
    assert "Hazards are black-and-yellow warning-marked floor zones" in prompt
    assert "coordinates" in prompt
    assert "bearings" in prompt
    assert "global preview" in prompt
    assert "intentionally omit" in prompt
    assert "secret_delivery_test" not in prompt
    assert "secret_condition" not in prompt
    assert "Hidden exact overlap check" not in prompt
    assert '"tests"' not in prompt


def test_task_child_command_disables_non_task_tools(tmp_path: Path) -> None:
    args = _build_task_agent_codex_args(
        task={
            "task_id": "visual-task",
            "instruction": "Reach the visible platform.",
            "max_steps": 400,
            "tests": [{"id": "hidden"}],
        },
        model="test-model",
        command="/usr/bin/python3",
        mcp_args=["-m", "environment_generation.mcp_server"],
        mcp_env={"ENVIRONMENT_GENERATION_TASK_JSON": "{}"},
        output_path=tmp_path / "last-message.txt",
    )

    assert "--ephemeral" in args
    assert args[args.index("-s") + 1] == "read-only"
    assert "--ignore-user-config" in args
    disabled = {
        args[index + 1]
        for index, value in enumerate(args[:-1])
        if value == "--disable"
    }
    assert {
        "apps",
        "browser_use",
        "browser_use_external",
        "computer_use",
        "plugins",
        "shell_tool",
        "unified_exec",
    } <= disabled
    assert '"id": "hidden"' not in args[-1]


def test_task_scene_frame_watcher_streams_only_valid_finite_transforms(tmp_path: Path) -> None:
    path = tmp_path / "scene_frames.jsonl"
    events: list[tuple[str, dict]] = []
    valid = {
        "total_step": 12,
        "simulation_time": 0.12,
        "status": "playing",
        "grounded": True,
        "objects": [
            {
                "id": "robot",
                "position": [1, 2, 0.6],
                "rotation_matrix": [1, 0, 0, 0, 1, 0, 0, 0, 1],
            }
        ],
        "mechanisms": [{"id": "door", "progress": 1.4, "active": True}],
    }
    path.write_text(json.dumps(valid) + "\nnot-json\n", encoding="utf-8")
    watcher = TaskSceneFrameWatcher(
        path=path,
        emit=lambda event, data: events.append((event, data)),
    )

    assert watcher.poll() == 1
    assert watcher.poll() == 0
    assert events == [
        (
            "task_scene_frame",
            {
                "step": 12,
                "simulation_time": 0.12,
                "status": "playing",
                "grounded": True,
                "objects": [
                    {
                        "id": "robot",
                        "position": [1.0, 2.0, 0.6],
                        "rotation_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    }
                ],
                "mechanisms": [
                    {
                        "id": "door",
                        "trigger_id": "",
                        "gate_id": "",
                        "active": True,
                        "progress": 1.0,
                    }
                ],
            },
        )
    ]

    invalid = {
        **valid,
        "total_step": 13,
        "objects": [
            {**valid["objects"][0], "position": [float("nan"), 0, 0]}
        ],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(invalid) + "\n")
    assert watcher.poll() == 0


def test_task_run_streams_and_persists_agent_activity(tmp_path: Path) -> None:
    scene_dir, task = _validated_scene(tmp_path)
    streamed: list[tuple[str, dict]] = []

    def child_executor(_scene, _task, _model, action_log, _frames, emit):
        assert emit is not None
        emit("text", {"delta": "I will move toward the goal."})
        emit(
            "tool_start",
            {
                "id": "tool-1",
                "name": "act_task_run",
                "displayName": "act_task_run",
                "input": {"forward": 1, "frames": 20},
                "message": "Controlling the robot...",
            },
        )
        emit(
            "tool_result",
            {
                "toolUseId": "tool-1",
                "name": "act_task_run",
                "displayName": "act_task_run",
                "isError": False,
                "summary": {"status": "success", "step": 20, "frames_advanced": 20},
                "message": "Advanced 20 frames at step 20.",
            },
        )
        action_log.write_text(
            json.dumps(
                {
                    "action": "controller",
                    "right": 0,
                    "forward": 0,
                    "camera_azimuth": -90,
                    "jump": False,
                    "frames": 20,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"exit_code": 0, "summary": "I will move toward the goal.", "stderr": []}

    result = run_validated_task(
        scene_dir=scene_dir,
        task_id=task["task_id"],
        child_executor=child_executor,
        emit=lambda event, data: streamed.append((event, data)),
    )

    assert [event for event, _data in streamed[:4]] == [
        "task_run",
        "text",
        "tool_start",
        "tool_result",
    ]
    run_dir = scene_dir / "tasks" / task["task_id"] / "runs" / result["run_id"]
    activity = json.loads((run_dir / "activity.json").read_text(encoding="utf-8"))
    assert activity["schema_version"] == "1.1"
    assert (run_dir / "activity.jsonl").is_file()
    assert [event["type"] for event in activity["events"]] == [
        "phase",
        "agent_message",
        "tool_start",
        "tool_result",
        "phase",
        "phase",
    ]
    assert [event["step"] for event in activity["events"]] == [0, 0, 0, 20, 20, 20]
    assert result["report"]["activity_url"].split("?", 1)[0].endswith(
        f"/{result['run_id']}/activity.json"
    )
    saved = read_task(scene_dir, task["task_id"])
    assert saved["latest_run"]["activity"][1]["message"] == "I will move toward the goal."


def test_task_run_history_keeps_each_replayable_trajectory(tmp_path: Path) -> None:
    scene_dir, task = _validated_scene(tmp_path)

    def child_executor(_scene, _task, _model, action_log, _frames, emit):
        assert emit is not None
        emit("text", {"delta": "Trying this route.", "step": 0})
        action_log.write_text(
            json.dumps(
                {
                    "action": "controller",
                    "right": 0,
                    "forward": 0,
                    "camera_azimuth": -90,
                    "jump": False,
                    "frames": 20,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"exit_code": 0, "summary": "Trying this route.", "stderr": []}

    first = run_validated_task(
        scene_dir=scene_dir,
        task_id=task["task_id"],
        child_executor=child_executor,
    )
    second = run_validated_task(
        scene_dir=scene_dir,
        task_id=task["task_id"],
        child_executor=child_executor,
    )

    saved = read_task(scene_dir, task["task_id"], include_staleness=False)
    runs = saved["run_summaries"]
    assert [run["run_id"] for run in runs] == [first["run_id"], second["run_id"]]
    assert len({run["trajectory_url"] for run in runs}) == 2
    assert all(
        run["activity_url"].split("?", 1)[0].endswith(f"/{run['run_id']}/activity.json")
        for run in runs
    )
    assert saved["latest_run"]["run_id"] == second["run_id"]
