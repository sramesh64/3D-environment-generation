"""Isolated Codex task runs with authoritative parent-process replay."""

from __future__ import annotations

import json
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .artifacts import ENV_SPEC_FILENAME, VISUAL_SCENE_FILENAME, WORLD_XML_FILENAME
from .env_tasks import (
    TASK_CONTROLLER_VERSION,
    read_task,
    task_artifact_url,
    task_definition_hash,
    task_dir,
    task_scene_hash,
    write_task,
)
from .runtime_config import runtime_env_key, runtime_env_value
from .task_agent import TASK_AGENT_OBSERVATION_MODE
from .task_oracle import replay_task_actions


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MCP_SERVER_NAME = "environment-generation"
TASK_RUNS_DIRNAME = "runs"
CHILD_TIMEOUT_SECONDS = 7 * 60
MAX_CHILD_OUTPUT_CHARS = 12_000
MAX_EVIDENCE_FRAMES = 6
MAX_ACTIVITY_EVENTS = 600
ACTIVITY_FILENAME = "activity.json"
ACTIVITY_LOG_FILENAME = "activity.jsonl"
TASK_AGENT_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "apps",
    "plugins",
)
TaskEmit = Callable[[str, dict[str, Any]], None]
ChildExecutor = Callable[[Path, dict[str, Any], str, Path, Path, TaskEmit | None], dict[str, Any]]


class TaskActivityRecorder:
    """Persist the public task-run trace while forwarding live events."""

    def __init__(self, *, run_dir: Path, emit: TaskEmit | None) -> None:
        self.run_dir = run_dir
        self.external_emit = emit
        self.events: list[dict[str, Any]] = []
        self.sequence = 0
        self.last_step = 0
        self.log_path = run_dir / ACTIVITY_LOG_FILENAME

    def emit(self, event: str, data: dict[str, Any]) -> None:
        self.record(event, data)
        _emit(self.external_emit, event, data)

    def record(self, event: str, data: dict[str, Any]) -> None:
        value = _public_task_activity_event(event, data, sequence=self.sequence + 1)
        if value is None:
            return
        step = _activity_event_step(value)
        if step is None:
            step = self.last_step
        else:
            self.last_step = max(self.last_step, step)
            step = self.last_step
        value["step"] = step
        self.sequence += 1
        self.events.append(value)
        if len(self.events) > MAX_ACTIVITY_EVENTS:
            self.events = self.events[-MAX_ACTIVITY_EVENTS:]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n")

    def has_agent_message(self, message: str) -> bool:
        normalized = " ".join(str(message or "").split())
        return any(
            event.get("type") == "agent_message"
            and " ".join(str(event.get("message") or "").split()) == normalized
            for event in self.events
        )

    def persist(self) -> Path:
        path = self.run_dir / ACTIVITY_FILENAME
        _write_json(
            path,
            {
                "schema_version": "1.1",
                "event_count": self.sequence,
                "events": self.events,
            },
        )
        return path


def run_validated_task(
    *,
    scene_dir: Path,
    task_id: str,
    model: str = "",
    emit: TaskEmit | None = None,
    child_executor: ChildExecutor | None = None,
) -> dict[str, Any]:
    """Run a validated task and score the child action log in a fresh simulation."""

    scene_dir = scene_dir.resolve()
    spec = _read_json(scene_dir / ENV_SPEC_FILENAME)
    if not spec:
        raise ValueError("finalized env_spec_3d.json is required before running a task")
    task = read_task(scene_dir, task_id, current_spec=spec)
    _require_validated_human_oracle(task)
    recover_abandoned_task_runs(scene_dir, task_id=task_id)
    if active := _active_manifest(scene_dir, task_id):
        raise ValueError(f"task run {active.get('run_id')} is already active")

    model = str(runtime_env_value("TASK_MODEL") or model or "").strip()
    run_id = _new_run_id(scene_dir, task_id)
    run_dir = task_dir(scene_dir, task_id) / TASK_RUNS_DIRNAME / run_id
    snapshot_dir = run_dir / "snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for filename in (ENV_SPEC_FILENAME, WORLD_XML_FILENAME, VISUAL_SCENE_FILENAME):
        source = scene_dir / filename
        if source.is_file():
            shutil.copy2(source, snapshot_dir / filename)
    _write_json(snapshot_dir / "task.json", task)

    manifest = {
        "schema_version": "1.0",
        "controller_version": TASK_CONTROLLER_VERSION,
        "run_id": run_id,
        "task_id": task_id,
        "env_id": str(task.get("env_id") or scene_dir.name),
        "created_at": _now(),
        "status": "running",
        "pid": os.getpid(),
        "env_spec_hash": task_scene_hash(spec),
        "task_definition_hash": task_definition_hash(task),
        "model": model or "default",
        "observation_mode": TASK_AGENT_OBSERVATION_MODE,
    }
    _write_json(run_dir / "manifest.json", manifest)
    activity = TaskActivityRecorder(run_dir=run_dir, emit=emit)
    activity.emit("task_run", {"status": "running", "manifest": manifest})

    action_log = run_dir / "child_actions.jsonl"
    child_frame_dir = run_dir / "child_frames"
    executor = child_executor or _run_codex_child
    try:
        child = executor(snapshot_dir, task, model, action_log, child_frame_dir, activity.emit)
        child_summary = str(child.get("summary") or "").strip()[:MAX_CHILD_OUTPUT_CHARS]
        if child_summary and not activity.has_agent_message(child_summary):
            activity.emit("text", {"delta": child_summary})
        actions = _read_action_log(action_log)
        _write_json(run_dir / "actions.json", {"actions": actions})
        manifest["status"] = "replaying"
        _write_json(run_dir / "manifest.json", manifest)
        activity.emit(
            "task_run",
            {"status": "replaying", "run_id": run_id, "action_count": len(actions)},
        )
        if not actions:
            report = _empty_run_report(task, child)
            trajectory: list[dict[str, Any]] = []
        else:
            replay = replay_task_actions(scene_dir=snapshot_dir, task=task, actions=actions)
            trajectory = replay.pop("trajectory", [])
            report = {
                **replay,
                "status": "passed" if replay.get("passed") else "failed",
                "child_summary": child_summary,
                "child_exit_code": int(child.get("exit_code") or 0),
                "child_stderr": [str(line)[:500] for line in child.get("stderr") or []][-20:],
            }
        _write_json(run_dir / "trajectory.json", {"task_id": task_id, "frames": trajectory})
        report.update(
            {
                "run_id": run_id,
                "task_id": task_id,
                "env_id": task.get("env_id"),
                "model": model or "default",
                "observation_mode": TASK_AGENT_OBSERVATION_MODE,
                "completed_at": _now(),
                "action_count": len(actions),
                "trajectory_url": task_artifact_url(scene_dir, run_dir / "trajectory.json"),
                "actions_url": task_artifact_url(scene_dir, run_dir / "actions.json"),
                "evidence_frames": _persist_evidence_frames(
                    scene_dir=scene_dir,
                    source_dir=child_frame_dir,
                    target_dir=run_dir / "evidence",
                    task_id=task_id,
                    run_id=run_id,
                ),
            }
        )
        activity.record(
            "task_run",
            {
                "status": report["status"],
                "run_id": run_id,
                "passed": bool(report.get("passed")),
                "step_count": int(report.get("step_count") or 0),
                "action_count": len(actions),
            },
        )
        activity_path = activity.persist()
        report["activity_url"] = task_artifact_url(scene_dir, activity_path)
        report["activity_count"] = activity.sequence
        current_spec = _read_json(scene_dir / ENV_SPEC_FILENAME)
        try:
            current_task = read_task(scene_dir, task_id, include_staleness=False)
        except (FileNotFoundError, ValueError):
            current_task = None
        stale = (
            current_spec is None
            or current_task is None
            or task_scene_hash(current_spec) != manifest["env_spec_hash"]
            or task_definition_hash(current_task) != manifest["task_definition_hash"]
        )
        report["stale"] = stale
        _write_json(run_dir / "report.json", report)
        report["report_url"] = task_artifact_url(scene_dir, run_dir / "report.json")

        if current_task is not None:
            summary = {
                "run_id": run_id,
                "created_at": manifest["created_at"],
                "completed_at": report["completed_at"],
                "status": report["status"],
                "passed": bool(report.get("passed")),
                "stale": stale,
                "step_count": int(report.get("step_count") or 0),
                "action_count": len(actions),
                "model": model or "default",
                "observation_mode": TASK_AGENT_OBSERVATION_MODE,
                "report_url": report["report_url"],
                "trajectory_url": report["trajectory_url"],
                "evidence_frames": report["evidence_frames"],
                "activity_url": report["activity_url"],
                "activity_count": report["activity_count"],
            }
            summaries = list(current_task.get("run_summaries") or [])
            summaries.append(summary)
            current_task["run_summaries"] = summaries[-20:]
            if not stale:
                current_task["latest_run"] = {
                    **summary,
                    "activity": activity.events[-80:],
                }
            write_task(scene_dir, current_task)

        manifest.update(
            {
                "status": report["status"],
                "completed_at": report["completed_at"],
                "stale": stale,
                "pid": None,
            }
        )
        _write_json(run_dir / "manifest.json", manifest)
        _emit(emit, "task_run", {"status": report["status"], "report": report})
        return {"status": "success", "run_id": run_id, "report": report, "manifest": manifest}
    except Exception as exc:
        manifest.update({"status": "error", "completed_at": _now(), "error": str(exc), "pid": None})
        _write_json(run_dir / "manifest.json", manifest)
        activity.record("task_run", {"status": "error", "run_id": run_id, "error": str(exc)})
        activity.persist()
        _emit(emit, "task_run", {"status": "error", "run_id": run_id, "error": str(exc)})
        raise


def recover_abandoned_task_runs(scene_dir: Path, *, task_id: str | None = None) -> None:
    roots = [task_dir(scene_dir, task_id)] if task_id else list((scene_dir / "tasks").glob("*"))
    for root in roots:
        runs = root / TASK_RUNS_DIRNAME
        if not runs.is_dir():
            continue
        for path in runs.glob("*/manifest.json"):
            manifest = _read_json(path)
            if not manifest or manifest.get("status") not in {"running", "replaying"}:
                continue
            if _process_is_alive(manifest.get("pid")):
                continue
            manifest.update(
                {
                    "status": "error",
                    "completed_at": _now(),
                    "error": "Studio restarted before this task run completed.",
                    "pid": None,
                }
            )
            _write_json(path, manifest)


def _require_validated_human_oracle(task: dict[str, Any]) -> None:
    if task.get("effective_status") != "validated":
        raise ValueError("record a current passing oracle before running this task")
    oracle = task.get("oracle") or {}
    if oracle.get("provenance") != "human_recording":
        raise ValueError("task validation requires a human-recorded oracle")
    if oracle.get("env_spec_hash") != task.get("env_spec_hash"):
        raise ValueError("the task oracle is stale for the current environment")
    if oracle.get("task_definition_hash") != task.get("task_definition_hash"):
        raise ValueError("the task oracle is stale for the current trajectory tests")


class TaskCodexEventTranslator:
    """Translate Codex JSONL into the public task activity stream."""

    def __init__(
        self,
        *,
        emit: TaskEmit | None,
        task: dict[str, Any],
        frame_url_prefix: str,
    ) -> None:
        self.emit = emit
        self.task = task
        self.frame_url_prefix = frame_url_prefix.rstrip("/")
        self.started_tools: set[str] = set()
        self.tool_inputs: dict[str, dict[str, Any]] = {}
        self.agent_messages: list[str] = []
        self.last_step = 0

    def handle_line(self, line: str) -> None:
        value = line.strip()
        if not value:
            return
        try:
            event = json.loads(value)
        except json.JSONDecodeError:
            return
        event_type = str(event.get("type") or "")
        if event_type in {"thread.started", "session.created", "turn.completed"}:
            return
        if event_type in {"item.started", "item.updated", "item.completed"}:
            item = event.get("item") if isinstance(event.get("item"), dict) else {}
            self._handle_item(event_type.rsplit(".", 1)[-1], item)
            return
        if event_type in {"turn.failed", "error"}:
            message = str(event.get("message") or event.get("error") or "Agent run failed.")
            _emit(self.emit, "task_error", {"message": message[:1000], "step": self.last_step})

    def _handle_item(self, phase: str, item: dict[str, Any]) -> None:
        item_type = str(item.get("type") or item.get("item_type") or "")
        if "reasoning" in item_type.lower():
            return
        if item_type in {"agent_message", "assistant", "assistant_message"}:
            if phase != "completed":
                return
            message = _codex_item_text(item).strip()
            if not message:
                return
            self.agent_messages.append(message)
            _emit(
                self.emit,
                "text",
                {"delta": message[:MAX_CHILD_OUTPUT_CHARS], "step": self.last_step},
            )
            return
        if item_type == "command_execution":
            item_id = str(item.get("id") or f"tool-{uuid.uuid4().hex[:8]}")
            arguments = {"command": str(item.get("command") or "")[:1000]}
            if item_id not in self.started_tools:
                self.started_tools.add(item_id)
                self.tool_inputs[item_id] = arguments
                _emit(
                    self.emit,
                    "tool_start",
                    {
                        "id": item_id,
                        "name": "shell",
                        "displayName": "shell",
                        "input": arguments,
                        "message": "Inspecting the task workspace...",
                        "step": self.last_step,
                    },
                )
            if phase == "completed":
                is_error = bool(item.get("exit_code"))
                _emit(
                    self.emit,
                    "tool_result",
                    {
                        "toolUseId": item_id,
                        "name": "shell",
                        "displayName": "shell",
                        "isError": is_error,
                        "summary": {"status": "error" if is_error else "success"},
                        "message": "Workspace inspection failed." if is_error else "Workspace inspection finished.",
                        "step": self.last_step,
                    },
                )
            return
        if item_type not in {"mcp_tool_call", "tool_call"}:
            return
        item_id = str(item.get("id") or f"tool-{uuid.uuid4().hex[:8]}")
        name = str(item.get("tool") or item.get("tool_name") or item.get("name") or "task_tool")
        arguments = _tool_arguments(item.get("arguments", item.get("input", {})))
        if item_id not in self.started_tools:
            self.started_tools.add(item_id)
            self.tool_inputs[item_id] = arguments
            _emit(
                self.emit,
                "tool_start",
                {
                    "id": item_id,
                    "name": name,
                    "displayName": name,
                    "input": arguments,
                    "message": _task_tool_start_message(name),
                    "step": self.last_step,
                },
            )
        if phase != "completed":
            return
        payload = _tool_result_payload(
            item.get("result", item.get("output", item.get("aggregated_output")))
        )
        is_error = item.get("status") in {"failed", "error"} or bool(item.get("error"))
        summary = _task_tool_result_summary(
            name=name,
            payload=payload,
            arguments=self.tool_inputs.get(item_id, arguments),
            previous_step=self.last_step,
            frame_url_prefix=self.frame_url_prefix,
        )
        if summary.get("step") is not None:
            self.last_step = max(self.last_step, int(summary["step"]))
        message = _task_tool_result_message(name=name, summary=summary, is_error=is_error)
        _emit(
            self.emit,
            "tool_result",
            {
                "toolUseId": item_id,
                "name": name,
                "displayName": name,
                "isError": is_error,
                "summary": summary,
                "message": message,
                "step": self.last_step,
            },
        )
        if summary.get("frame_url"):
            _emit(
                self.emit,
                "task_frame",
                {
                    "url": summary["frame_url"],
                    "step": summary.get("step"),
                    "reset_count": summary.get("resets_used"),
                    "renderer": summary.get("frame_renderer"),
                },
            )


class TaskSceneFrameWatcher:
    """Tail replay-format task states and publish them to the styled preview."""

    def __init__(self, *, path: Path, emit: TaskEmit | None) -> None:
        self.path = path
        self.emit = emit
        self.offset = 0
        self.buffer = ""

    def poll(self) -> int:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                handle.seek(self.offset)
                chunk = handle.read()
                self.offset = handle.tell()
        except OSError:
            return 0
        if not chunk:
            return 0
        self.buffer += chunk
        lines = self.buffer.split("\n")
        self.buffer = lines.pop()
        emitted = 0
        for line in lines:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            frame = _normalize_task_scene_frame(raw)
            if frame is None:
                continue
            _emit(self.emit, "task_scene_frame", frame)
            emitted += 1
        return emitted


def _codex_item_text(item: dict[str, Any]) -> str:
    for key in ("text", "message"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    content = item.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(value.get("text") or "")
            for value in content
            if isinstance(value, dict) and value.get("text")
        )
    return ""


def _tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value[:1000]}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {}


def _tool_result_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, list):
            text = "".join(
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
            parsed = _json_object(text)
            if parsed is not None:
                return _tool_result_payload(parsed)
        return value
    if isinstance(value, str):
        parsed = _json_object(value)
        return _tool_result_payload(parsed) if parsed is not None else {"message": value[:1000]}
    return {}


def _json_object(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _task_tool_result_summary(
    *,
    name: str,
    payload: dict[str, Any],
    arguments: dict[str, Any],
    previous_step: int,
    frame_url_prefix: str,
) -> dict[str, Any]:
    step = _optional_int(payload.get("steps_used", payload.get("step_count")))
    summary: dict[str, Any] = {
        "status": str(payload.get("status") or "success"),
        "observation_mode": str(payload.get("observation_mode") or ""),
        "outcome": str(payload.get("outcome") or ""),
        "step": step,
        "steps_remaining": _optional_int(payload.get("steps_remaining")),
        "resets_used": _optional_int(payload.get("resets_used", payload.get("reset_count"))),
        "terminal": bool(payload.get("terminal")),
        "termination_reason": str(payload.get("termination_reason") or ""),
    }
    if isinstance(payload.get("grounded"), bool):
        summary["grounded"] = payload["grounded"]
    if isinstance(payload.get("collision"), bool):
        summary["collision"] = payload["collision"]
    if isinstance(payload.get("passed"), bool):
        summary["passed"] = payload["passed"]
    if name == "act_task_run":
        last_action = payload.get("last_action") if isinstance(payload.get("last_action"), dict) else {}
        summary["frames_advanced"] = max(
            0,
            (step - previous_step)
            if step is not None
            else int(_optional_int(arguments.get("frames")) or 0),
        )
        summary["jump"] = bool(arguments.get("jump"))
        if isinstance(last_action.get("stopped_on_collision"), bool):
            summary["stopped_on_collision"] = last_action["stopped_on_collision"]
    report = payload.get("tests") if isinstance(payload.get("tests"), dict) else payload.get("report")
    if isinstance(report, dict):
        tests = [value for value in report.get("tests") or [] if isinstance(value, dict)]
        report_summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        summary["tests_total"] = int(report_summary.get("tests") or len(tests))
        summary["tests_passed"] = int(
            report_summary.get("passed_tests")
            or sum(1 for value in tests if value.get("passed") is True)
        )
        summary["passed"] = bool(report.get("passed"))
    frame = payload.get("frame") if isinstance(payload.get("frame"), dict) else {}
    frame_path = str(frame.get("path") or payload.get("path") or "")
    if frame_path.lower().endswith(".png"):
        summary["frame_url"] = f"{frame_url_prefix}/{Path(frame_path).name}"
        summary["frame_renderer"] = str(frame.get("renderer") or "")
    return {key: value for key, value in summary.items() if value not in {None, ""}}


def _task_tool_start_message(name: str) -> str:
    return {
        "start_task_run": "Starting the isolated task attempt...",
        "observe_task_run": "Observing the environment...",
        "act_task_run": "Controlling the robot...",
        "reset_task_run": "Resetting for another attempt...",
        "stop_task_run": "Checking the final task result...",
    }.get(name, f"Using {name}...")


def _task_tool_result_message(*, name: str, summary: dict[str, Any], is_error: bool) -> str:
    if is_error:
        return f"{name} failed."
    step = summary.get("step")
    progress = ""
    if summary.get("tests_total") is not None:
        progress = f" · {summary.get('tests_passed', 0)}/{summary['tests_total']} tests passing"
    if name == "start_task_run":
        return f"Task attempt started{f' at step {step}' if step is not None else ''}{progress}."
    if name == "observe_task_run":
        return f"Observed the environment{f' at step {step}' if step is not None else ''}{progress}."
    if name == "act_task_run":
        action = "Jumped and advanced" if summary.get("jump") else "Advanced"
        collision = " · stopped at collision" if summary.get("stopped_on_collision") else ""
        return f"{action} {summary.get('frames_advanced', 0)} frames{f' · step {step}' if step is not None else ''}{collision}{progress}."
    if name == "reset_task_run":
        attempt = int(summary.get("resets_used") or 0) + 1
        return f"Started attempt {attempt}{f' · step {step}' if step is not None else ''}."
    if name == "stop_task_run":
        state = "passed" if summary.get("passed") else "did not pass"
        return f"Task {state}{f' after {step} steps' if step is not None else ''}{progress}."
    return f"{name} finished{f' at step {step}' if step is not None else ''}."


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _activity_event_step(value: dict[str, Any]) -> int | None:
    direct = _optional_int(value.get("step"))
    if direct is not None:
        return max(0, direct)
    summary = value.get("summary") if isinstance(value.get("summary"), dict) else {}
    nested = _optional_int(summary.get("step"))
    if nested is not None:
        return max(0, nested)
    phase_step = _optional_int(value.get("step_count"))
    return max(0, phase_step) if phase_step is not None else None


def _child_frame_url_prefix(*, task: dict[str, Any], frame_dir: Path) -> str:
    return (
        f"/generated/{task.get('env_id')}/tasks/{task.get('task_id')}/"
        f"{TASK_RUNS_DIRNAME}/{frame_dir.parent.name}/{frame_dir.name}"
    )


def _normalize_task_scene_frame(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    objects = []
    for raw in value.get("objects") or []:
        if not isinstance(raw, dict) or len(objects) >= 256:
            continue
        object_id = str(raw.get("id") or "")[:160]
        position = _finite_vector(raw.get("position"), length=3)
        rotation = _finite_vector(raw.get("rotation_matrix"), length=9)
        if not object_id or position is None or rotation is None:
            continue
        objects.append({"id": object_id, "position": position, "rotation_matrix": rotation})
    if not objects:
        return None
    mechanisms = []
    for raw in value.get("mechanisms") or []:
        if not isinstance(raw, dict) or len(mechanisms) >= 64:
            continue
        mechanism_id = str(raw.get("id") or "")[:160]
        if not mechanism_id:
            continue
        progress = raw.get("progress")
        try:
            progress_value = float(progress)
        except (TypeError, ValueError):
            progress_value = 0.0
        if not math.isfinite(progress_value):
            progress_value = 0.0
        mechanisms.append(
            {
                "id": mechanism_id,
                "trigger_id": str(raw.get("trigger_id") or "")[:160],
                "gate_id": str(raw.get("gate_id") or "")[:160],
                "active": bool(raw.get("active")),
                "progress": max(0.0, min(1.0, progress_value)),
            }
        )
    return {
        "step": int(_optional_int(value.get("total_step", value.get("step"))) or 0),
        "simulation_time": _finite_float(value.get("simulation_time"), fallback=0.0),
        "status": str(value.get("status") or "")[:120],
        "grounded": bool(value.get("grounded")),
        "objects": objects,
        "mechanisms": mechanisms,
    }


def _finite_vector(value: Any, *, length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != length:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _finite_float(value: Any, *, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if math.isfinite(result) else fallback


def _run_codex_child(
    scene_dir: Path,
    task: dict[str, Any],
    model: str,
    action_log: Path,
    frame_dir: Path,
    emit: TaskEmit | None,
) -> dict[str, Any]:
    if shutil.which("codex") is None:
        raise RuntimeError("'codex' is not on PATH; install/login to the Codex CLI first")
    with tempfile.TemporaryDirectory(prefix="environment-generation-task-child-") as temporary:
        temp_dir = Path(temporary)
        output_path = temp_dir / "last_message.txt"
        command = sys.executable
        mcp_args = ["-m", "environment_generation.mcp_server"]
        mcp_env = {
            "PYTHONPATH": str(PROJECT_ROOT),
            runtime_env_key("TASK_SCENE_DIR"): str(scene_dir),
            runtime_env_key("TASK_JSON"): json.dumps(task, separators=(",", ":")),
            runtime_env_key("TASK_ACTION_LOG"): str(action_log),
            runtime_env_key("TASK_FRAME_DIR"): str(frame_dir),
            runtime_env_key("TASK_LIVE_STATE_LOG"): str(
                action_log.with_name("child_scene_frames.jsonl")
            ),
        }
        args = _build_task_agent_codex_args(
            task=task,
            model=model,
            command=command,
            mcp_args=mcp_args,
            mcp_env=mcp_env,
            output_path=output_path,
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        translator = TaskCodexEventTranslator(
            emit=emit,
            task=task,
            frame_url_prefix=_child_frame_url_prefix(task=task, frame_dir=frame_dir),
        )
        scene_watcher = TaskSceneFrameWatcher(
            path=action_log.with_name("child_scene_frames.jsonl"),
            emit=emit,
        )
        proc = subprocess.Popen(
            args,
            cwd=str(temp_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdout_queue: queue.Queue[str | None] = queue.Queue()
        stderr: list[str] = []

        def read_stdout() -> None:
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    stdout_queue.put(line)
            finally:
                stdout_queue.put(None)

        def read_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                value = line.rstrip()
                stderr.append(value)
                _emit(emit, "stderr", {"line": value[:1000]})

        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()
        deadline = time.monotonic() + CHILD_TIMEOUT_SECONDS
        timed_out = False
        try:
            while True:
                scene_watcher.poll()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    proc.terminate()
                    break
                try:
                    line = stdout_queue.get(timeout=min(0.25, remaining))
                except queue.Empty:
                    continue
                if line is None:
                    break
                translator.handle_line(line)
                scene_watcher.poll()
        except BaseException:
            _stop_child_process(proc)
            raise

        if timed_out:
            _stop_child_process(proc)
            exit_code = 124
            stderr.append(f"Child task run exceeded {CHILD_TIMEOUT_SECONDS} seconds.")
            _emit(
                emit,
                "task_error",
                {"message": f"Agent run exceeded the {CHILD_TIMEOUT_SECONDS}-second limit."},
            )
        else:
            try:
                exit_code = int(proc.wait(timeout=10))
            except subprocess.TimeoutExpired:
                _stop_child_process(proc)
                exit_code = 124
                stderr.append("Child task process did not exit after closing its event stream.")
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)
        scene_watcher.poll()
        if exit_code != 0 and not timed_out:
            _emit(
                emit,
                "task_error",
                {"message": stderr[-1] if stderr else f"Agent exited with status {exit_code}."},
            )
        try:
            summary = output_path.read_text(encoding="utf-8").strip()
        except OSError:
            summary = "\n".join(translator.agent_messages).strip()
        return {
            "exit_code": exit_code,
            "summary": summary[:MAX_CHILD_OUTPUT_CHARS],
            "stderr": stderr[-20:],
        }


def _stop_child_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _build_task_agent_codex_args(
    *,
    task: dict[str, Any],
    model: str,
    command: str,
    mcp_args: list[str],
    mcp_env: dict[str, str],
    output_path: Path,
) -> list[str]:
    args = [
        "codex",
        "exec",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--strict-config",
        "-s",
        "read-only",
        "-c",
        'approval_policy="never"',
        "-c",
        f"mcp_servers.{MCP_SERVER_NAME}.command={_toml_value(command)}",
        "-c",
        f"mcp_servers.{MCP_SERVER_NAME}.args={_toml_value(mcp_args)}",
        "-c",
        f'mcp_servers.{MCP_SERVER_NAME}.default_tools_approval_mode="approve"',
    ]
    for feature in TASK_AGENT_DISABLED_FEATURES:
        args.extend(["--disable", feature])
    for key, value in mcp_env.items():
        args.extend(["-c", f"mcp_servers.{MCP_SERVER_NAME}.env.{key}={_toml_value(value)}"])
    if model:
        args.extend(["-m", model])
    args.extend(["--output-last-message", str(output_path), _child_prompt(task)])
    return args


def _child_prompt(task: dict[str, Any]) -> str:
    evidence = json.dumps(
        {
            "task_id": task.get("task_id"),
            "instruction": task.get("instruction"),
            "max_steps": task.get("max_steps"),
        },
        indent=2,
    )
    return f"""\
You control one immutable Environment Generation benchmark task.

Task:
{evidence}

Visual vocabulary (appearance only, not extra task rules):
- Pushable boxes look like wooden supply crates.
- Neutral target regions are low cyan square outlines with a crossed diamond/package
  mark. They never have a tall beacon.
- Charging goals have a dark round plinth, bright gold rings, and a tall cyan beacon
  capped by a floating diamond. This is the only goal appearance.
- Hazards are black-and-yellow warning-marked floor zones and may look cracked,
  spiked, or liquid-filled.
- Floor switches are low yellow buttons; linked gates are tall brown barriers.

Use only start_task_run, observe_task_run, act_task_run, reset_task_run, and
stop_task_run. You cannot inspect or edit the workspace, scene, task, or hidden
tests. Start by observing and solve the task from the attached first-person images.
Each action returns one current view; earlier views remain in the conversation.
Compare them with the six recent actions to recognize landmarks, repeated locations,
and unproductive loops.
Observation responses intentionally omit coordinates, object labels, zone
identities, target bearings, mechanism state, maps, the global preview, and test
progress. They expose only human-like anonymous collision, grounded/airborne, and
zone-entry cues. Infer the layout and results of your actions visually. Prefer
36-90 frame movement batches over clear ground; use 6-24 frames near turns,
obstacles, hazards, movable objects, and target boundaries. Correct heading before
longer movement and settle before precise contacts. Do not spend repeated turns on
small forward steps or single-angle scans when the route is visibly clear. A movement action may
stop early when a solid collision blocks it; inspect the returned view and change
direction instead of repeating the same input. Follow only the task instruction;
visible goals, hazards, switches, and target regions have no extra implied rule
unless the instruction says so. The hidden deterministic evaluator reports only
whether the episode is still running or has ended.
You may reset only to begin a genuinely different attempt; a reset discards prior
attempt progress for final scoring. Stop when the episode reports a terminal
outcome, or when the bounded attempts are exhausted. End with a short factual
summary based only on visible evidence and the public outcome.
"""


def _empty_run_report(task: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "controller_version": TASK_CONTROLLER_VERSION,
        "observation_mode": TASK_AGENT_OBSERVATION_MODE,
        "env_spec_hash": task.get("env_spec_hash"),
        "task_definition_hash": task.get("task_definition_hash"),
        "status": "error",
        "passed": False,
        "step_count": 0,
        "tests": [],
        "summary": {"tests": len(task.get("tests") or []), "passed_tests": 0},
        "child_summary": str(child.get("summary") or "")[:MAX_CHILD_OUTPUT_CHARS],
        "child_exit_code": int(child.get("exit_code") or 1),
        "child_stderr": [str(line)[:500] for line in child.get("stderr") or []][-20:],
        "error": "The child agent produced no controller actions.",
    }


def _persist_evidence_frames(
    *,
    scene_dir: Path,
    source_dir: Path,
    target_dir: Path,
    task_id: str,
    run_id: str,
) -> list[dict[str, Any]]:
    if not source_dir.is_dir():
        return []
    manifest_by_name = _frame_manifest_by_name(source_dir / "frames.jsonl")
    paths = sorted(source_dir.glob("*.png"))
    if len(paths) > MAX_EVIDENCE_FRAMES:
        indexes = {
            round(index * (len(paths) - 1) / (MAX_EVIDENCE_FRAMES - 1))
            for index in range(MAX_EVIDENCE_FRAMES)
        }
        paths = [path for index, path in enumerate(paths) if index in indexes]
    target_dir.mkdir(parents=True, exist_ok=True)
    values = []
    for index, source in enumerate(paths):
        target = target_dir / f"frame_{index:03d}.png"
        shutil.copy2(source, target)
        source_metadata = manifest_by_name.get(source.name) or {}
        value = {
            "index": index,
            "url": (
                f"/generated/{scene_dir.name}/tasks/{task_id}/{TASK_RUNS_DIRNAME}/"
                f"{run_id}/evidence/{target.name}"
            ),
        }
        for key in ("step", "reset_count", "renderer", "camera"):
            if key in source_metadata:
                value[key] = source_metadata[key]
        values.append(value)
    return values


def _frame_manifest_by_name(path: Path) -> dict[str, dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    values: dict[str, dict[str, Any]] = {}
    for line in lines:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        name = Path(str(raw.get("path") or "")).name
        if name:
            values[name] = raw
    return values


def _read_action_log(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    values = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _active_manifest(scene_dir: Path, task_id: str) -> dict[str, Any] | None:
    root = task_dir(scene_dir, task_id) / TASK_RUNS_DIRNAME
    if not root.is_dir():
        return None
    for path in root.glob("*/manifest.json"):
        manifest = _read_json(path)
        if (
            manifest
            and manifest.get("status") in {"running", "replaying"}
            and _process_is_alive(manifest.get("pid"))
        ):
            return manifest
    return None


def _new_run_id(scene_dir: Path, task_id: str) -> str:
    root = task_dir(scene_dir, task_id) / TASK_RUNS_DIRNAME
    index = len([path for path in root.iterdir() if path.is_dir()]) + 1 if root.is_dir() else 1
    return f"run-{index:04d}-{uuid.uuid4().hex[:8]}"


def _process_is_alive(value: Any) -> bool:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _toml_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ",".join(_toml_value(item) for item in value) + "]"
    return json.dumps(str(value))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _public_task_activity_event(
    event: str,
    data: dict[str, Any],
    *,
    sequence: int,
) -> dict[str, Any] | None:
    base = {
        "sequence": sequence,
        "timestamp": _now(),
        "event": event,
    }
    if event == "text":
        message = str(data.get("delta") or "").strip()
        if not message:
            return None
        return {
            **base,
            "type": "agent_message",
            "label": "Agent note",
            "message": message,
            "step": data.get("step"),
        }
    if event == "tool_start":
        return {
            **base,
            "type": "tool_start",
            "id": str(data.get("id") or ""),
            "name": str(data.get("displayName") or data.get("name") or "task tool"),
            "label": "Action",
            "message": str(data.get("message") or ""),
            "input": data.get("input") if isinstance(data.get("input"), dict) else {},
            "step": data.get("step"),
        }
    if event == "tool_result":
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        return {
            **base,
            "type": "tool_result",
            "id": str(data.get("toolUseId") or ""),
            "name": str(data.get("displayName") or data.get("name") or "task tool"),
            "label": "Result",
            "message": str(data.get("message") or ""),
            "isError": bool(data.get("isError")),
            "step": data.get("step"),
            "summary": {
                key: summary[key]
                for key in (
                    "status",
                    "step",
                    "steps_remaining",
                    "resets_used",
                    "frames_advanced",
                    "grounded",
                    "collision",
                    "stopped_on_collision",
                    "tests_total",
                    "tests_passed",
                    "passed",
                    "terminal",
                    "termination_reason",
                    "frame_url",
                )
                if key in summary
            },
        }
    if event == "task_frame":
        return {
            **base,
            "type": "frame",
            "label": "Policy view",
            "message": "Updated first-person observation.",
            "url": str(data.get("url") or ""),
            "step": data.get("step"),
            "reset_count": data.get("reset_count"),
        }
    if event == "task_error":
        return {
            **base,
            "type": "error",
            "label": "Agent error",
            "message": str(data.get("message") or "Agent run failed."),
            "step": data.get("step"),
        }
    if event == "task_run":
        manifest = data.get("manifest") if isinstance(data.get("manifest"), dict) else {}
        return {
            **base,
            "type": "phase",
            "label": "Task run",
            "message": str(data.get("status") or "running").replace("_", " "),
            "run_id": str(data.get("run_id") or manifest.get("run_id") or ""),
            "status": str(data.get("status") or ""),
            "step_count": data.get("step_count"),
            "action_count": data.get("action_count"),
            "passed": data.get("passed"),
            "step": data.get("step", data.get("step_count")),
        }
    return None


def _emit(emit: TaskEmit | None, event: str, data: dict[str, Any]) -> None:
    if emit is not None:
        try:
            emit(event, data)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
