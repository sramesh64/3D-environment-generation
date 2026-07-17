"""Fixed-step interactive oracle recording and authoritative task replay."""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .env_tasks import (
    TASK_CONTROLLER_VERSION,
    TaskDefinitionError,
    new_attempt_id,
    normalize_task_definition,
    list_tasks,
    read_task,
    task_artifact_url,
    task_definition_hash,
    task_dir,
    task_scene_hash,
    with_task_staleness,
    write_task,
)
from .episode_runtime import decide_episode
from .player import PlayableSimulation
from .trajectory_assertions import TrajectoryRecorder


ORACLE_SESSION_TTL_SECONDS = 30 * 60
MAX_ORACLE_STEP_FRAMES = 12
MAX_ORACLE_ACTION_SEGMENTS = 5_000
ORACLE_SCHEMA_VERSION = "1.0"


@dataclass
class TaskOracleSession:
    id: str
    scene_dir: Path
    task: dict[str, Any]
    simulation: PlayableSimulation
    clock: Callable[[], float]
    lock: threading.Lock = field(default_factory=threading.Lock)
    created_at: float = field(init=False)
    last_access_at: float = field(init=False)
    actions: list[dict[str, Any]] = field(default_factory=list)
    reset_count: int = field(default=0)
    terminal_reason: str = field(default="")
    live_report: dict[str, Any] | None = field(default=None, init=False)
    live_report_step: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        now = self.clock()
        self.created_at = now
        self.last_access_at = now
        self.recorder = TrajectoryRecorder(self.simulation)

    def advance(
        self,
        *,
        right: float,
        forward: float,
        camera_azimuth: float,
        jump: bool,
        frames: int,
        evaluate_report: bool = False,
    ) -> dict[str, Any]:
        with self.lock:
            if self.terminal_reason:
                self.last_access_at = self.clock()
                return self.snapshot(evaluate_report=True)
            remaining = int(self.task["max_steps"]) - int(self.simulation.step_count)
            frames_to_advance = min(frames, max(0, remaining))
            advanced = 0
            for _ in range(frames_to_advance):
                self.simulation.step(
                    right=right,
                    forward=forward,
                    camera_azimuth=camera_azimuth,
                    jump=jump,
                )
                self.recorder.capture(reset_count=0)
                advanced += 1
                decision = decide_episode(
                    safety_failure=self.simulation.safety_failure_reason(),
                )
                if decision.terminal:
                    self.terminal_reason = decision.reason
                    break
            if self.simulation.step_count >= int(self.task["max_steps"]) and not self.terminal_reason:
                self.terminal_reason = "step_budget"
            if advanced:
                self._record_action(
                    right=right,
                    forward=forward,
                    camera_azimuth=camera_azimuth,
                    jump=jump,
                    frames=advanced,
                )
            self.last_access_at = self.clock()
            return self.snapshot(evaluate_report=evaluate_report or bool(self.terminal_reason))

    def reset(self) -> dict[str, Any]:
        with self.lock:
            self.simulation.reset()
            self.recorder = TrajectoryRecorder(self.simulation)
            self.actions = []
            self.reset_count += 1
            self.terminal_reason = ""
            self.live_report = None
            self.live_report_step = -1
            self.last_access_at = self.clock()
            return self.snapshot(evaluate_report=True)

    def snapshot(self, *, evaluate_report: bool = False) -> dict[str, Any]:
        if evaluate_report or self.live_report is None:
            self.live_report = self.recorder.report(task=self.task, final=False)
            self.live_report_step = int(self.simulation.step_count)
        report = self.live_report
        report_current = self.live_report_step == int(self.simulation.step_count)
        return {
            "session_id": self.id,
            "task_id": self.task["task_id"],
            "env_id": self.task["env_id"],
            "status": "failed" if self.terminal_reason else "recording",
            "terminal_reason": self.terminal_reason,
            "ready_to_validate": bool(report_current and report["passed"] and self.actions),
            "report_current": report_current,
            "report_step": self.live_report_step,
            "has_actions": bool(self.actions),
            "simulation_time": float(self.simulation.data.time),
            "timestep": float(self.simulation.model.opt.timestep),
            "steps": int(self.simulation.step_count),
            "steps_remaining": max(0, int(self.task["max_steps"]) - int(self.simulation.step_count)),
            "reset_count": self.reset_count,
            "grounded": self.simulation.is_grounded(),
            "agent_id": self.simulation.agent.object_id,
            "objects": self.simulation.body_transforms(),
            "mechanisms": self.simulation.mechanism_states(),
            "report": report,
        }

    def _record_action(
        self,
        *,
        right: float,
        forward: float,
        camera_azimuth: float,
        jump: bool,
        frames: int,
    ) -> None:
        action = {
            "right": round(float(right), 6),
            "forward": round(float(forward), 6),
            "camera_azimuth": round(float(camera_azimuth), 6),
            "jump": bool(jump),
            "frames": int(frames),
        }
        if self.actions and all(
            self.actions[-1].get(key) == action[key]
            for key in ("right", "forward", "camera_azimuth", "jump")
        ):
            self.actions[-1]["frames"] += int(frames)
            return
        if len(self.actions) >= MAX_ORACLE_ACTION_SEGMENTS:
            self.terminal_reason = "action_segment_limit"
            return
        self.actions.append(action)


class TaskOracleSessionManager:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        ttl_seconds: float = ORACLE_SESSION_TTL_SECONDS,
    ) -> None:
        self.clock = clock
        self.ttl_seconds = ttl_seconds
        self._sessions: dict[str, TaskOracleSession] = {}
        self._lock = threading.Lock()

    def start(self, *, scene_dir: Path, task_id: str) -> dict[str, Any]:
        scene_dir = scene_dir.resolve()
        task = _prepare_task_for_recording(scene_dir, task_id)
        simulation = PlayableSimulation.from_scene(scene_dir)
        session = TaskOracleSession(
            id=uuid.uuid4().hex,
            scene_dir=scene_dir,
            task=task,
            simulation=simulation,
            clock=self.clock,
        )
        with self._lock:
            self._remove_expired_locked()
            active = next(
                (
                    value
                    for value in self._sessions.values()
                    if value.scene_dir == scene_dir and value.task["task_id"] == task_id
                ),
                None,
            )
            if active is not None:
                raise ValueError("an oracle recording is already active for this task")
            self._sessions[session.id] = session
        task["status"] = "recording"
        task["active_oracle_session"] = {
            "session_id": session.id,
            "started_at": _iso_now(),
        }
        write_task(scene_dir, task)
        session.task = task
        return session.snapshot(evaluate_report=True)

    def step(
        self,
        session_id: str,
        *,
        right: Any,
        forward: Any,
        camera_azimuth: Any,
        jump: Any,
        frames: Any,
        evaluate_report: Any = False,
    ) -> dict[str, Any]:
        session = self._get(session_id)
        return session.advance(
            right=_bounded_float(right, "right", -1.0, 1.0),
            forward=_bounded_float(forward, "forward", -1.0, 1.0),
            camera_azimuth=_bounded_float(camera_azimuth, "camera_azimuth", -10_000.0, 10_000.0),
            jump=_boolean(jump, "jump"),
            frames=_bounded_int(frames, "frames", 1, MAX_ORACLE_STEP_FRAMES),
            evaluate_report=_boolean(evaluate_report, "evaluate_report"),
        )

    def reset(self, session_id: str) -> dict[str, Any]:
        return self._get(session_id).reset()

    def finish(self, session_id: str) -> dict[str, Any]:
        session = self._get(session_id)
        with session.lock:
            if not session.actions:
                raise ValueError("record at least one action before validating the oracle")
            current_task = read_task(session.scene_dir, session.task["task_id"], include_staleness=False)
            current_spec = _load_spec(session.scene_dir)
            if current_task.get("task_definition_hash") != session.task.get("task_definition_hash"):
                raise ValueError("task tests changed while the oracle was being recorded")
            if task_scene_hash(current_spec) != session.task.get("env_spec_hash"):
                raise ValueError("environment changed while the oracle was being recorded")
            replay = replay_task_actions(
                scene_dir=session.scene_dir,
                task=session.task,
                actions=session.actions,
            )
            attempt_id = new_attempt_id(session.scene_dir, session.task["task_id"])
            attempt_dir = task_dir(session.scene_dir, session.task["task_id"]) / "attempts" / attempt_id
            _atomic_write_json(
                attempt_dir / "actions.json",
                {
                    "schema_version": ORACLE_SCHEMA_VERSION,
                    "provenance": "human_recording",
                    "task_id": session.task["task_id"],
                    "env_id": session.task["env_id"],
                    "controller_version": TASK_CONTROLLER_VERSION,
                    "frames": sum(int(action["frames"]) for action in session.actions),
                    "actions": session.actions,
                },
            )
            _atomic_write_json(
                attempt_dir / "trajectory.json",
                {"task_id": session.task["task_id"], "frames": replay.pop("trajectory")},
            )
            _atomic_write_json(attempt_dir / "report.json", replay)
            passed = bool(replay["passed"])
            current_task.update(
                {
                    "status": "validated" if passed else "validation_failed",
                    "last_validation": replay,
                    "active_oracle_session": None,
                }
            )
            attempt_summary = {
                "attempt_id": attempt_id,
                "created_at": _iso_now(),
                "passed": passed,
                "step_count": replay["step_count"],
                "report_url": task_artifact_url(session.scene_dir, attempt_dir / "report.json"),
                "trajectory_url": task_artifact_url(session.scene_dir, attempt_dir / "trajectory.json"),
            }
            attempts = list(current_task.get("oracle_attempts") or [])
            attempts.append(attempt_summary)
            current_task["oracle_attempts"] = attempts[-20:]
            if passed:
                current_task["oracle"] = {
                    **attempt_summary,
                    "provenance": "human_recording",
                    "controller_version": TASK_CONTROLLER_VERSION,
                    "env_spec_hash": current_task["env_spec_hash"],
                    "task_definition_hash": current_task["task_definition_hash"],
                    "action_count": len(session.actions),
                    "frames": sum(int(action["frames"]) for action in session.actions),
                    "actions_url": task_artifact_url(session.scene_dir, attempt_dir / "actions.json"),
                }
            write_task(session.scene_dir, current_task)
        with self._lock:
            self._sessions.pop(session_id, None)
        return {
            "status": "validated" if passed else "validation_failed",
            "passed": passed,
            "task": current_task,
            "report": replay,
            "attempt": attempt_summary,
        }

    def cancel(self, session_id: str) -> dict[str, Any]:
        session = self._get(session_id)
        task = read_task(session.scene_dir, session.task["task_id"], include_staleness=False)
        task["status"] = "pending_oracle" if task.get("oracle") is None else "validated"
        task["active_oracle_session"] = None
        write_task(session.scene_dir, task)
        with self._lock:
            self._sessions.pop(session_id, None)
        return {"status": "cancelled", "task_id": task["task_id"]}

    def snapshot(self, session_id: str) -> dict[str, Any]:
        return self._get(session_id).snapshot(evaluate_report=True)

    def has_scene(self, scene_dir: Path) -> bool:
        scene_dir = scene_dir.resolve()
        with self._lock:
            self._remove_expired_locked()
            return any(session.scene_dir == scene_dir for session in self._sessions.values())

    def recover_scene(self, scene_dir: Path) -> None:
        """Release persisted recording states that have no live in-memory session."""

        scene_dir = scene_dir.resolve()
        with self._lock:
            self._remove_expired_locked()
            active_ids = {
                session.id
                for session in self._sessions.values()
                if session.scene_dir == scene_dir
            }
        for visible in list_tasks(scene_dir):
            if visible.get("status") != "recording":
                continue
            task = read_task(scene_dir, str(visible["task_id"]), include_staleness=False)
            session_id = str((task.get("active_oracle_session") or {}).get("session_id") or "")
            if session_id in active_ids:
                continue
            task["status"] = "pending_oracle" if task.get("oracle") is None else "validated"
            task["active_oracle_session"] = None
            task["recording_error"] = "Oracle recording was interrupted before validation."
            write_task(scene_dir, task)

    def _get(self, session_id: str) -> TaskOracleSession:
        with self._lock:
            self._remove_expired_locked()
            session = self._sessions.get(session_id)
        if session is None:
            raise ValueError("oracle session was not found or has expired")
        return session

    def _remove_expired_locked(self) -> None:
        cutoff = self.clock() - self.ttl_seconds
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if session.last_access_at < cutoff
        ]
        for session_id in expired:
            session = self._sessions.pop(session_id)
            try:
                task = read_task(session.scene_dir, session.task["task_id"], include_staleness=False)
                task["status"] = "pending_oracle" if task.get("oracle") is None else "validated"
                task["active_oracle_session"] = None
                task["recording_error"] = "Oracle recording expired before validation."
                write_task(session.scene_dir, task)
            except Exception:
                pass


def replay_task_actions(
    *,
    scene_dir: Path,
    task: dict[str, Any],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    simulation = PlayableSimulation.from_scene(scene_dir)
    recorder = TrajectoryRecorder(simulation)
    reset_count = 0
    total_step = 0
    for raw_action in actions:
        if raw_action.get("reset") or raw_action.get("action") == "reset":
            simulation.reset()
            reset_count += 1
            recorder = TrajectoryRecorder(simulation)
            recorder.frames[0]["reset_count"] = reset_count
            recorder.frames[0]["step"] = total_step
            recorder.trajectory[0]["total_step"] = total_step
            continue
        right = _bounded_float(raw_action.get("right", 0.0), "right", -1.0, 1.0)
        forward = _bounded_float(raw_action.get("forward", 0.0), "forward", -1.0, 1.0)
        camera_azimuth = _bounded_float(
            raw_action.get("camera_azimuth", 0.0),
            "camera_azimuth",
            -10_000.0,
            10_000.0,
        )
        jump = _boolean(raw_action.get("jump", False), "jump")
        frames = _bounded_int(raw_action.get("frames", 0), "frames", 1, int(task["max_steps"]))
        for _ in range(frames):
            if total_step >= int(task["max_steps"]):
                break
            simulation.step(
                right=right,
                forward=forward,
                camera_azimuth=camera_azimuth,
                jump=jump,
            )
            total_step += 1
            recorder.capture(reset_count=reset_count, total_step=total_step)
        if total_step >= int(task["max_steps"]):
            break
    report = recorder.report(task=task, final=True)
    report.update(
        {
            "schema_version": ORACLE_SCHEMA_VERSION,
            "controller_version": TASK_CONTROLLER_VERSION,
            "env_spec_hash": task.get("env_spec_hash"),
            "task_definition_hash": task.get("task_definition_hash"),
            "action_count": len(actions),
            "trajectory": list(recorder.trajectory),
        }
    )
    return report


def _prepare_task_for_recording(scene_dir: Path, task_id: str) -> dict[str, Any]:
    task = read_task(scene_dir, task_id, include_staleness=False)
    if task.get("status") == "compiling":
        raise ValueError("task tests are still being compiled")
    if task.get("status") == "error":
        raise ValueError("task tests must compile successfully before recording an oracle")
    spec = _load_spec(scene_dir)
    effective = with_task_staleness(task, spec)
    if effective.get("effective_status") == "stale":
        definition = normalize_task_definition(
            env_id=str(task["env_id"]),
            instruction=str(task["instruction"]),
            tests=task.get("tests"),
            max_steps=task.get("max_steps"),
            spec=spec,
        )
        task.update(definition)
        task.update(
            {
                "status": "pending_oracle",
                "env_spec_hash": task_scene_hash(spec),
                "controller_version": TASK_CONTROLLER_VERSION,
                "task_definition_hash": task_definition_hash(definition),
                "oracle": None,
            }
        )
        write_task(scene_dir, task)
    return task


def _load_spec(scene_dir: Path) -> dict[str, Any]:
    path = scene_dir / "env_spec_3d.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskDefinitionError("finalized env_spec_3d.json is required") from exc
    if not isinstance(value, dict):
        raise TaskDefinitionError("finalized env_spec_3d.json is malformed")
    return value


def _bounded_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in {0, 1}:
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
