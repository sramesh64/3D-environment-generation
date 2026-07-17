"""Restricted child-agent controller for one immutable validated task."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .episode_runtime import assertion_report_irreversibly_failed, decide_episode
from .env_tasks import TASK_CONTROLLER_VERSION
from .player import PlayableSimulation, camera_relative_velocity
from .styled_observation import (
    MUJOCO_OBSERVATION_RENDERER,
    STYLED_OBSERVATION_RENDERER,
    StyledObservationRenderer,
    StyledObservationUnavailable,
)
from .runtime_config import require_runtime_env, runtime_env_value
from .trajectory_assertions import TrajectoryRecorder


MAX_TASK_ACTION_FRAMES = 120
MAX_TASK_ACTION_SEGMENTS = 240
MAX_TASK_RESETS = 2
FIRST_PERSON_WIDTH = 640
FIRST_PERSON_HEIGHT = 360
LOOK_X_DEGREES = 30.0
LOOK_Y_DEGREES = 20.0
FRAME_MANIFEST_FILENAME = "frames.jsonl"
TASK_AGENT_OBSERVATION_MODE = "styled_first_person_with_proprioception"
FIRST_PERSON_FOV_DEGREES = 90.0
# The simulator position is the collider center; +0.42h places the eye 0.92h
# above the base, alongside the styled robot's head.
FIRST_PERSON_EYE_OFFSET_RATIO = 0.42
FIRST_PERSON_DEFAULT_ELEVATION = -8.0
MAX_RECENT_FRAMES = 3
MAX_RECENT_ACTIONS = 6
MAX_RECENT_EVENTS = 12
MAX_CONTACT_ACTION_FRAMES = 8


class TaskAgentSession:
    def __init__(
        self,
        *,
        scene_dir: Path,
        task: dict[str, Any],
        action_log_path: Path | None = None,
        frame_dir: Path | None = None,
        live_state_log_path: Path | None = None,
        render_frames: bool = True,
    ) -> None:
        self.scene_dir = scene_dir
        self.task = task
        self.simulation = PlayableSimulation.from_scene(scene_dir)
        self.action_log_path = action_log_path
        self.frame_dir = frame_dir
        self.live_state_log_path = live_state_log_path
        self.render_frames = render_frames
        self.max_steps = int(task.get("max_steps") or 6_000)
        self.total_steps = 0
        self.reset_count = 0
        self.initial_camera_azimuth = -90.0
        self.initial_camera_elevation = FIRST_PERSON_DEFAULT_ELEVATION
        self.camera_azimuth = self.initial_camera_azimuth
        self.camera_elevation = self.initial_camera_elevation
        self.actions: list[dict[str, Any]] = []
        if self.live_state_log_path is not None:
            self.live_state_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.live_state_log_path.write_text("", encoding="utf-8")
        self.recorder = TrajectoryRecorder(self.simulation)
        self._live_trajectory_cursor = 0
        self._live_last_step: int | None = None
        self._flush_live_trajectory(force=True)
        self.started = False
        self.stopped = False
        self.terminal = False
        self.termination_reason = ""
        self._renderer: Any | None = None
        self._scene_option: Any | None = None
        self._styled_renderer: StyledObservationRenderer | None = None
        self._styled_renderer_unavailable = False
        self._styled_renderer_error = ""
        self._last_frame_renderer = ""
        self._renderer_error = ""
        self._frame_sequence = 0
        self._recent_frames: list[dict[str, Any]] = []
        self._last_frame_signature: tuple[int, int, float, float] | None = None
        self._recent_events: list[dict[str, Any]] = []
        self._zone_event_cursor = 0
        self._last_grounded = self.simulation.has_ground_support()

    @classmethod
    def from_environment(cls) -> "TaskAgentSession":
        scene_dir = Path(require_runtime_env("TASK_SCENE_DIR")).expanduser().resolve()
        task = json.loads(require_runtime_env("TASK_JSON"))
        action_log = runtime_env_value("TASK_ACTION_LOG")
        frame_dir = runtime_env_value("TASK_FRAME_DIR")
        live_state_log = runtime_env_value("TASK_LIVE_STATE_LOG")
        return cls(
            scene_dir=scene_dir,
            task=task,
            action_log_path=Path(action_log) if action_log else None,
            frame_dir=Path(frame_dir) if frame_dir else None,
            live_state_log_path=Path(live_state_log) if live_state_log else None,
            render_frames=True,
        )

    def start(self) -> dict[str, Any]:
        self.started = True
        return self.observe()

    def observe(self) -> dict[str, Any]:
        self.started = True
        self._flush_live_trajectory(force=True)
        self._collect_zone_events()
        self._record_grounded_transition()
        response: dict[str, Any] = {
            "status": "success",
            "controller_version": TASK_CONTROLLER_VERSION,
            "observation_mode": TASK_AGENT_OBSERVATION_MODE,
            "task_id": self.task.get("task_id"),
            "instruction": self.task.get("instruction"),
            "steps_used": self.total_steps,
            "steps_remaining": max(0, self.max_steps - self.total_steps),
            "resets_used": self.reset_count,
            "resets_remaining": max(0, MAX_TASK_RESETS - self.reset_count),
            "grounded": self.simulation.has_ground_support(),
            "collision": self.simulation.has_blocking_contact(),
            "recent_events": list(self._recent_events),
            "recent_actions": [
                _public_action_context(action, index=index)
                for index, action in enumerate(
                    self.actions[-MAX_RECENT_ACTIONS:],
                    start=max(0, len(self.actions) - MAX_RECENT_ACTIONS),
                )
            ],
            "terminal": self.terminal,
            "outcome": _public_episode_outcome(
                terminal=self.terminal,
                stopped=self.stopped,
                termination_reason=self.termination_reason,
            ),
        }
        self._last_frame_renderer = ""
        path = self._render_first_person()
        if path is not None:
            frame = {
                "path": str(path),
                "width": FIRST_PERSON_WIDTH,
                "height": FIRST_PERSON_HEIGHT,
                "renderer": self._last_frame_renderer or STYLED_OBSERVATION_RENDERER,
            }
            response["frame"] = frame
            response["path"] = str(path)
            self._remember_frame(frame)
        elif self._renderer_error:
            response["frame_error"] = self._renderer_error
        response["recent_frames"] = [dict(frame) for frame in self._recent_frames]
        response["recent_frames_order"] = "oldest_to_current"
        if self.actions:
            response["last_action"] = _public_action_context(
                self.actions[-1], index=len(self.actions) - 1
            )
        return response

    def act(
        self,
        *,
        forward: Any = 0.0,
        right: Any = 0.0,
        look_x: Any = 0.0,
        look_y: Any = 0.0,
        jump: Any = False,
        frames: Any = 24,
    ) -> dict[str, Any]:
        if self.stopped:
            raise ValueError("task run is already stopped")
        if self.terminal:
            return self.observe()
        if len(self.actions) >= MAX_TASK_ACTION_SEGMENTS:
            self.terminal = True
            self.termination_reason = "action_segment_limit"
            return self.observe()
        forward_f = _bounded_float(forward, "forward", -1.0, 1.0)
        right_f = _bounded_float(right, "right", -1.0, 1.0)
        look_x_f = _bounded_float(look_x, "look_x", -1.0, 1.0)
        look_y_f = _bounded_float(look_y, "look_y", -1.0, 1.0)
        jump_b = _boolean(jump, "jump")
        requested_frames = _bounded_int(frames, "frames", 1, MAX_TASK_ACTION_FRAMES)
        frames_i = min(
            requested_frames,
            max(0, self.max_steps - self.total_steps),
        )
        if frames_i <= 0:
            self.terminal = True
            self.termination_reason = "step_budget"
            return self.observe()
        self.camera_azimuth += look_x_f * LOOK_X_DEGREES
        self.camera_elevation = max(
            -55.0,
            min(55.0, self.camera_elevation + look_y_f * LOOK_Y_DEGREES),
        )
        movement_requested = math.hypot(forward_f, right_f) > 1e-6
        limited_near_collision = movement_requested and self.simulation.has_blocking_contact()
        if limited_near_collision:
            frames_i = min(frames_i, MAX_CONTACT_ACTION_FRAMES)
        segment = {
            "action": "controller",
            "forward": forward_f,
            "right": right_f,
            "look_x": look_x_f,
            "look_y": look_y_f,
            "camera_azimuth": self.camera_azimuth,
            "camera_elevation": self.camera_elevation,
            "jump": jump_b,
            "frames": frames_i,
            "requested_frames": requested_frames,
        }
        advanced = 0
        collision_step: int | None = None
        for _ in range(frames_i):
            self.simulation.step(
                right=right_f,
                forward=forward_f,
                camera_azimuth=self.camera_azimuth,
                jump=jump_b,
            )
            self.total_steps += 1
            advanced += 1
            self.recorder.capture(reset_count=self.reset_count, total_step=self.total_steps)
            self._flush_live_trajectory()
            self._record_grounded_transition()
            self._collect_zone_events()
            movement_blocked = movement_requested and self.simulation.movement_is_blocked(
                right=right_f,
                forward=forward_f,
                camera_azimuth=self.camera_azimuth,
            )
            if movement_blocked and collision_step is None:
                collision_step = self.total_steps
            report = self.recorder.report(task=self.task, final=False)
            decision = decide_episode(
                safety_failure=self.simulation.safety_failure_reason(),
                explicit_failure=assertion_report_irreversibly_failed(report),
                objective_satisfied=bool(report["passed"]),
                failure_reason="task_test_failed",
                success_reason="task_satisfied",
            )
            if decision.terminal:
                self.terminal = True
                self.termination_reason = decision.reason
                break
            if self.total_steps >= self.max_steps:
                self.terminal = True
                self.termination_reason = "step_budget"
                break
            if movement_blocked:
                break
        segment["frames"] = advanced
        segment["frames_advanced"] = advanced
        segment["total_step"] = self.total_steps
        segment["stopped_on_collision"] = collision_step is not None
        segment["limited_near_collision"] = limited_near_collision
        self.actions.append(segment)
        if collision_step is not None:
            self._append_public_event("collision", step=collision_step)
        self._append_action(segment)
        return self.observe()

    def reset(self) -> dict[str, Any]:
        if self.reset_count >= MAX_TASK_RESETS:
            raise ValueError("task reset budget is exhausted")
        if self.stopped:
            raise ValueError("task run is already stopped")
        self.simulation.reset()
        self.reset_count += 1
        self.camera_azimuth = self.initial_camera_azimuth
        self.camera_elevation = self.initial_camera_elevation
        self._zone_event_cursor = 0
        self._recent_events.clear()
        self._last_grounded = self.simulation.has_ground_support()
        self.recorder = TrajectoryRecorder(self.simulation)
        self._live_trajectory_cursor = 0
        self.recorder.frames[0]["reset_count"] = self.reset_count
        self.recorder.frames[0]["step"] = self.total_steps
        self.recorder.trajectory[0]["total_step"] = self.total_steps
        self._flush_live_trajectory(force=True)
        self.terminal = False
        self.termination_reason = ""
        segment = {
            "action": "reset",
            "reset": True,
            "frames": 0,
            "total_step": self.total_steps,
        }
        self.actions.append(segment)
        self._append_action(segment)
        return self.observe()

    def stop(self) -> dict[str, Any]:
        self.stopped = True
        if not self.termination_reason:
            self.termination_reason = "child_stopped"
        report = self.recorder.report(task=self.task, final=True)
        self._flush_live_trajectory(force=True)
        return {
            "task_id": self.task.get("task_id"),
            "instruction": self.task.get("instruction"),
            "observation_mode": TASK_AGENT_OBSERVATION_MODE,
            "status": "passed" if report["passed"] else "failed",
            "passed": bool(report["passed"]),
            "termination_reason": self.termination_reason,
            "steps_used": self.total_steps,
            "reset_count": self.reset_count,
            "actions": list(self.actions),
            "report": report,
            "trajectory": list(self.recorder.trajectory),
        }

    def close(self) -> None:
        styled_renderer = self._styled_renderer
        self._styled_renderer = None
        if styled_renderer is not None:
            styled_renderer.close()
        renderer = self._renderer
        self._renderer = None
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass

    def _flush_live_trajectory(self, *, force: bool = False) -> None:
        if self.live_state_log_path is None:
            return
        frames = self.recorder.trajectory[self._live_trajectory_cursor :]
        if not frames:
            return
        selected: list[dict[str, Any]] = []
        for frame in frames:
            step = int(frame.get("total_step") or 0)
            if self._live_last_step is None or step - self._live_last_step >= self.recorder.sample_every:
                selected.append(frame)
                self._live_last_step = step
        if force and frames and (
            not selected
            or selected[-1].get("total_step") != frames[-1].get("total_step")
        ):
            selected.append(frames[-1])
            self._live_last_step = int(frames[-1].get("total_step") or 0)
        self._live_trajectory_cursor = len(self.recorder.trajectory)
        if not selected:
            return
        with self.live_state_log_path.open("a", encoding="utf-8") as handle:
            for frame in selected:
                handle.write(json.dumps(frame, separators=(",", ":")) + "\n")

    def _append_public_event(self, event_type: str, *, step: int) -> None:
        event = {"type": event_type, "step": int(step)}
        if self._recent_events and self._recent_events[-1] == event:
            return
        self._recent_events.append(event)
        self._recent_events = self._recent_events[-MAX_RECENT_EVENTS:]

    def _record_grounded_transition(self) -> None:
        grounded = self.simulation.has_ground_support()
        if grounded == self._last_grounded:
            return
        self._last_grounded = grounded
        self._append_public_event(
            "grounded" if grounded else "airborne",
            step=self.total_steps,
        )

    def _collect_zone_events(self) -> None:
        events = self.simulation.zone_events_since(self._zone_event_cursor)
        for event in events:
            self._zone_event_cursor = max(
                self._zone_event_cursor,
                int(event.get("sequence") or 0),
            )
            if event.get("type") == "zone_entered":
                self._append_public_event(
                    "zone_entered",
                    step=self.total_steps,
                )

    def _remember_frame(self, frame: dict[str, Any]) -> None:
        record = {
            "path": str(frame.get("path") or ""),
            "width": int(frame.get("width") or FIRST_PERSON_WIDTH),
            "height": int(frame.get("height") or FIRST_PERSON_HEIGHT),
            "renderer": str(frame.get("renderer") or ""),
            "step": self.total_steps,
            "reset_count": self.reset_count,
        }
        signature = (
            record["step"],
            record["reset_count"],
            round(self.camera_azimuth, 6),
            round(self.camera_elevation, 6),
        )
        if self._recent_frames and self._last_frame_signature == signature:
            self._recent_frames[-1] = record
        else:
            self._recent_frames.append(record)
        self._recent_frames = self._recent_frames[-MAX_RECENT_FRAMES:]
        self._last_frame_signature = signature

    def _render_first_person(self) -> Path | None:
        if not self.render_frames or self.frame_dir is None:
            return None
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        if not self._styled_renderer_unavailable:
            try:
                path = self._render_styled_first_person()
                self._last_frame_renderer = STYLED_OBSERVATION_RENDERER
                return path
            except StyledObservationUnavailable as exc:
                self._styled_renderer_error = str(exc)
                self._styled_renderer_unavailable = True
                if self._styled_renderer is not None:
                    self._styled_renderer.close()
                    self._styled_renderer = None
        path = self._render_mujoco_first_person()
        if path is not None:
            self._last_frame_renderer = MUJOCO_OBSERVATION_RENDERER
        return path

    def _render_styled_first_person(self) -> Path:
        if self.frame_dir is None:
            raise StyledObservationUnavailable("task frame directory is unavailable")
        if self._styled_renderer is None:
            self._styled_renderer = StyledObservationRenderer(
                visual_scene_path=self.scene_dir / "visual_scene.json",
                width=FIRST_PERSON_WIDTH,
                height=FIRST_PERSON_HEIGHT,
            )
        camera = self._first_person_camera_pose()
        trajectory_frame = self.recorder.trajectory[-1]
        path = self.frame_dir / f"frame_{self._frame_sequence:04d}.png"
        self._styled_renderer.render(
            output_path=path,
            objects=list(trajectory_frame.get("objects") or []),
            mechanisms=list(trajectory_frame.get("mechanisms") or []),
            camera=camera,
            hidden_source_ids=[self.simulation.agent.object_id],
        )
        self._frame_sequence += 1
        self._append_frame_manifest(path, renderer=STYLED_OBSERVATION_RENDERER)
        return path

    def _render_mujoco_first_person(self) -> Path | None:
        try:
            from PIL import Image

            if self._renderer is None:
                self.simulation.model.vis.global_.fovy = FIRST_PERSON_FOV_DEGREES
                self._renderer = self.simulation.mujoco.Renderer(
                    self.simulation.model,
                    height=FIRST_PERSON_HEIGHT,
                    width=FIRST_PERSON_WIDTH,
                )
                self._scene_option = self.simulation.mujoco.MjvOption()
                self._scene_option.geomgroup[5] = 0
            camera = self.simulation.mujoco.MjvCamera()
            camera.type = self.simulation.mujoco.mjtCamera.mjCAMERA_FREE
            pose = self._first_person_camera_pose()
            camera.lookat[:] = pose["target"]
            camera.distance = math.dist(pose["position"], pose["target"])
            camera.azimuth = -self.camera_azimuth - 90.0
            camera.elevation = self.camera_elevation
            self._renderer.update_scene(
                self.simulation.data,
                camera=camera,
                scene_option=self._scene_option,
            )
            pixels = self._renderer.render()
            path = self.frame_dir / f"frame_{self._frame_sequence:04d}.png"
            self._frame_sequence += 1
            Image.fromarray(pixels).save(path)
            self._append_frame_manifest(
                path,
                renderer=MUJOCO_OBSERVATION_RENDERER,
                renderer_error=self._styled_renderer_error,
            )
            return path
        except Exception as exc:
            self._renderer_error = str(exc)
            return None

    def _first_person_camera_pose(self) -> dict[str, Any]:
        position = self.simulation.agent_position()
        forward_xy = camera_relative_velocity(
            self.camera_azimuth,
            right=0.0,
            forward=1.0,
            speed=1.0,
        )
        eye_z = position[2] + self.simulation.agent.height * FIRST_PERSON_EYE_OFFSET_RATIO
        return {
            "position": [position[0], position[1], eye_z],
            "target": [
                position[0] + forward_xy[0],
                position[1] + forward_xy[1],
                eye_z + math.tan(math.radians(self.camera_elevation)),
            ],
            "fov_y_degrees": FIRST_PERSON_FOV_DEGREES,
        }

    def _append_frame_manifest(
        self,
        path: Path,
        *,
        renderer: str,
        renderer_error: str = "",
    ) -> None:
        if self.frame_dir is None:
            return
        record = {
            "path": str(path),
            "step": self.total_steps,
            "reset_count": self.reset_count,
            "renderer": renderer,
            "camera": {
                "azimuth": self.camera_azimuth,
                "elevation": self.camera_elevation,
                "fov_y_degrees": FIRST_PERSON_FOV_DEGREES,
            },
        }
        if renderer_error:
            record["renderer_error"] = renderer_error[:1000]
        with (self.frame_dir / FRAME_MANIFEST_FILENAME).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")

    def _append_action(self, action: dict[str, Any]) -> None:
        if self.action_log_path is None:
            return
        self.action_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.action_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(action, separators=(",", ":"), ensure_ascii=False) + "\n")


def _public_episode_outcome(*, terminal: bool, stopped: bool, termination_reason: str) -> str:
    if not terminal and not stopped:
        return "running"
    if termination_reason == "task_satisfied":
        return "passed"
    if termination_reason in {"step_budget", "action_segment_limit"}:
        return "budget_exhausted"
    return "failed"


def _public_action_context(action: dict[str, Any], *, index: int) -> dict[str, Any]:
    """Return action history without absolute camera pose or simulator state."""

    if action.get("reset") or action.get("action") == "reset":
        return {
            "index": int(index),
            "action": "reset",
            "step": int(action.get("total_step") or 0),
        }
    return {
        "index": int(index),
        "action": "controller",
        "forward": float(action.get("forward") or 0.0),
        "right": float(action.get("right") or 0.0),
        "look_x": float(action.get("look_x") or 0.0),
        "look_y": float(action.get("look_y") or 0.0),
        "jump": bool(action.get("jump")),
        "frames_requested": int(
            action.get("requested_frames", action.get("frames")) or 0
        ),
        "frames_advanced": int(action.get("frames_advanced", action.get("frames")) or 0),
        "stopped_on_collision": bool(action.get("stopped_on_collision")),
        "limited_near_collision": bool(action.get("limited_near_collision")),
        "step": int(action.get("total_step") or 0),
    }


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
