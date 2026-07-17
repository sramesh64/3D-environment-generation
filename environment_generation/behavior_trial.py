"""Fixed-step MuJoCo execution and evidence collection for behavior trials."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .behavior_guidance import (
    INTERACTION_RANGE,
    action_guidance,
    ground_route_guidance,
    interaction_guidance,
    object_clearance_xy,
    objective_focus,
    scene_context,
    semantic_affordances,
)
from .env_behavior_trials import (
    BEHAVIOR_CONTROLLER_VERSION,
    MAX_RESETS,
    MAX_STEPS,
    negative_search_evidence,
    negative_search_repair_hint,
    normalize_behavior_trial_for_runtime,
)
from .env_tasks import describe_assertion_condition
from .episode_runtime import decide_episode
from .env_verification import (
    SceneObject3D,
    scene_object_at,
    scene_objects,
)
from .mujoco_compile import floor_sensor_visual_box
from .player import PlayableSimulation, camera_relative_velocity
from .runtime_config import require_runtime_env, runtime_env_value
from .scene_geometry import volumes_overlap, yaw_from_rotation_matrix
from .trajectory_assertions import (
    AssertionSatisfactionMonitor,
    capture_trajectory_frame,
    evaluate_assertion_group,
)


FIRST_PERSON_WIDTH = 640
FIRST_PERSON_HEIGHT = 360
LOOK_X_DEGREES = 30.0
LOOK_Y_DEGREES = 20.0
TRAJECTORY_SAMPLE_SECONDS = 0.05
MAX_ACTION_FRAMES = 60
MAX_ASSISTED_ACTION_FRAMES = 300
# Fine control near hazards and interaction boundaries can legitimately require
# many short actions. Simulation-step budgets remain the authoritative limit.
MAX_ACTION_SEGMENTS = MAX_STEPS * (MAX_RESETS + 1)
SPAWN_SETTLE_MAX_STEPS = 120
SPAWN_SETTLE_STABLE_FRAMES = 3
FRAME_MANIFEST_FILENAME = "frames.jsonl"


class BehaviorTrialSession:
    """An isolated controller session over one immutable scene snapshot."""

    def __init__(
        self,
        *,
        scene_dir: Path,
        trial: dict[str, Any],
        action_log_path: Path | None = None,
        observation_log_path: Path | None = None,
        frame_dir: Path | None = None,
        render_frames: bool = True,
    ) -> None:
        self.scene_dir = scene_dir
        self.simulation = PlayableSimulation.from_scene(scene_dir)
        self.trial = normalize_behavior_trial_for_runtime(
            trial,
            spec=self.simulation.spec,
        )
        self.action_log_path = action_log_path
        self.observation_log_path = observation_log_path
        self.frame_dir = frame_dir
        self.render_frames = render_frames
        self.max_steps = int(
            self.trial.get("max_steps_per_attempt")
            or self.trial.get("max_steps")
            or 2400
        )
        max_resets = self.trial.get("max_resets")
        self.max_resets = int(1 if max_resets is None else max_resets)
        self.max_total_steps = int(
            self.trial.get("max_total_steps") or self.max_steps * (self.max_resets + 1)
        )
        self.total_steps = 0
        self.attempt_steps = 0
        self.reset_count = 0
        self.action_segments = 0
        self.initial_camera_azimuth = float(
            (self.trial.get("navigation") or {}).get("initial_camera_azimuth", -90.0)
        )
        self.initial_camera_elevation = float(
            (self.trial.get("navigation") or {}).get("initial_camera_elevation", 0.0)
        )
        self.camera_azimuth = self.initial_camera_azimuth
        self.camera_elevation = self.initial_camera_elevation
        self.started = False
        self.stopped = False
        self.terminal = False
        self.termination_reason = ""
        self.actions: list[dict[str, Any]] = []
        self.trajectory: list[dict[str, Any]] = []
        self.attempts: list[TrialAttemptTracker] = []
        self._frame_sequence = 0
        self._observation_sequence = 0
        self._renderer: Any | None = None
        self._scene_option: Any | None = None
        self._renderer_error = ""
        self._settle_steps: list[int] = []
        self._navigation_start_distances: dict[str, float] = {}
        self._frame_event_counts: dict[int, int] = {}
        timestep = float(self.simulation.model.opt.timestep)
        self._sample_every = max(1, int(round(TRAJECTORY_SAMPLE_SECONDS / timestep)))
        self._configure_policy_sensor_visuals()
        self._settle_spawn()
        self._start_attempt()

    @classmethod
    def from_environment(cls) -> "BehaviorTrialSession":
        scene_dir = Path(require_runtime_env("BEHAVIOR_SCENE_DIR")).expanduser().resolve()
        trial = json.loads(require_runtime_env("BEHAVIOR_TRIAL_JSON"))
        action_log = runtime_env_value("BEHAVIOR_ACTION_LOG")
        observation_log = runtime_env_value("BEHAVIOR_OBSERVATION_LOG")
        frame_dir = runtime_env_value("BEHAVIOR_FRAME_DIR")
        return cls(
            scene_dir=scene_dir,
            trial=trial,
            action_log_path=Path(action_log) if action_log else None,
            observation_log_path=Path(observation_log) if observation_log else None,
            frame_dir=Path(frame_dir) if frame_dir else None,
            render_frames=True,
        )

    def start(self) -> dict[str, Any]:
        self.started = True
        self._capture_trajectory(force=True)
        return self.observe()

    def observe(self) -> dict[str, Any]:
        if not self.started:
            self.started = True
        trial_state = self._combined_trial_state(final=self.stopped or self.terminal)
        objective = trial_state["objective"]
        guidance = self._guidance_state(objective)
        current_objects = guidance["objects"]
        focus = guidance["focus"]
        mechanisms = guidance["mechanisms"]
        route = guidance["route"]
        interaction = guidance["interaction"]
        agent = self._agent_telemetry()
        response: dict[str, Any] = {
            "status": "success",
            "controller_version": BEHAVIOR_CONTROLLER_VERSION,
            "trial_id": self.trial.get("id"),
            "attempt": len(self.attempts),
            "simulation_time": float(self.simulation.data.time),
            "steps_used": self.total_steps,
            "attempt_steps_used": self.attempt_steps,
            "steps_remaining": max(0, self.max_steps - self.attempt_steps),
            "total_steps_remaining": max(0, self.max_total_steps - self.total_steps),
            "resets_used": self.reset_count,
            "resets_remaining": max(0, self.max_resets - self.reset_count),
            "camera": {
                "azimuth": round(self.camera_azimuth, 4),
                "elevation": round(self.camera_elevation, 4),
            },
            "agent": agent,
            "scene_context": scene_context(
                spec=self.simulation.spec,
                trial=self.trial,
                objects=current_objects,
                focus=focus,
            ),
            "objective_focus": focus,
            "nearby_objects": self._nearby_objects(
                objects=current_objects,
                focus=focus,
                mechanisms=mechanisms,
            ),
            "navigation": self._navigation_telemetry(),
            "route_guidance": route,
            "interaction_guidance": interaction,
            "action_guidance": guidance["action"],
            "mechanisms": mechanisms,
            "recent_events": self.attempts[-1].events[-12:],
            "objective": objective,
            "constraints": trial_state["constraints"],
            "trial_satisfied": trial_state["satisfied"],
            "reward": (
                1.0
                if trial_state["satisfied"]
                else -1.0
                if self.termination_reason in {"constraint_failed", "out_of_bounds"}
                else 0.0
            ),
            "terminal": self.terminal,
            "termination_reason": self.termination_reason,
            "available_actions": {
                "forward": [-1.0, 1.0],
                "right": [-1.0, 1.0],
                "look_x": [-1.0, 1.0],
                "look_y": [-1.0, 1.0],
                "jump": "boolean",
                "assist": ["none", "ground_route"],
                "assist_target": "optional scene object id",
                "manual_frames": [1, MAX_ACTION_FRAMES],
                "assisted_frames": [1, MAX_ASSISTED_ACTION_FRAMES],
            },
        }
        current_attempt = self.attempts[-1]
        current_attempt_state = current_attempt.trial_state(final=self.stopped or self.terminal)
        frame_path = self._render_first_person(
            {
                "agent": agent,
                "navigation": response["navigation"],
                "objective_focus": focus,
                "interaction_guidance": interaction,
                "objective": _compact_evidence_group(objective),
                "constraints": _compact_evidence_group(trial_state["constraints"]),
                "attempt_objective": _compact_evidence_group(
                    current_attempt_state["objective"]
                ),
                "attempt_constraints": _compact_evidence_group(
                    current_attempt_state["constraints"]
                ),
                "mechanisms": mechanisms,
                "terminal": self.terminal,
                "termination_reason": self.termination_reason,
            }
        )
        if frame_path is not None:
            response["path"] = str(frame_path)
            response["frame"] = {"path": str(frame_path), "width": FIRST_PERSON_WIDTH, "height": FIRST_PERSON_HEIGHT}
        elif self._renderer_error:
            response["frame_error"] = self._renderer_error
        self._append_observation(response)
        return response

    def act(
        self,
        *,
        forward: Any = 0.0,
        right: Any = 0.0,
        look_x: Any = 0.0,
        look_y: Any = 0.0,
        jump: Any = False,
        frames: Any = 12,
        assist: Any = "none",
        target_id: Any = "",
        _defer_observation: bool = False,
    ) -> dict[str, Any]:
        if self.stopped:
            raise ValueError("behavior trial is already stopped")
        if self.terminal:
            return self.observe()
        assist_mode = str(assist or "none").strip().lower()
        if assist_mode not in {"none", "ground_route"}:
            raise ValueError("assist must be 'none' or 'ground_route'")
        if assist_mode == "ground_route":
            return self._follow_ground_route(frames, target_id=target_id)
        if str(target_id or "").strip():
            raise ValueError("target_id is only supported with ground_route assistance")
        if self.action_segments >= MAX_ACTION_SEGMENTS:
            self.terminal = True
            self.termination_reason = "action_segment_limit"
            return self.observe()
        forward_f = _bounded_float(forward, "forward", -1.0, 1.0)
        right_f = _bounded_float(right, "right", -1.0, 1.0)
        look_x_f = _bounded_float(look_x, "look_x", -1.0, 1.0)
        look_y_f = _bounded_float(look_y, "look_y", -1.0, 1.0)
        jump_b = _boolean(jump, "jump")
        frames_i = _bounded_int(frames, "frames", 1, MAX_ACTION_FRAMES)
        frames_i = min(
            frames_i,
            max(0, self.max_steps - self.attempt_steps),
            max(0, self.max_total_steps - self.total_steps),
        )
        if frames_i <= 0:
            self.terminal = True
            self.termination_reason = (
                "attempt_budget" if self.reset_count < self.max_resets else "step_budget"
            )
            return self.observe()

        self.camera_azimuth += look_x_f * LOOK_X_DEGREES
        self.camera_elevation = max(-55.0, min(55.0, self.camera_elevation + look_y_f * LOOK_Y_DEGREES))
        segment = {
            "action": "controller",
            "forward": forward_f,
            "right": right_f,
            "look_x": look_x_f,
            "look_y": look_y_f,
            "jump": jump_b,
            "frames": frames_i,
            "camera_azimuth": self.camera_azimuth,
            "camera_elevation": self.camera_elevation,
            "attempt": len(self.attempts),
        }
        advanced = 0
        jump_held = jump_b
        for frame_index in range(frames_i):
            before_vertical = self._vertical_velocity()
            self.simulation.step(
                right=right_f,
                forward=forward_f,
                camera_azimuth=self.camera_azimuth,
                jump=jump_held,
            )
            self.total_steps += 1
            self.attempt_steps += 1
            advanced += 1
            jump_started = jump_held and before_vertical <= 1.0 and self._vertical_velocity() > 3.0
            self.attempts[-1].record_step(
                total_step=self.total_steps,
                jump_started=jump_started,
            )
            safety_failure = self.simulation.safety_failure_reason()
            if safety_failure:
                self.attempts[-1].record_terminal_event(
                    total_step=self.total_steps,
                    outcome=safety_failure,
                )
            if self.total_steps % self._sample_every == 0:
                self._capture_trajectory()
            trial_satisfied = self._combined_trial_satisfied()
            decision = decide_episode(
                safety_failure=safety_failure,
                explicit_failure=self.attempts[-1].constraints_monitor.irreversibly_failed,
                objective_satisfied=trial_satisfied,
                failure_reason="constraint_failed",
                success_reason="objective_satisfied",
            )
            if decision.terminal:
                self.terminal = True
                self.termination_reason = decision.reason
                break
            if self.attempt_steps >= self.max_steps or self.total_steps >= self.max_total_steps:
                self.terminal = True
                self.termination_reason = (
                    "attempt_budget"
                    if self.reset_count < self.max_resets and self.total_steps < self.max_total_steps
                    else "step_budget"
                )
                break
            if frame_index == 0:
                jump_held = False

        segment["frames_advanced"] = advanced
        segment["total_step"] = self.total_steps
        self.action_segments += 1
        self.actions.append(segment)
        self._append_action(segment)
        self._capture_trajectory(force=True)
        if _defer_observation:
            return {
                "status": "success",
                "terminal": self.terminal,
                "termination_reason": self.termination_reason,
                "last_action": segment,
            }
        response = self.observe()
        response["last_action"] = segment
        return response

    def _follow_ground_route(self, frames: Any, *, target_id: Any = "") -> dict[str, Any]:
        requested = _bounded_int(
            frames,
            "frames",
            1,
            MAX_ASSISTED_ACTION_FRAMES,
        )
        advanced = 0
        segment_count = 0
        stop_reason = "frame_budget"
        route_target_id = str(target_id or "").strip()
        if route_target_id and not any(
            obj.id == route_target_id for obj in self.simulation.spec.objects
        ):
            raise ValueError(f"unknown ground-route target_id {route_target_id!r}")
        linked_mechanism_ids = {
            mechanism.id
            for mechanism in self.simulation.spec.mechanisms
            if mechanism.trigger_id == route_target_id
        }
        while advanced < requested and not self.terminal:
            if self.action_segments >= MAX_ACTION_SEGMENTS:
                self.terminal = True
                self.termination_reason = "action_segment_limit"
                stop_reason = self.termination_reason
                break
            objective = self._combined_trial_state(final=False)["objective"]
            guidance = self._guidance_state(
                objective,
                route_target_id=route_target_id,
            )
            route = guidance["route"]
            action = guidance["action"]
            if not route.get("available"):
                stop_reason = str(route.get("status") or "route_unavailable")
                break
            target_id = str(route.get("target_id") or "")
            target = next(
                (obj for obj in guidance["objects"] if obj.id == target_id),
                None,
            )
            target_clearance = action.get("current_target_clearance")
            target_reached = (
                target is not None
                and target_clearance is not None
                and float(target_clearance) <= 0.0
            )
            if target_reached and target.body_type == "sensor":
                if not linked_mechanism_ids:
                    stop_reason = "target_reached"
                    break
                mechanism_by_id = {
                    str(state.get("id") or ""): state
                    for state in guidance["mechanisms"]
                }
                if all(
                    float((mechanism_by_id.get(mechanism_id) or {}).get("progress") or 0.0)
                    >= 0.9
                    for mechanism_id in linked_mechanism_ids
                ):
                    stop_reason = "mechanism_ready"
                    break
            if (
                target is not None
                and target.body_type != "sensor"
                and target_clearance is not None
                and float(target_clearance) <= INTERACTION_RANGE
            ):
                stop_reason = "interaction_range"
                break
            relative = route.get("next_waypoint_relative") or {}
            heading_error = float(relative.get("heading_error_degrees") or 0.0)
            remaining = requested - advanced
            recommended = max(2, int(action.get("recommended_max_frames") or 12))
            segment_frames = min(remaining, recommended, MAX_ACTION_FRAMES)
            if abs(heading_error) > 35.0:
                forward = 0.0
                segment_frames = min(segment_frames, 3)
            elif (
                target is not None
                and target.body_type == "sensor"
                and target_reached
            ):
                # Sensor objectives such as switches may need time to latch or
                # open a mechanism after the actor overlaps their footprint.
                forward = 0.0
            else:
                forward = 1.0
            look_x = max(-1.0, min(1.0, heading_error / LOOK_X_DEGREES))
            response = self.act(
                forward=forward,
                right=0.0,
                look_x=look_x,
                look_y=0.0,
                jump=False,
                frames=segment_frames,
                assist="none",
                _defer_observation=True,
            )
            segment = response.get("last_action") or {}
            moved = int(segment.get("frames_advanced") or 0)
            advanced += moved
            segment_count += 1
            if moved <= 0:
                stop_reason = self.termination_reason or "no_progress"
                break
        if self.terminal:
            stop_reason = self.termination_reason or "terminal"
        response = self.observe()
        response["assisted_action"] = {
            "mode": "ground_route",
            "target_id": route_target_id or None,
            "requested_frames": requested,
            "frames_advanced": advanced,
            "controller_segments": segment_count,
            "stop_reason": stop_reason,
            "authoritative_replay_uses_logged_controller_segments": True,
        }
        return response

    def reset(self) -> dict[str, Any]:
        if self.reset_count >= self.max_resets:
            raise ValueError("behavior trial reset budget is exhausted")
        if self._combined_trial_satisfied():
            raise ValueError("the objective is already satisfied")
        reset_reason = self.termination_reason or "manual"
        self.attempts[-1].record_attempt_reset(
            total_step=self.total_steps,
            reason=reset_reason,
        )
        self.simulation.reset()
        self._settle_spawn()
        self.reset_count += 1
        self.attempt_steps = 0
        self.camera_azimuth = self.initial_camera_azimuth
        self.camera_elevation = self.initial_camera_elevation
        self.terminal = False
        self.termination_reason = ""
        self._navigation_start_distances.clear()
        segment = {
            "action": "reset",
            "reset": True,
            "frames": 0,
            "total_step": self.total_steps,
            "attempt": len(self.attempts) + 1,
        }
        self.actions.append(segment)
        self._append_action(segment)
        self._start_attempt()
        self._capture_trajectory(force=True)
        return self.observe()

    def stop(self) -> dict[str, Any]:
        self.stopped = True
        if not self.termination_reason:
            self.termination_reason = "child_stopped"
        self._capture_trajectory(force=True)
        return self.result()

    def result(self) -> dict[str, Any]:
        trial_state = self._combined_trial_state(final=True)
        objective = trial_state["objective"]
        expected = str(self.trial.get("expected_outcome") or "should_succeed")
        termination_reason = self.termination_reason or "not_stopped"
        search_evidence: dict[str, Any] | None = None
        if expected == "should_not_succeed":
            if trial_state["satisfied"]:
                status = "failed"
            else:
                search_evidence = negative_search_evidence(
                    trial=self.trial,
                    actions=self.actions,
                    objective=objective,
                    constraints=trial_state["constraints"],
                    termination_reason=termination_reason,
                )
                status = "passed" if search_evidence["valid"] else "inconclusive"
        else:
            status = "passed" if trial_state["satisfied"] else "inconclusive"
        result = {
            "trial_id": self.trial.get("id"),
            "instruction": self.trial.get("instruction"),
            "expected_outcome": expected,
            "severity": self.trial.get("severity") or "critical",
            "status": status,
            "passed": status == "passed",
            "non_failure": status == "passed",
            "objective": objective,
            "constraints": trial_state["constraints"],
            "reward": (
                1.0
                if trial_state["satisfied"]
                else -1.0
                if self.termination_reason in {"constraint_failed", "out_of_bounds"}
                else 0.0
            ),
            "termination_reason": termination_reason,
            "steps_used": self.total_steps,
            "reset_count": self.reset_count,
            "attempt_count": len(self.attempts),
            "settle_steps": list(self._settle_steps),
            "actions": list(self.actions),
            "events": [event for attempt in self.attempts for event in attempt.events],
            "attempts": [attempt.summary(final=True) for attempt in self.attempts],
            "trajectory": list(self.trajectory),
            "final_state": self._snapshot(),
            "repair_hints": _repair_hints(
                self.trial,
                status,
                objective,
                search_evidence=search_evidence,
            ),
            "preflight": self.trial.get("preflight") or {},
        }
        if search_evidence is not None:
            result["search_evidence"] = search_evidence
        return result

    def close(self) -> None:
        renderer = self._renderer
        self._renderer = None
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass

    def _start_attempt(self) -> None:
        self.attempts.append(
            TrialAttemptTracker(
                simulation=self.simulation,
                trial=self.trial,
                attempt=len(self.attempts) + 1,
                total_step=self.total_steps,
                reset_count=self.reset_count,
            )
        )

    def _configure_policy_sensor_visuals(self) -> None:
        changed = False
        for obj in self.simulation.spec.objects:
            visual_box = floor_sensor_visual_box(obj.model_dump(mode="json"))
            if visual_box is None:
                continue
            geom_id = int(
                self.simulation.mujoco.mj_name2id(
                    self.simulation.model,
                    self.simulation.mujoco.mjtObj.mjOBJ_GEOM,
                    obj.id,
                )
            )
            if geom_id < 0:
                continue
            position, size = visual_box
            self.simulation.model.geom_pos[geom_id, :3] = position
            self.simulation.model.geom_size[geom_id, :3] = [value / 2.0 for value in size]
            changed = True
        if changed:
            self.simulation.mujoco.mj_forward(self.simulation.model, self.simulation.data)

    def _settle_spawn(self) -> None:
        stable_frames = 0
        steps = 0
        for _ in range(SPAWN_SETTLE_MAX_STEPS):
            self.simulation.step(
                right=0.0,
                forward=0.0,
                camera_azimuth=self.initial_camera_azimuth,
                jump=False,
            )
            steps += 1
            vertical_speed = abs(self._vertical_velocity())
            if self.simulation.is_grounded() and vertical_speed <= 0.15:
                stable_frames += 1
                if stable_frames >= SPAWN_SETTLE_STABLE_FRAMES:
                    break
            else:
                stable_frames = 0
        self._settle_steps.append(steps)

    def _combined_trial_state(self, *, final: bool) -> dict[str, Any]:
        snapshots = [attempt.trial_state(final=final) for attempt in self.attempts]
        successful = next((item for item in snapshots if item["satisfied"]), None)
        selected = successful or max(
            snapshots,
            key=lambda item: (
                int(item["objective"].get("passed_count") or 0),
                int(item["constraints"].get("passed_count") or 0),
            ),
        )
        return {**selected, "attempt": snapshots.index(selected) + 1}

    def _combined_objective(self, *, final: bool) -> dict[str, Any]:
        return self._combined_trial_state(final=final)["objective"]

    def _combined_trial_satisfied(self) -> bool:
        return any(attempt.satisfied for attempt in self.attempts)

    def _snapshot(self) -> dict[str, Any]:
        return {
            "simulation_time": float(self.simulation.data.time),
            "status": self.simulation.status(),
            "grounded": self.simulation.is_grounded(),
            "objects": self.simulation.body_transforms(),
            "mechanisms": self.simulation.mechanism_states(),
            "camera": {
                "azimuth": self.camera_azimuth,
                "elevation": self.camera_elevation,
            },
        }

    def _capture_trajectory(self, *, force: bool = False) -> None:
        snapshot = {
            "total_step": self.total_steps,
            "attempt": len(self.attempts),
            **self._snapshot(),
        }
        if (
            self.trajectory
            and self.trajectory[-1]["total_step"] == self.total_steps
            and self.trajectory[-1]["attempt"] == len(self.attempts)
        ):
            if force:
                self.trajectory[-1] = snapshot
            return
        self.trajectory.append(snapshot)

    def _agent_telemetry(self) -> dict[str, Any]:
        position = self.simulation.agent_position()
        address = self.simulation.agent.qvel_address
        velocity = [float(value) for value in self.simulation.data.qvel[address : address + 3]]
        return {
            "id": self.simulation.agent.object_id,
            "position": [round(value, 5) for value in position],
            "velocity": [round(value, 5) for value in velocity],
            "grounded": self.simulation.is_grounded(),
            "status": self.simulation.status(),
        }

    def _current_scene_objects(self) -> list[SceneObject3D]:
        positions = _current_positions(self.simulation)
        values = []
        for obj in scene_objects(self.simulation.spec):
            body_id = self.simulation.dynamic_body_ids.get(obj.id)
            yaw = (
                yaw_from_rotation_matrix(
                    self.simulation.data.xmat[body_id],
                    fallback=obj.yaw,
                )
                if body_id is not None
                else obj.yaw
            )
            values.append(
                scene_object_at(
                    obj,
                    positions.get(obj.id, obj.position),
                    yaw=yaw,
                )
            )
        return values

    def _guidance_state(
        self,
        objective: dict[str, Any],
        *,
        route_target_id: str = "",
    ) -> dict[str, Any]:
        objects = self._current_scene_objects()
        focus = objective_focus(
            trial=self.trial,
            objective_state=objective,
            objects=objects,
        )
        mechanisms = self.simulation.mechanism_states()
        agent_position = self.simulation.agent_position()
        agent_radius = max(
            float(self.simulation.agent_spec.size[0]),
            float(self.simulation.agent_spec.size[1]),
        ) * 0.5
        interaction = interaction_guidance(
            objects=objects,
            focus=focus,
            agent_position=agent_position,
            agent_radius=agent_radius,
            contact_ids=_agent_contact_object_ids(self.simulation),
            camera_azimuth=self.camera_azimuth,
        )
        route_focus = dict(focus)
        if route_target_id:
            route_focus.update(
                {
                    "navigation_target_ids": [route_target_id],
                    "target_ids": [route_target_id],
                    "relation": "",
                }
            )
            route_focus.pop("approach_point", None)
        elif interaction.get("applicable") and interaction.get("staging_position"):
            route_focus["approach_point"] = interaction["staging_position"]
        route = ground_route_guidance(
            spec=self.simulation.spec,
            objects=objects,
            focus=route_focus,
            agent_position=agent_position,
            camera_azimuth=self.camera_azimuth,
            mechanism_states=mechanisms,
        )
        return {
            "objects": objects,
            "focus": focus,
            "mechanisms": mechanisms,
            "route": route,
            "interaction": interaction,
            "action": action_guidance(
                agent_position=agent_position,
                agent_radius=agent_radius,
                objects=objects,
                focus=route_focus,
                route=route,
            ),
        }

    def _nearby_objects(
        self,
        *,
        objects: list[SceneObject3D],
        focus: dict[str, Any],
        mechanisms: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        agent_position = self.simulation.agent_position()
        agent_radius = max(
            float(self.simulation.agent_spec.size[0]),
            float(self.simulation.agent_spec.size[1]),
        ) * 0.5
        forward_xy = camera_relative_velocity(self.camera_azimuth, right=0.0, forward=1.0, speed=1.0)
        right_xy = camera_relative_velocity(self.camera_azimuth, right=1.0, forward=0.0, speed=1.0)
        target_ids = set(focus.get("subject_ids") or []) | set(focus.get("target_ids") or [])
        contact_ids = _agent_contact_object_ids(self.simulation)
        agent_object = next(
            obj for obj in objects if obj.id == self.simulation.agent.object_id
        )
        active_zone_ids = {
            obj.id
            for obj in objects
            if obj.body_type == "sensor" and _scene_objects_overlap(agent_object, obj)
        }
        mechanism_progress = {
            str(state.get("gate_id")): float(state.get("progress") or 0.0)
            for state in mechanisms
        }
        values = []
        for obj in objects:
            if obj.id == self.simulation.agent.object_id:
                continue
            position = obj.position
            dx = position[0] - agent_position[0]
            dy = position[1] - agent_position[1]
            dz = position[2] - agent_position[2]
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            clearance = object_clearance_xy(agent_position, obj, radius=agent_radius)
            values.append(
                {
                    "id": obj.id,
                    "semantic_type": obj.semantic_type,
                    "body_type": obj.body_type,
                    "shape": obj.shape,
                    "position": [round(value, 4) for value in position],
                    "size": [round(float(value), 4) for value in obj.size],
                    "bounds": {key: round(value, 4) for key, value in obj.bounds.items()},
                    "affordances": semantic_affordances(
                        obj,
                        mechanism_progress=mechanism_progress.get(obj.id),
                    ),
                    "is_objective_target": obj.id in target_ids,
                    "in_contact": obj.id in contact_ids,
                    "agent_inside_zone": obj.id in active_zone_ids,
                    "distance": round(distance, 4),
                    "horizontal_clearance": round(clearance, 4),
                    "overlap_xy": clearance < 0.0,
                    "relative": {
                        "forward": round(dx * forward_xy[0] + dy * forward_xy[1], 4),
                        "right": round(dx * right_xy[0] + dy * right_xy[1], 4),
                        "up": round(dz, 4),
                    },
                }
            )
        values.sort(
            key=lambda item: (
                0 if item["is_objective_target"] else 1,
                0 if "failure_zone" in item["affordances"] else 1,
                item["distance"],
            )
        )
        return values[:24]

    def _navigation_telemetry(self) -> dict[str, Any]:
        navigation = self.trial.get("navigation") or {}
        target_id = self._current_navigation_target_id()
        if target_id is None and self.attempts:
            objective = self._combined_objective(final=False)
            if objective.get("satisfied"):
                return {
                    "primary_target_id": None,
                    "available": False,
                    "status": "objective_satisfied",
                }
        target_id = target_id or navigation.get("primary_target_id")
        if not target_id:
            return {"primary_target_id": None}
        positions = _current_positions(self.simulation)
        target_position = positions.get(str(target_id))
        if target_position is None:
            return {"primary_target_id": target_id, "available": False}
        agent_position = self.simulation.agent_position()
        dx = target_position[0] - agent_position[0]
        dy = target_position[1] - agent_position[1]
        center_distance_xy = math.hypot(dx, dy)
        target_spec = next((obj for obj in self.simulation.spec.objects if obj.id == target_id), None)
        agent_radius = max(float(self.simulation.agent_spec.size[0]), float(self.simulation.agent_spec.size[1])) * 0.5
        target_radius = (
            max(float(target_spec.size[0]), float(target_spec.size[1])) * 0.5
            if target_spec is not None
            else 0.0
        )
        distance_xy = max(0.0, center_distance_xy - agent_radius - target_radius)
        forward_xy = camera_relative_velocity(
            self.camera_azimuth,
            right=0.0,
            forward=1.0,
            speed=1.0,
        )
        right_xy = camera_relative_velocity(
            self.camera_azimuth,
            right=1.0,
            forward=0.0,
            speed=1.0,
        )
        if str(target_id) not in self._navigation_start_distances:
            self._navigation_start_distances[str(target_id)] = distance_xy
        initial_distance = self._navigation_start_distances[str(target_id)]
        desired_azimuth = math.degrees(math.atan2(-dx, -dy)) if center_distance_xy > 1e-9 else self.camera_azimuth
        heading_error = _wrapped_degrees(desired_azimuth - self.camera_azimuth)
        return {
            "primary_target_id": target_id,
            "available": True,
            "distance_xy": round(distance_xy, 5),
            "center_distance_xy": round(center_distance_xy, 5),
            "progress_xy": round(initial_distance - distance_xy, 5),
            "desired_camera_azimuth": round(desired_azimuth, 5),
            "heading_error_degrees": round(heading_error, 5),
            "within_interaction_range": distance_xy <= 0.4,
            "target_position": [round(float(value), 5) for value in target_position],
            "target_size": [round(float(value), 5) for value in target_spec.size] if target_spec else None,
            "relative": {
                "forward": round(dx * forward_xy[0] + dy * forward_xy[1], 5),
                "right": round(dx * right_xy[0] + dy * right_xy[1], 5),
                "up": round(target_position[2] - agent_position[2], 5),
            },
        }

    def _current_navigation_target_id(self) -> str | None:
        if not self.attempts:
            return None
        objective = self.attempts[-1].objective(final=False)
        focus = objective_focus(
            trial=self.trial,
            objective_state=objective,
            objects=self._current_scene_objects(),
        )
        targets = focus.get("navigation_target_ids") or focus.get("target_ids") or []
        return str(targets[0]) if targets else None

    def _vertical_velocity(self) -> float:
        return float(self.simulation.data.qvel[self.simulation.agent.qvel_address + 2])

    def _render_first_person(self, evidence_context: dict[str, Any]) -> Path | None:
        if not self.render_frames or self.frame_dir is None:
            return None
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image

            if self._renderer is None:
                self._renderer = self.simulation.mujoco.Renderer(
                    self.simulation.model,
                    height=FIRST_PERSON_HEIGHT,
                    width=FIRST_PERSON_WIDTH,
                )
                self._scene_option = self.simulation.mujoco.MjvOption()
                self._scene_option.geomgroup[5] = 0
            camera = self.simulation.mujoco.MjvCamera()
            camera.type = self.simulation.mujoco.mjtCamera.mjCAMERA_FREE
            position = self.simulation.agent_position()
            forward_xy = camera_relative_velocity(
                self.camera_azimuth,
                right=0.0,
                forward=1.0,
                speed=1.0,
            )
            head_z = position[2] + self.simulation.agent.height * 0.28
            eye_forward = float(self.simulation.agent_spec.size[0]) * 0.5 + 0.08
            camera.lookat[:] = [
                position[0] + forward_xy[0] * (1.0 + eye_forward),
                position[1] + forward_xy[1] * (1.0 + eye_forward),
                head_z + math.tan(math.radians(self.camera_elevation)),
            ]
            camera.distance = 1.0
            # Controller yaw is measured from screen-forward; MuJoCo's free camera
            # uses +x as azimuth zero.
            camera.azimuth = -self.camera_azimuth - 90.0
            camera.elevation = self.camera_elevation
            self._renderer.update_scene(
                self.simulation.data,
                camera=camera,
                scene_option=self._scene_option,
            )
            pixels = self._renderer.render()
            path = self.frame_dir / f"frame_{self._frame_sequence:04d}.png"
            frame_index = self._frame_sequence
            self._frame_sequence += 1
            Image.fromarray(pixels).save(path)
            self._append_frame_manifest(
                path=path,
                frame_index=frame_index,
                evidence_context=evidence_context,
            )
            return path
        except Exception as exc:
            self._renderer_error = str(exc)
            return None

    def _append_frame_manifest(
        self,
        *,
        path: Path,
        frame_index: int,
        evidence_context: dict[str, Any],
    ) -> None:
        if self.frame_dir is None:
            return
        events = self._frame_events(evidence_context.get("objective_focus") or {})
        record = {
            "index": frame_index,
            "path": str(path),
            "total_step": self.total_steps,
            "attempt_step": self.attempt_steps,
            "attempt": len(self.attempts),
            "camera": {
                "azimuth": round(self.camera_azimuth, 5),
                "elevation": round(self.camera_elevation, 5),
            },
            "grounded": self.simulation.is_grounded(),
            "events": events,
            **evidence_context,
        }
        manifest = self.frame_dir / FRAME_MANIFEST_FILENAME
        with manifest.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")

    def _frame_events(self, focus: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.attempts:
            return []
        tracker = self.attempts[-1]
        start = self._frame_event_counts.get(tracker.attempt, 0)
        self._frame_event_counts[tracker.attempt] = len(tracker.events)
        semantic_types = {
            obj.id: obj.semantic_type
            for obj in self.simulation.spec.objects
        }
        target_ids = {
            str(value)
            for value in [
                *(focus.get("subject_ids") or []),
                *(focus.get("target_ids") or []),
            ]
        }
        values = []
        for raw in tracker.events[start:]:
            event = dict(raw)
            object_id = str(event.get("object_id") or "")
            semantic_type = semantic_types.get(object_id, "")
            objective_relevant = object_id in target_ids
            event["semantic_type"] = semantic_type
            event["objective_relevant"] = objective_relevant
            event["routine"] = bool(
                event.get("type") == "contact_started"
                and semantic_type in {"ground", "platform", "ramp"}
                and not objective_relevant
            )
            values.append(event)
        return values

    def _append_action(self, action: dict[str, Any]) -> None:
        if self.action_log_path is None:
            return
        self.action_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.action_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(action, separators=(",", ":"), ensure_ascii=False) + "\n")

    def _append_observation(self, observation: dict[str, Any]) -> None:
        if self.observation_log_path is None:
            return
        self.observation_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "sequence": self._observation_sequence,
            **observation,
        }
        self._observation_sequence += 1
        with self.observation_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")


class TrialAttemptTracker:
    def __init__(
        self,
        *,
        simulation: PlayableSimulation,
        trial: dict[str, Any],
        attempt: int,
        total_step: int = 0,
        reset_count: int = 0,
    ) -> None:
        self.simulation = simulation
        self.trial = trial
        self.attempt = attempt
        self.objects = scene_objects(simulation.spec)
        self.initial_positions = _current_positions(simulation)
        self.max_agent_xy = 0.0
        self.max_agent_xyz = 0.0
        self.max_height_gain = 0.0
        self.jump_count = 0
        self.zone_entry_counts: dict[str, int] = {}
        self.contact_counts: dict[str, int] = {}
        self.active_zones: set[str] = set()
        self.active_contacts: set[str] = set()
        self.mechanism_active_ids: set[str] = set()
        self.terminal_event_counts: dict[str, int] = {}
        self.reset_counts_by_reason: dict[str, int] = {}
        self.terminal_outcome = ""
        self.reset_reason = ""
        self.events: list[dict[str, Any]] = []
        self.last_step = total_step
        initial_frame = capture_trajectory_frame(
            simulation,
            reset_count=reset_count,
            attempt=attempt,
            total_step=total_step,
        )
        self.frames = [initial_frame]
        constraints = self.trial.get("constraints") or {
            "mode": "all",
            "checks": [],
            "ordered_check_ids": [],
        }
        self.objective_monitor = AssertionSatisfactionMonitor(
            group=self.trial["objective"],
            objects=self.objects,
            initial_frame=initial_frame,
        )
        self.constraints_monitor = AssertionSatisfactionMonitor(
            group=constraints,
            objects=self.objects,
            initial_frame=initial_frame,
        )

    def record_step(self, *, total_step: int, jump_started: bool) -> None:
        self.last_step = total_step
        positions = _current_positions(self.simulation)
        start_agent = self.initial_positions[self.simulation.agent.object_id]
        current_agent = positions[self.simulation.agent.object_id]
        self.max_agent_xy = max(self.max_agent_xy, math.dist(start_agent[:2], current_agent[:2]))
        self.max_agent_xyz = max(self.max_agent_xyz, math.dist(start_agent, current_agent))
        self.max_height_gain = max(self.max_height_gain, current_agent[2] - start_agent[2])
        if jump_started:
            self.jump_count += 1
            self.events.append(_event("jump", total_step, self.attempt, self.simulation.agent.object_id))
        self._record_zone_entries(total_step)
        self._record_contacts(total_step)
        self._record_mechanisms(total_step)
        frame = capture_trajectory_frame(
            self.simulation,
            reset_count=self.attempt - 1,
            attempt=self.attempt,
            total_step=total_step,
        )
        self.frames.append(frame)
        self.objective_monitor.update(frame)
        self.constraints_monitor.update(frame)

    def objective(self, *, final: bool) -> dict[str, Any]:
        return self._evaluate_group(self.trial["objective"], final=final)

    def constraints(self, *, final: bool) -> dict[str, Any]:
        raw = self.trial.get("constraints") or {
            "mode": "all",
            "checks": [],
            "ordered_check_ids": [],
        }
        return self._evaluate_group(raw, final=final)

    def trial_state(self, *, final: bool) -> dict[str, Any]:
        objective = self.objective(final=final)
        constraints = self.constraints(final=final)
        raw_satisfied = bool(objective["satisfied"])
        terminal_failure = self.terminal_outcome if self.terminal_outcome == "out_of_bounds" else ""
        terminal_accepted = not terminal_failure or self._objective_accepts_terminal_failure(
            objective,
            terminal_failure,
        )
        if raw_satisfied and not terminal_accepted:
            objective = {
                **objective,
                "raw_satisfied": True,
                "satisfied": False,
                "disqualified_by_terminal_event": terminal_failure,
            }
        else:
            objective = {
                **objective,
                "raw_satisfied": raw_satisfied,
                "disqualified_by_terminal_event": None,
            }
        return {
            "satisfied": bool(objective["satisfied"] and constraints["satisfied"]),
            "objective": objective,
            "constraints": constraints,
            "terminal_outcome": self.terminal_outcome or None,
            "terminal_event_accepted": bool(terminal_accepted),
        }

    def record_terminal_event(
        self,
        *,
        total_step: int,
        outcome: str,
        object_id: str = "",
    ) -> None:
        if self.terminal_outcome:
            return
        self.terminal_outcome = outcome
        self.terminal_event_counts[outcome] = self.terminal_event_counts.get(outcome, 0) + 1
        self.events.append(
            _event(
                "terminal_event",
                total_step,
                self.attempt,
                object_id,
                outcome=outcome,
            )
        )

    def record_attempt_reset(self, *, total_step: int, reason: str) -> None:
        self.reset_reason = reason
        self.reset_counts_by_reason[reason] = self.reset_counts_by_reason.get(reason, 0) + 1
        self.events.append(
            _event(
                "attempt_reset",
                total_step,
                self.attempt,
                outcome=reason,
                reason=reason,
            )
        )
        frame = capture_trajectory_frame(
            self.simulation,
            reset_count=self.attempt,
            attempt=self.attempt,
            total_step=total_step,
            reset_reason=reason,
        )
        self.frames.append(frame)
        self.objective_monitor.update(frame)
        self.constraints_monitor.update(frame)

    @property
    def satisfied(self) -> bool:
        if not self.objective_monitor.satisfied or not self.constraints_monitor.satisfied:
            return False
        terminal_failure = self.terminal_outcome if self.terminal_outcome == "out_of_bounds" else ""
        if not terminal_failure:
            return True
        for check in (self.trial.get("objective") or {}).get("checks") or []:
            predicate = check.get("predicate") or {}
            if not self.objective_monitor.condition_passed(str(check.get("id") or "")):
                continue
            if predicate.get("type") == "terminal_event" and predicate.get("event") == terminal_failure:
                return True
            if predicate.get("type") == "reset_event" and predicate.get("reason") in {
                "any",
                terminal_failure,
            }:
                return True
        return False

    def _evaluate_group(self, group: dict[str, Any], *, final: bool) -> dict[str, Any]:
        return evaluate_assertion_group(
            group=group,
            frames=self.frames,
            objects=self.objects,
            final=final,
        )

    def summary(self, *, final: bool) -> dict[str, Any]:
        state = self.trial_state(final=final)
        return {
            "attempt": self.attempt,
            "objective": state["objective"],
            "constraints": state["constraints"],
            "terminal_outcome": state["terminal_outcome"],
            "terminal_event_accepted": state["terminal_event_accepted"],
            "reset_reason": self.reset_reason or None,
            "jump_count": self.jump_count,
            "zone_entry_counts": dict(self.zone_entry_counts),
            "contact_counts": dict(self.contact_counts),
            "terminal_event_counts": dict(self.terminal_event_counts),
            "reset_counts_by_reason": dict(self.reset_counts_by_reason),
            "max_agent_displacement_xy": round(self.max_agent_xy, 5),
            "max_agent_displacement_xyz": round(self.max_agent_xyz, 5),
            "max_agent_height_gain": round(self.max_height_gain, 5),
        }

    def _record_zone_entries(self, step: int) -> None:
        positions = _current_positions(self.simulation)
        current_objects = {
            obj.id: scene_object_at(obj, positions.get(obj.id, obj.position))
            for obj in self.objects
        }
        agent = current_objects[self.simulation.agent.object_id]
        active = {
            obj.id
            for obj in current_objects.values()
            if obj.body_type == "sensor" and _scene_objects_overlap(agent, obj)
        }
        for zone_id in sorted(active - self.active_zones):
            self.zone_entry_counts[zone_id] = self.zone_entry_counts.get(zone_id, 0) + 1
            self.events.append(_event("zone_entered", step, self.attempt, zone_id))
        self.active_zones = active

    def _record_contacts(self, step: int) -> None:
        active = _agent_contact_object_ids(self.simulation)
        for object_id in sorted(active - self.active_contacts):
            self.contact_counts[object_id] = self.contact_counts.get(object_id, 0) + 1
            self.events.append(_event("contact_started", step, self.attempt, object_id))
        self.active_contacts = active

    def _record_mechanisms(self, step: int) -> None:
        for state in self.simulation.mechanism_states():
            mechanism_id = str(state["id"])
            if state.get("active") and mechanism_id not in self.mechanism_active_ids:
                self.mechanism_active_ids.add(mechanism_id)
                self.events.append(_event("mechanism_activated", step, self.attempt, mechanism_id))

    def _objective_accepts_terminal_failure(
        self,
        objective: dict[str, Any],
        terminal_failure: str,
    ) -> bool:
        results = {result["id"]: result for result in objective.get("checks") or []}
        for check in (self.trial.get("objective") or {}).get("checks") or []:
            result = results.get(check.get("id")) or {}
            if not result.get("passed"):
                continue
            predicate = check.get("predicate") or {}
            if predicate.get("type") == "terminal_event" and predicate.get("event") == terminal_failure:
                return True
            if predicate.get("type") == "reset_event" and predicate.get("reason") in {
                "any",
                terminal_failure,
            }:
                return True
        return False


def replay_behavior_actions(
    *,
    scene_dir: Path,
    trial: dict[str, Any],
    actions: list[dict[str, Any]],
    frame_dir: Path | None = None,
) -> dict[str, Any]:
    session = BehaviorTrialSession(
        scene_dir=scene_dir,
        trial=trial,
        frame_dir=frame_dir,
        render_frames=frame_dir is not None,
    )
    session.start()
    try:
        for action in actions:
            if action.get("reset") or action.get("action") == "reset":
                if (
                    session.reset_count < session.max_resets
                    and not session._combined_trial_state(final=False)["satisfied"]
                ):
                    session.reset()
                continue
            if session.terminal:
                break
            session.act(
                forward=action.get("forward", 0.0),
                right=action.get("right", 0.0),
                look_x=action.get("look_x", 0.0),
                look_y=action.get("look_y", 0.0),
                jump=action.get("jump", False),
                frames=action.get("frames_advanced", action.get("frames", 12)),
            )
        return session.stop()
    finally:
        session.close()


def _compact_evidence_group(group: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": group.get("mode") or "all",
        "satisfied": bool(group.get("satisfied")),
        "raw_satisfied": bool(group.get("raw_satisfied", group.get("satisfied"))),
        "disqualified_by_terminal_event": group.get("disqualified_by_terminal_event"),
        "passed_count": int(group.get("passed_count") or 0),
        "total_count": int(group.get("total_count") or 0),
        "ordered_check_ids": list(group.get("ordered_check_ids") or []),
        "ordered_steps": list(group.get("ordered_steps") or []),
        "order_satisfied": bool(group.get("order_satisfied", True)),
        "checks": [
            {
                "id": check.get("id"),
                "type": check.get("type"),
                "description": check.get("description") or "",
                "passed": bool(check.get("passed")),
                "status": check.get("status"),
                "first_satisfied_step": check.get("first_satisfied_step"),
            }
            for check in group.get("checks") or []
            if isinstance(check, dict)
        ],
    }


def read_action_log(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    actions = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            actions.append(value)
    return actions[:MAX_ACTION_SEGMENTS + 4]


def _current_positions(simulation: PlayableSimulation) -> dict[str, tuple[float, float, float]]:
    positions = {
        obj.id: tuple(float(value) for value in obj.position)
        for obj in simulation.spec.objects
    }
    for object_id, body_id in simulation.dynamic_body_ids.items():
        positions[object_id] = tuple(float(value) for value in simulation.data.xpos[body_id])
    return positions


def _agent_contact_object_ids(simulation: PlayableSimulation) -> set[str]:
    active: set[str] = set()
    for index in range(int(simulation.data.ncon)):
        contact = simulation.data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 in simulation.agent.geom_ids:
            other = geom2
        elif geom2 in simulation.agent.geom_ids:
            other = geom1
        else:
            continue
        object_id = _geom_object_id(simulation, other)
        if object_id and object_id != simulation.agent.object_id:
            active.add(object_id)
    return active


def _scene_objects_overlap(left: SceneObject3D, right: SceneObject3D) -> bool:
    return volumes_overlap(left, right)


def _geom_object_id(simulation: PlayableSimulation, geom_id: int) -> str:
    mujoco = simulation.mujoco
    name = mujoco.mj_id2name(simulation.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    if any(obj.id == name for obj in simulation.spec.objects):
        return name
    body_id = int(simulation.model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(simulation.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    return body_name


def _wrapped_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _event(
    event_type: str,
    step: int,
    attempt: int,
    object_id: str = "",
    **details: Any,
) -> dict[str, Any]:
    event = {"type": event_type, "step": step, "attempt": attempt}
    if object_id:
        event["object_id"] = object_id
    event.update(details)
    return event


def _repair_hints(
    trial: dict[str, Any],
    status: str,
    objective: dict[str, Any],
    *,
    search_evidence: dict[str, Any] | None = None,
) -> list[str]:
    if status == "passed":
        return []
    failed = [check for check in objective.get("checks") or [] if not check.get("passed")]
    names = [describe_assertion_condition(check).rstrip(". ") for check in failed]
    if status == "failed":
        return ["The child demonstrated the forbidden behavior; change the relevant geometry or affordance."]
    if trial.get("expected_outcome") == "should_not_succeed" and search_evidence:
        return [negative_search_repair_hint(search_evidence)]
    if names:
        return ["Review or repair the unmet objective checks: " + ", ".join(names[:5]) + "."]
    return ["Review the trial route and rerun it before changing the environment."]


def _bounded_float(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return max(minimum, min(maximum, result))


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
