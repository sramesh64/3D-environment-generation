"""Deterministic trajectory capture and trusted assertion evaluation.

Tasks and behavior trials deliberately share this module. Callers define typed
subject/target predicates over immutable simulator frames; this module never
executes generated verifier code.
"""

from __future__ import annotations

import math
from typing import Any

from .env_verification import (
    SceneObject3D,
    scene_object_at,
    scene_objects,
    select_scene_objects,
    spatial_relation_holds,
)
from .player import PlayableSimulation
from .scene_geometry import volumes_overlap


TRAJECTORY_SAMPLE_SECONDS = 0.05


class AssertionSatisfactionMonitor:
    """Incrementally track whether one canonical assertion group is satisfied.

    Detailed reports still use the batch evaluator below. Runtime controllers use
    this monitor only to decide whether to stop on the current physics frame.
    """

    def __init__(
        self,
        *,
        group: dict[str, Any],
        objects: list[SceneObject3D],
        initial_frame: dict[str, Any],
    ) -> None:
        self.mode = str(group.get("mode") or "all")
        self.ordered_ids = list(
            group.get("ordered_condition_ids")
            or group.get("ordered_check_ids")
            or []
        )
        conditions = list(group.get("conditions") or group.get("checks") or [])
        self.conditions = [
            _ConditionSatisfactionState(
                condition=condition,
                objects=objects,
                initial=initial_frame,
            )
            for condition in conditions
        ]
        self._by_id = {state.id: state for state in self.conditions}
        self.frame_count = 0
        self.update(initial_frame)

    def update(self, frame: dict[str, Any]) -> None:
        frame_index = self.frame_count
        for state in self.conditions:
            state.update(frame=frame, frame_index=frame_index)
        self.frame_count += 1

    @property
    def satisfied(self) -> bool:
        values = [state.passed for state in self.conditions]
        base = all(values) if self.mode == "all" else any(values)
        ordered_candidates = [
            self._by_id[condition_id].satisfaction_indices
            if condition_id in self._by_id
            else []
            for condition_id in self.ordered_ids
        ]
        _, order_satisfied = _select_ordered_witnesses(ordered_candidates)
        return bool(base and order_satisfied)

    def condition_passed(self, condition_id: str) -> bool:
        state = self._by_id.get(str(condition_id))
        return bool(state and state.passed)

    @property
    def irreversibly_failed(self) -> bool:
        failures = [state.irreversibly_failed for state in self.conditions]
        if not failures:
            return False
        return any(failures) if self.mode == "all" else all(failures)


class _ConditionSatisfactionState:
    def __init__(
        self,
        *,
        condition: dict[str, Any],
        objects: list[SceneObject3D],
        initial: dict[str, Any],
    ) -> None:
        self.condition = condition
        self.id = str(condition.get("id") or "")
        self.objects = objects
        self.initial = initial
        self.predicate = condition["predicate"]
        self.subjects = select_scene_objects(
            objects,
            self.predicate.get("subject") or {},
        ) if self.predicate.get("subject") else []
        self.running_values = {subject.id: 0.0 for subject in self.subjects}
        self.true_count = 0
        self.transition_count = 0
        self.rising_edges: list[int] = []
        self.sustained_edges: list[int] = []
        self.current_streak = 0
        self.longest_streak = 0
        self.first_true_index: int | None = None
        self.first_sustained_index: int | None = None
        self.current_value = False
        self.frame_count = 0

    def update(self, *, frame: dict[str, Any], frame_index: int) -> None:
        value = bool(self._evaluate(frame=frame)["value"])
        if value:
            self.true_count += 1
            self.current_streak += 1
            self.longest_streak = max(self.longest_streak, self.current_streak)
            if self.first_true_index is None:
                self.first_true_index = frame_index
            required = int(self.condition.get("frames") or 1)
            if (
                self.condition.get("temporal") == "sustained"
                and self.first_sustained_index is None
                and self.current_streak >= required
            ):
                self.first_sustained_index = frame_index
            if (
                self.condition.get("temporal") == "sustained"
                and self.current_streak == required
            ):
                self.sustained_edges.append(frame_index)
        else:
            self.current_streak = 0
        if value and not self.current_value:
            self.transition_count += 1
            self.rising_edges.append(frame_index)
        self.current_value = value
        self.frame_count += 1

    @property
    def passed(self) -> bool:
        temporal = str(self.condition.get("temporal") or "eventually")
        if temporal == "eventually":
            return self.first_true_index is not None
        if temporal == "at_end":
            return self.current_value
        if temporal == "always":
            return self.true_count == self.frame_count
        if temporal == "never":
            return self.true_count == 0
        if temporal == "sustained":
            return self.longest_streak >= int(self.condition.get("frames") or 1)
        return _within(
            self.transition_count,
            self.condition.get("min_count"),
            self.condition.get("max_count"),
        )

    @property
    def satisfaction_indices(self) -> list[int]:
        if not self.passed:
            return []
        temporal = str(self.condition.get("temporal") or "eventually")
        if temporal == "eventually":
            return list(self.rising_edges)
        if temporal == "sustained":
            return list(self.sustained_edges)
        if temporal == "count":
            minimum = int(self.condition.get("min_count") or 0)
            if minimum:
                return list(self.rising_edges[minimum - 1 :])
        return [self.frame_count - 1]

    @property
    def irreversibly_failed(self) -> bool:
        if self.passed:
            return False
        temporal = str(self.condition.get("temporal") or "eventually")
        if temporal in {"always", "never"}:
            return True
        if temporal != "count" or self.condition.get("max_count") is None:
            return False
        return self.transition_count > int(self.condition["max_count"])

    def _evaluate(self, *, frame: dict[str, Any]) -> dict[str, Any]:
        predicate_type = self.predicate["type"]
        if predicate_type not in {"displacement", "axis_delta"}:
            return _evaluate_predicate(
                self.predicate,
                frame=frame,
                frame_index=0,
                frames=[frame],
                objects=self.objects,
                initial=self.initial,
            )

        metric = str(self.predicate.get("metric") or "maximum")
        values: dict[str, float] = {}
        for subject in self.subjects:
            start = self.initial["positions"].get(subject.id)
            position = frame["positions"].get(subject.id)
            if start is None:
                continue
            if predicate_type == "displacement":
                current = (
                    _distance(
                        start,
                        position,
                        space=self.predicate.get("space") or "xy",
                    )
                    if position is not None
                    else 0.0
                )
                self.running_values[subject.id] = (
                    current
                    if metric == "final"
                    else max(self.running_values[subject.id], current)
                )
            else:
                axis = {"x": 0, "y": 1, "z": 2}[self.predicate["axis"]]
                current = (
                    float(position[axis]) - float(start[axis])
                    if position is not None
                    else 0.0
                )
                if metric == "final":
                    self.running_values[subject.id] = current
                elif metric == "minimum":
                    self.running_values[subject.id] = min(
                        self.running_values[subject.id],
                        current,
                    )
                else:
                    self.running_values[subject.id] = max(
                        self.running_values[subject.id],
                        current,
                    )
            values[subject.id] = self.running_values[subject.id]
        matched = {
            object_id
            for object_id, value in values.items()
            if _within(
                value,
                self.predicate.get("min_value"),
                self.predicate.get("max_value"),
            )
        }
        return {
            "value": _quantified_subjects(
                self.subjects,
                matched,
                self.predicate,
            )
        }


class TrajectoryRecorder:
    """Capture exact evaluator frames and a smaller styled-replay trajectory."""

    def __init__(self, simulation: PlayableSimulation) -> None:
        self.simulation = simulation
        self.objects = scene_objects(simulation.spec)
        self.object_by_id = {obj.id: obj for obj in self.objects}
        self.frames: list[dict[str, Any]] = []
        self.trajectory: list[dict[str, Any]] = []
        timestep = float(simulation.model.opt.timestep)
        self.sample_every = max(1, int(round(TRAJECTORY_SAMPLE_SECONDS / timestep)))
        self.capture(reset_count=0, force_trajectory=True)

    def capture(
        self,
        *,
        reset_count: int,
        force_trajectory: bool = False,
        total_step: int | None = None,
    ) -> dict[str, Any]:
        frame = capture_trajectory_frame(self.simulation, reset_count=reset_count)
        if total_step is not None:
            frame["step"] = int(total_step)
        self.frames.append(frame)
        if force_trajectory or frame["step"] % self.sample_every == 0:
            self.capture_trajectory(frame=frame, force=force_trajectory)
        return frame

    def capture_trajectory(self, *, frame: dict[str, Any], force: bool = False) -> None:
        value = {
            "total_step": frame["step"],
            "simulation_time": frame["simulation_time"],
            "status": self.simulation.status(),
            "grounded": frame["grounded"],
            "objects": self.simulation.body_transforms(),
            "mechanisms": frame["mechanisms"],
            "events": frame["terminal_events"],
        }
        if self.trajectory and self.trajectory[-1]["total_step"] == value["total_step"]:
            if force:
                self.trajectory[-1] = value
            return
        self.trajectory.append(value)

    def report(self, *, task: dict[str, Any], final: bool) -> dict[str, Any]:
        if self.frames:
            self.capture_trajectory(frame=self.frames[-1], force=True)
        return evaluate_assertion_tests(
            task=task,
            frames=self.frames,
            objects=self.objects,
            final=final,
        )


def capture_trajectory_frame(
    simulation: PlayableSimulation,
    *,
    reset_count: int,
    attempt: int = 1,
    total_step: int | None = None,
    reset_reason: str = "",
) -> dict[str, Any]:
    positions = {
        obj.id: [float(value) for value in obj.position]
        for obj in simulation.spec.objects
    }
    rotations = {
        obj.id: _yaw_matrix(float(obj.yaw))
        for obj in simulation.spec.objects
    }
    velocities = {
        obj.id: {"linear": [0.0, 0.0, 0.0], "angular": [0.0, 0.0, 0.0]}
        for obj in simulation.spec.objects
    }
    for object_id, body_id in simulation.dynamic_body_ids.items():
        positions[object_id] = [float(value) for value in simulation.data.xpos[body_id]]
        rotations[object_id] = [float(value) for value in simulation.data.xmat[body_id]]
        spatial_velocity = [float(value) for value in simulation.data.cvel[body_id]]
        velocities[object_id] = {
            "angular": spatial_velocity[:3],
            "linear": spatial_velocity[3:6],
        }
    terminal_events: list[str] = []
    if simulation.active_zone_ids("goal"):
        terminal_events.append("goal")
    if simulation.active_zone_ids("hazard"):
        terminal_events.append("hazard")
    if simulation.failure_reason() == "out_of_bounds":
        terminal_events.append("out_of_bounds")
    return {
        "step": int(simulation.step_count if total_step is None else total_step),
        "attempt": int(attempt),
        "simulation_time": float(simulation.data.time),
        "positions": positions,
        "rotations": rotations,
        "velocities": velocities,
        "contacts": [list(pair) for pair in sorted(_contact_pairs(simulation))],
        "mechanisms": simulation.mechanism_states(),
        "jump_count": int(simulation.jump_count),
        "reset_count": int(reset_count),
        "reset_reason": str(reset_reason or ""),
        "grounded": bool(simulation.is_grounded()),
        "terminal_events": terminal_events,
        "play_bounds": [
            float(value)
            for value in (
                simulation.spec.game.play_bounds
                if simulation.spec.game is not None
                else simulation.spec.world_size
            )
        ],
    }


def evaluate_assertion_tests(
    *,
    task: dict[str, Any],
    frames: list[dict[str, Any]],
    objects: list[SceneObject3D] | None = None,
    final: bool = True,
) -> dict[str, Any]:
    if not frames:
        raise ValueError("task trajectory contains no frames")
    runtime_objects = objects or scene_objects_from_task(task)
    initial = frames[0]
    test_results = [
        _evaluate_test(test, frames=frames, objects=runtime_objects, initial=initial, final=final)
        for test in task.get("tests") or []
    ]
    passed = bool(test_results) and all(result["passed"] for result in test_results)
    condition_results = [
        condition
        for result in test_results
        for condition in result.get("conditions") or []
    ]
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "final": bool(final),
        "step_count": int(frames[-1]["step"]),
        "simulation_time": float(frames[-1]["simulation_time"]),
        "summary": {
            "tests": len(test_results),
            "passed_tests": sum(result["passed"] for result in test_results),
            "conditions": len(condition_results),
            "passed_conditions": sum(result["passed"] for result in condition_results),
        },
        "tests": test_results,
        "terminal_events": sorted(
            {
                str(event)
                for frame in frames
                for event in frame.get("terminal_events") or []
            }
        ),
    }


def evaluate_assertion_group(
    *,
    group: dict[str, Any],
    frames: list[dict[str, Any]],
    objects: list[SceneObject3D],
    final: bool = True,
) -> dict[str, Any]:
    """Evaluate one task- or behavior-style assertion group.

    Behavior artifacts retain the historical ``checks`` key while tasks use
    ``conditions``. Both contain the same canonical condition shape here.
    """

    conditions = list(group.get("conditions") or group.get("checks") or [])
    test = {
        "id": str(group.get("id") or "assertion_group"),
        "description": str(group.get("description") or ""),
        "mode": str(group.get("mode") or "all"),
        "conditions": conditions,
        "ordered_condition_ids": list(
            group.get("ordered_condition_ids")
            or group.get("ordered_check_ids")
            or []
        ),
    }
    report = evaluate_assertion_tests(
        task={"tests": [test]},
        frames=frames,
        objects=objects,
        final=final,
    )
    result = report["tests"][0]
    source_by_id = {
        str(condition.get("id")): condition
        for condition in conditions
        if isinstance(condition, dict)
    }
    checks = [
        _with_legacy_metrics({
            **source_by_id.get(str(condition.get("id")), {}),
            **condition,
            "type": source_by_id.get(str(condition.get("id")), {}).get("type")
            or condition.get("predicate_type"),
        })
        for condition in result.get("conditions") or []
    ]
    return {
        "mode": result.get("mode") or "all",
        "satisfied": bool(result.get("passed")),
        "passed_count": sum(bool(item.get("passed")) for item in checks),
        "total_count": len(checks),
        "ordered_check_ids": list(result.get("ordered_condition_ids") or []),
        "ordered_steps": list(result.get("ordered_steps") or []),
        "order_satisfied": bool(result.get("order_passed", True)),
        "checks": checks,
    }


def _with_legacy_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Keep saved-report/UI aliases while predicates remain authoritative."""

    metrics = dict(result.get("metrics") or {})
    legacy_type = str(result.get("legacy_type") or result.get("type") or "")
    if legacy_type in {"agent_displacement", "object_displacement"}:
        metrics["distance"] = float(metrics.get("value") or 0.0)
    elif legacy_type == "agent_height_gain":
        metrics["height_gain"] = float(metrics.get("value") or 0.0)
    elif legacy_type in {"zone_entry", "contact_count"}:
        metrics["count"] = int(metrics.get("transition_count") or 0)
        metrics["counts"] = dict(metrics.get("counts_by_target") or {})
        metrics["matched_ids"] = sorted(metrics["counts"])
    elif legacy_type == "jump_count":
        metrics["count"] = int(metrics.get("value") or 0)
    result["metrics"] = metrics
    return result


def scene_objects_from_task(task: dict[str, Any]) -> list[SceneObject3D]:
    spec = task.get("spec")
    if not isinstance(spec, dict):
        raise ValueError("objects must be supplied when task does not embed a spec")
    return scene_objects(spec)


def _evaluate_test(
    test: dict[str, Any],
    *,
    frames: list[dict[str, Any]],
    objects: list[SceneObject3D],
    initial: dict[str, Any],
    final: bool,
) -> dict[str, Any]:
    conditions = [
        _evaluate_condition(condition, frames=frames, objects=objects, initial=initial, final=final)
        for condition in test.get("conditions") or []
    ]
    mode = str(test.get("mode") or "all")
    base_passed = all(item["passed"] for item in conditions) if mode == "all" else any(
        item["passed"] for item in conditions
    )
    ordered_ids = list(test.get("ordered_condition_ids") or [])
    by_id = {str(item["id"]): item for item in conditions}
    ordered_candidates = [
        list(by_id.get(str(item_id), {}).get("_satisfaction_frames") or [])
        for item_id in ordered_ids
    ]
    ordered_frames, order_passed = _select_ordered_witnesses(ordered_candidates)
    ordered_steps = [
        int(frames[frame_index]["step"]) if frame_index is not None else None
        for frame_index in ordered_frames
    ]
    for condition in conditions:
        condition.pop("_satisfaction_frames", None)
    passed = bool(base_passed and order_passed)
    return {
        "id": test.get("id"),
        "description": test.get("description") or "",
        "mode": mode,
        "passed": passed,
        "status": "pass" if passed else "fail",
        "conditions": conditions,
        "ordered_condition_ids": ordered_ids,
        "ordered_steps": ordered_steps,
        "ordered_frames": ordered_frames,
        "order_passed": order_passed,
    }


def _evaluate_condition(
    condition: dict[str, Any],
    *,
    frames: list[dict[str, Any]],
    objects: list[SceneObject3D],
    initial: dict[str, Any],
    final: bool,
) -> dict[str, Any]:
    predicate = condition["predicate"]
    if predicate["type"] in {"displacement", "axis_delta"}:
        evaluations = _evaluate_running_metric_series(
            predicate,
            frames=frames,
            objects=objects,
            initial=initial,
        )
    else:
        evaluations = [
            _evaluate_predicate(
                predicate,
                frame=frame,
                frame_index=index,
                frames=frames,
                objects=objects,
                initial=initial,
            )
            for index, frame in enumerate(frames)
        ]
    values = [bool(item["value"]) for item in evaluations]
    temporal = condition["temporal"]
    first_true_index = next((index for index, value in enumerate(values) if value), None)
    count = _rising_edge_count(values)
    longest_streak, _ = _longest_true_streak(values)
    if temporal == "eventually":
        passed = first_true_index is not None
    elif temporal == "at_end":
        passed = values[-1]
    elif temporal == "always":
        passed = all(values)
    elif temporal == "never":
        passed = not any(values)
    elif temporal == "sustained":
        required = int(condition.get("frames") or 1)
        passed = longest_streak >= required
    elif temporal == "count":
        passed = _within(count, condition.get("min_count"), condition.get("max_count"))
    else:  # pragma: no cover - normalized task definitions prevent this
        passed = False
    satisfaction_frames = _condition_satisfaction_frames(
        values,
        temporal=temporal,
        condition=condition,
        passed=bool(passed),
    )
    first_satisfied_index = satisfaction_frames[0] if satisfaction_frames else None
    witness = evaluations[first_satisfied_index] if first_satisfied_index is not None else evaluations[-1]
    metrics = _condition_metrics(
        temporal=temporal,
        evaluations=evaluations,
        count=count,
        longest_streak=longest_streak,
        final=final,
    )
    return {
        "id": condition.get("id"),
        "description": condition.get("description") or "",
        "temporal": temporal,
        "predicate": dict(condition["predicate"]),
        "predicate_type": condition["predicate"]["type"],
        "passed": bool(passed),
        "status": "pass" if passed else "fail",
        "provisional": not final and temporal in {"always", "never", "at_end"},
        "first_satisfied_step": (
            int(frames[first_satisfied_index]["step"])
            if first_satisfied_index is not None
            else None
        ),
        "first_satisfied_frame": first_satisfied_index,
        "_satisfaction_frames": satisfaction_frames,
        "metrics": metrics,
        "witness": witness.get("details") or {},
    }


def _evaluate_running_metric_series(
    predicate: dict[str, Any],
    *,
    frames: list[dict[str, Any]],
    objects: list[SceneObject3D],
    initial: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate history-based metrics in one pass instead of rescanning prefixes."""

    subjects = select_scene_objects(objects, predicate.get("subject") or {})
    predicate_type = predicate["type"]
    metric = str(predicate.get("metric") or "maximum")
    running = {subject.id: 0.0 for subject in subjects}
    evaluations: list[dict[str, Any]] = []
    for frame in frames:
        values: dict[str, float] = {}
        for subject in subjects:
            start = initial["positions"].get(subject.id)
            position = frame["positions"].get(subject.id)
            if start is None:
                continue
            if predicate_type == "displacement":
                current = (
                    _distance(start, position, space=predicate.get("space") or "xy")
                    if position is not None
                    else 0.0
                )
                running[subject.id] = (
                    current
                    if metric == "final"
                    else max(running[subject.id], current)
                )
            else:
                axis = {"x": 0, "y": 1, "z": 2}[predicate["axis"]]
                current = (
                    float(position[axis]) - float(start[axis])
                    if position is not None
                    else 0.0
                )
                if metric == "final":
                    running[subject.id] = current
                elif metric == "minimum":
                    running[subject.id] = min(running[subject.id], current)
                else:
                    running[subject.id] = max(running[subject.id], current)
            values[subject.id] = running[subject.id]

        matched = {
            object_id
            for object_id, value in values.items()
            if _within(value, predicate.get("min_value"), predicate.get("max_value"))
        }
        if predicate_type == "displacement":
            details = {
                "value": round(max(values.values(), default=0.0), 6),
                "distances": _rounded_dict(values),
                "matched_subject_ids": sorted(matched),
            }
        else:
            aggregate = (
                min(values.values(), default=0.0)
                if metric == "minimum"
                else max(values.values(), default=0.0)
            )
            details = {
                "value": round(aggregate, 6),
                "axis": predicate["axis"],
                "metric": metric,
                "values": _rounded_dict(values),
                "matched_subject_ids": sorted(matched),
            }
        evaluations.append(
            {
                "value": _quantified_subjects(subjects, matched, predicate),
                "details": details,
            }
        )
    return evaluations


def _evaluate_predicate(
    predicate: dict[str, Any],
    *,
    frame: dict[str, Any],
    frame_index: int,
    frames: list[dict[str, Any]],
    objects: list[SceneObject3D],
    initial: dict[str, Any],
) -> dict[str, Any]:
    predicate_type = predicate["type"]
    subjects = select_scene_objects(objects, predicate.get("subject") or {}) if predicate.get("subject") else []
    targets = select_scene_objects(objects, predicate.get("target") or {}) if predicate.get("target") else []
    if predicate_type == "overlap":
        pairs = [
            [subject.id, target.id]
            for subject in subjects
            for target in targets
            if subject.id != target.id
            and _objects_overlap(
                _runtime_object(subject, frame),
                _runtime_object(target, frame),
            )
        ]
        matched = {pair[0] for pair in pairs}
        return {
            "value": _quantified_subjects(subjects, matched, predicate),
            "details": {"overlapping_pairs": pairs, "matched_subject_ids": sorted(matched)},
        }
    if predicate_type == "relation":
        pairs = []
        for subject in subjects:
            for target in targets:
                if subject.id == target.id:
                    continue
                if spatial_relation_holds(
                    _runtime_object(subject, frame),
                    _runtime_object(target, frame),
                    relation=predicate["relation"],
                    margin=float(predicate.get("margin") or 0.0),
                    min_distance=predicate.get("min_distance"),
                    max_distance=predicate.get("max_distance"),
                ):
                    pairs.append([subject.id, target.id])
        matched = {pair[0] for pair in pairs}
        return {
            "value": _quantified_subjects(subjects, matched, predicate),
            "details": {"matching_pairs": pairs, "matched_subject_ids": sorted(matched)},
        }
    if predicate_type == "displacement":
        distances: dict[str, float] = {}
        for subject in subjects:
            start = initial["positions"].get(subject.id)
            if not start:
                continue
            if predicate.get("metric") == "final":
                positions = [frame["positions"].get(subject.id)]
            else:
                positions = [item["positions"].get(subject.id) for item in frames[: frame_index + 1]]
            values = [
                _distance(start, position, space=predicate.get("space") or "xy")
                for position in positions
                if position is not None
            ]
            distances[subject.id] = max(values, default=0.0)
        value = max(distances.values(), default=0.0)
        matched = {
            object_id
            for object_id, distance in distances.items()
            if _within(distance, predicate.get("min_value"), predicate.get("max_value"))
        }
        return {
            "value": _quantified_subjects(subjects, matched, predicate),
            "details": {
                "value": round(value, 6),
                "distances": _rounded_dict(distances),
                "matched_subject_ids": sorted(matched),
            },
        }
    if predicate_type == "axis_delta":
        axis = {"x": 0, "y": 1, "z": 2}[predicate["axis"]]
        values: dict[str, float] = {}
        for subject in subjects:
            start = initial["positions"].get(subject.id)
            if start is None:
                continue
            if predicate.get("metric") == "final":
                samples = [frame["positions"].get(subject.id)]
            else:
                samples = [
                    item["positions"].get(subject.id)
                    for item in frames[: frame_index + 1]
                ]
            deltas = [
                float(position[axis]) - float(start[axis])
                for position in samples
                if position is not None
            ]
            metric = predicate.get("metric") or "maximum"
            values[subject.id] = (
                min(deltas, default=0.0)
                if metric == "minimum"
                else deltas[-1] if metric == "final" and deltas
                else max(deltas, default=0.0)
            )
        matched = {
            object_id
            for object_id, value in values.items()
            if _within(value, predicate.get("min_value"), predicate.get("max_value"))
        }
        aggregate = (
            min(values.values(), default=0.0)
            if (predicate.get("metric") or "maximum") == "minimum"
            else max(values.values(), default=0.0)
        )
        return {
            "value": _quantified_subjects(subjects, matched, predicate),
            "details": {
                "value": round(aggregate, 6),
                "axis": predicate["axis"],
                "metric": predicate.get("metric") or "maximum",
                "values": _rounded_dict(values),
                "matched_subject_ids": sorted(matched),
            },
        }
    if predicate_type == "contact":
        active = {tuple(sorted(pair)) for pair in frame.get("contacts") or []}
        pairs = [
            [subject.id, target.id]
            for subject in subjects
            for target in targets
            if subject.id != target.id and tuple(sorted((subject.id, target.id))) in active
        ]
        matched = {pair[0] for pair in pairs}
        return {
            "value": _quantified_subjects(subjects, matched, predicate),
            "details": {"contact_pairs": pairs, "matched_subject_ids": sorted(matched)},
        }
    if predicate_type == "axis_value":
        axis = {"x": 0, "y": 1, "z": 2}[predicate["axis"]]
        values = {
            subject.id: float(frame["positions"][subject.id][axis])
            for subject in subjects
            if subject.id in frame["positions"]
        }
        matched = {
            object_id
            for object_id, value in values.items()
            if _within(value, predicate.get("min_value"), predicate.get("max_value"))
        }
        return {
            "value": _quantified_subjects(subjects, matched, predicate),
            "details": {
                "axis": predicate["axis"],
                "values": _rounded_dict(values),
                "matched_subject_ids": sorted(matched),
            },
        }
    if predicate_type == "speed":
        component = predicate.get("component") or "linear"
        values = {
            subject.id: math.sqrt(
                sum(float(value) ** 2 for value in frame["velocities"][subject.id][component])
            )
            for subject in subjects
            if subject.id in frame["velocities"]
        }
        matched = {
            object_id
            for object_id, value in values.items()
            if _within(value, predicate.get("min_value"), predicate.get("max_value"))
        }
        return {
            "value": _quantified_subjects(subjects, matched, predicate),
            "details": {
                "component": component,
                "speeds": _rounded_dict(values),
                "matched_subject_ids": sorted(matched),
            },
        }
    if predicate_type == "settled":
        speeds = {}
        for subject in subjects:
            velocity = frame["velocities"].get(subject.id)
            if not velocity:
                continue
            linear = math.sqrt(sum(float(value) ** 2 for value in velocity["linear"]))
            angular = math.sqrt(sum(float(value) ** 2 for value in velocity["angular"]))
            speeds[subject.id] = {"linear": linear, "angular": angular}
        matched = {
            object_id
            for object_id, value in speeds.items()
            if value["linear"] <= float(predicate["max_linear_speed"])
            and value["angular"] <= float(predicate["max_angular_speed"])
        }
        passed = _quantified_subjects(subjects, matched, predicate)
        return {
            "value": passed,
            "details": {
                "speeds": {
                    key: {name: round(metric, 6) for name, metric in value.items()}
                    for key, value in speeds.items()
                },
                "matched_subject_ids": sorted(matched),
            },
        }
    if predicate_type == "mechanism_state":
        mechanism = next(
            (item for item in frame.get("mechanisms") or [] if item.get("id") == predicate["mechanism_id"]),
            None,
        )
        if predicate.get("state") == "active":
            passed = bool(mechanism and mechanism.get("active"))
        else:
            passed = bool(
                mechanism
                and float(mechanism.get("progress") or 0.0) >= float(predicate.get("min_progress") or 0.9)
            )
        return {"value": passed, "details": {"mechanism": mechanism or {}}}
    if predicate_type in {"jump_count", "step_count", "reset_count"}:
        key = predicate_type
        value = int(frame["jump_count"] if key == "jump_count" else frame["step"] if key == "step_count" else frame["reset_count"])
        return {
            "value": _within(value, predicate.get("min_value"), predicate.get("max_value")),
            "details": {"value": value},
        }
    if predicate_type == "reset_event":
        reason = str(predicate.get("reason") or "any")
        observed = str(frame.get("reset_reason") or "")
        return {
            "value": bool(observed) and (reason == "any" or observed == reason),
            "details": {"reason": reason, "observed_reason": observed},
        }
    if predicate_type == "terminal_event":
        event = predicate["event"]
        return {
            "value": event in set(frame.get("terminal_events") or []),
            "details": {"event": event, "active_events": frame.get("terminal_events") or []},
        }
    if predicate_type == "in_bounds":
        bounds = _play_bounds(frame=frame, objects=objects, subjects=subjects)
        values = {
            subject.id: _object_in_bounds(_runtime_object(subject, frame), bounds)
            for subject in subjects
        }
        matched = {object_id for object_id, value in values.items() if value}
        return {
            "value": _quantified_subjects(subjects, matched, predicate),
            "details": {"objects": values, "bounds": bounds, "matched_subject_ids": sorted(matched)},
        }
    if predicate_type == "grounded":
        agent_ids = {subject.id for subject in subjects if subject.semantic_type == "agent"}
        return {
            "value": bool(agent_ids) and bool(frame.get("grounded")),
            "details": {"grounded": bool(frame.get("grounded")), "matched_ids": sorted(agent_ids)},
        }
    return {"value": False, "details": {"error": f"unsupported predicate {predicate_type}"}}


def _condition_metrics(
    *,
    temporal: str,
    evaluations: list[dict[str, Any]],
    count: int,
    longest_streak: int,
    final: bool,
) -> dict[str, Any]:
    true_count = sum(bool(item["value"]) for item in evaluations)
    metrics: dict[str, Any] = {
        "frames_observed": len(evaluations),
        "true_frames": true_count,
        "transition_count": count,
        "longest_true_streak": longest_streak,
        "currently_true": bool(evaluations[-1]["value"]),
        "final_evaluation": bool(final),
        "operator": temporal,
    }
    details = evaluations[-1].get("details") or {}
    metrics.update(details)
    pair_key = (
        "overlapping_pairs"
        if any("overlapping_pairs" in (item.get("details") or {}) for item in evaluations)
        else "contact_pairs"
        if any("contact_pairs" in (item.get("details") or {}) for item in evaluations)
        else ""
    )
    if pair_key:
        pair_counts = _pair_transition_counts(
            [item.get("details", {}).get(pair_key) or [] for item in evaluations]
        )
        metrics["pair_transition_counts"] = {
            f"{subject_id}->{target_id}": value
            for (subject_id, target_id), value in sorted(pair_counts.items())
        }
        counts_by_subject: dict[str, int] = {}
        counts_by_target: dict[str, int] = {}
        for (subject_id, target_id), value in pair_counts.items():
            counts_by_subject[subject_id] = counts_by_subject.get(subject_id, 0) + value
            counts_by_target[target_id] = counts_by_target.get(target_id, 0) + value
        metrics["counts_by_subject"] = counts_by_subject
        metrics["counts_by_target"] = counts_by_target
    return metrics


def _pair_transition_counts(
    samples: list[list[list[str]]],
) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    previous: set[tuple[str, str]] = set()
    for sample in samples:
        active = {
            (str(pair[0]), str(pair[1]))
            for pair in sample
            if isinstance(pair, list) and len(pair) == 2
        }
        for pair in active - previous:
            counts[pair] = counts.get(pair, 0) + 1
        previous = active
    return counts


def _runtime_object(obj: SceneObject3D, frame: dict[str, Any]) -> SceneObject3D:
    position = frame["positions"].get(obj.id, obj.position)
    matrix = frame["rotations"].get(obj.id)
    yaw = _matrix_yaw(matrix) if matrix else obj.yaw
    return scene_object_at(obj, position, yaw=yaw)


def _objects_overlap(subject: SceneObject3D, target: SceneObject3D) -> bool:
    return volumes_overlap(subject, target)


def _contact_pairs(simulation: PlayableSimulation) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for index in range(int(simulation.data.ncon)):
        contact = simulation.data.contact[index]
        first = _geom_object_id(simulation, int(contact.geom1))
        second = _geom_object_id(simulation, int(contact.geom2))
        if first and second and first != second:
            pairs.add(tuple(sorted((first, second))))
    return pairs


def _geom_object_id(simulation: PlayableSimulation, geom_id: int) -> str:
    mujoco = simulation.mujoco
    name = mujoco.mj_id2name(simulation.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    if name in simulation.object_specs:
        return str(name)
    body_id = int(simulation.model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(simulation.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    return str(body_name) if body_name in simulation.object_specs else ""


def _play_bounds(
    *,
    frame: dict[str, Any],
    objects: list[SceneObject3D],
    subjects: list[SceneObject3D],
) -> list[float]:
    del subjects
    explicit = frame.get("play_bounds")
    if isinstance(explicit, list) and len(explicit) >= 3:
        return [float(explicit[0]), float(explicit[1]), float(explicit[2])]
    grounds = [obj for obj in objects if obj.semantic_type == "ground"]
    if grounds:
        ground = max(grounds, key=lambda item: item.size[0] * item.size[1])
        return [ground.size[0], ground.size[1], max(8.0, ground.size[2] + 8.0)]
    xs = [obj.bounds["x1"] for obj in objects] + [obj.bounds["x2"] for obj in objects]
    ys = [obj.bounds["y1"] for obj in objects] + [obj.bounds["y2"] for obj in objects]
    return [max(xs) - min(xs), max(ys) - min(ys), 8.0] if xs and ys else [24.0, 18.0, 8.0]


def _object_in_bounds(obj: SceneObject3D, bounds: list[float]) -> bool:
    value = obj.bounds
    return (
        value["x1"] >= -float(bounds[0]) * 0.5
        and value["x2"] <= float(bounds[0]) * 0.5
        and value["y1"] >= -float(bounds[1]) * 0.5
        and value["y2"] <= float(bounds[1]) * 0.5
        and value["z1"] >= -1.0
        and value["z2"] <= float(bounds[2])
    )


def _quantified_subjects(
    subjects: list[SceneObject3D],
    matched_ids: set[str],
    predicate: dict[str, Any],
) -> bool:
    if not subjects:
        return False
    if predicate.get("subject_quantifier") == "all":
        return all(subject.id in matched_ids for subject in subjects)
    return any(subject.id in matched_ids for subject in subjects)


def _distance(start: Any, end: Any, *, space: str) -> float:
    dimensions = 3 if space == "xyz" else 2
    return math.sqrt(
        sum((float(end[index]) - float(start[index])) ** 2 for index in range(dimensions))
    )


def _within(value: float | int, minimum: Any, maximum: Any) -> bool:
    if minimum is not None and float(value) < float(minimum):
        return False
    if maximum is not None and float(value) > float(maximum):
        return False
    return True


def _condition_satisfaction_frames(
    values: list[bool],
    *,
    temporal: str,
    condition: dict[str, Any],
    passed: bool,
) -> list[int]:
    if not passed:
        return []
    if temporal == "eventually":
        return _rising_edge_indices(values)
    if temporal == "sustained":
        return _sustained_end_indices(values, int(condition.get("frames") or 1))
    if temporal == "count":
        minimum = int(condition.get("min_count") or 0)
        if minimum:
            return _rising_edge_indices(values)[minimum - 1 :]
    return [len(values) - 1]


def _select_ordered_witnesses(
    candidate_frames: list[list[int]],
) -> tuple[list[int | None], bool]:
    """Choose the earliest monotonic witness sequence, allowing same-frame events."""

    selected: list[int | None] = []
    minimum_frame = 0
    for index, candidates in enumerate(candidate_frames):
        witness = next((frame for frame in candidates if frame >= minimum_frame), None)
        selected.append(witness)
        if witness is None:
            selected.extend([None] * (len(candidate_frames) - index - 1))
            return selected, False
        minimum_frame = witness
    return selected, True


def _rising_edge_indices(values: list[bool]) -> list[int]:
    indices: list[int] = []
    previous = False
    for index, value in enumerate(values):
        if value and not previous:
            indices.append(index)
        previous = value
    return indices


def _rising_edge_count(values: list[bool]) -> int:
    return len(_rising_edge_indices(values))


def _longest_true_streak(values: list[bool]) -> tuple[int, int | None]:
    current = 0
    longest = 0
    end: int | None = None
    for index, value in enumerate(values):
        current = current + 1 if value else 0
        if current > longest:
            longest = current
            end = index
    return longest, end


def _sustained_end_indices(values: list[bool], required: int) -> list[int]:
    indices: list[int] = []
    current = 0
    for index, value in enumerate(values):
        current = current + 1 if value else 0
        if current == required:
            indices.append(index)
    return indices


def _rounded_dict(values: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 6) for key, value in values.items()}


def _matrix_yaw(matrix: Any) -> float:
    if not isinstance(matrix, (list, tuple)) or len(matrix) < 9:
        return 0.0
    return math.atan2(float(matrix[3]), float(matrix[0]))


def _yaw_matrix(yaw: float) -> list[float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return [c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0]
