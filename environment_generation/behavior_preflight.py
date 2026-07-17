"""Adaptive validation and navigation setup for behavioral rollouts."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from .env_behavior_trials import (
    BEHAVIOR_CONTROLLER_VERSION,
    MAX_STEPS,
    normalize_behavior_trial_for_runtime,
)
from .env_verification import SceneObject3D, scene_objects, select_scene_objects
from .player import MOVE_SPEED, PlayableSimulation


NAVIGATION_HEADROOM_MULTIPLIER = 1.35
NAVIGATION_ACTION_HEADROOM_STEPS = 120
STABILITY_POST_ARRIVAL_STEPS = 80
MAX_STABILITY_PROBE_STEPS = 1200
STABILITY_MIN_XY_TOLERANCE = 0.25
STABILITY_MIN_Z_TOLERANCE = 0.25


def prepare_behavior_trial(
    *,
    scene_dir: Path,
    trial: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a runtime trial with an adaptive budget plus typed preflight evidence."""

    simulation = PlayableSimulation.from_scene(scene_dir)
    runtime_trial = copy.deepcopy(
        normalize_behavior_trial_for_runtime(trial, spec=simulation.spec)
    )
    objects = scene_objects(simulation.spec)
    object_by_id = {obj.id: obj for obj in objects}
    agent = object_by_id[simulation.agent.object_id]
    target_candidates = _objective_targets(objects, runtime_trial)
    targets = [candidate[1] for candidate in target_candidates]
    primary = (
        min(target_candidates, key=lambda candidate: (-candidate[0], _xy_edge_distance(agent, candidate[1])))[1]
        if target_candidates
        else None
    )
    timestep = float(simulation.model.opt.timestep)
    primary_distance = _xy_edge_distance(agent, primary) if primary is not None else 0.0
    estimated_travel_distance = _estimated_trial_distance(
        agent=agent,
        objects=objects,
        trial=runtime_trial,
    )
    minimum_travel_steps = math.ceil(
        estimated_travel_distance / max(1e-9, MOVE_SPEED * timestep)
    )
    configured_steps = int(runtime_trial.get("max_steps") or 2400)
    recommended_steps = configured_steps
    if primary is not None and estimated_travel_distance > 0.05:
        recommended_steps = max(
            configured_steps,
            math.ceil(minimum_travel_steps * NAVIGATION_HEADROOM_MULTIPLIER)
            + NAVIGATION_ACTION_HEADROOM_STEPS,
        )
    effective_steps = min(MAX_STEPS, recommended_steps)
    runtime_trial["configured_max_steps"] = configured_steps
    runtime_trial["max_steps"] = effective_steps
    runtime_trial["max_steps_per_attempt"] = effective_steps
    runtime_trial["max_total_steps"] = effective_steps * (int(runtime_trial.get("max_resets") or 0) + 1)
    runtime_trial["navigation"] = {
        "primary_target_id": primary.id if primary is not None else None,
        "target_ids": [target.id for target in targets],
        "initial_camera_azimuth": _controller_azimuth(agent, primary) if primary is not None else -90.0,
        "initial_camera_elevation": _camera_elevation(agent, primary) if primary is not None else 0.0,
        "initial_distance_xy": round(primary_distance, 5),
        "estimated_trial_travel_distance_xy": round(estimated_travel_distance, 5),
        "minimum_travel_steps": minimum_travel_steps,
        "recommended_steps_per_attempt": recommended_steps,
        "effective_steps_per_attempt": effective_steps,
        "primary_target": (
            {
                "id": primary.id,
                "semantic_type": primary.semantic_type,
                "body_type": primary.body_type,
                "position": list(primary.position),
                "size": list(primary.size),
            }
            if primary is not None
            else None
        ),
    }

    issues = _semantic_issues(
        runtime_trial,
        spec=simulation.spec,
        objects=objects,
        agent=agent,
    )
    warnings: list[str] = []
    if effective_steps > configured_steps:
        warnings.append(
            f"Expanded each attempt from {configured_steps} to {effective_steps} steps so the agent can reach "
            f"the primary target and still interact with it."
        )
    if recommended_steps > MAX_STEPS:
        issues.append(
            f"The target requires approximately {recommended_steps} steps per attempt, above the supported "
            f"limit of {MAX_STEPS}."
        )

    stability_targets = _stability_sensitive_targets(objects, runtime_trial)
    stability = _probe_target_stability(
        scene_dir=scene_dir,
        target_ids=[target.id for target in stability_targets],
        probe_steps=min(
            MAX_STABILITY_PROBE_STEPS,
            max(120, minimum_travel_steps + STABILITY_POST_ARRIVAL_STEPS),
        ),
    )
    unstable = [item for item in stability["targets"] if not item["stable"]]
    if unstable and not bool(runtime_trial.get("allow_target_motion")):
        names = ", ".join(item["id"] for item in unstable)
        issues.append(
            f"Target geometry ({names}) moves or collapses before a minimally fast agent can reach it. "
            "Use stable/anchored geometry, or set allow_target_motion only when target motion is intentional."
        )
    elif unstable:
        warnings.append("Target motion is intentional for this trial and was retained in the rollout.")

    preflight = {
        "schema_version": "1.0",
        "controller_version": BEHAVIOR_CONTROLLER_VERSION,
        "status": "invalid_setup" if issues else "ready",
        "configured_steps_per_attempt": configured_steps,
        "effective_steps_per_attempt": effective_steps,
        "max_attempts": int(runtime_trial.get("max_resets") or 0) + 1,
        "navigation": runtime_trial["navigation"],
        "stability": stability,
        "issues": issues,
        "warnings": warnings,
        "repair_hints": list(issues),
    }
    runtime_trial["preflight"] = preflight
    return runtime_trial, preflight


def _objective_targets(
    objects: list[SceneObject3D],
    trial: dict[str, Any],
) -> list[tuple[int, SceneObject3D]]:
    values: dict[str, tuple[int, SceneObject3D]] = {}
    for check in (trial.get("objective") or {}).get("checks") or []:
        if _is_max_only(check):
            continue
        priority = _navigation_priority(check, objects=objects)
        for target in _navigation_objects_for_check(objects, check):
            if target.semantic_type != "agent":
                current = values.get(target.id)
                if current is None or priority > current[0]:
                    values[target.id] = (priority, target)
    return list(values.values())


def _navigation_priority(
    check: dict[str, Any],
    *,
    objects: list[SceneObject3D],
) -> int:
    predicate = check.get("predicate") or {}
    if _is_manipulation_predicate(objects=objects, predicate=predicate):
        return 6
    if predicate.get("type") == "mechanism_state":
        return 5
    if predicate.get("type") == "relation" and check.get("temporal") == "at_end":
        return 4
    if predicate.get("type") in {"overlap", "displacement", "mechanism_state"}:
        return 3
    if predicate.get("type") == "contact":
        return 2
    return 1


def _stability_sensitive_targets(
    objects: list[SceneObject3D],
    trial: dict[str, Any],
) -> list[SceneObject3D]:
    values: dict[str, SceneObject3D] = {}
    for check in (trial.get("objective") or {}).get("checks") or []:
        predicate = check.get("predicate") or {}
        if predicate.get("type") != "relation" or predicate.get("relation") not in {
            "on_surface",
            "above",
            "below",
        }:
            continue
        selector = predicate.get("target")
        if selector is None:
            continue
        for target in select_scene_objects(objects, selector):
            if target.body_type == "dynamic":
                values[target.id] = target
    return list(values.values())


def _navigation_objects_for_check(
    objects: list[SceneObject3D],
    check: dict[str, Any],
) -> list[SceneObject3D]:
    predicate = check.get("predicate") or {}
    subjects = (
        select_scene_objects(objects, predicate.get("subject"))
        if isinstance(predicate.get("subject"), dict)
        else []
    )
    targets = (
        select_scene_objects(objects, predicate.get("target"))
        if isinstance(predicate.get("target"), dict)
        else []
    )
    movable_subjects = [
        subject
        for subject in subjects
        if subject.semantic_type != "agent" and subject.body_type == "dynamic"
    ]
    if movable_subjects:
        return movable_subjects
    if predicate.get("type") == "mechanism_state":
        selector = check.get("selector")
        return select_scene_objects(objects, selector) if isinstance(selector, dict) else []
    if any(subject.semantic_type == "agent" for subject in subjects):
        return targets
    return subjects or targets


def _estimated_trial_distance(
    *,
    agent: SceneObject3D,
    objects: list[SceneObject3D],
    trial: dict[str, Any],
) -> float:
    group = trial.get("objective") or {}
    checks_by_id = {
        str(check.get("id")): check
        for check in group.get("checks") or []
        if isinstance(check, dict)
    }
    ordered = [str(value) for value in group.get("ordered_check_ids") or []]
    ordered.extend(check_id for check_id in checks_by_id if check_id not in ordered)
    current = agent
    distance = 0.0
    for check_id in ordered:
        check = checks_by_id.get(check_id) or {}
        if _is_max_only(check):
            continue
        predicate = check.get("predicate") or {}
        subjects = (
            select_scene_objects(objects, predicate.get("subject"))
            if isinstance(predicate.get("subject"), dict)
            else []
        )
        targets = (
            select_scene_objects(objects, predicate.get("target"))
            if isinstance(predicate.get("target"), dict)
            else []
        )
        movable = next(
            (
                subject
                for subject in subjects
                if subject.semantic_type != "agent" and subject.body_type == "dynamic"
            ),
            None,
        )
        if movable is not None:
            distance += _xy_edge_distance(current, movable)
            if targets:
                distance += _xy_edge_distance(movable, targets[0])
                current = targets[0]
            else:
                current = movable
            continue
        navigation = _navigation_objects_for_check(objects, check)
        if navigation:
            distance += _xy_edge_distance(current, navigation[0])
            current = navigation[0]
    return distance


def _semantic_issues(
    trial: dict[str, Any],
    *,
    spec: Any,
    objects: list[SceneObject3D],
    agent: SceneObject3D,
) -> list[str]:
    issues: list[str] = []
    issues.extend(str(value) for value in trial.get("migration_issues") or [])
    if trial.get("controller_version") not in {None, BEHAVIOR_CONTROLLER_VERSION}:
        issues.append("The behavior plan uses an older controller contract and must be regenerated.")
    if trial.get("expected_outcome") == "should_not_succeed":
        for check in (trial.get("objective") or {}).get("checks") or []:
            if _is_max_only(check):
                issues.append(
                    f"Objective check {check.get('id')!r} describes an expected safety limit, not the prohibited "
                    "counterexample. Remove outcome limits, describe the forbidden outcome positively, and use "
                    "constraints only for genuine attempt rules such as no jumping."
                )
        max_height_gain = _constraint_max_height_gain(trial)
        if max_height_gain is not None:
            required_gain = _required_vertical_gain(objects, agent, trial)
            if required_gain is not None and max_height_gain + 0.05 < required_gain:
                issues.append(
                    f"The attempt limits agent height gain to {max_height_gain:.2f} m, but the prohibited target "
                    f"requires approximately {required_gain:.2f} m of height gain. This would make the "
                    "counterexample impossible by definition instead of testing the environment."
                )
    return issues


def _constraint_max_height_gain(trial: dict[str, Any]) -> float | None:
    limits = [
        float((check.get("predicate") or {})["max_value"])
        for check in (trial.get("constraints") or {}).get("checks") or []
        if (check.get("predicate") or {}).get("type") == "axis_delta"
        and (check.get("predicate") or {}).get("axis") == "z"
        and (check.get("predicate") or {}).get("max_value") is not None
    ]
    return min(limits) if limits else None


def _required_vertical_gain(
    objects: list[SceneObject3D],
    agent: SceneObject3D,
    trial: dict[str, Any],
) -> float | None:
    gains: list[float] = []
    for check in (trial.get("objective") or {}).get("checks") or []:
        predicate = check.get("predicate") or {}
        if predicate.get("type") != "relation" or predicate.get("relation") not in {"on_surface", "above"}:
            continue
        subjects = select_scene_objects(objects, predicate.get("subject") or {})
        if not any(subject.semantic_type == "agent" for subject in subjects):
            continue
        for target in select_scene_objects(objects, predicate.get("target") or {}):
            required_center = float(target.bounds["z2"]) + agent.size[2] * 0.5
            gains.append(max(0.0, required_center - agent.position[2]))
    return min(gains) if gains else None


def _is_max_only(check: dict[str, Any]) -> bool:
    temporal = str(check.get("temporal") or "eventually")
    predicate = check.get("predicate") or {}
    if temporal in {"always", "never"}:
        return True
    if temporal == "count":
        return "max_count" in check and int(check.get("min_count") or 0) <= 0
    return "max_value" in predicate and "min_value" not in predicate


def _is_manipulation_predicate(
    *,
    objects: list[SceneObject3D],
    predicate: dict[str, Any],
) -> bool:
    if predicate.get("type") not in {
        "overlap",
        "relation",
        "contact",
        "displacement",
        "axis_delta",
        "axis_value",
    }:
        return False
    return any(
        subject.semantic_type != "agent" and subject.body_type == "dynamic"
        for subject in select_scene_objects(objects, predicate.get("subject") or {})
    )


def _probe_target_stability(
    *,
    scene_dir: Path,
    target_ids: list[str],
    probe_steps: int,
) -> dict[str, Any]:
    if not target_ids:
        return {"probe_steps": 0, "seconds": 0.0, "targets": []}
    simulation = PlayableSimulation.from_scene(scene_dir)
    starts = {
        object_id: tuple(float(value) for value in simulation.data.xpos[simulation.dynamic_body_ids[object_id]])
        for object_id in target_ids
        if object_id in simulation.dynamic_body_ids
    }
    max_displacement = {object_id: 0.0 for object_id in starts}
    max_vertical_drop = {object_id: 0.0 for object_id in starts}
    for _ in range(probe_steps):
        simulation.step(right=0.0, forward=0.0, camera_azimuth=-90.0, jump=False)
        for object_id, start in starts.items():
            current = tuple(
                float(value)
                for value in simulation.data.xpos[simulation.dynamic_body_ids[object_id]]
            )
            max_displacement[object_id] = max(max_displacement[object_id], math.dist(start, current))
            max_vertical_drop[object_id] = max(max_vertical_drop[object_id], start[2] - current[2])
    spec_by_id = {obj.id: obj for obj in simulation.spec.objects}
    targets = []
    for object_id, start in starts.items():
        obj = spec_by_id[object_id]
        xy_tolerance = max(STABILITY_MIN_XY_TOLERANCE, min(obj.size[0], obj.size[1]) * 0.3)
        z_tolerance = max(STABILITY_MIN_Z_TOLERANCE, obj.size[2] * 0.3)
        displacement = max_displacement[object_id]
        vertical_drop = max_vertical_drop[object_id]
        targets.append(
            {
                "id": object_id,
                "stable": displacement <= xy_tolerance and vertical_drop <= z_tolerance,
                "max_displacement": round(displacement, 5),
                "max_vertical_drop": round(vertical_drop, 5),
                "displacement_tolerance": round(xy_tolerance, 5),
                "vertical_drop_tolerance": round(z_tolerance, 5),
                "start_position": list(start),
            }
        )
    return {
        "probe_steps": probe_steps,
        "seconds": round(probe_steps * float(simulation.model.opt.timestep), 5),
        "targets": targets,
    }


def _xy_edge_distance(subject: SceneObject3D, target: SceneObject3D | None) -> float:
    if target is None:
        return 0.0
    sx = (subject.bounds["x1"] + subject.bounds["x2"]) * 0.5
    sy = (subject.bounds["y1"] + subject.bounds["y2"]) * 0.5
    tx = (target.bounds["x1"] + target.bounds["x2"]) * 0.5
    ty = (target.bounds["y1"] + target.bounds["y2"]) * 0.5
    subject_radius = max(subject.bounds["x2"] - sx, subject.bounds["y2"] - sy)
    target_radius = max(target.bounds["x2"] - tx, target.bounds["y2"] - ty)
    return max(0.0, math.hypot(tx - sx, ty - sy) - subject_radius - target_radius)


def _controller_azimuth(subject: SceneObject3D, target: SceneObject3D) -> float:
    dx = target.position[0] - subject.position[0]
    dy = target.position[1] - subject.position[1]
    if math.hypot(dx, dy) <= 1e-9:
        return -90.0
    return math.degrees(math.atan2(-dx, -dy))


def _camera_elevation(subject: SceneObject3D, target: SceneObject3D) -> float:
    horizontal = max(0.5, math.hypot(
        target.position[0] - subject.position[0],
        target.position[1] - subject.position[1],
    ))
    vertical = target.position[2] - subject.position[2]
    return max(-15.0, min(15.0, math.degrees(math.atan2(vertical, horizontal))))
